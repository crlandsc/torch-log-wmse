"""The logWMSE metric and loss function.

Derived from nomonosound/log-wmse-audio-quality (Copyright 2023 Nomono), licensed under the
Apache License 2.0. Modified by Christopher Landschoot (SoundFoxLabs) in 2024-2026: ported from numpy to PyTorch,
FFT convolution in place of scipy.signal.oaconvolve, batched [batch, channel, stem, time] tensor
API, differentiable loss support, and the impulse-response/silence handling described in the
README and CHANGELOG.
"""
import torch
from torch import Tensor
from typing import Callable, Optional

from torch_log_wmse.constants import SCALER, EPS, RMS_EPS
from torch_log_wmse.freq_weighting_filter import HumanHearingSensitivityFilter
from torch_log_wmse.utils import VALID_REDUCTIONS, apply_reduction


def accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    """The dtype energies are summed in: never below float32.

    Energies are squares, so half precision gets strictly worse here than it is for the signal
    itself - and `RMS_EPS**2 = 1e-16`, the floor that keeps a silent mixture finite, underflows to
    exactly 0.0 in float16.
    """
    return dtype if dtype in (torch.float32, torch.float64) else torch.float32


def weighted_energy(
    signal: Tensor, filters: Callable, bypass_filter: bool = False
) -> Tensor:
    """Total energy of `signal` after frequency weighting, per [batch, channel, stem].

    `bypass_filter` takes the time-domain route on purpose rather than filtering with a flat
    response: it is the only path that works in half precision, because `torch.fft` has no half
    kernel on CPU or MPS.
    """
    if bypass_filter:
        return signal.to(accumulation_dtype(signal.dtype)).pow(2).sum(dim=-1)
    return filters(signal)


def per_element_mse(
    input_mean_square: Tensor,
    filters: Callable,
    processed_audio: Tensor,
    target_audio: Tensor,
    bypass_filter: bool = False,
) -> Tensor:
    """The frequency-weighted mean squared error, per [batch, channel, stem].

    This is the quantity everything downstream is a function of, and it is deliberately separate
    from the log: the two halves change independently. The filtering rewrite replaces how this is
    computed while leaving the score unchanged, and the pooling change replaces how these values
    combine while leaving this untouched.

    The floor is `RMS_EPS**2`, not `RMS_EPS`: the clamp applies to a MEAN SQUARE here, where it
    used to apply to an RMS that was subsequently squared. Getting that wrong scales a silent
    mixture by 1e4 instead of 1e8. RMS_EPS is a FLOOR rather than an addend, so the scaling stays
    exactly 1/rms for every non-degenerate input; adding it would bias very quiet mixtures.

    The scaling is FOLDED INTO THE DIVISOR rather than multiplied through the waveform. The filter
    is linear, so `E(filter(k*d)) == k^2 * E(filter(d))`, and dividing once at the end avoids an
    O(N) multiply and a full-size temporary.

    Both forms were measured, because dividing afterwards looks like it should be worse
    conditioned - the transform sees data spread over the input's full dynamic range instead of
    normalised to unit RMS. It is not. Exact joint-gain invariance holds either way (bit-identical
    at 2^-10 versus 2^10), the extremes behave identically, and on inputs that are NOT exactly
    proportional in binary - 1e-3 versus 1e3 - folding in is marginally the more accurate of the
    two (2.4e-07 against 7.2e-07). The residual difference there is the inputs, not the metric:
    `u * 1e-3` and `u * 1e3` are not exactly a factor of 1e6 apart in float32.

    Args:
        input_mean_square (Tensor): Mean square of the (weighted) mixture. Broadcasts against
            [batch, channel, stem].
        filters (Callable): The frequency-weighting filter, which returns ENERGY.
        processed_audio (Tensor): [batch, channel, stem, time].
        target_audio (Tensor): [batch, channel, stem, time].
        bypass_filter (bool): Skip the frequency weighting.

    Returns:
        Tensor: [batch, channel, stem].
    """
    differences = processed_audio - target_audio
    n_samples = differences.shape[-1]
    energy = weighted_energy(differences, filters, bypass_filter)
    floor = torch.clamp_min(input_mean_square.to(energy.dtype), RMS_EPS**2)
    return energy / (n_samples * floor)


def score_from_mse(mse: Tensor) -> Tensor:
    """Turn per-element MSE into per-element scores: SCALER * log(mse + EPS).

    Separate from `per_element_mse` because this is the half the aggregation change rewrites, and
    keeping the log out of the MSE computation is what lets pooling happen between the two.

    Args:
        mse (Tensor): Per-[batch, channel, stem] mean squared error.

    Returns:
        Tensor: Per-[batch, channel, stem] scores, higher is better.
    """
    # Take the log in at least float32, whatever the input dtype was. EPS = 1e-8 underflows to
    # exactly 0.0 in float16 (the smallest subnormal is 5.96e-8), so a bit-exact stem used to
    # give log(0) = -inf and the metric returned +inf - and "mean" then propagated that inf
    # across the whole batch. bfloat16 does not underflow but carries 8 mantissa bits, which is
    # not enough to hold mse + EPS apart from mse.
    #
    # Upcasting rather than using a per-dtype floor is a deliberate choice: it keeps the ceiling
    # at SCALER*log(EPS) = +73.6827 in EVERY dtype. A floor of finfo(float16).tiny would move
    # the fp16 ceiling to +38.8 and silently make fp16 and float32 runs incomparable, which is
    # a worse failure than the one being fixed because it looks like a valid number.
    if mse.dtype not in (torch.float32, torch.float64):
        mse = mse.to(torch.float32)

    return torch.log(mse + EPS) * SCALER


class LogWMSE(torch.nn.Module):
    """
    logWMSE is a custom metric and loss function for audio signals that calculates the logarithm
    of a frequency-weighted Mean Squared Error (MSE). It is designed to address several shortcomings
    of common audio metrics, most importantly the lack of support for digital silence targets.

    Key features of logWMSE:
    * Supports digital silence targets not supported by other audio metrics.
        i.e. (SI-)SDR, SIR, SAR, ISR, VISQOL_audio, STOI, CDPAM, and VISQOL.
    * Overcomes the small value range issue of MSE (i.e. between 1e-8 and 1e-3), making number
        formatting and sight-reading easier. Scaled similar to SI-SDR.
    * Scale-invariant, aligns with the frequency sensitivity of human hearing.
    * Logarithmic, reflecting the logarithmic sensitivity of human hearing.
    * Tailored specifically for audio signals.

    One instance serves ANY input length and any stem count. The transform size is derived from the
    input and its weights cached, so nothing is pinned at construction. Note the SAMPLE RATE cannot
    be inferred the same way - the impulse response is resampled to it - so it stays an argument.

    Args:
        sample_rate (int, optional): The sample rate of the audio signal in Hz. Defaults to 44100.
        impulse_response (Tensor, optional): The finite impulse response (FIR) filter for
            frequency weighting. If None (default), use built-in FIR. Currently only supports
            single-channel FIRs (applied to all batches & audio channels).
        impulse_response_sample_rate (int, optional): The sample rate of the FIR in Hz. Defaults to 44100.
        return_as_loss (bool, optional): Whether to return the loss value (i.e. negative of the metric). Defaults to True.
        bypass_filter (bool, optional): Whether to bypass the frequency weighting filter. Defaults to False.
        reduction (str, optional): How to aggregate the per-[batch, channel, stem] values.
            One of "mean" (default), "sum", or "none" to return them unreduced.
    """
    def __init__(
            self,
            sample_rate: int = 44100,
            impulse_response: Optional[Tensor] = None,
            impulse_response_sample_rate: int = 44100,
            return_as_loss: bool = True,
            bypass_filter: bool = False,
            reduction: str = "mean",
        ):
        super().__init__()
        self.sample_rate = sample_rate
        self.filters = HumanHearingSensitivityFilter(
            sample_rate=sample_rate,
            impulse_response=impulse_response,
            impulse_response_sample_rate=impulse_response_sample_rate,
        )
        self.return_as_loss = return_as_loss
        self.bypass_filter = bypass_filter
        if reduction not in VALID_REDUCTIONS:
            raise ValueError(f"reduction must be one of {VALID_REDUCTIONS}, got {reduction!r}")
        self.reduction = reduction

    def forward(self, unprocessed_audio: Tensor, processed_audio: Tensor, target_audio: Tensor) -> Tensor:
        # Validation raises rather than asserting: `python -O` strips assert statements, and these
        # checks are load-bearing. Without the batch/channel checks in particular, a mismatch
        # BROADCASTS and silently returns a plausible-looking number instead of failing.
        if unprocessed_audio.ndim != 3:
            raise ValueError(
                "unprocessed_audio must have shape [batch, channel, time], got "
                f"{tuple(unprocessed_audio.shape)}"
            )
        for name, t in (("processed_audio", processed_audio), ("target_audio", target_audio)):
            if t.ndim != 4:
                raise ValueError(
                    f"{name} must have shape [batch, channel, stem, time], got {tuple(t.shape)}"
                )
        if processed_audio.shape != target_audio.shape:
            raise ValueError(
                "processed_audio and target_audio must have the same shape, got "
                f"{tuple(processed_audio.shape)} and {tuple(target_audio.shape)}"
            )
        # Batch and channel must agree with unprocessed_audio. These were previously unchecked, so a
        # mono mixture against stereo stems (or a batch-size mismatch) broadcast silently.
        if unprocessed_audio.shape[0] != processed_audio.shape[0]:
            raise ValueError(
                f"batch size mismatch: unprocessed_audio has {unprocessed_audio.shape[0]}, "
                f"processed_audio has {processed_audio.shape[0]}"
            )
        if unprocessed_audio.shape[1] != processed_audio.shape[1]:
            raise ValueError(
                f"channel count mismatch: unprocessed_audio has {unprocessed_audio.shape[1]}, "
                f"processed_audio has {processed_audio.shape[1]}"
            )
        if unprocessed_audio.shape[-1] != processed_audio.shape[-1]:
            raise ValueError(
                f"length mismatch: unprocessed_audio has {unprocessed_audio.shape[-1]} samples, "
                f"processed_audio has {processed_audio.shape[-1]}"
            )
        # No length check against a configured audio_length: there is no configured length any more.
        # The transform size is derived from whatever arrives, so the whole error class - including
        # the floor()-versus-round() off-by-one it had to tolerate - is gone.

        # Mean square of the weighted mixture, per [batch, channel, stem=1] so it broadcasts.
        mixture = unprocessed_audio.unsqueeze(2)  # [batch, channel, time] -> [b, c, stem=1, time]
        input_mean_square = (
            weighted_energy(mixture, self.filters, self.bypass_filter) / mixture.shape[-1]
        )

        # Calculate the logWMSE
        values = self._calculate_log_wmse(
            input_mean_square,
            self.filters,
            processed_audio,
            target_audio,
            bypass_filter=self.bypass_filter,
        )

        # Apply reduction using the utility function
        reduced_values = apply_reduction(values, self.reduction)

        if self.return_as_loss:
            return -reduced_values
        else:
            return reduced_values

    @staticmethod
    def _calculate_log_wmse(
        input_mean_square: Tensor,
        filters: Callable,
        processed_audio: Tensor,
        target_audio: Tensor,
        bypass_filter: bool = False,
    ):
        """
        Calculate the logWMSE between the processed audio and target audio.

        Args:
            input_mean_square (Tensor): The mean square of the weighted mixture. Shape: [batch, channel, stem].
            filters (Callable): Returns the weighted ENERGY of a signal (i.e. HumanHearingSensitivityFilter).
            processed_audio (Tensor): The processed audio tensor. Shape: [batch, channel, stem, time].
            target_audio (Tensor): The target audio tensor. Shape: [batch, channel, stem, time].
            bypass_filter (bool, optional): Whether to bypass the frequency weighting filter. Defaults to False.

        Returns:
            Tensor: The logWMSE between the processed audio and target audio.
        """
        return score_from_mse(
            per_element_mse(input_mean_square, filters, processed_audio, target_audio, bypass_filter)
        )
