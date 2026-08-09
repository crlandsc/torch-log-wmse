"""The logWMSE metric and loss function.

Derived from nomonosound/log-wmse-audio-quality (Copyright 2023 Nomono), licensed under the
Apache License 2.0. Modified by Christopher Landschoot (SoundFoxLabs) in 2024-2026: ported from numpy to PyTorch,
FFT convolution in place of scipy.signal.oaconvolve, batched [batch, channel, stem, time] tensor
API, differentiable loss support, and the impulse-response/silence handling described in the
README and CHANGELOG.
"""
import math

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
    return torch.log(_shifted(mse)) * SCALER


def _shifted(mse: Tensor) -> Tensor:
    """`mse + EPS`, in at least float32. The one place EPS is ever added.

    EPS GOES INSIDE ANY POOLING THAT FOLLOWS, never after it. Adding it to a pooled value instead
    diverges from the per-element form by up to 32 units whenever a stem is perfect, and at
    p = 1/2 it produces a NaN GRADIENT on a bit-exact stem, because the derivative of sqrt is
    infinite at zero. That fires on a digitally silent target matched exactly - the package's
    headline case - and a forward-only test cannot see it, because the value stays finite.
    """
    if mse.dtype not in (torch.float32, torch.float64):
        mse = mse.to(torch.float32)
    return mse + EPS


def pool_mse(mse: Tensor, p: float) -> Tensor:
    """Pool per-[batch, channel, stem] MSE across CHANNEL and STEM, giving one score per batch item.

        M_p   = (mean over (channel, stem) of (mse + EPS)**p)**(1/p)
        score = SCALER * log(M_p)

    `p` selects how a spread of per-stem errors combines. Per-stem gradient ENERGY scales as
    `mse**(2p - 1)`, so it is equal across stems if and only if `p = 1/2` - a derived value rather
    than a tuned one. `p = 0` is the mean of logs, which is what every earlier version computed.

    p = 0 IS SPECIAL-CASED rather than computed as a limit. Writing it as `exp(mean(log(x)))` is
    mathematically the same but not bit-identical, and landing the machinery at p = 0 against an
    exact gate is what makes the later default change attributable to itself alone.

    The BATCH axis is deliberately excluded. Batch items are independent samples, so averaging
    their losses is what makes the objective an expectation over the data; pooling them
    non-linearly couples examples and breaks both gradient accumulation and DDP equivalence with a
    single large batch.

    Args:
        mse (Tensor): [batch, channel, stem].
        p (float): Power-mean exponent, >= 0.

    Returns:
        Tensor: [batch].
    """
    shifted = _shifted(mse)
    if p == 0.0:
        return torch.log(shifted).mean(dim=(1, 2)) * SCALER
    return torch.log(shifted.pow(p).mean(dim=(1, 2)).pow(1.0 / p)) * SCALER


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

    HIGHER IS BETTER. For training, use `LogWMSELoss`, which is this negated. The two were one
    class with a `return_as_loss` flag until 1.0.0; a flag that silently inverts a training
    objective is not worth the convenience of a single import.

    ALL ARGUMENTS ARE KEYWORD-ONLY. `audio_length` used to come first, so a positional call written
    for an earlier version would otherwise land it on `sample_rate` and quietly build a metric at
    1 Hz instead of raising.

    Args:
        sample_rate (int, optional): The sample rate of the audio signal in Hz. Defaults to 44100.
        p (float, optional): Power-mean exponent for pooling across channel and stem. See
            `pool_mse`. Defaults to 0.0, the mean of logs that every earlier version computed.
        impulse_response (Tensor, optional): The finite impulse response (FIR) filter for
            frequency weighting. If None (default), use built-in FIR. Currently only supports
            single-channel FIRs (applied to all batches & audio channels).
        impulse_response_sample_rate (int, optional): The sample rate of the FIR in Hz. Defaults to 44100.
        bypass_filter (bool, optional): Whether to bypass the frequency weighting filter. Defaults to False.
        reduction (str, optional): How to aggregate over the BATCH axis - and only the batch axis,
            as in any other torch loss. One of "mean" (default), "sum", or "none" for per-item
            values. Channel and stem are pooled by `p` before this applies.
    """
    def __init__(
            self,
            *,
            sample_rate: int = 44100,
            p: float = 0.0,
            impulse_response: Optional[Tensor] = None,
            impulse_response_sample_rate: int = 44100,
            bypass_filter: bool = False,
            reduction: str = "mean",
            audio_length: Optional[float] = None,
            return_as_loss: Optional[bool] = None,
        ):
        super().__init__()
        # Removed arguments are accepted only to reject them by name. Silently ignoring them would
        # let a caller believe a length was configured or a sign was chosen when neither happened.
        if audio_length is not None:
            raise TypeError(
                "audio_length was removed in 1.0.0. One instance now serves any input length: the "
                "transform size is derived per call. Drop the argument."
            )
        if return_as_loss is not None:
            raise TypeError(
                "return_as_loss was removed in 1.0.0. Use LogWMSE for the metric (higher is "
                "better) or LogWMSELoss for the loss (lower is better)."
            )

        p = float(p)
        if not math.isfinite(p) or p < 0:
            raise ValueError(f"p must be a finite, non-negative number, got {p!r}")
        self.p = p

        self.sample_rate = sample_rate
        self.filters = HumanHearingSensitivityFilter(
            sample_rate=sample_rate,
            impulse_response=impulse_response,
            impulse_response_sample_rate=impulse_response_sample_rate,
        )
        self.bypass_filter = bypass_filter
        if reduction not in VALID_REDUCTIONS:
            raise ValueError(f"reduction must be one of {VALID_REDUCTIONS}, got {reduction!r}")
        self.reduction = reduction

    def extra_repr(self) -> str:
        # sample_rate cannot be inferred from the input the way length now is - the impulse response
        # is resampled to it - so surface it, or a caller who has just learned that length is
        # automatic will assume the rate is too.
        return (f"sample_rate={self.sample_rate}, p={self.p}, reduction={self.reduction!r}"
                + (", bypass_filter=True" if self.bypass_filter else ""))

    def per_stem(self, unprocessed_audio: Tensor, processed_audio: Tensor,
                 target_audio: Tensor) -> Tensor:
        """Per-[batch, channel, stem] scores, unpooled and unreduced.

        These are the values `forward` pools. They are also the values that stay comparable with
        the original numpy implementation whatever `p` is, since `p` changes only how they combine
        - which is why the fidelity anchor is visible in the API rather than buried.

        Returns:
            Tensor: [batch, channel, stem]. Higher is better.
        """
        return score_from_mse(self._mse(unprocessed_audio, processed_audio, target_audio))

    def _mse(self, unprocessed_audio: Tensor, processed_audio: Tensor,
             target_audio: Tensor) -> Tensor:
        """Validated per-[batch, channel, stem] MSE. The shared front half of both entry points."""
        self._validate(unprocessed_audio, processed_audio, target_audio)

        # Mean square of the weighted mixture, per [batch, channel, stem=1] so it broadcasts.
        mixture = unprocessed_audio.unsqueeze(2)  # [batch, channel, time] -> [b, c, stem=1, time]
        input_mean_square = (
            weighted_energy(mixture, self.filters, self.bypass_filter) / mixture.shape[-1]
        )
        return per_element_mse(input_mean_square, self.filters, processed_audio, target_audio,
                               bypass_filter=self.bypass_filter)

    def forward(self, unprocessed_audio: Tensor, processed_audio: Tensor, target_audio: Tensor) -> Tensor:
        """The pooled score, reduced over the batch axis. Higher is better."""
        pooled = pool_mse(self._mse(unprocessed_audio, processed_audio, target_audio), self.p)
        return apply_reduction(pooled, self.reduction)

    @staticmethod
    def _validate(unprocessed_audio: Tensor, processed_audio: Tensor, target_audio: Tensor) -> None:
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


class LogWMSELoss(LogWMSE):
    """logWMSE as a training objective: the negated metric, so LOWER IS BETTER.

    A genuine subclass rather than a flag, and it negates at the OUTERMOST point, so
    `loss == -metric` holds for every reduction and for `per_stem` alike. Every other argument
    behaves identically.
    """

    def forward(self, unprocessed_audio: Tensor, processed_audio: Tensor,
                target_audio: Tensor) -> Tensor:
        return -super().forward(unprocessed_audio, processed_audio, target_audio)

    def per_stem(self, unprocessed_audio: Tensor, processed_audio: Tensor,
                 target_audio: Tensor) -> Tensor:
        return -super().per_stem(unprocessed_audio, processed_audio, target_audio)
