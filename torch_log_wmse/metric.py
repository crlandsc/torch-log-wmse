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

from torch_log_wmse.constants import ERROR_TOLERANCE_THRESHOLD, SCALER, EPS, RMS_EPS
from torch_log_wmse.freq_weighting_filter import HumanHearingSensitivityFilter
from torch_log_wmse.utils import VALID_REDUCTIONS, apply_reduction, calculate_rms


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
    * Invariant to the tiny errors of MSE that are inaudible to humans.
    * Logarithmic, reflecting the logarithmic sensitivity of human hearing.
    * Tailored specifically for audio signals.

    Args:
        audio_length (float): The length of the audio signal in seconds. May be fractional.
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
            audio_length: float,
            sample_rate: int = 44100,
            impulse_response: Optional[Tensor] = None,
            impulse_response_sample_rate: int = 44100,
            return_as_loss: bool = True,
            bypass_filter: bool = False,
            reduction: str = "mean",
        ):
        super().__init__()
        self.filters = HumanHearingSensitivityFilter(
            audio_length=audio_length,
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
        # The filter truncates to the length configured at construction, so a mismatch would silently
        # score a different window than the caller passed - and bypass_filter=True would not truncate
        # at all, making the two paths disagree on which samples they scored.
        # audio_length * sample_rate is rarely an exact integer, and floor() vs round() differ by one
        # sample for many fractional lengths (22 of the 399 hundredth-second values at 44.1 kHz). A
        # caller who sized their segment with round() is not making a mistake, so accept either.
        expected = self.filters.audio_length_samples
        if unprocessed_audio.shape[-1] not in (expected, expected + 1):
            raise ValueError(
                f"expected {expected} samples for the configured audio_length "
                f"(floor(audio_length * sample_rate); {expected + 1} is also accepted for callers who "
                f"round up), got {unprocessed_audio.shape[-1]}. Construct a LogWMSE for this length, "
                "or trim/pad the input to match."
            )

        if self.bypass_filter:
            input_rms = calculate_rms(unprocessed_audio.unsqueeze(2))  # [batch, channel, time] -> [batch, channel, stem=1, time]
        else:
            input_rms = calculate_rms(self.filters(unprocessed_audio.unsqueeze(2)))  # [batch, channel, time] -> [batch, channel, stem=1, time]

        # Calculate the logWMSE
        values = self._calculate_log_wmse(
            input_rms,
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
        input_rms: Tensor,
        filters: Callable,
        processed_audio: Tensor,
        target_audio: Tensor,
        bypass_filter: bool = False,
    ):
        """
        Calculate the logWMSE between the processed audio and target audio.

        Args:
            input_rms (Tensor): The root mean square of the input audio. Shape: [batch, channel, stem].
            filters (Callable): A function that applies a filter to the audio (i.e. HumanHearingSensitivityFilter).
            processed_audio (Tensor): The processed audio tensor. Shape: [batch, channel, stem, time].
            target_audio (Tensor): The target audio tensor. Shape: [batch, channel, stem, time].
            bypass_filter (bool, optional): Whether to bypass the frequency weighting filter. Defaults to False.

        Returns:
            Tensor: The logWMSE between the processed audio and target audio.
        """

        # Calculate the scaling factor based on the input RMS. RMS_EPS is a FLOOR rather than an
        # addend: clamping leaves the scaling factor exactly 1/input_rms for every non-degenerate
        # input, so joint-gain scale invariance holds exactly, whereas adding RMS_EPS biases very
        # quiet mixtures. Both forms give an identical value for a digitally silent mixture.
        scaling_factor = 1 / torch.clamp_min(input_rms, RMS_EPS)

        # Add extra dimension(s) to scaling_factor to match the shape of processed_audio and target_audio
        while scaling_factor.dim() < processed_audio.dim():
            scaling_factor = scaling_factor.unsqueeze(-1)

        # Calculate the frequency-weighted differences, ignoring small imperceptible differences.
        # The filter is linear, so filters(a) - filters(b) == filters(a - b); taking the difference first
        # halves the filtered signals and so removes one rfft/irfft pair per call.
        # Skip frequency weighting if bypass_filter is True.
        differences = (processed_audio - target_audio) * scaling_factor
        if not bypass_filter:
            differences = filters(differences)

        # Discard differences too small to be audible. torch.where rather than an in-place masked
        # assignment: it avoids a data-dependent scatter, which keeps the graph friendly to
        # torch.compile. Note it does NOT save memory - both forms build the full-size boolean
        # condition, and where() additionally allocates a new output buffer.
        differences = torch.where(
            torch.abs(differences) < ERROR_TOLERANCE_THRESHOLD,
            torch.zeros((), dtype=differences.dtype, device=differences.device),
            differences,
        )

        # Calculate the mean squared differences
        mean_diff = (differences**2).mean(dim=-1)

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
        if mean_diff.dtype not in (torch.float32, torch.float64):
            mean_diff = mean_diff.to(torch.float32)

        return torch.log(mean_diff + EPS) * SCALER
