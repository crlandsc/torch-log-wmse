"""Shared helpers for the logWMSE metric.

Derived from nomonosound/log-wmse-audio-quality (Copyright 2023 Nomono), licensed under the
Apache License 2.0. Modified by Christopher Landschoot (SoundFoxLabs) in 2024-2026: ported from numpy to PyTorch,
FFT convolution in place of scipy.signal.oaconvolve, batched [batch, channel, stem, time] tensor
API, differentiable loss support, and the impulse-response/silence handling described in the
README and CHANGELOG.
"""
import torch
from typing import Union

def calculate_rms(samples: torch.Tensor):
    """
    Calculates the Root Mean Square (RMS) power level of a tensor of audio samples.
        Args: samples (torch.Tensor): A tensor containing audio samples, with time samples in the last dimension.
        Returns: torch.Tensor: A tensor with the last dimension (time) reduced, containing the RMS power level of the audio samples.
    """
    # clamp_min, not `+ eps`: at exactly zero `sqrt` has infinite derivative, so a grad-requiring
    # all-silent input propagates NaN backwards while the forward value still looks finite.
    #
    # The floor is the dtype's smallest normal rather than a hard-coded constant. A literal like
    # 1e-24 underflows to 0.0 in float16, making the guard a silent no-op there, and it would also
    # rewrite any legitimate mean-square below it -- float32 represents mean-squares down to ~1e-38,
    # so a fixed 1e-24 would alter about seven decades of genuinely quiet audio. finfo().tiny clamps
    # only subnormals, so every normal value is untouched in every dtype.
    mean_square = torch.mean(torch.square(samples), dim=-1)
    return torch.sqrt(torch.clamp_min(mean_square, torch.finfo(mean_square.dtype).tiny))

def convert_decibels_to_amplitude_ratio(decibels: Union[torch.Tensor, float]):
    """
    Converts a tensor of decibel values into a tensor of amplitude ratios.
        Args: decibels (Union[torch.Tensor, float]): A tensor containing decibel values.
        Returns: torch.Tensor: A tensor of the same shape as the input, but containing amplitude ratio values.
    """
    if not isinstance(decibels, torch.Tensor):
        decibels = torch.tensor(decibels)
    return torch.pow(10, decibels / 20)

VALID_REDUCTIONS = ("none", "mean", "sum")


def apply_reduction(losses: torch.Tensor, reduction: str = "none") -> torch.Tensor:
    """Apply a reduction to a collection of losses.

    Args:
        losses (torch.Tensor): Per-[batch, channel, stem] values.
        reduction (str): One of "none", "mean" or "sum".

    Returns:
        torch.Tensor: The reduced values, or `losses` unchanged for "none".

    Raises:
        ValueError: If `reduction` is not one of VALID_REDUCTIONS. This matters because the function
            previously fell through silently for any unrecognised value, so a typo such as "Sum"
            behaved as "none" and returned an unreduced tensor.
    """
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    if reduction == "none":
        return losses
    raise ValueError(f"reduction must be one of {VALID_REDUCTIONS}, got {reduction!r}")
