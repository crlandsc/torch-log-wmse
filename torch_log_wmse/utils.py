"""Shared helpers for the logWMSE metric.

Derived from nomonosound/log-wmse-audio-quality (Copyright 2023 Nomono), licensed under the
Apache License 2.0. Modified by Christopher Landschoot (SoundFoxLabs) in 2024-2026: ported from
numpy to PyTorch; batched [batch, channel, stem, time] tensor API; differentiable loss support; the
weighted error energy computed in the frequency domain via one-sided Parseval rather than by
convolving with scipy.signal.oaconvolve; power-mean aggregation across channel and stem in place of
the mean of logs; the -68 dB per-sample inaudibility gate removed; and the impulse-response and
silence handling described in README.md and CHANGELOG.md.
"""
import torch
from typing import Union

# calculate_rms was removed in 1.0.0. It took the square root of a mean square, and at exactly zero
# `sqrt` has an infinite derivative - so a grad-requiring silent input propagated NaN backwards while
# the forward value still looked finite. Guarding that needed a floor of the dtype's smallest normal,
# chosen carefully enough to clamp subnormals without rewriting genuinely quiet audio.
#
# None of that is needed now. The metric works in the ENERGY domain and never takes a square root, so
# it clamps a mean square directly at RMS_EPS**2. The failure mode is not defended against, it is
# structurally absent - which is why the function goes rather than staying as an unused helper whose
# careful floor logic exists for a code path that no longer does.


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
