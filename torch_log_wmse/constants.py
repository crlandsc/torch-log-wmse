"""Constants for the logWMSE metric.

Derived from nomonosound/log-wmse-audio-quality (Copyright 2023 Nomono), licensed under the
Apache License 2.0. Modified by Whitebalance LLC in 2024-2026: ported from numpy to PyTorch,
FFT convolution in place of scipy.signal.oaconvolve, batched [batch, channel, stem, time] tensor
API, differentiable loss support, and the impulse-response/silence handling described in the
README and CHANGELOG.
"""
from torch_log_wmse.utils import convert_decibels_to_amplitude_ratio

# Error tolerance threshold, relative to 0 dB RMS
ERROR_TOLERANCE_THRESHOLD = convert_decibels_to_amplitude_ratio(-68.0)

# This scaler makes the scale of values closer to SDR, where an increase
# in the tenths place is a meaningful improvement. The goal is to make it easier to
# compare numbers at a glance, e.g. when numbers are presented in a table.
SCALER = -4.0

# Small constant to avoid taking log of zero
EPS = 1e-8

# Small constant to avoid division by zero in RMS scaling_factor calculation
RMS_EPS = 1e-8
