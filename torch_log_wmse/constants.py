"""Constants for the logWMSE metric.

Derived from nomonosound/log-wmse-audio-quality (Copyright 2023 Nomono), licensed under the
Apache License 2.0. Modified by Christopher Landschoot (SoundFoxLabs) in 2024-2026: ported from
numpy to PyTorch; batched [batch, channel, stem, time] tensor API; differentiable loss support; the
weighted error energy computed in the frequency domain via one-sided Parseval rather than by
convolving with scipy.signal.oaconvolve; power-mean aggregation across channel and stem in place of
the mean of logs; the -68 dB per-sample inaudibility gate removed; and the impulse-response and
silence handling described in README.md and CHANGELOG.md.
"""
# ERROR_TOLERANCE_THRESHOLD (a -68 dB per-sample inaudibility gate) was removed in 1.0.0. It could
# not survive the move to computing energy in the frequency domain - a per-SAMPLE gate needs a
# time-domain signal, and there no longer is one. Its measured effect across the reachable range was
# 0.000: identical values from -10 dB down to -40 dB residual, and reaching the band where it
# mattered at all needs 74-80 dB SI-SDR. Exact digital silence still returns the ceiling either way,
# so the headline feature never depended on it. Removing it also deletes a zero-gradient dead zone.

# This scaler makes the scale of values closer to SDR, where an increase
# in the tenths place is a meaningful improvement. The goal is to make it easier to
# compare numbers at a glance, e.g. when numbers are presented in a table.
SCALER = -4.0

# Small constant to avoid taking log of zero
EPS = 1e-8

# Small constant to avoid division by zero in RMS scaling_factor calculation
RMS_EPS = 1e-8
