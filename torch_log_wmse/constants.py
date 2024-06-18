from torch_log_wmse.utils import convert_decibels_to_amplitude_ratio

# FFT size for human hearing sensitivity filter impulse response calculation
N_FFT = 4096

# Error tolerance threshold, relative to 0 dB RMS
ERROR_TOLERANCE_THRESHOLD = convert_decibels_to_amplitude_ratio(-68.0)

# This scaler makes the scale of values closer to SDR, where an increase
# in the tenths place is a meaningful improvement. The goal is to make it easier to
# compare numbers at a glance, e.g. when numbers are presented in a table.
SCALER = -4.0

# Small constant to avoid taking log of zero
EPS = 1e-8