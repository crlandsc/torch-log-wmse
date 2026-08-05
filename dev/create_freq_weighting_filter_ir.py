"""Regenerate the bundled hearing-sensitivity impulse response.

VENDORED, WITH MODIFICATIONS, from nomonosound/log-wmse-audio-quality
(Copyright 2023 Nomono, Apache License 2.0). Modified by Whitebalance LLC: writes
torch_log_wmse/filter_ir.f32 as raw little-endian float32 instead of a pickle, and prints the
SHA-256 to paste into freq_weighting_filter.py.

This exists for PROVENANCE. Previously the filter shipped as an opaque binary with no generator in
the repo, so it could not be regenerated or independently reviewed. It is a dev-time script only and
is NOT part of the installed package -- it needs audiomentations and scipy, which are not runtime
dependencies:

    pip install "audiomentations>=0.31.0" scipy
    python dev/create_freq_weighting_filter_ir.py

Design credits carried over from upstream: the filter set is "Inspired by Fenton & Lee (2017)", and
the zero-phase-equivalent construction is credited to mmxgn (2023).

Regenerating will NOT reproduce the shipped bytes exactly. Current scipy preserves float32 through
freqz where the original produced float64, capping agreement at about 1.6e-08 -- measured during the
audit. Treat a mismatch at that scale as expected; anything larger means something changed.
"""
import hashlib
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.signal
from audiomentations import (
    Compose,
    HighPassFilter,
    HighShelfFilter,
    PeakingFilter,
    LowPassFilter,
)
from numpy.typing import NDArray

# N_FFT and INTERNAL_SAMPLE_RATE were removed from the package (N_FFT was dead code
# there); they live here now, where they are actually used.
N_FFT = 4096
INTERNAL_SAMPLE_RATE = 44100


def get_human_hearing_sensitivity_filter_set() -> (
    Callable[[NDArray[np.float32], int], NDArray[np.float32]]
):
    """Compose a set of filters that together form a frequency response that resembles
    human hearing sensitivity to different frequencies. Inspired by Fenton & Lee (2017).
    """
    return Compose(
        [
            HighShelfFilter(
                min_center_freq=1500.0,
                max_center_freq=1500.0,
                min_gain_db=5.0,
                max_gain_db=5.0,
                min_q=1 / np.sqrt(2),
                max_q=1 / np.sqrt(2),
                p=1.0,
            ),
            HighPassFilter(
                min_cutoff_freq=120.0,
                max_cutoff_freq=120.0,
                min_rolloff=12,
                max_rolloff=12,
                zero_phase=False,
                p=1.0,
            ),
            PeakingFilter(
                min_center_freq=500.0,
                max_center_freq=500.0,
                min_gain_db=2.5,
                max_gain_db=2.5,
                min_q=1.5 / np.sqrt(2),
                max_q=1.5 / np.sqrt(2),
                p=1.0,
            ),
            LowPassFilter(
                min_cutoff_freq=10_000,
                max_cutoff_freq=10_000,
                min_rolloff=12,
                max_rolloff=12,
                zero_phase=True,
                p=1.0,
            ),
        ]
    )


def get_zero_phase_equivalent_filter_impulse_response(
    filter_func: Callable[[NDArray[np.float32], int], NDArray[np.float32]],
    sample_rate: int,
    n_fft: int = 2 * N_FFT,
) -> NDArray:
    """Extract the target response from the given filter_func (which may be not
    zero-phase). The idea is to construct a zero-phase equivalent of the given
    filter_func. Credits: mmxgn (2023)"""
    # Get the impulse response of the filter
    delta = np.zeros(n_fft, dtype=np.float32)
    delta[len(delta) // 2] = 1.0
    impulse_response = filter_func(delta, sample_rate)

    w, h = scipy.signal.freqz(impulse_response, worN=n_fft // 2 + 1)
    linear_target_response = np.abs(h)

    # Compute impulse response
    impulse_response = np.fft.fftshift(np.fft.irfft(linear_target_response, n_fft))

    # Make it symmetric
    center_sample = len(impulse_response) // 2 + 1
    return impulse_response[center_sample - 2000 : center_sample + 2000]


if __name__ == "__main__":
    """Calculate and write the filter impulse response"""
    ir = get_zero_phase_equivalent_filter_impulse_response(
        get_human_hearing_sensitivity_filter_set(), sample_rate=INTERNAL_SAMPLE_RATE
    )
    here = Path(os.path.abspath(os.path.dirname(__file__)))
    package_dir = here.parent / "torch_log_wmse"
    shipped = package_dir / "filter_ir.f32"
    payload = np.asarray(ir, dtype="<f4").tobytes()
    digest = hashlib.sha256(payload).hexdigest()

    # Write beside the script, NOT over the shipped resource. Overwriting it would invalidate the
    # SHA-256 pinned in freq_weighting_filter.py, so every later LogWMSE(...) would raise an
    # integrity error -- and it would destroy the very bytes you regenerated in order to compare.
    out = here / "filter_ir.regenerated.f32"
    out.write_bytes(payload)
    print(f"wrote {out} ({len(payload)} bytes, {len(payload) // 4} float32 taps)")
    print(f"regenerated sha256 = {digest}")

    if shipped.exists():
        current = shipped.read_bytes()
        print(f"shipped      sha256 = {hashlib.sha256(current).hexdigest()}")
        if len(current) == len(payload):
            a = np.frombuffer(current, dtype="<f4")
            b = np.frombuffer(payload, dtype="<f4")
            print(f"max|regenerated - shipped| = {np.abs(a - b).max():.3e}  "
                  f"(expect ~1.6e-08; see the module docstring)")
        else:
            print(f"length differs: shipped {len(current)} bytes vs regenerated {len(payload)}")

    if "--overwrite" in sys.argv:
        shipped.write_bytes(payload)
        print(f"\nOVERWROTE {shipped}")
        print(f"Now update _IR_SHA256 in torch_log_wmse/freq_weighting_filter.py to:\n  {digest}")
    else:
        print("\nThe shipped resource was NOT modified. Pass --overwrite to replace it, and then "
              "update _IR_SHA256 in torch_log_wmse/freq_weighting_filter.py to the regenerated digest.")
