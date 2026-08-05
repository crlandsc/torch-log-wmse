"""Numerical parity against the original numpy implementation of logWMSE.

The rest of the suite checks internal consistency and closed-form identities. This module is the only
place that compares against an EXTERNAL reference, which is what makes it possible to claim this port
reproduces the metric rather than merely computing something self-consistent.

The reference is `log-wmse-audio-quality` (Iver Jordal / Nomono), the implementation this package is a
port of. It is an optional dependency: it needs scipy and soxr, so the whole module skips cleanly when
it is unavailable rather than failing.

    pip install log-wmse-audio-quality

Scope: parity is asserted at 44.1 kHz only. Below and above that the two implementations deliberately
diverge -- the original resamples the AUDIO to 44.1 kHz, this port resamples the IMPULSE RESPONSE to the
audio's rate -- so a parity assertion at other rates would encode the wrong expectation. See the README
section on sample rates.
"""
import math
import os
import sys

import numpy as np
import torch

# insert(0, ...) not append, so the working tree wins over any pip-installed copy of this package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from torch_log_wmse import LogWMSE

torch.set_num_threads(2)

try:
    from log_wmse_audio_quality import calculate_log_wmse as _upstream
except ImportError:  # pragma: no cover - exercised only when the optional dep is absent
    _upstream = None

SR = 44100
# Tolerance rationale: both implementations are float32 FFT convolutions, so they agree to a few
# float32 eps on the filtered signal. Measured worst case across the cases below is ~3.6e-07 on the
# filter output, which lands well inside 1e-4 on the log-domain metric.
TOL = 1e-4


def _torch_metric(unprocessed, processed, target, sample_rate=SR):
    """Run this port on numpy inputs shaped like upstream's, returning a float."""
    audio_length = unprocessed.shape[-1] / sample_rate
    m = LogWMSE(audio_length=audio_length, sample_rate=sample_rate, return_as_loss=False)
    u = torch.from_numpy(np.atleast_2d(unprocessed))
    p = torch.from_numpy(np.atleast_2d(processed))
    t = torch.from_numpy(np.atleast_2d(target))
    return float(m(u[None], p[None, :, None, :], t[None, :, None, :]))


@unittest.skipIf(_upstream is None, "log-wmse-audio-quality not installed (needs scipy + soxr)")
class TestUpstreamParity(unittest.TestCase):
    def _assert_parity(self, u, p, t, label):
        ref = float(_upstream(u, p, t, SR))
        got = _torch_metric(u, p, t)
        self.assertAlmostEqual(
            got, ref, delta=TOL,
            msg=f"{label}: this port {got!r} vs upstream {ref!r} (delta {abs(got - ref):.3e})")

    def test_scaled_input_against_silent_target(self):
        rng = np.random.default_rng(7)
        u = rng.uniform(-1, 1, (1, SR)).astype(np.float32)
        for k in (0.5, 0.1, 0.01):
            with self.subTest(k=k):
                self._assert_parity(u, (u * k).astype(np.float32), np.zeros_like(u), f"est={k}*input")

    def test_independent_signals(self):
        rng = np.random.default_rng(11)
        u = rng.uniform(-1, 1, (1, SR)).astype(np.float32)
        p = rng.uniform(-1, 1, (1, SR)).astype(np.float32)
        t = rng.uniform(-1, 1, (1, SR)).astype(np.float32)
        self._assert_parity(u, p, t, "independent est/target")

    def test_stereo(self):
        rng = np.random.default_rng(13)
        u = rng.uniform(-1, 1, (2, SR)).astype(np.float32)
        self._assert_parity(u, (u * 0.1).astype(np.float32), np.zeros_like(u), "stereo")

    def test_frequency_dependent_error(self):
        # A tone error exercises the weighting curve, so this would catch a filter that is subtly
        # wrong in shape even when broadband cases agree.
        rng = np.random.default_rng(17)
        u = rng.uniform(-1, 1, (1, SR)).astype(np.float32)
        t = (rng.uniform(-1, 1, (1, SR)) * 0.2).astype(np.float32)
        tt = np.arange(SR) / SR
        for freq in (60.0, 200.0, 1000.0, 3000.0, 7000.0):
            with self.subTest(freq=freq):
                p = (t + 0.01 * np.sin(2 * np.pi * freq * tt)).astype(np.float32)
                self._assert_parity(u, p, t, f"{freq:.0f} Hz tone error")

    def test_exact_match_and_all_silence(self):
        rng = np.random.default_rng(19)
        t = (rng.uniform(-1, 1, (1, SR)) * 0.1).astype(np.float32)
        self._assert_parity(t.copy(), t.copy(), t, "exact match")
        z = np.zeros((1, SR), dtype=np.float32)
        self._assert_parity(z, z.copy(), z.copy(), "all-silent triplet")

    def test_loud_and_quiet_inputs(self):
        rng = np.random.default_rng(23)
        base = rng.uniform(-1, 1, (1, SR)).astype(np.float32)
        for gain in (1e-3, 1e2):
            with self.subTest(gain=gain):
                u = (base * gain).astype(np.float32)
                self._assert_parity(u, (u * 0.1).astype(np.float32), np.zeros_like(u), f"gain={gain}")

    def test_upstream_documented_oracles(self):
        # The two values upstream's own test suite asserts, reproduced here as a cross-check that both
        # implementations agree with upstream's published expectations and not merely with each other.
        rng = np.random.default_rng(42)
        u = rng.uniform(-1, 1, (SR,)).astype(np.float32)
        est = (u * 0.1).astype(np.float32)
        self.assertAlmostEqual(_torch_metric(u, est, np.zeros_like(u)), 18.42, delta=0.01)
        z = np.zeros((SR,), dtype=np.float32)
        self.assertAlmostEqual(_torch_metric(z, z.copy(), z.copy()), 73.68, delta=0.01)


class TestDivergenceIsDeliberate(unittest.TestCase):
    """Below 44.1 kHz the two implementations are EXPECTED to differ, and this pins that expectation.

    If a future change made them agree at 16 kHz, that would mean the resampling strategy had changed,
    which is a decision that should be explicit rather than silent.
    """

    @unittest.skipIf(_upstream is None, "log-wmse-audio-quality not installed")
    def test_16k_diverges_from_upstream(self):
        sr = 16000
        rng = np.random.default_rng(29)
        u = rng.uniform(-1, 1, (1, sr)).astype(np.float32)
        t = (rng.uniform(-1, 1, (1, sr)) * 0.2).astype(np.float32)
        tt = np.arange(sr) / sr
        p = (t + 0.01 * np.sin(2 * np.pi * 7000.0 * tt)).astype(np.float32)
        ref = float(_upstream(u, p, t, sr))
        got = _torch_metric(u, p, t, sample_rate=sr)
        self.assertGreater(
            abs(got - ref), 0.1,
            "this port and upstream unexpectedly AGREE at 16 kHz; the resampling strategy may have "
            "changed. If that was intentional, update this test and the README.")

    def test_44k1_is_the_supported_rate(self):
        # Guards the assumption the parity tests above rest on: the bundled IR is designed at 44.1 kHz
        # and is used unresampled there.
        from torch_log_wmse.freq_weighting_filter import HumanHearingSensitivityFilter
        f = HumanHearingSensitivityFilter(audio_length=0.05, sample_rate=SR)
        self.assertEqual(f.impulse_response.shape[-1], 4000)


if __name__ == "__main__":
    unittest.main()
