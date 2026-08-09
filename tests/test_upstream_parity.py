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

import torch

# insert(0, ...) not append, so the working tree wins over any pip-installed copy of this package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from tests.conftest import SR, make_filter, make_metric

# numpy is imported HERE, not at module scope: it arrives with the reference package (via scipy) and is
# only ever used to talk to it. Importing it at module scope made this module fail COLLECTION in a bare
# environment instead of skipping, contradicting the docstring above.
try:
    import numpy as np

    from log_wmse_audio_quality import calculate_log_wmse as _upstream
except ImportError:  # pragma: no cover - exercised only when the optional dep is absent
    np = None
    _upstream = None

# Tolerance rationale: both implementations are float32 FFT convolutions, so they agree to a few
# float32 eps on the filtered signal. Measured worst case across the cases below is ~3.6e-07 on the
# filter output, which lands well inside 1e-4 on the log-domain metric.
TOL = 1e-4


def _as_batch(x):
    """numpy [channel, time] -> torch [1, channel, stem=1, time], and the mixture's [1, channel, time]."""
    return torch.from_numpy(np.atleast_2d(x))


def _torch_metric(unprocessed, processed, target, sample_rate=SR):
    """Run this port on numpy inputs shaped like upstream's, returning the POOLED float."""
    audio_length = unprocessed.shape[-1] / sample_rate
    m = make_metric(audio_length=audio_length, sample_rate=sample_rate)
    u, p, t = _as_batch(unprocessed), _as_batch(processed), _as_batch(target)
    return float(m(u[None], p[None, :, None, :], t[None, :, None, :]))


def _torch_per_element(unprocessed, processed, target, sample_rate=SR):
    """Per-[channel, stem] values from this port, as a nested list.

    `processed`/`target` may carry a stem axis: [channel, stem, time]. `unprocessed` is always
    [channel, time], since the mixture has no stems.

    This is the one place that reads unreduced values. When `reduction` narrows to the batch axis,
    this becomes `per_stem(...)` and nothing else in the module changes.
    """
    audio_length = unprocessed.shape[-1] / sample_rate
    m = make_metric(audio_length=audio_length, sample_rate=sample_rate, reduction="none")
    u = _as_batch(unprocessed)[None]
    p, t = torch.from_numpy(processed), torch.from_numpy(target)
    if p.ndim == 2:  # [channel, time] -> [channel, stem=1, time]
        p, t = p[:, None, :], t[:, None, :]
    return m(u, p[None], t[None])[0].tolist()  # [channel][stem]


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
        # NOTE: both channels get the same treatment, so this is blind to how channels COMBINE -
        # every aggregation rule passes it with delta exactly 0. TestUnequalChannelsAndStems below
        # is what actually covers that.
        rng = np.random.default_rng(13)
        u = rng.uniform(-1, 1, (2, SR)).astype(np.float32)
        self._assert_parity(u, (u * 0.1).astype(np.float32), np.zeros_like(u), "stereo")

    # Per-frequency tolerances, measured rather than guessed. A tone that starts and stops at the
    # buffer edges rings through the filter, and 1.0.0 counts that ring where upstream's trimmed
    # window discarded it. The effect is largest where the weighting attenuates hardest, because
    # the steady-state energy it is measured against is smallest there - 1.5e-02 at 60 Hz falling
    # to ~1e-04 above 1 kHz. See test_the_tone_divergence_is_an_edge_effect for the demonstration.
    #
    # These stay four orders of magnitude tighter than a genuinely wrong weighting curve, which
    # moves values by whole units, so the test keeps doing its job.
    TONE_TOL = {60.0: 3e-2, 200.0: 2e-3, 1000.0: 1e-3, 3000.0: 1e-3, 7000.0: 1e-3}

    def test_frequency_dependent_error(self):
        # A tone error exercises the weighting curve, so this would catch a filter that is subtly
        # wrong in shape even when broadband cases agree.
        rng = np.random.default_rng(17)
        u = rng.uniform(-1, 1, (1, SR)).astype(np.float32)
        t = (rng.uniform(-1, 1, (1, SR)) * 0.2).astype(np.float32)
        tt = np.arange(SR) / SR
        for freq, tol in self.TONE_TOL.items():
            with self.subTest(freq=freq):
                p = (t + 0.01 * np.sin(2 * np.pi * freq * tt)).astype(np.float32)
                ref, got = float(_upstream(u, p, t, SR)), _torch_metric(u, p, t)
                self.assertAlmostEqual(
                    got, ref, delta=tol,
                    msg=f"{freq:.0f} Hz tone: this port {got!r} vs upstream {ref!r} "
                        f"(delta {abs(got - ref):.3e}, allowed {tol:.0e})")

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


@unittest.skipIf(_upstream is None, "log-wmse-audio-quality not installed (needs scipy + soxr)")
class TestUnequalChannelsAndStems(unittest.TestCase):
    """How channels and stems COMBINE - the blind spot in every test above.

    `TestUpstreamParity.test_stereo` gives both channels the same treatment, so the two per-channel
    values are identical and EVERY aggregation rule reproduces upstream with delta exactly 0. That
    gap matters because aggregation is the one thing 1.0.0 deliberately changes.

    The durable anchor is the PER-ELEMENT comparison: each [channel, stem] value must equal upstream
    run on that pair alone. It survives the pooling change, because pooling consumes these values
    rather than producing them. The pooled comparison is the one that will legitimately move.
    """

    # A distinct residual level per (channel, stem), spanning 30 dB, so no two elements agree.
    LEVELS = ((0.316, 0.1, 0.0316), (0.0562, 0.0178, 0.01))

    def _case(self, seed=101, stems=3):
        rng = np.random.default_rng(seed)
        u = rng.uniform(-1, 1, (2, SR)).astype(np.float32)
        levels = np.array([row[:stems] for row in self.LEVELS], dtype=np.float32)
        p = (u[:, None, :] * levels[:, :, None]).astype(np.float32)
        t = np.zeros_like(p)
        return u, p, t

    def _upstream_element(self, u, p, t, c, s):
        """Upstream on one (channel, stem) pair, as mono."""
        return float(_upstream(u[c : c + 1], p[c, s][None], t[c, s][None], SR))

    def test_per_element_values_match_upstream(self):
        """THE fidelity anchor. Must still hold after the pooling rule changes."""
        u, p, t = self._case()
        got = _torch_per_element(u, p, t)
        for c in range(p.shape[0]):
            for s in range(p.shape[1]):
                with self.subTest(channel=c, stem=s):
                    ref = self._upstream_element(u, p, t, c, s)
                    self.assertAlmostEqual(
                        got[c][s], ref, delta=TOL,
                        msg=f"element [{c},{s}]: this port {got[c][s]!r} vs upstream {ref!r}")

    def test_elements_are_actually_unequal(self):
        """Guards the guard: if the levels ever collapse, this class silently stops testing anything."""
        u, p, t = self._case()
        flat = [v for row in _torch_per_element(u, p, t) for v in row]
        self.assertGreater(max(flat) - min(flat), 20.0,
                           f"per-element spread collapsed to {max(flat) - min(flat):.2f}; the case is "
                           "no longer exercising aggregation")

    def test_upstream_pools_channels_as_the_mean_of_logs(self):
        """Pins WHAT upstream does across channels, so our divergence from it stays a decision.

        Measured exact (delta 0.00e+00): upstream's stereo value is the arithmetic mean of its two
        mono values, i.e. mean-of-logs. That is the rule 1.0.0 replaces with a power mean.
        """
        u, p, t = self._case(stems=1)
        pooled = float(_upstream(u, p[:, 0], t[:, 0], SR))
        per_channel = [self._upstream_element(u, p, t, c, 0) for c in range(2)]
        self.assertAlmostEqual(pooled, sum(per_channel) / len(per_channel), delta=TOL)
        # And it is emphatically NOT a log of the mean MSE, which would read ~6.5 units lower here.
        mean_mse = sum(math.exp(v / -4.0) for v in per_channel) / len(per_channel)
        self.assertGreater(abs(pooled - (-4.0 * math.log(mean_mse))), 1.0)

    def test_pooled_value_matches_upstream_for_unequal_channels(self):
        """True while pooling is mean-of-logs (p = 0).

        EXPECTED TO CHANGE when the default p moves to 1/2: this is the assertion that proves the
        aggregation rule actually moved, so update it deliberately with the measured delta rather
        than deleting it.
        """
        u, p, t = self._case(stems=1)
        ref = float(_upstream(u, p[:, 0], t[:, 0], SR))
        got = _torch_metric(u, p[:, 0], t[:, 0])
        self.assertAlmostEqual(got, ref, delta=TOL,
                               msg=f"unequal-channel pooled: {got!r} vs upstream {ref!r}")


class TestDivergenceIsDeliberate(unittest.TestCase):
    """Below 44.1 kHz the two implementations are EXPECTED to differ, and this pins that expectation.

    If a future change made them agree at 16 kHz, that would mean the resampling strategy had changed,
    which is a decision that should be explicit rather than silent.
    """

    @unittest.skipIf(_upstream is None, "log-wmse-audio-quality not installed")
    def test_the_tone_divergence_is_an_edge_effect(self):
        """Demonstrates WHY tone errors need looser tolerances than broadband ones.

        1.0.0 scores the energy of the full linear convolution; upstream scores a trimmed window
        the same length as the input. The two differ by exactly the filter's ring-in and ring-out
        at the buffer edges - so confining the residual away from those edges must make the
        difference collapse, and it does: 1.5e-02 to 1.2e-03 at 60 Hz, 8.0e-04 to 8.8e-05 at
        200 Hz. Above 1 kHz both sit at the ~1e-04 float32 floor already.

        If this ever stops shrinking, the divergence is NOT the trim and something else is wrong.
        """
        rng = np.random.default_rng(17)
        u = rng.uniform(-1, 1, (1, SR)).astype(np.float32)
        t = (rng.uniform(-1, 1, (1, SR)) * 0.2).astype(np.float32)
        tt = np.arange(SR) / SR
        guard = np.ones(SR, dtype=np.float32)
        guard[:4100] = 0  # a little over the 4000-tap impulse response, at both ends
        guard[-4100:] = 0

        for freq in (60.0, 200.0):
            with self.subTest(freq=freq):
                tone = 0.01 * np.sin(2 * np.pi * freq * tt)
                to_edges = (t + tone).astype(np.float32)
                off_edges = (t + tone * guard).astype(np.float32)
                wide = abs(_torch_metric(u, to_edges, t) - float(_upstream(u, to_edges, t, SR)))
                narrow = abs(_torch_metric(u, off_edges, t) - float(_upstream(u, off_edges, t, SR)))
                self.assertLess(narrow * 5, wide,
                                f"{freq:.0f} Hz: keeping the residual off the edges barely helped "
                                f"({wide:.3e} -> {narrow:.3e}), so the divergence is not the trim")

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
        f = make_filter(audio_length=0.05, sample_rate=SR)
        self.assertEqual(f.impulse_response.shape[-1], 4000)


if __name__ == "__main__":
    unittest.main()
