"""Invariant and regression tests for logWMSE.

These complement tests/test_metric.py, which asserts mostly shapes and types. Every test here pins a
*behaviour* and is designed to fail if a specific property regresses. Where a test exists to kill a known
mutation, the mutation is named in the docstring.

Design notes:
* No matplotlib import, so this module runs without a plotting stack.
* Oracles are closed-form wherever possible rather than golden values recorded from this implementation,
  so they stay valid across torch versions and FFT round-off changes.
* Threads are capped so the suite stays polite on a shared machine.
"""
import math
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from torch_log_wmse import LogWMSE
from torch_log_wmse.constants import EPS, ERROR_TOLERANCE_THRESHOLD, SCALER
from torch_log_wmse.freq_weighting_filter import HumanHearingSensitivityFilter

torch.set_num_threads(2)

SR = 44100
CEILING = float(SCALER * math.log(EPS))  # +73.6827..., the value for an exact match


def _metric(audio_length=1.0, sample_rate=SR, **kw):
    kw.setdefault("return_as_loss", False)
    return LogWMSE(audio_length=audio_length, sample_rate=sample_rate, **kw)


class TestClosedFormOracle(unittest.TestCase):
    """The metric has an exact analytic value for est = k * unprocessed against a silent target.

    differences = filters(k*u*s) and s = 1/rms(filters(u)), so mse = k^2 exactly and the metric is
    SCALER*ln(k^2 + EPS). This is derivation-backed, unlike a golden value recorded from this
    implementation.

    The closed form holds only while k stays well clear of ERROR_TOLERANCE_THRESHOLD (3.98e-4): the
    scaled differences have RMS exactly k, so once k approaches the threshold a growing fraction of
    samples is legitimately zeroed and the measured mse drops below k^2. At k=1e-3 the threshold is
    0.4 * RMS and roughly a third of the samples are discounted, so the oracle is asserted only for
    k >= 1e-2, which is ~25x the threshold.
    """

    def test_scaled_input_against_silent_target(self):
        torch.manual_seed(0)
        m = _metric()
        u = (torch.rand(1, 1, SR) * 2 - 1)
        for k in (0.5, 0.25, 0.1, 0.05, 0.01):
            with self.subTest(k=k):
                p = (u * k).unsqueeze(2)
                t = torch.zeros_like(p)
                expected = SCALER * math.log(k * k + EPS)
                self.assertAlmostEqual(float(m(u, p, t)), expected, places=3)

    def test_threshold_discounts_error_below_its_level(self):
        # The complement of the oracle above: well below the threshold the metric must read BETTER than
        # the closed form, because sub-threshold samples are deliberately discounted. Measured offsets
        # from the closed form are +0.06 at k=1e-3, +0.82 at k=4e-4 and +2.77 at k=1e-4, so k=1e-4 gives
        # an unambiguous signal. (The offset stays small until k is comparable to the threshold because
        # the discarded samples are the low-ENERGY ones, not merely the numerous ones.)
        torch.manual_seed(0)
        m = _metric()
        u = (torch.rand(1, 1, SR) * 2 - 1)
        k = 1e-4
        p = (u * k).unsqueeze(2)
        got = float(m(u, p, torch.zeros_like(p)))
        naive = SCALER * math.log(k * k + EPS)
        self.assertGreater(got, naive + 1.0)

    def test_exact_match_hits_the_ceiling(self):
        torch.manual_seed(1)
        m = _metric()
        u = (torch.rand(1, 1, SR) * 2 - 1)
        t = (torch.rand(1, 1, 1, SR) * 2 - 1) * 0.3
        self.assertAlmostEqual(float(m(u, t.clone(), t)), CEILING, places=4)

    def test_ceiling_matches_the_constants(self):
        self.assertAlmostEqual(CEILING, 73.68272, places=4)


class TestFilterIsZeroPhase(unittest.TestCase):
    """A symmetric delta impulse response must pass a signal through unchanged.

    Kills M07 (shift removed), M08/M09 (shift +/-1 sample) and M20 (asymmetric padding), and would have
    caught the odd-length-IR misalignment (finding A2) that the original suite could not detect.
    """

    def _roundtrip_offset(self, ir_len, n=512):
        ir = torch.zeros(ir_len)
        ir[(ir_len - 1) // 2] = 1.0
        f = HumanHearingSensitivityFilter(
            audio_length=n / SR, sample_rate=SR,
            impulse_response=ir, impulse_response_sample_rate=SR)
        x = torch.zeros(1, 1, 1, n)
        x[0, 0, 0, n // 2] = 1.0
        y = f(x)
        return int(torch.argmax(torch.abs(y))) - n // 2, float((y - x[0, 0, 0]).abs().max())

    def test_delta_ir_is_transparent_for_even_and_odd_lengths(self):
        # Odd lengths are the regression guard for A2; even lengths cover the shipped 4000-tap IR.
        for ir_len in (2, 3, 4, 51, 100, 101, 999, 1000, 1001):
            with self.subTest(ir_len=ir_len):
                offset, err = self._roundtrip_offset(ir_len)
                self.assertEqual(offset, 0, f"delta IR of length {ir_len} shifted the signal by {offset}")
                self.assertLess(err, 1e-5)

    def test_builtin_filter_preserves_length_and_is_finite(self):
        f = HumanHearingSensitivityFilter(audio_length=0.05, sample_rate=SR)
        x = torch.randn(2, 2, 3, int(0.05 * SR))
        y = f(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())


class TestErrorToleranceThreshold(unittest.TestCase):
    """The -68 dB tolerance must actually discount sub-threshold error.

    Kills M24 (tolerance zeroing deleted) and M06 (threshold changed), which the original suite let through.
    """

    def test_uniform_subthreshold_error_reaches_the_ceiling(self):
        n = 4096
        m = _metric(audio_length=n / SR, bypass_filter=True)
        u = torch.ones(1, 1, n)
        t = torch.zeros(1, 1, 1, n)
        below = torch.full((1, 1, 1, n), float(ERROR_TOLERANCE_THRESHOLD) * 0.5)
        self.assertAlmostEqual(float(m(u, below, t)), CEILING, places=4)

    def test_error_exactly_at_threshold_is_not_discounted(self):
        # The comparison is a strict `<`, so a difference equal to the threshold must survive.
        n = 4096
        m = _metric(audio_length=n / SR, bypass_filter=True)
        u = torch.ones(1, 1, n)
        t = torch.zeros(1, 1, 1, n)
        at = torch.full((1, 1, 1, n), float(ERROR_TOLERANCE_THRESHOLD))
        self.assertLess(float(m(u, at, t)), CEILING - 1.0)

    def test_threshold_constant_is_minus_68_db(self):
        self.assertAlmostEqual(float(ERROR_TOLERANCE_THRESHOLD), 10 ** (-68.0 / 20), places=9)


class TestBypassFilter(unittest.TestCase):
    """bypass_filter=True must actually change the computation. Kills M23 (flag ignored)."""

    def test_bypass_differs_from_filtered(self):
        torch.manual_seed(2)
        n = SR
        u = (torch.rand(1, 1, n) * 2 - 1)
        p = (torch.rand(1, 1, 1, n) * 2 - 1) * 0.1
        t = torch.zeros_like(p)
        filtered = float(_metric(audio_length=n / SR)(u, p, t))
        bypassed = float(_metric(audio_length=n / SR, bypass_filter=True)(u, p, t))
        self.assertNotAlmostEqual(filtered, bypassed, places=3)

    def test_bypass_is_unaffected_by_the_impulse_response(self):
        # With the filter bypassed, a different IR must make no difference at all.
        torch.manual_seed(3)
        n = 4096
        u = (torch.rand(1, 1, n) * 2 - 1)
        p = (torch.rand(1, 1, 1, n) * 2 - 1) * 0.1
        t = torch.zeros_like(p)
        a = _metric(audio_length=n / SR, bypass_filter=True)
        b = _metric(audio_length=n / SR, bypass_filter=True,
                    impulse_response=torch.rand(2048), impulse_response_sample_rate=SR)
        self.assertAlmostEqual(float(a(u, p, t)), float(b(u, p, t)), places=5)


class TestSampleRateHandling(unittest.TestCase):
    """Non-44.1 kHz rates must exercise the resampling path. Covers M27 (resampling skipped),
    which had zero statement coverage in the original suite."""

    def test_resampled_impulse_response_length_tracks_the_rate(self):
        base = HumanHearingSensitivityFilter(audio_length=0.05, sample_rate=SR)
        n_base = base.impulse_response.shape[-1]
        for sr in (16000, 22050, 48000):
            with self.subTest(sample_rate=sr):
                f = HumanHearingSensitivityFilter(audio_length=0.05, sample_rate=sr)
                expected = round(n_base * sr / SR)
                self.assertAlmostEqual(f.impulse_response.shape[-1], expected, delta=4)

    def test_metric_is_finite_and_ordered_across_rates(self):
        for sr in (16000, 22050, 44100, 48000):
            with self.subTest(sample_rate=sr):
                m = _metric(audio_length=1.0, sample_rate=sr)
                torch.manual_seed(4)
                u = (torch.rand(1, 1, sr) * 2 - 1)
                t = torch.zeros(1, 1, 1, sr)
                good = float(m(u, (u * 0.01).unsqueeze(2), t))
                bad = float(m(u, (u * 0.5).unsqueeze(2), t))
                self.assertTrue(math.isfinite(good) and math.isfinite(bad))
                self.assertGreater(good, bad, "a quieter residual must score better")


class TestScaleInvariance(unittest.TestCase):
    """Gaining all three inputs by the same factor must not change the value."""

    def test_value_is_invariant_across_gain_decades(self):
        torch.manual_seed(5)
        n = SR
        m = _metric()
        u = (torch.rand(1, 1, n) * 2 - 1)
        t = (torch.rand(1, 1, 1, n) * 2 - 1) * 0.3
        p = t + torch.randn_like(t) * 0.01
        ref = float(m(u, p, t))
        for g in (1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3):
            with self.subTest(gain=g):
                self.assertAlmostEqual(float(m(u * g, p * g, t * g)), ref, places=3)


class TestNoSideEffects(unittest.TestCase):
    """The metric must not mutate its inputs. Upstream has this test; this port did not."""

    def test_inputs_are_unchanged(self):
        torch.manual_seed(6)
        n = SR
        u = (torch.rand(1, 1, n) * 2 - 1)
        p = (torch.rand(1, 1, 1, n) * 2 - 1) * 0.1
        t = torch.zeros_like(p)
        u0, p0, t0 = u.clone(), p.clone(), t.clone()
        for kw in ({}, {"bypass_filter": True}, {"reduction": "none"}, {"reduction": "sum"}):
            with self.subTest(**kw):
                _metric(**kw)(u, p, t)
                self.assertTrue(torch.equal(u, u0))
                self.assertTrue(torch.equal(p, p0))
                self.assertTrue(torch.equal(t, t0))

    def test_repeated_calls_are_deterministic(self):
        torch.manual_seed(7)
        n = 8192
        m = _metric(audio_length=n / SR)
        u = (torch.rand(1, 1, n) * 2 - 1)
        p = (torch.rand(1, 1, 1, n) * 2 - 1) * 0.1
        t = torch.zeros_like(p)
        first = float(m(u, p, t))
        for _ in range(3):
            self.assertEqual(float(m(u, p, t)), first)


class TestReductionSemantics(unittest.TestCase):
    """Reduction must pin the correct axis and relate mean/sum/none exactly.

    Kills M21 (mean over wrong dim) and M13 (mean/sum swapped).
    """

    def test_none_mean_sum_are_mutually_consistent(self):
        torch.manual_seed(8)
        b, c, s, n = 2, 2, 3, 8192
        u = (torch.rand(b, c, n) * 2 - 1)
        p = (torch.rand(b, c, s, n) * 2 - 1) * 0.1
        t = torch.zeros_like(p)
        none = _metric(audio_length=n / SR, reduction="none")(u, p, t)
        mean = float(_metric(audio_length=n / SR, reduction="mean")(u, p, t))
        total = float(_metric(audio_length=n / SR, reduction="sum")(u, p, t))
        self.assertEqual(none.shape, (b, c, s))
        self.assertAlmostEqual(float(none.mean()), mean, places=4)
        self.assertAlmostEqual(float(none.sum()), total, places=2)

    def test_per_entry_values_survive_batch_reduction(self):
        # The original suite asserted values only for batch size 1; this pins multi-entry reduction.
        torch.manual_seed(9)
        n = 8192
        u = (torch.rand(2, 1, n) * 2 - 1)
        t = torch.zeros(2, 1, 1, n)
        p = torch.stack([u[0] * 0.1, u[1] * 0.01]).unsqueeze(2)
        none = _metric(audio_length=n / SR, reduction="none")(u, p, t).flatten()
        self.assertAlmostEqual(float(none[0]), SCALER * math.log(0.1 ** 2), places=2)
        self.assertAlmostEqual(float(none[1]), SCALER * math.log(0.01 ** 2), places=2)
        mean = float(_metric(audio_length=n / SR, reduction="mean")(u, p, t))
        self.assertAlmostEqual(mean, float(none.mean()), places=4)


class TestLossOrientation(unittest.TestCase):
    """return_as_loss must negate, and the loss must decrease as the estimate improves.

    Kills M01 (SCALER sign flip) and M19 (negation removed).
    """

    def test_loss_is_the_negated_metric(self):
        torch.manual_seed(10)
        n = 8192
        u = (torch.rand(1, 1, n) * 2 - 1)
        p = (torch.rand(1, 1, 1, n) * 2 - 1) * 0.1
        t = torch.zeros_like(p)
        metric = float(_metric(audio_length=n / SR, return_as_loss=False)(u, p, t))
        loss = float(_metric(audio_length=n / SR, return_as_loss=True)(u, p, t))
        self.assertAlmostEqual(loss, -metric, places=5)

    def test_loss_decreases_as_the_estimate_improves(self):
        torch.manual_seed(11)
        n = 8192
        m = _metric(audio_length=n / SR, return_as_loss=True)
        u = (torch.rand(1, 1, n) * 2 - 1)
        t = (torch.rand(1, 1, 1, n) * 2 - 1) * 0.3
        worse = float(m(u, t + torch.randn_like(t) * 0.1, t))
        better = float(m(u, t + torch.randn_like(t) * 0.001, t))
        self.assertLess(better, worse)


class TestGradients(unittest.TestCase):
    """The loss must be differentiable end to end and produce finite, non-zero gradients."""

    def test_gradient_flows_to_the_estimate(self):
        torch.manual_seed(12)
        n = 4096
        m = _metric(audio_length=n / SR, return_as_loss=True)
        u = (torch.rand(1, 1, n) * 2 - 1)
        t = (torch.rand(1, 1, 1, n) * 2 - 1) * 0.3
        p = (t + torch.randn_like(t) * 0.05).requires_grad_(True)
        m(u, p, t).backward()
        self.assertIsNotNone(p.grad)
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad.norm()), 0.0)

    def test_gradcheck_in_float64(self):
        torch.manual_seed(13)
        n = 256
        m = _metric(audio_length=n / SR, sample_rate=4096, return_as_loss=True)
        u = (torch.rand(1, 1, n, dtype=torch.float64) * 2 - 1)
        t = (torch.rand(1, 1, 1, n, dtype=torch.float64) * 2 - 1)
        p = (torch.rand(1, 1, 1, n, dtype=torch.float64) * 2 - 1).requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(lambda x: m(u, x, t), (p,), eps=1e-6, atol=1e-4, rtol=1e-3))


class TestDigitalSilence(unittest.TestCase):
    """Digital silence must not produce NaN anywhere. Regression guard for the 0.2.8 bug and M14."""

    def test_all_silent_triplet_is_the_ceiling(self):
        n = 8192
        m = _metric(audio_length=n / SR)
        z3, z4 = torch.zeros(1, 1, n), torch.zeros(1, 1, 1, n)
        self.assertAlmostEqual(float(m(z3, z4, z4)), CEILING, places=4)

    def test_silent_entry_mixed_into_a_batch_is_not_nan(self):
        torch.manual_seed(14)
        n = 8192
        m = _metric(audio_length=n / SR, reduction="none")
        u = (torch.rand(2, 1, n) * 2 - 1)
        u[0] = 0.0
        p = (torch.rand(2, 1, 1, n) * 2 - 1) * 0.1
        p[0] = 0.0
        t = torch.zeros(2, 1, 1, n)
        out = m(u, p, t)
        self.assertTrue(torch.isfinite(out).all())
        self.assertAlmostEqual(float(out.flatten()[0]), CEILING, places=4)


if __name__ == "__main__":
    unittest.main()


class TestImpulseResponseValidation(unittest.TestCase):
    """A malformed impulse response must raise, not silently corrupt every result.

    Upstream gets this guard for free: scipy.signal.oaconvolve raises
    "in1 and in2 should have the same dimensionality" for a 2-D IR. The FFT-broadcast port lost it, so a
    2-D IR was silently expanded against the STEM axis and an all-zero IR made the metric report its
    ceiling for every input.
    """

    def _build(self, ir):
        return HumanHearingSensitivityFilter(
            audio_length=0.05, sample_rate=SR, impulse_response=ir, impulse_response_sample_rate=SR)

    def test_multichannel_ir_is_rejected(self):
        with self.assertRaises(ValueError):
            self._build(torch.rand(2, 4000))

    def test_all_zero_ir_is_rejected(self):
        with self.assertRaises(ValueError):
            self._build(torch.zeros(4000))

    def test_non_finite_ir_is_rejected(self):
        bad = torch.rand(4000)
        bad[7] = float("nan")
        with self.assertRaises(ValueError):
            self._build(bad)
        bad[7] = float("inf")
        with self.assertRaises(ValueError):
            self._build(bad)

    def test_degenerate_length_is_rejected(self):
        for ir in (torch.ones(1), torch.tensor(1.0)):
            with self.subTest(shape=tuple(ir.shape)):
                with self.assertRaises(ValueError):
                    self._build(ir)

    def test_singleton_dimensions_are_accepted(self):
        # [1, N] and [N, 1] are ordinary ways to hold a mono FIR and must still work.
        for ir in (torch.rand(1, 4000), torch.rand(4000, 1)):
            with self.subTest(shape=tuple(ir.shape)):
                f = self._build(ir)
                self.assertEqual(f.impulse_response.ndim, 1)
                self.assertEqual(f.impulse_response.numel(), 4000)

    def test_valid_ir_still_works_end_to_end(self):
        ir = torch.zeros(999)
        ir[499] = 1.0
        m = _metric(audio_length=0.05, impulse_response=ir, impulse_response_sample_rate=SR)
        n = int(0.05 * SR)
        torch.manual_seed(15)
        u = (torch.rand(1, 1, n) * 2 - 1)
        p = (torch.rand(1, 1, 1, n) * 2 - 1) * 0.1
        self.assertTrue(math.isfinite(float(m(u, p, torch.zeros_like(p)))))
