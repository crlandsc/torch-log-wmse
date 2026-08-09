"""Invariant and regression tests for logWMSE.

These complement tests/test_metric.py, which asserts mostly shapes and types. Every test here pins a
*behaviour* and is designed to fail if a specific property regresses. Where a test exists to kill a known
mutation, the mutation is named in the docstring.

Design notes:
* No matplotlib import, so this module runs without a plotting stack.
* Oracles are closed-form wherever possible rather than golden values recorded from this implementation,
  so they stay valid across torch versions and FFT round-off changes.
* Construction goes through tests/conftest.py, which also caps threads and owns the sys.path order.
"""
import math
import os
import sys

import torch

# insert(0, ...) not append: with append, a pip-installed copy of this package in
# site-packages shadows the working tree when this file is run directly
# (python tests/test_x.py puts tests/ on sys.path[0], not the repo root), so the
# suite would silently test the installed wheel instead of the code under edit.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

# conftest owns the sys.path insertion, the thread cap, SR/CEILING, and every construction of the
# metric and the filter. Constructing through it is what keeps the 1.0.0 constructor changes to one
# edit rather than forty.
from tests.conftest import CEILING, SR, make_filter, make_metric, per_element
from torch_log_wmse.constants import EPS, ERROR_TOLERANCE_THRESHOLD, SCALER

# Module-local alias, so the call sites below stay as they are.
_metric = make_metric


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
        f = make_filter(
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
        f = make_filter(audio_length=0.05, sample_rate=SR)
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
        base = make_filter(audio_length=0.05, sample_rate=SR)
        n_base = base.impulse_response.shape[-1]
        for sr in (16000, 22050, 48000):
            with self.subTest(sample_rate=sr):
                f = make_filter(audio_length=0.05, sample_rate=sr)
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
        # Includes 1e-5 and 1e-4, where an ADDITIVE rms epsilon biased the result (+0.0014 at 1e-4).
        # RMS_EPS is a floor, so invariance is exact to float32 noise across every decade here.
        for g in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3):
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
        # audio_length must be consistent with sample_rate, or the length validation rejects the input.
        sr = 4096
        m = _metric(audio_length=n / sr, sample_rate=sr, return_as_loss=True)
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



class TestImpulseResponseValidation(unittest.TestCase):
    """A malformed impulse response must raise, not silently corrupt every result.

    Upstream gets this guard for free: scipy.signal.oaconvolve raises
    "in1 and in2 should have the same dimensionality" for a 2-D IR. The FFT-broadcast port lost it, so a
    2-D IR was silently expanded against the STEM axis and an all-zero IR made the metric report its
    ceiling for every input.
    """

    def _build(self, ir):
        return make_filter(
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

class TestSilenceGradients(unittest.TestCase):
    """A digitally silent, grad-requiring input must not poison the graph.

    calculate_rms takes sqrt of a mean square; at exactly zero sqrt has infinite derivative, so the
    forward value stayed finite (RMS_EPS is applied afterwards) while the backward pass produced NaN.
    Reachable when the "unprocessed" signal is an upstream module's output in a cascaded system.
    """

    def test_rms_gradient_at_exact_zero_is_finite(self):
        from torch_log_wmse.utils import calculate_rms
        z = torch.zeros(1, 1, 1, 16, requires_grad=True)
        calculate_rms(z).sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(float(z.grad.abs().max()), 0.0)

    def test_grad_requiring_silent_mixture_does_not_produce_nan(self):
        n = 4096
        u = torch.zeros(1, 1, n, requires_grad=True)
        m = _metric(audio_length=n / SR, return_as_loss=True)
        m(u, torch.randn(1, 1, 1, n), torch.zeros(1, 1, 1, n)).backward()
        self.assertTrue(torch.isfinite(u.grad).all())

    def test_silent_mixture_scaling_is_pinned_to_rms_eps(self):
        """A silent mixture must scale by exactly 1/RMS_EPS, and the resulting value is pinned.

        This is a value assertion rather than a NaN check on purpose. The original 0.2.8 bug was
        1/0 -> inf -> 0*inf -> NaN, but the subnormal floor inside calculate_rms now prevents an
        exactly-zero RMS independently, so removing the RMS_EPS floor no longer produces NaN -- it
        produces a wrong number instead, tens of units below the correct one. Only pinning the value
        catches that. (The exact mutant value depends on calculate_rms's floor, so it is deliberately
        not quoted here; two fixes guard this failure by different mechanisms and each needs its own
        test, which is what the mutation study established.)
        """
        n = 4096
        m = _metric(audio_length=n / SR, reduction="none")
        z = torch.zeros(1, 1, n)
        t = torch.zeros(1, 1, 1, n)
        for level, expected in ((1e-3, -71.93113), (1e-2, -90.35181)):
            with self.subTest(level=level):
                got = float(m(z, torch.full((1, 1, 1, n), level), t))
                self.assertTrue(math.isfinite(got))
                self.assertAlmostEqual(got, expected, delta=0.01)
        # And an all-silent triplet still sits at the ceiling.
        self.assertAlmostEqual(float(m(z, t, t)), CEILING, places=4)


class TestInputValidation(unittest.TestCase):
    """Shape and option mistakes must raise, not broadcast into a plausible wrong number.

    These were bare `assert` statements, so `python -O` stripped them; and batch/channel agreement
    with unprocessed_audio was never checked at all, so a mono mixture against stereo stems silently
    broadcast and returned 18.41251.
    """

    def setUp(self):
        self.n = 4096
        self.m = _metric(audio_length=self.n / SR)
        torch.manual_seed(31)
        self.u = torch.rand(2, 2, self.n) * 2 - 1
        self.p = torch.rand(2, 2, 3, self.n) * 2 - 1
        self.t = torch.zeros(2, 2, 3, self.n)

    def test_accepts_the_documented_shapes(self):
        self.assertTrue(math.isfinite(float(self.m(self.u, self.p, self.t))))

    def test_channel_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "channel count mismatch"):
            self.m(torch.rand(2, 1, self.n), self.p, self.t)

    def test_batch_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "batch size mismatch"):
            self.m(torch.rand(1, 2, self.n), self.p, self.t)

    def test_wrong_ndim_raises(self):
        with self.assertRaisesRegex(ValueError, r"\[batch, channel, time\]"):
            self.m(self.p, self.p, self.t)
        with self.assertRaisesRegex(ValueError, r"\[batch, channel, stem, time\]"):
            self.m(self.u, self.u, self.t)

    def test_processed_target_shape_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            self.m(self.u, self.p, torch.zeros(2, 2, 1, self.n))

    def test_length_mismatch_with_configured_audio_length_raises(self):
        # Previously the filtered path silently scored only the first audio_length_samples while
        # bypass_filter=True scored everything, so the two disagreed by up to 26 units.
        longer = self.n * 2
        with self.assertRaisesRegex(ValueError, "expected"):
            self.m(torch.rand(2, 2, longer), torch.rand(2, 2, 3, longer), torch.zeros(2, 2, 3, longer))

    def test_bad_reduction_raises_at_construction(self):
        for bad in ("Mean", "average", "batchmean", "", None):
            with self.subTest(reduction=bad):
                with self.assertRaises(ValueError):
                    _metric(audio_length=self.n / SR, reduction=bad)

    def test_valid_reductions_are_accepted(self):
        for good in ("none", "mean", "sum"):
            with self.subTest(reduction=good):
                _metric(audio_length=self.n / SR, reduction=good)


class TestBundledImpulseResponse(unittest.TestCase):
    """The bundled FIR must load without a code-execution format, and must be the designed filter.

    It used to ship as a pickle, which `pickle.load` deserialises by resolving and calling arbitrary
    dotted global names, with no validation of the result. It is now raw float32 with a pinned length
    and digest.

    Note on what the response test below does and does not prove. Its expected values are
    measurements of this same artifact, so it is a CORRUPTION AND REGRESSION GUARD, not a provenance
    check -- it cannot distinguish "this is the designed filter" from "this is whatever blob was
    committed". Actual provenance is dev/create_freq_weighting_filter_ir.py, which regenerates the
    response from the documented audiomentations recipe and agrees to ~1.6e-08. The individual
    design parameters are also not directly readable off the response: the 1500 Hz shelf is +5 dB in
    isolation but measures +2.64 dB in the assembled cascade.
    """

    def test_loads_with_expected_shape_and_dtype(self):
        from torch_log_wmse.freq_weighting_filter import load_bundled_impulse_response
        ir = load_bundled_impulse_response()
        self.assertEqual(ir.ndim, 1)
        self.assertEqual(ir.numel(), 4000)
        self.assertEqual(ir.dtype, torch.float32)
        self.assertTrue(torch.isfinite(ir).all())

    def test_integrity_check_is_enforced(self):
        from torch_log_wmse import freq_weighting_filter as fwf
        original = fwf._IR_SHA256
        try:
            fwf._IR_SHA256 = "0" * 64
            with self.assertRaisesRegex(ValueError, "integrity check"):
                fwf.load_bundled_impulse_response()
        finally:
            fwf._IR_SHA256 = original
        # and it still loads once restored
        self.assertEqual(fwf.load_bundled_impulse_response().numel(), 4000)

    def test_no_deserialisation_format_is_used(self):
        """Regression guard: the loader must not reintroduce a code-execution format.

        Checked against the parsed AST rather than the source text, so prose in comments and
        docstrings that merely mentions these names does not trip it.
        """
        import ast
        import inspect
        from torch_log_wmse import freq_weighting_filter as fwf

        tree = ast.parse(inspect.getsource(fwf))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for banned in ("pickle", "dill", "joblib", "marshal", "shelve"):
            self.assertNotIn(banned, imported, f"{banned} reintroduced into the filter module")

        called = {
            ast.unparse(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name)
        }
        for banned in ("pickle.load", "pickle.loads", "torch.load", "numpy.load", "np.load"):
            self.assertNotIn(banned, called, f"{banned} reintroduced into the filter module")

    def test_is_symmetric_about_its_centre(self):
        # A zero-phase FIR must be symmetric; the shift compensation assumes a centre at (M-1)//2.
        from torch_log_wmse.freq_weighting_filter import load_bundled_impulse_response
        h = load_bundled_impulse_response()
        k = torch.arange(1, 2000)
        self.assertLess(float((h[1999 + k] - h[1999 - k]).abs().max()), 1e-7)
        self.assertEqual(int(h.abs().argmax()), 1999)

    def test_frequency_response_is_unchanged(self):
        """Corruption guard: the measured response must match the values recorded for this artifact.

        Not a provenance check -- see the class docstring. These numbers were measured from this
        blob, so they detect a swapped, truncated or resampled filter, not a wrong design.
        """
        from torch_log_wmse.freq_weighting_filter import load_bundled_impulse_response
        h = load_bundled_impulse_response()
        n = 1 << 16
        mag = 20 * torch.log10(torch.clamp_min(torch.fft.rfft(h, n).abs(), 1e-12))
        freqs = torch.fft.rfftfreq(n, 1 / SR)

        def at(hz):
            return float(mag[int(torch.argmin((freqs - hz).abs()))])

        # Measured from this artifact. The shape is consistent with the documented design (120 Hz
        # high-pass, 500 Hz peak, 1500 Hz shelf, 10 kHz low-pass) but these are cascade outputs, not
        # the individual filter parameters.
        for hz, expected in ((20, -30.97), (50, -15.39), (120, -2.89), (500, 2.53), (1000, 1.49),
                             (1500, 2.64), (3000, 4.24), (10000, -1.02), (20000, -30.94)):
            with self.subTest(hz=hz):
                self.assertAlmostEqual(at(hz), expected, delta=0.15)
        # -3 dB corners bracket the passband.
        self.assertAlmostEqual(float(freqs[int((mag > -3).to(torch.int8).argmax())]), 118.0, delta=3.0)
        peak = int(mag.argmax())
        self.assertAlmostEqual(float(freqs[peak]), 2971.0, delta=60.0)


class TestApplyReductionDirectly(unittest.TestCase):
    """apply_reduction is public and must validate on its own.

    LogWMSE validates `reduction` at construction, so the guard inside apply_reduction is never
    reached through the metric. It is still part of the public surface, and a mutation study showed
    that removing its `raise` was undetectable via LogWMSE alone.
    """

    def test_known_reductions(self):
        from torch_log_wmse.utils import apply_reduction
        x = torch.arange(6.0).reshape(2, 3)
        self.assertTrue(torch.equal(apply_reduction(x, "none"), x))
        self.assertAlmostEqual(float(apply_reduction(x, "mean")), float(x.mean()))
        self.assertAlmostEqual(float(apply_reduction(x, "sum")), float(x.sum()))

    def test_unknown_reduction_raises(self):
        from torch_log_wmse.utils import apply_reduction
        x = torch.arange(6.0).reshape(2, 3)
        for bad in ("Mean", "average", "batchmean", "", None, 0):
            with self.subTest(reduction=bad):
                with self.assertRaises(ValueError):
                    apply_reduction(x, bad)

    def test_valid_reductions_constant_is_exported(self):
        from torch_log_wmse.utils import VALID_REDUCTIONS
        self.assertEqual(set(VALID_REDUCTIONS), {"none", "mean", "sum"})


class TestConstructorValidation(unittest.TestCase):
    """audio_length and sample_rate must be validated, not left to fail opaquely later."""

    def test_degenerate_audio_length_raises(self):
        for bad in (0, -1.0, 1e-9):
            with self.subTest(audio_length=bad):
                with self.assertRaisesRegex(ValueError, "at least 1 sample"):
                    _metric(audio_length=bad)

    def test_non_positive_sample_rate_raises(self):
        for bad in (0, -44100):
            with self.subTest(sample_rate=bad):
                with self.assertRaisesRegex(ValueError, "sample_rate must be positive"):
                    _metric(audio_length=1.0, sample_rate=bad)

    def test_non_positive_impulse_response_sample_rate_raises(self):
        with self.assertRaisesRegex(ValueError, "impulse_response_sample_rate must be positive"):
            _metric(audio_length=1.0, impulse_response_sample_rate=0)

    def test_fractional_audio_length_accepts_floor_or_round(self):
        # floor() and round() of audio_length * sample_rate differ by one sample for many fractional
        # lengths (22 of the 399 hundredth-second values at 44.1 kHz). Rejecting the round() form
        # would fail callers who sized their segments the ordinary way.
        for length in (0.35, 0.57, 0.69, 0.7):
            with self.subTest(audio_length=length):
                m = _metric(audio_length=length)
                for n in (math.floor(length * SR), round(length * SR)):
                    out = m(torch.rand(1, 1, n), torch.rand(1, 1, 1, n), torch.zeros(1, 1, 1, n))
                    self.assertTrue(math.isfinite(float(out)))

    def test_length_still_rejects_a_real_mismatch(self):
        m = _metric(audio_length=1.0)
        for n in (SR // 2, SR * 2, SR + 100):
            with self.subTest(n=n):
                with self.assertRaisesRegex(ValueError, "expected"):
                    m(torch.rand(1, 1, n), torch.rand(1, 1, 1, n), torch.zeros(1, 1, 1, n))


class TestRmsFloorIsDtypeCorrect(unittest.TestCase):
    """The sqrt floor must be the dtype's smallest normal, not a hard-coded constant.

    A literal such as 1e-24 underflows to 0.0 in float16, making the guard a silent no-op there, and
    it would also rewrite any legitimate mean-square below it -- float32 represents mean-squares down
    to about 1e-38, so a fixed 1e-24 altered roughly seven decades of genuinely quiet audio.
    """

    def test_gradient_at_zero_is_finite_in_every_float_dtype(self):
        from torch_log_wmse.utils import calculate_rms
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            with self.subTest(dtype=dtype):
                z = torch.zeros(1, 1, 1, 16, dtype=dtype, requires_grad=True)
                calculate_rms(z).sum().backward()
                self.assertTrue(torch.isfinite(z.grad).all())
                self.assertEqual(float(z.grad.abs().max()), 0.0)

    def test_quiet_but_normal_amplitudes_are_preserved(self):
        from torch_log_wmse.utils import calculate_rms
        # 1e-13 is about -260 dBFS: absurdly quiet, but its mean-square is still a normal float32,
        # so it must pass through untouched. The old 1e-24 floor clamped it to 1e-12.
        for amp in (1e-6, 1e-10, 1e-13):
            with self.subTest(amp=amp):
                got = float(calculate_rms(torch.full((1, 1, 1, 64), amp)))
                self.assertAlmostEqual(got / amp, 1.0, places=3)


class TestAggregationContract(unittest.TestCase):
    """The four properties that make the pooled number MEAN something, stated without reference to
    how pooling is implemented.

    These are the contract for the aggregation change. Every one of them holds under mean-of-logs
    and under a power mean at any p, which is the point: if the redesign breaks one, the design is
    wrong and not the test, and the right response is to stop rather than relax the assertion.

    Levels are kept well clear of the -68 dB inaudibility gate, or "improving a stem" would
    saturate against the gate rather than the metric and monotonicity would be testing the wrong
    thing.
    """

    N = 2048
    # Both filter paths, and both float dtypes the filter supports. float16 cannot reach the
    # filtering path at all (no half FFT kernel on CPU or MPS) and is covered separately.
    VARIANTS = [(dt, bp) for dt in (torch.float32, torch.float64) for bp in (False, True)]

    def _graded(self, levels, dtype=torch.float32, seed=41):
        """levels[c][s] = that element's residual amplitude, relative to the mixture."""
        c, s = len(levels), len(levels[0])
        torch.manual_seed(seed)
        u = (torch.rand(1, c, self.N) * 2 - 1).to(dtype)
        t = (torch.rand(1, c, s, self.N) * 2 - 1).to(dtype)
        lv = torch.tensor(levels, dtype=dtype).reshape(1, c, s, 1)
        return u, t + u[:, :, None, :] * lv, t

    def test_bracketing(self):
        """The aggregate always lies between the best and worst per-element score.

        This is what licenses reading the pooled number as an average rather than an arbitrary
        function of the parts, and it is the property most at risk from a pooling change.
        """
        levels = [[0.316, 0.05, 0.01], [0.1, 0.02, 0.003]]
        for dtype, bypass in self.VARIANTS:
            with self.subTest(dtype=dtype, bypass_filter=bypass):
                u, p, t = self._graded(levels, dtype)
                kw = dict(audio_length=self.N / SR, bypass_filter=bypass)
                elements = per_element(u, p, t, **kw).flatten()
                pooled = float(_metric(**kw)(u, p, t))
                # Guards the guard: bracketing is trivially true when the elements are equal, so a
                # collapsed spread would leave this asserting nothing. Measured 37.25 units here.
                self.assertGreater(float(elements.max() - elements.min()), 20.0,
                                   "per-element spread collapsed; bracketing is now vacuous")
                self.assertGreaterEqual(pooled, float(elements.min()) - 1e-4,
                                        f"pooled {pooled} is below the worst element")
                self.assertLessEqual(pooled, float(elements.max()) + 1e-4,
                                     f"pooled {pooled} is above the best element")

    def test_monotonicity(self):
        """Improving any single element never worsens the aggregate."""
        base = [[0.2, 0.05], [0.1, 0.02]]
        for dtype, bypass in self.VARIANTS:
            kw = dict(audio_length=self.N / SR, bypass_filter=bypass)
            before = float(_metric(**kw)(*self._graded(base, dtype)))
            for c in range(2):
                for s in range(2):
                    with self.subTest(dtype=dtype, bypass_filter=bypass, element=(c, s)):
                        better = [row[:] for row in base]
                        better[c][s] *= 0.5  # halve one element's residual, leave the rest
                        after = float(_metric(**kw)(*self._graded(better, dtype)))
                        self.assertGreaterEqual(
                            after, before - 1e-4,
                            f"improving element ({c},{s}) moved the score {before} -> {after}")

    def test_equal_quality_agreement(self):
        """When every element is identical, the aggregate equals that element exactly.

        True under every pooling rule - a power mean of identical values is that value - which is
        why the familiar single-number reading survives the redesign unchanged.
        """
        for dtype, bypass in self.VARIANTS:
            with self.subTest(dtype=dtype, bypass_filter=bypass):
                torch.manual_seed(43)
                # Identical mixture across channels AND an identical residual on every element, so
                # the per-element values are bit-identical rather than merely close.
                u = ((torch.rand(1, 1, self.N) * 2 - 1).to(dtype)).expand(1, 2, self.N).contiguous()
                t = (torch.rand(1, 2, 3, self.N) * 2 - 1).to(dtype)
                d = (torch.rand(1, 1, 1, self.N) * 2 - 1).to(dtype) * 0.05
                p = t + d

                kw = dict(audio_length=self.N / SR, bypass_filter=bypass)
                elements = per_element(u, p, t, **kw).flatten()
                self.assertLess(float(elements.max() - elements.min()), 1e-4,
                                "the construction is not actually equal-quality")
                self.assertAlmostEqual(float(_metric(**kw)(u, p, t)), float(elements[0]), places=4)

    def test_single_element_identity(self):
        """Pooling one element is the identity - the mono, single-stem denoising case.

        Guards against an aggregation change silently altering the most common usage of all.
        """
        for dtype, bypass in self.VARIANTS:
            with self.subTest(dtype=dtype, bypass_filter=bypass):
                u, p, t = self._graded([[0.08]], dtype)
                kw = dict(audio_length=self.N / SR, bypass_filter=bypass)
                element = float(per_element(u, p, t, **kw).flatten()[0])
                self.assertAlmostEqual(float(_metric(**kw)(u, p, t)), element, places=5)


class TestGradientAccumulationEquivalence(unittest.TestCase):
    """Two micro-batches must give the same gradient as one full batch.

    This is the empirical reason the batch axis stays a conventional mean while channel and stem
    move to a power mean. Batch items are independent samples, so averaging their losses is what
    makes the objective an expectation over the data. Pooling batch non-linearly would COUPLE
    examples, and the first thing that breaks is this identity - which is also what makes DDP
    equivalent to a single large batch.

    It holds exactly today (measured 0.000e+00), so it is assertable rather than approximate, and it
    is the test that kills a mutant which pools the batch axis along with the others.
    """

    def _grad(self, m, u, p, t, scale=1.0):
        pe = p.detach().clone().requires_grad_(True)
        (m(u, pe, t) * scale).backward()
        return pe.grad

    def test_two_micro_batches_equal_one_full_batch(self):
        n = 2048
        torch.manual_seed(31)
        u = torch.rand(4, 2, n) * 2 - 1
        t = torch.rand(4, 2, 3, n) * 2 - 1
        p = t + (torch.rand(4, 2, 3, n) * 2 - 1) * 0.1
        m = _metric(audio_length=n / SR)

        full = self._grad(m, u, p, t)
        # Each half carries half the weight, which is exactly what an accumulation loop does.
        first = self._grad(m, u[:2], p[:2], t[:2], scale=0.5)
        second = self._grad(m, u[2:], p[2:], t[2:], scale=0.5)
        accumulated = torch.cat([first, second], dim=0)

        self.assertTrue(torch.isfinite(full).all())
        # torch.equal, not a tolerance: measured bit-identical today (max abs divergence 0.000e+00),
        # so anything else is a real change rather than float noise.
        self.assertTrue(
            torch.equal(full, accumulated),
            f"accumulation diverged by {float((full - accumulated).abs().max()):.3e}; the batch axis "
            "is no longer a plain mean, which also breaks DDP equivalence")

    def test_batch_items_do_not_influence_each_other(self):
        """The same statement from the forward side: one item's value cannot depend on its neighbours."""
        n = 2048
        torch.manual_seed(32)
        u = torch.rand(3, 1, n) * 2 - 1
        t = torch.rand(3, 1, 2, n) * 2 - 1
        p = t + (torch.rand(3, 1, 2, n) * 2 - 1) * 0.1
        m = _metric(audio_length=n / SR, reduction="none")

        together = m(u, p, t)
        for i in range(3):
            with self.subTest(item=i):
                alone = m(u[i : i + 1], p[i : i + 1], t[i : i + 1])
                self.assertTrue(torch.equal(together[i : i + 1], alone),
                                f"batch item {i} changed value depending on its neighbours")


@unittest.skipUnless(
    os.environ.get("CI") or os.environ.get("TLW_TEST_COMPILE"),
    "torch.compile takes ~8s and spawns compile workers; on by default in CI, opt in locally with "
    "TLW_TEST_COMPILE=1")
class TestTorchCompile(unittest.TestCase):
    """A compiled forward must agree with eager and must not graph-break on the per-size lookup.

    Fixed-length input is the supported case and is what this pins. Variable length is documented as
    causing recompilation: transform sizes sit 1-2% apart once they are 5-smooth rather than powers
    of two, so a variable-length workload blows past inductor's default cache_size_limit of 8.

    Inductor warns that it cannot generate code for complex operators, so the FFT stays in eager.
    That is expected and is why the tolerance here is float32 noise rather than zero.
    """

    def test_compiled_matches_eager(self):
        n = 2048
        torch.manual_seed(1)
        u, p, t = torch.rand(1, 1, n), torch.rand(1, 1, 2, n), torch.rand(1, 1, 2, n)
        m = _metric(audio_length=n / SR)
        eager = float(m(u, p, t))
        compiled = float(torch.compile(m)(u, p, t))
        self.assertAlmostEqual(compiled, eager, delta=1e-5,
                               msg=f"compiled {compiled} vs eager {eager}")


class TestLowPrecisionDtypes(unittest.TestCase):
    """float16 and bfloat16 must return a finite number, and the SAME ceiling as float32.

    The bug: EPS = 1e-8 underflows to exactly 0.0 in float16 (smallest subnormal 5.96e-8), so a
    bit-exact stem gave log(0) = -inf and the metric returned +inf - which "mean" then spread across
    the entire batch. bfloat16 does not underflow but has 8 mantissa bits, too few to keep mse + EPS
    distinct from mse.

    COVERAGE LIMIT, stated so nobody assumes otherwise: every test here runs with
    bypass_filter=True, because torch.fft has no half kernel on CPU or on MPS ("Unsupported dtype
    Half" on both). Half precision through the FILTERING path is unreachable on any hardware this
    project can test on, and whether CUDA provides a half FFT kernel is untested - there is no CUDA
    device available.
    """

    HALF_DTYPES = (torch.float16, torch.bfloat16)

    def _triplet(self, dtype, n=1024, exact_stem=True):
        torch.manual_seed(5)
        u = (torch.rand(1, 1, n) * 2 - 1).to(dtype)
        t = (torch.rand(1, 1, 2, n) * 2 - 1).to(dtype)
        p = t.clone()
        if not exact_stem:
            p = p + 0.1
        else:
            p[:, :, 1] = p[:, :, 1] + 0.1  # stem 0 exact, stem 1 not
        return u, p, t

    def test_bit_exact_stem_is_finite_and_hits_the_float32_ceiling(self):
        for dtype in self.HALF_DTYPES:
            with self.subTest(dtype=dtype):
                m = _metric(audio_length=1024 / SR, bypass_filter=True, reduction="none")
                u, p, t = self._triplet(dtype)
                got = m(u, p, t)
                self.assertTrue(torch.isfinite(got).all(), f"{dtype} produced {got}")
                # Stem 0 is bit-exact, so it must read the ceiling - the SAME ceiling float32 gives.
                self.assertAlmostEqual(float(got[0, 0, 0]), CEILING, places=3)

    def test_mean_reduction_does_not_propagate_an_infinity(self):
        """The failure that made this worth fixing: one exact stem used to poison the whole batch."""
        for dtype in self.HALF_DTYPES:
            with self.subTest(dtype=dtype):
                m = _metric(audio_length=1024 / SR, bypass_filter=True)
                got = m(*self._triplet(dtype))
                self.assertTrue(torch.isfinite(got).all(), f"{dtype} produced {got}")

    def test_agrees_with_float32_on_a_non_degenerate_case(self):
        """Half precision may be coarse, but it must not be WRONG."""
        m = _metric(audio_length=1024 / SR, bypass_filter=True)
        ref = float(m(*self._triplet(torch.float32, exact_stem=False)))
        for dtype in self.HALF_DTYPES:
            with self.subTest(dtype=dtype):
                got = float(m(*self._triplet(dtype, exact_stem=False)))
                self.assertAlmostEqual(got, ref, delta=0.5,
                                       msg=f"{dtype} read {got} vs float32 {ref}")

    def test_backward_is_finite_through_a_bit_exact_stem(self):
        """Forward-only checks miss this class entirely: the value can be finite while the gradient
        is not. Kept here because the same trap reappears when pooling gains a fractional power."""
        for dtype in self.HALF_DTYPES:
            with self.subTest(dtype=dtype):
                m = _metric(audio_length=1024 / SR, bypass_filter=True)
                u, p, t = self._triplet(dtype)
                p = p.detach().requires_grad_(True)
                m(u, p, t).backward()
                self.assertTrue(torch.isfinite(p.grad).all(), f"{dtype} grad: {p.grad}")


if __name__ == "__main__":
    unittest.main()
