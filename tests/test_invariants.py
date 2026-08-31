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
from torch_log_wmse import LogWMSE, LogWMSELoss
from torch_log_wmse.constants import EPS, SCALER
from torch_log_wmse.freq_weighting_filter import next_fft_friendly_size, parseval_weights
from torch_log_wmse.utils import VALID_REDUCTIONS

# Module-local alias, so the call sites below stay as they are.
_metric = make_metric


class TestClosedFormOracle(unittest.TestCase):
    """The metric has an exact analytic value for est = k * unprocessed against a silent target.

    differences = filters(k*u*s) and s = 1/rms(filters(u)), so mse = k^2 exactly and the metric is
    SCALER*ln(k^2 + EPS). This is derivation-backed, unlike a golden value recorded from this
    implementation.

    The oracle now holds at EVERY k. It used to break down below about 1e-3, because the -68 dB
    per-sample inaudibility gate zeroed a growing fraction of the samples and the measured mse fell
    below k^2. That gate is gone in 1.0.0, so the range where the derivation applies is no longer
    bounded from below - which is itself the cleanest evidence the gate really has been removed.
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

    def test_the_oracle_now_holds_below_the_old_inaudibility_gate(self):
        """The former complement of the test above, inverted by removing the gate.

        With the -68 dB gate in place, k = 1e-4 read +2.77 units BETTER than the closed form,
        because sub-threshold samples were discounted. It must now match the closed form like any
        other level. This is the regression guard against the gate coming back.
        """
        torch.manual_seed(0)
        m = _metric()
        u = (torch.rand(1, 1, SR) * 2 - 1)
        for k in (1e-3, 4e-4, 1e-4):
            with self.subTest(k=k):
                p = (u * k).unsqueeze(2)
                got = float(m(u, p, torch.zeros_like(p)))
                expected = SCALER * math.log(k * k + EPS)
                self.assertAlmostEqual(got, expected, places=2,
                                       msg=f"k={k}: {got} vs closed form {expected}; a "
                                           f"{got - expected:+.2f} offset means error is being discounted")

    def test_exact_match_hits_the_ceiling(self):
        torch.manual_seed(1)
        m = _metric()
        u = (torch.rand(1, 1, SR) * 2 - 1)
        t = (torch.rand(1, 1, 1, SR) * 2 - 1) * 0.3
        self.assertAlmostEqual(float(m(u, t.clone(), t)), CEILING, places=4)

    def test_ceiling_matches_the_constants(self):
        self.assertAlmostEqual(CEILING, 73.68272, places=4)


class TestFilterPhaseIsIrrelevant(unittest.TestCase):
    """The replacement for the old zero-phase tests, and a stronger check than they were.

    The score depends on the filter only through `|H|^2`, so the filter's PHASE cannot affect it at
    all. Two consequences worth asserting: a delta impulse response anywhere gives exactly the
    unfiltered score, and rolling the impulse response changes nothing.

    The delta-IR case reliably catches a wrong divisor and a too-short transform, both of which are
    otherwise near-invisible - one reads as a plausible number, the other as FFT noise.

    IT IS NOT A GUARD ON THE WEIGHT VECTOR'S SCALE, despite an earlier claim here that it caught a
    missing one-sided doubling "as a clean 4*ln(2) = 2.7726 offset". It cannot: the same weights
    divide out of `mse = weighted_error_energy / weighted_mixture_energy`, so ANY global factor on
    them cancels exactly. Dropping the doubling is not quite global - DC and Nyquist keep their
    coefficient either way - and the residual that leaves behind shrinks with buffer length, so it
    is caught here at n=512 and MISSED at n=44100. `TestParsevalWeights` is the real guard on the
    weights; this test is a guard on alignment and sizing.

    NOTE this is about the FILTER's phase. The metric is emphatically NOT phase-invariant with
    respect to the signals - the error is `estimate - target`, so a latency offset between the two
    is fully penalised.
    """

    def _case(self, n=512, seed=0):
        torch.manual_seed(seed)
        u = torch.rand(1, 1, n) * 2 - 1
        p = (torch.rand(1, 1, 1, n) * 2 - 1) * 0.1
        return u, p, torch.zeros_like(p)

    def test_a_delta_ir_anywhere_equals_the_unfiltered_score(self):
        u, p, t = self._case()
        reference = float(_metric(bypass_filter=True)(u, p, t))
        # Odd IR lengths were the regression guard for the group-delay misalignment (finding A2);
        # they now also cover the transform-size parity that the Nyquist weight depends on.
        for ir_len in (2, 3, 4, 51, 100, 101, 999, 1000, 1001):
            for position in (0, (ir_len - 1) // 2, ir_len - 1):
                with self.subTest(ir_len=ir_len, position=position):
                    ir = torch.zeros(ir_len)
                    ir[position] = 1.0
                    m = _metric(impulse_response=ir, impulse_response_sample_rate=SR)
                    self.assertAlmostEqual(
                        float(m(u, p, t)), reference, places=3,
                        msg=f"delta IR (len {ir_len}, tap at {position}) read "
                            f"{float(m(u, p, t))} against an unfiltered {reference}")

    def test_delaying_the_ir_changes_nothing(self):
        """A pure delay is a phase change and nothing else, so the score must not move.

        Delay by PREPENDING ZEROS, not by `torch.roll` on the tap array. Rolling wraps the last
        taps around to the front, and once that array is zero-padded to the transform size those
        taps sit in the wrong place entirely - a different filter with a different |H|, which
        legitimately scores differently (measured up to 0.15 units). That is a property of the
        test, not of the metric.
        """
        u, p, t = self._case(seed=1)
        torch.manual_seed(9)
        ir = torch.rand(256) - 0.5
        base = float(_metric(impulse_response=ir, impulse_response_sample_rate=SR)(u, p, t))
        for delay in (1, 7, 128, 1000):
            with self.subTest(delay=delay):
                delayed = torch.cat([torch.zeros(delay), ir])
                got = float(_metric(impulse_response=delayed, impulse_response_sample_rate=SR)(u, p, t))
                self.assertAlmostEqual(got, base, places=4,
                                       msg=f"a {delay}-sample delay moved the score {base} -> {got}")

    def test_builtin_filter_returns_finite_energy_per_element(self):
        f = make_filter()
        x = torch.randn(2, 2, 3, 2048)
        energy = f(x)
        self.assertEqual(energy.shape, (2, 2, 3))  # energy per element, not a filtered signal
        self.assertTrue(torch.isfinite(energy).all())
        self.assertTrue((energy > 0).all())


class TestParsevalWeights(unittest.TestCase):
    """Unit tests for the weight vector, where the one-sided sum is easy to get wrong in silence.

    `w[f] = |H(f)|^2 * c[f] / L`, with c = 1 at DC, 2 for interior bins, and 1 at the last bin for
    even L or 2 for odd L. Three separate mistakes live in that one line, so each is targeted by a
    signal that concentrates its energy exactly where the mistake would show:

      * interior bins not doubled -> a broadband signal is off by a factor of ~2
      * DC bin doubled            -> invisible on broadband (one bin in thousands), obvious on DC
      * Nyquist parity ignored    -> invisible unless L is odd AND the signal sits at the top bin
    """

    LENGTHS = (512, 513, 1024, 1025)  # both parities, deliberately

    def _signal(self, kind, n):
        if kind == "random":
            torch.manual_seed(5)
            return torch.rand(n) * 2 - 1
        if kind == "dc":
            return torch.ones(n)
        if kind == "nyquist":  # alternating +/-1: all energy in the top bin
            return torch.where(torch.arange(n) % 2 == 0, 1.0, -1.0)
        raise ValueError(kind)

    def test_weights_reproduce_the_time_domain_energy(self):
        torch.manual_seed(6)
        ir = torch.rand(64) - 0.5
        n = 256
        for length in self.LENGTHS:
            for kind in ("random", "dc", "nyquist"):
                with self.subTest(transform_size=length, signal=kind, parity=length % 2):
                    x = self._signal(kind, n)
                    spectrum = torch.fft.rfft(x, n=length)
                    got = float(((spectrum.real**2 + spectrum.imag**2)
                                 * parseval_weights(ir, length)).sum())
                    # Reference: the convolution actually performed, summed in the time domain.
                    filtered = torch.fft.irfft(
                        torch.fft.rfft(x, n=length) * torch.fft.rfft(ir, n=length), n=length)
                    want = float(filtered.pow(2).sum())
                    self.assertAlmostEqual(got / want, 1.0, places=4,
                                           msg=f"L={length} {kind}: {got} vs {want} "
                                               f"(ratio {got / want:.6f})")

    def test_a_missing_doubling_would_be_a_factor_of_two(self):
        """Pins the size of the mistake, so the failure above is recognisable when it happens."""
        # A delta IR has |H| = 1 in every bin, so w is exactly c / L and the coefficients are bare.
        w = parseval_weights(torch.tensor([1.0, 0.0]), 512)
        interior = w[1:-1] * 512
        self.assertTrue(torch.allclose(interior, torch.full_like(interior, 2.0), atol=1e-5),
                        "interior bins are not doubled; every value will be 4*ln(2) = 2.7726 low")
        self.assertAlmostEqual(float(w[0] * 512), 1.0, places=5, msg="DC must be counted once")
        self.assertAlmostEqual(float(w[-1] * 512), 1.0, places=5,
                               msg="Nyquist must be counted once for an EVEN transform size")
        odd = parseval_weights(torch.tensor([1.0, 0.0]), 513)
        self.assertAlmostEqual(float(odd[-1] * 513), 2.0, places=5,
                               msg="the top bin of an ODD transform is an ordinary conjugate pair")


class TestInaudibilityGateIsGone(unittest.TestCase):
    """The -68 dB per-sample gate was removed in 1.0.0. These pin its absence.

    It could not survive computing energy in the frequency domain - a per-SAMPLE gate needs a
    time-domain signal. Its measured effect across the reachable range was 0.000, and reaching the
    band where it mattered needs 74-80 dB SI-SDR.
    """

    def test_uniform_subthreshold_error_is_scored_not_discarded(self):
        n = 4096
        m = _metric(bypass_filter=True)
        u = torch.ones(1, 1, n)
        t = torch.zeros(1, 1, 1, n)
        k = 10 ** (-68.0 / 20) * 0.5  # half the old threshold: formerly zeroed outright
        below = torch.full((1, 1, 1, n), k)
        expected = SCALER * math.log(k * k + EPS)
        self.assertAlmostEqual(float(m(u, below, t)), expected, places=3)
        self.assertLess(float(m(u, below, t)), CEILING - 1.0,
                        "sub-threshold error is still being discarded")

    def test_the_constant_no_longer_exists(self):
        import torch_log_wmse.constants as constants

        self.assertFalse(hasattr(constants, "ERROR_TOLERANCE_THRESHOLD"))


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


class TestGradientAllocation(unittest.TestCase):
    """How `p` reshapes per-stem gradient allocation, and the exact boundary of the equal-gradient claim.

    Per-stem gradient energy scales as `G_s**2 * mse**(2p - 1)`. On residuals of MATCHED spectrum
    the `G_s**2` factors are equal, so `p = 1/2` cancels the exponent and every stem gets the same
    gradient energy - a derived value, not a tuned one, and the measurement below is exact. On
    residuals of DIFFERING spectrum it does not, which the second test pins so the claim is never
    overstated again.

    `p = 0` (the mean of logs) is the shipped default and the historically comparable one; `p = 1/2`
    is offered as the late-training gradient mitigation for issue #5, not as the default. Because
    the error is normalised by the MIXTURE, the largest-error stem is the LOUDEST, not the one that
    most needs gradient - so `p = 0` concentrating gradient away from it is a property to
    characterise, not a defect. These tests pin the allocation each `p` produces; they do not argue
    that one is correct.
    """

    N = 8192
    LEVELS = (0.316, 0.1, 0.0316, 0.0178)  # -10, -20, -30, -35 dB: a 25 dB spread

    def _shares(self, p, spectra="matched"):
        """Gradient-energy shares per stem.

        `spectra="matched"` gives every stem the SAME waveform at different gains, so their
        residual spectra are proportional. That is the condition under which the exponent law is
        the whole story - and asserting equality on it alone would be a degenerate test, so
        `spectra="coloured"` exists and is used by the test below.
        """
        torch.manual_seed(5)
        u = torch.rand(1, 1, self.N) * 2 - 1
        t = torch.rand(1, 1, 4, self.N) * 2 - 1
        levels = torch.tensor(self.LEVELS).reshape(1, 1, 4, 1)
        if spectra == "matched":
            residual = u[:, :, None, :] * levels
        else:  # each stem's residual occupies a different band
            time = torch.arange(self.N, dtype=torch.float32) / SR
            tones = torch.stack([torch.sin(2 * math.pi * f * time)
                                 for f in (200.0, 1000.0, 3000.0, 12000.0)])
            residual = (tones / tones.pow(2).mean(dim=-1, keepdim=True).sqrt()).reshape(1, 1, 4, -1) * levels
        estimate = (t + residual).detach().requires_grad_(True)
        _metric(p=p, return_as_loss=True)(u, estimate, t).backward()
        energy = torch.stack([estimate.grad[:, :, s].pow(2).sum() for s in range(4)])
        return energy / energy.sum(), float(estimate.grad.norm())

    def test_gradient_energy_is_equal_across_stems_of_matched_spectrum(self):
        """The exponent law, on the case where it is the whole story.

        Read the qualifier: MATCHED SPECTRUM. See the test below for what happens otherwise.
        """
        shares, _ = self._shares(0.5)
        for i, share in enumerate(shares):
            with self.subTest(stem=i, level_db=round(20 * math.log10(self.LEVELS[i]))):
                self.assertAlmostEqual(float(share), 0.25, places=3,
                                       msg=f"stem {i} takes {100 * float(share):.1f}% of the "
                                           "gradient energy, not 25%")

    def test_equalisation_does_NOT_survive_differing_residual_spectra(self):
        """The limit of the claim, pinned so it cannot quietly be overstated again.

        Gradient energy is `G_s^2 * mse_s^(2p-1)`, not `mse_s^(2p-1)` alone. `p = 1/2` cancels the
        exponent term and nothing else. `G_s^2` is an |H|^2-weighted mean of |H|^2 over that stem's
        own residual spectrum, and the shipped filter spans about 35 dB across 30 Hz - 20 kHz, so
        two stems with identical mse but residuals in different bands get very different gradient.

        This is asserted as a KNOWN LIMITATION rather than a defect: no single p can cancel a
        factor that does not depend on p. It exists so that "equal gradient energy" is never again
        documented without its qualifier - the original measurement used one waveform at four
        gains, which is exactly the degenerate case this guards against.
        """
        matched, _ = self._shares(0.5, spectra="matched")
        coloured, _ = self._shares(0.5, spectra="coloured")
        self.assertLess(float(matched.max() / matched.min()), 1.01,
                        "matched spectra should equalise almost exactly")
        self.assertGreater(float(coloured.max() / coloured.min()), 2.0,
                           "if coloured residuals now equalise too, the G^2 factor has been "
                           "cancelled somehow and the documented limitation should be revisited")

    def test_participation_ratio_reaches_the_stem_count(self):
        """1/sum(share^2) counts how many stems are effectively being trained. 4.00/4 is ideal."""
        shares, _ = self._shares(0.5)
        self.assertAlmostEqual(float(1.0 / shares.pow(2).sum()), 4.0, places=3)

    def test_p_zero_concentrates_gradient_away_from_the_worst_stem(self):
        """How `p = 0` (the default) allocates gradient, pinned so a pooling change cannot alter it silently.

        `p = 0` gives the largest-error stem under 1% of the gradient energy. Because error is
        normalised by the mixture, that stem is the loudest, not the neediest, so this is the
        allocation's shape rather than a defect; `p = 0.5` spreads it (see the tests above).
        """
        shares, _ = self._shares(0.0)
        self.assertLess(float(shares[0]), 0.01,
                        "p=0 no longer concentrates gradient away from the worst-error stem; if the "
                        "pooling changed, this test is the wrong oracle")
        self.assertLess(float(1.0 / shares.pow(2).sum()), 2.0)

    def test_p_one_concentrates_worse_than_p_zero(self):
        """Pinning the refutation of the obvious alternative.

        Pooling the MSE arithmetically looks like the natural fix and is worse than doing nothing:
        it chases the loudest residual even harder than the log-domain mean does. Measured
        participation ratio 1.23/4 against 1.66/4 for p=0 and 4.00/4 for p=1/2.
        """
        one, _ = self._shares(1.0)
        zero, _ = self._shares(0.0)
        self.assertLess(float(1.0 / one.pow(2).sum()), float(1.0 / zero.pow(2).sum()))

    def test_the_global_gradient_norm_shrinks_as_pooling_balances(self):
        """The other half of GH #5: a runaway norm as stems converge."""
        norms = [self._shares(p)[1] for p in (0.0, 0.5)]
        self.assertLess(norms[1], norms[0])


class TestIntegerAudioIsRejected(unittest.TestCase):
    """Integer PCM used to return the "perfect" ceiling for ANY input, including the worst one.

    Every tap of the hearing-sensitivity filter has magnitude below 1, so building the weight
    vector in the input's dtype truncated all 4000 taps to zero for an integer input. Error energy
    and mixture energy both came out 0 and the score pinned at +73.6827 - recreating, after the
    fact, exactly the failure the all-zero guard in the constructor exists to prevent.

    `soundfile.read(dtype='int16')` and `scipy.io.wavfile.read` both hand you this.
    """

    def test_integer_inputs_raise_instead_of_scoring_perfect(self):
        n = 4410
        mixture = torch.randint(-8000, 8000, (1, 1, n), dtype=torch.int16)
        target = torch.randint(-8000, 8000, (1, 1, 1, n), dtype=torch.int16)
        estimate = torch.zeros_like(target)  # the worst possible answer
        for dtype in (torch.int16, torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(TypeError, "must be a floating-point tensor"):
                    _metric()(mixture.to(dtype), estimate.to(dtype), target.to(dtype))

    def test_the_weight_vector_is_never_built_in_an_integer_dtype(self):
        """The structural half of the fix: `HumanHearingSensitivityFilter` is public, so the guard
        above is not the only way in."""
        f = make_filter()
        for dtype in (torch.int16, torch.int64, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                w = f.weights_for(4410, dtype)
                self.assertTrue(w.is_floating_point())
                self.assertGreater(float(w.abs().max()), 0.0,
                                   f"weights collapsed to zero for {dtype}")


class TestLargePIsNumericallySafe(unittest.TestCase):
    """`p` above about 5 used to return +-inf in float32, and the failure arrived LATE.

    `x.pow(p).mean().pow(1/p)` raises to the p-th power before the mean, so with a bit-exact stem
    (x = EPS = 1e-8) and p=6, x**p = 1e-48 underflows float32 to exactly 0 - mean 0, root 0,
    log(0), score +inf. In training that means the loss behaves normally and then dies once the
    model gets GOOD, and under reduction="mean" one such item poisons the whole batch.

    Pooling is now done in the log domain via logsumexp, which subtracts its own maximum.
    """

    P_VALUES = (0.5, 2.0, 5.0, 6.0, 8.0, 20.0, 64.0)

    def test_an_exact_match_returns_the_ceiling_at_every_p(self):
        torch.manual_seed(2)
        u = torch.rand(1, 1, 4096) * 2 - 1
        t = torch.rand(1, 1, 2, 4096) * 2 - 1
        for p in self.P_VALUES:
            with self.subTest(p=p):
                self.assertAlmostEqual(float(_metric(p=p)(u, t.clone(), t)), CEILING, places=3)

    def test_gradients_stay_finite_at_every_p(self):
        torch.manual_seed(3)
        u = torch.rand(1, 1, 4096) * 2 - 1
        t = torch.rand(1, 1, 2, 4096) * 2 - 1
        for p in self.P_VALUES:
            with self.subTest(p=p):
                estimate = t.clone().requires_grad_(True)  # bit-exact: the case that used to break
                _metric(p=p, return_as_loss=True)(u, estimate, t).backward()
                self.assertTrue(torch.isfinite(estimate.grad).all(), f"NaN/inf gradient at p={p}")

    def test_the_log_domain_form_agrees_with_the_direct_one_where_both_are_safe(self):
        """Guards the rewrite itself: within float32's safe range the two must still agree."""
        torch.manual_seed(4)
        mse = torch.rand(3, 2, 4) * 0.1 + 1e-4
        from torch_log_wmse.constants import EPS, SCALER
        from torch_log_wmse.metric import pool_mse
        for p in (0.25, 0.5, 1.0, 2.0, 4.0):
            with self.subTest(p=p):
                direct = torch.log((mse + EPS).pow(p).mean(dim=(1, 2)).pow(1.0 / p)) * SCALER
                self.assertTrue(torch.allclose(pool_mse(mse, p), direct, atol=1e-5))


class TestTheFilterActuallyShapesTheScore(unittest.TestCase):
    """Guards a structural blind spot: almost nothing else in this file can see the filter.

    Because `mse = weighted_error_energy / weighted_mixture_energy`, ANY linear weighting cancels
    out of the `estimate = k * mixture` oracle. Replacing the hearing-sensitivity curve with a flat
    response leaves TestClosedFormOracle, TestScaleInvariance, TestAggregationContract,
    TestGradientAllocation, TestEdgeRingIsNowCounted, TestGradients, TestDigitalSilence and
    TestReductionSemantics all passing - measured, not assumed.

    Only recorded goldens and the OPTIONAL upstream-parity module caught it, and a suite whose only
    guard on its central feature is a self-recorded number plus an optional dependency is one
    regeneration away from not testing it at all.
    """

    N = 8192

    def _score_for_tone(self, freq):
        torch.manual_seed(11)
        time = torch.arange(self.N, dtype=torch.float32) / SR
        u = torch.rand(1, 1, self.N) * 2 - 1
        t = torch.zeros(1, 1, 1, self.N)
        tone = torch.sin(2 * math.pi * freq * time)
        residual = (tone / tone.pow(2).mean().sqrt() * 0.01).reshape(1, 1, 1, -1)
        return float(_metric()(u, residual, t))

    def test_error_in_the_ear_s_sensitive_band_is_penalised_hardest(self):
        """An equal-energy error at 3 kHz must score WORSE than the same energy at 40 Hz or 16 kHz.

        This is the hearing curve's entire purpose, and it is what a flat response destroys.
        """
        mid = self._score_for_tone(3000.0)
        for freq in (40.0, 200.0, 16000.0):
            with self.subTest(freq=freq):
                self.assertLess(mid, self._score_for_tone(freq),
                                f"error at 3 kHz should be penalised more than at {freq:.0f} Hz; "
                                "a flat weighting would make these equal")

    def test_the_weighting_spans_a_wide_dynamic_range(self):
        """A near-flat response would pass the ordering test above while still being wrong."""
        spread = self._score_for_tone(16000.0) - self._score_for_tone(3000.0)
        self.assertGreater(spread, 5.0,
                           f"only {spread:.2f} units between 3 kHz and 16 kHz; the weighting has "
                           "been flattened")


class TestClassSplit(unittest.TestCase):
    """`LogWMSE` (higher is better) and `LogWMSELoss` (lower is better) replace `return_as_loss`.

    A flag that silently inverts a training objective is not worth the convenience of one import:
    forget it and the model optimises away from the target while every number still looks
    plausible. Two classes make the sign visible at the call site.
    """

    N = 4096

    def _case(self, seed=51):
        torch.manual_seed(seed)
        u = torch.rand(2, 2, self.N) * 2 - 1
        t = torch.rand(2, 2, 3, self.N) * 2 - 1
        return u, t + (torch.rand(2, 2, 3, self.N) * 2 - 1) * 0.1, t

    def test_the_loss_is_the_negated_metric_for_every_reduction(self):
        u, p, t = self._case()
        for reduction in VALID_REDUCTIONS:
            with self.subTest(reduction=reduction):
                metric = LogWMSE(reduction=reduction)
                loss = LogWMSELoss(reduction=reduction)
                self.assertTrue(torch.equal(loss(u, p, t), -metric(u, p, t)))

    def test_per_stem_negates_too(self):
        """Otherwise `loss.per_stem()` would silently disagree in sign with `loss()`."""
        u, p, t = self._case()
        self.assertTrue(torch.equal(LogWMSELoss().per_stem(u, p, t),
                                    -LogWMSE().per_stem(u, p, t)))

    def test_the_loss_is_a_real_subclass(self):
        # So isinstance checks, .to(), state_dict() and every other Module behaviour carry over
        # rather than being reimplemented.
        self.assertIsInstance(LogWMSELoss(), LogWMSE)

    def test_both_are_exported(self):
        import torch_log_wmse

        self.assertEqual(set(torch_log_wmse.__all__), {"LogWMSE", "LogWMSELoss"})

    def test_removed_arguments_raise_by_name(self):
        """Silently ignoring them would let a caller believe a length or a sign had been set."""
        with self.assertRaisesRegex(TypeError, "return_as_loss was removed"):
            LogWMSE(return_as_loss=True)
        with self.assertRaisesRegex(TypeError, "audio_length was removed"):
            LogWMSE(audio_length=1.0)

    def test_the_constructor_is_keyword_only(self):
        """The trap this closes: `audio_length` used to be FIRST.

        A positional call written for an earlier version would otherwise put a duration where
        `sample_rate` now is and quietly build a metric at 1 Hz, which raises nothing and returns
        numbers.
        """
        with self.assertRaises(TypeError):
            LogWMSE(1.0)
        with self.assertRaises(TypeError):
            LogWMSE(1.0, 44100)

    def test_p_is_validated(self):
        for bad in (-1.0, -0.5, float("nan"), float("inf")):
            with self.subTest(p=bad):
                with self.assertRaisesRegex(ValueError, "p must be a finite, non-negative"):
                    LogWMSE(p=bad)
        for good in (0, 0.5, 1, 2.0):
            with self.subTest(p=good):
                self.assertEqual(LogWMSE(p=good).p, float(good))

    def test_p_appears_in_the_repr_along_with_the_sample_rate(self):
        # sample_rate is the one thing that still cannot be inferred from the input, so a caller who
        # has just learned that length is automatic should be able to see the rate is not.
        text = repr(LogWMSE(sample_rate=48000, p=0.5))
        self.assertIn("sample_rate=48000", text)
        self.assertIn("p=0.5", text)


class TestReductionSemantics(unittest.TestCase):
    """`reduction` controls the BATCH axis and nothing else, as in any other torch loss.

    It used to reduce over [batch, channel, stem] together. Channel and stem are now pooled by `p`
    first, so "none" returns one value per batch item rather than one per element, and "sum" is a
    sum over the batch rather than over every element. Both changes are asserted below rather than
    merely observed, because "sum" quietly changing by a factor of channels x stems is exactly the
    sort of thing that reads as a plausible number downstream.

    Kills M13 (mean/sum swapped). The M21 reference here was stale - that mutant patched
    `mean_diff = (differences**2).mean(dim=-1)`, a line the frequency-domain rewrite deleted.
    """

    def test_none_mean_sum_are_mutually_consistent_over_the_batch(self):
        torch.manual_seed(8)
        b, c, s, n = 2, 2, 3, 8192
        u = (torch.rand(b, c, n) * 2 - 1)
        p = (torch.rand(b, c, s, n) * 2 - 1) * 0.1
        t = torch.zeros_like(p)
        none = _metric(reduction="none")(u, p, t)
        mean = float(_metric(reduction="mean")(u, p, t))
        total = float(_metric(reduction="sum")(u, p, t))
        self.assertEqual(none.shape, (b,), "reduction='none' must give one value per batch item")
        self.assertAlmostEqual(float(none.mean()), mean, places=4)
        self.assertAlmostEqual(float(none.sum()), total, places=2)
        # sum/mean is now the BATCH size. It was batch x channels x stems, i.e. 12 for this case.
        self.assertAlmostEqual(total / mean, b, places=3)

    def _graded_batch(self):
        """Per-element quality spread over 30 dB, so pooling rules actually disagree."""
        torch.manual_seed(21)
        b, c, s, n = 3, 2, 4, 4096
        u = torch.rand(b, c, n) * 2 - 1
        levels = torch.logspace(math.log10(0.316), math.log10(0.01), c * s).reshape(1, c, s, 1)
        return u, (u[:, :, None, :] * levels).contiguous(), torch.zeros(b, c, s, n)

    def test_p_zero_pools_as_the_mean_of_the_per_stem_values(self):
        """The identity that makes p=0 exactly the pre-1.0.0 behaviour, and the batch axis untouched.

        Asserted on a GRADED case: with near-equal elements every pooling rule agrees, so an
        equal-quality case would pass whatever p was in force and prove nothing.
        """
        u, p, t = self._graded_batch()
        m = _metric(p=0.0, reduction="none")
        self.assertTrue(
            torch.allclose(m(u, p, t), m.per_stem(u, p, t).mean(dim=(1, 2)), atol=1e-5),
            "at p=0 the pooled value must be the mean of the per-stem values")

    def test_the_default_pools_as_a_plain_mean(self):
        """The default is p=0, so the pooled value IS the mean of the per-stem values.

        This is the compatibility promise: multi-stem scores stay comparable with every version
        before 1.0.0 and with published logWMSE figures. If it fails, the default has moved.
        """
        u, p, t = self._graded_batch()
        m = _metric(reduction="none")
        self.assertEqual(m.p, 0.0, "the default p has changed; this test pins the compatibility promise")
        self.assertTrue(
            torch.allclose(m(u, p, t), m.per_stem(u, p, t).mean(dim=(1, 2)), atol=1e-5))

    def test_p_one_half_is_available_and_genuinely_different(self):
        """The knob has to do something, or exposing it is theatre.

        p=1/2 bounds the per-stem gradient as a stem converges, which is why it is offered; the
        price is that it no longer pools as a plain mean, so its numbers are not comparable with
        published figures.
        """
        u, p, t = self._graded_batch()
        m = _metric(p=0.5, reduction="none")
        gap = float((m(u, p, t) - m.per_stem(u, p, t).mean(dim=(1, 2))).abs().min())
        self.assertGreater(gap, 1.0, "p=0.5 is indistinguishable from the default")

    def test_per_entry_values_survive_batch_reduction(self):
        # The original suite asserted values only for batch size 1; this pins multi-entry reduction.
        torch.manual_seed(9)
        n = 8192
        u = (torch.rand(2, 1, n) * 2 - 1)
        t = torch.zeros(2, 1, 1, n)
        p = torch.stack([u[0] * 0.1, u[1] * 0.01]).unsqueeze(2)
        none = _metric(reduction="none")(u, p, t).flatten()
        self.assertAlmostEqual(float(none[0]), SCALER * math.log(0.1 ** 2), places=2)
        self.assertAlmostEqual(float(none[1]), SCALER * math.log(0.01 ** 2), places=2)
        mean = float(_metric(reduction="mean")(u, p, t))
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
    """Digital silence must not produce NaN anywhere.

    Regression guard for the 0.2.8 divide-by-zero. Its mutant, M14, was retired along with the
    time-domain scaling code it patched, so this test is now the only guard on that behaviour.
    """

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

    def test_grad_carrying_ir_is_rejected(self):
        """A learnable filter is not supported and must fail loudly, not silently.

        Both forms it arrives in are rejected: a plain tensor with requires_grad, and an
        nn.Parameter (which requires grad by default). Left through, the first poisons the weight
        cache - the cached weights hold the first call's graph, so the second backward raises - and
        the second is demoted to a plain tensor by as_tensor().squeeze(), giving zero parameters and
        an empty state_dict with no warning. Detaching is the documented way to use custom values.
        """
        for ir in (torch.rand(4000, requires_grad=True),
                   torch.nn.Parameter(torch.rand(4000))):
            with self.subTest(kind=type(ir).__name__):
                with self.assertRaises(ValueError):
                    self._build(ir)
        # The same values, detached, are accepted - rejection is about grad, not the values.
        self.assertEqual(self._build(torch.rand(4000, requires_grad=True).detach()).impulse_response.numel(),
                         4000)

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
    """A digitally silent, grad-requiring mixture must not poison the graph.

    The original failure: an RMS is the sqrt of a mean square, and at exactly zero sqrt has an
    infinite derivative, so the forward value stayed finite while the backward pass produced NaN.
    Reachable when the "unprocessed" signal is an upstream module's output in a cascaded system.

    1.0.0 closes it twice over. The metric never takes a square root - it clamps a MEAN SQUARE at
    RMS_EPS**2 - and the mixture's level is DETACHED, so no gradient reaches it at all.

    Detaching is a correctness fix, not a convenience. The mixture is the reference the error is
    measured against; left attached it is a gradient path, and `d(loss)/d(log mixture gain)` was
    measured at exactly -8.0. In any setup where the mixture carries gradient, the objective could
    then be reduced by making the MIXTURE LOUDER rather than the estimate better.
    """

    def test_the_mixture_receives_no_gradient(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                u = (torch.rand(1, 1, 4096, dtype=dtype) * 2 - 1).requires_grad_(True)
                t = torch.rand(1, 1, 2, 4096, dtype=dtype) * 2 - 1
                estimate = (t + 0.05).requires_grad_(True)
                _metric(return_as_loss=True)(u, estimate, t).backward()
                self.assertIsNone(u.grad, "the mixture is a reference level, not an optimisation "
                                          "target; a gradient here means the detach was lost")
                self.assertGreater(float(estimate.grad.norm()), 0.0,
                                   "the estimate must still receive gradient")

    def test_a_silent_grad_requiring_mixture_leaves_the_estimate_gradient_finite(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                z = torch.zeros(1, 1, 4096, dtype=dtype, requires_grad=True)
                t = torch.zeros(1, 1, 1, 4096, dtype=dtype)
                estimate = torch.randn(1, 1, 1, 4096, dtype=dtype).requires_grad_(True)
                _metric(return_as_loss=True)(z, estimate, t).backward()
                self.assertTrue(torch.isfinite(estimate.grad).all())

    def test_silent_mixture_scaling_is_pinned_to_rms_eps(self):
        """A silent mixture must scale by exactly 1/RMS_EPS, and the resulting value is pinned.

        This is a value assertion rather than a NaN check on purpose. The original 0.2.8 bug was
        1/0 -> inf -> 0*inf -> NaN, but the subnormal floor inside calculate_rms now prevents an
        exactly-zero RMS independently, so removing the RMS_EPS floor no longer produces NaN -- it
        produces a wrong number instead, tens of units below the correct one. Only pinning the value
        catches that. (The exact mutant value depends on calculate_rms's floor, so it is deliberately
        not quoted here; two fixes guard this failure by different mechanisms and each needs its own
        test, which is what the mutation study established.)

        BOTH VALUES MOVED BY EXACTLY -4*ln(2) = -2.7726 IN 1.0.0, AND THAT IS NOT THE PARSEVAL BUG.
        The estimate here is a CONSTANT, and the weighting filter heavily attenuates DC, so almost
        all of the filtered energy sits in the two edge transients where the constant switches on
        and off. The old centred window captured exactly half of each - a ratio of 2.0000 measured
        at n = 4096, 22050, 44100 and 220500 alike - and dropping the trim recovers the other half.
        A missing one-sided doubling produces the same 2.7726 signature but on EVERY input; this
        appears only on DC-dominated ones. If both this and the broadband cases shift by 2.7726,
        suspect the weights; if only this one does, it is the trim.
        """
        n = 4096
        m = _metric(audio_length=n / SR, reduction="none")
        z = torch.zeros(1, 1, n)
        t = torch.zeros(1, 1, 1, n)
        for level, expected in ((1e-3, -74.70372), (1e-2, -93.12440)):
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

    def test_any_length_is_accepted_by_the_same_instance(self):
        """The replacement for the old length-mismatch error, which no longer exists.

        `audio_length` used to pin one length per instance, and a mismatch had to raise because the
        filtered path silently scored only the first audio_length_samples while bypass_filter=True
        scored everything - the two disagreed by up to 26 units. The transform size is now derived
        from the input, so there is nothing to mismatch.
        """
        for n in (self.n, self.n * 2, 22050, 44100, 12345):
            with self.subTest(n=n):
                got = float(self.m(torch.rand(2, 2, n),
                                   torch.rand(2, 2, 3, n) * 0.1,
                                   torch.zeros(2, 2, 3, n)))
                self.assertTrue(math.isfinite(got))

    def test_the_two_paths_agree_on_which_samples_they_scored(self):
        """The property the old length check was protecting, asserted directly.

        A delta impulse response makes the filtered path mathematically identical to the bypass
        path, so any disagreement about the scored window shows up immediately.
        """
        for n in (1024, 4096, 44100):
            with self.subTest(n=n):
                torch.manual_seed(n)
                u = torch.rand(1, 1, n) * 2 - 1
                p = (torch.rand(1, 1, 1, n) * 2 - 1) * 0.1
                t = torch.zeros_like(p)
                delta = torch.tensor([1.0, 0.0])
                filtered = float(_metric(impulse_response=delta,
                                         impulse_response_sample_rate=SR)(u, p, t))
                bypassed = float(_metric(bypass_filter=True)(u, p, t))
                self.assertAlmostEqual(filtered, bypassed, places=3)

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
    """sample_rate must be validated, not left to fail opaquely later.

    The audio_length checks that used to live here are gone with the argument itself. A whole class
    of error disappeared with it: nothing can now be configured inconsistently with the input,
    because nothing about the input is configured.
    """

    def test_non_positive_sample_rate_raises(self):
        for bad in (0, -44100):
            with self.subTest(sample_rate=bad):
                with self.assertRaisesRegex(ValueError, "sample_rate must be positive"):
                    _metric(audio_length=1.0, sample_rate=bad)

    def test_non_positive_impulse_response_sample_rate_raises(self):
        with self.assertRaisesRegex(ValueError, "impulse_response_sample_rate must be positive"):
            _metric(audio_length=1.0, impulse_response_sample_rate=0)

    def test_one_instance_serves_lengths_that_used_to_need_separate_instances(self):
        """Including the floor()/round() pair that the old length check had to special-case.

        `audio_length * sample_rate` is rarely an exact integer, and floor() and round() differ by
        one sample for 22 of the 399 hundredth-second values at 44.1 kHz. That forced the old
        validation to accept both, which was a workaround for a constraint that no longer exists.
        """
        m = _metric()
        for length in (0.35, 0.57, 0.69, 0.7):
            for n in (math.floor(length * SR), round(length * SR)):
                with self.subTest(audio_length=length, n=n):
                    out = m(torch.rand(1, 1, n), torch.rand(1, 1, 1, n) * 0.1,
                            torch.zeros(1, 1, 1, n))
                    self.assertTrue(math.isfinite(float(out)))


class TestQuietMixturesAreNotRewritten(unittest.TestCase):
    """The mixture floor must engage only for genuinely degenerate input, not for quiet audio.

    This used to be about a hard-coded floor inside an RMS helper: a literal such as 1e-24
    underflows to 0.0 in float16, making the guard a silent no-op there, and it also rewrote any
    legitimate mean square below it - float32 holds mean squares down to about 1e-38, so a fixed
    1e-24 altered roughly seven decades of genuinely quiet audio.

    The helper is gone in 1.0.0, so the property is asserted where it now lives: RMS_EPS**2 is a
    floor on the MIXTURE's mean square, and a mixture above it must scale exactly as its own level
    dictates. Scale invariance is the observable form of that.
    """

    def test_scale_invariance_holds_for_very_quiet_mixtures(self):
        # RMS_EPS is 1e-8, so a mixture at 1e-6 is a hundred times above the floor: quiet, but not
        # degenerate, and its score must match the same material at unit level.
        torch.manual_seed(60)
        u = torch.rand(1, 1, 4096) * 2 - 1
        p = (torch.rand(1, 1, 1, 4096) * 2 - 1) * 0.1
        reference = float(_metric()(u, p, torch.zeros_like(p)))
        for amplitude in (1e-2, 1e-4, 1e-6):
            with self.subTest(amplitude=amplitude):
                got = float(_metric()(u * amplitude, p * amplitude, torch.zeros_like(p)))
                self.assertAlmostEqual(got, reference, places=2)

    def test_the_floor_engages_only_below_rms_eps(self):
        """Below the floor, scale invariance is expected to STOP holding - that is what a floor is."""
        torch.manual_seed(61)
        u = torch.rand(1, 1, 4096) * 2 - 1
        p = (torch.rand(1, 1, 1, 4096) * 2 - 1) * 0.1
        reference = float(_metric()(u, p, torch.zeros_like(p)))
        got = float(_metric()(u * 1e-10, p * 1e-10, torch.zeros_like(p)))
        self.assertGreater(abs(got - reference), 1.0)


class TestModuleContract(unittest.TestCase):
    """The metric must behave like any other nn.Module: `.to()` moves it, and it carries no state.

    The filter used to be a plain class, so `.to(device)` could not reach its tensors and `forward`
    compensated by comparing devices and reassigning module state mid-pass. Both halves of that are
    now gone, and these tests are what keep them gone.
    """

    N = 1024

    def _triplet(self, device="cpu"):
        # Generated on the CPU and MOVED, never generated on the target device: torch.rand seeds a
        # separate generator per device, so `torch.rand(..., device="mps")` after manual_seed gives
        # different numbers to the CPU call. Comparing those would compare different inputs and
        # report a ~0.5-unit "divergence" that is entirely the test's own doing.
        torch.manual_seed(71)
        out = (torch.rand(1, 1, self.N), torch.rand(1, 1, 2, self.N), torch.rand(1, 1, 2, self.N))
        return tuple(x.to(device) for x in out)

    def test_state_dict_is_empty(self):
        """Non-persistent buffers, so a model holding this as a submodule gains no checkpoint keys.

        If they became persistent, every downstream `load_state_dict(strict=True)` against a
        checkpoint saved before the change would fail on unexpected keys.
        """
        self.assertEqual(dict(_metric(audio_length=self.N / SR).state_dict()), {})

    def test_strict_load_round_trips_for_a_model_holding_the_metric(self):
        """The failure the previous test exists to prevent, exercised end to end."""
        class Model(torch.nn.Module):
            def __init__(self, n):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(3))
                self.loss = _metric(audio_length=n / SR)

        a, b = Model(self.N), Model(self.N)
        self.assertEqual(list(a.state_dict()), ["weight"])
        b.load_state_dict(a.state_dict(), strict=True)  # must not raise

    def test_to_moves_the_impulse_response_without_a_forward_pass(self):
        """`.to()` alone must relocate the buffers - no call needed to trigger it.

        The meta device is used because it is available everywhere, so this holds even where no
        accelerator exists. MPS covers the real thing below.
        """
        m = _metric(audio_length=self.N / SR)
        self.assertEqual(m.filters.impulse_response.device.type, "cpu")
        moved = m.to("meta")
        self.assertEqual(moved.filters.impulse_response.device.type, "meta")

    def test_half_and_bfloat16_do_not_corrupt_the_impulse_response(self):
        """`.half()` on a PARENT module must not destroy the filter.

        Every tap of the hearing curve has magnitude below 1, so casting the IR to float16 flushes
        more than half of them to exactly zero and bfloat16 mangles the rest. `_weight_dtype` already
        keeps the derived WEIGHTS at float32; this keeps the IR itself there too, so a `model.half()`
        that sweeps the loss up as a submodule leaves the filter intact. The supported
        mixed-precision path is torch.autocast, which never downcasts the module at all.

        The corruption divides out of a broadband energy ratio, so it is checked at the buffer, not
        through a score - a functional check would be the degenerate case this project keeps hitting.
        A device-only move must still apply and must NOT warn; that is covered by the `.to()` tests.
        """
        reference = _metric(audio_length=self.N / SR).filters.impulse_response.clone()

        for cast in ("half", "bfloat16"):
            with self.subTest(cast=cast):
                class Model(torch.nn.Module):
                    def __init__(self, n):
                        super().__init__()
                        self.loss = _metric(audio_length=n / SR)

                model = Model(self.N)
                with self.assertWarns(UserWarning):
                    getattr(model, cast)()
                ir = model.loss.filters.impulse_response
                self.assertEqual(ir.dtype, torch.float32,
                                 f"{cast}() downcast the impulse response to {ir.dtype}")
                self.assertTrue(torch.equal(ir, reference),
                                f"{cast}() changed impulse-response values")

    def test_forward_does_not_mutate_module_state(self):
        """No device reassignment, and nothing else rebound during the pass.

        Mutating module state in forward is not thread-safe and is awkward for torch.compile and
        graph capture, which is why the old compensation had to go rather than merely be tidied.
        """
        m = _metric(audio_length=self.N / SR)
        before = m.filters.impulse_response.data_ptr()
        m(*self._triplet())
        self.assertEqual(before, m.filters.impulse_response.data_ptr(), "forward rebound a buffer")

    def test_the_weight_cache_is_not_a_buffer(self):
        """DDP enumerates named_buffers() to build its broadcast list.

        A cache that grows lazily would give ranks divergent buffer lists - they process different
        data, so they reach different lengths at different times - and the collective then either
        hangs or mismatches. Non-persistent does not help: those still appear in named_buffers().
        """
        m = _metric(audio_length=self.N / SR)
        m(*self._triplet())
        self.assertTrue(m.filters._weights, "nothing was cached, so this test proves nothing")
        names = [n for n, _ in m.named_buffers()]
        self.assertEqual(names, ["filters.impulse_response"],
                         f"cache entries leaked into named_buffers(): {names}")

    @unittest.skipUnless(torch.backends.mps.is_available(), "no MPS device")
    def test_mps_matches_cpu(self):
        """A real non-CPU device: movement, execution, and agreement.

        MPS is the only accelerator available here - there is no CUDA device - so it is what stands
        in for "works off the CPU". It cannot cover float64 or half-precision FFT, and those limits
        are stated where they apply rather than assumed away.
        """
        m = _metric(audio_length=self.N / SR)
        cpu = float(m(*self._triplet()))
        gpu = float(m.to("mps")(*self._triplet("mps")))
        self.assertAlmostEqual(gpu, cpu, delta=1e-4, msg=f"mps {gpu} vs cpu {cpu}")


class TestEdgeRingIsNowCounted(unittest.TestCase):
    """The one semantic change in the filtering rewrite, pinned rather than left implicit.

    The metric moved from "energy of a window the same length as the input" to "energy of the full
    linear convolution", so the filter's ring-in and ring-out at the buffer edges are now counted
    instead of discarded.

    The user-visible consequence is an improvement, and it is the reason to state the change as a
    feature rather than an erratum: AN IDENTICAL ERROR NOW SCORES THE SAME WHEREVER IT OCCURS IN
    THE BUFFER. The old trimmed window kept only the half of the filter's ring that happened to
    fall inside it, so the same single-sample error read 0.7748 units BETTER at the buffer boundary
    than in the middle - purely because half its energy was discarded. Measured identically at
    n = 2048, 4096, 44100 and 220500, because the discarded pre-ring was a fixed fraction of that
    transient's own energy rather than of the window's; 100 samples in it was already down to
    0.0008, and by half the impulse response's length it was 0.0000.

    So the size of the difference from the old implementation depends on POSITION far more than on
    length or sparsity. "Sparse residuals shift" is misleading; "residuals at the buffer edges
    shift" is accurate.
    """

    N = 8192

    def _score_with_impulse_at(self, position, **kw):
        torch.manual_seed(77)
        u = torch.rand(1, 1, self.N) * 2 - 1
        t = torch.zeros(1, 1, 1, self.N)
        p = torch.zeros(1, 1, 1, self.N)
        p[0, 0, 0, position] = 0.01
        return float(_metric(**kw)(u, p, t))

    def test_the_score_no_longer_depends_on_where_the_residual_sits(self):
        """The property the trim used to break. Under it, position 0 read 0.7748 units better."""
        scores = {pos: self._score_with_impulse_at(pos)
                  for pos in (0, 1, 100, 2000, self.N // 2, self.N - 1)}
        spread = max(scores.values()) - min(scores.values())
        self.assertLess(spread, 1e-3,
                        f"an identical error scores differently by position: {scores}")

    def test_it_holds_for_a_burst_as_well_as_a_single_sample(self):
        torch.manual_seed(78)
        burst = torch.rand(64) * 0.02
        scores = []
        for start in (0, 1000, self.N // 2, self.N - 64):
            u = torch.rand(1, 1, self.N) * 2 - 1
            t = torch.zeros(1, 1, 1, self.N)
            p = torch.zeros(1, 1, 1, self.N)
            p[0, 0, 0, start:start + 64] = burst
            torch.manual_seed(77)
            u = torch.rand(1, 1, self.N) * 2 - 1
            scores.append(float(_metric()(u, p, t)))
        self.assertLess(max(scores) - min(scores), 1e-3, f"burst position changed the score: {scores}")


class TestTransformSizing(unittest.TestCase):
    """Size selection, and the cache keyed on it."""

    def test_next_size_is_the_smallest_even_smooth_value(self):
        """Checked against brute force, because an off-by-one here is silent: any size at or above
        the requirement is CORRECT, just slower, so only a size BELOW it breaks anything."""
        def brute(n):
            k = max(n, 2)
            while True:
                if k % 2 == 0:
                    r = k
                    for prime in (2, 3, 5):
                        while r % prime == 0:
                            r //= prime
                    if r == 1:
                        return k
                k += 1

        for n in list(range(2, 400)) + [4096, 12345, 48099, 92199, 65537]:
            with self.subTest(n=n):
                self.assertEqual(next_fft_friendly_size(n), brute(n))

    def test_the_headline_size_beats_the_power_of_two(self):
        # 1 s at 44.1 kHz needs 48099 samples: 48600 as a 5-smooth number against 65536 as a power
        # of two, cutting padding waste from 36% to 1%.
        self.assertEqual(next_fft_friendly_size(48099), 48600)

    def test_transform_is_never_shorter_than_the_linear_convolution(self):
        """`L >= n + m - 1` is correctness, not performance: Parseval over a shorter L returns the
        CIRCULAR convolution energy, which used to show as visible wrap-around and is now silent."""
        f = make_filter()
        m = f.impulse_response.shape[-1]
        for n in (1, 2, 100, 1024, 4096, 44100, 62000, 220500):
            with self.subTest(n=n):
                self.assertGreaterEqual(f.transform_size(n), n + m - 1)

    def test_lengths_that_share_a_transform_size_share_one_cache_entry(self):
        """Keyed on the transform size, not the input length.

        Keying on the input length is not wrong - the weights depend only on the transform size, so
        both give identical numbers - but it silently rebuilds the weights for every new length,
        which is the entire cost the cache exists to avoid. Only an entry count can see it.
        """
        f = make_filter()
        a, b = 44100, 44101
        self.assertEqual(f.transform_size(a), f.transform_size(b),
                         "pick two lengths that actually share a size, or this proves nothing")
        for n in (a, b):
            f(torch.rand(1, 1, 1, n))
        self.assertEqual(len(f._weights), 1, f"expected one cache entry, got {len(f._weights)}")

    def test_dtype_is_part_of_the_cache_key(self):
        """A float32 entry reused for a float64 call would silently downgrade the whole computation,
        and the float64 gradcheck along with it - while still returning a float64 tensor."""
        n = 1024
        shared = make_filter()
        x32 = torch.rand(1, 1, 1, n)
        shared(x32)                                   # populate with float32 first
        reused = shared(x32.double())
        fresh = make_filter()(x32.double())
        self.assertEqual(reused.dtype, torch.float64)
        self.assertTrue(torch.equal(reused, fresh),
                        "a float64 call after a float32 call did not match a fresh instance")
        self.assertEqual(len(shared._weights), 2)

    def test_the_weights_themselves_are_built_in_the_input_dtype(self):
        """Not just the cache KEY - the weight values.

        Building them from the float32 impulse response and letting promotion carry them into a
        float64 computation gives a float64 RESULT holding only float32 precision. Nothing
        observable changes by more than about 1e-8 relative, which is under the noise floor of every
        behavioural case here, so the property has to be asserted directly. This gap was found by a
        surviving mutant, not by a failing test.
        """
        f = make_filter()
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                f(torch.rand(1, 1, 1, 1024, dtype=dtype))
                weights = f.weights_for(1024, dtype)
                self.assertEqual(weights.dtype, dtype,
                                 f"weights for a {dtype} input were built as {weights.dtype}")

    def test_the_cache_is_capped_and_evicts_the_OLDEST_entry(self):
        """The cap alone is not worth asserting - a cache that stored nothing would satisfy it.

        Eviction ORDER is the part that matters and the part that was untested: swapping the FIFO
        `pop(next(iter(...)))` for `popitem()` turns an LRU into a LIFO that thrashes on any
        variable-length workload, and it used to pass the entire suite unchanged.
        """
        f = make_filter(cache_size=3)
        lengths = [1000, 2000, 4000, 8000, 16000, 32000]
        for n in lengths:
            f(torch.rand(1, 1, 1, n))
        self.assertEqual(len(f._weights), 3)
        # The three most RECENT sizes must be what survived.
        survived = {key[0] for key in f._weights}
        self.assertEqual(survived, {f.transform_size(n) for n in lengths[-3:]},
                         "eviction is not first-in-first-out; the newest entries were discarded")

    def test_eviction_tolerates_a_key_that_vanishes_mid_pop(self):
        """The cache is mutated inside forward, so eviction must survive a concurrent delete.

        Check-then-pop - `pop(next(iter(d)))` - is not atomic: a second thread can empty the dict
        between choosing the oldest key and popping it, and the pop then raises KeyError (or the
        lookup raises StopIteration on an empty dict). It reproduces under threads only with a tiny
        switch interval, which makes a threaded test flaky, so the exact interleaving is forced
        deterministically here with a dict that deletes each key as it is yielded. The fixed path
        uses defaulted pops and must not raise; the old path raised KeyError.
        """
        class VanishingDict(dict):
            def __iter__(self):
                for k in list(super().__iter__()):
                    self.pop(k, None)  # vanish before the caller can act on the key
                    yield k

        f = make_filter(cache_size=1)
        f(torch.rand(1, 1, 1, 1000))                 # one entry, so the next call evicts
        f._weights = VanishingDict(f._weights)
        f(torch.rand(1, 1, 1, 2000))                 # eviction runs against the vanishing dict
        # The new entry is still cached; no exception was raised reaching this line.
        self.assertIn(f.transform_size(2000), {key[0] for key in f._weights})

    def test_cache_size_is_validated(self):
        for bad in (0, -1):
            with self.subTest(cache_size=bad):
                with self.assertRaisesRegex(ValueError, "cache_size must be at least 1"):
                    make_filter(cache_size=bad)

    def test_moving_or_recasting_the_module_drops_the_cache(self):
        """A `.half()` after a forward pass used to leave float32-derived weights cached under a
        key that no longer described them, and the next call silently reused them."""
        f = make_filter()
        f(torch.rand(1, 1, 1, 2048))
        self.assertEqual(len(f._weights), 1)
        f.to(torch.float64)
        self.assertEqual(len(f._weights), 0, "cache survived a dtype change")

    def test_a_single_instance_reproduces_per_length_instances(self):
        """The whole point of dropping audio_length: one instance, any length, same numbers."""
        shared = _metric()
        for n in (512, 4096, 22050, 44100, 62000):
            with self.subTest(n=n):
                torch.manual_seed(n)
                u = torch.rand(1, 1, n) * 2 - 1
                p = (torch.rand(1, 1, 2, n) * 2 - 1) * 0.1
                t = torch.zeros_like(p)
                self.assertAlmostEqual(float(shared(u, p, t)), float(_metric()(u, p, t)), places=6)


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
    # Both filter paths, both float dtypes the filter supports, and three pooling exponents. float16
    # cannot reach the filtering path at all (no half FFT kernel on CPU or MPS) and is covered
    # separately. p is in the sweep because these properties have to hold for EVERY p - that is what
    # makes them the contract rather than a description of one setting, and it is what lets `p` be
    # changed from its default of 0 with the contract already proven at 0, 1/2 and 1.
    VARIANTS = [(dt, bp, p)
                for dt in (torch.float32, torch.float64)
                for bp in (False, True)
                for p in (0.0, 0.5, 1.0)]

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
        for dtype, bypass, power in self.VARIANTS:
            with self.subTest(dtype=dtype, bypass_filter=bypass, p=power):
                u, p, t = self._graded(levels, dtype)
                kw = dict(bypass_filter=bypass, p=power)
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
        for dtype, bypass, power in self.VARIANTS:
            kw = dict(bypass_filter=bypass, p=power)
            before = float(_metric(**kw)(*self._graded(base, dtype)))
            for c in range(2):
                for s in range(2):
                    with self.subTest(dtype=dtype, bypass_filter=bypass, p=power, element=(c, s)):
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
        for dtype, bypass, power in self.VARIANTS:
            with self.subTest(dtype=dtype, bypass_filter=bypass, p=power):
                torch.manual_seed(43)
                # Identical mixture across channels AND an identical residual on every element, so
                # the per-element values are bit-identical rather than merely close.
                u = ((torch.rand(1, 1, self.N) * 2 - 1).to(dtype)).expand(1, 2, self.N).contiguous()
                t = (torch.rand(1, 2, 3, self.N) * 2 - 1).to(dtype)
                d = (torch.rand(1, 1, 1, self.N) * 2 - 1).to(dtype) * 0.05
                p = t + d

                kw = dict(bypass_filter=bypass, p=power)
                elements = per_element(u, p, t, **kw).flatten()
                self.assertLess(float(elements.max() - elements.min()), 1e-4,
                                "the construction is not actually equal-quality")
                self.assertAlmostEqual(float(_metric(**kw)(u, p, t)), float(elements[0]), places=4)

    def test_single_element_identity(self):
        """Pooling one element is the identity - the mono, single-stem denoising case.

        Guards against an aggregation change silently altering the most common usage of all.
        """
        for dtype, bypass, power in self.VARIANTS:
            with self.subTest(dtype=dtype, bypass_filter=bypass, p=power):
                u, p, t = self._graded([[0.08]], dtype)
                kw = dict(bypass_filter=bypass, p=power)
                element = float(per_element(u, p, t, **kw).flatten()[0])
                self.assertAlmostEqual(float(_metric(**kw)(u, p, t)), element, places=5)

    # Small enough that a power mean is nearly a geometric mean, spread over a decade each step so
    # the trend is a trend and not two points and a tolerance.
    SMALL_P = (0.1, 0.01, 0.001)

    def test_the_score_is_continuous_in_p_at_zero(self):
        """A p that is merely small must land near the p = 0 value, and closer the smaller it gets.

        This is the seam test for pooling. `p = 0` is special-cased rather than computed as a limit,
        so the two branches meet at a seam, and the power mean being continuous at 0 - its limit
        there IS the geometric mean - is what says the seam is closed.

        What it catches in practice is EPS placed AFTER pooling rather than before. That misplacement
        is invisible almost everywhere: with every stem imperfect the two forms agree to ~1e-3 units,
        and at p >= 1/2 they agree to 0.008 even WITH a perfect stem. It only becomes large as p
        approaches 0, which is exactly where nothing was looking once the default moved to p = 0 and
        stopped exercising this branch at all. Measured gap against the p = 0 value at p = 0.01:
        0.40 units correct, 34.9 with EPS outside the pool - and outside the pool the gap GROWS as p
        shrinks rather than closing, which is what the monotonic assertion below pins.

        A BIT-EXACT STEM is what makes the property bite, since EPS only matters where mse is
        exactly 0. Hence the ceiling check first: without it this test would still pass while
        asserting nothing.

        Deliberately a value test, not a gradient test. The gradient at a bit-exact stem is exactly
        zero however EPS is placed, because the chain back to the waveform carries a factor of the
        residual and that residual is zero. Only the VALUE can see this.

        `bypass_filter` is not swept: `_graded` builds each residual as a scaled copy of the mixture,
        so the weighting cancels between numerator and denominator and both paths return the same
        numbers to the last bit. The filter has its own tests.
        """
        levels = [[0.0, 0.05, 0.01], [0.1, 0.02, 0.003]]
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                u, p, t = self._graded(levels, dtype)

                elements = per_element(u, p, t, p=0.0).flatten()
                self.assertAlmostEqual(
                    float(elements.max()), CEILING, places=3,
                    msg="no stem is bit-exact, so EPS placement is unobservable and this is vacuous")

                base = float(_metric(p=0.0)(u, p, t))
                gaps = [abs(float(_metric(p=power)(u, p, t)) - base) for power in self.SMALL_P]
                for (p_hi, gap_hi), (p_lo, gap_lo) in zip(zip(self.SMALL_P, gaps),
                                                          zip(self.SMALL_P[1:], gaps[1:])):
                    self.assertLess(gap_lo, gap_hi,
                                    f"gap to p=0 grew from {gap_hi:.4g} at p={p_hi} to "
                                    f"{gap_lo:.4g} at p={p_lo}; it must close as p -> 0")
                # Measured 0.041 here. The bound is loose on purpose - the assertion that discriminates
                # is the one above; this one only stops the trend from converging to the wrong place.
                self.assertLess(gaps[-1], 0.5,
                                f"p={self.SMALL_P[-1]} sits {gaps[-1]:.4g} from the p=0 value")


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
                m = _metric(bypass_filter=True)
                u, p, t = self._triplet(dtype)
                got = m.per_stem(u, p, t)
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
        is not.

        The trap did reappear when pooling gained a fractional power, exactly as this test was
        written to anticipate. At p = 1/2 the derivative of sqrt is infinite at zero, so putting EPS
        outside the pool gives `grad = [nan, ...]` on a stem matched bit-for-bit - which is the
        digital-silence case the package exists for - while the forward value stays finite.

        float32 and float64 are swept alongside the half dtypes: nothing about that failure is
        low-precision-specific, and the original version of this test only covered the half dtypes
        because they were what motivated it.
        """
        for dtype in self.HALF_DTYPES + (torch.float32, torch.float64):
            for p_exponent in (0.0, 0.5, 1.0):
                with self.subTest(dtype=dtype, p=p_exponent):
                    m = _metric(bypass_filter=True, p=p_exponent)
                    u, p, t = self._triplet(dtype)
                    p = p.detach().requires_grad_(True)
                    m(u, p, t).backward()
                    self.assertTrue(torch.isfinite(p.grad).all(), f"{dtype} p={p_exponent}: {p.grad}")


if __name__ == "__main__":
    unittest.main()
