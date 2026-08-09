"""Characterization tests: the recorded behaviour of the metric across a 36-case input matrix.

The rest of the suite asserts properties. This module asserts VALUES, which is a different and
complementary job: the 1.0.0 redesign moves values in exactly two places on purpose, and the only
way to show it moves them nowhere else is to have written down what they were.

Two levels of strictness, deliberately:

* **This module** compares with a tolerance, because it runs in CI on a different BLAS to the
  machine the values were recorded on, and float32 FFT reductions are not bit-identical across
  implementations. 1e-4 is the same tolerance the upstream-parity module uses successfully against
  an entirely separate implementation, and it is still 80x tighter than the smallest delta any
  planned step is expected to produce.

* **The dry run**, on the machine doing the work, is the bit-identical check:

      OMP_NUM_THREADS=2 .audit/venv/bin/python -m tests.golden_cases

  For any step that claims to preserve values, that must print "no change across N cases" - which
  is agreement to 1e-6, not to 1e-4. Use it before every commit in steps 2 and 4.

Never regenerate to make this pass. Read the deltas: every step's expected change is stated in the
implementation plan, and anything else is a bug.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from tests.conftest import CEILING
from tests.golden_cases import GOLDEN_PATH, cases, evaluate, load

TOL = 1e-4
GRAD_RTOL = 1e-3


class TestGoldenValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = load()

    def test_every_case_reproduces(self):
        for spec in cases():
            name = spec["name"]
            with self.subTest(case=name):
                self.assertIn(name, self.golden,
                              f"{name} has no recorded value; regenerate with "
                              f"`python -m tests.golden_cases --write` after reviewing the delta")
                want, got = self.golden[name], evaluate(spec)

                self.assertAlmostEqual(
                    got["pooled"], want["pooled"], delta=TOL,
                    msg=f"{name}: pooled {got['pooled']} vs recorded {want['pooled']} "
                        f"(delta {got['pooled'] - want['pooled']:+.6f})")

                flat_got = _flatten(got["per_element"])
                flat_want = _flatten(want["per_element"])
                self.assertEqual(len(flat_got), len(flat_want), f"{name}: shape changed")
                for i, (a, b) in enumerate(zip(flat_got, flat_want)):
                    self.assertAlmostEqual(
                        a, b, delta=TOL,
                        msg=f"{name}: element {i} {a} vs recorded {b} (delta {a - b:+.6f})")

                if want["grad_norms"] is not None:
                    self.assertIsNotNone(got["grad_norms"], f"{name}: gradients no longer recorded")
                    for i, (a, b) in enumerate(zip(got["grad_norms"], want["grad_norms"])):
                        self.assertAlmostEqual(
                            a, b, delta=max(GRAD_RTOL * abs(b), TOL),
                            msg=f"{name}: grad norm {i} {a} vs recorded {b}")

    def test_recorded_file_has_no_orphans(self):
        """A case removed from the matrix but left in the file would quietly stop being checked."""
        extra = sorted(set(self.golden) - {s["name"] for s in cases()})
        self.assertEqual(extra, [], f"{GOLDEN_PATH} has entries with no matching case: {extra}")


class TestGoldenMatrixIsMeaningful(unittest.TestCase):
    """Guards on the case matrix itself. A characterization suite that records degenerate values is
    worse than none: it looks like coverage and asserts nothing."""

    @classmethod
    def setUpClass(cls):
        cls.golden = load()

    def test_residual_shape_cases_are_clear_of_the_ceiling(self):
        """The trap this caught once already.

        Residual-shape cases exist to measure what dropping the time-domain trim costs for
        differently concentrated residuals. Placed too quietly, the -68 dB inaudibility gate zeroes
        them and every case pins to +73.68 - measuring the gate, not the shape, and guaranteeing a
        huge spurious delta the moment the gate is removed.
        """
        for name, rec in self.golden.items():
            if name.startswith("shape_"):
                with self.subTest(case=name):
                    self.assertLess(
                        rec["pooled"], CEILING - 20.0,
                        f"{name} sits at {rec['pooled']:.2f}, near the {CEILING:.2f} ceiling: the "
                        "residual is below the inaudibility gate and this case measures nothing")

    def test_graded_cases_have_a_wide_per_element_spread(self):
        """Pooling is only exercised when the elements DIFFER; every rule agrees when they match."""
        for name, rec in self.golden.items():
            if name.startswith("graded_"):
                with self.subTest(case=name):
                    flat = _flatten(rec["per_element"])
                    self.assertGreater(max(flat) - min(flat), 20.0,
                                       f"{name} spread collapsed to {max(flat) - min(flat):.2f}")

    def test_gain_cases_are_scale_invariant(self):
        """Recorded rather than derived, so a regression in scale invariance shows up as a value."""
        self.assertAlmostEqual(self.golden["gain0.001"]["pooled"],
                               self.golden["gain1000"]["pooled"], delta=TOL)

    def test_recorded_values_are_all_finite(self):
        for name, rec in self.golden.items():
            with self.subTest(case=name):
                values = _flatten(rec["per_element"]) + [rec["pooled"]] + (rec["grad_norms"] or [])
                self.assertTrue(all(_finite(v) for v in values), f"{name} recorded a non-finite value")


class TestPooledMatchesPerElement(unittest.TestCase):
    """Today pooling is the mean over every axis, so the pooled value IS the mean of the elements.

    This survives the redesign as `pooled(p=0) == per_stem().mean(dim=(1,2))`, which is the exact
    identity that makes the pooling machinery landable at p=0 with a bit-exact gate before the
    default moves to 1/2.
    """

    def test_pooled_is_the_mean_of_the_elements(self):
        for name, rec in load().items():
            with self.subTest(case=name):
                flat = _flatten(rec["per_element"])
                self.assertAlmostEqual(rec["pooled"], sum(flat) / len(flat), delta=1e-4)


def _flatten(x):
    if isinstance(x, list):
        return [v for item in x for v in _flatten(item)]
    return [x]


def _finite(v):
    return v == v and abs(v) != float("inf")


if __name__ == "__main__":
    unittest.main()
