"""The golden case matrix: input SPECS in code, expected VALUES in tests/goldens.json.

Why the split. The 1.0.0 redesign moves values in exactly two places on purpose (the filtering
rewrite, and the pooling default) and must move them nowhere else. Proving that needs a recorded
baseline covering enough of the input space that an accidental change has nowhere to hide.

Inputs are regenerated from a seed rather than stored, so the file stays small and diffable - which
matters, because the release gate is "review each delta", not "regenerate and move on".

Both PER-ELEMENT and POOLED values are recorded. Pooled-only would conflate "the filter changed"
with "the pooling changed", destroying the attributability that the whole step ordering exists to
buy: per-element values survive the pooling change, pooled values do not.

Regenerate with:

    OMP_NUM_THREADS=2 .audit/venv/bin/python -m tests.golden_cases --write

Never regenerate to make a test pass. Read the printed diff first: the expected shape of a change
is stated in the implementation plan for every step, and anything else is a bug.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from tests.conftest import SR, make_metric

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens.json")

# Values are recorded to this many decimals. float32 FFT convolution reproduces to roughly 1e-6 on
# the log-domain metric, so 6 decimals records everything real and nothing that is pure noise.
PLACES = 6


# --------------------------------------------------------------------------------------------
# Input builders. Every one is deterministic given the spec.
# --------------------------------------------------------------------------------------------

def _rand(shape, seed, scale=1.0):
    """U(-1, 1) * scale. torch's generator only - numpy is not a dependency of the package."""
    torch.manual_seed(seed)
    return (torch.rand(*shape) * 2 - 1) * scale


def _residual(shape, kind, seed):
    """A residual of a given SHAPE, normalised to UNIT RMS. The caller sets the level.

    The shape matters because dropping the time-domain trim in the filtering rewrite discards the
    filter's pre-ring, and how much energy that costs depends entirely on how concentrated the
    residual is: negligible for white noise, ~0.775 score units for a single sample.

    Unit RMS rather than a fixed absolute energy, so the caller can place the residual well above
    the -68 dB inaudibility gate. Spreading a fixed small energy over a long window puts it BELOW
    that gate, which zeroes it and pins every case to the ceiling - measuring the gate instead of
    the shape, and guaranteeing a huge spurious delta when the gate is removed.
    """
    n = shape[-1]
    if kind == "white":
        r = _rand(shape, seed)
    elif kind == "tone":
        t = torch.arange(n, dtype=torch.float32) / SR
        r = torch.sin(2 * math.pi * 10000.0 * t).expand(shape).clone()
    elif kind == "click_start":
        r = torch.zeros(shape)
        r[..., :64] = _rand((*shape[:-1], 64), seed)
    elif kind == "click_end":
        r = torch.zeros(shape)
        r[..., -64:] = _rand((*shape[:-1], 64), seed)
    elif kind == "single":
        r = torch.zeros(shape)
        r[..., n // 2] = 1.0
    else:
        raise ValueError(f"unknown residual shape {kind!r}")
    return r / max(float(r.pow(2).mean().sqrt()), 1e-30)


def build(spec):
    """spec -> (unprocessed [b,c,t], processed [b,c,s,t], target [b,c,s,t])."""
    b, c, s, n = spec["batch"], spec["channels"], spec["stems"], spec["n"]
    seed, kind = spec["seed"], spec["kind"]
    u = _rand((b, c, n), seed)

    if kind == "scaled":
        # est = k * mixture against a silent target. Closed form: SCALER * ln(k^2 + EPS).
        p = (u[:, :, None, :] * spec["k"]).expand(b, c, s, n).clone()
        return u, p, torch.zeros(b, c, s, n)

    if kind == "graded":
        # A DIFFERENT level per (channel, stem), so pooling is actually exercised. Every aggregation
        # rule agrees when the elements are equal, which is what made the old stereo test blind.
        levels = torch.logspace(math.log10(spec["k_hi"]), math.log10(spec["k_lo"]), c * s)
        p = u[:, :, None, :] * levels.reshape(1, c, s, 1)
        return u, p.contiguous(), torch.zeros(b, c, s, n)

    if kind == "independent":
        return u, _rand((b, c, s, n), seed + 1), _rand((b, c, s, n), seed + 2)

    if kind == "exact":
        t = _rand((b, c, s, n), seed + 1)
        return u, t.clone(), t

    if kind == "silence":
        return torch.zeros(b, c, n), torch.zeros(b, c, s, n), torch.zeros(b, c, s, n)

    if kind == "leakage":
        # Silent target with a quiet estimate: the package's headline case.
        return u, _rand((b, c, s, n), seed + 1, spec["k"]), torch.zeros(b, c, s, n)

    if kind == "gain":
        # Joint gain on all three inputs. The metric is scale-invariant, so this must not move.
        g = spec["g"]
        return u * g, _rand((b, c, s, n), seed + 1) * g, _rand((b, c, s, n), seed + 2) * g

    if kind == "shape":
        # `level` is the residual's RMS relative to the mixture's, i.e. the residual level in dB.
        t = _rand((b, c, s, n), seed + 1, 0.2)
        r = _residual((b, c, s, n), spec["shape"], seed + 3)
        return u, t + r * spec["level"] * float(u.pow(2).mean().sqrt()), t

    if kind == "exact_one_stem":
        # One stem bit-exact, the rest not. This is the case that makes a badly placed EPS produce a
        # NaN gradient, and a forward-only test cannot see it.
        t = _rand((b, c, s, n), seed + 1)
        p = t + _rand((b, c, s, n), seed + 2, 0.05)
        p[:, :, 0] = t[:, :, 0]
        return u, p, t

    raise ValueError(f"unknown case kind {spec['kind']!r}")


# --------------------------------------------------------------------------------------------
# The matrix.
# --------------------------------------------------------------------------------------------

def _spec(name, kind, *, batch=1, channels=1, stems=1, n=SR, seed=0, grad=False, config=None, **kw):
    return dict(name=name, kind=kind, batch=batch, channels=channels, stems=stems, n=n,
                seed=seed, grad=grad, config=config or {}, **kw)


def cases():
    out = []

    # Residual level, -10 dB down to -60 dB. The closed form is SCALER*ln(k^2 + EPS), so these
    # double as an analytic cross-check on the whole pipeline.
    for db in (-10, -20, -30, -40, -50, -60):
        out.append(_spec(f"level{db}", "scaled", k=10 ** (db / 20.0), seed=1))

    # Shape of the input: channels, stems, batch. Cross-covered rather than fully crossed.
    out += [
        _spec("mono_1stem", "independent", batch=1, channels=1, stems=1, seed=2),
        _spec("stereo_1stem", "independent", batch=1, channels=2, stems=1, seed=2),
        _spec("stereo_2stem", "independent", batch=1, channels=2, stems=2, seed=2),
        _spec("stereo_4stem", "independent", batch=1, channels=2, stems=4, seed=2),
        _spec("batch4_stereo_4stem", "independent", batch=4, channels=2, stems=4, seed=3),
    ]

    # Unequal per-element quality. The pooled value here is what the p-flip moves; the per-element
    # values are what it must leave alone.
    out += [
        _spec("graded_stereo_2stem", "graded", channels=2, stems=2, k_hi=0.316, k_lo=0.01, seed=4),
        _spec("graded_stereo_4stem", "graded", channels=2, stems=4, k_hi=0.5, k_lo=0.003, seed=4,
              grad=True),
    ]

    # Length. Sub-second, one second, and one whose required transform size (n + 3999) lands just
    # above a power of two, so today's padding is near-maximal and the smooth-size change is large.
    for n in (4096, 22050, SR, 62000):
        out.append(_spec(f"len{n}", "independent", channels=2, stems=2, n=n, seed=5))

    # Sample rate. 48 kHz forces the impulse response through the resampler.
    out.append(_spec("sr48000", "independent", channels=2, stems=2, n=48000, seed=6,
                     config={"sample_rate": 48000}))

    # Degenerate and headline cases.
    out += [
        _spec("exact_match", "exact", channels=2, stems=2, seed=7),
        _spec("all_silence", "silence", channels=2, stems=2),
        _spec("silent_target_leakage", "leakage", channels=2, stems=2, k=0.01, seed=8),
        _spec("exact_one_stem", "exact_one_stem", channels=2, stems=4, seed=9, grad=True),
    ]

    # Joint gain. Scale invariance is exact, so these must equal the ungained value.
    for g in (1e-3, 1e3):
        out.append(_spec(f"gain{g:g}", "gain", channels=2, stems=2, g=g, seed=10))

    # Residual SHAPE at two lengths. These are the cases where dropping the trim moves values, and
    # each one's delta is pinned separately rather than hidden inside a blanket tolerance.
    # -30 dB puts them ~79x above the inaudibility gate, so the gate is inert and its later removal
    # does not contaminate the measurement of the trim.
    for shape in ("white", "tone", "click_start", "click_end", "single"):
        for n in (4096, SR):
            out.append(_spec(f"shape_{shape}_n{n}", "shape", shape=shape, n=n, level=0.0316, seed=11))

    # bypass_filter is a separate code path and must be covered on both sides.
    out += [
        _spec("bypass_independent", "independent", channels=2, stems=2, seed=12,
              config={"bypass_filter": True}),
        _spec("bypass_level-20", "scaled", k=0.1, seed=12, config={"bypass_filter": True}),
    ]

    return out


# --------------------------------------------------------------------------------------------
# THE ADAPTER. Everything about how the metric is built and called lives in this one function, so
# the 1.0.0 API changes touch it and nothing else in the golden machinery.
# --------------------------------------------------------------------------------------------

def evaluate(spec):
    """Run one case. Returns {"per_element": [[...]], "pooled": float, "grad_norms": [...] | None}.

    Values are from the POSITIVE metric, so the recorded numbers read the same way as published
    logWMSE figures.
    """
    u, p, t = build(spec)
    config = dict(spec["config"])
    sample_rate = config.pop("sample_rate", SR)
    audio_length = spec["n"] / sample_rate

    unreduced = make_metric(audio_length=audio_length, sample_rate=sample_rate,
                            reduction="none", **config)(u, p, t)
    pooled = make_metric(audio_length=audio_length, sample_rate=sample_rate,
                         reduction="mean", **config)(u, p, t)

    result = {
        # Recorded per [batch][channel][stem]. Flattened batch-first so a diff points at one element.
        "per_element": _round(unreduced.tolist()),
        "pooled": _round(float(pooled)),
        "grad_norms": None,
    }

    if spec["grad"]:
        # Per-stem gradient norms w.r.t. the estimate. This is what makes the gradient-allocation
        # claim checkable rather than asserted, and it is the only recorded quantity that would
        # catch a NaN gradient - the forward value stays finite.
        pe = p.detach().clone().requires_grad_(True)
        make_metric(audio_length=audio_length, sample_rate=sample_rate,
                    reduction="mean", **config)(u, pe, t).backward()
        result["grad_norms"] = _round(
            [float(pe.grad[:, :, s].norm()) for s in range(pe.shape[2])])

    return result


def _round(x):
    if isinstance(x, list):
        return [_round(v) for v in x]
    return round(float(x), PLACES)


def generate():
    return {spec["name"]: evaluate(spec) for spec in cases()}


def load():
    if not os.path.exists(GOLDEN_PATH):
        raise FileNotFoundError(
            f"{GOLDEN_PATH} is missing. It is a committed baseline, not a build artifact - restore "
            "it from git rather than regenerating, or the characterization tests will happily "
            "record whatever the code currently does and assert nothing.")
    with open(GOLDEN_PATH) as fh:
        return json.load(fh)


def _main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="overwrite tests/goldens.json")
    args = ap.parse_args()

    fresh = generate()
    old = load() if os.path.exists(GOLDEN_PATH) else {}

    changed = [n for n in fresh if n in old and fresh[n] != old[n]]
    added = [n for n in fresh if n not in old]
    removed = [n for n in old if n not in fresh]

    for name in changed:
        a, b = old[name]["pooled"], fresh[name]["pooled"]
        print(f"  CHANGED {name:28s} pooled {a:+.6f} -> {b:+.6f}   delta {b - a:+.6f}")
    for name in added:
        print(f"  ADDED   {name:28s} pooled {fresh[name]['pooled']:+.6f}")
    for name in removed:
        print(f"  REMOVED {name}")
    if not (changed or added or removed):
        print(f"  no change across {len(fresh)} cases")

    if args.write:
        with open(GOLDEN_PATH, "w") as fh:
            json.dump(fresh, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print(f"\nwrote {GOLDEN_PATH} ({len(fresh)} cases)")
    elif changed or added or removed:
        print("\n(dry run - pass --write to record, but read the deltas first)")


if __name__ == "__main__":
    _main()
