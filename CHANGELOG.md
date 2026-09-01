# Changelog

## 0.1.0 (2024-05-15)

Initial release

## 0.1.1 (2024-05-15)

#### Bug Fix
Removed device assignment that was causing errors on distributed training setups

## 0.1.2 (2024-05-16)

#### Bug Fix
Error in tensor shapes. Was processing as the incorrect [batch, channels, stem, samples] instead of the correct [batch, stem, channels, samples] in some places.

## 0.1.3 (2024-05-16)

#### Include Image
Moved frequency weighting image to be included in the package.

## 0.1.4 (2024-05-16)

#### Image Bug
Moving image again and correcting reference.

## 0.1.5 (2024-05-17)

#### Added Logo
Added repo logo.

## 0.1.6 (2024-05-20)

#### Convolution Bug
The convolution operation was previously introducing an unintended time shift due to incorrect padding and trimming. This was causing models to inadvertently learn these time shifts when the operation was used as a loss function. This issue has now been corrected. The convolution operation is now time-invariant, meaning it will not introduce any unwanted time shifts.

## 0.1.7 (2024-05-22)

#### RMS Bug
If the unprocessed audio was silent, a value was immediately being returned unrelated to the model predictions. While this is how the original metric was implemented [here](https://github.com/nomonosound/log-wmse-audio-quality), the RMS value should actually be replaced with epsilon so that the difference between a non-silent output and silent output can be measured.

Added test for silent input & target.

Adjusted circular shift to account for IR with odd number of samples.

## 0.1.8 (2024-05-22)

#### Minimum threshold fix
Changed scaling factor so RMS doesn't need to = 0, rather just be lower than the error threshold to replace with min value. If it only could be 0, then very tiny numbers even closer to 0 would still go through.

## 0.1.9 (2024-06-18)

#### Package name update
Publishing as `torch-log-wmse` (for brevity) as well as `torch-log-wmse-audio-quality`.

## 0.2.0 (2024-06-18)

#### Finalizing name change
Updated all references to **`torch_log_wmse`** independent of installation name (i.e. `pip install torch-log-wmse` or `pip install torch-log-wmse-audio-quality`).

Imports now **MUST** be done as the following:
```
from torch_log_wmse import LogWMSE
```

## 0.2.1 (2024-06-18)

#### Updated badges
Updated badge references in the `README`.

## 0.2.2 (2024-06-18)

#### Changed GitHub repo name & references
Updated GitHub repo name to appropriate `torch-log-wmse`. `torch-log-wmse-audio-quality` can still be installed, but everything will reference the new name (`torch-log-wmse`) moving forward.

## 0.2.3 (2024-06-19)

#### Weighted filter reference bug
Corrected file reference to pkl filter file.

## 0.2.4-0.2.7 (2024-06-21)

#### Added `torch_log_wmse_audio_quality` alias
added alias file so imports can either be `torch_log_wmse` or `torch_log_wmse_audio_quality`.

## 0.2.8 (2024-07-31)

#### Added ability to bypass frequency weightning + bug fixes
Added `bypass_filter` argument that will bypass frequency weighting if `True`.

Fixed bug that returns NaN when one of the entries in the batch is a digital silence triplet - Thanks to Iver Jordal for the [issue](https://github.com/crlandsc/torch-log-wmse/issues/2) & [PR](https://github.com/crlandsc/torch-log-wmse/pull/3)!

## 0.2.9 (2024-07-31)

#### Updated README
Updated README to reflect 0.2.8 `bypass_filter` update.

## 0.3.0 (2025-04-24)

#### Added reduction argument and fixed tensor shape handling
- Added `reduction` argument to control how the loss/metric is aggregated, supporting 'mean' (default), 'sum', and 'none' options
- Fixed tensor shape inconsistencies throughout the codebase to consistently use [batch, channel, stem, time] format
- Updated docstrings and comments to correctly document expected tensor shapes
- Optimized scaling_factor broadcasting
- Added comprehensive test for all reduction options

## 0.3.1 (2026-01-29)

#### Centralized version management
- Version now only needs to be updated in one location (`torch_log_wmse/__init__.py`)
- `setup.cfg` now references version via `attr: torch_log_wmse.__version__`
- `torch_log_wmse_audio_quality` inherits version through import
## 1.0.0 (2026-08-31)

A full adversarial audit of the library against its upstream, `nomonosound/log-wmse-audio-quality`, followed by a redesign of the API and the internals, and then a second adversarial audit of that redesign.

**The API is a breaking change in almost every direction. The numbers are not.** Aggregation across stems is unchanged, so multi-stem and stereo scores stay comparable with earlier versions and with published logWMSE figures. Values move only from two removed mechanisms — a per-sample inaudibility gate and a window trim — and for ordinary broadband material that is under 6e-04.

This is where the `0.x` "anything may change" signal stops being true. Every change below is backed by a reproduced measurement; where a previously suspected problem turned out not to be real, or where a claim in an earlier draft of these notes turned out to be wrong, that is recorded too. Several of them looked convincing.

### Migrating from 0.x

| Before | Now |
|---|---|
| `LogWMSE(audio_length=1.0, sample_rate=44100)` | `LogWMSE(sample_rate=44100)` — one instance serves any length |
| `LogWMSE(..., return_as_loss=True)` | `LogWMSELoss(...)` |
| `LogWMSE(..., return_as_loss=False)` | `LogWMSE(...)` — this class is now the metric, higher is better |
| `LogWMSE(1.0, 44100)` | keyword-only: `LogWMSE(sample_rate=44100)` |
| `reduction="none"` for per-stem values | `per_stem()` → `[batch, channel, stem]` |
| `reduction="none"` | still valid, now returns `[batch]` |
| `from torch_log_wmse_audio_quality import ...` | `from torch_log_wmse import ...` |

Both removed constructor arguments raise `TypeError` naming their replacement rather than being ignored. The constructor is keyword-only specifically because `audio_length` used to come first: a positional call written for 0.x would otherwise put a duration where `sample_rate` now is and quietly build a metric at 1 Hz.

**Aggregation is unchanged.** The new `p` parameter defaults to `0`, which is the mean-of-logs every earlier version used, so multi-stem and stereo scores stay comparable.

### The new `p` parameter, and why the default is 0

A multi-stem model produces one error per stem and they have to become one number. `p` is the exponent of a power mean over them. `p=0` is the mean of logs — the historical behaviour and the default.

The one alternative worth knowing about is **`p=0.5`**, which changes how gradient behaves as a stem converges. The gradient blow-up in a log-domain loss comes from the logarithm itself, since the derivative of `log(x)` is `1/x`: measured at `p=0`, the gradient grows about **900×** from -10 dB to -70 dB of residual, peaks near -80 dB where the error reaches the metric's floor, then falls back, while at `p=0.5` it stays essentially flat. That is the mechanism behind issue #5 ("huge gradients when training"), and `p=0.5` is the mitigation. (In practice a separation model rarely converges far enough to reach the peak.)

It is not the default because it is not free: it changes multi-stem values, so published figures stop being comparable, and on the evidence available its allocation across stems is **worse** for per-stem dB metrics than `p=0` — which won a deterministic effort-allocation comparison in 4 of 4 shapes. `p=0` equalises pressure per unit of *relative* improvement, which is the objective a decibel-domain judge grades against.

> **Neither value is validated by a real training run.** An A/B on a real separation model is still outstanding, as is a promising third option: `p=0` with a raised gradient floor, which in a preliminary probe preserved `p=0`'s allocation to within 3% while cutting the gradient peak a hundredfold. If either changes the default, that is a major version bump.

> **`p=0.5` does not equalise gradient across stems in general.** Per-stem gradient energy is `G² · mse^(2p−1)`; `p=0.5` cancels only the second factor. `G²` is set by where a stem's error sits in the spectrum, and the weighting filter spans about 35 dB across the audible range — so two stems with identical error in different bands were measured to differ by hundreds of times (the exact factor depends on buffer length and how narrow the bands are, from roughly 70× to over 1000×), and the disparity persists at any `p`. An earlier draft of these notes claimed equalisation without that qualifier; the measurement behind it used one waveform at four gains, which is exactly the case where the qualifier does not bite.

### BREAKING — API

- **`LogWMSE` is the metric and `LogWMSELoss` is the loss.** `return_as_loss` is removed. A flag that silently inverts a training objective is not worth the convenience of one import: forget it when evaluating and your numbers are negated; forget it when training and you minimise quality. `LogWMSELoss` is a real subclass negating at the outermost point, so `loss == -metric` holds for every reduction and for `per_stem` alike.
- **`audio_length` is removed** and the constructor is keyword-only. The transform size is derived from the input and its weights cached, so one instance serves any length and any stem count. An entire error class goes with it, including the `floor()`-versus-`round()` off-by-one the old length check had to tolerate for callers who sized their segments the ordinary way.
- **`reduction` covers the batch axis only**, like any other torch loss. `"none"` returns one value per batch item rather than one per `[batch, channel, stem]`, and `"sum"` changes by a factor of channels × stems. Use `per_stem()` for per-element values — and note those are the values that remain comparable with the original numpy implementation at any `p`, so the fidelity anchor is now visible in the API rather than buried in the test suite.
- **`p` is new but its default preserves existing behaviour.** See above. Not a breaking change unless you set it.
- **The `torch_log_wmse_audio_quality` package and distribution are both discontinued.** A shim forwarding a *changed* API is worse than no shim: old code imports cleanly and then behaves differently, which is the failure mode hardest to notice. `import torch_log_wmse_audio_quality` now raises `ImportError`, at the point of the problem. The old distribution stops at 0.3.1 and will not be published again; install `torch-log-wmse` and import `torch_log_wmse`.
- **The filter no longer follows the input's device.** It is an `nn.Module` holding the impulse response as a non-persistent buffer, so `.to(device)` moves it like anything else and `state_dict()` stays empty — holding one as a submodule adds no checkpoint keys. Previously it reassigned its own tensors mid-forward, which is not thread-safe and is awkward for graph capture.
- **`calculate_rms` is removed.** It took the square root of a mean square, and at exactly zero `sqrt` has an infinite derivative, so a grad-requiring silent input propagated NaN backwards while the forward value looked finite. The metric now works in the energy domain and never takes a square root, so that failure mode is structurally absent rather than defended against.

### Fixed by a second adversarial audit, of the redesign itself

Four independent agents reviewed the redesigned code, each on a different mandate. All four found something real, in code that had already been verified and carried a 33-mutant gate.

- **Integer audio returned a *perfect* score for any input.** Every tap of the hearing-sensitivity filter has magnitude below 1, so building the weight vector in an integer dtype truncated all 4000 of them to zero — error energy and mixture energy both came out 0, and the score pinned at the `+73.6827` ceiling. Digital silence against a loud target read as flawless. Reachable from `soundfile.read(dtype='int16')` or `scipy.io.wavfile.read`. Integer inputs now raise a `TypeError` naming the conversion, and the weight dtype is forced to floating point independently.
- **`p >= 6` returned `±inf` in float32, and the failure arrived late.** Raising to the p-th power before the mean underflows: with a bit-exact stem, `EPS**6 = 1e-48` goes to zero, and `log(0)` follows. The loss trained normally and died once the model got *good*, and under `reduction="mean"` one such batch item poisoned the whole batch. Pooling now runs in the log domain via `logsumexp`, verified finite to `p=64`.
- **The mixture is now detached.** `d(loss)/d(log mixture gain)` was exactly `-8.0`, so anywhere the mixture carried gradient — a cascaded system, learned augmentation — the objective could be reduced by making the *mixture louder* rather than the estimate better.
- **`calculate_rms` is removed.** It took the square root of a mean square, and at exactly zero `sqrt` has an infinite derivative, so a grad-requiring silent input propagated NaN backwards while the forward value looked finite. The metric now works in the energy domain and never takes a square root, so the failure mode is structurally absent rather than guarded.
- Smaller: the weight cache survived `.to()`/`.half()` with stale entries; `cache_size=0` raised a bare `StopIteration`; `Resample` ran before validation, so a list or numpy impulse response worked at 44.1 kHz and threw at every other rate.

### BREAKING — values

- **The -68 dB per-sample inaudibility gate is removed.** It cannot survive computing energy in the frequency domain, because a per-sample gate needs a time-domain signal. Its measured effect across the reachable range was **0.000** — identical values from -10 dB to -40 dB residual — and reaching the band where it mattered needs 74-80 dB SI-SDR. Exact digital silence still returns the ceiling, so the headline feature never depended on it. Scores below roughly -50 dB residual now differ slightly from earlier versions, and the closed-form oracle holds at every level instead of breaking down below 1e-3.
- **An identical error now scores the same wherever it falls in the buffer.** The metric measures the energy of the full linear convolution rather than of a window the same length as the input, so the filter's ring at the buffer edges is counted instead of discarded. The old behaviour scored a single-sample error **0.77 units better at the buffer boundary than in the middle**, purely because half its ring fell outside the window. Broadband residuals move by under 6e-04; the difference is material only for errors concentrated at the very edges.

### Performance

**roughly 2-5× faster** end to end on stereo 4-stem audio, depending on clip length, batch size and machine load.

- **The weighted error energy is computed in the frequency domain.** One-sided Parseval gives it from the forward transform alone, so the inverse transform, the group-delay correction and the trim all disappear — along with the class of alignment bug the audit had already found once in the group-delay handling.
- **Transform sizes are 2/3/5-smooth**, not the next power of two, chosen from a ~5% geometric ladder rounded up to fit (so nearby lengths share a size). One second at 44.1 kHz needs 48099 samples: 48600 as a smooth number against 65536 as a power of two, cutting padding waste from 36% to 1%.
- Fixed-length input is the supported path and produces exactly one cache entry, which is what `torch.compile` wants. Variable-length input is correct but pays a small per-length setup and produces many distinct transform sizes, which defeats compilation caching.

### Added

- **`per_stem()`** returning `[batch, channel, stem]` scores.
- **`p`** exposed and documented, including `p=0` as the compatibility path.
- **`py.typed`**, so the type hints are visible to mypy and pyright rather than only to people reading the source.

### BREAKING — from the audit

- **Invalid inputs now raise `ValueError` instead of returning a number.** Batch and channel agreement between `unprocessed_audio` and `processed_audio` was never checked, and because the scaling factor broadcasts, a mismatch produced a plausible result rather than an error: a mono mixture against stereo stems returned `18.41251`, a batch-1 mixture against batch-4 stems `18.37790`. The audit also added a check that the input length matched the length configured at construction, closing a path where the filtered branch silently scored only the first `audio_length_samples` while `bypass_filter=True` scored everything — the two disagreed by up to 26 units on identical data. 1.0.0 removes both the check and the constructor argument it guarded: with the transform size derived from the input, there is nothing left to mismatch.
- **Validation no longer uses `assert`.** `python -O` strips assertions, so under optimisation a stem-count mismatch returned `0.02163` and a length mismatch `0.40525`.
- **`reduction` is validated** at construction and in `apply_reduction`, which previously fell through silently: `"Mean"`, `"average"`, `"batchmean"`, `""` and `None` all behaved as `"none"` and returned an unreduced tensor.
- **A malformed `impulse_response` now raises.** Previously unvalidated, so an all-zero impulse response made the metric report its `+73.6827` "perfect" ceiling for **every input, forever** — a truncated `filter_ir.pkl` would have silently reported flawless quality. A 2-D response was broadcast against the stem axis instead of raising, giving per-stem values wrong by ~0.04 with the same output shape. Upstream gets this guard free from `scipy.signal.oaconvolve`; the move to FFT convolution lost it.
- **`symmetric_ir` is removed.** `LogWMSE` never exposed it, and its `False` branch was a provable no-op (`F.pad(ir, (0, n))` then `rfft(n=fft_size)` is bit-identical to `rfft(ir, n=fft_size)`, because `rfft` already zero-pads on the right) that also skipped group-delay compensation.
- **`N_FFT` is removed.** Dead: grep found only its own definition. Its comment claimed it sized the filter's FFT, but the runtime size is computed from the audio length.
- **The bundled filter is no longer a pickle.** `filter_ir.pkl` is replaced by `filter_ir.f32`, raw little-endian float32 with its length and SHA-256 pinned in the loader. `pickle.load` resolves and calls arbitrary dotted global names, and nothing validated the result; the payload is 4000 floats and needs none of that. The bytes are the float32 cast of the original array, which is exactly what the old loader produced, so **metric values are unchanged** (verified bitwise equal). The file also halves, 32151 to 16000 bytes. Upstream's generator is now vendored at `dev/create_freq_weighting_filter_ir.py`, so the filter can be regenerated and compared rather than trusted, and upstream's design credits (Fenton & Lee 2017; mmxgn 2023) are carried over.
- **numpy is no longer a runtime dependency.** It was never imported by this package; it was required only so the pickle could reconstruct a numpy array. Verified by blocking every numpy import and running a full forward pass.
- **Dependency floors raised** to `torch>=2.0`, `torchaudio>=2.0`, `python_requires>=3.9`. The old floors could not all hold at once: torchaudio has no 1.x release, so `torchaudio>=1.8.0` already meant `>=2.0.1`, which pins torch `>=2.0` and made the advertised `torch>=1.8.0` unreachable.
- **The two distributions could destroy each other's files.** Through 0.3.1 both `torch-log-wmse` and `torch-log-wmse-audio-quality` shipped both package directories, and both `RECORD` files claimed the same 11 paths — so installing both and uninstalling either deleted the survivor's files while `pip list` still reported it installed and `import torch_log_wmse` raised `ModuleNotFoundError`. Cross-installing also silently overwrote files a hash-pinned distribution owned. The audit's fix was to make the second distribution metadata-only; 1.0.0 goes further and discontinues it entirely (see above), which makes the collision structurally impossible rather than merely avoided.

### Fixed

- **Group-delay compensation for odd-length impulse responses.** The shift used a parity-dependent formula that was one sample short for odd lengths, so any odd-length FIR passed via `impulse_response=` produced a filter that was not zero-phase — and since a time shift is learnable, a model trained against it could absorb a spurious one-sample offset. A delta-IR round trip shifted by +1 sample at every odd length tested (3, 51, 101, 999, 1001) and is now exact at all lengths. The shipped 4000-tap response is even, so **no published value changes** (verified `max|delta| = 0.000e+00`).
- **`RMS_EPS` is now a floor rather than an addend.** `1 / (input_rms + RMS_EPS)` perturbed quiet mixtures even when `input_rms` was far above the epsilon; `1 / clamp_min(input_rms, RMS_EPS)` leaves the factor exactly `1/input_rms` for every non-degenerate input. Joint-gain scale invariance, a documented headline property, becomes exact: deviation at gain 1e-5 goes from +0.0145 to 0. This is also what the 0.1.7 and 0.1.8 notes always described.
- **`torch.sqrt` at exactly zero no longer produces NaN gradients.** In `calculate_rms` the forward value stayed finite (the epsilon is applied afterwards) while the backward pass silently filled the graph with NaN for a grad-requiring digitally-silent input. The floor is the dtype's smallest normal rather than a constant, so it is correct in float16/bfloat16 and leaves every normal value untouched.
- **`torch_log_wmse_audio_quality.__version__`** raised `AttributeError`, because `from torch_log_wmse import *` skips underscore-prefixed names.
- **Tests could verify the wrong code.** Both test modules used `sys.path.append`, placing the working tree after `site-packages`, so running a test file directly imported any pip-installed copy of the package rather than the code under edit.

### Performance

- **One filtered difference instead of two filtered signals.** The filter is linear, so `filters(a) - filters(b) == filters(a - b)`. Filter work per call drops from `(1 + 2S)` to `(1 + S)` stem-convolutions for `S` stems. Measured on an idle machine: 1 s at batch 8 x 2ch x 4 stems, 32.26 ms to 18.54 ms (**1.74x**); 10 s, 292.17 ms to 169.37 ms (**1.72x**); 1 s at batch 32 x 2ch x 1 stem, 42.40 ms to 28.58 ms (**1.48x**). Values are identical to within one float32 ULP (exactly 0.000e+00 on the regression fixture, at most 3.9e-06 elsewhere). Peak memory is essentially unchanged, because torch's caching allocator already recycled the buffers.
- The in-place masked assignment is now `torch.where`, avoiding a data-dependent scatter. Note this does not save memory: both forms build the full-size boolean condition, and `torch.where` allocates an additional output buffer.

### Documentation

- **The README usage example printed a number ~18 units from its own comment.** It claimed `-18.42` but produced about `0.00`, because commit `acf8ac8` replaced the estimate `unprocessed * 0.1` with independent full-scale noise while fixing tensor ordering, stranding the comment. Independent noise against a silent target is analytically 0 and demonstrates nothing. The example now uses a 20 dB residual against a digital-silence target, which yields exactly `-18.4207` for **every** seed by scale invariance and matches upstream's own oracle. It also no longer rebinds `log_wmse`, which made a second call raise `TypeError`.
- **The sample-rate claim described upstream's behaviour, not this library's.** It said the metric "performs an internal resampling to 44.1kHz"; in fact the *impulse response* is resampled to the audio's rate. Now documents what the code does and the two consequences: below 44.1 kHz the designed curve is truncated at the new Nyquist, and results diverge from the original numpy implementation by tens of units at 16 kHz for error near Nyquist (about 42 units at 7.9 kHz — see the README's Frequency Weighting section for which implementation stops seeing the error and why).
- **New "Using logWMSE as a loss" section** covering the `+73.6827` ceiling, the `reduction` argument (public since 0.3.0 but previously undocumented), the -68 dB error tolerance, and the gradient regime — the gradient grows as the estimate improves, like SI-SDR and unlike MSE, and the value is scale-invariant while the gradient is not, so the effective learning rate depends on audio level with no indication in the loss curve. This is the subject of the previously unanswered issue #5.
- The two scale-invariance statements that read as contradictory are now distinguished: invariant to gaining all three inputs together, not invariant to scaling the estimate alone.
- The note claiming the original used "time-domain convolution" was wrong — `scipy.signal.oaconvolve` is FFT overlap-add, so both are FFT convolutions, and they agree to about 3 float32 eps at 44.1 kHz.
- Apache-2.0 section 4(b) modified-file notices added to the four derived modules.

### Tests and CI

- **New test CI** (`.github/workflows/ci.yml`) on push and pull request, with one numpy 1.x leg and two numpy 2.x legs, plus a build job running `twine check`. There was previously no test workflow at all, and the publish workflow had no test gate.
- **The publish workflow is rebuilt**: it now depends on the test workflow, uses trusted publishing (OIDC) instead of a long-lived token on the command line, builds each distribution to its own directory so one upload cannot re-send the other's artifacts, cleans `dist/` first, verifies the two distributions share no files, runs `twine check`, and treats TestPyPI and PyPI as separate targets.
- **New invariant, regression and upstream-parity suites**, from 10 test functions to 148 (920 subtests). A mutation study drove this: of the semantically distinct mutants tried, the original suite failed to detect a filter misaligned by one sample, the error tolerance disabled outright, `bypass_filter` ignored, and the entire non-44.1 kHz resampling path. All are now caught, and the numeric oracles are closed-form rather than values recorded from this implementation.

### Investigated and found sound

Recorded because each looked like a defect and was not, and re-deriving them would waste effort:

- **The error-tolerance dead zone is not a training hazard.** (Moot in 1.0.0, which removes the gate entirely — but the analysis is why removing it was known to be safe.) The zero-gradient region is exactly `argmin(loss)` — every point in it attains `-73.682724`, bit-identical to a perfect estimate, and an optimisation started inside it converges to the global minimum with gap `+0.00e+00`. It is also unreachable by 50-70 dB: it needs roughly 74-80 dB SI-SDR, against 5-25 dB for published separation and enhancement. At attainable error levels the threshold is numerically invisible (gradient cosine similarity 1.000000). Replacing the hard threshold with smooth shrinkage would have changed nothing reachable while breaking bit-parity with upstream.
- **Gradient magnitude is normal for this class of loss.** With `bypass_filter=True`, `‖∇‖ = 8/(√N · rms(F(p−t)))`; with the weighting filter the constant differs by up to about 1.6× depending on where the error sits in the spectrum. Either way the mixture's scaling factor cancels, so mixture level is irrelevant. Growing gradients as the estimate improves, and `1/g` scaling under joint gain, are both shared with SI-SDR (measured within ~30% at every operating point).
- **Per-stem mean-of-logs does not hide a failed stem** — upheld, but the reasoning behind it went through two corrections worth recording, because both were instructive.

  The original measurement used one totally failed stem against three *perfect* ones. That case cannot discriminate: stems with zero error contribute no gradient, so every aggregation rule agrees on it. A second measurement with four stems at *different partial* levels of convergence appeared to overturn the conclusion — the largest-error stem received only 0.2% of the gradient energy.

  That reading was itself wrong. Because the error is normalised by the **mixture**, a large error means a **loud** stem, not a badly separated one. Four stems separated to identical quality but at different levels get wildly different gradient shares, and mean-of-logs gives most of it to the *quietest* — the opposite of neglect. What the second measurement actually showed is that mean-of-logs equalises pressure per unit of **relative** improvement, which is a defensible objective and the one a decibel-domain judge grades against.

  Two degenerate test cases in a row, each producing a confident conclusion in a different direction. The lesson that survives is about method, not about aggregation: a case where every hypothesis predicts the same outcome is not evidence for any of them.
- **44.1 kHz parity with the original numpy implementation is float32-close.** For broadband errors and the exact-match and all-silence cases it is `0.000e+00`, or within a few times float32 epsilon. For pure tones the two now differ by up to about 1.5e-2 at 60 Hz (smaller higher up), because 1.0.0 dropped the window trim the original still applies — a deliberate change, not a regression. The shipped `frequency_weighting.png` still matches the shipped filter, and the filter is reproducible from upstream's documented recipe to 1.6e-08.

## 1.1.0 (unreleased)

### Added

- **`grad_scale` parameter** on `LogWMSE` and `LogWMSELoss` (default `1.0`, off). It multiplies the gradient magnitude by a constant while leaving the score value bit-exact, so you can shrink the reported gradient to keep gradient clipping from over-clipping. The constant cancels under Adam and other adaptive optimizers, so it changes nothing about training there; under plain SGD it is a learning-rate rescale. logWMSE's gradient grows as the estimate improves, so watch the gradient magnitude when you clip — loosen the clip threshold or lower `grad_scale`.
