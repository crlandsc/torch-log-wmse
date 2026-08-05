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
## 0.4.0 (unreleased)

Outcome of a full adversarial audit of the library against its upstream, `nomonosound/log-wmse-audio-quality`. Every change below is backed by a reproduced measurement; where a previously suspected problem turned out not to be real, that is recorded too, because several of them looked convincing.

### BREAKING

- **Invalid inputs now raise `ValueError` instead of returning a number.** Batch and channel agreement between `unprocessed_audio` and `processed_audio` was never checked, and because the scaling factor broadcasts, a mismatch produced a plausible result rather than an error: a mono mixture against stereo stems returned `18.41251`, a batch-1 mixture against batch-4 stems `18.37790`. Input length is now also checked against the length configured at construction, closing a path where the filtered branch silently scored only the first `audio_length_samples` while `bypass_filter=True` scored everything, so the two disagreed by up to 26 units on identical data.
- **Validation no longer uses `assert`.** `python -O` strips assertions, so under optimisation a stem-count mismatch returned `0.02163` and a length mismatch `0.40525`.
- **`reduction` is validated** at construction and in `apply_reduction`, which previously fell through silently: `"Mean"`, `"average"`, `"batchmean"`, `""` and `None` all behaved as `"none"` and returned an unreduced tensor.
- **A malformed `impulse_response` now raises.** Previously unvalidated, so an all-zero impulse response made the metric report its `+73.6827` "perfect" ceiling for **every input, forever** — a truncated `filter_ir.pkl` would have silently reported flawless quality. A 2-D response was broadcast against the stem axis instead of raising, giving per-stem values wrong by ~0.04 with the same output shape. Upstream gets this guard free from `scipy.signal.oaconvolve`; the move to FFT convolution lost it.
- **`symmetric_ir` is removed.** `LogWMSE` never exposed it, and its `False` branch was a provable no-op (`F.pad(ir, (0, n))` then `rfft(n=fft_size)` is bit-identical to `rfft(ir, n=fft_size)`, because `rfft` already zero-pads on the right) that also skipped group-delay compensation.
- **`N_FFT` is removed.** Dead: grep found only its own definition. Its comment claimed it sized the filter's FFT, but the runtime size is computed from the audio length.
- **Dependency floors raised** to `torch>=2.0`, `torchaudio>=2.0`, `python_requires>=3.9`, and numpy re-capped to `>=1.22,<3`. The old floors could not all hold at once: torchaudio has no 1.x release, so `torchaudio>=1.8.0` already meant `>=2.0.1`, which pins torch `>=2.0` and made the advertised `torch>=1.8.0` unreachable. The numpy cap matters because the bundled filter reconstructs through `numpy.core.multiarray._reconstruct`, and numpy states `numpy.core` will be removed.
- **`torch-log-wmse-audio-quality` is now a metadata-only distribution.** It ships no code and depends on `torch-log-wmse` at the same version. Both names previously shipped both package directories and both `RECORD` files claimed the same 11 paths, so installing both and uninstalling either deleted the survivor's files while `pip list` still reported it installed and `import torch_log_wmse` raised `ModuleNotFoundError`. Cross-installing also silently overwrote files a hash-pinned distribution owned. `pip install torch-log-wmse-audio-quality` still works and still provides both import names.

### Fixed

- **Group-delay compensation for odd-length impulse responses.** The shift used a parity-dependent formula that was one sample short for odd lengths, so any odd-length FIR passed via `impulse_response=` produced a filter that was not zero-phase — and since a time shift is learnable, a model trained against it could absorb a spurious one-sample offset. A delta-IR round trip shifted by +1 sample at every odd length tested (3, 51, 101, 999, 1001) and is now exact at all lengths. The shipped 4000-tap response is even, so **no published value changes** (verified `max|delta| = 0.000e+00`).
- **`RMS_EPS` is now a floor rather than an addend.** `1 / (input_rms + RMS_EPS)` perturbed quiet mixtures even when `input_rms` was far above the epsilon; `1 / clamp_min(input_rms, RMS_EPS)` leaves the factor exactly `1/input_rms` for every non-degenerate input. Joint-gain scale invariance, a documented headline property, becomes exact: deviation at gain 1e-5 goes from +0.0145 to 0. This is also what the 0.1.7 and 0.1.8 notes always described.
- **`torch.sqrt` at exactly zero no longer produces NaN gradients.** In `calculate_rms` the forward value stayed finite (the epsilon is applied afterwards) while the backward pass silently filled the graph with NaN for a grad-requiring digitally-silent input.
- **`torch_log_wmse_audio_quality.__version__`** raised `AttributeError`, because `from torch_log_wmse import *` skips underscore-prefixed names.
- **Tests could verify the wrong code.** Both test modules used `sys.path.append`, placing the working tree after `site-packages`, so running a test file directly imported any pip-installed copy of the package rather than the code under edit.

### Performance

- **One filtered difference instead of two filtered signals.** The filter is linear, so `filters(a) - filters(b) == filters(a - b)`. Filter work per call drops from `(1 + 2S)` to `(1 + S)` stem-convolutions for `S` stems. Measured on an idle machine: 1 s at batch 8 x 2ch x 4 stems, 32.26 ms to 18.54 ms (**1.74x**); 10 s, 292.17 ms to 169.37 ms (**1.72x**); 1 s at batch 32 x 2ch x 1 stem, 42.40 ms to 28.58 ms (**1.48x**). Values are bit-identical. Peak memory is essentially unchanged, because torch's caching allocator already recycled the buffers.
- The in-place masked assignment is now `torch.where`, avoiding a data-dependent scatter and a full-size boolean index.

### Documentation

- **The README usage example printed a number ~18 units from its own comment.** It claimed `-18.42` but produced about `0.00`, because commit `acf8ac8` replaced the estimate `unprocessed * 0.1` with independent full-scale noise while fixing tensor ordering, stranding the comment. Independent noise against a silent target is analytically 0 and demonstrates nothing. The example now uses a 20 dB residual against a digital-silence target, which yields exactly `-18.4207` for **every** seed by scale invariance and matches upstream's own oracle. It also no longer rebinds `log_wmse`, which made a second call raise `TypeError`.
- **The sample-rate claim described upstream's behaviour, not this library's.** It said the metric "performs an internal resampling to 44.1kHz"; in fact the *impulse response* is resampled to the audio's rate. Now documents what the code does and the two consequences: below 44.1 kHz the designed curve is truncated at the new Nyquist, and results diverge from the original numpy implementation by up to ~1.5 units at 16 kHz for error near Nyquist.
- **New "Using logWMSE as a loss" section** covering the `+73.6827` ceiling, the `reduction` argument (public since 0.3.0 but previously undocumented), the -68 dB error tolerance, and the gradient regime — the gradient grows as the estimate improves, like SI-SDR and unlike MSE, and the value is scale-invariant while the gradient is not, so the effective learning rate depends on audio level with no indication in the loss curve. This is the subject of the previously unanswered issue #5.
- The two scale-invariance statements that read as contradictory are now distinguished: invariant to gaining all three inputs together, not invariant to scaling the estimate alone.
- The note claiming the original used "time-domain convolution" was wrong — `scipy.signal.oaconvolve` is FFT overlap-add, so both are FFT convolutions, and they agree to about 3 float32 eps at 44.1 kHz.
- Apache-2.0 section 4(b) modified-file notices added to the four derived modules.

### Tests and CI

- **New test CI** (`.github/workflows/ci.yml`) on push and pull request, with one numpy 1.x leg and two numpy 2.x legs, plus a build job running `twine check`. There was previously no test workflow at all, and the publish workflow had no test gate.
- **The publish workflow is rebuilt**: it now depends on the test workflow, uses trusted publishing (OIDC) instead of a long-lived token on the command line, builds each distribution to its own directory so one upload cannot re-send the other's artifacts, cleans `dist/` first, verifies the two distributions share no files, runs `twine check`, and treats TestPyPI and PyPI as separate targets.
- **New invariant, regression and upstream-parity suites**, 10 tests to 60 (110 subtests). A mutation study drove this: of the semantically distinct mutants tried, the original suite failed to detect a filter misaligned by one sample, the error tolerance disabled outright, `bypass_filter` ignored, and the entire non-44.1 kHz resampling path. All are now caught, and the numeric oracles are closed-form rather than values recorded from this implementation.

### Investigated and found sound

Recorded because each looked like a defect and was not, and re-deriving them would waste effort:

- **The error-tolerance dead zone is not a training hazard.** The zero-gradient region is exactly `argmin(loss)` — every point in it attains `-73.682724`, bit-identical to a perfect estimate, and an optimisation started inside it converges to the global minimum with gap `+0.00e+00`. It is also unreachable by 50-70 dB: it needs roughly 74-80 dB SI-SDR, against 5-25 dB for published separation and enhancement. At attainable error levels the threshold is numerically invisible (gradient cosine similarity 1.000000). Replacing the hard threshold with smooth shrinkage would have changed nothing reachable while breaking bit-parity with upstream.
- **Gradient magnitude is normal for this class of loss.** `‖∇‖ = 8/(√N · rms(F(p−t)))`; the scaling factor cancels, so mixture level is irrelevant. Growing gradients as the estimate improves, and `1/g` scaling under joint gain, are both shared with SI-SDR (measured within ~30% at every operating point).
- **Per-stem mean-of-logs does not hide a failed stem.** One failed stem of four costs exactly `ceiling/K` = 18.42 units, the maximum a single stem can inflict, and it receives 100% of the gradient energy because correct stems have zero difference. Mean-over-sources is also upstream's own convention and the audio-ML norm.
- **44.1 kHz parity with the original numpy implementation is exact** — `0.000e+00` across mono, stereo, independent signals, exact match, all-silence, extreme gains, and tone errors from 60 Hz to 7 kHz. The shipped `frequency_weighting.png` still matches the shipped filter, and the filter is reproducible from upstream's documented recipe to 1.6e-08.
