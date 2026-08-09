![torch-log-wmse-logo](https://raw.githubusercontent.com/crlandsc/torch-log-wmse/main/images/logo.png)

[![LICENSE](https://img.shields.io/github/license/crlandsc/torch-log-wmse)](https://github.com/crlandsc/torch-log-wmse/blob/main/LICENSE) [![GitHub Repo stars](https://img.shields.io/github/stars/crlandsc/torch-log-wmse)](https://github.com/crlandsc/torch-log-wmse/stargazers) <!-- [![GitHub forks](https://img.shields.io/github/forks/crlandsc/torch-log-wmse)](https://github.com/crlandsc/torch-log-wmse/forks) -->

This repository contains the torch implementation of an audio quality metric, [logWMSE](https://github.com/nomonosound/log-wmse-audio-quality), originally proposed by [Iver Jordal](https://github.com/iver56). In addition to the original metric, this implementation can also be used as a loss function for training audio separation and denoising models.

logWMSE is a custom metric and loss function for audio signals that calculates the logarithm (log) of a frequency-weighted (W) Mean Squared Error (MSE). It is designed to address several shortcomings of common audio metrics, most importantly the lack of support for digital silence targets.

## Installation

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch-log-wmse)](https://pypi.org/project/torch-log-wmse/) [![PyPI - Version](https://img.shields.io/pypi/v/torch-log-wmse)](https://pypi.org/project/torch-log-wmse/) [![Downloads](https://img.shields.io/pepy/dt/torch-log-wmse)](https://pepy.tech/project/torch-log-wmse)


`pip install torch-log-wmse`

## Usage Example

```python
import torch
from torch_log_wmse import LogWMSE, LogWMSELoss

sample_rate = 44100
audio_stems = 4    # 4 audio stems (e.g. vocals, drums, bass, other)
audio_channels = 2 # stereo
batch = 4          # batch size
samples = sample_rate  # 1 second

# LogWMSE is the METRIC: higher is better. LogWMSELoss is the same thing negated, for training.
# All arguments are keyword-only, and every one of them is optional.
metric = LogWMSE(
    sample_rate=sample_rate,
    p=0.5,               # how per-stem errors combine; see "Combining stems and channels"
    bypass_filter=False, # skip the frequency weighting
    reduction="mean",    # over the BATCH axis: "mean", "sum", or "none"
)

# One instance handles any input length and any number of stems.
unprocessed_audio = 2 * torch.rand(batch, audio_channels, samples) - 1

# The target is digital silence, and the estimate leaves 20 dB of residual leakage.
# Supporting a digital-silence target is the main thing logWMSE offers over (SI-)SDR.
processed_audio = unprocessed_audio.unsqueeze(2).expand(-1, -1, audio_stems, -1) * 0.1
target_audio = torch.zeros(batch, audio_channels, audio_stems, samples)

print(metric(unprocessed_audio, processed_audio, target_audio))
# tensor(18.4207)

print(metric.per_stem(unprocessed_audio, processed_audio, target_audio).shape)
# torch.Size([4, 2, 4])  - one score per [batch, channel, stem]

loss = LogWMSELoss(sample_rate=sample_rate)
print(loss(unprocessed_audio, processed_audio, target_audio))
# tensor(-18.4207)  - exactly the negated metric, for every reduction
```

The value above is not a sampling artifact. Because the estimate is a fixed multiple of the mixture,
`mse` is exactly `0.1**2` and the score is exactly `-4*ln(0.01) = 18.4207` for any random draw.

> **Upgrading from 0.x?** 1.0.0 is a breaking release: `LogWMSE` changed sign, `return_as_loss` and
> `audio_length` are gone, `reduction` now covers only the batch axis, and multi-stem values changed.
> The [CHANGELOG](CHANGELOG.md) has a line-by-line migration guide.

logWMSE accepts three torch tensors of the following shapes:
- unprocessed_audio: `[batch, audio_channels, samples]`
- processed_audio: `[batch, audio_channels, audio_stems, samples]`
- target_audio: `[batch, audio_channels, audio_stems, samples]`

Each dimension being:
- `batch`: Number of audio files in a batch (i.e. batch size).
- `audio_channels`: Number of channels (i.e. 1 for mono and 2 for stereo).
- `audio_stems`: Number of separate audio sources. For source separation, this could be multiple different instruments, vocals, etc. For denoising audio, this will be 1.
- `samples`: Number of audio samples (e.g. 1 second of audio @ 44.1kHz is 44100 samples).

## Motivation
The goal of this metric is to account for several factors not present in current audio evaluation metrics, such as dealing with digital silence. Mean Squared Error (MSE) is well-defined for digital silence targets, but has its own set of drawbacks. Attempting to mitigate these issues, the following are some attributes of logWMSE:

- Supports digital silence targets not supported by other audio metrics.
    i.e. (SI-)SDR, SIR, SAR, ISR, VISQOL_audio, STOI, CDPAM, and VISQOL.
- Overcomes the small value range issue of MSE (i.e. between 1e-8 and 1e-3), making number formatting and sight-reading easier. It is scaled similarly to SI-SDR for consistency with current benchmark metrics (i.e. 3 is poor, 30 is very good).
- Scale-invariant, aligns with the frequency sensitivity of human hearing.
- Logarithmic, reflecting the logarithmic sensitivity of human hearing.
- Tailored specifically for audio signals.

##### Frequency Weighting
To measure the frequencies of a signal closer to that of human hearing, the following frequency weighting is applied. This helps the model effectively pay less attention to errors at frequencies that humans are not sensitive to (e.g. 50 Hz) and give more weight to those that we are acutely tuned to (e.g. 3kHz).

![Frequency Weighting](https://raw.githubusercontent.com/crlandsc/torch-log-wmse/main/images/frequency_weighting.png)

This metric has been constructed with high-fidelity audio in mind (sample rates &ge; 44.1kHz), and the frequency weighting above is designed at 44.1kHz.

For other sample rates, the **impulse response is resampled to the audio's rate** rather than the audio being resampled to 44.1kHz. Two consequences are worth knowing. Below 44.1kHz the designed curve is truncated at the new Nyquist, so at 16kHz (Nyquist 8kHz) everything above 8kHz of the weighting — including the 10kHz lowpass corner — no longer exists. And because the original numpy implementation resamples the *audio* to 44.1kHz instead, results agree with it to float32 precision at 44.1kHz but diverge at other rates. At 16kHz the disagreement grows sharply towards Nyquist: about 0.12 units for error at 1kHz, 1.5 units at 7kHz, and **tens of units** in the last few hundred Hz below Nyquist (measured 33 units at 7.9kHz, where the original still sees the error and this implementation's downsampled response no longer does). Treat sub-44.1kHz scores as internally consistent but not comparable to 44.1kHz scores or to the original implementation.

##### Inputs
Unlike many audio quality metrics, logWMSE accepts 3 audio inputs rather than 2:

- Unprocessed audio (e.g. raw, noisy audio)
- Processed audio (e.g. denoised or separated audio)
- Target audio (e.g. ground truth, clean audio)

Typically audio loss functions only use the processed audio and target audio to compare against one another. However, logWMSE requires the initial, unprocessed audio because it needs to be able to measure how well the processed audio was attenuated from the unprocessed version. This adds a factor that accounts for when the input contains silence (digital zero).

This also adds a factor of scale invariance in the sense that the processed audio needs to be scaled appropriately relative to both the unprocessed audio and ground truth. Conceptually, this means that if all 3 inputs are gained by the same arbitrary amount, the metric score will stay the same.

Note that this is invariance to **joint** gain, which is a different property from the one in Limitations below. Gaining all three inputs together leaves the score unchanged; scaling **only the estimate** does change it. Unlike SI-SDR, logWMSE does not solve for an optimal estimate scale, so an estimate that is correct apart from a gain error is penalised for that gain error.

##### Combining stems and channels
A multi-stem model produces one error per stem, and they have to become one number. How they combine is the `p` parameter, and it matters more than it sounds like it should.

Averaging in the log domain — what every version before 1.0.0 did, and what `p=0` still does — **starves the stem that most needs gradient**. With four stems spread over 25 dB, the worst one receives 0.2% of the gradient energy while the best-converged takes 74%. The model is then trained almost entirely on the parts it has already learned.

`p` is the exponent of a power mean over the per-stem errors, and per-stem gradient energy scales as `mse^(2p-1)`. That is equal across stems at exactly one value, `p = 1/2`, which is the default:

| `p` | gradient shares across 4 stems spread over 25 dB | effective stems trained |
|---|---|---|
| `0` (pre-1.0.0) | 0.2% / 2.3% / 23.5% / 74.0% | 1.66 of 4 |
| **`0.5` (default)** | **25% / 25% / 25% / 25%** | **4.00 of 4** |
| `1` | 89.8% / 9.0% / 0.9% / 0.3% | 1.23 of 4 |

You should not need to change it. Set `p=0` if you need numbers comparable with previously published logWMSE figures. Single-stem mono models are unaffected by `p` entirely — with one value to combine, every setting agrees.

> `p = 1/2` is derived from gradient analysis rather than validated by a training run. An A/B against `p = 0` on a real separation model is in progress; if it moves the default, that will be a major version bump.

##### Using logWMSE as a loss
`LogWMSELoss` is the negated metric, so lower is better and it can be minimised directly. A few properties are worth knowing before training against it:

- **The value is bounded above at +73.6827** (`-4 * ln(EPS)`). An exact match, or an all-silent triplet, saturates there. A score pinned at that ceiling means "no measurable error", not a bug.
- **Gradient magnitude grows as the estimate improves**, scaling as `1 / (absolute filtered error RMS)`. This is the same behaviour as SI-SDR and the opposite of plain MSE, whose gradient vanishes near the optimum. Gradient clipping is still recommended, though the default `p=1/2` makes this far tamer than it was: on a converging 4-stem case the global gradient norm settles near 0.94 where log-domain averaging drove it to 3.59 and climbing.
- **The gradient is not scale-invariant even though the value is.** Gaining all three inputs by `g` leaves the score identical but scales the gradient by `1/g`, so the effective learning rate depends on your audio level while the loss curve gives no indication of it. Normalise your audio to a consistent level.
- **`reduction`** controls aggregation over the **batch axis only**, like any other torch loss: `"mean"` (default), `"sum"`, or `"none"` for one value per batch item. For per-stem values use `per_stem()`, which returns `[batch, channel, stem]`.
- **Move it like any other module.** `LogWMSELoss` is an `nn.Module` holding the impulse response as a non-persistent buffer, so `.to(device)` works and `state_dict()` stays empty — holding one as a submodule adds no checkpoint keys.

##### Performance
Transform sizes are the smallest even 2/3/5-smooth value that fits the input, and the weighted error energy is computed in the frequency domain, so there is no inverse transform. One second of stereo 4-stem audio at 44.1 kHz transforms at 48600 points rather than 65536.

Fixed-length input is the supported path and produces exactly one cache entry, which is also what `torch.compile` wants. Variable-length input is correct but pays a small setup cost per new length and produces many distinct transform sizes, which defeats compilation caching — pad to a fixed length if you intend to compile.

##### Limitations
- The metric isn't invariant to scaling, polarity inversion, or offsets applied to the estimated audio alone (as distinct from the joint-gain invariance described above).
- Although it incorporates frequency filtering inspired by human auditory sensitivity, it doesn't fully model human auditory perception. For instance, it doesn't consider auditory masking.
- **Stem-count dilution.** Perfect stems still inflate the score, by `8*ln(S)`. A 4-stem and a 16-stem model are not directly comparable. This is much better than the pre-1.0.0 behaviour (`+41.4` at 4 stems) but not eliminated.
- **Per-stem values are the upstream-comparable ones.** `per_stem()` matches the original numpy implementation at 44.1 kHz whatever `p` is; the pooled value deliberately does not, for unequal stems or channels.
- Results match the original numpy implementation to float32 precision at 44.1 kHz for broadband errors. At other sample rates the two diverge, because the original resamples the audio to 44.1 kHz while this implementation resamples the impulse response to the audio's rate.
- **fp16 requires `bypass_filter=True`.** `torch.fft` has no half-precision kernel on CPU or MPS, so the filtered path needs float32 or better.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or new features to suggest.


## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
