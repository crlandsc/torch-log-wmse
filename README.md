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
batch, channels, stems = 4, 2, 4   # e.g. 4 stems: vocals, drums, bass, other
samples = sample_rate              # 1 second

# LogWMSE is the metric (higher is better). LogWMSELoss is the same thing negated, for training.
# One instance handles any length and any number of stems.
metric = LogWMSE(sample_rate=sample_rate)

unprocessed = 2 * torch.rand(batch, channels, samples) - 1            # the mixture
processed = unprocessed.unsqueeze(2).expand(-1, -1, stems, -1) * 0.1  # an estimate: 20 dB of residual
target = torch.zeros(batch, channels, stems, samples)                # a digital-silence target

print(metric(unprocessed, processed, target))
# tensor(18.4207)

loss = LogWMSELoss(sample_rate=sample_rate)
print(loss(unprocessed, processed, target))
# tensor(-18.4207)
```

> **Upgrading from 0.x?** The API changed in 1.0.0. `LogWMSE` is now the positive metric and
> `LogWMSELoss` is the loss (the old `return_as_loss` flag is gone), `audio_length` is no longer
> needed, and all arguments are keyword-only. Multi-stem scores are unchanged by default. The
> [CHANGELOG](CHANGELOG.md) has a full migration guide.

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

This metric is built for high-fidelity audio (sample rates &ge; 44.1kHz), and the weighting above is designed at 44.1kHz. It still works at other rates — the weighting filter is resampled to match your audio — but scores at other rates are internally consistent rather than comparable to 44.1kHz. See [how it works](docs/design-and-behavior.md#other-sample-rates) for what changes.

##### Inputs
Unlike many audio quality metrics, logWMSE accepts 3 audio inputs rather than 2:

- Unprocessed audio (e.g. raw, noisy audio)
- Processed audio (e.g. denoised or separated audio)
- Target audio (e.g. ground truth, clean audio)

Typically audio loss functions only use the processed audio and target audio to compare against one another. However, logWMSE requires the initial, unprocessed audio because it needs to be able to measure how well the processed audio was attenuated from the unprocessed version. This adds a factor that accounts for when the input contains silence (digital zero).

This also adds a factor of scale invariance: the processed audio needs to be scaled appropriately relative to both the unprocessed audio and the ground truth. Conceptually, if all 3 inputs are gained by the same arbitrary amount, the score stays the same.

##### Using it as a loss
`LogWMSELoss` is the negated metric, so lower is better and you can minimise it directly. Two things are worth knowing up front:

- The score is **bounded above at +73.6827** — a perfect estimate, or an all-silent triplet, lands there.
- The **gradient grows as the estimate improves** (the same behaviour as SI-SDR, the opposite of plain MSE), so you should **use gradient clipping**.

A `p` argument controls how per-stem errors combine; its default reproduces the aggregation every earlier version used, so you can ignore it to start. Mixed precision works through `torch.autocast`. The [training guide](docs/training-guide.md) covers all of this in the detail that matters inside a real training loop.

##### Limitations
- **This is a perceptual objective, not a signal-fidelity one.** The weighting deliberately discounts what the ear is less sensitive to, so training against logWMSE will generally cost you SDR relative to an unweighted loss. That is the trade the metric exists to make; if SDR is your target, use an SDR-matched loss.
- The metric isn't invariant to scaling, polarity inversion, or offsets applied to the estimate alone (as distinct from the joint-gain invariance above).
- Although it incorporates frequency filtering inspired by human auditory sensitivity, it doesn't fully model human auditory perception. For instance, it doesn't consider auditory masking.

More on these, plus sample-rate behaviour and how to compare scores across models, is in [how it works and behaves](docs/design-and-behavior.md).

## Documentation
- **[Using logWMSE as a loss](docs/training-guide.md)** — the gradient regime and why it grows, gradient clipping, mixed precision (including Apple Silicon / MPS), and the `p` stem-combining knob.
- **[How it works and how it behaves](docs/design-and-behavior.md)** — the frequency weighting, the three-input design, scale invariance, other sample rates, and comparing scores.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or new features to suggest.


## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
