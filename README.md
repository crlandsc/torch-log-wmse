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
from torch_log_wmse import LogWMSE

# Tensor shapes
audio_length = 1.0
sample_rate = 44100
audio_stems = 4 # 4 audio stems (e.g. vocals, drums, bass, other)
audio_channels = 2 # stereo
batch = 4 # batch size

# Instantiate logWMSE
# Set `return_as_loss=False` to return as a positive metric (Default: True)
# Set `bypass_filter=True` to bypass frequency weighting (Default: False)
# Set `reduction` to "mean" (default), "sum", or "none" for per-[batch, channel, stem] values
log_wmse = LogWMSE(
    audio_length=audio_length,
    sample_rate=sample_rate,
    return_as_loss=True, # optional
    bypass_filter=False, # optional
    reduction="mean",    # optional
)

# Generate a random mixture (scale between -1 and 1)
audio_lengths_samples = int(audio_length * sample_rate)
unprocessed_audio = 2 * torch.rand(batch, audio_channels, audio_lengths_samples) - 1

# The target is digital silence, and the estimate leaves 20 dB of residual leakage.
# Supporting a digital-silence target is the main thing logWMSE offers over (SI-)SDR.
processed_audio = unprocessed_audio.unsqueeze(2).expand(-1, -1, audio_stems, -1) * 0.1
target_audio = torch.zeros(batch, audio_channels, audio_stems, audio_lengths_samples)

score = log_wmse(unprocessed_audio, processed_audio, target_audio)
print(score)  # -18.4207, and seed-independent: the metric is exactly scale-invariant
```

The value above is not a sampling artifact. Because the estimate is a fixed multiple of the mixture,
`mse` is exactly `0.1**2` and the loss is exactly `4*ln(0.01) = -18.4207` for any random draw.

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
- Invariant to the tiny errors of MSE that are inaudible to humans.
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

##### Using logWMSE as a loss
`return_as_loss=True` (the default) returns the negated metric, so lower is better and it can be minimised directly. A few properties are worth knowing before training against it:

- **The value is bounded above at +73.6827** (`-4 * ln(EPS)`). An exact match, or an all-silent triplet, saturates there. A score pinned at that ceiling means "no measurable error", not a bug.
- **Gradient magnitude grows as the estimate improves**, scaling as `1 / (absolute filtered error RMS)`. This is the same behaviour as SI-SDR and the opposite of plain MSE, whose gradient vanishes near the optimum. Gradient clipping is recommended.
- **The gradient is not scale-invariant even though the value is.** Gaining all three inputs by `g` leaves the score identical but scales the gradient by `1/g`, so the effective learning rate depends on your audio level while the loss curve gives no indication of it. Normalise your audio to a consistent level.
- **Errors below -68 dB relative to the input RMS are treated as inaudible and discounted**, so the metric saturates at the ceiling once the residual falls below roughly -80 dB. That threshold is far beyond what separation or enhancement models reach in practice (it would require an SI-SDR of about 74-80 dB), so it does not affect normal training.
- **`reduction`** controls aggregation: `"mean"` (default), `"sum"`, or `"none"` for per-`[batch, channel, stem]` values. Use `"none"` when you want to report or inspect individual stems rather than a single averaged number.

##### Limitations
- The metric isn't invariant to scaling, polarity inversion, or offsets applied to the estimated audio alone (as distinct from the joint-gain invariance described above).
- Although it incorporates frequency filtering inspired by human auditory sensitivity, it doesn't fully model human auditory perception. For instance, it doesn't consider auditory masking.
- Results match the original numpy implementation to float32 precision at 44.1 kHz. At other sample rates the two diverge, because the original resamples the audio to 44.1 kHz while this implementation resamples the impulse response to the audio's rate.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or new features to suggest.


## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
