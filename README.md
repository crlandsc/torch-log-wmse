# torch-log-wmse-audio-quality

This repository contains the torch implementation of an audio quality metric, [logWMSE](https://github.com/nomonosound/log-wmse-audio-quality), originally suggested by [Iver Jordal](https://github.com/iver56) of [Nomono](https://nomono.co/). In addition to the original metric, this implementation can also be used as a loss function for training audio models.

logWMSE is a custom metric and loss function for audio signals that calculates the logarithm (log)
of a frequency-weighted (W) Mean Squared Error (MSE). It is designed to address several shortcomings of common audio metrics, most importantly the lack of support for digital silence targets.

## Installation

To use this repository, clone it and install the required dependencies:

```bash
git clone https://github.com/your-username/torch-log-wmse-audio-quality.git
cd torch-log-wmse-audio-quality
pip install -r requirements.txt
```

## Usage Example

```python
import torch
from torch_log_wmse_audio_quality import LogWMSE

# Tensor shapes
audio_length = 1.0
sample_rate = 44100
audio_channels = 2 # stereo
audio_stems = 3 # 3 audio stems
batch = 4 # batch size

# Instantiate logWMSELoss
log_wmse_loss = LogWMSE(audio_length=audio_length, sample_rate=sample_rate, return_as_loss=True)

# Generate random inputs (scale between -1 and 1)
audio_lengths_samples = int(audio_length * sample_rate)
unprocessed_audio = 2 * torch.rand(batch, audio_channels, audio_lengths_samples) - 1
processed_audio = unprocessed_audio.unsqueeze(2).expand(-1, -1, audio_stems, -1) * 0.1
target_audio = torch.zeros(batch, audio_channels, audio_stems, audio_lengths_samples)

log_wmse = log_wmse_loss(unprocessed_audio, processed_audio, target_audio)
print(log_wmse)  # Expected output: approx. -18.42
```

## Motivation
* Supports digital silence targets not supported by other audio metrics.
    i.e. (SI-)SDR, SIR, SAR, ISR, VISQOL_audio, STOI, CDPAM, and VISQOL.
* Overcomes the small value range issue of MSE (i.e. between 1e-8 and 1e-3), making number 
    formatting and sight-reading easier. Scaled similar to SI-SDR.
* Scale-invariant, aligns with the frequency sensitivity of human hearing.
* Invariant to the tiny errors of MSE that are inaudible to humans.
* Logarithmic, reflecting the logarithmic sensitivity of human hearing.
* Tailored specifically for audio signals.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or new features to suggest.


## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.


## Acknowledgements
Thanks to [Whitebalance](https://www.whitebalance.co/) for backing this project.
