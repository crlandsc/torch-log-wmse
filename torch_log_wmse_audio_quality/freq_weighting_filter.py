import torch
from torch import Tensor
from torchaudio.transforms import Resample
import importlib.resources as resources
import pickle
import math


def prepare_impulse_response_fft(impulse_response, fft_size):
    """
    Prepares the FFT of the impulse response for convolution.

    Parameters:
    - impulse_response: The impulse response signal, a 1D tensor of shape [kernel_size].
    - fft_size: The size of FFT to use, typically a power of two that is at least
                as large as the sum of the signal length and kernel_size minus one.

    Returns:
    - A 2D tensor of shape [1, 1, fft_size // 2 + 1] representing the FFT of the impulse
      response, ready for broadcasting across batches and channels during convolution.
    """
    # Pad the impulse response to FFT size (N+M-1)
    total_padding = fft_size - impulse_response.shape[0]
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding
    impulse_response = torch.nn.functional.pad(impulse_response, (left_padding, right_padding))

    # Compute the FFT of the impulse response
    impulse_response_fft = torch.fft.rfft(impulse_response, n=fft_size)

    # Adjust shape for broadcasting across batches, channels, & stems
    impulse_response_fft = impulse_response_fft.unsqueeze(0).unsqueeze(0)

    return impulse_response_fft


def fft_convolve(audio_batch, impulse_response_fft, fft_size):
    """
    Performs FFT convolution on a batch of audio signals using a precomputed impulse response FFT.

    Parameters:
    - audio_batch: A batch of audio signals, with shape [batch_size, channels, signal_length].
    - impulse_response_fft: The precomputed FFT of the impulse response, with shape [1, 1, fft_size // 2 + 1].

    Returns:
    - A tensor of convolved audio signals with the same shape as audio_batch.
    """
    # Perform the FFT on the audio batch
    signal_fft = torch.fft.rfft(audio_batch, n=fft_size)

    # Apply the convolution in the frequency domain
    result_fft = signal_fft * impulse_response_fft

    # Perform the inverse FFT to obtain the convolved signals
    convolved_audio = torch.fft.irfft(result_fft, n=fft_size)

    return convolved_audio


class HumanHearingSensitivityFilter:
    """
    A filter that applies human hearing sensitivity weighting to audio signals.
    
    This class implements a frequency weighting filter that mimics human hearing sensitivity. 
    It uses predefined finite impulse responses (FIR) to simulate how human ears perceive different frequencies.
    
    Attributes:
        sample_rate (int): The sample rate of the audio signal.
        impulse_response (torch.Tensor): The FIR used for filtering.
        impulse_response_fft (torch.Tensor): The FFT of the impulse response used for efficient convolution.
    """
    def __init__(self, audio_length: int = 1, sample_rate: int = 44100, impulse_response: Tensor = None, impulse_response_sample_rate: int = 44100):
        # Load the impulse response if not provided
        if impulse_response is None:
            with resources.open_binary("torch_log_wmse_audio_quality", "filter_ir.pkl") as f:
                impulse_response = torch.tensor(pickle.load(f), dtype=torch.float32)

        # Resample the impulse response if necessary
        if impulse_response_sample_rate != sample_rate:
            self.resampler = Resample(orig_freq=impulse_response_sample_rate, new_freq=sample_rate)
            impulse_response = self.resampler(impulse_response)

        # Remove any singleton dimensions
        self.impulse_response = impulse_response.squeeze()

        # Calculate minimum FFT size (N+M-1) - make a power of 2 for FFT efficiency
        self.audio_length_samples = math.floor(audio_length * sample_rate)
        min_fft_size = self.audio_length_samples + impulse_response.shape[-1] - 1
        self.fft_size = 2 ** math.ceil(math.log2(min_fft_size))

        # Compute the FFT of the impulse response
        self.impulse_response_fft = prepare_impulse_response_fft(impulse_response, self.fft_size)


    def __call__(self, audio: Tensor) -> Tensor:
        """
        Applies the human hearing sensitivity filter to the input audio via frequency domain convolution.

        NOTE: The original logWMSE metric implementation in numpy used time-domain convolution for
              single-channel/single-batch/single-stem audio. This torch implementation uses FFT convolution
              for efficiency. This will result in slightly different outputs due to the different convolution
              methods.
        
        Args: audio (torch.Tensor): A tensor containing the audio signal to be filtered. 
                                    Expected shape is [batch, stem, channels, time].
        
        Returns: torch.Tensor: The filtered audio signal with the same shape as the input.
        """
        # Ensure audio has the correct dimensions: [batch, stem, channels, time]
        if audio.ndim != 4:
            raise ValueError("Audio input must have dimensions [batch, stem, channels, time].")

        # Move impulse response to audio device if necessary
        if self.impulse_response_fft.device != audio.device:
            self.impulse_response_fft = self.impulse_response_fft.to(audio.device)

        # Pad audio to match padded FFT size (N+M-1)
        audio = torch.nn.functional.pad(audio, (0, self.fft_size - audio.shape[-1]))

        # Apply FFT convolution
        filtered_audio = fft_convolve(audio, self.impulse_response_fft, self.fft_size)

        # Trim the filtered audio to match the original length
        start_index = self.fft_size // 2
        end_index = start_index + self.audio_length_samples

        return filtered_audio[..., start_index:end_index]