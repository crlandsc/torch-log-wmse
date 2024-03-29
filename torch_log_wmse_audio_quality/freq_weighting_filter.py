import torch
import torch.fft
from torch import Tensor
from torchaudio.transforms import Resample
import importlib.resources as resources
import pickle

# # imports for conv1d implementation
# from einops import rearrange, repeat
# from torch.nn.functional import conv1d

def fft_convolve(audio, impulse_response):
    # Move the impulse response to the same device as the audio
    impulse_response = impulse_response.to(audio.device)

    # Flatten the batch, channel, and stem dimensions
    original_shape = audio.shape
    flattened_audio = audio.reshape(-1, original_shape[-1])
    
    # Initialize an output tensor
    convolved_audio = torch.empty_like(flattened_audio)
    
    # Apply FFT convolution to each sequence
    for i in range(flattened_audio.shape[0]):
        signal_fft = torch.fft.rfft(flattened_audio[i], n=flattened_audio.shape[-1] + impulse_response.size(-1) - 1)
        impulse_response_fft = torch.fft.rfft(impulse_response, n=flattened_audio.shape[-1] + impulse_response.size(-1) - 1)
        result_fft = signal_fft * impulse_response_fft
        result = torch.fft.irfft(result_fft, n=flattened_audio.shape[-1] + impulse_response.size(-1) - 1)
        
        # Trim the result to match the original signal length
        start = impulse_response.size(-1) // 2
        end = start + original_shape[-1]
        convolved_audio[i] = result[start:end]
    
    # Reshape the convolved audio back to the original shape
    convolved_audio = convolved_audio.reshape(original_shape)
    
    return convolved_audio


class HumanHearingSensitivityFilter:
    def __init__(self, sample_rate: int = 44100, impulse_response: Tensor = None, impulse_response_sample_rate: int = 44100):
        # Load the impulse response if not provided
        if impulse_response is None:
            with resources.open_binary("torch_log_wmse_audio_quality", "filter_ir.pkl") as f:
                impulse_response = torch.tensor(pickle.load(f), dtype=torch.float32)

        # Move impulse response to available device - also dynamically set during fft_convolve
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        impulse_response = impulse_response.to(device=device)

        # Resample the impulse response if necessary
        if impulse_response_sample_rate != sample_rate:
            self.resampler = Resample(orig_freq=impulse_response_sample_rate, new_freq=sample_rate)
            impulse_response = self.resampler(impulse_response)

        # Adjust impulse response to match the expected dimensions for conv1d
        self.impulse_response = impulse_response.squeeze() #.unsqueeze(0)  # Add a batch dimension for grouped convolution

    def __call__(self, audio: Tensor) -> Tensor:
        '''
        Apply the filter to the given audio. The sample rate of the audio
        '''
        # Ensure audio has the correct dimensions: [batch, channels, stem, time]
        if audio.ndim != 4:
            raise ValueError("Audio input must have dimensions [batch, channels, stem, time].")

        # Apply FFT convolution
        # Originally implemented conv1d, but FFT convolution is more efficient with kernel size of 4000
        filtered_audio = fft_convolve(audio, self.impulse_response)

        # # torch convolution with conv1d - deprecated
        # # Use einops to reshape audio by combining batch and channels dimensions
        # einops:
        # # - b: batch
        # # - c: channels
        # # - s: stem
        # # - t: time
        # audio_reshaped = rearrange(audio, 'b c s t -> (b c) s t')

        # # Apply grouped convolution
        # impulse_response_repeated = repeat(self.impulse_response, '1 t -> s1 s2 t', s1=audio.shape[2], s2=audio.shape[2])
        # padding_size = self.impulse_response.shape[-1] // 2
        # filtered_audio = conv1d(audio_reshaped, impulse_response_repeated, padding=padding_size)[..., :audio.shape[-1]]

        # # Use einops to reshape back to original dimensions: [batch, channels, stem, time]
        # filtered_audio = rearrange(filtered_audio, '(b c) s t -> b c s t', b=audio.shape[0], c=audio.shape[1])

        return filtered_audio