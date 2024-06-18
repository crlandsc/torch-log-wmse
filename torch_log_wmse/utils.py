import torch
from typing import Union

def calculate_rms(samples: torch.Tensor):
    """
    Calculates the Root Mean Square (RMS) power level of a tensor of audio samples.
        Args: samples (torch.Tensor): A tensor containing audio samples.
        Returns: torch.Tensor: A tensor of the same shape as the input, but containing the RMS power level of the audio samples.
    """
    return torch.sqrt(torch.mean(torch.square(samples), dim=-1))

def convert_decibels_to_amplitude_ratio(decibels: Union[torch.Tensor, float]):
    """
    Converts a tensor of decibel values into a tensor of amplitude ratios.
        Args: decibels (Union[torch.Tensor, float]): A tensor containing decibel values.
        Returns: torch.Tensor: A tensor of the same shape as the input, but containing amplitude ratio values.
    """
    if not isinstance(decibels, torch.Tensor):
        decibels = torch.tensor(decibels)
    return torch.pow(10, decibels / 20)