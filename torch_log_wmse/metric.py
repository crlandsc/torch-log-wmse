import torch
from torch import Tensor
from typing import Callable, Optional

from torch_log_wmse.constants import (
    ERROR_TOLERANCE_THRESHOLD,
    SCALER,
    EPS,
)
from torch_log_wmse.freq_weighting_filter import HumanHearingSensitivityFilter
from torch_log_wmse.utils import calculate_rms


class LogWMSE(torch.nn.Module):
    """
    logWMSE is a custom metric and loss function for audio signals that calculates the logarithm
    of a frequency-weighted Mean Squared Error (MSE). It is designed to address several shortcomings 
    of common audio metrics, most importantly the lack of support for digital silence targets.

    Key features of logWMSE:
    * Supports digital silence targets not supported by other audio metrics.
        i.e. (SI-)SDR, SIR, SAR, ISR, VISQOL_audio, STOI, CDPAM, and VISQOL.
    * Overcomes the small value range issue of MSE (i.e. between 1e-8 and 1e-3), making number 
        formatting and sight-reading easier. Scaled similar to SI-SDR.
    * Scale-invariant, aligns with the frequency sensitivity of human hearing.
    * Invariant to the tiny errors of MSE that are inaudible to humans.
    * Logarithmic, reflecting the logarithmic sensitivity of human hearing.
    * Tailored specifically for audio signals.

    Args:
        audio_length (int): The length of the audio signal in seconds.
        sample_rate (int, optional): The sample rate of the audio signal in Hz. Defaults to 44100.
        impulse_response (Tensor, optional): The finite impulse response (FIR) filter for 
            frequency weighting. If None (default), use built-in FIR. Currently only supports
            single-channel FIRs (applied to all batches & audio channels).
        impulse_response_sample_rate (int, optional): The sample rate of the FIR in Hz. Defaults to 44100.
        return_as_loss (bool, optional): Whether to return the loss value (i.e. negative of the metric). Defaults to True.
    """
    def __init__(
            self,
            audio_length: int,
            sample_rate: int = 44100,
            impulse_response: Optional[Tensor] = None,
            impulse_response_sample_rate: int = 44100,
            return_as_loss: bool = True,
        ):
        super().__init__()
        self.filters = HumanHearingSensitivityFilter(
            audio_length=audio_length,
            sample_rate=sample_rate,
            impulse_response=impulse_response,
            impulse_response_sample_rate=impulse_response_sample_rate
        )
        self.return_as_loss = return_as_loss

    def forward(self, unprocessed_audio: Tensor, processed_audio: Tensor, target_audio: Tensor):
        assert unprocessed_audio.ndim == 3 # unprocessed_audio audio shape: [batch, channel, time]
        assert processed_audio.ndim == 4 # processed_audio audio shape: [batch, channel, stem, time]
        assert target_audio.ndim == 4 # target_audio audio shape: [batch, channel, stem, time]
        assert processed_audio.shape == target_audio.shape # processed_audio and target_audio should have the same shape
        assert processed_audio.shape[-1] == target_audio.shape[-1] == unprocessed_audio.shape[-1] # all should have the same length

        input_rms = calculate_rms(self.filters(unprocessed_audio.unsqueeze(1))) # unsqueeze to add "stem" dimension

        # Calculate the logWMSE
        values = self._calculate_log_wmse(
            input_rms,
            self.filters,
            processed_audio,
            target_audio,
        )

        if self.return_as_loss:
            return -torch.mean(values)
        else:
            return torch.mean(values)

    @staticmethod
    def _calculate_log_wmse(
        input_rms: Tensor,
        filters: Callable,
        processed_audio: Tensor,
        target_audio: Tensor,
    ):
        """
        Calculate the logWMSE between the processed audio and target audio.

        Args:
            input_rms (Tensor): The root mean square of the input audio.
            filters (Callable): A function that applies a filter to the audio (i.e. HumanHearingSensitivityFilter).
            processed_audio (Tensor): The processed audio tensor.
            target_audio (Tensor): The target audio tensor.

        Returns:
            Tensor: The logWMSE between the processed audio and target audio.
        """

        # Add EPS if input_rms is 0 (silence), or close to it, to avoid NaNs
        if input_rms.sum() < ERROR_TOLERANCE_THRESHOLD:
            input_rms = torch.ones_like(input_rms) * ERROR_TOLERANCE_THRESHOLD

        # Calculate the scaling factor based on the input RMS
        scaling_factor = 1 / input_rms

        # Add extra dimensions to scaling_factor to match the shape of processed_audio and target_audio
        if scaling_factor.dim() == 2:
            scaling_factor = scaling_factor.unsqueeze(1)
        while scaling_factor.dim() < processed_audio.dim():
            scaling_factor = scaling_factor.unsqueeze(-1)

        # Expand scaling_factor to match the shape of processed_audio and target_audio
        scaling_factor = scaling_factor.expand(*processed_audio.shape)

        # Calculate the frequency-weighted differences, ignoring small imperceptible differences
        differences = filters(processed_audio * scaling_factor) - filters(target_audio * scaling_factor)
        differences[torch.abs(differences) < ERROR_TOLERANCE_THRESHOLD] = 0.0

        # Calculate the mean squared differences
        mean_diff = (differences**2).mean(dim=-1)

        return torch.log(mean_diff + EPS) * SCALER