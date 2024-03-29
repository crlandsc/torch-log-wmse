import torch
from torch import Tensor
from typing import Callable, Optional

from torch_log_wmse_audio_quality.constants import (
    ERROR_TOLERANCE_THRESHOLD,
    SCALER,
    EPS,
)
from torch_log_wmse_audio_quality.freq_weighting_filter import HumanHearingSensitivityFilter
from torch_log_wmse_audio_quality.utils import calculate_rms


class LogWMSELoss(torch.nn.Module):
    """
    A custom audio metric, logWMSE, that tries to fix a few shortcomings of common metrics.
    The most important property is the support for digital silence in the target, which
    (SI-)SDR, SIR, SAR, ISR, VISQOL_audio, STOI, CDPAM and VISQOL do not support.

    MSE is well-defined for digital silence targets, but has a bunch of other issues:
    * The values are commonly ridiculously small, like between 1e-8 and 1e-3, which
        makes number formatting and sight-reading hard
    * It's not tailored for audio
    * It's not scale-invariant
    * It doesn't align with frequency sensitivity of human hearing
    * It's not invariant to tiny errors that don't matter because humans can't hear
        those errors anyway
    * It's not logarithmic, like human hearing is

    So this custom metric attempts to solve all the problems mentioned above.
    It's essentially the log of a frequency-weighted MSE, with a few bells and whistles.
    """
    def __init__(
            self,
            sample_rate: int = 44100,
            impulse_response: Optional[Tensor] = None,
            impulse_response_sample_rate: int = 44100
        ):
        super().__init__()
        self.filters = HumanHearingSensitivityFilter(
            sample_rate=sample_rate,
            impulse_response=impulse_response,
            impulse_response_sample_rate=impulse_response_sample_rate
        )

    def forward(self, unprocessed_audio: Tensor, processed_audio: Tensor, target_audio: Tensor):
        assert unprocessed_audio.ndim == 3 # unprocessed_audio audio shape: [batch, channel, time]
        assert processed_audio.ndim == 4 # processed_audio audio shape: [batch, channel, stem, time]
        assert target_audio.ndim == 4 # target_audio audio shape: [batch, channel, stem, time]
        assert processed_audio.shape == target_audio.shape # processed_audio and target_audio should have the same shape
        assert processed_audio.shape[-1] == target_audio.shape[-1] == unprocessed_audio.shape[-1] # all should have the same length

        input_rms = calculate_rms(self.filters(unprocessed_audio.unsqueeze(2))) # unsqueeze to add "stem" dimension

        values = self._calculate_log_wmse(
            input_rms,
            self.filters,
            processed_audio,
            target_audio,
        )

        return torch.mean(values)

    @staticmethod
    def _calculate_log_wmse(
        input_rms: Tensor,
        filters: Callable,
        processed_audio: Tensor,
        target_audio: Tensor,
    ):
        zero_mask = torch.sum(input_rms, dim=-1, keepdim=True) == 0
        scaling_factor = torch.where(zero_mask, torch.zeros_like(input_rms), 1 / input_rms)

        # Add extra dimensions to scaling_factor to match the shape of processed_audio and target_audio
        while scaling_factor.dim() < processed_audio.dim():
            scaling_factor = scaling_factor.unsqueeze(-1)

        # Expand scaling_factor to match the shape of processed_audio and target_audio
        scaling_factor = scaling_factor.expand(*processed_audio.shape)

        # TODO: scaling factor is shape [2, 2]. each value needs to be multiplied to all corresponding calues for [i, i, :, :]
        differences = filters(processed_audio * scaling_factor) - filters(target_audio * scaling_factor)

        differences[torch.abs(differences) < ERROR_TOLERANCE_THRESHOLD] = 0.0
        mean_diff = (differences**2).mean(dim=-1)
        zero_mask = (mean_diff == 0)

        return torch.where(zero_mask, torch.log(torch.tensor(EPS)) * SCALER, torch.log((differences**2).mean(dim=-1) + EPS) * SCALER)
