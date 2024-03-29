from typing import Callable

import numpy as np
import soxr
from numpy.typing import NDArray

from log_wmse_audio_quality.constants import (
    ERROR_TOLERANCE_THRESHOLD,
    SCALER,
    EPS,
    INTERNAL_SAMPLE_RATE,
)
from log_wmse_audio_quality.freq_weighting_filter import HumanHearingSensitivityFilter
from log_wmse_audio_quality.utils import calculate_rms


def _calculate_single_channel_log_wmse(
    input_rms: np.float32,
    filters: Callable,
    processed_audio: NDArray[np.float32],
    target_audio: NDArray[np.float32],
):
    """
    Calculate the logWMSE metric for a single channel of audio. This is considered an
    internal function used by calculate_wlog_mse.
    """
    if np.sum(input_rms) == 0:
        return np.log(EPS) * SCALER
    scaling_factor = 1 / input_rms

    differences = filters(processed_audio * scaling_factor) - filters(
        target_audio * scaling_factor
    )

    # Tolerate minor errors, because those can't be heard anyway
    differences[np.abs(differences) < ERROR_TOLERANCE_THRESHOLD] = 0.0

    return np.log((differences**2).mean() + EPS) * SCALER


def calculate_log_wmse(
    unprocessed_audio: NDArray,
    processed_audio: NDArray,
    target_audio: NDArray,
    sample_rate: int,
):
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
    # A custom audio quality metric that
    assert unprocessed_audio.ndim in (1, 2)
    assert processed_audio.ndim in (1, 2)
    assert target_audio.ndim in (1, 2)
    assert processed_audio.shape == target_audio.shape
    is_unprocessed_audio_mono = (
        unprocessed_audio.ndim == 1 or unprocessed_audio.shape[0] == 1
    )
    is_processed_audio_mono = processed_audio.ndim == 1 or processed_audio.shape[0] == 1
    assert is_unprocessed_audio_mono == is_processed_audio_mono

    if sample_rate != INTERNAL_SAMPLE_RATE:
        # Resample to a common sample rate that the filters are designed for
        unprocessed_audio = soxr.resample(
            unprocessed_audio.T,
            sample_rate,
            INTERNAL_SAMPLE_RATE,
            quality="HQ",
        ).T
        processed_audio = soxr.resample(
            processed_audio.T,
            sample_rate,
            INTERNAL_SAMPLE_RATE,
            quality="HQ",
        ).T
        target_audio = soxr.resample(
            target_audio.T,
            sample_rate,
            INTERNAL_SAMPLE_RATE,
            quality="HQ",
        ).T

    # Get filter for human hearing sensitivity
    filters = HumanHearingSensitivityFilter()

    input_rms = calculate_rms(filters(unprocessed_audio))

    if processed_audio.ndim == 2:
        values = np.zeros(shape=input_rms.shape, dtype=np.float32)
        for chn_idx in range(input_rms.shape[0]):
            if processed_audio.ndim == 2:
                values[chn_idx] = _calculate_single_channel_log_wmse(
                    input_rms[chn_idx],
                    filters,
                    processed_audio[chn_idx],
                    target_audio[chn_idx],
                )

        return np.mean(values)
    else:
        return _calculate_single_channel_log_wmse(
            input_rms, filters, processed_audio, target_audio
        )
