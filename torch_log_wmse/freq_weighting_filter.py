"""Human hearing sensitivity weighting filter.

Derived from nomonosound/log-wmse-audio-quality (Copyright 2023 Nomono), licensed under the
Apache License 2.0. Modified by Christopher Landschoot (SoundFoxLabs) in 2024-2026: ported from numpy to PyTorch,
FFT convolution in place of scipy.signal.oaconvolve, batched [batch, channel, stem, time] tensor
API, differentiable loss support, and the impulse-response/silence handling described in the
README and CHANGELOG.
"""
import hashlib
import math
from importlib.resources import files
from typing import Optional

import torch
from torch import Tensor
from torchaudio.transforms import Resample

# Raw little-endian float32 samples of the bundled hearing-sensitivity FIR. Previously a pickle, which
# is a code-execution format: `pickle.load` resolves arbitrary dotted global names and calls them, and
# nothing validated what came back. A flat array needs none of that, and dropping the pickle also
# removes numpy as a runtime dependency -- it was required only to reconstruct the pickled ndarray, and
# is not imported anywhere in this package.
#
# The bytes are the float32 cast of the original float64 pickle, which is exactly what the old loader
# produced (verified bitwise equal), so metric values are unchanged.
_IR_RESOURCE = "filter_ir.f32"
_IR_TAPS = 4000
_IR_SHA256 = "bc9950e4f6be2846bf178017a5cca1e5407244d7b255d60de17b882f304998ab"


def load_bundled_impulse_response(verify: bool = True) -> Tensor:
    """Load the bundled human-hearing-sensitivity impulse response.

    Args:
        verify: If True (default), check the payload length and SHA-256 against the values pinned
            above. This is an integrity check against a corrupted or truncated install, not a
            security boundary -- anyone who can modify the resource can modify this module too. It is
            worth having because a silently truncated response would still "work": an all-zero or
            short FIR produces plausible numbers rather than an error.

    Returns:
        Tensor: 1-D float32 tensor of `_IR_TAPS` samples.
    """
    data = files("torch_log_wmse").joinpath(_IR_RESOURCE).read_bytes()
    if verify:
        expected_bytes = _IR_TAPS * 4
        if len(data) != expected_bytes:
            raise ValueError(
                f"{_IR_RESOURCE} should be {expected_bytes} bytes ({_IR_TAPS} float32 samples), "
                f"got {len(data)}. The install looks corrupt."
            )
        digest = hashlib.sha256(data).hexdigest()
        if digest != _IR_SHA256:
            raise ValueError(
                f"{_IR_RESOURCE} failed its integrity check: expected sha256 {_IR_SHA256}, got {digest}."
            )
    # bytearray() because frombuffer requires a writable buffer; the copy is 16 KB and happens once.
    return torch.frombuffer(bytearray(data), dtype=torch.float32)


def prepare_impulse_response_fft(impulse_response, fft_size):
    """
    Prepares the FFT of the impulse response for convolution.

    The impulse response is centre-padded, which pairs with the group-delay compensation in
    HumanHearingSensitivityFilter.forward to make the filter zero-phase.

    Args:
    - impulse_response: The impulse response signal, a 1D tensor of shape [kernel_size].
    - fft_size: The size of FFT to use, typically a power of two that is at least
                as large as the sum of the signal length and kernel_size minus one.

    Returns:
    - A complex tensor of shape [1, 1, fft_size // 2 + 1] (three dimensions, dtype complex64 for a
      float32 impulse response) holding the half-spectrum of the impulse response, shaped for
      broadcasting across batch, channel and stem axes during convolution.
    """
    # Centre-pad the impulse response to FFT size (N+M-1)
    total_padding = fft_size - impulse_response.shape[-1]
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

    Args:
    - audio_batch: A batch of time-domain audio signals.
                   Expected shape: [batch, channel, signal_length] or [batch, channel, stem, signal_length].
    - impulse_response_fft: The precomputed FFT of the impulse response (frequency domain), with shape [1, 1, fft_size // 2 + 1].
    - fft_size: The FFT size the impulse response was prepared at. The audio must already be padded
                to this length.

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


class HumanHearingSensitivityFilter(torch.nn.Module):
    """
    A filter that applies human hearing sensitivity weighting to audio signals.

    This class implements a frequency weighting filter that mimics human hearing sensitivity.
    It uses predefined finite impulse responses (FIR) to simulate how human ears perceive different frequencies.

    An nn.Module rather than a plain class, so it honours the standard torch contract: `.to(device)`
    and `.cuda()` move its tensors. As a plain class they were unreachable by `.to()`, and `__call__`
    compensated by comparing devices and REASSIGNING module state mid-forward - not thread-safe, and
    awkward for torch.compile and graph capture. That compensation is gone; the buffers move.

    Both tensors are registered `persistent=False`. They are derived constants, regenerable from the
    shipped `filter_ir.f32`, so they must not enter `state_dict()`: any model holding a LogWMSE as a
    submodule would otherwise gain checkpoint keys and break `load_state_dict(strict=True)` for
    everyone who saved before, and couple saved models to the package's filter data.

    Attributes:
        sample_rate (int): The sample rate of the audio signal.
        impulse_response (torch.Tensor): The FIR used for filtering. Non-persistent buffer.
        impulse_response_fft (torch.Tensor): The FFT of the impulse response used for efficient
            convolution. Non-persistent buffer.
        fft_size (int): The size of the FFT used for convolution. Signals will be padded to this size.
        audio_length_samples (int): The length of the audio signal in samples.

    Args:
        audio_length (float): The length of the audio signal in seconds. May be fractional.
        sample_rate (int, optional): The sample rate of the audio signal in Hz. Defaults to 44100.
        impulse_response (torch.Tensor, optional): The FIR filter for frequency weighting.
        impulse_response_sample_rate (int, optional): The sample rate of the FIR in Hz. Defaults to 44100.
    """
    def __init__(
            self,
            audio_length: float = 1,
            sample_rate: int = 44100,
            impulse_response: Optional[torch.Tensor] = None,
            impulse_response_sample_rate: int = 44100,
        ):
        super().__init__()
        # Validate the rates first: Resample() below would otherwise fail with a less useful error,
        # and math.log2 further down raises a bare "math domain error" for a non-positive length.
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if impulse_response_sample_rate <= 0:
            raise ValueError(
                f"impulse_response_sample_rate must be positive, got {impulse_response_sample_rate}"
            )

        # Load the impulse response if not provided
        if impulse_response is None:
            impulse_response = load_bundled_impulse_response()

        # Resample the impulse response if necessary
        if impulse_response_sample_rate != sample_rate:
            resampler = Resample(orig_freq=impulse_response_sample_rate, new_freq=sample_rate)
            impulse_response = resampler(impulse_response)

        # Remove any singleton dimensions, then validate. Without these checks a wrong-rank impulse
        # response is silently broadcast against the stem axis instead of raising, and a degenerate one
        # (all zeros, or containing NaN) corrupts every result: an all-zero IR makes the metric report
        # its "perfect" ceiling for every input, forever.
        impulse_response = torch.as_tensor(impulse_response, dtype=torch.float32).squeeze()
        if impulse_response.ndim != 1:
            raise ValueError(
                "impulse_response must be one-dimensional after squeezing singleton dimensions, got "
                f"shape {tuple(impulse_response.shape)}. Multi-channel FIRs are not supported."
            )
        if impulse_response.numel() < 2:
            raise ValueError(
                f"impulse_response must have at least 2 taps, got {impulse_response.numel()}."
            )
        if not torch.isfinite(impulse_response).all():
            raise ValueError("impulse_response contains NaN or infinite values.")
        if not torch.any(impulse_response != 0):
            raise ValueError(
                "impulse_response is all zeros, which would make every comparison report a perfect score."
            )
        # persistent=False: a derived constant, not learned state. See the class docstring.
        self.register_buffer("impulse_response", impulse_response, persistent=False)

        # Calculate minimum FFT size (N+M-1) - make a power of 2 for FFT efficiency
        self.audio_length_samples = math.floor(audio_length * sample_rate)
        if self.audio_length_samples < 1:
            raise ValueError(
                f"audio_length * sample_rate must be at least 1 sample, got {audio_length} x "
                f"{sample_rate} = {audio_length * sample_rate}. A negative or near-zero audio_length "
                "otherwise fails later with an opaque math domain error."
            )
        min_fft_size = self.audio_length_samples + self.impulse_response.shape[-1] - 1
        self.fft_size = 2 ** math.ceil(math.log2(min_fft_size))

        # Compute the FFT of the impulse response (will be padded to fft_size before FFT).
        # Note this uses the squeezed, validated IR - passing the raw argument here would reintroduce
        # the rank mismatch the validation above exists to prevent.
        self.register_buffer(
            "impulse_response_fft",
            prepare_impulse_response_fft(self.impulse_response, self.fft_size),
            persistent=False,
        )

    def forward(self, audio: Tensor) -> Tensor:
        """
        Applies the human hearing sensitivity filter to the input audio via frequency domain convolution.

        NOTE: The original numpy implementation convolves with scipy.signal.oaconvolve(..., "same"),
              which is also an FFT method (overlap-add), so both implementations are FFT convolutions.
              At 44.1 kHz the two agree to float32 precision - measured max absolute difference 3.6e-07
              on white noise, about 3 float32 eps. They diverge only at other sample rates, because the
              original resamples the AUDIO to 44.1 kHz while this implementation resamples the IMPULSE
              RESPONSE to the audio's rate; see the README on sample rates.

        Args: audio (torch.Tensor): A tensor containing the audio signal to be filtered.
                                    Expected shape is [batch, channel, stem, time].

        Returns: torch.Tensor: The filtered audio signal with the same shape as the input.
        """
        # Ensure audio has the correct dimensions: [batch, channel, stem, time]
        if audio.ndim != 4:
            raise ValueError("Audio input must have dimensions [batch, channel, stem, time].")

        # No device reassignment here. The buffers follow `.to(device)` like any other module's, so
        # mutating module state during a forward pass is neither needed nor wanted. A device
        # mismatch now raises torch's standard "Expected all tensors to be on the same device"
        # instead of being silently papered over.

        # Pad audio to match padded FFT size (N+M-1)
        padding = self.fft_size - audio.shape[-1]
        audio = torch.nn.functional.pad(audio, (0, padding))

        # Apply FFT convolution
        filtered_audio = fft_convolve(audio, self.impulse_response_fft, self.fft_size)

        # Circularly shift the signal to undo the symmetric IR's group delay.
        # The IR is centre-padded to fft_size, so its centre of symmetry lands at
        #   left_padding + (M - 1) // 2   where   left_padding = (fft_size - M) // 2,
        # which simplifies to fft_size // 2 - 1 for BOTH even and odd M. This matches the offset
        # scipy.signal.oaconvolve(..., "same") uses upstream, so the two agree at any IR length.
        shift = self.fft_size // 2 - 1
        filtered_audio = torch.roll(filtered_audio, -shift, dims=-1)

        return filtered_audio[..., :self.audio_length_samples]
