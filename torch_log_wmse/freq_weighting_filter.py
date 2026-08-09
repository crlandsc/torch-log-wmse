"""Human hearing sensitivity weighting filter.

Derived from nomonosound/log-wmse-audio-quality (Copyright 2023 Nomono), licensed under the
Apache License 2.0. Modified by Christopher Landschoot (SoundFoxLabs) in 2024-2026: ported from
numpy to PyTorch; batched [batch, channel, stem, time] tensor API; differentiable loss support; the
weighted error energy computed in the frequency domain via one-sided Parseval rather than by
convolving with scipy.signal.oaconvolve; power-mean aggregation across channel and stem in place of
the mean of logs; the -68 dB per-sample inaudibility gate removed; and the impulse-response and
silence handling described in README.md and CHANGELOG.md.
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


# A ~5% geometric ladder. The required transform length is rounded up to a rung before a size is
# chosen, so nearby input lengths collapse onto the same transform and reuse one cache entry.
#
# Do not expect this to restore the coarse bucketing that powers of two gave. Even 2/3/5-smooth
# values sit 1-2% apart, so a 0.5-1.5 s sweep produces 54 distinct sizes; the ladder brings that to
# 21, still above torch.compile's default cache_size_limit of 8. Fixed-length input is the supported
# case and produces exactly ONE entry either way. The ladder costs nothing in accuracy - any L at or
# above N + M - 1 gives the same linear convolution - and measured faster at 2 s (1.77x vs 1.51x),
# because 96000 is a friendlier size than 93312.
_LADDER_STEP = 1.05


def next_fft_friendly_size(n: int) -> int:
    """Smallest EVEN 2/3/5-smooth integer >= n.

    Mixed-radix FFTs run fast on any product of small primes, while a power of two is smooth but
    often forces rounding up much further: 1 s at 44.1 kHz needs 48099 samples, which is 48600 as a
    5-smooth number against 65536 as a power of two, cutting padding waste from 36% to 1%.

    EVEN is a performance choice, not a correctness one. It used to be mandatory because the
    group-delay correction assumed it; that correction is gone. Parity still matters for the
    Nyquist weight, which `parseval_weights` handles for both cases.
    """
    if n <= 2:
        return 2
    best = 1 << (n - 1).bit_length()  # next power of two: an upper bound, and even for n >= 2
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            m = 2  # >= 2 keeps the result even
            while m * p35 < n:
                m *= 2
            best = min(best, m * p35)
            p35 *= 3
        p5 *= 5
    return best


def _quantized_size(required: int) -> int:
    """Round `required` up the ladder, then to the next FFT-friendly size."""
    if required <= 2:
        return 2
    rung = int(math.ceil(_LADDER_STEP ** math.ceil(math.log(required) / math.log(_LADDER_STEP))))
    return next_fft_friendly_size(max(rung, required))


def parseval_weights(impulse_response: Tensor, transform_size: int) -> Tensor:
    """Per-bin weights `w[f] = |H(f)|^2 * c[f] / L` for one-sided Parseval.

    The metric only ever needs the ENERGY of the frequency-weighted error, and Parseval gives that
    from the forward transform alone:

        sum_n y[n]^2 = ( |Y[0]|^2 + |Y[-1]|^2 + 2 * sum_{k=1..-2} |Y[k]|^2 ) / L

    so with `Y = X * H` the whole thing collapses to `sum_f |X(f)|^2 * w[f]`. No inverse transform,
    no group-delay correction, no trim.

    THE DOUBLING AND THE 1/L ARE FOLDED INTO w ON PURPOSE. A naive `sum(|Y|^2) / L` over an rfft
    half-spectrum undercounts by 2x, which surfaces as a constant offset of exactly 4*ln(2) = 2.7726
    on every value. Putting the correction here makes that bug impossible at call sites and testable
    in one isolated unit.

    `c` is 1 at DC, 2 for interior bins, and at the last bin 1 for EVEN L (a real Nyquist bin,
    counted once) or 2 for odd L (where the last bin is an ordinary conjugate pair).

    The impulse response is zero-padded on the right rather than centre-padded. Centre-padding
    existed to place the group delay where the correction expected it; with `|H|^2` the filter's
    phase cannot affect the result at all, so the padding position is now irrelevant.

    NOTE this makes the FILTER's phase irrelevant, not the metric's. The error is `estimate -
    target`, so a latency offset between the two is still fully penalised.
    """
    spectrum = torch.fft.rfft(impulse_response, n=transform_size)
    weights = spectrum.real**2 + spectrum.imag**2
    coefficients = torch.full_like(weights, 2.0)
    coefficients[0] = 1.0
    if transform_size % 2 == 0:
        coefficients[-1] = 1.0
    return weights * coefficients / transform_size


class HumanHearingSensitivityFilter(torch.nn.Module):
    """
    Human hearing sensitivity weighting, reported as ENERGY rather than as a filtered signal.

    A predefined finite impulse response (FIR) models how the ear's sensitivity varies with
    frequency. The metric only ever needs the total energy of the weighted signal, and Parseval
    gives that from the forward transform alone - so the filtered waveform is never materialised.
    That removes the inverse transform, the group-delay correction and the trim, along with the
    entire class of alignment bug the audit had already found once in the group-delay handling.

    One instance serves ANY input length. The transform size is derived per call and its weight
    vector cached, so nothing pins a length at construction.

    An nn.Module, so it honours the standard torch contract: `.to(device)` and `.cuda()` move its
    tensors. As a plain class they were unreachable by `.to()`, and the forward pass compensated by
    comparing devices and REASSIGNING module state - not thread-safe, and awkward for torch.compile
    and graph capture.

    `impulse_response` is registered `persistent=False`: a derived constant, regenerable from the
    shipped `filter_ir.f32`, so it must not enter `state_dict()`. Any model holding a LogWMSE as a
    submodule would otherwise gain checkpoint keys and break `load_state_dict(strict=True)` for
    everyone who saved before.

    The weight cache is a PLAIN DICT and deliberately not a buffer. Non-persistent buffers still
    appear in `named_buffers()`, which DDP enumerates to build its broadcast list, so a cache that
    grows lazily and unevenly across ranks would give them divergent buffer lists.

    Attributes:
        sample_rate (int): The sample rate of the audio signal.
        impulse_response (torch.Tensor): The FIR used for weighting. Non-persistent buffer.

    Args:
        sample_rate (int, optional): The sample rate of the audio signal in Hz. Defaults to 44100.
        impulse_response (torch.Tensor, optional): The FIR filter for frequency weighting.
        impulse_response_sample_rate (int, optional): The sample rate of the FIR in Hz. Defaults to 44100.
        cache_size (int, optional): How many transform sizes to keep weights for. Defaults to 8,
            which is ample for the fixed-length case that produces exactly one entry.
    """
    def __init__(
            self,
            sample_rate: int = 44100,
            impulse_response: Optional[torch.Tensor] = None,
            impulse_response_sample_rate: int = 44100,
            cache_size: int = 8,
        ):
        super().__init__()
        # Validate the rates first: Resample() below would otherwise fail with a less useful error.
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

        self.sample_rate = sample_rate
        self.cache_size = cache_size
        # Plain dict, never a buffer - see the class docstring on DDP. Keyed on the TRANSFORM SIZE
        # (not the input length, which several lengths share) plus device and dtype, so a float32
        # entry can never be handed to a float64 call and silently downgrade its precision.
        self._weights = {}

    def transform_size(self, n_samples: int) -> int:
        """The transform length used for an input of `n_samples`.

        `>= n + m - 1` is not a performance preference, it is a correctness requirement: Parseval
        over an L-point DFT gives the energy of the CIRCULAR convolution, so a shorter L silently
        returns a wrong number. It used to show up as visible time-domain wrap-around; with the
        inverse transform gone there is nothing to see.
        """
        return _quantized_size(n_samples + self.impulse_response.shape[-1] - 1)

    def weights_for(self, n_samples: int, dtype: torch.dtype) -> Tensor:
        """Cached Parseval weights for this input length, dtype and device.

        Resolved OUTSIDE any compiled region and passed to `forward` as a plain tensor, so the dict
        lookup never enters the graph.

        Built in the input's real dtype: a float32 weight vector handed to a float64 signal would
        silently downgrade the whole computation, and the float64 gradcheck along with it.
        """
        length = self.transform_size(n_samples)
        key = (length, self.impulse_response.device, dtype)
        cached = self._weights.get(key)
        if cached is None:
            cached = parseval_weights(self.impulse_response.to(dtype), length)
            if len(self._weights) >= self.cache_size:
                self._weights.pop(next(iter(self._weights)))  # oldest out; insertion-ordered dict
            self._weights[key] = cached
        return cached

    def forward(self, audio: Tensor) -> Tensor:
        """Total energy of the frequency-weighted signal, per [batch, channel, stem].

        NOT the filtered waveform. Nothing downstream needs the samples, only their energy, and
        computing the energy directly is what removes the inverse transform, the group-delay
        correction and the trim.

        This is where values differ from a same-window implementation: the result is the energy of
        the FULL linear convolution, so the filter's ring-in and ring-out at the buffer edges are
        counted rather than discarded. The difference is negligible for broadband residuals
        (5e-05 score units at 1 s) and material only for very sparse ones (0.775 for a single
        non-zero sample), because the discarded pre-ring is a fixed fraction of that transient's
        own energy rather than of the window's.

        Args:
            audio (torch.Tensor): [batch, channel, stem, time].

        Returns:
            torch.Tensor: [batch, channel, stem].
        """
        if audio.ndim != 4:
            raise ValueError("Audio input must have dimensions [batch, channel, stem, time].")

        # No device reassignment here. The buffers follow `.to(device)` like any other module's, so
        # mutating module state during a forward pass is neither needed nor wanted. A device
        # mismatch now raises torch's standard "Expected all tensors to be on the same device"
        # instead of being silently papered over.
        n_samples = audio.shape[-1]
        length = self.transform_size(n_samples)
        weights = self.weights_for(n_samples, audio.real.dtype)

        # Flattened to 2-D for the transform, then restored. `rfft(n=length)` zero-pads internally,
        # and MPS has no native constant-pad above 3 dimensions - it falls back to View Ops and
        # warns about the performance. A 2-D pad takes the native path on every backend, and the
        # energy sum collapses the time axis anyway, so the leading shape is just carried across.
        leading = audio.shape[:-1]
        spectrum = torch.fft.rfft(audio.reshape(-1, n_samples), n=length)
        energy = ((spectrum.real**2 + spectrum.imag**2) * weights).sum(dim=-1)
        return energy.reshape(leading)
