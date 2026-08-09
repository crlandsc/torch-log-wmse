import os
import sys
# insert(0, ...) not append: with append, a pip-installed copy of this package in
# site-packages shadows the working tree when this file is run directly
# (python tests/test_x.py puts tests/ on sys.path[0], not the repo root), so the
# suite would silently test the installed wheel instead of the code under edit.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest
import torch
# No numpy and no matplotlib at module scope. The package itself needs neither, so requiring them to
# COLLECT the suite means the suite cannot run in a bare install of what it is testing. matplotlib is
# imported lazily inside the one plotting block that uses it (disabled by default).
# conftest owns the sys.path insertion, the thread cap, and every construction of the metric and the
# filter. Constructing through it is what keeps the 1.0.0 constructor changes to one edit.
from tests.conftest import CEILING, make_filter, make_loss, make_metric
from torch_log_wmse.utils import calculate_rms, convert_decibels_to_amplitude_ratio

# Test alias package
# from torch_log_wmse_audio_quality import LogWMSE
# from torch_log_wmse_audio_quality.utils import calculate_rms, convert_decibels_to_amplitude_ratio
# from torch_log_wmse_audio_quality.freq_weighting_filter import prepare_impulse_response_fft, HumanHearingSensitivityFilter

class TestLogWMSELoss(unittest.TestCase):
    def setUp(self):
        pass # Anything shared between tests

    def test_calculate_rms(self):
        print("Test calculate_rms")
        for i in range(10):
            with self.subTest(i=i):
                torch.manual_seed(i)
                samples = torch.rand(2, 2, 44100)
                rms = calculate_rms(samples)

                self.assertIsInstance(rms, torch.Tensor)
                self.assertEqual(rms.shape, (2, 2))

                print(f"Test {i}, RMS Value: {rms.mean()}")

    def test_per_stem(self):
        """Replaces a test that reached into a private static with a hand-built input_rms.

        `per_stem` is the public way to get the same values, so the test no longer has to
        reconstruct the metric's internal call signature to check a shape.
        """
        print("Test per_stem")
        metric = make_metric(sample_rate=44100)
        unprocessed_audio = torch.rand(2, 2, 44100)          # [batch, channel, time]
        processed_audio = torch.ones(2, 2, 3, 44100)         # [batch, channel, stem, time]
        target_audio = torch.ones(2, 2, 3, 44100)

        values = metric.per_stem(unprocessed_audio, processed_audio, target_audio)

        self.assertIsInstance(values, torch.Tensor)
        self.assertEqual(values.shape, (2, 2, 3))  # [batch, channel, stem]
        # An exact match on every stem, so every element is the ceiling.
        self.assertTrue(torch.allclose(values, torch.full_like(values, CEILING), atol=1e-3))

        print(f"Values: {values}")

    def test_forward(self):
        print("Test forward")
        audio_lengths = [0.1, 0.5, 1.0]  # Different audio lengths
        sample_rate = 44100
        audio_channels = 2 # stereo
        audio_stems = 3 # 3 audio stems
        batch = 4 # batch size

        for i, audio_length in enumerate(audio_lengths):
            log_wmse_loss = make_loss(audio_length=audio_length, sample_rate=sample_rate)
            for j in range(3):
                with self.subTest(i=i, j=j):
                    torch.manual_seed((i+1)*(j+1)) # Ensure reproducibility

                    # Generate random inputs (scale between -1 and 1)
                    audio_lengths_samples = int(audio_length * sample_rate)
                    unprocessed_audio = 2 * torch.rand(batch, audio_channels, audio_lengths_samples) - 1
                    processed_audio = (2 * torch.rand(batch, audio_channels, audio_stems, audio_lengths_samples) - 1) * 0.1
                    target_audio = torch.zeros(batch, audio_channels, audio_stems, audio_lengths_samples)

                    loss = log_wmse_loss(unprocessed_audio, processed_audio, target_audio)

                    self.assertIsInstance(loss, torch.Tensor)
                    self.assertEqual(loss.ndim, 0)

                    print(f"Test {i}, Subtest {j}, Audio Length: {audio_length}, Loss: {loss}, Seed: {(i+1)*(j+1)}")

    def test_bypass_filter(self):
        print("Test forward with bypassing the frequency weighting filter")
        audio_lengths = [0.1, 0.5, 1.0]  # Different audio lengths
        sample_rate = 44100
        audio_channels = 2 # stereo
        audio_stems = 3 # 3 audio stems
        batch = 4 # batch size

        for i, audio_length in enumerate(audio_lengths):
            log_wmse_loss = make_loss(audio_length=audio_length, sample_rate=sample_rate, bypass_filter=True)
            for j in range(3):
                with self.subTest(i=i, j=j):
                    torch.manual_seed((i+1)*(j+1)) # Ensure reproducibility

                    # Generate random inputs (scale between -1 and 1)
                    audio_lengths_samples = int(audio_length * sample_rate)
                    unprocessed_audio = 2 * torch.rand(batch, audio_channels, audio_lengths_samples) - 1
                    processed_audio = (2 * torch.rand(batch, audio_channels, audio_stems, audio_lengths_samples) - 1) * 0.1
                    target_audio = torch.zeros(batch, audio_channels, audio_stems, audio_lengths_samples)

                    loss = log_wmse_loss(unprocessed_audio, processed_audio, target_audio)

                    self.assertIsInstance(loss, torch.Tensor)
                    self.assertEqual(loss.ndim, 0)

                    print(f"Test {i}, Subtest {j}, Audio Length: {audio_length}, Loss: {loss}, Seed: {(i+1)*(j+1)}")

    # Test forward with silence
    def test_forward_silence(self):
        print("Test forward with silence")
        audio_lengths = [0.1, 0.5, 1.0]
        sample_rate = 44100
        audio_channels = 2 # stereo
        audio_stems = 3 # 3 audio stems
        batch = 4 # batch size

        for i, audio_length in enumerate(audio_lengths):
            log_wmse_loss = make_loss(audio_length=audio_length, sample_rate=sample_rate)
            for j in range(3):
                with self.subTest(i=i, j=j):
                    torch.manual_seed((i+1)*(j+1)) # Ensure reproducibility

                    # Generate random inputs (scale between -1 and 1)
                    audio_lengths_samples = int(audio_length * sample_rate)
                    unprocessed_audio = torch.rand(batch, audio_channels, audio_lengths_samples) * convert_decibels_to_amplitude_ratio(-75)
                    processed_audio = torch.rand(batch, audio_channels, audio_stems, audio_lengths_samples) * convert_decibels_to_amplitude_ratio(-60)
                    target_audio = torch.zeros(batch, audio_channels, audio_stems, audio_lengths_samples)

                    loss = log_wmse_loss(unprocessed_audio, processed_audio, target_audio)

                    self.assertIsInstance(loss, torch.Tensor)
                    self.assertEqual(loss.ndim, 0)

                    print(f"Test {i}, Subtest {j}, Audio Length: {audio_length}, Loss: {loss}, Seed: {(i+1)*(j+1)}")

    def test_logWMSE_metric_comparison(self):
        """For comparison with the original logWMSE metric implementation in numpy."""
        print("Test logWMSE metric comparison")
        audio_lengths = [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]  # Different audio lengths
        channels = 2
        stems = 4
        for i, audio_length in enumerate(audio_lengths):
            log_wmse_loss = make_loss(audio_length=audio_length, sample_rate=44100)
            for j in range(3):
                with self.subTest(i=i, j=j):
                    torch.manual_seed((i+1)*(j+1))  # Ensure reproducibility

                    # Generate random inputs. torch.rand rather than np.random.rand: same U[0,1)
                    # float32 draw, and this test only asserts the return type and rank, so the
                    # generator identity is immaterial.
                    audio_lengths_samples = int(audio_length * 44100)
                    # Create [batch=1, channel=2, time] tensor for unprocessed_audio
                    unprocessed_audio = torch.rand(channels, audio_lengths_samples)[None, ...]  # [1, 2, time]
                    # Create [batch=1, channel=2, stem=4, time] tensors for processed/target audio
                    processed_audio = torch.rand(channels, audio_lengths_samples)[None, :, None, :].repeat(1, 1, stems, 1)  # [1, 2, 4, time]
                    target_audio = torch.rand(channels, audio_lengths_samples)[None, :, None, :].repeat(1, 1, stems, 1)  # [1, 2, 4, time]

                    loss = log_wmse_loss(unprocessed_audio, processed_audio, target_audio)

                    self.assertIsInstance(loss, torch.Tensor)
                    self.assertEqual(loss.ndim, 0)

                    print(f"Test {i}, Subtest {j}, Audio Length: {audio_length}, Loss: {loss}, Seed: {(i+1)*(j+1)}")

    def test_digital_silence_in_batch(self):
        loss_function = make_loss(audio_length=1)
        torch.manual_seed(0)
        # raw: [batch, channel, time]
        raw = torch.randn(2, 1, 44100, dtype=torch.float32)
        raw[0] = 0.0
        # est and gt: [batch, channel, time]
        est = torch.randn(2, 1, 44100, dtype=torch.float32)
        est[0] = 0.0
        gt = torch.randn(2, 1, 44100, dtype=torch.float32)
        gt[0] = 0.0
        # Transform to [batch, channel, stem=1, time]
        loss0 = loss_function(raw[0:1], est[0:1].unsqueeze(2), gt[0:1].unsqueeze(2)).detach().item()
        self.assertAlmostEqual(loss0, -73.6827, places=4)
        loss1 = loss_function(raw[1:2], est[1:2].unsqueeze(2), gt[1:2].unsqueeze(2)).detach().item()
        self.assertAlmostEqual(loss1, 2.7475, places=4)
        loss_combined = loss_function(raw, est.unsqueeze(2), gt.unsqueeze(2))
        self.assertFalse(torch.isnan(loss_combined))

    def test_reduction_options(self):
        print("Test reduction options")
        # Test shapes and data
        batch = 2
        channels = 2
        stems = 3
        samples = 44100
        
        # Create test tensors
        torch.manual_seed(42)
        unprocessed_audio = torch.rand(batch, channels, samples)
        processed_audio = torch.rand(batch, channels, stems, samples)
        target_audio = torch.rand(batch, channels, stems, samples)
        
        # Test mean reduction (default)
        mean_log_wmse = make_loss(audio_length=1.0, sample_rate=44100, reduction="mean")
        mean_loss = mean_log_wmse(unprocessed_audio, processed_audio, target_audio)
        self.assertIsInstance(mean_loss, torch.Tensor)
        self.assertEqual(mean_loss.ndim, 0)  # Should be a scalar
        
        # Test sum reduction
        sum_log_wmse = make_loss(audio_length=1.0, sample_rate=44100, reduction="sum")
        sum_loss = sum_log_wmse(unprocessed_audio, processed_audio, target_audio)
        self.assertIsInstance(sum_loss, torch.Tensor)
        self.assertEqual(sum_loss.ndim, 0)  # Should be a scalar
        
        # Test no reduction
        none_log_wmse = make_loss(audio_length=1.0, sample_rate=44100, reduction="none")
        none_loss = none_log_wmse(unprocessed_audio, processed_audio, target_audio)
        self.assertIsInstance(none_loss, torch.Tensor)
        # BREAKING CHANGE, asserted deliberately rather than relaxed. `reduction` now controls the
        # BATCH axis only, like any other torch loss; channel and stem are pooled first. So "none"
        # is one value per batch item, not one per [batch, channel, stem]. Use per_stem() for those.
        self.assertEqual(none_loss.shape, (batch,))
        self.assertEqual(
            none_log_wmse.per_stem(unprocessed_audio, processed_audio, target_audio).shape,
            (batch, channels, stems))

        # And sum/mean is now the BATCH size, where it was batch x channels x stems (12 here).
        self.assertAlmostEqual(sum_loss.item() / mean_loss.item(), batch, delta=0.1)

        print(f"Mean reduction loss: {mean_loss.item()}")
        print(f"Sum reduction loss: {sum_loss.item()}")
        print(f"No reduction loss shape: {none_loss.shape}")

class TestFreqWeightingFilter(unittest.TestCase):
    """The filter reports weighted ENERGY per [batch, channel, stem], not a filtered waveform.

    The waveform is never materialised: the metric only needs its energy, and Parseval gives that
    from the forward transform alone. That is what removes the inverse transform, the group-delay
    correction and the trim.

    The plotting block that used to live here went with the waveform, which also leaves this suite
    with no matplotlib reference of any kind. `tests/test_invariants.py` carries the behavioural
    checks - a delta impulse response reproducing the unfiltered score, the one-sided Parseval
    weights, and the per-size cache.
    """

    def setUp(self):
        self.sample_rate = 44100
        self.audio_length = 3.7516936
        tone = 440  # sine wave in Hz
        # float64 deliberately: it keeps this test exercising dtype promotion against a float32
        # impulse response, and the weights being built in the INPUT's dtype rather than the IR's.
        t = torch.arange(int(self.audio_length * self.sample_rate), dtype=torch.float64) / self.sample_rate
        self.audio = 0.5 * torch.sin(2 * math.pi * tone * t)
        # Shape to [batch=1, channel=1, stem=1, time]
        self.audio = self.audio[None, None, None, :]

    def test_returns_energy_per_element(self):
        energy = make_filter(sample_rate=self.sample_rate)(self.audio)
        self.assertEqual(energy.shape, self.audio.shape[:-1])  # the time axis is consumed
        self.assertEqual(energy.dtype, torch.float64)  # follows the input, not the float32 IR
        self.assertTrue(torch.isfinite(energy).all())
        self.assertGreater(float(energy), 0.0)

    def test_a_440_hz_tone_is_attenuated_less_than_a_10_khz_one(self):
        """A shape check on the weighting curve itself: hearing sensitivity peaks in the low kHz,
        so an equal-amplitude 10 kHz tone must come back with less energy than 440 Hz."""
        f = make_filter(sample_rate=self.sample_rate)
        n = self.audio.shape[-1]
        t = torch.arange(n, dtype=torch.float64) / self.sample_rate
        high = (0.5 * torch.sin(2 * math.pi * 10000 * t))[None, None, None, :]
        self.assertGreater(float(f(self.audio)), float(f(high)))


if __name__ == "__main__":
    unittest.main()