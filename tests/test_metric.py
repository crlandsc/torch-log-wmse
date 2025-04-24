import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_log_wmse import LogWMSE
from torch_log_wmse.utils import calculate_rms, convert_decibels_to_amplitude_ratio
from torch_log_wmse.freq_weighting_filter import prepare_impulse_response_fft, HumanHearingSensitivityFilter

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

    def test_calculate_log_wmse(self):
        print("Test calculate_log_wmse")
        log_wmse_loss = LogWMSE(audio_length=1.0, sample_rate=44100)
        input_rms = torch.ones(2, 2)
        processed_audio = torch.ones(2, 2, 3, 44100)  # [batch, channel, stem, time]
        target_audio = torch.ones(2, 2, 3, 44100)  # [batch, channel, stem, time]

        values = log_wmse_loss._calculate_log_wmse(
            input_rms,
            log_wmse_loss.filters,
            processed_audio,
            target_audio,
        )

        self.assertIsInstance(values, torch.Tensor)
        self.assertEqual(values.shape, (2, 2, 3))  # [batch, channel, stem]

        print(f"Values: {values}")

    def test_forward(self):
        print("Test forward")
        audio_lengths = [0.1, 0.5, 1.0]  # Different audio lengths
        sample_rate = 44100
        audio_channels = 2 # stereo
        audio_stems = 3 # 3 audio stems
        batch = 4 # batch size

        for i, audio_length in enumerate(audio_lengths):
            log_wmse_loss = LogWMSE(audio_length=audio_length, sample_rate=sample_rate)
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
            log_wmse_loss = LogWMSE(audio_length=audio_length, sample_rate=sample_rate, bypass_filter=True)
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
            log_wmse_loss = LogWMSE(audio_length=audio_length, sample_rate=sample_rate)
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
            log_wmse_loss = LogWMSE(audio_length=audio_length, sample_rate=44100)
            for j in range(3):
                with self.subTest(i=i, j=j):
                    torch.manual_seed((i+1)*(j+1))  # Ensure reproducibility
                    np.random.seed((i+1)*(j+1))  # to make the test reproducible

                    # Generate random inputs
                    audio_lengths_samples = int(audio_length * 44100)
                    # Create [batch=1, channel=2, time] tensor for unprocessed_audio
                    unprocessed_audio = torch.from_numpy(np.random.rand(channels, audio_lengths_samples).astype(np.float32))[None, ...]  # [1, 2, time]
                    # Create [batch=1, channel=2, stem=4, time] tensors for processed/target audio
                    processed_audio = torch.from_numpy(np.random.rand(channels, audio_lengths_samples).astype(np.float32))[None, :, None, :].repeat(1, 1, stems, 1)  # [1, 2, 4, time]
                    target_audio = torch.from_numpy(np.random.rand(channels, audio_lengths_samples).astype(np.float32))[None, :, None, :].repeat(1, 1, stems, 1)  # [1, 2, 4, time]

                    loss = log_wmse_loss(unprocessed_audio, processed_audio, target_audio)

                    self.assertIsInstance(loss, torch.Tensor)
                    self.assertEqual(loss.ndim, 0)

                    print(f"Test {i}, Subtest {j}, Audio Length: {audio_length}, Loss: {loss}, Seed: {(i+1)*(j+1)}")

    def test_digital_silence_in_batch(self):
        loss_function = LogWMSE(audio_length=1, return_as_loss=True)
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
        mean_log_wmse = LogWMSE(audio_length=1.0, sample_rate=44100, reduction="mean")
        mean_loss = mean_log_wmse(unprocessed_audio, processed_audio, target_audio)
        self.assertIsInstance(mean_loss, torch.Tensor)
        self.assertEqual(mean_loss.ndim, 0)  # Should be a scalar
        
        # Test sum reduction
        sum_log_wmse = LogWMSE(audio_length=1.0, sample_rate=44100, reduction="sum")
        sum_loss = sum_log_wmse(unprocessed_audio, processed_audio, target_audio)
        self.assertIsInstance(sum_loss, torch.Tensor)
        self.assertEqual(sum_loss.ndim, 0)  # Should be a scalar
        
        # Test no reduction
        none_log_wmse = LogWMSE(audio_length=1.0, sample_rate=44100, reduction="none")
        none_loss = none_log_wmse(unprocessed_audio, processed_audio, target_audio)
        self.assertIsInstance(none_loss, torch.Tensor)
        self.assertEqual(none_loss.shape, (batch, channels, stems))  # Should preserve dimensions
        
        # Verify mathematical relationship between mean and sum
        # The sum should be approximately batch*channels*stems times the mean
        expected_factor = batch * channels * stems
        self.assertAlmostEqual(
            sum_loss.item() / mean_loss.item(),
            expected_factor,
            delta=0.1  # Allow some tolerance for floating point differences
        )
        
        print(f"Mean reduction loss: {mean_loss.item()}")
        print(f"Sum reduction loss: {sum_loss.item()}")
        print(f"No reduction loss shape: {none_loss.shape}")

class TestFreqWeightingFilter(unittest.TestCase):
    def setUp(self):
        # Example audio data, replace with actual audio loading if needed
        self.plot_output = False
        self.sample_rate = 44100
        self.audio_length = 3.7516936
        tone = 440 # sine wave in Hz
        t = np.arange(0, int(self.audio_length*self.sample_rate)) / self.sample_rate
        self.audio = torch.tensor(0.5 * np.sin(2 * np.pi * tone * t)) # create sine wave
        # Shape to [batch=1, channel=1, stem=1, time]
        self.audio = self.audio[None, None, None, :]

    def test_prepare_impulse_response_fft(self):
        print("Test prepare_impulse_response_fft")
        ir = torch.rand(512)  # Example impulse response
        fft_size = 1024
        ir_fft = prepare_impulse_response_fft(ir, fft_size)
        self.assertEqual(ir_fft.shape[-1], fft_size//2+1)

    def test_HumanHearingSensitivityFilter(self):
        print("Test HumanHearingSensitivityFilter")
        plot_upper_bound = 500
        hhs_filter = HumanHearingSensitivityFilter(audio_length=self.audio_length, sample_rate=self.sample_rate)
        # Add zeros at index 50-100 to demonstrate time alignment
        self.audio[:, :, :, 50:100] = 0
        self.audio[:, :, :, 101:125] = 0.5
        self.audio[:, :, :, 126:150] = -0.5
        self.audio[:, :, :, 151:200] = 0

        filtered_audio = hhs_filter(self.audio)

        # Plot the first 1000 samples before and after filtering
        if self.plot_output:
            fig, axs = plt.subplots(2, 1, figsize=(12, 8))
            axs[0].plot(self.audio.squeeze()[:plot_upper_bound])
            axs[0].set_title(f'Original Audio (First {plot_upper_bound} Samples)')
            axs[0].set_ylim(-1, 1)
            axs[1].plot(filtered_audio.squeeze()[:plot_upper_bound])
            axs[1].set_title(f'Filtered Audio (First {plot_upper_bound} Samples)')
            axs[1].set_ylim(-1, 1)
            plt.tight_layout()
            plt.show()
        else:
            print("Plotting disabled.")

        self.assertEqual(filtered_audio.shape, self.audio.shape)


if __name__ == "__main__":
    unittest.main()