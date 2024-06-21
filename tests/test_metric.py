# Add directory to python path if needed
import sys
sys.path.append("/Users/chris/Desktop/Whitebalance/torch-log-wmse")

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
        processed_audio = torch.ones(2, 3, 2, 44100)
        target_audio = torch.ones(2, 3, 2, 44100)

        values = log_wmse_loss._calculate_log_wmse(
            input_rms,
            log_wmse_loss.filters,
            processed_audio,
            target_audio,
        )

        self.assertIsInstance(values, torch.Tensor)
        self.assertEqual(values.shape, (2, 3, 2))

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
                    processed_audio = unprocessed_audio.unsqueeze(1).expand(-1, audio_stems, -1, -1) * 0.1
                    target_audio = torch.zeros(batch, audio_stems, audio_channels, audio_lengths_samples)

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
                    processed_audio = torch.rand(batch, audio_stems, audio_channels, audio_lengths_samples) * convert_decibels_to_amplitude_ratio(-60)
                    target_audio = torch.zeros(batch, audio_stems, audio_channels, audio_lengths_samples)

                    loss = log_wmse_loss(unprocessed_audio, processed_audio, target_audio)

                    self.assertIsInstance(loss, torch.Tensor)
                    self.assertEqual(loss.ndim, 0)

                    print(f"Test {i}, Subtest {j}, Audio Length: {audio_length}, Loss: {loss}, Seed: {(i+1)*(j+1)}")

    def test_logWMSE_metric_comparison(self):
        """For comparison with the original logWMSE metric implementation in numpy."""
        print("Test logWMSE metric comparison")
        audio_lengths = [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]  # Different audio lengths
        for i, audio_length in enumerate(audio_lengths):
            log_wmse_loss = LogWMSE(audio_length=audio_length, sample_rate=44100)
            for j in range(3):
                with self.subTest(i=i, j=j):
                    torch.manual_seed((i+1)*(j+1))  # Ensure reproducibility
                    np.random.seed((i+1)*(j+1))  # to make the test reproducible

                    # Generate random inputs
                    audio_lengths_samples = int(audio_length * 44100)
                    unprocessed_audio = torch.from_numpy(np.random.rand(2, audio_lengths_samples).astype(np.float32))[None, ...]
                    processed_audio = torch.from_numpy(np.random.rand(2, audio_lengths_samples).astype(np.float32))[None, None, ...].repeat(1, 4, 1, 1)
                    target_audio = torch.from_numpy(np.random.rand(2, audio_lengths_samples).astype(np.float32))[None, None, ...].repeat(1, 4, 1, 1)

                    loss = log_wmse_loss(unprocessed_audio, processed_audio, target_audio)

                    self.assertIsInstance(loss, torch.Tensor)
                    self.assertEqual(loss.ndim, 0)

                    print(f"Test {i}, Subtest {j}, Audio Length: {audio_length}, Loss: {loss}, Seed: {(i+1)*(j+1)}")

class TestFreqWeightingFilter(unittest.TestCase):
    def setUp(self):
        # Example audio data, replace with actual audio loading if needed
        self.plot_output = False
        self.sample_rate = 44100
        self.audio_length = 3.7516936
        tone = 440 # sine wave in Hz
        t = np.arange(0, int(self.audio_length*self.sample_rate)) / self.sample_rate
        self.audio = torch.tensor(0.5 * np.sin(2 * np.pi * tone * t)) # create sine wave
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