import unittest
import sys
sys.path.append("/Users/chris/Desktop/Whitebalance/torch-log-wmse-audio-quality")
import torch
import numpy as np
from torch_log_wmse_audio_quality import LogWMSE
from torch_log_wmse_audio_quality.utils import calculate_rms

class TestLogWMSELoss(unittest.TestCase):
    def setUp(self):
        pass # Anything shared between tests

    def test_calculate_rms(self):
        for i in range(10):
            with self.subTest(i=i):
                torch.manual_seed(i)
                samples = torch.rand(2, 2, 44100)
                rms = calculate_rms(samples)

                self.assertIsInstance(rms, torch.Tensor)
                self.assertEqual(rms.shape, (2, 2))

                print(f"Test {i}, RMS Value: {rms.mean()}")

    def test_calculate_log_wmse(self):
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
        audio_lengths = [0.1, 0.5, 1.0]  # Different audio lengths
        sample_rate = 44100
        audio_channels = 2 # stereo
        audio_stems = 3 # 3 audio stems
        batch = 4 # batch size

        for i, audio_length in enumerate(audio_lengths):
            log_wmse_loss = LogWMSE(audio_length=audio_length, sample_rate=sample_rate)
            for j in range(3):
                with self.subTest(i=i, j=j):
                    torch.manual_seed((i+1)*(j+1))  # Ensure reproducibility

                    # Generate random inputs (scale between -1 and 1)
                    audio_lengths_samples = int(audio_length * sample_rate)
                    unprocessed_audio = 2 * torch.rand(batch, audio_channels, audio_lengths_samples) - 1
                    processed_audio = unprocessed_audio.unsqueeze(1).expand(-1, audio_stems, -1, -1) * 0.1
                    target_audio = torch.zeros(batch, audio_stems, audio_channels, audio_lengths_samples)

                    loss = log_wmse_loss(unprocessed_audio, processed_audio, target_audio)

                    self.assertIsInstance(loss, torch.Tensor)
                    self.assertEqual(loss.ndim, 0)

                    print(f"Test {i}, Subtest {j}, Audio Length: {audio_length}, Loss: {loss}, Seed: {(i+1)*(j+1)}")

    def test_logWMSE_metric_comparison(self):
        """For comparison with the original logWMSE metric implementation in numpy."""
        audio_lengths = [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 50.0]  # Different audio lengths
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

if __name__ == "__main__":
    unittest.main()