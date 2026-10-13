import numpy as np


def iq_to_spectrogram_np(segments, n_fft=128, stft_hop=32):

    # NumPy version of dataset.iq_to_spectrogram (same output), so
    # edge inference does not need PyTorch.
    # segments: (num_segments, 2, segment_length) array of I and Q

    segments = np.asarray(segments, dtype=np.float32)

    # Complex signal I + jQ keeps positive and negative Doppler
    signal = segments[:, 0] + 1j * segments[:, 1]

    # Same framing as torch.stft(center=True, pad_mode="reflect")
    pad = n_fft // 2
    signal = np.pad(signal, ((0, 0), (pad, pad)), mode="reflect")

    num_frames = 1 + (signal.shape[1] - n_fft) // stft_hop

    index = (
        np.arange(n_fft)[None, :]
        + stft_hop * np.arange(num_frames)[:, None]
    )

    # Periodic Hann window, as torch.hann_window
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)

    frames = signal[:, index] * window

    # (num_segments, n_fft, num_frames)
    stft = np.fft.fft(frames, axis=-1).transpose(0, 2, 1)

    # Put zero Doppler in the centre of the frequency axis
    stft = np.fft.fftshift(stft, axes=1)

    # Log-magnitude in dB
    spectrogram = 20 * np.log10(np.abs(stft) + 1e-6)

    # Per-segment z-score (unbiased std, as torch.std)
    mean = spectrogram.mean(axis=(1, 2), keepdims=True)
    std = spectrogram.std(axis=(1, 2), ddof=1, keepdims=True)

    spectrogram = (spectrogram - mean) / (std + 1e-8)

    # Shape: (num_segments, 1, n_fft, num_frames)
    return spectrogram[:, None].astype(np.float32)


if __name__ == "__main__":

    import os
    import sys

    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from dataset import iq_to_spectrogram

    rng = np.random.default_rng(0)

    segments = rng.standard_normal((16, 2, 1024)).astype(np.float32)

    reference = iq_to_spectrogram(torch.from_numpy(segments)).numpy()
    result = iq_to_spectrogram_np(segments)

    print("Shape                 :", result.shape)
    print("Max diff vs PyTorch   :", f"{np.abs(result - reference).max():.2e}")

    assert result.shape == reference.shape
    assert np.abs(result - reference).max() < 1e-3
