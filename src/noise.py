import zlib

import numpy as np

from config import RANDOM_SEED, SAMPLING_RATE, TONE_FREQUENCY
from preprocessing import RadarPreprocessor

_preprocessor = RadarPreprocessor(SAMPLING_RATE, TONE_FREQUENCY)


def target_power(signal):

    # Power of the moving-target (micro-Doppler) part of the signal:
    # shift zero Doppler to DC and remove the stationary return, as the
    # preprocessing does. Same definition as the original noise test.
    baseband = _preprocessor.shift_to_baseband(signal)

    return float(np.mean(np.abs(baseband - baseband.mean()) ** 2))


def noise_seed(file_path, snr_db, seed=RANDOM_SEED):

    # Fixed seed per (file, SNR): the same file at the same SNR always
    # gets the same noise, whatever the run order. zlib.crc32 is stable
    # between Python runs (the built-in hash() is not).
    key = zlib.crc32(file_path.replace("\\", "/").encode())

    return np.random.SeedSequence([seed, key, int(round(snr_db * 100)) + 100000])


def add_awgn(signal, snr_db, rng):

    # Complex circular white Gaussian noise; SNR relative to target power.
    # Returns a NEW array; the input signal is not modified.
    signal = np.asarray(signal)

    noise_power = target_power(signal) / 10 ** (snr_db / 10)

    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
    )

    return signal + noise


def measured_snr_db(clean, noisy):

    # SNR actually obtained: target power / added noise power
    noise = np.asarray(noisy) - np.asarray(clean)

    return 10 * np.log10(target_power(clean) / np.mean(np.abs(noise) ** 2))


class AWGN:

    # signal_transform for RadarSegmentDataset: (signal, row) -> noisy copy

    def __init__(self, snr_db):

        self.snr_db = snr_db

    def __call__(self, signal, row):

        rng = np.random.default_rng(noise_seed(row["file_path"], self.snr_db))

        return add_awgn(signal, self.snr_db, rng)

    def __repr__(self):

        return f"AWGN({self.snr_db} dB)"
