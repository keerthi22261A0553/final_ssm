import numpy as np
from scipy.signal import istft, stft

from config import DENOISER, SAMPLING_RATE


class STFTWienerDenoiser:

    # Blind STFT-domain Wiener filter (spectral subtraction family).
    #
    # Operates on the raw complex radar signal, before preprocessing:
    #   1. complex STFT (Hann window, 75 % overlap, two-sided spectrum)
    #   2. noise power per bin estimated from the MEDIAN of all bin powers:
    #      white noise has the same expected power in every bin and an
    #      exponential distribution, median = sigma^2 * ln 2; micro-Doppler
    #      and the stationary return occupy only a minority of bins
    #   3. gain G = max(gain_floor, 1 - sigma^2 / |X|^2) per bin
    #   4. inverse STFT back to a complex signal of the same length
    #
    # Uses only the signal itself: no labels, no clean reference, no
    # parameters fitted to any data.

    name = "stft_wiener"

    def __init__(self, nperseg=256, overlap=0.75, gain_floor=0.0):

        self.nperseg = nperseg
        self.noverlap = int(nperseg * overlap)
        self.gain_floor = gain_floor

    def _stft(self, signal):

        return stft(
            signal,
            fs=SAMPLING_RATE,
            window="hann",
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            return_onesided=False,
            boundary="even",
            padded=True,
        )[2]

    def noise_power(self, spectrum):

        # Per-bin noise power estimate from the median bin power
        return np.median(np.abs(spectrum) ** 2) / np.log(2)

    def gain(self, spectrum):

        power = np.abs(spectrum) ** 2

        gain = 1 - self.noise_power(spectrum) / np.maximum(power, 1e-30)

        return np.clip(gain, self.gain_floor, 1.0)

    def __call__(self, signal, row=None):

        # Returns a NEW complex array of the same length
        signal = np.asarray(signal)

        spectrum = self._stft(signal)

        _, denoised = istft(
            spectrum * self.gain(spectrum),
            fs=SAMPLING_RATE,
            window="hann",
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            input_onesided=False,
            boundary=True,
        )

        return denoised[:len(signal)].astype(signal.dtype)

    def __repr__(self):

        return (f"STFTWienerDenoiser(nperseg={self.nperseg}, "
                f"noverlap={self.noverlap}, gain_floor={self.gain_floor})")


DENOISERS = {STFTWienerDenoiser.name: STFTWienerDenoiser}


def build_denoiser(settings=DENOISER):

    # Replaceable: add a class with __call__(signal, row) -> new signal
    # to DENOISERS and select it in config.DENOISER
    settings = dict(settings)

    method = settings.pop("method")

    return DENOISERS[method](**settings)


class NoisyThenDenoised:

    # signal_transform for RadarSegmentDataset: add noise, then denoise
    # THAT SAME noisy signal (the noisy-only condition uses the same
    # noise transform, so both conditions see identical noisy input)

    def __init__(self, noise_transform, denoiser):

        self.noise_transform = noise_transform
        self.denoiser = denoiser

    def __call__(self, signal, row):

        return self.denoiser(self.noise_transform(signal, row), row)

    def __repr__(self):

        return f"{self.noise_transform!r} -> {self.denoiser!r}"
