import numpy as np


class RadarPreprocessor:

    # TSMS-Drone CW radar (ADALM-Pluto): 65,536 Hz sampling rate.
    # Zero Doppler (direct return / stationary clutter) sits at +1,024 Hz.

    def __init__(self, sampling_rate=65536, tone_frequency=1024):

        self.sampling_rate = sampling_rate
        self.tone_frequency = tone_frequency

    def shift_to_baseband(self, complex_signal):

        complex_signal = np.asarray(complex_signal).squeeze()

        n = np.arange(len(complex_signal))

        # Mix down so that zero Doppler lands at 0 Hz
        return complex_signal * np.exp(
            -2j * np.pi * self.tone_frequency * n / self.sampling_rate
        )

    def separate_iq(self, complex_signal):

        complex_signal = np.asarray(complex_signal).squeeze()

        if not np.iscomplexobj(complex_signal):
            raise ValueError("Input signal is not complex-valued.")

        i_signal = np.real(complex_signal)
        q_signal = np.imag(complex_signal)

        return i_signal, q_signal

    def remove_dc(self, signal_data):

        signal_data = np.asarray(signal_data, dtype=np.float32)

        return signal_data - np.mean(signal_data)

    def normalize(self, signal_data):

        signal_data = np.asarray(signal_data, dtype=np.float32)

        mean = np.mean(signal_data)
        std = np.std(signal_data)

        if std < 1e-8:
            raise ValueError("Standard deviation is (near) zero.")

        return (signal_data - mean) / std

    def process(self, complex_signal):

        if not np.iscomplexobj(complex_signal):
            raise ValueError("Input signal is not complex-valued.")

        # Filtering: after the shift, the stationary return is at DC,
        # so removing the mean of I and Q is a clutter-removal filter
        baseband = self.shift_to_baseband(complex_signal)

        i_signal, q_signal = self.separate_iq(baseband)

        i_norm = self.normalize(self.remove_dc(i_signal))
        q_norm = self.normalize(self.remove_dc(q_signal))

        return i_norm, q_norm


if __name__ == "__main__":

    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from data_loader import RadarDataLoader

    # Load first radar file
    loader = RadarDataLoader()
    record = loader.find_mat_files()[0]
    signal = loader.load_mat_file(record["file_path"])

    print("File:", record["file_path"])

    # Create preprocessor
    preprocessor = RadarPreprocessor()

    # -------------------------------------------------
    # Step 0: Shift zero Doppler (+1,024 Hz tone) to 0 Hz
    # -------------------------------------------------

    freqs = np.fft.fftfreq(len(signal), 1 / preprocessor.sampling_rate)

    baseband = preprocessor.shift_to_baseband(signal)

    print("Strongest frequency before shift:", freqs[np.abs(np.fft.fft(signal)).argmax()], "Hz")
    print("Strongest frequency after shift :", freqs[np.abs(np.fft.fft(baseband)).argmax()], "Hz")

    # -------------------------------------------------
    # Step 1: Separate I and Q
    # -------------------------------------------------

    i_signal, q_signal = preprocessor.separate_iq(baseband)

    print("Complex signal shape:", signal.shape)
    print("I shape:", i_signal.shape)
    print("Q shape:", q_signal.shape)

    print("\nFirst 5 complex samples:")
    print(signal[:5])

    print("\nFirst 5 I samples:")
    print(i_signal[:5])

    print("\nFirst 5 Q samples:")
    print(q_signal[:5])

    # -------------------------------------------------
    # Step 2: Clutter removal (remove DC after the shift)
    # -------------------------------------------------

    i_dc = preprocessor.remove_dc(i_signal)
    q_dc = preprocessor.remove_dc(q_signal)

    print("\nMean before clutter removal:")
    print("I mean:", np.mean(i_signal))
    print("Q mean:", np.mean(q_signal))

    print("\nMean after clutter removal:")
    print("I mean:", np.mean(i_dc))
    print("Q mean:", np.mean(q_dc))

    removed_db = 10 * np.log10(
        (np.mean(i_dc ** 2) + np.mean(q_dc ** 2))
        / np.mean(np.abs(signal) ** 2)
    )

    print(f"Power after clutter removal: {removed_db:.1f} dB (stationary tone removed)")

    # -------------------------------------------------
    # Step 3: Z-score normalization
    # -------------------------------------------------

    i_norm = preprocessor.normalize(i_dc)
    q_norm = preprocessor.normalize(q_dc)

    print("\nAfter normalization:")

    print("I mean:", np.mean(i_norm))
    print("I standard deviation:", np.std(i_norm))

    print("Q mean:", np.mean(q_norm))
    print("Q standard deviation:", np.std(q_norm))

    print("\nFirst 5 normalized I samples:")
    print(i_norm[:5])

    print("\nFirst 5 normalized Q samples:")
    print(q_norm[:5])

    # -------------------------------------------------
    # Full pipeline check
    # -------------------------------------------------

    i_proc, q_proc = preprocessor.process(signal)

    print("\nprocess() matches step-by-step:",
          np.allclose(i_proc, i_norm) and np.allclose(q_proc, q_norm))