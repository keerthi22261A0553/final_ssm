import numpy as np
import torch
from torch.utils.data import Dataset

from block_split import load_split, load_splits, verify_no_overlap
from config import (
    N_FFT,
    SAMPLING_RATE,
    SEGMENT_HOP,
    SEGMENT_LENGTH,
    STFT_HOP,
    TONE_FREQUENCY,
)
from data_loader import RadarDataLoader
from preprocessing import RadarPreprocessor
from segmentation import RadarSegmenter


def iq_to_spectrogram(segments, n_fft=N_FFT, stft_hop=STFT_HOP):

    # Unchanged from the original project.
    # segments: (num_segments, 2, segment_length) tensor of I and Q

    # Complex signal I + jQ keeps positive and negative Doppler
    complex_segments = torch.complex(segments[:, 0], segments[:, 1])

    stft = torch.stft(
        complex_segments,
        n_fft=n_fft,
        hop_length=stft_hop,
        window=torch.hann_window(n_fft),
        return_complex=True,
    )

    # Put zero Doppler in the centre of the frequency axis
    stft = torch.fft.fftshift(stft, dim=1)

    # Log-magnitude in dB
    spectrogram = 20 * torch.log10(stft.abs() + 1e-6)

    # Per-segment z-score normalization
    mean = spectrogram.mean(dim=(1, 2), keepdim=True)
    std = spectrogram.std(dim=(1, 2), keepdim=True)

    spectrogram = (spectrogram - mean) / (std + 1e-8)

    # Shape: (num_segments, 1, n_fft, num_frames)
    return spectrogram.unsqueeze(1)


class RadarSegmentDataset(Dataset):

    # Segments of the whole files listed in data/splits/<distance>/<split>.csv.
    #
    # Order of operations (split first, segmentation second):
    #   split csv (whole files) -> load file -> [signal_transform on a copy]
    #   -> preprocessing -> segmentation -> spectrogram
    #
    # signal_transform(signal, row) -> new signal is optional; the noise
    # and denoising experiments use it. The .mat file is only read.

    def __init__(self, distance, split, signal_transform=None):

        self.distance = distance
        self.split = split
        self.signal_transform = signal_transform

        # Refuse to build anything if the saved split leaks
        verify_no_overlap(load_splits(distance))

        self.records = load_split(distance, split)

        wrong = {r["distance"] for r in self.records} - {distance}

        if wrong:
            raise ValueError(f"{split}.csv of {distance} lists other distances: {wrong}")

        self.loader = RadarDataLoader()
        self.preprocessor = RadarPreprocessor(SAMPLING_RATE, TONE_FREQUENCY)
        self.segmenter = RadarSegmenter(SEGMENT_LENGTH, SEGMENT_HOP)

        # Build all spectrograms once and keep them in memory
        self.inputs, self.labels, self.file_index = self._build()

    def process_signal(self, signal, row):

        if self.signal_transform is not None:

            # Work on a copy so the loaded clean signal is never changed
            signal = self.signal_transform(np.array(signal, copy=True), row)

        i_norm, q_norm = self.preprocessor.process(signal)

        # (num_segments, 2, segment_length)
        return self.segmenter.segment_iq(i_norm, q_norm)

    def _build(self):

        inputs = []
        labels = []
        file_index = []

        for index, row in enumerate(self.records):

            signal = self.loader.load_mat_file(row["abs_path"])

            segments = torch.from_numpy(
                self.process_signal(signal, row).astype(np.float32)
            )

            # Spectrograms per file keep peak memory low; the STFT works
            # on each segment separately, so the result is identical
            inputs.append(iq_to_spectrogram(segments))

            # Every segment carries its file: all segments of one file
            # stay in this split, and file-level voting is exact
            labels.extend([row["label"]] * len(segments))
            file_index.extend([index] * len(segments))

        return (
            torch.cat(inputs),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(file_index, dtype=torch.long),
        )

    def file_labels(self):

        return np.array([r["label"] for r in self.records])

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        return self.inputs[index], self.labels[index]


class NoiseAugmentedDataset(RadarSegmentDataset):

    # Training files only, for the optional noise-augmented model.
    # resample(epoch) draws new noise for every file: with probability
    # clean_fraction the file stays clean, otherwise complex AWGN at an
    # SNR drawn uniformly from snr_range_db (same SNR definition as the
    # noise test). The draw depends only on (seed, epoch, file), so a
    # resumed run gets exactly the same noise.

    def __init__(self, distance, snr_range_db, clean_fraction, seed):

        self.snr_range_db = snr_range_db
        self.clean_fraction = clean_fraction
        self.seed = seed
        self.epoch = None

        super().__init__(distance, "train")

        # Raw signals kept in memory (read once; files never modified)
        self.signals = [
            self.loader.load_mat_file(r["abs_path"]).astype(np.complex64)
            for r in self.records
        ]

    def resample(self, epoch):

        from noise import add_awgn

        if epoch == self.epoch:
            return

        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 7, epoch]))

        inputs = []

        for signal in self.signals:

            if rng.random() >= self.clean_fraction:
                signal = add_awgn(signal, rng.uniform(*self.snr_range_db), rng)

            i_norm, q_norm = self.preprocessor.process(signal)

            segments = self.segmenter.segment_iq(i_norm, q_norm)

            inputs.append(iq_to_spectrogram(torch.from_numpy(segments.astype(np.float32))))

        # Same files, same order, same labels; only the inputs change
        self.inputs = torch.cat(inputs)
        self.epoch = epoch


if __name__ == "__main__":

    import time

    from spectrogram import iq_to_spectrogram_np

    distance = RadarDataLoader().selected_distances()[0]

    start = time.time()

    dataset = RadarSegmentDataset(distance, "val")

    counts = torch.bincount(dataset.file_index)

    print(f"Distance / split : {distance} / val")
    print(f"Build time       : {time.time() - start:.1f} s")
    print(f"Files            : {len(dataset.records)}")
    print(f"Segments         : {len(dataset)}")
    print(f"Segments per file: {counts.min().item()}-{counts.max().item()}")
    print(f"Input shape      : {tuple(dataset[0][0].shape)}")
    print(f"Files per class  : {np.bincount(dataset.file_labels()).tolist()}")
    print(f"Segments / class : {torch.bincount(dataset.labels).tolist()}")

    # Same result as the NumPy spectrogram used for edge inference
    row = dataset.records[0]
    segments = dataset.process_signal(dataset.loader.load_mat_file(row["abs_path"]), row)
    reference = iq_to_spectrogram_np(segments)
    built = dataset.inputs[dataset.file_index == 0].numpy()

    print(f"Max diff vs NumPy spectrogram: {np.abs(built - reference).max():.2e}")
