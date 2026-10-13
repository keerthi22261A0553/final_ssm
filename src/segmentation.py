import numpy as np


class RadarSegmenter:

    def __init__(self, segment_length=1024, hop_length=None):

        # Non-overlapping segments by default
        if hop_length is None:
            hop_length = segment_length

        if segment_length <= 0:
            raise ValueError("Segment length must be positive.")

        if hop_length <= 0:
            raise ValueError("Hop length must be positive.")

        self.segment_length = segment_length
        self.hop_length = hop_length

    def segment(self, signal_data):

        signal_data = np.asarray(signal_data).squeeze()

        if signal_data.ndim != 1:
            raise ValueError("Input signal must be 1-D.")

        # Segment length valid? (Figure A-1)
        if self.segment_length > len(signal_data):
            raise ValueError(
                f"Segment length {self.segment_length} is longer "
                f"than signal length {len(signal_data)}."
            )

        num_segments = (
            (len(signal_data) - self.segment_length)
            // self.hop_length
            + 1
        )

        starts = np.arange(num_segments) * self.hop_length

        # Shape: (num_segments, segment_length)
        return np.stack([
            signal_data[s:s + self.segment_length]
            for s in starts
        ])

    def segment_iq(self, i_signal, q_signal):

        i_segments = self.segment(i_signal)
        q_segments = self.segment(q_signal)

        # Shape: (num_segments, 2, segment_length)
        return np.stack([i_segments, q_segments], axis=1)


if __name__ == "__main__":

    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from data_loader import RadarDataLoader
    from preprocessing import RadarPreprocessor

    loader = RadarDataLoader()
    preprocessor = RadarPreprocessor()
    segmenter = RadarSegmenter(segment_length=1024)

    record = loader.find_mat_files()[0]
    signal = loader.load_mat_file(record["file_path"])

    i_norm, q_norm = preprocessor.process(signal)

    segments = segmenter.segment_iq(i_norm, q_norm)

    print("File           :", record["file_path"])
    print("Signal length  :", len(signal))
    print("Segment length :", segmenter.segment_length)
    print("Hop length     :", segmenter.hop_length)
    print("Segments shape :", segments.shape)
