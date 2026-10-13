# Tests use the real TSMS-Drone files only (no generated data).

import numpy as np
import pytest
import torch

from block_split import SPLIT_NAMES, load_splits
from data_loader import RadarDataLoader
from dataset import RadarSegmentDataset


@pytest.fixture(scope="module")
def distance():
    return RadarDataLoader().selected_distances()[0]


@pytest.fixture(scope="module")
def test_set(distance):
    return RadarSegmentDataset(distance, "test")


def test_only_files_of_the_split(distance, test_set):

    splits = load_splits(distance)
    expected = [r["file_path"] for r in splits["test"]]

    assert [r["file_path"] for r in test_set.records] == expected

    others = {r["file_path"] for name in SPLIT_NAMES if name != "test" for r in splits[name]}
    assert not others & set(expected)


def test_segments_stay_with_their_file(test_set):

    # Every segment points to one file of this split, labels agree
    assert test_set.file_index.min() == 0
    assert test_set.file_index.max() == len(test_set.records) - 1

    file_labels = torch.from_numpy(test_set.file_labels())
    assert torch.equal(test_set.labels, file_labels[test_set.file_index])

    # Segments of a file are contiguous (no mixing between files)
    assert torch.all(test_set.file_index[1:] >= test_set.file_index[:-1])


def test_transform_gets_a_copy_and_file_is_unchanged(distance):

    seen = []

    def transform(signal, row):
        seen.append(row["file_path"])
        signal *= 0.5          # in-place change on the copy
        return signal

    dataset = RadarSegmentDataset(distance, "val", signal_transform=transform)

    assert seen == [r["file_path"] for r in dataset.records]

    # Re-reading the file gives the original data
    loader = RadarDataLoader()
    row = dataset.records[0]
    before = loader.load_mat_file(row["abs_path"])
    dataset.process_signal(before, row)
    after = loader.load_mat_file(row["abs_path"])

    assert np.array_equal(before, after)


def test_shapes(test_set):

    assert test_set.inputs.shape[1:] == (1, 128, 33)
    assert len(test_set) == len(test_set.labels) == len(test_set.file_index)
