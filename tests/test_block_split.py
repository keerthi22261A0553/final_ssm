# Tests use the real TSMS-Drone file lists only (no generated data).

import os
import random

import pytest

from block_split import (
    SPLIT_NAMES,
    LeakageError,
    chunk_sizes,
    load_splits,
    split_class,
    split_dir,
    split_distance,
    verify_no_overlap,
)
from config import CLASSES
from data_loader import RadarDataLoader, file_sort_key


@pytest.fixture(scope="module")
def loader():
    return RadarDataLoader()


@pytest.fixture(scope="module")
def distance(loader):
    return loader.selected_distances()[0]


def class_paths(loader, distance, class_name=CLASSES[0]):
    return loader.list_class_files(distance, class_name)


def numbers(rows):
    return sorted(int(r["file_number"]) for r in rows)


def test_chunks_are_consecutive_whole_files(loader, distance):

    splits, _ = split_distance(loader.find_mat_files(distance))

    for class_name in CLASSES:

        rows = {k: [r for r in v if r["class_name"] == class_name] for k, v in splits.items()}

        assert numbers(rows["train"]) == list(range(1, 351))
        assert numbers(rows["val"]) == list(range(351, 426))
        assert numbers(rows["test"]) == list(range(426, 501))


def test_70_15_15_sizes():

    assert chunk_sizes(500) == {"train": 350, "val": 75, "test": 75}

    # Every file assigned exactly once for any count
    for n in range(7, 1001):
        assert sum(chunk_sizes(n).values()) == n


def test_every_saved_distance_is_disjoint(loader):

    for distance in loader.selected_distances():

        splits = load_splits(distance)
        counts = verify_no_overlap(splits)

        assert all(n == 0 for n in counts.values())
        assert sum(len(v) for v in splits.values()) == len(loader.find_mat_files(distance))

        # Only the current split files exist (no stale lists)
        assert sorted(os.listdir(split_dir(distance))) == sorted(f"{n}.csv" for n in SPLIT_NAMES)


def test_order_does_not_depend_on_input_order(loader, distance):

    paths = class_paths(loader, distance)
    shuffled = paths[:]
    random.Random(0).shuffle(shuffled)

    assert split_class(paths)[0] == split_class(shuffled)[0]


def test_numeric_not_text_sort(loader, distance):

    paths = class_paths(loader, distance)

    assert [file_sort_key(p)[0] for p in sorted(paths, key=file_sort_key)] == list(range(1, 501))


def test_different_file_count_adapts(loader, distance):

    # 437 real files: 305 / 65 / 67, all assigned, still consecutive
    assignment, _ = split_class(class_paths(loader, distance)[:437])

    assert {k: len(v) for k, v in assignment.items()} == {"train": 305, "val": 65, "test": 67}
    assert [file_sort_key(p)[0] for k in SPLIT_NAMES for p in assignment[k]] == list(range(1, 438))


def test_too_few_files_is_refused(loader, distance):

    with pytest.raises(ValueError):
        split_class(class_paths(loader, distance)[:3])


def test_leak_is_detected(loader, distance):

    splits, _ = split_distance(loader.find_mat_files(distance))
    splits["train"].append(dict(splits["test"][0]))

    with pytest.raises(LeakageError):
        verify_no_overlap(splits)


def test_mixed_distances_refused(loader):

    first, second = loader.selected_distances()[:2]

    with pytest.raises(ValueError):
        split_distance(loader.find_mat_files(first) + loader.find_mat_files(second))
