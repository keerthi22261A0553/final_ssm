# Tests use real TSMS-Drone test files only; noise is added to copies.

import numpy as np
import pytest

from block_split import load_split
from config import SNR_LEVELS
from data_loader import RadarDataLoader
from noise import AWGN, measured_snr_db


@pytest.fixture(scope="module")
def rows():
    loader = RadarDataLoader()
    test_rows = load_split(loader.selected_distances()[0], "test")
    # First file of each class
    return [next(r for r in test_rows if r["label"] == k) for k in range(4)]


@pytest.fixture(scope="module")
def signals(rows):
    loader = RadarDataLoader()
    return [loader.load_mat_file(r["abs_path"]) for r in rows]


@pytest.mark.parametrize("snr_db", SNR_LEVELS)
def test_measured_snr_matches_request(rows, signals, snr_db):

    for row, clean in zip(rows, signals):
        noisy = AWGN(snr_db)(clean, row)
        assert abs(measured_snr_db(clean, noisy) - snr_db) < 0.15


def test_original_is_not_modified(rows, signals):

    clean = signals[0]
    before = clean.copy()

    AWGN(5)(clean, rows[0])

    assert np.array_equal(clean, before)

    # And the file on disk reads back the same
    assert np.array_equal(RadarDataLoader().load_mat_file(rows[0]["abs_path"]), before)


def test_same_file_same_snr_same_noise(rows, signals):

    a = AWGN(10)(signals[0], rows[0])
    b = AWGN(10)(signals[0], rows[0])

    assert np.array_equal(a, b)


def test_noise_differs_between_files_and_snrs(rows, signals):

    n0 = AWGN(10)(signals[0], rows[0]) - signals[0]
    n1 = AWGN(10)(signals[1], rows[1]) - signals[1]
    n0_20 = AWGN(20)(signals[0], rows[0]) - signals[0]

    assert not np.allclose(n0, n1)
    assert not np.allclose(n0 / np.std(n0), n0_20 / np.std(n0_20))


def test_lower_snr_means_more_noise(rows, signals):

    powers = [np.mean(np.abs(AWGN(s)(signals[0], rows[0]) - signals[0]) ** 2)
              for s in sorted(SNR_LEVELS, reverse=True)]

    assert powers == sorted(powers)
