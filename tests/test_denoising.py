# Tests use real TSMS-Drone validation files only (never test files).

import numpy as np
import pytest

from block_split import load_split
from data_loader import RadarDataLoader
from denoising import NoisyThenDenoised, STFTWienerDenoiser, build_denoiser
from noise import AWGN


@pytest.fixture(scope="module")
def row():
    return load_split(RadarDataLoader().selected_distances()[0], "val")[0]


@pytest.fixture(scope="module")
def clean(row):
    return RadarDataLoader().load_mat_file(row["abs_path"])


def test_output_shape_and_type(clean, row):

    out = build_denoiser()(AWGN(10)(clean, row), row)

    assert out.shape == clean.shape
    assert np.iscomplexobj(out)


def test_input_not_modified(clean, row):

    noisy = AWGN(10)(clean, row)
    before = noisy.copy()

    build_denoiser()(noisy, row)

    assert np.array_equal(noisy, before)


def test_unit_gain_is_exact_identity(clean):

    assert np.allclose(STFTWienerDenoiser(gain_floor=1.0)(clean), clean, atol=1e-12)


def test_denoising_uses_the_same_noisy_signal(clean, row):

    # Noisy condition and noisy+denoised condition share the noisy input
    noise = AWGN(5)
    denoiser = build_denoiser()

    assert np.array_equal(NoisyThenDenoised(noise, denoiser)(clean, row),
                          denoiser(noise(clean, row), row))


def test_deterministic(clean, row):

    noisy = AWGN(5)(clean, row)

    assert np.array_equal(build_denoiser()(noisy), build_denoiser()(noisy))
