import os

import torch

from block_split import split_fingerprint
from config import CHECKPOINT_DIR, RESULTS_DIR
from model import UIBSSMClassifier


def run_paths(distance, run_name="clean"):

    # One folder per distance and model; results of different distances
    # and different models never share a folder
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, distance, run_name)
    results_dir = os.path.join(RESULTS_DIR, distance, run_name)

    return {
        "checkpoint_dir": checkpoint_dir,
        "results_dir": results_dir,
        "best": os.path.join(checkpoint_dir, "best.pt"),
        "last": os.path.join(checkpoint_dir, "last.pt"),
        "done": os.path.join(checkpoint_dir, "done.json"),
        "history": os.path.join(results_dir, "training_history.json"),
        "log": os.path.join(results_dir, "train_log.txt"),
    }


def condition_name(snr_db=None, denoised=False):

    # Test condition folder name: test_clean, test_snr10_noisy, ...
    if snr_db is None:
        return "test_clean_denoised" if denoised else "test_clean"

    return f"test_snr{snr_db:g}_{'denoised' if denoised else 'noisy'}"


def condition_dir(distance, snr_db=None, denoised=False, run_name="clean"):

    return os.path.join(run_paths(distance, run_name)["results_dir"],
                        condition_name(snr_db, denoised))


def is_trained(distance, run_name="clean"):

    return os.path.exists(run_paths(distance, run_name)["done"])


def load_trained_model(distance, run_name="clean"):

    # Best checkpoint of a finished run, after checking that it was
    # trained on this distance with the split files that exist now
    paths = run_paths(distance, run_name)

    if not is_trained(distance, run_name):
        raise FileNotFoundError(f"{distance}: no finished '{run_name}' model.")

    checkpoint = torch.load(paths["best"], map_location="cpu")

    config = checkpoint["config"]

    if config["distance"] != distance:
        raise RuntimeError(f"Checkpoint is for {config['distance']}, not {distance}.")

    if config["split_fingerprint"] != split_fingerprint(distance):
        raise RuntimeError(
            f"{distance}: split files changed after training; evaluating would "
            "risk testing on training files. Retrain this distance."
        )

    model = UIBSSMClassifier(**config["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, checkpoint
