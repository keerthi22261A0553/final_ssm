import argparse
import hashlib
import json
import os
import sys
import time

import torch
from torch.utils.data import ConcatDataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from block_split import SPLIT_NAMES, load_splits, split_fingerprint
from checkpoints import condition_dir, condition_name
from config import (
    CHECKPOINT_DIR,
    CLASSES,
    CROSS_DISTANCE,
    DENOISER,
    MODEL,
    RANDOM_SEED,
    RESULTS_DIR,
    SNR_LEVELS,
    TRAINING,
)
from data_loader import RadarDataLoader, distance_metres
from dataset import RadarSegmentDataset
from denoising import NoisyThenDenoised, build_denoiser
from evaluate_noise import SNR_DEFINITION, RecordedAWGN
from metrics import evaluate_model, save_results, summary_line
from model import UIBSSMClassifier
from train import open_log, fit, signal_config
from utils import relative, set_seed

# Separate experiment: results never share a folder with the per-distance runs


def experiment_name(train_distances, test_distance):

    return f"train_{'-'.join(train_distances)}__test_{test_distance}"


def experiment_paths(name):

    checkpoint_dir = os.path.join(CHECKPOINT_DIR, "cross_distance", name)
    results_dir = os.path.join(RESULTS_DIR, "cross_distance", name)

    return {
        "checkpoint_dir": checkpoint_dir,
        "results_dir": results_dir,
        "best": os.path.join(checkpoint_dir, "best.pt"),
        "last": os.path.join(checkpoint_dir, "last.pt"),
        "done": os.path.join(checkpoint_dir, "done.json"),
        "history": os.path.join(results_dir, "training_history.json"),
        "log": os.path.join(results_dir, "train_log.txt"),
    }


def combined_fingerprint(distances):

    # Split files of every distance involved (training and test)
    digest = hashlib.sha256()

    for d in sorted(distances, key=distance_metres):
        digest.update(f"{d}:{split_fingerprint(d)};".encode())

    return digest.hexdigest()


def check_separation(train_distances, test_distance):

    # The test distance must be unseen, and no test file may be a
    # training or validation file of any training distance
    if test_distance in train_distances:
        raise ValueError(f"Test distance {test_distance} is also a training distance.")

    available = RadarDataLoader().selected_distances()

    missing = [d for d in train_distances + [test_distance] if d not in available]

    if missing:
        raise ValueError(f"Distances not in the dataset: {missing}")

    used = {
        r["file_path"]
        for d in train_distances
        for name in SPLIT_NAMES if name != "test"
        for r in load_splits(d)[name]
    }

    test_files = {r["file_path"] for r in load_splits(test_distance)["test"]}

    if used & test_files:
        raise RuntimeError(f"{len(used & test_files)} test files also used for training.")


def train(train_distances, test_distance, paths, config, force):

    if os.path.exists(paths["done"]) and not force:
        print(f"Already trained ({relative(paths['done'])}), skipped.")
        return

    log, log_file = open_log(paths["log"])

    set_seed(RANDOM_SEED)

    log(f"\n=== Cross-distance: train {', '.join(train_distances)} -> test {test_distance} ===")
    log("Loading train and validation files of the training distances only...")

    start = time.time()

    train_sets = [RadarSegmentDataset(d, "train") for d in train_distances]
    val_sets = [RadarSegmentDataset(d, "val") for d in train_distances]

    for d, t, v in zip(train_distances, train_sets, val_sets):
        log(f"  {d:>4s}: train {len(t.records)} files / {len(t)} segments, "
            f"val {len(v.records)} files / {len(v)} segments")

    log(f"  built in {time.time() - start:.0f} s")

    fit(ConcatDataset(train_sets), ConcatDataset(val_sets), paths, config, log, force)

    log_file.close()


def evaluate(train_distances, test_distance, paths, config):

    checkpoint = torch.load(paths["best"], map_location="cpu")

    if checkpoint["config"]["split_fingerprint"] != config["split_fingerprint"]:
        raise RuntimeError("Split files changed after training; retrain first.")

    model = UIBSSMClassifier(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    denoiser = build_denoiser()

    def run(snr_db, denoised, transform):

        test_set = RadarSegmentDataset(test_distance, "test", signal_transform=transform)

        results = evaluate_model(model, test_set)

        results.update({
            "experiment": "cross_distance",
            "train_distances": train_distances,
            "test_distance": test_distance,
            "condition": "clean" if snr_db is None else ("noisy+denoised" if denoised else "noisy"),
            "snr_db": snr_db,
            "snr_definition": None if snr_db is None else SNR_DEFINITION,
            "denoiser": DENOISER if denoised else None,
            "checkpoint_epoch": checkpoint["epoch"],
            "test_files": len(test_set.records),
            "test_segments": len(test_set),
            "class_names": CLASSES,
        })

        folder = os.path.join(paths["results_dir"], condition_name(snr_db, denoised))

        save_results(results, folder, f"cross-distance -> {test_distance} {condition_name(snr_db, denoised)}")

        return results

    table = {"clean": run(None, False, None)}

    for snr_db in SNR_LEVELS:

        a, b = RecordedAWGN(snr_db), RecordedAWGN(snr_db)

        table[f"{snr_db:g}dB"] = run(snr_db, False, a)
        table[f"{snr_db:g}dB+den"] = run(snr_db, True, NoisyThenDenoised(b, denoiser))

        if a.checksums != b.checksums:
            raise RuntimeError(f"{snr_db} dB: noisy inputs differ between conditions.")

    return table


def write_summary(train_distances, test_distance, paths, table):

    # Cross-distance model next to the per-distance model of the test
    # distance (same 300 test files), when that model's results exist
    # Same column order as the main table
    keys = (["clean"] + [f"{s:g}dB" for s in SNR_LEVELS]
            + [f"{s:g}dB+den" for s in SNR_LEVELS])

    def own(key):

        snr = None if key == "clean" else float(key.split("dB")[0])
        path = os.path.join(condition_dir(test_distance, snr, key.endswith("+den")), "metrics.json")

        if not os.path.exists(path):
            return None

        with open(path) as f:
            return json.load(f)

    lines = [
        f"# Cross-distance: train {', '.join(train_distances)} -> test {test_distance}",
        "",
        f"Test files: the {table['clean']['test_files']} test files of {test_distance} "
        f"(never seen in training). Accuracy in percent; n/a = not run.",
        "",
    ]

    for level in ("segment_level", "file_level"):

        lines += [
            f"## {level.replace('_', '-')}",
            "",
            "| Model | " + " | ".join(keys) + " |",
            "|---|" + "---|" * len(keys),
            f"| Cross-distance (unseen {test_distance}) | "
            + " | ".join(f"{100 * table[k][level]['accuracy']:.2f}" for k in keys) + " |",
            f"| Per-distance model trained at {test_distance} | "
            + " | ".join("n/a" if own(k) is None else f"{100 * own(k)[level]['accuracy']:.2f}"
                         for k in keys) + " |",
            "",
        ]

    path = os.path.join(paths["results_dir"], "summary.md")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path


def main():

    parser = argparse.ArgumentParser(
        description="Optional: train on some distances, test on an unseen distance."
    )
    parser.add_argument("--train", nargs="+", default=CROSS_DISTANCE["train"])
    parser.add_argument("--test", default=CROSS_DISTANCE["test"])
    parser.add_argument("--force", action="store_true", help="Retrain from scratch")

    args = parser.parse_args()

    train_distances = sorted(args.train, key=distance_metres)

    check_separation(train_distances, args.test)

    paths = experiment_paths(experiment_name(train_distances, args.test))

    config = {
        "experiment": "cross_distance",
        "run_name": "cross_distance",
        "train_distances": train_distances,
        "test_distance": args.test,
        "seed": RANDOM_SEED,
        "split_fingerprint": combined_fingerprint(train_distances + [args.test]),
        "model": MODEL,
        "training": TRAINING,
        "signal": signal_config(),
    }

    train(train_distances, args.test, paths, config, args.force)

    table = evaluate(train_distances, args.test, paths, config)

    for key, results in table.items():
        print(f"  {key:>9s}: {summary_line(results)}")

    print("\nSummary:", relative(write_summary(train_distances, args.test, paths, table)))


if __name__ == "__main__":
    main()
