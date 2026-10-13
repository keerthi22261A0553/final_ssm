import argparse
import os
import sys
import time

import torch
from torch.utils.data import ConcatDataset, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkpoints import condition_dir
from config import CHECKPOINT_DIR, CLASSES, DENOISER, MODEL, RANDOM_SEED, SNR_LEVELS, TRAINING
from cross_distance import combined_fingerprint
from data_loader import RadarDataLoader
from dataset import RadarSegmentDataset
from denoising import NoisyThenDenoised, build_denoiser
from evaluate_noise import SNR_DEFINITION, RecordedAWGN
from metrics import evaluate_model, save_results, summary_line
from model import UIBSSMClassifier
from train import fit, open_log, signal_config
from utils import relative, set_seed

# Optional experiment: ONE model trained on the train files of every
# distance, then tested on each distance's own test files. Results go to
# results/<distance>/all_distances/<condition>/, next to (never mixed
# with) the per-distance model in results/<distance>/clean/.

RUN_NAME = "all_distances"


def paths():

    checkpoint_dir = os.path.join(CHECKPOINT_DIR, RUN_NAME)

    return {
        "checkpoint_dir": checkpoint_dir,
        "results_dir": checkpoint_dir,
        "best": os.path.join(checkpoint_dir, "best.pt"),
        "last": os.path.join(checkpoint_dir, "last.pt"),
        "done": os.path.join(checkpoint_dir, "done.json"),
        "history": os.path.join(checkpoint_dir, "training_history.json"),
        "log": os.path.join(checkpoint_dir, "train_log.txt"),
    }


class HalfStored(Dataset):

    # All 15 distances do not fit in 7.7 GB RAM as float32 (5.7 GB), so
    # spectrograms are kept as float16 (z-scored values, ~3 significant
    # digits) and converted back to float32 per item for the model

    def __init__(self, dataset):

        self.inputs = dataset.inputs.half()
        self.labels = dataset.labels

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        return self.inputs[index].float(), self.labels[index]


def build(distances, split, log):

    parts = []

    for d in distances:

        # Built one distance at a time and reduced at once, to keep peak memory low
        dataset = RadarSegmentDataset(d, split)
        parts.append(HalfStored(dataset))

        log(f"  {d:>4s} {split}: {len(dataset.records)} files, {len(dataset)} segments")

        del dataset

    return ConcatDataset(parts)


def train(distances, config, run_paths, force):

    if os.path.exists(run_paths["done"]) and not force:
        print(f"Already trained ({relative(run_paths['done'])}), skipped.")
        return

    log, log_file = open_log(run_paths["log"])

    set_seed(RANDOM_SEED)

    log(f"\n=== All-distance model: {', '.join(distances)} ===")
    log("Loading train and validation files of every distance (no test files)...")

    start = time.time()

    train_set = build(distances, "train", log)
    val_set = build(distances, "val", log)

    log(f"  total: train {len(train_set)} segments, val {len(val_set)} segments, "
        f"built in {time.time() - start:.0f} s")

    fit(train_set, val_set, run_paths, config, log, force)

    log_file.close()


def evaluate(distances, config, run_paths):

    checkpoint = torch.load(run_paths["best"], map_location="cpu")

    if checkpoint["config"]["split_fingerprint"] != config["split_fingerprint"]:
        raise RuntimeError("Split files changed after training; retrain first.")

    model = UIBSSMClassifier(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    denoiser = build_denoiser()

    for distance in distances:

        def run(snr_db, denoised, transform):

            test_set = RadarSegmentDataset(distance, "test", signal_transform=transform)

            results = evaluate_model(model, test_set)

            results.update({
                "distance": distance,
                "model": RUN_NAME,
                "train_distances": config["train_distances"],
                "condition": "clean" if snr_db is None else ("noisy+denoised" if denoised else "noisy"),
                "snr_db": snr_db,
                "snr_definition": None if snr_db is None else SNR_DEFINITION,
                "denoiser": DENOISER if denoised else None,
                "checkpoint_epoch": checkpoint["epoch"],
                "test_files": len(test_set.records),
                "test_segments": len(test_set),
                "class_names": CLASSES,
            })

            save_results(results, condition_dir(distance, snr_db, denoised, RUN_NAME),
                         f"{distance} all-distance model, "
                         + ("clean" if snr_db is None else f"{snr_db:g} dB "
                            + ("noisy+denoised" if denoised else "noisy")))

            return results

        clean = run(None, False, None)

        print(f"{distance}:\n  clean      : {summary_line(clean)}")

        for snr_db in SNR_LEVELS:

            a, b = RecordedAWGN(snr_db), RecordedAWGN(snr_db)

            noisy = run(snr_db, False, a)
            den = run(snr_db, True, NoisyThenDenoised(b, denoiser))

            if a.checksums != b.checksums:
                raise RuntimeError(f"{distance} {snr_db} dB: noisy inputs differ between conditions.")

            print(f"  {snr_db:>3g} dB noisy : {summary_line(noisy)}")
            print(f"  {snr_db:>3g} dB denois: {summary_line(den)}")


def main():

    parser = argparse.ArgumentParser(
        description="Optional: one UIB-SSM model for all distances, tested per distance."
    )
    parser.add_argument("--force", action="store_true", help="Retrain from scratch")

    args = parser.parse_args()

    distances = RadarDataLoader().selected_distances()

    run_paths = paths()

    config = {
        "experiment": RUN_NAME,
        "run_name": RUN_NAME,
        "train_distances": distances,
        "seed": RANDOM_SEED,
        "split_fingerprint": combined_fingerprint(distances),
        "input_storage": "float16",
        "model": MODEL,
        "training": TRAINING,
        "signal": signal_config(),
    }

    train(distances, config, run_paths, args.force)

    evaluate(distances, config, run_paths)


if __name__ == "__main__":
    main()
