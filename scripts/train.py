import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from block_split import split_fingerprint
from checkpoints import run_paths
from config import (
    MODEL,
    N_FFT,
    NOISE_AUG,
    RANDOM_SEED,
    SAMPLING_RATE,
    SEGMENT_HOP,
    SEGMENT_LENGTH,
    STFT_HOP,
    TONE_FREQUENCY,
    TRAINING,
)
from data_loader import RadarDataLoader
from dataset import NoiseAugmentedDataset, RadarSegmentDataset
from model import UIBSSMClassifier, count_parameters
from utils import relative, set_seed

MODELS = ("clean", "noise_aug")


def signal_config():

    return {
        "sampling_rate": SAMPLING_RATE,
        "tone_frequency": TONE_FREQUENCY,
        "segment_length": SEGMENT_LENGTH,
        "segment_hop": SEGMENT_HOP,
        "n_fft": N_FFT,
        "stft_hop": STFT_HOP,
    }


def run_config(distance, run_name="clean"):

    # Everything needed to rebuild and check the model later
    return {
        "distance": distance,
        "run_name": run_name,
        "seed": RANDOM_SEED,
        "split_fingerprint": split_fingerprint(distance),
        "model": MODEL,
        "training": TRAINING,
        "noise_augmentation": NOISE_AUG if run_name == "noise_aug" else None,
        "signal": signal_config(),
    }


def run_epoch(model, loader, criterion, optimizer=None):

    # Training when an optimizer is given, validation otherwise
    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    correct = 0
    count = 0

    with torch.set_grad_enabled(training):

        for inputs, labels in loader:

            logits = model(inputs)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += len(labels)

    return total_loss / count, correct / count


def open_log(path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    log_file = open(path, "a")

    def log(text=""):
        print(text, flush=True)
        log_file.write(text + "\n")
        log_file.flush()

    return log, log_file


def fit(train_set, val_set, paths, config, log, force=False):

    # Shared training loop (per-distance and cross-distance models):
    # AdamW, LR halved on plateau, early stopping on VALIDATION loss,
    # exact resume from last.pt. Test files are never touched here.
    os.makedirs(paths["checkpoint_dir"], exist_ok=True)

    train_loader = DataLoader(train_set, batch_size=TRAINING["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=TRAINING["batch_size"])

    model = UIBSSMClassifier(**MODEL)

    log(f"Model parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=TRAINING["lr_factor"],
        patience=TRAINING["lr_patience"],
        min_lr=TRAINING["min_lr"],
    )

    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_epoch = 1

    if os.path.exists(paths["last"]) and not force:

        # Exact resume: model, optimizer, scheduler, counters, shuffle state
        state = torch.load(paths["last"], map_location="cpu")

        if state["config"]["split_fingerprint"] != config["split_fingerprint"]:
            raise RuntimeError(
                "Split files changed since last.pt was saved. "
                "Delete the checkpoint folder to retrain."
            )

        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        torch.set_rng_state(state["rng_state"])

        history = state["history"]
        best_val_loss = state["best_val_loss"]
        epochs_without_improvement = state["epochs_without_improvement"]
        start_epoch = state["epoch"] + 1

        log(f"Resuming after epoch {state['epoch']}.")

    log("\nEpoch | Train loss  acc  | Val loss  acc   | LR      | Time  | Note")

    stopped_early = False

    for epoch in range(start_epoch, TRAINING["max_epochs"] + 1):

        if epochs_without_improvement >= TRAINING["patience"]:
            stopped_early = True
            break

        epoch_start = time.time()

        # Noise-augmented training: fresh noise on the training files
        if hasattr(train_set, "resample"):
            train_set.resample(epoch)

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        elapsed = time.time() - epoch_start

        # Learning rate used for this epoch
        lr = optimizer.param_groups[0]["lr"]

        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr,
            "time_s": elapsed,
        })

        # Improvement only if lower by at least min_delta
        if val_loss < best_val_loss - TRAINING["min_delta"]:

            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": config,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, paths["best"])

            note = "saved best"

        else:

            epochs_without_improvement += 1
            note = f"no improvement {epochs_without_improvement}/{TRAINING['patience']}"

        log(
            f"{epoch:5d} | {train_loss:.4f}  {train_acc:.4f} | "
            f"{val_loss:.4f}  {val_acc:.4f} | {lr:.1e} | {elapsed:4.0f}s | {note}"
        )

        with open(paths["history"], "w") as f:
            json.dump({"config": config, "history": history}, f, indent=2)

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": config,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "rng_state": torch.get_rng_state(),
        }, paths["last"])

    else:
        stopped_early = epochs_without_improvement >= TRAINING["patience"]

    # Report the epoch actually saved in best.pt (with min_delta this can
    # be earlier than the epoch with the lowest validation loss)
    saved_epoch = torch.load(paths["best"], map_location="cpu")["epoch"]

    best = next(h for h in history if h["epoch"] == saved_epoch)

    if stopped_early:
        log(f"\nEarly stopping after epoch {history[-1]['epoch']}.")

    log(
        f"Best epoch {best['epoch']}: val loss {best['val_loss']:.4f}, "
        f"val acc {best['val_acc']:.4f}"
    )
    log(f"Checkpoint: {relative(paths['best'])}")

    with open(paths["done"], "w") as f:
        json.dump({
            "distance": config.get("distance", config.get("train_distances")),
            "run_name": config["run_name"],
            "best_epoch": best["epoch"],
            "best_val_loss": best["val_loss"],
            "best_val_acc": best["val_acc"],
            "epochs_run": history[-1]["epoch"],
            "split_fingerprint": config["split_fingerprint"],
        }, f, indent=2)


def train_distance(distance, run_name="clean", force=False):

    paths = run_paths(distance, run_name)

    if os.path.exists(paths["done"]) and not force:
        print(f"{distance}: already trained ({relative(paths['done'])}), skipped.")
        return

    config = run_config(distance, run_name)

    log, log_file = open_log(paths["log"])

    # Same seed for every distance: each run is reproducible on its own
    set_seed(RANDOM_SEED)

    log(f"\n=== Distance {distance} ({run_name}) ===")
    log("Loading data (train and validation files only)...")

    start = time.time()

    if run_name == "noise_aug":
        train_set = NoiseAugmentedDataset(
            distance, NOISE_AUG["snr_range_db"], NOISE_AUG["clean_fraction"], RANDOM_SEED
        )
        log(f"  noise augmentation on training files only: {NOISE_AUG}")
    else:
        train_set = RadarSegmentDataset(distance, "train")

    # Validation stays clean in both models
    val_set = RadarSegmentDataset(distance, "val")

    log(f"  train: {len(train_set.records)} files, {len(train_set)} segments")
    log(f"  val  : {len(val_set.records)} files, {len(val_set)} segments")
    log(f"  built in {time.time() - start:.0f} s")

    fit(train_set, val_set, paths, config, log, force)

    log_file.close()


def main():

    parser = argparse.ArgumentParser(
        description="Train one UIB-SSM model per distance (train files only)."
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        help="Distances to train, e.g. 2m 14m (default: all selected distances)",
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        default="clean",
        help="clean (baseline) or noise_aug (optional, kept separate)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain from scratch even if a finished or partial run exists",
    )

    args = parser.parse_args()

    available = RadarDataLoader().selected_distances()

    distances = args.distances or available

    unknown = [d for d in distances if d not in available]

    if unknown:
        parser.error(f"Unknown distances: {unknown}. Available: {available}")

    for distance in distances:
        train_distance(distance, args.model, force=args.force)


if __name__ == "__main__":
    main()
