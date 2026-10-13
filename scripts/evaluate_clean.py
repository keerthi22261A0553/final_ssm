import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from checkpoints import condition_dir, is_trained, load_trained_model
from config import CLASSES
from data_loader import RadarDataLoader
from dataset import RadarSegmentDataset
from metrics import evaluate_model, save_results, summary_line
from utils import relative


def evaluate_distance(distance, run_name="clean"):

    model, checkpoint = load_trained_model(distance, run_name)

    # Test files of this distance only, unmodified
    test_set = RadarSegmentDataset(distance, "test")

    results = evaluate_model(model, test_set)

    results.update({
        "distance": distance,
        "condition": "clean",
        "model": run_name,
        "checkpoint_epoch": checkpoint["epoch"],
        "split_fingerprint": checkpoint["config"]["split_fingerprint"],
        "test_files": len(test_set.records),
        "test_segments": len(test_set),
        "class_names": CLASSES,
    })

    folder = condition_dir(distance, run_name=run_name)

    save_results(results, folder, f"{distance} clean test ({run_name} model)")

    return results, folder


def main():

    parser = argparse.ArgumentParser(
        description="Clean baseline: each distance's model on its own test files."
    )
    parser.add_argument("--distances", nargs="+")
    parser.add_argument("--model", choices=("clean", "noise_aug"), default="clean")

    args = parser.parse_args()

    distances = args.distances or RadarDataLoader().selected_distances()

    for distance in distances:

        if not is_trained(distance, args.model):
            print(f"{distance:>4s}: not trained yet, skipped.")
            continue

        results, folder = evaluate_distance(distance, args.model)

        print(f"{distance:>4s}: {summary_line(results)}  -> {relative(folder)}")

        print("      per-class file recall: " + ", ".join(
            f"{name} {results['file_level']['per_class'][name]['recall']:.3f}"
            for name in CLASSES
        ))


if __name__ == "__main__":
    main()
