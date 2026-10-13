import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SNR_LEVELS
from data_loader import RadarDataLoader
from run_all import has_metrics, run


def main_experiment_complete(distances):

    return all(
        has_metrics(d) and all(has_metrics(d, s, den) for s in SNR_LEVELS for den in (False, True))
        for d in distances
    )


def model_complete(distance, run_name):

    from checkpoints import condition_dir

    def exists(snr=None, den=False):
        return os.path.exists(os.path.join(condition_dir(distance, snr, den, run_name), "metrics.json"))

    return exists() and all(exists(s, den) for s in SNR_LEVELS for den in (False, True))


def main():

    parser = argparse.ArgumentParser(
        description="Optional experiments, only after the per-distance clean-model "
                    "experiments are complete: noise-augmented training per distance, then "
                    "the all-distance model, then cross-distance. Finished steps are skipped."
    )
    parser.add_argument("--skip-all-distances", action="store_true")
    parser.add_argument("--skip-cross-distance", action="store_true")
    parser.add_argument("--skip-noise-aug", action="store_true")

    args = parser.parse_args()

    distances = RadarDataLoader().selected_distances()

    if not main_experiment_complete(distances):
        sys.exit("The per-distance clean / noisy / denoised experiments are not complete; "
                 "run scripts/run_all.py first.")

    start = time.time()

    # Order chosen by the user: noise-augmented first (per distance, results
    # every hour), then the all-distance model, then cross-distance
    if not args.skip_noise_aug:

        for distance in distances:

            print(f"\n{'=' * 60}\n{distance} noise-augmented  "
                  f"({(time.time() - start) / 3600:.1f} h elapsed)\n{'=' * 60}")

            run("train.py", "--distances", distance, "--model", "noise_aug")

            if not model_complete(distance, "noise_aug"):
                run("evaluate_clean.py", "--distances", distance, "--model", "noise_aug")
                run("evaluate_noise.py", "--distances", distance, "--model", "noise_aug")

            run("make_report.py")

    if not args.skip_all_distances:

        # One model for every distance (trains once, then tests each distance)
        if not all(model_complete(d, "all_distances") for d in distances):
            run("all_distances.py")

        run("make_report.py")

    if not args.skip_cross_distance:
        run("cross_distance.py")

    print(f"\nOptional experiments finished in {(time.time() - start) / 3600:.1f} h.")


if __name__ == "__main__":
    main()
