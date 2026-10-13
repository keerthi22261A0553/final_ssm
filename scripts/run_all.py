import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from block_split import SPLIT_NAMES, split_dir
from checkpoints import condition_dir
from config import PROJECT_ROOT, SNR_LEVELS
from data_loader import RadarDataLoader


def run(script, *args):

    # Each step in its own process, same Python; stops on the first error
    command = [sys.executable, "-u", os.path.join("scripts", script), *args]

    print(f"\n>>> {' '.join(command[2:])}", flush=True)

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def has_metrics(distance, snr_db=None, denoised=False):

    return os.path.exists(os.path.join(condition_dir(distance, snr_db, denoised), "metrics.json"))


def main():

    parser = argparse.ArgumentParser(
        description="Whole experiment: splits -> per distance (train, clean test, "
                    "noisy / denoised test) -> report. Finished steps are skipped."
    )
    parser.add_argument("--distances", nargs="+", help="Default: all selected distances")
    parser.add_argument("--re-evaluate", action="store_true",
                        help="Evaluate again even if metrics already exist")

    args = parser.parse_args()

    distances = args.distances or RadarDataLoader().selected_distances()

    start = time.time()

    run("inspect_dataset.py")

    if not all(os.path.exists(os.path.join(split_dir(d), f"{n}.csv"))
               for d in distances for n in SPLIT_NAMES):
        run("make_splits.py")

    for distance in distances:

        print(f"\n{'=' * 60}\n{distance}  ({(time.time() - start) / 3600:.1f} h elapsed)\n{'=' * 60}")

        # Skips by itself when the distance is already trained
        run("train.py", "--distances", distance)

        if args.re_evaluate or not has_metrics(distance):
            run("evaluate_clean.py", "--distances", distance)

        noise_done = all(has_metrics(distance, s, den) for s in SNR_LEVELS for den in (False, True))

        if args.re_evaluate or not noise_done:
            run("evaluate_noise.py", "--distances", distance)

        # Refresh tables and graphs after every distance
        run("make_report.py")

    print(f"\nFinished in {(time.time() - start) / 3600:.1f} h.")


if __name__ == "__main__":
    main()
