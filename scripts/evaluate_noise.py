import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from checkpoints import condition_dir, is_trained, load_trained_model
from config import CLASSES, DENOISER, SNR_LEVELS
from data_loader import RadarDataLoader
from dataset import RadarSegmentDataset
from denoising import NoisyThenDenoised, build_denoiser
from metrics import evaluate_model, save_results, summary_line
from noise import AWGN
from utils import relative

SNR_DEFINITION = (
    "complex AWGN added to a copy of each raw test signal; SNR relative to the "
    "target (micro-Doppler) power after zero-Doppler shift and DC removal; "
    "seed per (file, SNR) from RANDOM_SEED"
)


class RecordedAWGN(AWGN):

    # AWGN that also keeps a checksum of every noisy signal it produces,
    # to prove the noisy and denoised conditions got identical input

    def __init__(self, snr_db):

        super().__init__(snr_db)
        self.checksums = {}

    def __call__(self, signal, row):

        noisy = super().__call__(signal, row)

        self.checksums[row["file_path"]] = hashlib.sha256(noisy.tobytes()).hexdigest()

        return noisy


def evaluate_condition(model, checkpoint, distance, snr_db, denoised, transform, run_name="clean"):

    # Test files of this distance; the transform only changes in-memory copies
    test_set = RadarSegmentDataset(distance, "test", signal_transform=transform)

    results = evaluate_model(model, test_set)

    results.update({
        "distance": distance,
        "condition": "noisy+denoised" if denoised else "noisy",
        "snr_db": snr_db,
        "snr_definition": SNR_DEFINITION,
        "denoiser": DENOISER if denoised else None,
        "model": run_name,
        "checkpoint_epoch": checkpoint["epoch"],
        "split_fingerprint": checkpoint["config"]["split_fingerprint"],
        "test_files": len(test_set.records),
        "test_segments": len(test_set),
        "class_names": CLASSES,
    })

    folder = condition_dir(distance, snr_db, denoised, run_name)

    label = f"{distance} {snr_db:g} dB {'noisy+denoised' if denoised else 'noisy'}"

    save_results(results, folder, label)

    return results, folder


def evaluate_distance(distance, denoiser, run_name="clean"):

    # Same trained model for every SNR and both conditions
    model, checkpoint = load_trained_model(distance, run_name)

    rows = []

    for snr_db in SNR_LEVELS:

        noisy_noise = RecordedAWGN(snr_db)
        denoised_noise = RecordedAWGN(snr_db)

        noisy, noisy_dir = evaluate_condition(
            model, checkpoint, distance, snr_db, False, noisy_noise, run_name
        )

        denoised, denoised_dir = evaluate_condition(
            model, checkpoint, distance, snr_db, True,
            NoisyThenDenoised(denoised_noise, denoiser), run_name
        )

        # Both conditions must have seen exactly the same noisy signals
        if noisy_noise.checksums != denoised_noise.checksums:
            raise RuntimeError(f"{distance} {snr_db} dB: noisy inputs differ between conditions.")

        print(f"  {snr_db:>3g} dB noisy   : {summary_line(noisy)}")
        print(f"  {snr_db:>3g} dB denoised: {summary_line(denoised)}")
        print(f"         same noisy input verified for {len(noisy_noise.checksums)} files"
              f"  -> {relative(os.path.dirname(noisy_dir))}")

        rows.append((snr_db, noisy, denoised))

    return rows


def main():

    parser = argparse.ArgumentParser(
        description="A trained model (clean by default) on noisy and noisy+denoised test files."
    )
    parser.add_argument("--distances", nargs="+")
    parser.add_argument("--model", choices=("clean", "noise_aug"), default="clean")

    args = parser.parse_args()

    distances = args.distances or RadarDataLoader().selected_distances()

    denoiser = build_denoiser()

    print(f"SNR levels: {SNR_LEVELS} dB   Denoiser: {denoiser!r}\n")

    for distance in distances:

        if not is_trained(distance, args.model):
            print(f"{distance:>4s}: not trained yet, skipped.")
            continue

        print(f"{distance}:")

        evaluate_distance(distance, denoiser, args.model)


if __name__ == "__main__":
    main()
