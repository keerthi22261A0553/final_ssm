import argparse
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from block_split import chunk_sizes
from config import CLASSES, EXCLUDED_CLASSES, SUMMARY_DIR
from data_loader import FILE_NUMBER_PATTERN, RadarDataLoader
from utils import relative


def file_numbers(paths):

    numbers = []

    for path in paths:
        match = FILE_NUMBER_PATTERN.search(os.path.basename(path))
        if match:
            numbers.append(int(match.group(1)))

    return numbers


def numbering_gaps(numbers):

    # Recording numbers missing between the first and the last file
    if not numbers:
        return []

    return sorted(set(range(min(numbers), max(numbers) + 1)) - set(numbers))


def main():

    parser = argparse.ArgumentParser(
        description="Inventory of the distances, classes and .mat files present."
    )
    parser.add_argument(
        "--check-signals",
        action="store_true",
        help="Also load every usable file and report signal lengths (slower)",
    )

    args = parser.parse_args()

    loader = RadarDataLoader()

    distances = loader.selected_distances()

    print("Dataset folder:", relative(loader.data_dir))
    print(f"Distances found ({len(distances)}):", ", ".join(distances))
    print("Classes used   :", ", ".join(CLASSES))
    print("Excluded       :", ", ".join(EXCLUDED_CLASSES))

    rows = []
    lengths = collections.Counter()
    problems = []

    print(f"\n{'Distance':>8s} | " + " | ".join(f"{c[:13]:>13s}" for c in CLASSES)
          + " | Excluded (not used)")

    for distance in distances:

        cells = []

        for class_name in CLASSES:

            paths = loader.list_class_files(distance, class_name)
            gaps = numbering_gaps(file_numbers(paths))

            rows.append({
                "distance": distance,
                "class_name": class_name,
                "used": "yes",
                "total_files": len(paths),
                **{f"planned_{k}": v for k, v in chunk_sizes(len(paths)).items()},
                "numbering_gaps": len(gaps),
            })

            cells.append(f"{len(paths):13d}")

            if gaps:
                problems.append(f"{distance} {class_name}: missing numbers {gaps[:10]}")

            if len(paths) != len(loader.list_class_files(distances[0], CLASSES[0])):
                problems.append(
                    f"{distance} {class_name}: {len(paths)} files (differs from "
                    f"{distances[0]} {CLASSES[0]})"
                )

            if args.check_signals:
                for path in paths:
                    try:
                        signal = loader.load_mat_file(path)
                        lengths[len(signal)] += 1
                    except (OSError, KeyError, ValueError) as error:
                        problems.append(f"{relative(path)}: {error}")

        excluded = []

        for class_name in EXCLUDED_CLASSES:

            count = len(loader.list_class_files(distance, class_name))
            excluded.append(f"{class_name} {count}")

            rows.append({
                "distance": distance,
                "class_name": class_name,
                "used": "no (excluded)",
                "total_files": count,
                **{f"planned_{k}": "" for k in chunk_sizes(0)},
                "numbering_gaps": "",
            })

        print(f"{distance:>8s} | " + " | ".join(cells) + " | " + ", ".join(excluded))

        # Folders present on disk but neither used nor excluded
        known = set(CLASSES) | set(EXCLUDED_CLASSES)
        other = [
            name for name in os.listdir(os.path.join(loader.data_dir, distance))
            if name not in known
        ]

        if other:
            problems.append(f"{distance}: unrecognised folders ignored: {other}")

    used = [r for r in rows if r["used"] == "yes"]

    print(f"\nUsable files (4 classes, all distances): {sum(r['total_files'] for r in used)}")

    if args.check_signals:
        print("\nSignal lengths (samples: files):")
        for length, count in sorted(lengths.items()):
            print(f"  {length}: {count}")

    print("\nProblems:" if problems else "\nProblems: none")

    for problem in problems:
        print("  -", problem)

    os.makedirs(SUMMARY_DIR, exist_ok=True)

    output_path = os.path.join(SUMMARY_DIR, "dataset_inventory.csv")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", relative(output_path))


if __name__ == "__main__":
    main()
