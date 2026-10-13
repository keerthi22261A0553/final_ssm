import csv
import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from block_split import (
    SPLIT_NAMES,
    LeakageError,
    load_splits,
    overlap_counts,
    save_splits,
    split_dir,
    split_distance,
    verify_no_overlap,
)
from config import CLASSES, SPLIT_FRACTIONS, SUMMARY_DIR
from data_loader import RadarDataLoader
from utils import relative


def class_report(distance, class_name, report, splits):

    rows = {name: [r for r in splits[name] if r["class_name"] == class_name] for name in splits}

    print(f"\nDistance: {distance}")
    print(f"Class: {class_name}\n")

    for name, count, first, last in report:
        print(f"{name:6s}: {count:3d} files  (files {first:03d}-{last:03d})")

    print(f"\nTotal files: {sum(len(v) for v in rows.values())}")

    print("\nOverlap check:")

    for pair, count in overlap_counts(rows).items():
        print(f"{pair.upper()}: {count}")

    return rows


def main():

    loader = RadarDataLoader()

    distances = loader.selected_distances()

    log = io.StringIO()

    summary = []
    seen_paths = {}

    print("Whole-file split in recording order: "
          + ", ".join(f"{n} {SPLIT_FRACTIONS[n]:.0%}" for n in SPLIT_NAMES))
    print(f"Distances ({len(distances)}): {', '.join(distances)}\n")

    for distance in distances:

        records = loader.find_mat_files(distance)

        splits, reports = split_distance(records)

        save_splits(distance, splits)

        # Check the files that were actually written, not just memory
        saved = load_splits(distance)

        verify_no_overlap(saved)

        assigned = sum(len(rows) for rows in saved.values())

        if assigned != len(records):
            raise LeakageError(
                f"{distance}: {len(records)} files found but {assigned} assigned."
            )

        # The same file must never belong to two distances
        for rows in saved.values():
            for r in rows:
                if r["file_path"] in seen_paths:
                    raise LeakageError(
                        f"{r['file_path']} in {seen_paths[r['file_path']]} and {distance}"
                    )
                seen_paths[r["file_path"]] = distance

        with redirect_stdout(log):

            print("=" * 60)

            for class_name in CLASSES:

                rows = class_report(distance, class_name, reports[class_name], saved)

                summary.append({
                    "distance": distance,
                    "class_name": class_name,
                    "total_files": sum(len(v) for v in rows.values()),
                    **{name: len(rows[name]) for name in SPLIT_NAMES},
                })

        counts = "  ".join(f"{name} {len(saved[name]):4d}" for name in SPLIT_NAMES)

        print(f"  {distance:>4s}: {counts}  overlaps {sum(overlap_counts(saved).values())}"
              f"  -> {relative(split_dir(distance))}")

    os.makedirs(SUMMARY_DIR, exist_ok=True)

    report_path = os.path.join(SUMMARY_DIR, "split_report.txt")

    with open(report_path, "w") as f:
        f.write(log.getvalue())

    summary_path = os.path.join(SUMMARY_DIR, "dataset_summary.csv")

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print("\nAll overlap checks passed (train/val/test disjoint, every file in one set,")
    print("no file shared between distances).")
    print("\nPer-class report :", relative(report_path))
    print("Dataset summary  :", relative(summary_path))

    # Show one full class report on screen as an example
    print("\nExample (first class, first distance):")
    print("\n".join(log.getvalue().splitlines()[1:15]))


if __name__ == "__main__":
    main()
