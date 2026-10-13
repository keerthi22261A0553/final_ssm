import csv
import hashlib
import math
import os

from config import CLASSES, PROJECT_ROOT, SPLIT_FRACTIONS, SPLIT_ORDER, SPLITS_DIR
from data_loader import FILE_NUMBER_PATTERN, file_sort_key
from utils import relative

SPLIT_NAMES = tuple(SPLIT_ORDER)

SPLIT_COLUMNS = ["file_path", "distance", "class_name", "label", "block", "file_number"]


class LeakageError(Exception):

    # Raised when a file would end up in more than one set
    pass


def check_split_config():

    if sorted(SPLIT_ORDER) != sorted(SPLIT_FRACTIONS):
        raise ValueError("SPLIT_ORDER and SPLIT_FRACTIONS must name the same splits.")

    if abs(sum(SPLIT_FRACTIONS.values()) - 1) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1: {SPLIT_FRACTIONS}")


def chunk_sizes(num_files):

    # Whole files per split: floor for every split but the last, which
    # takes the rest, so every file is assigned exactly once.
    # 500 files -> train 350, val 75, test 75
    sizes = {
        name: math.floor(num_files * SPLIT_FRACTIONS[name])
        for name in SPLIT_ORDER[:-1]
    }

    sizes[SPLIT_ORDER[-1]] = num_files - sum(sizes.values())

    return sizes


def split_class(paths):

    # Files of ONE class at ONE distance -> consecutive chunks of whole
    # files in recording order. Returns {split: [paths]} and a report.
    check_split_config()

    paths = sorted(paths, key=file_sort_key)

    sizes = chunk_sizes(len(paths))

    empty = [name for name, size in sizes.items() if size == 0]

    if empty:
        raise ValueError(
            f"Only {len(paths)} files; splits {empty} would be empty."
        )

    assignment = {}
    report = []
    start = 0

    for block, name in enumerate(SPLIT_ORDER, start=1):

        chunk = paths[start:start + sizes[name]]

        assignment[name] = chunk

        report.append((
            name.upper(),
            len(chunk),
            file_number(chunk[0]),
            file_number(chunk[-1]),
        ))

        start += sizes[name]

    return assignment, report


def file_number(path):

    match = FILE_NUMBER_PATTERN.search(os.path.basename(path))

    return int(match.group(1)) if match else ""


def split_distance(records):

    # Records of ONE distance (from RadarDataLoader.find_mat_files)
    distances = {r["distance"] for r in records}

    if len(distances) != 1:
        raise ValueError(f"Expected one distance, got {sorted(distances)}.")

    distance = distances.pop()

    splits = {name: [] for name in SPLIT_NAMES}
    reports = {}

    for label, class_name in enumerate(CLASSES):

        paths = [r["file_path"] for r in records if r["class_name"] == class_name]

        assignment, report = split_class(paths)

        reports[class_name] = report

        for block, name in enumerate(SPLIT_NAMES, start=1):
            for path in assignment[name]:
                splits[name].append({
                    "file_path": relative(path),
                    "distance": distance,
                    "class_name": class_name,
                    "label": label,
                    "block": block,
                    "file_number": file_number(path),
                })

    verify_no_overlap(splits)

    return splits, reports


def overlap_counts(splits):

    # Number of shared file paths for every pair of sets
    paths = {
        name: {os.path.normcase(os.path.normpath(r["file_path"])) for r in rows}
        for name, rows in splits.items()
    }

    names = list(paths)

    return {
        f"{a} vs {b}": len(paths[a] & paths[b])
        for i, a in enumerate(names)
        for b in names[i + 1:]
    }


def verify_no_overlap(splits):

    # 1) no file path in two sets
    counts = overlap_counts(splits)

    leaks = {pair: n for pair, n in counts.items() if n}

    if leaks:
        raise LeakageError(f"Files shared between sets: {leaks}")

    # 2) no file path twice anywhere (also catches duplicates inside a set)
    all_paths = [
        os.path.normcase(os.path.normpath(r["file_path"]))
        for rows in splits.values()
        for r in rows
    ]

    if len(all_paths) != len(set(all_paths)):
        raise LeakageError("A file path occurs more than once.")

    # 3) one distance per split table
    distances = {r["distance"] for rows in splits.values() for r in rows}

    if len(distances) > 1:
        raise LeakageError(f"Distances mixed in one split: {sorted(distances)}")

    return counts


def split_dir(distance):

    return os.path.join(SPLITS_DIR, distance)


def save_splits(distance, splits):

    folder = split_dir(distance)

    os.makedirs(folder, exist_ok=True)

    # Remove split files from an earlier split plan, so a stale list
    # can never be picked up
    for name in os.listdir(folder):
        if name.endswith(".csv") and name[:-4] not in SPLIT_NAMES:
            os.remove(os.path.join(folder, name))

    paths = {}

    for name in SPLIT_NAMES:

        paths[name] = os.path.join(folder, f"{name}.csv")

        with open(paths[name], "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SPLIT_COLUMNS)
            writer.writeheader()
            writer.writerows(splits[name])

    return paths


def load_split(distance, name):

    # Rows of one saved split; file_path made absolute again
    path = os.path.join(split_dir(distance), f"{name}.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run scripts/make_splits.py first.")

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        row["label"] = int(row["label"])
        row["block"] = int(row["block"])
        row["abs_path"] = os.path.join(PROJECT_ROOT, row["file_path"])

    return rows


def load_splits(distance):

    return {name: load_split(distance, name) for name in SPLIT_NAMES}


def split_fingerprint(distance):

    # SHA-256 of the saved split files. Stored in every checkpoint, so a
    # model is only ever evaluated on the split it was trained with.
    digest = hashlib.sha256()

    for name in SPLIT_NAMES:
        with open(os.path.join(split_dir(distance), f"{name}.csv"), "rb") as f:
            digest.update(name.encode() + b"\0" + f.read().replace(b"\r\n", b"\n"))

    return digest.hexdigest()
