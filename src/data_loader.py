import os
import re

import numpy as np
import scipy.io as sio

from config import CLASSES, DISTANCES, EXCLUDED_CLASSES, RAW_DATA_DIR

# Distance folders are named like "8m", "14m"
DISTANCE_PATTERN = re.compile(r"^(\d+)m$")

# Trailing file number, e.g. "Inspire 2_14m_003.mat" -> 3
FILE_NUMBER_PATTERN = re.compile(r"_(\d+)\.mat$", re.IGNORECASE)


def distance_metres(distance):

    return int(DISTANCE_PATTERN.match(distance).group(1))


def file_sort_key(file_path):

    # Sort by recording number, not by text ("_10" after "_9"),
    # with the file name as tie-breaker so the order is always the same
    name = os.path.basename(file_path)
    match = FILE_NUMBER_PATTERN.search(name)

    number = int(match.group(1)) if match else float("inf")

    return number, name


class RadarDataLoader:

    def __init__(self, data_dir=RAW_DATA_DIR):

        self.data_dir = data_dir

    def detect_distances(self):

        # Distance folders that actually exist, sorted 2m, 4m, ..., 30m
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Dataset folder not found: {self.data_dir}")

        found = [
            name for name in os.listdir(self.data_dir)
            if DISTANCE_PATTERN.match(name)
            and os.path.isdir(os.path.join(self.data_dir, name))
        ]

        return sorted(found, key=distance_metres)

    def selected_distances(self):

        # config.DISTANCES restricts the run; None means every distance found
        available = self.detect_distances()

        if DISTANCES is None:
            return available

        missing = [d for d in DISTANCES if d not in available]

        if missing:
            raise ValueError(f"Configured distances not in the dataset: {missing}")

        return sorted(DISTANCES, key=distance_metres)

    def class_folder(self, distance, class_name):

        return os.path.join(self.data_dir, distance, class_name, "Raw File")

    def list_class_files(self, distance, class_name):

        # Every .mat file of one class at one distance, deterministic order
        folder = self.class_folder(distance, class_name)

        if not os.path.isdir(folder):
            return []

        paths = [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.lower().endswith(".mat")
        ]

        return sorted(paths, key=file_sort_key)

    def find_mat_files(self, distance=None):

        # Records for ONE distance only, so distances are never mixed.
        # Without an argument the first selected distance is used.
        if distance is None:
            distance = self.selected_distances()[0]

        records = []

        for label, class_name in enumerate(CLASSES):

            for file_path in self.list_class_files(distance, class_name):

                records.append({
                    "file_path": file_path,
                    "distance": distance,
                    "class_name": class_name,
                    "label": label,
                })

        return records

    def load_mat_file(self, file_path):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read-only: the file on disk is never modified
        mat_data = sio.loadmat(file_path)

        if "data" not in mat_data:
            raise KeyError(f"'data' variable not found in {file_path}")

        signal = np.asarray(mat_data["data"]).squeeze()

        if not np.iscomplexobj(signal):
            raise ValueError(f"Radar data is not complex: {file_path}")

        return signal


if __name__ == "__main__":

    loader = RadarDataLoader()

    distances = loader.selected_distances()

    print("Dataset folder :", loader.data_dir)
    print("Distances      :", ", ".join(distances))
    print("Classes        :", ", ".join(CLASSES))
    print("Excluded       :", ", ".join(EXCLUDED_CLASSES))

    record = loader.find_mat_files(distances[0])[0]
    signal = loader.load_mat_file(record["file_path"])

    print("\nFirst file     :", os.path.basename(record["file_path"]))
    print("Samples        :", len(signal), signal.dtype)
