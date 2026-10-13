import os


# Single place for every experiment setting. Other files import from
# here; no value below is repeated anywhere else in the project.


# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# TSMS-Drone CW radar data, copied here by the user; only real recorded
# files are used and the code never writes to this folder.
# Expected layout: <RAW_DATA_DIR>/<distance>/<class>/Raw File/*.mat
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "CW Radar")

SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary")


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------

# None = detect the distance folders that actually exist in RAW_DATA_DIR.
# A list (e.g. ["8m", "14m"]) restricts the run to those distances.
DISTANCES = None

# Label order: index in this list = class label
CLASSES = [
    "Inspire 2",
    "Matrice 30",
    "Mavic 2 Pro",
    "Phantom 4 Pro",
]

EXCLUDED_CLASSES = ["Corner Reflector"]


# ---------------------------------------------------------------
# File-level split (done before any segmentation)
# ---------------------------------------------------------------

# Per distance and class, the files sorted by recording number are cut
# into consecutive chunks of whole files, in this order. With 500 files:
# 001-350 train, 351-425 validation, 426-500 test.
SPLIT_ORDER = ["train", "val", "test"]

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


# ---------------------------------------------------------------
# Signal processing (unchanged from the original UIB-SSM project)
# ---------------------------------------------------------------

SAMPLING_RATE = 65536
TONE_FREQUENCY = 1024

SEGMENT_LENGTH = 1024
SEGMENT_HOP = None          # None = non-overlapping

N_FFT = 128
STFT_HOP = 32


# ---------------------------------------------------------------
# Model and training (unchanged from the original UIB-SSM project)
# ---------------------------------------------------------------

MODEL = {
    "num_classes": len(CLASSES),
    "input_shape": [1, N_FFT, SEGMENT_LENGTH // STFT_HOP + 1],
    "uib_channels": [16, 24, 32, 48],
    "d_model": 128,
    "d_state": 16,
    "num_ssm_layers": 2,
    "dropout": 0.1,
}

TRAINING = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "max_epochs": 150,
    "patience": 7,          # early stopping on validation loss
    # A validation loss counts as an improvement only if it is lower by at
    # least min_delta. Without it, runs at 100 % validation accuracy kept
    # "improving" by 1e-10 and never stopped. Replay on the saved
    # histories: identical best epoch for 2-18 m; 20 m and 22 m retrained.
    "min_delta": 1e-4,
    "lr_factor": 0.5,       # halve LR when validation loss stalls
    "lr_patience": 2,
    "min_lr": 1e-5,
}


# ---------------------------------------------------------------
# Noise and denoising experiments
# ---------------------------------------------------------------

SNR_LEVELS = [20, 10, 5]    # dB, added AWGN, test files only

# Denoiser applied to the noisy complex signal before preprocessing.
# Parameters fixed in advance (standard defaults), never tuned on test data.
DENOISER = {
    "method": "stft_wiener",
    "nperseg": 256,          # Hann window length (samples)
    "overlap": 0.75,         # 75 % overlap: exact inverse STFT
    "gain_floor": 0.0,       # minimum gain per time-frequency bin
}


# ---------------------------------------------------------------
# Optional experiments (run after the per-distance experiments)
# ---------------------------------------------------------------

# Noise-augmented training: a separate model per distance. Each epoch,
# every TRAINING file gets fresh AWGN at an SNR drawn uniformly from
# snr_range_db (or stays clean with probability clean_fraction).
# Validation and test files are never augmented.
NOISE_AUG = {
    "snr_range_db": [0, 30],
    "clean_fraction": 0.25,
}

# Cross-distance generalisation: train on these distances, test on a
# distance that is not among them
CROSS_DISTANCE = {
    "train": ["8m", "10m", "12m", "14m"],
    "test": "16m",
}


# ---------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------

RANDOM_SEED = 42
