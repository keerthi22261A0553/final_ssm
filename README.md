# Lightweight UIB-SSM — Multi-Distance Noise / Denoising Experiment

Micro-Doppler classification of 4 UAV classes (Inspire 2, Matrice 30,
Mavic 2 Pro, Phantom 4 Pro) from TSMS-Drone **CW radar** data, at every
available target distance, with a whole-file split (70 / 15 / 15 train /
validation / test, in recording order), a clean baseline, AWGN robustness (20 / 10 / 5 dB) and a denoising comparison.

New, separate project. The earlier projects (`Lightweight_UIB_SSM`,
`Lightweight_UIB_SSM_8m`) are not modified. The real TSMS-Drone CW files are
copied by hand into `data/raw/CW Radar/<distance>/<class>/Raw File/`; no
synthetic data is generated, and the code never writes to that folder.

## Folder structure

```
final_ssm/
├── src/
│   ├── config.py          all settings (distances, classes, blocks, SNRs, seed)
│   ├── utils.py           seed + path helpers
│   ├── data_loader.py     find / load .mat files, any distance      (adapted)
│   ├── block_split.py     whole-file 70/15/15 split + leakage check  (new)
│   ├── preprocessing.py   shift, I/Q, DC removal, z-score            (unchanged copy)
│   ├── segmentation.py    1024-sample windows                        (unchanged copy)
│   ├── spectrogram.py     STFT log-magnitude, NumPy                  (unchanged copy)
│   ├── dataset.py         split csv -> segments -> spectrograms      (adapted)
│   ├── uib.py             UIB feature extractor                      (unchanged copy)
│   ├── ssm.py             feature-to-sequence + diagonal SSM          (unchanged copy)
│   ├── model.py           UIB-SSM classifier                         (unchanged copy)
│   ├── noise.py           AWGN on a temporary copy                   (new)
│   ├── denoising.py       modular signal-processing denoiser         (new)
│   └── metrics.py         accuracy / P / R / F1 / CM / file-level    (new, from evaluate.py)
├── scripts/
│   ├── inspect_dataset.py   step 0  inventory of distances and files
│   ├── make_splits.py       step 1  block split + leakage report
│   ├── train.py             step 2  clean model per distance
│   ├── evaluate_clean.py    step 3  clean baseline
│   ├── evaluate_noise.py    step 4-5 noisy and noisy+denoised (same noisy signal)
│   ├── make_report.py       step 6  tables A-E, graphs F
│   └── run_all.py           everything in order
├── tests/
├── data/raw/CW Radar/<distance>/<class>/Raw File/*.mat   (real data, read-only)
├── data/splits/<distance>/{train,val,test}.csv
├── checkpoints/<distance>/clean/
├── results/<distance>/{clean, snr_20dB_noisy, snr_20dB_denoised, ...}/
├── results/summary/       final tables and graphs
└── docs/
```

## Implementation plan (one file at a time)

1. `utils.py`
2. `data_loader.py`, `scripts/inspect_dataset.py`
3. `block_split.py`, `scripts/make_splits.py`, tests
4. `dataset.py`
5. `metrics.py`, `scripts/train.py`, `scripts/evaluate_clean.py`
6. `noise.py`, tests
7. `denoising.py`, tests
8. `scripts/evaluate_noise.py`
9. `scripts/make_report.py`, `scripts/run_all.py`

Optional, later and kept separate: cross-distance generalisation,
noise-augmented training.

## Data flow

```
distance -> class -> .mat files sorted by number (001-500)
  -> TRAIN 001-350 (70%) / VAL 351-425 (15%) / TEST 426-500 (15%)
  -> overlap check = 0
  -> [test only: AWGN copy -> optional denoising]
  -> preprocessing -> segmentation -> spectrogram -> UIB -> SSM -> FC -> prediction
```

## Running

```
python scripts/run_all.py        # 15 distances: train, clean / noisy / denoised test, report
python scripts/run_optional.py   # afterwards: cross-distance, then noise-augmented models
```

Both skip finished steps, so they can be restarted after an interruption.
Results: `results/summary/final_results.md` (tables A-E, observations) and the
PNG graphs next to it; per condition `results/<distance>/<model>/<condition>/`.

## Optional experiments (kept separate from the main results)

- **All-distance model** (`scripts/all_distances.py`): ONE model trained on the
  train files of all 15 distances (validation files for early stopping), tested
  on each distance's own test files under all 7 conditions. Results in
  `results/<distance>/all_distances/`, checkpoint in `checkpoints/all_distances/`.
  Spectrograms are stored as float16 to fit in RAM (max rounding error 0.002).
- **Cross-distance** (`scripts/cross_distance.py`, `config.CROSS_DISTANCE`):
  train on 8+10+12+14 m (their train/val files), test on the unseen 16 m test
  files under all 7 conditions. Results in `results/cross_distance/`, with the
  16 m per-distance model alongside in `summary.md`.
- **Noise-augmented training** (`train.py --model noise_aug`, `config.NOISE_AUG`):
  a second model per distance; each epoch every training file gets fresh AWGN
  (SNR uniform in 0-30 dB, 25 % left clean). Validation and test files are never
  augmented. Stored in `checkpoints/<distance>/noise_aug/`; the clean baseline is
  never overwritten. Reported in a separate section of `final_results.md`.
