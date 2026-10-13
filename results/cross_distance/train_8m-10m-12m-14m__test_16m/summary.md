# Cross-distance: train 8m, 10m, 12m, 14m -> test 16m

Test files: the 300 test files of 16m (never seen in training). Accuracy in percent; n/a = not run.

## segment-level

| Model | clean | 20dB | 10dB | 5dB | 20dB+den | 10dB+den | 5dB+den |
|---|---|---|---|---|---|---|---|
| Cross-distance (unseen 16m) | 53.85 | 52.31 | 53.98 | 60.35 | 49.92 | 49.94 | 50.04 |
| Per-distance model trained at 16m | 99.79 | 99.73 | 99.21 | 95.40 | 99.88 | 99.98 | 99.75 |

## file-level

| Model | clean | 20dB | 10dB | 5dB | 20dB+den | 10dB+den | 5dB+den |
|---|---|---|---|---|---|---|---|
| Cross-distance (unseen 16m) | 52.67 | 51.33 | 54.33 | 61.33 | 50.00 | 50.00 | 50.00 |
| Per-distance model trained at 16m | 100.00 | 100.00 | 99.33 | 99.00 | 100.00 | 100.00 | 100.00 |
