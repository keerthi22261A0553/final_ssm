import json
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from config import CLASSES, RESULTS_DIR
from data_loader import RadarDataLoader, distance_metres
from utils import relative

# A small set of presentation graphs, one message each, read only from the
# saved metrics.json files (no value is typed in by hand)

OUT = os.path.join(RESULTS_DIR, "presentation")

INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 12})

DISTANCES = sorted(RadarDataLoader().selected_distances(), key=distance_metres)
X = [distance_metres(d) for d in DISTANCES]

CONDITIONS = ["test_clean", "test_snr20_noisy", "test_snr10_noisy", "test_snr5_noisy",
              "test_snr20_denoised", "test_snr10_denoised", "test_snr5_denoised"]


def metrics(distance, run, condition):

    path = os.path.join(RESULTS_DIR, distance, run, condition, "metrics.json")

    with open(path) as f:
        return json.load(f)


def acc(distance, run, condition, level="segment_level"):

    return 100 * metrics(distance, run, condition)[level]["accuracy"]


def table(run, level="segment_level"):

    return np.array([[acc(d, run, c, level) for c in CONDITIONS] for d in DISTANCES])


def style(ax, ylabel="Test accuracy (%)"):

    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")

    ax.tick_params(colors=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)


def save(fig, name):

    fig.tight_layout()
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)

    return path


def graph_clean_per_distance(per, alld):

    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor=SURFACE)

    ax.plot(X, per[:, 0], color=BLUE, linewidth=2.5, marker="o", markersize=8,
            label=f"One model per distance (mean {per[:, 0].mean():.1f} %)")
    ax.plot(X, alld[:, 0], color=ORANGE, linewidth=2.5, marker="s", markersize=8,
            label=f"One model for all distances (mean {alld[:, 0].mean():.1f} %)")

    ax.axhline(25, color=MUTED, linestyle=":", linewidth=1)
    ax.text(X[0], 27, "chance (25 %)", color=MUTED, fontsize=10)

    style(ax)
    ax.set_xticks(X, [f"{x}" for x in X])
    ax.set_xlabel("Target distance (m)", color=INK_2)
    ax.set_ylim(0, 103)
    ax.set_title("Clean test accuracy at every distance", color=INK, loc="left")
    ax.legend(frameon=False, loc="lower left")

    return save(fig, "1_clean_accuracy_per_distance.png")


def graph_model_summary(per, aug, alld, alld_files, per_files):

    labels = ["One model\nper distance", "Noise-trained\nper distance", "One model for\nall distances"]
    seg = [per[:, 0].mean(), aug[:, 0].mean(), alld[:, 0].mean()]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)

    bars = ax.bar(labels, seg, color=[BLUE, AQUA, ORANGE], width=0.55)

    for bar, value in zip(bars, seg):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f} %",
                ha="center", color=INK, fontsize=14, fontweight="bold")

    style(ax, "Mean clean test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Clean accuracy, mean of 15 distances (segments)", color=INK, loc="left")

    ax.text(2, 12, f"per recording:\n{alld_files:.1f} %", ha="center", color="white",
            fontsize=12, fontweight="bold")
    ax.text(0, 12, f"per recording:\n{per_files:.1f} %", ha="center", color="white",
            fontsize=12, fontweight="bold")

    return save(fig, "2_clean_accuracy_models.png")


def graph_snr(per, aug, alld):

    snr_labels = ["Clean", "20 dB", "10 dB", "5 dB"]
    x = np.arange(4)

    curves = [
        (per, False, BLUE, "-", "Per-distance model"),
        (per, True, BLUE, "--", "Per-distance model + denoising"),
        (aug, False, AQUA, "-", "Noise-trained model"),
        (aug, True, AQUA, "--", "Noise-trained model + denoising"),
        (alld, False, ORANGE, "-", "All-distance model"),
    ]

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=SURFACE)

    for data, denoised, colour, dash, label in curves:

        mean = data.mean(axis=0)
        y = [mean[0]] + ([mean[4], mean[5], mean[6]] if denoised else [mean[1], mean[2], mean[3]])

        ax.plot(x, y, color=colour, linestyle=dash, linewidth=2.5, marker="o", markersize=8, label=label)
        ax.annotate(f"{y[-1]:.1f}", (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=11)

    ax.axhline(25, color=MUTED, linestyle=":", linewidth=1)

    style(ax, "Mean test accuracy (%)")
    ax.set_xticks(x, snr_labels)
    ax.set_xlim(-0.2, 3.5)
    ax.set_ylim(0, 103)
    ax.set_xlabel("Added noise (AWGN, SNR relative to the drone signal)", color=INK_2)
    ax.set_title("Accuracy under noise, mean of 15 distances", color=INK, loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=11)

    return save(fig, "3_accuracy_vs_noise.png")


def graph_denoising_gain(per):

    width = 0.27
    x = np.arange(len(DISTANCES))

    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=SURFACE)

    for i, (snr, colour) in enumerate([(20, "#86b6ef"), (10, "#3987e5"), (5, "#1c5cab")]):
        col = {20: (4, 1), 10: (5, 2), 5: (6, 3)}[snr]
        gain = per[:, col[0]] - per[:, col[1]]
        ax.bar(x + (i - 1) * width, gain, width, color=colour,
               label=f"{snr} dB (mean {gain.mean():+.1f} points)")

    ax.axhline(0, color=INK_2, linewidth=1)

    style(ax, "Accuracy change from denoising\n(percentage points)")
    ax.set_xticks(x, [f"{d}" for d in DISTANCES])
    ax.set_xlabel("Target distance", color=INK_2)
    ax.set_title("Denoising helps at strong noise, hurts at weak noise (per-distance model)",
                 color=INK, loc="left")
    ax.legend(frameon=False, loc="upper left")

    return save(fig, "4_denoising_gain.png")


def graph_cross_distance():

    folder = os.path.join(RESULTS_DIR, "cross_distance")
    name = next(n for n in os.listdir(folder) if n.startswith("train_"))

    with open(os.path.join(folder, name, "test_clean", "metrics.json")) as f:
        cross = 100 * json.load(f)["segment_level"]["accuracy"]

    test = name.split("__test_")[1]
    own = acc(test, "clean", "test_clean")
    alld = acc(test, "all_distances", "test_clean")

    labels = [f"Trained at {test}", "Trained on all\n15 distances", f"Trained on 8-14 m\n({test} never seen)"]
    values = [own, alld, cross]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)

    bars = ax.bar(labels, values, color=[BLUE, ORANGE, MUTED], width=0.55)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f} %",
                ha="center", color=INK, fontsize=14, fontweight="bold")

    ax.axhline(25, color=MUTED, linestyle=":", linewidth=1)

    style(ax, f"Clean test accuracy at {test} (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Does the model work at an unseen distance? (test at {test})", color=INK, loc="left")

    return save(fig, "5_unseen_distance.png")


def main():

    os.makedirs(OUT, exist_ok=True)

    per = table("clean")
    aug = table("noise_aug")
    alld = table("all_distances")

    per_files = np.mean([acc(d, "clean", "test_clean", "file_level") for d in DISTANCES])
    alld_files = np.mean([acc(d, "all_distances", "test_clean", "file_level") for d in DISTANCES])

    paths = [
        graph_clean_per_distance(per, alld),
        graph_model_summary(per, aug, alld, alld_files, per_files),
        graph_snr(per, aug, alld),
        graph_denoising_gain(per),
        graph_cross_distance(),
    ]

    for path in paths:
        print("Saved:", relative(path))


if __name__ == "__main__":
    main()
