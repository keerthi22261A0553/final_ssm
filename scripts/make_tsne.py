import os
import sys
import time

import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkpoints import load_trained_model
from config import CHECKPOINT_DIR, CLASSES, RANDOM_SEED, RESULTS_DIR
from data_loader import RadarDataLoader, distance_metres
from dataset import RadarSegmentDataset
from denoising import NoisyThenDenoised, build_denoiser
from model import UIBSSMClassifier
from noise import AWGN
from utils import relative

# t-SNE of the learned features: the 128-dim pooled SSM output, right
# before the final classification layer. One point per TEST recording
# (mean of its 16 segment features). Everything is computed from the
# trained checkpoints and the real test files.

OUT = os.path.join(RESULTS_DIR, "presentation")

INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"

# Categorical colours in fixed order + a different marker per class, so
# classes stay distinguishable without relying on colour alone
CLASS_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CLASS_MARKERS = ["o", "s", "^", "D"]

plt.rcParams.update({"font.size": 12, "axes.titlesize": 14})


@torch.no_grad()
def features(model, dataset, batch_size=512):

    # Same path as model.forward(), stopping before the classifier
    model.eval()

    feats, probs = [], []

    for inputs, _ in DataLoader(dataset, batch_size=batch_size):
        pooled = model.ssm(model.sequence(model.uib(inputs))).mean(dim=1)
        feats.append(pooled)
        probs.append(torch.softmax(model.classifier(pooled), dim=1))

    feats = torch.cat(feats).numpy()
    probs = torch.cat(probs).numpy()

    # One point per recording: mean over its segments
    index = dataset.file_index.numpy()
    n = len(dataset.records)

    file_feats = np.zeros((n, feats.shape[1]))
    file_probs = np.zeros((n, probs.shape[1]))
    np.add.at(file_feats, index, feats)
    np.add.at(file_probs, index, probs)

    counts = np.bincount(index, minlength=n)[:, None]

    return file_feats / counts, (file_probs / counts).argmax(axis=1), dataset.file_labels()


def tsne(x):

    # Standardise features, then t-SNE with a fixed seed (reproducible)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)

    return TSNE(n_components=2, perplexity=30, init="pca", random_state=RANDOM_SEED).fit_transform(x)


def style(ax, title):

    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_color("#e1e0d9")

    ax.set_title(title, color=INK, loc="left")


def scatter_classes(ax, xy, labels, wrong=None, size=14):

    for k, name in enumerate(CLASSES):
        m = labels == k
        ax.scatter(xy[m, 0], xy[m, 1], s=size, c=CLASS_COLOURS[k], marker=CLASS_MARKERS[k],
                   alpha=0.75, linewidths=0, label=name)

    if wrong is not None and wrong.any():
        ax.scatter(xy[wrong, 0], xy[wrong, 1], s=size * 4, facecolors="none",
                   edgecolors=INK, linewidths=0.9, label="misclassified")


def load_all_distance_model():

    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "all_distances", "best.pt"), map_location="cpu")

    model = UIBSSMClassifier(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state"])

    return model.eval()


def figure_all_distances(distances):

    model = load_all_distance_model()

    feats, preds, labels, dist = [], [], [], []

    for d in distances:
        f, p, y = features(model, RadarSegmentDataset(d, "test"))
        feats.append(f); preds.append(p); labels.append(y)
        dist.append(np.full(len(y), distance_metres(d)))

    feats, preds, labels, dist = map(np.concatenate, (feats, preds, labels, dist))

    xy = tsne(feats)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), facecolor=SURFACE)

    scatter_classes(axes[0], xy, labels, size=10)
    style(axes[0], "Coloured by drone")
    axes[0].legend(frameon=False, loc="best", markerscale=2)

    points = axes[1].scatter(xy[:, 0], xy[:, 1], s=10, c=dist, cmap="Blues", vmin=-4, vmax=30, linewidths=0)
    style(axes[1], "Coloured by distance")
    bar = fig.colorbar(points, ax=axes[1], fraction=0.04)
    bar.set_label("Distance (m)", color=INK_2)

    acc = 100 * (preds == labels).mean()

    fig.suptitle(f"t-SNE of learned features: one model for all distances "
                 f"({len(labels)} test recordings, recording accuracy {acc:.1f} %)",
                 color=INK, fontsize=14, x=0.01, ha="left")

    fig.tight_layout()
    path = os.path.join(OUT, "6_tsne_all_distance_model.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)

    return path


def figure_easy_vs_hard(easy, hard):

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), facecolor=SURFACE)

    for ax, d in zip(axes, (easy, hard)):

        model, _ = load_trained_model(d)
        f, p, y = features(model, RadarSegmentDataset(d, "test"))

        xy = tsne(f)
        wrong = p != y

        scatter_classes(ax, xy, y, wrong, size=30)
        style(ax, f"{d} model on {d} test recordings: {100 * (1 - wrong.mean()):.1f} % correct")

    axes[0].legend(frameon=False, loc="best")

    fig.suptitle("t-SNE of learned features: an easy and a hard distance (one model per distance)",
                 color=INK, fontsize=14, x=0.01, ha="left")

    fig.tight_layout()
    path = os.path.join(OUT, "7_tsne_easy_vs_hard_distance.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)

    return path


def figure_noise(distance, snr_db=5):

    model, _ = load_trained_model(distance)

    conditions = [
        ("Clean", None),
        (f"{snr_db} dB noise", AWGN(snr_db)),
        (f"{snr_db} dB noise + denoising", NoisyThenDenoised(AWGN(snr_db), build_denoiser())),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=SURFACE)

    for ax, (title, transform) in zip(axes, conditions):

        f, p, y = features(model, RadarSegmentDataset(distance, "test", signal_transform=transform))

        xy = tsne(f)
        wrong = p != y

        scatter_classes(ax, xy, y, wrong, size=30)
        style(ax, f"{title}: {100 * (1 - wrong.mean()):.1f} % correct")

    axes[0].legend(frameon=False, loc="best")

    fig.suptitle(f"t-SNE of learned features under noise ({distance} model, {distance} test recordings)",
                 color=INK, fontsize=14, x=0.01, ha="left")

    fig.tight_layout()
    path = os.path.join(OUT, "8_tsne_noise_and_denoising.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)

    return path


def figure_all_distance_noise(distances, snr_db=5):

    # All-distance model, all test recordings of every distance, under
    # clean / noisy / noisy+denoised input (same noisy signal per file)
    model = load_all_distance_model()

    conditions = [
        ("Clean", None),
        (f"{snr_db} dB noise", AWGN(snr_db)),
        (f"{snr_db} dB noise + denoising", NoisyThenDenoised(AWGN(snr_db), build_denoiser())),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), facecolor=SURFACE)

    for ax, (title, transform) in zip(axes, conditions):

        feats, preds, labels = [], [], []

        for d in distances:
            f, p, y = features(model, RadarSegmentDataset(d, "test", signal_transform=transform))
            feats.append(f); preds.append(p); labels.append(y)

        feats, preds, labels = map(np.concatenate, (feats, preds, labels))

        xy = tsne(feats)
        wrong = preds != labels

        scatter_classes(ax, xy, labels, wrong, size=8)
        style(ax, f"{title}: {100 * (1 - wrong.mean()):.1f} % correct")

    axes[0].legend(frameon=False, loc="best", markerscale=2)

    fig.suptitle(f"t-SNE of learned features under noise: one model for all distances "
                 f"({len(labels)} test recordings, all 15 distances; circles = misclassified)",
                 color=INK, fontsize=14, x=0.01, ha="left")

    fig.tight_layout()
    path = os.path.join(OUT, "9_tsne_all_distance_noise_and_denoising.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)

    return path


def main():

    import argparse

    parser = argparse.ArgumentParser(description="t-SNE figures of the learned features.")
    parser.add_argument("--only", type=int, choices=(6, 7, 8, 9),
                        help="Make only this figure number")

    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(RANDOM_SEED)

    distances = sorted(RadarDataLoader().selected_distances(), key=distance_metres)

    figures = {
        6: lambda: figure_all_distances(distances),
        7: lambda: figure_easy_vs_hard("20m", "14m"),
        8: lambda: figure_noise("8m"),
        9: lambda: figure_all_distance_noise(distances),
    }

    start = time.time()

    for number, make in figures.items():
        if args.only in (None, number):
            print("Saved:", relative(make()), f"({time.time() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
