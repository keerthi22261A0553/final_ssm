import argparse
import csv
import json
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from checkpoints import condition_dir
from config import CLASSES, DENOISER, NOISE_AUG, SNR_LEVELS, SUMMARY_DIR
from data_loader import RadarDataLoader, distance_metres
from utils import relative

# Chart colours (validated reference palette)
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
NOISY = "#2a78d6"        # categorical slot 1
DENOISED = "#eb6834"     # categorical slot 2
SNR_RAMP = ["#1c5cab", "#3987e5", "#86b6ef"]   # ordinal blue, high -> low SNR
HEAT = "Blues"

LEVELS = {"segment": "segment_level", "file": "file_level"}


# ---------------------------------------------------------------
# Collect results (only values produced by the experiments)
# ---------------------------------------------------------------

def conditions():

    # (key, column title, snr, denoised) in table order
    items = [("clean", "Clean", None, False)]
    items += [(f"{s:g}dB", f"{s:g} dB", s, False) for s in SNR_LEVELS]
    items += [(f"{s:g}dB+den", f"{s:g} dB + Denoising", s, True) for s in SNR_LEVELS]

    return items


def load_metrics(distance, snr_db, denoised, run_name="clean"):

    path = os.path.join(condition_dir(distance, snr_db, denoised, run_name), "metrics.json")

    if not os.path.exists(path):
        return None

    with open(path) as f:
        return json.load(f)


def collect(distances, run_name="clean"):

    return {
        d: {key: load_metrics(d, snr, den, run_name) for key, _, snr, den in conditions()}
        for d in distances
    }


def value(results, distance, key, level, field="accuracy"):

    m = results[distance][key]

    return None if m is None else m[LEVELS[level]][field]


def pct(v):

    return "n/a" if v is None else f"{100 * v:.2f}"


def delta(a, b):

    return None if a is None or b is None else a - b


def signed(v):

    return "n/a" if v is None else f"{100 * v:+.2f}"


# ---------------------------------------------------------------
# Tables
# ---------------------------------------------------------------

def md_table(header, rows):

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]

    return "\n".join(lines)


def write_csv(path, header, rows):

    # UTF-8 with BOM: keeps "Δ" readable, also when opened in Excel
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def dataset_table():

    path = os.path.join(SUMMARY_DIR, "dataset_summary.csv")

    if not os.path.exists(path):
        return ["Distance", "Class", "Total", "Train", "Val", "Test"], []

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    return (
        ["Distance", "Class", "Total Files", "Train", "Val", "Test"],
        [[r["distance"], r["class_name"], r["total_files"], r["train"], r["val"], r["test"]]
         for r in rows],
    )


def clean_table(results, distances, level):

    header = ["Distance", "Accuracy", "Precision", "Recall", "F1", "Files / segments"]
    rows = []

    for d in distances:

        m = results[d]["clean"]
        s = None if m is None else m[LEVELS[level]]

        rows.append([
            d,
            pct(s and s["accuracy"]),
            pct(s and s["precision_macro"]),
            pct(s and s["recall_macro"]),
            pct(s and s["f1_macro"]),
            "n/a" if m is None else f"{m['test_files']} / {m['test_segments']}",
        ])

    return header, rows


def noise_table(results, distances, level):

    header = ["Distance"] + [f"{s:g} dB" for s in SNR_LEVELS]

    rows = [[d] + [pct(value(results, d, f"{s:g}dB", level)) for s in SNR_LEVELS]
            for d in distances]

    return header, rows


def denoise_table(results, distances, level):

    header = ["Distance"]

    for s in SNR_LEVELS:
        header += [f"{s:g} dB Noisy", f"{s:g} dB Denoised", f"{s:g} dB Δ"]

    rows = []

    for d in distances:

        row = [d]

        for s in SNR_LEVELS:
            noisy = value(results, d, f"{s:g}dB", level)
            den = value(results, d, f"{s:g}dB+den", level)
            row += [pct(noisy), pct(den), signed(delta(den, noisy))]

        rows.append(row)

    return header, rows


def final_table(results, distances, level):

    header = ["Distance"] + [title for _, title, _, _ in conditions()]

    rows = [[d] + [pct(value(results, d, key, level)) for key, _, _, _ in conditions()]
            for d in distances]

    return header, rows


def per_class_recall(results, distances, key, level):

    # Mean per-class recall over the distances where the condition exists
    out = {}

    for name in CLASSES:

        values = [
            results[d][key][LEVELS[level]]["per_class"][name]["recall"]
            for d in distances if results[d][key] is not None
        ]

        out[name] = float(np.mean(values)) if values else None

    return out


def per_class_table(results, distances, level):

    header = ["Class", "Clean"]

    for s in SNR_LEVELS:
        header += [f"{s:g} dB Noisy", f"{s:g} dB Denoised", f"{s:g} dB Δ (den − noisy)"]

    clean = per_class_recall(results, distances, "clean", level)

    noisy = {s: per_class_recall(results, distances, f"{s:g}dB", level) for s in SNR_LEVELS}
    den = {s: per_class_recall(results, distances, f"{s:g}dB+den", level) for s in SNR_LEVELS}

    rows = []

    for name in CLASSES:

        row = [name, pct(clean[name])]

        for s in SNR_LEVELS:
            row += [pct(noisy[s][name]), pct(den[s][name]),
                    signed(delta(den[s][name], noisy[s][name]))]

        rows.append(row)

    return header, rows, clean, noisy, den


# ---------------------------------------------------------------
# Interpretation: statements computed from the measured values only
# ---------------------------------------------------------------

def interpretation(results, distances, level, clean_cls, noisy_cls, den_cls):

    lines = []

    clean = {d: value(results, d, "clean", level) for d in distances}
    done = [d for d in distances if clean[d] is not None]

    if not done:
        return ["No clean results yet."]

    worst = min(done, key=lambda d: clean[d])
    best = max(done, key=lambda d: clean[d])

    lines.append(
        f"Clean accuracy ranges from {pct(clean[worst])} % ({worst}) to "
        f"{pct(clean[best])} % ({best}) over {len(done)} distance(s)."
    )

    for s in SNR_LEVELS:

        pairs = [
            (d, value(results, d, f"{s:g}dB", level), value(results, d, f"{s:g}dB+den", level))
            for d in done
        ]
        pairs = [(d, n, k) for d, n, k in pairs if n is not None and k is not None]

        if not pairs:
            continue

        drops = [clean[d] - n for d, n, _ in pairs]
        gains = [k - n for _, n, k in pairs]

        helped = sum(g > 0 for g in gains)
        hurt = sum(g < 0 for g in gains)

        hardest = min(pairs, key=lambda p: p[1])

        lines.append(
            f"{s:g} dB: mean accuracy drop from clean {100 * np.mean(drops):.2f} points "
            f"(noisy, mean {100 * np.mean([n for _, n, _ in pairs]):.2f} %); "
            f"lowest noisy accuracy {pct(hardest[1])} % at {hardest[0]}. "
            f"Denoising changed accuracy by {100 * np.mean(gains):+.2f} points on average "
            f"(range {100 * min(gains):+.2f} to {100 * max(gains):+.2f}); "
            f"higher at {helped}, lower at {hurt}, equal at {len(gains) - helped - hurt} "
            f"of {len(gains)} distance(s)."
        )

    # Collapse check: a class that receives most predictions has a high
    # recall only because it is the default guess, not because it is robust
    for key, title, _, _ in conditions():

        collapsed = []

        for d in done:

            m = results[d][key]

            if m is None:
                continue

            predicted = np.array(m[LEVELS[level]]["confusion_matrix"]).sum(axis=0)
            share = predicted.max() / predicted.sum()

            if share > 0.5:
                collapsed.append(f"{d}: {CLASSES[predicted.argmax()]} {100 * share:.0f} %")

        if collapsed:
            lines.append(
                f"{title}: one class receives more than half of all predictions "
                f"({'; '.join(collapsed)}). Its recall there reflects a default guess, "
                "not robustness."
            )

    low = min(SNR_LEVELS)

    if all(v is not None for v in noisy_cls[low].values()) and all(
        v is not None for v in clean_cls.values()
    ):
        drop = {c: clean_cls[c] - noisy_cls[low][c] for c in CLASSES}
        affected = max(drop, key=drop.get)

        lines.append(
            f"Largest mean recall drop from clean to {low:g} dB noisy: {affected} "
            f"({100 * drop[affected]:.2f} points); smallest: {min(drop, key=drop.get)} "
            f"({100 * min(drop.values()):.2f} points)."
        )

        if all(v is not None for v in den_cls[low].values()):

            gain = {c: den_cls[low][c] - noisy_cls[low][c] for c in CLASSES}
            top = max(gain, key=gain.get)

            lines.append(
                f"Largest mean recall change from denoising at {low:g} dB: {top} "
                f"({100 * gain[top]:+.2f} points); smallest: {min(gain, key=gain.get)} "
                f"({100 * min(gain.values()):+.2f} points)."
            )

    return lines


# ---------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------

def style(ax, ylabel="Test accuracy (%)"):

    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")

    ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def series(results, distances, key, level):

    return [None if (v := value(results, distances[i], key, level)) is None else 100 * v
            for i in range(len(distances))]


def plot_line(ax, x, y, **kwargs):

    # Missing values are left as gaps, never interpolated
    y = np.array([np.nan if v is None else v for v in y], dtype=float)

    ax.plot(x, y, linewidth=2, markersize=6, **kwargs)

    return y


def plot_accuracy_vs_distance(results, distances, level, path):

    x = [distance_metres(d) for d in distances]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, facecolor=SURFACE)

    for ax, denoised, title in zip(axes, (False, True), ("Noisy", "Noisy + denoising")):

        clean = plot_line(ax, x, series(results, distances, "clean", level),
                          color=INK, marker="o", label="Clean")

        lines = [("Clean", clean, INK)]

        for s, colour in zip(SNR_LEVELS, SNR_RAMP):

            key = f"{s:g}dB+den" if denoised else f"{s:g}dB"
            y = plot_line(ax, x, series(results, distances, key, level),
                          color=colour, marker="s", label=f"{s:g} dB")
            lines.append((f"{s:g} dB", y, colour))

        # Lines end close together at 30 m; the legend names them (no end labels)

        ax.axhline(25, color=MUTED, linestyle=":", linewidth=1)
        ax.text(x[0], 26, "chance (25 %)", fontsize=7, color=MUTED)

        style(ax)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Target distance (m)", color=INK_2, fontsize=9)
        ax.set_xticks(x)
        ax.set_ylim(0, 102)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, frameon=False, loc="lower left")

    fig.suptitle(f"Accuracy vs distance ({level}-level, clean-trained model per distance)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def snr_axis():

    return ["Clean"] + [f"{s:g} dB" for s in SNR_LEVELS]


def condition_curve(results, distance, level, denoised):

    keys = ["clean"] + [f"{s:g}dB+den" if denoised else f"{s:g}dB" for s in SNR_LEVELS]

    return [value(results, distance, k, level) for k in keys]


def plot_accuracy_vs_snr(results, distances, level, path):

    x = np.arange(len(snr_axis()))

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)

    for denoised, colour, label in ((False, NOISY, "Noisy"), (True, DENOISED, "Noisy + denoising")):

        curves = np.array([
            [np.nan if v is None else 100 * v for v in condition_curve(results, d, level, denoised)]
            for d in distances
        ], dtype=float)

        if np.all(np.isnan(curves)):
            continue

        with np.errstate(all="ignore"):
            mean = np.nanmean(curves, axis=0)
            low = np.nanmin(curves, axis=0)
            high = np.nanmax(curves, axis=0)

        ax.fill_between(x, low, high, color=colour, alpha=0.12, linewidth=0)
        ax.plot(x, mean, color=colour, linewidth=2, marker="o", markersize=6,
                label=f"{label} (mean; band = min–max over distances)")

        ax.annotate(label, (x[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)

    ax.axhline(25, color=MUTED, linestyle=":", linewidth=1)

    style(ax)
    ax.set_xticks(x, snr_axis())
    ax.set_xlabel("Test condition (added AWGN, SNR relative to target signal)", color=INK_2, fontsize=9)
    ax.set_ylim(0, 102)
    ax.set_xlim(-0.2, len(x) - 0.4)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title(f"Accuracy vs SNR, all distances ({level}-level)", color=INK, fontsize=10, loc="left")

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_snr_per_distance(results, distances, level, path):

    cols = 5
    rows = int(np.ceil(len(distances) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.6 * rows), sharex=True,
                             sharey=True, facecolor=SURFACE, squeeze=False)

    x = np.arange(len(snr_axis()))

    for ax, d in zip(axes.flat, distances):

        for denoised, colour, label in ((False, NOISY, "Noisy"), (True, DENOISED, "Denoised")):
            plot_line(ax, x, [None if v is None else 100 * v
                              for v in condition_curve(results, d, level, denoised)],
                      color=colour, marker="o", label=label)

        style(ax, ylabel="")
        ax.set_title(d, color=INK, fontsize=9, loc="left")
        ax.set_xticks(x, snr_axis(), fontsize=7)
        ax.set_ylim(0, 102)

    for ax in axes.flat[len(distances):]:
        ax.axis("off")

    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy (%)", color=INK_2, fontsize=8)

    axes.flat[0].legend(fontsize=7, frameon=False, loc="lower left")

    fig.suptitle(f"Accuracy vs SNR with and without denoising, per distance ({level}-level)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_final_heatmap(results, distances, level, path):

    items = conditions()

    data = np.array([[np.nan if (v := value(results, d, key, level)) is None else 100 * v
                      for key, _, _, _ in items] for d in distances], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(distances) + 1.6), facecolor=SURFACE)

    image = ax.imshow(np.ma.masked_invalid(data), cmap=HEAT, vmin=25, vmax=100, aspect="auto")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = "n/a" if np.isnan(data[i, j]) else f"{data[i, j]:.1f}"
            dark = not np.isnan(data[i, j]) and data[i, j] > 70
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if dark else INK)

    ax.set_xticks(range(len(items)), [t.replace(" + ", "\n+ ") for _, t, _, _ in items], fontsize=8)
    ax.set_yticks(range(len(distances)), distances, fontsize=8)
    ax.tick_params(colors=INK_2, length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(image, ax=ax, fraction=0.03)
    bar.set_label("Accuracy (%)", color=INK_2, fontsize=8)
    bar.ax.tick_params(labelsize=7, colors=INK_2)

    ax.set_title(f"Distance × condition accuracy ({level}-level)", color=INK, fontsize=10, loc="left")

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_clean_confusions(results, distances, level, path):

    done = [d for d in distances if results[d]["clean"] is not None]

    if not done:
        return False

    cols = 5
    rows = int(np.ceil(len(done) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.1 * rows),
                             facecolor=SURFACE, squeeze=False)

    short = [c.replace(" Pro", "P").replace("Matrice ", "M").replace("Inspire ", "I")
             .replace("Mavic ", "Mv").replace("Phantom ", "Ph") for c in CLASSES]

    for ax, d in zip(axes.flat, done):

        matrix = np.array(results[d]["clean"][LEVELS[level]]["confusion_matrix"])

        ax.imshow(matrix, cmap=HEAT, vmin=0)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] > matrix.max() / 2 else INK)

        ax.set_xticks(range(len(CLASSES)), short, fontsize=7)
        ax.set_yticks(range(len(CLASSES)), short, fontsize=7)
        ax.set_title(d, color=INK, fontsize=9, loc="left")

    for ax in axes.flat[len(done):]:
        ax.axis("off")

    fig.suptitle(f"Clean test confusion matrices ({level}-level; rows true, columns predicted)\n"
                 + ", ".join(f"{s} = {c}" for s, c in zip(short, CLASSES)),
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    return True


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

# Optional models, each in its own report section and results folder
OTHER_MODELS = [
    ("all_distances", "Optional: one model trained on all distances",
     "One UIB-SSM trained on the train files of all distances together (validation "
     "files of all distances for early stopping), tested on each distance's own test "
     "files. Compared with the per-distance clean model on the same test files.",
     "M"),
    ("noise_aug", "Optional: noise-augmented training (separate model per distance)",
     f"Training files only get fresh AWGN every epoch ({NOISE_AUG}); validation and "
     "test files are never augmented. The clean-model tables above are unchanged.",
     "N"),
]


def model_section(clean, augmented, distances, run_name, title, note, prefix):

    label = {"all_distances": "all-distance", "noise_aug": "noise-aug"}[run_name]

    md = [f"# {title}", "", note, ""]

    for level in ("segment", "file"):

        tag = "" if level == "segment" else "_files"

        header, rows = final_table(augmented, distances, level)
        write_csv(os.path.join(SUMMARY_DIR, f"{prefix}_final_{run_name}{tag}.csv"), header, rows)
        md += [f"## {label} model, final table ({level}-level accuracy)", "",
               md_table(header, rows), ""]

        header = ["Distance", "Clean: per-distance", f"Clean: {label}"]

        for s in SNR_LEVELS:
            header += [f"{s:g} dB per-distance", f"{s:g} dB per-distance + den",
                       f"{s:g} dB {label}", f"{s:g} dB {label} + den"]

        rows = []

        for d in distances:
            row = [d, pct(value(clean, d, "clean", level)), pct(value(augmented, d, "clean", level))]
            for s in SNR_LEVELS:
                row += [pct(value(clean, d, f"{s:g}dB", level)),
                        pct(value(clean, d, f"{s:g}dB+den", level)),
                        pct(value(augmented, d, f"{s:g}dB", level)),
                        pct(value(augmented, d, f"{s:g}dB+den", level))]
            rows.append(row)

        write_csv(os.path.join(SUMMARY_DIR, f"{prefix}_model_comparison{tag}.csv"), header, rows)
        md += [f"## Per-distance clean model vs {label} model ({level}-level accuracy)", "",
               md_table(header, rows), ""]

        plot_model_comparison(clean, augmented, distances, level, label,
                              os.path.join(SUMMARY_DIR, f"accuracy_vs_snr_{run_name}{tag}.png"))

    return md


def plot_model_comparison(clean, augmented, distances, level, label, path):

    x = np.arange(len(snr_axis()))

    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=SURFACE)

    # Fixed categorical order: slot 1 blue, 2 orange, 3 aqua, 4 yellow
    curves = [
        (clean, False, NOISY, "Per-distance model, noisy"),
        (clean, True, DENOISED, "Per-distance model, noisy + denoising"),
        (augmented, False, "#1baf7a", f"{label.capitalize()} model, noisy"),
        (augmented, True, "#eda100", f"{label.capitalize()} model, noisy + denoising"),
    ]

    for results, denoised, colour, label in curves:

        data = np.array([
            [np.nan if v is None else 100 * v for v in condition_curve(results, d, level, denoised)]
            for d in distances
        ], dtype=float)

        # Only distances where both models have results, so means compare like with like
        if np.all(np.isnan(data)):
            continue

        with np.errstate(all="ignore"):
            mean = np.nanmean(data, axis=0)

        ax.plot(x, mean, color=colour, linewidth=2, marker="o", markersize=6, label=label)

    ax.axhline(25, color=MUTED, linestyle=":", linewidth=1)

    style(ax)
    ax.set_xticks(x, snr_axis())
    ax.set_ylim(0, 102)
    ax.set_xlabel("Test condition", color=INK_2, fontsize=9)
    ax.set_title(f"Per-distance vs {label} model, mean over distances ({level}-level)",
                 color=INK, fontsize=10, loc="left")

    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, frameon=False, loc="lower left")

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():

    parser = argparse.ArgumentParser(description="Final tables and graphs from saved metrics.")
    parser.parse_args()

    distances = sorted(RadarDataLoader().selected_distances(), key=distance_metres)

    results = collect(distances)

    os.makedirs(SUMMARY_DIR, exist_ok=True)

    md = [
        "# UIB-SSM multi-distance results",
        "",
        "All values are measured test results (percent) read from "
        "`results/<distance>/clean/<condition>/metrics.json`; `n/a` = not run yet.",
        "Model: clean-trained UIB-SSM, one per distance. Split: whole files in recording "
        "order, 70 % train / 15 % validation / 15 % test per class.",
        f"Noise: complex AWGN at {', '.join(f'{s:g}' for s in SNR_LEVELS)} dB, SNR relative to "
        "the target (micro-Doppler) power. Denoiser: "
        f"`{DENOISER}` applied to the same noisy signal.",
        "",
    ]

    header, rows = dataset_table()
    write_csv(os.path.join(SUMMARY_DIR, "A_dataset_summary.csv"), header, rows)
    md += ["## A. Dataset summary", "", md_table(header, rows), ""]

    for level in ("segment", "file"):

        tag = "" if level == "segment" else "_files"
        title = "segment-level" if level == "segment" else "file-level (mean of segment probabilities per file)"

        md += [f"# Results — {title}", ""]

        for name, label, builder in (
            ("B_clean", "B. Clean performance", clean_table),
            ("C_noise", "C. Noise robustness (accuracy)", noise_table),
            ("D_denoising", "D. Denoising comparison (accuracy; Δ = denoised − noisy, points)", denoise_table),
            ("E_final", "E. Final combined table (accuracy)", final_table),
        ):
            header, rows = builder(results, distances, level)
            write_csv(os.path.join(SUMMARY_DIR, f"{name}{tag}.csv"), header, rows)
            md += [f"## {label}", "", md_table(header, rows), ""]

        header, rows, clean_cls, noisy_cls, den_cls = per_class_table(results, distances, level)
        write_csv(os.path.join(SUMMARY_DIR, f"per_class_recall{tag}.csv"), header, rows)
        md += ["## Per-class recall (mean over distances)", "", md_table(header, rows), ""]

        md += ["## Observations (computed from the values above)", ""]
        md += [f"- {line}" for line in
               interpretation(results, distances, level, clean_cls, noisy_cls, den_cls)]
        md += [""]

        plot_accuracy_vs_distance(results, distances, level,
                                  os.path.join(SUMMARY_DIR, f"accuracy_vs_distance{tag}.png"))
        plot_accuracy_vs_snr(results, distances, level,
                             os.path.join(SUMMARY_DIR, f"accuracy_vs_snr{tag}.png"))
        plot_snr_per_distance(results, distances, level,
                              os.path.join(SUMMARY_DIR, f"accuracy_vs_snr_per_distance{tag}.png"))
        plot_final_heatmap(results, distances, level,
                           os.path.join(SUMMARY_DIR, f"final_table_heatmap{tag}.png"))
        plot_clean_confusions(results, distances, level,
                              os.path.join(SUMMARY_DIR, f"clean_confusion_matrices{tag}.png"))

    # Optional noise-augmented model: separate tables, never mixed into
    # the clean-model tables above
    for run_name, title, note, prefix in OTHER_MODELS:

        other = collect(distances, run_name)

        if any(other[d][k] is not None for d in distances for k, *_ in conditions()):
            md += model_section(results, other, distances, run_name, title, note, prefix)

    report_path = os.path.join(SUMMARY_DIR, "final_results.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    finished = sum(results[d][k] is not None for d in distances for k, *_ in conditions())

    print(f"Conditions with results: {finished} of {len(distances) * len(conditions())}")
    print("Report:", relative(report_path))
    print("Tables and graphs:", relative(SUMMARY_DIR))


if __name__ == "__main__":
    main()
