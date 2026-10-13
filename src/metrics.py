import json
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from config import CLASSES


@torch.no_grad()
def predict_probabilities(model, dataset, batch_size=256):

    # Softmax probabilities for every segment, in dataset order
    model.eval()

    probs = [
        torch.softmax(model(inputs), dim=1)
        for inputs, _ in DataLoader(dataset, batch_size=batch_size)
    ]

    return torch.cat(probs).numpy()


def file_probabilities(probs, file_index, num_files):

    # Mean of the segment probabilities of each file (exact per file,
    # no assumption about the number of segments per file)
    file_index = np.asarray(file_index)

    sums = np.zeros((num_files, probs.shape[1]))
    np.add.at(sums, file_index, probs)

    counts = np.bincount(file_index, minlength=num_files)[:, None]

    return sums / counts


def classification_metrics(y_true, y_pred):

    labels = list(range(len(CLASSES)))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=CLASSES,
        output_dict=True, zero_division=0,
    )

    return {
        "count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": int(report[name]["support"]),
            }
            for name in CLASSES
        },
    }


def evaluate_model(model, dataset):

    # Segment-level and file-level metrics for one dataset
    probs = predict_probabilities(model, dataset)

    segment = classification_metrics(dataset.labels.numpy(), probs.argmax(axis=1))

    file_probs = file_probabilities(probs, dataset.file_index.numpy(), len(dataset.records))

    file = classification_metrics(dataset.file_labels(), file_probs.argmax(axis=1))

    return {"segment_level": segment, "file_level": file}


def summary_line(metrics):

    s, f = metrics["segment_level"], metrics["file_level"]

    return (
        f"segments {s['accuracy']:.4f} (F1 {s['f1_macro']:.4f}) | "
        f"files {f['accuracy']:.4f} (F1 {f['f1_macro']:.4f})"
    )


def plot_confusion_matrix(matrix, title, path):

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(matrix, cmap="Blues", vmin=0)

    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, matrix[i, j],
                ha="center", va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_results(results, folder, title):

    # metrics.json + segment / file confusion matrices for one condition
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    for level in ("segment_level", "file_level"):
        plot_confusion_matrix(
            results[level]["confusion_matrix"],
            f"{title} ({level.split('_')[0]}s)",
            os.path.join(folder, f"confusion_matrix_{level.split('_')[0]}s.png"),
        )
