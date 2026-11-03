"""Shared metrics and result logging.

Every experiment should call `compute_metrics` and `log_run`, so all results land
in results/runs/ in the same format and `python -m scripts.collect_results`
can build the comparison tables for the slides.

Each run is written as its own JSON file (not appended to one CSV) so six people
can commit results without merge conflicts.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.config import FIGURES_DIR, LABEL_FAKE, RUNS_DIR


def compute_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    y_score: Sequence[float] | None = None,
    *,
    task: str = "binary",
) -> dict:
    """Standard metric set.

    Args:
        y_true, y_pred: gold and predicted labels.
        y_score: probability of the FAKE class (binary only), enables ROC-AUC.
        task: "binary" (real/fake) or "multiclass" (LIAR 6-way).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    metrics = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f),
    }
    if task == "binary":
        metrics["f1_fake"] = float(f1_score(y_true, y_pred, pos_label=LABEL_FAKE, zero_division=0))
        if y_score is not None and len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, np.asarray(y_score)))
    else:
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    return metrics


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


def log_run(
    *,
    model: str,
    dataset: str,
    split: str,
    metrics: dict,
    train_dataset: str | None = None,
    params: dict | None = None,
    runtime_s: float | None = None,
    author: str | None = None,
    notes: str = "",
    runs_dir: Path = RUNS_DIR,
) -> Path:
    """Save one evaluation result as results/runs/<timestamp>_<model>_<train>-to-<test>_<split>.json.

    Args:
        model: short name, e.g. "tfidf-lr", "bilstm", "distilbert".
        dataset: dataset evaluated on, e.g. "liar" or "liar6".
        split: "val" or "test".
        train_dataset: dataset trained on (defaults to `dataset`; set it for cross-dataset tests).
        params: hyperparameters worth reporting.
        runtime_s: training (or inference) time in seconds.
        author: your name, so contributions are traceable for the group reflection.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = train_dataset or dataset
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    record = {
        "timestamp": stamp,
        "model": model,
        "train_dataset": train_dataset,
        "dataset": dataset,
        "split": split,
        "author": author,
        "runtime_s": runtime_s,
        "params": params or {},
        "notes": notes,
        **metrics,
    }
    path = runs_dir / f"{stamp}_{_slug(model)}_{_slug(train_dataset)}-to-{_slug(dataset)}_{split}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def plot_confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence,
    display_labels: Sequence[str],
    title: str,
    filename: str,
    figures_dir: Path = FIGURES_DIR,
) -> Path:
    """Save a confusion-matrix PNG to results/figures/ (for slides)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    fig, ax = plt.subplots(figsize=(4 + 0.4 * len(labels), 3.5 + 0.4 * len(labels)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), display_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), display_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
