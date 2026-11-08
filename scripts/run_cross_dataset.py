"""Cross-dataset test: train on one dataset, test on the others (real vs fake only).

Answers "does a fake-news detector trained on one kind of text work on another?"
(rubric: how the model could be extrapolated). Uses fixed default settings, so
nothing is tuned on the other datasets' test sets.

Each model is trained on: ISOT, LIAR, COVID, and (with --combined) all three together.
Every train set is scored on all three TEST splits. Off-diagonal cells are logged to
results/runs/ with train_dataset set, so the main results table ignores them.

Outputs:
  results/report/cross_dataset_<model>.csv      macro-F1 matrix (rows = trained on, cols = tested on)
  results/figures/cross_dataset_<model>.png     heatmap of the same matrix

Usage (from the repo root):
    python -m scripts.run_cross_dataset --author "Your Name"
    python -m scripts.run_cross_dataset --models tfidf-lr-sanity tfidf-svm --combined --author "Your Name"
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from src.config import DATA_PROCESSED, FIGURES_DIR, LABEL_FAKE, RESULTS_DIR
from src.data.loaders import load_splits
from src.evaluation.metrics import compute_metrics, log_run
from src.models.baseline import build_tfidf_lr
from src.models.classical import build_classical

DATASETS = ["isot", "liar", "covid"]  # binary real/fake only (LIAR-6 has different labels)
NICE = {"isot": "ISOT (articles)", "liar": "LIAR (claims)", "covid": "COVID (posts)", "combined": "All three"}
MODELS = {
    "tfidf-lr-sanity": build_tfidf_lr,                                  # same as the sanity baseline
    "tfidf-svm": lambda: build_classical("svm", "tfidf"),               # Role B defaults
    "bow-nb": lambda: build_classical("nb", "bow"),                     # Role B defaults
}


def evaluate(model, df: pd.DataFrame) -> dict:
    score = model.predict_proba(df["text"])[:, list(model.classes_).index(LABEL_FAKE)]
    return compute_metrics(df["label"], model.predict(df["text"]), score, task="binary")


def plot_matrix(matrix: pd.DataFrame, model_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 0.9 * len(matrix) + 1.6))
    im = ax.imshow(matrix.values, cmap="RdYlGn", vmin=0.3, vmax=1.0)
    ax.set_xticks(range(matrix.shape[1]), [NICE[c] for c in matrix.columns])
    ax.set_yticks(range(matrix.shape[0]), [NICE[r] for r in matrix.index])
    ax.set_xlabel("Tested on")
    ax.set_ylabel("Trained on")
    for i, row in enumerate(matrix.index):
        for j, col in enumerate(matrix.columns):
            v = matrix.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                    fontweight="bold" if row == col else "normal")
    fig.colorbar(im, ax=ax, fraction=0.046, label="macro-F1 (0.5 ≈ guessing)")
    ax.set_title(f"Cross-dataset test: {model_name}\n(bold = trained and tested on the same dataset)")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"cross_dataset_{model_name}.png", dpi=150)
    plt.close(fig)


def run_model(name: str, splits: dict, *, combined: bool, author: str | None) -> pd.DataFrame:
    train_sets = {d: splits[d]["train"] for d in DATASETS}
    if combined:
        train_sets["combined"] = pd.concat([splits[d]["train"] for d in DATASETS], ignore_index=True)

    matrix = pd.DataFrame(index=list(train_sets), columns=DATASETS, dtype=float)
    for train_name, train_df in train_sets.items():
        model = MODELS[name]()
        t0 = time.perf_counter()
        model.fit(train_df["text"], train_df["label"])
        runtime = round(time.perf_counter() - t0, 2)
        for test_name in DATASETS:
            m = evaluate(model, splits[test_name]["test"])
            matrix.loc[train_name, test_name] = m["f1_macro"]
            if train_name != test_name:  # in-domain scores are already logged by the main runs
                log_run(model=name, dataset=test_name, split="test", metrics=m,
                        train_dataset=train_name, runtime_s=runtime, author=author,
                        notes="cross-dataset: default settings, no tuning")
        print(f"[{name}] trained on {train_name:8s} -> "
              + "  ".join(f"{t}={matrix.loc[train_name, t]:.3f}" for t in DATASETS))
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=["tfidf-lr-sanity"])
    parser.add_argument("--combined", action="store_true", help="also train on all three datasets together")
    parser.add_argument("--author", default=None)
    args = parser.parse_args()

    missing = [d for d in DATASETS if not (DATA_PROCESSED / d / "train.csv").exists()]
    if missing:
        raise SystemExit(f"Missing processed data for {missing}: run `python -m scripts.prepare_data` first.")
    splits = {d: load_splits(d) for d in DATASETS}

    out_dir = RESULTS_DIR / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        matrix = run_model(name, splits, combined=args.combined, author=args.author)
        matrix.round(4).to_csv(out_dir / f"cross_dataset_{name}.csv")
        plot_matrix(matrix, name)
        print(f"\n{name}: macro-F1 matrix (rows = trained on, columns = tested on)")
        print(matrix.round(3).to_string())
        print(f"Saved: {out_dir / f'cross_dataset_{name}.csv'} and {FIGURES_DIR / f'cross_dataset_{name}.png'}\n")


if __name__ == "__main__":
    main()
