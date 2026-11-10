"""Generate reproducible LIME and SHAP figures for saved binary models.

Uses a fixed validation example per dataset. Select a row with --row after
inspecting its text; do not select examples by looking at test labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import FIGURES_DIR, MODELS_DIR
from src.data.loaders import load_split
from src.explain.words import lime_words, shap_words


def plot_words(words, values, title: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = sorted(zip(words, values), key=lambda x: abs(x[1]), reverse=True)[:12][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh([x[0] for x in pairs], [x[1] for x in pairs],
            color=["#cf5555" if x[1] > 0 else "#4a9e68" for x in pairs])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution toward P(fake) (red) or P(real) (green)")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


def get_model(dataset: str, family: str):
    if family == "distilbert":
        from src.models.transformers_ft import TransformerClassifier
        path = MODELS_DIR / "transformers" / dataset / "distilbert"
        if not (path / "wrapper.json").exists():
            raise FileNotFoundError(path / "wrapper.json")
        return TransformerClassifier.load(path, device="cpu")
    import joblib
    name = "bow-nb" if dataset == "liar" else "tfidf-svm"
    path = MODELS_DIR / "classical" / dataset / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)  # trusted team-supplied checkpoint


def run(dataset, family, row, method, lime_samples, shap_evals):
    frame = load_split(dataset, "val")
    text = str(frame.iloc[row]["text"])
    model = get_model(dataset, family)
    stem = f"{dataset}_{family}_val{row}"
    if method in ("lime", "both"):
        weights = lime_words(model, text, num_samples=lime_samples)
        plot_words([word for word, _ in weights], [value for _, value in weights],
                   f"LIME: {dataset} / {family} (validation row {row})",
                   FIGURES_DIR / f"xai_lime_{stem}.png")
    if method in ("shap", "both"):
        explanation = shap_words(model, text, max_evals=shap_evals)
        words = np.asarray(explanation.data[0]).reshape(-1)
        values = np.asarray(explanation.values[0]).reshape(-1)
        plot_words(words, values, f"SHAP: {dataset} / {family} (validation row {row})",
                   FIGURES_DIR / f"xai_shap_{stem}.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=["isot", "liar", "covid"],
                   default=["isot", "liar", "covid"])
    p.add_argument("--families", nargs="+", choices=["classical", "distilbert"],
                   default=["classical", "distilbert"])
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--method", choices=["lime", "shap", "both"], default="both")
    p.add_argument("--lime-samples", type=int, default=500)
    p.add_argument("--shap-evals", type=int, default=120)
    args = p.parse_args()
    for dataset in args.datasets:
        for family in args.families:
            run(dataset, family, args.row, args.method, args.lime_samples, args.shap_evals)


if __name__ == "__main__":
    main()
