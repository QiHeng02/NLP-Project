"""Traditional ML runs (Role B): NB / LR / SVM on TF-IDF, BoW and GloVe-averaged features.

Every (features, model, dataset) combination is trained on train, scored on val (and on test
with --test), and logged to results/runs/ via log_run. Tune on val; use --test once per final config.

Usage:
    python -m scripts.run_classical --author "Sharmayn"                       # all combos, all 4 tasks, val
    python -m scripts.run_classical --models lr --features tfidf --datasets liar
    python -m scripts.run_classical --models svm --features tfidf --datasets liar --C 0.1 --notes "tune: C"
    python -m scripts.run_classical --models svm --features tfidf --datasets liar --stopwords --lemmatize
    python -m scripts.run_classical --models svm --features tfidf --datasets covid --test --save --confusion

Datasets: isot, liar (real/fake), liar6 (6 labels), covid, or any other folder in data/processed/
(e.g. isot_nostyle, isot_title).
"""
from __future__ import annotations

import argparse
import time

from src.config import DATA_PROCESSED, LABEL_FAKE, LIAR_LABELS_6, MODELS_DIR
from src.data.loaders import load_splits
from src.evaluation.metrics import compute_metrics, log_run, plot_confusion_matrix
from src.models.classical import FEATURES, MODELS, build_classical, model_name

TASKS = ("isot", "liar", "liar6", "covid")


def _task(tag: str) -> tuple[str, str, str]:
    """dataset tag -> (processed folder, label column, task type)."""
    if tag == "liar6":
        return "liar", "label_6", "multiclass"
    return tag, "label", "binary"


def _evaluate(model, X, y, *, task: str) -> tuple[dict, object]:
    pred = model.predict(X)
    score = None
    if task == "binary":
        score = model.predict_proba(X)[:, list(model.classes_).index(LABEL_FAKE)]
    return compute_metrics(y, pred, score, task=task), pred


def run_one(
    model: str, features: str, tag: str, splits: dict, *, cfg: dict, test: bool = False,
    save: bool = False, confusion: bool = False, author: str | None = None, notes: str = "",
) -> dict:
    """Train one combination, log val (and test) results, return the val metrics."""
    _, label_col, task = _task(tag)
    pipe = build_classical(model, features, **cfg)
    name = model_name(model, features)

    t0 = time.perf_counter()
    pipe.fit(splits["train"]["text"].tolist(), splits["train"][label_col].tolist())
    runtime = round(time.perf_counter() - t0, 2)

    params = {"model": model, "features": features, **cfg}
    params["ngram_range"] = list(cfg["ngram_range"])
    if features == "glove":
        for k in ("ngram_range", "max_features", "min_df"):
            params.pop(k)
        params["glove_coverage"] = round(pipe.named_steps["features"].coverage(splits["val"]["text"]), 3)
    if model != "nb":
        params.pop("alpha")
    else:
        params.pop("C")

    val_metrics = None
    for split in ("val", "test") if test else ("val",):
        X, y = splits[split]["text"].tolist(), splits[split][label_col]
        t1 = time.perf_counter()
        m, pred = _evaluate(pipe, X, y, task=task)
        params["latency_ms_per_input"] = round(1000 * (time.perf_counter() - t1) / max(len(X), 1), 3)
        log_run(model=name, dataset=tag, split=split, metrics=m, params=params,
                runtime_s=runtime, author=author, notes=notes)
        print(f"[{tag:6s}] {name:10s} {split:4s} acc={m['accuracy']:.3f} f1_macro={m['f1_macro']:.3f}"
              f"  ({runtime:.1f}s train)")
        if split == "val":
            val_metrics = m
        if split == "test" and confusion:
            labels = LIAR_LABELS_6 if task == "multiclass" else [0, 1]
            names = LIAR_LABELS_6 if task == "multiclass" else ["real", "fake"]
            out = plot_confusion_matrix(y, pred, labels, names, title=f"{name} on {tag} (test)",
                                        filename=f"classical_cm_{name}_{tag}.png")
            print(f"          confusion matrix -> {out}")

    if save:
        import joblib

        out_dir = MODELS_DIR / "classical" / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, out_dir / f"{name}.joblib")
        print(f"          saved -> {out_dir / f'{name}.joblib'}")
    return val_metrics


def load_task_splits(tag: str, cache: dict) -> dict | None:
    folder, _, _ = _task(tag)
    if not (DATA_PROCESSED / folder / "train.csv").exists():
        print(f"[{tag}] skipped: run `python -m scripts.prepare_data` first")
        return None
    if folder not in cache:
        cache[folder] = load_splits(folder)
    return cache[folder]


def combos(models, features):
    """All valid (model, features) pairs; NB is skipped for GloVe (negative features)."""
    return [(m, f) for f in features for m in models if not (m == "nb" and f == "glove")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument("--features", nargs="+", default=list(FEATURES), choices=FEATURES)
    p.add_argument("--datasets", nargs="+", default=list(TASKS))
    p.add_argument("--ngram", nargs=2, type=int, default=(1, 2), metavar=("MIN", "MAX"))
    p.add_argument("--max-features", type=int, default=50_000, help="0 = no limit")
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1.0, help="NB smoothing")
    p.add_argument("--stopwords", action="store_true", help="remove English stop words")
    p.add_argument("--lemmatize", action="store_true", help="WordNet lemmatisation")
    p.add_argument("--test", action="store_true", help="also score the test set (final configs only)")
    p.add_argument("--save", action="store_true", help="save the fitted pipeline to models/classical/")
    p.add_argument("--confusion", action="store_true", help="save a test confusion matrix PNG")
    p.add_argument("--author", default="Sharmayn")
    p.add_argument("--notes", default="")
    args = p.parse_args()

    cfg = dict(ngram_range=tuple(args.ngram), max_features=args.max_features or None,
               min_df=args.min_df, C=args.C, alpha=args.alpha,
               stopwords=args.stopwords, lemmatize=args.lemmatize)
    cache: dict = {}
    for tag in args.datasets:
        splits = load_task_splits(tag, cache)
        if splits is None:
            continue
        for model, features in combos(args.models, args.features):
            run_one(model, features, tag, splits, cfg=cfg, test=args.test, save=args.save,
                    confusion=args.confusion, author=args.author, notes=args.notes)


if __name__ == "__main__":
    main()
