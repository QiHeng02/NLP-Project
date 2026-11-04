"""Sanity check: TF-IDF + Logistic Regression on every prepared dataset.

Confirms the data pipeline works end to end and gives the first reference numbers.
With --leakage it also reproduces the ISOT leakage evidence for the slides:
  * metadata-only model (uses ONLY the `subject` column)
  * TF-IDF on raw, uncleaned ISOT with a naive split
  * TF-IDF on the cleaned ISOT split

Usage:
    python -m scripts.run_sanity_baseline
    python -m scripts.run_sanity_baseline --datasets liar covid --author "Your Name"
    python -m scripts.run_sanity_baseline --datasets isot --leakage
    python -m scripts.run_sanity_baseline --datasets isot_nostyle   # any folder in data/processed/
"""
from __future__ import annotations

import argparse
import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import DATA_PROCESSED, DATA_RAW, DATASETS, LABEL_FAKE, SEED
from src.data.loaders import load_isot_raw, load_splits
from src.evaluation.metrics import compute_metrics, log_run
from src.models.baseline import build_tfidf_lr

MODEL_NAME = "tfidf-lr-sanity"


def _evaluate(model, X, y, *, task: str) -> dict:
    pred = model.predict(X)
    score = None
    if task == "binary":
        score = model.predict_proba(X)[:, list(model.classes_).index(LABEL_FAKE)]
    return compute_metrics(y, pred, score, task=task)


def run_dataset(name: str, author: str | None) -> None:
    if not (DATA_PROCESSED / name / "train.csv").exists():
        print(f"[{name}] skipped: run `python -m scripts.prepare_data` first")
        return
    splits = load_splits(name)
    tasks = [("binary", "label", name)]
    if name == "liar":
        tasks.append(("multiclass", "label_6", "liar6"))
    for task, label_col, tag in tasks:
        model = build_tfidf_lr()
        t0 = time.perf_counter()
        model.fit(splits["train"]["text"], splits["train"][label_col])
        runtime = time.perf_counter() - t0
        for split in ("val", "test"):
            m = _evaluate(model, splits[split]["text"], splits[split][label_col], task=task)
            log_run(model=MODEL_NAME, dataset=tag, split=split, metrics=m,
                    runtime_s=round(runtime, 2), author=author)
            print(f"[{tag:6s}] {split:4s} acc={m['accuracy']:.3f} f1_macro={m['f1_macro']:.3f}")


def run_isot_leakage(author: str | None) -> None:
    raw = load_isot_raw(DATA_RAW / "isot").fillna({"text": "", "subject": ""})
    train, test = train_test_split(raw, test_size=0.2, stratify=raw["label"], random_state=SEED)

    meta = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=1000))
    meta.fit(train[["subject"]], train["label"])
    m = _evaluate(meta, test[["subject"]], test["label"], task="binary")
    log_run(model="subject-only-lr", dataset="isot-raw", split="test", metrics=m, author=author,
            notes="Leakage check: only the subject metadata column is used")
    print(f"[leakage] subject column only      f1_macro={m['f1_macro']:.3f}")

    model = build_tfidf_lr().fit(train["text"], train["label"])
    m = _evaluate(model, test["text"], test["label"], task="binary")
    log_run(model=MODEL_NAME, dataset="isot-raw", split="test", metrics=m, author=author,
            notes="Leakage check: raw text, duplicates kept, naive 80/20 split")
    print(f"[leakage] raw text, naive split     f1_macro={m['f1_macro']:.3f}")
    print("[leakage] compare with the cleaned ISOT result above / in results/summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS),
                        help="folder names in data/processed/ (default: isot liar covid)")
    parser.add_argument("--leakage", action="store_true", help="also run the ISOT leakage checks")
    parser.add_argument("--author", default=None)
    args = parser.parse_args()
    for name in args.datasets:
        run_dataset(name, args.author)
    if args.leakage:
        run_isot_leakage(args.author)


if __name__ == "__main__":
    main()
