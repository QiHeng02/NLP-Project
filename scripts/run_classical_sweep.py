"""Role B tuning: change one setting at a time from the defaults, val only.

For each task, the two best (model, features) pairs from Phase 1 (`--notes "phase1: defaults"`)
are swept over:
    ngram     (1,1) (1,2)* (1,3)                  - tfidf / bow only
    max_feat  10k  50k*  no limit                  - tfidf / bow only
    C         0.1 0.3 1* 3 10   (lr / svm)
    alpha     0.1 0.3 1* 3      (nb)
    prep      stopwords, lemmatize, stopwords + lemmatize
(* = default, already logged in Phase 1, so not re-run.)
Each run is logged with notes "sens: <axis>", so Role A can leave them out of the main table.

Usage:
    python -m scripts.run_classical_sweep
    python -m scripts.run_classical_sweep --datasets liar covid
    python -m scripts.run_classical_sweep --dry-run
"""
from __future__ import annotations

import argparse

from scripts.run_classical import TASKS, load_task_splits, run_one

# Top-2 val macro-F1 per task from Phase 1 (results/runs/*_val.json, notes "phase1: defaults")
TOP2 = {
    "isot": [("svm", "tfidf"), ("lr", "bow")],
    "liar": [("nb", "bow"), ("lr", "tfidf")],
    "liar6": [("lr", "bow"), ("lr", "tfidf")],
    "covid": [("svm", "tfidf"), ("svm", "bow")],
}
DEFAULT = dict(ngram_range=(1, 2), max_features=50_000, min_df=2, C=1.0, alpha=1.0,
               stopwords=False, lemmatize=False)


def variants(model: str, features: str):
    """(axis, overrides) pairs, one setting changed at a time."""
    out = []
    if features != "glove":
        out += [("ngram", {"ngram_range": r}) for r in [(1, 1), (1, 3)]]
        out += [("max_features", {"max_features": n}) for n in [10_000, None]]
    if model == "nb":
        out += [("alpha", {"alpha": a}) for a in [0.1, 0.3, 3.0]]
    else:
        out += [("C", {"C": c}) for c in [0.1, 0.3, 3.0, 10.0]]
    out += [("prep", {"stopwords": True}), ("prep", {"lemmatize": True}),
            ("prep", {"stopwords": True, "lemmatize": True})]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(TASKS))
    p.add_argument("--author", default="Sharmayn")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cache: dict = {}
    best: dict = {}
    for tag in args.datasets:
        splits = None if args.dry_run else load_task_splits(tag, cache)
        if splits is None and not args.dry_run:
            continue
        for model, features in TOP2[tag]:
            for axis, over in variants(model, features):
                cfg = {**DEFAULT, **over}
                label = ", ".join(f"{k}={v}" for k, v in over.items())
                if args.dry_run:
                    print(f"[{tag}] {features}-{model}  sens: {axis}  ({label})")
                    continue
                m = run_one(model, features, tag, splits, cfg=cfg, author=args.author,
                            notes=f"sens: {axis} ({label})")
                key = (tag, f"{features}-{model}")
                if key not in best or m["f1_macro"] > best[key][0]:
                    best[key] = (m["f1_macro"], label)

    if best:
        print("\nBest sweep variant per (task, model)  [compare with the phase1 default]")
        for (tag, name), (f1, label) in sorted(best.items()):
            print(f"  {tag:6s} {name:10s} f1_macro={f1:.3f}  ({label})")


if __name__ == "__main__":
    main()
