"""Merge every results/runs/*.json into results/summary.csv plus a test-set pivot table.

Usage:
    python -m scripts.collect_results
    python -m scripts.collect_results --metric accuracy
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import RESULTS_DIR, RUNS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="f1_macro")
    args = parser.parse_args()

    files = sorted(RUNS_DIR.glob("*.json"))
    if not files:
        print(f"No runs found in {RUNS_DIR}")
        return
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    df = pd.json_normalize(rows, sep=".")
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    test = df[(df["split"] == "test") & (df["train_dataset"] == df["dataset"])]
    if args.metric in test and not test.empty:
        # latest run per (model, dataset)
        latest = test.sort_values("timestamp").groupby(["model", "dataset"]).tail(1)
        pivot = latest.pivot(index="model", columns="dataset", values=args.metric).round(3)
        pivot.to_csv(RESULTS_DIR / f"pivot_test_{args.metric}.csv")
        print(pivot.to_string())
    print(f"\n{len(df)} runs -> {RESULTS_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
