"""Download, clean and split all datasets into data/processed/<dataset>/{train,val,test}.csv.

Usage (from the repo root):
    python -m scripts.prepare_data                 # all three datasets
    python -m scripts.prepare_data --datasets liar covid
    python -m scripts.prepare_data --isot-include-title --isot-strip-style   # ablations

The output is deterministic (fixed seed), so every member can regenerate
identical splits locally instead of passing data files around.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import DATA_PROCESSED, DATASETS, SPLITS
from src.data.cleaning import clean_isot, normalize_whitespace
from src.data.download import DOWNLOADERS
from src.data.loaders import load_covid_raw, load_isot_raw, load_liar_raw
from src.data.splits import stratified_split


def _save(df: pd.DataFrame, name: str, report: dict) -> None:
    out = DATA_PROCESSED / name
    out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        part = df[df["split"] == split].drop(columns="split")
        part.to_csv(out / f"{split}.csv", index=False)
        report.setdefault("split_sizes", {})[split] = int(len(part))
        report.setdefault("split_label_counts", {})[split] = {
            int(k): int(v) for k, v in part["label"].value_counts().sort_index().items()
        }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[{name}] saved -> {out}  sizes={report['split_sizes']}")


def _basic_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """LIAR/COVID keep their official splits; only drop empty texts."""
    report = {"rows_raw": int(len(df))}
    df = df.assign(text=df["text"].map(normalize_whitespace))
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    report["rows_clean"] = int(len(df))
    return df, report


def prepare(dataset: str, *, include_title: bool, strip_style: bool) -> None:
    raw_dir = DOWNLOADERS[dataset]()
    if raw_dir is None:  # ISOT manual download still pending
        return
    if dataset == "isot":
        clean, report = clean_isot(
            load_isot_raw(raw_dir), include_title=include_title, strip_style=strip_style
        )
        report["options"] = {"include_title": include_title, "strip_style": strip_style}
        _save(stratified_split(clean), "isot", report)
    elif dataset == "liar":
        df, report = _basic_clean(load_liar_raw(raw_dir))
        report["binary_mapping"] = "fake = pants-fire, false, barely-true"
        _save(df, "liar", report)
    elif dataset == "covid":
        df, report = _basic_clean(load_covid_raw(raw_dir))
        _save(df, "covid", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--isot-include-title", action="store_true")
    parser.add_argument("--isot-strip-style", action="store_true")
    args = parser.parse_args()
    for name in args.datasets:
        prepare(name, include_title=args.isot_include_title, strip_style=args.isot_strip_style)


if __name__ == "__main__":
    main()
