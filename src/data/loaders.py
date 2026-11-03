"""Load raw files into one shared schema, and load the processed splits.

Shared schema (every dataset): id, text, label (0 = real, 1 = fake), split.
LIAR additionally keeps `label_6` (string, see config.LIAR_LABELS_6).
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_PROCESSED,
    DATA_RAW,
    LABEL_FAKE,
    LABEL_REAL,
    LIAR_FAKE_LABELS,
    LIAR_LABELS_6,
    SPLITS,
)

LIAR_COLUMNS = [
    "id", "label_6", "statement", "subject", "speaker", "job_title", "state", "party",
    "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts",
    "pants_on_fire_counts", "context",
]


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m scripts.prepare_data")
    return path


# ---------------------------------------------------------------- raw loaders
def load_isot_raw(raw_dir: Path = DATA_RAW / "isot") -> pd.DataFrame:
    """Fake.csv + True.csv -> id, title, text, subject, date, label. No cleaning applied."""
    frames = []
    for fname, label in (("True.csv", LABEL_REAL), ("Fake.csv", LABEL_FAKE)):
        df = pd.read_csv(_require(raw_dir / fname))
        df["label"] = label
        df["id"] = [f"isot_{'real' if label == LABEL_REAL else 'fake'}_{i}" for i in range(len(df))]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)[["id", "title", "text", "subject", "date", "label"]]


def liar_to_binary(label_6: str) -> int:
    return LABEL_FAKE if label_6 in LIAR_FAKE_LABELS else LABEL_REAL


def load_liar_raw(raw_dir: Path = DATA_RAW / "liar") -> pd.DataFrame:
    """Official TSV splits -> id, text, label, label_6, split (statement text only)."""
    frames = []
    for fname, split in (("train.tsv", "train"), ("valid.tsv", "val"), ("test.tsv", "test")):
        # QUOTE_NONE: statements contain unbalanced quote characters
        df = pd.read_csv(
            _require(raw_dir / fname), sep="\t", header=None, names=LIAR_COLUMNS,
            quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False,
        )
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    unknown = set(df["label_6"]) - set(LIAR_LABELS_6)
    if unknown:
        raise ValueError(f"Unexpected LIAR labels: {unknown}")
    df["text"] = df["statement"].map(str.strip)
    df["label"] = df["label_6"].map(liar_to_binary)
    df["id"] = "liar_" + df["id"].str.replace(".json", "", regex=False)
    return df[["id", "text", "label", "label_6", "split"]]


def load_covid_raw(raw_dir: Path = DATA_RAW / "covid") -> pd.DataFrame:
    """Constraint@AAAI-2021 official splits -> id, text, label, split."""
    frames = []
    for split in SPLITS:
        df = pd.read_csv(_require(raw_dir / f"{split}.csv"))
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    label_map = {"real": LABEL_REAL, "fake": LABEL_FAKE}
    df["label"] = df["label"].str.strip().str.lower().map(label_map)
    if df["label"].isna().any():
        raise ValueError("Unexpected COVID labels")
    df["text"] = df["tweet"].astype(str).str.strip()
    df["id"] = "covid_" + df["split"] + "_" + df["id"].astype(str)
    return df[["id", "text", "label", "split"]]


# ---------------------------------------------------------------- processed splits
def load_split(dataset: str, split: str, processed_dir: Path = DATA_PROCESSED) -> pd.DataFrame:
    """Load one processed split, e.g. load_split('liar', 'train')."""
    return pd.read_csv(_require(processed_dir / dataset / f"{split}.csv"), keep_default_na=False)


def load_splits(dataset: str, processed_dir: Path = DATA_PROCESSED) -> dict[str, pd.DataFrame]:
    """Load train/val/test for a dataset as a dict of DataFrames."""
    return {s: load_split(dataset, s, processed_dir) for s in SPLITS}
