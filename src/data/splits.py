"""Train/val/test splitting for datasets without an official split (ISOT)."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ISOT_TEST_SIZE, ISOT_VAL_SIZE, SEED


def stratified_split(
    df: pd.DataFrame,
    *,
    val_size: float = ISOT_VAL_SIZE,
    test_size: float = ISOT_TEST_SIZE,
    seed: int = SEED,
    label_col: str = "label",
) -> pd.DataFrame:
    """Return a copy of df with a `split` column (train/val/test), stratified by label.

    Deduplicate BEFORE calling this, or copies of the same text land in both train and test.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=seed
    )
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val[label_col], random_state=seed
    )
    return pd.concat(
        [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")],
        ignore_index=True,
    )
