"""Text cleaning, including the ISOT leakage fixes.

Why ISOT needs special treatment (see PROJECT_PLAN.md, section 3):
  * the `subject` column alone predicts the label perfectly
  * ~99% of real articles contain a "(Reuters)" dateline; fake ones almost never do
  * many fake articles are exact duplicates, so naive splits leak test data
Every step below is counted in a report dict, which feeds the
"before vs after cleaning" slide.
"""
from __future__ import annotations

import re

import pandas as pd

from src.config import ISOT_MIN_CHARS

# "WASHINGTON (Reuters) - ", "SEOUL/BEIJING (Reuters) – ", or a bare "(Reuters) - "
_REUTERS_DATELINE = re.compile(r"^\s*(?:[A-Z][\w.'’/,\- ]{0,80}?\s)?\(Reuters\)\s*[-–—]\s*")
_REUTERS_TOKEN = re.compile(r"\breuters\b", flags=re.IGNORECASE)
# Optional: web-publishing residue that mostly appears in ISOT's fake class.
_STYLE_MARKERS = re.compile(
    r"(featured image (via|by|credit)[^\n]*$)|(\bgetty images\b)|(\bscreenshot\b)",
    flags=re.IGNORECASE | re.MULTILINE,
)
_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_whitespace(text: str) -> str:
    return _WS.sub(" ", str(text)).strip()


def dedup_key(text: str) -> str:
    """Lower-cased, punctuation-free form used to detect duplicate documents."""
    return _NON_ALNUM.sub(" ", str(text).lower()).strip()


def strip_source_markers(text: str) -> str:
    """Remove the leading Reuters dateline and any remaining 'Reuters' tokens."""
    text = _REUTERS_DATELINE.sub("", str(text))
    return _REUTERS_TOKEN.sub(" ", text)


def strip_style_markers(text: str) -> str:
    return _STYLE_MARKERS.sub(" ", str(text))


def clean_isot(
    df: pd.DataFrame,
    *,
    include_title: bool = False,
    strip_style: bool = False,
    min_chars: int = ISOT_MIN_CHARS,
) -> tuple[pd.DataFrame, dict]:
    """Apply the leakage fixes to raw ISOT rows.

    Args:
        df: output of `loaders.load_isot_raw` (columns id, title, text, subject, date, label).
        include_title: prepend the headline. Off by default: headline style is another cue.
        strip_style: also remove "Featured image via ..." style residue (an ablation option).
        min_chars: drop bodies shorter than this after cleaning.

    Returns:
        (clean_df with columns id, text, label), report with counts per step.
    """
    report = {"rows_raw": int(len(df))}
    out = df.copy()

    # 1. metadata columns are dropped entirely (subject => perfect label leak)
    body = out["text"].fillna("").astype(str).map(strip_source_markers)
    if strip_style:
        body = body.map(strip_style_markers)
    if include_title:
        body = out["title"].fillna("").astype(str) + ". " + body
    out["text"] = body.map(normalize_whitespace)

    # 2. near-empty bodies
    short = out["text"].str.len() < min_chars
    report["dropped_short"] = int(short.sum())
    out = out.loc[~short]

    # 3. duplicates: drop texts that appear under BOTH labels, then keep one copy of the rest
    out = out.assign(_key=out["text"].map(dedup_key))
    conflicting = out.groupby("_key")["label"].transform("nunique") > 1
    report["dropped_label_conflicts"] = int(conflicting.sum())
    out = out.loc[~conflicting]
    before = len(out)
    out = out.drop_duplicates("_key", keep="first")
    report["dropped_duplicates"] = int(before - len(out))

    out = out[["id", "text", "label"]].reset_index(drop=True)
    report["rows_clean"] = int(len(out))
    report["label_counts"] = {int(k): int(v) for k, v in out["label"].value_counts().items()}
    return out, report
