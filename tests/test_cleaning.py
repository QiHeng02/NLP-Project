import pandas as pd

from src.config import LABEL_FAKE, LABEL_REAL
from src.data.cleaning import clean_isot, strip_source_markers
from src.data.loaders import liar_to_binary
from src.data.splits import stratified_split

BODY = "The committee voted on the proposal after a long debate on Tuesday afternoon."


def test_strips_reuters_dateline_variants():
    assert "Reuters" not in strip_source_markers("WASHINGTON (Reuters) - " + BODY)
    assert strip_source_markers("SEOUL/BEIJING (Reuters) – " + BODY).strip() == BODY
    assert strip_source_markers("(Reuters) - " + BODY).strip() == BODY
    assert "reuters" not in strip_source_markers("Officials told Reuters on Monday.").lower()
    assert "reuters" not in strip_source_markers("Follow @ReutersPolitics and #ReutersPoll").lower()


def _isot(rows):
    df = pd.DataFrame(rows, columns=["title", "text", "subject", "date", "label"])
    df["id"] = [f"r{i}" for i in range(len(df))]
    return df


def test_clean_isot_drops_leaks_duplicates_and_short_rows():
    raw = _isot([
        ("t", "WASHINGTON (Reuters) - " + BODY, "politicsNews", "d", LABEL_REAL),
        ("t", BODY + " Extra fake-only sentence here.", "News", "d", LABEL_FAKE),
        ("t", BODY + " Extra fake-only sentence here.", "News", "d", LABEL_FAKE),   # duplicate
        ("t", "too short", "News", "d", LABEL_FAKE),                                # short
        ("t", "Shared text " + BODY, "News", "d", LABEL_FAKE),                      # conflict...
        ("t", "Shared text " + BODY, "worldnews", "d", LABEL_REAL),                 # ...pair
    ])
    clean, report = clean_isot(raw)
    assert list(clean.columns) == ["id", "text", "label"]  # subject/date gone
    assert not clean["text"].str.contains("Reuters").any()
    assert report["dropped_short"] == 1
    assert report["dropped_label_conflicts"] == 2
    assert report["dropped_duplicates"] == 1
    assert report["rows_clean"] == 2


def test_liar_binary_mapping():
    assert liar_to_binary("pants-fire") == LABEL_FAKE
    assert liar_to_binary("barely-true") == LABEL_FAKE
    assert liar_to_binary("half-true") == LABEL_REAL
    assert liar_to_binary("true") == LABEL_REAL


def test_stratified_split_is_deterministic_and_disjoint():
    df = pd.DataFrame({"id": range(200), "text": ["x"] * 200, "label": [0, 1] * 100})
    a, b = stratified_split(df), stratified_split(df)
    assert a.equals(b)
    ids = {s: set(a.loc[a["split"] == s, "id"]) for s in ("train", "val", "test")}
    assert not (ids["train"] & ids["test"]) and not (ids["train"] & ids["val"])
    assert len(ids["test"]) == 20 and len(ids["val"]) == 20
