"""Build ISOT ablation variants from the main processed ISOT splits.

Each variant keeps the SAME articles in the SAME splits as data/processed/isot/;
only the text changes. So any change in score is caused by that change alone.

Variants:
  nostyle : remove web clutter (links, embedded tweets and code, photo credits, handles)
  title   : add the headline in front of the article (headline style may be another clue)

Usage (after `python -m scripts.prepare_data --datasets isot`):
    python -m scripts.make_isot_ablation --variant nostyle
    python -m scripts.run_sanity_baseline --datasets isot_nostyle --author "Your Name"
"""
from __future__ import annotations

import argparse
import json

from src.config import DATA_PROCESSED, ISOT_MIN_CHARS, SPLITS
from src.data.cleaning import normalize_whitespace, strip_source_markers, strip_style_markers
from src.data.loaders import load_isot_raw, load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["nostyle", "title"], required=True)
    args = parser.parse_args()

    if not (DATA_PROCESSED / "isot" / "train.csv").exists():
        raise SystemExit("Run `python -m scripts.prepare_data --datasets isot` first.")

    titles = None
    if args.variant == "title":
        titles = load_isot_raw().set_index("id")["title"].fillna("").map(strip_source_markers)

    out = DATA_PROCESSED / f"isot_{args.variant}"
    out.mkdir(parents=True, exist_ok=True)
    report = {"variant": args.variant, "source": "isot", "changed_rows": {}, "now_short_rows": {}}

    for split in SPLITS:
        df = load_split("isot", split)
        if args.variant == "nostyle":
            new_text = df["text"].map(strip_style_markers)
        else:
            new_text = df["id"].map(titles).fillna("") + ". " + df["text"]
        new_text = new_text.map(normalize_whitespace)
        report["changed_rows"][split] = int((new_text != df["text"]).sum())
        # rows are kept (same articles as the main split) but we report any that became near-empty
        report["now_short_rows"][split] = int((new_text.str.len() < ISOT_MIN_CHARS).sum())
        df.assign(text=new_text).to_csv(out / f"{split}.csv", index=False)

    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[isot_{args.variant}] saved -> {out}")
    print(f"  rows changed per split: {report['changed_rows']}")
    print(f"  rows now under {ISOT_MIN_CHARS} chars: {report['now_short_rows']}")


if __name__ == "__main__":
    main()
