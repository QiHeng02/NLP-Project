"""Phase 2 for Role C: BiLSTM sensitivity sweep + ISOT checks, in one command.

Changes ONE setting at a time from the Phase 1 defaults, so each plot isolates one effect.
The default value of each setting is not re-run: it is the Phase 1 BiLSTM result.
Val only. Every run is tagged in `notes` ("sens: <setting>" or "check: <name>").

Usage:
    python -m scripts.run_sequential_sweep --author "Your Name"
    python -m scripts.run_sequential_sweep --only dropout hidden     # a subset
    python -m scripts.run_sequential_sweep --dry-run                 # list the runs, train nothing

Needs the ISOT variants for the checks (built once):
    python -m scripts.make_isot_ablation --variant nostyle
    python -m scripts.make_isot_ablation --variant title
"""
from __future__ import annotations

import argparse
import time

from scripts.run_sequential import run
from src.config import SEED
from src.models.sequential import GLOVE_PATH, load_glove

ARCH = "bilstm"  # best sequential model in Phase 1

# group -> list of (datasets, overrides). Overrides change one setting from the defaults.
SWEEP = {
    "max_len": [
        (["liar"], {"max_len": 32}),
        (["liar"], {"max_len": 128}),
        (["isot"], {"max_len": 128}),
        (["isot"], {"max_len": 256}),
        (["isot"], {"max_len": 1024}),
    ],
    "hidden": [(["liar", "isot"], {"hidden": 64}), (["liar", "isot"], {"hidden": 256})],
    "dropout": [(["liar", "isot"], {"dropout": 0.1}), (["liar", "isot"], {"dropout": 0.5})],
    "embeddings": [(["liar", "isot"], {"trainable_embeddings": True})],
    # ISOT checks: does the score come from the ending, web clutter or headlines?
    "truncate": [(["isot"], {"truncate": "head_tail"})],
    "nostyle": [(["isot_nostyle"], {})],
    "title": [(["isot_title"], {})],
}
CHECKS = {"truncate", "nostyle", "title"}

# Phase 1 defaults (same as scripts/run_sequential.py)
DEFAULTS = dict(max_len=None, truncate="head", hidden=128, dropout=0.3, trainable_embeddings=False,
                lr=1e-3, batch_size=64, epochs=20, patience=3, seed=SEED, test=False, save=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(SWEEP), default=list(SWEEP))
    parser.add_argument("--glove-path", default=str(GLOVE_PATH))
    parser.add_argument("--author", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = [(group, tag, overrides) for group in args.only
            for datasets, overrides in SWEEP[group] for tag in datasets]
    for i, (group, tag, overrides) in enumerate(jobs, 1):
        print(f"{i:2d}. {group:10s} {tag:13s} {overrides}")
    if args.dry_run:
        return

    print(f"\nLoading GloVe from {args.glove_path} ...")
    glove = load_glove(args.glove_path)  # once for the whole sweep
    t0 = time.perf_counter()
    for i, (group, tag, overrides) in enumerate(jobs, 1):
        prefix = "check" if group in CHECKS else "sens"
        run_args = argparse.Namespace(**{**DEFAULTS, **overrides}, author=args.author,
                                      notes=f"{prefix}: {group}")
        print(f"\n=== run {i}/{len(jobs)} ({(time.perf_counter() - t0) / 60:.0f} min elapsed)")
        run(ARCH, tag, glove, run_args)
    print(f"\nSweep done: {len(jobs)} runs in {(time.perf_counter() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
