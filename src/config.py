"""Project-wide constants: paths, random seed, and label conventions.

Import from here instead of hard-coding paths or label values in your own code,
so every member's experiments stay comparable.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"  # saved checkpoints (git-ignored)

# ---------------------------------------------------------------- reproducibility
SEED = 42

# ---------------------------------------------------------------- datasets
DATASETS = ("isot", "liar", "covid")
SPLITS = ("train", "val", "test")

# Binary label convention used by EVERY dataset and model in this repo.
LABEL_REAL = 0
LABEL_FAKE = 1
LABEL_NAMES = {LABEL_REAL: "real", LABEL_FAKE: "fake"}

# LIAR 6-way truthfulness scale, ordered from least to most truthful.
LIAR_LABELS_6 = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
# Binary mapping for LIAR (documented in PROJECT_PLAN.md; change only as a team).
LIAR_FAKE_LABELS = {"pants-fire", "false", "barely-true"}

# ---------------------------------------------------------------- ISOT cleaning defaults
ISOT_MIN_CHARS = 50        # drop near-empty article bodies
ISOT_TEST_SIZE = 0.10      # ISOT has no official split: 80/10/10 stratified
ISOT_VAL_SIZE = 0.10
