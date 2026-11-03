"""Download the raw datasets into data/raw/<dataset>/.

LIAR and COVID download automatically. ISOT is hosted on Kaggle and needs a
Kaggle account: either configure the Kaggle CLI (kaggle.json) or download the
zip manually (see data/README.md).
"""
from __future__ import annotations

import io
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from src.config import DATA_RAW

_UA = {"User-Agent": "Mozilla/5.0 (EE6405 course project)"}

COVID_URLS = {
    # Mirror of the Constraint@AAAI-2021 English data (Patwa et al., 2021).
    # Official source needs a CodaLab sign-up; cite the original paper.
    "train.csv": "https://raw.githubusercontent.com/diptamath/covid_fake_news/main/data/Constraint_Train.csv",
    "val.csv": "https://raw.githubusercontent.com/diptamath/covid_fake_news/main/data/Constraint_Val.csv",
    "test.csv": "https://raw.githubusercontent.com/diptamath/covid_fake_news/main/data/english_test_with_labels.csv",
}

LIAR_ZIP_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"  # official (Wang, 2017)
LIAR_MIRROR = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/{name}"
LIAR_FILES = ("train.tsv", "valid.tsv", "test.tsv")

ISOT_KAGGLE_SLUG = "clmentbisaillon/fake-and-real-news-dataset"
ISOT_FILES = ("Fake.csv", "True.csv")


def _fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _have_all(folder: Path, names: tuple[str, ...]) -> bool:
    return all((folder / n).exists() for n in names)


def download_covid(raw_dir: Path = DATA_RAW) -> Path:
    out = raw_dir / "covid"
    out.mkdir(parents=True, exist_ok=True)
    if _have_all(out, tuple(COVID_URLS)):
        print(f"[covid] already present in {out}")
        return out
    for name, url in COVID_URLS.items():
        print(f"[covid] downloading {name} ...")
        (out / name).write_bytes(_fetch(url))
    return out


def download_liar(raw_dir: Path = DATA_RAW) -> Path:
    out = raw_dir / "liar"
    out.mkdir(parents=True, exist_ok=True)
    if _have_all(out, LIAR_FILES):
        print(f"[liar] already present in {out}")
        return out
    try:
        print("[liar] downloading official zip ...")
        with zipfile.ZipFile(io.BytesIO(_fetch(LIAR_ZIP_URL))) as zf:
            for name in LIAR_FILES:
                (out / name).write_bytes(zf.read(name))
    except Exception as err:  # official host is sometimes unreachable
        print(f"[liar] official source failed ({err!r}); using GitHub mirror")
        for name in LIAR_FILES:
            (out / name).write_bytes(_fetch(LIAR_MIRROR.format(name=name)))
    return out


def download_isot(raw_dir: Path = DATA_RAW) -> Path | None:
    out = raw_dir / "isot"
    out.mkdir(parents=True, exist_ok=True)
    if _have_all(out, ISOT_FILES):
        print(f"[isot] already present in {out}")
        return out
    if shutil.which("kaggle"):
        print("[isot] downloading via Kaggle CLI ...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", ISOT_KAGGLE_SLUG, "-p", str(out), "--unzip"],
            check=True,
        )
        if _have_all(out, ISOT_FILES):
            return out
    print(
        "[isot] MANUAL STEP NEEDED:\n"
        f"  1. Download https://www.kaggle.com/datasets/{ISOT_KAGGLE_SLUG}\n"
        f"  2. Unzip Fake.csv and True.csv into {out}\n"
        "  3. Re-run: python -m scripts.prepare_data"
    )
    return None


DOWNLOADERS = {"covid": download_covid, "liar": download_liar, "isot": download_isot}
