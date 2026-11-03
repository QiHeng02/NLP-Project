# Data

Everything in `data/raw/` and `data/processed/` is **git-ignored**. Regenerate it locally with:

```bash
python -m scripts.prepare_data
```

The output is deterministic (seed 42), so every member gets identical splits.

## Sources

| Dataset | How it's obtained | Expected raw files |
|---|---|---|
| **LIAR** | Automatic: official zip from UCSB, with a GitHub mirror as fallback | `raw/liar/train.tsv`, `valid.tsv`, `test.tsv` |
| **COVID-19** (Constraint@AAAI-2021) | Automatic: GitHub mirror of the official English data | `raw/covid/train.csv`, `val.csv`, `test.csv` |
| **ISOT** | **Manual** (Kaggle login needed), see below | `raw/isot/Fake.csv`, `raw/isot/True.csv` |

### ISOT: one-time manual step

**Option 1: browser**
1. Sign in to Kaggle and open https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Click **Download** and unzip `Fake.csv` and `True.csv` into `data/raw/isot/`
3. Run `python -m scripts.prepare_data --datasets isot`

**Option 2: Kaggle CLI**
1. `pip install kaggle`, then put your API token at `%USERPROFILE%\.kaggle\kaggle.json` (Windows)
2. `python -m scripts.prepare_data --datasets isot` downloads it automatically

## Processed output

`data/processed/<dataset>/`:
- `train.csv`, `val.csv`, `test.csv` with columns `id, text, label` (LIAR also has `label_6`)
- `report.json`: rows removed at each cleaning step and label counts per split (use it for the leakage slide)

## Citation / licence notes

- **LIAR:** Wang (2017), ACL. Research use.
- **COVID-19:** Patwa et al. (2021), CONSTRAINT @ AAAI. Official data is distributed via CodaLab; we use a public mirror and cite the original paper.
- **ISOT:** Ahmed, Traore & Saad (2017). University of Victoria ISOT lab; Kaggle redistribution.
