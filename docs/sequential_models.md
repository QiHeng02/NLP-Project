# Sequential models (Role C): LSTM / BiLSTM / GRU

Three recurrent models that read text **word by word, in order**, on top of pretrained GloVe word
vectors. They answer the project question *"does word order help?"*: Role B's GloVe-averaged models
use the same word vectors but ignore order, so B vs C isolates the effect of order.

| File | What it is |
|---|---|
| [`src/models/sequential.py`](../src/models/sequential.py) | The model code (import it, don't run it) |
| [`scripts/run_sequential.py`](../scripts/run_sequential.py) | The experiment runner (trains, scores, logs) |
| [`scripts/run_sequential_sweep.py`](../scripts/run_sequential_sweep.py) | Phase 2: all 18 BiLSTM sensitivity + ISOT check runs in one command |
| [`notebooks/02_AR_sequential_sensitivity.ipynb`](../notebooks/02_AR_sequential_sensitivity.ipynb) | Charts and tables for Phase 1 + 2 (reads `results/runs/`, no training). Saves `results/figures/seq_*.png` |

## Which roles need this

| Role | Needs it? | What for | See |
|---|---|---|---|
| **A. Data & evaluation** | **Yes** | Reading C's result JSONs into the results tables | [§3.1](#31-role-a-data--evaluation) |
| **B. Traditional ML** | **Yes** (GloVe only) | Use the same GloVe file and loader, so B vs C is a fair comparison | [§3.2](#32-role-b-traditional-ml) |
| **C. Sequential DL** | Owner | Everything in this file | [§1](#1-how-it-works), [§2](#2-setup-and-usage) |
| **D. Graph models** | Optional | GloVe if your graph nodes use word vectors | [§3.3](#33-role-d-graph-models-and-role-e-transformers) |
| **E. Transformers** | No | Nothing from Role C | [§3.3](#33-role-d-graph-models-and-role-e-transformers) |
| **F. LLM + XAI + demo** | **Yes** | Loading C's saved model for LIME/SHAP and the Streamlit app | [§3.4](#34-role-f-llm--xai--demo) |

---

## 1. How it works

```
"Says the vaccine contains microchips."
  │ tokenize           → ["says", "the", "vaccine", "contains", "microchips", "."]
  │ vocab (train only) → [15, 2, 803, 1204, 1, 3]          0 = padding, 1 = unknown word
  │ GloVe embedding    → 6 × 100 numbers (word meanings)
  │ LSTM / BiLSTM / GRU→ reads the 6 vectors in order, keeps a 128-number "memory"
  │ dropout + linear   → [P(real), P(fake)]
```

### `src/models/sequential.py`

| Piece | Purpose |
|---|---|
| `tokenize` | Lower-case and split off punctuation, matching how GloVe 6B was built |
| `build_vocab` | Word → id for words seen ≥ 2 times in **train** (max 50k). Never built from val/test |
| `load_glove` | Reads `glove.6B.100d.txt` into `{word: vector}` (~15 s, so load once) |
| `embedding_matrix` | One GloVe row per vocab id; words GloVe lacks get small random vectors. Also returns coverage |
| `RNNClassifier` | The network: embedding → RNN → dropout → linear. `arch` picks LSTM / BiLSTM / GRU. Padding is skipped (packed sequences) so short texts aren't diluted |
| `SequentialClassifier` | Wrapper with the team's shared interface (`fit` / `predict` / `predict_proba` / `save` / `load`). Handles batching, training and early stopping |

**Training (`fit`)**: Adam, batches of 64, cross-entropy loss, gradient clipping at 1.0. After each
epoch it scores val macro-F1; it stops after 3 epochs with no improvement and **rolls back to the
best epoch**.

### `scripts/run_sequential.py`

For every (model, dataset) pair: load splits → build `SequentialClassifier` → `fit` on train with
val for early stopping → score → `log_run` to `results/runs/` → optionally save the model.

| Task tag | Data folder | Label column | Type | Default max length |
|---|---|---|---|---|
| `isot` | isot | `label` | binary | 512 |
| `liar` | liar | `label` | binary | 64 |
| `liar6` | liar | `label_6` | 6-class | 64 |
| `covid` | covid | `label` | binary | 128 |
| `isot_nostyle` | isot_nostyle | `label` | binary | 512 |
| `isot_title` | isot_title | `label` | binary | 512 |

Max lengths come from Role A's token-length table in `notebooks/01_QH_eda.ipynb`. Note that this
model counts **words** (its own tokeniser), while A's table counts BERT sub-word tokens, so 512 words
covers more of an article than BERT's 512-token limit.
`isot_nostyle` (web clutter removed) and `isot_title` (headline added) are Role A's ISOT ablation
variants from PR #2: same articles and splits, only the text changes. Build them once with
`python -m scripts.make_isot_ablation --variant nostyle` / `--variant title`. They only run when named
in `--datasets`.

### Command-line options

| Option | Default | Notes |
|---|---|---|
| `--models` | `lstm bilstm gru` | any subset |
| `--datasets` | `isot liar liar6 covid` | any subset, plus `isot_nostyle` / `isot_title` |
| `--max-len` | per dataset (table above) | words read per text |
| `--truncate` | `head` | long texts: keep the first `max_len` words (`head`) or first half + last half (`head_tail`) |
| `--hidden` | 128 | RNN memory size |
| `--dropout` | 0.3 | |
| `--trainable-embeddings` | off (GloVe frozen) | fine-tune GloVe |
| `--lr` / `--batch-size` | 1e-3 / 64 | |
| `--epochs` / `--patience` | 20 / 3 | max epochs / early-stopping patience |
| `--seed` | 42 | |
| `--glove-path` | `data/raw/glove/glove.6B.100d.txt` | only if GloVe is stored elsewhere |
| `--test` | off | also score test. **Final runs only** |
| `--save` | off | save to `models/sequential/<task>/<model>.pt` |
| `--author` / `--notes` | | `--notes "sens: dropout"` tags sensitivity runs |

Defaults = PROJECT_PLAN.md §5 (GloVe 100d, hidden 128, dropout 0.3, Adam 1e-3, early stopping on val macro-F1).

---

## 2. Setup and usage

### Dependencies (on top of `requirements.txt`)

| Dependency | Needed by | How to get it |
|---|---|---|
| PyTorch (CUDA build) | **C**, **F** (to load C's model) | See step 1 below. `requirements.txt` does not install it |
| Processed data | **C** (and every role) | `python -m scripts.prepare_data`, see [`data/README.md`](../data/README.md) |
| GloVe 6B 100d (~330 MB) | **C**, **B** (GloVe-avg features), D (optional) | See step 3 below. Git-ignored |
| C's saved model `.pt` | **F** | Zip shared by C (see §3.4); checkpoints are git-ignored |

1. **PyTorch with CUDA**:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   python -c "import torch; print(torch.cuda.is_available())"   # should print True
   ```
   CPU also works, but training ISOT at 512 words will be slow.
2. **Processed data**: `python -m scripts.prepare_data`.
3. **GloVe 6B 100d**:
   ```bash
   mkdir data/raw/glove
   curl -L -o data/raw/glove/glove.6B.zip https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
   tar -xf data/raw/glove/glove.6B.zip -C data/raw/glove glove.6B.100d.txt
   rm data/raw/glove/glove.6B.zip        # Windows cmd: del data\raw\glove\glove.6B.zip
   ```
   `nlp.stanford.edu/data/...` often times out; use the `downloads.cs.stanford.edu` link above.
   Mirror: `https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip`.

No `gensim` needed: GloVe is read directly from the text file.

### Commands (Role C)

```bash
# Phase 1: main results, default settings, val only
python -m scripts.run_sequential --author "<name>"                          # 3 models × 4 tasks
python -m scripts.run_sequential --models bilstm --datasets liar --author "<name>"

# Phase 2: all 18 sensitivity + ISOT check runs (BiLSTM, val only, ~60–75 min)
python -m scripts.run_sequential_sweep --author "<name>"
python -m scripts.run_sequential_sweep --only dropout embeddings      # a subset
python -m scripts.run_sequential_sweep --dry-run                      # list runs, train nothing

# ...or a single sensitivity run by hand
python -m scripts.run_sequential --models bilstm --datasets isot --max-len 256 --notes "sens: max_len"

# Final: chosen settings, score test once, save the model (the exact commands behind the test results)
python -m scripts.run_sequential --models bilstm --datasets isot --truncate head_tail --test --save --author "<name>"
python -m scripts.run_sequential --models bilstm --datasets liar --dropout 0.5 --test --save --author "<name>"
python -m scripts.run_sequential --models bilstm --datasets liar6 covid --test --save --author "<name>"
```

Each epoch prints `train_loss`, `val_f1_macro` and time. Each run writes one JSON to `results/runs/`
with the scores **and all settings** in `params` (plus `best_epoch`, `epochs_run`, `glove_coverage`).

### Hardware

All Role C runs use C's own laptop GPU: **NVIDIA GeForce RTX 4050 Laptop GPU (6 GB)**, with
PyTorch 2.6 + CUDA 12.4. No shared cluster is needed.

| | Measured per epoch | Per run (~6–12 epochs) |
|---|---|---|
| LIAR / LIAR-6 (64 words) | ~1–2 s | < 1 min |
| ISOT, BiLSTM (512 words) | 27.5 s | ~3–6 min |

Plus ~15 s per command to load GloVe. Actual totals: Phase 1 (12 runs) ~13 min, Phase 2 (18 runs)
~71 min, final runs (4) ~10 min, so ~1.6 h of training overall.

---

## 3. Instructions for other roles

### 3.1 Role A (Data & evaluation)

**Needs:** nothing to install. C's results land in `results/runs/` in the shared format.

- `python -m scripts.collect_results` picks C's runs up automatically.
- `model` is `lstm`, `bilstm` or `gru`; `dataset` is `isot` / `liar` / `liar6` / `covid`.
- Only final runs use `--test`, so the latest test JSON per (model, dataset) is the reported number.
- Sensitivity runs are **val-only** and tagged in `notes` (e.g. `"sens: dropout"`). Leave them out
  of the main results table.
- `params.max_len` follows A's EDA token-length table (LIAR 64, COVID 128, ISOT 512).

### 3.2 Role B (Traditional ML)

**Needs:** the GloVe file (dependency step 3), for the plan's "GloVe-averaged features + LR/SVM".

- File: `data/raw/glove/glove.6B.100d.txt` (6B, 100d). Please use the **same file** as C. Then
  "B: GloVe-avg + LR" vs "C: GloVe + BiLSTM" differs only in word order, which is the comparison
  for the video.
- You can reuse C's helpers instead of gensim:
  ```python
  import numpy as np
  from src.models.sequential import load_glove, tokenize

  glove = load_glove()                      # {word: np.ndarray(100,)}
  def avg_vector(text):
      vecs = [glove[w] for w in tokenize(text) if w in glove]
      return np.mean(vecs, axis=0) if vecs else np.zeros(100, dtype=np.float32)
  ```
- No PyTorch needed for this. `src.models.sequential` imports torch at the top, though, so either
  have torch installed or copy the two functions.

### 3.3 Role D (Graph models) and Role E (Transformers)

**Needs:** no C code.

- **Role D:** if your TextGCN/GAT word nodes use pretrained vectors, the same GloVe file and
  `load_glove()` (§3.2) are available.
- **Role E:** nothing needed from Role C.

### 3.4 Role F (LLM + XAI + demo)

**Needs:** PyTorch installed (CPU build is fine), and the `.pt` file for C's best model per dataset
from C. Checkpoints are git-ignored (`models/*`), so they are shared outside git. **No GloVe file
needed**: the embeddings are stored inside the checkpoint.

**Setup**
1. Get C's code. Before PR #3 is merged: `git fetch origin` then `git checkout feat/sequential-arshad`.
   After it is merged: `git pull` on `main`.
2. Install PyTorch if needed (CPU is fine): `pip install torch`.
3. Unzip C's shared `sequential_models_<date>.zip` into the repo's `models/` folder, so you get
   `models/sequential/isot/bilstm.pt`, `.../liar/bilstm.pt`, `.../liar6/bilstm.pt`, `.../covid/bilstm.pt`.

**Usage.** The model takes **raw strings** and does its own tokenising (no preprocessing needed on
your side), so it plugs straight into LIME:

```python
from src.models.sequential import SequentialClassifier

model = SequentialClassifier.load("models/sequential/liar/bilstm.pt")   # device="cpu" to force CPU
model.predict_proba(["Says the vaccine contains microchips."])           # -> array (n, 2), column 1 = P(fake)
model.predict([...])                                                     # -> array of 0/1 (0 = real, 1 = fake)
model.classes_                                                           # column order of predict_proba

# LIME
from lime.lime_text import LimeTextExplainer
exp = LimeTextExplainer(class_names=["real", "fake"]).explain_instance(text, model.predict_proba)
```

- Expected paths: `models/sequential/<isot|liar|liar6|covid>/<lstm|bilstm|gru>.pt`.
- For `liar6`, `classes_` is the 6 label strings in alphabetical order, not the truthfulness order.
  Use `classes_` to label the columns.
- CPU speed per single input (final models): ISOT 19 ms, LIAR / LIAR-6 1.4 ms, COVID 2.9 ms, well under
  target R4 (< 1 s). LIME calls `predict_proba` on thousands of perturbed texts per explanation, so ISOT
  explanations take longer than single predictions.
- Final models: `models/sequential/<isot|liar|liar6|covid>/bilstm.pt` (~32 MB total, 29 MB zipped),
  shared with Role F as a zip.

---

## 4. Status (Wed 4 Nov)

**Done (Role C)**
- [x] Environment: PyTorch 2.6 + CUDA 12.4 on NVIDIA GeForce RTX 4050 Laptop GPU (6 GB)
- [x] Timing check: ISOT BiLSTM at 512 words = 27.5 s per epoch, so no cluster needed
- [x] Data prepared; sanity baseline reproduces Role A's numbers exactly (ISOT 0.980, LIAR 0.592, LIAR-6 0.252, COVID 0.929)
- [x] GloVe 6B 100d downloaded
- [x] `src/models/sequential.py` and `scripts/run_sequential.py` written
- [x] Smoke test (2 epochs, not real results): all 3 models train on GPU for LIAR and LIAR-6; save → load gives identical predictions; `predict_proba` handles raw and empty strings
- [x] **Phase 1 complete:** 3 models × 4 tasks = 12 val runs, default settings (~13 min total)
- [x] Added `--truncate head|head_tail`, the `isot_nostyle` / `isot_title` tasks, and `scripts/run_sequential_sweep.py`
- [x] **Phase 2 runs complete:** 18 BiLSTM val runs (15 sensitivity + 3 ISOT checks)
- [x] **Phase 2 plots:** `notebooks/02_AR_sequential_sensitivity.ipynb` → `results/figures/`:
  `seq_phase1_models.png`, `seq_sens_max_len.png`, `seq_sens_settings.png`, `seq_isot_checks.png`
- [x] **Phase 3 complete:** 4 final BiLSTM runs scored on test once, models saved to `models/sequential/`
- [x] Checks: saved models reproduce the logged test scores on CPU; test used only by the 4 final runs;
  `python -m scripts.collect_results` picks up exactly the 4 final runs

**Next (Role C)**
- [x] Shared the 4 `.pt` files with Role F (zip)
- [ ] Get PR #3 reviewed and merged (by Mon 9 Nov freeze)

### Phase 1 results (val)

Macro-F1, except LIAR-6, which shows accuracy (macro-F1 in brackets). Baseline = Role A's TF-IDF + LR, **val** split.

| Task | Baseline | LSTM | BiLSTM | GRU |
|---|---|---|---|---|
| ISOT | 0.981 | 0.995 | **0.997** | **0.997** |
| LIAR | 0.619 | 0.617 | 0.613 | **0.621** |
| LIAR-6 | 0.260 (0.238) | 0.265 (0.263) | **0.272 (0.269)** | 0.252 (0.229) |
| COVID | 0.924 | 0.917 | **0.955** | 0.929 |

Settings: GloVe 100d frozen, hidden 128, dropout 0.3, Adam 1e-3, patience 3; max length ISOT 512 /
LIAR 64 / COVID 128. Runtime: ISOT 2.5–5.5 min per model, all others < 30 s.

**Takeaways**
- **BiLSTM is the best sequential model** (best on 3 of 4 tasks), so it is used for Phase 2.
- **COVID:** BiLSTM 0.955 vs TF-IDF 0.924 (+0.031). LSTM (one direction) scored 0.917, below TF-IDF.
- **LIAR:** all three are ~0.61–0.62, level with TF-IDF (0.619). BiLSTM peaked at epoch 1 and stopped
  after 4, so it overfits LIAR's small training set almost immediately. Dropout and frozen-vs-trainable
  embeddings are tested in Phase 2.
- **LIAR-6:** BiLSTM accuracy 0.272 vs 0.260; macro-F1 0.269 vs 0.238 (val; see test results below).
- **ISOT: near-perfect (0.995–0.997), likely too easy.** Role A's PR #2 found TF-IDF does not depend on
  web clutter (0.980 on `isot_nostyle`). Phase 2 checks the same for BiLSTM (it doesn't either; see below).
- GloVe coverage of each vocabulary: LIAR 98%, ISOT 90%, COVID 89% (hashtags, URLs, slang).

### Phase 2 results (val, BiLSTM)

One setting changed at a time; everything else at the Phase 1 defaults. **Bold** = default (the
Phase 1 BiLSTM result). Macro-F1. Run: `python -m scripts.run_sequential_sweep`.

**LIAR** (default 0.613)

| Setting | Values → val macro-F1 |
|---|---|
| Max length | 32 → 0.618 · **64 → 0.613** · 128 → 0.617 |
| Hidden size | 64 → 0.611 · **128 → 0.613** · 256 → 0.613 |
| Dropout | 0.1 → 0.616 · **0.3 → 0.613** · 0.5 → 0.628 |
| GloVe | **frozen → 0.613** · trainable → 0.619 |

**ISOT** (default 0.997)

| Setting | Values → val macro-F1 | Runtime |
|---|---|---|
| Max length | 128 → 0.987 · 256 → 0.995 · **512 → 0.997** · 1024 → 0.998 | 1.3 / 4.8 / 5.4 / 15.2 min |
| Hidden size | 64 → 0.996 · **128 → 0.997** · 256 → 0.996 | |
| Dropout | 0.1 → 0.996 · **0.3 → 0.997** · 0.5 → 0.997 | |
| GloVe | **frozen → 0.997** · trainable → 0.998 | |

**ISOT checks** (default settings)

| Variant | val macro-F1 | Tests |
|---|---|---|
| Default: start of article, body only | 0.997 | |
| `isot_nostyle`: web clutter removed | 0.997 | Does BiLSTM rely on links / photo credits / embed code? |
| `--truncate head_tail`: start + end | 0.999 | Does the end of the article add clues? |
| `isot_title`: headline added | 0.998 | Do headlines add clues? |

**Takeaways**
- **LIAR is insensitive to every setting** (0.611–0.619), except dropout 0.5 (0.628, best epoch moved
  from 1 to 4, so it overfits more slowly). Val has 1,284 claims, so differences of ~0.01 are within
  run-to-run noise. The limit is the information in a ~22-word claim, not model size or tuning.
- **ISOT: only max length clearly matters.** 0.987 at 128 words → 0.997 at 512. Gains flatten after
  256, and 1024 words costs ~3× the time for +0.001. Even 128 words beats TF-IDF (0.981).
- **Removing web clutter doesn't change BiLSTM's score:** `isot_nostyle` scores the same (0.997), matching
  Role A's TF-IDF result. The most likely explanation is that real and fake articles differ in topic and
  writing style in the body text itself (not measured directly).
- Start + end (0.999) and headlines (0.998) add a little. At this ceiling, 0.001 ≈ 4 of 3,823 articles.
- Hidden size, dropout and trainable GloVe make no meaningful difference on ISOT.

Charts for both phases: `notebooks/02_AR_sequential_sensitivity.ipynb` → `results/figures/seq_*.png`.

### Final settings (chosen on val, before test was run)

For each task, the setting with the highest val score from Phases 1–2.

| Task | Model | Settings |
|---|---|---|
| ISOT | BiLSTM | defaults + `--truncate head_tail` (val 0.999) |
| LIAR | BiLSTM | defaults + `--dropout 0.5` (val 0.628) |
| LIAR-6 | BiLSTM | defaults (not swept; best in Phase 1, val acc 0.272) |
| COVID | BiLSTM | defaults (not swept; best in Phase 1, val 0.955) |

### Phase 3: final test results

Each run: `--test --save`, test scored once. Baseline = Role A's TF-IDF + LR **test** scores.

| Task | Metric | Val | **Test** | TF-IDF test | Difference | Plan target |
|---|---|---|---|---|---|---|
| ISOT | macro-F1 | 0.999 | **0.996** | 0.980 | +0.017 | ≥ 0.99 (R2d) met |
| COVID | macro-F1 | 0.955 | **0.951** | 0.929 | +0.022 | ≥ 0.95 (R2a) met |
| LIAR | macro-F1 | 0.628 | **0.595** | 0.592 | +0.003 | ≥ 0.63 (R2b) not met |
| LIAR-6 | accuracy | 0.272 | **0.246** | 0.252 | −0.005 | ≥ 0.27 (R2c) not met |
| LIAR-6 | macro-F1 | 0.269 | **0.240** | 0.219 | +0.021 | |

Differences are computed before rounding. Plan targets are **team** targets ("best test" score across
all model families), not per role.

**Bootstrap check** (1,000 resamples of the test set, BiLSTM − TF-IDF):

| Task | Difference | 95% range | BiLSTM ahead in |
|---|---|---|---|
| COVID macro-F1 | +0.022 | +0.011 to +0.033 | 100% |
| ISOT macro-F1 | +0.017 | +0.012 to +0.021 | 100% |
| LIAR macro-F1 | +0.003 | −0.026 to +0.033 | 55% |
| LIAR-6 macro-F1 | +0.021 | −0.009 to +0.052 | 93% |
| LIAR-6 accuracy | −0.005 | −0.036 to +0.026 | 35% |

**LIAR-6 predictions** (test, 1,283 claims): TF-IDF predicts the common labels (half-true, false) most
often and catches 3% of pants-fire claims; BiLSTM spreads its predictions across all 6 labels and catches
26% of pants-fire claims. This is why its macro-F1 is higher while accuracy is slightly lower.

**Val → test:** ISOT −0.003 and COVID −0.004. LIAR drops the most (0.628 → 0.595); dropout 0.5 was
chosen on a small val margin. TF-IDF also drops on LIAR (0.619 → 0.592).

**Checks**
- Test was scored only by these 4 runs (all 34 other Role C runs are val-only).
- Reloading each saved model on CPU reproduces the logged test scores exactly.
- Exact text overlap between train and test: ISOT 0, LIAR 4 of 1,283, COVID 2 of 2,140 (from the
  official splits).
- `python -m scripts.collect_results` lists exactly these 4 runs as the `bilstm` row.

### Conclusion

**Best of LSTM / BiLSTM / GRU:** BiLSTM (best on 3 of 4 tasks).

- **COVID and ISOT:** BiLSTM scores higher than TF-IDF (+0.022, +0.017), ahead in every bootstrap resample.
- **LIAR:** +0.003; ahead in about half the resamples, so no difference can be claimed.
- **LIAR-6:** accuracy −0.005, macro-F1 +0.021. BiLSTM predicts the rarer labels more often.
- **Settings:** most changes moved val scores by ~0.01 or less. The number of words read changed the ISOT
  score the most (0.987 at 128 words → 0.997 at 512).
- **ISOT:** removing web clutter didn't change the score, and no test articles appear in train.
- **Why BiLSTM scores higher on COVID and ISOT is not established.** It differs from TF-IDF in both word
  order and GloVe word meanings. Role B's GloVe-averaged model (same vectors, no order) would separate the two.

**Possible reasons LIAR shows no gain (theory, not tested here)**
- The label depends on real-world facts that are not in the claim text.
- Claims are ~18–22 words, so there is little sequence structure for an RNN to use.
- The TF-IDF baseline already uses word pairs (`ngram_range=(1, 2)`), which capture short phrases.
- ~10k training claims favour a simpler, lower-variance model (BiLSTM's best epoch was 1–4).
- GloVe represents numbers and specific figures poorly, and many claims hinge on them.
- LIAR-6 labels are ordered, but both models treat them as unrelated classes; neighbouring labels
  (e.g. half-true vs mostly-true) are close calls even for fact-checkers.

**Limitations**
- One random seed.
- BiLSTM was tuned a little; the TF-IDF baseline was not tuned.
- LIAR-6 and COVID use default settings (not swept).
- No cross-dataset test, so how far the ISOT result transfers to other news sources is untested.
- Max length counts words, not BERT tokens; keep this in mind when comparing with Role E on ISOT.
