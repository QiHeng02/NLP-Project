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

Max lengths come from Role A's token-length table in `notebooks/01_QH_eda.ipynb`.
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
| C's saved model `.pt` | **F** | Ask C; checkpoints are git-ignored |

1. **PyTorch with CUDA**:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   python -c "import torch; print(torch.cuda.is_available())"   # should print True
   ```
   CPU also works, but ISOT at 512 tokens will be slow.
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

# Final: chosen settings, score test once, save the model
python -m scripts.run_sequential --models bilstm --datasets liar --test --save --author "<name>"
```

Each epoch prints `train_loss`, `val_f1_macro` and time. Each run writes one JSON to `results/runs/`
with the scores **and all settings** in `params` (plus `best_epoch`, `epochs_run`, `glove_coverage`).

### Hardware

All Role C runs use C's own laptop GPU: **NVIDIA GeForce RTX 4050 Laptop GPU (6 GB)**, with
PyTorch 2.6 + CUDA 12.4. No shared cluster is needed.

| | Measured per epoch | Per run (~6–12 epochs) |
|---|---|---|
| LIAR / LIAR-6 (64 tokens) | ~1–2 s | < 1 min |
| ISOT, BiLSTM (512 tokens) | 27.5 s | ~3–6 min |

Plus ~15 s per command to load GloVe. All ~30 planned runs total roughly 1.5–2 h.

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

The model takes **raw strings** and does its own tokenising, so it plugs straight into LIME:

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
- Should be fast enough for the demo on CPU (target R4: < 1 s per input). Not yet timed; C will confirm with the final model.

---

## 4. Status (Wed 4 Nov)

**Done (Role C)**
- [x] Environment: PyTorch 2.6 + CUDA 12.4 on NVIDIA GeForce RTX 4050 Laptop GPU (6 GB)
- [x] Timing check: ISOT BiLSTM at 512 tokens = 27.5 s per epoch, so no cluster needed
- [x] Data prepared; sanity baseline reproduces Role A's numbers exactly (ISOT 0.980, LIAR 0.592, LIAR-6 0.252, COVID 0.929)
- [x] GloVe 6B 100d downloaded
- [x] `src/models/sequential.py` and `scripts/run_sequential.py` written
- [x] Smoke test (2 epochs, not real results): all 3 models train on GPU for LIAR and LIAR-6; save → load gives identical predictions; `predict_proba` handles raw and empty strings
- [x] **Phase 1 complete:** 3 models × 4 tasks = 12 val runs, default settings (~13 min total)
- [x] Added `--truncate head|head_tail`, the `isot_nostyle` / `isot_title` tasks, and `scripts/run_sequential_sweep.py`
- [x] **Phase 2 runs complete:** 18 BiLSTM val runs (15 sensitivity + 3 ISOT checks)
- [x] **Phase 2 plots:** `notebooks/02_AR_sequential_sensitivity.ipynb` → `results/figures/`:
  `seq_phase1_models.png`, `seq_sens_max_len.png`, `seq_sens_settings.png`, `seq_isot_checks.png`

**Next (Role C)**
- [ ] Phase 3: final test runs with the settings below, save best models and send `.pt` files to Role F, open PR (by Mon 9 Nov freeze)

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
- **COVID: word order helps.** BiLSTM 0.955 vs 0.924, already above target R2a (≥ 0.95) on val.
- **LIAR: no gain over TF-IDF.** All three are ~0.61–0.62 (target R2b ≥ 0.63 not met). BiLSTM peaked at
  epoch 1 and stopped after 4, so it overfits LIAR's small training set almost immediately. Dropout and
  frozen-vs-trainable embeddings are tested in Phase 2.
- **LIAR-6: small gain.** BiLSTM accuracy 0.272 meets target R2c (≥ 0.27); macro-F1 0.269 vs 0.238.
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
- **BiLSTM does not rely on web clutter:** `isot_nostyle` scores the same (0.997). This matches
  Role A's TF-IDF result. ISOT is easy because real and fake articles differ in topic and writing
  style in the body text itself.
- Start + end (0.999) and headlines (0.998) add a little. At this ceiling, 0.001 ≈ 4 of 3,823 articles.
- Hidden size, dropout and trainable GloVe make no meaningful difference on ISOT.

Charts for both phases: `notebooks/02_AR_sequential_sensitivity.ipynb` → `results/figures/seq_*.png`.

### Conclusion (Phase 1 + 2, val)

**Does reading words in order help?** It depends on the type of text, and the dataset matters more
than the model.

1. **BiLSTM is the best sequential model** (best on 3 of 4 tasks).
2. **Word order helps on social-media posts:** COVID 0.955 vs TF-IDF 0.924.
3. **Word order does not help on short political claims:** LIAR stays at ~0.61–0.63, level with TF-IDF,
   and no setting changes that. A ~22-word claim doesn't contain the evidence of whether it's true.
   LIAR is the honest measure of how hard fake-news detection is.
4. **ISOT's near-perfect score (0.997–0.999) reflects an easy dataset, not a clever model.** It is not
   overfitting (val is unseen data) and not web clutter (`isot_nostyle` scores the same). Real and fake
   articles come from different sources with different topics and writing styles, so the model can
   separate them without checking facts, and may not transfer to other news sources.
5. **Tuning barely matters; reading length does.** Hidden size, dropout and embeddings move scores by
   ≤ ~0.01. Only max length on long articles clearly matters, and 512 words is the best score/time trade-off.

**Limitations**
- Val scores only; final test scores come once, in Phase 3.
- LIAR differences of ~0.01 are within run-to-run noise (1,284 val claims, one seed).
- LIAR-6 and COVID were not swept; they use the Phase 1 defaults.
- No cross-dataset test, so how far the ISOT result transfers to other news sources is untested.

### Planned final settings (Phase 3, chosen on val)

| Task | Model | Settings |
|---|---|---|
| ISOT | BiLSTM | defaults + `--truncate head_tail` (val 0.999) |
| LIAR | BiLSTM | defaults + `--dropout 0.5` (val 0.628) |
| LIAR-6 | BiLSTM | defaults (not swept; best in Phase 1, val acc 0.272) |
| COVID | BiLSTM | defaults (not swept; best in Phase 1, val 0.955) |

Test scores are not run yet. They come once, in the final runs.
