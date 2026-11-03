# Project Plan: Fake News Detection Across Text Formats

**Course:** EE6405 NLP · **Weight:** 25% · **Deadline:** **Sun 15 Nov 2026, 11:59 pm** (target: submit Sat 14 Nov)
**Team:** 6 members · **Repo:** https://github.com/QiHeng02/NLP-Project

---

## 1. Summary

| Item | Decision |
|---|---|
| **Task** | Binary fake news detection (real vs fake), plus 6-class truthfulness on LIAR |
| **Research question** | Which approach detects misinformation best, and does that change across long articles, short claims and social media posts? Why do models label something as fake? |
| **Datasets** | ISOT (long articles), LIAR (short claims), COVID-19 Constraint (social posts) |
| **Model families** | Traditional ML · sequential DL · graph-based · transformers · zero-shot LLM |
| **Explainability** | LIME / SHAP on the best models |
| **Demo** | Streamlit app: paste text → prediction + highlighted words |
| **Primary metric** | **Macro-F1.** LIAR is imbalanced, and macro-F1 treats both classes equally |

---

## 2. Quantified requirements (draft; confirm as a team by Wed 4 Nov)

Targets are set from the sanity baseline (TF-IDF + LR) numbers and published results.

| ID | Requirement | Target | Baseline / reference |
|---|---|---|---|
| **R1** | Algorithms compared | ≥ 5 model families, each on all 3 datasets | Rubric asks for ≥ 2 |
| **R2a** | COVID: best test macro-F1 | ≥ 0.95 | Sanity baseline 0.929 |
| **R2b** | LIAR binary: best test macro-F1 | ≥ 0.63 | Sanity baseline 0.592 |
| **R2c** | LIAR 6-class: best test accuracy | ≥ 0.27 | Sanity baseline 0.252; Wang (2017) ≈ 0.27 |
| **R2d** | ISOT (cleaned): best test macro-F1 | Set on Wed 4 Nov after the ISOT baseline (expected ~0.98) | Leakage audit reports ~0.98 for LR after cleaning |
| **R3** | Data integrity | 0 duplicate texts across ISOT train/test; no metadata features; leakage table reported | Raw ISOT: subject column alone gives F1 = 1.0 |
| **R4** | Efficiency | Demo prediction < 1 s per input on CPU; training time logged for every model | n/a |
| **R5** | Interpretability | LIME/SHAP for the best classical and best transformer model on every dataset | n/a |
| **R6** | Reproducibility | Every reported number can be regenerated from `scripts/` with seed 42 | n/a |

---

## 3. Datasets and preprocessing decisions

| Dataset | Split | Preprocessing (shared, in `scripts/prepare_data.py`) |
|---|---|---|
| **ISOT** | Our own 80/10/10 stratified split, seed 42, **after** deduplication | See ISOT leakage fixes below |
| **LIAR** | Official | Statement text only (no speaker/party metadata). Binary: fake = {pants-fire, false, barely-true} |
| **COVID** | Official | Whitespace normalisation only |

**ISOT leakage fixes** (from a published audit of this exact corpus):

1. **Drop `subject` and `date`.** Subject values are completely disjoint between classes, so subject alone gives F1 = 1.0.
2. **Strip the "(Reuters)" dateline and the word "Reuters"** (present in ~99% of real articles and ~0% of fake ones).
3. **Drop near-empty bodies** (< 50 characters); almost all are fake.
4. **Remove duplicates before splitting.** A naive split puts ~19% of test articles verbatim into training. Texts that appear under both labels are dropped entirely.
5. **Body text only by default.** Headlines carry style cues (fake headlines are often ALL CAPS). `--isot-include-title` exists as an ablation.
6. Style residue ("Featured image via…", "Getty Images") is **kept by default** and can be removed with `--isot-strip-style` as an ablation experiment.

**Model-specific preprocessing** (stopwords, lemmatisation, tokenisation) belongs to each model owner, because different models need different inputs. Compare with vs without stopword removal and lemmatisation in the classical models.

**Long documents:** ISOT articles average ~400 words. Transformers truncate at 512 tokens (head truncation by default; head+tail is an optional experiment).

---

## 4. Model ladder and owners

| Family | Models | Key design choices to explain in the video | Owner |
|---|---|---|---|
| **Traditional ML** | TF-IDF / BoW + Naive Bayes, Logistic Regression, Linear SVM; GloVe-averaged features + LR/SVM | n-gram range, feature count, C; effect of stopwords/lemmatisation | B |
| **Sequential DL** | LSTM, BiLSTM, GRU on GloVe embeddings | max length per dataset, hidden size, dropout, frozen vs trainable embeddings; **sensitivity plots** (Week 5) | C |
| **Graph** | TextGCN (word–document graph, TF-IDF + PMI edges); GAT variant | Graph construction, edge weights; ISOT subsample (~10k docs) for memory | D |
| **Transformers** | BERT-base, DistilBERT, RoBERTa-base (fine-tuned) | Learning rate, epochs, max length (128 for LIAR/COVID, 512 for ISOT); accuracy vs speed trade-off | E |
| **LLM zero-shot** | Instruction-tuned LLM (local open model or API) | Prompt design; evaluated on a stratified sample (e.g., 500 per test set; state this clearly) | F |
| **Explainability** | LIME, SHAP | Which words drive "fake"; how this differs across the three formats | F |

**Starting hyperparameters** (tune on val):
- Transformers: lr 2e-5, batch 16, 3 epochs, warmup 10%, weight decay 0.01
- BiLSTM/GRU: GloVe 100d, hidden 128, dropout 0.3, Adam lr 1e-3, early stopping on val macro-F1
- TextGCN: 2 layers, hidden 200, dropout 0.5, lr 0.02 (Yao et al., 2019 defaults)

---

## 5. Experiment protocol (everyone follows this)

1. **Data:** use only `load_splits(<dataset>)`. Never re-split or re-clean on your own.
2. **Tuning:** on `val` only. Run `test` **once per final configuration**, at the results freeze.
3. **Seed:** 42 everywhere (`src.config.SEED`). If time permits, report transformer mean ± std over 3 seeds.
4. **Metrics** (`compute_metrics`):
   - Binary: accuracy, macro precision/recall/F1, fake-class F1, ROC-AUC
   - LIAR 6-class: accuracy, macro-F1, weighted-F1
   - Confusion matrix (`plot_confusion_matrix`) for your best model per dataset
5. **Logging:** every evaluation goes through `log_run(..., author="Your Name")`, then commit the JSON.
6. **Efficiency:** record `runtime_s` (training time) and, for demo candidates, per-input inference latency.
7. **Stretch goal:** cross-dataset tests (train on one dataset, test on another) for the best classical and best transformer models. Log them with `train_dataset=` set.

---

## 6. Roles and responsibilities

| Role | Core work | Also owns (to balance workload) | Video segment |
|---|---|---|---|
| **A. Data & evaluation lead** | ISOT download and cleaning, EDA (lengths, class balance), leakage table, results tables and cross-format comparison | **Video editing and integration** (data work finishes early) | Intro, datasets, leakage, results |
| **B. Traditional ML** | `src/models/classical.py`: all classical models + preprocessing ablations | **Slide template + references slide** | Classical models |
| **C. Sequential DL** | `src/models/sequential.py`: LSTM/BiLSTM/GRU + sensitivity plots | Hyperparameter-tuning theory slide (Week 5) | Sequential models |
| **D. Graph models** | `src/models/graph.py`: TextGCN/GAT + graph diagrams | Group reflection document (collects everyone's input) | Graph models |
| **E. Transformers** | `src/models/transformers_ft.py`: BERT variants + efficiency comparison | Exports the demo model for F | Transformers |
| **F. LLM + XAI + demo** | Zero-shot LLM, LIME/SHAP, Streamlit app | GAI declaration form (compiles `docs/ai_usage_log.md`) | XAI + live demo |

**Everyone:** writes the slides for their own part, appears in the video, reviews one other member's PR, and logs AI usage.

---

## 7. Timeline (13 days)

| Date | Milestone | Who | Done when… |
|---|---|---|---|
| **Tue 3 Nov** | Kick-off | All | Repo pushed; roles assigned; everyone has run `prepare_data` + `run_sanity_baseline` |
| **Wed 4 Nov** | Data locked | A | ISOT cleaned and split; leakage table; EDA plots; R2d target set |
| **Week 12 consultation** (date TBC) | Consultation | All | Bring this plan, the baseline table and §10 questions |
| **Thu 5 – Fri 6 Nov** | First results | B–F | Every family has a **val** result on ≥ 1 dataset |
| **Sat 7 – Sun 8 Nov** | Tuning | B–F | Every family has val results on all 3 datasets; sensitivity plots started |
| **Mon 9 Nov** | 🔒 **Results freeze 1** | B–F | Test results for every family × dataset logged; best models identified |
| **Tue 10 Nov** | XAI + demo | F, A | LIME/SHAP figures; Streamlit app runs with the best model; slides drafted |
| **Wed 11 Nov** | 🔒 **Final freeze** | All | No new experiments; slides complete; script written (~1,400 words for 10 min) |
| **Thu 12 Nov** | Recording | All | Each member records their segment (voice + code walkthrough) |
| **Fri 13 Nov** | Editing | A | Full first cut ≤ 10 min; reflection draft; AI usage log compiled |
| **Sat 14 Nov** | Final checks + **submit** | All | §9 checklist complete; signatures collected; submitted |
| **Sun 15 Nov** | Buffer | — | Deadline 11:59 pm. Do the peer evaluation when it is released |

---

## 8. Video plan (≤ 10 min, every member appears, **no AI voice**)

| Time | Segment | Presenter | Show |
|---|---|---|---|
| 0:00–0:50 | Problem, research question, targets (R1–R6) | A | Slides |
| 0:50–1:50 | Datasets, cleaning, **leakage table** | A | Slides + `prepare_data` code |
| 1:50–3:00 | Traditional ML | B | Theory slide + code + results |
| 3:00–4:10 | Sequential DL + hyperparameter sensitivity | C | Architecture diagram + plots |
| 4:10–5:20 | Graph models | D | Graph construction diagram + code |
| 5:20–6:30 | Transformers + efficiency | E | Architecture comparison + results |
| 6:30–7:40 | LLM zero-shot + explainability | F | Prompt + LIME/SHAP figures |
| 7:40–8:40 | Cross-format comparison + interpretation | A | Summary table + key insights |
| 8:40–9:40 | Live Streamlit demo | F | App |
| 9:40–10:00 | Conclusion, limitations, references | B | Slides |

---

## 9. Rubric and submission checklist

**Rubric (aim for Exemplary on every row):**

- [ ] **Requirements:** R1–R6 stated up front and each one shown as met (or explained) at the end
- [ ] **Application & datasets:** 3 datasets in 3 formats; cross-format comparison is a core finding
- [ ] **Framework:** model choices justified, plus documented tuning effort (sensitivity plots, ablations)
- [ ] **Experiment design:** multiple metrics; every model compared against the sanity baseline and the other families
- [ ] **Interpretation:** LIME/SHAP; *why* results differ by format (length, style, topic); what transfers beyond these datasets
- [ ] **Readability:** docstrings, README, consistent structure
- [ ] **Efficiency:** no brute force; runtimes reported
- [ ] **Reusability:** shared loaders, metrics and model interface
- [ ] **Presentation:** working Streamlit demo
- [ ] **Video skills:** all 6 members appear; smooth handovers

**Submission (one per team):**

- [ ] Video ≤ 10 min: shows Python code, theory slides and references; no GAI voice; no MATLAB
- [ ] Code (`.py` / `.ipynb`): zip of the repo, or confirm with the lecturer whether a GitHub link is acceptable
- [ ] Group reflection with each member's contributions, signed by all (`docs/contributions.md`)
- [ ] GAI declaration form (course template), built from `docs/ai_usage_log.md`. Do **not** tick "generated purely by AI tools"
- [ ] Peer evaluation on NTULearn: **every member** (−5% if skipped)

---

## 10. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| ISOT needs a Kaggle login | Blocks ISOT work | A downloads it on Day 1 and shares the zip in the team drive |
| Inflated ISOT results from leakage | Credibility | Cleaning in `prepare_data`; leakage table in the video |
| GPU time (BERT on ISOT) | Delays | Start with DistilBERT; use EEE GPUs or a local RTX 40-series; subsample ISOT if needed |
| TextGCN memory on ISOT | Crash | ~10k-document subsample; sparse adjacency matrix |
| Low LIAR scores | Looks like failure | Frame against Wang (2017, ≈ 0.27 accuracy); it's a hard task |
| Late integration | Rushed video | Two hard freezes (Mon 9, Wed 11) |
| AI-policy penalty | Grade | AI usage log; every member can explain their own code |
| Uneven contribution | Peer-eval penalty | Clear owners; `author` field in results; PR history |

## 11. Questions for the Week 12 consultation

1. Does fake news detection count as a "different" or a "standard" application under the rubric?
2. Is the cross-format comparison (articles vs claims vs posts) enough to count as interpretation "beyond the chosen application"?
3. Is evaluating the zero-shot LLM on a stratified sample of each test set acceptable?
4. Code submission: a zip file, or is a GitHub link acceptable?
