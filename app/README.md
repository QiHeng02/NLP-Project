# Streamlit fake news model explorer

**Goal (rubric: Presentation):** paste a news article, claim or post → get real/fake, a confidence
score, and the words that drove the prediction (LIME highlights).

## Features
1. Format selector (article / claim / social post) chooses the matching dataset.
2. Saved DistilBERT or classical model selector, result card, fake score and CPU latency.
3. Optional side-by-side comparison of the two model families on the same input.
4. LIME highlights (green = pushes toward real, red = pushes toward fake).
5. Validation examples with dataset labels shown after prediction.

## Requirements
- Load a saved model from `models/` that follows the shared interface (`predict_proba(list[str])`, see `src/models/README.md`)
- Prediction under 1 s on CPU (requirement R4). DistilBERT or the classical model qualify; check latency before choosing.

Install `requirements.txt` and place the trusted team checkpoints at:

| Format | DistilBERT | Classical |
|---|---|---|
| Article | `models/transformers/isot/distilbert/` | `models/classical/isot/tfidf-svm.joblib` |
| Claim | `models/transformers/liar/distilbert/` | `models/classical/liar/bow-nb.joblib` |
| Post | `models/transformers/covid/distilbert/` | `models/classical/covid/tfidf-svm.joblib` |

Run from the repository root:
```bash
streamlit run app/streamlit_app.py
```

The UI loads only saved binary models. LIAR-6 is included in the experiment runner,
but its six-class output is not shown as a fake/real probability.

The zero-shot evaluation and explanation figure workflow is documented in
[`docs/llm_explainability_demo.md`](../docs/llm_explainability_demo.md).
