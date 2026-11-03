# Models

One file per model family, owned by one member:

| File | Family | Owner |
|---|---|---|
| `baseline.py` | TF-IDF + LR sanity baseline (reference only) | shared |
| `classical.py` | NB / LR / SVM on TF-IDF, BoW, GloVe | B |
| `sequential.py` | LSTM / BiLSTM / GRU | C |
| `graph.py` | TextGCN / GAT | D |
| `transformers_ft.py` | BERT / DistilBERT / RoBERTa fine-tuning | E |
| `llm_zero_shot.py` | Zero-shot LLM classifier | F |

## Shared interface

LIME, SHAP and the Streamlit app need **one common way to call any model**. Each family should
expose an object (or wrapper) with:

```python
model.fit(texts: list[str], labels: list[int])           # or a train(...) function
model.predict(texts: list[str]) -> np.ndarray            # labels, 0 = real, 1 = fake
model.predict_proba(texts: list[str]) -> np.ndarray      # shape (n, 2); column 1 = P(fake)
```

A scikit-learn `Pipeline` already satisfies this. For PyTorch/Hugging Face models, write a small
wrapper class whose `predict_proba` takes **raw strings** and handles tokenisation internally.

## Conventions

- Hyperparameters as function arguments with sensible defaults (no magic numbers in the body).
- Save checkpoints to `models/<family>/<dataset>/` (git-ignored).
- Evaluate with `src.evaluation.metrics.compute_metrics` and record with `log_run(..., author=...)`.
- Exploratory work goes in `notebooks/`; move reusable code into these files.
