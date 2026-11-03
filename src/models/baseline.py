"""Reference model: TF-IDF + Logistic Regression.

This is a pipeline sanity check and the template for the shared model interface,
not the Traditional ML owner's final model. That owner extends it (NB, SVM,
tuning, preprocessing variants) in src/models/classical.py.

Shared interface every model family should provide (see src/models/README.md):
    fit(texts, labels)
    predict(texts)        -> np.ndarray of labels
    predict_proba(texts)  -> np.ndarray of shape (n, n_classes)  # needed for LIME and Streamlit
A scikit-learn Pipeline already satisfies this.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import SEED


def build_tfidf_lr(
    *,
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    C: float = 1.0,
) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                lowercase=True, max_features=max_features, ngram_range=ngram_range,
                min_df=min_df, sublinear_tf=True,
            )),
            ("clf", LogisticRegression(C=C, max_iter=2000, random_state=SEED)),
        ]
    )
