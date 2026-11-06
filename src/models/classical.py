"""Traditional ML: Naive Bayes, Logistic Regression and Linear SVM (Role B).

Features:
    tfidf  - TF-IDF word n-grams (sublinear tf), same settings as the sanity baseline by default
    bow    - raw word-count n-grams (bag of words)
    glove  - average of the GloVe 6B 100d vectors of the words in the text (no word order)

`glove + lr` vs Role C's `bilstm` (same GloVe file, same tokeniser) isolates the effect of word order.

Every model is a scikit-learn Pipeline, so it already follows the shared interface
(fit / predict / predict_proba on raw strings; see src/models/README.md).
LinearSVC has no predict_proba, so it is wrapped in CalibratedClassifierCV.

Preprocessing ablations (plan §3): `stopwords=True` drops sklearn's English stop-word list,
`lemmatize=True` maps each token to its WordNet lemma (NLTK).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import DATA_RAW, SEED

MODELS = ("nb", "lr", "svm")
FEATURES = ("tfidf", "bow", "glove")

# Same file and tokeniser as Role C (src/models/sequential.py), copied so this module needs no torch.
GLOVE_PATH = DATA_RAW / "glove" / "glove.6B.100d.txt"
_GLOVE_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")
# sklearn's default word pattern, so TF-IDF without ablations matches the baseline exactly
_SKLEARN_TOKEN = re.compile(r"(?u)\b\w\w+\b")


def glove_tokenize(text: str) -> list[str]:
    return _GLOVE_TOKEN.findall(str(text).lower())


@lru_cache(maxsize=1)
def load_glove(path: Path = GLOVE_PATH) -> dict[str, np.ndarray]:
    """Read the GloVe text file into {word: vector}. Cached: loaded once per process (~15 s)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. See docs/sequential_models.md, dependency step 3")
    vectors = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, *values = line.rstrip().split(" ")
            vectors[word] = np.asarray(values, dtype=np.float32)
    return vectors


# ---------------------------------------------------------------- preprocessing
@lru_cache(maxsize=1)
def _lemmatizer():
    import nltk
    from nltk.stem import WordNetLemmatizer

    for pkg in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    return WordNetLemmatizer()


@lru_cache(maxsize=200_000)
def lemma(word: str) -> str:
    return _lemmatizer().lemmatize(word)


class Tokenizer:
    """Picklable tokeniser with optional stop-word removal and lemmatisation.

    Args:
        style: "sklearn" (default TF-IDF word pattern) or "glove" (the GloVe/BiLSTM tokeniser).
    """

    def __init__(self, *, stopwords: bool = False, lemmatize: bool = False, style: str = "sklearn"):
        self.stopwords, self.lemmatize, self.style = stopwords, lemmatize, style

    def __call__(self, text: str) -> list[str]:
        text = str(text).lower()
        toks = _SKLEARN_TOKEN.findall(text) if self.style == "sklearn" else _GLOVE_TOKEN.findall(text)
        if self.stopwords:
            toks = [t for t in toks if t not in ENGLISH_STOP_WORDS]
        if self.lemmatize:
            toks = [lemma(t) for t in toks]
        return toks


class GloveAverager(BaseEstimator, TransformerMixin):
    """Text -> mean GloVe vector of its in-vocabulary tokens (zeros if none).

    The vectors are not stored in the pickled model: a loaded model re-reads GLOVE_PATH.
    """

    def __init__(self, path: str = str(GLOVE_PATH), stopwords: bool = False, lemmatize: bool = False):
        self.path, self.stopwords, self.lemmatize = path, stopwords, lemmatize

    def fit(self, X, y=None):
        self.dim_ = len(next(iter(load_glove(Path(self.path)).values())))
        return self

    def transform(self, X):
        glove = load_glove(Path(self.path))
        tok = Tokenizer(stopwords=self.stopwords, lemmatize=self.lemmatize, style="glove")
        out = np.zeros((len(X), self.dim_), dtype=np.float32)
        for i, text in enumerate(X):
            vecs = [glove[w] for w in tok(text) if w in glove]
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return out

    def coverage(self, X) -> float:
        """Share of tokens that have a GloVe vector (reported in params)."""
        glove = load_glove(Path(self.path))
        tok = Tokenizer(stopwords=self.stopwords, lemmatize=self.lemmatize, style="glove")
        total = hit = 0
        for text in X:
            ts = tok(text)
            total += len(ts)
            hit += sum(w in glove for w in ts)
        return hit / total if total else 0.0


# ---------------------------------------------------------------- builders
def _vectorizer(features: str, *, ngram_range, max_features, min_df, stopwords, lemmatize):
    if features == "glove":
        return GloveAverager(stopwords=stopwords, lemmatize=lemmatize)
    kwargs = dict(lowercase=True, ngram_range=ngram_range, max_features=max_features, min_df=min_df)
    if stopwords or lemmatize:
        kwargs.update(tokenizer=Tokenizer(stopwords=stopwords, lemmatize=lemmatize), token_pattern=None)
    if features == "tfidf":
        return TfidfVectorizer(sublinear_tf=True, **kwargs)
    if features == "bow":
        return CountVectorizer(**kwargs)
    raise ValueError(f"features must be one of {FEATURES}, got {features!r}")


def _classifier(model: str, features: str, *, C: float, alpha: float):
    if model == "nb":
        if features == "glove":
            raise ValueError("MultinomialNB needs non-negative features; use lr or svm with glove")
        return MultinomialNB(alpha=alpha)
    if model == "lr":
        return LogisticRegression(C=C, max_iter=2000, random_state=SEED)
    if model == "svm":
        return CalibratedClassifierCV(LinearSVC(C=C, max_iter=5000, random_state=SEED), cv=3)
    raise ValueError(f"model must be one of {MODELS}, got {model!r}")


def build_classical(
    model: str = "lr",
    features: str = "tfidf",
    *,
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int | None = 50_000,
    min_df: int = 2,
    C: float = 1.0,
    alpha: float = 1.0,
    stopwords: bool = False,
    lemmatize: bool = False,
) -> Pipeline:
    """Build an unfitted Pipeline. Defaults (lr + tfidf) equal the sanity baseline.

    Args:
        model: "nb", "lr" or "svm".
        features: "tfidf", "bow" or "glove".
        ngram_range, max_features, min_df: vectoriser settings (ignored for glove).
        C: inverse regularisation strength for lr / svm.
        alpha: additive smoothing for nb.
        stopwords, lemmatize: preprocessing ablations.
    """
    steps = [("features", _vectorizer(features, ngram_range=ngram_range, max_features=max_features,
                                      min_df=min_df, stopwords=stopwords, lemmatize=lemmatize))]
    if features == "glove":
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", _classifier(model, features, C=C, alpha=alpha)))
    return Pipeline(steps)


def model_name(model: str, features: str) -> str:
    """Name used in results/runs, e.g. 'tfidf-svm', 'glove-lr'."""
    return f"{features}-{model}"
