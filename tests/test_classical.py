import numpy as np
import pytest

from src.models import classical
from src.models.classical import GloveAverager, Tokenizer, build_classical

TEXTS = ["the senate passed the budget bill today", "shocking secret cure doctors hate",
         "officials confirmed the report on monday", "you won't believe this miracle trick",
         "the committee approved the new budget", "aliens secretly control the weather"] * 5
LABELS = [0, 1, 0, 1, 0, 1] * 5


@pytest.mark.parametrize("model,features", [("nb", "tfidf"), ("lr", "tfidf"), ("svm", "tfidf"),
                                            ("nb", "bow"), ("lr", "bow"), ("svm", "bow")])
def test_shared_interface(model, features):
    pipe = build_classical(model, features, min_df=1).fit(TEXTS, LABELS)
    proba = pipe.predict_proba(TEXTS[:4])
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1)
    assert set(pipe.predict(TEXTS)) <= {0, 1}


def test_nb_rejects_glove():
    with pytest.raises(ValueError):
        build_classical("nb", "glove")


def test_tokenizer_ablations():
    assert "the" not in Tokenizer(stopwords=True)("The cats are running")
    assert "cat" in Tokenizer(lemmatize=True)("cats")


def test_glove_averager(monkeypatch):
    fake = {"budget": np.ones(3, np.float32), "bill": np.full(3, 3, np.float32)}
    monkeypatch.setattr(classical, "load_glove", lambda path=None: fake)
    avg = GloveAverager().fit(["x"])
    out = avg.transform(["budget bill", "unknown words"])
    assert np.allclose(out[0], 2) and np.allclose(out[1], 0)
    assert avg.coverage(["budget unknown"]) == 0.5
