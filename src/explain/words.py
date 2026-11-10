"""LIME and SHAP explain how masking words changes a model's fake score.

These are local model-behaviour explanations, not evidence that a claim is true
or false. Both methods use the shared predict_proba(raw_texts) interface.
"""
from __future__ import annotations

import html
import re

import numpy as np


def fake_probability(model, texts):
    classes = list(model.classes_)
    if set(classes) != {0, 1}:
        raise ValueError(f"Expected binary classes 0/1, got {classes}")
    probabilities = np.asarray(model.predict_proba(list(texts)))
    if probabilities.shape != (len(texts), 2):
        raise ValueError("predict_proba must return one pair per text")
    return probabilities[:, classes.index(1)]


def ordered_probabilities(model, texts):
    fake = fake_probability(model, texts)
    return np.column_stack((1 - fake, fake))


def lime_words(model, text: str, *, num_samples: int = 500,
               num_features: int = 12, seed: int = 42):
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=["real", "fake"], random_state=seed)
    explanation = explainer.explain_instance(text, lambda xs: ordered_probabilities(model, xs),
                                             labels=(1,), num_samples=num_samples,
                                             num_features=num_features)
    return explanation.as_list(label=1)


def highlighted_html(text: str, weights: list[tuple[str, float]]) -> str:
    """Escape user text and highlight approximate word matches; red raises P(fake)."""
    lookup = {word.casefold(): weight for word, weight in weights}
    def render(match):
        word = match.group(0)
        weight = lookup.get(word.casefold(), 0)
        if not weight:
            return html.escape(word)
        color = "#f8b4b4" if weight > 0 else "#a7e7b0"
        return f'<mark style="background:{color};padding:1px 2px">{html.escape(word)}</mark>'
    return re.sub(r"\w+|\W+", lambda m: render(m) if m.group(0)[0].isalnum()
                  else html.escape(m.group(0)), text).replace("\n", "<br>")


def shap_words(model, text: str, *, max_evals: int = 120):
    """Return SHAP's word-level fake probability attributions for one document."""
    import shap

    masker = shap.maskers.Text(r"\W+")
    # The callable returns one scalar per text, so SHAP sees a single output.
    # Supplying output_names as a list makes recent SHAP versions expect a
    # multi-output model and raises an AssertionError at explanation time.
    explainer = shap.Explainer(lambda xs: fake_probability(model, xs), masker,
                               algorithm="partition")
    return explainer([text], max_evals=max_evals)
