"""Research demo: run with `streamlit run app/streamlit_app.py`."""
from __future__ import annotations

import html
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR
from src.data.loaders import load_split
from src.explain.words import fake_probability, highlighted_html, lime_words

FORMATS = {"News article": "isot", "Short claim": "liar", "Social post": "covid"}


@st.cache_resource(show_spinner="Loading saved model…")
def load_model(family: str, path: str):
    if family == "DistilBERT":
        from src.models.transformers_ft import TransformerClassifier
        return TransformerClassifier.load(path, device="cpu")
    import joblib
    return joblib.load(path)  # only load trusted team checkpoints


def available(dataset: str) -> dict[str, Path]:
    transformer = MODELS_DIR / "transformers" / dataset / "distilbert"
    classical = MODELS_DIR / "classical" / dataset
    paths = {}
    if (transformer / "wrapper.json").exists():
        paths["DistilBERT"] = transformer
    preferred = {"isot": "tfidf-svm", "liar": "bow-nb", "covid": "tfidf-svm"}[dataset]
    for file in sorted(classical.glob("*.joblib"), key=lambda p: (p.stem != preferred, p.name)):
        paths[f"Classical · {file.stem}"] = file
    return paths


def predict_one(family: str, path: Path, text: str) -> dict:
    model = load_model("DistilBERT" if family == "DistilBERT" else "classical", str(path))
    start = time.perf_counter()
    fake = float(fake_probability(model, [text])[0])
    return {"name": family, "fake": fake, "ms": 1000 * (time.perf_counter() - start)}


def result_card(result: dict, selected: bool):
    score = result["fake"]
    label, tone = ("FAKE", "fake") if score >= 0.5 else ("REAL", "real")
    st.markdown(
        f'<div class="result-card {tone}">'
        f'<div class="eyebrow">{html.escape(result["name"])}'
        f'{" · selected" if selected else " · comparison"}</div>'
        f'<div class="decision">{label}</div>'
        f'<div>Model fake score <strong>{score:.1%}</strong></div>'
        f'<div class="track"><div class="fill" style="width:{score * 100:.1f}%"></div></div>'
        f'<div class="scale"><span>REAL</span><span>FAKE</span></div>'
        f'<div class="latency">Prediction: {result["ms"]:.0f} ms on this CPU</div>'
        '</div>', unsafe_allow_html=True,
    )


st.set_page_config(page_title="Signal | Model explorer", page_icon="📰", layout="wide")
st.markdown("""
<style>
  .block-container { max-width: 1200px; padding-top: 2rem; }
  .hero { background: linear-gradient(120deg, #0c2438, #16536a); color: white;
          border-radius: 20px; padding: 30px 36px; margin-bottom: 25px; }
  .hero h1 { color: white; font-size: clamp(2rem, 4vw, 3.2rem);
             letter-spacing: -.04em; margin: .2rem 0 .5rem; }
  .hero p { color: #d9e8ed; margin: 0; max-width: 750px; }
  .kicker { color: #7ce0cf; font-size: .78rem; font-weight: 800; letter-spacing: .13em; }
  .section { color: inherit; opacity: .8; font-size: .76rem; font-weight: 800;
             letter-spacing: .12em; }
  .result-card { background: white; color: #203746; border: 1px solid #dce5eb; border-radius: 17px;
                 padding: 23px 26px; margin-bottom: 13px; box-shadow: 0 8px 25px #1a3c5510; }
  .result-card.fake { border-top: 5px solid #cf5961; }
  .result-card.real { border-top: 5px solid #338e75; }
  .eyebrow, .latency { color: #526b7a; font-size: .84rem; }
  .decision { font-size: 2.5rem; font-weight: 800; letter-spacing: -.04em; }
  .fake .decision { color: #a53640; } .real .decision { color: #226b57; }
  .track { background: #d8eae4; height: 12px; border-radius: 20px;
           overflow: hidden; margin-top: 12px; }
  .fill { background: #d05a62; height: 100%; border-radius: 20px; }
  .scale { display: flex; justify-content: space-between; color: #617887;
           font-size: .7rem; font-weight: 800; margin-top: 5px; }
  .latency { border-top: 1px solid #e7eef2; margin-top: 16px; padding-top: 12px; }
  .explanation { background: white; color: #203746; border: 1px solid #dce5eb;
                 border-radius: 16px; padding: 20px 24px; line-height: 2.1;
                 overflow-wrap: anywhere; }
  .legend { background: #f8b4b4; color: #203746; padding: 2px 6px; border-radius: 4px; }
  .legend.green { background: #a7e7b0; }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="hero"><div class="kicker">EE6405 · MODEL EXPLORER</div>
<h1>Can a model spot misinformation?</h1>
<p>Compare articles, claims and social posts. Inspect the model's decision and the words
that change its score.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Explore")
    format_name = st.selectbox("1 · Text format", list(FORMATS))
    dataset = FORMATS[format_name]
    st.caption({"isot": "Long-form news · ISOT", "liar": "Political claims · LIAR",
                "covid": "Social posts · COVID-19"}[dataset])
    paths = available(dataset)
    if not paths:
        st.error(f"No saved {dataset} model found. Check app/README.md for the expected folders.")
        st.stop()
    family = st.selectbox("2 · Selected model", list(paths))
    alternative = next((name for name in paths if
                        (name == "DistilBERT") != (family == "DistilBERT")), None)
    compare = st.checkbox("Compare with the other model family", disabled=alternative is None)
    st.divider()
    st.caption("Research demo only. A model prediction is not a fact check.")

if st.session_state.get("active_dataset") != dataset:
    st.session_state["active_dataset"] = dataset
    st.session_state["demo_text"] = ""
    st.session_state["example_index"] = 0
    for name in ("prediction", "explanation", "example"):
        st.session_state.pop(name, None)

left, right = st.columns([1.15, .85], gap="large")
with left:
    st.markdown('<div class="section">01 · INPUT</div>', unsafe_allow_html=True)
    st.subheader(f"Try a {format_name.lower()}")
    if st.button("Load a validation example", help="Click again for the next example"):
        try:
            frame = load_split(dataset, "val")
            row = frame.iloc[st.session_state["example_index"] % len(frame)]
            st.session_state["example_index"] += 1
            st.session_state["demo_text"] = str(row["text"])
            st.session_state["example"] = {"text": str(row["text"]), "label": int(row["label"])}
            st.session_state.pop("prediction", None)
            st.session_state.pop("explanation", None)
        except (FileNotFoundError, IndexError, ZeroDivisionError):
            st.warning("Validation data unavailable. Prepare the data or paste your own text.")
    text = st.text_area("Text to classify", key="demo_text", height=275,
                        placeholder="Paste an article, claim or post here…")
    st.caption(f"{len(text.split())} words · Using the {dataset.upper()} model")
    if st.button("Analyse text", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Enter text or load a validation example first.")
        else:
            try:
                selected = predict_one(family, paths[family], text)
                other = predict_one(alternative, paths[alternative], text) if compare else None
                st.session_state["prediction"] = {
                    "key": (dataset, family, alternative if compare else None, text),
                    "selected": selected, "other": other,
                }
                st.session_state.pop("explanation", None)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

key = (dataset, family, alternative if compare else None, text)
prediction = st.session_state.get("prediction")
if prediction and prediction["key"] != key:
    prediction = None

with right:
    st.markdown('<div class="section">02 · RESULT</div>', unsafe_allow_html=True)
    st.subheader("Model decision")
    if prediction:
        result_card(prediction["selected"], selected=True)
        if prediction["other"]:
            result_card(prediction["other"], selected=False)
        example = st.session_state.get("example")
        if example and text == example["text"]:
            label = "FAKE" if example["label"] else "REAL"
            st.info(f"Dataset label for this validation example: **{label}**")
        st.caption("The fake score is a model output, not a verified truth probability. "
                   "Timing excludes model loading and explanation.")
    else:
        st.info("Choose a format, enter text, then click **Analyse text**.")

if prediction:
    st.divider()
    st.markdown('<div class="section">03 · EXPLANATION</div>', unsafe_allow_html=True)
    st.subheader("What influenced this prediction?")
    st.caption(f"LIME perturbs the text to approximate the selected {family} model locally. "
               "This takes longer than one prediction.")
    if st.button("Reveal word highlights"):
        try:
            model = load_model("DistilBERT" if family == "DistilBERT" else "classical",
                               str(paths[family]))
            with st.spinner("Estimating word contributions…"):
                weights = lime_words(model, text, num_samples=300)
            st.session_state["explanation"] = {"key": key, "weights": weights}
        except Exception as exc:
            st.error(f"Explanation failed: {exc}")
    explanation = st.session_state.get("explanation")
    if explanation and explanation["key"] == key:
        weights = explanation["weights"]
        st.markdown('<span class="legend">Red</span> pushes toward fake · '
                    '<span class="legend green">Green</span> pushes toward real',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="explanation">{highlighted_html(text, weights)}</div>',
                    unsafe_allow_html=True)
        with st.expander("See word contributions"):
            st.dataframe({"Word": [word for word, _ in weights],
                          "Effect on fake score": [value for _, value in weights]},
                         hide_index=True, use_container_width=True)
        st.caption("These highlights describe model behaviour; they do not establish factual truth.")
