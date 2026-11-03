# Streamlit demo (owner: F)

**Goal (rubric: Presentation):** paste a news article, claim or post → get real/fake, a confidence
score, and the words that drove the prediction (LIME highlights).

## Planned features
1. Text box + "format" selector (article / claim / social post) that picks the model trained on the matching dataset
2. Prediction + probability bar
3. LIME word highlights (green = real, red = fake)
4. Optional: side-by-side comparison of the classical model vs the transformer on the same input
5. Example inputs from each test set (one click to load)

## Requirements
- Load a saved model from `models/` that follows the shared interface (`predict_proba(list[str])`, see `src/models/README.md`)
- Prediction under 1 s on CPU (requirement R4). DistilBERT or the classical model qualify; check latency before choosing.

Run (once `streamlit_app.py` exists):
```bash
streamlit run app/streamlit_app.py
```
