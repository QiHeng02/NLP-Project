# Zero-shot LLM, explainability, and Streamlit demo

This work adds a prompt-based zero-shot classifier, LIME and SHAP explanations
for the saved binary classical and DistilBERT models, and an interactive demo
for articles, claims and social posts. The scripts write measured outputs to
`results/runs/` and `results/figures/`; this document does not substitute
unverified scores for those files.

Run the lightweight interface checks with
`python -m unittest tests.test_llm_and_explanations`.

## Zero-shot LLM evaluation

Install dependencies from `requirements.txt` and prepare the shared data. The default
LLM is `Qwen/Qwen2.5-0.5B-Instruct`, loaded locally from Hugging Face on first run.
Its parameters require a separate download. No examples or metadata enter the prompt.
The same locked prompt is used for all three binary datasets; LIAR-6 has its own
six-label prompt. Generation is greedy and restricted to the task's label token
sequences, preventing invented labels. These are still zero-shot predictions.

```bash
python -m scripts.run_llm_zero_shot --split val --sample-n 500
python -m scripts.run_llm_zero_shot --split test --sample-n 500
```

Use validation to check the protocol; only run the final test after the prompt and
sample size are fixed. Results are logged through `log_run`, with the model ID,
prompt, seed, population size and sample size. Predictions are checkpointed in
`results/llm_cache/` (ignored by Git), so rerunning resumes the same sample.
Unexpected token sequences stop the run for inspection. The output is a class label and
has **no calibrated probability** or ROC-AUC. Report sampled macro-F1 and
accuracy distinctly from teammates' full-test metrics. LIAR-6's headline metric
is accuracy. Time spent on a resumed run excludes earlier work.

## LIME, SHAP, and the demo

After receiving the trained checkpoints, generate figures using validation rows:

```bash
python -m scripts.make_xai_figures --method both
streamlit run app/streamlit_app.py
```

This produces a LIME and SHAP figure for each best classical and DistilBERT binary
model on ISOT, LIAR and COVID (12 figures). Both perturb the input and measure
changes in the fake probability. For a shorter CPU run, use `--families classical`
or `--method lime`; transformer SHAP can require many forward passes. Use the same
validation row across families for each dataset. A positive word weight increases
the fake score in the local approximation. It does not establish that a word or
claim is objectively false. For ISOT, discuss possible source/style cues even
after cleaning; for LIAR, short claims and missing context limit explanation.

Model files are ignored by Git and must be obtained from the owners. Their reported
latencies are checkpoint-specific; measure the demo's prediction time on the actual
machine before claiming the under-one-second target. Do not claim zero-shot or XAI
results until the respective commands complete successfully.
