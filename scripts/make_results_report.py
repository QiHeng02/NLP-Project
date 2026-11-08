"""Build the slide-ready results report from results/runs/*.json.

Only FINAL runs count: split == "test", trained and tested on the same dataset, and the
latest run per (model, dataset). Val-only tuning runs, cross-dataset runs and the ISOT
leakage/ablation variants (isot-raw, isot_nostyle, isot_title) are left out.

Outputs:
  results/report/main_results.csv   one row per model, test score per task
  results/report/main_results.md    same table as Markdown, best score per column in bold
  results/report/targets.md         R2 targets: best model, met / not met
  results/figures/results_by_task.png       one panel per task, target as a dashed line
  results/figures/gain_over_baseline.png    best model per family minus the TF-IDF baseline

Usage (from the repo root):
    python -m scripts.make_results_report
"""
from __future__ import annotations

import json

import pandas as pd

from src.config import FIGURES_DIR, RESULTS_DIR, RUNS_DIR

REPORT_DIR = RESULTS_DIR / "report"
BASELINE = "tfidf-lr-sanity"

# (dataset tag, metric, column label)
TASKS = [
    ("isot", "f1_macro", "ISOT F1"),
    ("covid", "f1_macro", "COVID F1"),
    ("liar", "f1_macro", "LIAR F1"),
    ("liar6", "accuracy", "LIAR-6 Acc"),
    ("liar6", "f1_macro", "LIAR-6 F1"),
]
# PROJECT_PLAN.md section 2: column label -> (target id, threshold)
TARGETS = {
    "ISOT F1": ("R2d", 0.99),
    "COVID F1": ("R2a", 0.95),
    "LIAR F1": ("R2b", 0.63),
    "LIAR-6 Acc": ("R2c", 0.27),
}

FAMILY_ORDER = ["Baseline", "Traditional ML", "Sequential DL", "Graph", "Transformer", "LLM", "Other"]
FAMILY_COLORS = {
    "Baseline": "#8C8C8C", "Traditional ML": "#4C72B0", "Sequential DL": "#DD8452",
    "Graph": "#55A868", "Transformer": "#C44E52", "LLM": "#8172B2", "Other": "#937860",
}
# keyword -> family, checked in order (first match wins)
FAMILY_RULES = [
    (("gcn", "gat", "graph"), "Graph"),
    (("bert", "roberta", "deberta", "electra", "xlnet"), "Transformer"),
    (("llm", "gpt", "llama", "qwen", "mistral", "gemma", "flan", "zero-shot", "zeroshot", "claude"), "LLM"),
    (("lstm", "gru", "rnn"), "Sequential DL"),
    (("tfidf", "tf-idf", "bow", "nb", "naive", "svm", "logreg", "lr", "glove-avg", "classical"), "Traditional ML"),
]
PRETTY = {BASELINE: "TF-IDF + LR (baseline)", "bilstm": "BiLSTM", "lstm": "LSTM", "gru": "GRU"}


def family_of(model: str) -> str:
    if model == BASELINE:
        return "Baseline"
    name = model.lower()
    for keywords, family in FAMILY_RULES:
        if any(k in name for k in keywords):
            return family
    return "Other"


def load_final_runs() -> pd.DataFrame:
    files = sorted(RUNS_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"No runs found in {RUNS_DIR}")
    runs = pd.DataFrame([json.loads(f.read_text(encoding="utf-8")) for f in files])
    datasets = {d for d, _, _ in TASKS}
    final = runs[
        (runs["split"] == "test")
        & (runs["train_dataset"] == runs["dataset"])
        & (runs["dataset"].isin(datasets))
    ]
    return final.sort_values("timestamp").groupby(["model", "dataset"]).tail(1)


def build_table(final: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(index=sorted(final["model"].unique()))
    for dataset, metric, label in TASKS:
        scores = final[final["dataset"] == dataset].set_index("model")[metric]
        table[label] = scores
    table.insert(0, "family", [family_of(m) for m in table.index])
    table.insert(1, "author", final.groupby("model")["author"].agg(
        lambda a: ", ".join(sorted({x for x in a if x}))))
    order = {f: i for i, f in enumerate(FAMILY_ORDER)}
    table = table.sort_values(by="family", key=lambda s: s.map(order), kind="stable")
    table.index = [PRETTY.get(m, m) for m in table.index]
    table.index.name = "model"
    return table


def to_markdown(table: pd.DataFrame) -> str:
    score_cols = [label for _, _, label in TASKS]
    best = {c: table[c].max() for c in score_cols}
    header = "| Model | Family | " + " | ".join(score_cols) + " |"
    lines = [header, "|" + "---|" * (len(score_cols) + 2)]
    for model, row in table.iterrows():
        cells = []
        for c in score_cols:
            v = row[c]
            if pd.isna(v):
                cells.append("–")
            else:
                cells.append(f"**{v:.3f}**" if v == best[c] else f"{v:.3f}")
        lines.append(f"| {model} | {row['family']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Test set, macro-F1 unless marked Acc (accuracy). **Bold** = best per column.")
    return "\n".join(lines)


def targets_markdown(table: pd.DataFrame) -> str:
    lines = ["| Target | Task | Needed | Best so far | Model | Baseline | Status |", "|---|---|---|---|---|---|---|"]
    base_name = PRETTY[BASELINE]
    for label, (tid, threshold) in TARGETS.items():
        col = table[label].dropna()
        if col.empty:
            lines.append(f"| {tid} | {label} | ≥ {threshold} | – | – | – | ⏳ no results |")
            continue
        best_model, best_val = col.idxmax(), col.max()
        base = table.loc[base_name, label] if base_name in table.index else float("nan")
        status = "✅ met" if best_val >= threshold else "❌ not met"
        lines.append(f"| {tid} | {label} | ≥ {threshold} | {best_val:.3f} | {best_model} | {base:.3f} | {status} |")
    return "\n".join(lines)


def plot_by_task(table: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = list(TARGETS)  # one panel per target metric
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 0.45 * len(table) + 1.8))
    for ax, label in zip(axes, panels):
        data = table[label].dropna()[::-1]  # reversed so the first model is on top
        colors = [FAMILY_COLORS[table.loc[m, "family"]] for m in data.index]
        bars = ax.barh(data.index, data.values, color=colors)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in data.values], padding=2, fontsize=8)
        threshold = TARGETS[label][1]
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1)
        ax.text(threshold, 1.01, f"target {threshold}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8)
        lo = min(data.min(), threshold) - 0.03
        hi = min(1.0, max(data.max(), threshold) + 0.02)
        ax.set_xlim(max(0.0, lo), hi)
        ax.set_title(label, pad=14)
        ax.tick_params(axis="y", labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in FAMILY_ORDER
               if f in set(table["family"])]
    fig.legend(handles, [f for f in FAMILY_ORDER if f in set(table["family"])],
               loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.suptitle("Test results by task (x-axis zoomed per panel)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "results_by_task.png", dpi=150)
    plt.close(fig)


def plot_gain(table: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_name = PRETTY[BASELINE]
    if base_name not in table.index:
        print("Baseline missing; skipping gain chart")
        return
    cols = ["ISOT F1", "COVID F1", "LIAR F1", "LIAR-6 F1"]
    families = [f for f in FAMILY_ORDER if f not in ("Baseline",) and f in set(table["family"])]
    gains = pd.DataFrame(index=cols, columns=families, dtype=float)
    for fam in families:
        best = table[table["family"] == fam][cols].max()
        gains[fam] = best - table.loc[base_name, cols]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.8 / max(1, len(families))
    for i, fam in enumerate(families):
        xs = [j + (i - (len(families) - 1) / 2) * width for j in range(len(cols))]
        bars = ax.bar(xs, gains[fam].values, width, color=FAMILY_COLORS[fam], label=fam)
        ax.bar_label(bars, labels=["" if pd.isna(v) else f"{v:+.3f}" for v in gains[fam].values],
                     fontsize=7, padding=2, rotation=90)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.margins(y=0.15)
    ax.set_xticks(range(len(cols)), [c.replace(" F1", "") for c in cols])
    ax.set_ylabel("macro-F1 gain over TF-IDF baseline")
    ax.set_title("How much each model family beats the simple baseline (best model per family)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gain_over_baseline.png", dpi=150)
    plt.close(fig)


def main() -> None:
    table = build_table(load_final_runs())
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(REPORT_DIR / "main_results.csv")
    (REPORT_DIR / "main_results.md").write_text(to_markdown(table), encoding="utf-8")
    targets = targets_markdown(table)
    (REPORT_DIR / "targets.md").write_text(targets, encoding="utf-8")
    plot_by_task(table)
    plot_gain(table)

    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(table.drop(columns="author").round(3).to_string(na_rep="–"))
    print("\n" + targets)
    print(f"\nSaved: {REPORT_DIR}/main_results.(csv|md), targets.md")
    print(f"       {FIGURES_DIR}/results_by_task.png, gain_over_baseline.png")


if __name__ == "__main__":
    main()
