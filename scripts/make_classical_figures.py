"""Slide figures for Role B (Traditional ML). No training: reads saved models and logged runs.

Outputs in results/figures/:
    classical_top_words.png / .csv  - words that push the best model towards "fake" vs "real"
    classical_models_val.png         - all 8 models per task (val, default settings)
    classical_prep_ablation.png      - effect of stop-word removal / lemmatisation (val)
    classical_final_test.png         - best classical vs baseline vs BiLSTM vs GloVe-avg (test)

Usage:
    python -m scripts.make_classical_figures
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, MODELS_DIR, RUNS_DIR

# Reference palette (dataviz skill, light mode); validated for colour-blind separation
REAL, FAKE, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"

TASKS = ["isot", "liar", "liar6", "covid"]
TITLES = {"isot": "ISOT (articles)", "liar": "LIAR (claims, real/fake)",
          "liar6": "LIAR-6 (6 levels)", "covid": "COVID (posts)"}
# Best model per binary dataset (final test runs) -> saved pipeline
BEST = {"isot": "tfidf-svm", "liar": "bow-nb", "covid": "tfidf-svm"}
N_WORDS = 15

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
})


def _runs() -> pd.DataFrame:
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in RUNS_DIR.glob("*.json")]
    return pd.DataFrame(rows)


def _grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, name):
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


# ---------------------------------------------------------------- 1. top words
def word_weights(pipe) -> pd.Series:
    """Weight per word; positive = pushes towards fake (label 1)."""
    names = pipe.named_steps["features"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_log_prob_"):  # Naive Bayes: log-odds fake vs real
        i_fake, i_real = list(clf.classes_).index(1), list(clf.classes_).index(0)
        w = clf.feature_log_prob_[i_fake] - clf.feature_log_prob_[i_real]
    elif hasattr(clf, "calibrated_classifiers_"):  # calibrated LinearSVC: mean over CV folds
        w = np.mean([c.estimator.coef_[0] for c in clf.calibrated_classifiers_], axis=0)
    else:
        w = clf.coef_[0]
    return pd.Series(w, index=names)


def top_words() -> None:
    fig, axes = plt.subplots(1, len(BEST), figsize=(13, 6.5))
    rows = []
    for ax, (tag, name) in zip(axes, BEST.items()):
        path = MODELS_DIR / "classical" / tag / f"{name}.joblib"
        if not path.exists():
            ax.set_visible(False)
            print(f"[{tag}] skipped: {path} missing (run run_classical --test --save)")
            continue
        w = word_weights(joblib.load(path))
        fake, real = w.nlargest(N_WORDS), w.nsmallest(N_WORDS)
        for word, v in fake.items():
            rows.append({"dataset": tag, "model": name, "side": "fake", "word": word, "weight": v})
        for word, v in real.items():
            rows.append({"dataset": tag, "model": name, "side": "real", "word": word, "weight": v})

        sel = pd.concat([real.iloc[::-1], fake.iloc[::-1]])  # real at the bottom, strongest fake on top
        y = np.arange(len(sel))
        colors = [FAKE if v > 0 else REAL for v in sel.values]
        ax.barh(y, sel.values, color=colors, height=0.72, edgecolor=SURFACE, linewidth=1)
        span = np.abs(sel.values).max()
        for yi, (word, v) in zip(y, sel.items()):
            ax.text(v + (0.02 if v > 0 else -0.02) * span, yi, word, va="center",
                    ha="left" if v > 0 else "right", fontsize=8, color=INK)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_xlim(-span * 1.55, span * 1.55)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        _grid(ax)
        label = "log-odds (NB)" if name.endswith("nb") else "SVM weight"
        ax.set_xlabel(f"← real      {label}      fake →")
        ax.set_title(f"{TITLES[tag]}\n{name}", loc="left")
    fig.suptitle("Words that push the best classical model towards fake or real",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "classical_top_words.png")
    pd.DataFrame(rows).to_csv(FIGURES_DIR / "classical_top_words.csv", index=False)


# ---------------------------------------------------------------- 2. model comparison (val)
def models_val(df: pd.DataFrame) -> None:
    mine = df[(df.author == "Sharmayn") & (df.notes == "phase1: defaults") & (df.split == "val")]
    base = df[(df.model == "tfidf-lr-sanity") & (df.split == "val")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, tag in zip(axes, TASKS):
        d = mine[mine.dataset == tag].sort_values("f1_macro")
        y = np.arange(len(d))
        best = d.f1_macro.max()
        ax.hlines(y, d.f1_macro.min() - 0.02, d.f1_macro, color=GRID, linewidth=1)
        ax.scatter(d.f1_macro, y, s=46, color=[FAKE if v == best else REAL for v in d.f1_macro],
                   zorder=3, edgecolor=SURFACE, linewidth=1.5)
        for yi, v in zip(y, d.f1_macro):
            ax.text(v + 0.004 * (1 if tag != "liar6" else 1), yi, f" {v:.3f}", va="center", fontsize=8, color=INK2)
        b = base[base.dataset == tag].f1_macro
        if len(b):
            ax.axvline(b.iloc[0], color=MUTED, linestyle="--", linewidth=1)
            ax.text(b.iloc[0], len(d) - 0.4, " baseline", color=MUTED, fontsize=8, va="bottom")
        ax.set_yticks(y, d.model)
        lo, hi = d.f1_macro.min(), d.f1_macro.max()
        ax.set_xlim(lo - 0.02, hi + (hi - lo) * 0.35 + 0.01)
        ax.set_ylim(-0.6, len(d) - 0.1)
        _grid(ax)
        ax.set_title(TITLES[tag], loc="left")
        ax.set_xlabel("val macro-F1")
    fig.suptitle("All classical models, default settings (val). Orange = best per task",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "classical_models_val.png")


# ---------------------------------------------------------------- 3. preprocessing ablation
def prep_ablation(df: pd.DataFrame) -> None:
    mine = df[(df.author == "Sharmayn") & (df.split == "val")]
    variants = [("stopwords=True)", "stop words removed", REAL),
                ("lemmatize=True)", "lemmatised", FAKE),
                ("stopwords=True, lemmatize=True)", "both", THIRD)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, tag in zip(axes, TASKS):
        d = mine[mine.dataset == tag]
        models = sorted(d[d.notes.str.startswith("sens: prep")].model.unique())
        width = 0.26
        for k, (suffix, label, color) in enumerate(variants):
            deltas = []
            for m in models:
                base = d[(d.model == m) & (d.notes == "phase1: defaults")].f1_macro.iloc[0]
                v = d[(d.model == m) & (d.notes == f"sens: prep ({suffix}")].f1_macro.iloc[0]
                deltas.append(v - base)
            x = np.arange(len(models)) + (k - 1) * width
            ax.bar(x, deltas, width=width - 0.03, color=color, label=label)
            for xi, v in zip(x, deltas):
                ax.text(xi, v + (0.0006 if v >= 0 else -0.0006), f"{v:+.3f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=7, color=INK2)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_xticks(np.arange(len(models)), models)
        _grid(ax, "y")
        ax.set_title(TITLES[tag], loc="left")
    axes[0].set_ylabel("change in val macro-F1\nvs no preprocessing")
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo * 1.15, hi * 1.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Removing stop words mostly hurts; lemmatising barely matters (val)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(handles, labels, frameon=False, loc="upper right", ncol=3, fontsize=9,
               bbox_to_anchor=(0.99, 1.0))
    _save(fig, "classical_prep_ablation.png")


# ---------------------------------------------------------------- 4. final test comparison
def final_test(df: pd.DataFrame) -> None:
    test = df[(df.split == "test") & (df.train_dataset == df.dataset)].sort_values("timestamp")
    best_name = {"isot": "tfidf-svm", "liar": "bow-nb", "liar6": "tfidf-lr", "covid": "tfidf-svm"}
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.0))
    for ax, tag in zip(axes, TASKS):
        t = test[test.dataset == tag]
        entries = [("TF-IDF + LR baseline", "tfidf-lr-sanity", MUTED),
                   ("GloVe-avg + LR (no word order)", "glove-lr", THIRD),
                   ("BiLSTM (Arshad)", "bilstm", REAL),
                   ("Mine (best classical)", best_name[tag], FAKE)]
        vals = []
        for label, model, color in entries:
            r = t[t.model == model]
            vals.append((label, r.f1_macro.iloc[-1] if len(r) else np.nan, color))
        y = np.arange(len(vals))
        xs = [v for _, v, _ in vals]
        lo, hi = np.nanmin(xs), np.nanmax(xs)
        pad = (hi - lo) * 0.45 + 0.005
        for yi, (label, v, color) in zip(y, vals):
            ax.hlines(yi, lo - pad, v, color=GRID, linewidth=1)
            ax.scatter(v, yi, s=50, color=color, zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.text(v, yi + 0.28, f"{v:.3f}", ha="center", fontsize=8, color=INK2)
        ax.set_yticks(y, [l for l, _, _ in vals] if tag == "isot" else [""] * len(vals))
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.5, len(vals) - 0.3)
        _grid(ax)
        ax.set_title(f"{TITLES[tag]}\nmine = {best_name[tag]}", loc="left")
        ax.set_xlabel("test macro-F1")
    fig.suptitle("Final test results: classical vs baseline vs BiLSTM",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "classical_final_test.png")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = _runs()
    df["notes"] = df["notes"].fillna("")
    top_words()
    models_val(df)
    prep_ablation(df)
    final_test(df)


if __name__ == "__main__":
    main()
