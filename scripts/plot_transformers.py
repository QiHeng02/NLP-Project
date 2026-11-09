"""Charts for the transformer results (Role E), drawn from results/runs/*.json.

Each chart is skipped (with a message) until the runs it needs exist, so this can be re-run
at any point:  after the val runs, after the checks, and after the final --test runs.

    python -m scripts.plot_transformers                 # all charts -> results/figures/tr_*.png
    python -m scripts.plot_transformers --author Menujaa

Charts
    tr_val_models.png      val score per task: BERT vs DistilBERT vs RoBERTa
    tr_speed.png           val score vs training time (and CPU latency once --latency runs exist)
    tr_curves.png          val macro-F1 after every epoch
    tr_lr.png              learning-rate sensitivity (runs tagged --notes "sens: lr")
    tr_isot_checks.png     ISOT shortcut checks (start only, start + end, no web clutter, headline + article)
    tr_test_models.png     final TEST score per task: BERT vs DistilBERT vs RoBERTa
    tr_bars_<split>.png    grouped bars of the score per task (axis from 0)
    tr_table_<split>.png   score table, best model per task starred
    tr_time.png            training time per task
    (<split> = val, and also test once --test runs exist)

Same look as Arshad's sequential charts (notebooks/02_AR_sequential_sensitivity.ipynb).
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from matplotlib.ticker import MaxNLocator  # noqa: E402

from src.config import FIGURES_DIR, RUNS_DIR  # noqa: E402

TRANSFORMERS = [("bert", "BERT"), ("distilbert", "DistilBERT"), ("roberta", "RoBERTa")]
TASKS = [("isot", "ISOT (articles)"), ("liar", "LIAR (claims)"),
         ("liar6", "LIAR-6 (6 levels)"), ("covid", "COVID (posts)")]
# LIAR-6 is reported as accuracy (target R2c), everything else as macro-F1
METRIC = {"isot": "f1_macro", "liar": "f1_macro", "liar6": "accuracy", "covid": "f1_macro"}
DEFAULT_LR = 2e-5
DEFAULT_TRUNC = {"isot": "head_tail"}  # the others use plain head truncation

COLORS = {"bert": "#2a78d6", "distilbert": "#1baf7a", "roberta": "#eb6834"}
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "lines.linewidth": 2, "lines.markersize": 8,
})


def metric_of(task: str) -> str:
    return METRIC.get(task, "f1_macro")  # ISOT variants etc. use macro-F1


def metric_label(task: str, split: str = "val") -> str:
    return f"{split} accuracy" if metric_of(task) == "accuracy" else f"{split} macro-F1"


def load_runs() -> pd.DataFrame:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUNS_DIR.glob("*.json"))]
    df = pd.json_normalize(rows, sep=".")
    for col in ("notes", "author", "params.lr", "params.truncation", "params.seed"):
        if col not in df:
            df[col] = None
    df["notes"] = df["notes"].fillna("")
    return df[df["train_dataset"] == df["dataset"]]  # cross-dataset runs are not plotted here


def main_runs(df: pd.DataFrame, split: str, author: str | None) -> pd.DataFrame:
    """The headline transformer run per (model, task), latest first.

    Either a default run (lr 2e-5, no notes) or a rerun tagged --notes "final..." (e.g. with the
    best-val learning rate); checks, sensitivity and cross-dataset runs are left out.
    """
    t = df[df["model"].isin([m for m, _ in TRANSFORMERS]) & (df["split"] == split)]
    if author:
        t = t[t["author"] == author]
    is_final = t["notes"].str.startswith("final")
    is_default = ((t["params.lr"].fillna(DEFAULT_LR).round(10) == DEFAULT_LR)
                  & ~t["notes"].str.startswith(("check", "sens", "cross")))
    t = t[(is_final | is_default) & (t["params.seed"].fillna(42) == 42)]
    if t.empty:
        return t
    ok = [r["params.truncation"] in (None, DEFAULT_TRUNC.get(r["dataset"], "head")) for _, r in t.iterrows()]
    t = t[ok]
    return t.sort_values("timestamp").groupby(["model", "dataset"]).tail(1)


def save(fig, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def _dot_panel(ax, labels_vals_colors, refs, task, split):
    """One task panel: a dot per model (refs: optional dashed reference lines)."""
    vals = [v for _, v, _ in labels_vals_colors]
    for y, (label, v, color) in enumerate(labels_vals_colors):
        ax.plot(v, y, "o", color=color, ms=9)
        ax.annotate(f"{v:.3f}", (v, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    ref_vals = []
    for model, style in refs:
        v = refs[(model, style)]
        if v is not None:
            ax.axvline(v, color=COLORS[model], ls=style, lw=1.5)
            ref_vals.append(v)
    lo, hi = min(vals + ref_vals), max(vals + ref_vals)
    # at least 0.02 wide, so a 0.001 gap does not look like a big difference
    half = max(0.7 * (hi - lo), 0.01)
    centre = min((lo + hi) / 2, 1.0 - 0.85 * half)  # keep scores near 1.0 inside the axis
    ax.set_xlim(centre - half, centre + half)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_yticks(range(len(labels_vals_colors)), [lab for lab, _, _ in labels_vals_colors])
    ax.set_ylim(len(labels_vals_colors) - 0.4, -0.7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(metric_label(task, split))


# ------------------------------------------------------------------ charts
def plot_val_models(df, author):
    main = main_runs(df, "val", author)
    tasks = [(t, title) for t, title in TASKS if not main[main["dataset"] == t].empty]
    if not tasks:
        print("tr_val_models: no transformer val runs yet, skipped")
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.3 * len(tasks), 2.9), squeeze=False)
    for ax, (task, title) in zip(axes[0], tasks):
        rows = [(name, float(main.query("model == @m and dataset == @task")[metric_of(task)].iloc[0]),
                 COLORS[m]) for m, name in TRANSFORMERS
                if not main.query("model == @m and dataset == @task").empty]
        _dot_panel(ax, rows, {}, task, "val")
        ax.set_title(title, loc="left")
    fig.suptitle("Fine-tuned transformers on validation data", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "tr_val_models.png")


def plot_speed(df, author):
    main = main_runs(df, "val", author)
    tasks = [(t, title) for t, title in TASKS if not main[main["dataset"] == t].empty]
    if not tasks:
        print("tr_speed: no transformer val runs yet, skipped")
        return
    test = main_runs(df, "test", author)
    has_latency = "params.latency_ms_cpu" in test and test["params.latency_ms_cpu"].notna().any()
    nrows = 2 if has_latency else 1
    fig, axes = plt.subplots(nrows, len(tasks), figsize=(3.3 * len(tasks), 2.9 * nrows), squeeze=False)
    for col, (task, title) in enumerate(tasks):
        ax = axes[0][col]
        for m, name in TRANSFORMERS:
            r = main.query("model == @m and dataset == @task")
            if r.empty:
                continue
            x, y = r["runtime_s"].iloc[0] / 60, r[metric_of(task)].iloc[0]
            ax.plot(x, y, "o", color=COLORS[m], ms=9)
            ax.annotate(name, (x, y), xytext=(6, -3), textcoords="offset points", fontsize=8, color=INK)
        ax.set_title(title, loc="left")
        ax.set_xlabel("training time (min)")
        ax.set_ylabel(metric_label(task) if col == 0 else "")
        ax.set_xlim(left=0)
        ax.margins(x=0.35, y=0.3)
        if has_latency:
            ax = axes[1][col]
            for m, name in TRANSFORMERS:
                r = test.query("model == @m and dataset == @task")
                if r.empty or pd.isna(r["params.latency_ms_cpu"].iloc[0]):
                    continue
                x, y = r["params.latency_ms_cpu"].iloc[0], r[metric_of(task)].iloc[0]
                ax.plot(x, y, "s", color=COLORS[m], ms=8)
                ax.annotate(name, (x, y), xytext=(6, -3), textcoords="offset points", fontsize=8,
                            color=INK)
            ax.axvline(1000, color="#c0392b", ls="--", lw=1)  # R4: < 1 s per input on CPU
            ax.set_xlabel("CPU time per input (ms)")
            ax.set_ylabel(metric_label(task, "test") if col == 0 else "")
            ax.set_xlim(left=0)
            ax.margins(x=0.35, y=0.3)
    fig.suptitle("Accuracy vs cost" + ("  (red line = 1 s demo limit, R4)" if has_latency else ""),
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "tr_speed.png")


def plot_curves(df, author):
    main = main_runs(df, "val", author)
    if "params.history" not in main or main["params.history"].isna().all():
        print("tr_curves: no per-epoch history yet, skipped")
        return
    tasks = [(t, title) for t, title in TASKS if not main[main["dataset"] == t].empty]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.3 * len(tasks), 2.9), squeeze=False)
    for ax, (task, title) in zip(axes[0], tasks):
        for m, name in TRANSFORMERS:
            r = main.query("model == @m and dataset == @task")
            if r.empty or not isinstance(r["params.history"].iloc[0], list):
                continue
            hist = pd.DataFrame(r["params.history"].iloc[0])
            if "val_f1_macro" not in hist:
                continue
            ax.plot(hist["epoch"], hist["val_f1_macro"], "o-", color=COLORS[m], label=name, ms=6)
        ax.set_title(title, loc="left")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val macro-F1" if ax is axes[0][0] else "")
        ax.set_xticks(range(1, 1 + int(main["params.epochs"].max() or 3)))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Learning curves (best epoch is kept)", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "tr_curves.png")


def plot_lr(df, author):
    t = df[df["model"].isin([m for m, _ in TRANSFORMERS]) & (df["split"] == "val")
           & df["dataset"].isin(["liar", "liar6"])]
    if author:
        t = t[t["author"] == author]
    t = t[t["notes"].str.startswith("sens: lr") | (t["notes"] == "")]
    if t["params.lr"].nunique() < 2:
        print("tr_lr: needs runs at 2+ learning rates (--notes \"sens: lr\"), skipped")
        return
    t = t.sort_values("timestamp").groupby(["model", "dataset", "params.lr"]).tail(1)
    tasks = [(k, v) for k, v in TASKS if k in t["dataset"].unique()]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 2.9), squeeze=False)
    for ax, (task, title) in zip(axes[0], tasks):
        for m, name in TRANSFORMERS:
            r = t[(t["model"] == m) & (t["dataset"] == task)].sort_values("params.lr")
            if r.empty:
                continue
            ax.plot(r["params.lr"], r[metric_of(task)], "o-", color=COLORS[m], label=name, ms=6)
        ax.set_xscale("log")
        lrs = sorted(t["params.lr"].unique())
        ax.set_xticks(lrs, [f"{v:.0e}" for v in lrs])
        ax.minorticks_off()
        ax.set_title(title, loc="left")
        ax.set_xlabel("learning rate")
        ax.set_ylabel(metric_label(task))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Learning-rate sensitivity", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "tr_lr.png")


def plot_isot_checks(df, author):
    t = df[df["model"].isin([m for m, _ in TRANSFORMERS]) & (df["split"] == "val")]
    if author:
        t = t[t["author"] == author]
    t = t[t["dataset"].isin(["isot", "isot_nostyle", "isot_title"])]

    def pick(model, dataset, trunc):
        r = t[(t["model"] == model) & (t["dataset"] == dataset)
              & (t["params.truncation"] == trunc) & (t["params.lr"].round(10) == DEFAULT_LR)]
        return None if r.empty else float(r.sort_values("timestamp").iloc[-1]["f1_macro"])

    variants = [("Start + end (main)", "isot", "head_tail"), ("Start only", "isot", "head"),
                ("No web clutter", "isot_nostyle", "head_tail"), ("Headline + article", "isot_title", "head")]
    model = "distilbert"
    vals = [(label, pick(model, d, tr)) for label, d, tr in variants]
    vals = [(label, v) for label, v in vals if v is not None]
    if len(vals) < 2:
        print("tr_isot_checks: needs the DistilBERT ISOT check runs, skipped")
        return
    fig, ax = plt.subplots(figsize=(6.2, 0.55 * len(vals) + 1.2))
    for y, (label, v) in enumerate(vals):
        ax.plot(v, y, "o", color=COLORS[model], ms=9)
        ax.annotate(f"{v:.3f}", (v, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(vals)), [label for label, _ in vals])
    ax.set_ylim(len(vals) - 0.4, -0.7)
    pts = [v for _, v in vals]
    # at least 0.02 wide, so near-identical scores are not stretched across the chart
    half = max(0.7 * (max(pts) - min(pts)), 0.01)
    centre = min((max(pts) + min(pts)) / 2, 1.0 - 0.85 * half)
    ax.set_xlim(centre - half, centre + half)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("val macro-F1 (DistilBERT)")
    ax.set_title("ISOT checks: what is the model relying on?", loc="left")
    fig.tight_layout()
    save(fig, "tr_isot_checks.png")


def plot_test_models(df, author):
    test = main_runs(df, "test", author)
    tasks = [(t, title) for t, title in TASKS if not test[test["dataset"] == t].empty]
    if not tasks:
        print("tr_test_models: no transformer TEST runs yet (use --test), skipped")
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.3 * len(tasks), 2.9), squeeze=False)
    for ax, (task, title) in zip(axes[0], tasks):
        rows = [(name, float(test.query("model == @m and dataset == @task")[metric_of(task)].iloc[0]),
                 COLORS[m]) for m, name in TRANSFORMERS
                if not test.query("model == @m and dataset == @task").empty]
        _dot_panel(ax, rows, {}, task, "test")
        ax.set_title(title, loc="left")
    fig.suptitle("Fine-tuned transformers on test data", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "tr_test_models.png")


def _score_grid(df, author, split):
    """3 x 4 array of scores (models x tasks); NaN where a run is missing."""
    import numpy as np

    runs = main_runs(df, split, author)
    grid = np.full((len(TRANSFORMERS), len(TASKS)), np.nan)
    for i, (m, _) in enumerate(TRANSFORMERS):
        for j, (task, _) in enumerate(TASKS):
            r = runs[(runs["model"] == m) & (runs["dataset"] == task)]
            if not r.empty:
                grid[i, j] = r[metric_of(task)].iloc[0]
    return grid, runs


def _task_label(task, title, two_lines=True):
    name, kind = title.split(" (")
    if task == "liar6":
        kind = "6 levels, accuracy)"
    return f"{name}\n({kind}" if two_lines else f"{name} ({kind}"


def plot_bars(df, author):
    import numpy as np

    for split in ("val", "test"):
        grid, runs = _score_grid(df, author, split)
        if np.isnan(grid).all():
            if split == "val":
                print("tr_bars: no transformer runs yet, skipped")
            continue
        fig, ax = plt.subplots(figsize=(8, 3.4))
        x, w = np.arange(len(TASKS)), 0.26
        for i, (m, name) in enumerate(TRANSFORMERS):
            xs = x + (i - 1) * w
            ax.bar(xs, np.nan_to_num(grid[i]), w, color=COLORS[m], label=name)
            for xi, v in zip(xs, grid[i]):
                if not np.isnan(v):
                    ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=7, color=INK)
        ax.set_xticks(x, [_task_label(t, title) for t, title in TASKS])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(f"{split} score")
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
        fig.tight_layout()
        save(fig, f"tr_bars_{split}.png")


def plot_table(df, author):
    import numpy as np

    for split in ("val", "test"):
        grid, _ = _score_grid(df, author, split)
        if np.isnan(grid).all():
            if split == "val":
                print("tr_table: no transformer runs yet, skipped")
            continue
        # shade each column from its worst (light) to its best (dark) model
        lo, hi = np.nanmin(grid, axis=0), np.nanmax(grid, axis=0)
        shade = (grid - lo) / np.where(hi - lo > 0, hi - lo, 1)
        fig, ax = plt.subplots(figsize=(6.6, 2.6))
        ax.imshow(np.nan_to_num(shade), cmap="Blues", vmin=-0.3, vmax=1.3, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if np.isnan(v):
                    ax.text(j, i, "n/a", ha="center", va="center", color=MUTED)
                    continue
                best = v == hi[j]
                ax.text(j, i, f"{v:.3f}" + (" \u2605" if best else ""), ha="center", va="center",
                        fontsize=10, fontweight="bold" if best else "normal", color=INK)
        ax.set_xticks(range(len(TASKS)), [_task_label(t, title, False) for t, title in TASKS], fontsize=8.5)
        ax.set_yticks(range(len(TRANSFORMERS)), [name for _, name in TRANSFORMERS])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{split.capitalize()} scores (\u2605 = best per task; shading compares within each column)",
                     loc="left", fontsize=10)
        fig.tight_layout()
        save(fig, f"tr_table_{split}.png")


def plot_time(df, author):
    import numpy as np

    runs = main_runs(df, "val", author)
    if runs.empty:
        print("tr_time: no transformer runs yet, skipped")
        return
    fig, ax = plt.subplots(figsize=(8, 3.0))
    x, w = np.arange(len(TASKS)), 0.26
    for i, (m, name) in enumerate(TRANSFORMERS):
        mins = []
        for task, _ in TASKS:
            r = runs[(runs["model"] == m) & (runs["dataset"] == task)]
            mins.append(r["runtime_s"].iloc[0] / 60 if not r.empty else np.nan)
        xs = x + (i - 1) * w
        ax.bar(xs, np.nan_to_num(mins), w, color=COLORS[m], label=name)
        for xi, v in zip(xs, mins):
            if not np.isnan(v):
                ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, [title.split(" (")[0] for _, title in TASKS])
    ax.set_ylabel("training time (min)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    save(fig, "tr_time.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--author", default=None, help="only use transformer runs logged by this author")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_runs()
    print(f"{len(df)} runs loaded from {RUNS_DIR}")
    for fn in (plot_val_models, plot_speed, plot_curves, plot_lr, plot_isot_checks, plot_test_models,
               plot_bars, plot_table, plot_time):
        fn(df, args.author)


if __name__ == "__main__":
    main()
