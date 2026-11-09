"""Fine-tune and evaluate the transformers (BERT-base / DistilBERT / RoBERTa-base), Role E.

Every (model, dataset) pair is trained once and its VAL score is logged to results/runs/.
Test is only scored with --test, so all tuning choices are made on val.

Usage:
    python -m scripts.run_transformers --models distilbert --datasets covid          # quick first check
    python -m scripts.run_transformers                                               # 3 models x 4 tasks
    python -m scripts.run_transformers --models bert --datasets isot --truncate head_tail
    python -m scripts.run_transformers --models roberta --datasets liar --lr 1e-5 --notes "sens: lr"
    python -m scripts.run_transformers --models distilbert --datasets covid --test --save --latency  # final run
    python -m scripts.run_transformers --models bert --datasets isot --test --cross-eval liar covid  # stretch goal

Out of GPU memory on ISOT (512 tokens)?  --batch-size 8 --grad-accum 2  (same effective batch of 16)
"""
from __future__ import annotations

import argparse
import time

import torch

from src.config import DATA_PROCESSED, LABEL_FAKE, MODELS_DIR, SEED
from src.data.loaders import load_splits
from src.evaluation.metrics import compute_metrics, log_run, plot_confusion_matrix
from src.models.transformers_ft import MODEL_IDS, TRUNCATIONS, TransformerClassifier

# task tag -> (processed folder, label column, metric task, default max_len in sub-word tokens)
# Lengths follow Role A's EDA: LIAR 64, COVID 128, ISOT 512 (BERT's maximum)
TASKS = {
    "isot": ("isot", "label", "binary", 512),
    "liar": ("liar", "label", "binary", 64),
    "liar6": ("liar", "label_6", "multiclass", 64),
    "covid": ("covid", "label", "binary", 128),
    "isot_nostyle": ("isot_nostyle", "label", "binary", 512),
    "isot_title": ("isot_title", "label", "binary", 512),
}
DEFAULT_TASKS = ["isot", "liar", "liar6", "covid"]


def _evaluate(model, X, y, *, task: str) -> tuple[dict, list]:
    proba = model.predict_proba(X)
    pred = model.classes_[proba.argmax(axis=1)]
    score = proba[:, list(model.classes_).index(LABEL_FAKE)] if task == "binary" else None
    return compute_metrics(y, pred, score, task=task), pred


def run(name: str, tag: str, args: argparse.Namespace) -> None:
    folder, label_col, task, default_len = TASKS[tag]
    short = name.rstrip("/").split("/")[-1]  # "google/electra-base" -> "electra-base" for file names
    if not (DATA_PROCESSED / folder / "train.csv").exists():
        print(f"[{tag}] skipped: run `python -m scripts.prepare_data` first")
        return
    splits = load_splits(folder)
    if args.limit:  # smoke test on a few rows
        splits = {k: v.sample(min(args.limit, len(v)), random_state=SEED) for k, v in splits.items()}

    model = TransformerClassifier(
        name, max_len=args.max_len or default_len, truncation=args.truncate, head_frac=args.head_frac,
        lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
        warmup_ratio=args.warmup, weight_decay=args.weight_decay, grad_accum=args.grad_accum,
        fp16=not args.no_fp16, seed=args.seed,
    )
    print(f"\n[{short} | {tag}] {model.hf_id} max_len={model.max_len} ({model.truncation}) "
          f"lr={model.lr} batch={model.batch_size}x{model.grad_accum} epochs={model.epochs} "
          f"device={model.device} fp16={model._amp}", flush=True)

    t0 = time.perf_counter()
    model.fit(splits["train"]["text"], splits["train"][label_col],
              splits["val"]["text"], splits["val"][label_col])
    runtime = round(time.perf_counter() - t0, 2)

    params = {**model.get_params(), "hf_id": model.hf_id, "best_epoch": model.best_epoch_,
              "epochs_run": len(model.history_), "history": model.history_,
              "truncated_share": round(model.truncated_share_, 3)}
    if args.latency:  # per-input inference time on CPU, as in the demo (R4)
        train_device = model.device
        model.model_.to("cpu")
        model.device = torch.device("cpu")
        params["latency_ms_cpu"] = model.measure_latency(splits["test"]["text"])
        model.model_.to(train_device)  # back to the GPU for the evaluation below
        model.device = train_device
        print(f"[{short} | {tag}] CPU latency: {params['latency_ms_cpu']} ms per input")

    for split in ("val", "test") if args.test else ("val",):
        m, pred = _evaluate(model, splits[split]["text"], splits[split][label_col], task=task)
        log_run(model=short, dataset=tag, split=split, metrics=m, params=params,
                runtime_s=runtime, author=args.author, notes=args.notes)
        print(f"[{short} | {tag}] {split:4s} acc={m['accuracy']:.3f} f1_macro={m['f1_macro']:.3f} "
              f"(best epoch {model.best_epoch_}, {runtime:.0f}s)", flush=True)
        if split == "test":
            labels = list(model.classes_)
            names = ["real", "fake"] if task == "binary" else [str(c) for c in labels]
            plot_confusion_matrix(splits[split][label_col], pred, labels, names,
                                  title=f"{short} on {tag} (test)", filename=f"cm_{short}_{tag}.png")

    # stretch goal: train on this dataset, test on another (binary tasks only)
    if task == "binary":
        for other in args.cross_eval or []:
            if other == tag or TASKS[other][2] != "binary":
                continue
            o = load_splits(TASKS[other][0])["test"]
            m, _ = _evaluate(model, o["text"], o["label"], task="binary")
            log_run(model=short, dataset=other, train_dataset=tag, split="test", metrics=m,
                    params=params, runtime_s=runtime, author=args.author, notes="cross-dataset")
            print(f"[{short} | {tag} -> {other}] test f1_macro={m['f1_macro']:.3f}")

    if args.save:
        path = model.save(MODELS_DIR / "transformers" / tag / short)
        print(f"[{short} | {tag}] saved -> {path}")

    del model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODEL_IDS),
                        help=f"{list(MODEL_IDS)} or any Hugging Face model id")
    parser.add_argument("--datasets", nargs="+", choices=list(TASKS), default=DEFAULT_TASKS)
    # hyperparameters (defaults = PROJECT_PLAN.md section 4)
    parser.add_argument("--max-len", type=int, default=None, help="default: per dataset, see TASKS")
    parser.add_argument("--truncate", choices=TRUNCATIONS, default="head",
                        help="long texts: keep the start (head) or start + end (head_tail)")
    parser.add_argument("--head-frac", type=float, default=0.25,
                        help="head_tail only: share of the token budget taken from the start")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1, help="effective batch = batch-size x grad-accum")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--no-fp16", action="store_true", help="disable mixed precision on GPU")
    parser.add_argument("--seed", type=int, default=SEED)
    # run options
    parser.add_argument("--test", action="store_true", help="also score the test split (final runs only)")
    parser.add_argument("--save", action="store_true", help="save to models/transformers/<dataset>/<model>/")
    parser.add_argument("--latency", action="store_true", help="measure CPU ms per single input (R4)")
    parser.add_argument("--cross-eval", nargs="+", choices=["isot", "liar", "covid"],
                        help="also test on these datasets (needs --test)")
    parser.add_argument("--limit", type=int, default=None, help="use only N rows per split (smoke test)")
    parser.add_argument("--author", default=None)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    for tag in args.datasets:
        for name in args.models:
            run(name, tag, args)


if __name__ == "__main__":
    main()
