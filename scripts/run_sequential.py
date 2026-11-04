"""Train and evaluate the sequential models (LSTM / BiLSTM / GRU on GloVe), Role C.

Every (model, dataset) pair is trained once and its VAL score is logged to results/runs/.
Test is only scored with --test, so all tuning choices are made on val.

Usage:
    python -m scripts.run_sequential --models bilstm --datasets liar
    python -m scripts.run_sequential                                   # all 3 models x 4 tasks
    python -m scripts.run_sequential --models bilstm --datasets isot --max-len 256 --notes "sens: max_len"
    python -m scripts.run_sequential --models bilstm --datasets liar --test --save   # final run
"""
from __future__ import annotations

import argparse
import time

from src.config import DATA_PROCESSED, LABEL_FAKE, MODELS_DIR, SEED
from src.data.loaders import load_splits
from src.evaluation.metrics import compute_metrics, log_run
from src.models.sequential import ARCHS, GLOVE_PATH, SequentialClassifier, load_glove

# task tag -> (processed folder, label column, metric task, default max_len from the EDA token lengths)
TASKS = {
    "isot": ("isot", "label", "binary", 512),
    "liar": ("liar", "label", "binary", 64),
    "liar6": ("liar", "label_6", "multiclass", 64),
    "covid": ("covid", "label", "binary", 128),
}


def _evaluate(model, X, y, *, task: str) -> dict:
    proba = model.predict_proba(X)
    pred = model.classes_[proba.argmax(axis=1)]
    score = proba[:, list(model.classes_).index(LABEL_FAKE)] if task == "binary" else None
    return compute_metrics(y, pred, score, task=task)


def run(arch: str, tag: str, glove: dict, args: argparse.Namespace) -> None:
    folder, label_col, task, default_len = TASKS[tag]
    if not (DATA_PROCESSED / folder / "train.csv").exists():
        print(f"[{tag}] skipped: run `python -m scripts.prepare_data` first")
        return
    splits = load_splits(folder)
    model = SequentialClassifier(
        arch, glove=glove, max_len=args.max_len or default_len, hidden_size=args.hidden,
        dropout=args.dropout, freeze_embeddings=not args.trainable_embeddings, lr=args.lr,
        batch_size=args.batch_size, max_epochs=args.epochs, patience=args.patience, seed=args.seed,
    )
    print(f"\n[{arch} | {tag}] max_len={model.max_len} hidden={model.hidden_size} "
          f"dropout={model.dropout} frozen={model.freeze_embeddings} device={model.device}")

    t0 = time.perf_counter()
    model.fit(splits["train"]["text"], splits["train"][label_col],
              splits["val"]["text"], splits["val"][label_col])
    runtime = round(time.perf_counter() - t0, 2)

    params = {**model.get_params(), "best_epoch": model.best_epoch_,
              "epochs_run": len(model.history_), "glove_coverage": round(model.glove_coverage_, 3)}
    for split in ("val", "test") if args.test else ("val",):
        m = _evaluate(model, splits[split]["text"], splits[split][label_col], task=task)
        log_run(model=arch, dataset=tag, split=split, metrics=m, params=params,
                runtime_s=runtime, author=args.author, notes=args.notes)
        print(f"[{arch} | {tag}] {split:4s} acc={m['accuracy']:.3f} f1_macro={m['f1_macro']:.3f} "
              f"(best epoch {model.best_epoch_}, {runtime:.0f}s)")

    if args.save:
        path = model.save(MODELS_DIR / "sequential" / tag / f"{arch}.pt")
        print(f"[{arch} | {tag}] saved -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=ARCHS, default=list(ARCHS))
    parser.add_argument("--datasets", nargs="+", choices=list(TASKS), default=list(TASKS))
    # hyperparameters (defaults = PROJECT_PLAN.md section 5)
    parser.add_argument("--max-len", type=int, default=None, help="default: per dataset, see TASKS")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--trainable-embeddings", action="store_true", help="fine-tune GloVe (default: frozen)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20, help="max epochs (early stopping usually ends sooner)")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    # run options
    parser.add_argument("--glove-path", default=str(GLOVE_PATH))
    parser.add_argument("--test", action="store_true", help="also score the test split (final runs only)")
    parser.add_argument("--save", action="store_true", help="save the model to models/sequential/<dataset>/")
    parser.add_argument("--author", default=None)
    parser.add_argument("--notes", default="", help="e.g. 'sens: dropout' to tag sensitivity runs")
    args = parser.parse_args()

    print(f"Loading GloVe from {args.glove_path} ...")
    glove = load_glove(args.glove_path)  # load once, shared by every run below
    for tag in args.datasets:
        for arch in args.models:
            run(arch, tag, glove, args)


if __name__ == "__main__":
    main()
