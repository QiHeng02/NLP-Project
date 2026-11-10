"""Evaluate a zero-shot local LLM on deterministic stratified samples.

Run validation first; use --split test only for the final locked prompt. A JSONL
checkpoint lets an interrupted run resume without repeating generated answers.
The 500-row default is a subset, and its metrics must never be presented as
full-test-set results.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import RESULTS_DIR, SEED
from src.data.loaders import load_split
from src.evaluation.metrics import compute_metrics, log_run
from src.models.llm_zero_shot import DEFAULT_MODEL, ZeroShotLLM


def sample_rows(frame, label_col: str, n: int, seed: int):
    """Preserve original row indices for resumable evaluation and auditability."""
    if n <= 0:
        raise ValueError("sample size must be positive")
    if n >= len(frame):
        return frame
    selected, _ = train_test_split(np.arange(len(frame)), train_size=n,
                                   stratify=frame[label_col], random_state=seed)
    return frame.iloc[np.sort(selected)]


def run(dataset: str, split: str, n: int, model_id: str, seed: int,
        checkpoint_dir: Path, author: str):
    folder, label_col, task = ("liar", "label_6", "liar6") if dataset == "liar6" else (dataset, "label", "binary")
    frame = load_split(folder, split)
    sample = sample_rows(frame, label_col, n, seed)
    classifier = ZeroShotLLM(model_id, task=task)
    # Checkpoint identity includes the exact sample and prompt to avoid stale answers.
    import hashlib
    signature = json.dumps({"model": model_id, "dataset": dataset, "split": split,
                            "seed": seed, "ids": sample["id"].astype(str).tolist(),
                            "prompt": classifier.prompt("EXAMPLE")[0],
                            "max_input_tokens": classifier.max_input_tokens,
                            "max_new_tokens": classifier.max_new_tokens,
                            "truncation_policy": classifier.TRUNCATION_POLICY,
                            "decoding_policy": classifier.DECODING_POLICY}, sort_keys=True)
    digest = hashlib.sha256(signature.encode()).hexdigest()[:16]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{dataset}_{split}_{digest}.jsonl"
    answers = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            answers[str(row["id"])] = row["label"]
    start = time.perf_counter()
    with checkpoint.open("a", encoding="utf-8") as out:
        for _, row in sample.iterrows():
            key = str(row["id"])
            if key in answers:
                continue
            label = classifier.classify(str(row["text"]))
            answers[key] = label
            out.write(json.dumps({"id": key, "label": label}) + "\n")
            out.flush()
            print(f"[{dataset}] {len(answers)}/{len(sample)}", flush=True)
    elapsed = time.perf_counter() - start
    labels = [answers[str(i)] for i in sample["id"]]
    pred = [int(x == "fake") for x in labels] if task == "binary" else labels
    metrics = compute_metrics(sample[label_col], pred,
                              task="binary" if task == "binary" else "multiclass")
    path = log_run(model="llm-zero-shot", dataset=dataset, split=split, metrics=metrics,
                   runtime_s=round(elapsed, 2), author=author,
                   params={"model_id": model_id, "prompt": classifier.prompt("EXAMPLE")[0],
                           "seed": seed, "sample_n": len(sample), "population_n": len(frame),
                           "max_input_tokens": classifier.max_input_tokens,
                           "truncation_policy": classifier.TRUNCATION_POLICY,
                           "decoding_policy": classifier.DECODING_POLICY,
                           "max_new_tokens": classifier.max_new_tokens},
                   notes=f"Stratified sample of {len(sample)}/{len(frame)}; zero-shot; uncalibrated labels")
    print(f"{dataset} {split}: macro-F1={metrics['f1_macro']:.3f}, "
          f"accuracy={metrics['accuracy']:.3f} ({len(sample)}/{len(frame)} rows) -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=["isot", "liar", "liar6", "covid"],
                   default=["isot", "liar", "liar6", "covid"])
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--sample-n", type=int, default=500)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--author", default="Keith")
    p.add_argument("--checkpoint-dir", type=Path, default=RESULTS_DIR / "llm_cache")
    args = p.parse_args()
    for dataset in args.datasets:
        run(dataset, args.split, args.sample_n, args.model_id, args.seed,
            args.checkpoint_dir, args.author)


if __name__ == "__main__":
    main()
