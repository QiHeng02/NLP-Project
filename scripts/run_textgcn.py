
"""Run the TextGCN experiment on the COVID dataset."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.data.loaders import load_splits
from src.evaluation.metrics import compute_metrics, log_run
from src.models.graph import (
    build_vocabulary,
    build_tfidf_matrix,
    build_document_word_adjacency,
    build_word_word_pmi,
    combine_graph_adjacencies,
    normalize_adjacency,
    scipy_to_torch_sparse,
    train_textgcn,
)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Load the existing dataset splits.
    print("Loading COVID dataset...")
    splits = load_splits("covid")

    train_texts = splits["train"]["text"].tolist()
    val_texts = splits["val"]["text"].tolist()
    test_texts = splits["test"]["text"].tolist()

    train_labels = splits["train"]["label"].to_numpy()
    val_labels = splits["val"]["label"].to_numpy()

    num_train = len(train_texts)
    num_val = len(val_texts)

    # Include all document text nodes in the transductive graph.
    all_texts = train_texts + val_texts + test_texts

    train_indices = np.arange(num_train)
    val_indices = np.arange(num_train, num_train + num_val)

    print(f"Train documents: {num_train}")
    print(f"Validation documents: {num_val}")
    print(f"Test documents: {len(test_texts)}")

    # 2. Build vocabulary and TF-IDF features.
    print("\nBuilding vocabulary...")
    vocabulary = build_vocabulary(
        all_texts,
        min_df=5,
        max_features=20000,
    )
    print(f"Vocabulary size: {len(vocabulary)}")

    print("\nBuilding TF-IDF matrix...")
    tfidf_matrix, _ = build_tfidf_matrix(all_texts, vocabulary)

    # 3. Construct the TextGCN graph.
    print("\nBuilding document-word graph...")
    document_word = build_document_word_adjacency(tfidf_matrix)

    print("\nBuilding word-word PMI graph...")
    word_word = build_word_word_pmi(
        all_texts,
        vocabulary,
        window_size=20,
    )

    print("\nCombining and normalizing graph...")
    adjacency = combine_graph_adjacencies(document_word, word_word)
    adjacency = normalize_adjacency(adjacency)

    print(f"Graph shape: {adjacency.shape}")
    print(f"Graph non-zero entries: {adjacency.nnz}")

    print("\nConverting graph to PyTorch sparse tensor...")
    adjacency_tensor = scipy_to_torch_sparse(adjacency)

    # 4. Train using training labels; validation labels select the checkpoint.
    print("\nStarting TextGCN training...")
    start_time = time.perf_counter()

    model, history = train_textgcn(
        adjacency=adjacency_tensor,
        train_labels=train_labels,
        val_labels=val_labels,
        train_indices=train_indices,
        val_indices=val_indices,
        hidden_dim=200,
        dropout=0.5,
        learning_rate=0.02,
        epochs=100,
        patience=10,
    )

    training_runtime = time.perf_counter() - start_time

    # 5. Evaluate on validation data only.
    model.eval()
    with torch.no_grad():
        logits = model(adjacency_tensor)
        val_logits = logits[val_indices]
        val_predictions = val_logits.argmax(dim=1).cpu().numpy()
        val_probabilities = torch.softmax(
            val_logits, dim=1
        )[:, 1].cpu().numpy()

    val_metrics = compute_metrics(
        val_labels,
        val_predictions,
        y_score=val_probabilities,
    )

    print("\nValidation results:")
    for name, value in val_metrics.items():
        print(f"{name}: {value:.4f}" if isinstance(value, float)
              else f"{name}: {value}")

    # 6. Save checkpoint in the ignored models/ directory.
    checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "textgcn_covid.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_nodes": adjacency.shape[0],
            "vocabulary": vocabulary,
            "hidden_dim": 200,
            "num_classes": 2,
            "dropout": 0.5,
            "best_val_f1": val_metrics["f1_macro"],
        },
        checkpoint_path,
    )

    # 7. Save training history.
    output_dir = Path("results/graph")
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "textgcn_history.json"
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    # 8. Log validation metrics in the team's shared format.
    run_path = log_run(
        model="textgcn",
        dataset="covid",
        split="val",
        metrics=val_metrics,
        params={
            "hidden_dim": 200,
            "dropout": 0.5,
            "learning_rate": 0.02,
            "epochs": len(history),
            "patience": 10,
            "min_df": 5,
            "max_features": 20000,
            "window_size": 20,
            "seed": 42,
        },
        runtime_s=training_runtime,
        author="Sneha",
        notes=(
            "Transductive TextGCN. All document texts included in graph "
            "construction; only training labels used for gradient updates. "
            "Validation labels used for checkpoint selection."
        ),
    )

    print("\nExperiment completed!")
    print(f"Training time: {training_runtime:.2f} seconds")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training history: {history_path}")
    print(f"Validation run record: {run_path}")


if __name__ == "__main__":
    main()
