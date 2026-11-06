
"""Run the Top-20 GAT experiment on the COVID dataset."""

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
    build_gat_edge_index,
    train_gat,
)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Load the COVID dataset splits.
    print("Loading COVID dataset...")
    splits = load_splits("covid")

    train_texts = splits["train"]["text"].tolist()
    val_texts = splits["val"]["text"].tolist()
    test_texts = splits["test"]["text"].tolist()

    train_labels = splits["train"]["label"].to_numpy()
    val_labels = splits["val"]["label"].to_numpy()

    num_train = len(train_texts)
    num_val = len(val_texts)

    # Include all document texts in the transductive graph.
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

    # 3. Construct the unnormalised graph.
    print("\nBuilding document-word graph...")
    document_word = build_document_word_adjacency(tfidf_matrix)

    print("\nBuilding word-word PMI graph...")
    word_word = build_word_word_pmi(
        all_texts,
        vocabulary,
        window_size=20,
    )

    print("\nCombining graph...")
    adjacency = combine_graph_adjacencies(
        document_word,
        word_word,
    )

    print(f"Original graph shape: {adjacency.shape}")
    print(f"Original graph edges: {adjacency.nnz}")

    # 4. Prune graph using Top-20 neighbours per node.
    print("\nBuilding Top-20 sparse GAT edge list...")
    edge_index = build_gat_edge_index(
        adjacency,
        top_k=20,
    )

    num_nodes = adjacency.shape[0]

    print(f"Number of nodes: {num_nodes}")
    print(f"GAT edge index shape: {tuple(edge_index.shape)}")
    print(f"Number of GAT edges: {edge_index.shape[1]}")

    # 5. Train GAT.
    print("\nStarting Top-20 GAT training...")
    start_time = time.perf_counter()

    model, history = train_gat(
        edge_index=edge_index,
        num_nodes=num_nodes,
        train_labels=train_labels,
        val_labels=val_labels,
        train_indices=train_indices,
        val_indices=val_indices,
        hidden_dim=64,
        dropout=0.5,
        learning_rate=0.001,
        epochs=100,
        patience=15,
    )

    training_runtime = time.perf_counter() - start_time

    # 6. Evaluate on validation data only.
    model.eval()

    with torch.no_grad():
        logits = model(edge_index)
        val_logits = logits[val_indices]

        val_predictions = (
            val_logits.argmax(dim=1).cpu().numpy()
        )

        val_probabilities = (
            torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
        )

    val_metrics = compute_metrics(
        val_labels,
        val_predictions,
        y_score=val_probabilities,
    )

    print("\nTop-20 GAT validation results:")
    for name, value in val_metrics.items():
        print(
            f"{name}: {value:.4f}"
            if isinstance(value, float)
            else f"{name}: {value}"
        )

    # 7. Save checkpoint separately.
    checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "gat_lr001_top20_covid.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_nodes": num_nodes,
            "vocabulary": vocabulary,
            "hidden_dim": 64,
            "num_classes": 2,
            "dropout": 0.5,
            "top_k": 20,
            "learning_rate": 0.001,
            "best_val_f1": val_metrics["f1_macro"],
        },
        checkpoint_path,
    )

    # 8. Save separate training history.
    output_dir = Path("results/graph")
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "gat_lr001_top20_history.json"

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    # 9. Log validation metrics.
    run_path = log_run(
        model="gat",
        dataset="covid",
        split="val",
        metrics=val_metrics,
        params={
            "experiment": "top20_neighbours",
            "hidden_dim": 64,
            "dropout": 0.5,
            "learning_rate": 0.001,
            "epochs": len(history),
            "patience": 15,
            "min_df": 5,
            "max_features": 20000,
            "window_size": 20,
            "top_k": 20,
            "seed": 42,
        },
        runtime_s=training_runtime,
        author="Sneha",
        notes=(
            "Transductive sparse GAT with learning rate 0.001 "
            "and top-20 neighbours per node. All document texts "
            "included in graph construction; only training labels "
            "used for gradient updates. Validation labels used "
            "for checkpoint selection."
        ),
    )

    print("\nTop-20 GAT experiment completed!")
    print(f"Training time: {training_runtime:.2f} seconds")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training history: {history_path}")
    print(f"Validation run record: {run_path}")


if __name__ == "__main__":
    main()
