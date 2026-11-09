"""Run TextGCN or GAT on a supported fake-news detection task."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.config import LIAR_LABELS_6
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
    build_gat_edge_index,
    build_gat_node_features,
    train_textgcn,
    train_gat,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a TextGCN or GAT graph experiment."
    )
    parser.add_argument(
        "--model",
        choices=["textgcn", "gat"],
        required=True,
    )
    parser.add_argument(
        "--dataset",
        choices=["covid", "liar", "isot"],
        required=True,
    )
    parser.add_argument(
        "--task",
        choices=["binary", "multiclass"],
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--gat-features", choices=["random", "tfidf-svd"], default="random")
    parser.add_argument("--dropout", type=float, default=None, help="Override the model dropout rate.")
    parser.add_argument(
        "--isot-sample-size",
        type=int,
        default=10000,
        help="Number of ISOT training documents to sample; validation and test remain complete.",
    )
    return parser.parse_args()


def get_labels(splits, dataset, task):
    if task == "multiclass":
        if dataset != "liar":
            raise ValueError("The multiclass task is currently supported only for LIAR.")

        label_to_id = {
            label: index for index, label in enumerate(LIAR_LABELS_6)
        }

        def encode(frame):
            labels = frame["label_6"].astype(str).str.strip().str.lower()
            unknown = sorted(set(labels) - set(label_to_id))
            if unknown:
                raise ValueError(f"Unknown LIAR labels: {unknown}")
            return labels.map(label_to_id).to_numpy(dtype=np.int64)

        return (
            encode(splits["train"]),
            encode(splits["val"]),
            len(LIAR_LABELS_6),
        )

    return (
        splits["train"]["label"].to_numpy(dtype=np.int64),
        splits["val"]["label"].to_numpy(dtype=np.int64),
        2,
    )


def main():
    args = parse_args()

    if args.dataset in ("covid", "isot") and args.task != "binary":
        raise ValueError(f"{args.dataset.upper()} currently supports binary classification only.")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    task_name = (
        "liar6" if args.dataset == "liar" and args.task == "multiclass"
        else args.dataset
    )

    # Model-specific settings.
    if args.model == "textgcn":
        hidden_dim = 200
        dropout = 0.5
        learning_rate = 0.02
        patience = 10
        epochs = args.epochs or 100
    else:
        hidden_dim = 64
        dropout = 0.5
        learning_rate = 0.001
        patience = 15
        epochs = args.epochs or 100

    if args.dropout is not None:
        dropout = args.dropout

    num_classes = 6 if args.task == "multiclass" else 2

    print(f"\nModel: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Number of classes: {num_classes}")

    # 1. Load the existing official splits.
    print("\nLoading dataset...")
    splits = load_splits(args.dataset)

    # Reproducible, stratified sampling of the ISOT training split only.
    # Keep the complete official validation and test splits unchanged.
    if args.dataset == "isot":
        if args.isot_sample_size < 2:
            raise ValueError("ISOT training sample size must be at least 2.")

        train_frame = splits["train"]

        if args.isot_sample_size > len(train_frame):
            raise ValueError(
                f"Requested {args.isot_sample_size} ISOT training rows, "
                f"but only {len(train_frame)} are available."
            )

        sampled_train, _ = train_test_split(
            train_frame,
            train_size=args.isot_sample_size,
            stratify=train_frame["label"],
            random_state=42,
        )

        splits["train"] = sampled_train.sort_index().reset_index(drop=True)

        print(
            "ISOT split sizes:",
            {name: len(frame) for name, frame in splits.items()},
        )

    train_texts = splits["train"]["text"].fillna("").astype(str).tolist()
    val_texts = splits["val"]["text"].fillna("").astype(str).tolist()
    test_texts = splits["test"]["text"].fillna("").astype(str).tolist()

    train_labels, val_labels, num_classes = get_labels(
        splits, args.dataset, args.task
    )

    num_train = len(train_texts)
    num_val = len(val_texts)

    # Include all document texts in the transductive graph.
    all_texts = train_texts + val_texts + test_texts

    train_indices = np.arange(num_train)
    val_indices = np.arange(num_train, num_train + num_val)

    print(f"Train documents: {num_train}")
    print(f"Validation documents: {num_val}")
    print(f"Test documents: {len(test_texts)}")
    print(f"Total graph documents: {len(all_texts)}")

    # 2. Build vocabulary and TF-IDF features.
    print("\nBuilding vocabulary...")
    vocabulary = build_vocabulary(
        all_texts,
        min_df=args.min_df,
        max_features=args.max_features,
    )
    print(f"Vocabulary size: {len(vocabulary)}")

    if not vocabulary:
        raise ValueError("Vocabulary is empty. Check the dataset and min_df.")

    print("\nBuilding TF-IDF matrix...")
    tfidf_matrix, _ = build_tfidf_matrix(all_texts, vocabulary)

    # 3. Build the document-word and word-word graph.
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

    print(f"Graph shape: {adjacency.shape}")
    print(f"Graph non-zero entries: {adjacency.nnz}")

    # 4. Train the selected graph model.
    start_time = time.perf_counter()

    if args.model == "textgcn":
        print("\nConverting graph to PyTorch sparse tensor...")
        graph_input = scipy_to_torch_sparse(
            normalize_adjacency(adjacency)
        )

        print("\nStarting TextGCN training...")
        model, history = train_textgcn(
            adjacency=graph_input,
            train_labels=train_labels,
            val_labels=val_labels,
            train_indices=train_indices,
            val_indices=val_indices,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            patience=patience,
            num_classes=num_classes,
        )

    else:
        print("\nBuilding GAT edge index...")
        edge_index = build_gat_edge_index(
            adjacency,
            top_k=args.top_k,
        )
        print(f"Number of directed edges: {edge_index.shape[1]}")

        node_features = None
        if args.gat_features == "tfidf-svd":
            print("\nBuilding SVD node features...")
            node_features = build_gat_node_features(
                tfidf_matrix,
                n_components=hidden_dim,
            )
            print("Node feature shape:", tuple(node_features.shape))

        print("\nStarting GAT training...")
        model, history = train_gat(
            edge_index=edge_index,
            node_features=node_features,
            num_nodes=adjacency.shape[0],
            train_labels=train_labels,
            val_labels=val_labels,
            train_indices=train_indices,
            val_indices=val_indices,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            patience=patience,
            num_classes=num_classes,
        )

    training_runtime = time.perf_counter() - start_time

    # 5. Evaluate on validation data only.
    # The test split is included in the transductive graph as unlabeled text,
    # but its labels are not used for model selection or evaluation here.
    model.eval()
    with torch.no_grad():
        if args.model == "textgcn":
            logits = model(graph_input)
        else:
            logits = model(edge_index)

        val_logits = logits[val_indices]
        val_predictions = val_logits.argmax(dim=1).cpu().numpy()

        val_probabilities = None
        if args.task == "binary":
            val_probabilities = torch.softmax(
                val_logits, dim=1
            )[:, 1].cpu().numpy()

    val_metrics = compute_metrics(
        val_labels,
        val_predictions,
        y_score=val_probabilities,
        task=args.task,
    )

    print("\nValidation results:")
    for name, value in val_metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    # 6. Save a uniquely named checkpoint.
    checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"{args.model}_{task_name}"
    if args.model == "gat":
        run_name += f"_top{args.top_k}_lr{learning_rate:g}"
        if args.gat_features != "random":
            run_name += f"_{args.gat_features}"

    checkpoint_path = checkpoint_dir / f"{run_name}.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_nodes": adjacency.shape[0],
        "vocabulary": vocabulary,
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
        "dropout": dropout,
        "dataset": args.dataset,
        "task": args.task,
        "best_val_f1": val_metrics["f1_macro"],
        "model": args.model,
    }

    if args.model == "gat":
        checkpoint["top_k"] = args.top_k
        checkpoint["gat_features"] = args.gat_features

    torch.save(checkpoint, checkpoint_path)

    # 7. Save training history without overwriting older experiments.
    output_dir = Path("results/graph")
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / f"{run_name}_history.json"
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    # 8. Log the validation results using the shared project format.
    params = {
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "patience": patience,
        "min_df": args.min_df,
        "max_features": args.max_features,
        "window_size": 20,
        "seed": 42,
        "num_classes": num_classes,
    }

    if args.model == "gat":
        params["top_k"] = args.top_k

    run_path = log_run(
        model=args.model,
        dataset=task_name,
        split="val",
        metrics=val_metrics,
        params=params,
        runtime_s=training_runtime,
        author="Sneha",
        notes=(
            "Transductive graph construction includes train, validation "
            "and test document texts. Only training labels are used for "
            "gradient updates. Validation labels select the checkpoint. "
            "Test labels are not used for evaluation in this validation run."
        ),
    )

    print("\nExperiment completed!")
    print(f"Training time: {training_runtime:.2f} seconds")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    print(f"Validation run record: {run_path}")


if __name__ == "__main__":
    main()
