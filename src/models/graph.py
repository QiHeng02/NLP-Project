
from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from math import log

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix, eye
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    """Convert a document into lowercase word tokens."""
    if not isinstance(text, str):
        return []

    return _TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(
    texts: list[str],
    min_df: int = 5,
    max_features: int | None = None,
) -> dict[str, int]:
    """Build a deterministic word-to-index vocabulary."""
    document_frequency = Counter()

    for text in texts:
        tokens = set(tokenize(text))
        document_frequency.update(tokens)

    words = [
        (word, frequency)
        for word, frequency in document_frequency.items()
        if frequency >= min_df
    ]

    words.sort(key=lambda item: (-item[1], item[0]))

    if max_features is not None:
        words = words[:max_features]

    return {word: index for index, (word, _) in enumerate(words)}


def build_tfidf_matrix(
    texts: list[str],
    vocabulary: dict[str, int],
):
    """Build a sparse document-word TF-IDF matrix."""
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        vocabulary=vocabulary,
    )

    return vectorizer.fit_transform(texts), vectorizer


def build_document_word_adjacency(tfidf_matrix):
    """Create a bipartite adjacency matrix from document-word TF-IDF."""
    num_docs, num_words = tfidf_matrix.shape

    doc_word = tfidf_matrix.tocoo()

    rows = list(doc_word.row)
    cols = list(doc_word.col + num_docs)
    values = list(doc_word.data)

    # Add reverse edges to make the graph undirected.
    rows += cols.copy()
    cols += list(doc_word.row)
    values += list(doc_word.data)

    adjacency = coo_matrix(
        (values, (rows, cols)),
        shape=(num_docs + num_words, num_docs + num_words),
    ).tocsr()

    return adjacency


def build_word_word_pmi(
    texts: list[str],
    vocabulary: dict[str, int],
    window_size: int = 20,
):
    """Build a symmetric word-word PPMI matrix using sliding windows."""
    if window_size < 2:
        raise ValueError("window_size must be at least 2")

    word_frequency = Counter()
    pair_frequency = Counter()
    total_windows = 0

    for text in texts:
        tokens = [word for word in tokenize(text) if word in vocabulary]

        if not tokens:
            continue

        if len(tokens) <= window_size:
            windows = [tokens]
        else:
            windows = [
                tokens[i:i + window_size]
                for i in range(len(tokens) - window_size + 1)
            ]

        for window in windows:
            unique_words = sorted(set(window))

            if not unique_words:
                continue

            total_windows += 1
            word_frequency.update(unique_words)
            pair_frequency.update(combinations(unique_words, 2))

    rows = []
    cols = []
    values = []

    if total_windows == 0:
        return coo_matrix(
            (len(vocabulary), len(vocabulary))
        ).tocsr()

    for (word_a, word_b), pair_count in pair_frequency.items():
        pmi = log(
            (pair_count * total_windows)
            / (word_frequency[word_a] * word_frequency[word_b])
        )

        if pmi > 0:
            i = vocabulary[word_a]
            j = vocabulary[word_b]

            rows.extend([i, j])
            cols.extend([j, i])
            values.extend([pmi, pmi])

    adjacency = coo_matrix(
        (values, (rows, cols)),
        shape=(len(vocabulary), len(vocabulary)),
    ).tocsr()

    return adjacency


def combine_graph_adjacencies(document_word_adjacency, word_word_pmi):
    """Combine document-word and word-word edges into one graph."""
    num_docs = document_word_adjacency.shape[0] - word_word_pmi.shape[0]

    if num_docs < 0:
        raise ValueError("Incompatible adjacency matrix dimensions")

    if document_word_adjacency.shape != (
        num_docs + word_word_pmi.shape[0],
        num_docs + word_word_pmi.shape[0],
    ):
        raise ValueError("Document-word adjacency has an invalid shape")

    pmi = word_word_pmi.tocoo()

    word_word_full = coo_matrix(
        (
            pmi.data,
            (pmi.row + num_docs, pmi.col + num_docs),
        ),
        shape=document_word_adjacency.shape,
    ).tocsr()

    adjacency = (document_word_adjacency + word_word_full).tocsr()

    return adjacency


def normalize_adjacency(adjacency):
    """Add self-loops and apply symmetric GCN normalization."""
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency matrix must be square")

    adjacency = adjacency.tocsr()

    adjacency_with_self_loops = adjacency + eye(
        adjacency.shape[0],
        format="csr",
    )

    degrees = np.asarray(
        adjacency_with_self_loops.sum(axis=1)
    ).flatten()

    inv_sqrt_degrees = np.zeros_like(degrees, dtype=float)
    nonzero = degrees > 0
    inv_sqrt_degrees[nonzero] = degrees[nonzero] ** -0.5

    normalized = adjacency_with_self_loops.multiply(
        inv_sqrt_degrees[:, None]
    ).multiply(
        inv_sqrt_degrees[None, :]
    )

    return normalized.tocsr()


def scipy_to_torch_sparse(matrix):
    """Convert a SciPy sparse matrix into a PyTorch sparse tensor."""
    matrix = matrix.tocoo()

    indices = np.vstack((matrix.row, matrix.col)).astype(np.int64)
    values = matrix.data.astype(np.float32, copy=False)

    return torch.sparse_coo_tensor(
        torch.from_numpy(indices),
        torch.from_numpy(values),
        size=matrix.shape,
        dtype=torch.float32,
        check_invariants=False,
    ).coalesce()


class TextGCN(nn.Module):
    """Two-layer TextGCN for graph-based classification."""

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 200,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.node_embeddings = nn.Parameter(
            torch.empty(num_nodes, hidden_dim)
        )
        nn.init.xavier_uniform_(self.node_embeddings)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, adjacency):
        hidden = torch.sparse.mm(
            adjacency,
            self.node_embeddings,
        )
        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)

        hidden = torch.sparse.mm(adjacency, hidden)

        logits = self.classifier(hidden)
        return logits


def train_textgcn(
    adjacency,
    train_labels,
    val_labels,
    train_indices,
    val_indices,
    hidden_dim=200,
    dropout=0.5,
    learning_rate=0.02,
    epochs=100,
    patience=10,
    num_classes=2,
):
    """Train TextGCN using training labels and validation Macro-F1."""

    torch.manual_seed(42)
    device = torch.device("cpu")

    adjacency = adjacency.to(device)

    train_labels = torch.tensor(
        train_labels, dtype=torch.long, device=device
    )
    val_labels = torch.tensor(
        val_labels, dtype=torch.long, device=device
    )

    train_indices = torch.tensor(
        train_indices, dtype=torch.long, device=device
    )
    val_indices = torch.tensor(
        val_indices, dtype=torch.long, device=device
    )

    model = TextGCN(
        num_nodes=adjacency.shape[0],
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(adjacency)

        train_loss = criterion(
            logits[train_indices],
            train_labels,
        )

        train_loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            val_logits = model(adjacency)
            val_predictions = val_logits[val_indices].argmax(dim=1)

            val_f1 = f1_score(
                val_labels.cpu().numpy(),
                val_predictions.cpu().numpy(),
                average="macro",
                zero_division=0,
            )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss.item(),
            "val_macro_f1": val_f1,
        })

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {train_loss.item():.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1

            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation Macro-F1: {best_val_f1:.4f}")

    return model, history


def build_gat_edge_index(adjacency, top_k=10):
    """
    Keep the strongest top_k neighbours per node, then make the
    graph symmetric and add self-loops.

    Returns edge_index where row 0 = destination and row 1 = source.
    """
    adjacency = adjacency.tocsr()
    num_nodes = adjacency.shape[0]

    rows = []
    cols = []
    values = []

    for node in range(num_nodes):
        start = adjacency.indptr[node]
        end = adjacency.indptr[node + 1]

        neighbours = adjacency.indices[start:end]
        weights = adjacency.data[start:end]

        if len(neighbours) > top_k:
            selected = np.argpartition(weights, -top_k)[-top_k:]
            neighbours = neighbours[selected]
            weights = weights[selected]

        rows.extend([node] * len(neighbours))
        cols.extend(neighbours.tolist())
        values.extend(weights.tolist())

    pruned = sp.csr_matrix(
        (values, (rows, cols)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )

    pruned = pruned.maximum(pruned.T)
    pruned.setdiag(1.0)
    pruned.eliminate_zeros()

    coo = pruned.tocoo()

    edge_index = np.vstack((coo.row, coo.col)).astype(np.int64)

    return torch.from_numpy(edge_index)


class GATLayer(nn.Module):
    """Single-head sparse Graph Attention layer."""

    def __init__(self, input_dim, output_dim, dropout=0.5):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim, bias=False)

        self.attention_source = nn.Parameter(torch.empty(output_dim))
        self.attention_target = nn.Parameter(torch.empty(output_dim))

        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attention_source.unsqueeze(0))
        nn.init.xavier_uniform_(self.attention_target.unsqueeze(0))

    def forward(self, features, edge_index):
        num_nodes = features.shape[0]

        hidden = self.linear(self.dropout(features))

        destination = edge_index[0]
        source = edge_index[1]

        source_score = (hidden * self.attention_source).sum(dim=1)
        target_score = (hidden * self.attention_target).sum(dim=1)

        scores = nn.functional.leaky_relu(
            target_score[destination] + source_score[source],
            negative_slope=0.2,
        )

        max_scores = torch.full(
            (num_nodes,),
            float("-inf"),
            device=features.device,
        )

        max_scores.scatter_reduce_(
            0,
            destination,
            scores.detach(),
            reduce="amax",
            include_self=True,
        )

        exp_scores = torch.exp(scores - max_scores[destination])

        denominator = torch.zeros(
            num_nodes,
            device=features.device,
        ).index_add_(0, destination, exp_scores)

        attention = exp_scores / (denominator[destination] + 1e-16)

        messages = attention.unsqueeze(1) * hidden[source]

        output = torch.zeros(
            (num_nodes, hidden.shape[1]),
            device=features.device,
        ).index_add_(0, destination, messages)

        return output


def build_gat_node_features(tfidf_matrix, n_components=64):
    """Create shared SVD features for document and word nodes."""
    from sklearn.decomposition import TruncatedSVD

    num_documents, vocab_size = tfidf_matrix.shape
    n_components = min(n_components, num_documents - 1, vocab_size - 1)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    document_features = svd.fit_transform(tfidf_matrix).astype(np.float32)
    word_features = (
        svd.components_.T * svd.singular_values_
    ).astype(np.float32)

    document_features /= np.maximum(
        np.linalg.norm(document_features, axis=1, keepdims=True), 1e-12
    )
    word_features /= np.maximum(
        np.linalg.norm(word_features, axis=1, keepdims=True), 1e-12
    )

    return torch.tensor(
        np.vstack([document_features, word_features]),
        dtype=torch.float32,
    )


class GraphAttentionNetwork(nn.Module):
    """Two-layer sparse GAT for graph-based classification."""

    def __init__(
        self,
        num_nodes,
        hidden_dim=64,
        num_classes=2,
        dropout=0.5,
        node_features=None,
    ):
        super().__init__()

        if node_features is None:
            self.node_embeddings = nn.Parameter(
                torch.empty(num_nodes, hidden_dim)
            )
            nn.init.xavier_uniform_(self.node_embeddings)
            self.register_buffer("node_features", None)
        else:
            if node_features.shape != (num_nodes, hidden_dim):
                raise ValueError(
                    f"Expected node features with shape "
                    f"({num_nodes}, {hidden_dim}), got "
                    f"{tuple(node_features.shape)}"
                )
            self.node_embeddings = None
            self.register_buffer("node_features", node_features)

        self.dropout = nn.Dropout(dropout)
        self.gat1 = GATLayer(hidden_dim, hidden_dim, dropout)
        self.gat2 = GATLayer(hidden_dim, num_classes, dropout)

    def forward(self, edge_index):
        features = (
            self.node_embeddings
            if self.node_features is None
            else self.node_features
        )
        hidden = self.gat1(features, edge_index)
        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)
        return self.gat2(hidden, edge_index)


def train_gat(
    edge_index,
    num_nodes,
    train_labels,
    val_labels,
    train_indices,
    val_indices,
    hidden_dim=64,
    dropout=0.5,
    learning_rate=0.005,
    epochs=100,
    patience=10,
    num_classes=2,
    node_features=None,
):
    """Train GAT using training labels and validation Macro-F1."""

    torch.manual_seed(42)
    device = torch.device("cpu")

    edge_index = edge_index.to(device)

    train_labels = torch.tensor(
        train_labels, dtype=torch.long, device=device
    )
    val_labels = torch.tensor(
        val_labels, dtype=torch.long, device=device
    )

    train_indices = torch.tensor(
        train_indices, dtype=torch.long, device=device
    )
    val_indices = torch.tensor(
        val_indices, dtype=torch.long, device=device
    )

    model = GraphAttentionNetwork(
        num_nodes=num_nodes,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
        node_features=node_features,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=5e-4,
    )

    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(edge_index)

        train_loss = criterion(
            logits[train_indices],
            train_labels,
        )

        train_loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            val_logits = model(edge_index)
            val_predictions = val_logits[val_indices].argmax(dim=1)

            val_f1 = f1_score(
                val_labels.cpu().numpy(),
                val_predictions.cpu().numpy(),
                average="macro",
                zero_division=0,
            )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss.item(),
            "val_macro_f1": val_f1,
        })

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {train_loss.item():.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1

            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation Macro-F1: {best_val_f1:.4f}")

    return model, history
