"""Sequential deep learning: LSTM, BiLSTM and GRU on GloVe embeddings (Role C).

Pipeline:  text -> tokens -> word ids -> GloVe vectors -> RNN -> linear -> class
The RNN reads words in order, so unlike TF-IDF it can use word order.

`SequentialClassifier` follows the shared interface (see src/models/README.md):
    fit(texts, labels, val_texts, val_labels)
    predict(texts)        -> np.ndarray of labels
    predict_proba(texts)  -> np.ndarray (n, n_classes), columns ordered like `classes_`
It takes raw strings, so LIME/SHAP and the Streamlit app can call it directly.
"""
from __future__ import annotations

import copy
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.config import DATA_RAW, SEED

GLOVE_PATH = DATA_RAW / "glove" / "glove.6B.100d.txt"
ARCHS = ("lstm", "bilstm", "gru")
TRUNCATIONS = ("head", "head_tail")  # keep the first max_len words, or first half + last half
PAD, UNK = 0, 1  # reserved word ids: padding and unknown word

# GloVe 6B is lower-cased with punctuation split off, so tokenise the same way
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(str(text).lower())


def load_glove(path: Path = GLOVE_PATH) -> dict[str, np.ndarray]:
    """Read the GloVe text file into {word: vector}. Slow (~15 s), so load once and reuse."""
    vectors = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, *values = line.rstrip().split(" ")
            vectors[word] = np.asarray(values, dtype=np.float32)
    return vectors


def build_vocab(texts, min_freq: int = 2, max_size: int | None = 50_000) -> dict[str, int]:
    """Map frequent words to ids, starting at 2 (0 = PAD, 1 = UNK). Build from TRAIN only."""
    counts = Counter(tok for t in texts for tok in tokenize(t))
    words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
    return {w: i for i, w in enumerate(words, start=2)}


def embedding_matrix(vocab: dict[str, int], glove: dict[str, np.ndarray], dim: int,
                     seed: int = SEED) -> tuple[np.ndarray, float]:
    """One row per word id: its GloVe vector, or a small random vector if GloVe lacks it.

    Returns (matrix, coverage) where coverage = share of vocab words found in GloVe.
    """
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.1, (len(vocab) + 2, dim)).astype(np.float32)
    matrix[PAD] = 0
    hits = 0
    for word, i in vocab.items():
        if word in glove:
            matrix[i] = glove[word]
            hits += 1
    return matrix, hits / max(len(vocab), 1)


class RNNClassifier(nn.Module):
    """Embedding -> LSTM / BiLSTM / GRU -> dropout -> linear."""

    def __init__(self, embeddings: np.ndarray, arch: str, hidden_size: int, n_classes: int,
                 dropout: float, freeze_embeddings: bool):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(embeddings), freeze=freeze_embeddings, padding_idx=PAD)
        rnn_cls = nn.GRU if arch == "gru" else nn.LSTM
        bidirectional = arch == "bilstm"
        self.rnn = rnn_cls(embeddings.shape[1], hidden_size, batch_first=True,
                           bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), n_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embedding(ids))
        # packing makes the RNN skip padding, so its final state comes from the real last word
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        if isinstance(h, tuple):  # LSTM returns (hidden, cell); GRU returns hidden only
            h = h[0]
        # h is (directions, batch, hidden): join forward and backward final states
        h = torch.cat(list(h), dim=1)
        return self.fc(self.dropout(h))


class SequentialClassifier:
    """Shared-interface wrapper: handles tokenising, batching, training and early stopping."""

    _PARAM_NAMES = ("arch", "embed_dim", "hidden_size", "dropout", "max_len", "truncation",
                    "freeze_embeddings", "lr", "batch_size", "max_epochs", "patience", "min_freq",
                    "max_vocab", "grad_clip", "seed")

    def __init__(
        self,
        arch: str = "bilstm",
        *,
        glove: dict[str, np.ndarray] | None = None,
        embed_dim: int = 100,
        hidden_size: int = 128,
        dropout: float = 0.3,
        max_len: int = 128,
        truncation: str = "head",
        freeze_embeddings: bool = True,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 20,
        patience: int = 3,
        min_freq: int = 2,
        max_vocab: int | None = 50_000,
        grad_clip: float = 1.0,
        seed: int = SEED,
        device: str | None = None,
        verbose: bool = True,
    ):
        if arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}, got {arch!r}")
        if truncation not in TRUNCATIONS:
            raise ValueError(f"truncation must be one of {TRUNCATIONS}, got {truncation!r}")
        self.arch, self.glove, self.embed_dim, self.hidden_size = arch, glove, embed_dim, hidden_size
        self.dropout, self.max_len, self.truncation = dropout, max_len, truncation
        self.freeze_embeddings = freeze_embeddings
        self.lr, self.batch_size, self.max_epochs, self.patience = lr, batch_size, max_epochs, patience
        self.min_freq, self.max_vocab, self.grad_clip, self.seed = min_freq, max_vocab, grad_clip, seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose

    def get_params(self) -> dict:
        """Hyperparameters, for log_run and for saving checkpoints."""
        return {k: getattr(self, k) for k in self._PARAM_NAMES}

    # ------------------------------------------------------------ data
    def _truncate(self, tokens: list[str]) -> list[str]:
        if len(tokens) <= self.max_len:
            return tokens
        if self.truncation == "head_tail":  # start + end of long articles
            head = self.max_len // 2
            return tokens[:head] + tokens[-(self.max_len - head):]
        return tokens[: self.max_len]

    def _encode(self, texts) -> list[list[int]]:
        """Text -> list of word ids, cut to max_len."""
        seqs = []
        for t in texts:
            ids = [self.vocab_.get(tok, UNK) for tok in self._truncate(tokenize(t))]
            seqs.append(ids or [UNK])  # empty text -> one unknown token
        return seqs

    def _batches(self, seqs, labels=None, rng=None):
        """Yield (ids, lengths, labels) batches, padded to the longest text in each batch."""
        order = rng.permutation(len(seqs)) if rng is not None else np.arange(len(seqs))
        for start in range(0, len(seqs), self.batch_size):
            idx = order[start:start + self.batch_size]
            lengths = torch.tensor([len(seqs[i]) for i in idx])
            ids = torch.full((len(idx), int(lengths.max())), PAD, dtype=torch.long)
            for row, i in enumerate(idx):
                ids[row, : len(seqs[i])] = torch.tensor(seqs[i])
            y = None if labels is None else torch.as_tensor(labels[idx], device=self.device)
            yield ids.to(self.device), lengths, y

    def _build(self, embeddings: np.ndarray) -> None:
        self.model_ = RNNClassifier(embeddings, self.arch, self.hidden_size, len(self.classes_),
                                    self.dropout, self.freeze_embeddings).to(self.device)

    # ------------------------------------------------------------ training
    def fit(self, texts, labels, val_texts=None, val_labels=None) -> "SequentialClassifier":
        """Train with Adam. If val data is given, early-stop on val macro-F1 and keep the best epoch."""
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        texts = list(texts)
        # like sklearn: remember the original labels, train on 0..k-1
        self.classes_, y = np.unique(np.asarray(labels), return_inverse=True)

        self.vocab_ = build_vocab(texts, self.min_freq, self.max_vocab)
        glove = self.glove if self.glove is not None else load_glove()
        embeddings, self.glove_coverage_ = embedding_matrix(self.vocab_, glove, self.embed_dim, self.seed)
        self._build(embeddings)

        params = [p for p in self.model_.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        train_seqs = self._encode(texts)
        val_labels = None if val_labels is None else np.asarray(val_labels)

        best_f1, best_state, bad_epochs = -1.0, None, 0
        self.history_, self.best_epoch_ = [], self.max_epochs
        for epoch in range(1, self.max_epochs + 1):
            t0 = time.perf_counter()
            self.model_.train()
            total_loss = 0.0
            for ids, lengths, yb in self._batches(train_seqs, y, rng=rng):
                optimizer.zero_grad()
                loss = loss_fn(self.model_(ids, lengths), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(params, self.grad_clip)  # guards against exploding gradients
                optimizer.step()
                total_loss += loss.item() * len(yb)

            record = {"epoch": epoch, "train_loss": round(total_loss / len(train_seqs), 4)}
            if val_texts is not None:
                record["val_f1_macro"] = round(
                    f1_score(val_labels, self.predict(val_texts), average="macro"), 4)
            record["time_s"] = round(time.perf_counter() - t0, 1)
            self.history_.append(record)
            if self.verbose:
                print("   ", "  ".join(f"{k}={v}" for k, v in record.items()))

            if val_texts is None:
                continue
            # early stopping: stop after `patience` epochs with no val improvement
            if record["val_f1_macro"] > best_f1:
                best_f1, bad_epochs, self.best_epoch_ = record["val_f1_macro"], 0, epoch
                best_state = copy.deepcopy(self.model_.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)  # roll back to the best epoch
        return self

    # ------------------------------------------------------------ prediction
    @torch.no_grad()
    def predict_proba(self, texts) -> np.ndarray:
        self.model_.eval()
        seqs = self._encode(list(texts))
        probs = [torch.softmax(self.model_(ids, lengths), dim=1).cpu()
                 for ids, lengths, _ in self._batches(seqs)]
        return torch.cat(probs).numpy()

    def predict(self, texts) -> np.ndarray:
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]

    # ------------------------------------------------------------ checkpoints
    def save(self, path: Path) -> Path:
        """Save weights + vocab + settings (no GloVe file needed to reload)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.get_params(), "vocab": self.vocab_,
                    "classes": self.classes_.tolist(), "state_dict": self.model_.state_dict()}, path)
        return path

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "SequentialClassifier":
        ckpt = torch.load(path, map_location="cpu")
        model = cls(**ckpt["config"], device=device, verbose=False)
        model.vocab_, model.classes_ = ckpt["vocab"], np.array(ckpt["classes"])
        # placeholder embeddings of the right shape; real weights come from the state dict
        model._build(np.zeros((len(model.vocab_) + 2, model.embed_dim), dtype=np.float32))
        model.model_.load_state_dict(ckpt["state_dict"])
        return model
