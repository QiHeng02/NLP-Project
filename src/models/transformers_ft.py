"""Transformers: fine-tuned BERT-base, DistilBERT and RoBERTa-base (Role E).

Pipeline:  text -> sub-word tokens -> pretrained transformer encoder -> [CLS] vector -> linear -> class
Unlike TF-IDF or GloVe + RNN, every word's vector depends on the whole sentence around it
(self-attention), and the encoder was pretrained on billions of words before we fine-tune it.

`TransformerClassifier` follows the shared interface (see src/models/README.md):
    fit(texts, labels, val_texts, val_labels)
    predict(texts)        -> np.ndarray of labels
    predict_proba(texts)  -> np.ndarray (n, n_classes), columns ordered like `classes_`
It takes raw strings and tokenises internally, so LIME/SHAP and the Streamlit app can call it directly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from src.config import SEED

# short name -> Hugging Face model id
MODEL_IDS = {
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
}
TRUNCATIONS = ("head", "head_tail")  # keep the first max_len tokens, or first part + last part


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TransformerClassifier:
    """Shared-interface wrapper around a Hugging Face sequence classifier.

    Handles tokenising (incl. head + tail truncation for long articles), batching,
    mixed precision, warmup + linear decay, and early stopping on val macro-F1.
    """

    _PARAM_NAMES = ("model_name", "max_len", "truncation", "head_frac", "lr", "batch_size",
                    "eval_batch_size", "epochs", "patience", "warmup_ratio", "weight_decay",
                    "grad_accum", "grad_clip", "fp16", "seed")

    def __init__(
        self,
        model_name: str = "distilbert",
        *,
        max_len: int = 128,
        truncation: str = "head",
        head_frac: float = 0.25,
        lr: float = 2e-5,
        batch_size: int = 16,
        eval_batch_size: int = 64,
        epochs: int = 3,
        patience: int = 2,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        grad_accum: int = 1,
        grad_clip: float = 1.0,
        fp16: bool = True,
        seed: int = SEED,
        device: str | None = None,
        verbose: bool = True,
    ):
        if truncation not in TRUNCATIONS:
            raise ValueError(f"truncation must be one of {TRUNCATIONS}, got {truncation!r}")
        self.model_name, self.max_len, self.truncation, self.head_frac = model_name, max_len, truncation, head_frac
        self.lr, self.batch_size, self.eval_batch_size = lr, batch_size, eval_batch_size
        self.epochs, self.patience = epochs, patience
        self.warmup_ratio, self.weight_decay = warmup_ratio, weight_decay
        self.grad_accum, self.grad_clip, self.fp16, self.seed = grad_accum, grad_clip, fp16, seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose

    @property
    def hf_id(self) -> str:
        """Short names (bert/distilbert/roberta) map to HF ids; anything else is used as given."""
        return MODEL_IDS.get(self.model_name, self.model_name)

    @property
    def _amp(self) -> bool:
        return self.fp16 and self.device.type == "cuda"

    def get_params(self) -> dict:
        """Hyperparameters, for log_run and for saving checkpoints."""
        return {k: getattr(self, k) for k in self._PARAM_NAMES}

    # ------------------------------------------------------------ data
    def _encode(self, texts) -> list[list[int]]:
        """Text -> token ids incl. special tokens ([CLS] ... [SEP]), at most max_len long."""
        texts = [str(t) if str(t).strip() else "[empty]" for t in texts]
        if self.truncation == "head":  # fast path: let the tokenizer cut
            enc = self.tokenizer_(texts, truncation=True, max_length=self.max_len)
            return enc["input_ids"]
        # head_tail: tokenise in full, then keep the start and the end of long texts
        # (Sun et al., 2019, "How to Fine-Tune BERT for Text Classification?")
        prefix, suffix = self._special_tokens()
        budget = self.max_len - len(prefix) - len(suffix)
        raw = self.tokenizer_(texts, add_special_tokens=False, truncation=False,
                              verbose=False)["input_ids"]
        n_head = int(round(budget * self.head_frac))
        out = []
        for ids in raw:
            if len(ids) > budget:
                ids = ids[:n_head] + ids[len(ids) - (budget - n_head):]
            out.append(prefix + ids + suffix)
        return out

    def _special_tokens(self) -> tuple[list[int], list[int]]:
        """Ids the tokenizer adds around one text, e.g. [CLS] / [SEP] (BERT) or <s> / </s> (RoBERTa).

        Found by comparing an encoding with and without special tokens, so it works for any
        model and transformers version.
        """
        plain = self.tokenizer_("hello", add_special_tokens=False)["input_ids"]
        full = self.tokenizer_("hello")["input_ids"]
        for i in range(len(full) - len(plain) + 1):
            if full[i:i + len(plain)] == plain:
                return full[:i], full[i + len(plain):]
        raise ValueError("could not locate the special tokens of this tokenizer")

    def _batches(self, seqs, labels=None, batch_size=16, rng=None):
        """Yield (input_ids, attention_mask, labels), padded to the longest text in each batch."""
        if rng is not None:
            order = rng.permutation(len(seqs))
        else:  # inference: sort by length so batches need little padding (faster), restore order later
            order = np.argsort([len(s) for s in seqs], kind="stable")
        pad = self.tokenizer_.pad_token_id
        for start in range(0, len(seqs), batch_size):
            idx = order[start:start + batch_size]
            width = max(len(seqs[i]) for i in idx)
            ids = torch.full((len(idx), width), pad, dtype=torch.long)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            for row, i in enumerate(idx):
                ids[row, : len(seqs[i])] = torch.tensor(seqs[i])
                mask[row, : len(seqs[i])] = 1
            y = None if labels is None else torch.as_tensor(labels[idx])
            yield idx, ids.to(self.device), mask.to(self.device), y

    def _build(self, source: str | Path) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer_ = AutoTokenizer.from_pretrained(source)
        self.model_ = AutoModelForSequenceClassification.from_pretrained(
            source, num_labels=len(self.classes_), ignore_mismatched_sizes=True).to(self.device)

    # ------------------------------------------------------------ training
    def fit(self, texts, labels, val_texts=None, val_labels=None) -> "TransformerClassifier":
        """Fine-tune with AdamW + linear warmup/decay. With val data, keep the best epoch by val macro-F1."""
        from transformers import get_linear_schedule_with_warmup

        _set_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        # like sklearn: remember the original labels, train on 0..k-1
        self.classes_, y = np.unique(np.asarray(labels), return_inverse=True)
        self._build(self.hf_id)

        t0 = time.perf_counter()
        train_seqs = self._encode(list(texts))
        self.tokenize_s_ = round(time.perf_counter() - t0, 1)
        lengths = np.array([len(s) for s in train_seqs])
        self.truncated_share_ = float((lengths >= self.max_len).mean())  # how many texts hit the limit
        if self.verbose:
            print(f"    tokenised {len(train_seqs)} texts in {self.tokenize_s_}s; "
                  f"{self.truncated_share_:.0%} reach max_len={self.max_len}")

        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        groups = [
            {"params": [p for n, p in self.model_.named_parameters() if not any(k in n for k in no_decay)],
             "weight_decay": self.weight_decay},
            {"params": [p for n, p in self.model_.named_parameters() if any(k in n for k in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(groups, lr=self.lr)
        steps_per_epoch = int(np.ceil(len(train_seqs) / self.batch_size / self.grad_accum))
        total = steps_per_epoch * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(self.warmup_ratio * total), total)
        scaler = torch.amp.GradScaler("cuda", enabled=self._amp)
        loss_fn = nn.CrossEntropyLoss()
        val_labels = None if val_labels is None else np.asarray(val_labels)

        best_f1, best_state, bad_epochs = -1.0, None, 0
        self.history_, self.best_epoch_ = [], self.epochs
        for epoch in range(1, self.epochs + 1):
            t0 = time.perf_counter()
            self.model_.train()
            total_loss, n_seen = 0.0, 0
            optimizer.zero_grad()
            for step, (_, ids, mask, yb) in enumerate(
                    self._batches(train_seqs, y, self.batch_size, rng=rng), start=1):
                yb = yb.to(self.device)
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=self._amp):
                    logits = self.model_(input_ids=ids, attention_mask=mask).logits
                    loss = loss_fn(logits.float(), yb) / self.grad_accum
                scaler.scale(loss).backward()
                if step % self.grad_accum == 0 or step * self.batch_size >= len(train_seqs):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model_.parameters(), self.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                total_loss += loss.item() * self.grad_accum * len(yb)
                n_seen += len(yb)
                if self.verbose and step % 200 == 0:
                    print(f"      step {step}  loss={total_loss / n_seen:.4f}", flush=True)

            record = {"epoch": epoch, "train_loss": round(total_loss / n_seen, 4)}
            if val_texts is not None:
                record["val_f1_macro"] = round(
                    f1_score(val_labels, self.predict(val_texts), average="macro"), 4)
            record["time_s"] = round(time.perf_counter() - t0, 1)
            self.history_.append(record)
            if self.verbose:
                print("   ", "  ".join(f"{k}={v}" for k, v in record.items()), flush=True)

            if val_texts is None:
                continue
            if record["val_f1_macro"] > best_f1:
                best_f1, bad_epochs, self.best_epoch_ = record["val_f1_macro"], 0, epoch
                # keep a CPU copy of the best weights (BERT-base ~440 MB)
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
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
        out = np.zeros((len(seqs), len(self.classes_)), dtype=np.float32)
        for idx, ids, mask, _ in self._batches(seqs, batch_size=self.eval_batch_size):
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self._amp):
                logits = self.model_(input_ids=ids, attention_mask=mask).logits
            out[idx] = torch.softmax(logits.float(), dim=1).cpu().numpy()
        return out

    def predict(self, texts) -> np.ndarray:
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]

    def measure_latency(self, texts, n: int = 50) -> float:
        """Median milliseconds to classify ONE input (as the Streamlit demo does). Requirement R4."""
        texts = list(texts)[:n]
        self.predict_proba(texts[:2])  # warm-up
        times = []
        for t in texts:
            t0 = time.perf_counter()
            self.predict_proba([t])
            times.append(time.perf_counter() - t0)
        return round(1000 * float(np.median(times)), 1)

    # ------------------------------------------------------------ checkpoints
    def save(self, path: str | Path) -> Path:
        """Save to a folder: HF weights + tokenizer + our settings (classes, truncation, max_len)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model_.save_pretrained(path)
        self.tokenizer_.save_pretrained(path)
        meta = {"config": self.get_params(), "classes": self.classes_.tolist()}
        (path / "wrapper.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "TransformerClassifier":
        path = Path(path)
        meta = json.loads((path / "wrapper.json").read_text(encoding="utf-8"))
        model = cls(**meta["config"], device=device, verbose=False)
        model.classes_ = np.array(meta["classes"])
        model._build(path)
        model.model_.eval()
        return model
