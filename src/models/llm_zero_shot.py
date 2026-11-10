"""Deterministic, label-only zero-shot classification with a local instruction model.

No examples, gold labels, or dataset metadata are sent to the model. A generation
must be exactly one allowed label; malformed answers raise instead of silently
being counted as a class. This is a decision prediction, not a calibrated score.
"""
from __future__ import annotations

import re

from src.config import LIAR_LABELS_6

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class ZeroShotLLM:
    TRUNCATION_POLICY = "head_tail_preserve_assistant_v2"
    DECODING_POLICY = "constrained_label_tokens_v1"

    def __init__(self, model_id: str = DEFAULT_MODEL, *, task: str = "binary",
                 max_input_tokens: int = 1024, max_new_tokens: int = 12,
                 generator=None):
        if task not in ("binary", "liar6"):
            raise ValueError("task must be binary or liar6")
        self.model_id, self.task = model_id, task
        self.max_input_tokens, self.max_new_tokens = max_input_tokens, max_new_tokens
        self._generator = generator

    @property
    def labels(self):
        return ("real", "fake") if self.task == "binary" else tuple(LIAR_LABELS_6)

    def prompt(self, text: str) -> tuple[str, str]:
        if self.task == "binary":
            rule = ("Classify the text as real or fake. 'Fake' means it makes a false "
                    "or misleading factual claim; 'real' means it does not. "
                    "Judge the claim, not the writing style. If uncertain, choose the more likely label.")
        else:
            rule = ("Rate the factual truthfulness of the claim using exactly one label: "
                    "pants-fire, false, barely-true, half-true, mostly-true, true. "
                    "Choose the most likely label if uncertain.")
        system = rule + " Reply with only the label, no explanation."
        return system, f"Text to classify:\n{text}"

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _generate(self, system: str, user: str) -> str:
        import torch

        if not hasattr(self, "model"):
            self._load()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(self.device)
        # Right truncation can remove the final assistant header from a long
        # article. The LLM then continues the article instead of answering.
        ids = inputs["input_ids"]
        if ids.shape[-1] > self.max_input_tokens:
            if self.max_input_tokens < 64:
                raise ValueError("max_input_tokens must be at least 64")
            head = self.max_input_tokens // 4
            tail = self.max_input_tokens - head
            inputs["input_ids"] = torch.cat((ids[:, :head], ids[:, -tail:]), dim=-1)
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        prompt_length = inputs["input_ids"].shape[-1]
        candidates = {
            tuple(self.tokenizer.encode(label, add_special_tokens=False)): label
            for label in self.labels
        }
        if len(candidates) != len(self.labels) or any(not seq for seq in candidates):
            raise ValueError("The tokenizer cannot distinguish all task labels")
        if self.max_new_tokens <= max(map(len, candidates)):
            raise ValueError("max_new_tokens must fit the longest label plus EOS")

        def allowed_tokens(batch_id, sequence):
            prefix = tuple(sequence[prompt_length:].tolist())
            allowed = {seq[len(prefix)] for seq in candidates
                       if len(seq) > len(prefix) and seq[:len(prefix)] == prefix}
            if prefix in candidates:
                allowed.add(self.tokenizer.eos_token_id)
            return sorted(allowed)

        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                         do_sample=False,
                                         prefix_allowed_tokens_fn=allowed_tokens,
                                         pad_token_id=self.tokenizer.eos_token_id)
        generated = output[0, prompt_length:].tolist()
        if self.tokenizer.eos_token_id in generated:
            generated = generated[:generated.index(self.tokenizer.eos_token_id)]
        key = tuple(generated)
        if key not in candidates:
            raise ValueError(f"Constrained decoding returned invalid token ids: {key}")
        return candidates[key]

    def classify(self, text: str) -> str:
        system, user = self.prompt(str(text))
        raw = (self._generator or self._generate)(system, user)
        label = re.sub(r"[.!\s]+$", "", str(raw).strip().lower())
        if label not in self.labels:
            raise ValueError(f"Invalid LLM label {raw!r}; expected one of {self.labels}")
        return label

    def predict(self, texts):
        labels = [self.classify(text) for text in texts]
        if self.task == "binary":
            return [0 if label == "real" else 1 for label in labels]
        return labels
