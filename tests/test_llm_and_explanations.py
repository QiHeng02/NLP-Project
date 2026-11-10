"""Zero-shot and explanation interface tests without model downloads."""
import html
import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.explain.words import fake_probability, highlighted_html, ordered_probabilities, shap_words
from src.models.llm_zero_shot import ZeroShotLLM


class ReversedModel:
    classes_ = np.array([1, 0])

    def predict_proba(self, texts):
        return np.tile([0.8, 0.2], (len(texts), 1))


class LLMExplanationTests(unittest.TestCase):
    def test_binary_label_only(self):
        model = ZeroShotLLM(generator=lambda system, user: "fake.")
        self.assertEqual(model.predict(["a claim"]), [1])
        self.assertIn("a claim", model.prompt("a claim")[1])
        with self.assertRaises(ValueError):
            ZeroShotLLM(generator=lambda *_: "I think fake because...").classify("x")

    def test_llm_generation_accepts_tokenizer_mapping(self):
        calls = {}
        class Inputs(dict):
            def to(self, device):
                return self

        class Tokenizer:
            eos_token_id = 0

            def apply_chat_template(self, messages, **kwargs):
                calls["return_dict"] = kwargs["return_dict"]
                return Inputs(input_ids=np.array([[11, 12]]), attention_mask=np.array([[1, 1]]))

            def encode(self, label, **kwargs):
                return {"real": [13], "fake": [14]}[label]

        class Model:
            def generate(self, **kwargs):
                calls["input_ids"] = kwargs["input_ids"]
                allowed = kwargs["prefix_allowed_tokens_fn"]
                assert allowed(0, np.array([11, 12])) == [13, 14]
                assert allowed(0, np.array([11, 12, 13])) == [0]
                return np.array([[11, 12, 13, 0]])

        llm = ZeroShotLLM()
        llm.tokenizer, llm.model, llm.device = Tokenizer(), Model(), "cpu"
        with patch.dict(sys.modules, {"torch": SimpleNamespace(inference_mode=nullcontext)}):
            self.assertEqual(llm.classify("example"), "real")
        self.assertTrue(calls["return_dict"])
        np.testing.assert_array_equal(calls["input_ids"], [[11, 12]])

    def test_long_prompt_keeps_answer_cue_at_the_end(self):
        calls = {}
        class Inputs(dict):
            def to(self, device):
                return self

        class Tokenizer:
            eos_token_id = 0

            def apply_chat_template(self, messages, **kwargs):
                self.ids = np.arange(100).reshape(1, -1)
                return Inputs(input_ids=self.ids, attention_mask=np.ones_like(self.ids))

            def encode(self, label, **kwargs):
                return {"real": [101], "fake": [100]}[label]

        class Model:
            def generate(self, **kwargs):
                calls["ids"] = kwargs["input_ids"]
                calls["mask"] = kwargs["attention_mask"]
                assert set(kwargs["prefix_allowed_tokens_fn"](0, kwargs["input_ids"][0])) == {100, 101}
                return np.concatenate((kwargs["input_ids"], [[100, 0]]), axis=1)

        llm = ZeroShotLLM(max_input_tokens=64)
        llm.tokenizer, llm.model, llm.device = Tokenizer(), Model(), "cpu"
        fake_torch = SimpleNamespace(inference_mode=nullcontext,
                                     cat=lambda arrays, dim: np.concatenate(arrays, axis=dim),
                                     ones_like=np.ones_like)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(llm.classify("long article"), "fake")
        self.assertEqual(calls["ids"].shape, (1, 64))
        np.testing.assert_array_equal(calls["ids"][0, :16], np.arange(16))
        np.testing.assert_array_equal(calls["ids"][0, -48:], np.arange(52, 100))
        np.testing.assert_array_equal(calls["mask"], np.ones((1, 64)))

    def test_six_class(self):
        model = ZeroShotLLM(task="liar6", generator=lambda *_: "half-true")
        self.assertEqual(model.predict(["claim"]), ["half-true"])

    def test_six_class_constrained_generation_excludes_invented_label(self):
        class Inputs(dict):
            def to(self, device):
                return self

        class Tokenizer:
            eos_token_id = 0

            def apply_chat_template(self, messages, **kwargs):
                return Inputs(input_ids=np.array([[11, 12]]), attention_mask=np.array([[1, 1]]))

            def encode(self, label, **kwargs):
                return [list(ZeroShotLLM(task="liar6").labels).index(label) + 20]

        class Model:
            def generate(self, **kwargs):
                allowed = kwargs["prefix_allowed_tokens_fn"]
                assert 99 not in allowed(0, np.array([11, 12]))
                assert allowed(0, np.array([11, 12, 23])) == [0]
                return np.array([[11, 12, 23, 0]])

        llm = ZeroShotLLM(task="liar6")
        llm.tokenizer, llm.model, llm.device = Tokenizer(), Model(), "cpu"
        with patch.dict(sys.modules, {"torch": SimpleNamespace(inference_mode=nullcontext)}):
            self.assertEqual(llm.classify("example"), "half-true")

    def test_probability_class_order(self):
        self.assertAlmostEqual(fake_probability(ReversedModel(), ["x"])[0], 0.8)
        np.testing.assert_allclose(ordered_probabilities(ReversedModel(), ["x"]), [[0.2, 0.8]])

    def test_html_escapes_input(self):
        markup = highlighted_html("<script>alert(1)</script> fake", [("fake", 0.2)])
        self.assertNotIn("<script>", markup)
        self.assertIn(html.escape("<script>"), markup)
        self.assertIn("<mark", markup)

    def test_shap_scalar_output_has_no_output_names(self):
        calls = {}
        class Explainer:
            def __init__(self, predict, masker, **kwargs):
                calls.update(kwargs)
                np.testing.assert_allclose(predict(["x"]), [0.8])

            def __call__(self, texts, **kwargs):
                return texts, kwargs

        fake_shap = SimpleNamespace(maskers=SimpleNamespace(Text=lambda pattern: pattern),
                                    Explainer=Explainer)
        with patch.dict(sys.modules, {"shap": fake_shap}):
            result = shap_words(ReversedModel(), "a claim", max_evals=120)
        self.assertNotIn("output_names", calls)
        self.assertEqual(result, (["a claim"], {"max_evals": 120}))


if __name__ == "__main__":
    unittest.main()
