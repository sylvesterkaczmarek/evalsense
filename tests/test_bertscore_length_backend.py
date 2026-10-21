"""Offline tokenizer and BERTScore checks using a tiny randomly initialised BERT."""

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, cast
import unittest
from unittest.mock import Mock, patch

from evalsense.evaluation.evaluators import BertScoreCalculator
from evalsense.evaluation.evaluators.bertscore import (
    _encoded_length,
    _input_limit,
    _load_length_context,
)

EXTRAS = all(
    importlib.util.find_spec(name) for name in ("bert_score", "torch", "transformers")
)


@unittest.skipUnless(EXTRAS, "transformers extra not installed")
class NativeLengthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import BertConfig, BertModel, BertTokenizer

        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name)
        vocab = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "patient",
            "has",
            "no",
            "pain",
            "fever",
            "anti",
            "##co",
            "##agulation",
            "cafe",
            "药",
        ]
        vocab_file = cls.path / "vocab.txt"
        vocab_file.write_text("\n".join(vocab) + "\n")
        tokenizer = BertTokenizer(vocab_file=str(vocab_file), model_max_length=8)
        tokenizer.save_pretrained(cls.path)
        torch.manual_seed(0)
        BertModel(
            BertConfig(
                vocab_size=len(vocab),
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                max_position_embeddings=8,
            )
        ).save_pretrained(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_boundary_and_full_tokenization_match_slow_and_fast_backends(self):
        from bert_score.utils import sent_encode

        for fast in (False, True):
            with self.subTest(fast=fast):
                tokenizer, limit = _load_length_context(str(self.path), fast)
                self.assertEqual(limit, 8)
                for text in (
                    "",
                    "  \n ",
                    " patient has no pain ",
                    "café 药",
                    "patient " * 6,
                ):
                    self.assertEqual(
                        _encoded_length(tokenizer, text),
                        len(sent_encode(tokenizer, text)),
                    )
                self.assertEqual(_encoded_length(tokenizer, "patient " * 6), 8)
                self.assertEqual(_encoded_length(tokenizer, "patient " * 7), 9)
                self.assertEqual(len(sent_encode(tokenizer, "patient " * 7)), 8)

    def test_special_tokens_and_subwords_are_counted_not_words(self):
        tokenizer, _ = _load_length_context(str(self.path), False)
        self.assertEqual(_encoded_length(tokenizer, "anticoagulation"), 5)
        self.assertEqual(
            _encoded_length(tokenizer, "anticoagulation anticoagulation"), 8
        )
        self.assertEqual(
            _encoded_length(tokenizer, "anticoagulation anticoagulation pain"), 9
        )

    def test_512_limit_is_not_hardcoded(self):
        for limit in (8, 512, 4096):
            with self.subTest(limit=limit):
                self.assertEqual(
                    _input_limit(
                        SimpleNamespace(model_max_length=limit),
                        SimpleNamespace(max_position_embeddings=limit),
                    ),
                    limit,
                )

    def test_smallest_finite_declared_limit_wins(self):
        cases = [
            (8, 16, None, 8),
            (16, 8, None, 8),
            (10**30, 8, None, 8),
            (10**30, None, 16, 16),
            (8, None, None, 8),
        ]
        for token_limit, model_limit, positions, expected in cases:
            with self.subTest(case=(token_limit, model_limit, positions)):
                result = _input_limit(
                    SimpleNamespace(model_max_length=token_limit),
                    SimpleNamespace(
                        max_position_embeddings=model_limit, n_positions=positions
                    ),
                )
                self.assertEqual(result, expected)

    def test_sentinels_and_invalid_limits_are_not_treated_as_capacity(self):
        for value in (None, True, False, 0, -1, "512", 512.0, float("inf"), 10**30):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "finite BERTScore input limit"),
            ):
                _input_limit(SimpleNamespace(model_max_length=value), SimpleNamespace())

    def test_model_config_is_used_when_tokenizer_has_no_finite_limit(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.path, use_fast=False)
        tokenizer.model_max_length = 10**30
        with patch("bert_score.utils.get_tokenizer", return_value=tokenizer):
            loaded, limit = _load_length_context(str(self.path), False)
        self.assertIs(loaded, tokenizer)
        self.assertEqual(limit, 8)
        self.assertEqual(tokenizer.model_max_length, 10**30)

    def test_loader_uses_resolved_tokenizer_path_and_fast_flag(self):
        fake = SimpleNamespace(name_or_path="resolved-model", model_max_length=16)
        with (
            patch(
                "bert_score.utils.get_tokenizer", return_value=fake
            ) as load_tokenizer,
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(max_position_embeddings=8),
            ) as load_config,
        ):
            _, limit = _load_length_context("scibert-fixture-alias", True)
        load_tokenizer.assert_called_once_with("scibert-fixture-alias", use_fast=True)
        load_config.assert_called_once_with("resolved-model")
        self.assertEqual(limit, 8)

    def test_preflight_does_not_modify_tokenizer_truncation_settings(self):
        from bert_score.utils import sent_encode

        tokenizer, _ = _load_length_context(str(self.path), True)
        tokenizer.truncation_side = "left"
        self.assertEqual(_encoded_length(tokenizer, "patient " * 7), 9)
        self.assertEqual(tokenizer.truncation_side, "left")
        self.assertEqual(tokenizer.model_max_length, 8)
        self.assertEqual(len(sent_encode(tokenizer, "patient " * 7)), 8)

    def test_gpt2_and_roberta_prefix_space_matches_native_tokenization(self):
        from bert_score.utils import sent_encode
        from tokenizers import ByteLevelBPETokenizer
        from transformers import GPT2Tokenizer, RobertaTokenizer

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bpe = ByteLevelBPETokenizer()
            bpe.train_from_iterator(
                [" patient has no pain", " patient has fever"] * 10,
                vocab_size=300,
                min_frequency=1,
                special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
            )
            bpe.save_model(tmp)
            for cls in (GPT2Tokenizer, RobertaTokenizer):
                tokenizer = cls(
                    vocab_file=str(directory / "vocab.json"),
                    merges_file=str(directory / "merges.txt"),
                    model_max_length=32,
                )
                for text in ("patient", " patient has no pain ", ""):
                    with self.subTest(tokenizer=cls.__name__, text=text):
                        self.assertEqual(
                            _encoded_length(tokenizer, text),
                            len(sent_encode(tokenizer, text)),
                        )
                self.assertGreater(_encoded_length(tokenizer, "patient " * 40), 32)
                self.assertEqual(len(sent_encode(tokenizer, "patient " * 40)), 32)

    def test_native_truncation_can_hide_a_changed_tail_and_guard_blocks_it(self):
        from bert_score import BERTScorer
        from torch import Tensor

        native = BERTScorer(
            model_type=str(self.path), num_layers=1, device="cpu", idf=False
        )
        prefix = "patient " * 6
        prediction, reference = prefix + "pain", prefix + "fever"
        # Both tails lie beyond the six content tokens that fit with CLS and SEP.
        p, r, f = cast(
            tuple[Tensor, Tensor, Tensor], native.score([prediction], [reference])
        )
        self.assertGreater(float(f.item()), 0.999)
        metric = Mock()
        with patch(
            "evalsense.evaluation.evaluators.bertscore.evaluate.load",
            return_value=metric,
        ):
            calculator = BertScoreCalculator(model_type=str(self.path), num_layers=1)
        with self.assertRaisesRegex(ValueError, "prediction has 9 tokens"):
            calculator.calculate(prediction=prediction, reference=reference)
        metric.compute.assert_not_called()

    def test_in_limit_scores_match_native_backend_without_pretruncating_text(self):
        from bert_score import BERTScorer
        from torch import Tensor

        native = BERTScorer(
            model_type=str(self.path), num_layers=1, device="cpu", idf=False
        )
        seen = []

        def compute(**kwargs: Any) -> dict[str, Any]:
            seen.append(kwargs)
            p, r, f = cast(
                tuple[Tensor, Tensor, Tensor],
                native.score(kwargs["predictions"], kwargs["references"]),
            )
            return {
                "precision": p.tolist(),
                "recall": r.tolist(),
                "f1": f.tolist(),
                "hashcode": native.hash,
            }

        metric = Mock()
        metric.compute.side_effect = compute
        with patch(
            "evalsense.evaluation.evaluators.bertscore.evaluate.load",
            return_value=metric,
        ):
            calculator = BertScoreCalculator(model_type=str(self.path), num_layers=1)
        prediction, reference = " patient has pain ", "patient has no pain"
        p, r, f = cast(
            tuple[Tensor, Tensor, Tensor], native.score([prediction], [reference])
        )
        score = calculator.calculate(
            prediction=prediction, reference=reference, device="cpu"
        )
        self.assertEqual(
            score.value,
            {
                "BERTScore Precision": float(p.item()),
                "BERTScore Recall": float(r.item()),
                "BERTScore F1": float(f.item()),
            },
        )
        self.assertEqual(score.metadata, {"hashcode": native.hash})
        self.assertEqual(seen[0]["predictions"], [prediction])
        self.assertEqual(seen[0]["references"], [reference])


if __name__ == "__main__":
    unittest.main()
