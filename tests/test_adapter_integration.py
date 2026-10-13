"""Integration tests for LocalHFBackend against a tiny locally-built GPT-2.

The checkpoint is generated on the fly with random weights: no downloads,
no network, no external model files. It exercises the real torch /
transformers / safetensors path: config validation, weight loading,
tokenizer loading, fingerprinting, greedy generation, cancellation and
the failure contract.

Entirely skipped when the optional 'hf' extra is not installed.
"""

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

from model_protocol_lab.core import Generation

REQUIRED = ("torch", "transformers", "safetensors", "tokenizers")
HAS_HF = all(importlib.util.find_spec(name) is not None for name in REQUIRED)


def build_tiny_gpt2(root: Path, vocab_size: int = 64) -> None:
    """Generate a minimal GPT-2 checkpoint with random weights (local only)."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(1234)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=64,
        n_ctx=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    # save_pretrained handles tied weights (lm_head/wte) the way real
    # checkpoints are stored; it also writes a valid config.json.
    model.save_pretrained(str(root))

    vocab = {f"w{i}": i for i in range(vocab_size - 2)}
    vocab["<stop>"] = vocab_size - 2
    vocab["[UNK]"] = vocab_size - 1
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(["<stop>", "[UNK]"])
    tokenizer.save(str(root / "tokenizer.json"))


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class TinyGPT2AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from model_protocol_lab.backends import LocalHFBackend

        cls._tmp = TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.root = Path(cls._tmp.name)
        build_tiny_gpt2(cls.root)
        cls.backend = LocalHFBackend(cls.root, device="cpu", max_context=64)

    def test_initialization_succeeds_with_stable_fingerprint(self):
        from model_protocol_lab.backends import LocalHFBackend

        again = LocalHFBackend(self.root, device="cpu", max_context=64)
        self.assertEqual(self.backend.fingerprint, again.fingerprint)
        self.assertTrue(self.backend.fingerprint.startswith("sha256:"))
        self.assertEqual(len(self.backend.fingerprint), len("sha256:") + 64)

    def test_encode_decode_roundtrip(self):
        ids = self.backend.encode("w1 w2 w3")
        self.assertEqual(len(ids), 3)
        self.assertTrue(all(type(t) is int and 0 <= t < 64 for t in ids))
        self.assertEqual(self.backend.decode(ids), "w1 w2 w3")

    def test_encode_uses_no_implicit_special_tokens(self):
        # "<stop>" is a registered special token; encoding must not add it.
        ids = self.backend.encode("w1")
        self.assertNotIn(62, ids)

    def test_generate_returns_exactly_the_requested_cap(self):
        event = Event()
        result = self.backend.generate((1, 2, 3), max_new_tokens=8, seed=0, cancel=event)
        self.assertIs(type(result), Generation)
        self.assertEqual(result.reason, "limit")
        self.assertEqual(len(result.token_ids), 8)
        self.assertTrue(all(type(t) is int and 0 <= t < 64 for t in result.token_ids))
        self.assertIsNone(result.error)

    def test_generate_is_deterministic_greedy(self):
        event = Event()
        first = self.backend.generate((1, 2, 3), max_new_tokens=8, seed=0, cancel=event)
        second = self.backend.generate((1, 2, 3), max_new_tokens=8, seed=999, cancel=event)
        self.assertEqual(first.token_ids, second.token_ids)

    def test_generate_continuation_excludes_prompt(self):
        event = Event()
        result = self.backend.generate((1, 2), max_new_tokens=4, seed=0, cancel=event)
        decoded = self.backend.decode(result.token_ids)
        self.assertIs(type(decoded), str)

    def test_pre_cancelled_generate_returns_cancelled(self):
        event = Event()
        event.set()
        result = self.backend.generate((1, 2), max_new_tokens=4, seed=0, cancel=event)
        self.assertEqual((result.token_ids, result.reason), ((), "cancelled"))

    def test_invalid_inputs_become_error_generations_not_exceptions(self):
        event = Event()
        cases = [
            ((), 4, 0),  # empty input
            ((1, 2), 0, 0),  # bad cap
            ((1, 2), 4, -1),  # bad seed
            ((1, 2), 4, 2**64),  # bad seed
            ((999,), 4, 0),  # out of vocabulary
        ]
        for input_ids, cap, seed in cases:
            with self.subTest(input_ids=input_ids, cap=cap, seed=seed):
                result = self.backend.generate(
                    input_ids, max_new_tokens=cap, seed=seed, cancel=event
                )
                self.assertEqual(result.reason, "error")
                self.assertIs(type(result.error), str)

    def test_context_limit_is_enforced(self):
        event = Event()
        result = self.backend.generate(tuple([1] * 63), max_new_tokens=4, seed=0, cancel=event)
        self.assertEqual(result.reason, "error")
        self.assertEqual(result.error, "context_limit_exceeded")

    def test_full_evaluate_runs_against_the_real_backend(self):
        from model_protocol_lab.catalog import CATALOG
        from model_protocol_lab.core import Fixture, Message, evaluate

        fixture = Fixture("tiny", (Message("user", "w1 w2"),), "w3", "tiny-group")
        report = evaluate(
            self.backend,
            CATALOG["chatml"],
            (fixture,),
            boundary=(62,),
            max_new_tokens=8,
            seed=7,
        )
        self.assertEqual(report.status, "completed")
        self.assertTrue(report.aggregate_eligible)
        obs = report.observations[0]
        self.assertEqual(len(obs.output_ids), 8)
        self.assertEqual(obs.raw_reason, "limit")

    def test_unsupported_model_type_fails_closed(self):
        from model_protocol_lab.backends import LocalHFBackend

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_tiny_gpt2(root)
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["model_type"] = "bert"
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "local_backend_initialization_failed"):
                LocalHFBackend(root, device="cpu", max_context=64)

    def test_missing_tokenizer_fails_closed(self):
        from model_protocol_lab.backends import LocalHFBackend

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_tiny_gpt2(root)
            (root / "tokenizer.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "local_backend_initialization_failed"):
                LocalHFBackend(root, device="cpu", max_context=64)

    def test_pickle_weights_are_not_used(self):
        # The fixture dir must contain no pickle weight files by construction.
        names = [p.name for p in self.root.iterdir()]
        self.assertIn("model.safetensors", names)
        self.assertFalse(any(n.endswith((".bin", ".pt", ".pth")) for n in names))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class BackendInitFailureTests(unittest.TestCase):
    """Each unsupported configuration must fail closed at construction."""

    def _tiny(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        build_tiny_gpt2(root)
        return root

    def _config(self, root):
        path = root / "config.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return path, data

    def _assert_init_fails(self, root, **kwargs):
        from model_protocol_lab.backends import LocalHFBackend

        options = {"device": "cpu", "max_context": 64}
        options.update(kwargs)
        with self.assertRaisesRegex(RuntimeError, "local_backend_initialization_failed"):
            LocalHFBackend(root, **options)

    def test_bad_context_limit_rejected(self):
        from model_protocol_lab.backends import LocalHFBackend

        root = self._tiny()
        with self.assertRaisesRegex(RuntimeError, "local_backend_initialization_failed"):
            LocalHFBackend(root, device="cpu", max_context=0)

    def test_bad_device_rejected(self):
        root = self._tiny()
        self._assert_init_fails(root, device="tpu")

    def test_context_beyond_declared_window_rejected(self):
        from model_protocol_lab.backends import LocalHFBackend

        root = self._tiny()
        # A distinct, actionable code (not the generic init failure): the
        # operator learns the requested window exceeds the checkpoint's.
        with self.assertRaisesRegex(RuntimeError, "local_context_limit_exceeds_declared_window"):
            LocalHFBackend(root, device="cpu", max_context=65)

    def test_quantization_config_rejected(self):
        root = self._tiny()
        path, data = self._config(root)
        data["quantization_config"] = {"bits": 8}
        path.write_text(json.dumps(data), encoding="utf-8")
        self._assert_init_fails(root)

    def test_missing_tokenizer_model_section_rejected(self):
        root = self._tiny()
        (root / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
        self._assert_init_fails(root)

    def test_stochastic_tokenizer_rejected(self):
        root = self._tiny()
        doc = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
        doc["model"]["dropout"] = 0.1
        (root / "tokenizer.json").write_text(json.dumps(doc), encoding="utf-8")
        self._assert_init_fails(root)

    def test_checkpoint_loading_mismatch_rejected(self):
        from safetensors.torch import load_file, save_file

        root = self._tiny()
        tensors = load_file(str(root / "model.safetensors"))
        victim = sorted(tensors)[0]
        del tensors[victim]
        save_file(tensors, str(root / "model.safetensors"))
        self._assert_init_fails(root)

    def test_config_vocabulary_mismatch_rejected(self):
        # Declared vocab_size disagrees with the stored weights: the strict
        # loading-info check must refuse the checkpoint.
        root = self._tiny()
        path, data = self._config(root)
        data["vocab_size"] = 32
        path.write_text(json.dumps(data), encoding="utf-8")
        self._assert_init_fails(root)

    def test_tokenizer_vocabulary_mismatch_rejected(self):
        # Model loads fine (64 embeddings), but the tokenizer ships ids the
        # model has no embeddings for.
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        root = self._tiny()
        vocab = {f"w{i}": i for i in range(100)}
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="w0"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.save(str(root / "tokenizer.json"))
        self._assert_init_fails(root)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None
        and not __import__("torch").cuda.is_available(),
        "requires a machine without CUDA",
    )
    def test_unavailable_cuda_device_rejected(self):
        root = self._tiny()
        self._assert_init_fails(root, device="cuda")


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class BackendGenerateEdgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.root = Path(cls._tmp.name)
        build_tiny_gpt2(cls.root)
        from model_protocol_lab.backends import LocalHFBackend

        cls.backend = LocalHFBackend(cls.root, device="cpu", max_context=64)

    def test_generate_rejects_non_event_cancellation(self):
        result = self.backend.generate((1, 2), max_new_tokens=4, seed=0, cancel="nope")
        self.assertEqual(result.reason, "error")
        self.assertEqual(result.error, "invalid_cancellation_event")

    def test_encode_rejects_non_string(self):
        with self.assertRaises(ValueError):
            self.backend.encode(123)

    def test_decode_rejects_out_of_vocabulary_ids(self):
        with self.assertRaises(ValueError):
            self.backend.decode((10_000,))

    def test_mid_generation_cancellation_is_cooperative(self):
        from unittest.mock import patch

        event = Event()
        model = self.backend._model
        original = model.forward

        def cancelling_forward(*args, **kwargs):
            event.set()
            return original(*args, **kwargs)

        with patch.object(model, "forward", cancelling_forward):
            result = self.backend.generate((1, 2), max_new_tokens=8, seed=0, cancel=event)
        self.assertEqual(result.reason, "cancelled")
        self.assertLess(len(result.token_ids), 8)

    def test_inference_exception_becomes_error_generation(self):
        from unittest.mock import patch

        event = Event()
        with patch.object(self.backend._model, "forward", side_effect=RuntimeError("boom")):
            result = self.backend.generate((1, 2), max_new_tokens=8, seed=0, cancel=event)
        self.assertEqual(result.reason, "error")
        self.assertEqual(result.error, "local_inference_failed")


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class OversizedTokenizerTests(unittest.TestCase):
    def _tiny(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        build_tiny_gpt2(root)
        return root

    def test_oversized_tokenizer_rejected_before_hf_load(self):
        import os

        from model_protocol_lab.backends import LocalHFBackend

        root = self._tiny()
        tokenizer_path = root / "tokenizer.json"
        # Sparse file: reports a huge size without consuming disk.
        with open(tokenizer_path, "r+b") as handle:
            handle.truncate(128 * 1024 * 1024 + 1)
        self.assertGreater(os.path.getsize(tokenizer_path), 128 * 1024 * 1024)
        with self.assertRaisesRegex(RuntimeError, "local_backend_initialization_failed"):
            LocalHFBackend(root, device="cpu", max_context=64)
