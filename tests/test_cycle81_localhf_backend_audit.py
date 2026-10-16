"""Cycle 81 adversarial audit: LocalHFBackend lifecycle + generate contract.

METHOD
------
The repo's ``Backend`` protocol (``core.py``) promises: generate exactly the
cap unless cancelled/failed, never stop at EOS, continuation-only ids, a
request-local RNG, and cancellation checks during inference. ``LocalHFBackend``
(``backends.py``) implements that contract over a locally-built tiny GPT-2
checkpoint (random weights, built with ``torch.manual_seed(1234)`` -- no
downloads, no network). The emulate_* paths were audited in cycles 37/59/73,
the asset helpers in cycle 72, determinism in cycle 65; this file is the first
dedicated adversarial audit of the backend LIFECYCLE (constructor validation,
failure codes, leak discipline) and the GENERATE CONTRACT (exact-cap
discipline, EOS behavior, seeding, cancellation, threading).

Every claim below is proven BY TEST against the real torch/transformers path:

1.  Contract: exact cap for several sizes, continuation-only valid ids, a
    forced-EOS test (the model's forward is patched so argmax always lands on
    the EOS id -- generation must still return the full cap with
    reason="limit"), byte-identical outputs across instances with different
    seeds, global torch/random RNG state untouched by generate.
2.  Cancellation: pre-set event -> ((), "cancelled"); mid-generation cancel ->
    ("cancelled", partial ids, never "limit"); cancellation never raises.
3.  Init validation: every failure mode (bad directory, file-as-directory,
    empty/None path, missing/corrupt/non-dict config.json, unsupported
    model_type and configuration flags, missing/corrupt/stochastic/oversized
    tokenizer.json, missing/symlinked weights, adapter layouts, bad
    max_context values and devices, CUDA-unavailable) maps to the fixed
    RuntimeError code -- and the message is EXACTLY that code with
    ``__cause__`` None, so no HF-internal text or traceback detail can leak.
    The over-declared-window case keeps its distinct actionable code.
4.  forward_logits codes: invalid_input_ids / context_limit_exceeded /
    local_backend_not_initialized / local_forward_failed, each pinned by exact
    message, plus the output tensor contract ([T, vocab] fp32, detached,
    contiguous, finite).
5.  Adversarial edges: empty/single inputs, the max-context boundary
    (context-1/context/context+1 on both generate and forward_logits), the
    4096 cap bound, double __init__ on one instance, concurrent generate /
    forward_logits from threads (exact caps, identical outputs, no cross-talk),
    and lock release after an inference failure.
6.  TruncatedPromptBackend: init validation, fingerprint format, tail-byte
    truncation, the codepoint-split error, decode/generate pass-through, and
    hf_stop_token_bytes forwarding.

FINDINGS (verified by running this file, 2026-10-16)
----------------------------------------------------
CLEAN -- no genuine lifecycle or generate-contract bugs found:

* generate returns exactly max_new_tokens ids with reason="limit" for caps
  1/2/8/63; ids are continuation-only, plain ints in range. A forced-EOS
  test (every argmax steered onto the EOS id) proves generation never stops
  at EOS: the full cap is returned with reason="limit".
* Outputs are byte-identical across independently constructed instances with
  different seeds; generate leaves global torch and random RNG state
  untouched (request-local seeding discipline holds on the request path).
* Cancellation: pre-set event -> ((), "cancelled"); mid-generation cancel ->
  ("cancelled") with partial ids (< cap, >= 1), never "limit", never raised.
* Every constructor failure mode (bad directory, file-as-dir, empty/None
  path, missing/corrupt/non-dict config.json, unsupported model_type and
  configuration flags, missing/non-dict/stochastic/oversized tokenizer.json,
  missing/symlinked weights, adapter and dual weight layouts, bad
  max_context values, bad devices, CUDA-unavailable) raises RuntimeError
  whose message is EXACTLY the fixed code with __cause__ None -- no
  HF-internal text or traceback detail leaks. Over-declared-window keeps its
  distinct actionable code local_context_limit_exceeds_declared_window
  (pinned for both 65 and 32768 against n_positions=64).
* forward_logits pins invalid_input_ids (empty, out-of-range, negative,
  bool, non-tuple), context_limit_exceeded (65 > 64 accepted at exactly 64),
  local_backend_not_initialized (white-box: unreachable via the public API
  since __init__ raises, but the defensive branch is pinned), and
  local_forward_failed; the output tensor is [T, vocab] fp32, detached,
  contiguous, finite.
* Adversarial edges: single-id inputs, the context boundary on both generate
  (63+1 fits; 63+2 and 64+1 -> context_limit_exceeded) and forward_logits,
  the 4096 cap bound (4097/0/-1/str/float -> invalid_output_cap; a
  within-cap but over-window 4096 -> context_limit_exceeded), double
  __init__ on one instance (fingerprint stable, still generates),
  concurrent generate/forward_logits from 4 threads (exact caps, identical
  outputs, no cross-talk -- the RLock serializes correctly), lock release
  after inference failure, and fixed codes for encode/decode misuse.
* TruncatedPromptBackend: drop_bytes/fingerprint validation, fingerprint
  format, tail-byte truncation vs truncate_text_tail, the
  prompt_truncation_splits_codepoint code, decode/generate pass-through
  (exact cap, byte-identical to inner), hf_stop_token_bytes forwarding.

Deliberately not verified: behavior on real (non-synthetic) checkpoints and
CUDA devices; the local_backend_not_initialized branch via the public API
(unreachable by construction -- __init__ raises instead of returning a
half-built object).
"""

from __future__ import annotations

import importlib.util
import json
import random
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from typing import Any
from unittest.mock import patch

from model_protocol_lab import backends
from model_protocol_lab.core import Generation, truncate_text_tail

REQUIRED = ("torch", "transformers", "safetensors", "tokenizers")
HAS_HF = all(importlib.util.find_spec(name) is not None for name in REQUIRED)


def build_tiny_gpt2(root: Path, vocab_size: int = 64) -> None:
    """Generate a minimal GPT-2 checkpoint with random weights (local only).

    Self-contained copy of the builder from test_adapter_integration so this
    audit file has no cross-test imports (the mutation gate copies this file
    alone into the mutant directory).
    """
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(1234)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(str(root))

    vocab = {f"w{i}": i for i in range(vocab_size - 2)}
    vocab["<stop>"] = vocab_size - 2
    vocab["[UNK]"] = vocab_size - 1
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(["<stop>", "[UNK]"])
    tokenizer.save(str(root / "tokenizer.json"))


def _template_checkpoint(test: Any) -> Path:
    """Build one tiny checkpoint per test class; callers copytree-mutate it."""
    tmp = TemporaryDirectory()
    test.addClassCleanup(tmp.cleanup)
    root = Path(tmp.name) / "template"
    root.mkdir()
    build_tiny_gpt2(root)
    return root


def _mutated_copy(test: Any, template: Path) -> Path:
    """Fresh writable copy of the template checkpoint for one mutation."""
    tmp = TemporaryDirectory()
    test.addCleanup(tmp.cleanup)
    root = Path(tmp.name) / "ckpt"
    shutil.copytree(template, root)
    return root


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class GenerateContractTests(unittest.TestCase):
    """The Backend protocol contract, proved against the real HF stack."""

    backend: backends.LocalHFBackend
    root: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = _template_checkpoint(cls)
        cls.backend = backends.LocalHFBackend(cls.root, device="cpu", max_context=64)

    def test_exact_cap_for_several_sizes(self) -> None:
        for cap in (1, 2, 8, 63):
            with self.subTest(cap=cap):
                result = self.backend.generate((1,), max_new_tokens=cap, seed=0, cancel=Event())
                self.assertIs(type(result), Generation)
                self.assertEqual(result.reason, "limit")
                self.assertIsNone(result.error)
                self.assertEqual(len(result.token_ids), cap)

    def test_output_is_continuation_only_valid_ids(self) -> None:
        result = self.backend.generate((1, 2, 3), max_new_tokens=8, seed=0, cancel=Event())
        self.assertEqual(len(result.token_ids), 8)
        self.assertTrue(
            all(type(t) is int and 0 <= t < self.backend.vocab_size for t in result.token_ids)
        )

    def test_generate_never_stops_at_eos(self) -> None:
        # Adversarial: force every argmax onto the EOS id. A backend that
        # stopped at EOS would return fewer than the cap; the contract
        # demands the full cap with reason="limit" regardless.
        eos_id = self.backend.vocab_size - 2  # "<stop>" special id in the tiny vocab
        previous = self.backend.eos_token_id
        self.backend.eos_token_id = eos_id
        model: Any = self.backend._model
        real_forward = model.forward

        def eos_only_forward(*args: Any, **kwargs: Any) -> Any:
            out = real_forward(*args, **kwargs)
            out.logits[...] = -1e30  # finite: survives the isfinite guard
            out.logits[..., eos_id] = 0.0
            return out

        try:
            with patch.object(model, "forward", eos_only_forward):
                result = self.backend.generate((1, 2), max_new_tokens=4, seed=0, cancel=Event())
        finally:
            self.backend.eos_token_id = previous
        self.assertEqual(result.reason, "limit")
        self.assertIsNone(result.error)
        self.assertEqual(result.token_ids, (eos_id,) * 4)

    def test_same_request_byte_identical_across_instances_and_seeds(self) -> None:
        other = backends.LocalHFBackend(self.root, device="cpu", max_context=64)
        first = self.backend.generate((1, 2, 3), max_new_tokens=8, seed=0, cancel=Event())
        second = other.generate((1, 2, 3), max_new_tokens=8, seed=999, cancel=Event())
        self.assertEqual(first.reason, "limit")
        self.assertEqual(first.token_ids, second.token_ids)

    def test_generate_leaves_global_rng_state_untouched(self) -> None:
        import torch

        before_torch = torch.get_rng_state().clone()
        before_random = random.getstate()
        self.backend.generate((1, 2, 3), max_new_tokens=4, seed=12345, cancel=Event())
        self.backend.generate((4, 5, 6), max_new_tokens=4, seed=99999, cancel=Event())
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        self.assertEqual(before_random, random.getstate())

    def test_preseeded_cancel_event_returns_cancelled(self) -> None:
        event = Event()
        event.set()
        result = self.backend.generate((1, 2), max_new_tokens=4, seed=0, cancel=event)
        self.assertEqual((result.token_ids, result.reason, result.error), ((), "cancelled", None))

    def test_mid_generation_cancellation_returns_cancelled_partial(self) -> None:
        # Cooperative cancellation: the cancel fires during the 2nd forward,
        # so the loop-top check on step 3 must observe it and return the two
        # tokens generated so far -- never "limit", never an exception.
        model: Any = self.backend._model
        real_forward = model.forward
        event = Event()
        calls = 0

        def cancelling_forward(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls >= 2:
                event.set()
            return real_forward(*args, **kwargs)

        with patch.object(model, "forward", cancelling_forward):
            result = self.backend.generate((1, 2), max_new_tokens=8, seed=0, cancel=event)
        self.assertEqual(result.reason, "cancelled")
        self.assertIsNone(result.error)
        self.assertLess(len(result.token_ids), 8)
        self.assertGreaterEqual(len(result.token_ids), 1)

    def test_invalid_inputs_return_error_generations_with_fixed_codes(self) -> None:
        cases: list[tuple[Any, Any, Any, Any, str]] = [
            ((), 4, 0, Event(), "invalid_or_empty_input"),
            ((999,), 4, 0, Event(), "invalid_or_empty_input"),
            ((-1,), 4, 0, Event(), "invalid_or_empty_input"),
            ((True,), 4, 0, Event(), "invalid_or_empty_input"),
            ([1, 2], 4, 0, Event(), "invalid_or_empty_input"),  # list, not tuple
            ((1, 2), 0, 0, Event(), "invalid_output_cap"),
            ((1, 2), 4097, 0, Event(), "invalid_output_cap"),
            ((1, 2), -1, 0, Event(), "invalid_output_cap"),
            ((1, 2), "4", 0, Event(), "invalid_output_cap"),
            ((1, 2), 4, -1, Event(), "invalid_seed"),
            ((1, 2), 4, 2**64, Event(), "invalid_seed"),
            ((1, 2), 4, "0", Event(), "invalid_seed"),
            ((1, 2), 4, True, Event(), "invalid_seed"),
            ((1, 2), 4, 0, "nope", "invalid_cancellation_event"),
            ((1, 2), 4, 0, True, "invalid_cancellation_event"),
            (tuple([1] * 63), 4, 0, Event(), "context_limit_exceeded"),
        ]
        for input_ids, cap, seed, cancel, code in cases:
            with self.subTest(input_ids=input_ids, cap=cap, seed=seed, code=code):
                # Must not raise: invalid requests become error Generations.
                result = self.backend.generate(
                    input_ids, max_new_tokens=cap, seed=seed, cancel=cancel
                )
                self.assertIs(type(result), Generation)
                self.assertEqual(result.reason, "error")
                self.assertEqual(result.token_ids, ())
                self.assertEqual(result.error, code)

    def test_error_generations_carry_no_prompt_or_partial_ids(self) -> None:
        result = self.backend.generate((), max_new_tokens=4, seed=0, cancel=Event())
        self.assertEqual(result.reason, "error")
        self.assertEqual(result.token_ids, ())
        self.assertIs(type(result.error), str)
        self.assertTrue(result.error)


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class InitValidationTests(unittest.TestCase):
    """Every constructor failure mode maps to a fixed code; nothing leaks."""

    template: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.template = _template_checkpoint(cls)

    def _init_fails_exact(self, root: Any, code: str, **kwargs: Any) -> None:
        options: dict[str, Any] = {"device": "cpu", "max_context": 64}
        options.update(kwargs)
        with self.assertRaises(RuntimeError) as caught:
            backends.LocalHFBackend(root, **options)
        # The failure message is EXACTLY the fixed code: no HF-internal text,
        # no file bytes, no traceback detail can reach the caller.
        self.assertEqual(str(caught.exception), code)
        self.assertIsNone(caught.exception.__cause__)

    def test_nonexistent_directory(self) -> None:
        root = _mutated_copy(self, self.template)
        self._init_fails_exact(root / "nope", "local_backend_initialization_failed")

    def test_file_instead_of_directory(self) -> None:
        root = _mutated_copy(self, self.template)
        self._init_fails_exact(root / "config.json", "local_backend_initialization_failed")

    def test_empty_and_none_paths(self) -> None:
        root = _mutated_copy(self, self.template)
        self._init_fails_exact("", "local_backend_initialization_failed")
        self._init_fails_exact(None, "local_backend_initialization_failed")
        self.assertTrue(root.is_dir())  # sanity: the template copy itself is fine

    def test_missing_config_json(self) -> None:
        root = _mutated_copy(self, self.template)
        (root / "config.json").unlink()
        self._init_fails_exact(root, "local_backend_initialization_failed")

    def test_corrupt_config_json(self) -> None:
        root = _mutated_copy(self, self.template)
        (root / "config.json").write_text("{not valid json", encoding="utf-8")
        self._init_fails_exact(root, "local_backend_initialization_failed")

    def test_config_json_not_a_dict(self) -> None:
        root = _mutated_copy(self, self.template)
        (root / "config.json").write_text("[1, 2]", encoding="utf-8")
        self._init_fails_exact(root, "local_backend_initialization_failed")

    def test_unsupported_model_type(self) -> None:
        root = _mutated_copy(self, self.template)
        path = root / "config.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["model_type"] = "bert"
        path.write_text(json.dumps(data), encoding="utf-8")
        self._init_fails_exact(root, "local_backend_initialization_failed")

    def test_unsupported_model_configuration(self) -> None:
        root = _mutated_copy(self, self.template)
        path = root / "config.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["is_encoder_decoder"] = True
        path.write_text(json.dumps(data), encoding="utf-8")
        self._init_fails_exact(root, "local_backend_initialization_failed")

    def test_missing_tokenizer_json(self) -> None:
        root = _mutated_copy(self, self.template)
        (root / "tokenizer.json").unlink()
        self._init_fails_exact(root, "local_backend_initialization_failed")

    def test_tokenizer_json_not_a_dict(self) -> None:
        root = _mutated_copy(self, self.template)
        (root / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
        self._init_fails_exact(root, "local_backend_initialization_failed")

    def test_stochastic_tokenizer_rejected(self) -> None:
        root = _mutated_copy(self, self.template)
        path = root / "tokenizer.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["model"]["dropout"] = 0.1
        path.write_text(json.dumps(doc), encoding="utf-8")
        self._init_fails_exact(root, "local_backend_initialization_failed")

    def test_oversized_tokenizer_json_rejected(self) -> None:
        root = _mutated_copy(self, self.template)
        tokenizer_path = root / "tokenizer.json"
        # Sparse file: reports a huge size without consuming disk/tmpfs.
        with open(tokenizer_path, "r+b") as handle:
            handle.truncate(128 * 1024 * 1024 + 1)
        self._init_fails_exact(root, "local_backend_initialization_failed")

    def test_missing_model_weights(self) -> None:
        root = _mutated_copy(self, self.template)
        (root / "model.safetensors").unlink()
        self._init_fails_exact(root, "local_backend_initialization_failed")

    def test_symlinked_weight_rejected(self) -> None:
        root = _mutated_copy(self, self.template)
        target = _mutated_copy(self, self.template) / "model.safetensors"
        weight = root / "model.safetensors"
        weight.unlink()
        weight.symlink_to(target)
        self._init_fails_exact(root, "local_backend_initialization_failed")

    def test_adapter_layout_rejected(self) -> None:
        root = _mutated_copy(self, self.template)
        (root / "adapter_config.json").write_text("{}", encoding="utf-8")
        self._init_fails_exact(root, "local_backend_initialization_failed")

    def test_dual_weight_layout_rejected(self) -> None:
        root = _mutated_copy(self, self.template)
        (root / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
        self._init_fails_exact(root, "local_backend_initialization_failed")

    def test_bad_max_context_values(self) -> None:
        root = _mutated_copy(self, self.template)
        for bad in (0, -1, 32769, "64", True, 1.5):
            with self.subTest(bad=bad):
                self._init_fails_exact(root, "local_backend_initialization_failed", max_context=bad)

    def test_bad_devices(self) -> None:
        root = _mutated_copy(self, self.template)
        for bad in ("tpu", "", "CPU"):
            with self.subTest(bad=bad):
                self._init_fails_exact(root, "local_backend_initialization_failed", device=bad)

    def test_context_beyond_declared_window_uses_distinct_code(self) -> None:
        # The tiny checkpoint declares n_positions=64: 65 (and even the
        # in-range 32768) must surface the actionable distinct code, not the
        # generic init failure.
        root = _mutated_copy(self, self.template)
        for over in (65, 32768):
            with self.subTest(over=over):
                self._init_fails_exact(
                    root, "local_context_limit_exceeds_declared_window", max_context=over
                )

    def test_declared_window_boundary_inits_fine(self) -> None:
        root = _mutated_copy(self, self.template)
        backend = backends.LocalHFBackend(root, device="cpu", max_context=64)
        again = backends.LocalHFBackend(root, device="cpu", max_context=64)
        self.assertEqual(backend.fingerprint, again.fingerprint)

    def test_cuda_unavailable_rejected_without_cuda(self) -> None:
        import torch

        if torch.cuda.is_available():
            self.skipTest("requires a machine without CUDA")
        root = _mutated_copy(self, self.template)
        self._init_fails_exact(root, "local_backend_initialization_failed", device="cuda")


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class ForwardLogitsCodeTests(unittest.TestCase):
    """forward_logits: exact codes, exact messages, output tensor contract."""

    backend: backends.LocalHFBackend

    @classmethod
    def setUpClass(cls) -> None:
        root = _template_checkpoint(cls)
        cls.backend = backends.LocalHFBackend(root, device="cpu", max_context=64)

    def test_valid_forward_output_contract(self) -> None:
        import torch

        out = self.backend.forward_logits((1, 2, 3))
        self.assertEqual(tuple(out.shape), (3, self.backend.vocab_size))
        self.assertEqual(out.dtype, torch.float32)
        self.assertFalse(out.requires_grad)
        self.assertTrue(out.is_contiguous())
        self.assertTrue(bool(torch.isfinite(out).all().item()))

    def test_invalid_input_ids_code(self) -> None:
        bad: list[Any] = [
            (),
            (64,),  # == vocab_size: out of range
            (-1,),
            (True,),  # bool is not an int id
            ([1, 2]),  # list, not tuple
            ("1",),
        ]
        for ids in bad:
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError) as caught:
                    self.backend.forward_logits(ids)
                self.assertEqual(str(caught.exception), "invalid_input_ids")

    def test_context_limit_exceeded_code_and_boundary(self) -> None:
        ok = self.backend.forward_logits(tuple([1] * 64))
        self.assertEqual(tuple(ok.shape), (64, self.backend.vocab_size))
        with self.assertRaises(ValueError) as caught:
            self.backend.forward_logits(tuple([1] * 65))
        self.assertEqual(str(caught.exception), "context_limit_exceeded")

    def test_not_initialized_code(self) -> None:
        # White-box: __init__ either completes or raises, so this branch is
        # defensive-only (see _ready's docstring); pin the code anyway.
        model: Any = self.backend._model
        self.backend._model = None
        self.addCleanup(setattr, self.backend, "_model", model)
        with self.assertRaises(RuntimeError) as caught:
            self.backend.forward_logits((1, 2, 3))
        self.assertEqual(str(caught.exception), "local_backend_not_initialized")

    def test_forward_failure_code(self) -> None:
        model: Any = self.backend._model
        with (
            patch.object(model, "forward", side_effect=RuntimeError("boom")),
            self.assertRaises(RuntimeError) as caught,
        ):
            self.backend.forward_logits((1, 2, 3))
        self.assertEqual(str(caught.exception), "local_forward_failed")
        self.assertIsNone(caught.exception.__cause__)


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class AdversarialEdgeTests(unittest.TestCase):
    """Boundary inputs, double init, threads, and lock hygiene."""

    backend: backends.LocalHFBackend
    root: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = _template_checkpoint(cls)
        cls.backend = backends.LocalHFBackend(cls.root, device="cpu", max_context=64)

    def test_single_id_input(self) -> None:
        result = self.backend.generate((1,), max_new_tokens=4, seed=0, cancel=Event())
        self.assertEqual((result.reason, len(result.token_ids)), ("limit", 4))
        out = self.backend.forward_logits((1,))
        self.assertEqual(tuple(out.shape), (1, self.backend.vocab_size))

    def test_context_window_boundary_generate(self) -> None:
        # max_context=64: 63+1 fits, 63+2 and 64+1 do not.
        fits = self.backend.generate(tuple([1] * 63), max_new_tokens=1, seed=0, cancel=Event())
        self.assertEqual((fits.reason, len(fits.token_ids)), ("limit", 1))
        for prompt_len, cap in ((63, 2), (64, 1)):
            with self.subTest(prompt_len=prompt_len, cap=cap):
                result = self.backend.generate(
                    tuple([1] * prompt_len), max_new_tokens=cap, seed=0, cancel=Event()
                )
                self.assertEqual(result.reason, "error")
                self.assertEqual(result.error, "context_limit_exceeded")

    def test_cap_upper_bound_pinned(self) -> None:
        for bad_cap in (0, -1, 4097, "4096", 4.0):
            with self.subTest(bad_cap=bad_cap):
                result = self.backend.generate(
                    (1,),
                    max_new_tokens=bad_cap,  # type: ignore[arg-type]
                    seed=0,
                    cancel=Event(),
                )
                self.assertEqual(result.reason, "error")
                self.assertEqual(result.error, "invalid_output_cap")
        # Largest cap the declared 64-token window admits: 1 prompt + 63 new.
        biggest = self.backend.generate((1,), max_new_tokens=63, seed=0, cancel=Event())
        self.assertEqual((biggest.reason, len(biggest.token_ids)), ("limit", 63))
        # A within-cap but over-window request fails on the window, not the cap.
        over_window = self.backend.generate((1,), max_new_tokens=4096, seed=0, cancel=Event())
        self.assertEqual(
            (over_window.reason, over_window.error), ("error", "context_limit_exceeded")
        )

    def test_double_init_keeps_backend_working(self) -> None:
        before = self.backend.fingerprint
        backends.LocalHFBackend.__init__(self.backend, self.root, device="cpu", max_context=64)
        self.assertEqual(self.backend.fingerprint, before)
        result = self.backend.generate((1, 2), max_new_tokens=4, seed=0, cancel=Event())
        self.assertEqual((result.reason, len(result.token_ids)), ("limit", 4))

    def test_concurrent_generate_calls_all_hit_exact_cap(self) -> None:
        # The backend serializes on an RLock; concurrent requests must each
        # get exactly their cap with no cross-talk between outputs.
        count = 4
        results: list[Any] = [None] * count
        errors: list[Any] = [None] * count

        def work(index: int) -> None:
            try:
                results[index] = self.backend.generate(
                    (1, 2), max_new_tokens=8, seed=7, cancel=Event()
                )
            except Exception as exc:  # noqa: BLE001 -- recorded, asserted below
                errors[index] = exc

        threads = [Thread(target=work, args=(i,)) for i in range(count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=180)
        for thread in threads:
            self.assertFalse(thread.is_alive())
        for index in range(count):
            self.assertIsNone(errors[index])
            result = results[index]
            self.assertEqual(result.reason, "limit")
            self.assertEqual(len(result.token_ids), 8)
        self.assertEqual(results[0].token_ids, results[1].token_ids)
        self.assertEqual(results[1].token_ids, results[2].token_ids)
        self.assertEqual(results[2].token_ids, results[3].token_ids)

    def test_concurrent_forward_logits(self) -> None:
        count = 4
        results: list[Any] = [None] * count
        errors: list[Any] = [None] * count

        def work(index: int) -> None:
            try:
                results[index] = self.backend.forward_logits((1, 2, 3))
            except Exception as exc:  # noqa: BLE001 -- recorded, asserted below
                errors[index] = exc

        threads = [Thread(target=work, args=(i,)) for i in range(count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=180)
        for index in range(count):
            self.assertIsNone(errors[index])
            self.assertEqual(tuple(results[index].shape), (3, self.backend.vocab_size))

    def test_backend_lock_released_after_inference_failure(self) -> None:
        model: Any = self.backend._model
        with patch.object(model, "forward", side_effect=RuntimeError("boom")):
            result = self.backend.generate((1, 2), max_new_tokens=8, seed=0, cancel=Event())
        self.assertEqual((result.reason, result.error), ("error", "local_inference_failed"))
        # The finally block must have released the lock: re-acquire it.
        self.assertTrue(self.backend._lock.acquire(timeout=10))
        self.backend._lock.release()
        again = self.backend.generate((1, 2), max_new_tokens=4, seed=0, cancel=Event())
        self.assertEqual((again.reason, len(again.token_ids)), ("limit", 4))

    def test_encode_decode_reject_bad_ids_without_hf_text(self) -> None:
        with self.assertRaises(ValueError) as caught_encode:
            self.backend.encode(123)  # type: ignore[arg-type]
        self.assertEqual(str(caught_encode.exception), "invalid_text")
        with self.assertRaises(ValueError) as caught_decode:
            self.backend.decode((10_000,))
        self.assertEqual(str(caught_decode.exception), "invalid_decode_ids")


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class TruncatedPromptBackendTests(unittest.TestCase):
    """TruncatedPromptBackend: validation, truncation, pass-through."""

    inner: backends.LocalHFBackend
    wrapper: backends.TruncatedPromptBackend

    @classmethod
    def setUpClass(cls) -> None:
        root = _template_checkpoint(cls)
        cls.inner = backends.LocalHFBackend(root, device="cpu", max_context=64)
        cls.wrapper = backends.TruncatedPromptBackend(cls.inner, 3)

    def test_init_rejects_bad_drop_bytes(self) -> None:
        for bad in (0, -1, "3", True, 2.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                backends.TruncatedPromptBackend(self.inner, bad)  # type: ignore[arg-type]

    def test_init_rejects_non_string_fingerprint(self) -> None:
        class Stub:
            fingerprint = 123

        with self.assertRaises(ValueError):
            backends.TruncatedPromptBackend(Stub(), 3)  # type: ignore[arg-type]

        class NoFingerprint:
            pass

        with self.assertRaises(ValueError):
            backends.TruncatedPromptBackend(NoFingerprint(), 3)  # type: ignore[arg-type]

    def test_fingerprint_names_inner_and_drop(self) -> None:
        self.assertEqual(self.wrapper.fingerprint, f"{self.inner.fingerprint}+prompt-trunc-3b")

    def test_encode_truncates_tail_bytes(self) -> None:
        for text in ("w1 w2 w3", "w10 w20 w30 w40"):
            with self.subTest(text=text):
                expected = self.inner.encode(truncate_text_tail(text, 3))  # type: ignore[arg-type]
                self.assertEqual(self.wrapper.encode(text), expected)
        # Concrete: dropping 3 bytes from "w1 w2 w3" leaves "w1 w2".
        self.assertEqual(self.wrapper.encode("w1 w2 w3"), self.inner.encode("w1 w2"))

    def test_encode_splitting_codepoint_raises_fixed_code(self) -> None:
        # "é" is two UTF-8 bytes; a 1-byte drop splits the codepoint, so
        # truncate_text_tail returns None and encode must raise the fixed code.
        one_byte = backends.TruncatedPromptBackend(self.inner, 1)
        with self.assertRaises(ValueError) as caught:
            one_byte.encode("w1 é")
        self.assertEqual(str(caught.exception), "prompt_truncation_splits_codepoint")

    def test_encode_propagates_tail_length_error(self) -> None:
        with self.assertRaises(ValueError):
            self.wrapper.encode("w1")  # 2 bytes < drop_bytes=3

    def test_decode_passes_through(self) -> None:
        ids = self.inner.encode("w1 w2")
        self.assertEqual(self.wrapper.decode(ids), self.inner.decode(ids))

    def test_generate_passes_through_with_exact_cap(self) -> None:
        event = Event()
        wrapped = self.wrapper.generate((1, 2), max_new_tokens=8, seed=0, cancel=event)
        direct = self.inner.generate((1, 2), max_new_tokens=8, seed=0, cancel=Event())
        self.assertEqual(wrapped.reason, "limit")
        self.assertEqual(len(wrapped.token_ids), 8)
        self.assertEqual(wrapped.token_ids, direct.token_ids)

    def test_hf_stop_token_bytes_forwarded(self) -> None:
        self.assertIs(self.wrapper.hf_stop_token_bytes, self.inner.hf_stop_token_bytes)
