"""Cycle 192: the real LocalHFBackend batch path (wave-8 rank #4).

Cycle 153's batch-invariance audit probed for an optional
``generate_batch`` no in-tree backend implemented, so on the real
backend it honestly recorded ``batch_unsupported``. This cycle builds
the path: left-padded batched greedy over the same fp32 full-prefix
recompute discipline as solo ``generate``. The contract under test is
docs/BATCH_PATH_DESIGN.md (red-team adjudicated): equivalence with
solo is measured token-exactly, never assumed; the stage feeds the
batch path only solo-evaluable prompts; an oversize batch is a
labelled byte-budget refusal, never an OOM; cancel raises, never
returns partial rows.

Self-contained: the tiny-checkpoint builder is copied from
test_cycle81 (test files carry no cross-test imports -- mutation
gates copy files alone).
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from model_protocol_lab import backends, experiments
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


class PeakEstimateTests(unittest.TestCase):
    """The byte-budget arithmetic, pinned against the design formula."""

    def test_formula_terms(self) -> None:
        # logits 4*rows*L*vocab; scores 8*rows*heads*L^2 (matmul and
        # softmax outputs co-live); additive mask 4*rows*L^2.
        self.assertEqual(backends._batch_peak_bytes(2, 10, 64, 2), 5120 + 3200 + 800)
        self.assertEqual(backends._batch_peak_bytes(1, 1, 1, 1), 4 + 8 + 4)
        self.assertEqual(backends._batch_peak_bytes(3, 7, 128, 4), 10752 + 4704 + 588)

    def test_budget_constant(self) -> None:
        self.assertEqual(backends._BATCH_FORWARD_BYTE_BUDGET, 1 << 33)

    def test_default_caller_scale_fits(self) -> None:
        # The audit's default caller scale (100-fixture split,
        # paragraph prompts, GPT-2 vocab) must sit under the budget:
        # the bound refuses pathological shapes, not ordinary runs.
        peak = backends._batch_peak_bytes(100, 264 + 64, 50257, 12)
        self.assertLess(peak, backends._BATCH_FORWARD_BYTE_BUDGET)


class _StubBackend:
    """Scripted solo/batch backend for stage tests (no model needed)."""

    def __init__(
        self,
        solo_fail: frozenset[str] = frozenset(),
        batch_error: BaseException | None = None,
    ) -> None:
        self.solo_fail = solo_fail
        self.batch_error = batch_error
        self.fed: tuple[tuple[int, ...], ...] | None = None
        self.batch_calls = 0

    @staticmethod
    def generate(
        input_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        seed: int,
        cancel: Event,
    ) -> Generation:
        # The stage keys solo failures by prompt id via its own loop;
        # here the prompt content encodes the id (see stage tests).
        if input_ids and input_ids[0] == 99:
            return Generation((), "error", "local_inference_failed")
        return Generation(tuple(7 for _ in range(max_new_tokens)), "limit")

    def generate_batch(
        self,
        prompts: tuple[tuple[int, ...], ...],
        *,
        max_new_tokens: int,
        seed: int,
        cancel: Event,
    ) -> tuple[tuple[int, ...], ...]:
        self.batch_calls += 1
        self.fed = prompts
        if self.batch_error is not None:
            raise self.batch_error
        return tuple(tuple(7 for _ in range(max_new_tokens)) for _ in prompts)


class StageFeedingTests(unittest.TestCase):
    """F1 regression: the stage feeds the batch path evaluable rows only."""

    def test_solo_failed_prompt_is_not_fed(self) -> None:
        backend = _StubBackend()
        payload = experiments.batch_stage(
            backend,  # type: ignore[arg-type]
            {"a": (1, 2), "b": (99, 3), "c": (4,)},
            max_new_tokens=3,
            seed=0,
        )
        self.assertTrue(payload["available"])
        self.assertEqual(backend.fed, ((1, 2), (4,)))
        self.assertEqual(payload["n_evaluable"], 2)
        self.assertEqual(payload["n_failed"], 1)
        self.assertEqual(payload["per_prompt"]["b"]["error"], "solo_failed")
        self.assertEqual(payload["verdict"], "batch_invariant")

    def test_all_solo_failed_skips_the_batch_call(self) -> None:
        backend = _StubBackend(batch_error=RuntimeError("must not be called"))
        payload = experiments.batch_stage(
            backend,  # type: ignore[arg-type]
            {"a": (99,), "b": (99, 1)},
            max_new_tokens=3,
            seed=0,
        )
        self.assertTrue(payload["available"])
        self.assertEqual(backend.batch_calls, 0)
        self.assertEqual(payload["n_evaluable"], 0)
        self.assertEqual(payload["verdict"], "unavailable")
        self.assertIsNone(payload["divergence_rate"])

    def test_size_refusal_keeps_its_own_detail(self) -> None:
        backend = _StubBackend(batch_error=ValueError("batch_size_exceeded"))
        payload = experiments.batch_stage(
            backend,  # type: ignore[arg-type]
            {"a": (1, 2)},
            max_new_tokens=3,
            seed=0,
        )
        self.assertEqual(payload, {"available": False, "detail": "batch_size_exceeded"})

    def test_other_value_error_is_batch_failed(self) -> None:
        backend = _StubBackend(batch_error=ValueError("context_limit_exceeded"))
        payload = experiments.batch_stage(
            backend,  # type: ignore[arg-type]
            {"a": (1, 2)},
            max_new_tokens=3,
            seed=0,
        )
        self.assertEqual(payload, {"available": False, "detail": "batch_failed"})


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class BatchPathTests(unittest.TestCase):
    """The real batched path against the tiny GPT-2 checkpoint."""

    backend: backends.LocalHFBackend
    root: Path

    @classmethod
    def setUpClass(cls) -> None:
        tmp = TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name) / "ckpt"
        cls.root.mkdir()
        build_tiny_gpt2(cls.root)
        cls.backend = backends.LocalHFBackend(cls.root, device="cpu", max_context=64)

    def solo(self, prompt: tuple[int, ...], cap: int) -> tuple[int, ...]:
        generation = self.backend.generate(prompt, max_new_tokens=cap, seed=0, cancel=Event())
        self.assertEqual(generation.reason, "limit")
        return generation.token_ids

    def batch(self, prompts: tuple[tuple[int, ...], ...], cap: int) -> tuple[tuple[int, ...], ...]:
        return self.backend.generate_batch(prompts, max_new_tokens=cap, seed=0, cancel=Event())

    def test_equivalence_sizes_and_mixed_lengths(self) -> None:
        pool = [(1,), (2, 3), (0,), (0, 0, 5), (7, 0, 9, 11, 13)]
        for size in (1, 2, 3, 5):
            prompts = tuple(pool[:size])
            with self.subTest(size=size):
                result = self.batch(prompts, 5)
                self.assertEqual(len(result), size)
                for prompt, row in zip(prompts, result, strict=True):
                    self.assertEqual(row, self.solo(prompt, 5))

    def test_equivalence_32_rows(self) -> None:
        prompts = tuple((i % 61 + 1,) * (i % 8 + 1) for i in range(32))
        result = self.batch(prompts, 4)
        self.assertEqual(len(result), 32)
        for prompt, row in zip(prompts, result, strict=True):
            self.assertEqual(row, self.solo(prompt, 4))

    def test_permutation_invariance(self) -> None:
        prompts = ((3, 1), (9,), (0, 4, 4), (17, 2))
        forward = self.batch(prompts, 4)
        backward = self.batch(tuple(reversed(prompts)), 4)
        self.assertEqual(forward, tuple(reversed(backward)))
        for prompt, row in zip(prompts, forward, strict=True):
            self.assertEqual(row, self.solo(prompt, 4))

    def test_output_shape_contract(self) -> None:
        result = self.batch(((5, 5), (1,)), 3)
        self.assertIs(type(result), tuple)
        for row in result:
            self.assertIs(type(row), tuple)
            self.assertEqual(len(row), 3)
            for token in row:
                self.assertIs(type(token), int)
                self.assertGreaterEqual(token, 0)
                self.assertLess(token, 64)

    def test_one_forward_per_step_over_the_whole_batch(self) -> None:
        # I3 anti-theater: a wrapper that secretly looped solo calls
        # would show len(prompts) forwards per step with width-1 rows;
        # the real path shows one forward per step whose width grows
        # by exactly the appended column.
        shapes: list[tuple[int, ...]] = []
        original = self.backend._model.forward

        def counting(*args: Any, **kwargs: Any) -> Any:
            shapes.append(tuple(kwargs["input_ids"].shape))
            return original(*args, **kwargs)

        with patch.object(self.backend._model, "forward", counting):
            result = self.batch(((1, 2), (3, 4, 5, 6, 7), (8,)), 3)
        self.assertEqual(shapes, [(3, 5), (3, 6), (3, 7)])
        self.assertEqual(len(result), 3)

    def test_head_count_falls_back_to_gpt2_n_head(self) -> None:
        # GPT-2 configs name the query-head count n_head; a config
        # view hiding the unified name must still size the bound
        # (transformers v5 aliases the two names inside
        # __getattribute__, so the fallback is exercised through a
        # delegating view, not by mutating the real config).
        expected = self.solo((2, 3), 3)
        view = _ConfigView(self.backend._model.config, hide=frozenset({"num_attention_heads"}))
        with patch.object(self.backend._model, "config", view):
            result = self.batch(((2, 3),), 3)
        self.assertEqual(result, (expected,))

    def test_config_without_any_head_count_fails_closed(self) -> None:
        view = _ConfigView(
            self.backend._model.config, hide=frozenset({"num_attention_heads", "n_head"})
        )
        with (
            patch.object(self.backend._model, "config", view),
            self.assertRaises(RuntimeError) as caught,
        ):
            self.batch(((2, 3),), 3)
        self.assertEqual(str(caught.exception), "local_batch_failed")

    def test_wrappers_do_not_forward_the_batch_path(self) -> None:
        # Deliberate: the wrappers' solo contract rewrites inputs the
        # batch path would bypass, so they stay batch_unsupported.
        truncated = backends.TruncatedPromptBackend(self.backend, 4)
        healing = backends.TokenHealingBackend(self.backend)
        self.assertFalse(callable(getattr(truncated, "generate_batch", None)))
        self.assertFalse(callable(getattr(healing, "generate_batch", None)))

    def test_stage_end_to_end_on_the_real_backend(self) -> None:
        payload = experiments.batch_stage(
            self.backend, {"a": (1, 2, 3), "b": (4, 5)}, max_new_tokens=4, seed=0
        )
        self.assertTrue(payload["available"])
        self.assertEqual(payload["n_evaluable"], 2)
        self.assertEqual(payload["verdict"], "batch_invariant")

    def test_cli_probe_records_the_available_payload(self) -> None:
        # The probe composition -- render with the first candidate's
        # format, encode on the real backend, run the real stage --
        # pinned end-to-end: cycle 153 pins the probe with a scripted
        # backend and this file pins the stage directly, but the
        # composed path is what `run --batch-invariance` executes.
        from model_protocol_lab import cli, core

        fmt = core.Format(name="t", bos="", user_prefix="U:", assistant_prefix="A:", turn_end="\n")
        cases = (
            core.Fixture(
                id="f1",
                messages=(core.Message("user", "w1 w2"),),
                expected="x",
                group="g",
            ),
            core.Fixture(
                id="f2",
                messages=(core.Message("user", "w3"),),
                expected="y",
                group="g",
            ),
        )
        result: dict[str, Any] = {}
        cli._run_batch_probe(self.backend, (fmt,), result, cases, max_new_tokens=4, seed=0)
        payload = result["batch_invariance"]
        self.assertTrue(payload["available"])
        self.assertEqual(payload["verdict"], "batch_invariant")
        self.assertEqual(payload["n_prompts"], 2)
        self.assertEqual(payload["n_evaluable"], 2)


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class BatchValidationTests(unittest.TestCase):
    """The pinned validation precedence of docs/BATCH_PATH_DESIGN.md."""

    backend: backends.LocalHFBackend
    root: Path

    @classmethod
    def setUpClass(cls) -> None:
        tmp = TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name) / "ckpt"
        cls.root.mkdir()
        build_tiny_gpt2(cls.root)
        cls.backend = backends.LocalHFBackend(cls.root, device="cpu", max_context=64)

    def call(self, prompts: Any, **overrides: Any) -> tuple[tuple[int, ...], ...]:
        kwargs: dict[str, Any] = {"max_new_tokens": 2, "seed": 0, "cancel": Event()}
        kwargs.update(overrides)
        return self.backend.generate_batch(prompts, **kwargs)

    def assert_code(self, code: str, prompts: Any, **overrides: Any) -> None:
        with self.assertRaises((ValueError, RuntimeError)) as caught:
            self.call(prompts, **overrides)
        self.assertEqual(str(caught.exception), code)

    def test_invalid_input_ids(self) -> None:
        cases = [
            [(1, 2)],  # container is a list, not a tuple
            ([1, 2],),  # a prompt that is a list
            ((),),  # an empty prompt
            ((1, True),),  # a bool id
            ((1, -1),),  # a negative id
            ((1, 64),),  # an id at the vocab bound
            ((1, "2"),),  # a non-int id
            "not-a-tuple",
            None,
        ]
        for prompts in cases:
            with self.subTest(prompts=prompts):
                self.assert_code("invalid_input_ids", prompts)

    def test_invalid_cap_seed_cancel(self) -> None:
        for cap in (True, 0, -1, 4097, "3", 2.0):
            with self.subTest(cap=cap):
                self.assert_code("invalid_output_cap", ((1,),), max_new_tokens=cap)
        for seed in (True, -1, 2**64, "0"):
            with self.subTest(seed=seed):
                self.assert_code("invalid_seed", ((1,),), seed=seed)
        for cancel in (None, object(), "event"):
            with self.subTest(cancel=cancel):
                self.assert_code("invalid_cancellation_event", ((1,),), cancel=cancel)

    def test_precedence_multi_violation(self) -> None:
        # Bad ids beat a bad cap; a bad cap beats a bad seed; a
        # pre-set cancel beats the context check (solo's order).
        self.assert_code("invalid_input_ids", ((999,),), max_new_tokens=0)
        self.assert_code("invalid_output_cap", ((1,),), max_new_tokens=0, seed=-1)
        cancel = Event()
        cancel.set()
        self.assert_code("local_batch_cancelled", ((1,) * 70,), cancel=cancel)

    def test_cancel_pre_set(self) -> None:
        cancel = Event()
        cancel.set()
        with self.assertRaises(RuntimeError) as caught:
            self.call(((1, 2),), cancel=cancel)
        self.assertEqual(str(caught.exception), "local_batch_cancelled")

    def test_context_limit(self) -> None:
        self.assert_code("context_limit_exceeded", ((1,) * 63,), max_new_tokens=2)
        # Exactly at the limit is allowed.
        result = self.call(((1,) * 62,), max_new_tokens=2)
        self.assertEqual(len(result), 1)

    def test_empty_batch(self) -> None:
        self.assertEqual(self.call(()), ())
        self.assert_code("invalid_output_cap", (), max_new_tokens=0)
        cancel = Event()
        cancel.set()
        self.assert_code("local_batch_cancelled", (), cancel=cancel)

    def test_unready_backend(self) -> None:
        backend = backends.LocalHFBackend(self.root, device="cpu", max_context=64)
        backend._model = None
        backend._tokenizer = None
        with self.assertRaises(RuntimeError) as caught:
            backend.generate_batch(((1,),), max_new_tokens=2, seed=0, cancel=Event())
        self.assertEqual(str(caught.exception), "local_backend_not_initialized")
        # The empty shortcut sits after _ready(): unready wins.
        with self.assertRaises(RuntimeError) as caught:
            backend.generate_batch((), max_new_tokens=2, seed=0, cancel=Event())
        self.assertEqual(str(caught.exception), "local_backend_not_initialized")

    def test_size_bound_refusal_and_boundary(self) -> None:
        prompts = ((1, 2), (3,))
        peak = backends._batch_peak_bytes(2, 2 + 3, 64, 2)
        with patch.object(backends, "_BATCH_FORWARD_BYTE_BUDGET", peak - 1):
            self.assert_code("batch_size_exceeded", prompts, max_new_tokens=3)
        with patch.object(backends, "_BATCH_FORWARD_BYTE_BUDGET", peak):
            result = self.call(prompts, max_new_tokens=3)
            self.assertEqual(len(result), 2)

    def test_size_refusal_happens_before_any_forward(self) -> None:
        calls: list[int] = []
        original = self.backend._model.forward

        def counting(*args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            return original(*args, **kwargs)

        with (
            patch.object(backends, "_BATCH_FORWARD_BYTE_BUDGET", 1),
            patch.object(self.backend._model, "forward", counting),
        ):
            self.assert_code("batch_size_exceeded", ((1, 2),), max_new_tokens=2)
        self.assertEqual(calls, [])


class _ConfigView:
    """Delegating view over a real config that hides chosen attributes.

    Everything else forwards to the real config, so the model's own
    forward (which reads config.return_dict and friends) keeps working
    while generate_batch's head-count lookup sees the hidden names as
    absent.
    """

    def __init__(self, real: Any, hide: frozenset[str]) -> None:
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_hide", hide)

    def __getattr__(self, name: str) -> Any:
        if name in object.__getattribute__(self, "_hide"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_real"), name, value)


class _CountingEvent(Event):
    """An Event whose is_set() turns True after a fixed number of calls.

    Cancellation checks in generate_batch run in a pinned order; a
    plain Event cannot be set *between* two of them from a test, but
    counting the checks can: the Nth check is the first to see True.
    """

    def __init__(self, true_after: int) -> None:
        super().__init__()
        self._calls = 0
        self._true_after = true_after

    def is_set(self) -> bool:
        self._calls += 1
        return self._calls > self._true_after or super().is_set()


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class BatchFailureTests(unittest.TestCase):
    """Cancel and inference-failure behaviour of the real path."""

    backend: backends.LocalHFBackend
    root: Path

    @classmethod
    def setUpClass(cls) -> None:
        tmp = TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name) / "ckpt"
        cls.root.mkdir()
        build_tiny_gpt2(cls.root)
        cls.backend = backends.LocalHFBackend(cls.root, device="cpu", max_context=64)

    def test_cancel_seen_at_the_post_acquire_check(self) -> None:
        # Checks in order: pre-validation (1), lock poll (2),
        # post-acquire (3). Turning True at the third check must
        # raise before any forward runs.
        cancel = _CountingEvent(true_after=2)
        with self.assertRaises(RuntimeError) as caught:
            self.backend.generate_batch(((1, 2),), max_new_tokens=2, seed=0, cancel=cancel)
        self.assertEqual(str(caught.exception), "local_batch_cancelled")

    def test_cancel_seen_at_a_later_step_pre_forward_check(self) -> None:
        # Checks 4 and 5 are step 0's pre- and post-forward checks;
        # check 6 is step 1's pre-forward check. Turning True there
        # must still raise the cancelled code with no partial result.
        cancel = _CountingEvent(true_after=5)
        with self.assertRaises(RuntimeError) as caught:
            self.backend.generate_batch(((1, 2), (3,)), max_new_tokens=3, seed=0, cancel=cancel)
        self.assertEqual(str(caught.exception), "local_batch_cancelled")

    def test_cancel_during_first_forward(self) -> None:
        # Cancel raised inside the inference try surfaces as the
        # cancelled code, never eaten by the failure conversion, and
        # no partial rows are returned (the call raises).
        cancel = Event()
        original = self.backend._model.forward

        def cancelling(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            cancel.set()
            return result

        with (
            patch.object(self.backend._model, "forward", cancelling),
            self.assertRaises(RuntimeError) as caught,
        ):
            self.backend.generate_batch(((1, 2), (3,)), max_new_tokens=4, seed=0, cancel=cancel)
        self.assertEqual(str(caught.exception), "local_batch_cancelled")

    def test_cancel_while_waiting_for_the_lock(self) -> None:
        cancel = Event()
        captured: list[BaseException] = []

        def run() -> None:
            try:
                self.backend.generate_batch(((1, 2),), max_new_tokens=2, seed=0, cancel=cancel)
            except BaseException as exc:  # noqa: BLE001 -- captured for assertion
                captured.append(exc)

        self.backend._lock.acquire()
        try:
            worker = Thread(target=run)
            worker.start()
            cancel.set()
            worker.join(timeout=10)
        finally:
            self.backend._lock.release()
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(captured), 1)
        self.assertIsInstance(captured[0], RuntimeError)
        self.assertEqual(str(captured[0]), "local_batch_cancelled")

    def test_raising_forward_becomes_the_fixed_code(self) -> None:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("raw torch complaint with internals")

        with (
            patch.object(self.backend._model, "forward", boom),
            self.assertRaises(RuntimeError) as caught,
        ):
            self.backend.generate_batch(((1, 2),), max_new_tokens=2, seed=0, cancel=Event())
        self.assertEqual(str(caught.exception), "local_batch_failed")

    def test_non_finite_logits_fail_the_batch(self) -> None:
        import torch

        def nan_forward(*args: Any, **kwargs: Any) -> Any:
            rows = kwargs["input_ids"].shape[0]
            logits = torch.full((rows, 3, 64), float("nan"))
            return SimpleNamespace(logits=logits)

        with (
            patch.object(self.backend._model, "forward", nan_forward),
            self.assertRaises(RuntimeError) as caught,
        ):
            self.backend.generate_batch(((1, 2), (3,)), max_new_tokens=2, seed=0, cancel=Event())
        self.assertEqual(str(caught.exception), "local_batch_failed")

    def test_mis_shaped_logits_fail_the_batch(self) -> None:
        import torch

        def short_forward(*args: Any, **kwargs: Any) -> Any:
            rows = kwargs["input_ids"].shape[0]
            logits = torch.zeros((rows, 3, 63))
            return SimpleNamespace(logits=logits)

        with (
            patch.object(self.backend._model, "forward", short_forward),
            self.assertRaises(RuntimeError) as caught,
        ):
            self.backend.generate_batch(((1, 2), (3,)), max_new_tokens=2, seed=0, cancel=Event())
        self.assertEqual(str(caught.exception), "local_batch_failed")


if __name__ == "__main__":
    unittest.main()
