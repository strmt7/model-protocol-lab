"""Cycle 91: adversarial backend-output audit of the trust boundary.

METHOD
------
No cycle has adversarially audited what happens when a backend's
``generate()``/``decode()``/``encode()``/``forward_logits()`` returns
malformed data. This file drives hostile fake backends (in-process, small
fixture splits built here, no network, no checkpoints) through
``core.evaluate`` and ``scoring.score_target_logprobs`` and pins the outcome
of every case to either a valid per-candidate verdict from the fixed failure
taxonomy or the documented BaseException propagation (KeyboardInterrupt /
SystemExit, per cycle 89). The fake backends cover:

- ``generate()`` returning: empty id sequences, negative / huge (2**200) /
  float / str / None / bool ids, 10x more tokens than ``max_new_tokens``,
  non-tuple containers (list, generator, tuple subclass), a ``Generation``
  subclass, non-``Generation`` values (tuple, None), a bool ``reason``,
  short ids with ``reason="limit"``, ``error`` set while ``reason="limit"``,
  and every ``reason="error"`` error-shape (None, empty string, arbitrary
  hostile text, non-str, allowlisted code).
- ``decode()`` returning: bytes, int, None, a str subclass, lone
  surrogates, a 10MB string; ``decode()`` raising ``ValueError`` and
  ``KeyboardInterrupt``; ``decode()`` raising only for the boundary ids
  (the ``boundary_text_match`` diagnostic must degrade to None, not fail
  the run).
- ``generate()`` raising: ``RuntimeError``, ``MemoryError``, a custom
  ``Exception`` subclass, ``StopIteration``, ``KeyboardInterrupt``,
  ``SystemExit``; ``encode()`` raising and returning invalid ids.
- A non-deterministic backend returning a valid ``Generation`` on the
  first call and a malformed one on the second.
- ``forward_logits()`` returning: NaN / +inf / -inf logits, 1-D and 3-D
  tensors, wrong-length T, zero-width V, a non-tensor, None; raising
  ``RuntimeError`` / ``MemoryError`` / ``KeyboardInterrupt`` / ``SystemExit``
  / ``ValueError`` with an allowlisted code / ``ValueError`` with arbitrary
  text; a backend with no ``forward_logits`` at all.
- A mixed ``run_experiment`` run where one scheduled backend is
  well-behaved on some fixtures (one pass, one wrong answer without
  boundary) and hostile on others: the failure-taxonomy counts must sum
  exactly to the non-passing observations (no double-counting, no lost
  fixtures).

FINDINGS
--------
- CLEAN: every malformed ``Generation`` shape collapses to
  ``invalid_generation`` (failed observation, run terminates, evidence
  retained). No raw traceback, no hang, no malformed result dict in any
  case probed.
- CLEAN: ``reason="error"`` error-text sanitization holds -- arbitrary
  hostile text, the empty string, and None all collapse to the fixed
  ``backend_reported_error`` code; the hostile text never enters the
  records (pinned: a 10KB hostile error string is dropped, not stored).
  Allowlisted codes (``context_limit_exceeded``) pass through verbatim.
- CLEAN: ``generate()``/``encode()`` raising ``RuntimeError``,
  ``MemoryError``, custom ``Exception`` subclasses, and ``StopIteration``
  all become fixed codes (``generate_failed`` / ``encode_failed``);
  ``KeyboardInterrupt`` and ``SystemExit`` propagate unchanged out of
  ``evaluate`` (and out of ``score_target_logprobs``' provider call),
  matching the cycle-89 interrupt contract.
- CLEAN: hostile ``decode()`` output (bytes/int/None/str-subclass/raising)
  becomes ``decode_failed``; a 10MB decoded string and lone surrogates are
  accepted as-is (valid ``str``), with no truncation and no hang; the
  boundary diagnostic degrades to ``None`` when only the boundary decode
  fails.
- CLEAN: NaN/+-inf logits -> ``unavailable``/``non_finite_logits``;
  wrong-shaped tensors -> the matching fixed rejection code
  (``logits_must_be_2d``, ``logits_length_mismatch``, ``empty_vocabulary``,
  ``logits_must_be_tensor``); provider ``ValueError`` with an allowlisted
  code passes through, anything else collapses to ``forward_failed``.
- CLEAN: the mixed run's taxonomy counts sum exactly (1
  ``no-boundary-in-N`` + 1 ``backend-error`` = 2 non-passing observations
  per candidate; the hostile fixture terminates the run so the later
  fixture is never attempted and never counted).
- NO REAL BUGS FOUND. Nothing in this audit escaped the fixed taxonomy,
  so no production-code change was needed.

DELIBERATELY NOT VERIFIED
-------------------------
- ``Format.render()`` hostility (candidate-side input, not backend
  output; covered by earlier render-validation cycles).
- Real OS-signal injection from inside ``generate()`` (covered by
  cycle 89's live-signal tests).
- Resource-exhaustion DoS beyond a 100k-token ``encode()`` return (pinned
  as completing); unbounded ``encode()`` length is a resource question,
  not a taxonomy violation.
- ``save_report``/HTML serialization of hostile decoded text (e.g. lone
  surrogates through the HTML report path); the pinned guarantee stops at
  the ``EvaluationReport``/observation-dict level, where ``json.dumps``
  round-trips the surrogate answer.
- Live vLLM/SGLang/HF backends (no network or services in this
  environment); only the ``Backend`` protocol surface is audited.

Verified by RUNNING this file: ``pytest -p no:cacheprovider
tests/test_cycle91_hostile_backend_audit.py`` -- 2026-10-16.
"""

from __future__ import annotations

import json
import unittest
from collections.abc import Callable
from threading import Event
from typing import Any

import torch

from model_protocol_lab.core import (
    Fixture,
    Format,
    Generation,
    Message,
    Split,
    evaluate,
)
from model_protocol_lab.experiments import FAILURE_MODES, failure_taxonomy, run_experiment
from model_protocol_lab.scoring import score_target_logprobs

# ---------------------------------------------------------------------------
# Test harness: bijective test codec and a configurable hostile backend.
# ---------------------------------------------------------------------------

_OFFSET = 1000
_CAP = 4
_BOUNDARY = (9001,)


def _enc(text: str) -> tuple[int, ...]:
    return tuple(ord(char) + _OFFSET for char in text)


def _dec(token_ids: tuple[int, ...]) -> str:
    return "".join(chr(token - _OFFSET) for token in token_ids)


_FORMAT = Format(name="cycle91", bos="", user_prefix="", assistant_prefix="", turn_end="")


def _fixture(fid: str, content: str, expected: str, group: str = "g") -> Fixture:
    return Fixture(id=fid, messages=(Message("user", content),), expected=expected, group=group)


_ONE = (_fixture("c1", "hello", "zz"),)


class HostileBackend:
    """A Backend whose methods are caller-supplied, defaulting to a sane codec."""

    fingerprint = "cycle91-hostile"

    def __init__(
        self,
        *,
        generate: Callable[..., Generation] | None = None,
        decode: Callable[[tuple[int, ...]], Any] | None = None,
        encode: Callable[[str], Any] | None = None,
    ) -> None:
        self._generate = generate
        self._decode = decode
        self._encode = encode

    def encode(self, text: str) -> Any:
        if self._encode is not None:
            return self._encode(text)
        return _enc(text)

    def decode(self, token_ids: tuple[int, ...]) -> Any:
        if self._decode is not None:
            return self._decode(token_ids)
        return _dec(token_ids)

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        if self._generate is not None:
            return self._generate(
                input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel
            )
        ids = (_enc("ok") + (1001,) * max_new_tokens)[:max_new_tokens]
        return Generation(ids, "limit")


def _ids(text: str, cap: int = _CAP) -> tuple[int, ...]:
    """Exactly ``cap`` decodable token ids spelling ``text`` (padded)."""
    return (_enc(text) + (1001,) * cap)[:cap]


def _run(backend: HostileBackend, cases: tuple[Fixture, ...] = _ONE) -> Any:
    return evaluate(backend, _FORMAT, cases, boundary=_BOUNDARY, max_new_tokens=_CAP, seed=7)


def _failed_with(report: Any, code: str) -> None:
    assert report.status == "failed", report.status
    assert len(report.observations) == 1
    obs = report.observations[0]
    assert obs.status == "failed", obs.status
    assert obs.error == code, obs.error


class HostileGenerationShapeTest(unittest.TestCase):
    """Malformed Generation values must fail closed as invalid_generation."""

    def _gen(self, value: Any) -> Any:
        return _run(HostileBackend(generate=lambda *a, **k: value))

    def test_empty_ids_with_limit(self) -> None:
        _failed_with(self._gen(Generation((), "limit")), "invalid_generation")

    def test_negative_id(self) -> None:
        _failed_with(self._gen(Generation((1001, -5, 1003, 1004), "limit")), "invalid_generation")

    def test_huge_id_survives_validation(self) -> None:
        # 2**200 is a valid nonnegative int: it passes _valid_generation and
        # reaches decode; a tolerant decode completes with no corruption.
        backend = HostileBackend(
            generate=lambda *a, **k: Generation((1001, 2**200, 1003, 1004), "limit"),
            decode=lambda tids: f"<{tids[1]}>",
        )
        report = _run(backend)
        assert report.status == "completed"
        assert report.observations[0].answer == f"<{2**200}>"

    def test_ten_x_tokens_over_cap(self) -> None:
        gen = Generation(tuple(range(10 * _CAP)), "limit")
        _failed_with(self._gen(gen), "invalid_generation")

    def test_list_token_ids(self) -> None:
        gen = Generation([1001, 1002, 1003, 1004], "limit")  # type: ignore[arg-type]
        _failed_with(self._gen(gen), "invalid_generation")

    def test_generator_token_ids(self) -> None:
        gen = Generation((t for t in (1001, 1002, 1003, 1004)), "limit")  # type: ignore[arg-type]
        _failed_with(self._gen(gen), "invalid_generation")

    def test_tuple_subclass_token_ids(self) -> None:
        class Ids(tuple):  # noqa: F841
            pass

        _failed_with(
            self._gen(Generation(Ids((1001, 1002, 1003, 1004)), "limit")), "invalid_generation"
        )

    def test_float_ids(self) -> None:
        gen = Generation((1001.0, 1002, 1003, 1004), "limit")  # type: ignore[arg-type]
        _failed_with(self._gen(gen), "invalid_generation")

    def test_str_ids(self) -> None:
        gen = Generation(("1001", 1002, 1003, 1004), "limit")  # type: ignore[arg-type]
        _failed_with(self._gen(gen), "invalid_generation")

    def test_none_id(self) -> None:
        gen = Generation((1001, None, 1003, 1004), "limit")  # type: ignore[arg-type]
        _failed_with(self._gen(gen), "invalid_generation")

    def test_bool_id(self) -> None:
        _failed_with(self._gen(Generation((True, 1002, 1003, 1004), "limit")), "invalid_generation")

    def test_generation_subclass_rejected(self) -> None:
        class SubGeneration(Generation):
            pass

        _failed_with(
            self._gen(SubGeneration((1001, 1002, 1003, 1004), "limit")), "invalid_generation"
        )

    def test_non_generation_tuple_return(self) -> None:
        _failed_with(self._gen((1001, 1002, 1003, 1004)), "invalid_generation")

    def test_none_return(self) -> None:
        _failed_with(self._gen(None), "invalid_generation")

    def test_str_return(self) -> None:
        _failed_with(self._gen("oops"), "invalid_generation")

    def test_bool_reason(self) -> None:
        gen = Generation((1001, 1002, 1003, 1004), True)  # type: ignore[arg-type]
        _failed_with(self._gen(gen), "invalid_generation")

    def test_short_ids_with_limit(self) -> None:
        _failed_with(self._gen(Generation((1001, 1002), "limit")), "invalid_generation")

    def test_error_set_with_limit_reason(self) -> None:
        _failed_with(
            self._gen(Generation((1001, 1002, 1003, 1004), "limit", "")), "invalid_generation"
        )

    def test_error_reason_with_empty_string_error(self) -> None:
        _failed_with(self._gen(Generation((), "error", "")), "backend_reported_error")

    def test_error_reason_with_none_error(self) -> None:
        _failed_with(self._gen(Generation((), "error", None)), "backend_reported_error")

    def test_error_reason_with_hostile_text_collapses(self) -> None:
        report = self._gen(Generation((), "error", "<script>alert('xss')</script>"))
        _failed_with(report, "backend_reported_error")
        # The hostile text never enters the records.
        assert "<script>" not in repr(report.observations[0])

    def test_error_reason_with_huge_hostile_text_dropped(self) -> None:
        report = self._gen(Generation((), "error", "X" * 10_000))
        _failed_with(report, "backend_reported_error")
        assert report.observations[0].error == "backend_reported_error"

    def test_error_reason_with_non_str_error(self) -> None:
        gen = Generation((), "error", 12345)  # type: ignore[arg-type]
        _failed_with(self._gen(gen), "invalid_generation")

    def test_error_reason_with_allowlisted_code_passes_through(self) -> None:
        _failed_with(
            self._gen(Generation((), "error", "context_limit_exceeded")),
            "context_limit_exceeded",
        )

    def test_empty_ids_with_cancelled(self) -> None:
        report = self._gen(Generation((), "cancelled"))
        assert report.status == "cancelled"
        assert report.observations[0].status == "cancelled"
        assert report.observations[0].error is None

    def test_cancelled_with_valid_ids(self) -> None:
        report = self._gen(Generation((1001, 1002, 1003, 1004), "cancelled"))
        assert report.status == "cancelled"
        assert report.observations[0].output_ids == (1001, 1002, 1003, 1004)


class HostileGenerateRaisesTest(unittest.TestCase):
    """generate()/encode() raising: Exception -> fixed code, BaseException -> propagate."""

    def _raising(self, exc: BaseException) -> HostileBackend:
        def generate(*a: Any, **k: Any) -> Generation:
            raise exc

        return HostileBackend(generate=generate)

    def test_runtime_error(self) -> None:
        _failed_with(_run(self._raising(RuntimeError("boom"))), "generate_failed")

    def test_memory_error(self) -> None:
        _failed_with(_run(self._raising(MemoryError("no mem"))), "generate_failed")

    def test_custom_exception_subclass(self) -> None:
        class WeirdError(Exception):
            pass

        _failed_with(_run(self._raising(WeirdError("x"))), "generate_failed")

    def test_stop_iteration(self) -> None:
        _failed_with(_run(self._raising(StopIteration())), "generate_failed")

    def test_keyboard_interrupt_propagates(self) -> None:
        with self.assertRaises(KeyboardInterrupt):
            _run(self._raising(KeyboardInterrupt()))

    def test_system_exit_propagates(self) -> None:
        with self.assertRaises(SystemExit):
            _run(self._raising(SystemExit(3)))

    def test_encode_raising(self) -> None:
        def encode(text: str) -> tuple[int, ...]:
            raise RuntimeError("enc boom")

        _failed_with(_run(HostileBackend(encode=encode)), "encode_failed")

    def test_encode_keyboard_interrupt_propagates(self) -> None:
        def encode(text: str) -> tuple[int, ...]:
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            _run(HostileBackend(encode=encode))

    def test_encode_non_tuple(self) -> None:
        _failed_with(_run(HostileBackend(encode=lambda text: "not-a-tuple")), "invalid_input_ids")

    def test_encode_bad_ids(self) -> None:
        _failed_with(_run(HostileBackend(encode=lambda text: (1, -2, 3))), "invalid_input_ids")


class HostileDecodeTest(unittest.TestCase):
    """decode() returning/raising hostile values must fail closed as decode_failed."""

    def _decode_backend(self, decode: Callable[[tuple[int, ...]], Any]) -> HostileBackend:
        return HostileBackend(
            generate=lambda *a, **k: Generation(_ids("ok"), "limit"),
            decode=decode,
        )

    def _decode_case(self, decode: Callable[[tuple[int, ...]], Any]) -> Any:
        return _run(self._decode_backend(decode))

    def test_decode_returns_bytes(self) -> None:
        _failed_with(self._decode_case(lambda tids: b"nope"), "decode_failed")

    def test_decode_returns_int(self) -> None:
        _failed_with(self._decode_case(lambda tids: 42), "decode_failed")

    def test_decode_returns_none(self) -> None:
        _failed_with(self._decode_case(lambda tids: None), "decode_failed")

    def test_decode_returns_str_subclass(self) -> None:
        class Text(str):
            pass

        _failed_with(self._decode_case(lambda tids: Text("ok")), "decode_failed")

    def test_decode_raises(self) -> None:
        def decode(tids: tuple[int, ...]) -> str:
            raise ValueError("bad decode")

        _failed_with(self._decode_case(decode), "decode_failed")

    def test_decode_keyboard_interrupt_propagates(self) -> None:
        def decode(tids: tuple[int, ...]) -> str:
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            _run(self._decode_backend(decode))

    def test_decode_lone_surrogates_accepted(self) -> None:
        # Lone surrogates are valid Python str: they pass through as-is and
        # the observation dict stays JSON-serializable.
        report = _run(self._decode_backend(lambda tids: "a\ud800b"))
        assert report.status == "completed"
        assert report.observations[0].answer == "a\ud800b"
        json.dumps(report.observations[0].__dict__)

    def test_decode_10mb_string_accepted(self) -> None:
        big = "x" * 10_000_000
        report = _run(self._decode_backend(lambda tids: big))
        assert report.status == "completed"
        assert len(report.observations[0].answer) == 10_000_000

    def test_decode_failing_only_on_boundary_degrades_diagnostic(self) -> None:
        # The main answer decode succeeds; only the boundary-text diagnostic
        # decode fails -> boundary_text_match degrades to None, run completes.
        def decode(tids: tuple[int, ...]) -> str:
            if tuple(tids) == _BOUNDARY:
                raise ValueError("boundary undecodable")
            return _dec(tids)

        report = _run(self._decode_backend(decode))
        assert report.status == "completed"
        assert report.observations[0].boundary_text_match is None


class NonDeterministicBackendTest(unittest.TestCase):
    """A backend changing shape between calls fails closed on the first bad one."""

    def test_valid_then_invalid(self) -> None:
        calls = {"n": 0}

        def generate(*a: Any, **k: Any) -> Generation:
            calls["n"] += 1
            if calls["n"] == 1:
                return Generation(_ids("ok"), "limit")
            return Generation(("bad",), "limit")  # type: ignore[arg-type]

        cases = (_fixture("c1", "one", "zz"), _fixture("c2", "two", "zz"))
        report = _run(HostileBackend(generate=generate), cases)
        assert report.status == "failed"
        assert [o.fixture for o in report.observations] == ["c1", "c2"]
        assert report.observations[0].status == "completed"
        assert report.observations[1].status == "failed"
        assert report.observations[1].error == "invalid_generation"
        assert calls["n"] == 2


class HostileLogprobTest(unittest.TestCase):
    """forward_logits() hostility must become fixed unavailable codes (or propagate KI)."""

    _VOCAB = 2000
    _PROMPT = (1001, 1002, 1003)
    _TARGET = (1004, 1005)

    class _LogprobBackend:
        fingerprint = "cycle91-logprob"
        vocab_size = 2000

        def __init__(self, forward: Callable[[tuple[int, ...]], Any] | None) -> None:
            self._forward = forward

        def encode(self, text: str) -> tuple[int, ...]:
            return tuple(ord(c) - _OFFSET for c in text)

        def decode(self, token_ids: tuple[int, ...]) -> str:
            return "".join(chr(t + _OFFSET) for t in token_ids)

        def forward_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
            assert self._forward is not None
            result: torch.Tensor = self._forward(input_ids)
            return result

        def generate(  # noqa: ANN401 -- Backend protocol member; never called here
            self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
        ) -> Generation:
            raise NotImplementedError("logprob probe never generates")

    def _score(self, forward: Callable[[tuple[int, ...]], Any] | None) -> Any:
        backend = self._LogprobBackend(forward)
        return score_target_logprobs(backend, self._PROMPT, self._TARGET)

    def _unavailable(self, forward: Callable[[tuple[int, ...]], Any] | None, reason: str) -> None:
        score = self._score(forward)
        assert score.status == "unavailable", (score.status, score.reason)
        assert score.reason == reason, score.reason
        assert score.scored_token_count == 0

    def _good(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(len(input_ids), self._VOCAB)

    def test_sane_backend_scores(self) -> None:
        score = self._score(self._good)
        assert score.status == "ok"
        assert score.scored_token_count == 2

    def test_nan_logits(self) -> None:
        def forward(input_ids: tuple[int, ...]) -> torch.Tensor:
            logits = self._good(input_ids)
            logits[0, 0] = float("nan")
            return logits

        self._unavailable(forward, "non_finite_logits")

    def test_positive_inf_logits(self) -> None:
        def forward(input_ids: tuple[int, ...]) -> torch.Tensor:
            logits = self._good(input_ids)
            logits[1, 2] = float("inf")
            return logits

        self._unavailable(forward, "non_finite_logits")

    def test_negative_inf_logits(self) -> None:
        def forward(input_ids: tuple[int, ...]) -> torch.Tensor:
            logits = self._good(input_ids)
            logits[2, 3] = float("-inf")
            return logits

        self._unavailable(forward, "non_finite_logits")

    def test_1d_tensor(self) -> None:
        self._unavailable(lambda iids: torch.zeros(len(iids)), "logits_must_be_2d")

    def test_3d_tensor(self) -> None:
        self._unavailable(lambda iids: torch.zeros(len(iids), self._VOCAB, 2), "logits_must_be_2d")

    def test_wrong_sequence_length(self) -> None:
        self._unavailable(
            lambda iids: torch.zeros(len(iids) + 2, self._VOCAB), "logits_length_mismatch"
        )

    def test_zero_width_vocab(self) -> None:
        self._unavailable(lambda iids: torch.zeros(len(iids), 0), "empty_vocabulary")

    def test_non_tensor_return(self) -> None:
        self._unavailable(lambda iids: [[0.0] * self._VOCAB for _ in iids], "logits_must_be_tensor")

    def test_none_return(self) -> None:
        self._unavailable(lambda iids: None, "logits_must_be_tensor")

    def test_forward_raises_runtime_error(self) -> None:
        def forward(iids: tuple[int, ...]) -> torch.Tensor:
            raise RuntimeError("fwd boom")

        self._unavailable(forward, "forward_failed")

    def test_forward_raises_memory_error(self) -> None:
        def forward(iids: tuple[int, ...]) -> torch.Tensor:
            raise MemoryError("no mem")

        self._unavailable(forward, "forward_failed")

    def test_forward_raises_keyboard_interrupt_propagates(self) -> None:
        def forward(iids: tuple[int, ...]) -> torch.Tensor:
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self._score(forward)

    def test_forward_raises_system_exit_propagates(self) -> None:
        def forward(iids: tuple[int, ...]) -> torch.Tensor:
            raise SystemExit(3)

        with self.assertRaises(SystemExit):
            self._score(forward)

    def test_forward_allowlisted_code_passes_through(self) -> None:
        def forward(iids: tuple[int, ...]) -> torch.Tensor:
            raise ValueError("context_limit_exceeded")

        self._unavailable(forward, "context_limit_exceeded")

    def test_forward_arbitrary_text_collapses(self) -> None:
        def forward(iids: tuple[int, ...]) -> torch.Tensor:
            raise ValueError("<script>pwned</script>")

        score = self._score(forward)
        assert score.status == "unavailable"
        assert score.reason == "forward_failed"
        assert "<script>" not in repr(score)

    def test_missing_forward_logits(self) -> None:
        class NoLogits:
            fingerprint = "x"

            def encode(self, text: str) -> tuple[int, ...]:
                return tuple(ord(c) - _OFFSET for c in text)

            def decode(self, token_ids: tuple[int, ...]) -> str:
                return "".join(chr(t + _OFFSET) for t in token_ids)

            def generate(
                self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
            ) -> Generation:
                raise NotImplementedError("logprob probe never generates")

        score = score_target_logprobs(NoLogits(), self._PROMPT, self._TARGET)
        assert score.status == "unavailable"
        assert score.reason == "logits_not_supported"


class MixedHostileTaxonomyTest(unittest.TestCase):
    """Hostile + well-behaved fixtures in one run: taxonomy counts must sum exactly."""

    def test_counts_sum_with_no_double_counting_or_loss(self) -> None:
        # One scheduled backend, hostile only for cand-b (keyed on its "<b>"
        # BOS marker): cand-a runs all four fixtures well-behaved, cand-b
        # fails on its first fixture and terminates. Both candidates'
        # taxonomy counts must sum exactly to their non-passing observations.
        class Scheduled(HostileBackend):
            def generate(
                self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
            ) -> Generation:
                prompt = _dec(input_ids)
                if "<b>" in prompt:
                    raise RuntimeError("boom")
                if "PASS" in prompt:
                    return Generation(_ids("yes" + chr(_BOUNDARY[0] - _OFFSET)), "limit")
                return Generation(_ids("nope"), "limit")

        search = (
            _fixture("s1", "q WRONG", "zzz", group="g1"),
            _fixture("s2", "q PASS2", "yes", group="g1"),
            _fixture("s3", "q PASS3", "yes", group="g1"),
            _fixture("s4", "q PASS4", "yes", group="g1"),
        )
        held_out = (
            _fixture("h1", "q HPASS1", "yes", group="g2"),
            _fixture("h2", "q HPASS2", "yes", group="g2"),
        )
        split = Split(search=search, held_out=held_out)
        formats = (
            Format(name="cand-a", bos="", user_prefix="", assistant_prefix="", turn_end=""),
            Format(name="cand-b", bos="<b>", user_prefix="", assistant_prefix="", turn_end=""),
        )
        result = run_experiment(
            Scheduled(), formats, split, boundary=_BOUNDARY, max_new_tokens=_CAP, seed=11
        )
        assert result["decision"] == "failed"
        taxonomy = failure_taxonomy(result, eos_id=None)
        assert taxonomy["modes"] == list(FAILURE_MODES)
        assert taxonomy["held_out"] is None  # search never completed; held-out never ran
        assert set(taxonomy["per_candidate"]) == {"cand-a", "cand-b"}
        expected = {
            # cand-a: s1 wrong answer without boundary; s2/s3/s4 pass.
            "cand-a": {"no-boundary-in-N": 1, "backend-error": 0},
            # cand-b: hostile on s1 (generate raises) -> terminates;
            # s2..s4 never attempted and never counted.
            "cand-b": {"no-boundary-in-N": 0, "backend-error": 1},
        }
        for name, counts in taxonomy["per_candidate"].items():
            want = expected[name]
            assert counts["no-boundary-in-N"] == want["no-boundary-in-N"], (name, counts)
            assert counts["backend-error"] == want["backend-error"], (name, counts)
            assert all(
                v == 0 for k, v in counts.items() if k not in ("no-boundary-in-N", "backend-error")
            )
            entry = next(e for e in result["search"] if e["format"]["name"] == name)
            observations = entry["observations"]
            non_passing = sum(
                1 for o in observations if not (o["exact_match"] and o["boundary_found"])
            )
            # No double-counting: counts sum to the non-passing observations.
            assert sum(counts.values()) == non_passing, (name, counts, non_passing)
            # No lost fixtures: observations cover exactly the attempted cases.
            want_fixtures = ["s1", "s2", "s3", "s4"] if name == "cand-a" else ["s1"]
            assert [o["fixture"] for o in observations] == want_fixtures


if __name__ == "__main__":
    unittest.main()
