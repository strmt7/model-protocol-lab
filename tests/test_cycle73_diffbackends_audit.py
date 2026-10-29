"""Cycle 73: adversarial audit of ``diffbackends.py`` comparison semantics.

Cycle 37 audited the divergence classification with the *real* ``emulate_*``
functions as ground truth. This cycle audits the comparison layer *below* the
emulations: every ``emulate_*`` entry point in ``model_protocol_lab.diffbackends``
is stubbed out, so each per-cell result is a fully synthetic, hand-controlled
``(available, fired, detail, truncated_ids, returned_text)`` tuple -- no real
models, no network, no tokenizer. That lets the audit drive cell states the
real emulations never emit (fired-but-no-step contradictions, malformed
``truncated_ids``, unavailable mixes, decode failures) and pin the comparison
rules as pure functions of their inputs.

Method:

- ``_oracle_divergences`` is an independent oracle derived from the module
  docstring's divergence-class rules, written with brute-force loops (no set
  comprehensions, no tuple-key tricks) over plain per-cell snapshots.
- ``MinimalPairTests`` hand-rolls agree/diverge verdict flips that differ by
  one input fact (one firing step, one character of stop text or returned
  text, one decode-map entry).
- ``AdversarialCellTests`` feeds contradictory / malformed / boundary inputs
  (empty id tuples, cap-boundary id lists, unicode stop strings, fired cells
  with no firing step, non-list ``truncated_ids``, truthy non-bool flags,
  decode raising or returning non-str, backends without ``decode``).
- ``ValidatorTests`` pins the fail-closed validation vocabulary: every
  rejection raises ``ValueError`` with a fixed code, never a traceback.
- ``FuzzTests`` runs 300 random synthetic backend-result pairs through both
  the implementation and the oracle and requires identical verdict sets.
- ``SensitivityTests`` copies the package to PID-scoped ``/tmp`` dirs (never
  mutating the repo), plants 6 behavior mutants in the copies, and shows this
  file's tests fail on every mutant while passing on the pristine copy.

Deterministic: all randomness uses string seeds with ``random.Random``.
Local-only: no models, no downloads, no network.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from model_protocol_lab import diffbackends as db_module
from model_protocol_lab.diffbackends import (
    EOS_APPENDED,
    HF_STOP_STRING,
    MAX_DIVERGENCE_OBSERVATIONS,
    MAX_OUTPUT_IDS,
    MAX_PROMPT_IDS,
    SGLANG_STOP,
    VLLM_HOLDBACK_KEEP,
    VLLM_HOLDBACK_STRIP,
    _checked_ids,
    _checked_stop_texts,
    _effective_text,
    _firing_step,
    _observation_ids,
    diff_backends,
    divergence_section_data,
    run_diff_backends,
)

_STOP4 = (HF_STOP_STRING, VLLM_HOLDBACK_STRIP, VLLM_HOLDBACK_KEEP, SGLANG_STOP)
_ALL_SEMANTICS = _STOP4 + (EOS_APPENDED,)
_DISPOSITION = {
    HF_STOP_STRING: "kept",
    VLLM_HOLDBACK_STRIP: "stripped",
    VLLM_HOLDBACK_KEEP: "kept",
    SGLANG_STOP: "stripped",
    EOS_APPENDED: None,
}

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Synthetic cell outcomes and stub emulations.
# ---------------------------------------------------------------------------


@dataclass
class _SynthOutcome:
    """Hand-controlled stand-in for ``backends.StopOutcome``."""

    available: bool = True
    fired: bool = False
    detail: str = "no_stop_within_cap"
    truncated_ids: tuple[int, ...] = ()
    returned_text: str | None = None


def _outcome(
    *,
    available: bool = True,
    fired: bool = False,
    step: int | None = None,
    text: str | None = None,
    ids: tuple[int, ...] = (1,),
    detail: str | None = None,
) -> _SynthOutcome:
    if detail is None:
        detail = f"fired_at_step_{step}" if (fired and step is not None) else "no_stop_within_cap"
    return _SynthOutcome(
        available=available, fired=fired, detail=detail, truncated_ids=ids, returned_text=text
    )


class _FakeBackend:
    """Decode-only backend driven by an explicit id->text map. No models."""

    def __init__(
        self,
        texts: dict[tuple[int, ...], str] | None = None,
        *,
        default: str = "<unmapped>",
        raises: bool = False,
        non_str: bool = False,
    ) -> None:
        self._texts = dict(texts or {})
        self._default = default
        self._raises = raises
        self._non_str = non_str
        self.calls = 0

    def decode(self, token_ids: tuple[int, ...]) -> Any:
        self.calls += 1
        if self._raises:
            raise RuntimeError("c73 fake decode failure")
        text = self._texts.get(tuple(token_ids), self._default)
        return 12345 if self._non_str else text


def _stub_patches(
    spec: dict[tuple[str, str], Any],
    ungated: dict[str, Any],
    eos_spec: dict[tuple[str, str], Any],
) -> list[Any]:
    """Redirect the module's ``emulate_*`` names to synthetic lookups."""

    def fake_hf(
        backend: Any,
        prompt_ids: tuple[int, ...],
        output_ids: tuple[int, ...],
        stop_text: str,
        token_bytes: Any = None,
    ) -> Any:
        return spec.get((HF_STOP_STRING, stop_text), _SynthOutcome())

    def fake_vllm(
        backend: Any,
        output_ids: tuple[int, ...],
        stop_text: str,
        include_stop_str_in_output: bool = False,
        min_tokens: int = 0,
    ) -> Any:
        # The MIN_TOKENS_GATED diagnostic re-run is indistinguishable from
        # a main strip call with min_tokens=0 -- same emulation, same
        # inputs -- so both consult the ungated plan, exactly as reality.
        if min_tokens == 0 and not include_stop_str_in_output and stop_text in ungated:
            return ungated[stop_text]
        key = VLLM_HOLDBACK_KEEP if include_stop_str_in_output else VLLM_HOLDBACK_STRIP
        return spec.get((key, stop_text), _SynthOutcome())

    def fake_sglang(backend: Any, output_ids: tuple[int, ...], stop_text: str) -> Any:
        return spec.get((SGLANG_STOP, stop_text), _SynthOutcome())

    def fake_eos(output_ids: tuple[int, ...], eos_id: int | None) -> Any:
        # The EOS emulation does not depend on the stop text: one shared
        # synthetic outcome per run.
        return eos_spec.get((EOS_APPENDED, "*"), _SynthOutcome())

    return [
        mock.patch.object(db_module, "emulate_hf_stop_string", fake_hf),
        mock.patch.object(db_module, "emulate_vllm_holdback", fake_vllm),
        mock.patch.object(db_module, "emulate_sglang_stop", fake_sglang),
        mock.patch.object(db_module, "emulate_eos_stop", fake_eos),
    ]


# ---------------------------------------------------------------------------
# Independent oracle, derived from the module docstring's class rules.
# ---------------------------------------------------------------------------

_ORACLE_STEP_RE = re.compile(r"fired_at_step_([0-9]+)")


def _oracle_step(detail: str) -> int | None:
    """1-based firing step iff detail is exactly ``fired_at_step_N``."""
    match = _ORACLE_STEP_RE.fullmatch(detail)
    if match is None:
        return None
    return int(match.group(1))


class _Snap:
    """The oracle's evidence snapshot of one synthetic cell."""

    def __init__(self, semantic: str, outcome: Any) -> None:
        self.semantic = semantic
        self.available = bool(outcome.available)
        self.fired = bool(outcome.fired)
        self.step = _oracle_step(outcome.detail)
        self.ids = tuple(outcome.truncated_ids)
        self.text = outcome.returned_text
        self.disposition = _DISPOSITION[semantic]


def _oracle_text(decode: Callable[[tuple[int, ...]], Any], snap: _Snap) -> Any:
    """Text the semantic would return: exposed text, else decoded ids."""
    if snap.text is not None:
        return snap.text
    try:
        result = decode(snap.ids)
    except Exception:  # noqa: BLE001 -- oracle mirrors the degrade contract
        return None
    return result if type(result) is str else None


def _oracle_divergences(
    stops: list[str],
    spec: dict[tuple[str, str], Any],
    ungated: dict[str, Any],
    eos_spec: dict[tuple[str, str], Any],
    min_tokens: int,
    decode: Callable[[tuple[int, ...]], Any],
) -> list[tuple[str, str | None]]:
    """Brute-force expected ``(class, stop_text)`` verdicts per the docstring."""
    # With min_tokens=0 the strip cell IS the ungated emulation (same
    # inputs), so the oracle reads it from the ungated plan, like the stub.
    eff_spec = dict(spec)
    if min_tokens == 0:
        for stop_text in stops:
            if stop_text in ungated:
                eff_spec[(VLLM_HOLDBACK_STRIP, stop_text)] = ungated[stop_text]
    out: list[tuple[str, str | None]] = []
    per_stop: dict[str, dict[str, _Snap]] = {}
    for stop_text in stops:
        cells: dict[str, _Snap] = {}
        for semantic in _ALL_SEMANTICS:
            if semantic == EOS_APPENDED:
                outcome = eos_spec.get((EOS_APPENDED, "*"), _SynthOutcome())
            else:
                outcome = eff_spec.get((semantic, stop_text), _SynthOutcome())
            cells[semantic] = _Snap(semantic, outcome)
        per_stop[stop_text] = cells
        live: list[_Snap] = []
        for semantic in _STOP4:
            snap = cells[semantic]
            if snap.available:
                live.append(snap)

        # FIRING_STEP_DIFFERS: >1 distinct firing step among live cells
        # (None = did not fire counts as a value).
        distinct: list[int | None] = []
        for snap in live:
            if snap.step not in distinct:
                distinct.append(snap.step)
        if len(distinct) > 1:
            out.append(("FIRING_STEP_DIFFERS", stop_text))

        # KEPT_VS_STRIPPED: both fired, kept text established and different.
        hf = cells[HF_STOP_STRING]
        strip = cells[VLLM_HOLDBACK_STRIP]
        if hf.available and hf.fired and strip.available and strip.fired:
            kept = _oracle_text(decode, hf)
            if kept is not None and kept != strip.text:
                out.append(("KEPT_VS_STRIPPED", stop_text))

        # MIN_TOKENS_GATED (+ nested NO_MIN_TOKENS_GATE): shared ungated
        # diagnostic re-run.
        if min_tokens > 0 and strip.available and not strip.fired:
            un = ungated.get(stop_text, _SynthOutcome())
            if bool(un.available) and bool(un.fired):
                out.append(("MIN_TOKENS_GATED", stop_text))
                sglang = cells[SGLANG_STOP]
                if sglang.available and sglang.fired and sglang.step is not None:
                    out.append(("NO_MIN_TOKENS_GATE", stop_text))

        # EOS_PRECEDENCE: EOS at/before the minimum stop-semantic step.
        eos = cells[EOS_APPENDED]
        if eos.available and eos.fired and eos.step is not None:
            earliest: int | None = None
            for snap in live:
                if (
                    snap.fired
                    and snap.step is not None
                    and (earliest is None or snap.step < earliest)
                ):
                    earliest = snap.step
            if earliest is not None and eos.step <= earliest:
                out.append(("EOS_PRECEDENCE", stop_text))

        # STR_OVER_EOS_PRECEDENCE: sglang and EOS tie on the same step.
        sglang = cells[SGLANG_STOP]
        if (
            sglang.available
            and sglang.fired
            and sglang.step is not None
            and eos.available
            and eos.fired
            and eos.step == sglang.step
        ):
            out.append(("STR_OVER_EOS_PRECEDENCE", stop_text))

        # HARNESS_NORMALIZATION_RELIED: a stripped-disposition cell whose
        # text still contains the stop string.
        for snap in live:
            if not snap.fired:
                continue
            if snap.disposition != "stripped":
                continue
            text = _oracle_text(decode, snap)
            if text is not None and stop_text in text:
                out.append(("HARNESS_NORMALIZATION_RELIED", stop_text))

    # MULTI_STOP_SELECTION: sglang's first-listed rule vs vLLM's
    # earliest-completion rule, with exact tie-breaks.
    sglang_best_key: tuple[int, int] | None = None
    sglang_best: tuple[str, int] | None = None
    vllm_best_key: tuple[int, int, int] | None = None
    vllm_best: tuple[str, int] | None = None
    for list_index, stop_text in enumerate(stops):
        cells = per_stop[stop_text]
        sglang = cells[SGLANG_STOP]
        strip = cells[VLLM_HOLDBACK_STRIP]
        keep = cells[VLLM_HOLDBACK_KEEP]
        if sglang.available and sglang.fired and sglang.step is not None:
            skey = (sglang.step, list_index)
            if sglang_best_key is None or skey < sglang_best_key:
                sglang_best_key = skey
                sglang_best = (stop_text, sglang.step)
        if (
            strip.available
            and strip.fired
            and strip.step is not None
            and keep.available
            and keep.fired
            and keep.text is not None
        ):
            vkey = (strip.step, len(keep.text), list_index)
            if vllm_best_key is None or vkey < vllm_best_key:
                vllm_best_key = vkey
                vllm_best = (stop_text, strip.step)
    if sglang_best != vllm_best:
        if sglang_best is not None:
            winner_stop: str | None = sglang_best[0]
        elif vllm_best is not None:
            winner_stop = vllm_best[0]
        else:
            winner_stop = None
        out.append(("MULTI_STOP_SELECTION", winner_stop))
    return sorted(out, key=lambda pair: (pair[0], pair[1] or ""))


def _div_pairs(result: dict[str, Any]) -> list[tuple[str, str | None]]:
    return sorted(
        ((finding["class"], finding["stop_text"]) for finding in result["divergences"]),
        key=lambda pair: (pair[0], pair[1] or ""),
    )


def _run(
    stops: tuple[str, ...] | list[str],
    spec: dict[tuple[str, str], Any],
    *,
    eos_spec: dict[tuple[str, str], Any] | None = None,
    ungated: dict[str, Any] | None = None,
    min_tokens: int = 0,
    eos_id: int | None = None,
    decode_map: dict[tuple[int, ...], str] | None = None,
    decode_default: str = "<dflt>",
    decode_raises: bool = False,
    decode_non_str: bool = False,
    prompt_ids: tuple[int, ...] = (7, 8),
    output_ids: tuple[int, ...] = (9,),
) -> tuple[dict[str, Any], Callable[[tuple[int, ...]], Any]]:
    """Run ``diff_backends`` with stubbed emulations; return result + decode."""
    backend = _FakeBackend(
        decode_map, default=decode_default, raises=decode_raises, non_str=decode_non_str
    )
    with contextlib.ExitStack() as stack:
        for patcher in _stub_patches(dict(spec), dict(ungated or {}), dict(eos_spec or {})):
            stack.enter_context(patcher)
        result = diff_backends(
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            stop_texts=stops,
            eos_id=eos_id,
            min_tokens=min_tokens,
            backend=backend,
        )
    return result, backend.decode


def _check(
    testcase: unittest.TestCase,
    stops: tuple[str, ...] | list[str],
    spec: dict[tuple[str, str], Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Assert the implementation's verdict set equals the oracle's."""
    result, decode = _run(stops, spec, **kwargs)
    expected = _oracle_divergences(
        list(stops),
        dict(spec),
        dict(kwargs.get("ungated") or {}),
        dict(kwargs.get("eos_spec") or {}),
        int(kwargs.get("min_tokens", 0)),
        decode,
    )
    testcase.assertEqual(_div_pairs(result), expected)
    return result


# ---------------------------------------------------------------------------
# Fail-closed validation: fixed error vocabulary, never tracebacks.
# ---------------------------------------------------------------------------


class ValidatorTests(unittest.TestCase):
    @staticmethod
    def _good_kwargs() -> dict[str, Any]:
        return {
            "prompt_ids": (1, 2),
            "output_ids": (3,),
            "stop_texts": ("STOP",),
            "eos_id": None,
            "min_tokens": 0,
            "backend": _FakeBackend(),
        }

    def _expect_code(self, code: str, **overrides: Any) -> None:
        kwargs = self._good_kwargs()
        kwargs.update(overrides)
        with self.assertRaises(ValueError) as ctx, contextlib.ExitStack() as stack:
            for patcher in _stub_patches({}, {}, {}):
                stack.enter_context(patcher)
            diff_backends(**kwargs)
        self.assertEqual(str(ctx.exception), code)

    def test_checked_ids_rejects_non_tuple(self) -> None:
        self._expect_code("invalid_prompt_ids", prompt_ids=[1, 2])
        self._expect_code("invalid_output_ids", output_ids="ab")

    def test_checked_ids_rejects_bool_negative_huge(self) -> None:
        self._expect_code("invalid_prompt_ids", prompt_ids=(1, True))
        self._expect_code("invalid_prompt_ids", prompt_ids=(1, -2))
        self._expect_code("invalid_output_ids", output_ids=(2**31,))
        self._expect_code("invalid_prompt_ids", prompt_ids=(1, 2**31 - 1, 2**31))

    def test_checked_ids_boundary_values(self) -> None:
        self.assertEqual(_checked_ids((0, 2**31 - 1), "prompt_ids", cap=8), (0, 2**31 - 1))
        self.assertEqual(_checked_ids((), "output_ids", cap=8), ())

    def test_checked_ids_cap_boundary(self) -> None:
        at_cap = _checked_ids((0,) * MAX_OUTPUT_IDS, "output_ids", cap=MAX_OUTPUT_IDS)
        self.assertEqual(len(at_cap), MAX_OUTPUT_IDS)
        with self.assertRaises(ValueError) as ctx:
            _checked_ids((0,) * (MAX_OUTPUT_IDS + 1), "output_ids", cap=MAX_OUTPUT_IDS)
        self.assertEqual(str(ctx.exception), "output_ids_too_long")
        with self.assertRaises(ValueError) as ctx:
            _checked_ids((0,) * (MAX_PROMPT_IDS + 1), "prompt_ids", cap=MAX_PROMPT_IDS)
        self.assertEqual(str(ctx.exception), "prompt_ids_too_long")
        self._expect_code("prompt_ids_too_long", prompt_ids=(0,) * (MAX_PROMPT_IDS + 1))
        self._expect_code("output_ids_too_long", output_ids=(0,) * (MAX_OUTPUT_IDS + 1))

    def test_stop_texts_vocabulary(self) -> None:
        self._expect_code("invalid_stop_texts", stop_texts="STOP")
        self._expect_code("invalid_stop_texts", stop_texts={"STOP"})
        self._expect_code("invalid_stop_texts", stop_texts=[""])
        self._expect_code("invalid_stop_texts", stop_texts=[123])
        self._expect_code("invalid_stop_texts", stop_texts=[None])
        self._expect_code("stop_text_too_long", stop_texts=["x" * 1025])
        self._expect_code("empty_stop_texts", stop_texts=[])
        self._expect_code("empty_stop_texts", stop_texts=())
        self._expect_code("too_many_stop_texts", stop_texts=[f"stop{i}" for i in range(17)])

    def test_stop_texts_dedup_and_bounds(self) -> None:
        self.assertEqual(_checked_stop_texts(["A", "B", "A", "B", "C"]), ("A", "B", "C"))
        self.assertEqual(_checked_stop_texts(("x" * 1024,)), ("x" * 1024,))
        self.assertEqual(len(_checked_stop_texts([f"s{i}" for i in range(16)])), 16)
        self.assertEqual(_checked_stop_texts(["unicode-停止-🔚"]), ("unicode-停止-🔚",))

    def test_eos_id_and_min_tokens_vocabulary(self) -> None:
        self._expect_code("invalid_eos_id", eos_id=-1)
        self._expect_code("invalid_eos_id", eos_id=True)
        self._expect_code("invalid_eos_id", eos_id="5")
        self._expect_code("invalid_min_tokens", min_tokens=-1)
        self._expect_code("invalid_min_tokens", min_tokens=True)
        self._expect_code("invalid_min_tokens", min_tokens=1.5)

    def test_backend_vocabulary(self) -> None:
        self._expect_code("invalid_backend", backend=object())
        self._expect_code("invalid_backend", backend=SimpleNamespace(decode=None))
        self._expect_code("invalid_backend", backend=SimpleNamespace(decode="not-callable"))

    def test_firing_step_vocabulary(self) -> None:
        cases = {
            "fired_at_step_1": 1,
            "fired_at_step_0": 0,
            "fired_at_step_007": 7,
            "fired_at_step_4096": 4096,
            "no_stop_within_cap": None,
            "gated_by_min_tokens": None,
            "decode_unavailable": None,
            "invalid_stop_text": None,
            "fired_at_step_": None,
            "fired_at_step_12x": None,
            "xfired_at_step_1": None,
            "fired_at_step_1 ": None,
            " fired_at_step_1": None,
            "Fired_At_Step_1": None,
            "FIRED_AT_STEP_1": None,
            "": None,
            "fired_at_step_-1": None,
            "fired_at_step_1.5": None,
        }
        for detail, expected in cases.items():
            with self.subTest(detail=detail):
                self.assertEqual(_firing_step(detail), expected)

    def test_effective_text_paths(self) -> None:
        from model_protocol_lab.diffbackends import CellResult

        boom = _FakeBackend(raises=True)
        # vLLM-style: returned_text wins, decode never consulted.
        cell = CellResult("v", True, True, 2, (1,), "rt", "stripped")
        self.assertEqual(_effective_text(boom, cell), "rt")
        self.assertEqual(boom.calls, 0)
        # Decode path.
        cell = CellResult("h", True, True, 2, (4,), None, "kept")
        self.assertEqual(_effective_text(_FakeBackend({(4,): "dec"}), cell), "dec")
        # Decode raising degrades to None.
        self.assertIsNone(_effective_text(boom, cell))
        # Decode returning non-str degrades to None.
        self.assertIsNone(_effective_text(_FakeBackend(non_str=True), cell))
        # No decode attribute degrades to None.
        self.assertIsNone(_effective_text(object(), cell))

    def test_observation_ids(self) -> None:
        self.assertEqual(_observation_ids({"output_ids": [1, 2]}, "output_ids"), (1, 2))
        self.assertEqual(_observation_ids({"output_ids": (1, 2)}, "output_ids"), (1, 2))
        self.assertEqual(_observation_ids({"input_ids": []}, "input_ids"), ())
        self.assertIsNone(_observation_ids({"output_ids": [1.5]}, "output_ids"))
        self.assertIsNone(_observation_ids({"output_ids": "ab"}, "output_ids"))
        self.assertIsNone(_observation_ids({"output_ids": [True]}, "output_ids"))
        self.assertIsNone(_observation_ids({"output_ids": [2**31]}, "output_ids"))
        self.assertIsNone(_observation_ids({"output_ids": None}, "output_ids"))
        self.assertIsNone(_observation_ids({}, "output_ids"))
        self.assertIsNone(
            _observation_ids({"output_ids": [0] * (MAX_OUTPUT_IDS + 1)}, "output_ids")
        )
        self.assertIsNone(_observation_ids({"input_ids": [0] * (MAX_PROMPT_IDS + 1)}, "input_ids"))

    def test_cell_converts_truncated_ids_to_tuple(self) -> None:
        # A malformed (list) truncated_ids is normalized, not fatal.
        bad = SimpleNamespace(
            available=True,
            fired=True,
            detail="fired_at_step_2",
            truncated_ids=[1, 2],
            returned_text="t",
        )
        with contextlib.ExitStack() as stack:
            for patcher in _stub_patches({(VLLM_HOLDBACK_STRIP, "S"): bad}, {}, {}):
                stack.enter_context(patcher)
            result = diff_backends(
                prompt_ids=(1,),
                output_ids=(2,),
                stop_texts=("S",),
                eos_id=None,
                min_tokens=0,
                backend=_FakeBackend(),
            )
        cell = result["cells"][0]["semantics"][VLLM_HOLDBACK_STRIP]
        self.assertEqual(cell["truncated_ids"], [1, 2])
        self.assertTrue(cell["fired"])

    def test_tokenizer_backend_decode_validation(self) -> None:
        from model_protocol_lab.diffbackends import _TokenizerBackend

        class _Tok:
            @staticmethod
            def decode(ids: list[int], **kwargs: Any) -> str:
                return "".join(f"<{i}>" for i in ids)

        backend = _TokenizerBackend(_Tok(), 8)
        self.assertEqual(backend.decode((1, 2)), "<1><2>")
        with self.assertRaises(ValueError) as ctx:
            backend.decode([1, 2])  # type: ignore[arg-type]
        self.assertEqual(str(ctx.exception), "invalid_decode_ids")
        with self.assertRaises(ValueError) as ctx:
            backend.decode((8,))  # at the vocab bound
        self.assertEqual(str(ctx.exception), "invalid_decode_ids")
        with self.assertRaises(ValueError) as ctx:
            _TokenizerBackend(_Tok(), 0)
        self.assertEqual(str(ctx.exception), "invalid_vocab_size")
        with self.assertRaises(ValueError) as ctx:
            _TokenizerBackend(object(), 8)
        self.assertEqual(str(ctx.exception), "tokenizer_missing_decode")


# ---------------------------------------------------------------------------
# Minimal pairs: one input fact flips the verdict.
# ---------------------------------------------------------------------------


class MinimalPairTests(unittest.TestCase):
    def test_all_agree_baseline(self) -> None:
        decode_map = {(10,): "out", (11,): "x", (12,): "x", (13,): "x"}
        spec = {
            (HF_STOP_STRING, "S"): _outcome(fired=True, step=2, ids=(10,)),
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(fired=True, step=2, text="out", ids=(11,)),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(fired=True, step=2, text="outS", ids=(12,)),
            (SGLANG_STOP, "S"): _outcome(fired=True, step=2, text="out", ids=(13,)),
        }
        result = _check(self, ("S",), spec, decode_map=decode_map)
        self.assertEqual(result["verdict"], "agree")
        self.assertEqual(result["divergences"], [])
        self.assertEqual(result["prompt_tokens"], 2)
        self.assertEqual(result["output_tokens"], 1)

    def test_firing_step_differs_minimal(self) -> None:
        base = {
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(fired=True, step=2, text="out"),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(fired=True, step=2, text="out"),
            (SGLANG_STOP, "S"): _outcome(fired=True, step=2, text="out"),
        }
        agree_spec = {**base, (HF_STOP_STRING, "S"): _outcome(fired=True, step=2)}
        result = _check(self, ("S",), agree_spec, decode_map={(1,): "out"})
        self.assertEqual(result["verdict"], "agree")
        div_spec = {**base, (HF_STOP_STRING, "S"): _outcome(fired=True, step=1)}
        result = _check(self, ("S",), div_spec, decode_map={(1,): "out"})
        self.assertEqual(_div_pairs(result), [("FIRING_STEP_DIFFERS", "S")])

    def test_firing_step_all_none_is_one_value(self) -> None:
        spec = {
            (HF_STOP_STRING, "S"): _outcome(),
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(),
            (SGLANG_STOP, "S"): _outcome(),
        }
        result = _check(self, ("S",), spec)
        self.assertEqual(result["verdict"], "agree")

    def test_kept_vs_stripped_one_char_flip(self) -> None:
        decode_map = {(10,): "out"}
        base = {
            (HF_STOP_STRING, "S"): _outcome(fired=True, step=2, ids=(10,)),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(fired=True, step=2, text="out"),
            (SGLANG_STOP, "S"): _outcome(fired=True, step=2, text="out"),
        }
        div_spec = {**base, (VLLM_HOLDBACK_STRIP, "S"): _outcome(fired=True, step=2, text="ou")}
        result = _check(self, ("S",), div_spec, decode_map=decode_map)
        self.assertEqual(_div_pairs(result), [("KEPT_VS_STRIPPED", "S")])
        finding = result["divergences"][0]
        self.assertEqual(finding["class"], "KEPT_VS_STRIPPED")
        self.assertEqual(finding["stop_text"], "S")
        agree_spec = {**base, (VLLM_HOLDBACK_STRIP, "S"): _outcome(fired=True, step=2, text="out")}
        result = _check(self, ("S",), agree_spec, decode_map=decode_map)
        self.assertEqual(result["verdict"], "agree")

    def test_kept_text_unavailable_degrades(self) -> None:
        spec = {
            (HF_STOP_STRING, "S"): _outcome(fired=True, step=2, ids=(10,)),
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(fired=True, step=2, text="different"),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(fired=True, step=2),
            (SGLANG_STOP, "S"): _outcome(fired=True, step=2, text="different"),
        }
        # Decode explodes -> kept text cannot be established -> no KEPT_VS_STRIPPED.
        result = _check(self, ("S",), spec, decode_raises=True)
        self.assertNotIn(("KEPT_VS_STRIPPED", "S"), _div_pairs(result))

    def test_harness_strip_contains_stop(self) -> None:
        spec = {
            (HF_STOP_STRING, "S"): _outcome(fired=True, step=2, ids=(10,)),
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(fired=True, step=2, text="xSTOPx"),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(fired=True, step=2, text="xSTOPx"),
            (SGLANG_STOP, "S"): _outcome(fired=True, step=2, text="clean"),
        }
        result = _check(self, ("S",), spec, decode_map={(10,): "xSTOPx"})
        pairs = _div_pairs(result)
        self.assertEqual(pairs, [("HARNESS_NORMALIZATION_RELIED", "S")])
        # Exactly one HARNESS finding: the strip cell only (kept cells never flagged).
        self.assertEqual(
            [p for p in pairs if p[0] == "HARNESS_NORMALIZATION_RELIED"],
            [("HARNESS_NORMALIZATION_RELIED", "S")],
        )

    def test_harness_sglang_via_decode_path(self) -> None:
        spec = {
            (HF_STOP_STRING, "S"): _outcome(fired=True, step=2, ids=(10,)),
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(fired=True, step=2, text="clean"),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(fired=True, step=2),
            (SGLANG_STOP, "S"): _outcome(fired=True, step=2, ids=(14,)),
        }
        result = _check(
            self, ("S",), spec, decode_map={(10,): "kept text", (14,): "has STOP inside"}
        )
        self.assertIn(("HARNESS_NORMALIZATION_RELIED", "S"), _div_pairs(result))

    def test_min_tokens_gated_minimal(self) -> None:
        spec = {
            (HF_STOP_STRING, "S"): _outcome(),
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(),  # gated: not fired
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(),
            (SGLANG_STOP, "S"): _outcome(),
        }
        ungated = {"S": _outcome(fired=True, step=3)}
        result = _check(self, ("S",), spec, ungated=ungated, min_tokens=2)
        self.assertEqual(_div_pairs(result), [("MIN_TOKENS_GATED", "S")])
        self.assertIn("vLLM PR #22014", result["divergences"][0]["detail"])

    def test_no_min_tokens_gate_nested(self) -> None:
        spec = {
            (HF_STOP_STRING, "S"): _outcome(),
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(),
            (SGLANG_STOP, "S"): _outcome(fired=True, step=3, text="clean"),
        }
        ungated = {"S": _outcome(fired=True, step=3)}
        result = _check(self, ("S",), spec, ungated=ungated, min_tokens=2)
        # The gate scenario inherently splits firing steps (strip gated at
        # None while sglang fires) and leaves vLLM without a winner, so
        # FIRING_STEP_DIFFERS and MULTI_STOP_SELECTION co-fire.
        self.assertEqual(
            _div_pairs(result),
            [
                ("FIRING_STEP_DIFFERS", "S"),
                ("MIN_TOKENS_GATED", "S"),
                ("MULTI_STOP_SELECTION", "S"),
                ("NO_MIN_TOKENS_GATE", "S"),
            ],
        )

    def test_min_tokens_gate_needs_ungated_fire(self) -> None:
        spec = {
            (HF_STOP_STRING, "S"): _outcome(),
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(),
            (SGLANG_STOP, "S"): _outcome(),
        }
        # min_tokens>0, strip not fired, but the ungated re-run also does
        # not fire -> no gate classes.
        result = _check(self, ("S",), spec, ungated={"S": _outcome()}, min_tokens=2)
        self.assertEqual(result["verdict"], "agree")

    def test_min_tokens_zero_never_gates(self) -> None:
        spec = {
            (HF_STOP_STRING, "S"): _outcome(),
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(),
            (SGLANG_STOP, "S"): _outcome(),
        }
        result = _check(self, ("S",), spec, min_tokens=0)
        self.assertEqual(result["verdict"], "agree")

    def test_eos_precedence_at_and_before(self) -> None:
        def spec_with(eos_step: int) -> dict[tuple[str, str], _SynthOutcome]:
            return {
                (HF_STOP_STRING, "S"): _outcome(fired=True, step=2),
                (VLLM_HOLDBACK_STRIP, "S"): _outcome(fired=True, step=2),
                (VLLM_HOLDBACK_KEEP, "S"): _outcome(fired=True, step=2),
                (SGLANG_STOP, "S"): _outcome(fired=True, step=3, text="clean"),
            }

        eos_at = {(EOS_APPENDED, "*"): _outcome(fired=True, step=2)}
        result = _check(self, ("S",), spec_with(2), eos_spec=eos_at)
        self.assertIn(("EOS_PRECEDENCE", "S"), _div_pairs(result))
        eos_before = {(EOS_APPENDED, "*"): _outcome(fired=True, step=1)}
        result = _check(self, ("S",), spec_with(1), eos_spec=eos_before)
        self.assertIn(("EOS_PRECEDENCE", "S"), _div_pairs(result))
        # Off by one: EOS strictly after the earliest stop -> silent.
        eos_after = {(EOS_APPENDED, "*"): _outcome(fired=True, step=3)}
        result = _check(self, ("S",), spec_with(3), eos_spec=eos_after)
        self.assertNotIn(("EOS_PRECEDENCE", "S"), _div_pairs(result))

    def test_eos_precedence_uses_min_over_stop_cells(self) -> None:
        spec = {
            (HF_STOP_STRING, "S"): _outcome(fired=True, step=5),
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(fired=True, step=2),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(fired=True, step=5),
            (SGLANG_STOP, "S"): _outcome(fired=True, step=5, text="clean"),
        }
        eos_spec = {(EOS_APPENDED, "*"): _outcome(fired=True, step=3)}
        result = _check(self, ("S",), spec, eos_spec=eos_spec)
        self.assertNotIn(("EOS_PRECEDENCE", "S"), _div_pairs(result))
        eos_spec = {(EOS_APPENDED, "*"): _outcome(fired=True, step=2)}
        result = _check(self, ("S",), spec, eos_spec=eos_spec)
        self.assertIn(("EOS_PRECEDENCE", "S"), _div_pairs(result))

    def test_str_over_eos_tie(self) -> None:
        spec = {
            (HF_STOP_STRING, "S"): _outcome(fired=True, step=2),
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(fired=True, step=2),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(fired=True, step=2),
            (SGLANG_STOP, "S"): _outcome(fired=True, step=2, text="clean"),
        }
        eos_spec = {(EOS_APPENDED, "*"): _outcome(fired=True, step=2)}
        result = _check(self, ("S",), spec, eos_spec=eos_spec)
        pairs = _div_pairs(result)
        self.assertIn(("STR_OVER_EOS_PRECEDENCE", "S"), pairs)
        self.assertIn(("EOS_PRECEDENCE", "S"), pairs)  # co-fires on the tie

    def test_str_over_eos_without_eos_precedence(self) -> None:
        # HF fires strictly earlier via prompt overhang; sglang and EOS tie
        # at step 2 -> STR_OVER_EOS_PRECEDENCE without EOS_PRECEDENCE.
        spec = {
            (HF_STOP_STRING, "S"): _outcome(fired=True, step=1),
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(fired=True, step=2),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(fired=True, step=2),
            (SGLANG_STOP, "S"): _outcome(fired=True, step=2, text="clean"),
        }
        eos_spec = {(EOS_APPENDED, "*"): _outcome(fired=True, step=2)}
        result = _check(self, ("S",), spec, eos_spec=eos_spec)
        pairs = _div_pairs(result)
        self.assertIn(("STR_OVER_EOS_PRECEDENCE", "S"), pairs)
        self.assertNotIn(("EOS_PRECEDENCE", "S"), pairs)

    def test_multi_stop_selection_diverge(self) -> None:
        # SGLang: AAA@2 (list 0) beats B@2 on list order. vLLM: B completes
        # earlier (shorter keep text) -> winners differ.
        spec = {
            (SGLANG_STOP, "AAA"): _outcome(fired=True, step=2, text="t"),
            (VLLM_HOLDBACK_STRIP, "AAA"): _outcome(fired=True, step=2, text="t"),
            (VLLM_HOLDBACK_KEEP, "AAA"): _outcome(fired=True, step=2, text="x" * 20),
            (SGLANG_STOP, "B"): _outcome(fired=True, step=2, text="t"),
            (VLLM_HOLDBACK_STRIP, "B"): _outcome(fired=True, step=2, text="t"),
            (VLLM_HOLDBACK_KEEP, "B"): _outcome(fired=True, step=2, text="yyyy"),
        }
        result = _check(self, ("AAA", "B"), spec)
        pairs = _div_pairs(result)
        self.assertIn(("MULTI_STOP_SELECTION", "AAA"), pairs)
        winners = [d for d in result["divergences"] if d["class"] == "MULTI_STOP_SELECTION"]
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0]["stop_text"], "AAA")  # sglang pick preferred
        self.assertIn("sglang_stop selects", winners[0]["detail"])
        self.assertIn("vllm_holdback_strip selects", winners[0]["detail"])

    def test_multi_stop_selection_agree(self) -> None:
        spec = {
            (SGLANG_STOP, "AAA"): _outcome(fired=True, step=2, text="t"),
            (VLLM_HOLDBACK_STRIP, "AAA"): _outcome(fired=True, step=2, text="t"),
            (VLLM_HOLDBACK_KEEP, "AAA"): _outcome(fired=True, step=2, text="zzzz"),
            (SGLANG_STOP, "B"): _outcome(fired=True, step=3, text="t"),
            (VLLM_HOLDBACK_STRIP, "B"): _outcome(fired=True, step=3, text="t"),
            (VLLM_HOLDBACK_KEEP, "B"): _outcome(fired=True, step=3, text="w"),
        }
        result = _check(self, ("AAA", "B"), spec)
        self.assertNotIn("MULTI_STOP_SELECTION", [p[0] for p in _div_pairs(result)])

    def test_multi_stop_sglang_tiebreak_is_list_order(self) -> None:
        # Both stops fire at step 2 under sglang; the first-listed must win.
        # vLLM agrees (identical completion keys) -> no divergence.
        spec = {
            (SGLANG_STOP, "X"): _outcome(fired=True, step=2, text="t"),
            (VLLM_HOLDBACK_STRIP, "X"): _outcome(fired=True, step=2, text="t"),
            (VLLM_HOLDBACK_KEEP, "X"): _outcome(fired=True, step=2, text="qqqqq"),
            (SGLANG_STOP, "Y"): _outcome(fired=True, step=2, text="t"),
            (VLLM_HOLDBACK_STRIP, "Y"): _outcome(fired=True, step=2, text="t"),
            (VLLM_HOLDBACK_KEEP, "Y"): _outcome(fired=True, step=2, text="wwwww"),
        }
        result = _check(self, ("X", "Y"), spec)
        self.assertNotIn("MULTI_STOP_SELECTION", [p[0] for p in _div_pairs(result)])

    def test_multi_stop_neither_fires(self) -> None:
        result = _check(self, ("A", "B"), {})
        self.assertNotIn("MULTI_STOP_SELECTION", [p[0] for p in _div_pairs(result)])

    def test_multi_stop_vllm_needs_keep_text(self) -> None:
        # Strip fired but keep unavailable -> vLLM cannot name a winner;
        # sglang fired -> winners differ (sglang pick vs no stop).
        spec = {
            (SGLANG_STOP, "S"): _outcome(fired=True, step=2, text="t"),
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(fired=True, step=2, text="t"),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(available=False),
        }
        result = _check(self, ("S",), spec)
        pairs = _div_pairs(result)
        self.assertIn(("MULTI_STOP_SELECTION", "S"), pairs)
        winners = [d for d in result["divergences"] if d["class"] == "MULTI_STOP_SELECTION"]
        self.assertIn("no stop", winners[0]["detail"])

    def test_stop_text_one_char_diff(self) -> None:
        # The strip text contains "STOP" but not "STQP": HARNESS fires for
        # the first stop only -- a one-character stop difference flips it.
        spec = {
            (VLLM_HOLDBACK_STRIP, "STOP"): _outcome(fired=True, step=2, text="aSTOPb"),
            (VLLM_HOLDBACK_STRIP, "STQP"): _outcome(fired=True, step=2, text="aSTOPb"),
        }
        result = _check(self, ("STOP", "STQP"), spec)
        pairs = _div_pairs(result)
        self.assertIn(("HARNESS_NORMALIZATION_RELIED", "STOP"), pairs)
        self.assertNotIn(("HARNESS_NORMALIZATION_RELIED", "STQP"), pairs)

    def test_result_key_sets(self) -> None:
        result = _check(self, ("S",), {})
        self.assertEqual(
            set(result),
            {
                "prompt_tokens",
                "output_tokens",
                "min_tokens",
                "eos_id",
                "stop_texts",
                "cells",
                "divergences",
                "verdict",
            },
        )
        record = result["cells"][0]
        self.assertEqual(set(record), {"stop_text", "semantics"})
        self.assertEqual(
            set(record["semantics"]),
            {HF_STOP_STRING, VLLM_HOLDBACK_STRIP, VLLM_HOLDBACK_KEEP, SGLANG_STOP, EOS_APPENDED},
        )
        cell = record["semantics"][HF_STOP_STRING]
        self.assertEqual(
            set(cell),
            {"available", "fired", "firing_step", "truncated_ids", "returned_text", "disposition"},
        )
        self.assertEqual(result["stop_texts"], ["S"])


# ---------------------------------------------------------------------------
# Adversarial cells: contradictory, malformed, and boundary inputs.
# ---------------------------------------------------------------------------


class AdversarialCellTests(unittest.TestCase):
    def test_fired_without_step_is_none_valued(self) -> None:
        # Contradictory input the real emulations never emit: fired=True but
        # the detail carries no step. None counts as a distinct step value.
        spec = {
            (HF_STOP_STRING, "S"): _outcome(fired=True, detail="no_stop_within_cap"),
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(fired=True, step=2),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(fired=True, step=2),
            (SGLANG_STOP, "S"): _outcome(fired=True, step=2, text="clean"),
        }
        result = _check(self, ("S",), spec)
        self.assertIn(("FIRING_STEP_DIFFERS", "S"), _div_pairs(result))

    def test_eos_fired_without_step_never_counts(self) -> None:
        spec = {
            (HF_STOP_STRING, "S"): _outcome(fired=True, step=2),
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(fired=True, step=2),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(fired=True, step=2),
            (SGLANG_STOP, "S"): _outcome(fired=True, step=3, text="clean"),
        }
        eos_spec = {(EOS_APPENDED, "*"): _outcome(fired=True, detail="no_stop_within_cap")}
        result = _check(self, ("S",), spec, eos_spec=eos_spec)
        pairs = _div_pairs(result)
        self.assertNotIn(("EOS_PRECEDENCE", "S"), pairs)
        self.assertNotIn(("STR_OVER_EOS_PRECEDENCE", "S"), pairs)

    def test_unavailable_cells_excluded(self) -> None:
        spec = {
            (HF_STOP_STRING, "S"): _outcome(available=False, fired=True, step=9),
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(fired=True, step=2, text="out"),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(fired=True, step=2, text="out"),
            (SGLANG_STOP, "S"): _outcome(fired=True, step=2, text="out"),
        }
        result = _check(self, ("S",), spec, decode_map={(1,): "zzz"})
        self.assertNotIn(("FIRING_STEP_DIFFERS", "S"), _div_pairs(result))

    def test_all_unavailable_agrees(self) -> None:
        spec = {
            (HF_STOP_STRING, "S"): _outcome(available=False),
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(available=False),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(available=False),
            (SGLANG_STOP, "S"): _outcome(available=False),
        }
        eos_spec = {(EOS_APPENDED, "*"): _outcome(available=False)}
        result = _check(self, ("S",), spec, eos_spec=eos_spec)
        self.assertEqual(result["verdict"], "agree")

    def test_strip_fired_without_returned_text(self) -> None:
        # kept "x" vs strip.returned_text None: the inequality still fires
        # KEPT_VS_STRIPPED (None is a legitimate "could not establish" only
        # on the kept side).
        spec = {
            (HF_STOP_STRING, "S"): _outcome(fired=True, step=2, ids=(10,)),
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(fired=True, step=2, text=None),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(fired=True, step=2),
            (SGLANG_STOP, "S"): _outcome(fired=True, step=2, text="clean"),
        }
        result = _check(self, ("S",), spec, decode_map={(10,): "x"})
        self.assertIn(("KEPT_VS_STRIPPED", "S"), _div_pairs(result))

    def test_truthy_non_bool_flags_coerced(self) -> None:
        bad = SimpleNamespace(
            available=1,
            fired=1,
            detail="fired_at_step_2",
            truncated_ids=(1,),
            returned_text="t",
        )
        result, _ = _run(("S",), {(VLLM_HOLDBACK_STRIP, "S"): bad})
        cell = result["cells"][0]["semantics"][VLLM_HOLDBACK_STRIP]
        self.assertIs(cell["available"], True)
        self.assertIs(cell["fired"], True)
        self.assertEqual(cell["firing_step"], 2)

    def test_unicode_stop_strings(self) -> None:
        spec = {
            (VLLM_HOLDBACK_STRIP, "停止"): _outcome(fired=True, step=2, text="a停止b"),
            (VLLM_HOLDBACK_STRIP, "fin🔚"): _outcome(fired=True, step=2, text="clean"),
        }
        result = _check(self, ("停止", "fin🔚"), spec)
        pairs = _div_pairs(result)
        self.assertIn(("HARNESS_NORMALIZATION_RELIED", "停止"), pairs)
        self.assertNotIn(("HARNESS_NORMALIZATION_RELIED", "fin🔚"), pairs)

    def test_empty_id_tuples(self) -> None:
        result = _check(self, ("S",), {}, prompt_ids=(), output_ids=())
        self.assertEqual(result["prompt_tokens"], 0)
        self.assertEqual(result["output_tokens"], 0)
        self.assertEqual(result["verdict"], "agree")

    def test_cap_boundary_id_lengths_accepted(self) -> None:
        # Exactly at the caps: valid (stubs keep it fast).
        result, _ = _run(
            ("S",), {}, prompt_ids=(0,) * MAX_PROMPT_IDS, output_ids=(0,) * MAX_OUTPUT_IDS
        )
        self.assertEqual(result["prompt_tokens"], MAX_PROMPT_IDS)
        self.assertEqual(result["output_tokens"], MAX_OUTPUT_IDS)

    def test_stop_dedup_runs_once(self) -> None:
        result = _check(self, ["A", "B", "A"], {})
        self.assertEqual(result["stop_texts"], ["A", "B"])
        self.assertEqual(len(result["cells"]), 2)

    def test_list_stop_texts_accepted(self) -> None:
        result = _check(self, ["S"], {})
        self.assertEqual(result["stop_texts"], ["S"])

    def test_hf_returned_text_preferred_over_decode(self) -> None:
        # When the HF outcome exposes returned_text directly, a raising
        # decode must not matter.
        spec = {
            (HF_STOP_STRING, "S"): _outcome(fired=True, step=2, text="kept!", ids=(10,)),
            (VLLM_HOLDBACK_STRIP, "S"): _outcome(fired=True, step=2, text="kept?"),
            (VLLM_HOLDBACK_KEEP, "S"): _outcome(fired=True, step=2),
            (SGLANG_STOP, "S"): _outcome(fired=True, step=2, text="clean"),
        }
        result = _check(self, ("S",), spec, decode_raises=True)
        self.assertIn(("KEPT_VS_STRIPPED", "S"), _div_pairs(result))


# ---------------------------------------------------------------------------
# divergence_section_data: aggregation, truncation, degradation.
# ---------------------------------------------------------------------------


def _run_section(
    observations: list[Any] | tuple[Any, ...],
    stops: tuple[str, ...] | list[str],
    spec: dict[tuple[str, str], Any],
    *,
    eos_spec: dict[tuple[str, str], Any] | None = None,
    ungated: dict[str, Any] | None = None,
    min_tokens: int = 0,
    eos_id: int | None = None,
    decode_map: dict[tuple[int, ...], str] | None = None,
) -> Any:
    backend = _FakeBackend(decode_map)
    with contextlib.ExitStack() as stack:
        for patcher in _stub_patches(dict(spec), dict(ungated or {}), dict(eos_spec or {})):
            stack.enter_context(patcher)
        return divergence_section_data(
            observations=observations,
            stop_texts=stops,
            eos_id=eos_id,
            min_tokens=min_tokens,
            backend=backend,
        )


def _kept_diverge_spec(stop: str = "S") -> dict[tuple[str, str], _SynthOutcome]:
    return {
        (HF_STOP_STRING, stop): _outcome(fired=True, step=2, ids=(10,)),
        (VLLM_HOLDBACK_STRIP, stop): _outcome(fired=True, step=2, text="stripped"),
        (VLLM_HOLDBACK_KEEP, stop): _outcome(fired=True, step=2, text="kept!"),
        (SGLANG_STOP, stop): _outcome(fired=True, step=2, text="clean"),
    }


class SectionDataTests(unittest.TestCase):
    def test_aggregation_matrix_and_severities(self) -> None:
        observations = [
            {"fixture": "a", "input_ids": [1], "output_ids": [2]},
            {"fixture": "b", "input_ids": [1], "output_ids": [2]},
        ]
        section = _run_section(
            observations, ("S",), _kept_diverge_spec(), decode_map={(10,): "kept!"}
        )
        assert section is not None
        self.assertEqual(section["verdict"], "diverge")
        self.assertEqual(
            section["pairs_verdict"],
            "HF vs vLLM vs SGLang stop semantics: 0/2 agree; 2 divergences",
        )
        self.assertEqual(len(section["per_fixture"]), 2)
        self.assertEqual(section["per_fixture"][0]["fixture"], "a")
        self.assertEqual(section["per_fixture"][1]["fixture"], "b")
        for entry in section["per_fixture"]:
            self.assertEqual(entry["verdict"], "diverge")
            self.assertEqual(len(entry["divergences"]), 1)
            finding = entry["divergences"][0]
            self.assertEqual(finding["class"], "KEPT_VS_STRIPPED")
            self.assertEqual(finding["severity"], "high")
            self.assertEqual(finding["stop_text"], "S")
        # Matrix counts per finding, not per fixture entry.
        self.assertEqual(
            section["matrix"],
            {"KEPT_VS_STRIPPED": {"hf_stop_string": 2, "vllm_holdback_strip": 2}},
        )
        self.assertEqual(section["severities"], {"KEPT_VS_STRIPPED": "high"})
        self.assertEqual(section["observations_total"], 2)
        self.assertEqual(section["observations_processed"], 2)
        self.assertFalse(section["observations_truncated"])

    def test_pairs_verdict_singular(self) -> None:
        observations = [{"input_ids": [1], "output_ids": [2]}]
        section = _run_section(
            observations, ("S",), _kept_diverge_spec(), decode_map={(10,): "kept!"}
        )
        assert section is not None
        self.assertIn("1 divergence", section["pairs_verdict"])
        self.assertNotIn("1 divergences", section["pairs_verdict"])

    def test_all_agree_verdict(self) -> None:
        observations = [{"input_ids": [1], "output_ids": [2]}]
        section = _run_section(observations, ("S",), {})
        assert section is not None
        self.assertEqual(section["verdict"], "agree")
        self.assertEqual(
            section["pairs_verdict"],
            "HF vs vLLM vs SGLang stop semantics: 1/1 agree; 0 divergences",
        )
        self.assertEqual(section["matrix"], {})
        self.assertEqual(section["severities"], {})

    def test_skips_and_fixture_naming(self) -> None:
        observations: list[Any] = [
            "not-a-dict",
            {"input_ids": [1.5], "output_ids": [2]},  # bad ids
            {"input_ids": [1]},  # missing output_ids
            {"output_ids": [2]},  # missing input_ids
            {"input_ids": [1], "output_ids": [2]},  # valid, unnamed
            {"fixture": "", "input_ids": [1], "output_ids": [2]},  # empty name
            {"fixture": "named", "input_ids": [1], "output_ids": [2]},
        ]
        section = _run_section(observations, ("S",), {})
        assert section is not None
        self.assertEqual(
            [entry["fixture"] for entry in section["per_fixture"]],
            ["fixture_4", "fixture_5", "named"],
        )
        self.assertEqual(section["observations_total"], 7)
        self.assertEqual(section["observations_processed"], 3)

    def test_empty_and_missing_observations(self) -> None:
        # Missing keys -> skipped.
        self.assertIsNone(_run_section([{}], ("S",), {}))
        self.assertIsNone(_run_section([], ("S",), {}))
        self.assertIsNone(_run_section(["x", None, 42], ("S",), {}))
        # Empty-but-valid id lists -> processed.
        section = _run_section([{"input_ids": [], "output_ids": []}], ("S",), {})
        assert section is not None
        self.assertEqual(section["observations_processed"], 1)
        self.assertEqual(section["per_fixture"][0]["fixture"], "fixture_0")

    def test_tuple_observations_accepted(self) -> None:
        section = _run_section(({"input_ids": [1], "output_ids": [2]},), ("S",), {})
        assert section is not None
        self.assertEqual(section["observations_processed"], 1)

    def test_truncation_flags(self) -> None:
        observations = [
            {"input_ids": [1], "output_ids": [2]} for _ in range(MAX_DIVERGENCE_OBSERVATIONS + 1)
        ]
        section = _run_section(observations, ("S",), {})
        assert section is not None
        self.assertTrue(section["observations_truncated"])
        self.assertEqual(section["observations_total"], MAX_DIVERGENCE_OBSERVATIONS + 1)
        self.assertEqual(section["observations_processed"], MAX_DIVERGENCE_OBSERVATIONS)
        self.assertEqual(len(section["per_fixture"]), MAX_DIVERGENCE_OBSERVATIONS)

    def test_no_truncation_at_boundary(self) -> None:
        observations = [
            {"input_ids": [1], "output_ids": [2]} for _ in range(MAX_DIVERGENCE_OBSERVATIONS)
        ]
        section = _run_section(observations, ("S",), {})
        assert section is not None
        self.assertFalse(section["observations_truncated"])
        self.assertEqual(section["observations_processed"], MAX_DIVERGENCE_OBSERVATIONS)

    def test_over_cap_ids_skipped(self) -> None:
        observations = [
            {"input_ids": [1], "output_ids": [0] * (MAX_OUTPUT_IDS + 1)},
            {"input_ids": [0] * (MAX_PROMPT_IDS + 1), "output_ids": [2]},
            {"input_ids": [1], "output_ids": [2]},
        ]
        section = _run_section(observations, ("S",), {})
        assert section is not None
        self.assertEqual(section["observations_total"], 3)
        self.assertEqual(section["observations_processed"], 1)
        self.assertEqual(section["per_fixture"][0]["fixture"], "fixture_2")

    def test_section_validation_vocabulary(self) -> None:
        good = [{"input_ids": [1], "output_ids": [2]}]
        with self.assertRaises(ValueError) as ctx:
            _run_section({"a": 1}, ("S",), {})  # type: ignore[arg-type]
        self.assertEqual(str(ctx.exception), "invalid_observations")
        with self.assertRaises(ValueError) as ctx:
            _run_section("nope", ("S",), {})  # type: ignore[arg-type]
        self.assertEqual(str(ctx.exception), "invalid_observations")
        with self.assertRaises(ValueError) as ctx:
            _run_section(good, ("",), {})
        self.assertEqual(str(ctx.exception), "invalid_stop_texts")
        with self.assertRaises(ValueError) as ctx:
            _run_section(good, ("S",), {}, eos_id=-1)
        self.assertEqual(str(ctx.exception), "invalid_eos_id")
        with self.assertRaises(ValueError) as ctx:
            _run_section(good, ("S",), {}, min_tokens=-1)
        self.assertEqual(str(ctx.exception), "invalid_min_tokens")
        with self.assertRaises(ValueError) as ctx:
            divergence_section_data(
                observations=good,
                stop_texts=("S",),
                eos_id=None,
                min_tokens=0,
                backend=object(),
            )
        self.assertEqual(str(ctx.exception), "invalid_backend")

    def test_run_parameters_echoed(self) -> None:
        section = _run_section(
            [{"input_ids": [1], "output_ids": [2]}],
            ["S", "T"],
            {},
            eos_id=5,
            min_tokens=3,
        )
        assert section is not None
        self.assertEqual(section["stop_texts"], ["S", "T"])
        self.assertEqual(section["eos_id"], 5)
        self.assertEqual(section["min_tokens"], 3)


# ---------------------------------------------------------------------------
# Differential fuzz: implementation vs independent oracle, 300 random pairs.
# ---------------------------------------------------------------------------

_FUZZ_STOP_POOL = ["STOP", "END", "。", "fin🔚", "x", "STOP "]
_FUZZ_TEXT_POOL = ["plain output", "has STOP inside", "END", "。", "", "out", "outSTOP", "clean"]


def _fuzz_case(rng: random.Random) -> dict[str, Any]:
    n_stops = rng.randint(1, 4)
    stops = rng.sample(_FUZZ_STOP_POOL, n_stops)
    spec: dict[tuple[str, str], _SynthOutcome] = {}
    ungated: dict[str, _SynthOutcome] = {}
    decode_map: dict[tuple[int, ...], str] = {}
    counter = [1000]

    def fresh_ids(text: str) -> tuple[int, ...]:
        counter[0] += 1
        ids = (counter[0],)
        decode_map[ids] = text
        return ids

    for stop in stops:
        for semantic in _STOP4:
            available = rng.random() < 0.9
            fired = available and rng.random() < 0.55
            step = rng.randint(1, 5) if fired else None
            ids = fresh_ids(rng.choice(_FUZZ_TEXT_POOL))
            returned_text: str | None = None
            if semantic in (VLLM_HOLDBACK_STRIP, VLLM_HOLDBACK_KEEP):
                roll = rng.random()
                if roll < 0.7:
                    returned_text = rng.choice(
                        [f"pre {stop} post", "clean text", decode_map[ids], ""]
                    )
                # else None: exercises the decode path for vLLM cells too
            elif semantic == SGLANG_STOP:
                returned_text = rng.choice([f"trimmed {stop}", "clean", None])
            elif rng.random() < 0.2:
                returned_text = rng.choice(_FUZZ_TEXT_POOL)
            detail = (
                f"fired_at_step_{step}"
                if (fired and step is not None)
                else rng.choice(["no_stop_within_cap", "gated_by_min_tokens"])
            )
            if fired and rng.random() < 0.08:
                # Adversarial: fired but the detail carries no step.
                detail = "no_stop_within_cap"
                step = None
            spec[(semantic, stop)] = _SynthOutcome(
                available=available,
                fired=fired,
                detail=detail,
                truncated_ids=ids,
                returned_text=returned_text,
            )
        # The ungated (min_tokens=0) emulation of the strip cell.
        u_available = rng.random() < 0.9
        u_fired = u_available and rng.random() < 0.5
        u_step = rng.randint(1, 5) if u_fired else None
        u_ids = fresh_ids(rng.choice(_FUZZ_TEXT_POOL))
        ungated[stop] = _SynthOutcome(
            available=u_available,
            fired=u_fired,
            detail=f"fired_at_step_{u_step}" if u_fired else "no_stop_within_cap",
            truncated_ids=u_ids,
            returned_text=rng.choice([f"u {stop}", "clean", None]),
        )
    e_available = rng.random() < 0.9
    e_fired = e_available and rng.random() < 0.5
    e_step = rng.randint(1, 5) if e_fired else None
    eos_spec = {
        (EOS_APPENDED, "*"): _SynthOutcome(
            available=e_available,
            fired=e_fired,
            detail=f"fired_at_step_{e_step}" if e_fired else "no_stop_within_cap",
        )
    }
    return {
        "stops": stops,
        "spec": spec,
        "ungated": ungated,
        "eos_spec": eos_spec,
        "decode_map": decode_map,
        "min_tokens": rng.choice([0, 0, 1, 2, 5]),
        "eos_id": rng.choice([None, 7, 99]),
        "as_list": rng.random() < 0.3,
        "decode_raises": rng.random() < 0.05,
        "decode_non_str": rng.random() < 0.05,
        "decode_default": rng.choice(_FUZZ_TEXT_POOL),
    }


class FuzzTests(unittest.TestCase):
    FUZZ_CASES = 300

    def test_implementation_matches_oracle(self) -> None:
        for index in range(self.FUZZ_CASES):
            rng = random.Random(f"c73-fuzz-{index}")  # noqa: S311 -- deterministic test seed
            case = _fuzz_case(rng)
            stops = case["stops"]
            if case["as_list"]:
                stops = list(stops)
            with self.subTest(case=index):
                result, decode = _run(
                    stops,
                    case["spec"],
                    eos_spec=case["eos_spec"],
                    ungated=case["ungated"],
                    min_tokens=case["min_tokens"],
                    eos_id=case["eos_id"],
                    decode_map=case["decode_map"],
                    decode_default=case["decode_default"],
                    decode_raises=case["decode_raises"],
                    decode_non_str=case["decode_non_str"],
                )
                expected = _oracle_divergences(
                    list(case["stops"]),
                    dict(case["spec"]),
                    dict(case["ungated"]),
                    dict(case["eos_spec"]),
                    case["min_tokens"],
                    decode,
                )
                self.assertEqual(_div_pairs(result), expected)
                # Determinism spot-check: identical inputs, identical output.
                if index % 50 == 0:
                    again, _ = _run(
                        stops,
                        case["spec"],
                        eos_spec=case["eos_spec"],
                        ungated=case["ungated"],
                        min_tokens=case["min_tokens"],
                        eos_id=case["eos_id"],
                        decode_map=case["decode_map"],
                        decode_default=case["decode_default"],
                        decode_raises=case["decode_raises"],
                        decode_non_str=case["decode_non_str"],
                    )
                    self.assertEqual(
                        json.dumps(result, sort_keys=True),
                        json.dumps(again, sort_keys=True),
                    )


# ---------------------------------------------------------------------------
# CLI wiring: run_diff_backends validation and failure paths.
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Tokenizer double: no weights, no network, fixed encode/decode maps."""

    def __init__(
        self,
        encodings: dict[str, tuple[int, ...]],
        decodings: dict[int, str],
        encode_raises: bool = False,
    ) -> None:
        self._encodings = dict(encodings)
        self._decodings = dict(decodings)
        self._encode_raises = encode_raises

    def encode(self, text: str, **kwargs: Any) -> tuple[int, ...]:
        if self._encode_raises:
            raise RuntimeError("c73 fake encode failure")
        return self._encodings.get(text, (0,))

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        return "".join(self._decodings[i] for i in token_ids)

    def get_vocab(self) -> dict[str, int]:
        return {f"tok{i}": i for i in self._decodings}


class RunDiffBackendsTests(unittest.TestCase):
    @staticmethod
    def _args(**overrides: Any) -> argparse.Namespace:
        base: dict[str, Any] = {
            "min_tokens": 0,
            "eos_id": None,
            "stops": "STOP",
            "prompt_text": "ab",
            "output_text": "abSTOPab",
            "model": "fake-model",
        }
        base.update(overrides)
        return argparse.Namespace(**base)

    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        return argparse.ArgumentParser(prog="mpl diff-backends")

    def _run_cli(self, args: argparse.Namespace) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = run_diff_backends(args, self._parser())
            except SystemExit as exc:
                return int(exc.code or 0), out.getvalue(), err.getvalue()
        return code, out.getvalue(), err.getvalue()

    def test_negative_min_tokens_exits_2(self) -> None:
        code, _, _ = self._run_cli(self._args(min_tokens=-1))
        self.assertEqual(code, 2)

    def test_negative_eos_id_exits_2(self) -> None:
        code, _, _ = self._run_cli(self._args(eos_id=-1))
        self.assertEqual(code, 2)

    def test_empty_stops_exit_2(self) -> None:
        for stops in ("", "a,,b", ",a"):
            with self.subTest(stops=stops):
                code, _, _ = self._run_cli(self._args(stops=stops))
                self.assertEqual(code, 2)

    def test_oversize_texts_exit_2(self) -> None:
        code, _, _ = self._run_cli(self._args(prompt_text="x" * 1_000_001))
        self.assertEqual(code, 2)
        code, _, _ = self._run_cli(self._args(output_text="x" * 1_000_001))
        self.assertEqual(code, 2)

    @staticmethod
    def _patch_tokenizer(tokenizer: _FakeTokenizer) -> Any:
        import model_protocol_lab.doctor as doctor

        return mock.patch.object(doctor, "_build_primary_tokenizer", return_value=tokenizer)

    def test_end_to_end_with_fake_tokenizer(self) -> None:
        tokenizer = _FakeTokenizer(
            {"ab": (0, 1), "abSTOPab": (0, 1, 2, 1, 0)},
            {0: "a", 1: "b", 2: "STOP"},
        )
        with self._patch_tokenizer(tokenizer):
            code, out, _ = self._run_cli(self._args())
        self.assertEqual(code, 0)
        result = json.loads(out)
        self.assertEqual(result["prompt_tokens"], 2)
        self.assertEqual(result["output_tokens"], 5)
        self.assertEqual(result["stop_texts"], ["STOP"])
        self.assertIn(result["verdict"], ("agree", "diverge"))
        self.assertEqual(len(result["cells"]), 1)

    def test_encode_failure_is_fixed_code(self) -> None:
        tokenizer = _FakeTokenizer({}, {}, encode_raises=True)
        with self._patch_tokenizer(tokenizer):
            code, _, err = self._run_cli(self._args())
        self.assertEqual(code, 1)
        self.assertIn("diff-backends: encode_failed", err)

    def test_loader_failure_sanitized(self) -> None:
        import model_protocol_lab.doctor as doctor

        with mock.patch.object(
            doctor, "_build_primary_tokenizer", side_effect=ValueError("raw \x00 bytes")
        ):
            code, _, err = self._run_cli(self._args())
        self.assertEqual(code, 1)
        self.assertIn("diff-backends: tokenizer_load_failed", err)
        self.assertNotIn("raw", err)

    def test_loader_fixed_code_passes_through(self) -> None:
        import model_protocol_lab.doctor as doctor

        with mock.patch.object(
            doctor,
            "_build_primary_tokenizer",
            side_effect=ValueError("oversized_tokenizer_file"),
        ):
            code, _, err = self._run_cli(self._args())
        self.assertEqual(code, 1)
        self.assertIn("diff-backends: oversized_tokenizer_file", err)

    def test_diff_backends_value_error_is_fixed_code(self) -> None:
        # A 2000-char stop passes CLI splitting but fails diff_backends
        # validation -> exit 1 with the fixed vocabulary, no traceback.
        tokenizer = _FakeTokenizer({"ab": (1,)}, {1: "a"})
        with self._patch_tokenizer(tokenizer):
            code, _, err = self._run_cli(self._args(stops="x" * 2000))
        self.assertEqual(code, 1)
        self.assertIn("diff-backends: invalid input: stop_text_too_long", err)


# ---------------------------------------------------------------------------
# Mutation gate: 6 behavior mutants in PID-scoped /tmp copies must all die.
# ---------------------------------------------------------------------------

_MUTANTS = [
    # FIRING_STEP_DIFFERS: two distinct steps no longer enough.
    ("firing-step-gt1", "if len(steps) > 1:", "if len(steps) > 2:"),
    # EOS_PRECEDENCE: at/before weakened to strictly before.
    ("eos-lt", "eos.firing_step <= min(stop_steps)", "eos.firing_step < min(stop_steps)"),
    # KEPT_VS_STRIPPED: text inequality inverted.
    ("kept-eq", "kept_text != strip.returned_text", "kept_text == strip.returned_text"),
    # MULTI_STOP_SELECTION: sglang tie-break prefers the LAST-listed stop.
    (
        "sglang-tiebreak",
        "(sglang_cell.firing_step, list_index)",
        "(sglang_cell.firing_step, -list_index)",
    ),
    # HARNESS_NORMALIZATION_RELIED: kept-disposition cells also flagged.
    (
        "harness-kept",
        'if not cell.fired or cell.disposition != "stripped":',
        "if not cell.fired:",
    ),
    # MIN_TOKENS_GATED: gate condition inverted (fires when ungated silent).
    ("ungated-invert", "if ungated_step is not None:", "if ungated_step is None:"),
]


class SensitivityTests(unittest.TestCase):
    @contextlib.contextmanager
    def _scratch(self) -> Any:
        path = Path(tempfile.mkdtemp(prefix=f"mpl_c73_{os.getpid()}_"))
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _copy_package(dest: Path) -> None:
        shutil.copytree(
            REPO_ROOT / "model_protocol_lab",
            dest / "model_protocol_lab",
            ignore=shutil.ignore_patterns("__pycache__"),
        )

    @staticmethod
    def _run_suite(pkg_parent: Path) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(pkg_parent) + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--import-mode=importlib",
            "-p",
            "no:cacheprovider",
            "-q",
            "-k",
            "not SensitivityTests",
            os.path.abspath(__file__),
        ]
        return subprocess.run(
            cmd,
            cwd=str(pkg_parent),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )

    def test_pristine_copy_passes(self) -> None:
        with self._scratch() as parent:
            self._copy_package(parent)
            proc = self._run_suite(parent)
        tail = (proc.stdout + proc.stderr)[-3000:]
        self.assertEqual(proc.returncode, 0, msg=f"pristine copy failed:\n{tail}")

    def test_mutants_killed(self) -> None:
        for name, old, new in _MUTANTS:
            with self.subTest(mutant=name):
                with self._scratch() as parent:
                    self._copy_package(parent)
                    target = parent / "model_protocol_lab" / "diffbackends.py"
                    src = target.read_text()
                    self.assertEqual(src.count(old), 1, f"mutant anchor not unique: {name}")
                    target.write_text(src.replace(old, new))
                    proc = self._run_suite(parent)
                tail = (proc.stdout + proc.stderr)[-2000:]
                self.assertNotEqual(proc.returncode, 0, msg=f"mutant survived: {name}\n{tail}")


if __name__ == "__main__":
    unittest.main()
