"""Cycle 37: differential verification of ``diffbackends.py`` classification internals.

Cycle 15 verified the top-level divergence classification of ``diff_backends``
against an independent oracle. This cycle verifies the *remaining internals*,
which cycle 15 touched only lightly or not at all:

- the ``DIVERGENCE_SEVERITY`` mapping: its docstring-derived values, and its
  *usage* -- per-finding ``severity`` in ``divergence_section_data`` and the
  ``severities`` map (the report's severity badges consume exactly these);
- per-class predicate details: the exact firing conditions of
  ``KEPT_VS_STRIPPED`` (both cells fired, text comparison, degrade when the
  kept text cannot be established), ``EOS_PRECEDENCE`` (at/before, min over
  the stop semantics), ``STR_OVER_EOS_PRECEDENCE`` (same-step tie between
  the *sglang* cell and EOS), ``MULTI_STOP_SELECTION`` (winner selection,
  tie-breaks, winner-preference for the carried ``stop_text``),
  ``FIRING_STEP_DIFFERS`` (unavailable cells excluded, ``None`` = did not
  fire), ``MIN_TOKENS_GATED`` / ``NO_MIN_TOKENS_GATE`` (the shared ungated
  diagnostic re-run; the SGLang column is nested inside the gate firing),
  ``HARNESS_NORMALIZATION_RELIED`` (``"kept"``-disposition cells never
  flagged; vacuous for the sglang cell by construction);
- the class x semantic attribution (``_CLASS_SEMANTICS``) and how the count
  matrix consumes it (per *finding*, not per fixture);
- finding construction: dict key sets, ``stop_text`` carriage per class,
  detail anchors, divergence ordering;
- edge cases: unavailable cells, empty outputs, non-str decodes,
  stateful/adversarial backends, the byte-space HF path, stop-text dedup.

The oracle (``_DocstringOracle``) is written from the module and class
*docstrings*, not from the implementation's comparison code, and is
structurally different from both the implementation and cycle 15's oracle:
each divergence class is a declarative rule predicate over plain-dict
evidence snapshots (one ``_Snap`` per semantic per stop), with separate
severity/attribution tables derived from the docstring sentences. The cell
layer -- the ``emulate_*`` functions in ``backends.py`` -- is reused as
ground truth, exactly as in cycle 15: what is under test here is the
classification built on top of it.

Doc discrepancies found (code behavior pinned by tests; source NOT changed):

1. (CYCLE 37 NOTE, FIXED IN CYCLE 37.) The module docstring used to say
   ``STR_OVER_EOS_PRECEDENCE`` "Co-fires with ``EOS_PRECEDENCE`` on the
   tie". The code fires ``EOS_PRECEDENCE`` only when the EOS step is
   at/before the *minimum* stop-semantic firing step, while
   ``STR_OVER_EOS_PRECEDENCE`` only compares the sglang cell against EOS.
   When HF fires strictly earlier via prompt overhang (prompt ``"ST"`` +
   output ``"OP"`` completing ``"STOP"`` at step 1) while sglang and EOS
   tie at step 2, ``STR_OVER_EOS_PRECEDENCE`` fires *without*
   ``EOS_PRECEDENCE``. The docstring (and the matching code comment) were
   corrected in cycle 37: co-firing holds unless another stop cell fired
   strictly earlier. Pinned by
   ``ClassificationDetailTests.test_str_over_eos_without_eos_precedence``.
2. (CYCLE 37 NOTE, FIXED IN CYCLE 37.) The ``diff_backends`` docstring
   used to describe the return dict as carrying "stop_texts, eos_id,
   min_tokens, per-stop cells, divergences and verdict" but the returned
   dict also carries ``prompt_tokens`` and ``output_tokens``. The
   docstring was corrected in cycle 37. Pinned by ``test_result_key_sets``.

Sensitivity: ``SensitivityTests`` copies the package to ``/tmp`` (never
mutating the repo), applies behavior mutations to the copy, and shows this
file's tests fail on each mutant while passing on the pristine copy.

Deterministic: all fuzzing uses string seeds with ``random.Random``. No
network, no model weights, no torch.
"""

from __future__ import annotations

import os
import random
import re
import shutil
import subprocess
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import model_protocol_lab
from model_protocol_lab.backends import (
    emulate_eos_stop,
    emulate_hf_stop_string,
    emulate_sglang_stop,
    emulate_vllm_holdback,
)
from model_protocol_lab.diffbackends import (
    _CLASS_SEMANTICS,
    DIVERGENCE_SEVERITY,
    EOS_APPENDED,
    HF_STOP_STRING,
    MAX_DIVERGENCE_OBSERVATIONS,
    MAX_OUTPUT_IDS,
    MAX_PROMPT_IDS,
    MAX_STOP_TEXT_LEN,
    MAX_STOP_TEXTS,
    SGLANG_STOP,
    VLLM_HOLDBACK_KEEP,
    VLLM_HOLDBACK_STRIP,
    _effective_text,
    _firing_step,
    diff_backends,
    divergence_section_data,
)
from model_protocol_lab.diffbackends import (
    CellResult as _ImplCellResult,
)

_STOP4 = (HF_STOP_STRING, VLLM_HOLDBACK_STRIP, VLLM_HOLDBACK_KEEP, SGLANG_STOP)


# ---------------------------------------------------------------------------
# Fake backends.
# ---------------------------------------------------------------------------


class _ConcatBackend:
    """Decode is exactly incremental concatenation of per-token texts.

    Unknown ids raise ``KeyError`` (exercises the unavailable-cell path);
    ids map 1..n for output tokens, 1001.. for prompt tokens.
    """

    fingerprint = "cycle37-concat-backend-not-a-model"

    def __init__(self, texts: dict[int, str]) -> None:
        self._texts = dict(texts)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(self._texts[token] for token in token_ids)


def _make_backend(
    output_texts: tuple[str, ...], prompt_texts: tuple[str, ...] = ()
) -> tuple[_ConcatBackend, tuple[int, ...], tuple[int, ...]]:
    mapping = {index + 1: text for index, text in enumerate(output_texts)}
    prompt_ids = tuple(1001 + index for index in range(len(prompt_texts)))
    for pid, text in zip(prompt_ids, prompt_texts, strict=True):
        mapping[pid] = text
    return _ConcatBackend(mapping), prompt_ids, tuple(range(1, len(output_texts) + 1))


class _BytePathBackend(_ConcatBackend):
    """Concat backend exposing ``hf_stop_token_bytes`` so the HF cell runs
    the byte-space path (no per-token decode during the HF emulation).
    """

    def hf_stop_token_bytes(self, token_id: int) -> bytes | None:
        return self._texts[token_id].encode("utf-8")


class _CountingBackend(_ConcatBackend):
    """Concat backend with a decode-call budget; calls past the budget raise.

    Red-team tool: lets the test arm a decode failure *after* the cell
    emulations have run, probing the classification's degrade paths.
    """

    def __init__(self, texts: dict[int, str], budget: int) -> None:
        super().__init__(texts)
        self.budget = budget
        self.calls = 0

    def decode(self, token_ids: tuple[int, ...]) -> str:
        self.calls += 1
        if self.calls > self.budget:
            raise RuntimeError("cycle37-armed-decode-failure")
        return super().decode(token_ids)


# ---------------------------------------------------------------------------
# Independent oracle, derived from the docstrings.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Snap:
    """The oracle's evidence snapshot of one emulated semantic for one stop."""

    semantic: str
    available: bool
    fired: bool
    step: int | None
    ids: tuple[int, ...]
    text: str | None  # returned_text from the emulation, if any
    disposition: str | None


_STEP_RE = re.compile(r"fired_at_step_(\d+)")


def _step_from_detail(detail: str) -> int | None:
    """1-based firing step iff the detail is exactly ``fired_at_step_N``."""
    match = _STEP_RE.fullmatch(detail)
    return int(match.group(1)) if match is not None else None


def _readable_text(backend: Any, snap: _Snap) -> str | None:
    """The text the semantic would return: the exposed returned text, else
    the decode of the truncated ids -- None when it cannot be established.
    """
    if snap.text is not None:
        return snap.text
    decode = getattr(backend, "decode", None)
    if not callable(decode):
        return None
    try:
        out = decode(snap.ids)
    except Exception:  # noqa: BLE001 -- text checks degrade, never fail the oracle
        return None
    return out if type(out) is str else None


def _valid_id_seq(value: object, name: str, cap: int) -> tuple[int, ...]:
    if type(value) is not tuple or any(
        type(token) is not int or not 0 <= token < 2**31 for token in value
    ):
        raise ValueError(f"invalid_{name}")
    if len(value) > cap:
        raise ValueError(f"{name}_too_long")
    return value


def _valid_stop_list(value: object) -> tuple[str, ...]:
    if type(value) is tuple:
        items: tuple[object, ...] = value
    elif type(value) is list:
        items = tuple(value)
    else:
        raise ValueError("invalid_stop_texts")
    deduped: list[str] = []
    seen: set[str] = set()
    for stop in items:
        if type(stop) is not str or not stop:
            raise ValueError("invalid_stop_texts")
        if len(stop) > MAX_STOP_TEXT_LEN:
            raise ValueError("stop_text_too_long")
        if stop not in seen:
            seen.add(stop)
            deduped.append(stop)
    if not deduped:
        raise ValueError("empty_stop_texts")
    if len(deduped) > MAX_STOP_TEXTS:
        raise ValueError("too_many_stop_texts")
    return tuple(deduped)


def _valid_obs_ids(observation: dict[str, Any], key: str, cap: int) -> tuple[int, ...] | None:
    raw: object = observation.get(key)
    if type(raw) is list:
        raw = tuple(raw)
    try:
        return _valid_id_seq(raw, key, cap)
    except ValueError:
        return None


# Severity, derived from the DIVERGENCE_SEVERITY pin comment: "high" classes
# silently change the returned text or the stopping precedence (score
# corruption); "medium" classes change *when* stopping happens; "low" flags
# cases where correct scoring would depend on harness-side normalization.
_ORACLE_SEVERITY: dict[str, str] = {
    "KEPT_VS_STRIPPED": "high",  # returned text differs: kept vs stripped
    "EOS_PRECEDENCE": "high",  # stopping precedence: EOS vs stop string
    "STR_OVER_EOS_PRECEDENCE": "high",  # precedence disagreement SGLang-vs-TGI
    "MULTI_STOP_SELECTION": "high",  # which stop wins changes the outcome
    "FIRING_STEP_DIFFERS": "medium",  # when stopping happens
    "MIN_TOKENS_GATED": "medium",  # when stopping happens (min_tokens gate)
    "NO_MIN_TOKENS_GATE": "medium",  # the ungated SGLang column of the gate
    "HARNESS_NORMALIZATION_RELIED": "low",  # needs harness-side re-stripping
}

# Attribution, derived from each class's docstring paragraph: the emulated
# semantics whose disagreement the class measures.
_ORACLE_ATTRIBUTION: dict[str, tuple[str, ...]] = {
    "KEPT_VS_STRIPPED": (HF_STOP_STRING, VLLM_HOLDBACK_STRIP),
    "FIRING_STEP_DIFFERS": (HF_STOP_STRING, VLLM_HOLDBACK_STRIP, VLLM_HOLDBACK_KEEP, SGLANG_STOP),
    "MIN_TOKENS_GATED": (VLLM_HOLDBACK_STRIP,),
    "NO_MIN_TOKENS_GATE": (SGLANG_STOP, VLLM_HOLDBACK_STRIP),
    "EOS_PRECEDENCE": (
        EOS_APPENDED,
        HF_STOP_STRING,
        VLLM_HOLDBACK_STRIP,
        VLLM_HOLDBACK_KEEP,
        SGLANG_STOP,
    ),
    "STR_OVER_EOS_PRECEDENCE": (SGLANG_STOP, EOS_APPENDED),
    "MULTI_STOP_SELECTION": (SGLANG_STOP, VLLM_HOLDBACK_STRIP),
    "HARNESS_NORMALIZATION_RELIED": (VLLM_HOLDBACK_STRIP, SGLANG_STOP),
}


def _rule_firing_steps_differ(
    stop: str, snaps: dict[str, _Snap], ctx: SimpleNamespace
) -> list[str]:
    """Docstring: "the stop-string semantics disagree on the completing output
    step (None = did not fire)". Unavailable cells are excluded.
    """
    live = [snap for key, snap in snaps.items() if key in _STOP4 and snap.available]
    return [stop] if len({snap.step for snap in live}) > 1 else []


def _rule_kept_vs_stripped(stop: str, snaps: dict[str, _Snap], ctx: SimpleNamespace) -> list[str]:
    """Docstring: "HF always keeps the stop string while vLLM's default
    (include_stop_str_in_output=False) strips it". Both cells must have
    fired; when the kept text cannot be established the rule degrades to
    silence rather than failing.
    """
    hf = snaps[HF_STOP_STRING]
    strip = snaps[VLLM_HOLDBACK_STRIP]
    if not (hf.available and hf.fired and strip.available and strip.fired):
        return []
    kept = _readable_text(ctx.backend, hf)
    if kept is None or kept == strip.text:
        return []
    return [stop]


def _rule_min_tokens_gated(stop: str, snaps: dict[str, _Snap], ctx: SimpleNamespace) -> list[str]:
    """Docstring: "a stop that fires with min_tokens=0 does not fire under
    the given min_tokens (vLLM PR #22014)". The shared ungated diagnostic
    re-run decides whether the gate is what silenced the strip cell.
    """
    strip = snaps[VLLM_HOLDBACK_STRIP]
    if not (ctx.min_tokens > 0 and strip.available and not strip.fired):
        return []
    rerun = emulate_vllm_holdback(ctx.backend, ctx.output_ids, stop, min_tokens=0)
    if rerun.available and rerun.fired and _step_from_detail(rerun.detail) is not None:
        return [stop]
    return []


def _rule_no_min_tokens_gate(
    stop: str, snaps: dict[str, _Snap], ctx: SimpleNamespace, gated: bool
) -> list[str]:
    """Docstring: "the SGLang column on the MIN_TOKENS_GATED class: SGLang
    has no min_tokens equivalent ... so a stop that vLLM's min_tokens gate
    blocks still fires under the SGLang emulation". Nested inside the gate
    firing by construction.
    """
    if not gated:
        return []
    sglang = snaps[SGLANG_STOP]
    if sglang.available and sglang.fired and sglang.step is not None:
        return [stop]
    return []


def _rule_eos_precedence(stop: str, snaps: dict[str, _Snap], ctx: SimpleNamespace) -> list[str]:
    """Docstring: "EOS fires at/before the stop-string firing step". The
    stop-string step is the minimum over the available, fired stop-semantic
    cells.
    """
    eos = snaps[EOS_APPENDED]
    if not (eos.available and eos.fired and eos.step is not None):
        return []
    steps = [
        snap.step
        for key, snap in snaps.items()
        if key in _STOP4 and snap.available and snap.fired and snap.step is not None
    ]
    if not steps:
        return []
    return [stop] if eos.step <= min(steps) else []


def _rule_str_over_eos(stop: str, snaps: dict[str, _Snap], ctx: SimpleNamespace) -> list[str]:
    """Docstring: "stop string and EOS fire in the same step, so precedence
    -- not firing order -- decides the outcome". The code compares the
    *sglang* cell's step against EOS; it does NOT require EOS_PRECEDENCE
    (see the module docstring discrepancy pinned by the tie tests).
    """
    sglang = snaps[SGLANG_STOP]
    eos = snaps[EOS_APPENDED]
    if (
        sglang.available
        and sglang.fired
        and sglang.step is not None
        and eos.available
        and eos.fired
        and eos.step == sglang.step
    ):
        return [stop]
    return []


def _rule_harness_relied(stop: str, snaps: dict[str, _Snap], ctx: SimpleNamespace) -> list[str]:
    """Docstring: 'a "stripped"-disposition cell's returned text still
    contains the stop string: the strip failed ... Cells whose disposition
    is "kept" keep the stop string by definition and are not flagged.'
    """
    hits: list[str] = []
    for key, snap in snaps.items():
        if key not in _STOP4 or not snap.available or not snap.fired:
            continue
        if snap.disposition != "stripped":
            continue
        text = _readable_text(ctx.backend, snap)
        if text is not None and stop in text:
            hits.append(stop)
    return hits


def _rule_multi_stop_selection(
    per_stop: list[tuple[str, dict[str, _Snap]]], ctx: SimpleNamespace
) -> list[str]:
    """Docstring scope limits: "the SGLang winner is the fired stop with the
    smallest firing step, ties broken by list order ..., while the vLLM
    winner ports PR #49391 -- smallest (firing step, stop_index +
    len(stop_str)), ties by list order, read off the strip/keep cells'
    firing steps and trimmed texts" (the keep cell's returned text is the
    firing step's decoded text cut at the match end, so its length IS
    stop_index + len(stop_str)). Fires when the (stop, step) winners differ;
    the carried stop_text prefers the sglang winner.
    """
    sglang_best: tuple[tuple[int, int], str] | None = None
    vllm_best: tuple[tuple[int, int, int], str] | None = None
    for list_index, (stop, snaps) in enumerate(per_stop):
        sglang = snaps[SGLANG_STOP]
        strip = snaps[VLLM_HOLDBACK_STRIP]
        keep = snaps[VLLM_HOLDBACK_KEEP]
        if sglang.available and sglang.fired and sglang.step is not None:
            key = (sglang.step, list_index)
            if sglang_best is None or key < sglang_best[0]:
                sglang_best = (key, stop)
        if (
            strip.available
            and strip.fired
            and strip.step is not None
            and keep.available
            and keep.fired
            and keep.text is not None
        ):
            key = (strip.step, len(keep.text), list_index)
            if vllm_best is None or key < vllm_best[0]:
                vllm_best = (key, stop)
    sglang_winner = (sglang_best[1], sglang_best[0][0]) if sglang_best is not None else None
    vllm_winner = (vllm_best[1], vllm_best[0][0]) if vllm_best is not None else None
    if sglang_winner == vllm_winner:
        return []
    if sglang_best is not None:
        return [sglang_best[1]]
    return [vllm_best[1]]  # reachable only when the sglang cell never fired


class _DocstringOracle:
    """The independent oracle: validate, snapshot the emulations, classify."""

    @staticmethod
    def _snapshot(
        backend: Any,
        prompt_ids: tuple[int, ...],
        output_ids: tuple[int, ...],
        stop: str,
        eos_id: int | None,
        min_tokens: int,
    ) -> dict[str, _Snap]:
        token_bytes = getattr(backend, "hf_stop_token_bytes", None)

        def snap(semantic: str, outcome: Any, disposition: str | None) -> _Snap:
            return _Snap(
                semantic=semantic,
                available=bool(outcome.available),
                fired=bool(outcome.fired),
                step=_step_from_detail(outcome.detail),
                ids=tuple(outcome.truncated_ids),
                text=outcome.returned_text,
                disposition=disposition,
            )

        return {
            HF_STOP_STRING: snap(
                HF_STOP_STRING,
                emulate_hf_stop_string(
                    backend, prompt_ids, output_ids, stop, token_bytes=token_bytes
                ),
                "kept",
            ),
            VLLM_HOLDBACK_STRIP: snap(
                VLLM_HOLDBACK_STRIP,
                emulate_vllm_holdback(
                    backend,
                    output_ids,
                    stop,
                    include_stop_str_in_output=False,
                    min_tokens=min_tokens,
                ),
                "stripped",
            ),
            VLLM_HOLDBACK_KEEP: snap(
                VLLM_HOLDBACK_KEEP,
                emulate_vllm_holdback(
                    backend,
                    output_ids,
                    stop,
                    include_stop_str_in_output=True,
                    min_tokens=min_tokens,
                ),
                "kept",
            ),
            SGLANG_STOP: snap(
                SGLANG_STOP, emulate_sglang_stop(backend, output_ids, stop), "stripped"
            ),
            EOS_APPENDED: snap(EOS_APPENDED, emulate_eos_stop(output_ids, eos_id), None),
        }

    def classify(
        self,
        *,
        prompt_ids: tuple[int, ...],
        output_ids: tuple[int, ...],
        stop_texts: tuple[str, ...] | list[str],
        eos_id: int | None = None,
        min_tokens: int = 0,
        backend: Any,
    ) -> dict[str, Any]:
        prompt_ids = _valid_id_seq(prompt_ids, "prompt_ids", MAX_PROMPT_IDS)
        output_ids = _valid_id_seq(output_ids, "output_ids", MAX_OUTPUT_IDS)
        stops = _valid_stop_list(stop_texts)
        if eos_id is not None and (type(eos_id) is not int or eos_id < 0):
            raise ValueError("invalid_eos_id")
        if type(min_tokens) is not int or min_tokens < 0:
            raise ValueError("invalid_min_tokens")
        if not callable(getattr(backend, "decode", None)):
            raise ValueError("invalid_backend")
        ctx = SimpleNamespace(backend=backend, output_ids=output_ids, min_tokens=min_tokens)
        findings: list[tuple[str, str]] = []
        per_stop: list[tuple[str, dict[str, _Snap]]] = []
        for stop in stops:
            snaps = self._snapshot(backend, prompt_ids, output_ids, stop, eos_id, min_tokens)
            per_stop.append((stop, snaps))
            gated = _rule_min_tokens_gated(stop, snaps, ctx)
            per_class: list[tuple[str, list[str]]] = [
                ("FIRING_STEP_DIFFERS", _rule_firing_steps_differ(stop, snaps, ctx)),
                ("KEPT_VS_STRIPPED", _rule_kept_vs_stripped(stop, snaps, ctx)),
                ("MIN_TOKENS_GATED", gated),
                ("NO_MIN_TOKENS_GATE", _rule_no_min_tokens_gate(stop, snaps, ctx, bool(gated))),
                ("EOS_PRECEDENCE", _rule_eos_precedence(stop, snaps, ctx)),
                ("STR_OVER_EOS_PRECEDENCE", _rule_str_over_eos(stop, snaps, ctx)),
                ("HARNESS_NORMALIZATION_RELIED", _rule_harness_relied(stop, snaps, ctx)),
            ]
            for kind, hits in per_class:
                findings.extend((kind, hit) for hit in hits)
        findings.extend(
            ("MULTI_STOP_SELECTION", hit) for hit in _rule_multi_stop_selection(per_stop, ctx)
        )
        return {
            "stop_texts": list(stops),
            "snaps": dict(per_stop),
            "findings": findings,
            "verdict": "diverge" if findings else "agree",
        }

    def summarize(
        self,
        *,
        observations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        stop_texts: tuple[str, ...] | list[str],
        eos_id: int | None,
        min_tokens: int,
        backend: Any,
    ) -> dict[str, Any] | None:
        if type(observations) is tuple:
            candidates: list[Any] = list(observations)
        elif type(observations) is list:
            candidates = list(observations)
        else:
            raise ValueError("invalid_observations")
        observations_total = len(candidates)
        truncated = observations_total > MAX_DIVERGENCE_OBSERVATIONS
        if truncated:
            candidates = candidates[:MAX_DIVERGENCE_OBSERVATIONS]
        stops = _valid_stop_list(stop_texts)
        if eos_id is not None and (type(eos_id) is not int or eos_id < 0):
            raise ValueError("invalid_eos_id")
        if type(min_tokens) is not int or min_tokens < 0:
            raise ValueError("invalid_min_tokens")
        if not callable(getattr(backend, "decode", None)):
            raise ValueError("invalid_backend")
        per_fixture: list[dict[str, Any]] = []
        matrix: dict[str, dict[str, int]] = {}
        for index, observation in enumerate(candidates):
            if type(observation) is not dict:
                continue
            prompt_ids = _valid_obs_ids(observation, "input_ids", MAX_PROMPT_IDS)
            output_ids = _valid_obs_ids(observation, "output_ids", MAX_OUTPUT_IDS)
            if prompt_ids is None or output_ids is None:
                continue
            name = observation.get("fixture")
            fixture = name if type(name) is str and name else f"fixture_{index}"
            outcome = self.classify(
                prompt_ids=prompt_ids,
                output_ids=output_ids,
                stop_texts=stops,
                eos_id=eos_id,
                min_tokens=min_tokens,
                backend=backend,
            )
            entries = [
                {"class": kind, "stop_text": stop, "severity": _ORACLE_SEVERITY[kind]}
                for kind, stop in outcome["findings"]
            ]
            for kind, _stop in outcome["findings"]:
                row = matrix.setdefault(kind, {})
                for semantic in _ORACLE_ATTRIBUTION[kind]:
                    row[semantic] = row.get(semantic, 0) + 1
            per_fixture.append(
                {"fixture": fixture, "verdict": outcome["verdict"], "divergences": entries}
            )
        if not per_fixture:
            return None
        total = len(per_fixture)
        agreed = sum(1 for entry in per_fixture if entry["verdict"] == "agree")
        total_findings = sum(len(entry["divergences"]) for entry in per_fixture)
        noun = "divergence" if total_findings == 1 else "divergences"
        return {
            "verdict": "agree" if agreed == total else "diverge",
            "pairs_verdict": (
                "HF vs vLLM vs SGLang stop semantics: "
                f"{agreed}/{total} agree; {total_findings} {noun}"
            ),
            "per_fixture": per_fixture,
            "matrix": matrix,
            "severities": {kind: _ORACLE_SEVERITY[kind] for kind in matrix},
            "stop_texts": list(stops),
            "eos_id": eos_id,
            "min_tokens": min_tokens,
            "observations_truncated": truncated,
            "observations_total": observations_total,
            "observations_processed": len(per_fixture),
        }


# ---------------------------------------------------------------------------
# Hand-derived cases: (name, prompt texts, output texts, stops, eos id,
# min_tokens, divergence classes expected to fire per the docstrings).
# ---------------------------------------------------------------------------
_HAND_CASES: tuple[
    tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...], int | None, int, frozenset[str]],
    ...,
] = (
    ("agree_clean", (), ("a", "b", "c"), ("zzz",), None, 0, frozenset()),
    (
        "kept_vs_stripped",
        (),
        ("a", "STOP", "b"),
        ("STOP",),
        None,
        0,
        frozenset({"KEPT_VS_STRIPPED"}),
    ),
    (
        "tie_str_eos",
        (),
        ("a", "STOP"),
        ("STOP",),
        2,
        0,
        frozenset({"STR_OVER_EOS_PRECEDENCE", "EOS_PRECEDENCE"}),
    ),
    # HF fires at step 1 via prompt overhang ("ST"+"OP"); sglang and EOS tie
    # at step 2. STR_OVER_EOS_PRECEDENCE fires WITHOUT EOS_PRECEDENCE --
    # the module docstring's "Co-fires with EOS_PRECEDENCE on the tie"
    # does not hold here; the test pins the code behavior.
    (
        "prompt_overhang_str_tie",
        ("ST",),
        ("OP", "xSTOP"),
        ("STOP",),
        2,
        0,
        frozenset({"STR_OVER_EOS_PRECEDENCE"}),
    ),
    # min_tokens=5 gates both vLLM cells; hf/sglang fire at step 1.
    (
        "gated_combo",
        (),
        ("aSTOP",),
        ("STOP",),
        None,
        5,
        frozenset(
            {
                "FIRING_STEP_DIFFERS",
                "MIN_TOKENS_GATED",
                "NO_MIN_TOKENS_GATE",
                "MULTI_STOP_SELECTION",
            }
        ),
    ),
    # The stop completes after the gate: the gate changes nothing.
    (
        "post_gate_fire",
        (),
        ("a", "STOP"),
        ("STOP",),
        None,
        1,
        frozenset({"KEPT_VS_STRIPPED"}),
    ),
    # Windowed vLLM find matches "ab" at index 3 but an earlier occurrence
    # survives in the stripped text: the strip failed (vLLM #46113 class).
    (
        "harness_anomaly",
        (),
        ("ab", "Xab"),
        ("ab",),
        None,
        1,
        frozenset({"HARNESS_NORMALIZATION_RELIED"}),
    ),
    # Both stops fire at step 1: sglang takes the first-listed ("ab"),
    # vLLM takes the earliest-completing ("xa", match ends at index 2).
    (
        "multi_tie",
        (),
        ("xab",),
        ("ab", "xa"),
        None,
        0,
        frozenset({"MULTI_STOP_SELECTION"}),
    ),
    # Full tie: both rules pick "ab" (same step, same match end, list order).
    (
        "multi_full_tie",
        (),
        ("xab",),
        ("ab", "b"),
        None,
        0,
        frozenset({"KEPT_VS_STRIPPED"}),
    ),
)

# Every divergence class, fired by construction in at least one hand case.
_CLASS_CASES = {
    "KEPT_VS_STRIPPED": "kept_vs_stripped",
    "FIRING_STEP_DIFFERS": "gated_combo",
    "MIN_TOKENS_GATED": "gated_combo",
    "NO_MIN_TOKENS_GATE": "gated_combo",
    "EOS_PRECEDENCE": "tie_str_eos",
    "STR_OVER_EOS_PRECEDENCE": "tie_str_eos",
    "HARNESS_NORMALIZATION_RELIED": "harness_anomaly",
    "MULTI_STOP_SELECTION": "multi_tie",
}


# skipcq: PTC-W0046 - shared base: subclasses carry the test methods; the base runs none by design
class _Cycle37Base(unittest.TestCase):
    def setUp(self) -> None:
        self.oracle = _DocstringOracle()

    def _run_case(self, name: str) -> tuple[dict[str, Any], dict[str, Any], frozenset[str]]:
        for case in _HAND_CASES:
            if case[0] == name:
                _, prompt_texts, output_texts, stops, eos_id, min_tokens, expected = case
                break
        else:  # pragma: no cover -- test data bug, not product code
            raise AssertionError(f"unknown hand case {name!r}")
        backend, prompt_ids, output_ids = _make_backend(output_texts, prompt_texts)
        kwargs: dict[str, Any] = {
            "prompt_ids": prompt_ids,
            "output_ids": output_ids,
            "stop_texts": stops,
            "eos_id": eos_id,
            "min_tokens": min_tokens,
            "backend": backend,
        }
        impl = diff_backends(**kwargs)
        oracle_result = self.oracle.classify(**kwargs)
        return impl, oracle_result, expected

    def assert_matches_oracle(
        self, impl: dict[str, Any], oracle_result: dict[str, Any], label: str
    ) -> None:
        """The classification contract: (class, stop_text) multiset, cell
        records, echoed stop texts and verdict all agree; details are
        non-empty strings.
        """
        impl_pairs = sorted((d["class"], d["stop_text"]) for d in impl["divergences"])
        oracle_pairs = sorted(oracle_result["findings"])
        self.assertEqual(impl_pairs, oracle_pairs, label)
        self.assertEqual(impl["verdict"], oracle_result["verdict"], label)
        self.assertEqual(impl["stop_texts"], oracle_result["stop_texts"], label)
        for finding in impl["divergences"]:
            self.assertIsInstance(finding["detail"], str, label)
            self.assertTrue(finding["detail"], label)
        impl_cells = {record["stop_text"]: record["semantics"] for record in impl["cells"]}
        self.assertEqual(set(impl_cells), set(oracle_result["snaps"]), label)
        for stop_text, snaps in oracle_result["snaps"].items():
            for semantic, snap in snaps.items():
                icell = impl_cells[stop_text][semantic]
                where = (label, stop_text, semantic)
                self.assertEqual(icell["available"], snap.available, where)
                self.assertEqual(icell["fired"], snap.fired, where)
                self.assertEqual(icell["firing_step"], snap.step, where)
                self.assertEqual(tuple(icell["truncated_ids"]), snap.ids, where)
                self.assertEqual(icell["returned_text"], snap.text, where)
                self.assertEqual(icell["disposition"], snap.disposition, where)


class OracleDifferentialTests(_Cycle37Base):
    """``diff_backends`` vs the docstring-derived oracle, hand cases + fuzz."""

    def test_hand_cases_match_oracle(self) -> None:
        for name, _, _, _, _, _, expected in _HAND_CASES:
            with self.subTest(name=name):
                impl, oracle_result, _ = self._run_case(name)
                self.assert_matches_oracle(impl, oracle_result, name)
                kinds = {d["class"] for d in impl["divergences"]}
                for kind in expected:
                    self.assertIn(kind, kinds, f"{name}: {kind} must fire by construction")
                if not expected:
                    self.assertEqual(impl["divergences"], [], name)
                    self.assertEqual(impl["verdict"], "agree", name)

    def test_each_class_fires_by_construction(self) -> None:
        self.assertEqual(set(_CLASS_CASES), set(_ORACLE_SEVERITY))
        self.assertEqual(set(_CLASS_CASES), set(_CLASS_SEMANTICS))
        for kind, case_name in _CLASS_CASES.items():
            with self.subTest(kind=kind):
                impl, _, _ = self._run_case(case_name)
                self.assertIn(kind, {d["class"] for d in impl["divergences"]}, case_name)

    def test_severity_table_matches(self) -> None:
        # The oracle's docstring-derived severity table equals the module's
        # DIVERGENCE_SEVERITY for every class, keys and values.
        self.assertEqual(_ORACLE_SEVERITY, DIVERGENCE_SEVERITY)

    def test_attribution_table_matches(self) -> None:
        # The oracle's docstring-derived attribution equals the module's
        # _CLASS_SEMANTICS entry for every class.
        self.assertEqual(_ORACLE_ATTRIBUTION, _CLASS_SEMANTICS)

    def test_fuzz_classification_matches_oracle(self) -> None:
        rng = random.Random("cycle37-diffbackends-fuzz")  # noqa: S311 -- seeded, not crypto
        token_pool = [
            "a",
            "b",
            "ab",
            "STOP",
            "<stop>",
            "",
            "x",
            "é",
            " ",
            "\n",
            "aa",
            "Xab",
            "bc",
            "STOPSTOP",
            "xa",
            "aba",
        ]
        stop_pool = [
            "STOP",
            "ab",
            "a",
            "<stop>",
            "é",
            "b",
            "aa",
            "bc",
            "abc",
            "xab",
            "zzz",
            " ",
            "\n",
            "Xab",
            "STOPSTOP",
            "xa",
        ]
        for trial in range(350):
            n_tokens = rng.randint(0, 6)
            texts = tuple(rng.choice(token_pool) for _ in range(n_tokens))
            prompt_texts: tuple[str, ...] = (
                tuple(rng.choice(token_pool) for _ in range(rng.randint(0, 2)))
                if rng.random() < 0.3
                else ()
            )
            backend, prompt_ids, output_ids = _make_backend(texts, prompt_texts)
            if rng.random() < 0.08 and n_tokens:
                # Unknown token id: decode raises, the cells degrade to
                # unavailable and are excluded from the divergence checks.
                output_ids = output_ids + (777,)
            stops = rng.sample(stop_pool, rng.randint(1, 3))
            if rng.random() < 0.15:
                stops.append(rng.choice(stops))  # duplicate exercises dedup
            eos_roll = rng.random()
            if eos_roll < 0.35:
                eos_id = None
            elif eos_roll < 0.7 and n_tokens:
                eos_id = rng.randint(1, n_tokens)  # present in the output
            else:
                eos_id = 900 + rng.randint(0, 5)  # absent
            min_tokens = rng.randint(0, 4)
            with self.subTest(trial=trial):
                kwargs: dict[str, Any] = {
                    "prompt_ids": prompt_ids,
                    "output_ids": output_ids,
                    "stop_texts": stops,
                    "eos_id": eos_id,
                    "min_tokens": min_tokens,
                    "backend": backend,
                }
                impl = diff_backends(**kwargs)
                oracle_result = self.oracle.classify(**kwargs)
                self.assert_matches_oracle(impl, oracle_result, f"fuzz-{trial}")
                # Structural pin: NO_MIN_TOKENS_GATE is nested inside the
                # MIN_TOKENS_GATED firing, so it can never appear alone for
                # the same stop text.
                by_stop: dict[str, set[str]] = {}
                for finding in impl["divergences"]:
                    by_stop.setdefault(finding["stop_text"], set()).add(finding["class"])
                for stop_text, kinds in by_stop.items():
                    if "NO_MIN_TOKENS_GATE" in kinds:
                        self.assertIn("MIN_TOKENS_GATED", kinds, f"fuzz-{trial}: {stop_text}")
                # Structural pin: MULTI_STOP_SELECTION always carries a
                # winner's stop text (the winners differ, so one fired).
                for finding in impl["divergences"]:
                    if finding["class"] == "MULTI_STOP_SELECTION":
                        self.assertIsNotNone(finding["stop_text"], f"fuzz-{trial}")

    def test_fuzz_byte_path_matches_oracle(self) -> None:
        # The HF byte-space path (hf_stop_token_bytes) changes which decode
        # calls happen; classification must still match the oracle.
        rng = random.Random("cycle37-diffbackends-bytepath")  # noqa: S311 -- seeded
        pool = ["a", "b", "ab", "STOP", "x", "é", " "]
        for trial in range(40):
            n_tokens = rng.randint(1, 4)
            texts = tuple(rng.choice(pool) for _ in range(n_tokens))
            mapping = {i + 1: t for i, t in enumerate(texts)}
            backend = _BytePathBackend(mapping)
            output_ids = tuple(range(1, n_tokens + 1))
            stops = rng.sample(["STOP", "ab", "a", "x", "zzz"], rng.randint(1, 2))
            with self.subTest(trial=trial):
                kwargs: dict[str, Any] = {
                    "prompt_ids": (),
                    "output_ids": output_ids,
                    "stop_texts": stops,
                    "eos_id": None,
                    "min_tokens": rng.randint(0, 2),
                    "backend": backend,
                }
                impl = diff_backends(**kwargs)
                oracle_result = self.oracle.classify(**kwargs)
                self.assert_matches_oracle(impl, oracle_result, f"bytepath-{trial}")

    def test_stop_text_dedup(self) -> None:
        backend, prompt_ids, output_ids = _make_backend(("aSTOP",))
        impl = diff_backends(
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            stop_texts=["STOP", "STOP", "P", "P"],
            backend=backend,
        )
        oracle_result = self.oracle.classify(
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            stop_texts=["STOP", "STOP", "P", "P"],
            backend=backend,
        )
        self.assertEqual(impl["stop_texts"], ["STOP", "P"])
        self.assertEqual(len(impl["cells"]), 2)
        self.assert_matches_oracle(impl, oracle_result, "dedup")

    def test_invalid_inputs_raise_identical_codes(self) -> None:
        backend, _, _ = _make_backend(("a",))
        base: dict[str, Any] = {
            "prompt_ids": (),
            "output_ids": (1,),
            "stop_texts": ("s",),
            "eos_id": None,
            "min_tokens": 0,
            "backend": backend,
        }
        cases: list[tuple[str, dict[str, Any], str]] = [
            ("prompt_ids_list", {**base, "prompt_ids": [1]}, "invalid_prompt_ids"),
            ("output_ids_bool", {**base, "output_ids": (True,)}, "invalid_output_ids"),
            (
                "output_ids_too_long",
                {**base, "output_ids": tuple(range(MAX_OUTPUT_IDS + 1))},
                "output_ids_too_long",
            ),
            ("stop_texts_empty", {**base, "stop_texts": ()}, "empty_stop_texts"),
            ("stop_texts_str", {**base, "stop_texts": "s"}, "invalid_stop_texts"),
            ("eos_id_bool", {**base, "eos_id": True}, "invalid_eos_id"),
            ("min_tokens_bool", {**base, "min_tokens": True}, "invalid_min_tokens"),
            ("backend_no_decode", {**base, "backend": object()}, "invalid_backend"),
        ]
        for name, kwargs, code in cases:
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as impl_cm:
                    diff_backends(**kwargs)
                with self.assertRaises(ValueError) as oracle_cm:
                    self.oracle.classify(**kwargs)
                self.assertEqual(str(impl_cm.exception), code, name)
                self.assertEqual(str(oracle_cm.exception), code, name)


class ClassificationDetailTests(_Cycle37Base):
    """Per-class predicate details, finding construction, and edge cases."""

    @staticmethod
    def _kinds(impl: dict[str, Any]) -> set[str]:
        return {d["class"] for d in impl["divergences"]}

    @staticmethod
    def _details(impl: dict[str, Any], kind: str) -> list[str]:
        return [d["detail"] for d in impl["divergences"] if d["class"] == kind]

    # -- EOS_PRECEDENCE vs STR_OVER_EOS_PRECEDENCE ---------------------------

    def test_str_over_eos_without_eos_precedence(self) -> None:
        # The docstring claims STR_OVER_EOS_PRECEDENCE "Co-fires with
        # EOS_PRECEDENCE on the tie"; the code only requires the sglang/EOS
        # step tie, while EOS_PRECEDENCE needs eos_step <= min(stop steps).
        # HF firing earlier via prompt overhang breaks the co-firing: the
        # code behavior is pinned here, the doc overclaims.
        impl, oracle_result, _ = self._run_case("prompt_overhang_str_tie")
        self.assert_matches_oracle(impl, oracle_result, "prompt_overhang_str_tie")
        kinds = self._kinds(impl)
        self.assertIn("STR_OVER_EOS_PRECEDENCE", kinds)
        self.assertNotIn("EOS_PRECEDENCE", kinds)

    def test_str_over_eos_cofires_when_sglang_is_earliest(self) -> None:
        impl, oracle_result, _ = self._run_case("tie_str_eos")
        self.assert_matches_oracle(impl, oracle_result, "tie_str_eos")
        kinds = self._kinds(impl)
        self.assertIn("STR_OVER_EOS_PRECEDENCE", kinds)
        self.assertIn("EOS_PRECEDENCE", kinds)

    def test_eos_absent_means_no_eos_classes(self) -> None:
        # eos_id=None makes the EOS cell unavailable: neither EOS class can
        # fire however the stop semantics disagree.
        backend, prompt_ids, output_ids = _make_backend(("a", "STOP", "b"))
        impl = diff_backends(
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            stop_texts=("STOP",),
            eos_id=None,
            min_tokens=0,
            backend=backend,
        )
        kinds = self._kinds(impl)
        self.assertNotIn("EOS_PRECEDENCE", kinds)
        self.assertNotIn("STR_OVER_EOS_PRECEDENCE", kinds)
        self.assertIn("KEPT_VS_STRIPPED", kinds)

    def test_eos_precedence_needs_a_fired_stop(self) -> None:
        # EOS fires but no stop semantic fires: no EOS_PRECEDENCE (the
        # min() over an empty stop-step set must not fire).
        backend, prompt_ids, output_ids = _make_backend(("a", "b"))
        impl = diff_backends(
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            stop_texts=("zzz",),
            eos_id=2,
            min_tokens=0,
            backend=backend,
        )
        self.assertEqual(impl["divergences"], [])
        self.assertEqual(impl["verdict"], "agree")

    # -- KEPT_VS_STRIPPED -----------------------------------------------------

    def test_kept_vs_stripped_needs_both_fired(self) -> None:
        # HF fired but the strip cell is min_tokens-gated: no KEPT finding
        # (the gate finding fires instead).
        impl, oracle_result, _ = self._run_case("gated_combo")
        self.assert_matches_oracle(impl, oracle_result, "gated_combo")
        self.assertNotIn("KEPT_VS_STRIPPED", self._kinds(impl))
        self.assertIn("MIN_TOKENS_GATED", self._kinds(impl))

    def test_kept_text_degrade_suppresses_silently(self) -> None:
        # Red-team: the backend's decode works during cell building but the
        # KEPT_VS_STRIPPED text re-decode fails afterwards (stateful
        # backend; HF on the byte path so its emulation needs no decode).
        # The rule must degrade to silence, never fail the run.
        texts = {1: "aSTOP", 2: "b"}

        def byte_map(token_id: int) -> bytes | None:
            return texts[token_id].encode("utf-8")

        # Generous budget: count every decode the run performs.
        probe = _CountingBackend(texts, 10**9)
        probe.hf_stop_token_bytes = byte_map  # type: ignore[method-assign]
        control = diff_backends(
            prompt_ids=(), output_ids=(1, 2), stop_texts=("STOP",), backend=probe
        )
        self.assertIn("KEPT_VS_STRIPPED", {d["class"] for d in control["divergences"]})
        total = probe.calls
        self.assertGreater(total, 0)
        # Arm the failure at the LAST decode of the run: the kept-text
        # re-decode inside KEPT_VS_STRIPPED (the HARNESS checks reuse
        # returned_text, so they need no decode).
        armed = _CountingBackend(texts, total - 1)
        armed.hf_stop_token_bytes = byte_map  # type: ignore[method-assign]
        result = diff_backends(
            prompt_ids=(), output_ids=(1, 2), stop_texts=("STOP",), backend=armed
        )
        kinds = {d["class"] for d in result["divergences"]}
        self.assertNotIn("KEPT_VS_STRIPPED", kinds)
        self.assertEqual(result["divergences"], [])
        self.assertEqual(result["verdict"], "agree")
        # The oracle degrades identically on a freshly armed backend.
        fresh = _CountingBackend(texts, total - 1)
        fresh.hf_stop_token_bytes = byte_map  # type: ignore[method-assign]
        oracle_result = self.oracle.classify(
            prompt_ids=(), output_ids=(1, 2), stop_texts=("STOP",), backend=fresh
        )
        self.assertEqual(
            sorted((d["class"], d["stop_text"]) for d in result["divergences"]),
            sorted(oracle_result["findings"]),
        )
        self.assertEqual(oracle_result["verdict"], "agree")

    # -- MIN_TOKENS_GATED / NO_MIN_TOKENS_GATE ---------------------------------

    def test_post_gate_fire_has_no_gate_finding(self) -> None:
        # The stop completes after the gate: min_tokens changes nothing, so
        # the diagnostic re-run must not fire MIN_TOKENS_GATED.
        impl, oracle_result, _ = self._run_case("post_gate_fire")
        self.assert_matches_oracle(impl, oracle_result, "post_gate_fire")
        self.assertNotIn("MIN_TOKENS_GATED", self._kinds(impl))
        self.assertNotIn("NO_MIN_TOKENS_GATE", self._kinds(impl))

    def test_min_tokens_zero_never_runs_diagnostic(self) -> None:
        # min_tokens=0 with the strip cell silent (stop absent): no
        # diagnostic re-run, no gate findings.
        backend, prompt_ids, output_ids = _make_backend(("a", "b"))
        impl = diff_backends(
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            stop_texts=("zzz",),
            min_tokens=0,
            backend=backend,
        )
        self.assertEqual(impl["divergences"], [])

    def test_gated_combo_carries_sglang_winner(self) -> None:
        # min_tokens gates the vLLM side entirely: the sglang-only winner
        # still triggers MULTI_STOP_SELECTION, carrying sglang's stop.
        impl, _, _ = self._run_case("gated_combo")
        multi = [d for d in impl["divergences"] if d["class"] == "MULTI_STOP_SELECTION"]
        self.assertEqual(len(multi), 1)
        self.assertEqual(multi[0]["stop_text"], "STOP")

    # -- HARNESS_NORMALIZATION_RELIED ------------------------------------------

    def test_harness_never_flags_kept_cells(self) -> None:
        # "kept"-disposition cells keep the stop string by definition: even
        # when their text trivially contains it, they are not flagged.
        backend, prompt_ids, output_ids = _make_backend(("aSTOP",))
        impl = diff_backends(
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            stop_texts=("STOP",),
            backend=backend,
        )
        kinds = self._kinds(impl)
        self.assertIn("KEPT_VS_STRIPPED", kinds)  # hf text contains the stop...
        self.assertNotIn("HARNESS_NORMALIZATION_RELIED", kinds)  # ...but is "kept"
        # And the keep variant's returned text also contains the stop.
        keep_text = impl["cells"][0]["semantics"]["vllm_holdback_keep"]["returned_text"]
        self.assertIn("STOP", keep_text)

    def test_harness_strip_anomaly_detail(self) -> None:
        impl, oracle_result, _ = self._run_case("harness_anomaly")
        self.assert_matches_oracle(impl, oracle_result, "harness_anomaly")
        details = self._details(impl, "HARNESS_NORMALIZATION_RELIED")
        self.assertEqual(len(details), 1)
        self.assertTrue(details[0].startswith("vllm_holdback_strip is strip-disposition"))
        self.assertIn("'ab'", details[0])

    def test_harness_vacuous_for_sglang(self) -> None:
        # The DivergenceClass comment: "Vacuous for SGLang by construction:
        # text trimmed at the leftmost stop-string occurrence." A leftmost
        # trim cannot leave the stop string in the returned text, so no
        # HARNESS finding may ever attribute to sglang_stop.
        rng = random.Random("cycle37-harness-vacuity")  # noqa: S311 -- seeded
        pool = ["a", "b", "ab", "Xab", "xab", "STOP", "aa", "aba", "bab", "abX", ""]
        stop_pool = ["ab", "a", "b", "Xab", "aba", "STOP", "aa"]
        hits = 0
        for trial in range(1500):
            with self.subTest(trial=trial):
                n_tokens = rng.randint(1, 4)
                texts = tuple(rng.choice(pool) for _ in range(n_tokens))
                backend, prompt_ids, output_ids = _make_backend(texts)
                stops = rng.sample(stop_pool, rng.randint(1, 3))
                impl = diff_backends(
                    prompt_ids=prompt_ids,
                    output_ids=output_ids,
                    stop_texts=stops,
                    min_tokens=rng.randint(0, 2),
                    backend=backend,
                )
                for finding in impl["divergences"]:
                    if finding["class"] != "HARNESS_NORMALIZATION_RELIED":
                        continue
                    hits += 1
                    self.assertFalse(
                        finding["detail"].startswith("sglang_stop"),
                        f"trial {trial}: HARNESS attributed to sglang",
                    )
        self.assertGreater(hits, 0, "vacuity pin needs HARNESS to fire at all")

    # -- MULTI_STOP_SELECTION ----------------------------------------------------

    def test_multi_tie_prefers_sglang_winner_stop_text(self) -> None:
        impl, oracle_result, _ = self._run_case("multi_tie")
        self.assert_matches_oracle(impl, oracle_result, "multi_tie")
        multi = [d for d in impl["divergences"] if d["class"] == "MULTI_STOP_SELECTION"]
        self.assertEqual(len(multi), 1)
        # SGLang's first-listed rule picks "ab"; the finding carries the
        # sglang winner's stop text even though vLLM picks "xa".
        self.assertEqual(multi[0]["stop_text"], "ab")
        self.assertIn("stop 'ab' at step 1", multi[0]["detail"])
        self.assertIn("stop 'xa' at step 1", multi[0]["detail"])

    def test_multi_full_tie_no_finding(self) -> None:
        impl, oracle_result, _ = self._run_case("multi_full_tie")
        self.assert_matches_oracle(impl, oracle_result, "multi_full_tie")
        self.assertNotIn("MULTI_STOP_SELECTION", self._kinds(impl))

    def test_multi_no_fired_stops_no_finding(self) -> None:
        backend, prompt_ids, output_ids = _make_backend(("a", "b"))
        impl = diff_backends(
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            stop_texts=("zzz", "yyy"),
            backend=backend,
        )
        self.assertNotIn("MULTI_STOP_SELECTION", self._kinds(impl))
        self.assertEqual(impl["verdict"], "agree")

    # -- FIRING_STEP_DIFFERS -----------------------------------------------------

    def test_firing_step_differs_excludes_unavailable(self) -> None:
        # Unknown token id: every decode-dependent cell is unavailable; the
        # step set is empty, so no FIRING_STEP_DIFFERS and no crash.
        backend = _ConcatBackend({1: "a"})
        impl = diff_backends(
            prompt_ids=(), output_ids=(1, 777), stop_texts=("zzz",), backend=backend
        )
        cells = impl["cells"][0]["semantics"]
        for semantic in ("hf_stop_string", "vllm_holdback_strip", "sglang_stop"):
            self.assertFalse(cells[semantic]["available"], semantic)
        self.assertNotIn("FIRING_STEP_DIFFERS", self._kinds(impl))
        self.assertEqual(impl["verdict"], "agree")

    def test_firing_step_parsing(self) -> None:
        cases: list[tuple[str, int | None]] = [
            ("fired_at_step_1", 1),
            ("fired_at_step_007", 7),
            ("fired_at_step_4096", 4096),
            ("no_stop_within_cap", None),
            ("gated_by_min_tokens", None),
            ("eos_id_unavailable", None),
            ("decode_unavailable", None),
            ("token_text_unavailable", None),
            ("invalid_emulation_input", None),
            ("invalid_eos_id", None),
            ("", None),
            ("fired_at_step_", None),
            ("fired_at_step_1x", None),
            ("xfired_at_step_1", None),
            ("fired_at_step_-1", None),
            ("fired_at_step_1 ", None),
        ]
        for detail, expected in cases:
            with self.subTest(detail=detail):
                self.assertEqual(_firing_step(detail), expected, detail)

    # -- _effective_text paths -----------------------------------------------------

    def test_effective_text_paths(self) -> None:
        raising = _ConcatBackend({})
        # returned_text wins even when decode would fail.
        cell = _ImplCellResult(
            semantic="vllm_holdback_strip",
            available=True,
            fired=True,
            firing_step=1,
            truncated_ids=(1,),
            returned_text="kept-as-is",
            disposition="stripped",
        )

        def _boom(ids: tuple[int, ...]) -> str:
            raise RuntimeError("must not be called")

        raising.decode = _boom  # type: ignore[method-assign]
        self.assertEqual(_effective_text(raising, cell), "kept-as-is")
        # Otherwise the truncated ids are decoded...
        backend, _, _ = _make_backend(("hello",))
        cell2 = _ImplCellResult(
            semantic="hf_stop_string",
            available=True,
            fired=True,
            firing_step=1,
            truncated_ids=(1,),
            returned_text=None,
            disposition="kept",
        )
        self.assertEqual(_effective_text(backend, cell2), "hello")

        # ...degrading to None when decode raises, returns non-str, or is
        # absent -- the text checks degrade, never fail the diff.
        class _Raises:
            @staticmethod
            def decode(ids: tuple[int, ...]) -> str:
                raise RuntimeError("boom")

        class _NonStr:
            @staticmethod
            def decode(ids: tuple[int, ...]) -> int:
                return 42

        self.assertIsNone(_effective_text(_Raises(), cell2))
        self.assertIsNone(_effective_text(_NonStr(), cell2))
        self.assertIsNone(_effective_text(object(), cell2))

    # -- finding construction ------------------------------------------------------

    def test_result_key_sets(self) -> None:
        backend, prompt_ids, output_ids = _make_backend(("aSTOP",))
        impl = diff_backends(
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            stop_texts=("STOP",),
            eos_id=None,
            min_tokens=0,
            backend=backend,
        )
        # The docstring lists "stop_texts, eos_id, min_tokens, per-stop
        # cells, divergences and verdict" but the dict also carries
        # prompt_tokens/output_tokens: pinned here (doc discrepancy #2).
        self.assertEqual(
            set(impl),
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
        self.assertEqual(impl["prompt_tokens"], 0)
        self.assertEqual(impl["output_tokens"], 1)
        cell = impl["cells"][0]["semantics"]["hf_stop_string"]
        self.assertEqual(
            set(cell),
            {
                "available",
                "fired",
                "firing_step",
                "truncated_ids",
                "returned_text",
                "disposition",
            },
        )
        for finding in impl["divergences"]:
            self.assertEqual(set(finding), {"class", "stop_text", "detail"})
        self.assertEqual(impl["verdict"], "diverge")

    def test_divergence_ordering(self) -> None:
        # Observed construction order: per-stop classes in code order, then
        # the cross-stop MULTI_STOP_SELECTION last.
        impl, _, _ = self._run_case("gated_combo")
        self.assertEqual(
            [d["class"] for d in impl["divergences"]],
            [
                "FIRING_STEP_DIFFERS",
                "MIN_TOKENS_GATED",
                "NO_MIN_TOKENS_GATE",
                "MULTI_STOP_SELECTION",
            ],
        )

    def test_detail_anchors(self) -> None:
        # The details cite the research anchors the docstrings pin per class.
        impl, _, _ = self._run_case("gated_combo")
        for detail in self._details(impl, "MIN_TOKENS_GATED"):
            self.assertIn("vLLM PR #22014", detail)
        for detail in self._details(impl, "NO_MIN_TOKENS_GATE"):
            self.assertIn("vLLM PR #22014", detail)
        impl, _, _ = self._run_case("tie_str_eos")
        for detail in self._details(impl, "EOS_PRECEDENCE"):
            self.assertIn("SGLang #28802", detail)
        for detail in self._details(impl, "STR_OVER_EOS_PRECEDENCE"):
            self.assertIn("28802", detail)
        impl, _, _ = self._run_case("multi_tie")
        for detail in self._details(impl, "MULTI_STOP_SELECTION"):
            self.assertIn("vLLM PR #49391", detail)


class SectionAggregationTests(_Cycle37Base):
    """``divergence_section_data`` aggregation vs the oracle's ``summarize``."""

    def assert_section_matches(
        self, impl: dict[str, Any] | None, oracle_result: dict[str, Any] | None, label: str
    ) -> None:
        self.assertEqual(impl is None, oracle_result is None, label)
        if impl is None or oracle_result is None:
            return
        for key in (
            "verdict",
            "pairs_verdict",
            "matrix",
            "severities",
            "stop_texts",
            "eos_id",
            "min_tokens",
            "observations_truncated",
            "observations_total",
            "observations_processed",
        ):
            self.assertEqual(impl[key], oracle_result[key], (label, key))
        self.assertEqual(len(impl["per_fixture"]), len(oracle_result["per_fixture"]), label)
        for ientry, oentry in zip(impl["per_fixture"], oracle_result["per_fixture"], strict=True):
            self.assertEqual(ientry["fixture"], oentry["fixture"], label)
            self.assertEqual(ientry["verdict"], oentry["verdict"], label)
            ipairs = sorted(
                (d["class"], d["stop_text"], d["severity"]) for d in ientry["divergences"]
            )
            opairs = sorted(
                (d["class"], d["stop_text"], d["severity"]) for d in oentry["divergences"]
            )
            self.assertEqual(ipairs, opairs, label)
            for finding in ientry["divergences"]:
                self.assertEqual(set(finding), {"class", "stop_text", "detail", "severity"}, label)
                self.assertTrue(finding["detail"], label)

    @staticmethod
    def _section_kwargs(
        observations: list[Any] | tuple[Any, ...], backend: _ConcatBackend, **extra: Any
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "observations": observations,
            "stop_texts": ("STOP",),
            "eos_id": None,
            "min_tokens": 0,
            "backend": backend,
        }
        kwargs.update(extra)
        return kwargs

    def test_section_agree_diverge_exact(self) -> None:
        backend = _ConcatBackend({1: "a", 2: "STOP", 3: "b"})
        observations = [
            {"fixture": "agree_fx", "input_ids": [], "output_ids": [1]},
            {"fixture": "diverge_fx", "input_ids": [], "output_ids": [1, 2, 3]},
        ]
        kwargs = self._section_kwargs(observations, backend)
        impl = divergence_section_data(**kwargs)
        oracle_result = self.oracle.summarize(**kwargs)
        self.assert_section_matches(impl, oracle_result, "agree-diverge")
        assert impl is not None
        self.assertEqual(impl["verdict"], "diverge")
        self.assertEqual(
            impl["pairs_verdict"],
            "HF vs vLLM vs SGLang stop semantics: 1/2 agree; 1 divergence",
        )
        self.assertEqual(
            impl["matrix"],
            {"KEPT_VS_STRIPPED": {"hf_stop_string": 1, "vllm_holdback_strip": 1}},
        )
        self.assertEqual(impl["severities"], {"KEPT_VS_STRIPPED": "high"})
        self.assertFalse(impl["observations_truncated"])
        self.assertEqual(impl["observations_total"], 2)
        self.assertEqual(impl["observations_processed"], 2)

    def test_matrix_counts_per_finding_not_per_fixture(self) -> None:
        # Two stops fire KEPT_VS_STRIPPED in one fixture: each *finding*
        # contributes its class's attributed semantics to the matrix.
        backend = _ConcatBackend({1: "aSTOP"})
        observations = [{"fixture": "f", "input_ids": [], "output_ids": [1]}]
        kwargs = self._section_kwargs(observations, backend, stop_texts=("STOP", "P"))
        impl = divergence_section_data(**kwargs)
        oracle_result = self.oracle.summarize(**kwargs)
        self.assert_section_matches(impl, oracle_result, "per-finding-counts")
        assert impl is not None
        self.assertEqual(
            impl["matrix"],
            {"KEPT_VS_STRIPPED": {"hf_stop_string": 2, "vllm_holdback_strip": 2}},
        )
        # Severities cover exactly the fired classes (the report's badges
        # look up each matrix row's severity).
        self.assertEqual(set(impl["severities"]), set(impl["matrix"]))
        self.assertEqual(impl["severities"], {"KEPT_VS_STRIPPED": "high"})

    def test_severity_values_are_badge_vocab(self) -> None:
        # reporting.py maps severity -> badge via {"high": "fail",
        # "medium": "neutral", "low": "neutral"}; unknown values would
        # silently degrade to neutral, so the mapping's range is pinned.
        self.assertEqual(set(DIVERGENCE_SEVERITY.values()), {"high", "medium", "low"})

    def test_pairs_verdict_pluralization(self) -> None:
        backend = _ConcatBackend({1: "x"})
        observations = [{"fixture": "f", "input_ids": [], "output_ids": [1]}]
        kwargs = self._section_kwargs(observations, backend, stop_texts=("zzz",))
        impl = divergence_section_data(**kwargs)
        assert impl is not None
        self.assertEqual(
            impl["pairs_verdict"],
            "HF vs vLLM vs SGLang stop semantics: 1/1 agree; 0 divergences",
        )
        backend2 = _ConcatBackend({1: "aSTOP"})
        kwargs = self._section_kwargs(observations, backend2, stop_texts=("STOP", "P", "aS"))
        impl = divergence_section_data(**kwargs)
        assert impl is not None
        # 3x KEPT_VS_STRIPPED plus MULTI_STOP_SELECTION (sglang takes the
        # first-listed "STOP", vLLM the earliest-completing "aS").
        self.assertEqual(
            impl["pairs_verdict"],
            "HF vs vLLM vs SGLang stop semantics: 0/1 agree; 4 divergences",
        )

    def test_fixture_fallback_uses_candidate_index(self) -> None:
        # Skipped observations still consume candidate indices for the
        # fixture_N fallback names (observed behavior).
        backend = _ConcatBackend({1: "x"})
        observations: list[Any] = [
            {"fixture": "bad"},  # no ids: skipped by omission
            "junk",  # not a dict: skipped
            {"input_ids": [], "output_ids": [1]},
        ]
        kwargs = self._section_kwargs(observations, backend)
        impl = divergence_section_data(**kwargs)
        oracle_result = self.oracle.summarize(**kwargs)
        self.assert_section_matches(impl, oracle_result, "fallback-index")
        assert impl is not None
        self.assertEqual([e["fixture"] for e in impl["per_fixture"]], ["fixture_2"])
        self.assertEqual(impl["observations_processed"], 1)

    def test_section_truncation(self) -> None:
        backend = _ConcatBackend({1: "x", 2: "y"})
        observations = [
            {"fixture": f"obs_{i}", "input_ids": [], "output_ids": [1, 2]}
            for i in range(MAX_DIVERGENCE_OBSERVATIONS + 8)
        ]
        kwargs = self._section_kwargs(observations, backend)
        impl = divergence_section_data(**kwargs)
        oracle_result = self.oracle.summarize(**kwargs)
        self.assert_section_matches(impl, oracle_result, "truncation")
        assert impl is not None
        self.assertTrue(impl["observations_truncated"])
        self.assertEqual(impl["observations_total"], MAX_DIVERGENCE_OBSERVATIONS + 8)
        self.assertEqual(impl["observations_processed"], MAX_DIVERGENCE_OBSERVATIONS)
        self.assertEqual(
            [e["fixture"] for e in impl["per_fixture"]],
            [f"obs_{i}" for i in range(MAX_DIVERGENCE_OBSERVATIONS)],
        )

    def test_section_none_when_nothing_usable(self) -> None:
        backend = _ConcatBackend({1: "x"})
        kwargs = self._section_kwargs([{"nope": 1}, "junk"], backend)
        self.assertIsNone(divergence_section_data(**kwargs))
        self.assertIsNone(self.oracle.summarize(**kwargs))

    def test_section_fuzz_matches_oracle(self) -> None:
        rng = random.Random("cycle37-diffbackends-section")  # noqa: S311 -- seeded
        token_pool = ["a", "b", "ab", "STOP", "<stop>", "", "x", "é", " ", "\n"]
        stop_pool = ["STOP", "ab", "a", "<stop>", "é", "b", "zzz", " "]
        for trial in range(60):
            n_tokens = rng.randint(1, 5)
            texts = tuple(rng.choice(token_pool) for _ in range(n_tokens))
            backend, _, _ = _make_backend(texts)
            observations: list[Any] = []
            for _ in range(rng.randint(0, 8)):
                roll = rng.random()
                if roll < 0.6:
                    observations.append(
                        {
                            "fixture": f"fx_{trial}",
                            "input_ids": [rng.randint(1, n_tokens) for _ in range(2)],
                            "output_ids": [
                                rng.randint(1, n_tokens) for _ in range(rng.randint(0, 5))
                            ],
                        }
                    )
                elif roll < 0.75:
                    observations.append(rng.choice(["junk", 42, None]))
                else:
                    observations.append({"input_ids": [], "output_ids": ["bad"]})
            if rng.random() < 0.3:
                observations = tuple(observations)
            stops = rng.sample(stop_pool, rng.randint(1, 3))
            with self.subTest(trial=trial):
                kwargs: dict[str, Any] = {
                    "observations": observations,
                    "stop_texts": stops,
                    "eos_id": None if rng.random() < 0.5 else rng.randint(1, n_tokens + 1),
                    "min_tokens": rng.randint(0, 3),
                    "backend": backend,
                }
                impl = divergence_section_data(**kwargs)
                oracle_result = self.oracle.summarize(**kwargs)
                self.assert_section_matches(impl, oracle_result, f"section-{trial}")

    def test_section_invalid_inputs_raise_identical_codes(self) -> None:
        backend = _ConcatBackend({1: "x"})
        base: dict[str, Any] = {
            "observations": [],
            "stop_texts": ("s",),
            "eos_id": None,
            "min_tokens": 0,
            "backend": backend,
        }
        cases: list[tuple[str, dict[str, Any], str]] = [
            ("observations_dict", {**base, "observations": {"a": 1}}, "invalid_observations"),
            ("stop_texts_empty", {**base, "stop_texts": ()}, "empty_stop_texts"),
            ("eos_id_negative", {**base, "eos_id": -1}, "invalid_eos_id"),
            ("min_tokens_negative", {**base, "min_tokens": -1}, "invalid_min_tokens"),
            ("backend_no_decode", {**base, "backend": object()}, "invalid_backend"),
        ]
        for name, kwargs, code in cases:
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as impl_cm:
                    divergence_section_data(**kwargs)
                with self.assertRaises(ValueError) as oracle_cm:
                    self.oracle.summarize(**kwargs)
                self.assertEqual(str(impl_cm.exception), code, name)
                self.assertEqual(str(oracle_cm.exception), code, name)


# ---------------------------------------------------------------------------
# Mutation sensitivity: run this file against mutated COPIES in /tmp.
# ---------------------------------------------------------------------------

# PID-scoped so two pytest processes running this file concurrently stage
# into disjoint directories (a fixed name raced -> FileExistsError).
_SCRATCH_ROOT = f"/tmp/cycle37_diffbackends_mutants_{os.getpid()}"  # noqa: S108 -- task-mandated /tmp scratch

_MUTANTS: tuple[tuple[str, str, str], ...] = (
    # (name, exact source snippet, replacement)
    (
        "severity-flip",
        '"KEPT_VS_STRIPPED": "high",',
        '"KEPT_VS_STRIPPED": "medium",',
    ),
    (
        "attribution-drop",
        '"KEPT_VS_STRIPPED": (HF_STOP_STRING, VLLM_HOLDBACK_STRIP),',
        '"KEPT_VS_STRIPPED": (HF_STOP_STRING,),',
    ),
    (
        "eos-strict",
        "if stop_steps and eos.firing_step <= min(stop_steps):",
        "if stop_steps and eos.firing_step < min(stop_steps):",
    ),
    (
        "firing-step-off-by-one",
        "return int(match.group(1)) if match is not None else None",
        "return int(match.group(1)) + 1 if match is not None else None",
    ),
    (
        "multi-tiebreak-list-first",
        "vkey = (strip_cell.firing_step, end_index, list_index)",
        "vkey = (strip_cell.firing_step, list_index, end_index)",
    ),
    (
        "harness-flags-kept",
        'if not cell.fired or cell.disposition != "stripped":',
        "if not cell.fired:",
    ),
    (
        "str-over-eos-wrong-cell",
        "and eos.firing_step == sglang.firing_step",
        "and eos.firing_step == hf.firing_step",
    ),
)


class SensitivityTests(unittest.TestCase):
    """Prove this file catches real bugs: each mutant is a copy of the
    package in /tmp with one behavior change; the suite must fail on every
    mutant and pass on the pristine copy. Never mutates the repo.
    """

    def test_mutants_are_caught(self) -> None:
        if os.environ.get("C37_UNDER_MUTANT") == "1":
            self.skipTest("running under a mutant copy")
        scratch = _SCRATCH_ROOT
        shutil.rmtree(scratch, ignore_errors=True)
        os.makedirs(scratch)
        try:
            package_dir = os.path.dirname(model_protocol_lab.__file__)
            test_src = os.path.abspath(__file__)
            python = sys.executable
            results: list[tuple[str, bool, str]] = []
            # Baseline first: the pristine copy must pass fully.
            baseline_dir = os.path.join(scratch, "baseline")
            self._stage_copy(scratch, baseline_dir, package_dir, test_src, None, None)
            baseline_ok = self._run_suite(baseline_dir, python, fast=False)
            results.append(("baseline-pristine", baseline_ok, "must pass"))
            for name, old, new in _MUTANTS:
                mutant_dir = os.path.join(scratch, f"mutant_{name}")
                self._stage_copy(scratch, mutant_dir, package_dir, test_src, old, new)
                caught = not self._run_suite(mutant_dir, python, fast=True)
                results.append((name, caught, "must fail"))
        finally:
            # Do not accumulate package copies in /tmp across runs (tmpfs is
            # small; leftovers starve the 128 MiB fixture tests of disk).
            shutil.rmtree(scratch, ignore_errors=True)
        failures = [name for name, ok, _ in results if not ok]
        detail = "; ".join(
            f"{name}: {'OK' if ok else 'NOT CAUGHT'} ({note})" for name, ok, note in results
        )
        self.assertEqual(failures, [], f"mutants not caught: {detail}")

    def _stage_copy(
        self,
        scratch: str,
        dest: str,
        package_dir: str,
        test_src: str,
        old: str | None,
        new: str | None,
    ) -> None:
        os.makedirs(dest)
        shutil.copytree(
            package_dir,
            os.path.join(dest, "model_protocol_lab"),
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy(test_src, os.path.join(dest, "test_cycle37_diffbackends_verify.py"))
        if old is not None:
            target = os.path.join(dest, "model_protocol_lab", "diffbackends.py")
            with open(target, encoding="utf-8") as handle:
                text = handle.read()
            self.assertIn(old, text, f"mutant snippet not found: {old!r}")
            self.assertEqual(text.count(old), 1, f"mutant snippet not unique: {old!r}")
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(text.replace(old, new or ""))

    @staticmethod
    def _run_suite(workdir: str, python: str, *, fast: bool) -> bool:
        env = dict(os.environ)
        env["PYTHONPATH"] = workdir
        env["C37_UNDER_MUTANT"] = "1"
        cmd = [
            python,
            "-m",
            "pytest",
            "test_cycle37_diffbackends_verify.py",
            "-q",
            "-p",
            "no:cacheprovider",
        ]
        if fast:
            cmd.append("-x")
        proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell, local pytest
            cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=900, check=False
        )
        return proc.returncode == 0


if __name__ == "__main__":
    unittest.main()
