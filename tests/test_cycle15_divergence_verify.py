"""Cycle 15: differential verification of the divergence classification.

``diff_backends`` (``model_protocol_lab/diffbackends.py``) runs the lab's
five stop-semantics emulations over one recorded token sequence and
classifies disagreements into eight divergence classes;
``divergence_section_data`` aggregates per-observation results into the
report's "Backend divergence" section (verdict, pairs_verdict,
per_fixture, class x semantic count matrix, severities, truncation
accounting).

This file verifies the *classification and aggregation* layers against an
independent oracle. The cell layer -- the ``emulate_*`` functions in
``model_protocol_lab/backends.py`` -- was differentially verified in
cycles 9-11 and is reused here as ground truth: the oracle runs the same
emulations, maps their outcomes to its own cell records, then applies its
own independently written classification rules (derived from the class
semantics documented in the ``diffbackends`` module docstring, not from
the implementation's comparison code) and its own independently written
section aggregation.

Verified properties:
- per-observation classification: the oracle's ``(class, stop_text)``
  multiset, cell records, echoed stop texts and verdict match
  ``diff_backends`` on hand-derived adversarial cases plus a seeded fuzz
  sweep (random stop texts incl. duplicates, EOS present/absent,
  min_tokens 0..3, prompts, unknown token ids);
- every divergence class fires at least once by construction, and the
  oracle's attributed semantics for each firing equal the module's
  ``_CLASS_SEMANTICS`` entry (the class x semantic matrix pin);
- severities: the oracle's docstring-derived severity pin equals
  ``DIVERGENCE_SEVERITY``;
- aggregation: the oracle's section data matches
  ``divergence_section_data`` on hand cases plus a seeded fuzz sweep over
  valid/invalid/mixed observations (non-int ids, over-cap ids, non-dict
  entries, missing keys, tuple input);
- truncation accounting: >512 observations exercise the truncation path;
  ``observations_processed`` counts only emulated observations;
- invalid inputs raise identical fixed codes in implementation and oracle;
- sensitivity: a mutated oracle attribution (removed semantic; swapped
  classes) fails the attribution check, and a mutated classifier
  predicate is caught by the differential comparison.

Deterministic: all fuzzing uses fixed seeds. No network, no model
weights, no torch.
"""

from __future__ import annotations

import random
import re
import unittest
from dataclasses import dataclass
from typing import Any

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
    diff_backends,
    divergence_section_data,
)

# ---------------------------------------------------------------------------
# Concatenative fake backend.
# ---------------------------------------------------------------------------


class _ConcatBackend:
    """Backend whose decode is exactly incremental concatenation.

    ``decode(ids)`` joins the per-token texts; unknown ids raise KeyError,
    which the emulations (and the oracle's text helper) degrade on -- this
    exercises the unavailable-cell path in the fuzz.
    """

    fingerprint = "cycle15-divergence-concat-backend-not-a-model"

    def __init__(self, texts: dict[int, str]) -> None:
        self._texts = dict(texts)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        """Decode."""
        return "".join(self._texts[token] for token in token_ids)


# ---------------------------------------------------------------------------
# Independent oracle: classification re-derived from the documented class
# semantics, aggregation re-derived from the section-data contract.
# ---------------------------------------------------------------------------

_STOP_SEMANTICS = (HF_STOP_STRING, VLLM_HOLDBACK_STRIP, VLLM_HOLDBACK_KEEP, SGLANG_STOP)


@dataclass(frozen=True)
class _OracleCell:
    """The oracle's own record of one emulated semantic's outcome."""

    semantic: str
    available: bool
    fired: bool
    firing_step: int | None
    truncated_ids: tuple[int, ...]
    returned_text: str | None
    disposition: str | None


@dataclass(frozen=True)
class _OracleDivergence:
    """One classified disagreement, with the oracle's semantic attribution."""

    kind: str
    stop_text: str | None
    attributed: tuple[str, ...]


class _DivergenceOracle:
    """Independent re-derivation of the divergence classification.

    Every rule below is written from the class semantics documented in the
    ``diffbackends`` module docstring (the research-log-anchored
    descriptions of the eight classes), not from the implementation's
    comparison code. ``CLASS_SEMANTICS`` is the oracle's own attribution
    table -- the tests pin each entry against the module's
    ``_CLASS_SEMANTICS`` -- and ``SEVERITY`` is derived from the severity
    pin comment ("high": silently changes the returned text or the
    stopping precedence; "medium": changes *when* stopping happens;
    "low": correct scoring would depend on harness-side normalization).
    """

    CLASS_SEMANTICS: dict[str, tuple[str, ...]] = {
        "KEPT_VS_STRIPPED": (HF_STOP_STRING, VLLM_HOLDBACK_STRIP),
        "FIRING_STEP_DIFFERS": (
            HF_STOP_STRING,
            VLLM_HOLDBACK_STRIP,
            VLLM_HOLDBACK_KEEP,
            SGLANG_STOP,
        ),
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

    SEVERITY: dict[str, str] = {
        "KEPT_VS_STRIPPED": "high",
        "EOS_PRECEDENCE": "high",
        "STR_OVER_EOS_PRECEDENCE": "high",
        "MULTI_STOP_SELECTION": "high",
        "FIRING_STEP_DIFFERS": "medium",
        "MIN_TOKENS_GATED": "medium",
        "NO_MIN_TOKENS_GATE": "medium",
        "HARNESS_NORMALIZATION_RELIED": "low",
    }

    def __init__(
        self,
        *,
        class_semantics: dict[str, tuple[str, ...]] | None = None,
        severity: dict[str, str] | None = None,
    ) -> None:
        # Copies, so the sensitivity controls can mutate one oracle
        # without touching the class-level derivation.
        self.class_semantics = dict(
            self.CLASS_SEMANTICS if class_semantics is None else class_semantics
        )
        self.severity = dict(self.SEVERITY if severity is None else severity)

    # -- input validation (the lab's fixed-code contract, re-derived) --

    @staticmethod
    def _checked_ids(value: object, name: str, cap: int) -> tuple[int, ...]:
        """Checked ids."""
        if type(value) is not tuple or any(
            type(token) is not int or not 0 <= token < 2**31 for token in value
        ):
            raise ValueError(f"invalid_{name}")
        if len(value) > cap:
            raise ValueError(f"{name}_too_long")
        return value

    @staticmethod
    def _checked_stop_texts(value: object) -> tuple[str, ...]:
        """Checked stop texts."""
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

    def _observation_ids(
        self, observation: dict[str, Any], key: str, cap: int
    ) -> tuple[int, ...] | None:
        """Observation ids."""
        raw: object = observation.get(key)
        if type(raw) is list:
            raw = tuple(raw)
        try:
            return self._checked_ids(raw, key, cap)
        except ValueError:
            return None

    # -- cell layer (ground-truth emulations, oracle-owned records) --

    @staticmethod
    def _parse_firing_step(detail: str) -> int | None:
        """Parse firing step."""
        match = re.fullmatch(r"fired_at_step_(\d+)", detail)
        return int(match.group(1)) if match is not None else None

    @staticmethod
    def _safe_decode(backend: Any, ids: tuple[int, ...]) -> str | None:
        """Safe decode."""
        try:
            text = backend.decode(ids)
        except Exception:  # noqa: BLE001 -- text checks degrade, never fail the oracle
            return None
        return text if type(text) is str else None

    def _cell_text(self, backend: Any, cell: _OracleCell) -> str | None:
        """Cell text."""
        if cell.returned_text is not None:
            return cell.returned_text
        return self._safe_decode(backend, cell.truncated_ids)

    def _build_cells(
        self,
        *,
        prompt_ids: tuple[int, ...],
        output_ids: tuple[int, ...],
        stop_text: str,
        eos_id: int | None,
        min_tokens: int,
        backend: Any,
    ) -> dict[str, _OracleCell]:
        """Build cells."""
        token_bytes = getattr(backend, "hf_stop_token_bytes", None)

        def make(semantic: str, outcome: Any, disposition: str | None) -> _OracleCell:
            """Make."""
            return _OracleCell(
                semantic=semantic,
                available=bool(outcome.available),
                fired=bool(outcome.fired),
                firing_step=self._parse_firing_step(outcome.detail),
                truncated_ids=tuple(outcome.truncated_ids),
                returned_text=outcome.returned_text,
                disposition=disposition,
            )

        return {
            HF_STOP_STRING: make(
                HF_STOP_STRING,
                emulate_hf_stop_string(
                    backend, prompt_ids, output_ids, stop_text, token_bytes=token_bytes
                ),
                "kept",
            ),
            VLLM_HOLDBACK_STRIP: make(
                VLLM_HOLDBACK_STRIP,
                emulate_vllm_holdback(
                    backend,
                    output_ids,
                    stop_text,
                    include_stop_str_in_output=False,
                    min_tokens=min_tokens,
                ),
                "stripped",
            ),
            VLLM_HOLDBACK_KEEP: make(
                VLLM_HOLDBACK_KEEP,
                emulate_vllm_holdback(
                    backend,
                    output_ids,
                    stop_text,
                    include_stop_str_in_output=True,
                    min_tokens=min_tokens,
                ),
                "kept",
            ),
            SGLANG_STOP: make(
                SGLANG_STOP, emulate_sglang_stop(backend, output_ids, stop_text), "stripped"
            ),
            EOS_APPENDED: make(EOS_APPENDED, emulate_eos_stop(output_ids, eos_id), None),
        }

    # -- per-class firing predicates, each derived from the module docstring --

    @staticmethod
    def _firing_steps_differ(live: list[_OracleCell]) -> bool:
        """The stop-string semantics disagree on the completing output step."""
        return len({cell.firing_step for cell in live}) > 1

    def _kept_text_differs(self, backend: Any, hf: _OracleCell, strip: _OracleCell) -> bool:
        """HF keeps the stop string while vLLM's default strips it."""
        if not (hf.available and hf.fired and strip.available and strip.fired):
            return False
        kept_text = self._cell_text(backend, hf)
        return kept_text is not None and kept_text != strip.returned_text

    @staticmethod
    def _ungated_strip_fires(
        backend: Any,
        output_ids: tuple[int, ...],
        stop_text: str,
        min_tokens: int,
        strip: _OracleCell,
    ) -> bool:
        """The diagnostic re-run: the stop fires with min_tokens=0, so the
        given min_tokens is what silenced the strip cell (vLLM PR #22014).
        """
        if not (min_tokens > 0 and strip.available and not strip.fired):
            return False
        ungated = emulate_vllm_holdback(backend, output_ids, stop_text, min_tokens=0)
        return bool(ungated.available and ungated.fired)

    @staticmethod
    def _eos_precedence_fires(eos: _OracleCell, live: list[_OracleCell]) -> bool:
        """EOS fires at/before the stop-string firing step (SGLang #28802)."""
        if not (eos.available and eos.fired and eos.firing_step is not None):
            return False
        stop_steps = [
            cell.firing_step for cell in live if cell.fired and cell.firing_step is not None
        ]
        return bool(stop_steps) and eos.firing_step <= min(stop_steps)

    @staticmethod
    def _str_over_eos_fires(sglang: _OracleCell, eos: _OracleCell) -> bool:
        """Stop string and EOS fire in the same step: precedence, not firing
        order, decides the outcome -- and the engines disagree (SGLang's
        shipped #28802 fix: stop wins; TGI's documented precedence: EOS
        wins).
        """
        return (
            sglang.available
            and sglang.fired
            and sglang.firing_step is not None
            and eos.available
            and eos.fired
            and eos.firing_step == sglang.firing_step
        )

    # skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
    def _classify_stop(
        self,
        *,
        stop_text: str,
        cells: dict[str, _OracleCell],
        backend: Any,
        output_ids: tuple[int, ...],
        min_tokens: int,
    ) -> list[_OracleDivergence]:
        """Classify stop."""
        kinds: list[str] = []
        live = [cells[semantic] for semantic in _STOP_SEMANTICS if cells[semantic].available]
        hf = cells[HF_STOP_STRING]
        strip = cells[VLLM_HOLDBACK_STRIP]
        sglang = cells[SGLANG_STOP]
        eos = cells[EOS_APPENDED]

        if self._firing_steps_differ(live):
            kinds.append("FIRING_STEP_DIFFERS")
        if self._kept_text_differs(backend, hf, strip):
            kinds.append("KEPT_VS_STRIPPED")
        if self._ungated_strip_fires(backend, output_ids, stop_text, min_tokens, strip):
            kinds.append("MIN_TOKENS_GATED")
            # NO_MIN_TOKENS_GATE: SGLang has no min_tokens equivalent, so
            # its cell fires anyway while vLLM's gate silences the strip.
            if sglang.available and sglang.fired and sglang.firing_step is not None:
                kinds.append("NO_MIN_TOKENS_GATE")
        if self._eos_precedence_fires(eos, live):
            kinds.append("EOS_PRECEDENCE")
        if self._str_over_eos_fires(sglang, eos):
            kinds.append("STR_OVER_EOS_PRECEDENCE")
        # HARNESS_NORMALIZATION_RELIED: a strip-disposition cell's returned
        # text still contains the stop string -- the strip failed, so
        # correct scoring would depend on harness-side re-stripping.
        # Kept-disposition cells keep it by definition and are skipped.
        for cell in live:
            if not cell.fired or cell.disposition != "stripped":
                continue
            text = self._cell_text(backend, cell)
            if text is not None and stop_text in text:
                kinds.append("HARNESS_NORMALIZATION_RELIED")

        return [
            _OracleDivergence(kind=kind, stop_text=stop_text, attributed=self.class_semantics[kind])
            for kind in kinds
        ]

    # skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
    def _classify_selection(
        self, per_stop: list[tuple[str, _OracleCell, _OracleCell, _OracleCell]]
    ) -> _OracleDivergence | None:
        """Cross-stop MULTI_STOP_SELECTION from the documented selection rules.

        SGLang: the fired stop with the smallest firing step, ties broken
        by list order (first-listed rule). vLLM (PR #49391): smallest
        (firing step, stop_index + len(stop_str)), ties by list order --
        where stop_index + len(stop_str) is the length of the keep cell's
        returned text (the firing step's decoded text cut at the match end).
        """
        sglang_best: tuple[tuple[int, int], tuple[str, int]] | None = None
        vllm_best: tuple[tuple[int, int, int], tuple[str, int]] | None = None
        for list_index, (stop_text, sglang, strip, keep) in enumerate(per_stop):
            if sglang.available and sglang.fired and sglang.firing_step is not None:
                key = (sglang.firing_step, list_index)
                if sglang_best is None or key < sglang_best[0]:
                    sglang_best = (key, (stop_text, sglang.firing_step))
            if (
                strip.available
                and strip.fired
                and strip.firing_step is not None
                and keep.available
                and keep.fired
                and keep.returned_text is not None
            ):
                vkey = (strip.firing_step, len(keep.returned_text), list_index)
                if vllm_best is None or vkey < vllm_best[0]:
                    vllm_best = (vkey, (stop_text, strip.firing_step))
        sglang_winner = sglang_best[1] if sglang_best is not None else None
        vllm_winner = vllm_best[1] if vllm_best is not None else None
        if sglang_winner == vllm_winner:
            return None
        if sglang_winner is not None:
            winner_stop: str | None = sglang_winner[0]
        elif vllm_winner is not None:
            winner_stop = vllm_winner[0]
        else:  # unreachable: the winners differ, so one of them fired
            winner_stop = None
        return _OracleDivergence(
            kind="MULTI_STOP_SELECTION",
            stop_text=winner_stop,
            attributed=self.class_semantics["MULTI_STOP_SELECTION"],
        )

    def run(
        self,
        *,
        prompt_ids: tuple[int, ...],
        output_ids: tuple[int, ...],
        stop_texts: tuple[str, ...] | list[str],
        eos_id: int | None = None,
        min_tokens: int = 0,
        backend: Any,
    ) -> dict[str, Any]:
        """The oracle's per-observation pipeline: validate, emulate, classify."""
        prompt_ids = self._checked_ids(prompt_ids, "prompt_ids", MAX_PROMPT_IDS)
        output_ids = self._checked_ids(output_ids, "output_ids", MAX_OUTPUT_IDS)
        stops = self._checked_stop_texts(stop_texts)
        if eos_id is not None and (type(eos_id) is not int or eos_id < 0):
            raise ValueError("invalid_eos_id")
        if type(min_tokens) is not int or min_tokens < 0:
            raise ValueError("invalid_min_tokens")
        if not callable(getattr(backend, "decode", None)):
            raise ValueError("invalid_backend")

        cell_records: dict[str, dict[str, _OracleCell]] = {}
        selection_inputs: list[tuple[str, _OracleCell, _OracleCell, _OracleCell]] = []
        divergences: list[_OracleDivergence] = []
        for stop_text in stops:
            cells = self._build_cells(
                prompt_ids=prompt_ids,
                output_ids=output_ids,
                stop_text=stop_text,
                eos_id=eos_id,
                min_tokens=min_tokens,
                backend=backend,
            )
            cell_records[stop_text] = cells
            divergences.extend(
                self._classify_stop(
                    stop_text=stop_text,
                    cells=cells,
                    backend=backend,
                    output_ids=output_ids,
                    min_tokens=min_tokens,
                )
            )
            selection_inputs.append(
                (
                    stop_text,
                    cells[SGLANG_STOP],
                    cells[VLLM_HOLDBACK_STRIP],
                    cells[VLLM_HOLDBACK_KEEP],
                )
            )
        selection = self._classify_selection(selection_inputs)
        if selection is not None:
            divergences.append(selection)
        return {
            "stop_texts": list(stops),
            "cells": cell_records,
            "divergences": divergences,
            "verdict": "diverge" if divergences else "agree",
        }

    # skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
    def section_data(
        self,
        *,
        observations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        stop_texts: tuple[str, ...] | list[str],
        eos_id: int | None,
        min_tokens: int,
        backend: Any,
    ) -> dict[str, Any] | None:
        """The oracle's independent aggregation of the section data."""
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
        stops = self._checked_stop_texts(stop_texts)
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
            prompt_ids = self._observation_ids(observation, "input_ids", MAX_PROMPT_IDS)
            output_ids = self._observation_ids(observation, "output_ids", MAX_OUTPUT_IDS)
            if prompt_ids is None or output_ids is None:
                continue
            name = observation.get("fixture")
            fixture = name if type(name) is str and name else f"fixture_{index}"
            outcome = self.run(
                prompt_ids=prompt_ids,
                output_ids=output_ids,
                stop_texts=stops,
                eos_id=eos_id,
                min_tokens=min_tokens,
                backend=backend,
            )
            findings: list[dict[str, Any]] = []
            for divergence in outcome["divergences"]:
                kind = divergence.kind
                findings.append(
                    {
                        "class": kind,
                        "stop_text": divergence.stop_text,
                        "severity": self.severity[kind],
                    }
                )
                row = matrix.setdefault(kind, {})
                for semantic in self.class_semantics[kind]:
                    row[semantic] = row.get(semantic, 0) + 1
            per_fixture.append(
                {"fixture": fixture, "verdict": outcome["verdict"], "divergences": findings}
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
            "severities": {kind: self.severity[kind] for kind in matrix},
            "stop_texts": list(stops),
            "eos_id": eos_id,
            "min_tokens": min_tokens,
            "observations_truncated": truncated,
            "observations_total": observations_total,
            "observations_processed": len(per_fixture),
        }


# ---------------------------------------------------------------------------
# Hand-derived adversarial cases.
# ---------------------------------------------------------------------------
# (name, output token texts, stop texts, eos_id, min_tokens,
#  divergence classes expected to fire)
_HAND_CASES: tuple[tuple[str, list[str], list[str], int | None, int, tuple[str, ...]], ...] = (
    ("agree_all", ["a", "b", "c"], ["zzz"], None, 0, ()),
    ("agree_eos_absent", ["a", "b", "c"], ["zzz"], 99, 0, ()),
    ("kept_vs_stripped", ["a", "STOP", "b"], ["STOP"], None, 0, ("KEPT_VS_STRIPPED",)),
    (
        "firing_step_differs",
        ["a", "STOP", "b"],
        ["STOP"],
        None,
        2,
        ("FIRING_STEP_DIFFERS",),
    ),
    (
        "min_tokens_gated",
        ["a", "STOP"],
        ["STOP"],
        None,
        5,
        ("MIN_TOKENS_GATED", "NO_MIN_TOKENS_GATE"),
    ),
    ("eos_precedence", ["a", "E", "STOP"], ["STOP"], 2, 0, ("EOS_PRECEDENCE",)),
    (
        "str_over_eos_precedence",
        ["a", "STOP"],
        ["STOP"],
        2,
        0,
        ("STR_OVER_EOS_PRECEDENCE", "EOS_PRECEDENCE"),
    ),
    (
        "harness_normalization",
        ["ab", "Xab"],
        ["ab"],
        None,
        1,
        ("HARNESS_NORMALIZATION_RELIED",),
    ),
    (
        "multi_stop_selection",
        ["abc"],
        ["bc", "ab"],
        None,
        0,
        ("MULTI_STOP_SELECTION",),
    ),
)

# Divergence class -> hand case that fires it by construction.
_CLASS_CASES = {
    "KEPT_VS_STRIPPED": "kept_vs_stripped",
    "FIRING_STEP_DIFFERS": "firing_step_differs",
    "MIN_TOKENS_GATED": "min_tokens_gated",
    "NO_MIN_TOKENS_GATE": "min_tokens_gated",
    "EOS_PRECEDENCE": "eos_precedence",
    "STR_OVER_EOS_PRECEDENCE": "str_over_eos_precedence",
    "HARNESS_NORMALIZATION_RELIED": "harness_normalization",
    "MULTI_STOP_SELECTION": "multi_stop_selection",
}


class _StrictEosOracle(_DivergenceOracle):
    """Sensitivity mutant: EOS_PRECEDENCE needs a *strictly* earlier EOS step.

    The documented rule is "at/before", so this mutant must be caught by
    the differential comparison on a same-step tie case.
    """

    @staticmethod
    def _eos_precedence_fires(eos: _OracleCell, live: list[_OracleCell]) -> bool:
        """Eos precedence fires."""
        if not (eos.available and eos.fired and eos.firing_step is not None):
            return False
        stop_steps = [
            cell.firing_step for cell in live if cell.fired and cell.firing_step is not None
        ]
        return bool(stop_steps) and eos.firing_step < min(stop_steps)


class DivergenceClassificationTests(unittest.TestCase):
    """``diff_backends`` divergence classification vs the independent oracle."""

    def setUp(self) -> None:
        """Set up the test fixture."""
        self.oracle = _DivergenceOracle()

    @staticmethod
    def _backend(
        texts: list[str], prompt_texts: list[str] | tuple[str, ...] = ()
    ) -> tuple[_ConcatBackend, tuple[int, ...], tuple[int, ...]]:
        """Backend."""
        mapping = {index + 1: text for index, text in enumerate(texts)}
        prompt_ids = tuple(1001 + index for index in range(len(prompt_texts)))
        for pid, text in zip(prompt_ids, prompt_texts, strict=True):
            mapping[pid] = text
        return _ConcatBackend(mapping), tuple(range(1, len(texts) + 1)), prompt_ids

    def _run_hand_case(self, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run hand case."""
        for case in _HAND_CASES:
            if case[0] == name:
                _, texts, stops, eos_id, min_tokens, _ = case
                break
        else:  # pragma: no cover -- test data bug, not product code
            raise AssertionError(f"unknown hand case {name!r}")
        backend, output_ids, prompt_ids = self._backend(texts)
        kwargs: dict[str, Any] = {
            "prompt_ids": prompt_ids,
            "output_ids": output_ids,
            "stop_texts": stops,
            "eos_id": eos_id,
            "min_tokens": min_tokens,
            "backend": backend,
        }
        return diff_backends(**kwargs), self.oracle.run(**kwargs)

    def assert_matches_oracle(
        self, impl: dict[str, Any], oracle_result: dict[str, Any], label: str
    ) -> None:
        """The classification contract: (class, stop_text) multiset, cells,
        echoed stop texts and verdict all agree; details are non-empty.
        """
        impl_pairs = sorted((d["class"], d["stop_text"]) for d in impl["divergences"])
        oracle_pairs = sorted((d.kind, d.stop_text) for d in oracle_result["divergences"])
        self.assertEqual(impl_pairs, oracle_pairs, label)
        self.assertEqual(impl["verdict"], oracle_result["verdict"], label)
        self.assertEqual(impl["stop_texts"], oracle_result["stop_texts"], label)
        for finding in impl["divergences"]:
            self.assertIsInstance(finding["detail"], str, label)
            self.assertTrue(finding["detail"], label)
        # Cell-level agreement: the oracle's independent cell computation
        # matches the implementation's recorded cells.
        impl_cells = {record["stop_text"]: record["semantics"] for record in impl["cells"]}
        self.assertEqual(set(impl_cells), set(oracle_result["cells"]), label)
        for stop_text, ocells in oracle_result["cells"].items():
            for semantic, ocell in ocells.items():
                icell = impl_cells[stop_text][semantic]
                where = (label, stop_text, semantic)
                self.assertEqual(icell["available"], ocell.available, where)
                self.assertEqual(icell["fired"], ocell.fired, where)
                self.assertEqual(icell["firing_step"], ocell.firing_step, where)
                self.assertEqual(tuple(icell["truncated_ids"]), ocell.truncated_ids, where)
                self.assertEqual(icell["returned_text"], ocell.returned_text, where)
                self.assertEqual(icell["disposition"], ocell.disposition, where)

    def test_hand_cases_match_oracle(self) -> None:
        """Test hand cases match oracle."""
        for name, _, _, _, _, expected in _HAND_CASES:
            with self.subTest(name=name):
                impl, oracle_result = self._run_hand_case(name)
                self.assert_matches_oracle(impl, oracle_result, name)
                kinds = {d["class"] for d in impl["divergences"]}
                for kind in expected:
                    self.assertIn(kind, kinds, f"{name}: {kind} must fire by construction")
                if not expected:
                    self.assertEqual(impl["divergences"], [], name)
                    self.assertEqual(impl["verdict"], "agree", name)

    def test_each_class_attribution_matches_class_semantics(self) -> None:
        """Test each class attribution matches class semantics."""
        # Every divergence class fires at least once by construction; the
        # oracle's attributed semantics for each firing must equal the
        # module's _CLASS_SEMANTICS entry (the class x semantic matrix pin).
        self.assertEqual(set(_CLASS_CASES), set(_CLASS_SEMANTICS))
        for kind, case_name in _CLASS_CASES.items():
            with self.subTest(kind=kind):
                impl, oracle_result = self._run_hand_case(case_name)
                self.assertIn(kind, {d["class"] for d in impl["divergences"]}, case_name)
                fired = [d for d in oracle_result["divergences"] if d.kind == kind]
                self.assertTrue(fired, f"{kind} did not fire in {case_name}")
                for divergence in fired:
                    self.assertEqual(
                        divergence.attributed,
                        _CLASS_SEMANTICS[kind],
                        f"{kind} attribution",
                    )

    def test_severity_pin_matches(self) -> None:
        """Test severity pin matches."""
        # The oracle's docstring-derived severity pin equals the module's
        # DIVERGENCE_SEVERITY for every class.
        self.assertEqual(self.oracle.SEVERITY, DIVERGENCE_SEVERITY)

    def test_fuzz_classification_matches_oracle(self) -> None:
        """Test fuzz classification matches oracle."""
        rng = random.Random(20261017)  # noqa: S311 -- seeded deterministic fuzz, not crypto
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
        ]
        for trial in range(400):
            n_tokens = rng.randint(1, 6)
            texts = [rng.choice(token_pool) for _ in range(n_tokens)]
            prompt_texts: list[str] = (
                [rng.choice(token_pool) for _ in range(rng.randint(0, 2))]
                if rng.random() < 0.3
                else []
            )
            output_ids = tuple(range(1, n_tokens + 1))
            if rng.random() < 0.08:
                # Unknown token id: decode raises, the cells degrade to
                # unavailable and are excluded from the divergence checks.
                output_ids = output_ids + (777,)
            stops = rng.sample(stop_pool, rng.randint(1, 3))
            if rng.random() < 0.15:
                stops.append(rng.choice(stops))  # duplicate exercises dedup
            eos_roll = rng.random()
            if eos_roll < 0.35:
                eos_id = None
            elif eos_roll < 0.7:
                eos_id = rng.randint(1, n_tokens)  # present in the output
            else:
                eos_id = 900 + rng.randint(0, 5)  # absent
            min_tokens = rng.randint(0, 3)
            backend, _, prompt_ids = self._backend(texts, prompt_texts)
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
                oracle_result = self.oracle.run(**kwargs)
                self.assert_matches_oracle(impl, oracle_result, f"fuzz-{trial}")

    def test_invalid_inputs_raise_identical_codes(self) -> None:
        """Test invalid inputs raise identical codes."""
        backend, _, _ = self._backend(["a"])
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
            ("prompt_ids_bool", {**base, "prompt_ids": (True,)}, "invalid_prompt_ids"),
            ("prompt_ids_huge", {**base, "prompt_ids": (2**31,)}, "invalid_prompt_ids"),
            (
                "prompt_ids_too_long",
                {**base, "prompt_ids": tuple(range(MAX_PROMPT_IDS + 1))},
                "prompt_ids_too_long",
            ),
            (
                "output_ids_too_long",
                {**base, "output_ids": tuple(range(MAX_OUTPUT_IDS + 1))},
                "output_ids_too_long",
            ),
            ("stop_texts_empty", {**base, "stop_texts": ()}, "empty_stop_texts"),
            ("stop_texts_str", {**base, "stop_texts": "s"}, "invalid_stop_texts"),
            ("stop_texts_empty_str", {**base, "stop_texts": ("",)}, "invalid_stop_texts"),
            (
                "stop_texts_too_long",
                {**base, "stop_texts": ("x" * (MAX_STOP_TEXT_LEN + 1),)},
                "stop_text_too_long",
            ),
            (
                "stop_texts_too_many",
                {**base, "stop_texts": tuple(f"s{i}" for i in range(MAX_STOP_TEXTS + 1))},
                "too_many_stop_texts",
            ),
            ("eos_id_negative", {**base, "eos_id": -1}, "invalid_eos_id"),
            ("eos_id_bool", {**base, "eos_id": True}, "invalid_eos_id"),
            ("min_tokens_negative", {**base, "min_tokens": -1}, "invalid_min_tokens"),
            ("min_tokens_bool", {**base, "min_tokens": True}, "invalid_min_tokens"),
            ("backend_no_decode", {**base, "backend": object()}, "invalid_backend"),
        ]
        for name, kwargs, code in cases:
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as impl_cm:
                    diff_backends(**kwargs)
                with self.assertRaises(ValueError) as oracle_cm:
                    self.oracle.run(**kwargs)
                self.assertEqual(str(impl_cm.exception), code, name)
                self.assertEqual(str(oracle_cm.exception), code, name)

    def test_mutated_attribution_is_caught(self) -> None:
        """Test mutated attribution is caught."""
        # Sensitivity: remove one semantic from a class's attribution in a
        # COPY of the oracle -- the check against _CLASS_SEMANTICS must fail.
        mutated = _DivergenceOracle(
            class_semantics={
                **_DivergenceOracle.CLASS_SEMANTICS,
                "KEPT_VS_STRIPPED": (HF_STOP_STRING,),
            }
        )
        backend, output_ids, prompt_ids = self._backend(["a", "STOP", "b"])
        result = mutated.run(
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            stop_texts=("STOP",),
            backend=backend,
        )
        fired = [d for d in result["divergences"] if d.kind == "KEPT_VS_STRIPPED"]
        self.assertTrue(fired)
        with self.assertRaises(AssertionError):
            for divergence in fired:
                self.assertEqual(divergence.attributed, _CLASS_SEMANTICS[divergence.kind])

    def test_swapped_attribution_is_caught(self) -> None:
        """Test swapped attribution is caught."""
        # Sensitivity: swap two classes' attributions in a copy -- the
        # check against _CLASS_SEMANTICS must fail.
        swapped = dict(_DivergenceOracle.CLASS_SEMANTICS)
        swapped["KEPT_VS_STRIPPED"], swapped["HARNESS_NORMALIZATION_RELIED"] = (
            swapped["HARNESS_NORMALIZATION_RELIED"],
            swapped["KEPT_VS_STRIPPED"],
        )
        oracle = _DivergenceOracle(class_semantics=swapped)
        backend, output_ids, prompt_ids = self._backend(["ab", "Xab"])
        result = oracle.run(
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            stop_texts=("ab",),
            min_tokens=1,
            backend=backend,
        )
        fired = [d for d in result["divergences"] if d.kind == "HARNESS_NORMALIZATION_RELIED"]
        self.assertTrue(fired)
        with self.assertRaises(AssertionError):
            for divergence in fired:
                self.assertEqual(divergence.attributed, _CLASS_SEMANTICS[divergence.kind])

    def test_mutated_predicate_is_caught(self) -> None:
        """Test mutated predicate is caught."""
        # Sensitivity: a classifier mutant that requires a *strictly*
        # earlier EOS step misses EOS_PRECEDENCE on the same-step tie --
        # the differential comparison against diff_backends must fail.
        backend, output_ids, prompt_ids = self._backend(["a", "STOP"])
        kwargs: dict[str, Any] = {
            "prompt_ids": prompt_ids,
            "output_ids": output_ids,
            "stop_texts": ("STOP",),
            "eos_id": 2,
            "backend": backend,
        }
        impl = diff_backends(**kwargs)
        mutant_result = _StrictEosOracle().run(**kwargs)
        self.assertIn("EOS_PRECEDENCE", {d["class"] for d in impl["divergences"]}, "tie must fire")
        with self.assertRaises(AssertionError):
            self.assert_matches_oracle(impl, mutant_result, "strict-eos-mutant")


class DivergenceSectionTests(unittest.TestCase):
    """``divergence_section_data`` aggregation vs the independent oracle."""

    def setUp(self) -> None:
        """Set up the test fixture."""
        self.oracle = _DivergenceOracle()

    def assert_section_matches(
        self, impl: dict[str, Any] | None, oracle_result: dict[str, Any] | None, label: str
    ) -> None:
        """Assert section matches."""
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
        observations: list[Any] | tuple[Any, ...], backend: _ConcatBackend
    ) -> dict[str, Any]:
        """Section kwargs."""
        return {
            "observations": observations,
            "stop_texts": ("STOP",),
            "eos_id": None,
            "min_tokens": 0,
            "backend": backend,
        }

    def test_section_agree_and_diverge(self) -> None:
        """Test section agree and diverge."""
        backend = _ConcatBackend({1: "a", 2: "STOP", 3: "b"})
        observations = [
            {"fixture": "agree_fx", "input_ids": [], "output_ids": [1]},
            {"fixture": "diverge_fx", "input_ids": [], "output_ids": [1, 2, 3]},
        ]
        kwargs = self._section_kwargs(observations, backend)
        impl = divergence_section_data(**kwargs)
        oracle_result = self.oracle.section_data(**kwargs)
        self.assert_section_matches(impl, oracle_result, "agree-and-diverge")
        assert impl is not None
        # Exact aggregation pins for the one-divergence section.
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
        self.assertEqual(impl["per_fixture"][0]["fixture"], "agree_fx")
        self.assertEqual(impl["per_fixture"][0]["verdict"], "agree")
        self.assertEqual(impl["per_fixture"][0]["divergences"], [])
        self.assertEqual(impl["per_fixture"][1]["fixture"], "diverge_fx")
        self.assertEqual(impl["per_fixture"][1]["verdict"], "diverge")
        findings = impl["per_fixture"][1]["divergences"]
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["class"], "KEPT_VS_STRIPPED")
        self.assertEqual(findings[0]["stop_text"], "STOP")
        self.assertEqual(findings[0]["severity"], "high")
        self.assertFalse(impl["observations_truncated"])
        self.assertEqual(impl["observations_total"], 2)
        self.assertEqual(impl["observations_processed"], 2)

    def test_section_default_fixture_names(self) -> None:
        """Test section default fixture names."""
        backend = _ConcatBackend({1: "a"})
        observations = [
            {"input_ids": [], "output_ids": [1]},  # no fixture name
            {"fixture": 42, "input_ids": [], "output_ids": [1]},  # non-str name
            {"fixture": "", "input_ids": [], "output_ids": [1]},  # empty name
        ]
        kwargs = self._section_kwargs(observations, backend)
        impl = divergence_section_data(**kwargs)
        oracle_result = self.oracle.section_data(**kwargs)
        self.assert_section_matches(impl, oracle_result, "default-names")
        assert impl is not None
        self.assertEqual(
            [entry["fixture"] for entry in impl["per_fixture"]],
            ["fixture_0", "fixture_1", "fixture_2"],
        )

    @staticmethod
    def _random_observation(rng: random.Random, n_tokens: int) -> Any:
        """Random observation."""
        roll = rng.random()
        if roll < 0.62:
            observation: dict[str, Any] = {
                "input_ids": [rng.randint(1, n_tokens) for _ in range(rng.randint(0, 3))],
                "output_ids": [rng.randint(1, n_tokens) for _ in range(rng.randint(0, 5))],
            }
            if rng.random() < 0.7:
                observation["fixture"] = (
                    f"fx_{rng.randint(0, 999)}" if rng.random() < 0.8 else rng.randint(0, 9)
                )
            if rng.random() < 0.2:
                observation["input_ids"] = tuple(observation["input_ids"])
            return observation
        if roll < 0.72:
            return rng.choice(["junk", 42, None, [1, 2]])
        if roll < 0.82:
            return {"fixture": "missing-ids"}
        if roll < 0.9:
            bad = rng.choice([["a", "b"], [1.5, 2], [True, 2], [-1, 2], [2**31, 1], None, {}])
            return {"input_ids": [], "output_ids": bad}
        # Over the output-id cap: skipped without failing.
        return {"input_ids": [], "output_ids": list(range(MAX_OUTPUT_IDS + 5))}

    def test_section_fuzz_matches_oracle(self) -> None:
        """Test section fuzz matches oracle."""
        rng = random.Random(20261018)  # noqa: S311 -- seeded deterministic fuzz, not crypto
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
        ]
        stop_pool = ["STOP", "ab", "a", "<stop>", "é", "b", "aa", "zzz", " ", "\n"]
        for trial in range(80):
            n_tokens = rng.randint(1, 5)
            texts = [rng.choice(token_pool) for _ in range(n_tokens)]
            backend = _ConcatBackend({i + 1: text for i, text in enumerate(texts)})
            observations = [
                self._random_observation(rng, n_tokens) for _ in range(rng.randint(0, 10))
            ]
            if rng.random() < 0.3:
                observations = tuple(observations)
            stops = rng.sample(stop_pool, rng.randint(1, 3))
            eos_roll = rng.random()
            eos_id = None if eos_roll < 0.5 else rng.randint(1, n_tokens + 2)
            min_tokens = rng.randint(0, 3)
            with self.subTest(trial=trial):
                kwargs: dict[str, Any] = {
                    "observations": observations,
                    "stop_texts": stops,
                    "eos_id": eos_id,
                    "min_tokens": min_tokens,
                    "backend": backend,
                }
                impl = divergence_section_data(**kwargs)
                oracle_result = self.oracle.section_data(**kwargs)
                self.assert_section_matches(impl, oracle_result, f"section-fuzz-{trial}")

    def test_section_truncation_accounting(self) -> None:
        """Test section truncation accounting."""
        backend = _ConcatBackend({1: "x", 2: "y"})
        observations = [
            {"fixture": f"obs_{i}", "input_ids": [], "output_ids": [1, 2]}
            for i in range(MAX_DIVERGENCE_OBSERVATIONS + 8)
        ]
        kwargs = self._section_kwargs(observations, backend)
        impl = divergence_section_data(**kwargs)
        oracle_result = self.oracle.section_data(**kwargs)
        self.assert_section_matches(impl, oracle_result, "truncation")
        assert impl is not None
        self.assertTrue(impl["observations_truncated"])
        self.assertEqual(impl["observations_total"], MAX_DIVERGENCE_OBSERVATIONS + 8)
        self.assertEqual(impl["observations_processed"], MAX_DIVERGENCE_OBSERVATIONS)
        self.assertEqual(len(impl["per_fixture"]), MAX_DIVERGENCE_OBSERVATIONS)
        # First N kept, in input order.
        self.assertEqual(
            [entry["fixture"] for entry in impl["per_fixture"]],
            [f"obs_{i}" for i in range(MAX_DIVERGENCE_OBSERVATIONS)],
        )
        self.assertEqual(impl["verdict"], "agree")

    def test_section_truncation_with_skips(self) -> None:
        """Test section truncation with skips."""
        # Invalid observations inside the first N are skipped by omission;
        # observations_processed counts only the emulated ones.
        backend = _ConcatBackend({1: "x", 2: "y"})
        bad_indices = {0, 100, 200, 300, 400}
        observations: list[Any] = []
        for i in range(MAX_DIVERGENCE_OBSERVATIONS + 3):
            if i in bad_indices:
                observations.append({"fixture": f"bad_{i}"})  # no ids: skipped
            else:
                observations.append({"fixture": f"obs_{i}", "input_ids": [], "output_ids": [1, 2]})
        kwargs = self._section_kwargs(observations, backend)
        impl = divergence_section_data(**kwargs)
        oracle_result = self.oracle.section_data(**kwargs)
        self.assert_section_matches(impl, oracle_result, "truncation-with-skips")
        assert impl is not None
        self.assertTrue(impl["observations_truncated"])
        self.assertEqual(impl["observations_total"], MAX_DIVERGENCE_OBSERVATIONS + 3)
        self.assertEqual(
            impl["observations_processed"], MAX_DIVERGENCE_OBSERVATIONS - len(bad_indices)
        )
        self.assertEqual(impl["observations_processed"], len(impl["per_fixture"]))
        self.assertNotIn("bad_0", [entry["fixture"] for entry in impl["per_fixture"]])

    def test_section_no_usable_observations_returns_none(self) -> None:
        """Test section no usable observations returns none."""
        backend = _ConcatBackend({1: "x"})
        for label, observations in (
            ("empty", []),
            ("all_invalid", [{"nope": 1}, "junk", {"input_ids": [], "output_ids": ["bad"]}]),
        ):
            with self.subTest(label=label):
                kwargs = self._section_kwargs(observations, backend)
                self.assertIsNone(divergence_section_data(**kwargs))
                self.assertIsNone(self.oracle.section_data(**kwargs))

    def test_section_invalid_inputs_raise_identical_codes(self) -> None:
        """Test section invalid inputs raise identical codes."""
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
            ("observations_str", {**base, "observations": "nope"}, "invalid_observations"),
            # Invalid observations type beats invalid stop texts (checked first).
            (
                "observations_beats_stops",
                {**base, "observations": "nope", "stop_texts": ()},
                "invalid_observations",
            ),
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
                    self.oracle.section_data(**kwargs)
                self.assertEqual(str(impl_cm.exception), code, name)
                self.assertEqual(str(oracle_cm.exception), code, name)


if __name__ == "__main__":
    unittest.main()
