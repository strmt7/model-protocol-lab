"""Differential testing of emulated backend stop semantics.

``diff_backends`` runs the lab's stop emulations (see
:mod:`model_protocol_lab.backends`) -- HF ``StopStringCriteria``
(``emulate_hf_stop_string``), vLLM's detokenizer holdback
(``emulate_vllm_holdback``, strip and keep variants) and an EOS-appended
truncation (``emulate_eos_stop``) -- over ONE recorded token sequence and
reports where the emulated backends disagree, classified into divergence
classes drawn from the vendors' own bug trackers (research round R2a,
``docs/RESEARCH_LOG.md``):

- ``KEPT_VS_STRIPPED`` -- HF always keeps the stop string while vLLM's
  default (``include_stop_str_in_output=False``) strips it from the
  returned text. Spec anchors: vLLM/SGLang default strip; HF keeps the
  completing token.
- ``FIRING_STEP_DIFFERS`` -- the stop-string semantics disagree on which
  output step completes the stop.
- ``MIN_TOKENS_GATED`` -- a stop that fires with ``min_tokens=0`` does not
  fire under the given ``min_tokens`` (vLLM PR #22014,
  https://github.com/vllm-project/vllm/pull/22014).
- ``EOS_PRECEDENCE`` -- EOS fires at/before the stop-string firing step
  (the SGLang #28802 class: stop-string vs EOS precedence in one step,
  https://github.com/sgl-project/sglang/issues/28802; ports vLLM #23938 --
  both engines shipped the same bug independently).
- ``HARNESS_NORMALIZATION_RELIED`` -- a ``"stripped"``-disposition cell's
  returned text still contains the stop string: the strip failed (the
  vLLM #46113 / SGLang #28802 anomaly class), so correct scoring would
  depend on harness-side re-stripping (cf. lm-eval's
  ``postprocess_generated_text`` belt-and-suspenders re-strip; the lm-eval
  #3587 class,
  https://github.com/EleutherAI/lm-evaluation-harness/issues/3587, of
  stop-handling bugs destroying benchmark verdicts). Cells whose
  disposition is ``"kept"`` keep the stop string by definition and are not
  flagged.

Scope limits (deliberate, documented -- no semantics are invented here;
every cell is one of the existing emulations in
:mod:`model_protocol_lab.backends`, run with fixed inputs):

- One stop string per run. Multi-stop earliest-completion selection
  (vLLM #49391, https://github.com/vllm-project/vllm/pull/49391) is NOT
  emulated; callers (``mpl diff-backends``) run each stop text separately.
- Streaming vs non-streaming divergence (vLLM #47137,
  https://github.com/vllm-project/vllm/issues/47137) is not emulated: the
  lab's vLLM port is the non-streaming final-truncation path (the
  ``stop_buffer_length`` holdback only withholds streamed deltas).
- Match position leftmost/rightmost (vLLM #46113,
  https://github.com/vllm-project/vllm/issues/46113) is not differentially
  tested: the port follows the Python ``find`` path, not the Rust
  frontend's ``rposition``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .backends import (
    _hf_byte_token_mapper,
    emulate_eos_stop,
    emulate_hf_stop_string,
    emulate_vllm_holdback,
)

DivergenceClass = Literal[
    "KEPT_VS_STRIPPED",
    "FIRING_STEP_DIFFERS",
    "MIN_TOKENS_GATED",
    "EOS_PRECEDENCE",
    "HARNESS_NORMALIZATION_RELIED",
]

#: Semantic names used as keys in each stop record's ``semantics`` mapping.
HF_STOP_STRING = "hf_stop_string"
VLLM_HOLDBACK_STRIP = "vllm_holdback_strip"
VLLM_HOLDBACK_KEEP = "vllm_holdback_keep"
EOS_APPENDED = "eos_appended"

_STOP_SEMANTICS = (HF_STOP_STRING, VLLM_HOLDBACK_STRIP, VLLM_HOLDBACK_KEEP)

_FIRED_AT_STEP = re.compile(r"^fired_at_step_(\d+)$")


def _firing_step(detail: str) -> int | None:
    """Parse the 1-based firing step from an emulation ``detail`` string.

    Only ``fired_at_step_N`` carries a step; every other fixed detail
    (``no_stop_within_cap``, ``gated_by_min_tokens``, ``*_unavailable``,
    ``invalid_*``) parses to None.
    """
    match = _FIRED_AT_STEP.fullmatch(detail)
    return int(match.group(1)) if match is not None else None


@dataclass(frozen=True)
class CellResult:
    """One emulated semantic's outcome for one stop text.

    ``available=False`` means the emulation could not run (see
    ``StopOutcome``); such cells are recorded but excluded from the
    divergence checks. ``disposition`` is the stop-text disposition:
    ``"kept"`` when the semantic returns the stop string in its text (HF
    always keeps; vLLM with ``include_stop_str_in_output=True``),
    ``"stripped"`` for vLLM's default, and None for ``eos_appended`` (no
    stop text involved). ``returned_text`` is set only by the vLLM cells
    (see ``StopOutcome.returned_text``); HF/EOS text is the decode of
    ``truncated_ids`` and is computed by the caller when needed.
    """

    semantic: str
    available: bool
    fired: bool
    firing_step: int | None
    truncated_ids: tuple[int, ...]
    returned_text: str | None
    disposition: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "fired": self.fired,
            "firing_step": self.firing_step,
            "truncated_ids": list(self.truncated_ids),
            "returned_text": self.returned_text,
            "disposition": self.disposition,
        }


@dataclass(frozen=True)
class Divergence:
    """One classified disagreement between the emulated semantics."""

    kind: DivergenceClass
    stop_text: str | None
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"class": self.kind, "stop_text": self.stop_text, "detail": self.detail}


def _checked_ids(value: object, name: str) -> tuple[int, ...]:
    """Strict token-id tuple validation (mirrors the emulations' contract).

    Requires an actual tuple of plain ints (bool excluded), each a
    non-negative id below 2**31, matching the CLI's ``parse_boundary``
    bound. The total sequence is capped (4x the lab's max context window)
    because the emulations do per-step decodes; unbounded input would be
    an unadvertised DoS surface. Raises ValueError with a fixed code
    otherwise.
    """
    if type(value) is not tuple or any(
        type(token) is not int or not 0 <= token < 2**31 for token in value
    ):
        raise ValueError(f"invalid_{name}")
    if len(value) > 131072:
        raise ValueError(f"{name}_too_long")
    return value


def _checked_stop_texts(value: object) -> tuple[str, ...]:
    """Strict stop-text validation: tuple/list of non-empty strings.

    Duplicates are removed (order-preserving): running the same stop
    twice would only duplicate identical cells.
    """
    if type(value) is tuple:
        items: tuple[object, ...] = value
    elif type(value) is list:
        items = tuple(value)
    else:
        raise ValueError("invalid_stop_texts")
    strings: list[str] = []
    for stop in items:
        if type(stop) is not str or not stop:
            raise ValueError("invalid_stop_texts")
        strings.append(stop)
    deduped: list[str] = []
    for stop in strings:
        if stop not in deduped:
            deduped.append(stop)
    if not deduped:
        raise ValueError("empty_stop_texts")
    return tuple(deduped)


def _effective_text(backend: Any, cell: CellResult) -> str | None:
    """The text a semantic would return for a cell.

    vLLM cells expose it directly (``returned_text``); every other cell's
    text is the decode of ``truncated_ids``. None when the text cannot be
    established (vLLM did not fire and the decode failed).
    """
    if cell.returned_text is not None:
        return cell.returned_text
    decode = getattr(backend, "decode", None)
    if not callable(decode):
        return None
    try:
        text = decode(cell.truncated_ids)
    except Exception:  # noqa: BLE001 -- text checks degrade, never fail the diff
        return None
    return text if type(text) is str else None


def _cell(
    semantic: str,
    outcome: Any,
    disposition: str | None,
) -> CellResult:
    return CellResult(
        semantic=semantic,
        available=bool(outcome.available),
        fired=bool(outcome.fired),
        firing_step=_firing_step(outcome.detail),
        truncated_ids=tuple(outcome.truncated_ids),
        returned_text=outcome.returned_text,
        disposition=disposition,
    )


def diff_backends(
    *,
    prompt_ids: tuple[int, ...],
    output_ids: tuple[int, ...],
    stop_texts: tuple[str, ...] | list[str],
    eos_id: int | None = None,
    min_tokens: int = 0,
    backend: Any,
) -> dict[str, Any]:
    """Differential-test emulated backend stop semantics on one token sequence.

    Runs ``emulate_hf_stop_string`` (per stop text), ``emulate_vllm_holdback``
    in strip and keep variants (per stop text, with ``min_tokens``), and
    ``emulate_eos_stop`` (once per stop text; it does not depend on the stop
    text), then classifies disagreements into the divergence classes
    documented in the module docstring. Returns a JSON-serializable dict
    with ``stop_texts``, ``eos_id``, ``min_tokens``, per-stop ``cells``,
    ``divergences`` and ``verdict`` (``"agree"`` iff no divergences).

    ``backend`` needs only ``decode`` (plus the optional
    ``hf_stop_token_bytes`` byte-space mapping, used by the HF emulation
    exactly as the lab's stop matrix does). All inputs are strictly
    validated; ValueError with a fixed code on any violation. Cells whose
    emulation reports ``available=False`` are recorded but excluded from
    the divergence checks.
    """
    prompt_ids = _checked_ids(prompt_ids, "prompt_ids")
    output_ids = _checked_ids(output_ids, "output_ids")
    stops = _checked_stop_texts(stop_texts)
    if eos_id is not None and (type(eos_id) is not int or eos_id < 0):
        raise ValueError("invalid_eos_id")
    if type(min_tokens) is not int or min_tokens < 0:
        raise ValueError("invalid_min_tokens")
    if not callable(getattr(backend, "decode", None)):
        raise ValueError("invalid_backend")
    token_bytes = getattr(backend, "hf_stop_token_bytes", None)

    records: list[dict[str, Any]] = []
    divergences: list[Divergence] = []
    for stop_text in stops:
        hf = _cell(
            HF_STOP_STRING,
            emulate_hf_stop_string(
                backend, prompt_ids, output_ids, stop_text, token_bytes=token_bytes
            ),
            "kept",
        )
        strip = _cell(
            VLLM_HOLDBACK_STRIP,
            emulate_vllm_holdback(
                backend,
                output_ids,
                stop_text,
                include_stop_str_in_output=False,
                min_tokens=min_tokens,
            ),
            "stripped",
        )
        keep = _cell(
            VLLM_HOLDBACK_KEEP,
            emulate_vllm_holdback(
                backend,
                output_ids,
                stop_text,
                include_stop_str_in_output=True,
                min_tokens=min_tokens,
            ),
            "kept",
        )
        eos = _cell(EOS_APPENDED, emulate_eos_stop(output_ids, eos_id), None)
        semantics = {
            HF_STOP_STRING: hf,
            VLLM_HOLDBACK_STRIP: strip,
            VLLM_HOLDBACK_KEEP: keep,
            EOS_APPENDED: eos,
        }
        records.append(
            {
                "stop_text": stop_text,
                "semantics": {name: cell.as_dict() for name, cell in semantics.items()},
            }
        )

        stop_cells = [cell for name, cell in semantics.items() if name in _STOP_SEMANTICS]

        # FIRING_STEP_DIFFERS: the stop-string semantics disagree on the
        # completing output step (None = did not fire).
        live = [cell for cell in stop_cells if cell.available]
        steps = {cell.firing_step for cell in live}
        if len(steps) > 1:
            parts = []
            for cell in live:
                step = "did not fire" if cell.firing_step is None else f"step {cell.firing_step}"
                parts.append(f"{cell.semantic} {step}")
            divergences.append(
                Divergence(
                    "FIRING_STEP_DIFFERS",
                    stop_text,
                    "firing steps disagree: " + "; ".join(parts),
                )
            )

        # KEPT_VS_STRIPPED: vLLM-strip's returned text differs from HF-kept.
        if hf.available and hf.fired and strip.available and strip.fired:
            kept_text = _effective_text(backend, hf)
            if kept_text is not None and kept_text != strip.returned_text:
                divergences.append(
                    Divergence(
                        "KEPT_VS_STRIPPED",
                        stop_text,
                        f"vllm_holdback_strip returned {strip.returned_text!r} "
                        f"while hf_stop_string kept {kept_text!r}",
                    )
                )

        # MIN_TOKENS_GATED: the stop fires with min_tokens=0 but the given
        # min_tokens gates it (diagnostic re-run with the gate disabled).
        if min_tokens > 0 and strip.available and not strip.fired:
            ungated = emulate_vllm_holdback(backend, output_ids, stop_text, min_tokens=0)
            ungated_step = _firing_step(ungated.detail)
            if ungated.available and ungated.fired and ungated_step is not None:
                divergences.append(
                    Divergence(
                        "MIN_TOKENS_GATED",
                        stop_text,
                        f"stop {stop_text!r} fires at step {ungated_step} with "
                        f"min_tokens=0 but min_tokens={min_tokens} gates it "
                        "(vLLM PR #22014)",
                    )
                )

        # EOS_PRECEDENCE: EOS fires at/before the stop-string firing step
        # (the SGLang #28802 class).
        if eos.available and eos.fired and eos.firing_step is not None:
            stop_steps = [
                cell.firing_step for cell in live if cell.fired and cell.firing_step is not None
            ]
            if stop_steps and eos.firing_step <= min(stop_steps):
                divergences.append(
                    Divergence(
                        "EOS_PRECEDENCE",
                        stop_text,
                        f"eos_appended fired at step {eos.firing_step}, at/before the "
                        f"stop-string firing step {min(stop_steps)} for stop "
                        f"{stop_text!r} (SGLang #28802 class)",
                    )
                )

        # HARNESS_NORMALIZATION_RELIED: a "stripped"-disposition cell's
        # returned text still contains the stop string -- the strip failed
        # (vLLM #46113 / SGLang #28802 anomaly class), so correct scoring
        # would depend on harness-side re-stripping. "kept"-disposition
        # cells keep the stop string by definition and are not flagged.
        for cell in live:
            if not cell.fired or cell.disposition != "stripped":
                continue
            text = _effective_text(backend, cell)
            if text is not None and stop_text in text:
                divergences.append(
                    Divergence(
                        "HARNESS_NORMALIZATION_RELIED",
                        stop_text,
                        f"{cell.semantic} is strip-disposition but its returned "
                        f"text still contains stop text {stop_text!r}; correct "
                        "scoring depends on harness-side re-stripping (cf. "
                        "lm-eval postprocess_generated_text)",
                    )
                )

    return {
        "prompt_tokens": len(prompt_ids),
        "output_tokens": len(output_ids),
        "min_tokens": min_tokens,
        "eos_id": eos_id,
        "stop_texts": list(stops),
        "cells": records,
        "divergences": [finding.as_dict() for finding in divergences],
        "verdict": "diverge" if divergences else "agree",
    }


#: Severity of each divergence class, pinned for the report's severity
#: badges (infer-check/Csmith style). "high" classes silently change the
#: returned text or the stopping precedence (score corruption);
#: "medium" classes change *when* stopping happens; "low" flags cases
#: where correct scoring would depend on harness-side normalization.
DIVERGENCE_SEVERITY: dict[DivergenceClass, str] = {
    "KEPT_VS_STRIPPED": "high",
    "EOS_PRECEDENCE": "high",
    "FIRING_STEP_DIFFERS": "medium",
    "MIN_TOKENS_GATED": "medium",
    "HARNESS_NORMALIZATION_RELIED": "low",
}

#: Semantics each divergence class is attributed to in the class x
#: semantic count matrix: the emulated semantics whose disagreement the
#: class measures.
_CLASS_SEMANTICS: dict[DivergenceClass, tuple[str, ...]] = {
    "KEPT_VS_STRIPPED": (HF_STOP_STRING, VLLM_HOLDBACK_STRIP),
    "FIRING_STEP_DIFFERS": (HF_STOP_STRING, VLLM_HOLDBACK_STRIP, VLLM_HOLDBACK_KEEP),
    "MIN_TOKENS_GATED": (VLLM_HOLDBACK_STRIP,),
    "EOS_PRECEDENCE": (
        EOS_APPENDED,
        HF_STOP_STRING,
        VLLM_HOLDBACK_STRIP,
        VLLM_HOLDBACK_KEEP,
    ),
    "HARNESS_NORMALIZATION_RELIED": (VLLM_HOLDBACK_STRIP,),
}


def _observation_ids(observation: dict[str, Any], key: str) -> tuple[int, ...] | None:
    """Extract a validated token-id tuple from a recorded observation.

    Recorded observations come back JSON-round-tripped, so ids may be
    lists; anything that is not a list/tuple of plain ints (or exceeds the
    emulation's length bound) is unusable and returns None -- the caller
    skips such observations instead of failing the diagnosis.
    """
    raw: object = observation.get(key)
    if type(raw) is list:
        raw = tuple(raw)
    try:
        return _checked_ids(raw, key)
    except ValueError:
        return None


def divergence_section_data(
    *,
    observations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    stop_texts: tuple[str, ...] | list[str],
    eos_id: int | None,
    min_tokens: int,
    backend: Any,
) -> dict[str, Any] | None:
    """Build the report's "Backend divergence" section data.

    Runs :func:`diff_backends` once per recorded observation
    (``prompt_ids``/``output_ids`` pairs, as the stop-matrix path uses)
    with the given stop text(s), then aggregates into a JSON-serializable
    dict:

    - ``verdict``: ``"agree"`` iff no fixture diverged, else ``"diverge"``;
    - ``pairs_verdict``: one-line summary, e.g.
      ``"HF vs vLLM stop semantics: 3/4 agree; 2 divergences"``;
    - ``per_fixture``: ``[{fixture, verdict, divergences:
      [{class, stop_text, detail, severity}]}]`` in observation order;
    - ``matrix``: ``{divergence_class: {semantic: count}}`` counting, per
      fired divergence, the semantics the class is attributed to (see
      ``_CLASS_SEMANTICS``);
    - ``severities``: ``{divergence_class: severity}`` for the classes
      that fired (pins the report's severity badges);
    - ``stop_texts``, ``eos_id``, ``min_tokens``: the run parameters.

    Observations that are not dicts, or whose ``input_ids``/``output_ids``
    do not validate, are skipped. Returns None when no observation is
    usable, so the ``run`` diagnosis degrades gracefully (the section is
    simply absent from the report). ``stop_texts``/``eos_id``/
    ``min_tokens``/``backend`` are strictly validated like
    :func:`diff_backends`; ValueError with a fixed code on violation.
    Deterministic: identical inputs produce identical output.
    """
    if type(observations) is tuple:
        candidates: list[Any] = list(observations)
    elif type(observations) is list:
        candidates = list(observations)
    else:
        raise ValueError("invalid_observations")
    stops = _checked_stop_texts(stop_texts)
    if eos_id is not None and (type(eos_id) is not int or eos_id < 0):
        raise ValueError("invalid_eos_id")
    if type(min_tokens) is not int or min_tokens < 0:
        raise ValueError("invalid_min_tokens")
    if not callable(getattr(backend, "decode", None)):
        raise ValueError("invalid_backend")

    per_fixture: list[dict[str, Any]] = []
    matrix: dict[DivergenceClass, dict[str, int]] = {}
    for index, observation in enumerate(candidates):
        if type(observation) is not dict:
            continue
        prompt_ids = _observation_ids(observation, "input_ids")
        output_ids = _observation_ids(observation, "output_ids")
        if prompt_ids is None or output_ids is None:
            continue
        name = observation.get("fixture")
        fixture = name if type(name) is str and name else f"fixture_{index}"
        outcome = diff_backends(
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            stop_texts=stops,
            eos_id=eos_id,
            min_tokens=min_tokens,
            backend=backend,
        )
        findings: list[dict[str, Any]] = []
        for finding in outcome["divergences"]:
            kind: DivergenceClass = finding["class"]
            findings.append(
                {
                    "class": kind,
                    "stop_text": finding["stop_text"],
                    "detail": finding["detail"],
                    "severity": DIVERGENCE_SEVERITY[kind],
                }
            )
            row = matrix.setdefault(kind, {})
            for semantic in _CLASS_SEMANTICS[kind]:
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
            f"HF vs vLLM stop semantics: {agreed}/{total} agree; {total_findings} {noun}"
        ),
        "per_fixture": per_fixture,
        "matrix": matrix,
        "severities": {kind: DIVERGENCE_SEVERITY[kind] for kind in matrix},
        "stop_texts": list(stops),
        "eos_id": eos_id,
        "min_tokens": min_tokens,
    }


class _TokenizerBackend:
    """Decode-only backend adapter over a loaded HF fast tokenizer.

    Exposes the ``decode`` surface the stop emulations need, plus the
    ``hf_stop_token_bytes`` byte-space mapping (when determinable), so
    ``diff_backends`` runs the same measurement path as the lab's stop
    matrix. Never touches model weights: only the tokenizer is loaded.
    """

    def __init__(self, tokenizer: Any, vocab_size: int) -> None:
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError("invalid_vocab_size")
        if not callable(getattr(tokenizer, "decode", None)):
            raise ValueError("tokenizer_missing_decode")
        self._tokenizer = tokenizer
        self._vocab_size = vocab_size
        self.fingerprint = "diff-backends-tokenizer-only"
        self.hf_stop_token_bytes: Callable[[int], bytes | None] | None = _hf_byte_token_mapper(
            tokenizer
        )

    def decode(self, token_ids: tuple[int, ...]) -> str:
        if type(token_ids) is not tuple or any(
            type(token) is not int or not 0 <= token < self._vocab_size for token in token_ids
        ):
            raise ValueError("invalid_decode_ids")
        text: str = self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return text


def run_diff_backends(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Implement ``mpl diff-backends``; returns the process exit code.

    Loads the checkpoint's tokenizer only (never weights) via doctor's
    tokenizer-without-weights path, encodes the prompt/output texts with
    ``add_special_tokens=False``, runs :func:`diff_backends`, and prints the
    result JSON to stdout. Input validation fails fast (exit 2, argparse
    style); checkpoint/tokenizer problems exit 1 with a fixed message.
    """
    if args.min_tokens < 0:
        parser.error("--min-tokens must be a non-negative integer")
    if args.eos_id is not None and args.eos_id < 0:
        parser.error("--eos-id must be a non-negative integer")
    stops = args.stops.split(",")
    if any(not stop for stop in stops):
        parser.error("--stops must be comma-separated non-empty stop texts")
    # Bound the texts before encoding: the emulations do per-step decodes,
    # so unbounded input would be an unadvertised DoS surface.
    for label, text in (("--prompt-text", args.prompt_text), ("--output-text", args.output_text)):
        if len(text) > 1_000_000:
            parser.error(f"{label} must be at most 1MB")
    try:
        from .doctor import _build_primary_tokenizer

        tokenizer = _build_primary_tokenizer(args.model)
    except ImportError:
        print(
            "diff-backends: transformers is required for tokenizer loading",
            file=sys.stderr,
        )
        return 1
    except ValueError as exc:
        # Fixed vocabulary from the checkpoint checks; safe to show.
        print(f"diff-backends: invalid checkpoint: {exc}", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 -- sanitization boundary, never leak loader text
        print("diff-backends: tokenizer_load_failed", file=sys.stderr)
        return 1
    try:
        prompt_ids = tuple(
            tokenizer.encode(
                args.prompt_text, add_special_tokens=False, truncation=False, padding=False
            )
        )
        output_ids = tuple(
            tokenizer.encode(
                args.output_text, add_special_tokens=False, truncation=False, padding=False
            )
        )
    except Exception:  # noqa: BLE001 -- sanitization boundary, never leak tokenizer text
        print("diff-backends: encode_failed", file=sys.stderr)
        return 1
    backend = _TokenizerBackend(tokenizer, len(tokenizer.get_vocab()))
    try:
        result = diff_backends(
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            stop_texts=stops,
            eos_id=args.eos_id,
            min_tokens=args.min_tokens,
            backend=backend,
        )
    except ValueError as exc:
        # Fixed validation vocabulary from diff_backends; safe to show.
        print(f"diff-backends: invalid input: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0
