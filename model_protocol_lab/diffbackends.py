"""Differential testing of emulated backend stop semantics.

``diff_backends`` runs the lab's stop emulations (see
:mod:`model_protocol_lab.backends`) -- HF ``StopStringCriteria``
(``emulate_hf_stop_string``), vLLM's detokenizer holdback
(``emulate_vllm_holdback``, strip and keep variants), SGLang's str-based
finish (``emulate_sglang_stop``) and an EOS-appended truncation
(``emulate_eos_stop``) -- over ONE recorded token sequence and reports
where the emulated backends disagree, classified into divergence classes
drawn from the vendors' own bug trackers (research rounds R2a and Cycle 1,
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
- ``MULTI_STOP_SELECTION`` -- SGLang's multi-stop check selects the
  first-listed stop (``Req._check_str_based_finish`` iterates
  ``stop_strs`` in list order, ``schedule_batch.py:1836-1847``) while vLLM
  selects the earliest-completing one (``min(stop_index +
  len(stop_str))``, ties broken by list order; vLLM PR #49391,
  https://github.com/vllm-project/vllm/pull/49391, pinned by
  ``tests/detokenizer/test_check_stop_strings.py``). Fires when the two
  selection rules pick different winners across the run's stop texts --
  the hard divergence the research log records, with its caveat: with 1
  token/step the earlier-completing stop still fires at its own step and
  list order decides only when >=2 stops match within one check.
- ``NO_MIN_TOKENS_GATE`` -- the SGLang column on the ``MIN_TOKENS_GATED``
  class: SGLang has no ``min_tokens`` equivalent (``min_new_tokens`` only
  clips the length budget, ``scheduler.py:2631-2634``; the finish path
  never consults it), so a stop that vLLM's ``min_tokens`` gate blocks
  still fires under the SGLang emulation.
- ``STR_OVER_EOS_PRECEDENCE`` -- stop string and EOS fire in the same
  step, so precedence -- not firing order -- decides the outcome, and the
  engines disagree: SGLang's shipped fix for #28802 gives the stop string
  the win (``update_finish_state`` runs the str-based check before the
  token-based one, ``schedule_batch.py:1913-1950``), while TGI's
  documented precedence is length > EOS > stop-sequence, i.e. EOS beats
  the stop string (the reverse of SGLang). Co-fires with
  ``EOS_PRECEDENCE`` on the tie unless another stop cell fired strictly
  earlier (then EOS did not beat every stop, so only this class fires);
  this class records the precedence disagreement.

Deliberate non-cells (documented, not emulated -- no TGI column):

- No TGI cell: every TGI stop rule from the Cycle 1 research is covered
  by composition of the existing cells -- TGI keeps stop sequences (like
  HF's kept-disposition ``hf_stop_string`` cell), selects first-listed
  among multiple stops (like the ``sglang_stop`` selection rule), has no
  ``min_tokens`` equivalent (like ``sglang_stop``'s ungated finish path),
  and its EOS-beats-stop precedence is the documented contrast in
  ``STR_OVER_EOS_PRECEDENCE``. A TGI column would duplicate these cells
  without adding a distinguishable rule (the one genuine TGI quirk, the
  200-char match-buffer truncation, is too narrow for a cell and stays
  research-log-only).
- SGLang ``stop_regex`` (``MAX_STOP_REGEX_COUNT=32``,
  ``MAX_STOP_REGEX_LEN=256``) is outside stop-string scope: a USAGE.md
  note only, no cell.

Scope limits (deliberate, documented -- no semantics are invented here;
every cell is one of the existing emulations in
:mod:`model_protocol_lab.backends`, run with fixed inputs):

- Per-stop cells still run one stop string at a time; multi-stop
  SELECTION is compared across the run's stop texts
  (``MULTI_STOP_SELECTION``): the SGLang winner is the fired stop with
  the smallest firing step, ties broken by list order (the research-log
  caveat, exactly), while the vLLM winner ports PR #49391 -- smallest
  ``(firing step, stop_index + len(stop_str))``, ties by list order, read
  off the strip/keep cells' firing steps and trimmed texts.
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
    emulate_sglang_stop,
    emulate_vllm_holdback,
)

DivergenceClass = Literal[
    "KEPT_VS_STRIPPED",
    "FIRING_STEP_DIFFERS",
    "MIN_TOKENS_GATED",
    "EOS_PRECEDENCE",
    # Vacuous for SGLang by construction: text trimmed at the leftmost stop-string occurrence.
    "HARNESS_NORMALIZATION_RELIED",
    "MULTI_STOP_SELECTION",
    "NO_MIN_TOKENS_GATE",
    "STR_OVER_EOS_PRECEDENCE",
]

#: Semantic names used as keys in each stop record's ``semantics`` mapping.
HF_STOP_STRING = "hf_stop_string"
VLLM_HOLDBACK_STRIP = "vllm_holdback_strip"
VLLM_HOLDBACK_KEEP = "vllm_holdback_keep"
SGLANG_STOP = "sglang_stop"
EOS_APPENDED = "eos_appended"

_STOP_SEMANTICS = (HF_STOP_STRING, VLLM_HOLDBACK_STRIP, VLLM_HOLDBACK_KEEP, SGLANG_STOP)

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
        """Return the as dict."""
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
        """Return the as dict."""
        return {"class": self.kind, "stop_text": self.stop_text, "detail": self.detail}


#: Maximum length of an *output* (generated) token-id sequence accepted by
#: the stop emulations. The vLLM-holdback and SGLang emulations do one
#: ``backend.decode(prefix)`` per output step, so total cost is quadratic
#: in this length: at the cap a single stop completes in ~2-3 s on the
#: repo's linear reference stub (measured 2026-10-15; tens of minutes at
#: the old 131072 cap). The value also matches ``mpl run --max-new-tokens``' own
#: 4096 cap, so the lab can never record a longer generated output --
#: legitimate inputs are never rejected by this bound.
MAX_OUTPUT_IDS = 4096

#: Maximum length of a *prompt* (input) token-id sequence. Prompt tokens
#: are decoded once each (linear cost), so the bound stays generous: 4x
#: the lab's max context window (32768), which recorded observations can
#: legitimately approach.
MAX_PROMPT_IDS = 131072

#: Maximum length of one stop text, in characters. Real stop strings are
#: short phrases; the bound keeps ``json.dumps(result)`` small (each stop
#: is embedded verbatim in its record plus ``{stop_text!r}`` details) and
#: the per-step ``str.find`` window cheap. Far below the 1 MB prompt/
#: output text caps -- a stop text is a needle, not a document.
MAX_STOP_TEXT_LEN = 1024

#: Maximum number of (deduped) stop texts per call. Each stop adds five
#: emulated cells, two of them quadratic in the output length, so the
#: count multiplies the ``MAX_OUTPUT_IDS`` bound; at both caps the
#: absolute worst case is ~40 s on the linear reference stub. Real stop
#: lists are a handful of strings; 16 is generous headroom.
MAX_STOP_TEXTS = 16

#: Maximum number of recorded observations ``divergence_section_data``
#: processes per call. The ``run`` diagnosis path feeds the winner's
#: recorded observations only (search + held-out fixture outputs);
#: experiments.py rejects fixture splits above 100 search / 100 held-out
#: fixtures, so a legitimate run hands over at most 200 observations.
#: 512 is comfortably above that legitimate maximum (2.56x headroom) but
#: far below DoS territory: each observation costs one full
#: ``diff_backends`` emulation (~2-3 s worst case per stop text at
#: ``MAX_OUTPUT_IDS``), so a hostile report with millions of observations
#: would otherwise burn open-ended hours. Over-count input degrades by
#: omission, never by exception: the first 512 observations are processed
#: in input order and the output records ``observations_truncated: true``
#: plus the offered count.
MAX_DIVERGENCE_OBSERVATIONS = 512


def _checked_ids(value: object, name: str, *, cap: int) -> tuple[int, ...]:
    """Strict token-id tuple validation (mirrors the emulations' contract).

    Requires an actual tuple of plain ints (bool excluded), each a
    non-negative id below 2**31, matching the CLI's ``parse_boundary``
    bound. The total sequence is capped at ``cap`` (``MAX_OUTPUT_IDS``
    for generated sequences, ``MAX_PROMPT_IDS`` for prompts) because the
    emulations do per-step decodes; unbounded input would be an
    unadvertised DoS surface. Raises ValueError with a fixed code
    otherwise.
    """
    if type(value) is not tuple or any(
        type(token) is not int or not 0 <= token < 2**31 for token in value
    ):
        raise ValueError(f"invalid_{name}")
    if len(value) > cap:
        raise ValueError(f"{name}_too_long")
    return value


def _checked_stop_texts(value: object) -> tuple[str, ...]:
    """Strict stop-text validation: tuple/list of non-empty strings.

    Each stop is bounded by ``MAX_STOP_TEXT_LEN`` and the deduped list by
    ``MAX_STOP_TEXTS`` (see the constants for rationale): unbounded stops
    balloon ``json.dumps(result)`` and the per-stop emulation cost.
    Duplicates are removed (order-preserving, set-based so dedup itself
    is linear, not O(m^2)): running the same stop twice would only
    duplicate identical cells.
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
        if len(stop) > MAX_STOP_TEXT_LEN:
            raise ValueError("stop_text_too_long")
        strings.append(stop)
    deduped: list[str] = []
    seen: set[str] = set()
    for stop in strings:
        if stop not in seen:
            seen.add(stop)
            deduped.append(stop)
    if not deduped:
        raise ValueError("empty_stop_texts")
    if len(deduped) > MAX_STOP_TEXTS:
        raise ValueError("too_many_stop_texts")
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
    """Return the cell."""
    return CellResult(
        semantic=semantic,
        available=bool(outcome.available),
        fired=bool(outcome.fired),
        firing_step=_firing_step(outcome.detail),
        truncated_ids=tuple(outcome.truncated_ids),
        returned_text=outcome.returned_text,
        disposition=disposition,
    )


# skipcq: PY-R1000 - branches are frozen payload-contract cases pinned by tests
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
    in strip and keep variants (per stop text, with ``min_tokens``),
    ``emulate_sglang_stop`` (per stop text; SGLang has no ``min_tokens``
    equivalent, so no gate is applied) and ``emulate_eos_stop`` (once per
    stop text; it does not depend on the stop text), then classifies
    disagreements into the divergence classes documented in the module
    docstring -- including the cross-stop ``MULTI_STOP_SELECTION``
    comparison of SGLang's first-listed rule against vLLM's
    earliest-completion rule. Returns a JSON-serializable dict with
    ``stop_texts``, ``eos_id``, ``min_tokens``, per-stop ``cells``,
    ``divergences`` and ``verdict`` (``"agree"`` iff no divergences), plus
    the echoed ``prompt_tokens``/``output_tokens`` input lengths.

    ``backend`` needs only ``decode`` (plus the optional
    ``hf_stop_token_bytes`` byte-space mapping, used by the HF emulation
    exactly as the lab's stop matrix does). All inputs are strictly
    validated; ValueError with a fixed code on any violation. Cells whose
    emulation reports ``available=False`` are recorded but excluded from
    the divergence checks. Per-stop composition is deliberate: upstream's
    joint multi-stop criteria bounds the match walk by the longest stop,
    an implementation artifact the lab does not reproduce (see
    ``emulate_hf_stop_string``).
    """
    prompt_ids = _checked_ids(prompt_ids, "prompt_ids", cap=MAX_PROMPT_IDS)
    output_ids = _checked_ids(output_ids, "output_ids", cap=MAX_OUTPUT_IDS)
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
    # Per-stop cells retained for the cross-stop MULTI_STOP_SELECTION
    # comparison after the loop: (stop text, sglang, vllm strip, vllm
    # keep) in stop-list order.
    selection_cells: list[tuple[str, CellResult, CellResult, CellResult]] = []
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
        # SGLang has no min_tokens equivalent (min_new_tokens only clips
        # the length budget; the finish path never consults it), so no
        # gate is applied here -- that absence is what NO_MIN_TOKENS_GATE
        # measures.
        sglang = _cell(
            SGLANG_STOP,
            emulate_sglang_stop(backend, output_ids, stop_text),
            "stripped",
        )
        eos = _cell(EOS_APPENDED, emulate_eos_stop(output_ids, eos_id), None)
        semantics = {
            HF_STOP_STRING: hf,
            VLLM_HOLDBACK_STRIP: strip,
            VLLM_HOLDBACK_KEEP: keep,
            SGLANG_STOP: sglang,
            EOS_APPENDED: eos,
        }
        records.append(
            {
                "stop_text": stop_text,
                "semantics": {name: cell.as_dict() for name, cell in semantics.items()},
            }
        )
        selection_cells.append((stop_text, sglang, strip, keep))

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

        # MIN_TOKENS_GATED / NO_MIN_TOKENS_GATE share one ungated
        # diagnostic re-run: the stop fires with min_tokens=0 but the given
        # min_tokens gates vLLM's holdback (vLLM PR #22014). SGLang has no
        # min_tokens equivalent, so its column fires anyway -- that is the
        # NO_MIN_TOKENS_GATE class.
        ungated_step: int | None = None
        if min_tokens > 0 and strip.available and not strip.fired:
            ungated = emulate_vllm_holdback(backend, output_ids, stop_text, min_tokens=0)
            if ungated.available and ungated.fired:
                ungated_step = _firing_step(ungated.detail)
        if ungated_step is not None:
            divergences.append(
                Divergence(
                    "MIN_TOKENS_GATED",
                    stop_text,
                    f"stop {stop_text!r} fires at step {ungated_step} with "
                    f"min_tokens=0 but min_tokens={min_tokens} gates it "
                    "(vLLM PR #22014)",
                )
            )
            if sglang.available and sglang.fired and sglang.firing_step is not None:
                divergences.append(
                    Divergence(
                        "NO_MIN_TOKENS_GATE",
                        stop_text,
                        f"stop {stop_text!r} fires at step {ungated_step} with "
                        f"min_tokens=0 and at step {sglang.firing_step} under "
                        "sglang_stop (SGLang has no min_tokens equivalent: "
                        "min_new_tokens only clips the length budget, the "
                        "finish path never consults it) but "
                        f"min_tokens={min_tokens} gates vllm_holdback_strip "
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

        # STR_OVER_EOS_PRECEDENCE: the stop string and EOS fire in the same
        # step, so precedence -- not firing order -- decides the outcome,
        # and the engines disagree. SGLang's shipped fix for #28802 gives
        # the stop string the win (update_finish_state runs the str-based
        # check before the token-based one); TGI's documented precedence is
        # length > EOS > stop-sequence, i.e. EOS beats the stop string (the
        # reverse of SGLang). Co-fires with EOS_PRECEDENCE on the tie
        # unless another stop cell fired strictly earlier (then EOS did not
        # beat every stop, so only this class fires); this class records
        # the precedence disagreement.
        if (
            sglang.available
            and sglang.fired
            and sglang.firing_step is not None
            and eos.available
            and eos.fired
            and eos.firing_step == sglang.firing_step
        ):
            divergences.append(
                Divergence(
                    "STR_OVER_EOS_PRECEDENCE",
                    stop_text,
                    f"sglang_stop fired at step {sglang.firing_step} and "
                    f"eos_appended fired at step {eos.firing_step}: same-step "
                    "precedence decides the outcome -- SGLang's shipped fix "
                    "for #28802 gives the stop string the win "
                    "(update_finish_state runs the str-based check before "
                    "the token-based one), while TGI's documented precedence "
                    "is length > EOS > stop-sequence (EOS beats the stop "
                    "string)",
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

    # MULTI_STOP_SELECTION (cross-stop): SGLang's multi-stop check
    # selects the first-listed stop -- the winner is the fired stop with
    # the smallest firing step, ties broken by list order (the
    # research-log caveat: with 1 token/step the earlier-completing stop
    # still fires at its own step; list order decides only when >=2 stops
    # match within one check). vLLM instead selects the
    # earliest-completing stop: min(stop_index + len(stop_str)), ties by
    # list order (PR #49391). The keep cell's returned text is the firing
    # step's decoded text cut at the windowed find's match end, so its
    # length IS stop_index + len(stop_str).
    sglang_pick: tuple[tuple[int, int], tuple[str, int]] | None = None
    vllm_pick: tuple[tuple[int, int, int], tuple[str, int]] | None = None
    for list_index, (stop_text, sglang_cell, strip_cell, keep_cell) in enumerate(selection_cells):
        if sglang_cell.available and sglang_cell.fired and sglang_cell.firing_step is not None:
            key = (sglang_cell.firing_step, list_index)
            if sglang_pick is None or key < sglang_pick[0]:
                sglang_pick = (key, (stop_text, sglang_cell.firing_step))
        if (
            strip_cell.available
            and strip_cell.fired
            and strip_cell.firing_step is not None
            and keep_cell.available
            and keep_cell.fired
            and keep_cell.returned_text is not None
        ):
            end_index = len(keep_cell.returned_text)
            vkey = (strip_cell.firing_step, end_index, list_index)
            if vllm_pick is None or vkey < vllm_pick[0]:
                vllm_pick = (vkey, (stop_text, strip_cell.firing_step))
    sglang_winner = sglang_pick[1] if sglang_pick is not None else None
    vllm_winner = vllm_pick[1] if vllm_pick is not None else None
    if sglang_winner != vllm_winner:

        def _pick_text(pick: tuple[str, int] | None) -> str:
            """Return the pick text."""
            if pick is None:
                return "no stop"
            return f"stop {pick[0]!r} at step {pick[1]}"

        winner_stop: str | None
        if sglang_winner is not None:
            winner_stop = sglang_winner[0]
        elif vllm_winner is not None:
            winner_stop = vllm_winner[0]
        else:  # pragma: no cover -- the winners differ, so one of them fired
            winner_stop = None
        divergences.append(
            Divergence(
                "MULTI_STOP_SELECTION",
                winner_stop,
                f"sglang_stop selects {_pick_text(sglang_winner)} "
                "(first-listed rule, Req._check_str_based_finish) but "
                f"vllm_holdback_strip selects {_pick_text(vllm_winner)} "
                "(earliest-completion, min(stop_index + len(stop_str)), "
                "vLLM PR #49391)",
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
    "STR_OVER_EOS_PRECEDENCE": "high",
    "MULTI_STOP_SELECTION": "high",
    "FIRING_STEP_DIFFERS": "medium",
    "MIN_TOKENS_GATED": "medium",
    "NO_MIN_TOKENS_GATE": "medium",
    "HARNESS_NORMALIZATION_RELIED": "low",
}

#: Semantics each divergence class is attributed to in the class x
#: semantic count matrix: the emulated semantics whose disagreement the
#: class measures.
_CLASS_SEMANTICS: dict[DivergenceClass, tuple[str, ...]] = {
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
        # Recorded outputs are generated sequences (quadratic emulation
        # cost); recorded inputs are prompts (linear cost).
        cap = MAX_OUTPUT_IDS if key == "output_ids" else MAX_PROMPT_IDS
        return _checked_ids(raw, key, cap=cap)
    except ValueError:
        return None


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
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
      ``"HF vs vLLM vs SGLang stop semantics: 3/4 agree; 2 divergences"``;
    - ``per_fixture``: ``[{fixture, verdict, divergences:
      [{class, stop_text, detail, severity}]}]`` in observation order;
    - ``matrix``: ``{divergence_class: {semantic: count}}`` counting, per
      fired divergence, the semantics the class is attributed to (see
      ``_CLASS_SEMANTICS``);
    - ``severities``: ``{divergence_class: severity}`` for the classes
      that fired (pins the report's severity badges);
    - ``stop_texts``, ``eos_id``, ``min_tokens``: the run parameters;
    - ``observations_truncated``: True when the input list held more than
      ``MAX_DIVERGENCE_OBSERVATIONS`` entries;
    - ``observations_total``: the number of observation entries handed in;
    - ``observations_processed``: the number of observations actually
      emulated, i.e. ``len(per_fixture)`` (the bound when truncated *and*
      every truncated candidate validated -- skipped observations are not
      counted).

    Observations that are not dicts, or whose ``input_ids``/``output_ids``
    do not validate, are skipped. At most ``MAX_DIVERGENCE_OBSERVATIONS``
    entries are processed -- the first ones, in input order -- so a hostile
    report with millions of observations degrades to a fixed-vocabulary
    truncation note instead of burning open-ended emulation time; this
    preserves the "degrade by omission, never fail the run" contract.
    Returns None when no observation is
    usable, so the ``run`` diagnosis degrades gracefully (the section is
    simply absent from the report). ``stop_texts``/``eos_id``/
    ``min_tokens``/``backend`` are strictly validated like
    :func:`diff_backends`; ValueError with a fixed code on violation.
    Deterministic: identical inputs produce identical output (truncation
    takes the same first-N prefix every run).
    """
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
            f"HF vs vLLM vs SGLang stop semantics: {agreed}/{total} agree; {total_findings} {noun}"
        ),
        "per_fixture": per_fixture,
        "matrix": matrix,
        "severities": {kind: DIVERGENCE_SEVERITY[kind] for kind in matrix},
        "stop_texts": list(stops),
        "eos_id": eos_id,
        "min_tokens": min_tokens,
        "observations_truncated": truncated,
        "observations_total": observations_total,
        "observations_processed": len(per_fixture),
    }


class _TokenizerBackend:
    """Decode-only backend adapter over a loaded HF fast tokenizer.

    Exposes the ``decode`` surface the stop emulations need, plus the
    ``hf_stop_token_bytes`` byte-space mapping (when determinable), so
    ``diff_backends`` runs the same measurement path as the lab's stop
    matrix. Never touches model weights: only the tokenizer is loaded.
    """

    def __init__(self, tokenizer: Any, vocab_size: int) -> None:
        """Initialise the tokenizer-only diff backend."""
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
        """Decode token IDs without dropping special tokens."""
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
    """Implement ``model-protocol-lab diff-backends``; returns the process exit code.

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
        from .doctor import _build_primary_tokenizer, _loader_failure_code

        tokenizer = _build_primary_tokenizer(args.model)
    except ImportError:
        print(
            "diff-backends: transformers is required for tokenizer loading",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001 -- sanitization boundary, never leak loader text
        # Only exact-match fixed codes (e.g. oversized_tokenizer_file)
        # reach the terminal; a parse failure echoing checkpoint bytes
        # collapses to the fixed code.
        # skipcq: PYL-E0601 - bound by the try import whenever this handler runs
        print(f"diff-backends: {_loader_failure_code(exc)}", file=sys.stderr)
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
