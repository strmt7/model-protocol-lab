"""Scorer-lens regrade: how much of a template's failure is the scorer?

The lab's native verdict is byte-exact: a recorded answer passes a
fixture only when it equals the expected text exactly (and its
response boundary was found). That strictness is the point of the
lab -- but it means a template can look worse than a rival purely
because it formats answers differently: trailing punctuation, a
markdown fence, ``1,000`` instead of ``1000``. This module regrades
the *recorded* search answers under a frozen ladder of progressively
more permissive text normalisations, so a run can separate "the
template produced the wrong answer" from "the template produced the
right answer in a surface form the byte-exact scorer discards".

The ladder is a single pipeline -- each rung consumes the previous
rung's output, which makes monotonicity a theorem rather than a
hope, and R4 is a union with R3, never a replacement comparison:

- R0: identity. Reproduces the native verdicts (the S1 oracle).
- R1: trim; collapse runs of spaces/tabs to one space; collapse
  three or more consecutive newlines to two. Single and double
  newlines survive: line structure can be content (a two-line
  answer is not the same answer on one line).
- R2: NFC, casefold, then combining-mark removal (NFD, drop
  category Mn, recompose). The mark step is the refusals
  detector's fix for Turkish dotted capital I (``İ`` casefolds to
  ``i`` plus a combining dot, which plain casefold leaves unequal
  to ``i``). The decomposition is canonical (NFD), not the
  detector's compatibility NFKC: fullwidth digits and other
  compatibility forms are NOT folded at R2, consistent with R4's
  ASCII-only grammar. Casefold still equates German ``ß`` with ``ss`` --
  Unicode default folding working as specified; on name fixtures
  that can be a false recovery, and the payload's per-flip
  attribution says whether a flip needed more than NFC.
- R3: markdown-fence strip (line-anchored ``` fences, any info
  string discarded, applied to both sides), then iterative
  stripping of a small presentational punctuation set at the
  string edges. The set is data (see ``LADDER_SPEC``); it excludes
  ``-``/``+`` (signs are content), ``%`` (a unit, not decoration),
  and a leading ``.`` directly before a digit is never stripped
  (``.5`` is not ``5``). Removing punctuation at *all* positions
  is forbidden by the protocol -- it would equate ``3.5`` with
  ``35``.
- R4: R3, or both R3 forms match the frozen ASCII numeric grammar
  (plain digits or strict thousands grouping, optional fraction
  and exponent) and parse to finite, equal ``Decimal`` values.
  Underscores, non-ASCII digits, ``inf``/``NaN`` spellings and
  decimal commas never fire R4: ``1,5`` is one-and-a-half in much
  of the world, not ``15``, and the ladder has no locale input.
  Comparison is ``Decimal.__eq__`` on the parsed values, never
  ``normalize()`` (the default context would collapse distinct
  long decimals).

Two honesty boundaries. The ladder has **no label-strip
operation**: ``Answer: 42`` flips at no rung against expected
``42`` -- lead-in prose is not surface the ladder forgives, by
design. And a rung pass is a *text* statement only; the reported
verdict series always conjoins the recorded boundary fact, which
no rung can move.

Fail-closed disciplines, per the wave-7 design cross-check: at
rungs >= 1 a pass requires both normalised forms to be nonempty
(an erasure-to-empty is a no-flip, never a pass); fixtures whose
expected text normalises to empty at R1 are excluded from the lens
entirely, with the count reported -- the lens has nothing to say
about a fixture with no surface. The payload carries counts,
fixture IDs and rung numbers only: no answer text, no expected
text, no normalised forms, no lengths. The per-fixture ground
records let the diff-reports validator recompute every aggregate.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

LADDER_ID = "scorerlens-ladder-v1"

# The R3 edge-strip set, published as data: presentational marks
# only. ``-``/``+``/``%`` are deliberately absent (see module
# docstring); backtick is present so a stray `code` span that does
# not form a line fence still strips.
STRIP_CHARACTERS = "*_`\"'.,;:!?()[]{}"

# One ASCII-only full-match grammar for every R4 input (plain or
# strictly grouped mantissa, optional fraction/exponent, or a
# leading-dot fraction). ``[0-9]`` rather than ``\d``: ``\d``
# matches every Unicode Nd digit, which would smuggle the
# cross-script acceptance the grammar exists to exclude.
NUMERIC_GRAMMAR = (
    r"^[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?"
    r"(?:[eE][+-]?[0-9]+)?$"
    r"|^[+-]?\.[0-9]+(?:[eE][+-]?[0-9]+)?$"
)

_NUMERIC_RE = re.compile(NUMERIC_GRAMMAR)
_FENCE_OPEN_RE = re.compile(r"```[^`]*")
_SPACE_TAB_RUN = re.compile(r"[ \t]+")
_NEWLINE_RUN = re.compile(r"\n{3,}")

LADDER_SPEC: dict[str, Any] = {
    "protocol": LADDER_ID,
    "rungs": [
        {"rung": 0, "ops": ["identity"]},
        {
            "rung": 1,
            "ops": [
                "trim",
                "collapse-space-tab-runs",
                "collapse-3plus-newlines-to-2",
            ],
        },
        {
            "rung": 2,
            "ops": ["nfc", "casefold", "nfd-strip-mn", "nfc"],
        },
        {"rung": 3, "ops": ["fence-strip", "edge-punctuation-strip"]},
        {"rung": 4, "ops": ["union-numeric-equality"]},
    ],
    "strip_characters": sorted(STRIP_CHARACTERS),
    "numeric_grammar": NUMERIC_GRAMMAR,
    "empty_guard": "rungs >= 1 require nonempty normalised forms on both sides",
    "decisions": {
        "underscores": "rejected (not in the numeric grammar)",
        "leading_plus": "accepted (Decimal grammar)",
        "signed_zero": "-0 == 0 (Decimal value equality)",
        "non_ascii_digits": "rejected (grammar is [0-9] only)",
        "decimal_comma": "out of scope (no locale input)",
        "percent": "never stripped, never numeric",
        "label_prefix": "no label-strip operation exists",
    },
}

RUNGS = (0, 1, 2, 3, 4)


def ladder_sha256() -> str:
    """Digest of the ladder-as-data, recomputable from any payload."""
    canonical = json.dumps(LADDER_SPEC, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _norm_r1(text: str) -> str:
    """R1: trim, collapse space/tab runs, cap newline runs at two."""
    out = text.strip()
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = _SPACE_TAB_RUN.sub(" ", out)
    out = _NEWLINE_RUN.sub("\n\n", out)
    return out


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _norm_r2(text: str) -> str:
    """R2: NFC, casefold, combining-mark removal, recompose.

    The mark removal is what makes Turkish ``İ`` recover against
    ``i`` (casefold alone leaves a combining dot above); it mirrors
    the refusal detector's fix for the same trap. The decomposition
    is canonical (NFD), deliberately NOT the detector's NFKC: a
    compatibility fold would also equate fullwidth digits and
    other compatibility forms, and the ladder's job is to price
    surface differences, not erase them wholesale.
    """
    folded = _nfc(text).casefold()
    decomposed = unicodedata.normalize("NFD", folded)
    unmarked = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return _nfc(unmarked)


def _fence_strip(text: str) -> str:
    """Remove one pair of line-anchored ``` fences, if present.

    The opening fence may carry an info string (discarded -- it is
    presentation metadata, and both sides get the same treatment);
    the closing fence must be exactly ``` on its own line.
    """
    lines = text.split("\n")
    if len(lines) >= 3 and _FENCE_OPEN_RE.fullmatch(lines[0]) and lines[-1] == "```":
        return "\n".join(lines[1:-1]).strip("\n")
    return text


def _punct_strip(text: str) -> str:
    """Iteratively strip the frozen edge set from both ends.

    A leading ``.`` directly before an ASCII digit is never
    stripped: ``.5`` losing its dot would manufacture a 10x error.
    """
    out = text
    while out:
        stripped = False
        if out[0] in STRIP_CHARACTERS and not (
            out[0] == "." and len(out) > 1 and out[1] in "0123456789"
        ):
            out = out[1:]
            stripped = True
        if out and out[-1] in STRIP_CHARACTERS:
            out = out[:-1]
            stripped = True
        if not stripped:
            break
    return out


def _norm_r3(text: str) -> str:
    """R3: fence strip, then edge-punctuation strip, then trim."""
    return _punct_strip(_fence_strip(text)).strip()


def ladder_forms(text: str) -> tuple[str, str, str, str]:
    """The R0..R3 normalised forms of ``text`` (R4 is a comparison)."""
    form0 = text
    form1 = _norm_r1(form0)
    form2 = _norm_r2(form1)
    form3 = _norm_r3(form2)
    return form0, form1, form2, form3


def _decimal_value(form: str) -> Decimal | None:
    """Parse ``form`` under the frozen grammar, or return None.

    The grammar is the gate; ``Decimal`` only ever sees strings it
    fully matches, and the finite check stays as a second gate so
    no spelling of infinity or NaN can ever compare equal here.
    """
    if not _NUMERIC_RE.match(form):
        return None
    try:
        value = Decimal(form.replace(",", ""))
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def _form_match(a_form: str, e_form: str) -> bool:
    """Pure rung match: forms equal and both nonempty (the S3 guard)."""
    return bool(a_form) and bool(e_form) and a_form == e_form


def rung_match(answer: str, expected: str, rung: int) -> bool:
    """Whether ``answer`` passes ``expected`` at ``rung`` (text only).

    Cumulative by definition -- "passes under the ladder up to this
    rung": raw equality (the R0 component, vacuous empty-expected
    passes included, as S1 requires) counts at every rung, each
    rung's pure match is its normalised forms being equal and
    nonempty on both sides, and R4 adds the numeric union. The
    cumulative reading is what makes monotonicity (S2) hold even
    for pairs whose normalised forms later erase to empty: an
    identical all-punctuation pair passes at R0 and therefore at
    every rung, while a *distinct* pair that only meets at an
    empty form (``?`` vs ``!``) passes at none. The
    boundary/status conjuncts of a full verdict live in
    :func:`analyze_candidate`, not here.
    """
    if rung not in RUNGS:
        raise ValueError(f"unknown rung: {rung!r}")
    if answer == expected:
        return True
    a_forms = ladder_forms(answer)
    e_forms = ladder_forms(expected)
    for j in (1, 2, 3):
        if j > rung:
            break
        if _form_match(a_forms[j], e_forms[j]):
            return True
    if rung == 4:
        a_value = _decimal_value(a_forms[3])
        e_value = _decimal_value(e_forms[3])
        return a_value is not None and e_value is not None and a_value == e_value
    return False


def first_flip(answer: str, expected: str) -> tuple[int | None, str | None]:
    """First rung >= 1 at which a natively failing pair passes.

    Returns ``(rung, via)``; ``via`` names the R2 sub-operation that
    first achieved equality (``"nfc"`` or ``"casefold"``) and is
    None at other rungs. ``(None, None)`` when no rung recovers the
    pair. Callers pass only pairs that fail R0.
    """
    a_forms = ladder_forms(answer)
    e_forms = ladder_forms(expected)
    for rung in (1, 2, 3):
        if _form_match(a_forms[rung], e_forms[rung]):
            via: str | None = None
            if rung == 2:
                via = "nfc" if _nfc(a_forms[1]) == _nfc(e_forms[1]) else "casefold"
            return rung, via
    a_value = _decimal_value(a_forms[3])
    e_value = _decimal_value(e_forms[3])
    if a_value is not None and e_value is not None and a_value == e_value:
        return 4, None
    return None, None


def _aggregate(records: list[dict[str, Any]], *, eligible: bool) -> dict[str, Any]:
    """Derive every per-candidate aggregate from the ground records.

    This is the producer's half of the recompute contract: the
    diff-reports validator re-derives the same values from the
    same records with independent code, and a payload whose
    aggregates disagree with its records is corruption.
    """
    evaluable = [r for r in records if not r["excluded"]]
    n_evaluable = len(evaluable)
    match_counts: dict[str, int] = {}
    verdict_counts: dict[str, int] = {}
    for rung in RUNGS:
        key = str(rung)
        match_counts[key] = sum(
            1
            for r in evaluable
            if r["exact_match"]
            or (r["first_flip_rung"] is not None and r["first_flip_rung"] <= rung)
        )
        verdict_counts[key] = sum(
            1
            for r in evaluable
            if r["completed"]
            and r["boundary_found"]
            and (
                r["exact_match"]
                or (r["first_flip_rung"] is not None and r["first_flip_rung"] <= rung)
            )
        )
    n_failures = n_evaluable - verdict_counts["0"]
    n_recovered = verdict_counts["4"] - verdict_counts["0"]
    if not eligible:
        share: float | None = None
        share_note = "candidate_not_eligible"
    elif n_failures == 0:
        share = None
        share_note = "no_native_failures"
    else:
        share = n_recovered / n_failures
        share_note = ""
    flipped = [r for r in evaluable if r["first_flip_rung"] is not None]
    return {
        "n_evaluable": n_evaluable,
        "n_excluded_empty_expected": len(records) - n_evaluable,
        "n_boundary_blocked": sum(
            1 for r in evaluable if r["completed"] and not r["boundary_found"]
        ),
        "match_counts": match_counts,
        "verdict_counts": verdict_counts,
        "n_native_verdict_failures": n_failures,
        "n_recovered_by_r4": n_recovered,
        "surface_failure_share": share,
        "share_note": share_note,
        "n_flips_boundary_blocked": sum(1 for r in flipped if not r["boundary_found"]),
        "n_flips_expected_altered": sum(1 for r in flipped if r["expected_r3_altered"]),
        "n_r1_flips_newline_changed": sum(
            1 for r in flipped if r["first_flip_rung"] == 1 and r["newline_differs"]
        ),
    }


def analyze_candidate(
    observations: Any,
    expected_by_fixture: dict[str, str],
    *,
    eligible: bool,
) -> dict[str, Any]:
    """Regrade one candidate's recorded observations under the ladder.

    The join against ``expected_by_fixture`` is fail-closed: an
    observation naming a fixture outside the map (or a malformed
    observation) makes the whole candidate unavailable with a
    named detail -- a partial regrade in silence would manufacture
    denominators. Never raises on data.
    """
    if type(observations) is not list:
        return {"available": False, "detail": "observations_not_a_list"}
    records: list[dict[str, Any]] = []
    n_completed = 0
    for observation in observations:
        if type(observation) is not dict:
            return {"available": False, "detail": "observation_malformed"}
        fixture = observation.get("fixture")
        answer = observation.get("answer")
        if (
            type(fixture) is not str
            or type(answer) is not str
            or type(observation.get("boundary_found")) is not bool
            or type(observation.get("status")) is not str
        ):
            return {"available": False, "detail": "observation_malformed"}
        if fixture not in expected_by_fixture:
            return {"available": False, "detail": "unknown_fixture"}
        expected = expected_by_fixture[fixture]
        completed = observation.get("status") == "completed"
        if completed:
            n_completed += 1
        excluded = _norm_r1(expected) == ""
        exact = completed and answer == expected
        flip_rung: int | None = None
        flip_via: str | None = None
        expected_altered = False
        if not excluded and not exact and answer != expected:
            flip_rung, flip_via = first_flip(answer, expected)
            if flip_rung is not None:
                e_forms = ladder_forms(expected)
                expected_altered = e_forms[3] != e_forms[2]
        records.append(
            {
                "fixture": fixture,
                "completed": completed,
                "boundary_found": observation.get("boundary_found") is True,
                "exact_match": exact,
                "excluded": excluded,
                "first_flip_rung": flip_rung,
                "flip_via": flip_via,
                "expected_r3_altered": expected_altered,
                "newline_differs": answer.count("\n") != expected.count("\n"),
            }
        )
    record: dict[str, Any] = {
        "available": True,
        "eligible": eligible,
        "n_observations": len(observations),
        "n_completed": n_completed,
    }
    record.update(_aggregate(records, eligible=eligible))
    record["records"] = records
    return record
