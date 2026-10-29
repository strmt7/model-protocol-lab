"""Verbalized-confidence x template probe (cycle 199): pure payload computation.

The producer-side half of ``run --confidence`` (docs/CONFIDENCE_DESIGN.md).
Every search fixture's last user message gains one frozen suffix asking
the model to end its reply with a line of the form
``Confidence: <integer 0-100>``; the elicited generations are parsed
under a frozen grammar, and the stated numbers are compared against
the exact-match verdicts the lab already trusts. A second, generated
arm pair (planted-fact cases whose stated value is knowable vs a
donor-key control whose payload binds a *different* key) measures
whether the stated number tracks knowability at all.

What this module owns:

* the frozen instrument -- the suffix text and joiner, the line
  grammar and its total parser (five outcome codes, never an
  exception), the eligibility screens, and the generated case set;
* the protocol object and its fingerprint (seed, budget, fixture-set
  digest, per-candidate boundaries, backend fingerprint, the screen
  map, the generated cases' digests, the frozen bin edges);
* every derived readout, computed from digest-bound ground records:
  per-candidate compliance, the compliant correct/incorrect joint
  table, mean stated confidence per verdict class and their gap
  (descriptive, withheld with a named reason when a class is empty),
  the frozen four-bin table, elicitation-cost verdict flips against
  the recorded native run, and -- over the generated pair -- the
  knowability falsification boolean.

Honesty constraints, enforced by construction: a stated confidence is
model output text, not a probability; no readout here is a
calibration error and no verdict word claims the model is calibrated
or overconfident. Records carry digests, never answer text. The one
exception is a compliant cell's final line, carried verbatim: it is
one of exactly 101 strings and a bijection with the confidence scalar
the record already carries, so it discloses nothing new and lets the
diff validator *derive* the scalar instead of trusting it.
Malformed final lines are carried as a 32-character prefix plus the
full line length, which is exactly what re-verifying their class
takes (a grammar line is at most 15 characters). Everything else
about a generation -- finality, multiplicity, the span split -- is
stage-attested, the same residue class as every digest-bound probe
in the tree. This module never imports the backend or runs a model:
the stage in experiments.py drives evaluation and assembles records.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any

from .core import Fixture, Message
from .repetition import target_index

#: Protocol version for the confidence payload.
PROTOCOL_VERSION = 1

#: Records caps, enforced before any recompute loop (legal max is
#: 24 candidates x 100 fixtures -- the choice-audit precedent; the
#: generated records cap is 24 candidates x 24 cases).
MAX_RECORDS = 2400
MAX_CONTROL_RECORDS = 24 * 24
MAX_RECORDS_BYTES = 12 * 1024 * 1024

#: Generated planted-fact cases per candidate (the placement
#: precedent's generated-instrument size).
CASE_COUNT = 24

#: The falsification boolean is withheld below this many paired
#: compliant generated cells (placement's MIN_PAIRED_EVALUABLE = 6
#: precedent: floors gate readouts, never reported counts).
MIN_PAIRED_COMPLIANT = 6

#: Joiner between a fixture's last user content and the suffix
#: (dateline's JOINER precedent).
JOINER = "\n\n"

#: The frozen elicitation suffix, appended to the last user message
#: of every treated fixture and generated case. Its final line is a
#: placeholder, not a grammar line: "Confidence: <number>" does not
#: match the grammar below, so a model echoing the instruction
#: verbatim is malformed_line, never compliant.
SUFFIX = (
    "Answer the question, then state your confidence that your answer is "
    "correct. Put the confidence on its own final line as a whole number "
    "from 0 to 100, in exactly this form:\nConfidence: <number>"
)

#: The frozen line grammar (ASCII digits only, no sign, no leading
#: zeros, no surrounding whitespace). Load-bearing: int() accepts
#: "+85", "8_5", and non-ASCII digits, and an unflagged \d matches
#: fullwidth digits -- a strip-and-int parser would manufacture
#: compliance the grammar does not grant.
_GRAMMAR_RE = re.compile(r"^Confidence: (0|[1-9][0-9]?|100)$")

#: Parser outcome codes (a closed vocabulary; the parser is total).
CODES: tuple[str, ...] = (
    "ok",
    "missing_line",
    "malformed_line",
    "multiple_lines",
    "confidence_not_final",
)

#: Fixture screen classes (a screened fixture is never treated; its
#: record carries the class and no cell). Closed vocabulary.
SCREEN_CLASSES: tuple[str, ...] = (
    "expected_blank",
    "expected_confidence_shaped",
    "vacuous_last_user",
    "no_user_turn",
)

#: Frozen calibration-table bin edges over the stated integer.
BIN_EDGES: tuple[tuple[int, int], ...] = ((0, 24), (25, 49), (50, 74), (75, 100))

#: Characters of a malformed final line carried in the record (the
#: classification prefix "Confidence:" is 11; a grammar line is at
#: most 15, so 32 re-verifies the class exactly).
CARRIED_PREFIX_CHARS = 32

#: Gap withholding reasons (the precision WITHHELD_* discipline: a
#: withheld readout names why, in a closed vocabulary).
GAP_NO_COMPLIANT = "no_compliant_cells"
GAP_NO_CORRECT = "no_correct_compliant"
GAP_NO_INCORRECT = "no_incorrect_compliant"

#: Falsification withholding reason.
FALSIFICATION_INSUFFICIENT = "insufficient_paired_compliant"

_HEX64 = frozenset("0123456789abcdef")


def text_digest(text: str) -> str:
    """SHA-256 of a text (records carry digests, not texts).

    ``surrogatepass`` keeps a pathological text (a lone surrogate can
    survive the CSV loader) from crashing the probe; for every
    well-formed string the digest is the plain UTF-8 digest. Same
    construction as precision.text_digest.
    """
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def fixture_set_sha256(fixture_ids: Sequence[str]) -> str:
    """SHA-256 over the sorted fixture id set (leading's form)."""
    return hashlib.sha256("\n".join(sorted(fixture_ids)).encode("utf-8")).hexdigest()


def is_digest(value: Any) -> bool:
    """True for a lowercase hex SHA-256 digest string."""
    return type(value) is str and len(value) == 64 and all(c in _HEX64 for c in value)


def is_grammar_line(line: str) -> bool:
    """True when a single line matches the frozen grammar exactly."""
    return _GRAMMAR_RE.match(line) is not None


def _final_line(answer: str) -> str:
    r"""The answer's final line under the parser's line convention.

    Lines come from split("\n") with exactly one trailing empty
    element dropped (the text ended with exactly one trailing
    newline). splitlines() is never used: it also splits on U+2028,
    U+0085 and friends, which would manufacture a final line the
    frozen convention does not see.
    """
    lines = answer.split("\n")
    if answer.endswith("\n"):
        lines.pop()
    return lines[-1] if lines else ""


def parse_verbalized_confidence(answer: str) -> tuple[str, int | None, str]:
    """Classify one elicited generation: (code, confidence, span).

    Total over all strings, pure ``str`` (never encodes, so a lone
    surrogate in the text cannot raise). Precedence is final-line
    first, adjudicated in cycle 199: the model's last act decides --
    a final line matching the grammar classifies the cell (``ok``
    when it is the only grammar line, ``multiple_lines`` when any
    earlier line also matches); a prefix-shaped final line is
    ``malformed_line`` even when an earlier line matched; a grammar
    line anywhere else is ``confidence_not_final``; anything else is
    ``missing_line``. The span is the answer with the compliant line
    removed for ``ok`` cells, and the whole answer otherwise.
    """
    lines = answer.split("\n")
    if answer.endswith("\n"):
        lines.pop()
    final = lines[-1] if lines else ""
    grammar_lines = sum(1 for line in lines if _GRAMMAR_RE.match(line))
    match = _GRAMMAR_RE.match(final)
    if match is not None:
        if grammar_lines >= 2:
            return ("multiple_lines", None, answer)
        return ("ok", int(match.group(1)), "\n".join(lines[:-1]))
    if final.startswith("Confidence:") or final.lower().startswith("confidence:"):
        return ("malformed_line", None, answer)
    if grammar_lines >= 1:
        return ("confidence_not_final", None, answer)
    return ("missing_line", None, answer)


def carried_line(answer: str, code: str) -> tuple[str | None, int | None]:
    """The record's carried final line and its full length, per code.

    ``ok`` cells carry the grammar line verbatim (a bijection with
    the confidence scalar). ``malformed_line`` cells carry the first
    CARRIED_PREFIX_CHARS characters plus the full length. Every other
    code's final line is ordinary answer text and is never carried.
    """
    if code == "ok":
        line = _final_line(answer)
        return (line, len(line))
    if code == "malformed_line":
        line = _final_line(answer)
        return (line[:CARRIED_PREFIX_CHARS], len(line))
    return (None, None)


def screen_fixture(case: Fixture) -> str:
    """The fixture's screen class, or "" when it may be treated.

    A fixture is screened out of every elicitation population when
    its expected answer is blank under ``strip()`` (no honest span
    exists), when its expected answer already contains a grammar
    line anywhere (an honest generation would gain a second line
    through no act of the model's), or when its last user message
    is vacuous or absent (the suffix would be the whole turn, which
    measures suffix echo, not elicitation). Screens are properties
    of the fixture, identical for every candidate.
    """
    if not case.expected.strip():
        return "expected_blank"
    if any(_GRAMMAR_RE.match(line) for line in case.expected.split("\n")):
        return "expected_confidence_shaped"
    try:
        index = target_index(case)
    except ValueError:
        return "no_user_turn"
    if not case.messages[index].content.strip():
        return "vacuous_last_user"
    return ""


def treated_fixture(case: Fixture) -> Fixture:
    """The fixture with the frozen suffix appended to its last user message.

    Same id, group, and expected; only the target message's content
    changes (``content + JOINER + SUFFIX``), the dateline surgery
    pattern, so the per-fixture seed is identical across arms.
    """
    index = target_index(case)
    messages = list(case.messages)
    original = messages[index]
    messages[index] = type(original)(role=original.role, content=original.content + JOINER + SUFFIX)
    return Fixture(
        id=case.id,
        messages=tuple(messages),
        expected=case.expected,
        group=case.group,
    )


def _derive(seed: int, tag: str, index: int, attempt: int = 0) -> str:
    """Return the derive."""
    material = f"{seed}\0confidence-{tag}\0{index}\0{attempt}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def case_material(seed: int) -> list[tuple[str, str]]:
    """The generated cases' (key, value) pairs, derived from the seed.

    Keys are synthetic identifiers, values are 6-hex strings; both
    are pairwise distinct across the case set by construction (the
    deterministic re-derivation loop below), because a duplicated
    value would let the donor-key control be answered correctly by
    copying, and a duplicated key would corrupt a knowable arm.
    Stated values are unguessable (16^-6) and unmemorizable (seeded),
    so the knowable arm measures reading, the control its absence.
    """
    pairs: list[tuple[str, str]] = []
    used_keys: set[str] = set()
    used_values: set[str] = set()
    for index in range(CASE_COUNT):
        attempt = 0
        key = "key-" + _derive(seed, "key", index)[:8]
        while key in used_keys:
            attempt += 1
            key = "key-" + _derive(seed, "key", index, attempt)[:8]
        used_keys.add(key)
        attempt = 0
        value = _derive(seed, "value", index)[:6]
        while value in used_values:
            attempt += 1
            value = _derive(seed, "value", index, attempt)[:6]
        used_values.add(value)
        pairs.append((key, value))
    return pairs


def case_message(asked_key: str, bound_key: str, bound_value: str) -> str:
    """One generated case's user message: a binding, then the question."""
    return (
        f"Reference record: the code for {bound_key} is {bound_value}."
        f"\n\nWhat is the code for {asked_key}? Answer with the code only."
    )


def build_cases(seed: int, *, control: bool) -> tuple[Fixture, ...]:
    """The generated fixtures for one arm (knowable or donor-key control).

    Both arms share case ids and expected values; they differ only
    in which key the payload binds. The knowable arm binds the asked
    key to its true value. The control arm binds the *next* case's
    key to that case's value: nothing false is stated about the
    asked key, and no binding for it exists in context -- a model
    confidently answering the donor value there is copying the only
    value it can see, which is the answer-independence the arm
    exists to detect.
    """
    pairs = case_material(seed)
    out: list[Fixture] = []
    for index, (key, value) in enumerate(pairs):
        if control:
            bound_key, bound_value = pairs[(index + 1) % CASE_COUNT]
        else:
            bound_key, bound_value = key, value
        out.append(
            Fixture(
                id=f"confidence-case-{index:02d}",
                messages=(Message(role="user", content=case_message(key, bound_key, bound_value)),),
                expected=value,
                group="confidence-control",
            )
        )
    return tuple(out)


def case_protocol(seed: int) -> list[dict[str, Any]]:
    """The generated case set as protocol data: digests only, no texts."""
    pairs = case_material(seed)
    out: list[dict[str, Any]] = []
    for index, (key, value) in enumerate(pairs):
        donor_key, donor_value = pairs[(index + 1) % CASE_COUNT]
        out.append(
            {
                "id": f"confidence-case-{index:02d}",
                "key_digest": text_digest(key),
                "expected_digest": text_digest(value),
                "knowable_payload_digest": text_digest(case_message(key, key, value)),
                "control_payload_digest": text_digest(case_message(key, donor_key, donor_value)),
            }
        )
    return out


def protocol_object(
    *,
    seed: int,
    max_new_tokens: int,
    fixture_ids: Sequence[str],
    candidates: Sequence[str],
    boundaries: dict[str, Sequence[int]],
    backend_fingerprint: str,
    screens: dict[str, str],
    cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """The frozen protocol as data; the fingerprint covers it."""
    return {
        "version": PROTOCOL_VERSION,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "max_records": MAX_RECORDS,
        "max_control_records": MAX_CONTROL_RECORDS,
        "max_records_bytes": MAX_RECORDS_BYTES,
        "case_count": CASE_COUNT,
        "min_paired_compliant": MIN_PAIRED_COMPLIANT,
        "bin_edges": [list(edge) for edge in BIN_EDGES],
        "suffix_sha256": text_digest(SUFFIX),
        "fixture_set_sha256": fixture_set_sha256(fixture_ids),
        "candidates": list(candidates),
        "boundaries": {name: list(boundaries[name]) for name in candidates},
        "backend_fingerprint": backend_fingerprint,
        "screens": dict(screens),
        "control_cases": [dict(entry) for entry in cases],
    }


def protocol_fingerprint(protocol: dict[str, Any]) -> str:
    """SHA-256 over the canonical protocol object (fingerprint field excluded)."""
    body = {key: value for key, value in protocol.items() if key != "fingerprint"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cell_verdict(cell: dict[str, Any], expected_digest: str) -> bool:
    """The verdict binding: span-digest equality plus the boundary.

    One definition for every population: the parser's span (the
    answer with a compliant line removed, else the whole answer)
    must digest-match the expected answer and the response boundary
    must have been found. Compliance is a separate axis (the joint
    table keeps the two apart); a correct answer that never stated
    a number is correct *and* non-compliant.
    """
    return bool(cell["span_digest"] == expected_digest and cell["boundary_found"])


def _mean(values: list[int]) -> float | None:
    """Return the mean."""
    return (sum(values) / len(values)) if values else None


def _population_block(cells: Sequence[dict[str, Any]], n_population: int) -> dict[str, Any]:
    """Every readout over one population of completed cells.

    ``cells`` are the completed cells only (a fixture an arm never
    reached is absent, never a failure); ``n_population`` is the
    population's denominator (unscreened fixtures, or CASE_COUNT).
    """
    compliant = [cell for cell in cells if cell["code"] == "ok"]
    correct = [cell for cell in compliant if cell["verdict"]]
    incorrect = [cell for cell in compliant if not cell["verdict"]]
    mean_correct = _mean([int(cell["confidence"]) for cell in correct])
    mean_incorrect = _mean([int(cell["confidence"]) for cell in incorrect])
    if not compliant:
        gap: float | None = None
        gap_detail = GAP_NO_COMPLIANT
    elif mean_correct is None:
        gap = None
        gap_detail = GAP_NO_CORRECT
    elif mean_incorrect is None:
        gap = None
        gap_detail = GAP_NO_INCORRECT
    else:
        gap = mean_correct - mean_incorrect
        gap_detail = ""
    bins: list[dict[str, Any]] = []
    for lo, hi in BIN_EDGES:
        members = [cell for cell in compliant if lo <= int(cell["confidence"]) <= hi]
        n_correct = sum(1 for cell in members if cell["verdict"])
        bins.append(
            {
                "lo": lo,
                "hi": hi,
                "n": len(members),
                "n_correct": n_correct,
                "accuracy": (n_correct / len(members)) if members else None,
            }
        )
    return {
        "n_population": n_population,
        "n_completed": len(cells),
        "n_compliant": len(compliant),
        "compliance_rate": (len(compliant) / len(cells)) if cells else None,
        "mean_confidence": _mean([int(cell["confidence"]) for cell in compliant]),
        "mean_correct": mean_correct,
        "mean_incorrect": mean_incorrect,
        "gap": gap,
        "gap_detail": gap_detail,
        "joint": {
            "compliant_correct": len(correct),
            "compliant_incorrect": len(incorrect),
            "noncompliant": len(cells) - len(compliant),
        },
        "bins": bins,
    }


def _candidate_block(
    records: Sequence[dict[str, Any]],
    control_records: Sequence[dict[str, Any]],
    candidate: str,
    n_operator_population: int,
) -> dict[str, Any]:
    """Every per-candidate derived value, from the records alone."""
    own = [r for r in records if r["candidate"] == candidate]
    elicited = [r["elicited"] for r in own if type(r["elicited"]) is dict]
    operator = _population_block(elicited, n_operator_population)
    # skipcq: FLK-E501 - ruff's E501 exemptions cover this frozen, test-pinned literal
    flips = {"n_compared": 0, "pass_to_fail": 0, "fail_to_pass": 0}  # nosec B105 -- 'pass' keys are verdict-flip counts in the frozen payload schema, not credentials
    for record in own:
        cell = record["elicited"]
        native = record["native_verdict"]
        if type(cell) is not dict or type(native) is not bool:
            continue
        flips["n_compared"] += 1
        if native and not cell["verdict"]:
            flips["pass_to_fail"] += 1
        elif not native and cell["verdict"]:
            flips["fail_to_pass"] += 1
    operator["flips"] = flips
    own_control = [r for r in control_records if r["candidate"] == candidate]
    knowable_cells = [r["knowable"] for r in own_control if type(r["knowable"]) is dict]
    control_cells = [r["control"] for r in own_control if type(r["control"]) is dict]
    knowable = _population_block(knowable_cells, CASE_COUNT)
    control = _population_block(control_cells, CASE_COUNT)
    paired = [
        r
        for r in own_control
        if type(r["knowable"]) is dict
        and type(r["control"]) is dict
        and r["knowable"]["code"] == "ok"
        and r["control"]["code"] == "ok"
    ]
    if len(paired) >= MIN_PAIRED_COMPLIANT:
        knowable_mean = _mean([int(r["knowable"]["confidence"]) for r in paired])
        control_mean = _mean([int(r["control"]["confidence"]) for r in paired])
        knowability: dict[str, Any] = {
            "n_paired_compliant": len(paired),
            "knowable_mean": knowable_mean,
            "control_mean": control_mean,
            "difference": float(knowable_mean) - float(control_mean)
            if knowable_mean is not None and control_mean is not None
            else None,
            "falsified": bool(
                control_mean is not None
                and knowable_mean is not None
                and control_mean >= knowable_mean
            ),
            "detail": "",
        }
    else:
        knowability = {
            "n_paired_compliant": len(paired),
            "knowable_mean": None,
            "control_mean": None,
            "difference": None,
            "falsified": None,
            "detail": FALSIFICATION_INSUFFICIENT,
        }
    return {
        "operator": operator,
        "generated": {
            "knowable": knowable,
            "control": control,
            "knowability": knowability,
        },
    }


def compute_derived(
    records: Sequence[dict[str, Any]],
    control_records: Sequence[dict[str, Any]],
    candidates: Sequence[str],
    n_operator_population: int,
) -> dict[str, Any]:
    """All derived payload fields from the ground records.

    Everything a reader sees -- compliance, the joint table, means,
    the gap, the bins, flips, and the knowability readout -- is
    computed here from the records and recomputed by the diff
    validator's independent mirror with type-strict equality.
    """
    ordered = sorted(records, key=lambda r: (str(r["candidate"]), str(r["fixture"])))
    ordered_control = sorted(control_records, key=lambda r: (str(r["candidate"]), str(r["case"])))
    blocks = {
        name: _candidate_block(ordered, ordered_control, name, n_operator_population)
        for name in candidates
    }
    return {
        "candidates": blocks,
        "totals": {
            "n_records": len(records),
            "n_control_records": len(control_records),
        },
    }


__all__ = [
    "BIN_EDGES",
    "CARRIED_PREFIX_CHARS",
    "CASE_COUNT",
    "CODES",
    "FALSIFICATION_INSUFFICIENT",
    "GAP_NO_COMPLIANT",
    "GAP_NO_CORRECT",
    "GAP_NO_INCORRECT",
    "JOINER",
    "MAX_CONTROL_RECORDS",
    "MAX_RECORDS",
    "MAX_RECORDS_BYTES",
    "MIN_PAIRED_COMPLIANT",
    "PROTOCOL_VERSION",
    "SCREEN_CLASSES",
    "SUFFIX",
    "build_cases",
    "carried_line",
    "case_message",
    "case_protocol",
    "cell_verdict",
    "compute_derived",
    "fixture_set_sha256",
    "is_digest",
    "is_grammar_line",
    "parse_verbalized_confidence",
    "protocol_fingerprint",
    "protocol_object",
    "screen_fixture",
    "text_digest",
    "treated_fixture",
]
