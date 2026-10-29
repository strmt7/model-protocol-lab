r"""Instruction-distance probe (cycle 218, research wave-11
rank 2; fresh review BUILD-as-amended, coordinator #59).

One frozen format directive -- "Reply with only the number: no
units, no sentence, nothing else." -- governs a fixed
fact+question block on generated cases. Six arms per case vary
only where neutral filler sits relative to the directive and
the block it governs: adjacent (directive, block), separated at
two doses (directive, filler, block), displaced at the same two
doses (directive, block, filler), and a descriptive order
control (block, directive). "Distance" in this probe means
rendered tokens between the frozen directive and the question
block, verified per candidate on its own rendered tokens --
never edit distance. A dose is an exact rendered-token delta
over the case's adjacent render, sized from the neutral corpus
by the single-phase rule below; a cell whose delta cannot be
sized exactly is ``arm_unmatchable``, never approximated.

Binding design pins (docs/INSTRUCTIONDISTANCE_DESIGN.md, the
candidate text as amended by the fresh review's amendments
A--J):

* Scorer: the judged span is ``core.evaluate``'s pre-boundary
  answer. The byte-exact verdict (span == expected, conjoined
  with boundary presence) is descriptive only. The headline is
  compliance under the frozen grammar: a completed cell is
  compliant iff its span matches ``\A[0-9]+\Z`` (fullmatch;
  ASCII digits only -- amendment B2/B3). ``digit_count`` and
  first-run extraction use the same ASCII alphabet.
* Decomposition (exactly one class per completed cell):
  ``compliant_correct``, ``compliant_wrong``,
  ``noncompliant_correct_value`` (first maximal ASCII digit
  run equals the expected value), ``noncompliant_wrong``, and
  ``boundary_only``. Boundary disposition per negation section
  4.2 as shipped (amendment B1): boundary-missing cells are
  grammar-classified over the whole-output span; a
  boundary-missing cell whose span is compliant-correct is
  ``boundary_only`` -- boundary attrition only, never a
  compliance class, never counted toward S or T. Completion =
  a span was produced.
* Compliance is derived from digests, never from bare
  integers (amendment C): compliant iff ``answer_digest`` is
  not the digest of the empty string AND the
  non-digit-projection digest IS the digest of the empty
  string. One record field (``nondigit_projection_digest``)
  binds it; ``answer_len`` / ``digit_count`` are descriptive
  only. ``first_digit_run_digest`` is "" when the span has no
  ASCII digit run (amendment B3).
* Decision rule per candidate per dose, over cases completed
  in the adjacent, separated, and displaced arms of that
  dose: D = adjacent ``compliant_correct``; S = separated
  noncompliant in D; T = displaced noncompliant in D. Fire
  ``instruction_distance_drop_observed`` iff D >= 10 and
  S >= 6 and S - T >= 5; D < 10 -> ``insufficient``; else
  ``below_threshold``. The label is an excess-of-between-
  losses label, never a distance effect simpliciter: a pure
  constraint-to-generation effect yields S = T and reads
  ``below_threshold`` even at total collapse (amendment F).
  Reverse discordants (adjacent noncompliant, separated
  compliant) are printed beside the decision, never pooled
  into the rule. Discordance is never pooled across
  candidates. No winner/selection readout.
* The instrument is a per-dose large-drop falsifier
  (amendment F): exact power 0.9088 at a 50% conditional drop
  rate, 0.2460 at 0.25 -- the 0.25 row is below this panel's
  resolution and is labelled as such wherever the operating
  characteristics are printed. No curve, threshold, or decay
  language anywhere: three same-turn layouts at two doses.
* Screening (amendment D): per-cell ``arm_screened`` when a
  constructed filler span contains the case value
  (case-folded) or the candidate's decoded boundary text;
  stage-level, the frozen texts are audited for special-token
  literals against every candidate format and the generated
  texts for boundary-text containment, either failing the
  whole stage closed before any generation.
* Sizing economy (amendment J): 4 sized cells per
  candidate-case (2,304 panel-wide); expected about 56K
  tokenizer encodes at the shipped ``span_candidates``
  estimator's executed mean (24.5 probes per cell), worst case
  2,304 x 512 = 1,179,648 encodes, plus per-generation
  encodes. Generations: 6 x 24 x 24 = 3,456 maximum; records
  576. The sizing rule drops needle's in-span
  offset-verification phase: this instrument has no offset
  variable (filler sits wholly between the directive and the
  block, or wholly after it) and the "\n" joiner insulates
  the span, so delta-exactness on the final render is the
  only acceptance condition.
* Generated cases only, stage-local, never the user's Split.
  Records carry digests only. The frozen constraint is the
  only directive in any prompt, by construction.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

__all__ = [
    "ARM_KEYS",
    "CASE_COUNT",
    "CELL_DETAILS",
    "CONSTRAINT",
    "DECISIONS",
    "DECOMPOSITION_CLASSES",
    "DOSES",
    "FACT_TEMPLATE",
    "GENERATION_BUDGET",
    "GROUP",
    "JOINER",
    "MAX_GROUND_RECORDS",
    "MAX_RECORDS_BYTES",
    "MIN_D",
    "MIN_MARGIN",
    "MIN_S",
    "OPERATING_CHARACTERISTICS",
    "PROTOCOL_VERSION",
    "QUESTION_TEMPLATE",
    "SCREENING_RULE_TEXT",
    "SIZED_ARMS",
    "SIZING_ENCODE_BUDGET",
    "SIZING_RULE_TEXT",
    "STAGE_DETAILS",
    "STREAM_CYCLES",
    "TOPICS",
    "aggregate_candidate",
    "arm_content",
    "arm_dose",
    "block_text",
    "canonical_generated_rows",
    "case_material_sha256",
    "classify_instruction_distance",
    "decompose_cell",
    "decompose_span",
    "digest_ids",
    "digest_text",
    "digit_count",
    "filler_screened",
    "filler_stream",
    "first_digit_run",
    "first_digit_run_digest",
    "generate_cases",
    "ground_sha256",
    "nondigit_projection",
    "nondigit_projection_digest",
    "protocol_fingerprint",
    "protocol_object",
    "screening_rule_sha256",
    "sizing_rule_sha256",
    "span_compliant",
]

PROTOCOL_VERSION = 1

GROUP = "instruction_distance"

ARM_KEYS: tuple[str, ...] = (
    "adj",
    "sep_256",
    "sep_1024",
    "disp_256",
    "disp_1024",
    "adj_after",
)
BASELINE_ARM = "adj"
DOSES: tuple[int, ...] = (256, 1024)
#: Arms whose filler span is sized per candidate-case.
SIZED_ARMS: tuple[str, ...] = ("sep_256", "sep_1024", "disp_256", "disp_1024")
#: Arms entering a dose decision (the headline population is
#: completion in all three).
HEADLINE_ARMS: tuple[str, ...] = ("adj", "sep_256", "sep_1024", "disp_256", "disp_1024")

JOINER = "\n"

CASE_COUNT = 24

#: The frozen constraint (64 characters; design section 3.2 as
#: amended by amendment A). Byte-identical in every arm that
#: carries it; the only directive in any prompt.
CONSTRAINT = "Reply with only the number: no units, no sentence, nothing else."

FACT_TEMPLATE = "The {topic} number is {value}."
QUESTION_TEMPLATE = "What is the {topic} number?"

#: Frozen topics, one per case. Verified absent from the
#: neutral corpus as case-folded substrings (pinned by test):
#: a topic occurring in the filler could prime or mask the
#: fact the case plants.
TOPICS: tuple[str, ...] = (
    "harbor",
    "summit",
    "quarry",
    "tundra",
    "granite",
    "willow",
    "copper",
    "saddle",
    "walnut",
    "velvet",
    "cobalt",
    "marble",
    "cinder",
    "falcon",
    "garnet",
    "saffron",
    "indigo",
    "juniper",
    "lagoon",
    "magnet",
    "nickel",
    "onyx",
    "pepper",
    "quartz",
)

#: Decision floors, frozen by the design (section 6, derived
#: for this panel by exact enumeration). They gate the section
#: label only, never a reported count or rate.
MIN_D = 10
MIN_S = 6
MIN_MARGIN = 5

#: 24 candidates (run_experiment's operative cap,
#: experiments.py) x 24 cases; six generations per
#: (candidate, case).
MAX_GROUND_RECORDS = 24 * CASE_COUNT
GENERATION_BUDGET = len(ARM_KEYS) * MAX_GROUND_RECORDS
#: Records carry six cells with answer digests plus the
#: span-primitive digests; the bound is the house 12 MiB for
#: uniformity with the sibling probes.
MAX_RECORDS_BYTES = 12 * 1024 * 1024

#: The section decision vocabulary (a total function of D, S,
#: and T -- see classify_instruction_distance). Section labels
#: only, never cell or stage details.
DECISIONS: tuple[str, ...] = (
    "insufficient",
    "instruction_distance_drop_observed",
    "below_threshold",
)

#: The decomposition classes of a completed cell, exactly one
#: per cell (see decompose_cell).
DECOMPOSITION_CLASSES: tuple[str, ...] = (
    "compliant_correct",
    "compliant_wrong",
    "noncompliant_correct_value",
    "noncompliant_wrong",
    "boundary_only",
)

#: Classes that count as a compliance loss in S and T.
NONCOMPLIANT_CLASSES: tuple[str, ...] = (
    "noncompliant_correct_value",
    "noncompliant_wrong",
)
#: Classes that count as compliant (the 2x2's compliant row).
COMPLIANT_CLASSES: tuple[str, ...] = ("compliant_correct", "compliant_wrong")

CELL_DETAILS: tuple[str, ...] = (
    "",
    "arm_context_exceeded",
    "arm_evaluation_failed",
    "arm_unmatchable",
    "arm_screened",
)

STAGE_DETAILS: tuple[str, ...] = (
    "cancelled",
    "instruction_distance_record_budget_exceeded",
    "instruction_distance_screen_failed",
)

#: The sizing rule, frozen as text: the protocol carries the
#: SHA-256 of this string, so a bare version number never has
#: to stand in for the rule itself. Amendment J/A-K5c: needle's
#: in-span offset-verification phase is omitted BY NAME --
#: there is no offset variable here, and the joiner insulates
#: the span from the block.
SIZING_RULE_TEXT = (
    "instructiondistance-single-phase-exact-v1: spans are whole "
    "sentences plus one truncated suffix (span_text) over the "
    "cycled neutral corpus stream, every sized cell searching "
    "from the stream head; candidates are probed in "
    "span_candidates closeness order against the dose, each "
    "probe rendering the full arm content under the candidate's "
    "own template, accepting the first whose rendered token "
    "count minus the case's adjacent-arm rendered token count "
    "equals the dose exactly, verified on the accepted "
    "construction's final ids (the probe render IS the arm "
    "render, so acceptance and verification are one encode). "
    "One encode per probe; the per-cell encode budget is 512 "
    "and a cell with no exact construction inside it is "
    "arm_unmatchable, never approximated. Unlike needle's "
    "phased rule there is no in-span offset-verification phase: "
    "this instrument has no offset variable (the filler sits "
    "wholly between the directive and the block, or wholly "
    "after it) and the newline joiner insulates the span from "
    "the block it precedes or follows, so delta-exactness on "
    "the final render is the only acceptance condition."
)
SIZING_ENCODE_BUDGET = 512

#: The screening rule, frozen as text on the same terms
#: (amendment D: per-cell arm_screened plus the stage-level
#: fail-closed screens).
SCREENING_RULE_TEXT = (
    "instructiondistance-screen-v1: per constructed filler "
    "span, case-folded: the span is screened (the cell is "
    "arm_screened) when it contains the case value, or when "
    "the candidate's decoded boundary text occurs in it. The "
    "constraint and the case block are the treatment and are "
    "never screened per-cell. Stage-level, the frozen texts "
    "(constraint, fact and question templates, topics, neutral "
    "corpus) are audited for special-token literals against "
    "every candidate format, and the generated texts "
    "(constraint, templates with slots emptied, topics) are "
    "checked for containment of any candidate's decoded "
    "boundary text; a hit fails the whole stage closed before "
    "any generation."
)

#: The filler stream is the neutral corpus cycled this many
#: times (needle's settlement: one pass is about 1,278 tokens
#: under a typical tokenizer, so the stream must be long
#: enough that a 1024-dose construction never exhausts it).
STREAM_CYCLES = 6

#: Operating characteristics of the frozen (10, 6, 5) rule at
#: N = 24, from the design section 6 as corrected by amendment
#: E (exact enumeration; four independent derivations agree).
#: Printed beside every decision; never re-derived at build
#: time. The pure-length bound carries its premise: the
#: independent-cells figure is the worst case only under
#: non-negative correlation between the separated and
#: displaced losses; the negative-rho row is the executed
#: breach of the uncorrected bound.
OPERATING_CHARACTERISTICS: dict[str, Any] = {
    "random_labeller_fire_rate": 0.0013,
    "no_effect_competent_fire_rate": 0.0058,
    "no_effect_paired_max_fire_rate": 0.0077,
    "power_drop_rate_0_25": 0.246,
    "power_drop_rate_0_50": 0.9088,
    "power_drop_rate_0_75": 0.9993,
    "pure_length_independent_fire_rate": 0.0788,
    "pure_length_fully_correlated_fire_rate": 0.0,
    "pure_length_negative_rho_max_fire_rate": 0.133,
    "familywise_decisions": 48,
    "familywise_no_effect_expected": 0.28,
    "familywise_no_effect_p_at_least_one": 0.24,
    "familywise_frailty_expected_low": 0.40,
    "familywise_frailty_expected_high": 0.59,
    "familywise_frailty_p_low": 0.33,
    "familywise_frailty_p_high": 0.44,
    "familywise_pure_length_expected": 3.78,
    "familywise_pure_length_p_at_least_one": 0.98,
    "verdict_availability_at_adjacent_cc_0_40": 0.511,
    "verdict_availability_at_adjacent_cc_0_30": 0.153,
}

_ASCII_DIGITS = "0123456789"
_DIGIT_RUN_RE = re.compile(r"[0-9]+")


def digest_text(text: str) -> str:
    """SHA-256 hex digest of a text (the records' digest form)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_ids(ids: tuple[int, ...]) -> str:
    """SHA-256 hex digest of a render id sequence, so records
    carry the sequence's identity without its contents.
    """
    canonical = ",".join(str(i) for i in ids)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


EMPTY_DIGEST = digest_text("")


def _derive(seed: int, index: int, slot: int, counter: int) -> int:
    """One deterministic draw. The material string, the slot
    meaning (0 = value), and the redraw rule in
    :func:`generate_cases` are frozen together -- the diff
    validator's mirror reproduces them exactly.
    """
    material = f"{seed}|instructiondistance|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def generate_cases(seed: int) -> list[dict[str, Any]]:
    """The frozen generated case set for a run seed, in
    canonical index order. Pure and deterministic -- the same
    seed regenerates the identical set across runs and
    PYTHONHASHSEED values. A value colliding with an earlier
    case's value redraws at the next counter, so the 24 values
    (3-digit integers, 100--999) are pairwise distinct by
    construction and a closed-book guess is a 1-in-900 event
    before any model prior.
    """
    if type(seed) is not int:
        raise ValueError("instruction-distance seed must be an int")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(CASE_COUNT):
        topic = TOPICS[index]
        counter = 0
        while True:
            value = str(100 + _derive(seed, index, 0, counter) % 900)
            if value not in seen:
                break
            counter += 1
        seen.add(value)
        fact = FACT_TEMPLATE.format(topic=topic, value=value)
        question = QUESTION_TEMPLATE.format(topic=topic)
        cases.append(
            {
                "fixture_id": f"instructiondistance-{index:02d}",
                "topic": topic,
                "value": value,
                "fact": fact,
                "question": question,
                "block": fact + JOINER + question,
                "expected": value,
            }
        )
    return cases


def block_text(case: dict[str, Any]) -> str:
    """The case's governed block B: fact line + question line,
    byte-identical in every arm.
    """
    return str(case["fact"]) + JOINER + str(case["question"])


def arm_content(case: dict[str, Any], arm: str, span: str = "") -> str:
    r"""One arm's user-turn content. Blocks join with the
    distraction JOINER ("\n"); the constraint and the block
    are byte-identical wherever they appear. ``span`` is the
    sized filler text for the sized arms and is ignored by
    the unsized arms.
    """
    block = block_text(case)
    if arm == "adj":
        return CONSTRAINT + JOINER + block
    if arm in ("sep_256", "sep_1024"):
        return CONSTRAINT + JOINER + span + JOINER + block
    if arm in ("disp_256", "disp_1024"):
        return CONSTRAINT + JOINER + block + JOINER + span
    if arm == "adj_after":
        return block + JOINER + CONSTRAINT
    raise ValueError("unknown instruction-distance arm")


def arm_dose(arm: str) -> int:
    """The dose of a sized arm (0 for the unsized arms)."""
    if arm in ("sep_256", "disp_256"):
        return 256
    if arm in ("sep_1024", "disp_1024"):
        return 1024
    if arm in ARM_KEYS:
        return 0
    raise ValueError("unknown instruction-distance arm")


def canonical_generated_rows(seed: int) -> list[list[Any]]:
    """The canonical generated set the ground digest binds."""
    return [[case["fixture_id"], case["topic"], case["value"]] for case in generate_cases(seed)]


def ground_sha256(seed: int) -> str:
    """Digest of the canonical generated rows for a seed: two
    runs share an instrument only when this matches, because
    the generated set is the treatment.
    """
    canonical = json.dumps(canonical_generated_rows(seed), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def case_material_sha256() -> str:
    """Digest of the frozen case material (the constraint, the
    templates, the topics): the material is bound by digest in
    the protocol, so a payload cannot quietly carry a different
    instrument under the same version number.
    """
    canonical = json.dumps(
        [CONSTRAINT, FACT_TEMPLATE, QUESTION_TEMPLATE, list(TOPICS)],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sizing_rule_sha256() -> str:
    """Digest of the frozen sizing-rule text."""
    return digest_text(SIZING_RULE_TEXT)


def screening_rule_sha256() -> str:
    """Digest of the frozen screening-rule text."""
    return digest_text(SCREENING_RULE_TEXT)


def protocol_object(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The frozen instrument description a payload is checked
    against: arms, doses, case material, rule digests, floors,
    budgets, and the seed/backend pair that generated the set.
    """
    return {
        "version": PROTOCOL_VERSION,
        "arm_keys": list(ARM_KEYS),
        "doses": list(DOSES),
        "baseline_arm": BASELINE_ARM,
        "case_count": CASE_COUNT,
        "case_material_sha256": case_material_sha256(),
        "ground_sha256": ground_sha256(seed),
        "sizing_rule_sha256": sizing_rule_sha256(),
        "sizing_encode_budget": SIZING_ENCODE_BUDGET,
        "screening_rule_sha256": screening_rule_sha256(),
        "stream_cycles": STREAM_CYCLES,
        "min_d": MIN_D,
        "min_s": MIN_S,
        "min_margin": MIN_MARGIN,
        "generation_budget": GENERATION_BUDGET,
        "max_ground_records": MAX_GROUND_RECORDS,
        "max_records_bytes": MAX_RECORDS_BYTES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }


def protocol_fingerprint(seed: int, backend_fingerprint: str) -> str:
    """SHA-256 over the canonical protocol object."""
    canonical = json.dumps(
        protocol_object(seed, backend_fingerprint),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def filler_stream() -> tuple[str, ...]:
    """The filler sentence stream: the neutral corpus cycled
    ``STREAM_CYCLES`` times, in order.
    """
    from .distraction import NEUTRAL_CORPUS

    return tuple(
        NEUTRAL_CORPUS[i % len(NEUTRAL_CORPUS)] for i in range(STREAM_CYCLES * len(NEUTRAL_CORPUS))
    )


def filler_screened(span: str, value: str, boundary_text: str) -> bool:
    """Per-span screen (screening rule
    instructiondistance-screen-v1). True when the case value
    could be copied from this span (case-folded containment)
    or when the candidate's decoded boundary text occurs in
    it (the filler would carry the terminator). The value is
    a 3-digit string and the corpus carries no digits at all,
    so the value screen is a guard against a future corpus
    edit, not a live filter -- stated so nobody mistakes it
    for one.
    """
    if value and value.casefold() in span.casefold():
        return True
    return bool(boundary_text) and boundary_text in span


# ---------------------------------------------------------------------
# Span primitives (frozen grammar; ASCII digit alphabet only,
# amendments B2/B3). Every §5.2 class is a function of these
# primitives plus the case's expected value.
# ---------------------------------------------------------------------


def span_compliant(span: str) -> bool:
    r"""Grammar membership: the span matches ``\A[0-9]+\Z``
    (fullmatch). A trailing newline, a sign, grouping, or any
    non-digit character -- ASCII or not -- is noncompliance.
    """
    return re.fullmatch(r"[0-9]+", span) is not None


def first_digit_run(span: str) -> str | None:
    """The first maximal run of ASCII digits in the span, or
    None when the span has none. "1,417" yields "1": grouping
    is a format violation and its first run is not the value.
    """
    match = _DIGIT_RUN_RE.search(span)
    return match.group(0) if match is not None else None


def digit_count(span: str) -> int:
    r"""Total ASCII digit characters in the span (descriptive
    only; never a verdict input -- amendment C). Unicode
    digits do not count: ``str.isdigit`` and ``\d`` disagree
    with each other on spans like "4\u00b27", so the alphabet
    is pinned to [0-9] exactly.
    """
    return sum(1 for ch in span if ch in _ASCII_DIGITS)


def nondigit_projection(span: str) -> str:
    """The span with every ASCII digit removed. The span is
    all-digits (hence compliant, given non-emptiness) iff this
    projection is the empty string -- the digest of the
    projection is what amendment C binds compliance to.
    """
    return "".join(ch for ch in span if ch not in _ASCII_DIGITS)


def first_digit_run_digest(span: str) -> str:
    """Digest of the first maximal ASCII digit run; "" when
    the span has no run (amendment B3: the no-run field is the
    empty string, not the digest of the empty string).
    """
    run = first_digit_run(span)
    return digest_text(run) if run is not None else ""


def nondigit_projection_digest(span: str) -> str:
    """Digest of the non-digit projection (amendment C)."""
    return digest_text(nondigit_projection(span))


def _class_from_digests(
    *,
    answer_digest: Any,
    run_digest: Any,
    projection_digest: Any,
    expected_digest: str,
    boundary_found: bool,
) -> str:
    """The one decomposition definition, over digests. The
    stage classifies spans by digesting them into exactly
    these inputs, and the diff validator recomputes classes
    from the recorded digests alone, so the two can never
    disagree (the cycle-190 digest-binding discipline).
    """
    compliant = (
        type(answer_digest) is str
        and answer_digest != EMPTY_DIGEST
        and projection_digest == EMPTY_DIGEST
    )
    if compliant and answer_digest == expected_digest:
        # Amendment B1: a boundary-missing compliant-correct
        # cell is boundary attrition only -- it cannot count
        # toward D (which requires the adjacent cell
        # boundary-found) and never toward S or T.
        return "compliant_correct" if boundary_found else "boundary_only"
    if compliant:
        return "compliant_wrong"
    if type(run_digest) is str and run_digest != "" and run_digest == expected_digest:
        return "noncompliant_correct_value"
    return "noncompliant_wrong"


def decompose_span(span: str, expected: str, boundary_found: bool) -> str:
    """The decomposition class of one completed cell, from its
    span text (the stage's path).
    """
    return _class_from_digests(
        answer_digest=digest_text(span),
        run_digest=first_digit_run_digest(span),
        projection_digest=nondigit_projection_digest(span),
        expected_digest=digest_text(expected),
        boundary_found=boundary_found,
    )


def decompose_cell(cell: dict[str, Any], expected_digest: str) -> str:
    """The decomposition class of one completed cell, from its
    recorded digests alone (the validator's path; the report
    recomputes nothing from raw text because no raw text is
    carried). Callers must only invoke this on completed
    cells; a missing cell is attrition, not a class.
    """
    if cell is None:
        raise ValueError("decomposition requires a completed cell")
    return _class_from_digests(
        answer_digest=cell.get("answer_digest"),
        run_digest=cell.get("first_digit_run_digest"),
        projection_digest=cell.get("nondigit_projection_digest"),
        expected_digest=expected_digest,
        boundary_found=bool(cell.get("boundary_found")),
    )


def classify_instruction_distance(direct_passes: int, separated: int, displaced: int) -> str:
    """The section decision (total function of D, S, and T).

    D < 10 -> ``insufficient`` (withheld, never defaulted to
    acquittal). D >= 10, S >= 6, and S - T >= 5 ->
    ``instruction_distance_drop_observed`` (an
    excess-of-between-losses pattern at this panel's stated
    power, not a biconditional). Otherwise ->
    ``below_threshold`` (an observed null at this panel size,
    weak evidence, not proof of invariance). The floors gate
    this label only, never a reported count or rate.
    """
    if direct_passes < MIN_D:
        return "insufficient"
    if separated >= MIN_S and separated - displaced >= MIN_MARGIN:
        return "instruction_distance_drop_observed"
    return "below_threshold"


def _cell(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the cell."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _detail(record: dict[str, Any], key: str) -> str:
    """Return the detail."""
    details = record.get("cell_details")
    if not isinstance(details, dict):
        return ""
    detail = details.get(key, "")
    return detail if type(detail) is str else ""


def _rate(numerator: int, denominator: int) -> float | None:
    """Return the rate."""
    return (numerator / denominator) if denominator else None


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def aggregate_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-candidate aggregates over its ground records.

    Every count carries its denominator; a rate is None only
    when its denominator is 0. Per dose, the population is
    cases completed in the adjacent, separated, and displaced
    arms of that dose; D is the adjacent compliant-correct
    count inside it. Token-ID divergence is reported, not
    gated: same-ids between the adjacent arm and another arm
    is a rendering anomaly, and a same-ids compliance flip is
    a determinism finding. Discordance is never pooled across
    candidates: this function sees one candidate's records
    only.
    """
    arms: dict[str, Any] = {}
    for key in ARM_KEYS:
        n_completed = 0
        n_pass = 0
        n_compliant = 0
        n_context = 0
        n_failed = 0
        n_unmatchable = 0
        n_screened = 0
        decomposition: dict[str, int] = dict.fromkeys(DECOMPOSITION_CLASSES, 0)
        for record in records:
            detail = _detail(record, key)
            if detail == "arm_context_exceeded":
                n_context += 1
            elif detail == "arm_evaluation_failed":
                n_failed += 1
            elif detail == "arm_unmatchable":
                n_unmatchable += 1
            elif detail == "arm_screened":
                n_screened += 1
            cell = _cell(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
            cls = decompose_cell(cell, str(record.get("expected_digest", "")))
            decomposition[cls] = decomposition.get(cls, 0) + 1
            if cls in COMPLIANT_CLASSES:
                n_compliant += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": _rate(n_pass, n_completed),
            "n_compliant": n_compliant,
            "compliance_rate": _rate(n_compliant, n_completed),
            "n_context_exceeded": n_context,
            "n_evaluation_failed": n_failed,
            "n_unmatchable": n_unmatchable,
            "n_screened": n_screened,
            "decomposition": decomposition,
        }

    doses: dict[str, Any] = {}
    for dose in DOSES:
        sep = f"sep_{dose}"
        disp = f"disp_{dose}"
        n_population = 0
        direct_passes = 0
        separated_losses = 0
        displaced_losses = 0
        s_value_kept = 0
        s_value_lost = 0
        n_reverse = 0
        for record in records:
            adj = _cell(record, "adj")
            sep_cell = _cell(record, sep)
            disp_cell = _cell(record, disp)
            expected_digest = str(record.get("expected_digest", ""))
            if adj is not None and sep_cell is not None:
                adj_cls = decompose_cell(adj, expected_digest)
                sep_cls = decompose_cell(sep_cell, expected_digest)
                if adj_cls in NONCOMPLIANT_CLASSES and sep_cls in COMPLIANT_CLASSES:
                    n_reverse += 1
            if adj is None or sep_cell is None or disp_cell is None:
                continue
            n_population += 1
            if decompose_cell(adj, expected_digest) != "compliant_correct":
                continue
            direct_passes += 1
            sep_cls = decompose_cell(sep_cell, expected_digest)
            if sep_cls in NONCOMPLIANT_CLASSES:
                separated_losses += 1
                if sep_cls == "noncompliant_correct_value":
                    s_value_kept += 1
                else:
                    s_value_lost += 1
            if decompose_cell(disp_cell, expected_digest) in NONCOMPLIANT_CLASSES:
                displaced_losses += 1
        doses[str(dose)] = {
            "n_population": n_population,
            "direct_passes": direct_passes,
            "separated_losses": separated_losses,
            "displaced_losses": displaced_losses,
            "margin": separated_losses - displaced_losses,
            "s_value_kept": s_value_kept,
            "s_value_lost": s_value_lost,
            "n_reverse": n_reverse,
            "decision": classify_instruction_distance(
                direct_passes, separated_losses, displaced_losses
            ),
        }

    ids: dict[str, Any] = {}
    for key in ARM_KEYS:
        if key == BASELINE_ARM:
            continue
        deltas: list[int] = []
        n_ids_identical = 0
        n_same_ids_flips = 0
        for record in records:
            arm_tokens = record.get("arm_tokens") or {}
            id_digests = record.get("id_digests") or {}
            adj_tokens = arm_tokens.get(BASELINE_ARM)
            other_tokens = arm_tokens.get(key)
            if type(adj_tokens) is int and type(other_tokens) is int:
                deltas.append(other_tokens - adj_tokens)
            else:
                deltas.append(0)
            adj_digest = id_digests.get(BASELINE_ARM)
            other_digest = id_digests.get(key)
            if type(adj_digest) is str and adj_digest != "" and adj_digest == other_digest:
                n_ids_identical += 1
                adj_cell = _cell(record, BASELINE_ARM)
                other_cell = _cell(record, key)
                if adj_cell is not None and other_cell is not None:
                    expected_digest = str(record.get("expected_digest", ""))
                    adj_compliant = decompose_cell(adj_cell, expected_digest) in COMPLIANT_CLASSES
                    other_compliant = (
                        decompose_cell(other_cell, expected_digest) in COMPLIANT_CLASSES
                    )
                    if adj_compliant != other_compliant:
                        n_same_ids_flips += 1
        ids[key] = {
            "id_length_delta_by_case": deltas,
            "max_abs_id_delta": max((abs(d) for d in deltas), default=0),
            "n_ids_identical": n_ids_identical,
            "n_same_ids_compliance_flips": n_same_ids_flips,
        }

    return {
        "arms": arms,
        "doses": doses,
        "ids": ids,
        "operating_characteristics": dict(OPERATING_CHARACTERISTICS),
        "n_cases": len(records),
        "n_incomplete": sum(
            1 for record in records if any(_cell(record, arm) is None for arm in HEADLINE_ARMS)
        ),
    }
