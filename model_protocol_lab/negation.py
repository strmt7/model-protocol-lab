"""Antonym-negated definite-description probe (cycle 212, wave-10
slate rank 3).

Generated numform compare pairs rendered under three headline
instruction arms per candidate — direct ("larger"), negated ("not
the smaller"), flipped ("smaller") — plus one descriptive arm
(negated_alt, "not smaller than the other", never a decision
input). The measurement is per-case verdict discordance under
greedy decoding on one frozen phrasing, not a mechanism: the
section's only falsification-class label is
``negation_drop_observed``, defined exactly by the (8, 5) rule in
:func:`classify_negation`.

Binding design pins (docs/NEGATION_DESIGN.md):

* Frozen instruction strings, byte-identical reply contract in
  every arm, plain form only, operands in presentation order.
  The negated arm differs from direct in TWO lexical moves
  (negation token + larger-to-smaller antonym swap), so the claim
  is "antonym-negated definite-description discordance" and every
  headline rendering states the bundle. No sentence attributes a
  discordance to "negation" simpliciter.
* Expected answers are the ASCII labels ``first`` / ``second``
  (the numform compare scorer pin: compare's expected answer is
  a form-invariant label, never a rendered number).
* Distinctness is load-bearing: "the smaller" has a unique
  referent only because ``a != b``. The generator enforces it
  and the stage audit re-derives it; an equal pair fails the
  stage closed, never gets tie-broken.
* Scorer: ``core.evaluate``'s byte-exact verdict conjoined with
  boundary presence. No stripping, parsing, or normalisation on
  the headline path. ``negation_echo`` is a frozen subclass of
  ``non_label`` (exactly "not " + other label or "not the " +
  other label, byte-for-byte); it is printed beside negated-arm
  rates, never pooled into ``wrong_label`` and never a pass.
* Decision rule per candidate, over cases completed in all three
  headline arms and not ids-identical on the direct-negated
  pair: D = direct passes, J = cases in D where the negated cell
  is ``wrong_label`` AND the flipped cell passes on the same
  case. D < 8 -> ``insufficient``; D >= 8 and J >= 5 ->
  ``negation_drop_observed``; D >= 8 and J < 5 ->
  ``below_threshold`` (an observed null, not proof of
  invariance). The (8, 5) rule is frozen; it is never re-derived
  and no borrowed floor (6/2 or any other) is used anywhere in
  this panel.
* NO winner readout in v1 in any form. Flipped is a competence
  gate, not a control; a flipped pass rate on D below 0.5 carries
  the gate-weak qualifier ("negation-drop undiagnosed, not
  absent"), never an acquittal.
* Token-ID divergence is reported, not gated: same-ids cases are
  excluded from the headline population and counted as a
  rendering anomaly; a same-ids verdict flip is a determinism
  finding, never negation fragility. Token-count delta is NOT
  reported (the tree has no frozen local tokenizer); character
  delta and render-id length delta are the reported quantities.
* Generated cases only, stage-local, never the user's Split.
  Records carry digests only.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

PROTOCOL_VERSION = 1

GROUP = "negation"

ARM_KEYS: tuple[str, ...] = ("direct", "negated", "flipped", "negated_alt")
HEADLINE_ARMS: tuple[str, ...] = ("direct", "negated", "flipped")
DESCRIPTIVE_ARM = "negated_alt"

CASE_COUNT = 16

#: Decision floors, frozen by the design doc. They gate the
#: section label only, never a reported count or rate.
MIN_DIRECT_PASSES = 8
MIN_JOINT = 5

#: 24 candidates (run_experiment's cap) x 16 cases; four
#: generations per (candidate, case). Headline generations
#: (three arms) are 1,152; with the descriptive arm, 1,536.
MAX_GROUND_RECORDS = 24 * CASE_COUNT
GENERATION_BUDGET = len(ARM_KEYS) * MAX_GROUND_RECORDS
HEADLINE_GENERATIONS = len(HEADLINE_ARMS) * MAX_GROUND_RECORDS
#: Records carry four cells with answer digests plus four
#: content digests and four id digests; the bound is the house
#: 12 MiB for uniformity with the sibling probes.
MAX_RECORDS_BYTES = 12 * 1024 * 1024

#: The section decision vocabulary (a total function of D and
#: J -- see classify_negation). These are section labels only,
#: never cell or stage details.
DECISIONS: tuple[str, ...] = (
    "insufficient",
    "negation_drop_observed",
    "below_threshold",
)

CELL_DETAILS: tuple[str, ...] = (
    "",
    "arm_context_exceeded",
    "arm_evaluation_failed",
)

STAGE_DETAILS: tuple[str, ...] = (
    "cancelled",
    "negation_record_budget_exceeded",
    "negation_screen_failed",
)

#: Margin classes the generator emits. ``same_length_middle``
#: is a real classifier class the generator never emits, so a
#: record carrying it fails validation, as does ``equal``.
EMITTED_MARGIN_CLASSES: tuple[str, ...] = (
    "different_length",
    "same_length_early",
    "same_length_last_digit",
)

#: Frozen instruction strings (docs/NEGATION_DESIGN.md §2.2).
#: The reply contract is byte-identical in every arm, so the
#: answer vocabulary never varies. Operands are appended one
#: per line, in presentation order, exactly as numform does.
INSTRUCTIONS: dict[str, str] = {
    "direct": ("Which of these two numbers is larger? Reply with only the word first or second."),
    "negated": (
        "Which of these two numbers is not the smaller? Reply with only the word first or second."
    ),
    "flipped": ("Which of these two numbers is smaller? Reply with only the word first or second."),
    "negated_alt": (
        "Which of these two numbers is not smaller than the other? "
        "Reply with only the word first or second."
    ),
}

#: Operating characteristics of the frozen (8, 5) rule, from
#: the design doc §5/§7.3 (40,000-trial Monte Carlo, 16 cases).
#: Printed beside every decision; never re-derived at build
#: time. The random-labeller row is a noise-floor null, not a
#: model of a real checkpoint; the no-effect competent row is
#: the realistic false-fire row.
OPERATING_CHARACTERISTICS: dict[str, Any] = {
    "random_labeller_fire_rate": 0.04,
    "no_effect_competent_fire_rate": 0.007,
    "power_drop_rate_0_25": 0.21,
    "power_drop_rate_0_50": 0.84,
    "power_drop_rate_0_75": 0.99,
}

OPERATING_PARAGRAPH_TEMPLATE = (
    "At D = {d}, this panel (16 cases) falsifies only "
    "negation-drop patterns at least as lopsided as J >= 5 at "
    "D >= 8. It does not establish invariance. Measured "
    "operating characteristics of the (8,5) rule: a random "
    "labeller (independent 0.5 labels per arm) fires ~4% of "
    "the time; a competent candidate with no negation effect "
    "(direct/flipped pass 0.9, negated wrong-label rate 0.1) "
    "fires 0.7%; power is 84% at a 0.5 drop rate "
    "(direct/flipped pass 0.9) and 99%+ at 0.75, but only ~21% "
    "at a 0.25 drop rate. Small effects are therefore usually "
    "missed; a below_threshold on this panel is weak evidence."
)


def digest_text(text: str) -> str:
    """SHA-256 hex digest of a text (the records' digest form)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_ids(ids: tuple[int, ...]) -> str:
    """SHA-256 hex digest of a render id sequence, so records
    carry the sequence's identity without its contents.
    """
    canonical = ",".join(str(i) for i in ids)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def other_label(label: str) -> str:
    """The other ASCII label of the frozen two-label set."""
    if label == "first":
        return "second"
    if label == "second":
        return "first"
    raise ValueError("negation label must be first or second")


def expected_label(arm: str, larger_first: bool) -> str:
    """The expected ASCII label for one arm of one case.

    Direct, negated, and negated_alt all denote the larger
    operand (negated by the pair-closed complement rule, valid
    only given a != b); flipped denotes the smaller operand.
    """
    if arm not in ARM_KEYS:
        raise ValueError("unknown negation arm")
    larger = "first" if larger_first else "second"
    if arm == "flipped":
        return other_label(larger)
    return larger


def arm_content(arm: str, first: int, second: int) -> str:
    """One arm's user-turn content: the frozen instruction with
    the plain-form operands appended one per line, in
    presentation order. Only the instruction sentence changes
    across the arms of one case.
    """
    if arm not in INSTRUCTIONS:
        raise ValueError("unknown negation arm")
    return f"{INSTRUCTIONS[arm]}\n{first}\n{second}"


def echo_answers(expected: str) -> tuple[str, str]:
    """The frozen negation_echo answer set for an expected
    label: exactly "not " + other label and "not the " + other
    label, byte-for-byte. Any other negated form is
    ``non_label`` (other), never echo -- the set is frozen, and
    enlarging it voids the design verdict.
    """
    other = other_label(expected)
    return ("not " + other, "not the " + other)


def cases_for(seed: int) -> list[dict[str, Any]]:
    """The frozen generated case set for a run seed, in
    canonical index order. Pure and deterministic: the shipped
    numform compare generator (same seed, same value set),
    re-expressed as negation cases. A case carries its
    presentation operands, its larger-first flag, its margin
    stratum, and its band; per-arm contents and expected
    labels are derived from those, never stored as text.
    """
    from .numform import generate_value_set

    if type(seed) is not int:
        raise ValueError("negation seed must be an int")
    entries = generate_value_set(seed)["compare"]
    cases: list[dict[str, Any]] = []
    for entry in entries:
        operands = entry["operands"]
        first = int(operands[0])
        second = int(operands[1])
        if first == second:
            raise ValueError("negation case operands must differ")
        larger_first = bool(entry["larger_first"])
        cases.append(
            {
                "fixture_id": str(entry["value_id"]),
                "operands": [first, second],
                "larger_first": larger_first,
                "stratum": str(entry["stratum"]),
                "band": int(entry["band"]),
                "contents": {arm: arm_content(arm, first, second) for arm in ARM_KEYS},
                "expected": {arm: expected_label(arm, larger_first) for arm in ARM_KEYS},
            }
        )
    if len(cases) != CASE_COUNT:
        raise ValueError("negation case count must be 16")
    return cases


def canonical_generated_rows(seed: int) -> list[list[Any]]:
    """The canonical generated set the ground digest binds,
    including every arm's content digest and expected label,
    so the ground digest binds the treatment, not just the
    case skeleton.
    """
    rows: list[list[Any]] = []
    for case in cases_for(seed):
        rows.append(
            [
                case["fixture_id"],
                list(case["operands"]),
                case["larger_first"],
                case["stratum"],
                case["band"],
                {arm: digest_text(case["contents"][arm]) for arm in ARM_KEYS},
                dict(case["expected"]),
            ]
        )
    return rows


def ground_sha256(seed: int) -> str:
    """Digest of the canonical generated set: two runs share
    an instrument only when this matches, because the
    generated set is the treatment.
    """
    canonical = json.dumps(canonical_generated_rows(seed), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def case_material_sha256() -> str:
    """Digest of the frozen case material (the four
    instruction strings, bound by digest in the protocol, so
    a payload cannot quietly carry a different instrument
    under the same version number).
    """
    canonical = json.dumps(
        [INSTRUCTIONS[arm] for arm in ARM_KEYS],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def protocol_object(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The frozen instrument description a payload is checked
    against: arms, case material, floors, budgets, and the
    seed/backend pair that generated the set.
    """
    return {
        "version": PROTOCOL_VERSION,
        "arm_keys": list(ARM_KEYS),
        "headline_arms": list(HEADLINE_ARMS),
        "descriptive_arm": DESCRIPTIVE_ARM,
        "case_count": CASE_COUNT,
        "case_material_sha256": case_material_sha256(),
        "ground_sha256": ground_sha256(seed),
        "min_direct_passes": MIN_DIRECT_PASSES,
        "min_joint": MIN_JOINT,
        "generation_budget": GENERATION_BUDGET,
        "headline_generations": HEADLINE_GENERATIONS,
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


# ---------------------------------------------------------------------
# Cell decomposition and aggregation (the report recomputes
# nothing from raw text: every class below is decidable from
# the cell's answer digest and the case's expected label)
# ---------------------------------------------------------------------


def decompose_cell(cell: dict[str, Any], expected: str) -> str:
    """The decomposition class of one completed cell.

    ``exact_pass``: the byte-exact verdict passed.
    ``boundary_only``: the answer text equals the expected
    label but the boundary was missed -- a boundary effect,
    never a negation effect, and it cannot count toward J.
    ``wrong_label``: the answer digest equals the other
    label's digest -- the negation-drop signature.
    ``negation_echo``: the answer is exactly one of the frozen
    echo forms for the other label -- a subclass of non-label,
    never pooled into ``wrong_label``, never a pass.
    ``non_label``: anything else.
    Callers must only invoke this on completed cells (cell
    present); a missing cell is attrition, not a class.
    """
    if cell is None:
        raise ValueError("decomposition requires a completed cell")
    if bool(cell["verdict"]):
        return "exact_pass"
    digest = cell.get("answer_digest")
    if digest == digest_text(expected):
        return "boundary_only"
    if digest == digest_text(other_label(expected)):
        return "wrong_label"
    if digest in {digest_text(form) for form in echo_answers(expected)}:
        return "negation_echo"
    return "non_label"


def classify_negation(direct_passes: int, joint: int) -> str:
    """The section decision (total function of D and J).

    D < 8 -> ``insufficient`` (withheld, never defaulted to
    acquittal and never to a robustness claim). D >= 8 and
    J >= 5 -> ``negation_drop_observed`` (a sufficient-pattern
    detector, not a biconditional). D >= 8 and J < 5 ->
    ``below_threshold`` (an observed null at this panel size,
    not proof of invariance). The floors gate this label
    only, never a reported count or rate.
    """
    if direct_passes < MIN_DIRECT_PASSES:
        return "insufficient"
    if joint >= MIN_JOINT:
        return "negation_drop_observed"
    return "below_threshold"


def _cell(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the cell."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _expected(record: dict[str, Any], arm: str) -> str:
    """Return the expected."""
    larger = "first" if record.get("larger_first") is True else "second"
    if arm == "flipped":
        return other_label(larger)
    return larger


def _rate(numerator: int, denominator: int) -> float | None:
    """Return the rate."""
    return (numerator / denominator) if denominator else None


def aggregate_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-candidate aggregates over its ground records.

    Every count carries its denominator; a rate is None only
    when its denominator is 0. The headline population is
    cases completed in all three headline arms and NOT
    ids-identical on the direct-negated pair; same-ids cases
    are counted separately as a rendering anomaly, never as a
    pass or a fail. Discordance is never pooled across
    candidates or templates: this function sees one
    candidate's records only.
    """
    arms: dict[str, Any] = {}
    for key in ARM_KEYS:
        n_completed = 0
        n_pass = 0
        n_context = 0
        n_failed = 0
        decomposition: dict[str, int] = {}
        for record in records:
            cell = _cell(record, key)
            detail = (record.get("cell_details") or {}).get(key, "")
            if detail == "arm_context_exceeded":
                n_context += 1
            elif detail == "arm_evaluation_failed":
                n_failed += 1
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
            cls = decompose_cell(cell, _expected(record, key))
            decomposition[cls] = decomposition.get(cls, 0) + 1
        for cls in ("exact_pass", "wrong_label", "negation_echo", "non_label", "boundary_only"):
            decomposition.setdefault(cls, 0)
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": _rate(n_pass, n_completed),
            "n_context_exceeded": n_context,
            "n_evaluation_failed": n_failed,
            "decomposition": decomposition,
        }

    # Headline population: all three headline cells present
    # and the direct-negated render ids diverged. Population
    # entries carry their three cells, so downstream loops
    # never re-check a condition the population already
    # guarantees (an unreachable guard would be decoration).
    population: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    n_ids_identical = 0
    n_ids_divergent = 0
    n_same_ids_flips = 0
    deltas: list[int] = []
    for record in records:
        if record.get("ids_identical") is True:
            n_ids_identical += 1
            direct = _cell(record, "direct")
            negated = _cell(record, "negated")
            if (
                direct is not None
                and negated is not None
                and bool(direct["verdict"]) != bool(negated["verdict"])
            ):
                n_same_ids_flips += 1
        elif record.get("ids_identical") is False:
            n_ids_divergent += 1
        delta = record.get("id_length_delta")
        deltas.append(delta if type(delta) is int else 0)
        if record.get("ids_identical") is True:
            continue
        direct = _cell(record, "direct")
        negated = _cell(record, "negated")
        flipped = _cell(record, "flipped")
        if direct is None or negated is None or flipped is None:
            continue
        population.append((record, direct, negated, flipped))

    direct_passes = 0
    joint = 0
    flipped_passes_on_d = 0
    j_by_order: dict[str, int] = {"larger_first": 0, "larger_second": 0}
    j_by_stratum: dict[str, int] = {}
    echo_dominant_denominator = 0
    echo_dominant_numerator = 0
    for record, direct, negated, flipped in population:
        if not bool(direct["verdict"]):
            continue
        direct_passes += 1
        if bool(flipped["verdict"]):
            flipped_passes_on_d += 1
        if decompose_cell(negated, _expected(record, "negated")) == "wrong_label" and bool(
            flipped["verdict"]
        ):
            joint += 1
            order = "larger_first" if record.get("larger_first") is True else "larger_second"
            j_by_order[order] += 1
            stratum = str(record.get("stratum", ""))
            j_by_stratum[stratum] = j_by_stratum.get(stratum, 0) + 1
    flipped_rate_on_d = _rate(flipped_passes_on_d, direct_passes)
    gate_weak = direct_passes > 0 and flipped_rate_on_d is not None and flipped_rate_on_d < 0.5

    negated_arm = arms["negated"]
    negated_decomp = negated_arm["decomposition"]
    echo_dominant_numerator = negated_decomp["negation_echo"]
    echo_dominant_denominator = negated_arm["n_completed"] - negated_decomp["exact_pass"]
    echo_dominant = echo_dominant_numerator * 2 > echo_dominant_denominator > 0
    order_concentrated = joint > 0 and (
        j_by_order["larger_first"] == 0 or j_by_order["larger_second"] == 0
    )

    # Descriptive alt arm: wrong-label count on D and the
    # alt-joint count (alt wrong_label AND flipped passes on
    # the same case), over D cases where the alt cell
    # completed. Never a decision input, never pooled.
    alt_wrong_on_d = 0
    alt_joint = 0
    alt_completed_on_d = 0
    for record, direct, _negated, flipped in population:
        alt = _cell(record, "negated_alt")
        if alt is None:
            continue
        if not bool(direct["verdict"]):
            continue
        alt_completed_on_d += 1
        if decompose_cell(alt, _expected(record, "negated_alt")) == "wrong_label":
            alt_wrong_on_d += 1
            if bool(flipped["verdict"]):
                alt_joint += 1

    # Boundary-only vs exact-only discordance, over paired
    # direct-negated cases (both cells completed, ids
    # diverged): verdicts differ with equal answer digests is
    # boundary-only; verdicts differ with differing digests
    # and both boundaries found is exact-only; verdicts
    # differ otherwise is mixed (reported, never attributed).
    n_boundary_only = 0
    n_exact_only = 0
    n_mixed = 0
    for record in records:
        if record.get("ids_identical") is True:
            continue
        direct = _cell(record, "direct")
        negated = _cell(record, "negated")
        if direct is None or negated is None:
            continue
        if bool(direct["verdict"]) == bool(negated["verdict"]):
            continue
        if direct.get("answer_digest") == negated.get("answer_digest"):
            n_boundary_only += 1
        elif bool(direct.get("boundary_found")) and bool(negated.get("boundary_found")):
            n_exact_only += 1
        else:
            n_mixed += 1

    decision = classify_negation(direct_passes, joint)
    return {
        "arms": arms,
        "headline": {
            "n_population": len(population),
            "direct_passes": direct_passes,
            "joint": joint,
            "decision": decision,
            "j_by_order": j_by_order,
            "j_by_stratum": j_by_stratum,
            "flipped_passes_on_d": flipped_passes_on_d,
            "flipped_rate_on_d": flipped_rate_on_d,
            "gate_weak": gate_weak,
            "echo_dominant": echo_dominant,
            "order_concentrated": order_concentrated,
        },
        "descriptive": {
            "negated_alt_completed_on_d": alt_completed_on_d,
            "negated_alt_wrong_label_on_d": alt_wrong_on_d,
            "negated_alt_joint": alt_joint,
        },
        "ids": {
            "n_ids_identical": n_ids_identical,
            "n_ids_divergent": n_ids_divergent,
            "n_same_ids_verdict_flips": n_same_ids_flips,
            "id_length_delta_by_case": deltas,
            "max_abs_id_delta": max((abs(d) for d in deltas), default=0),
        },
        "discordance": {
            "boundary_only": n_boundary_only,
            "exact_only": n_exact_only,
            "mixed": n_mixed,
        },
        "operating_characteristics": dict(OPERATING_CHARACTERISTICS),
        "n_cases": len(records),
        "n_incomplete": sum(
            1 for record in records if any(_cell(record, arm) is None for arm in HEADLINE_ARMS)
        ),
    }
