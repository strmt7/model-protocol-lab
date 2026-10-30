"""Context rotation probe (cycle 203, research wave-9 rank 1).

Every other order probe in the lab moves one thing: needle
moves one fact through fixed filler, placement moves the
question around a frozen payload, position permutes answer
options. This one rotates the payload itself: five
fact-bearing blocks are cyclically rotated inside one user
turn -- same content multiset, same total text length, same
cyclic relative order -- and the readout is whether the
verdict, and the conservative selection, survive the rotation
per template family. The mirror arms (cyclic rotations of the
reversed block order, named by the asked block's slot) form a
matched control set: they place the asked block at the same
slot multiset {0,1,2,3,4} by non-cyclic reorderings, so a pure
absolute-slot effect moves the headline statistic and the
control statistic identically, case by case.

The measurement settlements were earned by two red teams and
a verifier before any code (docs/ROTATION_DESIGN.md carries
the full adjudication): the classification's control compares
the same statistic on both sides (any-flip over five arms,
same cases, matched slot multiset), because an any-of-5 count
against a single-pair discordance labels pure noise
``rotation_sensitive`` about 99.8% of the time (executed);
joint collapse is a dominance step before any invariance
acquittal (the placement settlement); the construction
classification (``id_exact``) is derived from a carried
primitive -- a per-arm token-inventory digest -- never stored,
so the diff validator recomputes the population gate from the
record itself (the numform F3 lesson, applied by construction).
Cases are generated from the run seed by a frozen SHA-256
derivation the diff validator mirrors exactly; records carry
digests and counts only, and the cell verdict is bound by
``verdict == (answer_digest == sha256(expected)) and
boundary_found``.

The falsification definition is exactly one formula, here and
in the design doc: ``invariance_falsified`` is true iff
``n_headline >= 6`` and ``n_flip_any >= 1``.

The position phenomenon is published prior art (Lost in the
Middle, Attention Sorting, The Order Effect, Context Rot --
see docs/RESEARCH_LOG.md cycle 203); the slot curve is
presented as a cross-check against needle's depth curve, not
a novelty claim. Each cell is one greedy generation on
generated fixtures: the readouts are counts over twenty cases
with printed denominators, not smooth functions.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = [
    "ARM_KEYS",
    "BASELINE_ARM",
    "CASE_COUNT",
    "CELL_DETAILS",
    "CLASSIFICATIONS",
    "CONDITION_ARMS",
    "CONTROL_ARMS",
    "CONSTRUCTION_RULE_TEXT",
    "FACT_TEMPLATE",
    "GENERATION_BUDGET",
    "GROUP",
    "MAX_GROUND_RECORDS",
    "MAX_RECORDS_BYTES",
    "MIN_CONTROL_POPULATION",
    "MIN_HEADLINE_POPULATION",
    "MIRROR_ARMS",
    "PROTOCOL_VERSION",
    "QUESTION_TEMPLATE",
    "ROTATION_ARMS",
    "SCREENING_RULE_TEXT",
    "SPAN_ARMS",
    "STAGE_DETAILS",
    "TOPICS",
    "WORDS",
    "aggregate_candidate",
    "arm_content",
    "arm_id_exact",
    "arm_order",
    "arm_selection",
    "asked_slot",
    "canonical_generated_rows",
    "case_material_sha256",
    "classify_rotation",
    "digest_text",
    "generate_cases",
    "ground_sha256",
    "invariance_falsified",
    "inventory_digest",
    "live_records",
    "protocol_fingerprint",
    "protocol_object",
    "screened",
    "boundary_screened",
    "shared_live_fixtures",
]

PROTOCOL_VERSION = 1
GROUP = "rotation"
JOINER = "\n\n"

CASE_COUNT = 20

FACT_TEMPLATE = "The {topic} access code is {value}."
QUESTION_TEMPLATE = "What is the {topic} access code?"

#: Case topics, frozen. Pinned properties (test-enforced):
#: phrases pairwise distinct, no topic a substring of another,
#: every token at least four characters, and no topic token
#: shares a four-character prefix with a code word or is a
#: substring of one (or vice versa) -- so the asked topic's
#: full phrase occurs in exactly one block of a case and the
#: question cannot leak a value by association.
TOPICS: tuple[str, ...] = (
    "archive vault",
    "boiler room",
    "loading dock",
    "records office",
    "north gate",
    "pump house",
    "storage annex",
    "mail room",
    "server closet",
    "roof access",
    "east stairwell",
    "water tower",
    "fuel depot",
    "tool library",
    "greenhouse wing",
    "ferry terminal",
    "signal cabin",
    "customs shed",
    "observatory dome",
    "print workshop",
    "harbor station",
    "tunnel control",
    "solar array",
    "wind turbine",
    "clock tower",
    "grain silo",
    "kiln workshop",
    "sorting office",
    "parcel depot",
    "marble quarry",
)

#: Code words, frozen. No word is a suffix of another, so one
#: block's value can never occur inside another block's value
#: (values share the hyphen and four-digit shape; a substring
#: hit would need aligned hyphens, hence a suffix pair with an
#: equal number) -- pinned by test over the pool.
WORDS: tuple[str, ...] = (
    "ochre",
    "velvet",
    "cobalt",
    "saffron",
    "tundra",
    "zephyr",
    "onyx",
    "pepper",
    "riddle",
    "saddle",
    "temple",
    "umbrella",
    "violet",
    "walnut",
    "yonder",
    "zipper",
    "anchor",
    "beacon",
    "cinder",
    "dagger",
    "ember",
    "falcon",
    "garnet",
    "helmet",
    "inkwell",
    "jubilee",
    "kestrel",
    "basalt",
    "magnet",
    "nectar",
    "obelisk",
    "pumice",
    "quiver",
    "rampart",
    "scepter",
    "sienna",
    "utensil",
    "vortex",
    "willow",
    "zenith",
    "almond",
    "birch",
    "copper",
    "drizzle",
    "glacier",
    "mauve",
    "ivory",
    "juniper",
    "linden",
    "meteor",
    "nickel",
    "opal",
    "pyrite",
    "quill",
    "russet",
    "sorrel",
    "topaz",
    "flint",
    "jasper",
    "nutmeg",
    "rowan",
    "timber",
    "umber",
    "dune",
)

#: Blocks are indexed [0 = asked, 1..4 = distractors,
#: 5 = replacement]; an arm order lists block indices in span
#: order.
BASE_ORDER: tuple[int, ...] = (0, 1, 2, 3, 4)
REVERSED_ORDER: tuple[int, ...] = (4, 3, 2, 1, 0)
ABSENT_ORDER: tuple[int, ...] = (5, 1, 2, 3, 4)

ROTATION_ARMS: tuple[str, ...] = tuple(f"rot_{r}" for r in range(5))
#: Mirror arms are named by the slot the asked block occupies.
MIRROR_ARMS: tuple[str, ...] = ("mir_1", "mir_2", "mir_3", "mir_4")
SPAN_ARMS: tuple[str, ...] = ROTATION_ARMS + MIRROR_ARMS
ARM_KEYS: tuple[str, ...] = SPAN_ARMS + ("absent", "closed_book")
#: The arms carrying a per-arm selection readout; the baseline
#: is rot_0 and mir_4 is the reversal condition (the placement
#: layout-arm pattern).
CONDITION_ARMS: tuple[str, ...] = ROTATION_ARMS + ("mir_4",)
BASELINE_ARM = "rot_0"
#: The classification's control set: rot_0 plus the mirrors --
#: the same asked-slot multiset {0,1,2,3,4} as the rotations.
CONTROL_ARMS: tuple[str, ...] = ("rot_0",) + MIRROR_ARMS


def _rotate(order: tuple[int, ...], r: int) -> tuple[int, ...]:
    """Cyclic rotation: block ``order[r]`` moves to the front."""
    return tuple(order[(r + i) % len(order)] for i in range(len(order)))


def arm_order(arm: str) -> tuple[int, ...]:
    """The block order of one span arm (frozen).

    Rotations rotate the base order; mirror arm ``mir_s`` is
    the rotation of the reversed order that places the asked
    block (index 0) at slot ``s``; the absent arm swaps the
    asked block for the replacement at slot 0.
    """
    if arm in ROTATION_ARMS:
        return _rotate(BASE_ORDER, int(arm.split("_")[1]))
    if arm in MIRROR_ARMS:
        slot = int(arm.split("_")[1])
        return _rotate(REVERSED_ORDER, 4 - slot)
    if arm == "absent":
        return ABSENT_ORDER
    raise ValueError(f"unknown rotation span arm: {arm!r}")


def asked_slot(arm: str) -> int:
    """The absolute slot of the asked block under one span arm."""
    return arm_order(arm).index(0)


#: Classification floors, frozen at the needle/placement
#: value. They gate the classification label and the
#: falsification boolean only, never a reported count or rate.
MIN_HEADLINE_POPULATION = 6
MIN_CONTROL_POPULATION = 6

#: 24 candidates (run_experiment's cap) x 20 cases; eleven
#: generations per (candidate, case).
MAX_GROUND_RECORDS = 24 * CASE_COUNT
GENERATION_BUDGET = len(ARM_KEYS) * MAX_GROUND_RECORDS
#: Records carry eleven cells with answer digests plus nine
#: inventory digests; the bound is the house 12 MiB for
#: uniformity with the sibling probes.
MAX_RECORDS_BYTES = 12 * 1024 * 1024

#: The classification vocabulary (a total function of the
#: headline and mirror counts -- see classify_rotation).
CLASSIFICATIONS: tuple[str, ...] = (
    "insufficient",
    "joint_collapse_dominant",
    "invariant_observed",
    "control_set_too_small",
    "within_mirror_control",
    "rotation_sensitive",
)

#: The per-cell detail vocabulary. A cell is present exactly
#: when its detail is "".
CELL_DETAILS: tuple[str, ...] = (
    "",
    "arm_context_exceeded",
    "arm_evaluation_failed",
    "arm_screened",
)
#: The stage-level detail vocabulary.
STAGE_DETAILS: tuple[str, ...] = (
    "cancelled",
    "rotation_record_budget_exceeded",
    "rotation_screen_failed",
)

#: The construction rule, frozen as text: the protocol carries
#: the SHA-256 of this string, so a bare version label never
#: has to stand in for the rule itself (the needle settlement).
CONSTRUCTION_RULE_TEXT = (
    "rotation-construction-v1: a span arm's inventory digest is the "
    "SHA-256 of the comma-joined decimal token ids of the arm's full "
    "render, sorted ascending; an arm is id_exact iff its inventory "
    "digest and its token count both equal rot_0's, and rot_0 is "
    "id_exact by definition. The classification is a claim about the "
    "token inventory only; segmentation at the seams is not verified."
)

#: The screening rule, frozen as text on the same terms.
SCREENING_RULE_TEXT = (
    "rotation-screen-v1: stage-level, the frozen texts (templates "
    "with slots emptied, topics, words) are audited for special-token "
    "literals against every candidate format, and every candidate's "
    "decoded boundary text is checked for containment in the frozen "
    "texts; a hit fails the whole stage closed before any generation. "
    "Per case and candidate, if any block text of the case contains "
    "the candidate's decoded boundary text, every cell of the case is "
    "arm_screened (the record is boundary-screened and enters no "
    "population). The closed-book screen is derived, never stored: a "
    "record whose closed-book cell passed is excluded from its "
    "candidate's rotation populations."
)


def digest_text(text: str) -> str:
    """SHA-256 hex digest of a text (the records' digest form)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inventory_digest(ids: tuple[int, ...]) -> str:
    """The token-inventory digest of one render: SHA-256 over
    the sorted token ids, comma-joined (the construction
    rule's canonical form).
    """
    return digest_text(",".join(str(i) for i in sorted(ids)))


def _derive(seed: int, index: int, slot: int, counter: int) -> int:
    """One deterministic draw. The material string, the slot
    meanings (block b's code word is slot 2b and its number is
    slot 2b + 1, for b in 0..5; the asked topic is slot 20;
    the distractor topics for blocks 1..4 are slots 21..24;
    the replacement topic is slot 25), and the redraw rules in
    :func:`generate_cases` are frozen together -- the diff
    validator's mirror reproduces them exactly.
    """
    material = f"{seed}|rotation|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def generate_cases(seed: int) -> list[dict[str, Any]]:
    """The frozen generated case set for a run seed, in
    canonical index order. Pure and deterministic -- the same
    seed regenerates the identical set across runs and
    PYTHONHASHSEED values.

    The asked topic redraws at its slot's next counter while
    its index is an earlier case's asked topic (without
    replacement across cases). Each distractor/replacement
    topic draws at its own slot's counter, redrawing while the
    topic is already chosen for this case, so a case's six
    topics are pairwise distinct. Values draw per block at
    that block's own counters; if any two of the case's six
    values collide, or the asked value repeats an earlier
    case's asked value, all six blocks' counters increment and
    all six values redraw (topic draws are never perturbed by
    a value redraw -- the streams are separate slots).
    """
    if type(seed) is not int:
        raise ValueError("rotation seed must be an int")
    cases: list[dict[str, Any]] = []
    used_asked: set[int] = set()
    seen_asked_values: set[str] = set()
    for index in range(CASE_COUNT):
        counter = 0
        while True:
            asked_index = _derive(seed, index, 20, counter) % len(TOPICS)
            if asked_index not in used_asked:
                break
            counter += 1
        used_asked.add(asked_index)
        topics = [TOPICS[asked_index]]
        chosen = {TOPICS[asked_index]}
        for slot in (21, 22, 23, 24, 25):
            counter = 0
            while True:
                topic = TOPICS[_derive(seed, index, slot, counter) % len(TOPICS)]
                if topic not in chosen:
                    break
                counter += 1
            chosen.add(topic)
            topics.append(topic)
        counters = [0] * 6
        while True:
            values = []
            for block in range(6):
                word = WORDS[_derive(seed, index, 2 * block, counters[block]) % len(WORDS)]
                number = 1000 + _derive(seed, index, 2 * block + 1, counters[block]) % 9000
                values.append(f"{word}-{number}")
            if len(set(values)) == 6 and values[0] not in seen_asked_values:
                break
            counters = [c + 1 for c in counters]
        seen_asked_values.add(values[0])
        blocks = [
            FACT_TEMPLATE.format(topic=topics[block], value=values[block]) for block in range(6)
        ]
        cases.append(
            {
                "fixture_id": f"rotation-{index:02d}",
                "topic": topics[0],
                "topics": topics,
                "values": values,
                "blocks": blocks,
                "question": QUESTION_TEMPLATE.format(topic=topics[0]),
                "expected": values[0],
            }
        )
    return cases


def arm_content(case: dict[str, Any], arm: str) -> str:
    """One arm's user-turn content for a case: the block span
    in the arm's order with the question pinned after it, or
    the question alone for the closed-book screen.
    """
    question = str(case["question"])
    if arm == "closed_book":
        return question
    blocks = case["blocks"]
    span = " ".join(str(blocks[block]) for block in arm_order(arm))
    return span + JOINER + question


def canonical_generated_rows(seed: int) -> list[list[Any]]:
    """The canonical generated set the ground digest binds."""
    return [
        [
            case["fixture_id"],
            list(case["topics"]),
            list(case["values"]),
        ]
        for case in generate_cases(seed)
    ]


def ground_sha256(seed: int) -> str:
    """Digest of the canonical generated rows for a seed: two
    runs share an instrument only when this matches, because
    the generated set is the treatment.
    """
    canonical = json.dumps(canonical_generated_rows(seed), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def case_material_sha256() -> str:
    """Digest of the frozen case material (templates, topics,
    words): the lists are bound by digest in the protocol, so
    a payload cannot quietly carry a different instrument
    under the same version number.
    """
    canonical = json.dumps(
        [FACT_TEMPLATE, QUESTION_TEMPLATE, list(TOPICS), list(WORDS)],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def construction_rule_sha256() -> str:
    """Digest of the frozen construction-rule text."""
    return digest_text(CONSTRUCTION_RULE_TEXT)


def screening_rule_sha256() -> str:
    """Digest of the frozen screening-rule text."""
    return digest_text(SCREENING_RULE_TEXT)


def protocol_object(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The frozen instrument description a payload is checked
    against: arms, case material, rule digests, floors,
    budgets, and the seed/backend pair.
    """
    return {
        "version": PROTOCOL_VERSION,
        "arm_keys": list(ARM_KEYS),
        "span_arms": list(SPAN_ARMS),
        "condition_arms": list(CONDITION_ARMS),
        "baseline_arm": BASELINE_ARM,
        "case_count": CASE_COUNT,
        "case_material_sha256": case_material_sha256(),
        "ground_sha256": ground_sha256(seed),
        "construction_rule_sha256": construction_rule_sha256(),
        "screening_rule_sha256": screening_rule_sha256(),
        "min_headline_population": MIN_HEADLINE_POPULATION,
        "min_control_population": MIN_CONTROL_POPULATION,
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


# ---------------------------------------------------------------------
# Record predicates (the validator's mirror derives the same way)
# ---------------------------------------------------------------------


def _cell(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the cell."""
    cells = record.get("cells")
    if type(cells) is not dict:
        return None
    cell = cells.get(key)
    return cell if type(cell) is dict else None


def _verdict(record: dict[str, Any], key: str) -> bool:
    """Return the verdict."""
    cell = _cell(record, key)
    return bool(cell["verdict"]) if cell is not None else False


def arm_id_exact(record: dict[str, Any], arm: str) -> bool:
    """The derived construction classification of one span
    arm: inventory digest and token count equal to rot_0's
    (rot_0 itself is exact by definition). Derived from the
    record's carried digests and counts, never stored.
    """
    if arm == "rot_0":
        return True
    digests = record.get("arm_inventory_digests")
    tokens = record.get("arm_tokens")
    if type(digests) is not dict or type(tokens) is not dict:
        return False
    return digests.get(arm) == digests.get("rot_0") and tokens.get(arm) == tokens.get("rot_0")


def screened(record: dict[str, Any]) -> bool:
    """Whether the closed-book screen excludes this record for
    its candidate: the closed-book cell completed and passed,
    so the answer was reachable without the span.
    """
    cell = _cell(record, "closed_book")
    return cell is not None and bool(cell["verdict"])


def boundary_screened(record: dict[str, Any]) -> bool:
    """Whether the per-case boundary screen fired for this
    record: every cell carries the arm_screened detail (the
    stage marks the case as a unit, so the state is derivable
    from the record).
    """
    details = record.get("cell_details")
    if type(details) is not dict:
        return False
    return all(details.get(arm) == "arm_screened" for arm in ARM_KEYS)


def live_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records that enter rotation populations: neither
    closed-book-screened nor boundary-screened.
    """
    return [r for r in records if not screened(r) and not boundary_screened(r)]


# ---------------------------------------------------------------------
# Classification (total function; the joint table is the design's
# Falsification section, pinned by an enumeration test)
# ---------------------------------------------------------------------


def classify_rotation(
    n_headline: int,
    n_flip_any: int,
    n_all_pass: int,
    n_all_fail: int,
    n_mirror_paired: int,
    n_flips_paired: int,
    n_flips_mirror: int,
) -> str:
    """The headline classification (total function).

    Decision list, first match wins (docs/ROTATION_DESIGN.md).
    The floors gate this label only. Step 2 is the joint-
    collapse dominance carve-out: a population whose constant
    verdicts are mostly all-fail is a capability story, not an
    invariance acquittal. Steps 5-6 compare the same statistic
    (any-flip over five arms) on the same cases, against a
    control set whose asked-slot multiset matches the
    rotations', so a pure slot effect ties the two sides for
    every case and can never reach step 6 alone.
    """
    if n_headline < MIN_HEADLINE_POPULATION:
        return "insufficient"
    if n_flip_any == 0 and n_all_fail > n_all_pass:
        return "joint_collapse_dominant"
    if n_flip_any == 0:
        return "invariant_observed"
    if n_mirror_paired < MIN_CONTROL_POPULATION:
        return "control_set_too_small"
    if n_flips_paired <= n_flips_mirror:
        return "within_mirror_control"
    return "rotation_sensitive"


def invariance_falsified(n_headline: int, n_flip_any: int) -> bool:
    """The pre-registered falsification boolean: true iff
    ``n_headline >= MIN_HEADLINE_POPULATION`` and at least one
    headline case flips across the five rotations. The mirror
    control does not acquit invariance -- it answers the
    separate cyclic-specificity question the classification
    reports.
    """
    return n_headline >= MIN_HEADLINE_POPULATION and n_flip_any >= 1


# ---------------------------------------------------------------------
# Per-candidate aggregation (the validator recomputes this with its
# own mirror)
# ---------------------------------------------------------------------


def _rate(n_pass: int, n: int) -> float | None:
    """Return the rate."""
    return (n_pass / n) if n else None


def aggregate_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-candidate readouts over its ground records.

    Every count carries its denominator; a rate is None only
    when its denominator is 0. The headline population is the
    live records whose five rotation cells are present and
    id_exact (derived); the mirror and reversal blocks narrow
    it further by their own construction gates.
    """
    live = live_records(records)
    arms: dict[str, Any] = {}
    for key in SPAN_ARMS:
        n_completed = 0
        n_pass = 0
        for record in live:
            cell = _cell(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": _rate(n_pass, n_completed),
        }
    cb_completed = 0
    cb_pass = 0
    for record in records:
        cell = _cell(record, "closed_book")
        if cell is None:
            continue
        cb_completed += 1
        if cell["verdict"]:
            cb_pass += 1
    absent_evaluable = 0
    absent_false = 0
    for record in live:
        cell = _cell(record, "absent")
        if cell is None:
            continue
        absent_evaluable += 1
        if cell["verdict"]:
            absent_false += 1
    headline_records = [
        record
        for record in live
        if all(_cell(record, arm) is not None for arm in ROTATION_ARMS)
        and all(arm_id_exact(record, arm) for arm in ROTATION_ARMS)
    ]
    n_headline = len(headline_records)
    n_all_pass = 0
    n_all_fail = 0
    n_r0_pass = 0
    n_flip_from_pass = 0
    slot_pass: dict[str, int] = {str(s): 0 for s in range(5)}
    for record in headline_records:
        verdicts = [_verdict(record, arm) for arm in ROTATION_ARMS]
        if all(verdicts):
            n_all_pass += 1
        elif not any(verdicts):
            n_all_fail += 1
        if verdicts[0]:
            n_r0_pass += 1
            if not all(verdicts):
                n_flip_from_pass += 1
        for r, verdict in enumerate(verdicts):
            if verdict:
                slot_pass[str((5 - r) % 5)] += 1
    n_invariant = n_all_pass + n_all_fail
    n_flip_any = n_headline - n_invariant
    rotation_present = 0
    rotation_exact = 0
    mirror_present = 0
    mirror_exact = 0
    for record in live:
        for arm in ROTATION_ARMS:
            if _cell(record, arm) is not None:
                rotation_present += 1
                if arm_id_exact(record, arm):
                    rotation_exact += 1
        for arm in MIRROR_ARMS:
            if _cell(record, arm) is not None:
                mirror_present += 1
                if arm_id_exact(record, arm):
                    mirror_exact += 1
    mirror_records = [
        record
        for record in headline_records
        if all(_cell(record, arm) is not None for arm in MIRROR_ARMS)
        and all(arm_id_exact(record, arm) for arm in MIRROR_ARMS)
    ]
    n_mirror_paired = len(mirror_records)
    n_flips_paired = 0
    n_flips_mirror = 0
    for record in mirror_records:
        rotation_verdicts = [_verdict(record, arm) for arm in ROTATION_ARMS]
        if len(set(rotation_verdicts)) > 1:
            n_flips_paired += 1
        control_verdicts = [_verdict(record, arm) for arm in CONTROL_ARMS]
        if len(set(control_verdicts)) > 1:
            n_flips_mirror += 1
    reversal_records = [
        record
        for record in headline_records
        if _cell(record, "mir_4") is not None and arm_id_exact(record, "mir_4")
    ]
    n_reversal_paired = len(reversal_records)
    n_discordant = sum(
        1 for record in reversal_records if _verdict(record, "mir_4") != _verdict(record, "rot_0")
    )
    n_live = len(live)
    return {
        "arms": arms,
        "closed_book": {
            "n_completed": cb_completed,
            "n_pass": cb_pass,
            "pass_rate": _rate(cb_pass, cb_completed),
        },
        "absent": {
            "n_evaluable": absent_evaluable,
            "n_false_pass": absent_false,
            "rate": _rate(absent_false, absent_evaluable),
        },
        "headline": {
            "n_headline": n_headline,
            "n_all_pass": n_all_pass,
            "n_all_fail": n_all_fail,
            "n_invariant": n_invariant,
            "invariance_rate": _rate(n_invariant, n_headline),
            "n_flip_any": n_flip_any,
            "n_r0_pass": n_r0_pass,
            "n_flip_from_pass": n_flip_from_pass,
            "flip_rate_from_pass": _rate(n_flip_from_pass, n_r0_pass),
            "evaluable_share": _rate(n_headline, n_live),
            "id_exact_share": _rate(rotation_exact, rotation_present),
        },
        "slot_curve": {
            "slots": slot_pass,
            "n_headline": n_headline,
        },
        "mirror": {
            "n_paired": n_mirror_paired,
            "n_flips_paired": n_flips_paired,
            "n_flips_mirror": n_flips_mirror,
            "id_exact_share": _rate(mirror_exact, mirror_present),
        },
        "reversal": {
            "n_paired": n_reversal_paired,
            "n_discordant": n_discordant,
            "discordance_rate": _rate(n_discordant, n_reversal_paired),
        },
        "classification": classify_rotation(
            n_headline,
            n_flip_any,
            n_all_pass,
            n_all_fail,
            n_mirror_paired,
            n_flips_paired,
            n_flips_mirror,
        ),
        "invariance_falsified": invariance_falsified(n_headline, n_flip_any),
        "n_fixtures": len(records),
        "n_screened": sum(1 for record in records if screened(record)),
        "n_boundary_screened": sum(1 for record in records if boundary_screened(record)),
        "n_closed_book_incomplete": sum(
            1
            for record in records
            if not boundary_screened(record) and _cell(record, "closed_book") is None
        ),
        "n_live": n_live,
    }


# ---------------------------------------------------------------------
# Cross-candidate readouts: per-arm selection over the shared live
# population, gated per arm on derived construction exactness
# ---------------------------------------------------------------------


def shared_live_fixtures(
    records_by_candidate: dict[str, list[dict[str, Any]]],
) -> frozenset[str]:
    """Fixture ids recorded for every candidate and live for
    all of them: the conditions population's base. A case any
    candidate answered closed-book, or that was
    boundary-screened for any candidate, is not a clean shared
    measurement, so the shared readout screens by union while
    the per-candidate blocks screen per candidate.
    """
    populations: list[frozenset[str]] = []
    for records in records_by_candidate.values():
        populations.append(frozenset(str(r.get("fixture")) for r in live_records(records)))
    if not populations:
        return frozenset()
    return frozenset.intersection(*populations)


def _selection_over(
    records_by_candidate: dict[str, list[dict[str, Any]]],
    key: str,
    fixtures: frozenset[str],
) -> dict[str, Any] | None:
    """The conservative selection for one arm restricted to
    ``fixtures``: None when any candidate's completed set on
    that restriction is empty or the restricted completed sets
    differ across candidates.
    """
    completed_sets: list[frozenset[str]] = []
    all_pass: dict[str, bool | None] = {}
    for name, records in records_by_candidate.items():
        completed = frozenset(
            str(record["fixture"])
            for record in records
            if record["fixture"] in fixtures and _cell(record, key) is not None
        )
        completed_sets.append(completed)
        if not completed:
            return None
        all_pass[name] = all(
            _verdict(record, key) for record in records if record["fixture"] in completed
        )
    if len(set(completed_sets)) > 1:
        return None
    passers = [name for name, ok in all_pass.items() if ok]
    if len(passers) == 1:
        return {"decision": "selected", "winner": passers[0]}
    if len(passers) > 1:
        return {"decision": "ambiguous", "winner": None}
    return {"decision": "no_candidate", "winner": None}


def arm_selection(
    records_by_candidate: dict[str, list[dict[str, Any]]],
    arm: str,
) -> dict[str, Any]:
    """Per-arm selection readout over the shared live
    population, additionally gated on the arm being id_exact
    (derived) for every candidate on the case -- needle's
    conditions range only over construction-verified cells
    because its unmatchable cells are never generated, and
    rotation's text_level cells are generated, so they are
    gated out here instead (rot_0's gate is vacuous: it is
    id_exact by definition). Withheld when the restricted
    completed sets differ across candidates or are empty; the
    baseline arm's own flip is False when not withheld, and a
    flip is None when either side is withheld. The flip
    compares against the rot_0 selection recomputed on the
    same restricted completed set.
    """
    shared = shared_live_fixtures(records_by_candidate)
    by_fixture: dict[str, dict[str, dict[str, Any]]] = {
        name: {str(record.get("fixture")): record for record in records}
        for name, records in records_by_candidate.items()
    }
    gated = frozenset(
        fixture
        for fixture in shared
        if all(arm_id_exact(by_fixture[name][fixture], arm) for name in by_fixture)
    )
    own = _selection_over(records_by_candidate, arm, gated)
    if own is None:
        return {
            "decision": "unavailable",
            "detail": "selection_withheld_mismatched_sets",
            "winner": None,
            "withheld": True,
            "flip": None,
        }
    if arm == BASELINE_ARM:
        flip: bool | None = False
    else:
        completed = frozenset(
            str(record["fixture"])
            for records in records_by_candidate.values()
            for record in records
            if record["fixture"] in gated and _cell(record, arm) is not None
        )
        baseline = _selection_over(records_by_candidate, BASELINE_ARM, completed)
        flip = None if baseline is None else own != baseline
    return {
        "decision": own["decision"],
        "detail": "",
        "winner": own["winner"],
        "withheld": False,
        "flip": flip,
    }
