"""Instruction placement probe protocol: generated cases, arms, aggregation.

Why this exists: prompts assembled from a retrieved passage
plus a question follow layout conventions copied from blog
posts measured on other models. Whether the question sits
before or after the payload is a property of the prompt, not
of the weights, so the lab can measure it exactly: generated
single-fact fixtures, the question moved around a frozen short
payload inside one user turn, crossed with the hand-rendered
template families, under exact-match greedy scoring with the
abstention rule. The adjacent axes are other probes' jobs --
hierarchy varies a privilege-marked block folded into the user
turn, position permutes answer options, distract splices
filler at token doses, needle buries a fact in long filler;
none varies question/payload order. The design contract, the
red-team record behind it, and the falsification statement
live in docs/PLACEMENT_DESIGN.md; the uniqueness pass is in
docs/RESEARCH_LOG.md (cycle 193).

Definitions that carry the design (red-team cross-checked,
cycle 193):

* The headline is the discordant pair block question-first vs
  payload-first, split by direction. A symmetric flip count
  hides the direction, which is the actionable part.
* The sandwich arm (question, payload, question repeated
  verbatim) is the control: its discordance against each
  layout, computed on the headline's own paired population,
  bounds what recency and exposure together produce on the
  same cases. It is reported beside the headline, never pooled
  into it, and it does not identify which of recency or
  exposure drives any excess -- it bounds their sum.
* The falsification definition is exactly one formula, here
  and in the design doc: ``placement_effect_falsified`` is
  true iff ``n_paired >= 6`` and not ``J`` and (``d == 0`` or
  (``n_control >= 6`` and ``control_level`` is not ``None``
  and ``d <= control_level``)), where ``d`` is the headline
  discordant total and ``J`` is joint collapse (both layouts
  failing the same cases). The control is deliberately
  conservative -- the bar sits at ``control_level + 1`` with
  ``control_level`` the maximum of two counts -- and the
  counts are always reported so a reader can apply a
  different bar.
* The closed-book arm (question alone) is a screen, not a
  layout: a candidate that answers a case without the payload
  excludes that case from its own placement populations. The
  exclusion is derived from the closed-book cell's verdict,
  never a stored flag.
* Records carry digests and counts only. A cell's verdict is
  recomputed by the diff validator as ``answer_digest ==
  expected_digest`` conjoined with the boundary flag, so there
  is no stored derived answer field to forge (the numform F3
  lesson, applied by construction).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .confidentiality import mcnemar_exact_p

PROTOCOL_VERSION = 1

GROUP = "placement"
JOINER = "\n\n"

CASE_COUNT = 24

#: Frozen topics, one per case. The topic tokens and the code
#: words below are stem-disjoint -- no topic token shares a
#: four-character prefix with a code word, and neither is a
#: prefix of the other -- so a question naming the topic cannot
#: leak the value's word by association (pinned by test).
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
    "greenhouse bay",
    "ferry office",
    "signal box",
    "customs shed",
    "observatory dome",
    "print workshop",
    "harbor office",
    "tunnel control",
    "solar array",
    "wind station",
)

#: The code-word pool for generated values.
WORDS: tuple[str, ...] = (
    "amber",
    "birch",
    "cobalt",
    "dune",
    "ember",
    "flint",
    "garnet",
    "jasper",
    "kestrel",
    "linden",
    "magnet",
    "nutmeg",
    "onyx",
    "sorrel",
    "quartz",
    "rowan",
    "saddle",
    "timber",
    "umber",
    "velvet",
    "walnut",
    "yonder",
    "zephyr",
    "copper",
)

#: The neutral payload sentences, written for this probe:
#: temporally inert scene sentences with no digits, no topic
#: tokens, no code words, and no part of the fact template's
#: carrier phrase, so the payload's only answer-bearing
#: sentence is the fact sentence (pinned by test).
NEUTRAL_POOL: tuple[str, ...] = (
    "The corridor lights hum softly after the evening shift ends.",
    "A kettle sits on the counter beside a chipped blue mug.",
    "Rain taps the window above the wooden bench all morning.",
    "The notice board by the door holds a faded timetable.",
    "Someone left a grey umbrella leaning against the wall.",
    "The floor tiles are cool even on the warmest afternoons.",
    "A narrow shelf holds jars of screws, washers, and twine.",
    "The clock over the sink runs a few minutes fast.",
    "Dust gathers on the ledge where the plants used to sit.",
    "The hallway smells faintly of soap and old paper.",
    "A folded map rests on top of the filing cabinet.",
    "The radiator clicks twice before the heat comes on.",
    "Paper clips and rubber bands fill the shallow drawer.",
    "The yard outside is quiet except for sparrows at dawn.",
    "A chipped enamel sign hangs crooked by the entrance.",
    "The mop bucket stands in the corner by the sink.",
)

FACT_TEMPLATE = "The {topic} inspection code is {value}."
QUESTION_TEMPLATE = "What is the {topic} inspection code?"

ARM_KEYS: tuple[str, ...] = (
    "question_first",
    "payload_first",
    "sandwich",
    "closed_book",
)
#: The arms that are layouts. The closed-book arm is a screen
#: and gets no selection readout.
LAYOUT_ARMS: tuple[str, ...] = ("question_first", "payload_first", "sandwich")
BASELINE_ARM = "payload_first"

#: Classification floors, frozen at the leading/dateline
#: values. They gate the classification label and the
#: falsification boolean only, never a reported count or rate.
MIN_PAIRED_EVALUABLE = 6
MIN_DISCORDANT = 2

#: 24 candidates (run_experiment's cap) x 24 cases; four
#: generations per (candidate, case). Generations and records
#: carry one constant each because the record count is the
#: generation count divided by the arm count.
MAX_GROUND_RECORDS = 24 * CASE_COUNT
GENERATION_BUDGET = len(ARM_KEYS) * MAX_GROUND_RECORDS
#: Records carry four cells with answer digests; the bound is
#: the house 12 MiB for uniformity with the sibling probes.
MAX_RECORDS_BYTES = 12 * 1024 * 1024

#: The classification vocabulary (a total function of the
#: paired counts -- see classify_placement).
CLASSIFICATIONS: tuple[str, ...] = (
    "insufficient",
    "no_discordance_observed",
    "control_set_too_small",
    "control_unavailable",
    "within_sandwich_control",
    "discordant_balanced",
    "discordant_asymmetric",
    "below_threshold",
)

CELL_DETAILS: tuple[str, ...] = (
    "",
    "arm_context_exceeded",
    "arm_evaluation_failed",
)


def digest_text(text: str) -> str:
    """SHA-256 hex digest of a text (the records' digest form)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _derive(seed: int, index: int, slot: int, counter: int) -> int:
    """One deterministic draw. The material string, the slot
    meanings (0 = code word, 1 = number, 2 = first neutral
    sentence, 3 = second neutral sentence), and the redraw
    rules in :func:`generate_cases` are frozen together -- the
    diff validator's mirror reproduces them exactly."""
    material = f"{seed}|placement|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def generate_cases(seed: int) -> list[dict[str, Any]]:
    """The frozen generated case set for a run seed, in
    canonical index order. Pure and deterministic -- the same
    seed regenerates the identical set across runs and
    PYTHONHASHSEED values. A value colliding with an earlier
    case's value redraws both value slots at the next counter,
    so the 24 values are pairwise distinct strings by
    construction. The two neutral sentences are distinct by a
    frozen adjustment: when the second draw lands on the first
    sentence's index, it advances by one (mod the pool size).
    """
    if type(seed) is not int:
        raise ValueError("placement seed must be an int")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(CASE_COUNT):
        topic = TOPICS[index]
        counter = 0
        while True:
            word = WORDS[_derive(seed, index, 0, counter) % len(WORDS)]
            number = 1000 + _derive(seed, index, 1, counter) % 9000
            value = f"{word}-{number}"
            if value not in seen:
                break
            counter += 1
        seen.add(value)
        first = _derive(seed, index, 2, 0) % len(NEUTRAL_POOL)
        second = _derive(seed, index, 3, 0) % len(NEUTRAL_POOL)
        if second == first:
            second = (first + 1) % len(NEUTRAL_POOL)
        fact = FACT_TEMPLATE.format(topic=topic, value=value)
        payload = f"{NEUTRAL_POOL[first]} {fact} {NEUTRAL_POOL[second]}"
        cases.append(
            {
                "fixture_id": f"placement-{index:02d}",
                "topic": topic,
                "value": value,
                "fact": fact,
                "payload": payload,
                "question": QUESTION_TEMPLATE.format(topic=topic),
                "expected": value,
                "sentence_indices": [first, second],
            }
        )
    return cases


def arm_content(case: dict[str, Any], arm: str) -> str:
    """One arm's user-turn content for a case: the same texts,
    differing only in where the question sits (and, for the
    sandwich, its deliberate verbatim repeat; for closed-book,
    the payload's absence)."""
    question = str(case["question"])
    payload = str(case["payload"])
    if arm == "question_first":
        return question + JOINER + payload
    if arm == "payload_first":
        return payload + JOINER + question
    if arm == "sandwich":
        return question + JOINER + payload + JOINER + question
    if arm == "closed_book":
        return question
    raise ValueError(f"unknown placement arm: {arm!r}")


def canonical_generated_rows(seed: int) -> list[list[Any]]:
    """The canonical generated set the ground digest binds."""
    return [
        [
            case["fixture_id"],
            case["topic"],
            case["value"],
            list(case["sentence_indices"]),
        ]
        for case in generate_cases(seed)
    ]


def ground_sha256(seed: int) -> str:
    """Digest of the canonical generated rows for a seed: two
    runs share an instrument only when this matches, because
    the generated set is the treatment."""
    canonical = json.dumps(canonical_generated_rows(seed), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def case_material_sha256() -> str:
    """Digest of the frozen case material (templates, topics,
    words, pool): the lists are bound by digest in the
    protocol, so a payload cannot quietly carry a different
    instrument under the same version number."""
    canonical = json.dumps(
        [FACT_TEMPLATE, QUESTION_TEMPLATE, list(TOPICS), list(WORDS), list(NEUTRAL_POOL)],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def protocol_object(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The frozen instrument description a payload is checked
    against: arms, case material, floors, budgets, and the
    seed/backend pair that generated the set."""
    return {
        "version": PROTOCOL_VERSION,
        "arm_keys": list(ARM_KEYS),
        "layout_arms": list(LAYOUT_ARMS),
        "baseline_arm": BASELINE_ARM,
        "case_count": CASE_COUNT,
        "case_material_sha256": case_material_sha256(),
        "ground_sha256": ground_sha256(seed),
        "min_paired_evaluable": MIN_PAIRED_EVALUABLE,
        "min_discordant": MIN_DISCORDANT,
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
# Aggregation (the validator recomputes this with its own mirror)
# ---------------------------------------------------------------------


def _cell(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _verdict(record: dict[str, Any], key: str) -> bool:
    cell = _cell(record, key)
    return bool(cell is not None and cell["verdict"])


def screened(record: dict[str, Any]) -> bool:
    """Whether the closed-book screen excludes this record for
    its candidate: the closed-book cell completed and passed,
    so the answer was reachable without the payload."""
    cell = _cell(record, "closed_book")
    return cell is not None and bool(cell["verdict"])


def live_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records that enter placement populations: not
    closed-book-screened."""
    return [r for r in records if not screened(r)]


def _pair_block(
    records: list[dict[str, Any]],
    key_a: str,
    key_b: str,
    restrict: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Discordant counts for one arm pair over the paired
    population (both cells completed, optionally restricted to
    a fixture set)."""
    n_paired = 0
    n_a_only = 0
    n_b_only = 0
    n_both = 0
    n_pass_a = 0
    n_pass_b = 0
    for record in records:
        if restrict is not None and record.get("fixture") not in restrict:
            continue
        cell_a = _cell(record, key_a)
        cell_b = _cell(record, key_b)
        if cell_a is None or cell_b is None:
            continue
        n_paired += 1
        verdict_a = bool(cell_a["verdict"])
        verdict_b = bool(cell_b["verdict"])
        if verdict_a:
            n_pass_a += 1
        if verdict_b:
            n_pass_b += 1
        if verdict_a and verdict_b:
            n_both += 1
        elif verdict_a:
            n_a_only += 1
        elif verdict_b:
            n_b_only += 1
    return {
        "n_paired": n_paired,
        "n_a_only": n_a_only,
        "n_b_only": n_b_only,
        "n_both": n_both,
        "n_neither": n_paired - n_a_only - n_b_only - n_both,
        "rate_a": (n_pass_a / n_paired) if n_paired else None,
        "rate_b": (n_pass_b / n_paired) if n_paired else None,
        "mcnemar_p": mcnemar_exact_p(n_a_only, n_b_only),
    }


def both_fail_dominant(n_qf_only: int, n_pf_only: int, n_neither: int) -> bool:
    """The joint-collapse state: both layouts fail the same
    cases at least as often as either direction flips, and at
    or above the discordance floor. A candidate in this state
    has a capability story, not a placement story, and never
    counts as falsified."""
    return n_neither >= max(n_qf_only, n_pf_only) and n_neither >= MIN_DISCORDANT


def classify_placement(
    n_paired: int,
    qf_only: int,
    pf_only: int,
    n_control: int,
    control_level: int | None,
    joint_collapse: bool,
) -> str:
    """The headline classification (total function).

    Decision list, first match wins (docs/PLACEMENT_DESIGN.md).
    The floors gate this label only. ``control_unavailable``
    sits after ``control_set_too_small`` deliberately: a
    control set below the floor reports its size problem
    before its absence, and the branch stays reachable for
    direct calls with a sufficient-but-empty control set,
    which real aggregates cannot produce (a nonempty control
    set always yields a level).
    """
    d = qf_only + pf_only
    if n_paired < MIN_PAIRED_EVALUABLE:
        return "insufficient"
    if d == 0 and not joint_collapse:
        return "no_discordance_observed"
    if n_control < MIN_PAIRED_EVALUABLE:
        return "control_set_too_small"
    if control_level is None:
        return "control_unavailable"
    if d <= control_level:
        return "within_sandwich_control"
    if qf_only == pf_only and qf_only >= MIN_DISCORDANT:
        return "discordant_balanced"
    if max(qf_only, pf_only) >= MIN_DISCORDANT:
        return "discordant_asymmetric"
    return "below_threshold"


def effect_falsified(
    n_paired: int,
    qf_only: int,
    pf_only: int,
    n_control: int,
    control_level: int | None,
    joint_collapse: bool,
) -> bool:
    """The pre-registered falsification boolean.

    True iff ``n_paired >= MIN_PAIRED_EVALUABLE`` and not
    joint collapse and (no discordance, or discordance at or
    below a sufficient control level). Exactly the
    ``no_discordance_observed`` / ``within_sandwich_control``
    outcomes at a sufficient population -- the boolean and the
    classification are two views of one condition.
    """
    if n_paired < MIN_PAIRED_EVALUABLE or joint_collapse:
        return False
    d = qf_only + pf_only
    if d == 0:
        return True
    return n_control >= MIN_PAIRED_EVALUABLE and control_level is not None and d <= control_level


def aggregate_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-candidate aggregates over its ground records.

    Every count carries its denominator; a rate is None only
    when its denominator is 0. The sandwich blocks are computed
    on the headline's paired population (the control set), so
    the control and the headline describe the same cases.
    """
    live = live_records(records)
    arms: dict[str, Any] = {}
    for key in ARM_KEYS:
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
            "rate": (n_pass / n_completed) if n_completed else None,
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
    headline_raw = _pair_block(live, "question_first", "payload_first")
    paired_ids = frozenset(
        str(record.get("fixture"))
        for record in live
        if _cell(record, "question_first") is not None
        and _cell(record, "payload_first") is not None
    )
    sb_pf = _pair_block(live, "sandwich", "payload_first", paired_ids)
    sb_qf = _pair_block(live, "sandwich", "question_first", paired_ids)
    n_control = sb_pf["n_paired"]
    control_level: int | None = None
    if n_control:
        control_level = max(
            sb_pf["n_a_only"] + sb_pf["n_b_only"],
            sb_qf["n_a_only"] + sb_qf["n_b_only"],
        )
    qf_only = headline_raw["n_a_only"]
    pf_only = headline_raw["n_b_only"]
    joint = both_fail_dominant(qf_only, pf_only, headline_raw["n_neither"])
    classification = classify_placement(
        headline_raw["n_paired"], qf_only, pf_only, n_control, control_level, joint
    )
    headline = {
        "n_paired": headline_raw["n_paired"],
        "n_qf_only": qf_only,
        "n_pf_only": pf_only,
        "n_both": headline_raw["n_both"],
        "n_neither": headline_raw["n_neither"],
        "rate_qf": headline_raw["rate_a"],
        "rate_pf": headline_raw["rate_b"],
        "mcnemar_p": headline_raw["mcnemar_p"],
        "both_fail_dominant": joint,
        "classification": classification,
    }
    return {
        "arms": arms,
        "closed_book": {
            "n_completed": cb_completed,
            "n_pass": cb_pass,
            "pass_rate": (cb_pass / cb_completed) if cb_completed else None,
        },
        "headline": headline,
        "sandwich": {
            "vs_payload_first": sb_pf,
            "vs_question_first": sb_qf,
        },
        "control_level": control_level,
        "n_control": n_control,
        "placement_effect_falsified": effect_falsified(
            headline_raw["n_paired"], qf_only, pf_only, n_control, control_level, joint
        ),
        "n_fixtures": len(records),
        "n_screened": len(records) - len(live),
        "n_closed_book_incomplete": sum(
            1 for record in records if _cell(record, "closed_book") is None
        ),
    }


def shared_live_fixtures(
    records_by_candidate: dict[str, list[dict[str, Any]]],
) -> frozenset[str]:
    """Fixture ids recorded for every candidate and screened
    for none of them: the conditions population. A case any
    candidate answered closed-book is not a clean shared
    measurement, so the shared readout screens by union while
    the per-candidate blocks screen per candidate."""
    populations: list[frozenset[str]] = []
    for records in records_by_candidate.values():
        populations.append(frozenset(str(r.get("fixture")) for r in records if not screened(r)))
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
    differ across candidates."""
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
    population, with the flip against the payload-first
    baseline recomputed on the same restricted completed set
    (comparing against the full-set baseline would mix set
    differences into the flip). Withheld when the restricted
    completed sets differ across candidates or are empty; the
    baseline arm's own flip is False when not withheld, and a
    flip is None when either side is withheld."""
    shared = shared_live_fixtures(records_by_candidate)
    own = _selection_over(records_by_candidate, arm, shared)
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
            if record["fixture"] in shared and _cell(record, arm) is not None
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
