"""Instruction-density probe (cycle 204, research wave-9 rank 3).

Every other adherence probe in the lab holds the instruction
count fixed: hierarchy pits two privileged instructions,
structured checks one schema, distraction adds demand-free
text. This one varies the count itself: twenty generated
cases each carry a frozen ladder of sixteen simultaneous,
mutually compatible constraints, and the arms present one,
two, four, eight, or all sixteen of them (plus a sham arm
that pads the block with neutral statements), so the readout
is how per-constraint satisfaction falls as density rises --
and which serial positions fall -- per template family.

The measurement settlements were earned by two red teams and
a verifier before any code (docs/DENSITY_DESIGN.md carries
the full adjudication): the constraint sets are jointly
satisfiable by construction (one shared start target, one
shared end target, every target group distinct, the count
floor above the distinct-target count), because an
unsatisfiable ladder falsifies a perfect instruction-follower
deterministically (executed); the falsification decision
aggregates discordance per case and runs an exact sign test
over cases, because the paired sign test on 320 pairs fires
on pure noise about half the time and case-clustered pairs
break the paired test's calibration (both executed); and
every satisfaction verdict is derived from one carried
primitive -- the ordered list of the answer's word hashes --
never stored, so the diff validator re-derives the verdicts
from the record itself (the rotation F4 settlement). The
position headline is band-pooled because a per-position cell
holds twenty cases and cannot support a finer ranking.

The falsification definition is exactly one rule, here and
in the design doc: per candidate, aggregate the eligible
solo-vs-k16 discordance per case into d_c; with at least
``MIN_DISCORDANT_CASES`` cases at d_c != 0, ``falsified`` is
true iff the pooled f exceeds the pooled p and the exact
two-sided sign test over the discordant cases gives
p <= ``ALPHA``; below that case floor the verdict is
``insufficient`` and no boolean is printed.

The phenomenon is published prior art (IFScale, Jaroslawicz
et al., arXiv:2507.11538 -- 500 keyword instructions on a
business report at API scale); this probe's claim is the
combination (density x serial position x template family,
local checkpoint, digest-decidable constraints), never first.
Each cell is one greedy generation on generated fixtures:
the readouts are counts over twenty cases with printed
denominators, not smooth functions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any

__all__ = [
    "ALPHA",
    "ARM_KEYS",
    "CASE_COUNT",
    "CHECKING_RULE_TEXT",
    "CONSTRAINT_COUNT",
    "COUNT_PARAMS",
    "DENSITIES",
    "GENERATION_RULE_TEXT",
    "GROUP",
    "KINDS",
    "MAX_DENSITY_PAYLOAD_BYTES",
    "MAX_DENSITY_RECORDS",
    "MAX_NEW_TOKENS",
    "MIN_DISCORDANT_CASES",
    "NEUTRAL_STATEMENTS",
    "PROTOCOL_VERSION",
    "RECORD_DETAILS",
    "RENDERING_RULE_TEXT",
    "STAGE_DETAILS",
    "TARGET_WORDS",
    "TOPICS",
    "WORDS",
    "aggregate_candidate",
    "arm_block_words",
    "arm_content",
    "canonical_generated_rows",
    "case_material_sha256",
    "checking_rule_sha256",
    "commitment_digest",
    "derive_from_hashes",
    "generate_cases",
    "generation_rule_sha256",
    "ground_sha256",
    "kind_at",
    "normalize_words",
    "protocol_fingerprint",
    "protocol_object",
    "rendering_rule_sha256",
    "satisfied",
    "sign_test_p",
    "task_present",
    "word_hashes",
]

PROTOCOL_VERSION = 1
GROUP = "density"
JOINER = "\n\n"

CASE_COUNT = 20
CONSTRAINT_COUNT = 16
DENSITIES: tuple[int, ...] = (1, 2, 4, 8, 16)
#: Kind cycle; a case's position kinds rotate by case index.
KINDS: tuple[str, ...] = ("include", "start", "end", "count")
#: Frozen word-count limits; the smallest (8) exceeds the
#: largest possible distinct-target count (7), which is half
#: of the satisfiability invariant.
COUNT_PARAMS: tuple[int, ...] = (8, 12, 16, 24)

SOLO_ARMS: tuple[str, ...] = tuple(f"solo_{j}" for j in range(1, CONSTRAINT_COUNT + 1))
DENSITY_ARMS: tuple[str, ...] = ("k2", "k4", "k8", "k16")
ARM_KEYS: tuple[str, ...] = SOLO_ARMS + DENSITY_ARMS + ("sham",)

MAX_NEW_TOKENS = 64
MAX_DENSITY_RECORDS = 21 * CASE_COUNT * 24
MAX_DENSITY_PAYLOAD_BYTES = 52 * 1024 * 1024

#: The falsification decision's frozen constants.
MIN_DISCORDANT_CASES = 6
ALPHA = 0.05

#: Stage-level fail-closed detail vocabulary.
STAGE_DETAILS: tuple[str, ...] = (
    "cancelled",
    "density_record_budget_exceeded",
    "density_screen_failed",
)
#: Per-record detail vocabulary; "" is a completed record.
RECORD_DETAILS: tuple[str, ...] = ("", "arm_evaluation_failed", "arm_context_exceeded")

TASK_FACT_TEMPLATE = "The {topic} signal word is {value}."
TASK_QUESTION_TEMPLATE = "What is the {topic} signal word?"
CONSTRAINT_TEMPLATES: dict[str, str] = {
    "include": "Include the word {target} in your answer.",
    "start": "Start your answer with the word {target}.",
    "end": "End your answer with the word {target}.",
    "count": "Use at most {limit} words in your answer.",
}
INSTRUCTION_LINE_TEMPLATE = "Instruction {position}: {text}"

#: Case topics, frozen. Pinned properties (test-enforced):
#: phrases pairwise distinct, no topic a substring of another,
#: and no component word of any topic is a task-value word or
#: a target word -- so a case's task text can collide with a
#: constraint target only through the task value, which the
#: redraw rule owns.
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

#: Task values (the signal words), frozen. Single words,
#: pairwise distinct, none equal to a target word or a topic
#: component (test-enforced).
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
)

#: Constraint target words, frozen: content words plus the
#: number words (an include target may be a number word; the
#: digit form is a different word by the frozen checking
#: rule -- a deliberate measurement choice, stated in the
#: design). Pairwise distinct and disjoint from WORDS and
#: from every topic component (test-enforced).
TARGET_WORDS: tuple[str, ...] = (
    "basket",
    "breeze",
    "candle",
    "carpet",
    "delta",
    "emerald",
    "feather",
    "garden",
    "compass",
    "island",
    "jacket",
    "kettle",
    "ladder",
    "meadow",
    "needle",
    "orchard",
    "pencil",
    "quilt",
    "ribbon",
    "bronze",
    "shadow",
    "timber",
    "copper",
    "window",
    "yellow",
    "drizzle",
    "bridge",
    "cloud",
    "drum",
    "eagle",
    "flame",
    "glove",
    "hammer",
    "ivory",
    "jungle",
    "lemon",
    "mirror",
    "night",
    "ocean",
    "palace",
    "quest",
    "river",
    "stone",
    "tiger",
    "urban",
    "valley",
    "winter",
    "xenon",
    "glacier",
    "lantern",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
)

#: Neutral statements for the sham arm, frozen. Pinned
#: property (test-enforced): no statement contains any task
#: value, target word, or topic component as a normalized
#: word -- a neutral can neither satisfy a constraint nor
#: prime the task answer.
NEUTRAL_STATEMENTS: tuple[str, ...] = (
    "The weather changed slowly during the afternoon hours.",
    "A wooden fence runs along the edge of the property.",
    "She placed the documents inside a plain folder yesterday.",
    "The train arrived later than the schedule suggested.",
    "Fresh bread was cooling on the counter near the sink.",
    "He repaired the bicycle with tools from the garage.",
    "The bookshop extends its opening hours during summer.",
    "Several chairs were arranged around the long table.",
    "The recipe calls for a pinch of salt and herbs.",
    "They walked along the path beside the lake at dawn.",
    "The committee postponed its decision until next week.",
    "A thick layer of dust covered the old furniture.",
    "The children played quietly in the courtyard after lunch.",
    "His handwriting was difficult to read in places.",
    "The farmer harvested the fields before the rain came.",
    "An old map hung framed on the wall of the hallway.",
    "The accountant reviewed the figures a second time.",
    "Birds gathered on the rooftops as the sun went down.",
    "The mechanic inspected the engine and changed the oil.",
    "A narrow staircase led up to the attic.",
    "The orchestra rehearsed the piece several times over.",
    "Someone left a bicycle leaning against the railing.",
    "The soup simmered gently on the stove all morning.",
    "Visitors signed the guest book near the entrance.",
)

#: How many neutrals a case draws (the sham arm uses a
#: prefix of them, in drawn order).
NEUTRALS_PER_CASE = 15


def kind_at(case_index: int, position: int) -> str:
    """The frozen kind of one position (1-based) of one case:
    the kind cycle rotated by the case index, so each position
    carries each kind exactly CASE_COUNT / 4 times.
    """
    return KINDS[(position - 1 + case_index) % len(KINDS)]


GENERATION_RULE_TEXT = (
    "Cases are derived from the run seed by SHA-256 draws "
    "material=f'{seed}|density|{index}|{slot}|{counter}'. Per case, "
    "slots are drawn in this order: 0 task topic, 1 task value, "
    "10..24 the case's fifteen neutral statements (distinct, in "
    "drawn order), then constraint parameters -- include targets "
    "at slots 30..33 in position order, the shared start target "
    "at slot 34, the shared end target at slot 35, count limits "
    "at slots 40..43 in position order, drawn without "
    "replacement from {8, 12, 16, 24}. A "
    "case's sixteen constraint kinds are fixed by the kind cycle "
    "(include, start, end, count) rotated by the case index. A "
    "word target redraws at its own slot's next counter while it "
    "equals the task value, equals another target group of the "
    "case, or -- for position 1's target only -- occurs in a "
    "neutral statement of the case; occurrence means normalized-"
    "word equality. All start constraints of a case share the "
    "start target and all end constraints share the end target, "
    "so a case has at most seven distinct word targets (four "
    "include, one start, one end, the task value), below the "
    "smallest count limit."
)

RENDERING_RULE_TEXT = (
    "A case's user turn is the task fact, the task question, "
    "and the arm's instruction block, joined by a blank line. "
    "A constraint renders through its kind template and is "
    "always labelled with its own position: 'Instruction "
    "{position}: {text}' -- in solo arms too, so a constraint's "
    "position label is identical in every arm that carries it. "
    "A density arm's block is the first K rendered constraints "
    "in position order. A solo arm's block is the single "
    "rendered constraint. The sham arm's block is constraint "
    "1's rendered line followed by bare neutral statements (no "
    "label), appended in drawn order, wrapping to the start of "
    "the drawn list, until the block's word count first reaches "
    "or exceeds the k16 block's word count; block word count is "
    "the normalized-word count of the block text."
)

CHECKING_RULE_TEXT = (
    "A word is a maximal run of alphanumeric characters, "
    "case-folded. The commitment of an answer is the ordered "
    "list of SHA-256 hashes of its words; the list digest is "
    "SHA-256 of the compact-JSON list. Satisfaction is derived "
    "from the list: include = the target's hash occurs in the "
    "list; start = the list's first hash is the target's; end "
    "= the list's last hash is the target's; count = the "
    "list's length is at most the limit. The task value is "
    "present iff its hash occurs in the list. Aggregation "
    "excludes zero-word live records from count-kind checked "
    "denominators. The falsification decision aggregates "
    "solo-vs-k16 discordance per case and applies the exact "
    "two-sided sign test over discordant cases."
)


def digest_text(text: str) -> str:
    """SHA-256 hex digest of a text (the records' digest form)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _derive(seed: int, index: int, slot: int, counter: int) -> int:
    """One deterministic draw. The material string, the slot
    meanings in GENERATION_RULE_TEXT, and the redraw rules in
    :func:`generate_cases` are frozen together -- the diff
    validator's mirror reproduces them exactly.
    """
    material = f"{seed}|density|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _draw_word(seed: int, index: int, slot: int, forbidden: set[str]) -> str:
    """Draw a target word at a slot, redrawing at the next
    counter while the word is forbidden for this case.
    """
    counter = 0
    while True:
        word = TARGET_WORDS[_derive(seed, index, slot, counter) % len(TARGET_WORDS)]
        if word not in forbidden:
            return word
        counter += 1


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def generate_cases(seed: int) -> list[dict[str, Any]]:
    """The frozen generated case set for a run seed, in
    canonical index order. Pure and deterministic -- the same
    seed regenerates the identical set across runs and
    PYTHONHASHSEED values.

    The task value is drawn before the targets, and every
    target group redraws against the value and the groups
    drawn before it, so a case's distinct word targets are
    pairwise distinct by construction (the satisfiability
    invariant). Neutral statements draw without replacement.
    """
    if type(seed) is not int:
        raise ValueError("density seed must be an int")
    cases: list[dict[str, Any]] = []
    for index in range(CASE_COUNT):
        topic = TOPICS[_derive(seed, index, 0, 0) % len(TOPICS)]
        value = WORDS[_derive(seed, index, 1, 0) % len(WORDS)]
        neutrals: list[str] = []
        chosen: set[int] = set()
        for k in range(NEUTRALS_PER_CASE):
            counter = 0
            while True:
                pick = _derive(seed, index, 10 + k, counter) % len(NEUTRAL_STATEMENTS)
                if pick not in chosen:
                    break
                counter += 1
            chosen.add(pick)
            neutrals.append(NEUTRAL_STATEMENTS[pick])
        neutral_words: set[str] = set()
        for statement in neutrals:
            neutral_words.update(normalize_words(statement))
        include_targets: list[str] = []
        forbidden: set[str] = {value}
        for slot in (30, 31, 32, 33):
            word = _draw_word(seed, index, slot, forbidden)
            include_targets.append(word)
            forbidden.add(word)
        start_target = _draw_word(seed, index, 34, forbidden)
        forbidden.add(start_target)
        end_target = _draw_word(seed, index, 35, forbidden)
        forbidden.add(end_target)
        # Position 1's target must not occur in the case's
        # neutrals (it is the only target sharing an arm with
        # them); redraw it at its slot's next counter. The
        # kind at position 1 rotates, so find which group owns
        # position 1 for this case.
        first_kind = kind_at(index, 1)
        if first_kind in ("include", "start", "end"):
            current = (
                include_targets[0]
                if first_kind == "include"
                else start_target
                if first_kind == "start"
                else end_target
            )
            if current in neutral_words:
                slot = 30 if first_kind == "include" else 34 if first_kind == "start" else 35
                fresh = _draw_word(seed, index, slot, forbidden | neutral_words)
                if first_kind == "include":
                    include_targets[0] = fresh
                elif first_kind == "start":
                    start_target = fresh
                else:
                    end_target = fresh
                forbidden.discard(current)
                forbidden.add(fresh)
        limit_pool = list(COUNT_PARAMS)
        count_limits: list[int] = []
        for slot in (40, 41, 42, 43):
            pick = _derive(seed, index, slot, 0) % len(limit_pool)
            count_limits.append(limit_pool.pop(pick))
        constraints: list[dict[str, Any]] = []
        pools = {
            "include": iter(include_targets),
            "start": iter([start_target] * 4),
            "end": iter([end_target] * 4),
            "count": iter(count_limits),
        }
        for position in range(1, CONSTRAINT_COUNT + 1):
            kind = kind_at(index, position)
            if kind == "count":
                constraints.append(
                    # skipcq: PTC-W0063 - kind cycle and pool sizes are both four by construction
                    {"position": position, "kind": kind, "limit": next(pools["count"])}
                )
            elif kind == "include":
                constraints.append(
                    # skipcq: PTC-W0063 - kind cycle and pool sizes are both four by construction
                    {"position": position, "kind": kind, "target": next(pools["include"])}
                )
            else:
                constraints.append(
                    # skipcq: PTC-W0063 - kind cycle and pool sizes are both four by construction
                    {"position": position, "kind": kind, "target": next(pools[kind])}
                )
        cases.append(
            {
                "fixture_id": f"density-{index:02d}",
                "topic": topic,
                "value": value,
                "fact": TASK_FACT_TEMPLATE.format(topic=topic, value=value),
                "question": TASK_QUESTION_TEMPLATE.format(topic=topic),
                "constraints": constraints,
                "neutrals": neutrals,
                "expected_word_digest": digest_text(value),
            }
        )
    return cases


def _render_constraint(constraint: dict[str, Any]) -> str:
    """One constraint's labelled instruction line (frozen)."""
    kind = str(constraint["kind"])
    if kind == "count":
        text = CONSTRAINT_TEMPLATES["count"].format(limit=constraint["limit"])
    else:
        text = CONSTRAINT_TEMPLATES[kind].format(target=constraint["target"])
    return INSTRUCTION_LINE_TEMPLATE.format(position=constraint["position"], text=text)


def _block_text(lines: list[str]) -> str:
    """Return the block text."""
    return "\n".join(lines)


def arm_block_words(case: dict[str, Any], arm: str) -> int:
    """The normalized-word count of one arm's instruction
    block (the sham construction's length measure).
    """
    return len(normalize_words(_arm_block(case, arm)))


def _arm_block(case: dict[str, Any], arm: str) -> str:
    """Return the arm block."""
    constraints = case["constraints"]
    if arm.startswith("solo_"):
        j = int(arm.split("_")[1])
        return _block_text([_render_constraint(constraints[j - 1])])
    if arm.startswith("k"):
        k = int(arm[1:])
        return _block_text([_render_constraint(c) for c in constraints[:k]])
    if arm == "sham":
        first = _render_constraint(constraints[0])
        target = len(normalize_words(_arm_block(case, "k16")))
        lines = [first]
        words = len(normalize_words(first))
        # Append the drawn neutrals in drawn order, wrapping
        # to the start of the drawn list, until the block
        # reaches the k16 block's word count. (Coordinator
        # amendment at build: the no-wrap rule left the sham
        # shorter in 3,992 of 4,000 measured cases -- the
        # length match the control exists for almost never
        # held. Wrapping adds no pool words: the neutrals are
        # pinned pool-free, so repetition cannot smuggle a
        # target into the block.)
        pool = list(case["neutrals"])
        index = 0
        while words < target:
            statement = pool[index % len(pool)]
            lines.append(statement)
            words += len(normalize_words(statement))
            index += 1
        return _block_text(lines)
    raise ValueError(f"unknown density arm: {arm!r}")


def arm_content(case: dict[str, Any], arm: str) -> str:
    """One arm's user-turn content for a case: the task fact,
    the task question, and the arm's instruction block.
    """
    return JOINER.join([str(case["fact"]), str(case["question"]), _arm_block(case, arm)])


def canonical_generated_rows(seed: int) -> list[list[Any]]:
    """The canonical generated set the ground digest binds."""
    rows: list[list[Any]] = []
    for case in generate_cases(seed):
        rows.append(
            [
                case["fixture_id"],
                case["topic"],
                case["value"],
                [
                    [c["position"], c["kind"], c.get("target"), c.get("limit")]
                    for c in case["constraints"]
                ],
                list(case["neutrals"]),
            ]
        )
    return rows


def ground_sha256(seed: int) -> str:
    """Digest of the canonical generated rows for a seed: two
    runs share an instrument only when this matches, because
    the generated set is the treatment.
    """
    canonical = json.dumps(canonical_generated_rows(seed), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def case_material_sha256() -> str:
    """Digest of the frozen case material (templates, topics,
    values, targets, neutrals): the lists are bound by digest
    in the protocol, so a payload cannot quietly carry a
    different instrument under the same version number.
    """
    canonical = json.dumps(
        [
            TASK_FACT_TEMPLATE,
            TASK_QUESTION_TEMPLATE,
            CONSTRAINT_TEMPLATES,
            INSTRUCTION_LINE_TEMPLATE,
            list(TOPICS),
            list(WORDS),
            list(TARGET_WORDS),
            list(NEUTRAL_STATEMENTS),
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generation_rule_sha256() -> str:
    """Digest of the frozen generation-rule text."""
    return digest_text(GENERATION_RULE_TEXT)


def rendering_rule_sha256() -> str:
    """Digest of the frozen rendering-rule text."""
    return digest_text(RENDERING_RULE_TEXT)


def checking_rule_sha256() -> str:
    """Digest of the frozen checking-rule text."""
    return digest_text(CHECKING_RULE_TEXT)


def protocol_object(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The frozen instrument description a payload is checked
    against: arms, case material, rule digests, floors,
    budgets, and the seed/backend pair.
    """
    return {
        "version": PROTOCOL_VERSION,
        "arm_keys": list(ARM_KEYS),
        "case_count": CASE_COUNT,
        "constraint_count": CONSTRAINT_COUNT,
        "densities": list(DENSITIES),
        "count_params": list(COUNT_PARAMS),
        "case_material_sha256": case_material_sha256(),
        "ground_sha256": ground_sha256(seed),
        "generation_rule_sha256": generation_rule_sha256(),
        "rendering_rule_sha256": rendering_rule_sha256(),
        "checking_rule_sha256": checking_rule_sha256(),
        "min_discordant_cases": MIN_DISCORDANT_CASES,
        "alpha": ALPHA,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_density_records": MAX_DENSITY_RECORDS,
        "max_density_payload_bytes": MAX_DENSITY_PAYLOAD_BYTES,
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
# Checking (the validator's mirror derives the same way)
# ---------------------------------------------------------------------


def normalize_words(text: str) -> list[str]:
    """The frozen word rule: a word is a maximal run of
    alphanumeric characters, case-folded. Every other
    character is a separator, so emphasis, brackets, and
    punctuation never hide a word, and a punctuation-only
    answer has no words.
    """
    words: list[str] = []
    current: list[str] = []
    for char in text:
        if char.isalnum():
            current.append(char)
        elif current:
            words.append("".join(current).casefold())
            current = []
    if current:
        words.append("".join(current).casefold())
    return words


def word_hashes(text: str) -> list[str]:
    """The commitment of an answer text: the ordered list of
    its words' SHA-256 hashes (multiplicity preserved).
    """
    return [digest_text(word) for word in normalize_words(text)]


def commitment_digest(hashes: list[str]) -> str:
    """SHA-256 of the commitment's canonical encoding (the
    compact-JSON hash list, frozen in the checking rule).
    """
    canonical = json.dumps(list(hashes), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def derive_from_hashes(hashes: list[str]) -> dict[str, Any]:
    """Everything satisfaction reads, derived from the
    commitment alone: the multiset, the count, and the
    first/last word hashes (None when the answer has no
    words).
    """
    return {
        "multiset": Counter(hashes),
        "word_count": len(hashes),
        "first": hashes[0] if hashes else None,
        "last": hashes[-1] if hashes else None,
    }


def satisfied(constraint: dict[str, Any], hashes: list[str]) -> bool:
    """One constraint's verdict, derived from the commitment.
    (The zero-word exclusion for count constraints is an
    aggregation rule, applied by the caller -- a zero-word
    record's count verdict is vacuous, not false.)
    """
    kind = str(constraint["kind"])
    if kind == "count":
        return len(hashes) <= int(constraint["limit"])
    target = digest_text(str(constraint["target"]))
    if kind == "include":
        return target in hashes
    if kind == "start":
        return bool(hashes) and hashes[0] == target
    if kind == "end":
        return bool(hashes) and hashes[-1] == target
    raise ValueError(f"unknown constraint kind: {kind!r}")


def task_present(case: dict[str, Any], hashes: list[str]) -> bool:
    """Whether the answer's commitment contains the case's
    expected task word (the base task's measurable trace).
    """
    return str(case["expected_word_digest"]) in hashes


def sign_test_p(n_pos: int, n_neg: int) -> float:
    """Exact two-sided sign-test p-value for n_pos positive
    and n_neg negative independent units under the symmetric
    null. This design's units are cases (independent
    generations), never the case-clustered pairs.
    """
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    tail: float = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


# ---------------------------------------------------------------------
# Aggregation (stage and validator share this derivation)
# ---------------------------------------------------------------------


def _live(record: dict[str, Any]) -> bool:
    """A record counts in readouts iff it completed, its case
    was not boundary-screened for the candidate, and its
    generation found the boundary.
    """
    return (
        record.get("detail") == ""
        and not record.get("case_screened")
        and bool(record.get("boundary_found"))
    )


def _rate(numerator: int, denominator: int) -> float | None:
    """A rate is None iff its denominator is 0 (the tree's
    global rule); denominators are always printed beside it.
    """
    return numerator / denominator if denominator else None


def _eligible(constraint: dict[str, Any], hashes: list[str]) -> bool:
    """Whether a (constraint, answer) pair enters a
    denominator at all. A zero-word answer makes a count
    constraint's verdict vacuous, so those pairs are excluded
    from every count-kind denominator (design section 6);
    every other pair is always eligible.
    """
    return not (constraint["kind"] == "count" and not hashes)


# skipcq: PY-R1000 - branches are frozen payload-contract cases pinned by tests
def aggregate_candidate(
    records: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """One candidate's density block, derived entirely from
    its records and the generated cases. The diff validator
    recomputes this from the payload's records and requires
    equality, so no field here may read anything else.
    """
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for record in records:
        by_key[(int(record["case_index"]), str(record["arm"]))] = record
    n_generated = len(records)
    n_boundary_screened = sum(1 for r in records if r.get("case_screened"))
    rest = [r for r in records if not r.get("case_screened")]
    n_detail_screened = sum(1 for r in rest if r.get("detail") != "")
    rest = [r for r in rest if r.get("detail") == ""]
    n_no_boundary = sum(1 for r in rest if not r.get("boundary_found"))
    live = [r for r in rest if r.get("boundary_found")]
    n_zero_word = sum(1 for r in live if len(r.get("word_hashes") or []) == 0)
    block: dict[str, Any] = {
        "available": bool(live),
        "detail": "" if live else "density_unavailable",
        "n_generated": n_generated,
        "n_boundary_screened": n_boundary_screened,
        "n_detail_screened": n_detail_screened,
        "n_no_boundary": n_no_boundary,
        "n_zero_word": n_zero_word,
        "n_live": len(live),
    }

    def hashes_of(record: dict[str, Any]) -> list[str]:
        """Return the word hashes recorded on a density record."""
        return list(record.get("word_hashes") or [])

    def case_of(record: dict[str, Any]) -> dict[str, Any]:
        """Return the case a density record indexes."""
        return cases[int(record["case_index"])]

    # Satisfaction by density, with the solo composition
    # control beside each point.
    density_rows: list[dict[str, Any]] = []
    for k in DENSITIES:
        arm = "solo_1" if k == 1 else f"k{k}"
        sat = checked = 0
        solo_sat = solo_checked = 0
        for record in live:
            if record["arm"] != arm:
                continue
            case = case_of(record)
            hashes = hashes_of(record)
            for constraint in case["constraints"][:k]:
                if not _eligible(constraint, hashes):
                    continue
                checked += 1
                sat += satisfied(constraint, hashes)
        for record in live:
            if not str(record["arm"]).startswith("solo_"):
                continue
            j = int(str(record["arm"]).split("_")[1])
            if j > k:
                continue
            case = case_of(record)
            constraint = case["constraints"][j - 1]
            if not _eligible(constraint, hashes_of(record)):
                continue
            solo_checked += 1
            solo_sat += satisfied(constraint, hashes_of(record))
        density_rows.append(
            {
                "density": k,
                "satisfied": sat,
                "checked": checked,
                "rate": _rate(sat, checked),
                "solo_satisfied": solo_sat,
                "solo_checked": solo_checked,
                "solo_rate": _rate(solo_sat, solo_checked),
            }
        )
    block["by_density"] = density_rows

    # Task-value presence per arm.
    presence_rows: list[dict[str, Any]] = []
    for arm in ARM_KEYS:
        present = total = 0
        for record in live:
            if record["arm"] != arm:
                continue
            total += 1
            present += task_present(case_of(record), hashes_of(record))
        presence_rows.append(
            {"arm": arm, "present": present, "live": total, "rate": _rate(present, total)}
        )
    block["task_presence"] = presence_rows

    # Position at full density: per-position rates (n = one
    # per case, descriptive) and the band-pooled headline.
    k16_live = [r for r in live if r["arm"] == "k16"]
    position_rows: list[dict[str, Any]] = []
    for position in range(1, CONSTRAINT_COUNT + 1):
        sat = total = 0
        for record in k16_live:
            constraint = case_of(record)["constraints"][position - 1]
            if not _eligible(constraint, hashes_of(record)):
                continue
            total += 1
            sat += satisfied(constraint, hashes_of(record))
        position_rows.append(
            {
                "position": position,
                "satisfied": sat,
                "checked": total,
                "rate": _rate(sat, total),
            }
        )
    block["position_k16"] = position_rows
    bands: list[dict[str, Any]] = []
    for label, lo, hi in (("early", 1, 5), ("mid", 6, 11), ("late", 12, 16)):
        sat = sum(r["satisfied"] for r in position_rows[lo - 1 : hi])
        total = sum(r["checked"] for r in position_rows[lo - 1 : hi])
        bands.append({"band": label, "satisfied": sat, "checked": total, "rate": _rate(sat, total)})
    block["position_bands"] = bands

    # Per-kind satisfaction, solo vs k16.
    kind_rows: list[dict[str, Any]] = []
    for kind in KINDS:
        row: dict[str, Any] = {"kind": kind}
        for label, arms in (("solo", SOLO_ARMS), ("k16", ("k16",))):
            sat = checked = 0
            for record in live:
                if record["arm"] not in arms:
                    continue
                case = case_of(record)
                hashes = hashes_of(record)
                if label == "solo":
                    j = int(str(record["arm"]).split("_")[1])
                    constraint = case["constraints"][j - 1]
                    if constraint["kind"] != kind:
                        continue
                    if not _eligible(constraint, hashes):
                        continue
                    checked += 1
                    sat += satisfied(constraint, hashes)
                else:
                    for constraint in case["constraints"]:
                        if constraint["kind"] != kind:
                            continue
                        if not _eligible(constraint, hashes):
                            continue
                        checked += 1
                        sat += satisfied(constraint, hashes)
            row[f"{label}_satisfied"] = sat
            row[f"{label}_checked"] = checked
            row[f"{label}_rate"] = _rate(sat, checked)
        kind_rows.append(row)
    block["by_kind"] = kind_rows

    # Joint satisfaction at k16.
    joint_sat = 0
    for record in k16_live:
        case = case_of(record)
        hashes = hashes_of(record)
        if all(satisfied(c, hashes) for c in case["constraints"]):
            joint_sat += 1
    block["joint_k16"] = {
        "satisfied": joint_sat,
        "live": len(k16_live),
        "rate": _rate(joint_sat, len(k16_live)),
    }

    # Sham readout over cases live in all three arms.
    sham_cases = [
        case_index
        for case_index in range(len(cases))
        if all(
            (case_index, arm) in by_key and _live(by_key[(case_index, arm)])
            for arm in ("solo_1", "sham", "k16")
        )
    ]
    sham_row: dict[str, Any] = {"cases": len(sham_cases)}
    for arm in ("solo_1", "sham", "k16"):
        sat = 0
        for case_index in sham_cases:
            record = by_key[(case_index, arm)]
            constraint = cases[case_index]["constraints"][0]
            if not _eligible(constraint, hashes_of(record)):
                continue
            sat += satisfied(constraint, hashes_of(record))
        sham_row[f"{arm}_satisfied"] = sat
        sham_row[f"{arm}_rate"] = _rate(sat, len(sham_cases))
    if sham_cases:
        sham_row["sham_block_words_mean"] = sum(
            arm_block_words(cases[i], "sham") for i in sham_cases
        ) / len(sham_cases)
        sham_row["k16_block_words_mean"] = sum(
            arm_block_words(cases[i], "k16") for i in sham_cases
        ) / len(sham_cases)
    else:
        sham_row["sham_block_words_mean"] = None
        sham_row["k16_block_words_mean"] = None
    block["sham"] = sham_row

    # Discordance: eligible solo-vs-k16 pairs, aggregated per
    # case for the decision.
    per_position: list[dict[str, Any]] = [
        {"position": position, "f": 0, "p": 0} for position in range(1, CONSTRAINT_COUNT + 1)
    ]
    total_f = total_p = 0
    case_discordance: list[int] = []
    for case_index, case in enumerate(cases):
        d_c = 0
        for position in range(1, CONSTRAINT_COUNT + 1):
            solo = by_key.get((case_index, f"solo_{position}"))
            full = by_key.get((case_index, "k16"))
            if solo is None or full is None or not _live(solo) or not _live(full):
                continue
            constraint = case["constraints"][position - 1]
            if not _eligible(constraint, hashes_of(solo)) or not _eligible(
                constraint, hashes_of(full)
            ):
                continue
            solo_ok = satisfied(constraint, hashes_of(solo))
            full_ok = satisfied(constraint, hashes_of(full))
            row = per_position[position - 1]
            if solo_ok and not full_ok:
                row["f"] += 1
                total_f += 1
                d_c += 1
            elif full_ok and not solo_ok:
                row["p"] += 1
                total_p += 1
                d_c -= 1
        case_discordance.append(d_c)
    block["discordance_by_position"] = per_position
    discordant_cases = [d for d in case_discordance if d != 0]
    n_pos = sum(1 for d in discordant_cases if d > 0)
    n_neg = sum(1 for d in discordant_cases if d < 0)
    p_value = sign_test_p(n_pos, n_neg)
    decision: dict[str, Any] = {
        "f": total_f,
        "p": total_p,
        "discordant_cases": len(discordant_cases),
        "cases_positive": n_pos,
        "cases_negative": n_neg,
        "sign_p": p_value,
    }
    if len(discordant_cases) < MIN_DISCORDANT_CASES:
        decision["verdict"] = "insufficient"
        decision["falsified"] = None
    else:
        falsified = total_f > total_p and p_value <= ALPHA
        decision["verdict"] = "falsified" if falsified else "not_falsified"
        decision["falsified"] = falsified
    block["decision"] = decision
    return block
