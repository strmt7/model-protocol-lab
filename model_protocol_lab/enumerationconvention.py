"""Enumeration-convention probe (cycle 233).

Design: docs/ENUMERATION_DESIGN.md (the wave-12 round-4
candidate as adjudicated BUILD-as-amended, amendments
A1-A9 folded in). The probe varies Sclar et al.'s
enumeration grammar along the dimension their section-4.3
distributional analysis identifies as active -- the
*indexing system* (F_item2) -- while holding the wrapper
dimension's representative and an index-free bullet as
expected-null arms.

A template author choosing one house list style for
prompts rendered across many chat templates learns, per
candidate (template family) and per enumeration arm,
whether changing only the list markers on identical list
content moves the checkpoint's answers at the frozen
floors, standardising on the incumbent arabic style
unless an arm fires. The probe measures the marker
convention on this generated population; it does not
measure a best style, multiple-choice option labels, or
list content or order (content and order are
byte-identical across arms by construction).

Frozen shape: a generated 24-case fact-recall panel
(stage-local, never the user's Split). Every case is a
frozen list of K = 9 fact sentences (inspection codes),
rendered in six arms that differ only in the per-item
marker: arabic ``1.`` (control), letters ``A.`` / ``a.``,
Roman ``I.``, the wrapper ``(1)``, and the index-free
bullet ``•``. Questions address their item by content
(the topic word) and by nothing else, enforced
mechanically by the section-4 validator (V1-V4).

Headline (design section 6): per (candidate, arm), over
the cases completed in both control and arm AND
id-divergent for that arm, D = control passes, J = of
those the arm cells decomposed ``wrong_value``. The
section-5 decomposition is arm-relative: marker echoes
are interpreted in the arm's own system, and (amendment
A1) a wrong value wearing exactly one marker-shaped
prefix or suffix counts toward J exactly as a bare wrong
value does. Labels are the sibling set only, guarded by
the completion rule over the id-divergent domain.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

__all__ = [
    "ARM_KEYS",
    "BULLET",
    "CASE_COUNT",
    "CELL_CLASSES",
    "CELL_DETAILS",
    "GENERATION_BUDGET",
    "GENERATOR_SPEC",
    "GROUP",
    "INDICES",
    "INTRO",
    "LABELS",
    "MARKER_TABLE",
    "MAX_CANDIDATES",
    "MAX_GROUND_RECORDS",
    "MAX_INCOMPLETE",
    "MAX_RECORDS_BYTES",
    "MIN_D",
    "MIN_J",
    "NEUTRAL_STATEMENTS",
    "OPERATING_CHARACTERISTICS",
    "PAIR_CLASSES",
    "POOL_DIGESTS",
    "PROTOCOL_VERSION",
    "REDRAW_LIMIT",
    "ROMAN_TABLE",
    "STAGE_DETAILS",
    "STRIDE",
    "TOPICS",
    "TREATED_ARMS",
    "WORDS",
    "RedrawExhausted",
    "aggregate_candidate",
    "arm_content",
    "case_material_sha256",
    "classify_headline",
    "decompose_cell",
    "decompose_pair",
    "digest_ids",
    "digest_text",
    "generate_case",
    "generate_cases",
    "ground_sha256",
    "item_sentence",
    "list_block",
    "marker_for",
    "protocol_fingerprint",
    "protocol_object",
    "question_text",
    "records_bytes",
    "validate_case",
    "validate_question",
]

PROTOCOL_VERSION = 1
GROUP = "enumeration_convention"
CASE_COUNT = 24

# The frozen pools (design section 3). TOPICS is the
# sibling 24-word ASCII topic list (the copies carried by
# confusableskeleton and compatibilityform; NOT
# canonicalform's diacritic list and NOT density's
# 30-entry list). WORDS is canonicalform's 24-word pool;
# NEUTRAL_STATEMENTS is density's 24-statement pool. All
# three are pinned by content digest in POOL_DIGESTS
# (amendment A8, extended to TOPICS at build).
TOPICS = (
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
    "pepper",
    "quartz",
    "birch",
)
WORDS = (
    "alder",
    "basalt",
    "cedar",
    "drift",
    "elm",
    "fern",
    "glade",
    "heath",
    "iris",
    "jade",
    "kelp",
    "loom",
    "moss",
    "nova",
    "ochre",
    "pine",
    "quill",
    "reed",
    "sage",
    "tide",
    "urn",
    "vale",
    "wren",
    "yew",
)
NEUTRAL_STATEMENTS = (
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


def _pool_digest(pool: tuple[str, ...]) -> str:
    """SHA-256 of the newline-joined pool (the A8 pin
    convention)."""
    return hashlib.sha256("\n".join(pool).encode("utf-8")).hexdigest()


POOL_DIGESTS = {
    "topics": _pool_digest(TOPICS),
    "words": _pool_digest(WORDS),
    "neutral_statements": _pool_digest(NEUTRAL_STATEMENTS),
}

INDICES = (1, 2, 3, 4, 5, 6, 7, 8, 9)  # K = len(INDICES) = 9 items
STRIDE = 5
BULLET = "•"  # U+2022 BULLET
INTRO = "Inspection codes:"

# The frozen marker rendering table (design section 3),
# indices 1..9; Roman in subtractive form (IV, IX).
ROMAN_TABLE: dict[int, str] = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
    8: "VIII",
    9: "IX",
}
ARM_KEYS = (
    "control_arabic",
    "alpha_upper",
    "alpha_lower",
    "roman",
    "wrapper",
    "bullet",
)
TREATED_ARMS = ARM_KEYS[1:]
MARKER_TABLE: dict[str, tuple[str, ...]] = {
    "control_arabic": tuple(f"{t}." for t in INDICES),
    "alpha_upper": tuple("ABCDEFGHI"[t - 1] + "." for t in INDICES),
    "alpha_lower": tuple("abcdefghi"[t - 1] + "." for t in INDICES),
    "roman": tuple(ROMAN_TABLE[t] + "." for t in INDICES),
    "wrapper": tuple(f"({t})" for t in INDICES),
    "bullet": tuple(BULLET for _t in INDICES),
}
ARM_SYSTEMS = {
    "control_arabic": "arabic",
    "alpha_upper": "alpha_upper",
    "alpha_lower": "alpha_lower",
    "roman": "roman",
    "wrapper": "wrapper",
    "bullet": "bullet",
}

MIN_D = 10
MIN_J = 6
MAX_INCOMPLETE = 6
REDRAW_LIMIT = 1000
GENERATOR_SPEC = (
    "draw(seed, index, slot, counter) = int.from_bytes("
    "sha256(f'{seed}|enumerationconvention|{index}|{slot}|{counter}')"
    ".digest()[:8], 'big'); the material index is the 0-based case "
    "number; slots 2 and 3 at counter 0: neutral indices = draw % "
    "len(NEUTRAL_STATEMENTS) (slot 2 is the first neutral, slot 3 "
    "the second; the second advances +1 mod len on collision); "
    "slot 20 at counter 0: target position = 1 + draw % "
    "len(INDICES); list topics are the stride-5 base sequence "
    "B_j = TOPICS[(index + 5*j) % len(TOPICS)] for j = 0..8 with "
    "the target B_0 placed at the drawn position and B_1..B_8 in "
    "order at the remaining positions; item values are drawn per "
    "position t = 1..9 in order (word slot 30 + (t-1): WORDS[draw "
    "% len(WORDS)]; number slot 40 + (t-1): 1000 + draw % 9000) "
    "with panel traversal in case-then-position order; a value "
    "colliding with any earlier value in the panel redraws that "
    "item's two slots together -- the item's word and number slots "
    "share one redraw counter -- and give-up raises when the "
    "counter exceeds 1000 (the stage fails closed)"
)
MAX_CANDIDATES = 24
MAX_GROUND_RECORDS = MAX_CANDIDATES * CASE_COUNT
GENERATION_BUDGET = len(ARM_KEYS) * MAX_GROUND_RECORDS
MAX_RECORDS_BYTES = 12 * 1024 * 1024

CELL_DETAILS = ("arm_context_exceeded", "arm_evaluation_failed")
STAGE_DETAILS = (
    "cancelled",
    "enumeration_convention_record_budget_exceeded",
    "enumeration_convention_screen_failed",
)
CELL_CLASSES = (
    "pass",
    "boundary_only",
    "wrong_value",
    "echo",
    "form_only",
    "non_answer",
)
PAIR_CLASSES = ("exact_only", "boundary_only", "mixed")
LABELS = (
    "insufficient",
    "no_discordance_observed",
    "discordant_asymmetric",
    "below_threshold",
)

# Design section 6 operating characteristics, printed
# beside every label (never re-derived at runtime; the
# build's tests re-derive every figure exactly, including
# the exact paired column, the guard worlds of section 6,
# and the corrected required-N values of section 3 /
# amendment A6).
OPERATING_CHARACTERISTICS: dict[str, Any] = {
    "fire_competent_independent": 0.0173,
    "fire_competent_paired": 0.0044,
    "fire_weak_independent": 0.1743,
    "fire_random_independent": 0.5503,
    "power_drop5_base90": 0.0947,
    "power_drop10_base90": 0.2542,
    "power_drop25_base90": 0.8159,
    "power_drop25_base85": 0.8761,
    "power_drop25_paired": 0.6981,
    "power_drop50_base90": 0.999,
    "sensitivity18_drop25": 0.5214,
    "sensitivity18_competent": 0.0038,
    "sensitivity12_drop25": 0.1379,
    "sensitivity12_competent": 0.0003,
    "transplant_fire_competent_independent": 0.8897,
    "familywise_competent_expected_120": 2.0729,
    "familywise_competent_any_120": 0.8764,
    "familywise_paired_expected_120": 0.5331,
    "familywise_paired_any_120": 0.4139,
    "familywise_random_expected_120": 66.0368,
    "familywise_random_any_120": 1.0,
    "world_a1_guarded_fire": 0.0055,
    "world_a1_unguarded_fire": 0.0186,
    "world_a1_guarded_insufficient": 0.9923,
    "world_a2_guarded_fire": 0.0173,
    "world_a2_unguarded_fire": 0.0173,
    "world_a2_guarded_insufficient": 0.2897,
    "world_b_unguarded_fire": 0.2551,
    "world_b_unguarded_below_threshold": 0.7277,
    "world_b_guarded_fire": 0.1731,
    "world_b_guarded_below_threshold": 0.0117,
    "world_b_guarded_insufficient": 0.8152,
    "world_b777_guarded_below_threshold": 0.011,
    "world_c_guarded_fire": 0.8893,
    "world_c_unguarded_fire": 0.9399,
    "world_c_guarded_below_threshold": 0.0016,
    "required_n_smallest": 117,
    "required_n_min_d": 0,
    "required_n_min_j": 18,
    "required_n_null": 0.0172,
    "required_n_power10": 0.8023,
    "required_n_grid120_power10": 0.7657,
    "required_n_grid126_power10": 0.8336,
    "required_n_grid132_power10": 0.8325,
    "required_n_cost": 16848,
}

_VALUE_RE = re.compile(r"[a-z]+-[0-9]{4}")
_ROMAN_RE = re.compile(r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})")
_ROMAN_VAL = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_ASCII_DIGITS = "0123456789"
_ASCII_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ASCII_LOWER = "abcdefghijklmnopqrstuvwxyz"


class RedrawExhausted(Exception):
    """The value-redraw give-up bound was reached.

    Counting pinned by amendment A8: the item's word and
    number slots share one redraw counter; the counter
    increments once per colliding redraw; exhaustion is
    raised when the counter exceeds REDRAW_LIMIT. The
    stage maps this to the fail-closed detail
    ``enumeration_convention_screen_failed``.
    """


def digest_text(text: str) -> str:
    """SHA-256 hex digest of a text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_ids(ids: Any) -> str:
    """SHA-256 hex digest of a render id sequence."""
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode("ascii")).hexdigest()


def _derive(seed: int, index: int, slot: int, counter: int) -> int:
    """The frozen SHA-256 draw (design section 3)."""
    material = f"{seed}|enumerationconvention|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _build_case(seed: int, index: int, seen: set[str]) -> dict[str, Any]:
    """One panel case, drawing against the panel-global
    ``seen`` value set (mutated in place).

    Draw order per case (design section 3): neutrals,
    target position, then item values in position order.
    A value colliding with any earlier value in the panel
    redraws that item's two slots at counter+1; exceeding
    REDRAW_LIMIT raises RedrawExhausted.
    """
    first = _derive(seed, index, 2, 0) % len(NEUTRAL_STATEMENTS)
    second = _derive(seed, index, 3, 0) % len(NEUTRAL_STATEMENTS)
    if second == first:
        second = (first + 1) % len(NEUTRAL_STATEMENTS)
    position = 1 + _derive(seed, index, 20, 0) % len(INDICES)
    base = [TOPICS[(index + STRIDE * j) % len(TOPICS)] for j in range(len(INDICES))]
    topics: list[str] = []
    rest = iter(base[1:])
    for t in INDICES:
        topics.append(base[0] if t == position else next(rest))
    values: list[str] = []
    for t in INDICES:
        counter = 0
        while True:
            word = WORDS[_derive(seed, index, 30 + (t - 1), counter) % len(WORDS)]
            number = 1000 + _derive(seed, index, 40 + (t - 1), counter) % 9000
            value = f"{word}-{number}"
            if value not in seen:
                break
            counter += 1
            if counter > REDRAW_LIMIT:
                raise RedrawExhausted
        seen.add(value)
        values.append(value)
    target_topic = TOPICS[index]
    return {
        "fixture_id": f"enumerationconvention-{index:02d}",
        "index": index,
        "topic": target_topic,
        "position": position,
        "topics": topics,
        "values": values,
        "expected": values[position - 1],
        "question": f"What is the {target_topic} inspection code?",
        "neutral_indices": [first, second],
    }


def generate_cases(seed: int) -> list[dict[str, Any]]:
    """The frozen 24-case panel for ``seed``.

    Values are pairwise distinct across the whole panel
    (216 values); the redraw discipline of design
    section 3 applies, and exhaustion raises
    RedrawExhausted.
    """
    seen: set[str] = set()
    return [_build_case(seed, index, seen) for index in range(CASE_COUNT)]


def generate_case(seed: int, index: int) -> dict[str, Any]:
    """One panel case: case ``index`` under ``seed``.

    The panel prefix 0..index is regenerated so the
    panel-global redraw discipline sees exactly the
    earlier values the full panel would see.
    """
    seen: set[str] = set()
    case: dict[str, Any] = {}
    for i in range(index + 1):
        case = _build_case(seed, i, seen)
    return case


def marker_for(arm: str, t: int) -> str:
    """The frozen marker of ``arm`` at index ``t``."""
    if arm not in MARKER_TABLE:
        raise ValueError(f"unknown enumeration-convention arm: {arm!r}")
    return MARKER_TABLE[arm][t - 1]


def item_sentence(topic: str, value: str) -> str:
    """The frozen per-item fact sentence."""
    return f"The {topic} inspection code is {value}."


def question_text(case: dict[str, Any]) -> str:
    """The frozen question template."""
    return f"What is the {case['topic']} inspection code?"


def list_block(case: dict[str, Any], arm: str) -> str:
    """The frozen list block for ``arm``: the intro line
    and the nine item lines, marker + one ASCII space +
    the item sentence, joined by single newlines.
    """
    if arm not in MARKER_TABLE:
        raise ValueError(f"unknown enumeration-convention arm: {arm!r}")
    lines = [
        f"{MARKER_TABLE[arm][t - 1]} {item_sentence(case['topics'][t - 1], case['values'][t - 1])}"
        for t in INDICES
    ]
    return INTRO + "\n" + "\n".join(lines)


def arm_content(case: dict[str, Any], arm: str) -> str:
    """The full prompt content of one case under ``arm``."""
    if arm not in MARKER_TABLE:
        raise ValueError(f"unknown enumeration-convention arm: {arm!r}")
    first, second = (int(i) for i in case["neutral_indices"])
    return (
        f"{NEUTRAL_STATEMENTS[first]} {list_block(case, arm)} "
        f"{NEUTRAL_STATEMENTS[second]} {question_text(case)}"
    )


# ------------------------------------------------------------------
# Section 4: the content-addressing validator (V1-V4).
#
# A question is admissible iff all of V1 (template), V2
# (blocklist), V3 (marker disjointness), V4 (answer
# invariance) hold. The sub-rules are separate functions
# so the build's tests pin each independently: a future
# template change cannot silently disarm the guard.
# ------------------------------------------------------------------

_FORBIDDEN: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ordinal",
        re.compile(
            r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|"
            r"ninth|tenth|last|next|previous|preceding|following|above|"
            r"below)\b",
            re.IGNORECASE,
        ),
    ),
    ("item-ref", re.compile(r"\bitem\s*[0-9]", re.IGNORECASE)),
    ("option-ref", re.compile(r"\boption\s*[A-Za-z0-9]", re.IGNORECASE)),
    ("line-ref", re.compile(r"\bline\s*[0-9]", re.IGNORECASE)),
    ("marker-word", re.compile(r"\bmarker\b", re.IGNORECASE)),
    ("bracket-or-bullet", re.compile(r"[\[\]()•]")),
)
_MARKER_TOKENS = frozenset(marker for arm in ARM_KEYS for marker in MARKER_TABLE[arm])
# The V3 bare-glyph set, completed by amendment A4: the
# digits 1-9, the letters A-I / a-i, the Roman strings
# for 1..9 in both cases, and every remaining glyph the
# section-5 scorer parses (X/x, L/l, C/c, D/d, M/m).
_BARE_GLYPHS = frozenset(
    {str(t) for t in INDICES}
    | set("ABCDEFGHIabcdefghi")
    | {ROMAN_TABLE[t] for t in INDICES}
    | {ROMAN_TABLE[t].lower() for t in INDICES}
    | set("XxLlCcDdMm")
)


def _forbidden_hits(question: str) -> list[str]:
    """V2 standalone: the names of every forbidden
    pattern the question matches."""
    return [name for name, pattern in _FORBIDDEN if pattern.search(question)]


def _marker_token_hit(question: str) -> bool:
    """V3 standalone: whether any whitespace-delimited
    token of the question is a marker rendering of any
    arm at any index, or a bare marker glyph."""
    return any(
        token in _MARKER_TOKENS or token in _BARE_GLYPHS
        for token in question.replace("?", " ").split()
    )


def _value_grammar_ok(value: Any) -> bool:
    """The frozen value grammar, fullmatch semantics:
    ``word-NNNN`` with word in WORDS and the number in
    [1000, 9999] (four ASCII digits, no leading zero in
    range)."""
    if type(value) is not str or not _VALUE_RE.fullmatch(value):
        return False
    word, _, number = value.partition("-")
    return word in WORDS and 1000 <= int(number) <= 9999


def validate_question(question: Any, target_topic: Any, expected: Any) -> bool:
    """V1-V4 for one (question, target topic, expected)
    triple. Never raises: any violation is False.
    """
    if type(question) is not str or type(target_topic) is not str:
        return False
    if question != f"What is the {target_topic} inspection code?":
        return False
    if _forbidden_hits(question):
        return False
    if _marker_token_hit(question):
        return False
    return _value_grammar_ok(expected)


def _normalise_markers(content: str, arm: str) -> str:
    """Replace the k-th list line's marker with the
    control marker for index k (by line order, never by
    search: the bullet marker is the same at every
    position). The result equals the control content iff
    only the marker differs across arms.
    """
    out: list[str] = []
    k = 0
    for line in content.split("\n"):
        hit: int | None = None
        for t in INDICES:
            if line.startswith(MARKER_TABLE[arm][t - 1] + " "):
                hit = t
                break
        if hit is None:
            out.append(line)
        else:
            k += 1
            out.append(
                MARKER_TABLE["control_arabic"][k - 1] + line[len(MARKER_TABLE[arm][hit - 1]) :]
            )
    return "\n".join(out)


def validate_case(case: dict[str, Any]) -> bool:
    """The full generator battery for one case (design
    sections 3-4): frozen shapes, the stride topic
    construction with the target at the drawn position,
    within-case value distinctness, the V1-V4 question
    validator, pure-ASCII expected, and the per-arm
    invariants (pairwise-distinct contents; marker
    normalisation reproduces the control content, so
    only the marker differs). Never raises: any
    violation is False.
    """
    try:
        return _validate_case(case)
    except Exception:  # noqa: BLE001
        return False


def _validate_case(case: dict[str, Any]) -> bool:
    """Validate case (worker for validate_case)."""
    fixture_id = case.get("fixture_id")
    if type(fixture_id) is not str or not fixture_id.startswith("enumerationconvention-"):
        return False
    index = int(fixture_id.rsplit("-", 1)[1])
    if not 0 <= index < CASE_COUNT or fixture_id != f"enumerationconvention-{index:02d}":
        return False
    if type(case.get("index")) is not int or case.get("index") != index:
        return False
    topic = case.get("topic")
    if topic != TOPICS[index]:
        return False
    position = case.get("position")
    if type(position) is not int or position not in INDICES:
        return False
    topics = case.get("topics")
    if type(topics) is not list or len(topics) != len(INDICES):
        return False
    base = [TOPICS[(index + STRIDE * j) % len(TOPICS)] for j in range(len(INDICES))]
    expected_topics: list[str] = []
    rest = iter(base[1:])
    for t in INDICES:
        expected_topics.append(base[0] if t == position else next(rest))
    if topics != expected_topics or len(set(topics)) != len(INDICES):
        return False
    values = case.get("values")
    if type(values) is not list or len(values) != len(INDICES):
        return False
    if any(not _value_grammar_ok(value) for value in values):
        return False
    if len(set(values)) != len(INDICES):
        return False
    expected = case.get("expected")
    if expected != values[position - 1] or type(expected) is not str or not expected.isascii():
        return False
    neutral_indices = case.get("neutral_indices")
    if (
        type(neutral_indices) is not list
        or len(neutral_indices) != 2
        or any(type(i) is not int or not 0 <= i < len(NEUTRAL_STATEMENTS) for i in neutral_indices)
        or neutral_indices[0] == neutral_indices[1]
    ):
        return False
    if case.get("question") != question_text(case):
        return False
    if not validate_question(case["question"], topic, expected):
        return False
    contents = {arm: arm_content(case, arm) for arm in ARM_KEYS}
    if len(set(contents.values())) != len(ARM_KEYS):
        return False
    control = contents["control_arabic"]
    return all(_normalise_markers(contents[arm], arm) == control for arm in TREATED_ARMS)


# ------------------------------------------------------------------
# Section 5: the decomposition (ordered decision list;
# total). Marker interpretation is arm-relative, glyphs
# are pinned to ASCII (amendment A2), echo is string
# identity with the arm's own marker (amendment A3), and
# class 3 reaches the once-marker-stripped answer as
# well as the bare answer (amendment A1).
# ------------------------------------------------------------------


def _is_ascii_digits(text: str) -> bool:
    """Whether every character is an ASCII digit (never
    Unicode-lenient: amendment A2)."""
    return bool(text) and all(char in _ASCII_DIGITS for char in text)


def _is_ascii_letters(text: str) -> bool:
    """Whether every character is an ASCII letter."""
    return bool(text) and all(char in _ASCII_UPPER or char in _ASCII_LOWER for char in text)


def parse_roman(glyph: str) -> int | None:
    """Subtractive parse of a canonical Roman numeral.

    None unless the glyph is a canonical Roman numeral
    (all-upper or all-lower ASCII; IIII and VX are not
    canonical, hence not Roman at all). ASCII-pinned
    (amendment A2): no Unicode case folding or digit
    test is involved.
    """
    if not glyph or not _is_ascii_letters(glyph):
        return None
    if glyph.upper() != glyph and glyph.lower() != glyph:
        return None  # mixed case is not a canonical rendering
    upper = glyph.upper()
    if any(char not in _ROMAN_VAL for char in upper):
        return None
    if not _ROMAN_RE.fullmatch(upper):
        return None
    total, prev = 0, 0
    for char in reversed(upper):
        value = _ROMAN_VAL[char]
        total += -value if value < prev else value
        prev = max(prev, value)
    return total


def glyph_ok(glyph: str) -> bool:
    """A glyph is marker-shaped iff it is arabic digits
    (1-2, ASCII), a single ASCII letter, or a canonical
    Roman numeral (all-upper or all-lower).
    Multi-letter non-Roman strings (IIII, AB) are not
    marker-shaped."""
    if _is_ascii_digits(glyph):
        return len(glyph) <= 2
    if not _is_ascii_letters(glyph):
        return False
    if len(glyph) == 1:
        return True
    if glyph.upper() != glyph and glyph.lower() != glyph:
        return False  # mixed case is not a canonical rendering
    return parse_roman(glyph) is not None


def split_token(head: str) -> tuple[str, str] | None:
    """Classify a candidate marker token's FORM, arm-free.

    Returns (form, glyph) or None if the token is not
    marker-shaped. Forms: dotted / wrapper-parens /
    bracket / close-paren / bullet. Digit tests are
    ASCII-only (amendment A2).
    """
    if head == BULLET:
        return ("bullet", BULLET)
    match = re.fullmatch(r"\(([0-9]{1,2})\)", head)
    if match:
        return ("wrapper-parens", match.group(1))
    match = re.fullmatch(r"\[([^\]]{1,6})\]", head)
    if match and glyph_ok(match.group(1)):
        return ("bracket", match.group(1))
    match = re.fullmatch(r"(.{1,6})([.)])", head)
    if match and glyph_ok(match.group(1)):
        return ("close-paren" if match.group(2) == ")" else "dotted", match.group(1))
    return None


def index_in_system(glyph: str, system: str) -> int | None:
    """The index a bare glyph denotes in ONE named
    system, or None. Case-sensitive except Roman (upper
    and lower both parse). ASCII-pinned (amendment A2).
    """
    if system in ("arabic", "wrapper"):
        return int(glyph) if _is_ascii_digits(glyph) else None
    if system == "alpha_upper":
        return ord(glyph) - ord("A") + 1 if len(glyph) == 1 and "A" <= glyph <= "Z" else None
    if system == "alpha_lower":
        return ord(glyph) - ord("a") + 1 if len(glyph) == 1 and "a" <= glyph <= "z" else None
    if system == "roman":
        return parse_roman(glyph)
    return None


def _own_marker_strings(arm: str, target: int) -> frozenset[str]:
    """The arm's own marker strings for the target index
    (amendment A3: echo is string identity with the
    marker table's form, or -- Roman only -- its
    all-lower variant)."""
    marker = MARKER_TABLE[arm][target - 1]
    if arm == "roman":
        return frozenset({marker, ROMAN_TABLE[target].lower() + "."})
    return frozenset({marker})


def is_echo(head: str, arm: str, target: int) -> bool:
    """Class-4 predicate: the token is the arm's OWN
    marker for the target index, string-identical
    (amendment A3 -- index-equality parsings such as
    ``03.`` or ``(03)`` are form_only, never echo)."""
    if arm not in MARKER_TABLE or target not in INDICES:
        return False
    return head in _own_marker_strings(arm, target)


def strip_one(text: str, where: str) -> str | None:
    """Remove exactly one marker-shaped token at the
    given edge plus its single separating space. Returns
    the remainder or None."""
    if where == "lead":
        if " " not in text:
            return None
        head, tail = text.split(" ", 1)
        return tail if split_token(head) is not None else None
    if " " not in text:
        return None
    body, tail = text.rsplit(" ", 1)
    return body if split_token(tail) is not None else None


def decompose_cell(
    cell: dict[str, Any], expected: str, expected_digest: str, arm: str, target: int
) -> str:
    """The ordered decision list of design section 5.

    Exactly one class per completed cell:

    1. pass -- the verdict passed.
    2. boundary_only -- the answer digest equals the
       expected digest; the boundary conjunct alone
       failed.
    3. wrong_value -- the answer, or the answer after
       removing exactly one marker-shaped token (leading
       or trailing, with its single separating space),
       parses under the frozen value grammar (fullmatch)
       and is not the expected value (amendment A1: a
       wrong value wearing a marker prefix counts toward
       J exactly as a bare one does; a stripped remainder
       equal to the expected value continues to classes
       4/5; at most one token is ever stripped;
       non-marker prefixes never strip).
    4. echo -- removing exactly one leading marker token
       yields exactly the expected value, and the token
       is the arm's own marker for the target index
       (string identity, amendment A3).
    5. form_only -- removing exactly one leading or
       trailing marker-shaped token yields exactly the
       expected value, but class 4 does not apply.
    6. non_answer -- everything else, including empty
       answers, bare markers, bare counts, article
       prefixes, doubled markers, no-space fusions,
       non-canonical Roman prefixes, non-ASCII digit
       heads (amendment A2), and out-of-grammar value
       changes.
    """
    if cell.get("verdict") is True:
        return "pass"
    digest = cell.get("answer_digest")
    if digest is not None and digest == expected_digest:
        return "boundary_only"
    text = cell.get("answer_text")
    if type(text) is not str:
        return "non_answer"
    if text != expected:
        if _value_grammar_ok(text):
            return "wrong_value"
        for where in ("lead", "trail"):
            remainder = strip_one(text, where)
            if remainder is not None and remainder != expected and _value_grammar_ok(remainder):
                return "wrong_value"
    lead = strip_one(text, "lead")
    if lead is not None and lead == expected:
        head = text.split(" ", 1)[0]
        return "echo" if is_echo(head, arm, target) else "form_only"
    if strip_one(text, "trail") == expected:
        return "form_only"  # trailing placement is never the arm marker
    return "non_answer"


def decompose_pair(control_cell: dict[str, Any], arm_cell: dict[str, Any]) -> str:
    """The sibling total pair partition (design section 5)."""
    if control_cell.get("answer_digest") == arm_cell.get("answer_digest"):
        return "boundary_only"
    if control_cell.get("boundary_found") and arm_cell.get("boundary_found"):
        return "exact_only"
    return "mixed"


def classify_headline(control_passes: int, wrong_values: int, n_incomplete: int) -> str:
    """The guarded label: a total function of D, J, and
    the domain incomplete count k (design section 6).
    """
    if n_incomplete > MAX_INCOMPLETE:
        return "insufficient"
    if control_passes < MIN_D:
        return "insufficient"
    if wrong_values >= MIN_J:
        return "discordant_asymmetric"
    if wrong_values == 0:
        return "no_discordance_observed" if n_incomplete == 0 else "insufficient"
    return "below_threshold" if wrong_values + n_incomplete < MIN_J else "insufficient"


def _cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the enumerationconvention cell get."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _detail_get(record: dict[str, Any], key: str) -> str:
    """Return the enumerationconvention detail get."""
    details = record.get("cell_details")
    if not isinstance(details, dict):
        return ""
    detail = details.get(key, "")
    return detail if type(detail) is str else ""


def _verdict(record: dict[str, Any], key: str) -> bool:
    """Return the enumerationconvention verdict."""
    cell = _cell_get(record, key)
    return bool(cell is not None and cell.get("verdict") is True)


def _ids_identical(record: dict[str, Any], arm: str) -> bool:
    """Derived from the recorded id digests alone."""
    digests = record.get("id_digests")
    if not isinstance(digests, dict):
        return False
    control = digests.get("control_arabic")
    return type(control) is str and control != "" and control == digests.get(arm)


def _rate(numerator: int, denominator: int) -> float | None:
    """Return the enumerationconvention rate."""
    return (numerator / denominator) if denominator else None


def aggregate_candidate(
    records: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Per-arm rates and decompositions, the id report,
    the per-position length join (amendment A5), and the
    guarded headlines for one candidate's ground records.
    """
    case_by_fixture = {str(case["fixture_id"]): case for case in cases}
    expected_by_fixture = {
        fixture: str(case["expected"]) for fixture, case in case_by_fixture.items()
    }
    arms: dict[str, Any] = {}
    for key in ARM_KEYS:
        n_completed = 0
        n_pass = 0
        n_context = 0
        n_failed = 0
        decomposition = dict.fromkeys(CELL_CLASSES, 0)
        for record in records:
            detail = _detail_get(record, key)
            if detail == "arm_context_exceeded":
                n_context += 1
            elif detail == "arm_evaluation_failed":
                n_failed += 1
            cell = _cell_get(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell.get("verdict") is True:
                n_pass += 1
            case = case_by_fixture.get(str(record.get("fixture", "")))
            target = int(case["position"]) if case is not None else 0
            decomposition[
                decompose_cell(
                    cell,
                    expected_by_fixture.get(str(record.get("fixture", "")), ""),
                    str(record.get("expected_digest", "")),
                    key,
                    target,
                )
            ] += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": _rate(n_pass, n_completed),
            "n_context_exceeded": n_context,
            "n_evaluation_failed": n_failed,
            "decomposition": decomposition,
        }
    ids: dict[str, Any] = {}
    for arm in TREATED_ARMS:
        deltas: list[int] = []
        n_ids_identical = 0
        n_same_ids_flips = 0
        for record in records:
            tokens = record.get("arm_tokens")
            tokens = tokens if isinstance(tokens, dict) else {}
            treated = tokens.get(arm)
            baseline = tokens.get("control_arabic")
            if type(treated) is int and type(baseline) is int:
                deltas.append(treated - baseline)
            else:
                deltas.append(0)
            if _ids_identical(record, arm):
                n_ids_identical += 1
                control_cell = _cell_get(record, "control_arabic")
                arm_cell = _cell_get(record, arm)
                if (
                    control_cell is not None
                    and arm_cell is not None
                    and (control_cell.get("verdict") is True) != (arm_cell.get("verdict") is True)
                ):
                    n_same_ids_flips += 1
        ids[arm] = {
            "n_ids_identical": n_ids_identical,
            "n_ids_divergent": len(records) - n_ids_identical,
            "id_length_delta": deltas,
            "max_abs_id_delta": max((abs(delta) for delta in deltas), default=0),
            "n_same_ids_verdict_flips": n_same_ids_flips,
        }
    headline: dict[str, Any] = {}
    positions: dict[str, Any] = {}
    for arm in TREATED_ARMS:
        domain = [record for record in records if not _ids_identical(record, arm)]
        paired = [
            record
            for record in domain
            if _cell_get(record, "control_arabic") is not None
            and _cell_get(record, arm) is not None
        ]
        n_incomplete = len(domain) - len(paired)
        control_passes = 0
        wrong_values = 0
        reverse = 0
        non_answer = 0
        pair_decomposition = dict.fromkeys(PAIR_CLASSES, 0)
        failing_classes = dict.fromkeys(CELL_CLASSES, 0)
        by_position: dict[int, dict[str, Any]] = {
            t: {"n_population": 0, "control_passes": 0, "wrong_values": 0, "deltas": []}
            for t in INDICES
        }
        for record in paired:
            control_cell = _cell_get(record, "control_arabic")
            arm_cell = _cell_get(record, arm)
            if control_cell is None or arm_cell is None:
                continue
            case = case_by_fixture.get(str(record.get("fixture", "")))
            target = int(case["position"]) if case is not None else 0
            expected = expected_by_fixture.get(str(record.get("fixture", "")), "")
            expected_digest = str(record.get("expected_digest", ""))
            arm_class = decompose_cell(arm_cell, expected, expected_digest, arm, target)
            if arm_class == "non_answer":
                non_answer += 1
            control_verdict = control_cell.get("verdict") is True
            arm_verdict = arm_cell.get("verdict") is True
            slot = by_position.get(target)
            if slot is not None:
                slot["n_population"] += 1
                tokens = record.get("arm_tokens")
                tokens = tokens if isinstance(tokens, dict) else {}
                treated_tokens = tokens.get(arm)
                baseline_tokens = tokens.get("control_arabic")
                if type(treated_tokens) is int and type(baseline_tokens) is int:
                    slot["deltas"].append(treated_tokens - baseline_tokens)
                if control_verdict:
                    slot["control_passes"] += 1
                    if arm_class == "wrong_value":
                        slot["wrong_values"] += 1
            if control_verdict:
                control_passes += 1
                if arm_class == "wrong_value":
                    wrong_values += 1
            elif arm_verdict:
                reverse += 1
            if control_verdict != arm_verdict:
                pair_decomposition[decompose_pair(control_cell, arm_cell)] += 1
                failing = arm_cell if control_verdict else control_cell
                failing_arm = arm if control_verdict else "control_arabic"
                failing_target = target
                failing_classes[
                    decompose_cell(failing, expected, expected_digest, failing_arm, failing_target)
                ] += 1
        headline[arm] = {
            "label": classify_headline(control_passes, wrong_values, n_incomplete),
            "n_population": len(paired),
            "n_incomplete": n_incomplete,
            "n_ids_divergent": len(domain),
            "control_passes": control_passes,
            "wrong_values": wrong_values,
            "n_reverse": reverse,
            "control_rate": _rate(control_passes, len(paired)),
            "arm_rate": _rate(
                sum(1 for record in paired if _verdict(record, arm)),
                len(paired),
            ),
            "n_non_answer": non_answer,
            "pair_decomposition": pair_decomposition,
            "failing_classes": failing_classes,
        }
        positions[arm] = [
            {
                "position": t,
                "n_population": by_position[t]["n_population"],
                "control_passes": by_position[t]["control_passes"],
                "wrong_values": by_position[t]["wrong_values"],
                "mean_id_length_delta": _rate(
                    sum(by_position[t]["deltas"]), len(by_position[t]["deltas"])
                ),
            }
            for t in INDICES
        ]
    return {
        "arms": arms,
        "ids": ids,
        "positions": positions,
        "headline": headline,
        "operating_characteristics": dict(OPERATING_CHARACTERISTICS),
        "n_cases": len(records),
        "n_incomplete": sum(
            1 for record in records if any(_cell_get(record, arm) is None for arm in ARM_KEYS)
        ),
    }


def case_material_sha256() -> str:
    """Digest of the frozen panel material."""
    material = {
        "topics": list(TOPICS),
        "words": list(WORDS),
        "neutral_statements": list(NEUTRAL_STATEMENTS),
        "arm_keys": list(ARM_KEYS),
        "marker_table": {arm: list(MARKER_TABLE[arm]) for arm in ARM_KEYS},
        "generator_spec": GENERATOR_SPEC,
        "case_count": CASE_COUNT,
        "indices": list(INDICES),
        "stride": STRIDE,
        "pool_digests": dict(POOL_DIGESTS),
        "min_d": MIN_D,
        "min_j": MIN_J,
        "max_incomplete": MAX_INCOMPLETE,
        "redraw_limit": REDRAW_LIMIT,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return digest_text(canonical)


def ground_sha256(seed: int) -> str:
    """Digest of the seed's generated ground rows."""
    rows = [
        [
            case["fixture_id"],
            case["topic"],
            case["position"],
            list(case["topics"]),
            list(case["values"]),
            list(case["neutral_indices"]),
        ]
        for case in generate_cases(seed)
    ]
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return digest_text(canonical)


def protocol_object(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The protocol block carried by every payload."""
    return {
        "version": PROTOCOL_VERSION,
        "arm_keys": list(ARM_KEYS),
        "treated_arms": list(TREATED_ARMS),
        "case_count": CASE_COUNT,
        "min_d": MIN_D,
        "min_j": MIN_J,
        "max_incomplete": MAX_INCOMPLETE,
        "case_material_sha256": case_material_sha256(),
        "ground_sha256": ground_sha256(seed),
        "pool_digests": dict(POOL_DIGESTS),
        "generation_budget": GENERATION_BUDGET,
        "max_ground_records": MAX_GROUND_RECORDS,
        "max_records_bytes": MAX_RECORDS_BYTES,
        "max_candidates": MAX_CANDIDATES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }


def protocol_fingerprint(seed: int, backend_fingerprint: str) -> str:
    """Digest of the protocol block (fingerprint field
    excluded, as in the siblings).
    """
    protocol = protocol_object(seed, backend_fingerprint)
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return digest_text(canonical)


def records_bytes(records: list[dict[str, Any]]) -> int:
    """The canonical records size (the budget measure)."""
    return len(json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))
