"""Compatibility-form substitution probe (cycle 231).

Design: docs/COMPATIBILITYFORM_DESIGN.md (the wave-12
round-3 candidate as adjudicated BUILD-as-amended,
amendments A1-A9 folded in). This is canonical-form's
missing third leg: the sibling instrument measures
canonical equivalence (NFC/NFD); this probe measures
compatibility equivalence -- the forms NFKC folds and
NFC/NFD keep distinct.

Anyone pasting typeset text (PDF extraction is full of
ligatures, enclosed digits, and vulgar fractions) into a
local model learns, per candidate (template family) and
per compatibility class, whether one compatibility-form
field in carrier text changes the model's answers at
all, measured with the tokenizer-regime gate making
tokenizer-side nulls legible as such. No direct LLM
prior exists for this measurement; a null result is a
legitimate pre-registered outcome (design section 7).

Frozen shape: a generated 24-case fact-recall panel
(stage-local, never the user's Split). Every case
carries a frozen marker sentence hosting exactly one
field per class, so every treated arm treats every case
at a uniform dose of exactly 1 frozen code point: the
arm replaces its own field's plain-text reading with the
frozen compatibility character, positionally, never by
search-replace. The population is 117 characters in six
classes derived from UnicodeData.txt 15.0.0 (SHA-256
pinned below): ligatures 7, enclosed 10, fractions 19,
superscripts 10, roman 16, katakana 55. Readings equal
raw NFKC for 98 members; the 19 fraction readings use
the plain-text solidus in place of U+2044 FRACTION SLASH
(the disclosed transcription rule of design section 2).

Headline (design section 6): per (candidate, arm), over
the cases completed in both control and arm AND
id-divergent for that arm, D = control passes, J = of
those the arm cells decomposed ``wrong_value``. Labels
are the sibling set only, guarded by the completion
rule: the incomplete count k ranges over exactly the
id-divergent domain (A1: the cap bounds k at 6 within
the domain; paired >= 18 is guaranteed only at the full
domain M = 24, and every headline prints its M).
``echo_compat`` / ``form_only`` decompose
compatibility-rendered correct values so they never
count toward J; the control arm's echo table is the
empty set (A3).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any

__all__ = [
    "ARM_KEYS",
    "CASE_COUNT",
    "CELL_CLASSES",
    "CELL_DETAILS",
    "CLASS_CHARS",
    "CLASS_TABLES",
    "ENCLOSED",
    "FRACTIONS",
    "GENERATION_BUDGET",
    "GENERATOR_SPEC",
    "GROUP",
    "KATAKANA",
    "LABELS",
    "LIGATURES",
    "MARKER_TEMPLATE",
    "MAX_CANDIDATES",
    "MAX_GROUND_RECORDS",
    "MAX_INCOMPLETE",
    "MAX_RECORDS_BYTES",
    "MIN_D",
    "MIN_J",
    "NEUTRAL_STATEMENTS",
    "PAIR_CLASSES",
    "POPULATION_AUDIT",
    "PROTOCOL_VERSION",
    "REDRAW_LIMIT",
    "ROMAN",
    "STAGE_DETAILS",
    "SUPERSCRIPTS",
    "TOPICS",
    "TREATED_ARMS",
    "UNICODEDATA_SHA256",
    "UNICODE_VERSION",
    "WORDS",
    "RedrawExhausted",
    "aggregate_candidate",
    "arm_content",
    "audit_population",
    "case_material_sha256",
    "classify_headline",
    "decompose_cell",
    "decompose_pair",
    "digest_ids",
    "digest_text",
    "fact_sentence",
    "generate_case",
    "generate_cases",
    "ground_sha256",
    "marker_sentence",
    "protocol_fingerprint",
    "protocol_object",
    "question_text",
    "records_bytes",
    "table_sha256",
    "validate_case",
]

PROTOCOL_VERSION = 1
GROUP = "compatibility_form"
CASE_COUNT = 24

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

# The frozen class tables (design section 2): (character,
# reading) pairs in codepoint order, which is the order
# the section-3 draws index. Readings equal raw NFKC for
# every member except the 19 fractions, whose readings
# replace U+2044 FRACTION SLASH with U+002F SOLIDUS (the
# disclosed transcription rule).
LIGATURES: tuple[tuple[str, str], ...] = (
    ("ĳ", "ij"),
    ("ﬀ", "ff"),
    ("ﬁ", "fi"),
    ("ﬂ", "fl"),
    ("ﬃ", "ffi"),
    ("ﬄ", "ffl"),
    ("ﬆ", "st"),
)
ENCLOSED: tuple[tuple[str, str], ...] = (
    ("①", "1"),
    ("②", "2"),
    ("③", "3"),
    ("④", "4"),
    ("⑤", "5"),
    ("⑥", "6"),
    ("⑦", "7"),
    ("⑧", "8"),
    ("⑨", "9"),
    ("⓪", "0"),
)
FRACTIONS: tuple[tuple[str, str], ...] = (
    ("¼", "1/4"),
    ("½", "1/2"),
    ("¾", "3/4"),
    ("⅐", "1/7"),
    ("⅑", "1/9"),
    ("⅒", "1/10"),
    ("⅓", "1/3"),
    ("⅔", "2/3"),
    ("⅕", "1/5"),
    ("⅖", "2/5"),
    ("⅗", "3/5"),
    ("⅘", "4/5"),
    ("⅙", "1/6"),
    ("⅚", "5/6"),
    ("⅛", "1/8"),
    ("⅜", "3/8"),
    ("⅝", "5/8"),
    ("⅞", "7/8"),
    ("↉", "0/3"),
)
SUPERSCRIPTS: tuple[tuple[str, str], ...] = (
    ("²", "2"),
    ("³", "3"),
    ("¹", "1"),
    ("⁰", "0"),
    ("⁴", "4"),
    ("⁵", "5"),
    ("⁶", "6"),
    ("⁷", "7"),
    ("⁸", "8"),
    ("⁹", "9"),
)
ROMAN: tuple[tuple[str, str], ...] = (
    ("Ⅰ", "I"),
    ("Ⅱ", "II"),
    ("Ⅲ", "III"),
    ("Ⅳ", "IV"),
    ("Ⅴ", "V"),
    ("Ⅵ", "VI"),
    ("Ⅶ", "VII"),
    ("Ⅷ", "VIII"),
    ("Ⅸ", "IX"),
    ("Ⅹ", "X"),
    ("Ⅺ", "XI"),
    ("Ⅻ", "XII"),
    ("Ⅼ", "L"),
    ("Ⅽ", "C"),
    ("Ⅾ", "D"),
    ("Ⅿ", "M"),
)
KATAKANA: tuple[tuple[str, str], ...] = (
    ("ｦ", "ヲ"),
    ("ｧ", "ァ"),
    ("ｨ", "ィ"),
    ("ｩ", "ゥ"),
    ("ｪ", "ェ"),
    ("ｫ", "ォ"),
    ("ｬ", "ャ"),
    ("ｭ", "ュ"),
    ("ｮ", "ョ"),
    ("ｯ", "ッ"),
    ("ｱ", "ア"),
    ("ｲ", "イ"),
    ("ｳ", "ウ"),
    ("ｴ", "エ"),
    ("ｵ", "オ"),
    ("ｶ", "カ"),
    ("ｷ", "キ"),
    ("ｸ", "ク"),
    ("ｹ", "ケ"),
    ("ｺ", "コ"),
    ("ｻ", "サ"),
    ("ｼ", "シ"),
    ("ｽ", "ス"),
    ("ｾ", "セ"),
    ("ｿ", "ソ"),
    ("ﾀ", "タ"),
    ("ﾁ", "チ"),
    ("ﾂ", "ツ"),
    ("ﾃ", "テ"),
    ("ﾄ", "ト"),
    ("ﾅ", "ナ"),
    ("ﾆ", "ニ"),
    ("ﾇ", "ヌ"),
    ("ﾈ", "ネ"),
    ("ﾉ", "ノ"),
    ("ﾊ", "ハ"),
    ("ﾋ", "ヒ"),
    ("ﾌ", "フ"),
    ("ﾍ", "ヘ"),
    ("ﾎ", "ホ"),
    ("ﾏ", "マ"),
    ("ﾐ", "ミ"),
    ("ﾑ", "ム"),
    ("ﾒ", "メ"),
    ("ﾓ", "モ"),
    ("ﾔ", "ヤ"),
    ("ﾕ", "ユ"),
    ("ﾖ", "ヨ"),
    ("ﾗ", "ラ"),
    ("ﾘ", "リ"),
    ("ﾙ", "ル"),
    ("ﾚ", "レ"),
    ("ﾛ", "ロ"),
    ("ﾜ", "ワ"),
    ("ﾝ", "ン"),
)

# Class name -> frozen table, in marker-field order
# (field indices 0-5 of design section 3).
CLASS_TABLES: dict[str, tuple[tuple[str, str], ...]] = {
    "ligature": LIGATURES,
    "enclosed": ENCLOSED,
    "fraction": FRACTIONS,
    "superscript": SUPERSCRIPTS,
    "roman": ROMAN,
    "katakana": KATAKANA,
}
CLASS_ORDER = tuple(CLASS_TABLES)
# Class name -> {character: reading} and {codepoint: reading}.
_READING_BY_CHAR: dict[str, dict[str, str]] = {
    name: dict(table) for name, table in CLASS_TABLES.items()
}
_READING_BY_CP: dict[str, dict[int, str]] = {
    name: {ord(char): reading for char, reading in table} for name, table in CLASS_TABLES.items()
}
_CHAR_BY_CP: dict[str, dict[int, str]] = {
    name: {ord(char): char for char, _reading in table} for name, table in CLASS_TABLES.items()
}

ARM_KEYS = ("control", *CLASS_ORDER)
TREATED_ARMS = ARM_KEYS[1:]

# The decomposition's class-4 character sets (design
# section 5): the arm's own frozen class table. The
# control arm's table is the EMPTY set (amendment A3):
# echo_compat never fires for control cells, and a folded
# control answer classifies form_only.
CLASS_CHARS: dict[str, frozenset[str]] = {
    arm: frozenset(char for char, _reading in CLASS_TABLES[arm])
    if arm != "control"
    else frozenset()
    for arm in ARM_KEYS
}

UNICODE_VERSION = "15.0.0"
UNICODEDATA_SHA256 = "806e9aed65037197f1ec85e12be6e8cd870fc5608b4de0fffd990f689f376a73"

MARKER_TEMPLATE = (
    "The marker line reads script {f0}, circle {f1}, share {f2}, "
    "power {f3}, regnal Henry {f4}, kana {f5}."
)

MIN_D = 10
MIN_J = 6
MAX_INCOMPLETE = 6
REDRAW_LIMIT = 1000
GENERATOR_SPEC = (
    "draw(seed, index, slot, counter) = int.from_bytes("
    "sha256(f'{seed}|compatfold|{index}|{slot}|{counter}')"
    ".digest()[:8], 'big'); slot 0: word = WORDS[draw % len(WORDS)]; "
    "slot 1: number = 1000 + draw % 9000; a value colliding with an "
    "earlier case redraws slots 0 and 1 only at counter+1 (slots 2-9 "
    "stay at counter 0), the counter incrementing once per colliding "
    "redraw and giving up when it exceeds 1000 (the stage fails "
    "closed); slots 2/3: neutral indices = draw % "
    "len(NEUTRAL_STATEMENTS), the second advancing +1 on collision; "
    "slot 4: ligature; slot 5: enclosed; slot 6: fraction; slot 7: "
    "superscript; slot 8: roman; slot 9: katakana, each draw % "
    "len(class table) over the codepoint-sorted frozen tables"
)
MAX_CANDIDATES = 24
MAX_GROUND_RECORDS = MAX_CANDIDATES * CASE_COUNT
GENERATION_BUDGET = len(ARM_KEYS) * MAX_GROUND_RECORDS
MAX_RECORDS_BYTES = 12 * 1024 * 1024

CELL_DETAILS = ("arm_context_exceeded", "arm_evaluation_failed")
STAGE_DETAILS = (
    "cancelled",
    "compatibility_form_record_budget_exceeded",
    "compatibility_form_screen_failed",
)
CELL_CLASSES = (
    "pass",
    "boundary_only",
    "wrong_value",
    "echo_compat",
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
# the A1 M-conditioned rows and the exact paired values).
OPERATING_CHARACTERISTICS: dict[str, Any] = {
    "fire_competent_independent": 0.0173,
    "fire_competent_paired": 0.0044,
    "fire_weak_independent": 0.1743,
    "fire_random_independent": 0.5503,
    "power_drop10_base90": 0.2542,
    "power_drop25_base90": 0.8159,
    "power_drop25_base85": 0.8761,
    "power_drop25_paired": 0.6981,
    "power_drop50_base90": 0.999,
    "sensitivity18_drop25": 0.5214,
    "sensitivity18_competent": 0.0038,
    "panel_sweep_16_drop25": 0.3904,
    "panel_sweep_20_drop25": 0.6395,
    "transplant_fire_competent_independent": 0.8897,
    "familywise_competent_expected_144": 2.4875,
    "familywise_competent_any_144": 0.9187,
    "familywise_paired_expected_144": 0.6397,
    "familywise_paired_any_144": 0.4733,
    "familywise_random_expected_144": 79.2441,
    "world_a_guarded_fire": 0.0037,
    "world_a_unguarded_fire": 0.0069,
    "world_a_guarded_insufficient": 0.9765,
    "world_b_unguarded_below_threshold": 0.7231,
    "world_b_guarded_below_threshold": 0.0763,
    "world_b_guarded_fire": 0.1046,
    "world_c_guarded_fire": 0.7719,
    "world_c_unguarded_fire": 0.7824,
    "m20_drop25": 0.6395,
    "m20_drop10": 0.1356,
    "m20_competent": 0.0068,
    "m18_drop25": 0.5214,
    "m18_drop10": 0.0889,
    "m18_competent": 0.0038,
    "m16_drop25": 0.3904,
    "m16_drop10": 0.0527,
    "m16_competent": 0.0019,
    "m12_drop25": 0.1379,
    "m12_drop10": 0.0113,
    "m12_competent": 0.0003,
    "m10_drop25": 0.0331,
    "m10_drop10": 0.0022,
    "m10_competent": 0.0001,
}

_VALUE_RE = re.compile(r"^[a-z]+-[0-9]{4}$")
_ARITHMETIC_RE = re.compile(r"\b(plus|times|sum|total|add|subtract|multiply|divide)\b|\+|\*")
_ROMAN_READINGS = frozenset(reading for _char, reading in ROMAN)
_ROMAN_CHARS = frozenset(char for char, _reading in ROMAN)


class RedrawExhausted(Exception):
    """The value-redraw give-up bound was reached.

    Counting pinned by amendment A2: the counter
    increments once per colliding redraw; exhaustion is
    raised when the counter exceeds REDRAW_LIMIT. The
    stage maps this to the fail-closed detail
    ``compatibility_form_screen_failed``.
    """


def digest_text(text: str) -> str:
    """SHA-256 hex digest of a text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_ids(ids: Any) -> str:
    """SHA-256 hex digest of a render id sequence."""
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode("ascii")).hexdigest()


def _derive(seed: int, index: int, slot: int, counter: int) -> int:
    """The frozen SHA-256 draw (design section 3)."""
    material = f"{seed}|compatfold|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def generate_cases(seed: int) -> list[dict[str, Any]]:
    """The frozen 24-case panel for ``seed``.

    Values are pairwise distinct across the panel: a
    collision redraws slots 0+1 only at counter+1 (slots
    2-9 stay at counter 0); the counter increments once
    per colliding redraw and exceeding REDRAW_LIMIT
    raises RedrawExhausted (amendment A2's counting).
    """
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(CASE_COUNT):
        counter = 0
        while True:
            word = WORDS[_derive(seed, index, 0, counter) % len(WORDS)]
            number = 1000 + _derive(seed, index, 1, counter) % 9000
            value = f"{word}-{number}"
            if value not in seen:
                break
            counter += 1
            if counter > REDRAW_LIMIT:
                raise RedrawExhausted
        seen.add(value)
        cases.append(generate_case(seed, index, value))
    return cases


def generate_case(seed: int, index: int, value: str | None = None) -> dict[str, Any]:
    """One panel case: case ``index`` under ``seed``.

    ``value`` is the (possibly redrawn) slot-0/1 value;
    when omitted it is read at counter 0. Slots 2-9 are
    always read at counter 0.
    """
    if value is None:
        word = WORDS[_derive(seed, index, 0, 0) % len(WORDS)]
        number = 1000 + _derive(seed, index, 1, 0) % 9000
        value = f"{word}-{number}"
    first = _derive(seed, index, 2, 0) % len(NEUTRAL_STATEMENTS)
    second = _derive(seed, index, 3, 0) % len(NEUTRAL_STATEMENTS)
    if second == first:
        second = (first + 1) % len(NEUTRAL_STATEMENTS)
    members: dict[str, int] = {}
    for offset, name in enumerate(CLASS_ORDER):
        table = CLASS_TABLES[name]
        drawn = table[_derive(seed, index, 4 + offset, 0) % len(table)]
        members[name] = ord(drawn[0])
    return {
        "fixture_id": f"compatibilityform-{index:02d}",
        "topic": TOPICS[index],
        "value": value,
        "expected": value,
        "neutral_indices": [first, second],
        "members": members,
    }


def fact_sentence(case: dict[str, Any]) -> str:
    """The frozen fact template."""
    return f"The {case['topic']} inspection code is {case['value']}."


def question_text(case: dict[str, Any]) -> str:
    """The frozen question template."""
    return f"What is the {case['topic']} inspection code?"


def marker_fields(case: dict[str, Any], arm: str) -> list[str]:
    """The six marker fields for ``arm``, positionally.

    Every field is its reading; the treated arm's own
    field is its frozen character instead (the
    superscript field is ``x`` + form in both shapes).
    """
    members = case["members"]
    fields: list[str] = []
    for name in CLASS_ORDER:
        codepoint = int(members[name])
        reading = _READING_BY_CP[name][codepoint]
        if arm == name:
            form = _CHAR_BY_CP[name][codepoint]
            fields.append(f"x{form}" if name == "superscript" else form)
        else:
            fields.append(f"x{reading}" if name == "superscript" else reading)
    return fields


def marker_sentence(case: dict[str, Any], arm: str = "control") -> str:
    """The frozen marker sentence for ``arm``."""
    fields = marker_fields(case, arm)
    named = {f"f{position}": field for position, field in enumerate(fields)}
    return MARKER_TEMPLATE.format(**named)


def arm_content(case: dict[str, Any], arm: str) -> str:
    """The full prompt content of one case under ``arm``."""
    if arm not in ARM_KEYS:
        raise ValueError(f"unknown compatibility-form arm: {arm!r}")
    first, second = (int(i) for i in case["neutral_indices"])
    return (
        f"{NEUTRAL_STATEMENTS[first]} {fact_sentence(case)} "
        f"{marker_sentence(case, arm)} {NEUTRAL_STATEMENTS[second]} "
        f"{question_text(case)}"
    )


def validate_case(case: dict[str, Any]) -> bool:
    """The full generator battery for one case (design
    sections 2-3): frozen shapes, pure-ASCII expected,
    the katakana confinement, the arithmetic screen with
    its pinned operator set (A7), the Roman enumeration
    screens, the per-arm dose and byte-divergence
    invariants. Never raises: any violation is False.
    """
    try:
        return _validate_case(case)
    except Exception:  # noqa: BLE001
        return False


def _validate_case(case: dict[str, Any]) -> bool:
    """Validate case (worker for validate_case)."""
    fixture_id = case.get("fixture_id")
    if type(fixture_id) is not str or not fixture_id.startswith("compatibilityform-"):
        return False
    index = int(fixture_id.rsplit("-", 1)[1])
    if not 0 <= index < CASE_COUNT or fixture_id != f"compatibilityform-{index:02d}":
        return False
    topic = case.get("topic")
    if topic != TOPICS[index]:
        return False
    value = case.get("value")
    expected = case.get("expected")
    if type(value) is not str or expected != value:
        return False
    if not expected.isascii() or not _VALUE_RE.fullmatch(expected):
        return False
    word, _, number = expected.partition("-")
    if word not in WORDS or not 1000 <= int(number) <= 9999:
        return False
    neutral_indices = case.get("neutral_indices")
    if (
        type(neutral_indices) is not list
        or len(neutral_indices) != 2
        or any(type(i) is not int or not 0 <= i < len(NEUTRAL_STATEMENTS) for i in neutral_indices)
        or neutral_indices[0] == neutral_indices[1]
    ):
        return False
    members = case.get("members")
    if type(members) is not dict or set(members) != set(CLASS_ORDER):
        return False
    for name in CLASS_ORDER:
        codepoint = members[name]
        if type(codepoint) is not int or codepoint not in _READING_BY_CP[name]:
            return False
    fact = fact_sentence(case)
    question = question_text(case)
    if not fact.isascii() or not question.isascii():
        return False
    control = arm_content(case, "control")
    kana_reading = _READING_BY_CP["katakana"][int(members["katakana"])]
    if {char for char in control if not char.isascii()} != {kana_reading}:
        return False
    control_fields = marker_fields(case, "control")
    contents: dict[str, str] = {}
    for arm in ARM_KEYS:
        content = arm_content(case, arm)
        contents[arm] = content
        if "\n" in content:
            return False
        marker = marker_sentence(case, arm)
        region = f"{fact} {marker} {question}"
        if _ARITHMETIC_RE.search(region):
            return False
        # Roman enumeration screens (design section 3):
        # no sentence's first token is a Roman reading or
        # character, and the marker carries the literal
        # regnal prefix immediately before the Roman field.
        for sentence in content.split(". "):
            tokens = sentence.split()
            if tokens and (tokens[0] in _ROMAN_READINGS or tokens[0] in _ROMAN_CHARS):
                return False
        fields = marker_fields(case, arm)
        if f"regnal Henry {fields[4]}" not in marker:
            return False
        if arm == "control":
            continue
        # Dose: exactly this arm's field differs from the
        # control fields, and the content differs from the
        # control content at exactly that field.
        differing = [
            position
            for position, (mine, base) in enumerate(zip(fields, control_fields, strict=True))
            if mine != base
        ]
        if differing != [CLASS_ORDER.index(arm)]:
            return False
        if content == control:
            return False
    return len(set(contents.values())) == len(ARM_KEYS)


def decompose_cell(cell: dict[str, Any], expected: str, expected_digest: str, arm: str) -> str:
    """The ordered decision list of design section 5.

    Exactly one class per completed cell; class 4 tests
    the answer against the arm's own frozen class table
    (the control arm's table is the empty set, A3).
    """
    if cell.get("verdict") is True:
        return "pass"
    digest = cell.get("answer_digest")
    if digest is not None and digest == expected_digest:
        return "boundary_only"
    text = cell.get("answer_text")
    if type(text) is not str:
        return "non_answer"
    if text != expected and _VALUE_RE.fullmatch(text):
        word, _, number = text.partition("-")
        if word in WORDS and 1000 <= int(number) <= 9999:
            return "wrong_value"
    if text != expected and unicodedata.normalize("NFKC", text) == expected:
        if any(char in CLASS_CHARS.get(arm, frozenset()) for char in text):
            return "echo_compat"
        return "form_only"
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


def _member_satisfies(name: str, char: str, reading: str) -> bool:
    """One frozen member against the pinned clauses,
    executed against the local unicodedata (15.0.0, the
    pinned file's own version -- design section 2).
    """
    decomposition = unicodedata.decomposition(char)
    parts = decomposition.split()
    if not parts or not parts[0].startswith("<") or parts[0] == "<wide>":
        return False
    if unicodedata.normalize("NFC", char) != char or unicodedata.normalize("NFD", char) != char:
        return False
    folded = unicodedata.normalize("NFKC", char)
    if folded == char:
        return False
    member_name = unicodedata.name(char, "")
    if name == "ligature":
        targets = "".join(chr(int(token, 16)) for token in parts[1:])
        return (
            member_name.startswith("LATIN SMALL LIGATURE")
            and targets.isascii()
            and reading == folded == targets
        )
    if name == "enclosed":
        return member_name.startswith("CIRCLED DIGIT") and reading == folded
    if name == "fraction":
        return member_name.startswith("VULGAR FRACTION") and reading == folded.replace(
            "\u2044", "/"
        )
    if name == "superscript":
        return (
            parts[0] == "<super>"
            and len(parts) == 2
            and reading == folded
            and reading in "0123456789"
        )
    if name == "roman":
        return member_name.startswith("ROMAN NUMERAL") and reading == folded
    if name == "katakana":
        return member_name.startswith("HALFWIDTH KATAKANA LETTER") and reading == folded
    return False


def audit_population() -> dict[str, Any]:
    """Execute the section-2 clauses over the frozen
    tables: every member satisfies its class clause and
    the general eligibility rule under the local
    unicodedata (15.0.0); classes are pairwise disjoint
    and codepoint-sorted; no reading contains the value
    grammar's separator.
    """
    classes: dict[str, Any] = {}
    seen: set[str] = set()
    ok = True
    for name in CLASS_ORDER:
        table = CLASS_TABLES[name]
        codepoints = [ord(char) for char, _reading in table]
        members_ok = all(_member_satisfies(name, char, reading) for char, reading in table)
        disjoint = not any(char in seen for char, _reading in table)
        seen.update(char for char, _reading in table)
        entry_ok = (
            members_ok
            and disjoint
            and codepoints == sorted(codepoints)
            and all("-" not in reading for _char, reading in table)
        )
        if name != "katakana":
            entry_ok = entry_ok and all(reading.isascii() for _char, reading in table)
        else:
            entry_ok = entry_ok and all(
                len(reading) == 1 and not reading.isascii() for _char, reading in table
            )
        ok = ok and entry_ok
        classes[name] = {
            "count": len(table),
            "members": [f"U+{codepoint:04X}" for codepoint in codepoints],
            "satisfies": entry_ok,
        }
    return {
        "classes": classes,
        "total": sum(len(CLASS_TABLES[name]) for name in CLASS_ORDER),
        "pairwise_disjoint": len(seen) == sum(len(CLASS_TABLES[name]) for name in CLASS_ORDER),
        "ok": ok,
        "unicode_version": unicodedata.unidata_version,
        "unicodedata_version": UNICODE_VERSION,
        "unicodedata_sha256": UNICODEDATA_SHA256,
    }


POPULATION_AUDIT = audit_population()


def table_sha256() -> str:
    """Digest of the frozen tables and their pins."""
    material = {
        "classes": {
            name: [[char, reading] for char, reading in CLASS_TABLES[name]] for name in CLASS_ORDER
        },
        "unicode_version": UNICODE_VERSION,
        "unicodedata_sha256": UNICODEDATA_SHA256,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return digest_text(canonical)


def case_material_sha256() -> str:
    """Digest of the frozen panel material."""
    material = {
        "topics": list(TOPICS),
        "words": list(WORDS),
        "neutral_statements": list(NEUTRAL_STATEMENTS),
        "arm_keys": list(ARM_KEYS),
        "marker_template": MARKER_TEMPLATE,
        "generator_spec": GENERATOR_SPEC,
        "case_count": CASE_COUNT,
        "table_sha256": table_sha256(),
        "min_d": MIN_D,
        "min_j": MIN_J,
        "max_incomplete": MAX_INCOMPLETE,
        "redraw_limit": REDRAW_LIMIT,
        "population_audit": POPULATION_AUDIT,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return digest_text(canonical)


def ground_sha256(seed: int) -> str:
    """Digest of the seed's generated ground rows."""
    rows = [
        [
            case["fixture_id"],
            case["topic"],
            case["value"],
            list(case["neutral_indices"]),
            [int(case["members"][name]) for name in CLASS_ORDER],
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
        "table_sha256": table_sha256(),
        "unicode_version": UNICODE_VERSION,
        "unicodedata_sha256": UNICODEDATA_SHA256,
        "population_audit": POPULATION_AUDIT,
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


def _cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the compatibilityform cell get."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _detail_get(record: dict[str, Any], key: str) -> str:
    """Return the compatibilityform detail get."""
    details = record.get("cell_details")
    if not isinstance(details, dict):
        return ""
    detail = details.get(key, "")
    return detail if type(detail) is str else ""


def _verdict(record: dict[str, Any], key: str) -> bool:
    """Return the compatibilityform verdict."""
    cell = _cell_get(record, key)
    return bool(cell is not None and cell.get("verdict") is True)


def _ids_identical(record: dict[str, Any], arm: str) -> bool:
    """Derived from the recorded id digests alone."""
    digests = record.get("id_digests")
    if not isinstance(digests, dict):
        return False
    control = digests.get("control")
    return type(control) is str and control != "" and control == digests.get(arm)


def _rate(numerator: int, denominator: int) -> float | None:
    """Return the compatibilityform rate."""
    return (numerator / denominator) if denominator else None


def aggregate_candidate(
    records: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Per-arm rates and decompositions, the id report,
    the per-arm dose deltas, and the guarded headlines
    (with each arm's divergent-domain size M, A1) for one
    candidate's ground records.
    """
    expected_by_fixture = {str(case["fixture_id"]): str(case["expected"]) for case in cases}
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
            decomposition[
                decompose_cell(
                    cell,
                    expected_by_fixture.get(str(record.get("fixture", "")), ""),
                    str(record.get("expected_digest", "")),
                    key,
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
            baseline = tokens.get("control")
            if type(treated) is int and type(baseline) is int:
                deltas.append(treated - baseline)
            else:
                deltas.append(0)
            if _ids_identical(record, arm):
                n_ids_identical += 1
                control_cell = _cell_get(record, "control")
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
    dose: dict[str, Any] = {}
    for arm in TREATED_ARMS:
        codepoint_deltas: list[int] = []
        byte_deltas: list[int] = []
        for case in cases:
            treated = arm_content(case, arm)
            control = arm_content(case, "control")
            codepoint_deltas.append(len(treated) - len(control))
            byte_deltas.append(len(treated.encode("utf-8")) - len(control.encode("utf-8")))
        dose[arm] = {
            "codepoint_delta_min": min(codepoint_deltas, default=0),
            "codepoint_delta_max": max(codepoint_deltas, default=0),
            "byte_delta_min": min(byte_deltas, default=0),
            "byte_delta_max": max(byte_deltas, default=0),
        }
    headline: dict[str, Any] = {}
    for arm in TREATED_ARMS:
        domain = [record for record in records if not _ids_identical(record, arm)]
        paired = [
            record
            for record in domain
            if _cell_get(record, "control") is not None and _cell_get(record, arm) is not None
        ]
        n_incomplete = len(domain) - len(paired)
        control_passes = 0
        wrong_values = 0
        reverse = 0
        non_answer = 0
        pair_decomposition = dict.fromkeys(PAIR_CLASSES, 0)
        failing_classes = dict.fromkeys(CELL_CLASSES, 0)
        for record in paired:
            control_cell = _cell_get(record, "control")
            arm_cell = _cell_get(record, arm)
            if control_cell is None or arm_cell is None:
                continue
            expected = expected_by_fixture.get(str(record.get("fixture", "")), "")
            expected_digest = str(record.get("expected_digest", ""))
            arm_class = decompose_cell(arm_cell, expected, expected_digest, arm)
            if arm_class == "non_answer":
                non_answer += 1
            control_verdict = control_cell.get("verdict") is True
            arm_verdict = arm_cell.get("verdict") is True
            if control_verdict:
                control_passes += 1
                if arm_class == "wrong_value":
                    wrong_values += 1
            elif arm_verdict:
                reverse += 1
            if control_verdict != arm_verdict:
                pair_decomposition[decompose_pair(control_cell, arm_cell)] += 1
                failing = arm_cell if control_verdict else control_cell
                failing_arm = arm if control_verdict else "control"
                failing_class = decompose_cell(failing, expected, expected_digest, failing_arm)
                failing_classes[failing_class] += 1
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
    return {
        "arms": arms,
        "ids": ids,
        "dose": dose,
        "headline": headline,
        "operating_characteristics": dict(OPERATING_CHARACTERISTICS),
        "n_cases": len(records),
        "n_incomplete": sum(
            1 for record in records if any(_cell_get(record, arm) is None for arm in ARM_KEYS)
        ),
    }
