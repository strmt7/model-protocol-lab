"""Typographic-emphasis probe (cycle 237, research
wave-12): does wrapping one topic span of a fact-recall
carrier in a typographic-emphasis treatment -- bold,
italic, monospace, or capitals -- move a checkpoint's
answers to wrong values, and can a candidate earn a
level-controlled certificate that the treatment does
not?

Design: docs/TYPOGRAPHIC_EMPHASIS_DESIGN.md (the
falsification track's predicted-null instrument, built
as adjudicated with amendments A1-A6). Frozen shape: a
generated 24-case fact-recall panel per candidate;
every case is one frozen topic, one word-number
inspection code, two frozen neutral carrier sentences,
one fact sentence of the form ``The {span} inspection
code is {value}.`` and one question naming the topic.
The main panel renders five arms that differ only in
the treatment of the fact sentence's topic span
(verbatim control; ``**span**`` bold; ``*span*``
italic; `` `span` `` monospace; the span uppercased
for caps) -- the question's topic occurrence stays
verbatim in every main arm, so exactly one occurrence
is treated. A positive-control (PC) panel of the same
24 cases asks which topic is written in bold over two
facts (the main case's own value under its topic,
plain or bold, and a distractor fact), and a PC-B arm
renders the main case with the topic wrapped in
``***...***`` at both its occurrences. Scored with the
tree's own exact-match-plus-boundary scorer
(``core.evaluate``); surgery is at the fixture level,
so request seeds are identical across arms.

The readout per (candidate, treated arm) is the
sibling completion-guarded classification over the
token-id-divergent paired population (fire floors:
at least 10 divergent control passes, at least 6
in-grammar wrong-value discordants), extended by the
certification event E = {D >= 10 and H <= 5 and
S <= 0} over the same domain (H = control-pass /
arm-fail pairs over every failure class, R =
control-fail / arm-pass pairs, S = H - R): the label
``equivalence_certified`` issues only when E holds,
the divergent domain is the full M = 24 (a guarded
certificate at M implies the full-data event at size
24 under worst-case completion, by componentwise
monotonicity of (D, H, S); design section 6.4), E is
stable under that completion (H + k <= 5 and
S + k <= 0), and the candidate's positive control
passed in the same run. The level is stated over the
heterogeneous class (per-pair joint laws arbitrary;
the null is a pooled marginal drop of at least
delta = 0.34): the completed extremal bound is
0.013826 per comparison at alpha = 0.02, printed in
the operating characteristics with its full name
beside every figure it could be confused with.
Records carry answer digests plus the answer text the
decomposition classes are functions of, bound by the
digest; the diff-reports mirror recomputes every
label -- including a certificate -- from the
committed counts, so a certificate cannot be asserted
by a record whose counts do not earn it.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

__all__ = [
    "ALPHA",
    "ARM_KEYS",
    "CASE_COUNT",
    "CELL_CLASSES",
    "CELL_DETAILS",
    "DELTA",
    "GENERATION_BUDGET",
    "GENERATOR_SPEC",
    "GROUP",
    "LABELS",
    "MAX_CANDIDATES",
    "MAX_GROUND_RECORDS",
    "MAX_INCOMPLETE",
    "MAX_RECORDS_BYTES",
    "MIN_D",
    "MIN_J",
    "OPERATING_CHARACTERISTICS",
    "PAIR_CLASSES",
    "PCB_ARM",
    "PC_ARM_KEYS",
    "PC_MIN_INFO_PASSES",
    "PC_MIN_MARGIN",
    "PC_QUESTION",
    "POOL_DIGESTS",
    "PROTOCOL_VERSION",
    "RECORD_ARMS",
    "STAGE_DETAILS",
    "TREATED_ARMS",
    "RedrawExhausted",
    "aggregate_candidate",
    "arm_content",
    "case_material_sha256",
    "certification_blockers",
    "classify_headline",
    "content_for",
    "decompose_cell",
    "decompose_pair",
    "digest_ids",
    "digest_text",
    "echo_set",
    "fact_sentence",
    "generate_case",
    "generate_cases",
    "ground_sha256",
    "pc_content",
    "pc_fact_sentences",
    "pc_gate_passed",
    "pcb_content",
    "protocol_fingerprint",
    "protocol_object",
    "question_text",
    "records_bytes",
    "strip_marks",
    "validate_case",
    "wrap_span",
]

PROTOCOL_VERSION = 1
GROUP = "typographic_emphasis"
CASE_COUNT = 24

# The frozen pools, carried verbatim from the sibling
# instruments (the topics of invisiblechars /
# confusableskeleton / compatibilityform, the words of
# canonicalform, the neutral carriers of density). The
# module carries its own copies; the pool digests pin
# the contents (data and digests, never a host or
# library version).
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

ARM_KEYS = ("control", "bold", "italic", "monospace", "caps")
TREATED_ARMS = ARM_KEYS[1:]
PC_ARM_KEYS = ("pc_plain", "pc_info")
PCB_ARM = "pc_b"
RECORD_ARMS = ARM_KEYS + PC_ARM_KEYS + (PCB_ARM,)
PC_QUESTION = "What is the inspection code of the topic written in bold?"

MIN_D = 10
MIN_J = 6
MAX_INCOMPLETE = 6
REDRAW_LIMIT = 1000

# The certification licence (design sections 6.2 and
# 7, as amended A1): the discriminability bound delta
# and the per-comparison level alpha. The level is
# stated over the heterogeneous class with arbitrary
# per-pair dependence; the completed extremal bound at
# delta is OPERATING_CHARACTERISTICS'
# certification_bound_completed_per_comparison.
DELTA = 0.34
ALPHA = 0.02

# Positive-control floors (design section 6.3): the
# info arm must pass at least 10 completed pairs and
# beat the plain arm's reverse discordants by a margin
# of at least 8, stable under the worst-case completion
# of the panel's incomplete pairs (the A5 guard).
PC_MIN_INFO_PASSES = 10
PC_MIN_MARGIN = 8

GENERATOR_SPEC = (
    "per case index i in 0..23 with run seed s: topic = TOPICS[i]; "
    "value drawn at slots 0 (word) and 1 (number) of the frozen draw "
    "int.from_bytes(sha256(f'{s}|typographicemphasis|{i}|{slot}|{counter}')"
    ".digest()[:8], 'big'), word = WORDS[draw % 24], number = 1000 + "
    "draw % 9000, redrawn at counter+1 while the value collides with an "
    "earlier case's value (give up past 1000: the stage fails closed); "
    "neutrals at slots 2 and 3 (second index +1 mod 24 on collision); "
    "positive-control distractor topic at slot 4 (draw % 24, +1 mod 24 "
    "when equal to i), distractor value at slots 5 and 6 (redrawn at "
    "counter+1 while equal to the case's own value, give up past 1000), "
    "position bit at slot 7 (0 = the target fact is printed first)"
)

MAX_CANDIDATES = 24
MAX_GROUND_RECORDS = MAX_CANDIDATES * CASE_COUNT
# 24 cases x 8 arms = 192 generations per candidate;
# 4,608 at the legal maximum of 24 candidates.
GENERATION_BUDGET = MAX_CANDIDATES * CASE_COUNT * len(RECORD_ARMS)
MAX_RECORDS_BYTES = 12 * 1024 * 1024

CELL_DETAILS = ("", "arm_context_exceeded", "arm_evaluation_failed")
STAGE_DETAILS = (
    "",
    "cancelled",
    "typographic_emphasis_record_budget_exceeded",
    "typographic_emphasis_screen_failed",
)
CELL_CLASSES = (
    "pass",
    "boundary_only",
    "wrong_value",
    "echo_bold",
    "echo_italic",
    "echo_monospace",
    "echo_caps",
    "form_only",
    "non_answer",
)
PAIR_CLASSES = ("boundary_only", "exact_only", "mixed")
LABELS = (
    "insufficient",
    "no_discordance_observed",
    "discordant_asymmetric",
    "below_threshold",
    "equivalence_certified",
)
CERTIFICATION_GATES = (
    "statistic_event",
    "domain_size",
    "completion_guard",
    "positive_control",
)

_VALUE_RE = re.compile(r"[a-z]+-[0-9]{4}")
_WORD_SET = frozenset(WORDS)
_TOPIC_SET = frozenset(TOPICS)

# The printed operating characteristics (design
# section 6; every figure reproduced from its own code
# by the cycle-237 figure tests). Names are full: three
# distinct quantities in this design print near 0.0122
# (the anchored certification probability at the old
# delta 0.28, the homogeneous uniform bound at delta
# 0.30, and the restricted-vertex heterogeneous bound
# at delta 0.34), and a bare 0.0122 anywhere is a
# defect.
OPERATING_CHARACTERISTICS: dict[str, Any] = {
    "delta": DELTA,
    "alpha": ALPHA,
    # The certification licence: the completed extremal
    # bound over the heterogeneous class at delta (the
    # figure section 7 and VALIDATION print).
    "certification_bound_completed_per_comparison": 0.013826,
    # The restricted-vertex bound at the same delta
    # (the adjudication's vertex curve endpoint; not
    # the licence).
    "certification_bound_restricted_vertex_delta_034": 0.012253,
    # Heterogeneous completed-class rows at smaller
    # domain sizes: M = 23 is the unique partial domain
    # the level supports; M = 22 fails it.
    "certification_bound_completed_m23": 0.017934,
    "certification_bound_completed_m22": 0.023163,
    # Family-wise price of the certification licence
    # over the 96 labelled comparisons (4 treated arms
    # x 24 candidates), independence convention.
    "familywise_certification_expected_96": 1.3273,
    "familywise_certification_any_96": 0.7373,
    # The competent panel certifies at this rate under
    # the anchored coupling at delta with control rate
    # 0.90 (the anchored curve's endpoint).
    "certify_anchored_delta_034_pc_090": 0.0034,
    # Context only, never the licence: the homogeneous
    # uniform bound at delta 0.34.
    "certification_bound_uniform_homogeneous_delta_034": 0.005414,
    # The real-table bound's counterexample world: a
    # per-pair dependence structure whose certification
    # probability at delta exceeds alpha, which is why
    # the licence is stated over the completed class.
    "rtb_counterexample_certification_probability": 0.022662,
    # The sibling fire rule's exact grid (independent
    # enumeration; the paired column couples outcomes
    # within a case at correlation 0.5).
    "fire_competent_independent": 0.017274,
    "fire_competent_paired": 0.004442,
    "fire_weak_independent": 0.1743,
    "fire_random_independent": 0.550306,
    "power_drop10_base90": 0.2542,
    "power_drop25_base90": 0.8159,
    "power_drop25_base85": 0.8761,
    "power_drop25_paired": 0.698078,
    "power_drop50_base90": 0.9990,
    "certify_paired_rho_05_delta_000": 0.6139,
    "certify_paired_rho_05_delta_028": 0.0044,
    "familywise_fire_expected_96": 1.6583,
    "familywise_fire_any_96": 0.8123,
    # The anchored certification probability at the
    # true null (control rate 0.90): the null yield --
    # a competent candidate certifies, it does not
    # merely fail to fire.
    "certify_anchored_delta_000_pc_090": 0.5974,
    # Positive control: the deaf-guesser probability of
    # passing the PC gate, at the independence point of
    # the discordant-cell parameter x and along its
    # dependence range (x is the probability of each
    # discordant cell, B and H alike; both-pass and
    # both-fail carry 0.5 - x each).
    "pc_gate_deaf_x_025_independence": 0.0147,
    "pc_gate_deaf_x_030": 0.0235,
    "pc_gate_deaf_x_035": 0.0332,
    "pc_gate_deaf_x_050": 0.0758,
    # The executed guard worlds (10,000 trials each at
    # the designer's seed): world A (zero-effect
    # manufacture), world B at delta and on the PASTA
    # profile, world C (true drop, fire side).
    "world_a_guarded_certification": 0.0016,
    "world_a_guarded_fire": 0.0047,
    "world_b_delta_unguarded_certification": 0.2709,
    "world_b_delta_guarded_certification": 0.0127,
    "world_b_pasta_guarded_certification": 0.0006,
    "world_c_guarded_fire": 0.6318,
}


class RedrawExhausted(Exception):
    """A frozen draw needed more redraws than the
    frozen limit allows; the stage fails closed.
    """


def digest_text(text: str) -> str:
    """SHA-256 hex digest of a text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_ids(ids: tuple[int, ...]) -> str:
    """Digest ids."""
    return digest_text(",".join(str(int(i)) for i in ids))


def _pool_digest(pool: tuple[str, ...]) -> str:
    """Pool digest."""
    canonical = json.dumps(list(pool), separators=(",", ":"), ensure_ascii=True)
    return digest_text(canonical)


POOL_DIGESTS = {
    "topics": _pool_digest(TOPICS),
    "words": _pool_digest(WORDS),
    "neutral_statements": _pool_digest(NEUTRAL_STATEMENTS),
}


def _derive(seed: int, index: int, slot: int, counter: int) -> int:
    """The frozen draw: first 8 bytes of the SHA-256 of
    the namespaced material, big-endian.
    """
    material = f"{seed}|typographicemphasis|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")


def _draw_value(seed: int, index: int, seen: set[str]) -> str:
    """Draw one case's value at slots 0 and 1, redrawing
    both at counter+1 while the value collides with an
    earlier case's value; give up past the frozen limit.
    """
    counter = 0
    while True:
        word = WORDS[_derive(seed, index, 0, counter) % len(WORDS)]
        number = 1000 + _derive(seed, index, 1, counter) % 9000
        value = f"{word}-{number}"
        if value not in seen:
            seen.add(value)
            return value
        counter += 1
        if counter > REDRAW_LIMIT:
            raise RedrawExhausted


def _draw_distractor_value(seed: int, index: int, target_value: str) -> str:
    """Draw the positive-control distractor value at
    slots 5 and 6, redrawing both at counter+1 while it
    equals the case's own value; give up past the
    frozen limit.
    """
    counter = 0
    while True:
        word = WORDS[_derive(seed, index, 5, counter) % len(WORDS)]
        number = 1000 + _derive(seed, index, 6, counter) % 9000
        value = f"{word}-{number}"
        if value != target_value:
            return value
        counter += 1
        if counter > REDRAW_LIMIT:
            raise RedrawExhausted


def generate_case(seed: int, index: int) -> dict[str, Any]:
    """One frozen case, regenerated standalone: the
    panel's earlier values are replayed so the
    collision domain matches the full panel exactly.
    """
    seen: set[str] = set()
    for earlier in range(index):
        _draw_value(seed, earlier, seen)
    value = _draw_value(seed, index, seen)
    first = _derive(seed, index, 2, 0) % len(NEUTRAL_STATEMENTS)
    second = _derive(seed, index, 3, 0) % len(NEUTRAL_STATEMENTS)
    if second == first:
        second = (first + 1) % len(NEUTRAL_STATEMENTS)
    distractor_index = _derive(seed, index, 4, 0) % len(TOPICS)
    if distractor_index == index:
        distractor_index = (index + 1) % len(TOPICS)
    return {
        "fixture_id": f"typographicemphasis-{index:02d}",
        "index": index,
        "topic": TOPICS[index],
        "expected": value,
        "neutral_indices": [first, second],
        "distractor_topic": TOPICS[distractor_index],
        "distractor_value": _draw_distractor_value(seed, index, value),
        "target_first": _derive(seed, index, 7, 0) % 2 == 0,
    }


def generate_cases(seed: int) -> list[dict[str, Any]]:
    """The frozen 24-case panel for a run seed."""
    return [generate_case(seed, index) for index in range(CASE_COUNT)]


def wrap_span(span: str, arm: str) -> str:
    """The arm's treatment of one span: bold wraps in
    two asterisks a side, italic in one, monospace in
    one backtick a side, caps uppercases, control is
    verbatim.
    """
    if arm == "bold":
        return f"**{span}**"
    if arm == "italic":
        return f"*{span}*"
    if arm == "monospace":
        return f"`{span}`"
    if arm == "caps":
        return span.upper()
    return span


def fact_sentence(case: dict[str, Any], arm: str) -> str:
    """The fact sentence with the topic span treated
    by the arm (the value is never treated).
    """
    return f"The {wrap_span(str(case['topic']), arm)} inspection code is {case['expected']}."


def question_text(case: dict[str, Any]) -> str:
    """The main panel's question: the topic occurrence
    here stays verbatim in every main arm.
    """
    return f"What is the {case['topic']} inspection code?"


def arm_content(case: dict[str, Any], arm: str) -> str:
    """One main-panel arm's full content: neutral,
    treated fact, neutral, question -- the fact
    sentence is the only part that moves across arms.
    """
    first, second = (int(i) for i in case["neutral_indices"])
    return (
        f"{NEUTRAL_STATEMENTS[first]} {fact_sentence(case, arm)} "
        f"{NEUTRAL_STATEMENTS[second]} {question_text(case)}"
    )


def pc_fact_sentences(case: dict[str, Any], arm: str) -> tuple[str, str]:
    """The positive control's two fact sentences in
    printed order: the target fact (its topic wrapped
    in bold only in the pc_info arm; verbatim in
    pc_plain) and the distractor fact (always
    verbatim), ordered by the case's frozen position
    bit.
    """
    topic = str(case["topic"])
    wrapped = f"**{topic}**" if arm == "pc_info" else topic
    target = f"The {wrapped} inspection code is {case['expected']}."
    distractor = f"The {case['distractor_topic']} inspection code is {case['distractor_value']}."
    if case["target_first"]:
        return target, distractor
    return distractor, target


def pc_content(case: dict[str, Any], arm: str) -> str:
    """One positive-control arm's full content: the
    main case's own two neutrals, the two facts, and
    the frozen PC question (byte-identical across the
    two PC arms except the target topic's wrap).
    """
    first, second = (int(i) for i in case["neutral_indices"])
    fact_a, fact_b = pc_fact_sentences(case, arm)
    return (
        f"{NEUTRAL_STATEMENTS[first]} {fact_a} {fact_b} {NEUTRAL_STATEMENTS[second]} {PC_QUESTION}"
    )


def pcb_content(case: dict[str, Any]) -> str:
    """The PC-B arm's content: the main case with the
    topic wrapped in ``***...***`` at both its
    occurrences (fact and question). Descriptive only.
    """
    first, second = (int(i) for i in case["neutral_indices"])
    topic = str(case["topic"])
    fact = f"The ***{topic}*** inspection code is {case['expected']}."
    question = f"What is the ***{topic}*** inspection code?"
    return f"{NEUTRAL_STATEMENTS[first]} {fact} {NEUTRAL_STATEMENTS[second]} {question}"


def content_for(case: dict[str, Any], arm: str) -> str:
    """Any record arm's full content for a case."""
    if arm in PC_ARM_KEYS:
        return pc_content(case, arm)
    if arm == PCB_ARM:
        return pcb_content(case)
    return arm_content(case, arm)


def _value_grammar_ok(text: Any) -> bool:
    """The frozen value grammar, fullmatch semantics
    only (a trailing newline or space is out of
    grammar): a pool word, a hyphen, and a number in
    [1000, 9999].
    """
    if type(text) is not str or _VALUE_RE.fullmatch(text) is None:
        return False
    word, _sep, number = text.partition("-")
    return word in _WORD_SET and 1000 <= int(number) <= 9999


def _valid_index(value: Any) -> bool:
    """Valid index."""
    return type(value) is int and not isinstance(value, bool)


def validate_case(case: Any) -> bool:
    """The section-4 invariant battery (design section
    4, build debt 2): exact key set and field shapes,
    the value under the frozen grammar, two distinct
    in-range neutrals, the distractor drawn off the
    case's own topic and value, and the generated-text
    invariants executed against the builders -- the
    expected value is ASCII; exactly one treated span
    occurrence per main arm (the fact's), the question
    and neutrals byte-identical across main arms; the
    PC plain arm carries no mark characters anywhere;
    the PC info arm's only marks are the target topic's
    wrap; the PC target value is the main case's value.
    Never raises.
    """
    if type(case) is not dict:
        return False
    if set(case) != {
        "fixture_id",
        "index",
        "topic",
        "expected",
        "neutral_indices",
        "distractor_topic",
        "distractor_value",
        "target_first",
    }:
        return False
    index = case["index"]
    if not _valid_index(index) or not 0 <= int(index) < CASE_COUNT:
        return False
    if case["fixture_id"] != f"typographicemphasis-{int(index):02d}":
        return False
    topic = case["topic"]
    if topic != TOPICS[int(index)]:
        return False
    expected = case["expected"]
    if not _value_grammar_ok(expected) or not str(expected).isascii():
        return False
    neutrals = case["neutral_indices"]
    if type(neutrals) is not list or len(neutrals) != 2:
        return False
    if any(not _valid_index(n) or not 0 <= int(n) < len(NEUTRAL_STATEMENTS) for n in neutrals):
        return False
    if neutrals[0] == neutrals[1]:
        return False
    distractor_topic = case["distractor_topic"]
    if distractor_topic not in _TOPIC_SET or distractor_topic == topic:
        return False
    if not _value_grammar_ok(case["distractor_value"]):
        return False
    if case["distractor_value"] == expected:
        return False
    if type(case["target_first"]) is not bool:
        return False
    try:
        contents = {arm: arm_content(case, arm) for arm in ARM_KEYS}
        control = contents["control"]
        if control.count(str(topic)) != 2:
            return False
        for arm in TREATED_ARMS:
            content = contents[arm]
            wrapped = wrap_span(str(topic), arm)
            if content.count(wrapped) != 1:
                return False
            # The question and the two neutrals are
            # byte-identical across arms: removing the
            # fact sentence leaves the same frame.
            if (
                content.replace(fact_sentence(case, arm), fact_sentence(case, "control"), 1)
                != control
            ):
                return False
        plain = pc_content(case, "pc_plain")
        if "*" in plain or "`" in plain:
            return False
        info = pc_content(case, "pc_info")
        if info.count("*") != 4 or "`" in info:
            return False
        if info.replace(f"**{topic}**", str(topic), 1) != plain:
            return False
        if pc_content(case, "pc_plain").count(str(expected)) != 1:
            return False
        pcb = pcb_content(case)
        if pcb.count("***") != 4 or pcb.count(f"***{topic}***") != 2:
            return False
    except Exception:  # noqa: BLE001 -- a malformed case is invalid, never an exception
        return False
    return True


def echo_set(expected: str, arm: str) -> set[str]:
    """The arm's echo set for an expected value: the
    whole-value form (marks around the whole value;
    the whole value uppercased for caps) and the
    word-half form (marks around the word part; the
    word part uppercased for caps -- for caps the two
    forms coincide). The control's set is empty.
    """
    word, _sep, number = expected.partition("-")
    if arm == "bold":
        return {f"**{expected}**", f"**{word}**-{number}"}
    if arm == "italic":
        return {f"*{expected}*", f"*{word}*-{number}"}
    if arm == "monospace":
        return {f"`{expected}`", f"`{word}`-{number}"}
    if arm == "caps":
        return {expected.upper(), f"{word.upper()}-{number}"}
    return set()


def strip_marks(text: str, arm: str) -> str:
    """The arm's form strip: asterisks removed for
    bold and italic, backticks for monospace,
    lowercasing for caps, identity for control.
    """
    if arm in ("bold", "italic"):
        return text.replace("*", "")
    if arm == "monospace":
        return text.replace("`", "")
    if arm == "caps":
        return text.lower()
    return text


def decompose_cell(cell: dict[str, Any], expected: str, arm: str) -> str:
    """The section-5 ordered decision list (first
    match wins; total over completed cells):

    1 ``pass`` -- the verdict passed;
    2 ``boundary_only`` -- the answer digest equals
    the expected digest and the boundary failed (the
    remaining digest-equal combination, boundary
    found with a failed verdict, is unreachable: the
    verdict is exactly that conjunction);
    3 ``wrong_value`` -- the answer parses under the
    frozen grammar (fullmatch) and differs from the
    expected value; the only class counting toward J;
    4 ``echo_<arm>`` -- the answer is in the arm's
    echo set (the value intact, wearing the arm's own
    treatment);
    5 ``form_only`` -- stripping the arm's marks (for
    caps: lowercasing) yields exactly the expected
    value;
    6 ``non_answer`` -- everything else, including a
    marked topic offered as the answer and marks
    under the caps arm.

    A markdown-stripped echo is indistinguishable from
    the value itself and lands in ``pass`` by
    construction (disclosed in the design). For the
    control arm the echo and form classes are
    unreachable by construction (empty echo set,
    identity strip).
    """
    if cell.get("verdict") is True:
        return "pass"
    if (
        cell.get("answer_digest") == digest_text(expected)
        and cell.get("boundary_found") is not True
    ):
        return "boundary_only"
    text = cell.get("answer_text")
    if type(text) is not str:
        return "non_answer"
    if text != expected and _value_grammar_ok(text):
        return "wrong_value"
    if text in echo_set(expected, arm):
        return f"echo_{arm}"
    if text != expected and strip_marks(text, arm) == expected:
        return "form_only"
    return "non_answer"


def decompose_pair(control_cell: dict[str, Any], arm_cell: dict[str, Any]) -> str:
    """The sibling total pair partition (design
    section 5): equal answer digests are a
    boundary-only pair; differing digests with both
    boundaries found are exact-only; anything else is
    mixed. Reported, never attributed.
    """
    if control_cell.get("answer_digest") == arm_cell.get("answer_digest"):
        return "boundary_only"
    if control_cell.get("boundary_found") and arm_cell.get("boundary_found"):
        return "exact_only"
    return "mixed"


def pc_gate_passed(info_passes: int, margin_base: int, margin_harm: int, n_incomplete: int) -> bool:
    """The positive-control gate (design section 6.3
    with the A5 completion guard): over the panel's
    completed PC pairs, B = plain-fail / info-pass
    pairs and H_pc = plain-pass / info-fail pairs; the
    gate passes iff the info arm passed at least
    PC_MIN_INFO_PASSES pairs, the margin B - H_pc is
    at least PC_MIN_MARGIN, at most MAX_INCOMPLETE
    pairs are incomplete, and the margin survives the
    worst-case completion B - (H_pc + k_pc) >=
    PC_MIN_MARGIN (info_passes is unchanged under that
    completion by construction: a completed pass
    cannot be removed, and the adverse completion
    spends every incomplete pair on the margin's harm
    side).
    """
    if info_passes < PC_MIN_INFO_PASSES:
        return False
    if margin_base - margin_harm < PC_MIN_MARGIN:
        return False
    if n_incomplete > MAX_INCOMPLETE:
        return False
    return margin_base - (margin_harm + n_incomplete) >= PC_MIN_MARGIN


def certification_blockers(
    control_passes: int,
    harm: int,
    reverse: int,
    n_incomplete: int,
    domain_size: int,
    pc_passed: bool,
) -> list[str]:
    """The certification gates that fail for one
    (candidate, arm), in the design's printed order:
    the statistic event E = {D >= 10 and H <= 5 and
    S <= 0}; the domain-size gate (M = 24); the
    completion guard (H + k <= 5 and S + k <= 0); the
    positive-control gate. An empty list means every
    gate passes (the label function separately owns
    the fire and acquittal precedence).
    """
    blockers: list[str] = []
    if not (control_passes >= MIN_D and harm <= 5 and harm - reverse <= 0):
        blockers.append("statistic_event")
    if domain_size != CASE_COUNT:
        blockers.append("domain_size")
    if not (harm + n_incomplete <= 5 and harm - reverse + n_incomplete <= 0):
        blockers.append("completion_guard")
    if not pc_passed:
        blockers.append("positive_control")
    return blockers


def classify_headline(
    control_passes: int,
    wrong_values: int,
    harm: int,
    reverse: int,
    n_incomplete: int,
    domain_size: int,
    pc_passed: bool,
) -> str:
    """The combined guarded label (design sections 6.1
    and 6.2): a total function of the paired counts.
    The completion cap withholds every label above
    MAX_INCOMPLETE incomplete in the divergent domain;
    the fire rule issues on observed counts; a
    certificate issues only when every certification
    gate passes (see certification_blockers) -- and
    never beside a fire, since H <= 5 implies J <= 5;
    the sibling acquittals keep their guards
    (no-discordance only at k = 0, below-threshold
    only while J + k < MIN_J), else the arm is
    insufficient.
    """
    if n_incomplete > MAX_INCOMPLETE:
        return "insufficient"
    if control_passes < MIN_D:
        return "insufficient"
    if wrong_values >= MIN_J:
        return "discordant_asymmetric"
    if not certification_blockers(
        control_passes, harm, reverse, n_incomplete, domain_size, pc_passed
    ):
        return "equivalence_certified"
    if wrong_values == 0:
        return "no_discordance_observed" if n_incomplete == 0 else "insufficient"
    return "below_threshold" if wrong_values + n_incomplete < MIN_J else "insufficient"


def case_material_sha256() -> str:
    """Digest of the frozen pools and the generator
    spec (the panel's material identity, seed-free).
    """
    material = {
        "topics": list(TOPICS),
        "words": list(WORDS),
        "neutral_statements": list(NEUTRAL_STATEMENTS),
        "pool_digests": POOL_DIGESTS,
        "generator": GENERATOR_SPEC,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return digest_text(canonical)


def ground_sha256(seed: int) -> str:
    """Digest of one seed's generated panel (the
    ground the protocol commits to).
    """
    canonical = json.dumps(
        generate_cases(seed), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return digest_text(canonical)


def protocol_object(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The protocol block carried by every payload."""
    return {
        "version": PROTOCOL_VERSION,
        "arm_keys": list(ARM_KEYS),
        "treated_arms": list(TREATED_ARMS),
        "pc_arm_keys": list(PC_ARM_KEYS),
        "pcb_arm": PCB_ARM,
        "record_arms": list(RECORD_ARMS),
        "case_count": CASE_COUNT,
        "min_d": MIN_D,
        "min_j": MIN_J,
        "max_incomplete": MAX_INCOMPLETE,
        "delta": DELTA,
        "alpha": ALPHA,
        "pc_min_info_passes": PC_MIN_INFO_PASSES,
        "pc_min_margin": PC_MIN_MARGIN,
        "generator_spec": GENERATOR_SPEC,
        "pool_digests": POOL_DIGESTS,
        "case_material_sha256": case_material_sha256(),
        "ground_sha256": ground_sha256(seed),
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
    """Serialised size of a record list (the byte cap
    is measured on exactly this encoding).
    """
    return len(
        json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    )


def _cell_completed(cell: Any) -> bool:
    """Cell completed."""
    return type(cell) is dict


def _cell_verdict(cell: Any) -> bool:
    """Cell verdict."""
    return type(cell) is dict and cell.get("verdict") is True


def _ids_divergent_records(records: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    """Ids divergent records."""
    return [record for record in records if record["ids_identical"].get(arm) is False]


def aggregate_candidate(
    records: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Per-candidate readout over the ground records:
    per-arm rates with attrition (and, for the five
    main arms, the six-class completed-cell
    decomposition), the per-arm dose deltas, the
    per-arm token-ID report (the regime gate), the
    positive-control block (its counts and gate
    verdict), and per treated arm the combined guarded
    headline over the id-divergent paired population --
    the label with its printed companions (the
    divergent-domain size M, D, J, H, R, S, the
    incomplete count, the rates, the non-answer count,
    the pair decomposition, the failing cells' classes,
    and the certification licence beside the gates that
    blocked it). The mirror recomputes every field from
    the committed records.
    """
    by_fixture = {str(case["fixture_id"]): case for case in cases}
    arms: dict[str, Any] = {}
    for arm in RECORD_ARMS:
        completed = [record for record in records if _cell_completed(record["cells"].get(arm))]
        n_pass = sum(1 for record in completed if _cell_verdict(record["cells"][arm]))
        entry: dict[str, Any] = {
            "rate": (n_pass / len(completed)) if completed else None,
            "n_pass": n_pass,
            "n_completed": len(completed),
            "n_context_exceeded": sum(
                1 for record in records if record["cell_details"].get(arm) == "arm_context_exceeded"
            ),
            "n_evaluation_failed": sum(
                1
                for record in records
                if record["cell_details"].get(arm) == "arm_evaluation_failed"
            ),
        }
        if arm in ARM_KEYS:
            decomposition = dict.fromkeys(CELL_CLASSES, 0)
            for record in completed:
                case = by_fixture[str(record["fixture"])]
                cls = decompose_cell(record["cells"][arm], str(case["expected"]), arm)
                decomposition[cls] += 1
            entry["decomposition"] = decomposition
        arms[arm] = entry
    dose: dict[str, Any] = {}
    for arm in TREATED_ARMS:
        codepoint_deltas = [
            len(arm_content(case, arm)) - len(arm_content(case, "control")) for case in cases
        ]
        byte_deltas = [
            len(arm_content(case, arm).encode("utf-8"))
            - len(arm_content(case, "control").encode("utf-8"))
            for case in cases
        ]
        dose[arm] = {
            "codepoint_delta_min": min(codepoint_deltas, default=0),
            "codepoint_delta_max": max(codepoint_deltas, default=0),
            "byte_delta_min": min(byte_deltas, default=0),
            "byte_delta_max": max(byte_deltas, default=0),
        }
    ids: dict[str, Any] = {}
    for arm in TREATED_ARMS:
        divergent = _ids_divergent_records(records, arm)
        identical = [record for record in records if record["ids_identical"].get(arm) is True]
        flips = sum(
            1
            for record in identical
            if _cell_completed(record["cells"].get("control"))
            and _cell_completed(record["cells"].get(arm))
            and _cell_verdict(record["cells"]["control"]) != _cell_verdict(record["cells"][arm])
        )
        deltas = [int(record["id_length_delta"][arm]) for record in records]
        ids[arm] = {
            "n_ids_identical": len(identical),
            "n_ids_divergent": len(divergent),
            "id_length_delta": deltas,
            "max_abs_id_delta": max((abs(delta) for delta in deltas), default=0),
            "n_same_ids_verdict_flips": flips,
        }
    n_pc_pairs = 0
    pc_info_passes = 0
    pc_plain_fail_info_pass = 0
    pc_plain_pass_info_fail = 0
    for record in records:
        plain = record["cells"].get("pc_plain")
        info = record["cells"].get("pc_info")
        if _cell_completed(plain) and _cell_completed(info):
            n_pc_pairs += 1
            plain_verdict = _cell_verdict(plain)
            info_verdict = _cell_verdict(info)
            if info_verdict:
                pc_info_passes += 1
            if not plain_verdict and info_verdict:
                pc_plain_fail_info_pass += 1
            if plain_verdict and not info_verdict:
                pc_plain_pass_info_fail += 1
    pc_incomplete = CASE_COUNT - n_pc_pairs
    positive_control = {
        "n_pairs": n_pc_pairs,
        "n_incomplete": pc_incomplete,
        "info_passes": pc_info_passes,
        "plain_fail_info_pass": pc_plain_fail_info_pass,
        "plain_pass_info_fail": pc_plain_pass_info_fail,
        "margin": pc_plain_fail_info_pass - pc_plain_pass_info_fail,
        "passed": pc_gate_passed(
            pc_info_passes, pc_plain_fail_info_pass, pc_plain_pass_info_fail, pc_incomplete
        ),
    }
    headline: dict[str, Any] = {}
    for arm in TREATED_ARMS:
        domain = _ids_divergent_records(records, arm)
        domain_size = len(domain)
        paired = [
            record
            for record in domain
            if _cell_completed(record["cells"].get("control"))
            and _cell_completed(record["cells"].get(arm))
        ]
        n_paired = len(paired)
        n_incomplete = domain_size - n_paired
        control_passes = 0
        wrong_values = 0
        harm = 0
        reverse = 0
        arm_passes = 0
        n_non_answer = 0
        pair_decomposition = dict.fromkeys(PAIR_CLASSES, 0)
        failing_classes = dict.fromkeys(CELL_CLASSES, 0)
        for record in paired:
            case = by_fixture[str(record["fixture"])]
            expected = str(case["expected"])
            control_cell = record["cells"]["control"]
            arm_cell = record["cells"][arm]
            control_verdict = _cell_verdict(control_cell)
            arm_verdict = _cell_verdict(arm_cell)
            arm_class = decompose_cell(arm_cell, expected, arm)
            if control_verdict:
                control_passes += 1
                if arm_class == "wrong_value":
                    wrong_values += 1
                if not arm_verdict:
                    harm += 1
            elif arm_verdict:
                reverse += 1
            if arm_verdict:
                arm_passes += 1
            if arm_class == "non_answer":
                n_non_answer += 1
            if control_verdict != arm_verdict:
                pair_decomposition[decompose_pair(control_cell, arm_cell)] += 1
                failing = arm_cell if control_verdict else control_cell
                failing_arm = arm if control_verdict else "control"
                failing_classes[decompose_cell(failing, expected, failing_arm)] += 1
        headline[arm] = {
            "label": classify_headline(
                control_passes,
                wrong_values,
                harm,
                reverse,
                n_incomplete,
                domain_size,
                bool(positive_control["passed"]),
            ),
            "n_population": n_paired,
            "n_incomplete": n_incomplete,
            "n_ids_divergent": domain_size,
            "control_passes": control_passes,
            "wrong_values": wrong_values,
            "harm": harm,
            "n_reverse": reverse,
            "net_harm": harm - reverse,
            "control_rate": (control_passes / n_paired) if n_paired else None,
            "arm_rate": (arm_passes / n_paired) if n_paired else None,
            "n_non_answer": n_non_answer,
            "pair_decomposition": pair_decomposition,
            "failing_classes": failing_classes,
            "delta": DELTA,
            "alpha": ALPHA,
            "pc_gate_passed": positive_control["passed"],
            "certification_blocked_by": certification_blockers(
                control_passes,
                harm,
                reverse,
                n_incomplete,
                domain_size,
                bool(positive_control["passed"]),
            ),
        }
    return {
        "arms": arms,
        "dose": dose,
        "ids": ids,
        "positive_control": positive_control,
        "headline": headline,
        "operating_characteristics": dict(OPERATING_CHARACTERISTICS),
        "n_cases": len(records),
        "n_incomplete": sum(
            1 for record in records if any(record["cells"].get(arm) is None for arm in RECORD_ARMS)
        ),
    }
