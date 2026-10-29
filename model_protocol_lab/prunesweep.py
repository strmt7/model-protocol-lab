"""Prune-sweep (cycle 220): the probe that prunes interior
words from a frozen generated passage at exact word-count
doses and classifies the verdict drop per candidate.

A prompt engineer whose pipeline deletes interior passage
words -- a question-aware compressor of the LLMLingua class,
a middle-chunk RAG trimmer, a word-level summariser or
redactor -- uses this, per candidate, to learn whether
deleting half the non-answer-bearing words of a frozen
fact-recall passage (answer sentence, instruction, and
question byte-identical) produces a large verdict drop at
printed operating characteristics, before that trim ships.
Character caps and tail cuts are not the headline's subject;
they are the ``tail72`` contrast's.

The instrument (docs/PRUNESWEEP_DESIGN.md, as amended by the
fresh review's binding amendments A-I): a generated panel of
24 cases; each case is one user message -- a frozen
instruction, a 17-sentence passage, a frozen question,
joined by newlines. Sixteen context sentences of exactly
nine words carry 144 eligible words; the answer-bearing
fact sentence sits at passage index 8 (the ninth sentence),
is six words, and is never pruned; the instruction and the
question are never pruned. Doses 0 / 18 / 36 / 72 words (0 /
12.5 / 25 / 50 percent of the eligible words) are nested via
one sha256-ordered permutation per case: the position key is
``sha256(f"{seed}:{case_id}:{position}")`` over the
eligible-list index (0-143) in passage order (amendment E).
Six arms per case: the four headline doses, the descriptive
displaced control ``tail72`` (drops exactly the 72 eligible
words of passage sentences 9-16, the dose-matched tail
loss), and the screen arm ``closed_book`` (instruction and
question only; a case the candidate passes closed-book is
screened out of its headline population).

Scoring is the tree's own: ``core.evaluate``'s byte-exact
verdict conjoined with ``boundary_found``, on a Fixture with
the same id, group, and expected at every arm, so arm seeds
are identical by construction (``request_seed_for`` is
id-derived). Every completed cell is decomposed from its
answer digest alone (amendment B, the negation section 4.2
disposition): ``exact_pass`` -> ``boundary_only`` ->
``wrong_value`` -> ``non_value``, in that order; a
``boundary_only`` dose72 cell does NOT count toward the
drop count A -- it is printed beside A as boundary
attrition. The decision is a total function over the cases
completed in all four headline arms, not closed-book
screened, and not ids-identical between dose0 and dose72:
C = dose0 passes; A = cases in C whose dose72 cell fails as
``wrong_value`` or ``non_value``; B = cases failing dose0
and passing dose72. C < 12 -> ``insufficient``; C >= 12 and
A >= 5 and the two-sided exact McNemar p(A, B) < 0.05 ->
``prune_drop_observed``; else ``below_threshold``. The
(12, 5) floors are the rule's text; the McNemar conjunct is
load-bearing -- (5, 0) gives p = 0.0625 and does not fire,
while (6, 0) gives 0.03125. Of the (C, A, B) triples the
rule can see, exactly 492 fire, none with A <= B, the
minimum firing A is 6 and the maximum firing B is 6 (exact
enumeration, amendment A).

The label attributes the drop to the named word-loss
treatment AS A BUNDLE (amendment F): information loss +
prompt shortening + fact-position shift + sentence breakage
move together under the scattered operator. The executed
operator constants: at dose72 the fact sentence shifts
earlier by a mean of 36.0 words (hypergeometric sd 3.01)
against exactly 0 under tail72; an expected 15.95 of the 16
context sentences are partially broken at dose72 (0.02
removed whole, 0.02 intact) against 0 broken under tail72,
which instead relocates the fact to passage-final position.
The tail72 contrast therefore separates scatter-vs-block
jointly with fact position, never either alone; when the
two arms' discordance is within noise, the report prints
the word-count-loss reading in words (section 3.3 of the
design).

Operating characteristics are exact enumerations, not
Monte Carlo (amendment A; the candidate text's printed null
row of 0 / 40,000 was false and is deleted). Stated
conditional basis -- pass0 Bernoulli p0; fail72 given pass0
with probability q; pass72 given fail0 with probability r;
arms conditionally independent: random labeller (p0 = .5,
q = r = .5) fires 0.0121 per decision; competent no-effect
(p0 = .9, q = r = .1) fires 0.0143; jitter (p0 = .97,
q = r = .02) fires 0.000005; power at q72 = .25 / .50 / .75
is 0.3969 / 0.9601 / 0.9996 (graded and threshold-at-72 are
one computation printed twice -- the rule reads only dose0
and dose72). Exchangeable basis -- per-case joint
(pass, pass) = m - v, (pass, fail) = (fail, pass) = v,
(fail, fail) = 1 - m - v, with v the per-direction
discordance: competent (m = .9) fires 0.0030 at v = .09,
0.0004 at v = .05, 0.000004 at v = .02; the random labeller
(m = .5, v = .25) fires 0.0121, identical to the stated
row, as it must be. Family-wise over a 24-candidate panel:
stated basis E = 0.34 false labels, P(at least one) = 0.29;
exchangeable basis E = 0.07, P = 0.07. The enumerations are
computed on the unscreened 24-case population;
closed-book screening only shrinks the population (stated
basis: competent 0.0143 at N = 24 falls to 0.0017 at
N = 16; random 0.0121 falls to 0.0072 at N = 17 and 0.0019
at N = 14). Two candidate shape gates were executed and
rejected (amendment I; the graded interpolation
q_d = q72 * d / 72 is printed here because the candidate
text never printed it): a returns cap (R <= 1) collapses
graded q72 = .50 power from 0.9601 to 0.0941 (the candidate
text's printed 0.031 does not reproduce under the stated
interpolation and is corrected), and an intermediate-
support requirement (A36 >= 2) collapses threshold-at-72
power from 0.9601 to exactly 0. Shape is a readout, never
a gate; the word "sweep" in the flag name denotes the dose
ladder, never a curve shape.

Vocabulary is the tree's frozen fail-closed vocabulary:
labels {insufficient, prune_drop_observed,
below_threshold}; stage details {cancelled,
prune_record_budget_exceeded, prune_screen_failed}; cell
details {"", arm_context_exceeded, arm_evaluation_failed};
whole-stage failure {"available": False, "detail":
"prune_unavailable"}. Records carry digests only (per-arm
content digests, render-id digests, token counts, answer
digests) -- no raw answer text. Counts: 24 cases x 4
headline arms x 24 candidates = 2,304 headline generations;
24 x 6 x 24 = 3,456 legal maximum; 576 ground records; the
house 12 MiB records cap. No winner readout, no selection
re-application, no pooled discordance across candidates:
the panel is generated, so the numform settlement applies
with full force. Informational only: the stage never
affects selection and never fails the run.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .confidentiality import mcnemar_exact_p

__all__ = [
    "ARM_KEYS",
    "CASE_COUNT",
    "CELL_DETAILS",
    "CONTEXT_SENTENCES",
    "DECISIONS",
    "DESCRIPTIVE_ARM",
    "DOSES",
    "ELIGIBLE_WORDS",
    "EXPECTED_BROKEN_SENTENCES_DOSE72",
    "EXPECTED_REMOVED_SENTENCES_DOSE72",
    "FACT_SHIFT_MEAN_WORDS",
    "FACT_SHIFT_SD_WORDS",
    "FACT_WORDS",
    "GENERATION_BUDGET",
    "GROUP",
    "HEADLINE_ARMS",
    "HEADLINE_GENERATIONS",
    "INSTRUCTION",
    "MAX_CANDIDATES",
    "MAX_GROUND_RECORDS",
    "MAX_RECORDS_BYTES",
    "MIN_A",
    "MIN_C",
    "NEGATION_WORDS",
    "OBJECTS",
    "OPERATING_CHARACTERISTICS",
    "PLACES",
    "PROTOCOL_VERSION",
    "QUESTION_TEMPLATE",
    "SCREEN_ARM",
    "STAGE_DETAILS",
    "SUBJECTS",
    "TEMPLATES",
    "TIMES",
    "TOPICS",
    "VALUES",
    "WORDS_PER_SENTENCE",
    "aggregate_candidate",
    "arm_content",
    "case_material_sha256",
    "classify_decision",
    "decompose_cell",
    "digest_ids",
    "digest_text",
    "drop_order",
    "dropped_positions",
    "dropped_set_shares",
    "generate_case",
    "generate_cases",
    "ground_sha256",
    "protocol_fingerprint",
    "protocol_object",
    "validate_case",
]

PROTOCOL_VERSION = 1

GROUP = "prune"

CASE_COUNT = 24
CONTEXT_SENTENCES = 16
WORDS_PER_SENTENCE = 9
#: Eligible words per case: the context sentences' words.
#: The fact sentence, instruction, and question are never
#: eligible.
ELIGIBLE_WORDS = CONTEXT_SENTENCES * WORDS_PER_SENTENCE
#: Context sentences before the fact sentence; the fact is
#: passage sentence index 8 (the ninth of 17).
FACT_CONTEXT_SLOTS = 8
FACT_WORDS = 6

#: Word doses, as eligible-word counts (0 / 12.5 / 25 / 50
#: percent of the 144 eligible words).
DOSES: tuple[int, ...] = (0, 18, 36, 72)
HEADLINE_ARMS: tuple[str, ...] = ("dose0", "dose18", "dose36", "dose72")
DESCRIPTIVE_ARM = "tail72"
SCREEN_ARM = "closed_book"
ARM_KEYS: tuple[str, ...] = (*HEADLINE_ARMS, DESCRIPTIVE_ARM, SCREEN_ARM)
DOSE_BY_ARM: dict[str, int] = {
    "dose0": 0,
    "dose18": 18,
    "dose36": 36,
    "dose72": 72,
}

#: Decision floors, frozen by the design (section 5). The
#: McNemar conjunct is load-bearing: the minimum firing A
#: is 6, not 5.
MIN_C = 12
MIN_A = 5
SIGNIFICANCE = 0.05

MAX_CANDIDATES = 24
MAX_GROUND_RECORDS = MAX_CANDIDATES * CASE_COUNT
GENERATION_BUDGET = len(ARM_KEYS) * MAX_GROUND_RECORDS
HEADLINE_GENERATIONS = len(HEADLINE_ARMS) * MAX_GROUND_RECORDS
#: The house 12 MiB records bound, for uniformity with the
#: sibling probes.
MAX_RECORDS_BYTES = 12 * 1024 * 1024

#: Section labels (exactly the negation label pattern, with
#: ``prune_drop_observed`` formed as
#: ``negation_drop_observed`` was).
DECISIONS: tuple[str, ...] = (
    "insufficient",
    "prune_drop_observed",
    "below_threshold",
)

#: Decomposition of a completed cell, from its answer
#: digest alone, in this order (amendment B).
DECOMPOSITION_CLASSES: tuple[str, ...] = (
    "exact_pass",
    "boundary_only",
    "wrong_value",
    "non_value",
)

CELL_DETAILS: tuple[str, ...] = ("", "arm_context_exceeded", "arm_evaluation_failed")

STAGE_DETAILS: tuple[str, ...] = (
    "cancelled",
    "prune_record_budget_exceeded",
    "prune_screen_failed",
)

INSTRUCTION = "Read the passage and answer the question."
FACT_TEMPLATE = "The {topic} code is {value}."
QUESTION_TEMPLATE = "What is the {topic} code? Reply with only the code word."

#: Eight frozen two-word topics; case i uses TOPICS[i % 8].
TOPICS: tuple[str, ...] = (
    "harbor light",
    "summit trail",
    "quarry gate",
    "tundra post",
    "granite arch",
    "willow ford",
    "copper mill",
    "saddle pass",
)

#: Twenty-four distinct ASCII code words; case i's expected
#: value is VALUES[i].
VALUES: tuple[str, ...] = (
    "amber",
    "birch",
    "cobalt",
    "dune",
    "ember",
    "flint",
    "garnet",
    "hazel",
    "ivory",
    "jasper",
    "kelvin",
    "linden",
    "magnet",
    "nickel",
    "onyx",
    "pepper",
    "quartz",
    "rowan",
    "saffron",
    "timber",
    "umber",
    "violet",
    "walnut",
    "yonder",
)

#: Slot pools (amendments C and D): disjoint from the value
#: pool, from topic words, from fact-template words, and
#: from the instruction and question vocabulary.
SUBJECTS: tuple[str, ...] = (
    "archive",
    "ledger",
    "cabinet",
    "chapel",
    "meadow",
    "workshop",
    "station",
    "garden",
    "library",
    "market",
    "office",
    "depot",
)
OBJECTS: tuple[str, ...] = (
    "forms",
    "letters",
    "tools",
    "charts",
    "samples",
    "records",
    "parcels",
    "volumes",
    "tickets",
    "bundles",
    "reports",
    "crates",
)
PLACES: tuple[str, ...] = (
    "annex",
    "attic",
    "cellar",
    "landing",
    "terrace",
    "alcove",
    "gallery",
    "storeroom",
    "corridor",
    "balcony",
    "vestibule",
    "loft",
)
TIMES: tuple[str, ...] = (
    "morning",
    "midday",
    "evening",
    "weekdays",
    "season",
    "harvest",
    "holiday",
    "weekend",
    "noon",
    "dusk",
    "dawn",
    "closing",
)

#: Four frozen context templates, each exactly nine words
#: with the slots filled. The template for sentence slot s
#: is TEMPLATES[s % 4].
TEMPLATES: tuple[str, ...] = (
    "The {subject} holds {object} inside the {place} each {time}.",
    "A {subject} keeps {object} near the {place} during {time}.",
    "Workers quietly move {object} past the {place} during {time}.",
    "Each {time} the {subject} stores {object} under the {place}.",
)

#: Frozen negation word list (amendment D): no context
#: sentence may contain one of these words, so no dropped or
#: kept context word can negate the fact.
NEGATION_WORDS: frozenset[str] = frozenset(
    {
        "no",
        "not",
        "never",
        "none",
        "cannot",
        "without",
        "neither",
        "nor",
        "nothing",
        "nobody",
        "nowhere",
        "hardly",
        "barely",
        "scarcely",
    }
)

#: Amendment F operator constants, executed exactly on the
#: frozen operator (hypergeometric: 72 dropped of 144
#: eligible, 72 pre-fact eligible positions). Printed in
#: the report's bundle fencing.
FACT_SHIFT_MEAN_WORDS = 36.0
FACT_SHIFT_SD_WORDS = 3.01
EXPECTED_BROKEN_SENTENCES_DOSE72 = 15.95
EXPECTED_REMOVED_SENTENCES_DOSE72 = 0.02

#: Operating characteristics of the frozen rule by exact
#: enumeration (amendment A; see the module docstring for
#: the two bases' equations). Printed beside every
#: decision; never re-derived at run time.
OPERATING_CHARACTERISTICS: dict[str, Any] = {
    "null_random_stated": 0.0121,
    "null_competent_stated": 0.0143,
    "null_jitter_stated": 0.000005,
    "null_competent_exchangeable_v09": 0.003,
    "null_competent_exchangeable_v05": 0.0004,
    "null_competent_exchangeable_v02": 0.000004,
    "power_q25": 0.3969,
    "power_q50": 0.9601,
    "power_q75": 0.9996,
    "familywise_expected_stated": 0.34,
    "familywise_any_stated": 0.29,
    "familywise_expected_exchangeable": 0.07,
    "familywise_any_exchangeable": 0.07,
    "shape_returns_cap_graded": 0.094,
    "shape_intermediate_support_threshold": 0.0,
}


def digest_text(text: str) -> str:
    """SHA-256 hex digest of a text (the records' digest form)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_ids(ids: tuple[int, ...]) -> str:
    """SHA-256 hex digest of a render id sequence, so records
    carry the sequence's identity without its contents.
    """
    canonical = ",".join(str(i) for i in ids)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _sha(text: str) -> str:
    """Return the sha."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_VALUE_DIGESTS: frozenset[str] = frozenset(digest_text(value) for value in VALUES)
_POOL_WORDS: frozenset[str] = frozenset(
    word for pool in (SUBJECTS, OBJECTS, PLACES, TIMES) for word in pool
)


def _draw_sentence(seed: int, case_id: str, slot: int, seen: set[str]) -> str:
    """One context sentence by hash-ranked slot draw with
    distinctness rejection (the design section 3.4 rule): the
    draw digests sha256(f"{seed}:{case_id}:slot:{slot}:{counter}")
    for counter 0, 1, 2, ...; four hex windows select the
    subject, object, place, and time; a sentence already in
    ``seen`` is redrawn with the next counter.
    """
    template = TEMPLATES[slot % len(TEMPLATES)]
    counter = 0
    while counter < 10000:
        digest = _sha(f"{seed}:{case_id}:slot:{slot}:{counter}")
        text = template.format(
            subject=SUBJECTS[int(digest[0:8], 16) % len(SUBJECTS)],
            object=OBJECTS[int(digest[8:16], 16) % len(OBJECTS)],
            place=PLACES[int(digest[16:24], 16) % len(PLACES)],
            time=TIMES[int(digest[24:32], 16) % len(TIMES)],
        )
        if text not in seen:
            return text
        counter += 1
    raise ValueError("prune-sweep sentence draw did not terminate")


def generate_case(seed: int, index: int) -> dict[str, Any]:
    """The frozen generated case ``index`` (0-23) at ``seed``:
    16 drawn context sentences plus the fact sentence built
    from the case's topic and value.
    """
    case_id = f"prune-{index:02d}"
    topic = TOPICS[index % len(TOPICS)]
    value = VALUES[index]
    seen: set[str] = set()
    sentences: list[str] = []
    for slot in range(CONTEXT_SENTENCES):
        text = _draw_sentence(seed, case_id, slot, seen)
        seen.add(text)
        sentences.append(text)
    return {
        "fixture_id": case_id,
        "topic": topic,
        "value": value,
        "expected": value,
        "context_sentences": tuple(sentences),
        "fact_sentence": FACT_TEMPLATE.format(topic=topic, value=value),
    }


def generate_cases(seed: int) -> tuple[dict[str, Any], ...]:
    """The full 24-case panel at ``seed``, in case order."""
    return tuple(generate_case(seed, index) for index in range(CASE_COUNT))


def drop_order(seed: int, case_id: str) -> tuple[int, ...]:
    """The case's frozen drop permutation over eligible
    positions 0-143: positions sorted by the hex of
    sha256(f"{seed}:{case_id}:{position}") (amendment E --
    the position is the eligible-list index in passage
    order). Dose k drops the first k positions, so the drop
    sets are nested by construction.
    """
    return tuple(
        sorted(
            range(ELIGIBLE_WORDS),
            key=lambda position: _sha(f"{seed}:{case_id}:{position}"),
        )
    )


def dropped_positions(case_id: str, arm: str, seed: int) -> frozenset[int]:
    """The eligible positions dropped under ``arm``.
    ``tail72`` drops exactly positions 72-143 (the eligible
    words of passage sentences 9-16); the dose arms drop the
    first ``dose`` positions of the drop order;
    ``closed_book`` drops nothing (its message has no
    passage).
    """
    if arm == DESCRIPTIVE_ARM:
        return frozenset(range(ELIGIBLE_WORDS // 2, ELIGIBLE_WORDS))
    dose = DOSE_BY_ARM.get(arm)
    if dose is None:
        return frozenset()
    return frozenset(drop_order(seed, case_id)[:dose])


def _passage(case: dict[str, Any], dropped: frozenset[int]) -> str:
    """The passage text under a drop set: context sentences
    keep their surviving words in order, sentences that lose
    every word are omitted, and the fact sentence sits after
    the first eight context slots, byte-identical always.
    """
    kept: dict[int, str] = {}
    for slot, sentence in enumerate(case["context_sentences"]):
        words = sentence.split(" ")
        surviving = [
            word
            for offset, word in enumerate(words)
            if slot * WORDS_PER_SENTENCE + offset not in dropped
        ]
        if surviving:
            kept[slot] = " ".join(surviving)
    sentences = [kept[slot] for slot in range(FACT_CONTEXT_SLOTS) if slot in kept]
    sentences.append(str(case["fact_sentence"]))
    sentences.extend(
        kept[slot] for slot in range(FACT_CONTEXT_SLOTS, CONTEXT_SENTENCES) if slot in kept
    )
    return " ".join(sentences)


def arm_content(case: dict[str, Any], arm: str, seed: int) -> str:
    """The full user-message content of one arm: instruction,
    passage, and question joined by newlines (the passage is
    absent under ``closed_book``).
    """
    question = QUESTION_TEMPLATE.format(topic=case["topic"])
    if arm == SCREEN_ARM:
        return f"{INSTRUCTION}\n{question}"
    dropped = dropped_positions(str(case["fixture_id"]), arm, seed)
    return f"{INSTRUCTION}\n{_passage(case, dropped)}\n{question}"


def case_material_sha256() -> str:
    """Digest over every frozen text and list the generator
    draws from, so the protocol binds the population's
    definition, not just its seed.
    """
    material = {
        "instruction": INSTRUCTION,
        "fact_template": FACT_TEMPLATE,
        "question_template": QUESTION_TEMPLATE,
        "topics": list(TOPICS),
        "values": list(VALUES),
        "subjects": list(SUBJECTS),
        "objects": list(OBJECTS),
        "places": list(PLACES),
        "times": list(TIMES),
        "templates": list(TEMPLATES),
        "negation_words": sorted(NEGATION_WORDS),
        "doses": list(DOSES),
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ground_sha256(seed: int) -> str:
    """Digest over the seed's generated ground population
    (canonical rows binding fixture id, topic, value, the 16
    context sentences, and the fact sentence of every case).
    """
    rows = [
        [
            case["fixture_id"],
            case["topic"],
            case["value"],
            list(case["context_sentences"]),
            case["fact_sentence"],
        ]
        for case in generate_cases(seed)
    ]
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def protocol_object(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The frozen instrument description a payload is checked
    against: arms, doses, floors, budgets, the case-material
    and ground digests, and the seed / backend identity that
    generated the records.
    """
    return {
        "version": PROTOCOL_VERSION,
        "arm_keys": list(ARM_KEYS),
        "headline_arms": list(HEADLINE_ARMS),
        "descriptive_arm": DESCRIPTIVE_ARM,
        "screen_arm": SCREEN_ARM,
        "doses": list(DOSES),
        "case_count": CASE_COUNT,
        "eligible_words": ELIGIBLE_WORDS,
        "case_material_sha256": case_material_sha256(),
        "ground_sha256": ground_sha256(seed),
        "min_c": MIN_C,
        "min_a": MIN_A,
        "generation_budget": GENERATION_BUDGET,
        "headline_generations": HEADLINE_GENERATIONS,
        "max_ground_records": MAX_GROUND_RECORDS,
        "max_records_bytes": MAX_RECORDS_BYTES,
        "max_candidates": MAX_CANDIDATES,
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


def _bare(word: str) -> str:
    """A passage word with its sentence-final period removed."""
    return word[:-1] if word.endswith(".") else word


def _word_classes(word: str) -> tuple[bool, bool]:
    """(is_numeral, is_negation) for one eligible word
    (amendment D's dropped-set share classes).
    """
    bare = _bare(word).lower()
    return any(char.isdigit() for char in bare), bare in NEGATION_WORDS


def dropped_set_shares(case: dict[str, Any], seed: int) -> dict[str, Any]:
    """Per drop arm, the share of dropped words in the
    numeral and negation classes (amendment D; expected 0 by
    the screens, printed and validator-mirrored, never
    assumed).
    """
    words = [word for sentence in case["context_sentences"] for word in sentence.split(" ")]
    shares: dict[str, Any] = {}
    for arm in (*HEADLINE_ARMS[1:], DESCRIPTIVE_ARM):
        dropped = dropped_positions(str(case["fixture_id"]), arm, seed)
        numeral = negation = 0
        for position in dropped:
            is_numeral, is_negation = _word_classes(words[position])
            numeral += 1 if is_numeral else 0
            negation += 1 if is_negation else 0
        total = len(dropped)
        shares[arm] = {
            "numeral": numeral / total if total else 0.0,
            "negation": negation / total if total else 0.0,
        }
    return shares


def validate_case(case: dict[str, Any], seed: int) -> bool:
    """The pre-generation invariant battery over one case
    (design section 3.4 + amendments C and D), executed by
    the stage before any generation and re-executed on the
    printed lists in the tests: the case equals its
    regeneration from the seed; 144 eligible words in 16
    distinct 9-word sentences; the fact sentence is 6 words
    at passage index 8; the stem screen (the value appears
    nowhere in the context and shares no 4-character prefix
    with any context word); slot pools disjoint from the
    value pool, topic words, fact-template words, and the
    instruction and question vocabulary; no context word
    carries a digit or is a negation word; the acrostic
    screen (sentence-initial letters, consecutive or single,
    spell no pool value); drop sets nested; tail72 is
    exactly positions 72-143; the instruction, question, and
    fact sentence are byte-identical in every arm's content.
    """
    try:
        fixture_id = str(case["fixture_id"])
        index = int(fixture_id.rsplit("-", 1)[1])
        if not 0 <= index < CASE_COUNT or fixture_id != f"prune-{index:02d}":
            return False
        if generate_case(seed, index) != case:
            return False
        context = case["context_sentences"]
        if len(context) != CONTEXT_SENTENCES or len(set(context)) != CONTEXT_SENTENCES:
            return False
        words: list[str] = []
        for sentence in context:
            parts = sentence.split(" ")
            if len(parts) != WORDS_PER_SENTENCE:
                return False
            words.extend(parts)
        if len(words) != ELIGIBLE_WORDS:
            return False
        if len(str(case["fact_sentence"]).split(" ")) != FACT_WORDS:
            return False
        value = str(case["value"])
        bare_words = [_bare(word) for word in words]
        if value in bare_words or value in " ".join(context):
            return False
        if any(word[:4] == value[:4] for word in bare_words):
            return False
        topic_words = set(str(case["topic"]).split(" "))
        if _POOL_WORDS & set(VALUES) or _POOL_WORDS & topic_words:
            return False
        if _POOL_WORDS & {"the", "code", "is"}:
            return False
        instruction_words = set(INSTRUCTION.lower().replace(".", "").split(" "))
        question_words = set(
            QUESTION_TEMPLATE.format(topic=case["topic"])
            .lower()
            .replace(".", "")
            .replace("?", "")
            .split(" ")
        )
        if _POOL_WORDS & instruction_words or _POOL_WORDS & (question_words - topic_words):
            return False
        for word in bare_words:
            is_numeral, is_negation = _word_classes(word)
            if is_numeral or is_negation:
                return False
        initials = (
            "".join(sentence[0].lower() for sentence in context[:FACT_CONTEXT_SLOTS])
            + str(case["fact_sentence"])[0].lower()
            + "".join(sentence[0].lower() for sentence in context[FACT_CONTEXT_SLOTS:])
        )
        for candidate in VALUES:
            for start in range(len(initials)):
                for end in range(start + 1, len(initials) + 1):
                    if initials[start:end] == candidate:
                        return False
        order = drop_order(seed, fixture_id)
        if not set(order[:18]) < set(order[:36]) < set(order[:72]):
            return False
        if dropped_positions(fixture_id, DESCRIPTIVE_ARM, seed) != frozenset(
            range(ELIGIBLE_WORDS // 2, ELIGIBLE_WORDS)
        ):
            return False
        question = QUESTION_TEMPLATE.format(topic=case["topic"])
        for arm in ARM_KEYS:
            content = arm_content(case, arm, seed)
            lines = content.split("\n")
            if lines[0] != INSTRUCTION or lines[-1] != question:
                return False
            if arm != SCREEN_ARM and str(case["fact_sentence"]) not in lines[1]:
                return False
        base_words = len(arm_content(case, "dose0", seed).split("\n")[1].split(" "))
        for arm, dose in (("dose18", 18), ("dose36", 36), ("dose72", 72), ("tail72", 72)):
            pruned = len(arm_content(case, arm, seed).split("\n")[1].split(" "))
            if base_words - pruned != dose:
                return False
        return True
    except Exception:  # noqa: BLE001 -- a malformed case fails the screen closed
        return False


def decompose_cell(cell: dict[str, Any], expected_digest: str) -> str:
    """The decomposition class of one completed cell, from
    its answer digest alone, in the frozen order (amendment
    B): ``exact_pass`` (the verdict passed), then
    ``boundary_only`` (the digest is the expected one; the
    boundary event alone failed), then ``wrong_value`` (the
    digest is a different pool value's), else ``non_value``.
    """
    if cell.get("verdict") is True:
        return "exact_pass"
    digest = cell.get("answer_digest")
    if digest is not None and digest == expected_digest:
        return "boundary_only"
    if digest is not None and digest in _VALUE_DIGESTS:
        return "wrong_value"
    return "non_value"


def classify_decision(direct_passes: int, drop_cases: int, reverse: int) -> str:
    """The frozen decision (design section 5): C < 12 ->
    ``insufficient`` (withheld, never an acquittal);
    A >= 5 with the two-sided exact McNemar p(A, B) < 0.05
    -> ``prune_drop_observed``; else ``below_threshold`` (an
    observed null at this panel size, weak evidence).
    """
    if direct_passes < MIN_C:
        return "insufficient"
    p = mcnemar_exact_p(drop_cases, reverse)
    if drop_cases >= MIN_A and p is not None and p < SIGNIFICANCE:
        return "prune_drop_observed"
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


def _verdict(record: dict[str, Any], key: str) -> bool:
    """One record's verdict for one arm; False when the cell
    did not complete. (On a completed-restricted set this is
    exactly the cell's verdict.)
    """
    cell = _cell(record, key)
    return bool(cell is not None and cell["verdict"])


def _screened(record: dict[str, Any]) -> bool:
    """The closed-book screen: the candidate passed the case
    with no passage at all, so passage loss cannot move it.
    """
    cell = _cell(record, SCREEN_ARM)
    return bool(cell is not None and cell["verdict"])


def _ids_identical(record: dict[str, Any]) -> bool:
    """The ids-identical gate, derived from the recorded
    per-arm id digests alone: the dose0 and dose72 renders
    tokenise identically, so the pair is a rendering anomaly,
    not a measurement.
    """
    digests = record.get("id_digests")
    if not isinstance(digests, dict):
        return False
    return digests.get("dose0") is not None and digests.get("dose0") == digests.get("dose72")


def _first_flip(record: dict[str, Any]) -> str:
    """The first headline dose whose verdict fails, over
    ("dose18", "dose36", "dose72"); "never" when none does.
    """
    for arm in HEADLINE_ARMS[1:]:
        if not _verdict(record, arm):
            return arm
    return "never"


# skipcq: PY-R1000 - branches are frozen payload-contract cases pinned by tests
def aggregate_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-candidate aggregates over its ground records:
    per-arm completion, attrition, and decomposition; the
    headline decision over the screened, ids-diverged,
    fully completed population; the descriptive shape
    readout; the tail72 displaced-control contrast; the
    id-length report; and the dropped-set class shares.
    Every count carries its denominator; discordance is
    never pooled across candidates -- this function sees one
    candidate's records only.
    """
    arms: dict[str, Any] = {}
    for key in ARM_KEYS:
        n_completed = 0
        n_pass = 0
        n_context = 0
        n_failed = 0
        decomposition = dict.fromkeys(DECOMPOSITION_CLASSES, 0)
        for record in records:
            detail = _detail(record, key)
            if detail == "arm_context_exceeded":
                n_context += 1
            elif detail == "arm_evaluation_failed":
                n_failed += 1
            cell = _cell(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
            decomposition[decompose_cell(cell, str(record.get("expected_digest", "")))] += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": _rate(n_pass, n_completed),
            "n_context_exceeded": n_context,
            "n_evaluation_failed": n_failed,
            "decomposition": decomposition,
        }
    complete = [
        record for record in records if all(_cell(record, arm) is not None for arm in HEADLINE_ARMS)
    ]
    screened = [record for record in complete if _screened(record)]
    unscreened = [record for record in complete if not _screened(record)]
    ids_excluded = [record for record in unscreened if _ids_identical(record)]
    population = [record for record in unscreened if not _ids_identical(record)]
    direct = [record for record in population if _verdict(record, "dose0")]
    drop_cases = 0
    boundary_attrition = 0
    a_by_first_flip = dict.fromkeys(HEADLINE_ARMS[1:], 0)
    for record in direct:
        cell72 = _cell(record, "dose72")
        if cell72 is None or cell72["verdict"]:
            continue
        klass = decompose_cell(cell72, str(record.get("expected_digest", "")))
        if klass == "boundary_only":
            boundary_attrition += 1
        elif klass in ("wrong_value", "non_value"):
            drop_cases += 1
            a_by_first_flip[_first_flip(record)] += 1
    reverse = sum(
        1 for record in population if not _verdict(record, "dose0") and _verdict(record, "dose72")
    )
    headline = {
        "n_population": len(population),
        "n_headline_incomplete": len(records) - len(complete),
        "n_screened": len(screened),
        "n_closed_book_incomplete": sum(
            1 for record in records if _cell(record, SCREEN_ARM) is None
        ),
        "n_ids_identical": len(ids_excluded),
        "direct_passes": len(direct),
        "drop_cases": drop_cases,
        "boundary_attrition": boundary_attrition,
        "reverse": reverse,
        "decision": classify_decision(len(direct), drop_cases, reverse),
        "a_by_first_flip": a_by_first_flip,
    }
    histogram = dict.fromkeys(HEADLINE_ARMS[1:], 0)
    histogram["never"] = 0
    returns = 0
    for record in direct:
        histogram[_first_flip(record)] += 1
        if (not _verdict(record, "dose18") or not _verdict(record, "dose36")) and _verdict(
            record, "dose72"
        ):
            returns += 1
    shape = {
        "first_flip_histogram": histogram,
        "returns": returns,
        "dose_pass_rates": {
            arm: _rate(
                sum(1 for record in population if _verdict(record, arm)),
                len(population),
            )
            for arm in HEADLINE_ARMS
        },
    }
    n_compared = 0
    dose72_fails = 0
    tail72_fails = 0
    dose72_only = 0
    tail72_only = 0
    for record in population:
        if _cell(record, DESCRIPTIVE_ARM) is None:
            continue
        n_compared += 1
        fail72 = not _verdict(record, "dose72")
        fail_tail = not _verdict(record, DESCRIPTIVE_ARM)
        dose72_fails += 1 if fail72 else 0
        tail72_fails += 1 if fail_tail else 0
        if fail72 and not fail_tail:
            dose72_only += 1
        elif fail_tail and not fail72:
            tail72_only += 1
    p_tail = mcnemar_exact_p(dose72_only, tail72_only)
    rival = dose72_fails > 0 and (
        dose72_only + tail72_only == 0 or (p_tail is not None and p_tail >= SIGNIFICANCE)
    )
    tail_contrast = {
        "n_compared": n_compared,
        "dose72_fails": dose72_fails,
        "tail72_fails": tail72_fails,
        "dose72_only": dose72_only,
        "tail72_only": tail72_only,
        "rival_reading": rival,
    }
    ids: dict[str, Any] = {}
    for arm in (*HEADLINE_ARMS[1:], DESCRIPTIVE_ARM):
        deltas: list[int] = []
        n_ids_identical = 0
        n_same_ids_flips = 0
        for record in records:
            tokens = record.get("arm_tokens")
            tokens = tokens if isinstance(tokens, dict) else {}
            digests = record.get("id_digests")
            digests = digests if isinstance(digests, dict) else {}
            treated = tokens.get(arm)
            baseline = tokens.get("dose0")
            if type(treated) is int and type(baseline) is int:
                deltas.append(treated - baseline)
            else:
                deltas.append(0)
            base_digest = digests.get("dose0")
            if type(base_digest) is str and base_digest != "" and base_digest == digests.get(arm):
                n_ids_identical += 1
                base_cell = _cell(record, "dose0")
                arm_cell = _cell(record, arm)
                if (
                    base_cell is not None
                    and arm_cell is not None
                    and base_cell["verdict"] != arm_cell["verdict"]
                ):
                    n_same_ids_flips += 1
        ids[arm] = {
            "id_length_delta_by_case": deltas,
            "max_abs_id_delta": max((abs(delta) for delta in deltas), default=0),
            "n_ids_identical": n_ids_identical,
            "n_same_ids_flips": n_same_ids_flips,
        }
    share_arms = (*HEADLINE_ARMS[1:], DESCRIPTIVE_ARM)
    dropped_shares: dict[str, Any] = {}
    for arm in share_arms:
        numeral_total = 0.0
        negation_total = 0.0
        counted = 0
        for record in records:
            shares = record.get("dropped_shares")
            if not isinstance(shares, dict) or not isinstance(shares.get(arm), dict):
                continue
            numeral_total += float(shares[arm].get("numeral", 0.0))
            negation_total += float(shares[arm].get("negation", 0.0))
            counted += 1
        dropped_shares[arm] = {
            "numeral": numeral_total / counted if counted else 0.0,
            "negation": negation_total / counted if counted else 0.0,
        }
    return {
        "arms": arms,
        "headline": headline,
        "shape": shape,
        "tail_contrast": tail_contrast,
        "ids": ids,
        "dropped_shares": dropped_shares,
        "operating_characteristics": dict(OPERATING_CHARACTERISTICS),
        "n_cases": len(records),
        "n_incomplete": sum(
            1 for record in records if any(_cell(record, arm) is None for arm in ARM_KEYS)
        ),
    }
