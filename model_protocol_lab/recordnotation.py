r"""Record-notation (cycle 221): the descriptive probe that
renders the same generated data table in two frozen notations
-- prose sentences and canonical JSON -- inside the user
message, and reports how often the two notations' byte-exact
verdicts discord, per candidate.

A prompt-assembly owner deciding whether to switch the frozen
notation their prompt assembly uses for generated data tables
-- prose sentences or canonical JSON -- uses this, per
candidate and template family, to learn on how many of 24
paired generated tables the two notations' byte-exact verdicts
discord, in which direction, and whether the discordance is
exact-only, boundary-only, or mixed, before making that switch
in production prompts.

The instrument (docs/PAYLOADSERIALIZATION_DESIGN.md Part I, as
amended by the fresh review's binding amendments A-H): 24
generated cases per candidate, stage-local, never entering the
user's search/held-out Split. One table per case; the shape is
a frozen function of the case index (records 4 + (i mod 3);
fields {code, label, zone} on even indices, plus mark on odd).
Every value in every field matches ^[a-z]+-[0-9]{4}$ with its
word drawn from a frozen 24-word list and its number in
[1000, 9999] -- a code space of 216,000 -- and all values in a
table are pairwise distinct, so no value contains a space,
comma, period, quote, backslash, control, or non-ASCII
character, JSON escaping is unreachable over the legal
population, and the prose arm is injective over it: the
exhibited inverse (parse_prose) satisfies parse(render(t)) ==
t, executed over the generated panel and random legal tables
in the tests. The generator is a total function of (seed,
case index) with its RNG call sequence frozen verbatim
(amendment A): rng = random.Random(seed * 1000 + i); per value
draw, WORDS[rng.randrange(24)] first and
rng.randrange(1000, 10000) second; duplicate draws are
consumed and discarded; the table fills row-major, fields in
alphabetical order; the queried record index is the next
draw, rng.randrange(n_records). The question names that
record's zone value; the expected answer is its code value.

Exactly two headline arms: ``prose`` (one sentence per record,
fields in sorted order; see render_prose) and
``canonical_json`` (the tree's canonical JSON of the list of
record objects, in record sequence order -- the sequence, not
the multiset, is the held constant). Held constant across
arms: the record sequence, all values, the key and answer
fields, alphabetical field order, the question (byte-identical,
after the table in both arms), the output contract, greedy
decoding, the seed, and the template family per candidate. A
third arm, ``closed_book`` (question and contract only), is
the retained screen: a case a candidate passes closed-book is
excluded from that candidate's live populations, at a stated
cost of 576 of the 1,728 generations at the legal maximum; a
closed-book pass requires guessing one value of 216,000
(executed probability 1/216000 per case), so expected
exclusions are 0.000111 per candidate and 0.0027 per
24-candidate panel, and the arm's pass count is printed as a
leak readout. The headline readouts are also printed over the
unscreened paired set as a sensitivity, so the screen's effect
on any number is visible.

Scoring is the tree's own: a cell's verdict is
``core.evaluate``'s ``exact_match`` conjoined with
``boundary_found``, on a Fixture with the same id, group, and
expected at every arm, so request seeds are identical by
construction (``request_seed_for`` is id-derived). A stage
cell is recorded only from an aggregate-eligible report, whose
invariants force every recorded observation's status to
"completed", so digest equality conjoined with
``boundary_found`` recomputes the verdict exactly (the
aggregate-eligible-only recording invariant, amendment D).
Every completed cell decomposes as pass / boundary_only /
other_failure; every discordant pair decomposes by the
negation construction into exact_only / boundary_only /
mixed, a total partition. The two section 4.4 diagnostics --
the echo rate (the share of an arm's completed verdict
failures whose answer contains the expected value as a
substring) and the parsed-value rate (the share whose answer
parses via json.loads as an object whose value at the answer
key equals the expected value) -- are independent diagnostics:
neither is a subset of the other, because JSON string escapes
break the subset relation (an answer carrying the escape
\u002d for the value's hyphen parses to the expected value
while the literal value never appears in the text --
amendment C). Regrade sensitivities recompute pair discordance
with each answer replaced by scorerlens' R1 normalisation
and, separately, by its fence strip, the boundary conjunct
unchanged.

There is NO classification and no winner readout in v1, by
executed arithmetic, not by preference: at N = 24 every
count-based falsification rule is a false-falsification
machine under the executed null families (the canonical-form
transplant fires 0.8897 on a competent independent null, with
family-wise expectation 21.35 false labels per 24-candidate
panel; Rule A fires 0.6500), and every family-wise-controlled
rule is near-inert at the gaps that matter (the Bonferroni
sign rule's power is 0.0542 at a 25-point gap with marginals
.85/.60). The larger panel that would license a directional
claim (N = 96: Bonferroni false-fire 0.0008, power 0.7452 at
marginals .85/.60 and 0.8329 at the .90 base, 6,912
generations) buys a claim this instrument's utility does not
consume. Every power figure prints its marginals (amendment
E): gap alone does not determine power -- at the competent
base .90 a 25-point gap gives McNemar power 0.4000, not the
0.3547 of the .85/.60 column. A winner readout would be
undefined or abstention in the large majority of runs (a
candidate passes an arm only by passing every completed case:
0.95^24 = 0.292, 0.90^24 = 0.080, 0.80^24 = 0.005), so per-arm
accuracies are the selection-adjacent readout (the negation
precedent). The report prints the readout's resolution beside
its counts (amendment F): one paired case is 100 / n_paired
points of discordance rate, 4.2 points at the full panel.

Token-ID divergence is reported, not gated (the negation
section 3.4 disposition): the headline population is the live
both-completed pairs minus same-ids anomalies;
``n_ids_identical``, the per-case ``id_length_delta`` list
(canonical-JSON id length minus prose id length), the maximum
absolute delta, and ``n_same_ids_verdict_flips`` are printed
per candidate. A same-ids verdict flip is a determinism
finding, never notation fragility; a run landing 100 percent
ids-identical reads "instrument inapplicable to this
tokenizer", never "the model is robust to notation".

The answer-embedding audit's guard set is stated exactly
(amendment B): no field name, question-template token, or
contract token shares a 4-character prefix with, or is a
prefix of, any value word (executed over the frozen lists: 0
violations). The instantiated question is a different matter
and is not hidden: it embeds the key value, whose word shares
its own 4-character prefix in 24 of 24 possible key words, and
the answer's word equals the key's word with analytic
probability 8999/215999 (about 4.17 percent of cases). The
phenomenon is arm-symmetric -- the question is byte-identical
across arms -- so it cannot manufacture discordance; no redraw
rule is applied, because a redraw would change the population.

Budgets (amendment G, measured on this record shape): a
synthetic worst-case record with both headline answer texts
stressed to 2,048 characters serialises to 5,507 bytes, so
576 records are 3.03 MiB against the house 12 MiB cap -- a
3.96x headroom for ASCII text, not the 4.0x the candidate text
printed; and 2,048 characters is exactly the tree's 64-token
character bound (fieldorder), not far beyond it. Stressed
with 2,048 non-ASCII characters per headline answer text the
same record measures 25,987 bytes and the panel 14.28 MiB,
over the cap: such a panel fails closed with
``record_notation_record_budget_exceeded``, which the stage
enforces on the serialised records. A typical record
(bare-code answers) measures 0.79 MiB per panel.

Vocabulary is the tree's frozen fail-closed vocabulary:
stage details {cancelled, record_notation_record_budget_exceeded,
record_notation_screen_failed}; cell details {"",
arm_context_exceeded, arm_evaluation_failed}; whole-stage
failure {"available": False, "detail":
"record_notation_unavailable"}. No classification labels, no
verdict words beyond these. Records carry digests, with one
sanctioned exception: headline cells carry ``answer_text``
bound by ``answer_digest`` (the cycle-190 pattern), because
the section 4.4 diagnostics are functions of the answer text
and uncomputable from digests; reports render counts and
rates only, and answer text never reaches a report. Screen
cells carry no answer text, and no text-derived quantity is
reported for the screen arm. Counts: 24 cases x 3 arms x 24
candidates = 1,728 generations at the legal maximum (1,152
headline + 576 screen); 576 ground records; the house 12 MiB
records cap. Discordance is never pooled across candidates or
template families. Informational only: the stage never
affects selection and never fails the run.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

__all__ = [
    "ANSWER_FIELD",
    "ARM_KEYS",
    "CASE_COUNT",
    "CELL_DETAILS",
    "CODE_SPACE",
    "CONTRACT",
    "EMBEDDING_AUDIT",
    "FIELDS_3",
    "FIELDS_4",
    "GENERATION_BUDGET",
    "GENERATOR_SPEC",
    "GROUP",
    "HEADLINE_ARMS",
    "HEADLINE_GENERATIONS",
    "KEY_FIELD",
    "MAX_CANDIDATES",
    "MAX_GROUND_RECORDS",
    "MAX_RECORDS_BYTES",
    "OPERATING_CHARACTERISTICS",
    "PROTOCOL_VERSION",
    "QUESTION_TEMPLATE",
    "SCREEN_ARM",
    "SCREEN_GENERATIONS",
    "STAGE_DETAILS",
    "WORDS",
    "aggregate_candidate",
    "arm_content",
    "audit_answer_embedding",
    "case_material_sha256",
    "decompose_cell",
    "decompose_pair",
    "digest_ids",
    "digest_text",
    "generate_case",
    "generate_cases",
    "ground_sha256",
    "parse_prose",
    "protocol_fingerprint",
    "protocol_object",
    "records_bytes",
    "render_json",
    "render_prose",
    "table_shape",
    "validate_case",
]

PROTOCOL_VERSION = 1

GROUP = "record"

CASE_COUNT = 24
ANSWER_FIELD = "code"
KEY_FIELD = "zone"
FIELDS_3: tuple[str, ...] = ("code", "label", "zone")
FIELDS_4: tuple[str, ...] = ("code", "label", "mark", "zone")

#: The frozen 24-word list every value's word part is drawn
#: from (candidate section 2.2).
WORDS: tuple[str, ...] = (
    "amber",
    "birch",
    "cedar",
    "copper",
    "coral",
    "falcon",
    "garnet",
    "harbor",
    "indigo",
    "juniper",
    "lagoon",
    "magnet",
    "marble",
    "nickel",
    "onyx",
    "pepper",
    "quartz",
    "saddle",
    "saffron",
    "summit",
    "tundra",
    "velvet",
    "walnut",
    "willow",
)

#: The code space: 24 words x 9000 numbers.
CODE_SPACE = len(WORDS) * 9000

QUESTION_TEMPLATE = "What is the code of the record whose zone is {key}?"
CONTRACT = "Answer with ONLY the value."

HEADLINE_ARMS: tuple[str, ...] = ("prose", "canonical_json")
SCREEN_ARM = "closed_book"
ARM_KEYS: tuple[str, ...] = (*HEADLINE_ARMS, SCREEN_ARM)

#: The frozen RNG call sequence (amendment A), stated as text
#: so the protocol fingerprint binds the seed-to-table mapping
#: the validator's regeneration rests on, not just the seed.
GENERATOR_SPEC = (
    "rng = random.Random(seed * 1000 + i); per value draw, in order: "
    "word = WORDS[rng.randrange(24)], then number = rng.randrange(1000, 10000); "
    "a duplicate value is consumed and discarded; the table fills row-major, "
    "fields in alphabetical order; the queried record index is the next draw, "
    "rng.randrange(n_records), after the table is complete"
)

MAX_CANDIDATES = 24
MAX_GROUND_RECORDS = MAX_CANDIDATES * CASE_COUNT
GENERATION_BUDGET = len(ARM_KEYS) * MAX_GROUND_RECORDS
HEADLINE_GENERATIONS = len(HEADLINE_ARMS) * MAX_GROUND_RECORDS
SCREEN_GENERATIONS = MAX_GROUND_RECORDS
#: The house 12 MiB records bound, for uniformity with the
#: sibling probes.
MAX_RECORDS_BYTES = 12 * 1024 * 1024

CELL_DETAILS: tuple[str, ...] = ("", "arm_context_exceeded", "arm_evaluation_failed")

STAGE_DETAILS: tuple[str, ...] = (
    "cancelled",
    "record_notation_record_budget_exceeded",
    "record_notation_screen_failed",
)

#: Decomposition of a completed cell (candidate section 4.2).
CELL_CLASSES: tuple[str, ...] = ("pass", "boundary_only", "other_failure")

#: Decomposition of a discordant pair (candidate section 4.3,
#: the negation construction).
PAIR_CLASSES: tuple[str, ...] = ("exact_only", "boundary_only", "mixed")

_VALUE_RE = re.compile(r"^[a-z]+-[0-9]{4}$")

#: Operating characteristics of the classification rules this
#: instrument REJECTS, by exact multinomial enumeration over
#: per-case outcomes (amendment E: every power figure carries
#: its marginals; amendment A's census discipline applied to
#: a descriptive instrument -- the numbers are the reason no
#: classification ships). Printed with the report section;
#: never re-derived at run time.
OPERATING_CHARACTERISTICS: dict[str, Any] = {
    "transplant_fire_competent_independent": 0.8897,
    "transplant_fire_competent_paired_phi05": 0.5088,
    "transplant_fire_delta_005": 0.2276,
    "transplant_familywise_expected": 21.35,
    "transplant_familywise_any": 1.0,
    "rule_a_fire_competent_independent": 0.65,
    "rule_a_fire_competent_paired_phi05": 0.1648,
    "rule_a_fire_delta_005": 0.0298,
    "rule_a_familywise_expected": 0.71,
    "rule_a_familywise_any": 0.516,
    "mcnemar_fire_competent_independent": 0.0061,
    "mcnemar_fire_competent_paired_phi05": 0.0005,
    "mcnemar_power_gap25_marginals_85_60": 0.3547,
    "mcnemar_power_gap20_marginals_90_70": 0.2598,
    "mcnemar_power_gap25_base_90": 0.4,
    "sign_bonferroni_power_gap25_marginals_85_60": 0.0542,
    "sign_bonferroni_power_gap20_marginals_90_70": 0.0242,
    "panel96_bonferroni_false_fire": 0.0008,
    "panel96_bonferroni_power_gap25_marginals_85_60": 0.7452,
    "panel96_bonferroni_power_gap25_base_90": 0.8329,
    "panel96_generations": 6912,
    # skipcq: FLK-E501 - ruff's E501 exemptions cover this frozen, test-pinned literal
    "winner_pass_probability_95": 0.292,  # nosec B105 -- operating-characteristic probability in the frozen payload schema, not a credential
    # skipcq: FLK-E501 - ruff's E501 exemptions cover this frozen, test-pinned literal
    "winner_pass_probability_90": 0.08,  # nosec B105 -- operating-characteristic probability in the frozen payload schema, not a credential
    # skipcq: FLK-E501 - ruff's E501 exemptions cover this frozen, test-pinned literal
    "winner_pass_probability_80": 0.005,  # nosec B105 -- operating-characteristic probability in the frozen payload schema, not a credential
    "screen_guess_probability": 1 / 216000,
    "screen_expected_exclusions_per_candidate": 24 / 216000,
    "screen_expected_exclusions_per_panel": 576 / 216000,
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


def table_shape(index: int) -> tuple[int, tuple[str, ...]]:
    """The frozen shape of case ``index``: records
    4 + (index mod 3); three fields on even indices, four on
    odd. Over index mod 6 the shapes are (4,3), (5,4), (6,3),
    (4,4), (5,3), (6,4) -- every combination exactly once.
    """
    return 4 + (index % 3), (FIELDS_4 if index % 2 else FIELDS_3)


def generate_case(seed: int, index: int) -> dict[str, Any]:
    """The frozen generated case ``index`` (0-23) at ``seed``.

    The RNG call sequence is frozen (amendment A; see
    GENERATOR_SPEC): per value draw, the word first
    (``WORDS[rng.randrange(24)]``), then the number
    (``rng.randrange(1000, 10000)``); duplicates are consumed
    and discarded; the table fills row-major over the
    alphabetical fields; the queried record index is the next
    draw after the table is complete.
    """
    n_records, fields = table_shape(index)
    rng = random.Random(seed * 1000 + index)  # noqa: S311 # nosec B311
    values: list[str] = []
    seen: set[str] = set()
    while len(values) < n_records * len(fields):
        word = WORDS[rng.randrange(len(WORDS))]
        number = rng.randrange(1000, 10000)
        value = f"{word}-{number}"
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    table = [
        {field: values[r * len(fields) + j] for j, field in enumerate(fields)}
        for r in range(n_records)
    ]
    query_index = rng.randrange(n_records)
    return {
        "fixture_id": f"record-{index:02d}",
        "table": table,
        "query_index": query_index,
        "key": table[query_index][KEY_FIELD],
        "expected": table[query_index][ANSWER_FIELD],
    }


def generate_cases(seed: int) -> tuple[dict[str, Any], ...]:
    """The full 24-case panel at ``seed``, in case order."""
    return tuple(generate_case(seed, index) for index in range(CASE_COUNT))


def render_json(table: list[dict[str, str]]) -> str:
    """The canonical-JSON arm: the tree's canonical JSON
    (``structured._canonical_json``'s exact parameters) applied
    to the list of record objects in record sequence order.
    ``sort_keys`` orders keys inside each record object; it
    never reorders the list.
    """
    return json.dumps(table, sort_keys=True, separators=(", ", ": "), allow_nan=False)


def render_prose(table: list[dict[str, str]]) -> str:
    """The prose arm: one sentence per record, fields in
    sorted-key order -- 1 field ``The <f> is <v>.``; 2 fields
    ``The <f1> is <v1> and the <f2> is <v2>.``; k >= 3 fields
    comma-joined with ``, and the`` before the last. Sentences
    join with a single space. The function is total over
    record sequences and injective over the legal population
    (see parse_prose).
    """
    sentences: list[str] = []
    for record in table:
        parts = [f"the {field} is {record[field]}" for field in sorted(record)]
        if len(parts) == 1:
            sentence = f"The {parts[0][4:]}."
        elif len(parts) == 2:
            sentence = f"The {parts[0][4:]} and {parts[1]}."
        else:
            sentence = "The " + ", ".join([parts[0][4:], *parts[1:-1]]) + f", and {parts[-1]}."
        sentences.append(sentence)
    return " ".join(sentences)


def parse_prose(text: str) -> list[dict[str, str]]:
    """The exhibited inverse of render_prose (candidate
    section 3.2): split the rendering into sentences at the
    sentence boundaries; within a sentence, normalise the
    join forms to the separator ``, the `` and split; each
    segment has the unique form ``the <field> is <value>``.
    Segmentation is unique because every separator contains a
    space, a comma, a period, or an uppercase letter, and no
    legal value or field name contains any of those
    characters. Raises ValueError on text render_prose could
    not have produced.
    """
    if not text.endswith("."):
        raise ValueError("prose rendering must end with a period")
    table: list[dict[str, str]] = []
    for sentence in text[:-1].split(". "):
        if not sentence.startswith("The "):
            raise ValueError("prose sentence must open with 'The '")
        body = sentence[4:]
        body = body.replace(", and the ", ", the ").replace(" and the ", ", the ")
        record: dict[str, str] = {}
        for segment in body.split(", the "):
            field, sep, value = segment.partition(" is ")
            if not sep or not field or not value:
                raise ValueError("prose segment must have the form '<field> is <value>'")
            record[field] = value
        table.append(record)
    return table


def arm_content(case: dict[str, Any], arm: str) -> str:
    """The full user-message content of one arm: the rendering,
    a blank line, the question, a space, and the contract;
    the closed-book arm is the question and contract only.
    """
    question = QUESTION_TEMPLATE.format(key=case["key"])
    if arm == SCREEN_ARM:
        return f"{question} {CONTRACT}"
    if arm == "prose":
        rendering = render_prose(case["table"])
    elif arm == "canonical_json":
        rendering = render_json(case["table"])
    else:
        raise ValueError(f"unknown record-notation arm: {arm}")
    return f"{rendering}\n\n{question} {CONTRACT}"


def audit_answer_embedding() -> dict[str, Any]:
    """The answer-embedding audit over the frozen lists
    (amendment B, its guard set stated exactly).

    Template side: no field name, question-template token, or
    contract token shares a 4-character prefix with, or is a
    prefix of, any value word -- executed here over the frozen
    lists (0 violations, asserted by the stage). Instantiated
    side, printed and never hidden: the instantiated question
    embeds the key value, whose word shares its own 4-character
    prefix for every possible key word (24 of 24), and the
    answer's word equals the key's word with analytic
    probability 8999/215999 -- the answer value is uniform over
    the 215,999 values distinct from the key's, 8,999 of which
    share the key's word. Arm-symmetric (the question is
    byte-identical across arms); no redraw rule is applied.
    """
    tokens = set(FIELDS_4)
    tokens.update(re.findall(r"[a-z]+", QUESTION_TEMPLATE.replace("{key}", " ").lower()))
    tokens.update(re.findall(r"[a-z]+", CONTRACT.lower()))
    violations = 0
    for token in tokens:
        for word in WORDS:
            if token[:4] == word[:4] or word.startswith(token) or token.startswith(word):
                violations += 1
    return {
        "template_violations": violations,
        "key_word_prefix_cases": len(WORDS),
        "key_word_prefix_of": len(WORDS),
        "answer_key_same_word_rate": 8999 / 215999,
    }


#: The audit block, computed once over the frozen lists and
#: bound into the protocol (so the validator recomputes it
#: from its own literal copies, not from trust).
EMBEDDING_AUDIT: dict[str, Any] = audit_answer_embedding()


def case_material_sha256() -> str:
    """Digest over every frozen text and list the generator
    draws from, so the protocol binds the population's
    definition -- including the frozen RNG call sequence --
    not just its seed.
    """
    material = {
        "words": list(WORDS),
        "fields_3": list(FIELDS_3),
        "fields_4": list(FIELDS_4),
        "answer_field": ANSWER_FIELD,
        "key_field": KEY_FIELD,
        "question_template": QUESTION_TEMPLATE,
        "contract": CONTRACT,
        "generator_spec": GENERATOR_SPEC,
        "case_count": CASE_COUNT,
        "arm_keys": list(ARM_KEYS),
        "embedding_audit": EMBEDDING_AUDIT,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ground_sha256(seed: int) -> str:
    """Digest over the seed's generated ground population
    (canonical rows binding fixture id, the table in record
    sequence, the queried index, the key, and the expected
    answer of every case).
    """
    rows = [
        [
            case["fixture_id"],
            [[[field, record[field]] for field in sorted(record)] for record in case["table"]],
            case["query_index"],
            case["key"],
            case["expected"],
        ]
        for case in generate_cases(seed)
    ]
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def protocol_object(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The frozen instrument description a payload is checked
    against: arms, population, budgets, the case-material and
    ground digests, the embedding audit, and the seed /
    backend identity that generated the records.
    """
    return {
        "version": PROTOCOL_VERSION,
        "arm_keys": list(ARM_KEYS),
        "headline_arms": list(HEADLINE_ARMS),
        "screen_arm": SCREEN_ARM,
        "answer_field": ANSWER_FIELD,
        "key_field": KEY_FIELD,
        "case_count": CASE_COUNT,
        "code_space": CODE_SPACE,
        "case_material_sha256": case_material_sha256(),
        "ground_sha256": ground_sha256(seed),
        "embedding_audit": dict(EMBEDDING_AUDIT),
        "generation_budget": GENERATION_BUDGET,
        "headline_generations": HEADLINE_GENERATIONS,
        "screen_generations": SCREEN_GENERATIONS,
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


def records_bytes(records_by_candidate: dict[str, list[dict[str, Any]]]) -> int:
    """Canonical JSON size of the ground records, the quantity
    the payload byte cap is enforced against (the numform
    pattern).
    """
    return len(
        json.dumps(records_by_candidate, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    )


def validate_case(case: dict[str, Any], seed: int) -> bool:
    """The pre-generation invariant battery over one case
    (candidate sections 2 and 3), executed by the stage before
    any generation and re-executed on the frozen material in
    the tests: the case equals its regeneration from the seed
    (the frozen RNG sequence, amendment A); the shape schedule
    holds; every value conforms to the alphabet with its word
    in the frozen list and its number in [1000, 9999]; values
    are pairwise distinct; the key (zone) is unique; the
    expected answer appears in exactly one row; the two arm
    renderings are byte-divergent; the JSON rendering contains
    no backslash (escaping unreachable); parse_prose inverts
    render_prose and json.loads inverts render_json on this
    table; the arm contents carry the question and contract
    exactly; the embedding audit's template side is clean.
    """
    try:
        fixture_id = str(case["fixture_id"])
        index = int(fixture_id.rsplit("-", 1)[1])
        if not 0 <= index < CASE_COUNT or fixture_id != f"record-{index:02d}":
            return False
        if generate_case(seed, index) != case:
            return False
        n_records, fields = table_shape(index)
        table = case["table"]
        if len(table) != n_records:
            return False
        values: list[str] = []
        for record in table:
            if set(record) != set(fields):
                return False
            values.extend(str(record[field]) for field in fields)
        if len(set(values)) != len(values):
            return False
        for value in values:
            if not _VALUE_RE.match(value):
                return False
            word, _, number = value.partition("-")
            if word not in WORDS or not 1000 <= int(number) <= 9999:
                return False
        if len({str(record[KEY_FIELD]) for record in table}) != n_records:
            return False
        if sum(1 for record in table if record[ANSWER_FIELD] == case["expected"]) != 1:
            return False
        query = int(case["query_index"])
        if not 0 <= query < n_records:
            return False
        if table[query][KEY_FIELD] != case["key"] or table[query][ANSWER_FIELD] != case["expected"]:
            return False
        prose = render_prose(table)
        canonical = render_json(table)
        if prose == canonical or "\\" in canonical:
            return False
        if parse_prose(prose) != table or json.loads(canonical) != table:
            return False
        question = QUESTION_TEMPLATE.format(key=case["key"])
        if arm_content(case, "prose") != f"{prose}\n\n{question} {CONTRACT}":
            return False
        if arm_content(case, SCREEN_ARM) != f"{question} {CONTRACT}":
            return False
        return bool(EMBEDDING_AUDIT["template_violations"] == 0)
    except Exception:  # noqa: BLE001 -- a malformed case fails the screen closed
        return False


def decompose_cell(cell: dict[str, Any], expected_digest: str) -> str:
    """The decomposition class of one completed cell
    (candidate section 4.2): ``pass`` (the verdict passed),
    then ``boundary_only`` (the answer digest equals the
    expected digest; the boundary event alone failed -- a
    boundary effect, never a notation effect), else
    ``other_failure`` (counted, and characterised only through
    the section 4.4 diagnostics; no class word is coined).
    """
    if cell.get("verdict") is True:
        return "pass"
    digest = cell.get("answer_digest")
    if digest is not None and digest == expected_digest:
        return "boundary_only"
    return "other_failure"


def decompose_pair(prose_cell: dict[str, Any], json_cell: dict[str, Any]) -> str:
    """The decomposition class of one discordant pair
    (candidate section 4.3, the negation construction, which
    is total): verdicts differ and answer digests are equal
    -> ``boundary_only``; verdicts differ, digests differ, and
    both cells found their boundary -> ``exact_only``;
    verdicts differ otherwise -> ``mixed`` (reported, never
    attributed). Callers invoke this only on discordant
    pairs.
    """
    if prose_cell.get("answer_digest") == json_cell.get("answer_digest"):
        return "boundary_only"
    if prose_cell.get("boundary_found") and json_cell.get("boundary_found"):
        return "exact_only"
    return "mixed"


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
    """Return the verdict."""
    cell = _cell(record, key)
    return bool(cell is not None and cell["verdict"])


def _screened(record: dict[str, Any]) -> bool:
    """The closed-book screen: the candidate passed the case
    with no table at all, so the notation cannot move it.
    """
    cell = _cell(record, SCREEN_ARM)
    return bool(cell is not None and cell["verdict"])


def _ids_identical(record: dict[str, Any]) -> bool:
    """The ids-identical derivation, from the recorded per-arm
    id digests alone: the two headline renders tokenise
    identically, so the pair is a rendering anomaly, not a
    measurement.
    """
    digests = record.get("id_digests")
    if not isinstance(digests, dict):
        return False
    return digests.get("prose") is not None and digests.get("prose") == digests.get(
        "canonical_json"
    )


def _parsed_hit(answer_text: str, expected: str) -> bool:
    """The parsed-value diagnostic (candidate section 4.4):
    the answer parses via json.loads as an object whose value
    at the answer-field key equals the expected value. JSON
    string escapes are honoured by the parser, so a parsed hit
    need not be a substring hit (amendment C).
    """
    try:
        parsed = json.loads(answer_text)
    except ValueError:
        return False
    return type(parsed) is dict and parsed.get(ANSWER_FIELD) == expected


def _regrade_verdict(cell: dict[str, Any], expected: str, transform: Any) -> bool:
    """One cell's verdict under a scorer-lens regrade: the
    answer text is replaced by its transform, the boundary
    conjunct unchanged (candidate section 4.4).
    """
    text = cell.get("answer_text")
    if type(text) is not str:
        return False
    return transform(text) == expected and bool(cell.get("boundary_found"))


# skipcq: PY-R1000 - branches are frozen payload-contract cases pinned by tests
def aggregate_candidate(
    records: list[dict[str, Any]], cases: tuple[dict[str, Any], ...]
) -> dict[str, Any]:
    """Per-candidate aggregates over its ground records.

    ``cases`` is the regenerated panel (``generate_cases`` of
    the run seed): the section 4.4 diagnostics are functions
    of the expected value and the carried answer texts, so the
    aggregate takes the cases beside the records, matched by
    fixture id. Readouts, in the report's printed order: the
    audit counts; per-arm completion, pass rate, attrition,
    decomposition, and (headline arms) the echo and
    parsed-value rates; the headline population (live cases --
    not closed-book screened -- with both headline cells
    completed, minus same-ids anomalies) with its discordance
    count, direction, decomposition, per-arm paired rates, and
    resolution (amendment F); the id report; and the
    sensitivities (discordance over the unscreened paired set;
    R1 and fence regrade discordance over the headline
    population). Discordance is never pooled across candidates
    -- this function sees one candidate's records only.
    """
    from .scorerlens import _fence_strip, _norm_r1

    expected_by_fixture = {str(case["fixture_id"]): str(case["expected"]) for case in cases}
    arms: dict[str, Any] = {}
    for key in ARM_KEYS:
        n_completed = 0
        n_pass = 0
        n_context = 0
        n_failed = 0
        decomposition = dict.fromkeys(CELL_CLASSES, 0)
        n_failures = 0
        echo_hits = 0
        parsed_hits = 0
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
            if key in HEADLINE_ARMS and not cell["verdict"]:
                n_failures += 1
                text = cell.get("answer_text")
                expected = expected_by_fixture.get(str(record.get("fixture", "")), "")
                if type(text) is str:
                    if expected in text:
                        echo_hits += 1
                    if _parsed_hit(text, expected):
                        parsed_hits += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": _rate(n_pass, n_completed),
            "n_context_exceeded": n_context,
            "n_evaluation_failed": n_failed,
            "decomposition": decomposition,
            "n_failures": n_failures,
            "echo_rate": _rate(echo_hits, n_failures) if key in HEADLINE_ARMS else None,
            "parsed_rate": _rate(parsed_hits, n_failures) if key in HEADLINE_ARMS else None,
        }
    both = [
        record
        for record in records
        if _cell(record, "prose") is not None and _cell(record, "canonical_json") is not None
    ]
    unscreened_pairs = [record for record in both if not _ids_identical(record)]
    live_pairs = [record for record in unscreened_pairs if not _screened(record)]

    def _discordance(pairs: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the discordance."""
        prose_only = 0
        json_only = 0
        decomposition = dict.fromkeys(PAIR_CLASSES, 0)
        for record in pairs:
            prose_cell = _cell(record, "prose")
            json_cell = _cell(record, "canonical_json")
            if prose_cell is None or json_cell is None:
                continue
            if bool(prose_cell["verdict"]) == bool(json_cell["verdict"]):
                continue
            if prose_cell["verdict"]:
                prose_only += 1
            else:
                json_only += 1
            decomposition[decompose_pair(prose_cell, json_cell)] += 1
        return {
            "n_paired": len(pairs),
            "discordant": prose_only + json_only,
            "n_prose_only": prose_only,
            "n_canonical_json_only": json_only,
            "decomposition": decomposition,
        }

    headline_counts = _discordance(live_pairs)
    headline = {
        **headline_counts,
        "prose_rate": _rate(
            sum(1 for record in live_pairs if _verdict(record, "prose")), len(live_pairs)
        ),
        "canonical_json_rate": _rate(
            sum(1 for record in live_pairs if _verdict(record, "canonical_json")),
            len(live_pairs),
        ),
        "resolution_points": (100.0 / len(live_pairs)) if live_pairs else None,
    }
    deltas: list[int] = []
    n_same_ids_flips = 0
    for record in records:
        tokens = record.get("arm_tokens")
        tokens = tokens if isinstance(tokens, dict) else {}
        prose_tokens = tokens.get("prose")
        json_tokens = tokens.get("canonical_json")
        if type(prose_tokens) is int and type(json_tokens) is int:
            deltas.append(json_tokens - prose_tokens)
        else:
            deltas.append(0)
        if _ids_identical(record):
            prose_cell = _cell(record, "prose")
            json_cell = _cell(record, "canonical_json")
            if (
                prose_cell is not None
                and json_cell is not None
                and prose_cell["verdict"] != json_cell["verdict"]
            ):
                n_same_ids_flips += 1
    ids = {
        "n_ids_identical": sum(1 for record in records if _ids_identical(record)),
        "id_length_delta": deltas,
        "max_abs_id_delta": max((abs(delta) for delta in deltas), default=0),
        "n_same_ids_verdict_flips": n_same_ids_flips,
    }
    regrade: dict[str, int] = {}
    for label, transform in (("r1", _norm_r1), ("fence", _fence_strip)):
        count = 0
        for record in live_pairs:
            prose_cell = _cell(record, "prose")
            json_cell = _cell(record, "canonical_json")
            if prose_cell is None or json_cell is None:
                continue
            expected = expected_by_fixture.get(str(record.get("fixture", "")), "")
            if _regrade_verdict(prose_cell, expected, transform) != _regrade_verdict(
                json_cell, expected, transform
            ):
                count += 1
        regrade[label] = count
    sensitivities = {
        "unscreened": _discordance(unscreened_pairs),
        "regrade_r1_discordant": regrade["r1"],
        "regrade_fence_discordant": regrade["fence"],
    }
    return {
        "arms": arms,
        "audit": {
            "n_screened": sum(1 for record in records if _screened(record)),
            "n_closed_book_incomplete": sum(
                1 for record in records if _cell(record, SCREEN_ARM) is None
            ),
        },
        "headline": headline,
        "ids": ids,
        "sensitivities": sensitivities,
        "operating_characteristics": dict(OPERATING_CHARACTERISTICS),
        "n_cases": len(records),
        "n_incomplete": sum(
            1 for record in records if any(_cell(record, arm) is None for arm in ARM_KEYS)
        ),
    }
