r"""Invisible-character insertion (cycle 223): the probe that
inserts one frozen invisible code point at two frozen sites in
a generated fact sentence and classifies, per candidate and
per arm, whether the insertion moved the model's answers to
wrong values.

A team piping web/RAG text into a local checkpoint decides,
Monday morning, whether to add a pre-ingest sanitizer stripping
invisible code points, using this probe -- per candidate
(template family) and per invisible arm -- to learn whether a
single invisible insertion changes the model's answers on
ordinary generated tasks at all. The claim is scoped to that
first half of the sanitizer question: the arm set has no strip
arm, so sanitizer cost on clean text is not measured (on this
panel's clean text a sanitizer removing exactly the five frozen
code points is the identity function, by construction and by
the generator's assert battery); a strip-cost instrument over
naturally contaminated text is a separate candidate.

The instrument (docs/INVISIBLECHARS_DESIGN.md, the wave-12
candidate 1 design as amended by the fresh review's binding
amendments A1-A10): 24 generated cases per candidate,
stage-local, never entering the user's search/held-out Split.
Case i's topic is TOPICS[i] from a frozen 24-word ASCII list;
its value is ``word-number`` with the word from the canonical-
form 24-word ASCII pool and the number in [1000, 9999], values
pairwise distinct across the panel by redraw. Content is
``{neutral1} {fact} {neutral2} {question}`` with the fact
``The {topic} inspection code is {value}.``, the question
``What is the {topic} inspection code?``, and two distinct
neutral sentences from density's frozen pool. The draws are
SHA-256 derivations with the conversion pinned exactly
(amendment A1): material
``f"{seed}|invisiblechars|{index}|{slot}|{counter}"``, integer
``int.from_bytes(sha256(material).digest()[:8], "big")``; slots
0 = value word, 1 = number (a collision redraws both at
counter+1, with a frozen give-up bound of 1,000 consecutive
redraws whose exhaustion fails the stage closed), 2/3 =
neutral indices (the second advances +1 on collision). Every
expected value is pure ASCII, enforced by the generator
battery, so the byte-exact scorer can never manufacture
discordance out of the value itself.

Seven arms, fully crossed: ``control`` (verbatim) and six
treated arms each inserting exactly one code point at each of
two frozen sites inside the fact sentence only -- after
code-point index 2 of the topic, and after code-point index 2
of the value's word part. The five invisible arms are a frozen
LIST, each member verified against a stated normative
property (Default_Ignorable_Code_Point = Yes in UCD 15.0.0,
the local interpreter's Unicode version; General_Category in
{Cf, Mn}; canonical combining class 0): U+200B ZERO WIDTH
SPACE, U+FE0F VARIATION SELECTOR-16, U+E0001 LANGUAGE TAG,
U+00AD SOFT HYPHEN (the weakest member -- a renderer may show
a hyphen at a line-break opportunity; the frozen sites are
never at a line boundary), U+2060 WORD JOINER. The definition
is the list, never the predicate: the predicate alone admits
U+034F COMBINING GRAPHEME JOINER (Mn, ccc 0, DICP Yes), which
is excluded by list membership, as are the joiners U+200C /
U+200D (real joining semantics), U+FEFF (byte-order-mark
semantics), and the bidi controls (a direction axis, not
invisibility). The sixth treated arm, ``visible_hyphen``
(U+002D HYPHEN-MINUS at the same sites, same dose), is the
matched visible-insertion control: any insertion disrupts
tokenization at some rate, and an invisible arm is evidence
about invisibility only against this arm's behaviour. The
question and both neutrals are byte-identical across arms.

Scoring is the tree's own: a cell's verdict is
``core.evaluate``'s ``exact_match`` conjoined with
``boundary_found``, on a Fixture with the same id, group, and
expected at every arm, so request seeds are identical by
construction. A stage cell is recorded only from an
aggregate-eligible report, whose invariants force every
recorded observation's status to "completed", so digest
equality conjoined with ``boundary_found`` recomputes the
verdict exactly. Every completed cell decomposes by an
ORDERED decision list (amendment A5) -- pass, boundary_only,
wrong_value, echo_invisible, form_only, non_answer -- each
class reached only when every earlier predicate fails, with
``form_only`` additionally requiring the answer to differ
from the expected value; without the ordering the literal
predicates overlap (removal is the identity on a
code-point-free answer). ``wrong_value`` -- the only class
that counts toward the headline -- is scoped to the frozen
value grammar (amendment A7): a value change outside it (an
out-of-pool word, a leading-zero number) decomposes as
``non_answer``, is printed per arm, and never counts toward
the headline. For ``visible_hyphen``, ``form_only`` is empty
by construction: its character is the grammar's separator, so
stripping every hyphen can never yield a value whose
separator is a hyphen (amendment A6); cross-arm comparisons
use only the uniformly-defined classes.

Token-ID divergence is the regime gate, measured at encode
time on the full render id sequence per (case, candidate),
per treated arm against control: ``n_ids_identical``,
``n_ids_divergent``, the per-case ``id_length_delta`` list,
``max_abs_id_delta``, and ``n_same_ids_verdict_flips`` print
per (candidate, arm). The headline population per (candidate,
arm) is the cases completed in both control and arm AND
id-divergent for that arm; ids-identical arms are reported
instrument-inapplicable for that family, never as passes. The
pre-registered regime expectation, stated before any run and
discovered by measuring ids, never asserted per checkpoint:
SentencePiece pipelines normalise before tokenization with
``nmt_nfkc`` by default and UTS #39's internalSkeleton step 2
removes Default_Ignorable code points, so DICP arms are
expected to land ids-identical on normalizing pipelines,
while byte-level BPE without a normalizer passes the same
code points through as distinct bytes.

The headline is a guarded classification, per (candidate,
arm): D = control passes in the population; J = of those, arm
cells decomposed ``wrong_value``; floors MIN_D = 10 and
MIN_J = 6, derived under this instrument's own null (MIN_J
set by the floor search; MIN_D selected from the plateau
[9, 16] over which every printed operating characteristic is
identical, amendment A8). The labels: ``insufficient`` (D <
10, or the guard withholds), ``no_discordance_observed``
(D >= 10, J = 0, guard passes), ``discordant_asymmetric``
(D >= 10, J >= 6, guard passes -- the fire: the section 7
invariance claim is falsified for this arm),
``below_threshold`` (D >= 10, 0 < J < 6, guard passes -- an
observed null at this panel's resolution, not proof of
invariance). Reverse discordants (control fails, arm passes,
among paired cases) are counted as ``n_reverse`` beside every
label and never pooled into J: the population is conditioned
on control passes, so discordance has one direction by
construction. The two-directional canonical rule was rejected
by execution, not preference: at N = 24 the transplanted
floors (paired >= 6, discordant >= 2) fire at 0.8897 on the
competent independent null.

The completion guard closes the channel that defeated the
multi-question line twice. Incompletion -- a generation
producing no boundary-delimited span at all, or a missing
cell -- has its domain pinned (amendment A3): exactly the
cases in the pair's population domain (id-divergent, an
encode-time property decidable whether or not the generation
completed) with at least one missing cell; ids-identical
cases never enter a completion, in either direction. A fire
cannot be manufactured: J and D are absolute counts over
completed cases, and the fire issues on the observed counts.
An acquittal cannot be manufactured: an acquittal label
issues only if it is stable under the worst-case completion
(every incomplete case in the domain completed as
control-pass + arm ``wrong_value``). Consequences, printed
(amendment A4): ``no_discordance_observed`` requires zero
incomplete cases in the domain; ``below_threshold`` survives
only while J + n_incomplete < 6; and a paired-population
floor withholds every label when incomplete cases in the
domain exceed 6 (paired < 18, the smallest paired count with
printed operating characteristics). The guard's worlds,
executed (10,000 trials each, frozen seeds): in World A
(zero effect; incompletion at rate 1.0 on concordant
control-pass arm cells in the panel's first half) the
unguarded fire is 0.0168, the guarded fire 0.0014; in World B
(a true drop with sign-concentrated incompletion removing 60
percent of would-be J cells) the unguarded rule acquits at
0.7285 while the guarded rule's ``below_threshold`` is 0.0133
-- the displaced mass is withholding, not acquittal; in
World C (the same drop, flat incompletion 0.15) the guarded
fire is 0.8932 against the unguarded 0.9408. No sign test is
used anywhere in this instrument (its rule is a count rule
with exact enumerated operating characteristics), so the
completion-monotonicity failure of two-sided sign tests does
not arise.

Operating characteristics, one convention, stated (amendment
A2): the independent column is exact enumeration under
per-case independence with q the arm's marginal in-grammar
wrong-value rate -- a 10-point drop fires 0.2542, the
25-point target effect 0.8159 at base .90 and 0.8761 at base
.85, a 50-point drop 0.9990; the competent null fires 0.0173
independent and 0.0044 under the paired Gaussian copula at
rho 0.5 (exact bivariate computation; paired power at the
target effect 0.6981). The family-wise price prints beside
the licence in every report: over the 120 labelled invisible
comparisons the competent expectation is 2.07 false labels
(paired 0.53); over all 144 printed labels, 2.49. Not
falsified is not robust: a 10-point drop fires only 0.2542 of
the time, and the report prints each candidate's control rate
beside its label so the reader can see which null row applies
(a candidate passing only 80 percent of control cases with
independent arm noise fires spuriously about one time in six).

Budgets: 24 cases x 7 arms x 24 candidates = 4,032
generations at the legal maximum; 576 ground records; the
house 12 MiB records cap, enforced on the serialised records
with the frozen budget detail. Measured on this record shape:
a typical record (short answers) serialises to about 3,189
bytes, 1.75 MiB per panel; stressing every cell's answer text
to 2,048 ASCII characters measures 17,357 bytes per record,
9.53 MiB per panel, inside the cap; the same stress with
2,048 non-ASCII characters per answer measures 89,037 bytes
per record, 48.91 MiB per panel, over the cap -- such a panel
fails closed with
``invisible_chars_record_budget_exceeded``, which the stage
enforces on the serialised records. Records carry digests, with one
sanctioned exception: every cell carries ``answer_text`` bound
by ``answer_digest`` (the cycle-190 pattern), because the
decomposition classes are functions of the answer text and
uncomputable from digests; reports render counts and rates
only, and answer text never reaches a report. Per-arm content
digests are mandatory: the diff validator's mirror regenerates
all seven arm contents from the seed and checks each digest,
so the treatment itself -- not just token counts -- is
committed. Vocabulary is the tree's frozen fail-closed
vocabulary: stage details {cancelled,
invisible_chars_record_budget_exceeded,
invisible_chars_screen_failed}; cell details {"",
arm_context_exceeded, arm_evaluation_failed}; whole-stage
failure {"available": False, "detail":
"invisible_chars_unavailable"}; classification labels
{insufficient, no_discordance_observed,
discordant_asymmetric, below_threshold} and no other verdict
words. Discordance is never pooled across candidates or arms.
Informational only: the stage never affects selection and
never fails the run.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any

__all__ = [
    "ARM_CODEPOINTS",
    "ARM_KEYS",
    "CASE_COUNT",
    "CELL_CLASSES",
    "CELL_DETAILS",
    "GENERATION_BUDGET",
    "GENERATOR_SPEC",
    "GROUP",
    "INVISIBLE_ARMS",
    "LABELS",
    "MAX_CANDIDATES",
    "MAX_GROUND_RECORDS",
    "MAX_INCOMPLETE",
    "MAX_RECORDS_BYTES",
    "MIN_D",
    "MIN_J",
    "NEUTRAL_STATEMENTS",
    "OPERATING_CHARACTERISTICS",
    "PAIR_CLASSES",
    "POPULATION_AUDIT",
    "PROTOCOL_VERSION",
    "REDRAW_LIMIT",
    "STAGE_DETAILS",
    "TOPICS",
    "TREATED_ARMS",
    "WORDS",
    "aggregate_candidate",
    "arm_content",
    "audit_arm_population",
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
    "protocol_fingerprint",
    "protocol_object",
    "question_text",
    "records_bytes",
    "treated_fact",
    "validate_case",
]

PROTOCOL_VERSION = 1

GROUP = "invisible_chars"

CASE_COUNT = 24

#: The frozen 24-word topic list (design section 3): case i's
#: topic is TOPICS[i]. All ASCII, all length >= 5, so the
#: topic insertion site (after code-point index 2) always
#: exists.
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
    "pepper",
    "quartz",
    "birch",
)

#: The value-word pool: the canonical-form 24-word ASCII pool
#: (a literal copy, bound into the case material; a freeze
#: test pins it equal to canonicalform.WORDS as a drift
#: alarm). Minimum length 3 ("elm"/"urn"), so the payload
#: insertion site always exists.
WORDS: tuple[str, ...] = (
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

#: The neutral carrier sentences: density's frozen 24 ASCII
#: statements (a literal copy, bound into the case material;
#: a freeze test pins it equal to density.NEUTRAL_STATEMENTS
#: as a drift alarm).
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

#: The frozen arm list (design section 2): the code point each
#: treated arm inserts at both frozen sites; the control arm
#: inserts nothing. The five invisible members are each
#: Default_Ignorable_Code_Point = Yes in UCD 15.0.0 with
#: General_Category in {Cf, Mn} and combining class 0 -- the
#: predicate is a check on the list's members, never a
#: membership rule (the predicate alone admits U+034F, which
#: the list excludes).
ARM_CODEPOINTS: dict[str, str | None] = {
    "control": None,
    "zwsp": "\u200b",
    "vs16": "\ufe0f",
    "tag": "\U000e0001",
    "softhyphen": "\u00ad",
    "wj": "\u2060",
    "visible_hyphen": "-",
}

ARM_KEYS: tuple[str, ...] = (
    "control",
    "zwsp",
    "vs16",
    "tag",
    "softhyphen",
    "wj",
    "visible_hyphen",
)
TREATED_ARMS: tuple[str, ...] = ARM_KEYS[1:]
INVISIBLE_ARMS: tuple[str, ...] = ("zwsp", "vs16", "tag", "softhyphen", "wj")

#: The frozen floors and the paired-population cap (design
#: section 6, amendments A4 and A8).
MIN_D = 10
MIN_J = 6
MAX_INCOMPLETE = 6

#: The redraw give-up bound (amendment A1): consecutive
#: collision redraws for one case beyond which generation
#: fails and the stage fails closed. Unreachable under the
#: frozen derivation (24 draws from a 216,000-value space);
#: it exists so the generator is provably total.
REDRAW_LIMIT = 1000

#: The frozen draw derivation (amendment A1), stated as text
#: so the protocol fingerprint binds the seed-to-panel mapping
#: the validator's regeneration rests on, not just the seed.
GENERATOR_SPEC = (
    "draw(seed, index, slot, counter) = int.from_bytes("
    "sha256(f'{seed}|invisiblechars|{index}|{slot}|{counter}')"
    ".digest()[:8], 'big'); slot 0: word = WORDS[draw % len(WORDS)]; "
    "slot 1: number = 1000 + draw % 9000; a value colliding with an "
    "earlier case redraws both slots at counter+1, giving up after "
    "1000 consecutive redraws (the stage fails closed); slots 2/3: "
    "neutral indices = draw % len(NEUTRAL_STATEMENTS), the second "
    "advancing +1 on collision"
)

MAX_CANDIDATES = 24
MAX_GROUND_RECORDS = MAX_CANDIDATES * CASE_COUNT
GENERATION_BUDGET = len(ARM_KEYS) * MAX_GROUND_RECORDS
#: The house 12 MiB records bound, for uniformity with the
#: sibling probes.
MAX_RECORDS_BYTES = 12 * 1024 * 1024

CELL_DETAILS: tuple[str, ...] = ("", "arm_context_exceeded", "arm_evaluation_failed")

STAGE_DETAILS: tuple[str, ...] = (
    "cancelled",
    "invisible_chars_record_budget_exceeded",
    "invisible_chars_screen_failed",
)

#: Decomposition of a completed cell (design section 5, the
#: ordered decision list of amendment A5).
CELL_CLASSES: tuple[str, ...] = (
    "pass",
    "boundary_only",
    "wrong_value",
    "echo_invisible",
    "form_only",
    "non_answer",
)

#: Decomposition of a discordant pair (the recordnotation
#: total partition, mirrored).
PAIR_CLASSES: tuple[str, ...] = ("exact_only", "boundary_only", "mixed")

#: The classification vocabulary (design section 6).
LABELS: tuple[str, ...] = (
    "insufficient",
    "no_discordance_observed",
    "discordant_asymmetric",
    "below_threshold",
)

_VALUE_RE = re.compile(r"^[a-z]+-[0-9]{4}$")

#: Operating characteristics of the shipped rule and of the
#: rejected transplant, by exact enumeration under the
#: marginal convention (amendment A2) and the exact bivariate
#: paired computation (design section 6). Printed with the
#: report section; never re-derived at run time. The
#: cycle-223 test module reproduces every figure from its own
#: code (amendment A10).
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
    "familywise_competent_expected_120": 2.07,
    "familywise_competent_any_120": 0.8764,
    "familywise_competent_expected_144": 2.49,
    "familywise_competent_any_144": 0.9187,
    "familywise_paired_expected_120": 0.53,
    "familywise_paired_any_120": 0.4139,
    "familywise_random_expected_120": 66.04,
    "world_a_guarded_fire": 0.0014,
    "world_b_unguarded_below_threshold": 0.7285,
    "world_b_guarded_below_threshold": 0.0133,
    "world_c_guarded_fire": 0.8932,
    "world_c_unguarded_fire": 0.9408,
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


def _derive(seed: int, index: int, slot: int, counter: int) -> int:
    """One frozen draw (amendment A1): the SHA-256 derivation
    with its digest-to-integer conversion pinned exactly.
    """
    material = f"{seed}|invisiblechars|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


class _RedrawExhausted(Exception):
    """Internal: the frozen redraw give-up bound was reached
    for one case, so generation fails and the stage fails
    closed (amendment A1). Unreachable under the frozen
    derivation; it makes the generator provably total.
    """


def generate_cases(seed: int) -> tuple[dict[str, Any], ...]:
    """The full 24-case panel at ``seed``, in case order.

    Values are pairwise distinct across the panel: each case
    draws its word and number at counter 0 and, on a collision
    with an earlier case's value, redraws both slots at
    counter+1 (the frozen derivation of amendment A1), giving
    up after REDRAW_LIMIT consecutive redraws.
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
                raise _RedrawExhausted(f"redraw bound exhausted at case {index}")
        seen.add(value)
        first = _derive(seed, index, 2, 0) % len(NEUTRAL_STATEMENTS)
        second = _derive(seed, index, 3, 0) % len(NEUTRAL_STATEMENTS)
        if second == first:
            second = (first + 1) % len(NEUTRAL_STATEMENTS)
        cases.append(
            {
                "fixture_id": f"invisiblechars-{index:02d}",
                "topic": TOPICS[index],
                "value": value,
                "expected": value,
                "neutral_indices": [first, second],
            }
        )
    return tuple(cases)


def generate_case(seed: int, index: int) -> dict[str, Any]:
    """The frozen generated case ``index`` (0-23) at ``seed``.

    Panel-wide value distinctness makes a case a function of
    the whole panel derivation, so this regenerates the panel
    and returns its ``index`` member -- one code path, total.
    """
    return generate_cases(seed)[index]


def fact_sentence(case: dict[str, Any]) -> str:
    """The case's (untreated) fact sentence."""
    return f"The {case['topic']} inspection code is {case['value']}."


def question_text(case: dict[str, Any]) -> str:
    """The case's question -- byte-identical across arms by
    construction (insertions live in the fact sentence only).
    """
    return f"What is the {case['topic']} inspection code?"


def treated_fact(case: dict[str, Any], code_point: str) -> str:
    """The fact sentence with ``code_point`` inserted at both
    frozen sites (design section 4): after code-point index 2
    of the topic, and after code-point index 2 of the value's
    word part. Built directly on the two site words, which is
    byte-identical to inserting into the assembled sentence in
    reverse site order (the design evidence's two
    implementations agree on this population).
    """
    topic = str(case["topic"])
    word, _, number = str(case["value"]).partition("-")
    treated_topic = topic[:2] + code_point + topic[2:]
    treated_value = word[:2] + code_point + word[2:] + "-" + number
    return f"The {treated_topic} inspection code is {treated_value}."


def arm_content(case: dict[str, Any], arm: str) -> str:
    """The full user-message content of one arm:
    ``{neutral1} {fact} {neutral2} {question}``, with the
    arm's code point inserted at both frozen sites in the
    fact sentence for a treated arm.
    """
    first, second = (int(i) for i in case["neutral_indices"])
    code_point = ARM_CODEPOINTS[arm]
    fact = fact_sentence(case) if code_point is None else treated_fact(case, code_point)
    return f"{NEUTRAL_STATEMENTS[first]} {fact} {NEUTRAL_STATEMENTS[second]} {question_text(case)}"


def audit_arm_population() -> dict[str, Any]:
    """The arm-population audit over the frozen list (design
    section 2): for every invisible arm, the locally decidable
    predicate -- the code point is assigned, its
    General_Category is in {Cf, Mn}, and its canonical
    combining class is 0 -- executed against the local
    interpreter's unicodedata (Unicode 15.0.0). The
    Default_Ignorable column of the design's table is a cited
    UCD-file fact and is NOT checked here, because the stdlib
    does not expose it: the build never depends on a property
    the stdlib cannot check. ``ok`` is the conjunction over
    the five invisible arms; the visible control is reported
    for contrast (it satisfies none of the predicate, by
    design).
    """
    arms: dict[str, Any] = {}
    ok = True
    for arm in (*INVISIBLE_ARMS, "visible_hyphen"):
        code_point = ARM_CODEPOINTS[arm]
        if code_point is None:
            raise ValueError(f"treated arm {arm} carries no code point")
        try:
            name = unicodedata.name(code_point)
        except ValueError:
            name = ""
        category = unicodedata.category(code_point)
        combining = unicodedata.combining(code_point)
        satisfies = bool(name) and category in ("Cf", "Mn") and combining == 0
        if arm in INVISIBLE_ARMS and not satisfies:
            ok = False
        arms[arm] = {
            "code_point": f"U+{ord(code_point):04X}",
            "name": name,
            "category": category,
            "combining_class": combining,
            "satisfies_predicate": satisfies,
        }
    return {"arms": arms, "ok": ok, "unicode_version": unicodedata.unidata_version}


#: The population audit, computed once over the frozen list
#: and bound into the protocol (so the validator recomputes
#: it from its own literal copies, not from trust).
POPULATION_AUDIT: dict[str, Any] = audit_arm_population()


def case_material_sha256() -> str:
    """Digest over every frozen text, list, and constant the
    generator and the rule draw from, so the protocol binds
    the population's definition -- including the frozen draw
    conversion and the floors -- not just its seed.
    """
    material = {
        "topics": list(TOPICS),
        "words": list(WORDS),
        "neutral_statements": list(NEUTRAL_STATEMENTS),
        "arm_codepoints": {arm: ARM_CODEPOINTS[arm] for arm in ARM_KEYS},
        "generator_spec": GENERATOR_SPEC,
        "case_count": CASE_COUNT,
        "arm_keys": list(ARM_KEYS),
        "min_d": MIN_D,
        "min_j": MIN_J,
        "max_incomplete": MAX_INCOMPLETE,
        "redraw_limit": REDRAW_LIMIT,
        "population_audit": POPULATION_AUDIT,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ground_sha256(seed: int) -> str:
    """Digest over the seed's generated ground population
    (canonical rows binding fixture id, topic, value, and the
    neutral indices of every case).
    """
    rows = [
        [
            case["fixture_id"],
            case["topic"],
            case["value"],
            list(case["neutral_indices"]),
        ]
        for case in generate_cases(seed)
    ]
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def protocol_object(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The frozen instrument description a payload is checked
    against: arms, population, floors, budgets, the
    case-material and ground digests, the population audit,
    and the seed / backend identity that generated the
    records.
    """
    return {
        "version": PROTOCOL_VERSION,
        "arm_keys": list(ARM_KEYS),
        "treated_arms": list(TREATED_ARMS),
        "invisible_arms": list(INVISIBLE_ARMS),
        "case_count": CASE_COUNT,
        "min_d": MIN_D,
        "min_j": MIN_J,
        "max_incomplete": MAX_INCOMPLETE,
        "case_material_sha256": case_material_sha256(),
        "ground_sha256": ground_sha256(seed),
        "population_audit": POPULATION_AUDIT,
        "generation_budget": GENERATION_BUDGET,
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
    (design sections 2-4, amendment A9(iv)), executed by the
    stage before any generation and re-executed on the frozen
    material in the tests: the case equals its regeneration
    from the seed (the frozen derivation, amendment A1); the
    value conforms to the grammar with its word in the frozen
    pool and its number in [1000, 9999]; the expected value is
    pure ASCII; both site words have length >= 3; the neutral
    indices are valid and distinct; the control content is
    ASCII and contains none of the five invisible code points;
    every treated arm's content differs from the control's,
    carries exactly two more occurrences of its code point
    than the control carries, and ends with the same question;
    the population audit holds.
    """
    try:
        fixture_id = str(case["fixture_id"])
        index = int(fixture_id.rsplit("-", 1)[1])
        if not 0 <= index < CASE_COUNT or fixture_id != f"invisiblechars-{index:02d}":
            return False
        if generate_case(seed, index) != case:
            return False
        topic = str(case["topic"])
        value = str(case["value"])
        if topic != TOPICS[index] or case["expected"] != value:
            return False
        if not _VALUE_RE.match(value):
            return False
        word, _, number = value.partition("-")
        if word not in WORDS or not 1000 <= int(number) <= 9999:
            return False
        if not value.isascii() or len(topic) < 3 or len(word) < 3:
            return False
        first, second = (int(i) for i in case["neutral_indices"])
        if (
            first == second
            or not (0 <= first < len(NEUTRAL_STATEMENTS))
            or not (0 <= second < len(NEUTRAL_STATEMENTS))
        ):
            return False
        control = arm_content(case, "control")
        if not control.isascii():
            return False
        for arm in INVISIBLE_ARMS:
            code_point = ARM_CODEPOINTS[arm]
            if code_point is None or code_point in control:
                return False
        question = question_text(case)
        for arm in TREATED_ARMS:
            code_point = ARM_CODEPOINTS[arm]
            if code_point is None:
                return False
            content = arm_content(case, arm)
            if content == control or not content.endswith(question):
                return False
            if content.count(code_point) != control.count(code_point) + 2:
                return False
        return bool(POPULATION_AUDIT["ok"])
    except Exception:  # noqa: BLE001 -- a malformed case fails the screen closed
        return False


def decompose_cell(cell: dict[str, Any], expected: str, expected_digest: str, arm: str) -> str:
    """The decomposition class of one completed cell (design
    section 5): an ordered decision list (amendment A5) --
    ``pass`` (the verdict passed); then ``boundary_only``
    (the answer digest equals the expected digest; the
    boundary event alone failed); then ``wrong_value`` (the
    answer parses under the frozen value grammar with its
    word in WORDS and its number in [1000, 9999], and is not
    the expected value); then ``echo_invisible`` (the answer
    is the expected value with the arm's code point inserted
    at the payload site); then ``form_only`` (removing every
    occurrence of the arm's code point from the answer yields
    exactly the expected value, and the answer is not the
    expected value); else ``non_answer``. For the control arm
    the echo and form classes are empty by construction (it
    has no code point); for ``visible_hyphen`` both are empty
    by construction -- the echo and form steps run on the
    five invisible arms only, so the visible site echo (a
    hyphenated string outside the answer grammar) lands in
    ``non_answer`` and its ``form_only`` is empty because a
    visible insertion is not character-removable
    (amendment A6).
    """
    if cell.get("verdict") is True:
        return "pass"
    digest = cell.get("answer_digest")
    if digest is not None and digest == expected_digest:
        return "boundary_only"
    text = cell.get("answer_text")
    if type(text) is not str:
        return "non_answer"
    if text != expected and _VALUE_RE.match(text):
        word, _, number = text.partition("-")
        if word in WORDS and 1000 <= int(number) <= 9999:
            return "wrong_value"
    code_point = ARM_CODEPOINTS[arm]
    if arm in INVISIBLE_ARMS and code_point is not None:
        word, _, number = expected.partition("-")
        site_echo = word[:2] + code_point + word[2:] + "-" + number
        if text == site_echo:
            return "echo_invisible"
        if text != expected and text.replace(code_point, "") == expected:
            return "form_only"
    return "non_answer"


def decompose_pair(control_cell: dict[str, Any], arm_cell: dict[str, Any]) -> str:
    """The decomposition class of one discordant pair (the
    recordnotation construction, which is total): verdicts
    differ and answer digests are equal -> ``boundary_only``;
    verdicts differ, digests differ, and both cells found
    their boundary -> ``exact_only``; verdicts differ
    otherwise -> ``mixed`` (reported, never attributed).
    Callers invoke this only on discordant pairs.
    """
    if control_cell.get("answer_digest") == arm_cell.get("answer_digest"):
        return "boundary_only"
    if control_cell.get("boundary_found") and arm_cell.get("boundary_found"):
        return "exact_only"
    return "mixed"


def classify_headline(control_passes: int, wrong_values: int, n_incomplete: int) -> str:
    """The frozen headline label (design section 6, amendment
    A4), a total function of (D, J, n_incomplete):

    - the paired-population floor withholds every label when
      incomplete cases in the domain exceed MAX_INCOMPLETE;
    - D < MIN_D -> ``insufficient``;
    - J >= MIN_J -> ``discordant_asymmetric`` (the fire,
      issued on the observed counts; incompletion present
      does not by itself withhold a fire);
    - J = 0 -> ``no_discordance_observed`` only at
      n_incomplete = 0 (any incomplete case completes to
      J >= 1 under the worst-case completion, changing the
      label), else ``insufficient``;
    - 0 < J < MIN_J -> ``below_threshold`` only while
      J + n_incomplete < MIN_J (stability under the worst-case
      completion), else ``insufficient``.
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


def _ids_identical(record: dict[str, Any], arm: str) -> bool:
    """Whether one record's renders are ids-identical for
    ``arm`` against control, derived from the recorded id
    digests alone (an encode-time property, decidable whether
    or not either generation completed -- the guard domain of
    amendment A3).
    """
    digests = record.get("id_digests")
    if not isinstance(digests, dict):
        return False
    control = digests.get("control")
    return type(control) is str and control != "" and control == digests.get(arm)


def aggregate_candidate(
    records: list[dict[str, Any]], cases: tuple[dict[str, Any], ...]
) -> dict[str, Any]:
    """Per-candidate aggregates over its ground records.

    ``cases`` is the regenerated panel (``generate_cases`` of
    the run seed): the decomposition classes are functions of
    the expected value and the carried answer texts, so the
    aggregate takes the cases beside the records, matched by
    fixture id. Readouts: per-arm completion, pass rate,
    attrition, and the six-class decomposition; the per-arm
    id report (regime gate); and per treated arm the headline
    over the id-divergent paired population -- D, J,
    ``n_reverse``, ``n_incomplete`` in the pinned domain, the
    paired control and arm pass rates, the pair decomposition
    with the failing cells' classes, the arm's ``non_answer``
    count over the population, and the guarded label.
    Discordance is never pooled across candidates or arms --
    this function sees one candidate's records only.
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
                control_cell = _cell(record, "control")
                arm_cell = _cell(record, arm)
                if (
                    control_cell is not None
                    and arm_cell is not None
                    and control_cell["verdict"] != arm_cell["verdict"]
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
    for arm in TREATED_ARMS:
        domain = [record for record in records if not _ids_identical(record, arm)]
        paired = [
            record
            for record in domain
            if _cell(record, "control") is not None and _cell(record, arm) is not None
        ]
        n_incomplete = len(domain) - len(paired)
        control_passes = 0
        wrong_values = 0
        reverse = 0
        non_answer = 0
        pair_decomposition = dict.fromkeys(PAIR_CLASSES, 0)
        failing_classes = dict.fromkeys(CELL_CLASSES, 0)
        for record in paired:
            control_cell = _cell(record, "control")
            arm_cell = _cell(record, arm)
            if control_cell is None or arm_cell is None:
                continue
            expected = expected_by_fixture.get(str(record.get("fixture", "")), "")
            expected_digest = str(record.get("expected_digest", ""))
            arm_class = decompose_cell(arm_cell, expected, expected_digest, arm)
            if arm_class == "non_answer":
                non_answer += 1
            if control_cell["verdict"]:
                control_passes += 1
                if arm_class == "wrong_value":
                    wrong_values += 1
            elif arm_cell["verdict"]:
                reverse += 1
            if control_cell["verdict"] != arm_cell["verdict"]:
                pair_decomposition[decompose_pair(control_cell, arm_cell)] += 1
                failing = arm_cell if control_cell["verdict"] else control_cell
                failing_arm = arm if control_cell["verdict"] else "control"
                failing_classes[
                    decompose_cell(failing, expected, expected_digest, failing_arm)
                ] += 1
        headline[arm] = {
            "label": classify_headline(control_passes, wrong_values, n_incomplete),
            "n_population": len(paired),
            "n_incomplete": n_incomplete,
            "control_passes": control_passes,
            "wrong_values": wrong_values,
            "n_reverse": reverse,
            "control_rate": _rate(control_passes, len(paired)),
            "arm_rate": _rate(sum(1 for record in paired if _verdict(record, arm)), len(paired)),
            "n_non_answer": non_answer,
            "pair_decomposition": pair_decomposition,
            "failing_classes": failing_classes,
        }
    return {
        "arms": arms,
        "ids": ids,
        "headline": headline,
        "operating_characteristics": dict(OPERATING_CHARACTERISTICS),
        "n_cases": len(records),
        "n_incomplete": sum(
            1 for record in records if any(_cell(record, arm) is None for arm in ARM_KEYS)
        ),
    }
