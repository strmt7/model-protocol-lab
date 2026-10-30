r"""Confusable-skeleton substitution (cycle 224): the probe
that replaces Latin letters in a generated fact sentence with
their UTS #39 skeleton-equivalent Cyrillic/Greek letters and
classifies, per candidate and per arm, whether the
substitution moved the model's answers to wrong values.

An eval owner deciding, Monday morning, whether frozen
benchmark fixtures and ingested documents need a
confusable-screening pass (skeleton-based detection at
ingest) gets, per candidate (template family) and per
substitution arm, a number: how often replacing Latin letters
with their UTS #39 skeleton-equivalent Cyrillic/Greek letters
in ordinary generated fact-recall text changes what a local
model answers. If skeleton-class substitution never moves
answers at the frozen floors, the screening pass is
deprioritised for that family; if it fires, the pass -- cheap
and deterministic -- has a measured reason to exist. The
claim is "skeleton-class substitution" -- substitution inside
a standards-defined equivalence class (UTS #39 section 4
skeleton equality) -- never that a model is fooled like a
human: the model never sees glyphs. The probe measures the
effect of the substitution, not the accuracy of any screening
implementation and not human visual confusability.

The instrument (docs/CONFUSABLESKELETON_DESIGN.md, the
wave-12 candidate 2 design as amended by the fresh review's
binding amendments A1-A9): 24 generated cases per candidate,
stage-local, never entering the user's search/held-out Split.
The panel machinery is the sibling panel of canonical-form
and invisible-chars: case i's topic is TOPICS[i]; its value
is ``word-number`` with the word from the canonical-form
24-word ASCII pool and the number in [1000, 9999], values
pairwise distinct across the panel by redraw; content is
``{neutral1} {fact} {neutral2} {question}``. The draws are
SHA-256 derivations with the conversion pinned exactly:
material ``f"{seed}|confusableskeleton|{index}|{slot}|{counter}"``,
integer ``int.from_bytes(sha256(material).digest()[:8], "big")``;
slots 0 = value word, 1 = number (a collision redraws both at
counter+1, with a frozen give-up bound of 1,000 consecutive
redraws whose exhaustion fails the stage closed), 2/3 =
neutral indices (the second advances +1 on collision). The
namespace is this probe's own, so its panel is an independent
draw from the sibling's. Every expected value is pure ASCII,
enforced by the generator battery, so the byte-exact scorer
can never manufacture discordance out of the value itself.

The population is derived from confusables.txt pinned to
Unicode 15.0.0 -- the design-time interpreter's UCD version,
so the derivation is single-version -- with the DICP set of
skeleton step 2 from DerivedCoreProperties.txt 15.0.0; both
files' SHA-256 digests are frozen constants below, and UTS
#39 rev 27 (a 15.1.0 draft) is cited for the section 4
algorithm only, never for data, because skeletons are not
version-stable. The shipped tables are frozen literals: at
runtime the audit records the host's UCD version as read and
checks only properties of the frozen characters that are
stable across the supported interpreters' UCD versions
(14.0 / 15.0 / 15.1). A
code point is an eligible replacement for a panel letter iff
its skeleton equals the letter's under the pinned data, its
unicodedata name starts with CYRILLIC SMALL LETTER or GREEK
SMALL LETTER, it has no canonical decomposition, and it is
NFKC-invariant; the freeze is the smallest code point
satisfying the rule. Sixteen of the 25 panel letters freeze
(a to Greek alpha, r to Cyrillic ghe, y to Greek gamma -- the
population is normative, not visual); nine are zero-eligible
(b, f, g, k, l, m, n, t, z -- l has 73 skeleton-equal sources,
none in the frozen script class). Every one of the 48 panel
words carries at least one eligible letter, so no case is
untreated. Sequence-level equivalence (skeleton("rn") ==
skeleton("m") == "rn") is normatively real and is excluded
from v1 by the single-code-point rule -- a length-changing
mechanism is a different dose axis.

Four arms, fully crossed, substitutions inside the fact
sentence only (the topic word and the value's word part; the
question and both neutrals are byte-identical across arms):
``control`` (verbatim); ``single_site`` (the first eligible
letter of each treated word replaced by its frozen
confusable -- exactly 2 substitutions per case);
``whole_word`` (every eligible letter occurrence -- 4 to 10
per case; on seed 0's panel 171 of 249 letters in treated
words, 0.6867, the rest being zero-eligible letters no arm
can substitute); and ``rarity_control`` (exactly the
single_site sites, filled with a same-script non-confusable
letter -- the isolating control for rarity vs skeleton
class). The rarity controls are two characters, one per
script seat: U+0434 CYRILLIC SMALL LETTER DE for the Cyrillic
seat and U+03B6 GREEK SMALL LETTER ZETA for the Greek seat,
assigned per treated word by the script of that word's
single_site frozen replacement, script determined by the
unicodedata.name prefix (amendment A2 -- a code-point range
proxy is not the rule: the Cyrillic Supplement replacements
U+0501 and U+051B take the Cyrillic control). The property
relied on (amendment A7) is stated exactly: each control is
not a source in the pinned file, its skeleton is itself, and
it is skeleton-distinct from every panel letter and every
frozen replacement. U+0434 is wholly absent from the file;
U+03B6 is the target of five mathematical variants (a
six-member skeleton class, disclosed in the design doc) --
the operative separations hold; "the controls are
non-confusable" is not the claim. Substitution preserves
code-point count and adds one UTF-8 byte per substituted
letter (every replacement and both controls are 2 bytes),
a byte-length delta the rarity control matches exactly.

Scoring is the tree's own: a cell's verdict is
``core.evaluate``'s ``exact_match`` conjoined with
``boundary_found``, on a Fixture with the same id, group, and
expected at every arm, so request seeds are identical by
construction. Every completed cell decomposes by an ORDERED
decision list -- pass, boundary_only, wrong_value,
echo_confusable / echo_control, form_only, non_answer --
each class reached only when every earlier predicate fails.
``wrong_value`` -- the only class counting toward the
headline -- is scoped to the frozen value grammar: a value
change outside it decomposes as ``non_answer``, is printed
per arm, and never counts toward the headline. The echo
classes are scripted per arm (the answer equals the expected
value with that arm's own treatment applied to the value
word); ``form_only`` for the confusable arms maps every
frozen-table character in the answer back to its Latin
letter (the frozen-table scope: an echo in a non-frozen
skeleton-equal letter decomposes as ``non_answer``), and for
the rarity arm is the position-aware predicate of amendment
A3 -- same code-point length as the expected value, at least
one control-set character, and replacing every control-set
character by the expected value's character at the same
position yields exactly the expected value (a fixed inverse
is impossible there: the Cyrillic control stands for nine
letters across cases, the Greek for seven).

Token-ID divergence is the regime gate, measured at encode
time on the full render id sequence per (case, candidate),
per treated arm against control. The headline population per
(candidate, arm) is the cases completed in both control and
arm AND id-divergent for that arm; ids-identical arms are
reported instrument-inapplicable for that family, never as
passes. Pre-registered regime expectation: these characters
are not NFKC-foldable and not default-ignorable, so both
NFKC-regime and byte-level pipelines are expected to encode
treated text divergently; a family landing ids-identical has
a skeleton-screening or confusable-normalizing tokenizer --
itself the utility question's most direct evidence.

The headline is a guarded classification, per (candidate,
arm): D = control passes in the population; J = of those, arm
cells decomposed ``wrong_value``; floors MIN_D = 10 and
MIN_J = 6, derived under this instrument's own null (MIN_J
set by the floor search -- every MIN_J <= 5 pair breaches the
competent-null bound at every MIN_D; MIN_D selected from a
plateau, exact on [1, 6] and identical at printed precision
over [1, 14], the selected pair trailing the exact maximum
by under 3e-11). Labels: ``insufficient``,
``no_discordance_observed``, ``discordant_asymmetric`` (the
fire: the section 7 invariance claim is falsified for this
arm, under the guard), ``below_threshold``. Reverse
discordants are counted as ``n_reverse`` beside every label
and never pooled into J. The two-directional canonical rule
was rejected by execution: at N = 24 the transplanted floors
fire at 0.8897 on the competent independent null.

The completion guard closes the channel that defeated the
multi-question line twice, transferred with the sibling
round's amendments folded in. Incompletion's domain is
pinned: exactly the id-divergent cases (an encode-time
property, decidable whether or not the generation completed)
with at least one missing cell; ids-identical cases never
enter a completion. A fire cannot be manufactured: J and D
are absolute counts over completed cases, issued on observed
counts. An acquittal cannot be manufactured: acquittal labels
issue only if stable under the worst-case completion (every
incomplete case in the domain counted as control-pass + arm
``wrong_value`` -- a deliberate blanket over-count ranging
over observation-inconsistent completions for
partially-observed cases, disclosed in the design doc, the
computation unchanged). Consequences, printed:
``no_discordance_observed`` requires zero incomplete cases in
the domain; ``below_threshold`` survives only while
J + n_incomplete < 6; and every label is withheld when
incomplete cases in the domain exceed 6 (paired < 18, the
smallest paired count with printed operating
characteristics). The cap's only marginal effect in the
exhaustive reachable-state audit (D + k <= 24) is suppressing
264 completion-stable fire states. The guard's worlds,
executed at design time (10,000 trials each, ranging over all
incomplete cases -- the conservative direction; the build's
battery re-executes them under the pinned domain): World A
(zero effect, sign-concentrated incompletion) guarded fire
0.0055 against unguarded 0.0186; World B (a true drop, 60
percent of would-be discordants removed) unguarded
below-threshold 0.7277 against guarded 0.0117; World C (the
same drop, flat incompletion) guarded fire 0.8893 against
unguarded 0.9399. No sign test is used anywhere.

Operating characteristics, one convention, stated: the
independent column is exact enumeration under per-case
independence with q the arm's marginal in-grammar wrong-value
rate; the paired column is exact (per-case bivariate
quadrature at rho 0.5 with trinomial summation -- amendment
A6: the design-phase Monte Carlo understated it and is not
printed). The competent null fires 0.0173 independent and
0.004442 paired; the 25-point target effect fires 0.8159 at
base .90 and 0.8761 at base .85 (paired 0.698078); a 10-point
drop fires only 0.2542, so an acquittal is an observed null
at this panel's resolution, never proof of invariance. The
family-wise price prints beside the licence in every report:
over the 72 labelled comparisons (3 treated arms x 24
candidates) the competent expectation is 1.2437 false labels
independent (probability of at least one 0.7148) and 0.3199
paired (0.2743). The rarity arm receives its own label under
the identical rule: a single_site fire is read as
skeleton-class-specific only if rarity_control does not also
fire for the same candidate; for whole_word the isolation
reading does not extend -- a whole-word solo fire reads
"skeleton-class substitution at whole-word dose disrupts
this candidate; rarity at that dose is unmeasured by this
design" (amendment A5). No pooled headline and no winner
readout in v1.

Budgets: 24 cases x 4 arms x 24 candidates = 2,304
generations at the legal maximum; 576 ground records; the
house 12 MiB records cap, enforced on the serialised records
with the frozen budget detail. Records carry digests, with
one sanctioned exception: every cell carries ``answer_text``
bound by ``answer_digest`` (the cycle-190 pattern), because
the decomposition classes are functions of the answer text
and uncomputable from digests; reports render counts and
rates only, and answer text never reaches a report. Per-arm
content digests are mandatory: the diff validator's mirror
regenerates all four arm contents from the seed and checks
each digest, and each record carries the frozen-table
version pin (the confusables data version and the table's
own digest), so the treatment itself -- not just token
counts -- is committed. Vocabulary is the tree's frozen
fail-closed vocabulary: stage details {cancelled,
confusable_skeleton_record_budget_exceeded,
confusable_skeleton_screen_failed}; cell details {"",
arm_context_exceeded, arm_evaluation_failed}; whole-stage
failure {"available": False, "detail":
"confusable_skeleton_unavailable"}; classification labels
{insufficient, no_discordance_observed,
discordant_asymmetric, below_threshold} and no other verdict
words. Discordance is never pooled across candidates or
arms. Informational only: the stage never affects selection
and never fails the run.
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
    "CONFUSABLES_DERIVED_SHA256",
    "CONFUSABLES_SHA256",
    "CONFUSABLES_VERSION",
    "CONTROL_CHARACTERS",
    "FROZEN_REPLACEMENTS",
    "GENERATION_BUDGET",
    "GENERATOR_SPEC",
    "GROUP",
    "LABELS",
    "LETTER_SKELETONS",
    "MAX_CANDIDATES",
    "MAX_GROUND_RECORDS",
    "MAX_INCOMPLETE",
    "MAX_RECORDS_BYTES",
    "MIN_D",
    "MIN_J",
    "NEUTRAL_STATEMENTS",
    "OPERATING_CHARACTERISTICS",
    "PAIR_CLASSES",
    "PANEL_LETTERS",
    "POPULATION_AUDIT",
    "POPULATION_COUNTS",
    "PROTOCOL_VERSION",
    "RARITY_CONTROLS",
    "REDRAW_LIMIT",
    "STAGE_DETAILS",
    "TOPICS",
    "TREATED_ARMS",
    "WORDS",
    "ZERO_ELIGIBLE",
    "aggregate_candidate",
    "arm_content",
    "audit_population",
    "case_material_sha256",
    "classify_headline",
    "decompose_cell",
    "decompose_pair",
    "digest_ids",
    "digest_text",
    "eligible_count",
    "fact_sentence",
    "generate_case",
    "generate_cases",
    "ground_sha256",
    "protocol_fingerprint",
    "protocol_object",
    "question_text",
    "records_bytes",
    "skeleton_form",
    "table_sha256",
    "treated_fact",
    "treated_word",
    "validate_case",
]

PROTOCOL_VERSION = 1

GROUP = "confusable_skeleton"

CASE_COUNT = 24

#: The frozen 24-word topic list (design section 3): case i's
#: topic is TOPICS[i]. All ASCII, each with at least one
#: eligible letter (the generator screen asserts it).
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
#: alarm).
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

#: The pinned population data (design section 2): the
#: confusables data and the DICP source, both Unicode 15.0.0
#: (the local interpreter's UCD version), identified by the
#: SHA-256 digests executed against the publisher's files in
#: the design round. The population derived from them is
#: frozen in the tables below; a future Unicode bump is a
#: population change requiring re-derivation, never a silent
#: upgrade.
CONFUSABLES_VERSION = "15.0.0"
CONFUSABLES_SHA256 = "2b10130885c3370b101c52d7baedc452ab7f0e257b86c1e52ee657ecfc29ce64"
CONFUSABLES_DERIVED_SHA256 = "d367290bc0867e6b484c68370530bdd1a08b6b32404601b8c7accaf83e05628d"

#: The frozen replacement table (design section 2): for each
#: eligible panel letter, the smallest code point whose
#: skeleton equals the letter's under the pinned data, whose
#: name starts with CYRILLIC SMALL LETTER or GREEK SMALL
#: LETTER, with no canonical decomposition and NFKC identity.
FROZEN_REPLACEMENTS: dict[str, str] = {
    "a": "\u03b1",
    "c": "\u0441",
    "d": "\u0501",
    "e": "\u0435",
    "h": "\u04bb",
    "i": "\u03b9",
    "j": "\u0458",
    "o": "\u03bf",
    "p": "\u03c1",
    "q": "\u051b",
    "r": "\u0433",
    "s": "\u0455",
    "u": "\u03c5",
    "v": "\u03bd",
    "w": "\u0461",
    "y": "\u03b3",
}

#: The per-letter rarity control (design section 2, the
#: definition in full): the control character a letter's site
#: receives in the rarity arm -- U+0434 CYRILLIC SMALL LETTER
#: DE for every letter whose frozen replacement is Cyrillic,
#: U+03B6 GREEK SMALL LETTER ZETA for every letter whose
#: frozen replacement is Greek. Assignment is by the
#: unicodedata.name prefix of the frozen replacement
#: (amendment A2), never by code-point range: the Cyrillic
#: Supplement replacements U+0501 (d) and U+051B (q) take the
#: Cyrillic control.
RARITY_CONTROLS: dict[str, str] = {
    "a": "\u03b6",
    "c": "\u0434",
    "d": "\u0434",
    "e": "\u0434",
    "h": "\u0434",
    "i": "\u03b6",
    "j": "\u0434",
    "o": "\u03b6",
    "p": "\u03b6",
    "q": "\u0434",
    "r": "\u0434",
    "s": "\u0434",
    "u": "\u03b6",
    "v": "\u03b6",
    "w": "\u0434",
    "y": "\u03b6",
}

#: The control character set (the rarity arm's two seats).
CONTROL_CHARACTERS: tuple[str, ...] = ("\u0434", "\u03b6")

#: The skeleton of each panel letter under the pinned data:
#: the identity for every letter except m, whose prototype is
#: the two-character "rn" (the file's only ASCII-letter
#: source). A frozen replacement's skeleton is its letter's
#: skeleton (the population definition), so word skeletons on
#: the panel alphabet are computed from this table alone.
LETTER_SKELETONS: dict[str, str] = {
    "a": "a",
    "b": "b",
    "c": "c",
    "d": "d",
    "e": "e",
    "f": "f",
    "g": "g",
    "h": "h",
    "i": "i",
    "j": "j",
    "k": "k",
    "l": "l",
    "m": "rn",
    "n": "n",
    "o": "o",
    "p": "p",
    "q": "q",
    "r": "r",
    "s": "s",
    "t": "t",
    "u": "u",
    "v": "v",
    "w": "w",
    "y": "y",
    "z": "z",
}

#: The executed population audit counts (design section 2):
#: per panel letter, (skeleton-equal sources in the pinned
#: file, sources eligible under the frozen script rule).
POPULATION_COUNTS: dict[str, tuple[int, int]] = {
    "a": (23, 2),
    "b": (18, 0),
    "c": (21, 1),
    "d": (19, 1),
    "e": (19, 2),
    "f": (18, 0),
    "g": (18, 0),
    "h": (17, 1),
    "i": (37, 4),
    "j": (17, 1),
    "k": (13, 0),
    "l": (73, 0),
    "m": (17, 0),
    "n": (15, 0),
    "o": (75, 3),
    "p": (29, 2),
    "q": (16, 1),
    "r": (19, 1),
    "s": (20, 1),
    "t": (13, 0),
    "u": (27, 1),
    "v": (29, 2),
    "w": (22, 2),
    "y": (30, 3),
    "z": (16, 0),
}

#: The panel alphabet: the 25 distinct letters of the topics
#: and value words (a-z minus x).
PANEL_LETTERS: tuple[str, ...] = tuple(sorted(POPULATION_COUNTS))

#: The zero-eligible letters: no Cyrillic/Greek small-letter
#: skeleton equivalent in Unicode 15.0.0.
ZERO_ELIGIBLE: tuple[str, ...] = tuple(
    letter for letter in PANEL_LETTERS if letter not in FROZEN_REPLACEMENTS
)

ARM_KEYS: tuple[str, ...] = ("control", "single_site", "whole_word", "rarity_control")
TREATED_ARMS: tuple[str, ...] = ARM_KEYS[1:]
CONFUSABLE_ARMS: tuple[str, ...] = ("single_site", "whole_word")

#: The frozen floors and the paired-population cap (design
#: section 6).
MIN_D = 10
MIN_J = 6
MAX_INCOMPLETE = 6

#: The redraw give-up bound: consecutive collision redraws
#: for one case beyond which generation fails and the stage
#: fails closed. Unreachable under the frozen derivation (24
#: draws from a 216,000-value space); it exists so the
#: generator is provably total. The counting is the module's:
#: the counter increments once per colliding redraw and the
#: bound trips when the counter exceeds REDRAW_LIMIT.
REDRAW_LIMIT = 1000

#: The frozen draw derivation, stated as text so the protocol
#: fingerprint binds the seed-to-panel mapping the validator's
#: regeneration rests on, not just the seed.
GENERATOR_SPEC = (
    "draw(seed, index, slot, counter) = int.from_bytes("
    "sha256(f'{seed}|confusableskeleton|{index}|{slot}|{counter}')"
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
    "confusable_skeleton_record_budget_exceeded",
    "confusable_skeleton_screen_failed",
)

#: Decomposition of a completed cell (design section 5, the
#: ordered decision list). The echo class is scripted per
#: arm: ``echo_confusable`` on the two confusable arms,
#: ``echo_control`` on the rarity arm; both are empty on the
#: control arm by construction.
CELL_CLASSES: tuple[str, ...] = (
    "pass",
    "boundary_only",
    "wrong_value",
    "echo_confusable",
    "echo_control",
    "form_only",
    "non_answer",
)

#: Decomposition of a discordant pair (the sibling total
#: partition, mirrored).
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
#: marginal convention and the exact bivariate paired
#: computation (design section 6, amendment A6). Printed
#: with the report section; never re-derived at run time.
#: The cycle-224 test module reproduces every figure from
#: its own code.
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
    "familywise_competent_expected_72": 1.2437,
    "familywise_competent_any_72": 0.7148,
    "familywise_paired_expected_72": 0.3199,
    "familywise_paired_any_72": 0.2743,
    "familywise_random_expected_72": 39.6221,
    "world_a_guarded_fire": 0.0055,
    "world_a_unguarded_fire": 0.0186,
    "world_b_unguarded_below_threshold": 0.7277,
    "world_b_guarded_below_threshold": 0.0117,
    "world_c_guarded_fire": 0.8893,
    "world_c_unguarded_fire": 0.9399,
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
    """One frozen draw: the SHA-256 derivation with its
    digest-to-integer conversion pinned exactly.
    """
    material = f"{seed}|confusableskeleton|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


class _RedrawExhausted(Exception):
    """Internal: the frozen redraw give-up bound was reached
    for one case, so generation fails and the stage fails
    closed. Unreachable under the frozen derivation; it makes
    the generator provably total.
    """


def generate_cases(seed: int) -> tuple[dict[str, Any], ...]:
    """The full 24-case panel at ``seed``, in case order.

    Values are pairwise distinct across the panel: each case
    draws its word and number at counter 0 and, on a collision
    with an earlier case's value, redraws both slots at
    counter+1 (the frozen derivation), giving up after
    REDRAW_LIMIT consecutive redraws. The neutral indices are
    drawn at counter 0 unconditionally (amendment A9).
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
                "fixture_id": f"confusableskeleton-{index:02d}",
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
    construction (substitutions live in the fact sentence
    only).
    """
    return f"What is the {case['topic']} inspection code?"


def eligible_count(word: str) -> int:
    """The number of eligible letter occurrences in ``word``
    under the frozen table (its whole-word dose).
    """
    return sum(1 for char in word if char in FROZEN_REPLACEMENTS)


def treated_word(word: str, arm: str) -> str:
    """One word under one arm's treatment (design section 4):
    ``control`` is the identity; ``single_site`` replaces the
    first eligible letter by its frozen confusable;
    ``whole_word`` replaces every eligible letter;
    ``rarity_control`` replaces the first eligible letter by
    that letter's rarity control (its script seat's control
    character, amendment A2). A word with no eligible letter
    is the identity in every arm (dose 0; the frozen panel
    has none, and the generator screen asserts it).
    """
    if arm == "control":
        return word
    if arm == "whole_word":
        return "".join(FROZEN_REPLACEMENTS.get(char, char) for char in word)
    table = FROZEN_REPLACEMENTS if arm == "single_site" else RARITY_CONTROLS
    for index, char in enumerate(word):
        if char in table:
            return word[:index] + table[char] + word[index + 1 :]
    return word


def treated_fact(case: dict[str, Any], arm: str) -> str:
    """The fact sentence with the arm's treatment applied to
    the topic and to the value's word part.
    """
    word, _, number = str(case["value"]).partition("-")
    treated_topic = treated_word(str(case["topic"]), arm)
    treated_value = treated_word(word, arm) + "-" + number
    return f"The {treated_topic} inspection code is {treated_value}."


def arm_content(case: dict[str, Any], arm: str) -> str:
    """The full user-message content of one arm:
    ``{neutral1} {fact} {neutral2} {question}``, with the
    arm's substitution applied in the fact sentence for a
    treated arm.
    """
    first, second = (int(i) for i in case["neutral_indices"])
    fact = fact_sentence(case) if arm == "control" else treated_fact(case, arm)
    return f"{NEUTRAL_STATEMENTS[first]} {fact} {NEUTRAL_STATEMENTS[second]} {question_text(case)}"


def skeleton_form(word: str) -> str:
    """The word's skeleton on the panel alphabet, computed
    from the frozen tables alone: each panel letter maps to
    its frozen skeleton (the identity except m, whose
    prototype is "rn"), each frozen replacement maps to its
    letter's skeleton (the population definition), and every
    other character -- both rarity controls included, whose
    skeletons are themselves -- maps to itself. Valid exactly
    on strings over the panel alphabet plus the frozen
    replacements and controls, which is where the module and
    its mirror apply it (treated panel words); the full
    UTS #39 skeleton over arbitrary text needs the pinned
    data file and is a design-time derivation, not a runtime
    one.
    """
    out: list[str] = []
    inverse = {replacement: letter for letter, replacement in FROZEN_REPLACEMENTS.items()}
    for char in word:
        if char in LETTER_SKELETONS:
            out.append(LETTER_SKELETONS[char])
        elif char in inverse:
            out.append(LETTER_SKELETONS[inverse[char]])
        else:
            out.append(char)
    return "".join(out)


def _script_of(char: str) -> str:
    """The script seat of one character by its unicodedata
    name prefix (the design's own classification method):
    "CYRILLIC", "GREEK", or "".
    """
    name = unicodedata.name(char, "")
    if name.startswith("CYRILLIC"):
        return "CYRILLIC"
    if name.startswith("GREEK"):
        return "GREEK"
    return ""


def _replacement_satisfies(replacement: str, control: str) -> bool:
    """Whether one frozen replacement meets every stdlib-decidable
    eligibility predicate of design section 2, including the
    amendment-A2 rule that its rarity control seats in the same
    script (the caller supplies the letter's paired control).
    """
    return (
        unicodedata.name(replacement, "").startswith(
            ("CYRILLIC SMALL LETTER", "GREEK SMALL LETTER")
        )
        and unicodedata.category(replacement) == "Ll"
        and unicodedata.normalize("NFC", replacement)
        == unicodedata.normalize("NFD", replacement)
        == replacement
        and unicodedata.normalize("NFKC", replacement) == replacement
        and unicodedata.decomposition(replacement) == ""
        and len(replacement.encode("utf-8")) == 2
        and _script_of(control) == _script_of(replacement)
        and control in CONTROL_CHARACTERS
    )


def _control_satisfies(control: str) -> bool:
    """Whether one control character meets the amendment-A7
    properties in their stdlib-decidable form: named, category
    Ll, normalization identity, empty canonical decomposition,
    exactly 2 UTF-8 bytes, skeleton-distinct from every panel
    letter and frozen replacement, and its own skeleton.
    """
    distinct = (
        control not in FROZEN_REPLACEMENTS.values()
        and control not in LETTER_SKELETONS
        and skeleton_form(control) == control
    )
    return (
        bool(unicodedata.name(control, ""))
        and unicodedata.category(control) == "Ll"
        and unicodedata.normalize("NFC", control)
        == unicodedata.normalize("NFD", control)
        == control
        and unicodedata.normalize("NFKC", control) == control
        and unicodedata.decomposition(control) == ""
        and len(control.encode("utf-8")) == 2
        and distinct
    )


def audit_population() -> dict[str, Any]:
    """The population audit over the frozen tables (design
    section 2), executed against the local interpreter's
    unicodedata (Unicode 15.0.0): for every panel letter, the
    frozen replacement's name, script seat, and category, its
    per-letter rarity control, and the stdlib-decidable
    eligibility predicates (name prefix CYRILLIC/GREEK SMALL
    LETTER, category Ll, NFC == NFD == identity, empty
    canonical decomposition, NFKC identity, exactly 2 UTF-8
    bytes, and -- amendment A2 -- the control's script seat
    equal to the replacement's); for both controls, the
    amendment-A7 properties in their stdlib-decidable form
    (skeleton is itself under the frozen tables, distinct
    from every panel letter and every frozen replacement,
    2 UTF-8 bytes, normalization identity); and the panel
    invariants (every topic and value word carries at least
    one eligible letter; skeleton invariance of both
    confusable arms over all 48 words; the rarity arm's
    skeleton differs; substitution preserves code-point
    length). The file-membership facts (a replacement's
    skeleton equality in the pinned confusables.txt; a
    control's absence as a source) are design-time derivations
    bound into the protocol through the frozen tables and the
    pinned files' digests: the stdlib exposes neither the
    prototype map nor source membership, so the build never
    depends at run time on a property it cannot check.
    ``ok`` is the conjunction.
    """
    letters: dict[str, Any] = {}
    ok = True
    for letter in PANEL_LETTERS:
        replacement = FROZEN_REPLACEMENTS.get(letter)
        control = RARITY_CONTROLS.get(letter)
        skeleton_equal, eligible = POPULATION_COUNTS[letter]
        entry: dict[str, Any] = {
            "skeleton_equal_sources": skeleton_equal,
            "eligible_sources": eligible,
            "replacement": None,
            "replacement_name": "",
            "script": "",
            "control": None,
            "control_name": "",
            "satisfies": replacement is None,
        }
        if replacement is not None and control is not None:
            name = unicodedata.name(replacement, "")
            script = _script_of(replacement)
            control_name = unicodedata.name(control, "")
            satisfies = _replacement_satisfies(replacement, control)
            entry.update(
                {
                    "replacement": f"U+{ord(replacement):04X}",
                    "replacement_name": name,
                    "script": script,
                    "control": f"U+{ord(control):04X}",
                    "control_name": control_name,
                    "satisfies": satisfies,
                }
            )
            if not satisfies:
                ok = False
        letters[letter] = entry
    controls: dict[str, Any] = {}
    for control in CONTROL_CHARACTERS:
        name = unicodedata.name(control, "")
        distinct = (
            control not in FROZEN_REPLACEMENTS.values()
            and control not in LETTER_SKELETONS
            and skeleton_form(control) == control
        )
        satisfies = _control_satisfies(control)
        controls[f"U+{ord(control):04X}"] = {
            "name": name,
            "script": _script_of(control),
            "skeleton_distinct": distinct,
            "satisfies": satisfies,
        }
        if not satisfies:
            ok = False
    words_ok = all(eligible_count(word) >= 1 for word in (*TOPICS, *WORDS))
    invariance = all(
        skeleton_form(treated_word(word, arm)) == skeleton_form(word)
        for word in (*TOPICS, *WORDS)
        for arm in CONFUSABLE_ARMS
    )
    rarity_differs = all(
        skeleton_form(treated_word(word, "rarity_control")) != skeleton_form(word)
        for word in (*TOPICS, *WORDS)
    )
    length_ok = all(
        len(treated_word(word, arm)) == len(word)
        for word in (*TOPICS, *WORDS)
        for arm in ARM_KEYS[1:]
    )
    ok = ok and words_ok and invariance and rarity_differs and length_ok
    return {
        "letters": letters,
        "controls": controls,
        "every_word_eligible": words_ok,
        "skeleton_invariance": invariance,
        "rarity_skeleton_differs": rarity_differs,
        "length_preserved": length_ok,
        "ok": ok,
        "unicode_version": unicodedata.unidata_version,
        "confusables_version": CONFUSABLES_VERSION,
    }


#: The population audit, computed once over the frozen tables
#: and bound into the protocol (so the validator recomputes
#: it from its own literal copies, not from trust).
POPULATION_AUDIT: dict[str, Any] = audit_population()


def table_sha256() -> str:
    """Digest over the frozen population tables and the
    pinned data files' digests: the frozen-table version pin
    each record carries (design section 10), so a silent
    table or data change is a digest change, not a result
    change.
    """
    material = {
        "frozen_replacements": FROZEN_REPLACEMENTS,
        "rarity_controls": RARITY_CONTROLS,
        "control_characters": list(CONTROL_CHARACTERS),
        "letter_skeletons": LETTER_SKELETONS,
        "population_counts": {letter: list(counts) for letter, counts in POPULATION_COUNTS.items()},
        "confusables_version": CONFUSABLES_VERSION,
        "confusables_sha256": CONFUSABLES_SHA256,
        "confusables_derived_sha256": CONFUSABLES_DERIVED_SHA256,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def case_material_sha256() -> str:
    """Digest over every frozen text, list, and constant the
    generator and the rule draw from, so the protocol binds
    the population's definition -- including the frozen draw
    conversion, the tables, and the floors -- not just its
    seed.
    """
    material = {
        "topics": list(TOPICS),
        "words": list(WORDS),
        "neutral_statements": list(NEUTRAL_STATEMENTS),
        "arm_keys": list(ARM_KEYS),
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
    case-material and ground digests, the table pin, the
    population audit, and the seed / backend identity that
    generated the records.
    """
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
        "confusables_version": CONFUSABLES_VERSION,
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
    (design sections 2-4), executed by the stage before any
    generation and re-executed on the frozen material in the
    tests: the case equals its regeneration from the seed
    (the frozen derivation); the value conforms to the
    grammar with its word in the frozen pool and its number
    in [1000, 9999]; the expected value is pure ASCII; both
    treated words carry at least one eligible letter; the
    neutral indices are valid and distinct; the control
    content is ASCII; every treated arm's content differs
    from the control's and ends with the same question; per
    treated word and arm, the treatment is position-exact
    (each position carries the original character or the
    arm's frozen mapping of it), preserves code-point length,
    substitutes exactly the arm's dose (single_site and
    rarity_control: one per word; whole_word: the eligible
    count), keeps the confusable arms skeleton-invariant and
    moves the rarity arm's skeleton; the population audit
    holds.
    """
    try:
        fixture_id = str(case["fixture_id"])
        index = int(fixture_id.rsplit("-", 1)[1])
        if not 0 <= index < CASE_COUNT or fixture_id != f"confusableskeleton-{index:02d}":
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
        if not value.isascii() or eligible_count(topic) < 1 or eligible_count(word) < 1:
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
        question = question_text(case)
        for arm in TREATED_ARMS:
            content = arm_content(case, arm)
            if content == control or not content.endswith(question):
                return False
            for original in (topic, word):
                treated = treated_word(original, arm)
                if len(treated) != len(original):
                    return False
                table = (
                    FROZEN_REPLACEMENTS
                    if arm in CONFUSABLE_ARMS
                    else RARITY_CONTROLS
                    if arm == "rarity_control"
                    else {}
                )
                dose = 0
                for before, after in zip(original, treated, strict=True):
                    if after != before:
                        dose += 1
                        if table.get(before) != after:
                            return False
                if arm == "whole_word":
                    expected_dose = eligible_count(original)
                else:
                    expected_dose = min(1, eligible_count(original))
                if dose != expected_dose:
                    return False
                if arm in CONFUSABLE_ARMS and skeleton_form(treated) != skeleton_form(original):
                    return False
                if arm == "rarity_control" and skeleton_form(treated) == skeleton_form(original):
                    return False
        return bool(POPULATION_AUDIT["ok"])
    except Exception:  # noqa: BLE001 -- a malformed case fails the screen closed
        return False


def decompose_cell(cell: dict[str, Any], expected: str, expected_digest: str, arm: str) -> str:
    """The decomposition class of one completed cell (design
    section 5): an ordered decision list -- ``pass`` (the
    verdict passed); then ``boundary_only`` (the answer digest
    equals the expected digest; the boundary event alone
    failed); then ``wrong_value`` (the answer parses under
    the frozen value grammar with its word in WORDS and its
    number in [1000, 9999], and is not the expected value);
    then the arm's echo class (``echo_confusable`` on the
    confusable arms: the answer is the expected value with
    that arm's own treatment applied to the value word;
    ``echo_control`` on the rarity arm, analogously); then
    ``form_only`` (confusable arms: mapping every frozen-table
    character in the answer back to its Latin letter yields
    exactly the expected value, and the answer is not the
    expected value -- the frozen-table scope of amendment
    A8(iv); rarity arm: the position-aware predicate of
    amendment A3); else ``non_answer``. For the control arm
    the echo and form classes are empty by construction.
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
    if arm in CONFUSABLE_ARMS:
        label = _confusable_echo_or_form(text, expected, arm)
        if label is not None:
            return label
    elif arm == "rarity_control":
        label = _rarity_echo_or_form(text, expected, arm)
        if label is not None:
            return label
    return "non_answer"


def _confusable_echo_or_form(text: str, expected: str, arm: str) -> str | None:
    """The echo and form classes of a confusable arm, in the
    decision list's order: ``echo_confusable`` when the answer
    is the expected value with this arm's own treatment applied
    to the value word; else ``form_only`` when mapping every
    frozen-table character in the answer back to its Latin
    letter yields exactly the expected value (the frozen-table
    scope of amendment A8(iv)); else ``None``.
    """
    word, _, number = expected.partition("-")
    if text == treated_word(word, arm) + "-" + number:
        return "echo_confusable"
    if text != expected:
        inverse = {replacement: letter for letter, replacement in FROZEN_REPLACEMENTS.items()}
        if "".join(inverse.get(char, char) for char in text) == expected:
            return "form_only"
    return None


def _rarity_echo_or_form(text: str, expected: str, arm: str) -> str | None:
    """The echo and form classes of the rarity arm, in the
    decision list's order: ``echo_control`` when the answer is
    the expected value with the rarity treatment applied to the
    value word; else ``form_only`` under the position-aware
    predicate of amendment A3 (same length, at least one control
    character, and every position either the expected character
    or a control character); else ``None``.
    """
    word, _, number = expected.partition("-")
    if text == treated_word(word, arm) + "-" + number:
        return "echo_control"
    if (
        text != expected
        and len(text) == len(expected)
        and any(char in CONTROL_CHARACTERS for char in text)
        and all(
            char == expected_char or char in CONTROL_CHARACTERS
            for char, expected_char in zip(text, expected, strict=True)
        )
    ):
        return "form_only"
    return None


def decompose_pair(control_cell: dict[str, Any], arm_cell: dict[str, Any]) -> str:
    """The decomposition class of one discordant pair (the
    sibling construction, which is total): verdicts differ
    and answer digests are equal -> ``boundary_only``;
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
    """The frozen headline label (design section 6), a total
    function of (D, J, n_incomplete):

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
    or not either generation completed -- the pinned guard
    domain).
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
    attrition, and the seven-class decomposition; the per-arm
    id report (regime gate); the whole-word dose over the
    panel (eligible letters over all letters in treated
    words, the printed fraction of design section 4); and per
    treated arm the headline over the id-divergent paired
    population -- D, J, ``n_reverse``, ``n_incomplete`` in the
    pinned domain, the paired control and arm pass rates, the
    pair decomposition with the failing cells' classes, the
    arm's ``non_answer`` count over the population, and the
    guarded label. Discordance is never pooled across
    candidates or arms -- this function sees one candidate's
    records only.
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
    dose_eligible = 0
    dose_letters = 0
    for case in cases:
        word, _, _number = str(case["value"]).partition("-")
        for treated in (str(case["topic"]), word):
            dose_eligible += eligible_count(treated)
            dose_letters += len(treated)
    dose = {
        "whole_word": {
            "eligible_letters": dose_eligible,
            "total_letters": dose_letters,
            "fraction": _rate(dose_eligible, dose_letters),
        }
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
        "dose": dose,
        "headline": headline,
        "operating_characteristics": dict(OPERATING_CHARACTERISTICS),
        "n_cases": len(records),
        "n_incomplete": sum(
            1 for record in records if any(_cell(record, arm) is None for arm in ARM_KEYS)
        ),
    }
