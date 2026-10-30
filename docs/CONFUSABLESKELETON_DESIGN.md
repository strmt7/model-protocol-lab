# Confusable-skeleton substitution — design (ADJUDICATED: BUILD-as-amended)

## Round summary (coordinator #74, wave-12 design round 2, 2026-10-30)

Verdict: **BUILD-as-amended**. The candidate's three load-bearing
structures survived adversarial execution: the completion guard —
transferred from the invisible-chars round with its amendments
folded in — was re-attacked with an independent engine and the
coordinator's own state-space enumeration (zero manufactured
fires, zero unstable acquittals over the reachable (D, J, k)
space); the population is real — the frozen 16-letter table, the
zero-eligible audit, and skeleton invariance reproduce under
three independent implementations of UTS #39 §4 against the
pinned 15.0.0 data file, byte-identical (SHA-256) to the
publisher's; and the exact operating characteristics reproduce
digit-for-digit against the invisible-chars round's adjudicated
marginal grid, as they must for the shared rule shape.

What failed is the freeze, in this tree's ordinary classes, plus
one aggravated delivery defect: the delivered candidate file was
damaged — a 17,429-character span of its §2 was missing, replaced
by a literal truncation marker, and with it the entire textual
definition of the rarity-control arm; the executed rarity
assignment then contradicted the surviving text's script-matching
rule through a code-point range proxy; the rarity arm's
`form_only` class was undecidable as written; the paired column's
Monte Carlo understated the exact paired price; the cap's printed
state count ranged over unreachable states; and §8's isolation
reading extended a dose-matched argument to an arm whose dose it
does not match. Every defect is repaired by the amendments below,
which leave the decision rule, the floors, the panel, the
population rule, the headline scorer, and the §7 claim's
falsification condition untouched. This consolidated doc — not the
damaged candidate file — is the build's governing text.

## The amendments (folded into the text at the sections named)

- **A1 (§2).** This doc contains the rarity definition **in
  full**: controls U+0434 CYRILLIC SMALL LETTER DE (д) for the
  Cyrillic seat and U+03B6 GREEK SMALL LETTER ZETA (ζ) for the
  Greek seat; assignment per treated word by the script of that
  word's `single_site` frozen replacement, script determined by
  the `unicodedata.name` prefix (CYRILLIC / GREEK); the per-letter
  control table printed beside the frozen replacement table. The
  build verifies the definition's presence in this doc with a
  named test — the candidate file's damage is the reason the
  check exists.
- **A2 (§2, §3).** The code-point range proxy is replaced by the
  name-prefix rule in the generator; the rarity populations are
  regenerated under the corrected assignment (they differ from
  the design-phase scripts' rarity contents on 6 / 4 / 4 / 4 / 4
  cases per seed — the corrected populations are the frozen
  ones); the build's freeze tests assert, per treated site, that
  the control's script (by name prefix) equals the frozen
  replacement's script.
- **A3 (§5).** The rarity arm's `form_only` is pinned
  position-aware: the answer has the same code-point length as
  the expected value, contains at least one control-set
  character, and replacing every control-set character by the
  expected value's character **at the same position** yields
  exactly the expected value. The grammar battery pins the
  witness `дlder-1234` (vs `alder-1234`) to `form_only`.
- **A4 (§6).** The cap figure is **264** completion-stable fire
  states suppressed at k > 6, over the reachable state space
  **D + k ≤ 24**; the audit's bound is stated in the text.
- **A5 (§8).** The skeleton-specific isolation reading is
  licensed for `single_site` (the dose-matched arm). For
  `whole_word`, a solo fire reads: "skeleton-class substitution
  at whole-word dose disrupts this candidate; rarity at that
  dose is unmeasured by this design" — printed wherever the
  isolation reading is stated.
- **A6 (§6).** The paired column prints the exact bivariate
  values — competent null **0.0044** (0.004442), drop-25
  **0.6981** (0.698078) — and the family-wise paired row is
  computed from the exact rate: **E = 0.3199, P(≥ 1) = 0.2743**
  over 72 comparisons. The design-phase Monte Carlo (0.0036 /
  0.6918) was a faithful execution of its stated model but
  understates the exact price; it is not printed.
- **A7 (§2).** The control property is restated as the one
  relied on: each control is not a source in the pinned file,
  its skeleton is itself, and it is skeleton-distinct from
  every panel letter and every frozen replacement — with ζ's
  six-member skeleton class (itself + five mathematical
  variants U+1D6C7, U+1D701, U+1D73B, U+1D775, U+1D7AF)
  disclosed. The build's population verification asserts
  exactly these properties (not absence from the file).
- **A8 (§5, §6, §7).** (i) §6's worlds bullet carries the
  caveat: design-phase evidence ranged over all incomplete
  cases (no ids dimension — the conservative direction); the
  build's guard battery re-executes the worlds under the
  pinned domain. (ii) The completion sentence discloses that
  the (D+1, J+1) counting is a blanket over-count ranging over
  observation-inconsistent completions for partially-observed
  cases, deliberately conservative; the computation is
  unchanged. (iii) §7's claim sentence gains "under the §6
  guard". (iv) §5's `form_only` gloss states the frozen-table
  scope; echoes in non-frozen skeleton-equal letters decompose
  as `non_answer`, disclosed, never counted toward J. (v) The
  MIN_D plateau sentence gains its exactness qualifier
  (identical at printed precision over [1, 14]; exact plateau
  [1, 6]; the selected pair trails the exact maximum by
  < 3e-11).
- **A9 (§3, §13).** The build's generator freeze tests include
  a constructed colliding draw that forces the redraw path,
  pinning: slots 0+1 redraw at counter+1, neutrals remain at
  counter 0, and the give-up bound's counting as the module
  implements it. The build's OC script reproduces every §6
  figure from its own code, including the exact paired values
  and the N-sweep.

**Framing, stated once and binding on every section:** the
honest claim is **"skeleton-class substitution"** — substitution
inside a standards-defined equivalence class (UTS #39 §4
skeleton equality). The model never sees glyphs. Nothing in this
design claims a model is "fooled like a human", and uniqueness
is claimed for the combination only, never "first".

# The design as amended

## 1. Monday-morning utility

An eval owner deciding, Monday morning, whether frozen benchmark
fixtures and ingested documents need a **confusable-screening
pass** (skeleton-based detection at ingest) gets, per candidate
(template family) and per substitution arm, a number: how often
replacing Latin letters with their UTS #39 skeleton-equivalent
Cyrillic/Greek letters in ordinary generated fact-recall text
changes what a local model answers. If skeleton-class
substitution never moves answers at the frozen floors, the
screening pass is deprioritised for that family; if it fires, the
pass — which is cheap and deterministic, unlike a sanitizer whose
cost the sibling design had to scope away — has a measured reason
to exist.

Scope: the probe measures the *effect* of the substitution, not
the accuracy of any particular screening implementation, and not
human visual confusability. Both exclusions are in the §7 claim's
wording.

## 2. The population definition for "confusable" (the novel point)

**Version pin — decided and justified.** The population is
derived from **`confusables.txt` Version 15.0.0** (downloaded
from unicode.org/Public/15.0.0; SHA-256
`2b10130885c3370b101c52d7baedc452ab7f0e257b86c1e52ee657ecfc29ce64`),
with the DICP set of skeleton step 2 taken from
**`DerivedCoreProperties.txt` 15.0.0** (SHA-256
`d367290bc0867e6b484c68370530bdd1a08b6b32404601b8c7accaf83e05628d`;
executed DICP set size 4,174 code points). Reasons, in order of
weight:

1. **Same-version coherence.** The interpreter that runs the
   build has `unicodedata.unidata_version = 15.0.0` (Python
   3.12.3). Skeleton steps 1/4 (NFD), the script classification
   below (character names), and categories all come from that
   local UCD. Pinning the confusables data to the same version
   makes the whole derivation single-version; a 15.1.0 or 18.x
   data file against a 15.0.0 NFD/name table is a mixed-version
   population no single standard defines.
2. **The cited text and the data are different artefacts.** UTS
   #39 rev 27 (Version 15.1.0 draft 5, Proposed Update,
   2023-08-07) is cited for the *algorithm* — §4's four steps.
   The algorithm text is what the design relies on from rev 27;
   the *data* the algorithm consumes is pinned separately, to
   15.0.0, per reason 1. §4 of rev 27 itself warns "The data may
   change between versions" — skeletons are **not
   version-stable**, which is exactly why the pin must be stated
   and why a future Unicode bump is a population change requiring
   re-derivation, not a silent upgrade.
3. **Latest (18.0.0) rejected:** it would maximise reason 1's
   incoherence for zero design benefit — no arm, floor, or claim
   depends on post-15.0 additions.
4. Sibling precedent: the invisible-chars design pinned its
   property claims to UCD 15.0.0 for the same interpreter reason
   (adjudicated sound).

**Runtime posture (build clarification, cycle 224).** The
derivation above happened once, at design time, under UCD
15.0.0; the shipped module carries the population as frozen
literal tables and file digests — it does not re-derive
anything from the host's UCD at runtime. The runtime audit
records the host's `unicodedata.unidata_version` as read and
checks the frozen characters' names, categories and
normalization forms, every one of which was assigned long
before Unicode 14.0 and is unchanged in 14.0, 15.0 and 15.1;
the audit's verdicts are therefore the same on every
supported interpreter, and only the recorded version string
differs. A future Unicode version that changed one of those
frozen characters' properties would be a population change
under reason 2, caught by the audit going red.

**The skeleton function** (UTS #39 rev 27 §4, verbatim):
skeleton(X) = (1) NFD; (2) remove Default_Ignorable_Code_Point
characters; (3) concatenate per-character prototypes from the
pinned confusables data (single application — §4: mappings are
idempotent, implementations "do not have to recursively apply"
them); (4) NFD. X ≅ Y iff skeleton(X) = skeleton(Y). Executed
checks of the standard against itself: §4's own examples verify
— skeleton("paypal") = skeleton("pаypаl") (mixed-script, U+0430)
and skeleton("scope") = skeleton("ѕсоре") (whole-script) are
both **True** under the pinned 15.0.0 data. Data-file shape,
executed: **6,311 entries; every source is a single code point
(multi-code-point sources = 0)** — so the single-code-point-source
restriction below excludes nothing from the file; the restriction
that bites is on the *target* side and the script rule.

**Eligibility rule (frozen).** A code point `ch` is an eligible
replacement for a panel letter L iff **all** of:

- skeleton(ch) == skeleton(L) under the pinned data (normative
  membership — not visual judgement; see the r/y rows below);
- `unicodedata.name(ch)` starts with `CYRILLIC SMALL LETTER` or
  `GREEK SMALL LETTER` (the slate's cross-script class — UTS
  #39's own worked examples are Cyrillic/Greek; category is Ll
  by construction of the name, verified in the audit output);
- NFC(ch) == NFD(ch) == ch (no canonical decomposition — the
  canonical-form boundary, below);
- NFKC(ch) == ch (not a compatibility fold — the candidate-3
  boundary, below).

**Freeze rule:** one replacement per letter = the **smallest
code point** satisfying the rule. Deterministic, reproducible
from the pinned file alone.

**Executed audit over the whole panel alphabet.** The 24 topics
+ 24 WORDS use **25 distinct letters** (a–z minus x). For each:
skeleton-equal sources in the pinned file range from 13 (k, t)
to 75 (o) — raw skeleton equality admits digits, capitals, math
alphanumerics, symbols, and a dozen scripts. The script rule is
what makes this a homoglyph population rather than a skeleton
grab-bag. Result:

| letter | skeleton-equal | eligible | frozen replacement |
|---|---|---|---|
| a | 23 | 2 | **U+03B1 GREEK SMALL LETTER ALPHA (α)** |
| b | 18 | 0 | — none — |
| c | 21 | 1 | **U+0441 CYRILLIC SMALL LETTER ES (с)** |
| d | 19 | 1 | **U+0501 CYRILLIC SMALL LETTER KOMI DE (ԁ)** |
| e | 19 | 2 | **U+0435 CYRILLIC SMALL LETTER IE (е)** |
| f | 18 | 0 | — none — |
| g | 18 | 0 | — none — |
| h | 17 | 1 | **U+04BB CYRILLIC SMALL LETTER SHHA (һ)** |
| i | 37 | 4 | **U+03B9 GREEK SMALL LETTER IOTA (ι)** |
| j | 17 | 1 | **U+0458 CYRILLIC SMALL LETTER JE (ј)** |
| k | 13 | 0 | — none — |
| l | 73 | 0 | — none — |
| m | 17 | 0 | — none — |
| n | 15 | 0 | — none — |
| o | 75 | 3 | **U+03BF GREEK SMALL LETTER OMICRON (ο)** |
| p | 29 | 2 | **U+03C1 GREEK SMALL LETTER RHO (ρ)** |
| q | 16 | 1 | **U+051B CYRILLIC SMALL LETTER QA (ԛ)** |
| r | 19 | 1 | **U+0433 CYRILLIC SMALL LETTER GHE (г)** |
| s | 20 | 1 | **U+0455 CYRILLIC SMALL LETTER DZE (ѕ)** |
| t | 13 | 0 | — none — |
| u | 27 | 1 | **U+03C5 GREEK SMALL LETTER UPSILON (υ)** |
| v | 29 | 2 | **U+03BD GREEK SMALL LETTER NU (ν)** |
| w | 22 | 2 | **U+0461 CYRILLIC SMALL LETTER OMEGA (ѡ)** |
| y | 30 | 3 | **U+03B3 GREEK SMALL LETTER GAMMA (γ)** |
| z | 16 | 0 | — none — |

- **16 letters frozen; 9 zero-eligible: b, f, g, k, l, m, n, t,
  z.** These are not rare-source letters (l has 73 skeleton-equal
  sources; m, n, z have Armenian/Cherokee/Coptic candidates the
  script rule excludes) — they have no *Cyrillic/Greek
  small-letter* skeleton equivalent in Unicode 15.0.0. Any design
  assuming the slate's five example letters (a/e/o/p/c) suffice
  is wrong in both directions: three of the five freeze to Greek,
  not the slate's Cyrillic (smallest-code-point rule: α U+03B1 <
  а U+0430, ο < о, ρ < р), and eleven further letters are eligible
  that the example never mentions.
- **Normative, not visual — two rows prove the point.** r freezes
  to г (CYRILLIC SMALL LETTER GHE) and y to γ (GREEK SMALL LETTER
  GAMMA): a human reader would not pick either as the lookalike,
  but the pinned file's skeleton makes them equivalents, and the
  population rule is skeleton equality. The names are printed in
  the table precisely so a reviewer judges the real population,
  not the textbook subset (а/е/о/р/с all appear, but they are five
  of sixteen, and р is not among p's frozen pick).
- **Zero-eligible words: none.** Every one of the 48 words (24
  topics + 24 value words) contains at least one eligible letter
  (minimum: `elm`, 1 — executed per-word). Frozen rule for a
  hypothetical zero-eligible word (a future pool edit): it is
  treated as identity in both substitution arms (dose 0 for that
  word), the case stays in the panel, and the generator's screen
  asserts every topic and every WORDS entry has ≥ 1 eligible
  letter — a pool edit violating it fails closed
  (`confusable_skeleton_screen_failed`, §10) rather than silently
  shipping an untreated case. **Population effect under the
  frozen lists, in counts: 0 of 48 words untreated, 0 of 24
  cases dose-free**; the zero-eligible letters' real effect is
  on whole-word *dose fraction* (§4), not on case inclusion.
  Note: `summit` contains none of a/e/o/p/c — but it is not
  untreated: s, u, i are eligible (3 eligible occurrences,
  executed).
- **Skeleton invariance, proved:** for every topic and value
  word, skeleton(treated) == skeleton(original) under **both**
  substitution arms — 48 words × 2 arms, **0 failures** (rarity
  arm: skeleton differs, also executed — §4). Code-point length
  is preserved by substitution (asserted per word).

**The rarity control — the definition in full (amendment A1).**
The rarity arm isolates rarity from skeleton class: at exactly
the `single_site` sites, it substitutes a **same-script,
non-confusable letter** instead of the frozen confusable. There
are exactly two control characters, one per script seat:

- **U+0434 CYRILLIC SMALL LETTER DE (д)** — the control for the
  **Cyrillic seat**;
- **U+03B6 GREEK SMALL LETTER ZETA (ζ)** — the control for the
  **Greek seat**.

**Assignment rule (amendment A2):** the control for a treated
word is chosen **per word**, by the script of that word's
`single_site` frozen replacement — that is, by the script of the
frozen replacement of the word's first eligible letter. Script
is determined by the **`unicodedata.name` prefix** (`CYRILLIC` /
`GREEK`) — §2's own classification method, applied to the frozen
replacement character. (The design-phase generator's code-point
range proxy — U+0400–U+04FF ⇒ Cyrillic — is **wrong** and is not
the rule: the Cyrillic Supplement replacements ԁ U+0501 and ԛ
U+051B fall outside that range and must take the Cyrillic control
д, not ζ. The corrected assignment is the frozen one.)

**Per-letter control table** (the control a letter's site
receives when that letter is the word's first eligible letter;
it is a function of the frozen replacement's script alone):

| letter | frozen replacement (script) | rarity control |
|---|---|---|
| a | α U+03B1 (Greek) | ζ U+03B6 |
| c | с U+0441 (Cyrillic) | д U+0434 |
| d | ԁ U+0501 (Cyrillic) | д U+0434 |
| e | е U+0435 (Cyrillic) | д U+0434 |
| h | һ U+04BB (Cyrillic) | д U+0434 |
| i | ι U+03B9 (Greek) | ζ U+03B6 |
| j | ј U+0458 (Cyrillic) | д U+0434 |
| o | ο U+03BF (Greek) | ζ U+03B6 |
| p | ρ U+03C1 (Greek) | ζ U+03B6 |
| q | ԛ U+051B (Cyrillic) | д U+0434 |
| r | г U+0433 (Cyrillic) | д U+0434 |
| s | ѕ U+0455 (Cyrillic) | д U+0434 |
| u | υ U+03C5 (Greek) | ζ U+03B6 |
| v | ν U+03BD (Greek) | ζ U+03B6 |
| w | ѡ U+0461 (Cyrillic) | д U+0434 |
| y | γ U+03B3 (Greek) | ζ U+03B6 |

**How the controls were chosen (executed search):** per script,
the smallest code point in the script's small-letter range whose
name carries the script's `… SMALL LETTER` prefix, that is
**not a source in the pinned file** (its prototype is itself),
is category Ll, encodes to exactly 2 UTF-8 bytes (matching every
frozen replacement's byte length, so the rarity arm's byte-length
delta equals the confusable arms'), and satisfies NFC == NFD ==
NFKC == identity, with skeleton(control) distinct from the
skeleton of every panel letter. The Cyrillic search lands on
U+0434; the Greek search on U+03B6.

**The property relied on (amendment A7), stated exactly:** each
control (i) is **not a source** in the pinned file — no entry
maps *from* it, so its prototype is itself; (ii) its **skeleton
is itself** (skeleton(д) = д, skeleton(ζ) = ζ, executed); and
(iii) it is **skeleton-distinct from every panel letter and
every frozen replacement** — so a rarity-treated word's skeleton
differs from the original's (executed), which is the arm's whole
point. Two disclosures the property does **not** hide: U+0434 is
wholly absent from the pinned file (neither source nor target —
verified), while **U+03B6 is the target of five entries** — the
mathematical zeta variants U+1D6C7, U+1D701, U+1D73B, U+1D775,
U+1D7AF — so ζ is the prototype of a six-member skeleton class
under this design's own definition of the relation. The class
mates are mathematical alphanumerics, in no arm and no panel
word; the operative separations (i)–(iii) all hold and are
executed. The build's population verification asserts exactly
(i)–(iii) and the stdlib-decidable parts of the choice rule —
never bare absence from the file, which is false for ζ as a
target. An earlier draft's summary ("the controls are
non-confusable") is **not** the claim and must not be repeated:
the claim is (i)–(iii).

**What the control matches, and what it does not (§9 R2):** the
control matches the frozen replacement on script, UTF-8 byte
length (2 bytes), category (Ll), and the (i)–(iii) separation
properties — all checkable, all checked. It does not match
corpus frequency or tokenizer merge rank, quantities this tree
does not measure; the residual frequency confound is disclosed
in §9, not claimed away.

**The sequence question, executed.** All 6,311 pinned sources
are single code points, but targets are not all single: the file
maps U+006D (m) to the two-character prototype "rn", and executed
**skeleton("rn") == skeleton("m") == "rn" is True** — sequence-level
skeleton equivalence exists inside the normative class. **Design
consequence:** v1's population is single-code-point →
single-letter substitution only, so rn→m / m→rn sequence
substitutions are **outside** this probe's population (they change
code-point count — a length-changing mechanism, a different dose
axis). Stated here so no reader infers the probe covers the full
skeleton equivalence relation: it covers its single-letter,
cross-script (Cyrillic/Greek) slice, frozen above.

**Territory boundaries — each verified by execution:**

- **(a) vs canonical-form (§8.54):** every frozen replacement
  and control satisfies NFC(ch) == NFD(ch) == ch and has an
  empty canonical decomposition field. The confusables in this
  population have no canonical decomposition; no arm is an
  NFC/NFD arm.
- **(b) vs the NFKC leg (wave-12 candidate 3, unbuilt):**
  NFKC(ch) == ch executed for **every frozen replacement, both
  controls, and every panel letter**. No arm is silently an
  NFKC fold, and — the regime consequence — an NFKC-normalizing
  pipeline (e.g. SentencePiece `nmt_nfkc`) does **not** fold
  these characters. This probe's ids gate (§6) is therefore
  expected to find divergence where candidate 1's DICP arms
  found identity — a *difference between the sibling probes'
  regimes*, pre-registered in §6.
- **(c) vs invisible-chars (§8.60):** disjoint mechanisms and
  disjoint code points, executed: the frozen replacements are
  all category-Ll **visible letters**; their code-point set's
  intersection with {U+200B, U+FE0F, U+E0001, U+00AD, U+2060} is
  empty; substitution preserves code-point count where insertion
  adds one per site. (Skeleton step 2 removes DICP characters —
  the two probes sit on opposite sides of the skeleton function:
  candidate 1 inserts what skeleton deletes; candidate 2
  substitutes what skeleton equates.)
- **(d) vs perturbation's `unicode_variant` (fullwidth):**
  executed — **14 fullwidth sources** in the pinned file are
  skeleton-equal to a panel letter (e.g. U+FF41 → a). They are
  excluded **by the script-prefix rule** (their names start
  FULLWIDTH, not CYRILLIC/GREEK SMALL LETTER), verified. Reason:
  fullwidth forms are compatibility forms (NFKC-foldable),
  owned by perturbation's operator and by candidate 3's
  territory; admitting them would smuggle the NFKC axis into a
  skeleton probe and break boundary (b).

## 3. Frozen generated material

The sibling panel, unchanged in shape (shared with canonical-form
and invisible-chars; the slate expects shared panel machinery):

- **Cases:** 24, stage-local, generated only, never the user's
  Split. Case *i*: topic = `TOPICS[i]` (harbor, summit, quarry,
  tundra, granite, willow, copper, saddle, walnut, velvet, cobalt,
  marble, cinder, falcon, garnet, saffron, indigo, juniper, lagoon,
  magnet, nickel, pepper, quartz, birch). Value =
  `WORDS[draw] + "-" + number`, WORDS the canonical-form 24-word
  ASCII pool (alder … yew, read from `canonicalform.py` in the
  tree), number ∈ [1000, 9999], pairwise distinct by redraw. Two
  distinct neutrals from density's `NEUTRAL_STATEMENTS` (24 ASCII
  statements). Fact = `The {topic} inspection code is {value}.`
  Question = `What is the {topic} inspection code?` Content =
  `{neutral1} {fact} {neutral2} {question}`. Expected = value.
- **Scorer pin:** every expected value pure ASCII (asserted in
  the evidence run on all seeds); verdicts are `core.evaluate`'s
  byte-exact `answer == case.expected` (the `exact_match`
  conjunction, `core.py:827` in the current tree), conjoined with
  boundary presence as in every sibling stage. Substitutions live
  only in carrier text (the fact sentence); the question, both
  neutrals, and the expected value are byte-identical across
  arms — a model echoing the treated surface is separated by §5.
- **Draws — conversion pinned in text:** material
  `f"{seed}|confusableskeleton|{index}|{slot}|{counter}"`;
  integer `int.from_bytes(sha256(material).digest()[:8], "big")`;
  word = draw % len(WORDS) (slot 0); number = 1000 + draw % 9000
  (slot 1); neutrals = draw % len(NEUTRAL_STATEMENTS) (slots 2, 3;
  second advances +1 mod len on collision). A value collision
  redraws slots 0+1 at counter+1; **give-up bound 1,000
  consecutive redraws**, exhaustion fails the stage closed
  (`confusable_skeleton_screen_failed`). The counting is as the
  module implements it: the counter increments once per colliding
  redraw, and the bound trips when the counter exceeds 1,000 —
  i.e. after 1,001 colliding draws for one case (amendment A9:
  the build's freeze tests force a collision and pin the redraw
  at counter+1, neutrals at counter 0, and this counting).
- **Namespace: NEW** (`confusableskeleton`), not the sibling's
  `invisiblechars`. Each shipped probe's material names its own
  probe (canonicalform / invisiblechars precedent), so panels are
  independent draws: sharing the namespace would make this panel
  case-identical to candidate 1's at the same seed and invite
  cross-probe per-case comparisons the designs never licensed.
- **Panel-size derivation (executed):** under this candidate's
  own null and floors (§6), exact marginal power at the 25-point
  target is N=16: **0.3904**, N=20: **0.6395**, N=24: **0.8159** —
  24 is the smallest of the three house sizes reaching the
  design's 0.80 criterion, and it is the shared sibling size.
  The target effect (25 points) is the decision threshold of
  §1's utility: a screening decision does not turn on a 3-point
  wobble; smaller effects read `below_threshold` (§9), never
  "no effect".

**Generator determinism evidence.** The design-phase generator
implemented the frozen spec twice — string slicing and
code-point-list rebuild (independent site logic) — with
byte-identical populations for seeds {0, 1, 7, 42, 20261030} on
the `control`, `single_site`, and `whole_word` arms (red team B's
from-text implementation matched all three on all five seeds);
the `rarity_control` arm's populations are the ones regenerated
under amendment A2's corrected assignment, and the build's
freeze tests pin those. Byte-divergence: every treated arm
differs byte-wise from control on **24/24 cases, every seed**.
Substitution counts per case (topic + value word): `single_site`
exactly **2** (min = max = 2, every seed — both words always have
an eligible letter, §2); `rarity_control` exactly **2**;
`whole_word` min 4, max 10, mean 7.04–7.54 across seeds.
Exclusions: none, and none silent.

## 4. Frozen substitution sites and dose

**Arms (4, fully crossed):**

| arm | treatment (fact sentence only) |
|---|---|
| `control` | verbatim |
| `single_site` | the **first eligible letter** of the topic and the **first eligible letter** of the value word replaced by its frozen confusable |
| `whole_word` | **every eligible letter occurrence** in both words replaced |
| `rarity_control` | exactly the `single_site` sites, replaced by the per-letter rarity control (§2) |

**Dose, stated honestly.** Per case: `single_site` and
`rarity_control` substitute exactly **2 code points** (1 per
treated word, 2 treated words). `whole_word` substitutes every
eligible occurrence: executed range 4–10 per case; as a fraction,
on seed 0's drawn panel **171 of 249 letters** in treated words
are substituted (**0.6867**) — the remaining 31% are the
zero-eligible letters of §2, which no arm can substitute. The arm
is "whole-word" in *site scope* (both words, all eligible sites),
**not** a claim that every letter changes, and not UTS #39
whole-script confusability (a treated word is mixed-script by
construction whenever it contains a zero-eligible letter). Dose
escalation in response to a null is forbidden (§9).

**Sites.** Substitutions are inside the **fact sentence only**,
in the topic word and in the value's word part (before `-`). The
question and neutrals are byte-identical across arms by
construction (asserted). First-eligible position is a
deterministic function of the frozen table and the word — a
second implementation reproduces it byte-identically. The panel
is pure ASCII before treatment, so positions are code-point
indices equal to byte indices there; substitution preserves
code-point count and changes UTF-8 byte count (+1 byte per
substituted letter: every frozen replacement and both controls
are 2 UTF-8 bytes vs 1 for ASCII — executed; that byte-length
delta is part of the treatment and is matched by the rarity
control, §2).

**Budget (house pattern):** 24 cases × 4 arms × up to 24
candidates = **2,304 generations** worst case; 576 ground
records; house 12 MiB records cap.

## 5. Decomposition (ordered decision list; total)

The byte-exact scorer is the instrument Hua et al.
(arXiv:2509.01790, opened) warn about; no discordance is reported
undecomposed. Classes are an **ordered decision list** (the
literal predicates overlap, so order is part of the definition):
each class is reached only when every earlier predicate fails.

**Cell decomposition** — exactly one class per completed cell:

1. `pass` — verdict passed.
2. `boundary_only` — answer digest == expected digest; the
   boundary conjunct alone failed.
3. `wrong_value` — the answer parses under the frozen value
   grammar `^[a-z]+-[0-9]{4}$` with word ∈ WORDS and number ∈
   [1000, 9999], and is not the expected value. **The only class
   counting toward the headline J (§6).** Scope disclosed: J
   counts wrong values *inside* the frozen grammar; a value
   change outside it (out-of-pool word, leading-zero number)
   decomposes as `non_answer`, is printed per arm, and never
   counts toward J; the OC's q is the **in-grammar** wrong-value
   rate wherever it appears.
4. `echo_confusable` — **scripted per arm.** For the two
   confusable arms: the answer equals the expected value with
   that arm's own treatment applied to the value word (for
   `single_site`, its first eligible letter substituted; for
   `whole_word`, every eligible letter). For `rarity_control`
   the analogous class is named `echo_control` (the arm's
   control-treated expected). For `control` both are empty by
   construction. The model reproduced the treated surface; the
   value it read was correct.
5. `form_only` — additionally requiring answer ≠ expected. For
   the confusable arms: mapping every character of the answer
   that appears in the frozen replacement table back to its
   Latin letter yields exactly the expected value, and the
   answer is not in class 4. **Gloss scoped (amendment A8(iv)):**
   this is the **frozen-table** scope — an echo in a
   *non-frozen* skeleton-equal letter (e.g. U+0430 for a —
   eligible, not the frozen pick) is **not** mapped back and
   decomposes as `non_answer`, disclosed here, never counted
   toward J. For the rarity arm, class 5 is the
   **position-aware** predicate (amendment A3): the answer has
   the same code-point length as the expected value, contains
   at least one control-set character (д or ζ), and replacing
   every control-set character by the expected value's
   character **at the same position** yields exactly the
   expected value. (A fixed-inverse reading is impossible for
   the rarity arm: д stands for the 9 Cyrillic-seat letters and
   ζ for the 7 Greek-seat letters across cases; neither has a
   Latin letter to map back to.)
6. `non_answer` — everything else, including empty answers,
   answers with no parseable value, an answer carrying a
   substituted letter inside an otherwise malformed value, and
   out-of-grammar value changes (class 3's scope note).

**Grammar battery (frozen wrong shapes the tests must pin):**
the correct shape; a *wrong value in the right form*
(`wrong_value`); the *arm-site echo* and an *off-site/mixed
echo* (classes 4/5 — including an answer echoing the whole-word
surface under the single-site arm, which is class 5 there, not
4); the rarity witness **`дlder-1234`** against expected
`alder-1234` (class 5, `form_only`, under amendment A3's
position-aware definition — and `non_answer` under the
single-site arm, whose inverse is the frozen table); an answer
that is another case's value (still `wrong_value` — the grammar
defines the class); *empty* and *boundary-less* outputs
(`non_answer`, and §6 incompletion when no span exists); an
answer mixing scripts in the word part with a wrong number
(`non_answer` — reverse mapping yields a non-matching or
out-of-grammar string); a value with a Cyrillic lookalike digit
position (digits are never substituted; any non-ASCII digit
fails the grammar → `non_answer`). Precedence witnesses: the
exact expected answer (class 1, not 5 — reverse mapping is the
identity on ASCII); a digest-equal answer with a boundary
failure (class 2 before all form classes).

**Pair decomposition** — for discordant pairs, the sibling total
partition: answer digests equal → `boundary_only`; digests
differ, both boundaries found → `exact_only`; otherwise →
`mixed`. Cross-tabulated with the failing cell's class.

## 6. Headline statistic, decision vocabulary, floors, guard

**Regime gate first.** Id divergence measured at encode time on
the FULL render id sequence per (case, candidate), per arm
against control (canonical-form invariant 9 discipline).
Per-arm reporting: `n_ids_identical`, `n_ids_divergent`, the
per-case `id_length_delta` list, `max_abs_id_delta`, and
`n_same_ids_verdict_flips`, per (candidate, arm).
**Pre-registered regime expectation** (stated as expectation,
discovered by measuring ids, never asserted per checkpoint):
unlike candidate 1's DICP arms, these characters are **not**
NFKC-foldable (executed, §2b) and not default-ignorable, so both
NFKC-regime (SentencePiece-class) and byte-level-BPE pipelines
are expected to encode treated text **divergently**; a family
landing ids-identical has a skeleton-screening or
confusable-normalizing tokenizer — itself the §1 decision's most
direct evidence — and is reported instrument-inapplicable for
that arm, never as a pass.

**Headline population** per (candidate, arm): cases completed in
both control and arm **and** id-divergent for that arm. D =
control passes in the population; J = of those, arm cells
decomposed `wrong_value`.

**Decision vocabulary (total function of D, J, and the guard)** —
the tree's existing sibling set only, no invented words:

- `insufficient` — D < 10, **or** the guard withholds.
- `no_discordance_observed` — D ≥ 10, J = 0, guard passes.
- `discordant_asymmetric` — D ≥ 10, J ≥ 6, guard passes.
  (**Fires**: the §7 invariance claim is falsified for this arm.)
- `below_threshold` — D ≥ 10, 0 < J < 6, guard passes.

The population is conditioned on control passes (the
negation / instruction-distance / history-frame / invisible-chars
D-conditioning precedent), so discordance has one direction by
construction and `discordant_balanced` is structurally
unreachable. Reverse discordants are counted and printed beside
every label as `n_reverse`, never pooled into J. The
two-directional canonical rule was rejected by execution in the
sibling round (transplanted floors fire at 0.8897 on the
competent independent null at N = 24 — invisible-chars round,
executed there and reproduced in its adjudication); this design
does not re-litigate it. **No sign test is used anywhere** (the
tree's `density.sign_test_p` is two-sided and not
completion-monotone — the sibling settlement); a descriptive
McNemar figure, if printed, uses `confidentiality.mcnemar_exact_p`
over completed pairs only and is never a decision input.

**Floors — derived under this candidate's own null, frozen
here.** `MIN_D = 10`, `MIN_J = 6`, from this candidate's floor
search (exact enumeration over D ~ Bin(24, pc), J|D ~ Bin(D, q);
grid MIN_D ∈ 1..16 × MIN_J ∈ 2..8, requiring competent-null fire
≤ 0.02 and maximal target power). MIN_J = 6 is set by the search
(in a stronger form than claimed: every MIN_J ≤ 5 pair breaches
the competent-null bound at every MIN_D — minimum null 0.0591);
MIN_D = 10 is **selected from a plateau**: at printed (4 dp)
precision the competent-null and target figures are identical
for every MIN_D ∈ [1, 14] (executed), and 10 is chosen from that
plateau as the sibling-consistent point — stated as a plateau
selection, not a point derivation. **Exactness qualifier
(amendment A8(v)):** the plateau is exact only on [1, 6]; in
exact arithmetic the (10, 6) pair trails the plateau maximum by
3.3e-14 (build-phase correction, verified by exact rational
arithmetic in the cycle-224 test module: the adjudication's
2.8e-11 did not reproduce under either q convention; the
executed gap is 3.329e-14). No floor is borrowed: the search was run against this
candidate's rule and null, and its output happens to land on the
sibling's pair.

**Operating characteristics** (fire = `discordant_asymmetric`).
**Convention, stated in the heading:** the independent column is
**exact enumeration under the marginal convention** — per-case
independence; **q = 1 − pa, the arm's marginal in-grammar
wrong-value rate**, where pa is the arm's marginal pass rate.
The paired column is **exact** — per-case bivariate quadrature
with trinomial summation over the paired population (Gaussian
copula ρ = 0.5), computed independently by red team B, by the
coordinator, and in the sibling round's adjudication for the
same rule shape: three computations, one answer (amendment A6).
Cross-check: the exact independent figures reproduce, digit for
digit, the coordinator-adjudicated marginal grid of the sibling
round for the shared rows — the rule shape is the sibling's, so
the grid must agree, and it does.

| world | independent (exact, marginal) | paired ρ=0.5 (exact) |
|---|---|---|
| competent null (pc .90, q .10) | 0.0173 (0.017274) | **0.004442** |
| weak null (.80 / q .20) | 0.1743 (0.174277) | — |
| random labeller (.50 / q .50) | 0.5503 (0.550306) | — |
| drop 10pt (.90 / pa .80, q .20) | 0.2542 (0.254234) | — |
| **drop 25pt (.90 / pa .65, q .35) — target** | **0.8159 (0.815868)** | **0.698078** |
| drop 25pt at base .85 (.85 / pa .60, q .40) | 0.8761 (0.876145) | — |
| drop 50pt (.90 / pa .40, q .60) | 0.9990 (0.999044) | — |
| sensitivity: 18/24 paired, drop 25pt (marginal) | 0.5214 (0.521448) | — |
| sensitivity: 18/24 paired, competent null | 0.0038 (0.003803) | — |

The weak-null (0.1743) and random-labeller (0.5503) rows are
printed, not hidden: the label is comparative evidence at a
stated base rate, and each candidate's control rate is printed
beside its label so a reader can see which null row applies.

**Family-wise price** — counting **every arm that receives a
label: all 3 treated arms** (`single_site`, `whole_word`, and
`rarity_control`, which is labelled under the identical rule,
§8) × 24 candidates = **72 comparisons**:

| null | per-comparison (unrounded) | E[false labels] | P(≥ 1) |
|---|---|---|---|
| competent independent | 0.017274 | 1.2437 | 0.7148 |
| paired ρ=0.5 (exact, amendment A6) | 0.004442 | **0.3199** | **0.2743** |
| random labeller | 0.550306 | 39.6221 | 1.0000 |

Printed beside the licence in every report.

**Completion guard + incompletion perimeter.** Two channels, as
settled in the sibling round and transferred with its amendments
folded in from the start:

- *(a) Hidden slots* — completed generations whose span does not
  parse are completed cells, decomposed `non_answer` (§5),
  **inside** the perimeter; `n_non_answer` per arm is printed,
  and §9's pre-registered reading governs high-non-answer
  discordance. They never count toward J.
- *(b) Incompletion* — a generation producing no
  boundary-delimited span at all (or a missing cell) has no
  cell; the case leaves the paired population. The guard brings
  it inside the decision perimeter by construction:
  - **Domain pinned:** the worst-case completion and the cap
    range over exactly the incomplete cases **in the pair's
    population domain** — cases whose control and arm renders
    are id-divergent for that arm (an encode-time property,
    decidable whether or not the generation completed) and
    which have at least one missing cell. Ids-identical cases
    never enter the completion, in either direction.
  - *Fire cannot be manufactured:* J and D are absolute counts
    over completed cases; no completion raises observed J, and
    the guarded fire uses observed counts only.
  - *Acquittal cannot be manufactured:* an acquittal label
    (`no_discordance_observed` / `below_threshold`) issues
    **only if stable under the worst-case completion** — every
    incomplete case in the domain counted as control-pass +
    arm `wrong_value` (D+1, J+1 each). **Disclosure (amendment
    A8(ii)):** this (D+1, J+1) counting is a deliberate blanket
    over-count — it ranges over observation-inconsistent
    completions for partially-observed cases (a case with an
    observed control-fail cell is still counted as completing
    control-pass); it is conservative in exactly the direction
    the guard exists for, and the computation is the one all
    the evidence ran. Consequences disclosed: fires issue on
    observed counts (incompletion present does not by itself
    withhold a fire); `no_discordance_observed` issues only at
    n_incomplete = 0; `below_threshold` survives only while
    J + n_incomplete < 6.
  - **Cap, derived not copied:** if incomplete cases in the
    domain exceed **6** of 24 for a pair, the label is
    `insufficient` regardless. Derivation: 6 is the largest
    incomplete count leaving paired ≥ 18 — 18 being the
    smallest paired count whose operating characteristics this
    design prints (the sensitivity rows above). The cap is a
    paired-population quality floor; its only marginal effect
    in the exhaustive audit below is suppressing
    completion-stable fires, disclosed as such.
  - *Executed attack on this guard:* **exhaustive (D, J, k)
    audit** over the reachable state space — **0
    manufactured-fire violations, 0 unstable-acquittal
    violations**; fires issue at every k ∈ [0, 6]; the cap
    suppresses **264 completion-stable fire states at k > 6
    over the reachable state space D + k ≤ 24** (amendment A4;
    a wider D + k ≤ 25 bound counts 345 but ranges over 81
    unreachable states) — its only marginal effect. **World A**
    (zero-effect null; incompletion concentrated on concordant
    pass/pass cells, removed w.p. 0.5): guarded fire **0.0055**,
    unguarded 0.0186 (≈ the exact null 0.0173 — manufacture
    fails); guarded acquittals collapse into withholding
    (`insufficient` 0.9923). **World B** (true drop, q = 0.50;
    60% of would-be J cells removed): unguarded fire 0.2551
    with `below_threshold` at **0.7277 — a manufactured
    acquittal**; guarded fire 0.1731 with `below_threshold` at
    **0.0117** and 0.8152 withheld — displaced mass is
    withholding, not acquittal. **World C** (same drop, flat
    incompletion 0.15): guarded fire **0.8893** vs unguarded
    0.9399 — the guard costs little when incompletion is not
    sign-concentrated. (10,000 trials per world, seed 20261030.)
    **Caveat (amendment A8(i)):** these design-phase worlds
    range over *all* incomplete cases — they carry no ids
    dimension, so their k counts a stricter guard than the
    pinned text's (the conservative direction); the build's
    guard battery re-executes the worlds under the pinned
    id-divergent domain.
  - *Cost, printed:* in high-incompletion worlds the guard
    withholds aggressively (World A guarded `insufficient`
    0.9923). That is the chosen direction: incompletion buys
    silence, never a manufactured verdict in either direction;
    the precise label behaviour is the bullets above, not the
    slogan.

## 7. Pre-registered falsifiable claim

> Over generated fact-recall cases (§3) whose control and
> treated renders are **id-divergent** (§6 gate), for a given
> candidate and substitution arm, the arm's verdicts on
> control-passing cases show no in-grammar value-changing
> discordance at or above the frozen floor: conditioning on the
> id-divergent paired population, J < 6 whenever D ≥ 10,
> **under the §6 guard** (amendment A8(iii) — at k > 6 the
> sentence's antecedent labels are withheld without falsifying).

- **Falsified** for a (candidate, arm) exactly when its guarded
  label is `discordant_asymmetric`. That is the whole
  falsification condition; a fire proves only that *this* arm's
  skeleton-class substitutions moved *this* checkpoint's answers
  to *wrong in-grammar values* on *this* generated population —
  nothing about human visual confusion, weights, attacks, or
  other scripts/code points.
- **Not falsified ≠ robust.** `below_threshold` and
  `no_discordance_observed` acquit only at the printed OC: a
  10-point marginal drop fires only 0.2542 of the time.
  Ids-identical arms are instrument-inapplicable for that
  family, reported as such, never as passes.
- The claim is per-arm and per-candidate. **No pooled headline
  and no winner readout in v1** (sibling precedent).

## 8. Isolation rows

- The `rarity_control` arm is generated and classified under the
  identical rule and **receives its own label** (it is counted
  in the §6 family-wise table for exactly that reason). Its
  null behaviour is an isolation row: under the null process its
  fire rates are the §6 table's rows, printed per candidate in
  the report.
- **Pre-registered isolation reading (scoped by amendment A5):**
  a `single_site` fire is read as skeleton-class-specific only
  if `rarity_control` does not also fire for the same candidate
  — the rarity arm matches `single_site`'s dose exactly (2
  substitutions), so the reading is dose-matched there. If both
  fire, the reading is "non-Latin substitution at these sites
  disrupts this candidate" (a rarity/tokenization finding), and
  the confusable arm adds no skeleton-specific evidence. If the
  rarity arm fires and no confusable arm does, the panel has
  measured disruption that skeleton equivalence *avoids* —
  also decision-relevant for §1, printed as such. **For
  `whole_word` the isolation reading does not extend:** its
  dose (4–10 substitutions) has no dose-matched rarity baseline
  anywhere in this design, so a whole-word solo fire reads
  exactly: **"skeleton-class substitution at whole-word dose
  disrupts this candidate; rarity at that dose is unmeasured by
  this design"** — printed wherever the isolation reading is
  stated, including the report section.
- **Dose isolation:** `single_site` vs `whole_word` is a dose
  contrast at fixed mechanism (2 vs 4–10 substitutions, same
  table, same words); a fire only at `whole_word` is read as
  dose-dependent, not as a different phenomenon.
- Non-target arms/families under the null process: every arm
  notionally treated with a null substitution behaves as the
  competent/paired rows; the report's per-arm table (3 treated
  arms × candidates, each with label, D, J, `n_reverse`,
  `n_non_answer`) is itself the isolation grid. Cross-arm
  comparisons use only the uniformly-defined classes
  (`wrong_value`, `non_answer`, `boundary_only`); echo/form
  classes are reported per-arm as descriptive (the rarity arm's
  echo class is `echo_control`, not `echo_confusable`).

## 9. Risks and pre-registered readings

- **R1 — Attack endpoints ≠ benign effect.** Boucher et al.
  measure attacks (a single imperceptible injection
  significantly reduces vulnerable systems' performance; three
  break most); Sarabamoun's homoglyph figures (§14, caveated)
  are jailbreak ASR under a heuristic criterion. This probe
  substitutes **unoptimized, standards-frozen letters on benign
  recall tasks**. The benign effect size is genuinely unknown;
  a null is a legitimate pre-registered outcome, and **dose
  escalation in response to a null is forbidden**: no "add
  Armenian/Cherokee letters and retry" inside this probe — the
  zero-eligible letters stay zero-eligible in v1; broadening
  the script rule is a new design.
- **R2 — Rarity confound, honestly bounded.** The §2 control
  matches script, UTF-8 byte length, category, and the
  amendment-A7 separation properties — all checkable. It does
  not match corpus frequency or tokenizer merge rank, quantities
  this tree does not measure; "comparable rarity" in this
  design **means the checkable properties**, and the residual
  frequency confound is disclosed here rather than claimed away.
  The §8 isolation reading is written to survive it: a joint
  fire is reported as rarity/tokenization, not skeleton.
- **R3 — Mixed-script side effects.** Whole-word substitution
  changes script-run structure; some tokenizers treat script
  runs specially. Reported per family via the ids gate; never
  pooled blindly — there is no pooled headline to pool into (§7).
- **R4 — Partial whole-word dose.** §4's 0.6867 letter fraction
  (seed 0) is printed, not hidden: if `whole_word` nulls, the
  reading is "skeleton substitution at ~69% of letters in two
  carrier words did not fire", not "words are robust to
  confusables". The fraction is a per-panel executed quantity
  and is printed in the report beside the arm.
- **R5 — Version instability.** §2's pin is load-bearing: a
  Unicode version bump changes the population (UTS #39's own
  caveat). The build pins the data version in its case material
  (the table digest binds the frozen table and the pinned files'
  SHA-256 digests, §10) so a silent data change is a digest
  change, not a result change.
- **R6 — Scorer artefact.** Without §5 this probe counts a
  model echoing Cyrillic surface as a capability change; with
  it, the headline counts only `wrong_value`. The residual risk
  is masking via silence (`non_answer`): printed per arm, with
  the pre-registered reading that an arm whose discordance is
  dominated by `non_answer` is reported as "substitution
  disrupts output form", a distinct finding — never silently
  acquitted, never pooled into J.
- **R7 — Fully specified panel.** The panel's instruction /
  question / contract strings are frozen verbatim (§3) so the
  instrument measures the substitution, not ambiguity (the
  slate's framing result for all wave-12 candidates).

## 10. Fail-closed vocabulary, record shape, naming

- Whole-stage failure: `{"available": False, "detail":
  "confusable_skeleton_unavailable"}`. Stage details:
  `cancelled`, `confusable_skeleton_record_budget_exceeded`,
  `confusable_skeleton_screen_failed` (sibling pattern).
  Per-cell details: `""`, `arm_context_exceeded`,
  `arm_evaluation_failed`. Rates are None only when the
  denominator is 0. Reports render counts and rates only,
  every payload-derived string through `_escape_text`.
- Record per (candidate, case): fixture id, topic, expected
  digest, per-arm content digests (4), per-arm token counts
  and render-id digests, `ids_identical` per arm vs control,
  `id_length_delta` per arm, the frozen-table version pin
  (confusables data 15.0.0 + the table's own digest), cells
  and cell details per arm. Content digests per arm are
  mandatory (canonical-form invariant 10): the diff validator's
  mirror regenerates all four arm contents from the seed and
  checks each digest. The mirror decomposes pairs in the
  sibling pattern and recomputes skeleton equality for treated
  words from the pinned table it carries.
- **Naming audit (executed in the design round, real parser):**
  census via `cli.build_parser()` = **81 distinct long options**
  (at the cycle-223 tree); `--confusable-skeleton`,
  `--confusables`, `--confusable-substitution`,
  `--skeleton-substitution`, `--homoglyphs`, and
  `--confusable-chars` are all **FREE**; `allow_abbrev=False`
  on all 17 parsers walked. Shipped probe flags name the axis
  population in hyphenated noun form (`--canonical-form`,
  `--invisible-chars`, `--record-notation`). **Chosen:
  `--confusable-skeleton`** — it names the standards-defined
  class that defines the population (§2), in the shipped
  register; it avoids `--homoglyphs` (the human-visual framing
  §1/§7 forbid as the claim, and a nickname for the axis),
  `--confusables` (names the data file, not the equivalence),
  and `--confusable-substitution` / `--skeleton-substitution`
  (name the operation; every shipped flag names what varies,
  and what varies here is the skeleton class member, the
  substitution being the constant mechanism).
  USAGE section: **§8.61**, following §8.60 invisible
  characters. Module: `model_protocol_lab/confusableskeleton.py`;
  group `confusable_skeleton`; stage details as above.

## 11. Duplication check (verified against the tree)

Read from `docs/USAGE.md` §8.1–§8.60 (60 probes): no shipped
probe substitutes cross-script skeleton-equivalent letters.
Nearest neighbours, and why they are not this:
`canonical-form` (§8.54) is canonical equivalence (NFC/NFD) of
authored text — boundary (a) executed in §2 shows this
population is disjoint from it (no replacement decomposes);
invisible-chars (§8.60) inserts DICP code points — boundary
(c), disjoint mechanism and code points; `perturbation`
(§8.15) blends a fullwidth *compatibility* operator into a
noise rate — boundary (d), fullwidth excluded by rule, and no
paired classification or ids gate there; candidate 3 (NFKC
leg, unbuilt) owns compatibility foldables — boundary (b)
executed so no arm trespasses. Parked set (document-syntax,
instruction-language, directive, multi-question) shares no
axis with skeleton-class substitution; multi-question's
contribution is its settlement obligation, discharged in §6.
Uniqueness is claimed for the **combination only** — frozen
UTS #39 skeleton-class substitution arms × paired id-divergent
classification with derived floors × completion-guarded labels
× digest-committed treatments on local checkpoints — never
"first".

## 12. Rejected alternatives

- **Broadening the script rule** (any non-Latin small letter):
  executed consequence — f, g, n, z gain Armenian/Cherokee
  replacements (U+0584, U+0581, U+0578, U+AB93), b/k/l/m/t stay
  zero-eligible anyway, and the population stops being the
  class UTS #39's own examples and the slate define
  (Cyrillic/Greek), mixing scripts whose rarity/tokenization
  profiles differ from the frozen class. Rejected for v1;
  R1 forbids adopting it as a dose-escalation response.
  (Completeness note from the review: under the broadened rule
  j's smallest pick would also shift, to U+03F3; the list above
  is illustrative of the consequence class, not exhaustive.)
- **Sequence substitutions (rn↔m):** normatively real
  (executed, §2) but length-changing; a different dose axis.
  Outside v1's population by the single-code-point rule.
- **Mathematical alphanumeric / fullwidth confusables:**
  skeleton-equal in the file (dozens per letter) but they are
  compatibility forms — NFKC territory (candidate 3) and
  perturbation's width axis; boundary (b)/(d) discipline
  excludes them.
- **Uppercase arms:** the panel is lowercase by construction;
  capital-letter confusables are a second population (several
  panel letters' richest sources are capitals — l's 73 include
  U+0049/U+0406 class sources). A capital arm doubles the
  family-wise price to answer a question §1 does not ask.
- **Two-directional canonical classification:** rejected by
  execution in the sibling round (0.8897 transplant fire, §6).
- **Borrowed floors:** forbidden by the slate's method note;
  §6's floors are derived under this candidate's own null.
- **Sign-test headline:** two-sided, not completion-monotone;
  this design needs no test (§6).
- **A screening/detection arm in v1** (run a skeleton screener
  over the treated text and measure *it*): that measures an
  implementation, not a model; §1's decision needs the model
  effect first. Separate candidate if ever.
- **Winner readout:** §7 — not consumed by the utility;
  abstention-dominated at these rates.
- **A whole-word rarity arm** (considered in review, rejected):
  it would make §8's isolation reading dose-matched at
  whole-word dose, at the price of a fourth labelled arm and a
  changed family-wise count — fixing a prose scope by changing
  the instrument. Amendment A5 scopes the reading instead.

## 13. Build cost and build debts (discharged in cycle 224)

By sibling analogy (invisible-chars scale, adjusted for 4 arms
instead of 7 and a carried data table instead of a code-point
list): one new module
`model_protocol_lab/confusableskeleton.py` (~600–800 lines,
including the frozen 16-letter table, its per-letter control
table, and the 15.0.0 pin), stage wiring in `experiments.py`,
the diffreports mirror, the reporting section, USAGE §8.61,
and a test module — roughly 1,800–2,600 lines including tests.
Build debts the design and review create, all discharged by
named tests in the build cycle:

1. The A1 doc-completeness check (this doc carries the rarity
   definition in full — U+0434 / U+03B6 present, no truncation
   markers in shipped docs).
2. The A2 regenerated rarity populations with per-site
   script-agreement freeze tests.
3. The A3 battery witness (`дlder-1234` → `form_only`).
4. The A8(i) guard battery with the ids dimension,
   re-executing worlds A–C under the pinned domain at the
   frozen cap.
5. The A9 forced-collision freeze test and the OC reproduction
   script (every §6 figure, exact paired values and N-sweep
   included).
6. Generator asserts (expected ASCII; every topic/WORDS entry
   ≥ 1 eligible letter; skeleton invariance per treated word)
   as freeze tests.

## 14. Sources (every entry opened in the design round)

- UTS #39, Unicode Security Mechanisms, **rev 27**
  (unicode.org/reports/tr39/tr39-27.html — opened; header
  executed from the fetched text: Version **15.1.0 (draft 5)**,
  Proposed Update, dated 2023-08-07; it is a draft, cited for
  the algorithm text only, never for data). §4, read verbatim:
  skeleton(X) = NFD → remove Default_Ignorable_Code_Point →
  concatenate per-character prototypes → NFD; X ≅ Y iff
  skeleton(X) = skeleton(Y); the mixed-script example
  "paypal"/"pаypаl" (U+0430) and whole-script example
  "scope"/"ѕсоре"; the data-file format (source ; target ; MA,
  field 3 obsolete); mappings idempotent ("implementations do
  not have to recursively apply"); "The data may change
  between versions." The population *data* is pinned to
  15.0.0 (§2), not to this draft's version.
- Pinned data files (downloaded in the design round, SHA-256
  verified against the publisher's in the build cycle):
  `confusables.txt` (Version: 15.0.0, Date: 2022-08-26) and
  `DerivedCoreProperties.txt` (15.0.0) from unicode.org/Public
  — every population number in §2 is derived from these files
  by the design round's executed population script.
- UAX #15, Unicode Normalization Forms (unicode.org/reports/
  tr15/, rev. 58 / Unicode 18.0.0 text, opened): §1.1
  canonical vs compatibility equivalence — the boundary
  discipline of §2 (a)/(b): canonical equivalence is the
  fundamental same-abstract-character relation; compatibility
  equivalence is the weaker, appearance-variant relation NFKC
  folds. Used only for what this probe is not; all NFKC
  *values* in §2 are executed against the local 15.0.0
  interpreter, not taken from the live text.
- Boucher, Shumailov, Anderson, Papernot, "Bad Characters:
  Imperceptible NLP Attacks", arXiv:2106.09898 (v2, 11 Dec
  2021) — opened at arxiv.org/abs/2106.09898. Homoglyph
  substitution is one of its four imperceptible attack
  classes; a single imperceptible injection (invisible
  character, homoglyph, reordering, or deletion) significantly
  reduces vulnerable systems' performance, three break most.
  Attack endpoints, §9 R1.
- Sarabamoun, "Special-Character Adversarial Attacks on
  Open-Source Language Models", arXiv:2508.14070 — cited
  **only via the ar5iv rendering**
  (ar5iv.labs.arxiv.org/html/2508.14070, opened; homoglyph
  section located in the opened text), with all three
  caveats: **single author, arXiv-only, heuristic
  (keyword/policy-pattern) success criterion**. The figures
  used: homoglyph attacks 42.1–58.7% success depending on
  script combination, cross-script highest at 58.7%, up to
  92% on mistral:7b (ar5iv §4.1/Figure 2) — jailbreak attack
  success rates, cited as attack endpoints under §9 R1, never
  as benign-effect evidence. Per the slate's verified version
  trap, these percentages are not cited to arXiv HTML v2.
- Hua et al., "Flaw or Artifact? Rethinking Prompt Sensitivity
  in Evaluating LLMs", arXiv:2509.01790 (v1, 1 Sep 2025) —
  opened (abstract): much reported prompt sensitivity stems
  from heuristic evaluation including rigid answer matching
  across 7 LLMs / 6 benchmarks / 12 templates. The §5
  decomposition obligation.
- Tree: `core.py:827` (byte-exact `exact_match`);
  `density.py:790` (`sign_test_p`, two-sided — not used);
  `confidentiality.py:295` (`mcnemar_exact_p`, descriptive
  only); `canonicalform.py` (WORDS pool, population
  precedents); `density.py` NEUTRAL_STATEMENTS;
  `perturbation.py:104-107` (`unicode_variant` fullwidth
  operator — boundary (d)); `docs/USAGE.md` §8.1–§8.60
  (duplication ground truth); sibling design + adjudication
  in `~/workspace/agents_out/invisiblechars/` (guard
  settlement, A1–A10 amendment classes, the 0.8897 transplant
  figure); this round's record in
  `~/workspace/agents_out/confusables/` (candidate as
  delivered — damaged, see the round summary — adjudication,
  lane briefs, evidence scripts, pinned data files).

**Removals:** none required — every source the design relies
on resolved when opened. Corrections to the slate/tasking on
contact with the data, recorded in place: the slate's example
letters freeze to Greek for a/o/p under the deterministic rule
(§2); "whole-word" is a site-scope term, not a full
substitution (9 zero-eligible letters, §2/§4);
`summit` is treated despite lacking a/e/o/p/c (§2); the rn↔m
sequence equivalence is real and is *excluded by rule*, not
absent (§2); and candidate 1's NFKC-regime ids-identity
expectation does **not** transfer to this population (§2b/§6).
