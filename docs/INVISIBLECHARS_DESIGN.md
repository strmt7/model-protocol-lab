# Invisible-character insertion — design (ADJUDICATED: BUILD-as-amended)

## Round summary (coordinator #72, wave-12 design round 1, 2026-10-30)

- Candidate of record: `~/workspace/agents_out/invisiblechars/candidate.md`;
  adjudication: `~/workspace/agents_out/invisiblechars/ADJUDICATION.md`.
  Lanes: designer (design + executed evidence), red team A (mechanism
  disproof), red team B (numbers and freeze audit), verifier (script
  re-runs + citation resolution) — blind to each other; the coordinator
  adjudicated by its own execution (`coord_*` scripts). Evidence scripts
  in `~/workspace/agents_out/invisiblechars/scripts/`.
- **Verdict: BUILD-as-amended.** The completion guard survived red team
  A's exhaustive attack (118,430 latent removal patterns; 0 manufactured
  fires, 0 manufactured acquittals); the population verifies against the
  real UCD 15.0.0 file; the fabrication sweep is clean (9/9 citations,
  both halves of the Sarabamoun ar5iv/v2 trap executed). Amendments
  A1–A10 below are part of the design; none touches the decision rule,
  the floors, the panel, the arms, the headline scorer, or the §7 claim.
- This document is the build text of record: the candidate with A1–A10
  folded in (built in cycle 223 as `model-protocol-lab run
  --invisible-chars`).

## The amendments (folded into the text at the sections named)

- **A1 (freeze completed; §3).** The draw conversion is pinned exactly:
  `int.from_bytes(sha256(material).digest()[:8], "big")`; word =
  draw % len(WORDS); number = 1000 + draw % 9000; neutrals =
  draw % len(NEUTRAL_STATEMENTS); and a frozen redraw give-up bound
  (1,000 consecutive collision redraws for one case) whose exhaustion
  fails the stage closed with `invisible_chars_screen_failed`.
- **A2 (one convention, honest figures; §§3, 6, 7).** The independent
  column is recomputed under the marginal convention, stated in the
  table's heading: drop 10pt 0.2542; drop 25pt 0.8159 (base .90) /
  0.8761 (base .85); drop 50pt 0.9990; the 18/24 sensitivity drop row
  0.5214. Paired power prints as the exact bivariate 0.6981 (the
  designer's MC printed 0.6974). §3's size derivation is restated on
  the marginal figures; §7's 10-point figure is 0.2542.
- **A3 (guard domain pinned; §6).** The worst-case completion and the
  incompletion cap range over exactly the incomplete cases in the
  pair's population domain: id-divergent cases (an encode-time
  property) with at least one missing cell. Ids-identical cases never
  enter the completion, in either direction.
- **A4 (guard prose = the executed rule; §6).** Fires issue on
  observed counts; acquittal labels are stability-gated;
  `no_discordance_observed` requires zero incomplete cases in the
  domain; `below_threshold` requires J + n_incomplete < 6; the cap is
  a paired-population floor — withhold all labels when incomplete
  cases in the domain exceed 6 (paired < 18, the smallest paired count
  with printed OC). The build re-executes worlds A–C at the amended
  cap (cycle-223 tests).
- **A5 (decomposition is an ordered decision list; §5).** Classes are
  reached in the order pass → boundary_only → wrong_value →
  echo_invisible → form_only → non_answer, each only when every
  earlier predicate fails, with `form_only` additionally requiring
  answer ≠ expected. The battery gains the precedence witnesses.
- **A6 (visible arm honesty; §§5, 8).** The battery's example is the
  site-produced shape (`al-der-1234`); `form_only` is empty by
  construction for `visible_hyphen`; cross-arm readings compare only
  the uniformly-defined classes.
- **A7 (wrong_value scope disclosed; §§5, 6, 7).** J counts wrong
  values inside the frozen value grammar; out-of-grammar value
  changes decompose as `non_answer`, are printed per arm, and never
  count toward J; the OC's q is named the in-grammar wrong-value rate.
- **A8 (floor honesty; §6).** MIN_J = 6 is set by the floor search;
  MIN_D = 10 is selected from the plateau [9, 16] over which every
  printed OC figure is identical; the pair sits on the search's
  optimal frontier.
- **A9 (sentence-level truthfulness; §§1, 2, 4, 6, 14).** The
  combining-mark exclusion is by the frozen list (the predicate alone
  admits U+034F); World A is described as executed (first-half
  concentration); the UTS #39 step is internalSkeleton step 2; the
  generator asserts are build obligations pinned by freeze tests; the
  family-wise table gains the all-labelled 144-comparison row and
  unrounded per-comparison rates.
- **A10 (evidence completeness; §§3, 6).** The shipped OC reproduction
  (the cycle-223 test module) produces the §3 N-sweep and every §6
  figure from its own code; the adjudicated values are the ones
  printed here.

---

# The design as amended

## 1. Monday-morning utility

A team piping web/RAG text into a local checkpoint decides, Monday
morning, whether to add a pre-ingest sanitizer stripping invisible
code points, using this probe — per candidate (template family) and
per invisible arm — to learn whether a single invisible insertion
changes the model's answers on ordinary generated tasks at all.

**Scope correction to the slate's second half.** The slate's utility
sentence also promises "whether stripping them costs anything on
clean text". The arm set as specified **cannot answer that**: there
is no strip arm, and no arm applies a sanitizer to anything. What is
true by construction — and enforced by the shipped generator's
assert battery (§3; content ASCII and five-free, pinned by the
build's freeze tests, amendment A9(iv)) — is the
degenerate half: a sanitizer removing exactly the five frozen code
points is the identity function on the control arm's text, because
the control text contains none of them — so on *this panel's* clean
text stripping costs nothing, trivially. The real-world cost question
(stripping legitimate invisible code points — ZWJ sequences in emoji
and in scripts such as Persian, where the joiners this design
*excludes* in §2 carry meaning) is outside this instrument's
population and is **not** claimed. The headline claim in §7 is scoped
to the first half only. A future strip-cost arm (sanitizer applied to
naturally contaminated text) is a separate candidate, not an
amendment to this one.

## 2. The population definition for "invisible" (the novel point)

"Invisible" is **not** defined by a property sweep. It is defined by
a frozen list, each member of which is shown to satisfy a stated
normative property, and the generator enforces membership by assert
against the list — decidable, never authorial care.

**Frozen arm list** (Unicode 15.0.0, the version of the local
interpreter's data: `unicodedata.unidata_version = 15.0.0`, Python
3.12.3; property claims below are pinned to 15.0.0 because, per
UTS #39's own caveat, derived properties are not stable across
Unicode versions):

| arm key | code point | name | General_Category | ccc | Default_Ignorable_Code_Point (UCD 15.0.0) |
|---|---|---|---|---|---|
| `zwsp` | U+200B | ZERO WIDTH SPACE | Cf | 0 | Yes |
| `vs16` | U+FE0F | VARIATION SELECTOR-16 | Mn | 0 | Yes |
| `tag` | U+E0001 | LANGUAGE TAG | Cf | 0 | Yes |
| `softhyphen` | U+00AD | SOFT HYPHEN | Cf | 0 | Yes |
| `wj` | U+2060 | WORD JOINER | Cf | 0 | Yes |

Executed by `scripts/designer_properties.py`: category, combining
class, and name verified against local `unicodedata` for every row.
DICP is **not** exposed by Python's `unicodedata`; the DICP column is
a cited UCD-file fact (DerivedCoreProperties.txt, 15.0.0), and the
generator assert enforces only the locally decidable predicate —
`cp ∈ FROZEN_LIST ∧ category(cp) ∈ {Cf, Mn} ∧ ccc(cp) = 0 ∧ cp is
assigned` — so the build never depends on a property the stdlib
cannot check. The shared normative property is: **Default_Ignorable
with no combining behaviour** — a conformant default rendering may
ignore the code point entirely, and it never composes with a base
character. Per-arm notes, stated before any run:

- `vs16` (U+FE0F) is the Gao et al. mechanism class (variation
  selectors), frozen to one representative. After an ASCII letter it
  has no defined variation sequence, which is exactly the Gao
  condition (their Figure 2 point: selectors do not influence the
  appearance of standard alphabetic characters).
- `softhyphen` (U+00AD) is the weakest member and is labelled as
  such: it is DICP=Yes and Cf, but a renderer *may* show a hyphen at
  a line-break opportunity. Inside a word mid-line (the frozen sites
  of §4 are never at a line boundary) it is rendering-invisible in
  the ordinary case. If a red team judges it disqualifying, dropping
  it is a population change, not an amendment — recorded here so the
  judgement is made on the text.
- `tag` (U+E0001) represents the U+E0000 block; the tag characters
  are deprecated as a mechanism but remain assigned Cf code points.

**Excluded, with reasons** (each exclusion is by list; several
excluded code points satisfy the same category predicate, which is
why the definition cannot be the predicate alone — executed list in
`designer_properties.py`):

- U+200C ZWNJ / U+200D ZWJ: real joining semantics — they change
  correct rendering in Arabic-script and Indic text and in emoji
  sequences. Their "invisibility" is script-conditional; including
  them would confound invisibility with joining behaviour.
- U+FEFF: byte-order-mark semantics; its word-break behaviour
  differs (it is also ZERO WIDTH NO-BREAK SPACE) and it is a
  signature, not an insertion, in real pipelines.
- Bidi controls U+202A–U+202E and U+2066–U+2069: a different axis —
  direction overrides. They do not merely hide; they reorder
  displayed text. Display-side concerns own them (the tree's report
  sanitizers already strip them); pooling them with zero-advance
  code points would mix two mechanisms in one headline.
- Combining marks generally (e.g. U+034F COMBINING GRAPHEME
  JOINER): excluded **by the frozen list**, and the sentence says so
  plainly (amendment A9(i)): most combining marks have ccc ≠ 0, but
  the predicate alone does *not* exclude the class — U+034F is Mn,
  ccc = 0, DICP = Yes, so a property-only definition would wrongly
  admit it. List membership is the exclusion; the predicate is a
  check on the list's members, never a membership rule. This is the
  concrete reason the definition is a **list verified against
  properties**, not a property.
- U+061C ARABIC LETTER MARK, U+180E MONGOLIAN VOWEL SEPARATOR:
  direction/script-conditional visibility; same axis objection as
  the bidi controls.

**Visible control.** `visible_hyphen`: U+002D HYPHEN-MINUS
(category Pd — it does *not* satisfy the predicate, executed) at
exactly the same sites, same dose. It is the isolating control:
any insertion disrupts tokenization at some rate; an invisible arm
is evidence about invisibility only against this arm's behaviour
(§8, isolation rows).

## 3. Frozen generated material

Panel in the negation/numform shape, mirroring canonical-form's
fact-recall construction (the sibling instrument whose population
this panel reuses in ASCII):

- **Cases:** 24, stage-local, generated only, never the user's
  Split. Case *i*: topic = `TOPICS[i]` from a frozen 24-word ASCII
  list (harbor, summit, quarry, tundra, granite, willow, copper,
  saddle, walnut, velvet, cobalt, marble, cinder, falcon, garnet,
  saffron, indigo, juniper, lagoon, magnet, nickel, pepper, quartz,
  birch — all length ≥ 4, so the §4 sites always exist). Value =
  `WORDS[draw] + "-" + number`, WORDS the canonical-form 24-word
  ASCII pool, number ∈ [1000, 9999], pairwise distinct across cases
  by redraw. Two distinct neutral sentences from density's frozen
  `NEUTRAL_STATEMENTS` (ASCII). Fact =
  `The {topic} inspection code is {value}.` Question =
  `What is the {topic} inspection code?` Content =
  `{neutral1} {fact} {neutral2} {question}`. Expected = value.
- **Scorer pin (canonical-form invariant 2 analogue):** every
  expected value is pure ASCII, enforced by the generator
  (`expected.isascii()` asserted in the evidence run). Verdicts are
  `core.evaluate`'s byte-exact `answer == case.expected` (the
  `exact_match` conjunction at `core.py:827` in the current tree),
  conjoined with boundary presence as in every sibling stage.
  Inserted code points live only in carrier text (fact sentence),
  never in the expected value — a model echoing the treated fact
  flips its verdict with zero value change, which is precisely
  what the §5 decomposition exists to separate.
- **Draws (conversion pinned, amendment A1):** SHA-256 derive,
  material `f"{seed}|invisiblechars|{index}|{slot}|{counter}"`,
  converted to an integer by exactly
  `int.from_bytes(sha256(material.encode("ascii")).digest()[:8],
  "big")`. Slots: 0 = value word (`WORDS[draw % len(WORDS)]`),
  1 = number (`1000 + draw % 9000`) — a value collision against an
  earlier case redraws both slots at counter+1; 2/3 = neutral
  indices (`draw % len(NEUTRAL_STATEMENTS)`; the second advances
  +1 mod len(NEUTRAL_STATEMENTS) on collision). **Redraw give-up
  bound (A1):** 1,000 consecutive collision redraws for one case
  exhaust the derivation; generation then fails and the stage
  fails closed with `invisible_chars_screen_failed`. (The bound
  is unreachable under the frozen derivation — 24 draws from a
  216,000-value space — and exists so the generator is provably
  total, with its failure inside the frozen vocabulary.)
- **Panel size derivation (restated on the marginal figures,
  amendment A2).** 24 cases is not asserted; it is the size at
  which the §6 rule, derived under this candidate's own null in
  §6, reaches its printed OC: exact marginal power at the target
  effect (a 25-point drop, §6) under the independent null is
  0.3904 at N = 16, 0.6395 at N = 20, and **0.8159** at N = 24
  (base .90; 0.8761 at base .85), and 0.6981 under the paired
  null. (The design round's first computation, under a
  concentrated-dependence convention, printed 0.5100 for N = 16
  and 0.8956 for N = 24; the N-sweep under both conventions is
  produced by the shipped OC reproduction, amendment A10.) 24 is
  the smallest house-standard size reaching the design's own
  0.80 criterion. The target effect — 25 points — is the decision
  threshold of the utility: a sanitizer decision does not turn on
  a 3-point wobble, and the design says in advance (§9) that
  smaller effects read `below_threshold`, not "no effect".

**Generator determinism evidence (B-3 standard).**
`scripts/designer_generator.py` implements the frozen spec twice —
implementation A by string slicing on the assembled fact,
implementation B by tokenising the fact on spaces and inserting via
code-point lists (independent site logic) — and asserts
byte-identical populations for seeds {0, 1, 7, 42, 20261030}:
**A == B on all five seeds**. Byte-divergence: every non-control
arm differs byte-wise from control on **24/24 cases**, with exactly
**48 inserted code points per arm** (2 sites × 24 cases), on every
seed. Exclusions: none, and none silent — the assert fails the run
if any case/arm deviates. The question string is byte-identical
across arms for every case (asserted).

## 4. Frozen insertion sites and dose

**Dose:** single insertion per site; two sites per case; every
treated arm (invisible and visible) inserts at both sites. The
slate's "single-insertion dose" is honoured per site; the two-site
total (2 code points per case) is stated here, not hidden, because
a one-site reading would halve the treatment the OC in §6 was
derived for. Dose escalation in response to a null is forbidden
(§9).

**Site algorithm** (a second implementation can reproduce it
byte-identically — implementation B in the evidence script does):

1. Sites are inside the **fact sentence only**. The question and
   both neutral sentences are byte-identical across arms by
   construction.
2. *Topic site:* in the fact, the topic is the token after
   `The ` (fact code-point offset 4). Insert after code-point
   index 2 of the topic (i.e. between its 2nd and 3rd letters).
3. *Payload site:* in the fact, the value is the last token; its
   word part precedes `-`. Insert after code-point index 2 of
   the word part.
4. Insertions are applied in reverse site order (payload site
   first) so offsets never shift — an implementation detail the
   evidence script's first run proved load-bearing (applying
   topic-first shifted the payload offset by one and A ≠ B caught
   it; the fix is in the shipped script and this sentence).

**Boundaries.** The panel is pure ASCII before treatment, so no
site can fall inside a multi-byte character: insertion indices are
code-point indices into ASCII words, equal to byte indices there.
Every topic has length ≥ 5 and every value word length ≥ 3 (WORDS
minimum is "elm"/"urn", length 3 — index 2 exists: insertion after
the 2nd code point of a 3-letter word is between its 2nd and 3rd
letters). The shipped generator's screen asserts length ≥ 3 for
both site words (a build obligation under amendment A9(iv), pinned
by the build's freeze tests); a future word-list edit violating it
fails the screen closed
(`invisible_chars_screen_failed`, §10), never silently relocates
a site.

**Crossing:** arms × cases fully crossed — every arm treats every
case at both sites. No rotation: rotation would make per-arm
populations differ across cases and break the paired design.

**Budget (house pattern):** 24 cases × 7 arms × up to 24
candidates = **4,032 generations** worst case; 576 ground records;
house 12 MiB records cap. (Arms: `control`, `zwsp`, `vs16`, `tag`,
`softhyphen`, `wj`, `visible_hyphen`.)

## 5. Decomposition (total functions; "Flaw or Artifact?" discipline)

The byte-exact scorer is exactly the instrument Hua et al.
(arXiv:2509.01790) warn about: rigid answer matching counts
formatting wobbles as capability changes. No discordance in this
probe is ever reported undecomposed.

**Cell decomposition (an ordered decision list, amendment A5)** —
exactly one class per completed cell, decidable from the answer
digest plus the case's frozen strings (the record carries the
answer digest and the answer text bound by digest, the
record-notation amendment pattern, so the validator recomputes
every class). The classes are an **ordered list**: a cell takes
the first class whose predicate holds, each class reached only
when every earlier predicate fails — the sibling modules'
decomposition discipline, stated here because the literal
predicates overlap without it (an exact expected answer satisfies
the un-ordered `form_only` predicate, since removal is the
identity on a code-point-free string):

1. `pass` — verdict passed.
2. `boundary_only` — answer digest == expected digest; the boundary
   conjunct alone failed. A boundary effect, never an insertion
   effect (recordnotation `decompose_cell` discipline).
3. `wrong_value` — the answer parses under the frozen value grammar
   `^[a-z]+-[0-9]{4}$` with word ∈ WORDS and number ∈ [1000, 9999],
   and is not the expected value (guaranteed at this point in the
   list: a text equal to the expected value has its digest, so it
   was taken by class 1 or 2). **This is the only class that
   counts toward the headline J (§6).** Its scope is the frozen
   grammar, disclosed wherever J is glossed (amendment A7): a
   value change outside the grammar — an out-of-pool word
   (`zebra-1234`), a leading-zero or out-of-range number
   (`alder-0123`) — is a genuine value change that decomposes as
   `non_answer`, is printed per arm, and never counts toward J;
   the operating characteristics' q is the *in-grammar*
   wrong-value rate.
4. `echo_invisible` — the answer equals the expected value with the
   arm's own code point inserted at the §4 payload site (for the
   control arm this class is empty by construction — it has no
   code point). The model reproduced the treated surface; the
   value it read was correct. Steps 4–5 run on the five invisible
   arms only (amendment A6): the visible arm's site echo
   (`al-der-1234`) is a hyphenated string outside the answer
   grammar and classifies `non_answer` at step 6.
5. `form_only` — removing every occurrence of the arm's code point
   from the answer yields exactly the expected value, the answer
   is not `echo_invisible`, **and the answer is not the expected
   value** (the side condition that, with the ordering, makes the
   list a partition). Code point echoed at a different position
   or multiplicity. **For `visible_hyphen` this class is empty by
   construction** (amendment A6): its character is the grammar's
   separator, so stripping every hyphen from an answer can never
   yield a value whose separator is a hyphen; off-site hyphen
   echoes land in `non_answer` where the invisible arms'
   analogues land here. Cross-arm readings (§8) therefore compare
   only the uniformly-defined classes (`wrong_value`,
   `non_answer`, `boundary_only`); the echo/form classes are
   reported per arm as descriptive.
6. `non_answer` — everything else, **including** empty answers and
   answers with no parseable value. Completed-but-unparsed cells
   live here, inside the perimeter (§6 guard), never outside it.

**Grammar battery (frozen wrong shapes).** The battery the tests
must pin includes, beyond the correct shape: a *wrong value in the
right form* (`wrong_value`); the *site echo* and *off-site echo*
above; *empty* and *boundary-less* outputs (`non_answer`, and §6's
incompletion when no span exists at all); an answer carrying the
*visible* hyphen arm's insertion read as part of the value — the
shape the frozen site actually produces is `al-der-1234`
(insertion after code-point index 2 of "alder"; corrected,
amendment A6): it fails the value grammar → `non_answer`, never
`wrong_value`, because the grammar admits exactly one hyphen in
the word–number join and this string's parse has the hyphen inside
the word part; an answer that is another case's value (still
`wrong_value` — the grammar, not case identity, defines the
class); and the **precedence witnesses** (amendment A5): the exact
expected answer with a failed boundary (taken by `boundary_only`,
never `form_only`, although removal is the identity on it); a site
echo carrying an additional off-site code point (`form_only` by
multiplicity, not `echo_invisible`); and a discordant pair whose
control answer is the exact expected value while the arm answer
carries the code point off-site (the pair decomposition reads
`exact_only` when both boundaries are found — the digests differ,
so it is never misread as `boundary_only`).

**Pair decomposition** — for discordant pairs (verdicts differ),
the recordnotation total partition, mirrored: answer digests equal
→ `boundary_only`; digests differ, both boundaries found →
`exact_only`; otherwise → `mixed`. Cross-tabulated with the
failing cell's cell class in the report, so a reader can see
whether `exact_only` discordance is value change or form wobble
without trusting the headline.

## 6. Headline statistic, decision vocabulary, floors, guard

**Regime gate first.** Id divergence is measured at encode time on
the FULL render id sequence per (case, candidate), per arm against
control (canonical-form invariant 9). Per-arm reporting
(history-frame discipline): `n_ids_identical`, `n_ids_divergent`,
the per-case `id_length_delta` list (arm tokens − control tokens),
and `max_abs_id_delta`, per (candidate, arm). A same-ids verdict
flip is a determinism finding (`n_same_ids_verdict_flips`), never
insertion fragility. **Pre-registered regime split** (from the
slate addendum, stated as expectation, discovered by measuring
ids, never asserted per checkpoint): SentencePiece pipelines
normalise *before* tokenization with `nmt_nfkc` by default
(SentencePiece normalization doc, opened), and UTS #39's skeleton
step treats Default_Ignorable code points as removable, so for
SentencePiece/NFKC-regime families the DICP arms are expected to
land ids-identical (instrument-inapplicable); byte-level BPE
without a normalizer passes the same code points through as
distinct bytes, so ids are expected to diverge. `vs16` after an
ASCII letter has no NFKC decomposition, so its regime behaviour is
the least certain — which is a reason it is in the panel, not a
reason to predict it.

**Headline population** per (candidate, arm): cases completed in
both control and arm **and** id-divergent for that arm. Ids-
identical arms are reported instrument-inapplicable per family,
never as passes. D = control passes in the population; J = of
those, arm cells decomposed `wrong_value`.

**Decision vocabulary (total function of D, J, and the guard):**

- `insufficient` — D < 10, **or** the guard withholds (§6 guard).
- `no_discordance_observed` — D ≥ 10, J = 0, guard passes.
- `discordant_asymmetric` — D ≥ 10, J ≥ 6, guard passes.
  (**Fires**: the §7 invariance claim is falsified for this arm.)
- `below_threshold` — D ≥ 10, 0 < J < 6, guard passes (an observed
  null at this panel's resolution, not proof of invariance).

Justified variant, stated plainly: this is the canonical-form
label family with `discordant_balanced` structurally unreachable,
because the population is conditioned on control passes (the
negation / instruction-distance / history-frame D-conditioning
precedent), so discordance has one direction by construction.
Reverse discordants (control fails, arm passes, among paired
cases) are counted and printed beside every label as
`n_reverse`, with the mirrored rule applied descriptively; they
are never pooled into J. The two-directional canonical rule was
**rejected by execution, not preference**: at N = 24 the
transplanted floors (paired ≥ 6, discordant ≥ 2) fire at
**0.8897** on the competent independent null (exact enumeration,
`scripts/designer_oc.py` — the identical figure record-notation's
shipped operating characteristics print for the same transplant,
an independent cross-check of the enumeration), and even at
discordant ≥ 8 it retains only 0.4995 power at a 25-point drop.
The D-conditioned rule dominates it on both axes at this N.

**Floors — derived under this candidate's own null, frozen here.**
`MIN_D = 10`, `MIN_J = 6` (`scripts/designer_oc2.py`: exact
enumeration over D ~ Bin(24, pc), J|D ~ Bin(D, q); the paired
null by the exact bivariate computation of the per-case Gaussian
copula at ρ = 0.5, cross-checked by MC, 20,000 trials, seed
20261030). Floor honesty (amendment A8): `MIN_J = 6` is set by the
floor search; `MIN_D = 10` is **selected from the plateau
[9, 16]** over which every printed OC figure below is identical —
a plateau choice on the search's optimal frontier, not a point
derivation. No floor is borrowed: placement's (6, 2) and negation's
(8, 5) were both computed against in the floor search and rejected
by the numbers above/below.

**Operating characteristics** (fire = `discordant_asymmetric`;
every power figure prints its marginals). **Convention (amendment
A2, stated once for the whole table):** the independent column is
exact enumeration under per-case independence, D ~ Bin(N, pc),
J|D ~ Bin(D, q), where **q is the arm's marginal in-grammar
wrong-value rate** (amendment A7) — under independence the
conditional rate given a control pass equals the marginal rate, so
a drop row is fixed by the two marginals (control pass rate,
arm pass rate) alone. The paired column couples the control and
arm outcomes *within each case* by a Gaussian copula at ρ = 0.5
(cases independent), computed exactly through the bivariate
orthant probabilities; the arm's failure mass on control-passing
cases is counted as wrong-value (share 1.0, the conservative
reading). The design round's first table mixed conventions — its
drop rows held the joint flip mass pc·q constant, a
concentrated-dependence figure printed as "independent"
(0.8956 at both bases, 0.3441 at 10 points); those figures are
superseded by the marginal column below and survive only as a
dependence sensitivity, never as the headline power.

| world | independent (exact, marginal) | paired ρ=0.5 (exact) |
|---|---|---|
| competent null (control .90, arm wrong-value .10) | 0.0173 | 0.0044 |
| weak null (.80 / .20) | 0.1743 | — |
| random labeller (.50 / .50) | 0.5503 | — |
| drop 10pt (.90 / .80; q = .20) | 0.2542 | — |
| **drop 25pt (.90 / .65; q = .35) — target effect** | **0.8159** | **0.6981** |
| drop 25pt at base .85 (.85 / .60; q = .40) | 0.8761 | — |
| drop 50pt (.90 / .40; q = .60) | 0.9990 | — |
| sensitivity: 18/24 paired, drop 25pt | 0.5214 | — |
| sensitivity: 18/24 paired, competent null | 0.0038 | — |

The weak-null row (0.1743) and the random-labeller row (0.5503)
are printed, not hidden: a candidate passing only 80% of control
cases with independent arm noise fires spuriously about one time
in six, so the label is comparative evidence at a stated base
rate, and the report prints each candidate's control rate beside
its label so the reader can see which null row applies.

**Family-wise price** (5 invisible arms × 24 candidates = 120
comparisons; the visible arm is a control, not a comparison —
but the report also prints a label for it under the identical
rule (§8), so the all-labelled count is 144; both rows print,
with unrounded per-comparison rates, amendment A9(v)):

| null | comparisons | per-comparison (unrounded) | E[false labels] | P(≥ 1) |
|---|---|---|---|---|
| competent independent | 120 | 0.017274 | 2.07 | 0.8764 |
| competent independent, all labelled | 144 | 0.017274 | 2.49 | 0.9187 |
| paired ρ=0.5 | 120 | 0.004442 | 0.53 | 0.4139 |
| random labeller | 120 | 0.550306 | 66.04 | 1.0000 |

(The design round's table printed the paired P as 0.4144: that
figure is computed from the MC point estimate 89/20000 = 0.00445,
not from the exact rate. From the exact bivariate rate 0.004442
the reproducible figure is 0.4139, printed here; the licence's
decision-relevant quantity, the expectation 0.53, is unchanged.)

Printed beside the licence in every report, in the multi-question
discipline. The decision-relevant rows are the competent and
paired ones; the random row is a noise floor, not a model of a
checkpoint.

**Completion guard + incompletion perimeter (the decisive
settlement).** Two channels, two dispositions:

- *(a) Hidden slots* — completed generations whose span does not
  parse: these are completed cells, decomposed `non_answer`
  (§5), **inside** the perimeter. They count as arm failures for
  rates, never toward J (they are not value changes), and they
  cannot hide: `n_non_answer` per arm is printed, and a
  pre-registered reading (§9) governs high-non-answer discordance.
- *(b) Incompletion* — a generation producing **no
  boundary-delimited span at all** (or a missing cell) has no
  cell; the case leaves the paired population. This is the
  channel that defeated the multi-question guard (coordinator
  #70 adjudication, read in full): absent ≡ removed for any
  decision reading only completed pairs. This design brings
  incompletion **inside the decision perimeter by construction
  (path 1b)**, not by assumption:
  - *Fire cannot be manufactured:* J and D are absolute counts
    over completed cases. No completion of an incomplete case
    can raise the *observed* J, and the guarded fire uses the
    observed counts only. Incompletion removes evidence; it
    cannot create a wrong-value discordant.
  - *Acquittal cannot be manufactured:* an acquittal label
    (`no_discordance_observed` / `below_threshold`) is issued
    **only if it is stable under the worst-case completion** —
    every incomplete case in the domain completed as control-pass
    + arm `wrong_value` (D+1, J+1 each). If the worst-case
    completion changes the label, the label is `insufficient`
    (withheld), with `n_incomplete` printed.
  - *Completion domain (pinned, amendment A3):* the worst-case
    completion and the incompletion cap range over exactly the
    incomplete cases **in the pair's population domain** — cases
    whose control and arm renders are id-divergent for that arm
    (an encode-time property of the render, decidable whether or
    not the generation completed) and which have at least one
    missing cell. Ids-identical cases never enter the completion,
    in either direction: they could never enter the population in
    any completion. (The design-phase evidence ranged over all
    incomplete cases, the conservative direction; the build's
    guard battery re-attacks the worlds under the pinned domain
    with an ids dimension.)
  - *The cap is a paired-population floor (amendment A4):* if
    incomplete cases in the domain exceed **6** of 24 — paired
    population below 18, the smallest paired count whose OC this
    section prints (the sensitivity rows) — every label for the
    pair is `insufficient`, fires included. The cap is a
    population-quality floor, not an acquittal safeguard (the
    stability rule already withholds every unstable acquittal);
    its only marginal effect beyond the stability rule is
    suppressing completion-stable labels, fires included, on
    thin populations — stated here because the design round's
    first text placed the cap inside the acquittal bullet and
    underived the constant (8 in that text; 6 by the derivation
    above).
  - *Consequences of the stability rule, disclosed (A4):*
    `no_discordance_observed` issues only at n_incomplete = 0
    (any incomplete case in the domain completes to J ≥ 1 under
    the worst case, changing the label); `below_threshold`
    survives only while J + n_incomplete < 6. Fires, by contrast,
    issue on the observed counts whenever D ≥ 10 and J ≥ 6 and
    the cap holds — incompletion present does not by itself
    withhold a fire. The true summary sentence is the narrow one:
    **incompletion cannot manufacture a verdict** — it can only
    remove evidence, and on the acquittal side removal is priced
    as withholding. (The design round's first text printed the
    broader slogan "incompletion buys silence, never a verdict
    in either direction"; its own label function falsifies that
    slogan — fires issue with 1–6 incomplete cases present — and
    the slogan does not ship.)
  - *Executed attack on this guard* (`scripts/
    designer_incompletion.py`, 10,000 trials per world, frozen
    seeds; re-executed by the build at the amended cap, cycle-223
    tests): **World A** (zero-effect null; incompletion at rate
    1.0 on the arm cells of concordant control-pass cases **in
    the first half of the panel** — the executed concentration,
    stated exactly, amendment A9(ii): maximally shrinking D
    while leaving J untouched): unguarded fire 0.0168 ≈ the
    exact null 0.0173 (manufacture fails, as constructed);
    guarded fire 0.0095 at the design-phase cap and **0.0014 at
    the amended cap**, the balance withheld. **World B** (true
    drop, conditional wrong-value rate 0.50; realised-sign
    concentrated incompletion removing 60% of would-be J cells):
    unguarded fire collapses 0.9408-class to **0.2551 with
    `below_threshold` at 0.7285 — a manufactured acquittal**;
    guarded `below_threshold` stays **0.0133** at both caps (the
    stability rule, not the cap, does that work), guarded fire
    0.2356 at the design-phase cap and 0.1731 at the amended cap,
    the displaced mass withheld (0.8136). **World C** (same drop,
    flat incompletion 0.15): guarded fire 0.9355 vs unguarded
    0.9408 at the design-phase cap; 0.8932 at the amended cap,
    guarded `below_threshold` 0.0013 — the guard costs little
    when incompletion is not sign-concentrated, and what it
    costs at the amended cap is withholding (0.1055), never
    acquittal.
  - *Cost, printed:* in high-incompletion worlds the guard
    withholds aggressively (World A guarded `insufficient`
    0.9986 at the amended cap). That is the design's chosen
    direction, stated by the narrow sentence above.
  - No sign test is used anywhere in this design, so the
    one-sided/two-sided completion-monotonicity question (the
    adjudication's (6,17) 0.0347 → (6,16) 0.0525 counterexample,
    against the tree's two-sided `density.sign_test_p`,
    `density.py:790`) does not arise. The descriptive McNemar
    figure, if printed, uses `confidentiality.mcnemar_exact_p`
    (`confidentiality.py:295`) over completed pairs only and is
    never a decision input.

## 7. Pre-registered falsifiable claim

> Over generated fact-recall cases (§3) whose control and treated
> renders are **id-divergent** (§6 gate), for a given candidate
> and invisible arm, the arm's verdicts on control-passing cases
> show no value-changing discordance at or above the frozen
> floor: conditioning on the id-divergent paired population,
> J < 6 whenever D ≥ 10.

- **Falsified** for a (candidate, arm) exactly when its guarded
  label is `discordant_asymmetric`. That is the whole
  falsification condition; nothing else falsifies it, and a fire
  proves only that *this* arm moved *this* checkpoint's answers
  to *wrong values inside the frozen grammar* (amendment A7 —
  out-of-grammar value changes decompose as `non_answer` and are
  outside this claim) on *this* generated population — nothing
  about weights, attacks, or other code points.
- **Not falsified ≠ robust.** `below_threshold` and
  `no_discordance_observed` acquit only at the printed OC: a
  10-point drop fires only 0.2542 of the time (marginal
  convention, amendment A2). Ids-identical
  arms are instrument-inapplicable for that family, reported as
  such, never as passes. A run landing 100% ids-identical has
  measured the tokenizer's normalizer, full stop.
- The claim is per-arm and per-candidate. There is **no pooled
  headline and no winner readout in v1** (negation/record-notation
  precedent: the sanitizer decision is per checkpoint, and a
  candidate "passes" an arm only by passing every case, a
  readout that would be abstention-dominated at these rates).

## 8. Isolation rows

- The visible-hyphen arm is generated and classified under the
  identical rule. Its null behaviour *is* an isolation row
  (executed, `designer_oc2.py`): competent independent fire
  0.0173, paired 0.0044 — the same numbers as any arm under the
  null, printed as its own row in the report. Cross-arm
  comparisons of decomposition classes use only the
  uniformly-defined classes (`wrong_value`, `non_answer`,
  `boundary_only`): the visible arm's `form_only` is empty by
  construction (§5, amendment A6), so echo/form rates are
  reported per arm as descriptive, never compared across arms.
- **Pre-registered isolation reading:** an invisible arm's fire
  is read as invisibility-specific only if the visible arm does
  not also fire for the same candidate. If both fire, the
  reading is "insertion at these sites disrupts this candidate"
  (a tokenization-fragility finding), and the invisible arm adds
  no invisibility-specific evidence. If the visible arm fires
  and no invisible arm does, the panel has measured insertion
  disruption that invisibility *avoids* — also a decision-relevant
  outcome for the sanitizer question, printed as such.
- Non-target arms/families under the null process: every arm
  notionally "treated" with a null code point behaves as the
  competent/paired rows above; the report's per-arm table is
  itself the isolation grid (6 treated arms × candidates, each
  with its own label, D, J, `n_reverse`, `n_non_answer`).

## 9. Risks and pre-registered readings

- **R1 — Attack endpoints ≠ benign effect.** Boucher et al. and
  Gao et al. measure attacks (performance destruction on
  vulnerable NLP systems; jailbreak ASR via *optimized suffixes*
  of many selectors). This probe inserts **one code point per
  site, unoptimized, on benign recall tasks**. The benign effect
  size is genuinely unknown; a null result is a legitimate
  pre-registered outcome — a replication-style null in the
  tree's history-frame sense — and **dose escalation in response
  to a null is forbidden**: no "insert five and retry" inside
  this probe. A follow-up dose candidate is a new design.
- **R2 — Boucher's own negative result.** Boucher et al. (§VII-B,
  via the slate addendum's full-text read) found invisible-
  character attacks had **no effect** on their toxic-content,
  NER, and sentiment models, attributed to tokenizers ignoring
  the characters. Tokenizer-dependent nulls are the expected
  texture of this probe's results, not an anomaly — the regime
  gate (§6) exists to make them legible.
- **R3 — Regime shrinkage.** If most families land ids-identical
  for most arms, the classifiable population shrinks; the
  18/24-paired sensitivity rows (§6) price this in advance.
- **R4 — Scorer artefact.** Without §5, this probe is the
  "Flaw or Artifact?" cautionary tale with extra steps; with it,
  the headline counts only `wrong_value`. The residual risk is
  the opposite masking: a model that goes silent (`non_answer`)
  under insertion cannot fire J. Mitigation is disclosure, not
  a rule change: `n_non_answer` discordance is printed per arm,
  and a pre-registered reading states that an arm whose
  discordance is dominated by `non_answer` is reported as
  "insertion disrupts output form", a distinct finding from
  value change — never silently acquitted, never pooled into J.
- **R5 — Fully specified panel.** Per Pecher et al.
  (arXiv:2602.04297), underspecification inflates apparent
  sensitivity; the panel's instruction/question/contract strings
  are frozen verbatim (§3) so the instrument measures the
  insertion, not ambiguity.
- **R6 — Soft hyphen.** See §2: the weakest "invisible" member,
  labelled in advance.

## 10. Fail-closed vocabulary, record shape, naming

- Whole-stage failure: `{"available": False, "detail":
  "invisible_chars_unavailable"}`. Stage details: `cancelled`,
  `invisible_chars_record_budget_exceeded`,
  `invisible_chars_screen_failed` (placement pattern). Per-cell
  details: `""`, `arm_context_exceeded`,
  `arm_evaluation_failed`. Rates are None only when the
  denominator is 0. Reports render counts and rates only, every
  payload-derived string through `_escape_text`.
- Record per (candidate, case): fixture id, topic, expected
  digest, per-arm content digests (7), per-arm token counts and
  render-id digests, `ids_identical` per arm vs control,
  `id_length_delta` per arm, cells and cell details per arm.
  Content digests per arm are mandatory (canonical-form
  invariant 10): the diff validator's mirror regenerates all
  seven arm contents from the seed and checks each digest, so
  the treatment itself — not just token counts — is committed.
  The diffreports mirror decomposes pairs in the
  instructiondistance/recordnotation pattern.
- **Naming audit.** Census executed through the real parser
  (`cli.build_parser()`, `designer_properties.py`): **80
  distinct long options**; `--invisible-chars`,
  `--invisible-characters`, `--ghost-chars`, and
  `--invisible-insertion` are all FREE; `allow_abbrev=False` on
  the root parser. (No `docs/reference/cli.md` exists in the
  current tree — the CLI reference lives in `docs/USAGE.md` §8;
  the tasking's path was checked and is absent, recorded here
  rather than silently substituted.) Shipped probe flags name
  the *axis population* in hyphenated noun form
  (`--canonical-form`, `--history-frame`, `--prune-sweep`,
  `--record-notation`, `--instruction-distance`). **Chosen:
  `--invisible-chars`** — it names the population (§2), matches
  the shipped pattern's length and register, and avoids:
  `--invisible-characters` (same meaning, four syllables of
  pure length); `--ghost-chars` (a nickname, not a population —
  it would be the only slang flag in the parser); and
  `--invisible-insertion` (names the operation, not the axis;
  every shipped flag names what varies, and what varies here is
  which invisible character, the insertion being the constant
  mechanism). USAGE section: §8.60, following §8.59
  record-notation. Module: `model_protocol_lab/
  invisiblechars.py`; group `invisible_chars`; stage details as
  above.

## 11. Duplication check (verified against the tree)

Read from `docs/USAGE.md` §8.1–§8.59 (59 probes, executed listing
in this session): no shipped probe inserts invisible code points
into model inputs. Nearest neighbours, and why they are not this:
`perturbation` (§8.15) blends a fullwidth *compatibility* operator
into a noise rate — visible characters, informational readout, no
paired classification, no ids gate (the canonical-form design's
differentiation argument applies verbatim); `deliforge` (§8.26)
is reserved-token vs subword *authority* for forged markers;
`canonical-form` (§8.54) is canonical equivalence of authored
text, disjoint population (its characters are visible and
decomposable); the refusals detector and report sanitizers touch
format characters detection/display-side and measure no model
behaviour. Parked set: document-syntax, instruction-language,
directive, and multi-question share no axis with insertion of
rendering-invisible code points into a payload; multi-question's
contribution to this design is its *settlement obligation* (§6),
not its subject matter. Uniqueness is claimed for the
**combination only** — frozen invisible-insertion arms × paired
id-divergent classification with derived floors × completion-
guarded labels × digest-committed treatments on local
checkpoints — never "first".

## 12. Rejected alternatives

- **Two-directional canonical classification:** rejected by
  execution (§6: 0.8897 transplant fire; 0.4995 power at the
  retained floor).
- **Borrowed floors (placement 6/2, negation 8/5):** forbidden
  by the slate's method note and rejected by the floor search.
- **Sign-test headline:** the tree's sign test is two-sided
  (`density.py:790`); the multi-question adjudication proved
  the two-sided form is not completion-monotone, and this
  design needs no test — its rule is a count rule with exact
  enumerated OC.
- **Strip arm in v1:** it would answer §1's second half, but a
  strip arm over *this* panel is degenerate (identity on clean
  text, §1); the non-degenerate version needs a contaminated-
  text population, a different instrument.
- **Joiner/bidi arms:** §2 — different mechanisms; pooling them
  would make the headline uninterpretable.
- **Winner readout:** §7 — not consumed by the utility;
  abstention-dominated at these rates.
- **User-fixture arm:** shipped fixtures are pure ASCII and a
  fixture arm would insert into user data; generated-only is
  the sibling discipline (canonical-form invariant 3).

## 13. Build cost (estimate, executed in cycle 223)

By sibling analogy (canonical-form / record-notation scale):
one new module (~700–900 lines), stage wiring in
`experiments.py`, diffreports mirror, reporting section,
USAGE §8.60, and a test module — roughly 2,000–2,800 lines
including tests. An estimate for sequencing only, from analogy,
not measurement; cycle 223's build record (its final report)
states the measured cost. The build additionally owes, beyond
the ordinary gate: the A3 guard battery with an ids dimension;
the A4 re-execution of worlds A–C at the amended cap (executed;
figures in §6); the A9(iv) generator asserts pinned by freeze
tests; and the A10 OC reproduction
(`tests/test_cycle223_invisiblechars.py`) reproducing the
adjudicated figures from its own code.

## 14. Sources (every entry opened in this session)

- Boucher, Shumailov, Anderson, Papernot, "Bad Characters:
  Imperceptible NLP Attacks", arXiv:2106.09898 (v2, 11 Dec
  2021) — opened at arxiv.org/abs/2106.09898. Single
  imperceptible injection significantly reduces performance of
  vulnerable systems; three break most. Attack endpoints, §9 R1.
- Gao, Li, Du, Wang, Ma, Xia, Pang, "Imperceptible Jailbreaking
  against Large Language Models", arXiv:2510.05025 (v1, 6 Oct
  2025) — opened at arxiv.org/abs/2510.05025. Appended variation
  selectors leave prompts visually identical while tokenization
  is "secretly" altered; high ASR against four aligned LLMs.
  Optimized attack, §9 R1.
- Sarabamoun, "Special-Character Adversarial Attacks on
  Open-Source Language Models", arXiv:2508.14070 — cited **only
  via the ar5iv rendering** (ar5iv.labs.arxiv.org/html/2508.14070,
  opened), with all three caveats: **single author, arXiv-only,
  heuristic (keyword/policy-pattern) success criterion**. The
  figure used: zero-width injection 54.2% success (ar5iv §4.1),
  verified in the opened ar5iv text. Per the slate's verified
  version trap, this number does **not** appear in arXiv HTML
  v2 (a rewritten paper reporting different metrics) and is
  never cited to v2.
- SentencePiece, "Text Normalization in SentencePiece"
  (chromium.googlesource.com mirror of the SentencePiece doc,
  opened): normalization always happens *before* tokenization;
  default rule `nmt_nfkc`; `identity` alternative exists; the
  compiled rule is embedded in the model file. Basis of the §6
  regime pre-registration — an expectation about a *class* of
  pipelines, never an assertion about a specific checkpoint.
- UAX #15, Unicode Normalization Forms (unicode.org/reports/
  tr15/, rev. 58 / Unicode 18.0.0 text, opened): §1.1 canonical
  vs compatibility equivalence — used here only for the
  boundary discipline of §11/§12 (what this probe is not).
- UTS #39, Unicode Security Mechanisms (unicode.org/reports/
  tr39/, opened via search-surfaced text): step 2 of
  *internalSkeleton* — the function `skeleton` incorporates —
  removes Default_Ignorable_Code_Point
  characters (attribution corrected, amendment A9(iii)) — the standards basis for expecting DICP code
  points to fold in normalizing pipelines (§6); and the
  identifier tables' definition of Default_Ignorable via
  `Default_Ignorable_Code_Point = Yes` (§2). Property *values*
  for the five arms are pinned to UCD 15.0.0 (§2), not to the
  live UTS text's version.
- Hua et al., "Flaw or Artifact? Rethinking Prompt Sensitivity
  in Evaluating LLMs", arXiv:2509.01790 (v1, 1 Sep 2025) —
  opened (abstract/full-text surfaced): much reported prompt
  sensitivity stems from heuristic evaluation including rigid
  answer matching across 7 LLMs / 6 benchmarks / 12 templates.
  The §5 decomposition obligation.
- Pecher, Spiegel, Belanec, Cegin, "Revisiting Prompt
  Sensitivity in Large Language Models for Text
  Classification: The Role of Prompt Underspecification",
  arXiv:2602.04297 (v1) — opened: underspecified prompts show
  higher variance; instruction prompts less. The §9 R5
  fully-specified-panel obligation.
- Tree: `core.py:827` (byte-exact `exact_match`);
  `density.py:790` (`sign_test_p`, two-sided — not used);
  `confidentiality.py:295` (`mcnemar_exact_p`, descriptive
  only); `placement.py:172-173` (the floors *not* borrowed);
  `canonicalform.py` / `recordnotation.py` /
  `instructiondistance.py` (population, decomposition, and
  D-conditioning precedents); `docs/USAGE.md` §8.1–§8.59
  (duplication ground truth).

**Removals:** none required — every source the design relies on
resolved when opened. Two tasking-supplied references were
corrected on contact with the tree, recorded in place:
`docs/reference/cli.md` does not exist (§10), and the slate's
utility second half is scoped, not silently dropped (§1).
