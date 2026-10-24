# Context rotation design (cycle 203)

Status: v2 — amended after the full adversarial round
(red-team A: `agents_out/cycle203/redteam_a.md`, verdict DO
NOT BUILD AS SPECIFIED, 2 FATAL / 8 MAJOR / 5 MINOR; red-team
B: `redteam_b.md`, mandated to disprove A, confirmed A's core
by independent execution and added two findings; verifier:
`verifier.md`, design claims re-executed; coordinator
adjudication: `coordinator_adjudication.md`). v1's
classification compared two different statistics (A F1,
fatal) and its construction classification had no primitive
beneath it (A F4); v2 rebuilds both. The uniqueness pass is
the wave-9 research brief (§3.1), verified live by the
researcher against primary sources and recorded in
docs/RESEARCH_LOG.md (cycle 203 entry); the adjudicated
amendments from the wave-9 coordinator adjudication are
binding on this design: (a) the pass-by-absolute-slot headline
is presented as a cross-check against needle's depth curve,
not a novelty claim; (b) arXiv:2511.20836 is cited as v3
(retitled "Structured Prompts Improve Evaluation of Language
Models") or not at all — and its ranking-shift figure, if ever
used, is 5/7 per the v3 abstract; (c) Lialin
(arXiv:2404.02054) and Richardson (arXiv:2310.20081) are
struck as rotation sources — the citation autopsy in the
wave-9 research brief §1 shows neither supports a rotation,
block-ordering, or position-sensitivity claim, and no sentence
in this design rests on either name. The Chroma Context Rot
report is cited corporately (its page carries no byline).

## Utility

A practitioner whose RAG stack assembles a prompt from five
retrieved chunks learns, per template family on their own
checkpoint, whether cyclically rotating the same chunks —
same content, same total text length, same relative order —
changes the verdict, and whether the template they were about
to ship keeps the same winner under rotation, before a
retriever re-ranking silently reorders chunks in production
and flips their eval.

## What this probe is not (the boundary, in the design doc's words)

Four neighbouring probes vary neighbouring things; none varies
this one. Needle moves one fact through a fixed filler stream
at five depths and two doses; the filler is topically
dissimilar prose, everything else stays put, and the readout
is a recall-vs-depth curve for that fact. Position permutes
the answer options inside one multiple-choice question; there
is no multi-block context span. Placement moves the question
around a frozen three-sentence payload; the payload itself
never reorders. Distract splices filler at token doses. This
probe pins the question after the span and rotates the span's
five fact-bearing blocks themselves, holding the content
multiset, the total text length, and the cyclic relative
order constant across arms — the controlled sub-case of
reordering that separates an absolute-position effect from an
order effect. USAGE §8.50 repeats this paragraph's
distinctions in the same words.

## Construction

Cases are generated inside the stage from the run seed by the
frozen generator in `rotation.py` (SHA-256-derived draws, the
needle/placement discipline: the same seed regenerates the
identical set across runs and `PYTHONHASHSEED` values).
`CASE_COUNT = 20` is a construction constant; fixture ids are
`rotation-{index:02d}` in canonical index order.

Each case is a span of five blocks plus a question. Blocks are
indexed **[0 = asked, 1..4 = distractors, 5 = replacement]**;
every arm order below is a list of block indices in span
order.

- **Blocks.** One asked block and four distractor blocks.
  Every block is one fact sentence from a frozen template:
  `The {topic} access code is {value}.` A distractor block is
  fact-shaped — same carrier, different topic, different
  code — not prose filler, so the span reads as five retrieved
  chunks competing for the same question shape. A sixth block
  (the replacement) is generated per case for the absent arm.
- **Topics.** Six topics per case (asked, four distractors,
  one replacement), drawn from a frozen pool of thirty. The
  pool's pinned properties: topic phrases are pairwise
  distinct and no topic is a substring of another; every topic
  token has at least four characters; no topic token shares a
  four-character prefix with a code word, and no topic token
  is a substring of a code word or vice versa. A case's six
  topics are pairwise distinct, so the asked topic's full
  phrase occurs in exactly one block of the case and the
  question cannot leak a value by association.
- **Values.** Each block binds a code (`word-1000`..
  `word-9999` shape) drawn per block. All six values in a case
  are pairwise distinct, and asked values are distinct across
  cases (the draw rules below).
- **Question.** From a frozen template: `What is the {topic}
  access code?` — naming the asked topic, byte-identical in
  every arm, pinned after the block span. The expected answer
  is the asked block's value.
- **Span assembly.** Block texts joined by a single space;
  the question follows the span after a `"\n\n"` joiner. The
  span text of an arm is exactly the five block texts in the
  arm's order — text-level surgery before render, per
  candidate (the placement pattern).

### Draw rules (frozen; the diff validator's mirror reproduces them exactly)

One derivation primitive, the house form:
`_derive(seed, index, slot, counter)` is the big-endian first
eight bytes of SHA-256 over the ASCII material
`"{seed}|rotation|{index}|{slot}|{counter}"`. Slot meanings
are frozen: for block `b` in 0..5, the code word is slot
`2b` and the number is slot `2b + 1`; the asked topic is slot
20; the distractor topics for blocks 1..4 are slots 21..24;
the replacement topic is slot 25. Each slot has its own
counter, starting at 0 for each case. Topic counters and
value counters are independent streams: a value redraw never
perturbs a topic draw and vice versa.

- **Asked topic.** Draw `pool[_derive(seed, index, 20, c) %
  30]` at counter `c = 0, 1, ...`, taking the first index not
  used as an earlier case's asked topic (without replacement
  across cases; 20 cases from a pool of 30 always terminate).
- **Distractor and replacement topics.** For slot `s` in
  21..25, draw at that slot's own counter from 0, taking the
  first topic that is neither the case's asked topic nor
  already chosen for this case.
- **Values.** For block `b`, at its current counters: word =
  `WORDS[_derive(seed, index, 2b, c_b) % len(WORDS)]`, number
  = `1000 + _derive(seed, index, 2b + 1, c_b) % 9000`, value
  = `"{word}-{number}"`. All six blocks draw at counters
  starting at 0. If any two of the case's six values collide,
  or the asked value repeats an earlier case's asked value,
  all six blocks' counters increment by one and all six
  values redraw (the needle whole-set redraw rule, widened to
  the case's set and frozen here).

Per-case construction checks (generator-enforced, pinned by
test over the generator and seed sweeps): the six topics are
pairwise distinct; the six values are pairwise distinct; each
block's value occurs in exactly one block text of the case
(its own), and the asked topic's phrase occurs in exactly one
block text — the containment screen restated as a construction
property of the frozen pools, proven over the pools, not
inferred from the value format (A F15).

## Arms

Eleven cells per (candidate, case), all a single user turn:

- `rot_0`..`rot_4`: cyclic rotations of the block sequence.
  Rotation `r` moves block `r` to the front:
  order = `[r, r+1, ..., 4, 0, ..., r-1]`. Content multiset,
  total text length, and cyclic relative order are identical
  in all five; among internal block–block pairs exactly one
  adjacency is lost and one gained per rotation (the wrap
  pair) — two blocks additionally change predecessor context
  at the render's head and tail seams (A F14; the
  construction discipline below covers the whole render, not
  only internal pairs). The asked block sits at absolute slot
  `(5 - r) % 5` under rotation `r`, so across the five
  rotations the asked block occupies every slot exactly once.
- `mir_1`..`mir_4`: the mirror arms — cyclic rotations of the
  reversed block order `[4, 3, 2, 1, 0]`, named by the slot
  the asked block occupies: `mir_4` = `[4, 3, 2, 1, 0]` (the
  reversal arm), `mir_3` = `[3, 2, 1, 0, 4]`, `mir_2` =
  `[2, 1, 0, 4, 3]`, `mir_1` = `[1, 0, 4, 3, 2]`. Together
  with `rot_0` they form the classification's control set
  (see Classification): the same five asked slots {0,1,2,3,4},
  reached by non-cyclic reorderings. Reported beside the
  rotation readouts, never pooled into them.
- `absent`: the r=0 span with the asked block replaced by the
  case's sixth block (order `[5, 1, 2, 3, 4]`). The question
  still asks for the asked topic's code; a pass is recall
  without the fact present — the false-recall floor (the
  needle precedent).
- `closed_book`: the question alone. A screening arm, not a
  rotation (see Screening).

## Construction verification (the rotation-specific discipline)

Text-level rotation does not guarantee token-level rotation:
at each block joint, a candidate's tokenizer may merge the
last token of one block with the first token of the next, so
the rotated render's token inventory can differ from the
unrotated one even though the text multiset is identical.
Each span arm (`rot_0`..`rot_4`, `mir_1`..`mir_4`) of each
(candidate, case) is therefore measured at construction under
a frozen rule whose text the protocol carries by digest
(`CONSTRUCTION_RULE_TEXT` in `rotation.py`,
`rotation-construction-v1`):

- Render the arm once. `arm_tokens[arm]` is the render's
  token count. The arm's **inventory digest** is the SHA-256
  of the comma-joined decimal token ids of the full render,
  sorted ascending — a digest of the token multiset.
- An arm is `id_exact` iff its inventory digest and token
  count both equal `rot_0`'s; `rot_0` is `id_exact` by
  definition (it is the reference). `id_exact` is a **derived**
  classification everywhere — stage aggregation and the diff
  validator both compute it from the recorded digests and
  counts; no record stores the classification itself (A F4).
- The classification is a claim about the token inventory
  only: no seam event changed what the prompt is made of. It
  is **not** a claim that segmentation at the seams is
  unchanged — a merge at one seam compensated by a split at
  another restores the exact inventory while changing the
  segmentation at two joints, and the design states this
  openly rather than claiming a segment-level proof it does
  not perform (A F5a, open point 1).
- Paired readouts (headline population, slot curve, mirror
  and reversal blocks) run only over cases whose five
  rotation cells are all present and `id_exact`. The
  `id_exact` shares are themselves reported readouts per
  candidate, so a tokenizer whose merges shrink the paired
  population is visible in the output, never silently
  absorbed.
- The validator cannot recompute token ids without a
  tokenizer, so the inventory digests are stage-attested
  measurements — but they are primitives in the same trust
  class as `answer_digest`: every identity they imply
  (digest-and-count equality against `rot_0`, hence the
  derived classification and every population built on it) is
  checked and recomputed by the validator. This is a stronger
  settlement than needle's sizing residue, where the measured
  quantity has no digest-sized primitive at all.

Construction costs exactly one render per arm — eleven
renders per (candidate, case) — and no search, so there is
no encode budget to freeze.

## Screening

Three screens, in order:

1. **Frozen-text screen (stage-level, fail-closed).** Every
   generated text (templates with slots emptied, topics, code
   words) is checked with `audit_special_token_literals`
   against every candidate, and every candidate's decoded
   boundary text is checked for containment in the generated
   texts (the needle/placement pattern). A hit fails the
   whole stage closed with detail `rotation_screen_failed`
   before any generation. Stated limit (lead L3, accepted as
   the house pattern): the audit covers the frozen parts, not
   instantiated texts; a special-token literal spanning a slot
   boundary in an instantiated block would not be caught by
   it — the neighbours share this shape and no concrete
   literal is known for any shipped candidate.
2. **Pool screens (construction-level).** The topic pool's
   pinned properties above, plus the per-case construction
   checks (topic/value distinctness, single-occurrence
   containment). Pinned by test over the frozen pools and the
   generator.
3. **Closed-book screen (per candidate, per case, derived).**
   If a candidate's `closed_book` cell passes a case, the
   answer was reachable without the span for that candidate —
   the case is excluded from that candidate's rotation
   populations. The exclusion is derived in aggregation from
   the closed-book cell's verdict, never a stored flag; the
   diff validator re-derives the same exclusion when it
   recomputes.
   Per-case, before generation: if any block text of the case
   contains the candidate's decoded boundary text, every cell
   of the case for that candidate is marked `arm_screened`
   (the needle per-span screen, at case granularity because
   the blocks rotate as a unit).

**Populations, defined once.** A record is
**boundary-screened** for its candidate when the per-case
boundary screen fired (every cell `arm_screened`);
**closed-book-screened** when its closed-book cell passed;
**live** when it is neither. `n_screened` counts
closed-book-screened records; `n_boundary_screened` counts
boundary-screened records; `n_closed_book_incomplete` counts
records that are not boundary-screened and whose closed-book
cell never completed (screening status genuinely unknown —
boundary-screened records are a known exclusion and are never
counted here, A F7). Boundary-screened records enter no
readout population: not `n_live`, not the absent block, not
the conditions shared population.

## Records

One record per (candidate, case); a candidate's record list
carries exactly `CASE_COUNT` records in canonical generated
order (the validator rejects any other count or order):

- `fixture` (the case id), `topic` (the asked topic),
  `expected_digest` (SHA-256 of the asked value). No block,
  question, or value text anywhere — digests and counts only.
- `arm_tokens`: the rendered token length of each of the
  eleven arms under this candidate, measured at construction.
  For a completed cell, the cell's own `input_tokens` must
  equal `arm_tokens[arm]` (the needle/placement grounding:
  one id sequence per fixture, as recorded by
  `core.evaluate()`).
- `arm_inventory_digests`: per span arm (`rot_0`..`rot_4`,
  `mir_1`..`mir_4`), the inventory digest defined above (the
  absent arm has different content by design and carries no
  digest).
- `cells`: per arm, either a cell or `None`. A cell is
  `{verdict, boundary_found, boundary_text_match,
  input_tokens, answer_digest}` where `verdict` is derived:
  the validator recomputes it as `(answer_digest ==
  expected_digest) and boundary_found` (the numform F3
  lesson, applied by construction).
- `cell_details`: per arm, `""` when the cell completed, else
  one of `arm_context_exceeded`, `arm_evaluation_failed`,
  `arm_screened`. A cell is present iff its detail is `""`.

Stage-level details are frozen as `STAGE_DETAILS =
("cancelled", "rotation_record_budget_exceeded",
"rotation_screen_failed")`. Candidate names are unique by the
run-level guard (`experiments.py`, "formats must have unique
names"); the payload's candidates are a parsed JSON object,
in which a duplicate key cannot survive parsing — the
uniqueness the validator relies on is the stage's guard plus
the object shape, audited as A F10b (corrected at build time:
no sibling validator performs a separate duplicate-key
check, and neither does this one).

## Readouts

Per candidate, over its own records (the validator recomputes
every number). **Global denominator rule: every rate and
share in this section is `None` iff its denominator is 0**
(A F8); each readout below names its population explicitly.

- `arms`: per span arm (`rot_0`..`rot_4`, `mir_1`..`mir_4`)
  `{n_completed, n_pass, rate}` over live records. This table
  is descriptive and construction-ungated; the absent and
  closed-book arms are not rows here — their numbers live in
  their own blocks, so no number is printed twice (B-F2).
- `closed_book`: `{n_completed, n_pass, pass_rate}` over all
  records — the screening yield, reported as itself.
- Screen accounting: `n_records`, `n_screened`,
  `n_boundary_screened`, `n_closed_book_incomplete`,
  `n_live`.
- **Headline block** over the headline population — live
  records whose five rotation cells are all present and all
  `id_exact` (derived):
  - `n_headline`;
  - `n_all_pass` and `n_all_fail` (verdict constant across
    `rot_0`..`rot_4`, all-pass / all-fail), `n_invariant =
    n_all_pass + n_all_fail` and `invariance_rate`;
  - `n_flip_any = n_headline - n_invariant`;
  - the r=0-conditional block: over headline records whose
    `rot_0` cell passes, `n_r0_pass` and `n_flip_from_pass`
    (records whose verdict vector is not all-pass), with
    `flip_rate_from_pass`;
  - `evaluable_share = n_headline / n_live`, and
    `id_exact_share` over completed rotation cells of live
    records (present-and-`id_exact` cells over present
    rotation cells);
  - a classification (below) and the falsification boolean.
- **Slot curve**: for each absolute slot `s` in 0..4, the pass
  count over headline records of the one rotation that places
  the asked block at slot `s` — by construction every slot is
  observed exactly `n_headline` times, so the curve carries
  its denominator once, as `n_headline`. This is the rotation
  analogue of needle's depth curve and is presented as a
  cross-check against it (same functional question — does the
  fact's absolute position move the verdict — asked of a
  whole-prompt symmetry instead of a single planted fact),
  never as a novelty claim (adjudication amendment (a)).
- **Mirror block**: over headline records whose four mirror
  cells are all present and `id_exact` (the mirror-paired
  set): `n_paired`, `n_flips_paired` (records whose verdict
  is not constant across `rot_0`..`rot_4` — the headline
  statistic restricted to this set), `n_flips_mirror`
  (records whose verdict is not constant across the control
  set {`rot_0`, `mir_1`..`mir_4`}), and the mirror
  `id_exact_share` over completed mirror cells of live
  records.
- **Reversal block**: over headline records whose `mir_4`
  cell is present and `id_exact`: `n_paired`, `n_discordant`
  (the `mir_4` verdict differs from `rot_0`),
  `discordance_rate`. Reading: the reversal arm is one member
  of the mirror set; its discordance is the descriptive
  single-arm control level, printed beside the mirror block.
  The classification's control comparison is the mirror
  block's matched statistic, not this discordance (A F1).
- **Absent block**: over live records whose absent cell is
  present: `n_evaluable`, `n_false_pass`, `rate` — the
  false-recall floor, reported beside the headline (the
  needle pattern). In the reporting section text the absent
  numbers are labelled false recall wherever they appear
  (open point 4).

### Classification (total function)

Floors frozen at the needle/placement value:
`MIN_HEADLINE_POPULATION = 6` and `MIN_CONTROL_POPULATION =
6`. The floors gate this label and the falsification boolean
only, never a reported count or rate. Decision list, first
match wins, over the headline and mirror blocks with
`F = n_flip_any`:

1. `n_headline < 6` → `insufficient`
2. `F == 0` and `n_all_fail > n_all_pass` →
   `joint_collapse_dominant`
3. `F == 0` → `invariant_observed`
4. mirror `n_paired < 6` → `control_set_too_small`
5. `n_flips_paired <= n_flips_mirror` →
   `within_mirror_control`
6. otherwise → `rotation_sensitive`

Step 2 is placement's joint-collapse settlement in dominance
form (A F2, B's correction): a population whose constant
verdicts are mostly all-fail is a capability story, not an
invariance acquittal, so it never receives
`invariant_observed`. Steps 5–6 compare the same statistic on
both sides — any-flip over five arms, over the same cases —
and the two arm sets place the asked block at the same slot
multiset {0,1,2,3,4}: under a pure absolute-slot effect (the
verdict a function of the asked slot only) the two flip
indicators are equal for every case deterministically, so a
slot effect alone can never produce `rotation_sensitive`
(B-F1, answered by construction). Under the exchangeability
null the comparison is calibrated by symmetry (a tie goes to
`within_mirror_control`).

### Falsification

There is exactly one definition, and this is it (the same
formula is printed in the `rotation.py` module docstring; any
other sentence in this document is commentary on it, not a
second definition):

`invariance_falsified` is true iff `n_headline >= 6` and
`n_flip_any >= 1`.

Reading: the claim under test is invariance itself (the wave-9
brief §3.1(c)), so a single verified flip at a sufficient
population falsifies it. The mirror control does not acquit
invariance — a flip is a flip — it answers the separate
question of whether the effect is cyclic-specific or generic
reordering, which is what the classification's
`within_mirror_control` step reports.

The joint behaviour of the boolean and the classification is
a theorem of the decision list, stated as one (A F3):

| classification | falsified |
|---|---|
| `insufficient` | false only |
| `joint_collapse_dominant` | false only |
| `invariant_observed` | false only |
| `control_set_too_small` | **true only** |
| `within_mirror_control` | true only |
| `rotation_sensitive` | true only |

Two pairings are deliberate and the section text says both,
in this order: a candidate can be falsified and
within-control at once (the boolean does not wait for the
control), and a candidate can be falsified beside
`control_set_too_small`, whose meaning is only that the
control set was too small to judge cyclic-specificity — the
boolean never claimed otherwise. `joint_collapse_dominant`
pairs with falsified = false because the boolean is about
flips and there are none; the collapse caveat lives in the
label, and the section prints both. Below the floor the
boolean is false and the classification is `insufficient` —
withheld, never defaulted to acquittal. The enumeration of
the full feasible tuple space, proving totality, label
reachability, and this table, is pinned by a test in the
cycle's test plan (the placement precedent, extended).

**Dispositions of brief §3.1(c)'s other clauses (A F13).**
The brief's cross-candidate clause ("proved wrong if all
evaluable candidates share the same invariance rate within
their denominators") is dropped: its criterion defines no
decidable test ("within their denominators" names no
statistic), and no cross-candidate boolean is printed; the
per-candidate boolean plus the printed per-candidate rates
with denominators carry the evidence. The brief's
fact-absent-floor qualifier does not condition the boolean:
falsification is over the screened headline population, the
absent rate is a reported readout beside it, and a flip is a
within-case verdict change on present-fact arms, which false
recall does not manufacture.

### Conditions (per-rotation selection)

For each arm in {`rot_0`..`rot_4`, `mir_4` (the reversal)},
the conservative selection rule is re-applied over the arm's
completed cases restricted to the **shared live population**:
cases recorded for every candidate and live for all of them
(the placement union rule, with "live" as defined above — so
a case boundary-screened for one candidate never enters, A
F7). One amendment to v1 (A F9): each arm's population is
additionally gated on that arm being `id_exact` (derived)
for **every** candidate on the case — needle's conditions
range only over construction-verified cells because its
unmatchable cells are never generated, and rotation's
`text_level` cells, which are generated, must not enter the
selection readout at full weight beside a headline that
excludes them. (`rot_0`'s gate is vacuous: it is `id_exact`
by definition.) The readout is withheld with detail
`selection_withheld_mismatched_sets` when the restricted
completed sets differ across candidates or are empty (the
needle rule). Each arm's `flip` compares its selection to the
`rot_0` selection recomputed on the same restricted completed
set; `rot_0`'s own flip is `False` when not withheld, and a
flip is `None` when either side is withheld.

## Diff validator

`diff_rotation` in `diffreports.py`, full-recompute standard
(the wave-7 F6 settlement): `_rotation_recompute` carries its
own generator mirror (its own derive function, templates,
topic and word pools as literals — no imports from
`rotation.py`), regenerates the case set from the protocol's
seed, and compares every record against it (exactly
`CASE_COUNT` records per candidate in canonical order;
fixture id, asked topic, expected digest). The protocol is
rebuilt from frozen scalars plus the payload's seed and
backend fingerprint and its fingerprint recomputed. Every
cell's verdict is re-derived from its digest and boundary
flag; every completed cell's `input_tokens` must equal the
record's `arm_tokens`; cells and details must be consistent
(a cell is present iff its detail is `""`).
`arm_inventory_digests` are shape-checked (nine span arms,
64-hex) and the `id_exact` classification is **derived** from
digest-and-count equality with `rot_0`, never read from the
payload. Every aggregate, headline block, slot curve, mirror
block, reversal block, absent block, classification,
falsification flag, and the conditions are recomputed by
mirror code and compared exactly. A one-break-at-a-time
battery pins every reject branch.

## Budget

Worst case: 24 candidates (the run cap) × 20 cases × 11 arms
= **5,280 generations**, stated in the CLI help with the
forwards formula before the run (the wave-9 brief's 3,360
counted five arms of an earlier shape; the help says 5,280).
Construction costs exactly one render per arm beside the
generations (see Construction verification). The stage
threads the run's cancel event end-to-end and checks it
before and after every evaluation (the multiturn rule).
`MAX_GROUND_RECORDS = 480`, `MAX_RECORDS_BYTES = 12 MiB`
(the house bound).

## Pin blast radius

(Rewritten from executed greps of the placement footprint,
A F10; the v1 list's "VALIDATION §6 probe list" was a phantom
inherited from PLACEMENT_DESIGN.md — VALIDATION has no §6;
its probes are a prose list in the first Evidence bullet.
That PLACEMENT_DESIGN line is corrected in this cycle's push.)

- New pure module `rotation.py` → the test_cycle44 mypy
  source-file pin moves 59 → 60 in the same push. VALIDATION's
  first-Evidence-section probe list gains the rotation entry
  in prose (pinned by nobody verbatim); VALIDATION header
  range moves to cycles 10–203 with the test_cycle178 pin in
  the same push.
- New `--rotate` flag → the cycle-179 parametrized dispatch
  table and probe-wrapper audit gain a row; the cycle-31/64/84
  subcommand/flag audits and the args-Namespace contract gain
  a case.
- New diff section `rotation` → test_cycle67's three elements
  (fuzz oracle + `_fuzz_rotation_section`; the flip-identical
  gate anchor's original *and* mutant texts;
  `_identical_expected`'s key set) plus test_cycle27's full
  placement-style footprint: an independent `_or_rotation_*`
  oracle, its asymmetric computation, the two summary keys,
  and the section stub entry `"rotation": {"available": ...,
  "changed": False}` (grep-counted at build time, not
  assumed).
- New reporting section → reporting.py's three touchpoints
  (compute, TOC tuple, body concatenation).
- README Capabilities bullet + measured test count;
  VALIDATION counts and the CI-scope ruff file count;
  CHANGELOG entry; USAGE §8.50 (the boundary paragraph in the
  same words as above); RESEARCH_LOG cycle-203 entry before
  the code; the PLACEMENT_DESIGN §6-phantom correction.

## Open points — dispositions

1. *Inventory vs segment classification.* The `id_exact`
   definition is inventory equality, honestly labelled
   (segmentation at seams not verified). With the inventory
   digest carried as a primitive (A F4), the classification is
   derived, not attested-free; a segment-level check is not
   adopted (its cost is a per-block segment apparatus the
   inventory claim does not need).
2. *Falsification asymmetry.* The boolean and the
   classification are one story with a published joint table
   (Falsification). The substantive defects nearby were
   F1/F2, both fixed in the definitions.
3. *Conditions without construction gating.* Adopted: gated
   per arm on `id_exact` for every candidate (Conditions).
4. *The absent arm's expected value.* Sound as designed: the
   absent cell's verdict uses the asked value's digest, and
   the generator's distinctness rules guarantee the asked
   value occurs nowhere in the absent prompt, so a pass is
   false recall by construction. The one smuggle A found —
   the generic arms table listing the absent rate unlabelled
   — is closed by B-F2 (absent is not an arms-table row).
5. *Topic-pool repetition across cases.* Asked topics are
   drawn without replacement across cases, so the twenty
   asked topics are always distinct (lead L1 closed by
   construction, not by caveat).
6. *A slot-pinned mirror arm* (`[0,4,3,2,1]`, asked fixed at
   slot 0, distractors reversed — B-F1's suggestion): not
   adopted. The mirror set answers the slot confound by
   slot-multiset matching (Classification); a twelfth arm's
   marginal readout does not justify its cost. Recorded as
   considered-and-deferred; a future cycle may add it as a
   descriptive arm without touching the classification.
