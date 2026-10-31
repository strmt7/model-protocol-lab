# Instruction placement design (cycle 193)

Status: v2 — adjudicated, buildable as amended. Swarm record:
the researcher verified the uniqueness claim live (HOLDS WITH
NARROWING; the narrowed wording is in docs/RESEARCH_LOG.md,
cycle 193, with three corrections to the wave-8 brief — the
arXiv:2605.28188 instrument is FRAGILE — fact-preserved
framing variants (value-tinted narration, temporal slice,
narrative vividness), not a value anchor; the healthcare role-relocation result is
arXiv:2601.18334, recorded in docs/RESEARCH_LOG.md's
cycle-185 entry, on closed-weights hosted models (an earlier
printing of this summary named the wrong ID and the wrong
home; corrected in cycle 197); the framing paper's
mitigation is VALIGN, a representation-level method, never a
placement instrument (an earlier printing of this summary
carried the few-shot-exemplar / order-swap characterization
that cycle 207 withdrew in full; corrected in cycle 226). Red-team A returned DO NOT BUILD AS SPECIFIED (17
findings, 3 fatal). Red-team B cross-checked every finding with
its own enumerations and probes: it refuted A's F1/F8
contradiction claim under the design's operative reading,
refuted F9's seam evidence and F10's collision mechanism, and
confirmed the substantive core — control populations (F2),
duplicate candidate names (F7), the missing joint-collapse
state (F4/F5), uncalibrated-control disclosure (F6), question
duplication accounting (F9 residue), topic-mention screening
(F10 residue), closed-book screen accounting (F11), and the
missed cycle-44 §6 pin (F16). The nine must-fixes below are B's,
adopted wholesale; the body text is amended accordingly.

1. One falsification formula (R_d), printed verbatim in the
   Falsification section and in the module docstring; the
   ambiguous prose sentence is deleted.
2. Sandwich control blocks are computed on the headline paired
   population (the control set); the decision list gains a
   `control_set_too_small` step.
3. The stage rejects duplicate format names; `_valid_placement`
   states the unique-names clause.
4. The headline block carries `both_fail_dominant`; the
   falsification formula and classification rule 2 account for
   it.
5. The control's deliberate conservatism is disclosed with its
   measured acquittal rates (Falsification section).
6. Records carry `question_tokens`, recompute-checked.
7. The generator's topic and code-word pools are stem-disjoint
   (pinned), closing topic-mention leakage deterministically.
8. Aggregates carry closed-book incompleteness accounting
   (`n_screened` beside a screening-incomplete count).
9. The RESEARCH_LOG entry precedes the code (done, cycle 193).

## Utility

A practitioner assembling prompts from a retrieved passage plus
a question learns, per template family, whether putting the
question before or after the payload changes verdicts on their
checkpoint — and picks the layout their winning template
actually rewards — instead of copying a layout convention from
a blog post measured on a different model.

## What this probe is not (the boundary, in the design doc's words)

Four neighbouring probes vary neighbouring things; none varies
this one. Hierarchy varies the position of a folded
privilege-marked instruction block against a user instruction.
Position permutes the answer options of a multiple-choice
question. Distract splices filler into a message at token doses.
Needle buries a fact at measured depths inside long filler. This
probe varies where the task question sits relative to a short
payload inside one user turn — question-first versus
payload-first — crossed with the hand-rendered template family.
The payload is frozen short by design: this is placement, not
dilution (distract owns dose) and not depth (needle owns long
contexts). USAGE §8.46 repeats this paragraph's distinctions in
the same words.

## Construction

Cases are generated inside the stage from the run seed by the
frozen generator in `placement.py` (SHA-256-derived draws, the
numform/needle discipline: the same seed regenerates the
identical set across runs and `PYTHONHASHSEED` values).

Each case is a short passage plus a question:

- A frozen topic list, one topic per case (`CASE_COUNT = 24`).
  The topic list and the code-word pool are stem-disjoint — no
  topic word shares a stem with a code word — so a question
  naming the topic cannot leak the value's word by association
  (red-team F10 residue; the disjointness is a property of the
  frozen pools, pinned by test).
- A fact sentence from a frozen template: `The {topic}
  inspection code is {value}.` The value is a code word plus a
  number (`word-1000`..`word-9999` shape), drawn per case;
  values are pairwise distinct across cases by the needle
  redraw rule (a colliding draw redraws both slots at the next
  counter).
- A payload of exactly three sentences: a neutral sentence, the
  fact sentence, a neutral sentence. The fact sits at frozen
  sentence position 1 (the middle) in every case and every arm
  — the only thing that moves between arms is the question.
  The two neutral sentences are drawn per case from a frozen
  pool of scene sentences written for this probe (no digits, no
  topic words, no code words, temporally inert), distinct from
  each other.
- The question from a frozen template: `What is the {topic}
  inspection code?` The expected answer is the value. The
  question wording is frozen byte-for-byte across arms; any
  per-arm "naturalness" edit would void the contrast.

## Arms

Four cells per (candidate, case), all a single user turn,
text-level surgery pre-render, joiner `"\n\n"`:

- `question_first`: question, payload.
- `payload_first`: payload, question.
- `sandwich`: question, payload, question repeated verbatim.
  The sandwich is the control for the headline contrast: it
  separates "question at the end wins by recency" and "question
  twice wins by exposure" from a pure order effect. It is
  reported beside the headline, never pooled into it.
- `closed_book`: the question alone, no payload. A screening
  arm, not a layout (see Screening).

## Screening

Three screens, in order:

1. **Frozen-text screen (stage-level, fail-closed).** Every
   generated text (templates with slots emptied, topics, code
   words, neutral pool) is checked with
   `audit_special_token_literals` against every candidate, and
   every candidate's decoded boundary text is checked for
   containment in the generated texts (the needle pattern). A
   hit fails the whole stage closed with detail
   `placement_screen_failed` before any generation: a
   special-token literal in frozen text would forge turn
   structure in every cell of every run.
2. **Pool screen (construction-level).** No neutral-pool
   sentence contains a digit, a code word, a topic string, or
   the fact template's carrier phrase. This is a property of
   the frozen pool, pinned by test; it guarantees the payload's
   only answer-bearing sentence is the fact sentence.
3. **Closed-book screen (per candidate, per case, derived).**
   If a candidate's `closed_book` cell passes a case, the
   answer was reachable without the payload for that candidate
   — the case is excluded from that candidate's placement
   populations (pair blocks, arm rates, sandwich blocks). The
   exclusion is *derived in aggregation* from the closed-book
   cell's verdict, never a stored flag: the record carries the
   closed-book cell itself, and the diff validator re-derives
   the same exclusion when it recomputes. The screened count
   and the closed-book pass rate are reported per candidate,
   never silently dropped. A case whose closed-book cell did
   not complete is not screened (nothing is known), and is
   handled by the paired-population rules like any other
   incomplete cell.

## Records

One record per (candidate, case):

- `fixture` (the case id), `topic`, `expected_digest`
  (SHA-256 of the expected value). No payload, question, or
  value text anywhere in the payload — digests and counts only.
- `arm_tokens`: the rendered token length of each arm under
  this candidate, measured at construction. For a completed
  cell, the cell's own `input_tokens` (the evaluation's
  observation) must equal `arm_tokens[arm]`; a backend whose
  encode disagreed with itself surfaces as a validator
  rejection downstream, never a silently wrong record (the
  needle discipline). The equality's grounding is the tree's
  own contract: one id sequence per fixture, as recorded by
  `core.evaluate()` — the cell's `input_tokens` is read from
  the very encode the stage measures (red-team F3's citation,
  adopted).
- `question_tokens`: the rendered token length of the question
  alone under this candidate — identical by construction to
  `arm_tokens["closed_book"]`, and carried as its own field so
  the sandwich arm's duplicated question is accounted in the
  recompute-checked fields (the sandwich render contains the
  question twice at the text level; token counts at the seams
  are whatever the candidate's tokenizer does, and are
  recorded, not assumed).
- `cells`: per arm, either a cell or `None`. A cell is
  `{verdict, boundary_found, boundary_text_match,
  input_tokens, answer_digest}` where `verdict` is *derived*:
  the validator recomputes it as `(answer_digest ==
  expected_digest) and boundary_found`. There is no stored
  parsed answer to forge (the numform F3 lesson, applied by
  construction, as in dateline and needle).
- `cell_details`: per arm, `""` when the cell completed, else
  one of `arm_context_exceeded`, `arm_evaluation_failed`.

## Readouts

Per candidate, over its own records (the validator recomputes
every number):

- `arms`: per arm `{n_completed, n_pass, rate}` over live
  (non-screened) records; a rate is `None` only when its
  denominator is 0.
- `closed_book`: `{n_completed, n_pass, pass_rate}` over *all*
  records — the screening yield, reported as itself.
- `n_screened`: cases excluded by the closed-book screen,
  reported beside `n_closed_book_incomplete` — cases whose
  closed-book cell never completed, where the screen could not
  fire and nothing is known. A candidate with incomplete
  screening is visibly unscreened on those cases, never
  silently treated as screened (red-team F11).
- The **headline pair block** (`question_first` vs
  `payload_first`) over the paired population — live cases
  where both cells completed: `n_paired`, `n_qf_only`,
  `n_pf_only`, `n_both`, `n_neither`, per-arm pass rates over
  the paired population, the exact McNemar p (diagnostic only),
  `both_fail_dominant` (true iff `n_neither >=
  max(n_qf_only, n_pf_only)` and `n_neither >= MIN_DISCORDANT`:
  the joint-collapse state, where both layouts fail the same
  cases and a discordance label alone would hide it), and a
  classification (below).
- Two **sandwich pair blocks** (`sandwich` vs `payload_first`,
  `sandwich` vs `question_first`), computed on the **control
  set**: the headline paired population further restricted to
  cases whose sandwich cell also completed. The control is
  compared against the headline on the same cases or it is not
  a control (red-team F2, confirmed by B with a worked example
  where the unrestricted control flipped the falsification
  verdict). The headline block's rates stay over its own paired
  population.
- `control_level`: the larger discordant total
  (`n_a_only + n_b_only`) of the two sandwich blocks, or
  `None` when the control set is empty. `n_control` reports the
  control set's size.

### Classification (total function)

Floors frozen at the leading/dateline values:
`MIN_PAIRED_EVALUABLE = 6`, `MIN_DISCORDANT = 2`. The floors
gate this label only, never a reported count or rate. Decision
list, first match wins, over the headline block with
`d = n_qf_only + n_pf_only` and `J = both_fail_dominant`:

1. `n_paired < 6` → `insufficient`
2. `d == 0` and not `J` → `no_discordance_observed`
3. `n_control < 6` → `control_set_too_small`
4. `control_level is None` → `control_unavailable`
5. `d <= control_level` → `within_sandwich_control`
6. `n_qf_only == n_pf_only` and `n_qf_only >= 2` →
   `discordant_balanced`
7. `max(n_qf_only, n_pf_only) >= 2` → `discordant_asymmetric`
8. otherwise → `below_threshold`

### Falsification

There is exactly one definition, and this is it (the same
formula is printed in the `placement.py` module docstring; any
other sentence in this document is commentary on it, not a
second definition):

`placement_effect_falsified` is true iff `n_paired >= 6` and
not `J` and (`d == 0` or (`n_control >= 6` and `control_level`
is not `None` and `d <= control_level`)).

Reading: the falsified condition and the classification are
two views of one condition, with joint collapse as the one
stated exception. Absent joint collapse, falsified holds
exactly when the classification is
`no_discordance_observed` or `within_sandwich_control` at a
sufficient population (the control-set steps in between
report *why* a non-falsified candidate could not be
acquitted: no usable control). Under joint collapse the two
views part: a candidate whose discordance sits at or below a
usable control level is still classified
`within_sandwich_control` while the formula returns false.
Zero discordance falsifies with or without a control; joint
collapse (`J`) never falsifies — a candidate failing the same
cases under both layouts has a capability story, not an
acquittal. The shipped enumeration test pins the equivalence
in this joint-collapse-qualified form. Discordance above the control level with a
one-sided split is the only outcome that supports a directional
placement claim, and even then the claim is a verdict-flip
count on generated fixtures, not a recommendation (the utility
sentence says the practitioner *picks the layout their
template rewards* — on their own fixtures, with this
instrument as the template-axis evidence).

The control is deliberately conservative, and the design says
so with numbers (red-team F6, confirmed as a disclosure
obligation, not a redesign): the falsification bar sits at
`control_level + 1` discordant pairs, where `control_level` is
the *maximum* of two sandwich discordant counts. Under a
symmetric-noise null at n = 24 paired cases, a max-based bar
acquits 88.2% of the time at a true flip probability of 0.10,
69.2% at 0.15, and 15.6% at 0.25 (red-team A's enumeration,
reproduced by B). A supported directional claim must clear a
high bar; the price is that weak real effects are acquitted,
and the readout reports the counts so a reader can apply a
different bar. One mechanism note: the sandwich arm shares the
duplicated question with `question_first`, so its discordance
against `payload_first` partly tracks the headline pair's —
another reason the max is a bar, not an estimate.

### Conditions (per-arm selection)

For each layout arm (`question_first`, `payload_first`,
`sandwich` — the closed-book arm is a screen, not a layout, and
gets no selection readout), the conservative selection rule is
re-applied over the arm's completed cases restricted to the
**shared live population**: cases recorded for every candidate
and screened for none of them. The readout is withheld with
detail `selection_withheld_mismatched_sets` when the restricted
completed sets differ across candidates or are empty (the
needle rule). Each arm's `flip` compares its selection to the
`payload_first` selection recomputed on the same restricted
completed set (the conventional layout is the baseline); the
baseline arm's own flip is `False` when not withheld, and a
flip is `None` when either side is withheld.

## Diff validator

`diff_placement` in `diffreports.py`, full-recompute standard
(the wave-7 F6 settlement): `_placement_recompute` carries its
own generator mirror (its own derive function, templates, topic
list, and neutral pool as literals — no imports from
`placement.py`), regenerates the case set from the protocol's
seed, and compares every record's construction against it
(fixture id set, topic, expected digest). The protocol is
rebuilt from frozen scalars plus the payload's seed and backend
fingerprint and its fingerprint recomputed. Every cell's
verdict is re-derived from its digest and boundary flag; every
completed cell's `input_tokens` must equal the record's
`arm_tokens`; cells and details must be consistent (a cell is
present iff its detail is `""`). Every aggregate, pair block,
control level, classification, falsification flag, and the
conditions are recomputed by mirror code and compared exactly.
A one-break-at-a-time battery pins every reject branch.

## Budget

Worst case: 24 candidates (the run cap) × 24 cases × 4 arms =
**2,304 generations**, stated in the CLI help with the forwards
formula before the run (each generation is one greedy decode of
at most `--max-new-tokens` tokens over a short rendered prompt;
the payload is three sentences by construction). The stage
threads the run's cancel event end-to-end and checks it before
and after every evaluation (the multiturn rule: a cancel
observed during a generation is never mislabelled an evaluation
failure). `MAX_GROUND_RECORDS = 576`,
`MAX_RECORDS_BYTES = 12 MiB` (the house bound).

## Pin blast radius

- New pure module `placement.py` → the test_cycle44 mypy
  source-file pin moves 55 → 56 in the same push, and
  test_cycle44's enumeration of the VALIDATION Evidence section's
  first-paragraph probe list (pinned verbatim) gains the
  placement entry — VALIDATION's own text moves with it
  (red-team F16: this pin was missing from the v1 list; the
  `n_positions` name nearby in experiments.py is an unrelated
  local, a red herring). Correction (cycle 203): the original
  text of this bullet called that list "VALIDATION §6" — no §6
  exists in VALIDATION.md; the cycle-203 red-team traced the
  phantom to this bullet and it is corrected here, at its
  source.
- New `--placement` flag → the cycle-179 parametrized dispatch
  table (27 stage rows → 28, plus its probe-wrapper audit list,
  19 → 20), the cycle-31/64/84 subcommand/flag audits, and the
  args-Namespace contract gain a case.
- New diff section `placement` → test_cycle67's three elements
  (fuzz oracle + `_fuzz_placement_section`; the flip-identical
  gate anchor's original *and* mutant texts; `_identical_expected`'s
  key set) plus test_cycle27's oracle stub entry
  `"placement": {"available": ..., "changed": False}`.
- README Capabilities bullet + measured test count;
  VALIDATION counts and the CI-scope ruff file count;
  CHANGELOG entry; USAGE §8.46 (the boundary paragraph in the
  same words as above) and the §6/§12 inventories if they
  enumerate run stages; RESEARCH_LOG uniqueness entry (after
  the researcher's live verification lands).
- The sample report gains nothing: a section renders only when
  the run recorded its stage, so sample artifacts stay
  byte-identical (pinned by the existing sample tests).

## Adjudicated open points (from the v1 draft)

1. *New classification labels* (`within_sandwich_control`,
   `control_unavailable`, and v2's `control_set_too_small`):
   kept. §4's honesty vocabulary governs fail-closed *detail*
   words; classification vocabularies are per-probe (dateline's
   five labels are its own). The three labels name states no
   existing label names without conflation.
2. *Union screening for conditions*: kept, and pinned by a
   cross-candidate conditions test. The per-candidate blocks
   use per-candidate screening because they are per-candidate
   measurements; the conditions readout compares candidates,
   so its population must be one set — a case any candidate
   answered closed-book is not a clean shared measurement.
3. *Control populations*: resolved by must-fix 2 — the control
   is computed on the headline paired population.
4. *`arm_tokens` equality as a false-reject surface*: the
   grounding is the tree's own scoring contract (one id
   sequence per fixture — the recorded observation in
   `core.evaluate()`, with `scoring.py`'s alignment check
   failing closed on encode inconsistency), the same
   grounding needle already ships. A backend whose encode is
   non-deterministic within a run violates that contract
   everywhere, not just here; the rejection is the instrument
   working.
5. *Duplicate candidate names* (red-team F7, mechanism
   corrected by B): the stage raises
   `ValueError("formats must have unique names")` on duplicate
   format names — the placement stage's own guard, at its own
   boundary — and `_valid_placement` carries the unique-names
   clause. A two-same-named-formats test pins it. (The shipped
   needle stage keys candidates by name in a dict; placement
   states the precondition instead of inheriting the silence.)
