# Prune-sweep probe — design of record

## Status

BUILT in cycle 220 as `run --prune-sweep`, under the fresh
review's BUILD-as-amended verdict (coordinator #62, Part III
below): amendments A–I are incorporated into Part I, which
is the instrument as built. Part II is the round-1
adjudication (coordinator #45, NO-BUILD) whose §3
resubmission candidate the fresh review judged; the
candidate text of record is research §§3–10
(`~/workspace/agents_out/prunesweep/research.md`), of which
Part I is the as-amended, as-built form. Part III is the
fresh-review adjudication itself (coordinator #62).
Endpoints (the doses 0/18/36/72 words, the six arms as
decision structure, the label vocabulary), the population
(24 generated cases under the frozen construction rules,
C-conditioning as designed in §5), and the headline scorer
(`core.evaluate` byte-exact + boundary) are frozen — an
amendment touching any of them would have been a
resubmission, not this build.

# Part I — The instrument as built (candidate text as amended)

## 1. What exactly varies — the treatment dimension

Which *eligible words of a frozen fact-recall passage are
deleted*, at four nested doses, with the answer-bearing
sentence, the instruction, and the question held
byte-identical across every arm. The treatment is a bundle
(§6): information loss, prompt shortening, fact-position
shift, and sentence breakage move together under the
scattered operator, and the report says so in words rather
than letting the label imply a single channel. The
displaced control (`tail72`) deletes a different,
position-extreme set of the same size, so the panel can
say when the honest reading is word-count loss rather
than interior placement.

## 2. Population and frozen texts (amendment C: lists printed)

A generated, stage-local panel of 24 cases (`prune-00` …
`prune-23`, group label `prune`). Each case is a single
user message: a frozen instruction + a 17-sentence
passage + a frozen question, joined by `"\n"`. The
population is defined by the construction rules of §3
plus the invariant battery of §3.3; the lexical lists
below are the rules' deterministic output, printed here
as record-binding (the cycle-218 TOPICS precedent) and
re-verified by the battery on every run.

- Instruction: `Read the passage and answer the question.`
- Fact template: `The {topic} code is {value}.` (exactly
  6 words with a two-word topic; the fact sentence is
  passage sentence 9 of 17, index 8, and is never pruned).
- Question template: `What is the {topic} code? Reply
  with only the code word.`
- Topics (8, cycled over cases): harbor light, summit
  trail, quarry gate, tundra post, granite arch, willow
  ford, copper mill, saddle pass.
- Code values (24, `VALUES[case index]`): amber, birch,
  cobalt, dune, ember, flint, garnet, hazel, ivory,
  jasper, kelvin, linden, magnet, nickel, onyx, pepper,
  quartz, rowan, saffron, timber, umber, violet, walnut,
  yonder.
- Slot pools (12 each). Subjects: archive, ledger,
  cabinet, chapel, meadow, workshop, station, garden,
  library, market, office, depot. Objects: forms,
  letters, tools, charts, samples, records, parcels,
  volumes, tickets, bundles, reports, crates. Places:
  annex, attic, cellar, landing, terrace, alcove,
  gallery, storeroom, corridor, balcony, vestibule, loft.
  Times: morning, midday, evening, weekdays, season,
  harvest, holiday, weekend, noon, dusk, dawn, closing.
- Sentence templates (4, each exactly 9 words; slot `s`
  uses template `s mod 4`):
  `The {subject} holds {object} inside the {place} each {time}.`;
  `A {subject} keeps {object} near the {place} during {time}.`;
  `Workers quietly move {object} past the {place} during {time}.`;
  `Each {time} the {subject} stores {object} under the {place}.`
- Negation list (frozen, for the §3.3 screens and the §6
  dropped-set shares): no, not, never, none, cannot,
  without, neither, nor, nothing, nobody, nowhere,
  hardly, barely, scarcely.

Three list choices were forced by the screens and are
recorded here: `hazel` replaces a value that would have
shared its stem with a topic word; `chapel` replaces a
subject drawn from the topic vocabulary; `corridor`
replaces a place drawn from the instruction vocabulary,
and `harvest` replaces a time word sharing a 4-character
prefix with a code value — the battery (§3.3) rejects any
future list edit that reintroduces such a collision.

## 3. Construction (frozen)

### 3.1 Case generation

Per case, 16 context sentences are drawn by hash-ranked
selection with distinctness rejection: for slot `s` and
counter `c` from 0, `sha256(f"{seed}:{case_id}:slot:{s}:{c}")`
supplies four pool indices (hex slices 0:8, 8:16, 16:24,
24:32, each mod 12); the rendered sentence is kept if it
is not already in this case, else `c` increments and the
draw repeats. The fact sentence is inserted at passage
index 8. Distinctness is over rendered sentences.

### 3.2 The pruning operator (amendment E: key pinned)

The eligible set is the 144 words of the 16 context
sentences (16 × 9), indexed 0–143 in passage order
(sentence slot × 9 + word offset). One permutation per
case orders positions by `sha256(f"{seed}:{case_id}:{position}")`
hex ascending, where `position` is the eligible-list
index (amendment E). Dose arms drop the first 0 / 18 /
36 / 72 positions of that order — nested by construction,
so first-flip and returns are well-defined. `tail72`
drops positions 72–143 exactly (the eligible words of
passage sentences 10–17). The fact sentence, instruction,
and question are never pruned. The dose unit is *words*;
rendered-token deltas are reported per candidate via
`backend.encode`, never asserted and never gated.
Measured on the shipped lists (seed 0): the tail72 drop
set overlaps the dose72 drop set in 32 of 72 words on
case 0 (panel mean 35.92, min 29, max 41) — the control
is displaced, not disjoint, and the report does not
claim otherwise.

### 3.3 Invariant battery and screens (amendments D and C)

`validate_case` re-derives each case from the frozen
rules and asserts, per case: exactly 144 eligible words;
16 distinct context sentences of exactly 9 words; the
6-word fact sentence at index 8; protected parts
byte-identical in every arm; each dose arm removes
exactly its dose of words and nothing else; no context
word shares a 4-character prefix with the case's code
value; slot pools disjoint from the values, the topics,
the fact template's fixed words, and — amendment D —
the instruction and question vocabulary; no context
sentence contains a digit; no context sentence contains
a frozen-negation-list word; and an acrostic screen
(amendment D): the sentence-initial letters of the 16
context sentences must not spell any pool value, as a
whole string or in any single sentence pair window the
screen enumerates. Any failure excludes the case before
generation under the stage detail `prune_screen_failed`,
with the audit's `by_reason` naming the failing screen
(`case_invariant`, `special_token_literal`, or the
boundary-text containment screen, which reuses the
sibling stages' form: no arm content may contain the
rendered boundary text). Context sentences are
declarative statements from the frozen templates — the
templates carry no question, imperative, or negation
shapes (amendment D).

## 4. Arms and scorer

Six arms: `dose0`, `dose18`, `dose36`, `dose72` (the
headline ladder), `tail72` (descriptive displaced
control), `closed_book` (screen only: instruction +
question, no passage). The scorer is `core.evaluate`
byte-exact ∧ boundary, unchanged. Every arm of a case is
requested under the case's own fixture id, so request
seeds are identical across arms by construction. Records
carry answer digests only — no raw answer text — plus
per-arm token counts, per-arm rendered-id digests, the
derived `ids_identical` flag, per-arm content digests,
and the per-case dropped-set shares (§6).

## 5. Decision rule and operating characteristics (amendment A: corrected)

Per candidate, over the cases completed in all four
headline arms, minus closed-book-screened cases (the
closed-book cell completed and passed — passage loss
cannot move a case the candidate answers with no passage
at all), minus ids-identical cases (the dose0 and dose72
renders tokenise identically, derived from the recorded
id digests): C = dose0 passes; A = cases in C failing
dose72, decomposed as in §6 — a `boundary_only` dose72
cell does **not** count toward A (amendment B); B = cases
failing dose0 and passing dose72. Rule: C < 12 →
`insufficient`; C ≥ 12 ∧ A ≥ 5 ∧ two-sided exact McNemar
p(A, B) < 0.05 (`confidentiality.mcnemar_exact_p`) →
`prune_drop_observed`; else `below_threshold`. No winner
readout.

**Corrected operating characteristics.** The candidate
as submitted printed a 40,000-trial Monte Carlo claiming
zero fires on every null ("95% upper bound ≈ 0.00008";
"< 0.01 expected false labels per panel"). All three
sentences are false and are deleted wherever the
candidate printed them. The rule's discreteness is the
defect: at C = 24 the smallest firing A is 6, and
(A, B) = (6, 0) already gives p = 0.03125, so any null
with discordance mass fires at a per-decision rate of
about one percent. The figures below are exact
enumerations (integer arithmetic, independently
re-derived at build time), conditional on headline
completion, and are printed in the report section with
their conditioning stated.

Stated basis — pass0 ~ Bernoulli(p0); fail72|pass0 = q;
pass72|fail0 = r; the two arms conditionally independent
given the case class:

- random (p0 = 0.5, q = r = 0.5): **0.0121**
- competent (p0 = 0.9, q = r = 0.1): **0.0143**
- jitter (p0 = 0.97, q = r = 0.02): **0.000005**
- power at a 25 / 50 / 75 percent conditional drop
  (p0 = 0.9, r = 0.1): **0.3969 / 0.9601 / 0.9996**

Exchangeable basis — cases exchangeable with marginal
pass rate m and per-direction discordance v (joint
(pass, pass) = m − v, (pass, fail) = (fail, pass) = v,
(fail, fail) = 1 − m − v), competent m = 0.9:

- v = 0.09 / 0.05 / 0.02: **0.003 / 0.0004 / 0.000004**

Family-wise per 24-candidate panel: stated basis
E = 0.34, P(≥ 1) = 0.29; exchangeable basis E = 0.07,
P(≥ 1) = 0.07. On the stated basis the closed-book
screen shrinks the competent null from 0.0143 to 0.0017
at the frozen screen floor (16 unscreened passes). The
firing census is exact: 492 (C, A, B) triples fire, none
with A ≤ B. The conditioning matters and is printed:
under free completion a model that attempts every case
and fails often completes few joint cells, and these
rates rise toward the stated random row — the
enumeration's conditioning on completion is a disclosed
limitation of the printed figures, not a hidden one.

## 6. Report shape (printed, in this order, per candidate)

Per-arm pass rates with attrition beside them (context
overflows and evaluation failures counted, never
pooled); the per-arm four-way decomposition in one
frozen order over (verdict, boundary_found,
answer_digest) — `exact_pass`, `boundary_only`,
`wrong_value` (the digest is a different pool value's),
`non_value` (amendment B: decidable from digests alone);
the headline counts C, A, B with boundary attrition
printed beside A and the decision; A by first-flip dose;
the descriptive shape readout (first-flip histogram,
returns, per-dose pass rates — never gated, §8); the
tail72 contrast with its discordant counts; the id
report (per-arm id-length deltas, ids-identical counts,
same-ids flips); and the per-dose dropped-set shares for
the numeral and negation classes (expected 0 under the
§3.3 screens; printed and validator-mirrored, amendment
D).

**Bundle fencing (amendment F).** The section states
that the label attributes the drop to the named
word-loss treatment as a bundle — information loss +
prompt shortening + fact-position shift + sentence
breakage — and prints the executed operator constants:
under dose72 the fact sentence's word position shifts by
a mean of 36.0 words (sd 3.01, the exact hypergeometric
figure) while under tail72 the shift is exactly 0 and
the fact instead lands in passage-final position; an
expected 15.95 of the 16 context sentences are partially
broken at dose72 (0.02 removed whole, 0.02 intact) versus
0 under tail72. When the tail72 discordance is within
noise of dose72's (the frozen rival test: dose72 fails
exist, and either there are no discordants in either
direction or the two-sided McNemar p of the discordants
is ≥ 0.05), the section prints the sentence: the honest
reading of this panel is word-count loss at 50 percent
of the eligible words, not interior placement.

## 7. Utility and scope (amendment G)

The headline's subject is the class of operations that
delete *interior* passage words: question-aware
compressors of the LLMLingua class, middle-chunk RAG
trimming, and word-level summarisation/redaction
pipelines. Character caps and tail cuts — the operations
whose loss is positional by definition — are assigned
explicitly to the tail72 contrast, not to the headline.
The instrument cannot detect a trim that deletes the
answer itself: the fact sentence is protected precisely
so the measurement is sound, and answer-deleting trims
are named out of scope in the section's printed
limitations. Monday-morning sentence: a prompt engineer
whose pipeline deletes interior passage words uses this,
per candidate, to learn whether deleting half the
non-answer-bearing words of a frozen fact-recall passage
(answer sentence, instruction, and question
byte-identical) produces a large verdict drop at the
printed operating characteristics, before that trim
ships.

## 8. Shape gate — declined, with the executed figures (amendment I)

The shape readout is descriptive, never a gate. Under
the frozen graded alternative — a linear interpolation
of the conditional top-dose drop, q_d = q72 · d / 72, at
p0 = 0.9, r = 0.1, so the base 50-percent-drop power is
0.9601 — a returns cap collapses power to **0.094** (the
candidate printed 0.031; that figure does not reproduce
under the printed interpolation and is corrected here),
and an intermediate-support requirement collapses
threshold-drop power to exactly **0.000**. Any shape
gate makes a real shape invisible; the panel prints the
shape and declines to judge it.

## 9. Counts (frozen, with derivations)

24 cases × 6 arms = 144 generations per candidate;
2,304 headline generations per 24-candidate panel;
3,456 legal maximum generations (24 × 6 × 24); 576
records maximum; the house records cap of 12 MiB applies
unchanged. The stage is generation-based like its
siblings: it runs its own cells under the run's backend,
formats, seed, and cancellation, inside the run, and the
whole-stage failure payload is
`{"available": false, "detail": "prune_unavailable"}`
in the tree's frozen shape.

## 10. Records and validator mirror

Records are digest-only (answer digests, id digests,
content digests; no raw answers). The diff validator
holds an independent generator mirror: it rebuilds the
protocol whole (including the case-material digest over
all frozen lists and the ground digest over the
regenerated cases), regenerates every case, rebinds
every record's content digests to the regenerated arm
contents for all six arms, checks the recorded
`ids_identical` flag against its derivation from the
recorded id digests, recomputes the dropped-set shares,
binds every completed cell to its record's arm token
count, re-derives every verdict from digest identity,
and recomputes every aggregate — including the headline
population pipeline in its frozen order — before any
payload is called valid. Drift metrics per candidate:
decision, C, A, boundary attrition, B, population size,
the tail72 discordants, the rival reading, and same-ids
flips at dose72, plus protocol / generator / backend
drift and candidate added/removed sets.

## 11. The claim (only falsification-class sentence)

For this candidate, on this frozen panel, deleting half
the non-answer-bearing words of a fact-recall passage —
as the named bundle of §6 — did (or did not) produce a
large verdict drop at the operating characteristics
printed in §5. What would prove the instrument wrong: a
candidate whose dose72 verdicts fall while its tail72
verdicts hold, on a panel where the printed null rates
say the joint pattern is common — the contrast is
printed precisely so that reading can be checked.

## 12. Naming audit (amendment H, re-executed at build)

The flag is `run --prune-sweep`; the help text's first
sentence states that the probe prunes passage words.
Re-audit at cycle-220 build start, against the
post-cycle-219 namespace: the `run` subparser carries 61
optionals before this flag (argparse introspection of
`cli.build_parser()`); 70 `--flag` literals exist in
`cli.py` overall; `--prune-sweep` / `prune_sweep` occur
nowhere in the package or tests before this build;
"prune" occurs in no source file and once in the docs
(USAGE.md, in the candidate-set prose sense of selecting
survivors). No collision.

## 13. Uniqueness pass (primary sources)

The neighbours were resolved at primary in the design
rounds (Part II §5): LLMLingua (arXiv:2310.05736),
LongLLMLingua (arXiv:2310.06839), and LLMLingua-2
(arXiv:2403.12968, §4.2 for the word-level compression
unit τ) are *subjects* of the measurement class in §7,
not implementations of it: none ships a seeded,
nested-dose, answer-protected word-loss panel with a
derived decision rule, a displaced tail control, and
digest-only validator-mirrored records. In-tree
neighbours are bounded exactly in Part II §3 and the
round-1 text: density (character share of one field),
token-depth-sweep (prefix truncation), needle-depth
(insertion depth), distraction (added distractors),
surface-noise (character corruption), ablation (field
removal) — each varies a different dimension; this probe
is the only one that deletes interior words of a fixed
passage under protection of the answer sentence. The
claim is the combination, and it is scoped to the
sources surveyed in the design rounds, which did not
re-survey the general evaluation harnesses this round —
that limit is stated, not hidden.

# Part II — Round-1 adjudication (coordinator #45, NO-BUILD)

## Round summary (coordinator #45, design round only — nothing built)

Wave-11 slate rank 4: interior content-token loss (tentatively
`run --prune-sweep`). The candidate sketch
(`~/workspace/agents_out/wave11/slate_candidates.md` Candidate 4):
drop a seeded fraction of the *content* tokens of the last user
message (template skeleton untouched) at doses 10% / 25% / 50%,
seeded per fixture, and report the first-flip dose per candidate —
the dose at which any candidate's pass set changes on the common
completed set — as a truncation-budget readout, worst case
3 doses × 100 fixtures × 24 candidates = 7,200 generations.

- Tree at main `a6705d1b9607` (cycle 213 closed green, 8,442
  tests, precise coverage 95.3455%). This round was READ-ONLY
  except for this doc and the briefs; no suite was run, no push
  was made, no tag was cut.
- Specialists (blind to each other): researcher
  (`~/workspace/agents_out/prunesweep/research.md`), red-team A
  (`redteam_a.md`), red-team B (`redteam_b.md`), verifier
  (`verifier.md`). All four briefs landed; the verifier waited
  for all three briefs to stabilise, mechanically resolved every
  citation, checked every external source at primary, and
  re-executed every load-bearing PoC with its own scripts; the
  coordinator re-executed the load-bearing claims itself (§4).
  **Fabrication sweep: clean — a fifth consecutive round.** No
  invented symbol, paper, or count in any brief; the errors found
  were four substance-true line drifts in the researcher's
  citations and one labelling imprecision in red-team A's
  survival Monte Carlo (its fatal is stronger, not weaker, once
  corrected), all handled in §5.
- Convergence: **unanimous NO-BUILD for the sketch as
  specified** — researcher, both blind red teams, and the
  verifier each reached it independently, fatal by fatal, by
  execution. The two red teams, working blind, converged on the
  same core defects (evidence destruction, unfrozen policy,
  grammaticality collapse, the noise-firing falsifier) with
  independently executed numbers that agree (perfect-reader
  pass rates 0.86–0.89 / 0.74–0.75 / 0.49–0.50 at
  10% / 25% / 50%).
- The researcher additionally specifies a complete redesigned
  instrument (generated panel, word-level doses over a protected
  passage, nested draws, a derived decision rule). The verifier
  reproduced its decision math exactly and ruled it a
  **resubmission, not an amendment**, under this program's own
  test — the unit, the population, and the readout all change —
  and the researcher himself classifies it the same way. The
  coordinator concurs: the verdict is **NO-BUILD** — for the
  sketch, and as a build order for the redesigned instrument in
  this round. §3 records the resubmission candidate in full,
  including the two points no brief fully closed (the
  answer-protection utility wound, the absent
  length-held-constant arm).

## 1. Why NO-BUILD: the fatal defects of the candidate as specified

Each defect below is individually blocking and was established
by execution (attribution: R = researcher, A = red-team A,
B = red-team B, V = verifier, C = coordinator).

### F1 — Evidence destruction: the dose-response is answer-survival arithmetic, not robustness

The sketch holds `expected` fixed while pruning deletes the
tokens the expected answer is derived from, with no solvability
or redundancy screen. A perfect-reader stub through the real
`core.evaluate` passes at exactly the closed-form survival
probability of its evidence: 1−k/n for k evidence tokens among
n — executed at 0.86/0.75/0.49 (A) and 0.890/0.740/0.485 (B) at
10/25/50% for single-token evidence, rising to
1.000/0.995/0.910 when the same evidence is merely repeated 3×.
A prompt-ignoring stub passes all doses. The headline curve
therefore measures fixture redundancy, not template robustness:
two fixtures with identical templates and different evidence
redundancy produce different "first-flip doses" with no
template property involved. (A F1, B F1, V upheld, C
re-executed: survival = 1−dose to three decimals at n=20 and
n=100.)

### F2 — The pruning policy is an unfrozen second treatment riding on the dose

Which tokens are removed is not specified: policy, seed
derivation, dose-rounding, and nesting are all unfrozen, and
they are not cosmetic. At one fixed dose (drop 5 of 18 words),
three honest policies give opposite verdicts — uniform-random
broke the "code is" context around a surviving answer while
function-word removal passed (A F2, executed). The two red
teams themselves, instructed only by the sketch, chose
different roundings — the sketch does not determine its own
instrument. Rounding is load-bearing at the shipped scale:
floor(10%) of every shipped fixture's last user message is
**0 tokens** (C re-executed on `examples/fixtures.json`: spans
of 7, 8, 5, and 8 words), so the 10% arm is ids-identical filler
on the entire shipped population, while ceil-rounding silently
turns "10%" into 12.5–14.3%. Non-nested draws additionally make
the sketch's own flip-back clause ("a verdict that flips at
25% and flips back at 50% is noise") incoherent: with
independent draws per dose there is no fact of the matter about
which draw a return belongs to. (A F2, B F2, V upheld.)

### F3 — Grammaticality and semantics collapse: the construct is not "loss"

Pruned text is not shorter text; it is broken text, and the
breakage is not dose-monotone in any single construct.
Single-token deletions invert ground truth while `expected`
stays frozen — "is not locked" → "is locked",
"Ana, not to Ben" → "Ana, to Ben" (A F3, B F5, executed) — at
doses as low as one token (~7% of a 14-word span). A numeral
drop deletes the answer inside otherwise fluent text; a filler
drop is harmless. The sketch pools three different events —
evidence loss, meaning inversion, and harmless noise — under
one "content loss" headline with no decomposition, so the
construct it measures is robustness-to-ungrammaticality-and-
inversion, not robustness-to-loss. On the tree's own fixtures
the "content span" is an instruction sentence (F7), so the same
instrument would measure instruction mutilation there and
document loss elsewhere, pooled. (A F3, B F5, V upheld.)

### F4 — No decision rule: the first-flip falsifier fires on noise with P = 1.000

The sketch's falsifier — "the first dose at which any
candidate's pass set changes" — is the any-difference rule this
program has already killed twice (directive F2,
instruction-distance). Over the sketch's own 7,200-cell panel
it fires with P=1.000 at per-cell jitter ε ≥ 0.001 (A: 2000/2000
trials at ε ≥ 0.005; B: conditioned selection-flip 0.778–0.998;
C re-executed: 300/300 at each of ε = 0.001, 0.005, 0.01,
analytic 1−(1−ε)^7200 = 0.9993 at ε=0.001). The selection half
is a one-cell tripwire. The falsifier is also shape-blind and
self-contradictory: decay, threshold, single-dip, U/recovery,
and *improvement* all return the identical headline
"FALSIFIED", while the sketch's own text says a flip-and-return
"is noise, counted as such" — a clause the headline cannot
express. Borrowed sibling floors fire ≈1.000 at this panel's
N=100 (B, executed). The common-completed set the headline
conditions on is post-treatment in both directions, with no
attrition reporting. (A F4, B F4, V upheld.)

### F5 — The "sweep" headline is permutation-invariant, and one deletion bundles four channels

A first-flip dose is invariant to permuting the dose order:
graded decay, a threshold cliff, a single dip, and a U all
return the same number, so the word "sweep" promises a shape
the readout cannot distinguish (A F5, executed; the
instruction-distance F5 settlement). Separately, one interior
deletion simultaneously removes information, shortens the
prompt, shifts every downstream token's position, and breaks
syntax — four channels, one number, no length-matched
(pad / sham / placeholder) control to separate them, and no
such control exists anywhere in the sketch. The claim names
one cause ("loss fraction"); the treatment delivers a bundle.
(A F5, B F6 in part, V upheld.)

### F6 — The dose unit is unexecutable as specified: token fractions are per-checkpoint, and there is no frozen tokenizer

Doses are stated as fractions of *content tokens*, but the tree
has no per-candidate frozen tokenizer (the instruction-language
fertility settlement: no unexecuted token figures may be
printed). The same nominal 25% dose deletes different text
under different tokenizers — word-level: clean whole words;
character-level: the answer `4821` → `421` (A F6, executed).
A "10% dose" is therefore not one treatment across the
24-candidate panel; it is 24 different treatments sharing a
label, and any cross-candidate comparison of first-flip doses
compares unlike quantities. (A F6, V upheld.)

### F7 — The population is undefined, and empty on the shipped fixtures under any honest floor

The sketch says "100 fixtures" without stating the population
or the split scope. On the only population the tree ships, the
last user messages are 5–8-word *instructions that contain the
expected answer itself* ("Reply with exactly BLUE…",
"Return only the numeral 7.") — not the long documents of the
utility paragraph. C re-executed: all four shipped cases
(search 2, held-out 2) have last-user spans of 5–8 words;
floor(10%) = 0 on all four; any span floor ≥ 9 words — the
minimum at which a 10% dose is even one token with margin —
makes the evaluable share **0/2 search fixtures**. The utility
population (long pasted documents, support tickets) is a
different population, owned in-tree by the needle/distraction
family's generated corpora, and importing it is a new
instrument, not a reading of the sketch. The copied token-depth
pattern also drags in treated held-out winner evaluation,
which the sketch never scopes. (A F8, B F2/F7, V upheld.)

### F8 — The span machinery does not exist for this use, pruned id sequences are non-canonical, and the build surface is misstated

The sketch's span location rests on the behavioural template
probe's differencing, but `templateprobe.find_span`
(`templateprobe.py:283`) resolves a span only between
sentinels the probe itself inserted into synthetic probe
content; on real fixture renders it returns None for every
fixture × family (R S1, B F3, V and C verified: the function
counts sentinel occurrences and returns None when either does
not occur exactly once). No tree function maps a message's
character span to a token-id span in an arbitrary render. The
substrate is worse than unlocated: a seam token (`'\nR'`,
executed with a merging tokenizer) is simultaneously skeleton
and content, so no id interval equals the content span and
"skeleton untouched" is unachievable in id space; and pruned
id sequences are the tokenisation of *no text*
(decode→re-encode ≠ pruned ids, executed by A, B, and V) —
prompts no truncation tool emits, with the sketch's defence
being disclosure of the artefact share, not removal of the
artefact. The cited wrapper beside `backends.py:811` exposes
`encode(text)` only and cannot locate a last-user span from
text alone; delivery would need a plan-keyed surface. The
sketch's "M–L, a copy of the token-depth pattern" understates
the build surface on every axis. (R S1, A F9, B F3, V upheld.)

### F9 — No frozen fail-closed vocabulary, decomposition, or attrition discipline

The sketch freezes no vocabulary for span failure, seam
straddling, non-canonical renders, zero-eligible populations,
or per-dose attrition; no answer-decomposition (the negation
settlement's digest-derived classes); no paired-subset seed
discipline; no native-digest binding. Every one of these is a
place a build would have to invent verdict words, which the
program forbids. Verified by reading the full slate text
(B F7, V). (A F10, B F7, V upheld.)

### Partially upheld — duplication and utility (A F7, B F6)

Not a standalone fatal, recorded because it kills the sketch's
novelty paragraph: the instrument's *shape* — a dose ladder
over prompt ids with a first-flip headline — is the
token-depth sweep's (`experiments.py:3621`,
`backends.py:810`), and the production operations the utility
paragraph names (tail cuts, character caps) are already
shipped as `prompt_token_depth_sweep` /
`prompt_sensitivity_sweep` with the same first-flip headline.
Text-level interior deletion also already ships
(perturbation `punctuation_strip`, surface-noise). What is
genuinely unshipped is only the residue: dose-graded *interior*
deletion itself. On documents, uniform pruning is furthermore
distraction run in reverse — ~90% of drops hit filler (A,
executed). A resubmission must argue its utility against the
shipped tail-cut sweeps, naming the real operation interior
deletion models; the sketch does not.

## 2. Amendment or resubmission?

The settlement from the rank-2/rank-4 rounds applies: an
amendment corrects named defects of the *same* frozen
instrument (same population, unit, arms, scorer, and readout
shape); a redesign that changes the population, the unit, or
the headline readout is a resubmission candidate requiring a
fresh blind design review, not a build order. The researcher's
redesign changes all three — population (shipped fixtures →
generated 24-case panel), unit (token fraction of an id span →
word count over a protected passage, delivered by fixture
surgery), and readout (first-flip dose → a derived
McNemar-gated classification with descriptive shape) — and
additionally protects the answer-bearing sentence, which
changes the treatment itself. It is therefore recorded in §3
as a **resubmission candidate only**. No build cycle may act
on it without a fresh design round.

## 3. The resubmission candidate (recorded, NOT a build order)

The researcher's redesigned instrument (`research.md` §3–§10),
verifier-checked (decision math reproduced exactly; PoC2
population invariants and PoC3 scorer path re-run clean):

- **Population.** A generated, stage-local panel of 24 cases.
  Each case is a single user message: a frozen instruction + a
  17-sentence passage + a frozen question. Split hygiene and
  generator mirroring follow the tree's generated-panel
  precedents.
- **Pruned unit and protection.** The eligible set is all 144
  words of the 16 context sentences (16×9, construction-
  asserted). The answer-bearing fact sentence (fixed position
  9 of 17), the instruction, and the question are **never
  pruned** — this is what makes the instrument sound (F1/F3
  cannot fire by construction) and simultaneously its
  principal wound (below).
- **Dose ladder and operator.** Doses 0 / 18 / 36 / 72 words
  (0 / 12.5 / 25 / 50% of the eligible set), **nested** via one
  sha256-ordered permutation per case, so first-flip and
  returns are well-defined (curing F2's incoherence). The dose
  unit is *words* — executable and validator-mirrorable with
  no frozen tokenizer; rendered-token deltas are *reported*
  per candidate via `backend.encode` (`id_length_delta`),
  never asserted and never gated (the F6 settlement). Delivery
  is fixture surgery (the `perturbation.py` precedent): no
  backend wrapper, no span location, no id surgery — F8's
  machinery is never needed.
- **Controls.** `tail72`, a dose-matched displaced control
  (descriptive only), and a `closed_book` screen arm (the
  placement pattern).
- **Scorer.** `core.evaluate` byte-exact + boundary, unchanged.
  Decomposition (`exact_pass` / `wrong_value` / `non_value`)
  derived from answer digests only.
- **Decision rule (derived, never borrowed).** Per candidate:
  C ≥ 12, A ≥ 5, two-sided McNemar (`confidentiality.py:293`)
  p < 0.05 → `prune_drop_observed`, else `below_threshold` /
  `insufficient` in the frozen fail-closed vocabulary. 40,000-
  trial Monte Carlo: 0 fires on random, competent, and jitter
  nulls; power 0.40 at a 25% top-dose drop, 0.96 at 50% —
  printed in the brief as weak-evidence labelling for
  `below_threshold`, not hidden.
- **Shape.** A descriptive readout, *not* a gate — with
  executed proof: a returns-cap collapses graded-drop power
  0.960 → 0.031 and intermediate-support collapses
  threshold-drop power 0.960 → 0.000, so any shape gate makes
  a real shape invisible (curing F5 by declining to gate what
  cannot be distinguished honestly).
- **No winner readout** (the numform settlement — generated
  panel). Counts: 2,304 headline generations, 3,456 legal max.
- **Claim, scoped to the treatment actually delivered:**
  robustness to *collateral* word loss in non-answer-bearing
  context, per template family — never "truncation budget" in
  general.

**What the fresh review must settle head-on (no brief closed
these):**

1. **The answer-protection utility wound.** Protection empties
   the scariest cell by construction: the instrument measures
   robustness to collateral loss only and *cannot* detect a
   trim that deletes the answer itself — which is what
   production trimmers actually risk. The researcher's defence
   (the unprotected version manufactures unanswerable cases
   and pools three treatments — F1/F3) is a judgment, not an
   execution. The fresh review must decide whether the
   surviving utility (a collateral-loss curve per family)
   justifies a probe at all.
2. **No length-held-constant arm.** The redesign has a
   dose-matched *displaced* control but no placeholder /
   sham arm holding prompt length constant, so the
   shorter-prompt channel of F5's bundle is disclosed, not
   separated. Red-team B's resubmission conditions require one.
3. Red-team B's remaining conditions, in full: per-dose
   printed grammaticality / negation / numeral shares of the
   dropped set; per-dose attrition reporting; search-only
   scope stated; the utility paragraph argued against the
   shipped tail-cut sweeps (F-partial).

## 4. Coordinator re-verification (own execution, 2026-10-27)

- Shipped population (F2/F7): read `examples/fixtures.json`
  directly — 4 cases; last-user spans 7 (`search-copy-blue`),
  8 (`search-latest-green`), 5 (`held-number`), 8
  (`held-latest-word`) words; floor(10%) = 0 on all four.
- Span machinery (F8): `templateprobe.py:283` `find_span`
  counts left/right sentinel occurrences in the render and
  returns None unless each occurs exactly once — sentinel-
  dependent by construction, as R/B/V state.
- Falsifier (F4): own Monte Carlo, 300 trials per ε over the
  sketch's 7,200-comparison panel — fires 300/300 at each of
  ε = 0.001, 0.005, 0.01 (analytic 0.9993 at ε=0.001).
- Evidence survival (F1): own Monte Carlo — perfect-reader
  pass rates 0.902/0.753/0.503 (n=20) and 0.901/0.750/0.492
  (n=100) at 10/25/50%, matching 1−dose to simulation noise.
- Inversion (F3): "The door is not locked" minus `not` →
  "The door is locked" — ground truth inverted by one word.

## 5. Fabrication sweep and minor errors

Clean, fifth consecutive round. Every cited tree symbol
resolves; all three external citations resolve at primary
exactly as cited (LLMLingua, arXiv:2310.05736; LongLLMLingua,
arXiv:2310.06839; LLMLingua-2, arXiv:2403.12968, §4.2 for the
word-level compression unit). Non-load-bearing errors,
corrected here: the researcher's line citations drift by a
few lines in four places (perturbation operators at :33, not
:20; distraction levels at :74, not :57; needle DOSES at :102,
not :57; negation DECISIONS at :95, not :107) — substance
intact in each case; red-team A's survival Monte Carlo also
requires local context survival, so its pass rates are a lower
bound on the pure survival figure — F1 is stronger than A
stated, not weaker. The researcher's neighbour-tool re-survey
is flagged UNEXECUTED in its own brief; the uniqueness claim
stands only in its combination-scoped form and no stronger.

## 6. Queue state

Wave-11 adjudication so far: rank 1 (directive) NO-BUILD,
rank 2 (instruction-distance) NO-BUILD, rank 3 (history-frame)
NO-BUILD, rank 4 (prune-sweep) NO-BUILD — each with a
resubmission candidate recorded, none queued as a build. The
resubmission queue now holds seven candidates: payload §2,
answer-syntax `--document-syntax`, instruction-language §3,
directive §3, instruction-distance §3, history-frame §3, and
this round's §3 — none of them a build order. Next in the wave-11 slate: rank 5 (question
bundling, `--bundle`) design round.

# Part III — Fresh-review adjudication (coordinator #62)

# Fresh-review adjudication — prune-sweep §3 resubmission (coordinator #62, 2026-10-28)

**Verdict: NO-BUILD as written. BUILD-as-amended**, under the parent's
amendment test: every required amendment below is text, pins, reporting,
or record-binding; none touches the endpoints (doses 0/18/36/72 words,
the six arms as decision structure, the label vocabulary), the
population (24 generated cases under the frozen construction rules,
C-conditioning as designed in §5), or the headline scorer
(`core.evaluate` byte-exact + boundary; the rule C≥12 ∧ A≥5 ∧
`mcnemar_exact_p(A,B) < 0.05`). A build that omits any of amendments
A–I is not this verdict's build. If BUILD: do not build in this round —
the parent sequences build cycles.

**Two boundary calls, flagged for the parent.** (1) Amendment C
completes the population's lexical lists by disclosure (the candidate
defines the population by construction rules + asserted invariants and
explicitly defers the lists to the build); if the parent judges
list-completion to be population authoring rather than record-binding,
this verdict degrades to RESUBMISSION with amendments A–I as its
required settlements. (2) Amendment B (boundary disposition) is the
amendment nearest the line; it is justified by the candidate's own
negation citation and by the instruction-distance amendment-B1
precedent (identical defect, identical cure).

Specialists: researcher (`researcher.md`, lean BUILD-as-amended),
red team A (`redteam_a.md`, KILL), red team B (`redteam_b.md`, KILL) —
blind to each other. Every load-bearing number below was re-derived
by the coordinator's own execution (`coord_oc.py`, `coord_oc2.py`,
`coord_construction.py`, `coord_decide.py`, `coord_shape.py` in this
directory) or cross-executed by at least two parties with agreement.

## What survived blind attack (executed)

- **F1 is cured.** A perfect reader is flat across all arms: the
  researcher ran a perfect-reader stub through the real
  `core.evaluate` — 24/24 exact+boundary at dose0/18/36/72 and
  tail72, 0/24 closed-book (the screen demonstrably discriminates).
  The coordinator's own construction + stub executions agree. The
  original survival arithmetic (1−dose) does not apply: the evidence
  sentence is structurally unprunable. The unprotected counterfactual
  quantifies the protection's necessity: P(value word dropped at
  dose72) = 0.480, P(≥1 fact word dropped) = 0.9820 (researcher,
  closed-form plausible, coordinator-checked arithmetic).
- **Power rows are exact.** Four independent computations (coordinator
  exact enumeration + MC cross-check; A; B; researcher exact + MC)
  reproduce the printed table under the fingerprinted conditional
  model (p0=0.9, reverse=0.1): q72=.25 → 0.3969, q72=.50 → 0.9601,
  q72=.75 → 0.9996 (printed 0.397/0.960/0.9995). The rule is powered
  for large drops and prints its 25%-drop weakness (0.397) itself.
- **McNemar transplant is sound at its joints.** `confidentiality.py`'s
  `mcnemar_exact_p` is two-sided exact, `None` iff A+B=0 (unreachable
  under A≥5); all six printed worked values reproduce exactly
  ((5,0)=0.0625 no-fire, (6,0)=0.03125, (7,0)=0.015625, (8,1)=0.0391,
  (9,1)=0.0215, (10,2)=0.0386). Exhaustive feasible-triple scan
  (researcher, consistent with the coordinator's minimal-A table):
  492 firing triples, **zero with A ≤ B** — the rule cannot fire on
  improvement; effective floor A≥6; max firing B=6.
- **Decidability passes the Fatal-1 test.** The verdict classes are a
  total function of (answer_digest, boundary_found) given the frozen
  enumerable 24-value pool — no text parse is ever needed (contrast
  instruction-language class 5). Coordinator battery: 17/17
  stage/validator agreement under the boundary-first reading; B's
  battery: 13/14, the single disagreement being the boundary cell
  whose disposition the text under-pins (amendment B). Prose/value
  collisions inside the residual `non_value` class are the design's
  stated echo discipline, not undecidability.
- **Counts are clean.** Four parties swept: 16×9=144 eligible words,
  150-word passage, fact at position 9 of 17, doses 12.5/25/50%,
  2,304 headline + 576 + 576 = 3,456 generations, 576 records — no
  contradiction between design doc §3 and research.md §§3–10.
- **Duplication residue is genuine — conceded by both red teams.**
  Executed: all five shipped perturbation operators remove 0 interior
  words; `TokenTruncatedPromptBackend` is tail-only; distraction /
  density / instruction-distance only add; needle only moves an
  intact fact. No shipped operator can produce a dose72-equivalent
  fixture. The scaffolding is needle's (conceded); the residue is
  the operator + protection + derived classification, combination-
  scoped as the candidate claims.
- **Contamination CLEAN.** Generated stage-local panel; no reference
  to `examples/fixtures.json`/`.csv` or the demo population anywhere
  in the candidate path (B, grep-verified; coordinator concurred).
  The directive candidate's fatal is not repeated.
- **Vocabulary conformant.** Labels and stage/cell details match the
  negation shapes exactly (`negation.py:95-111`, verified); one new
  label formed on the `negation_drop_observed` pattern.
- **tail72 identity.** Coordinator + A + B: tail72's drop set is
  exactly the 72 eligible words of sentences 10–17 (the entire
  post-fact context); fact offset under tail72 is unchanged.
- **Uniqueness survey now executed.** The original brief's
  UNEXECUTED neighbour survey was run this round (researcher §6):
  LM Eval Harness, HELM, Inspect, OpenAI Evals, DSPy, promptfoo,
  PromptPerfect — none ships seeded, nested, answer-preserving
  interior word loss with a derived classification (HELM is nearest
  and differs on protection, doses, nesting, and classification
  simultaneously). All three LLMLingua citations re-resolved at
  primary, including LLMLingua-2 §4.2's word-defined τ.

## The decisive defect — the printed null OC is false (all parties agree)

The §5 table prints 0 fires in 40,000 Monte Carlo trials on the
random, competent, and jitter nulls, and derives "95% upper bound
≈ 0.00008" and "expected false labels per 24-candidate panel < 0.01".
Under the candidate's own stated model (research.md §10: pass0
Bernoulli; per-dose fail probabilities conditional on the dose0
outcome; arms conditionally independent) — the model that reproduces
the printed power rows to three decimals — exact enumeration gives:

- random labeller: **0.0121** per decision (≈484 expected fires per
  40,000; P(0/40,000) ≈ 10⁻²¹²);
- competent (p0=.9, q=r=.1): **0.0143** (≈573 expected; P ≈ 10⁻²⁵¹;
  A's independent MC: 568/40,000; researcher's: 562/40,000);
- jitter: 0.000005 — the only null row that reproduces.

Under the exchangeable marginal-preserving completion (competent:
fail|pass=.1, pass|fail=.9, discordance v=.09) the competent rate is
**0.0030** (three parties agree); symmetric-discordance variants give
0.000004 (v=.02) and 0.0004 (v=.05). No parameterisation any party
constructed reproduces 0/40,000 on the random or competent rows; a
0/40,000 result is obtainable only from a degenerate
perfectly-correlated null generation that §10's stated model
disavows. Corrected family-wise at the stated competent basis:
**E = 0.34 false labels per 24-candidate panel, P(≥1) = 0.29** (at
the exchangeable competent basis: E = 0.07, P(≥1) = 0.07) — within
this program's shipped tolerances (field-order prints ≈1.2 expected;
document-syntax's 0.369 was adjudicated not fatal), but the opposite
of the printed "< 0.01". Further executed sensitivities the printed
paragraph omits: candidate-level heterogeneity moves the rate by
construction (direction-symmetric mixtures 0.0021–0.0079, A;
stated-conditional beta-frailty 0.029–0.068 for κ=50→10,
coordinator); conditioning on C≥12 is not ancillary (B:
conditional fire 0.164/0.021/0.011/0.003 at per-case pass
.3/.5/.7/.9 under the exchangeable basis); screening shrinks the
effective N and moves both null fire and power down (A: competent
0.0143 at N=24 → 0.0017 at N=16; B: 0.0121 → 0.0072 → 0.0019 at
N=24→17→14, p=0.5 basis). The researcher's mitigation — that a
shared-label/jitter generation is the physically correct null for
greedy decoding and restores ≤0.0005 — is recorded but NOT credited
as load-bearing: greedy determinism bounds nothing about cross-
prompt verdict flips, and §10's stated model is the candidate's own.
The defect is confined to reporting text (the table, the §5 Reading
paragraph, §6 item 7's mandated prose); the rule, its constants,
and its true operating characteristics are unchanged by the
correction. Amendment A.

Two corollaries: (i) the graded and threshold power rows are one
computation printed twice (the rule reads only dose0/dose72) — the
corrected text must say so; (ii) design doc §3's preamble claim
"verifier-checked (decision math reproduced exactly)" is falsified
for the null rows — the round-1 verifier re-ran the researcher's
simulator, artefact included. Recorded as a side finding.

## Wounds (sustained, amendment-class)

- **Channel bundle (A-F1, B-F2).** Three construction-determined
  channels fire the rule at C=24, A=24, B=0, p=1.19×10⁻⁷ on stubs
  (both red teams, coordinator's context-depth stub agreeing):
  fact-position shift (Hypergeometric mean 36.0 words, sd 3.01,
  P(shift=0)=6.8×10⁻⁴³ at dose72; exactly 0 under tail72),
  prompt shortening (a pure length stub fires identically on
  dose72 and tail72), and sentence breakage (expected 15.95/16
  context sentences broken at dose72 vs 0 under tail72). B would
  make the missing length-held-constant arm fatal (design-doc §3
  settlement item 2; density's `sham` precedent). **Settled
  against fatality, with reasons:** the claim (§4) names the
  *operation* — removing the first 72 words of the frozen drop
  order — not a channel; shortening, fact shift, and breakage are
  constituents of that operation as any production word-deletion
  delivers it, not rival causes of it (contrast
  instruction-distance, whose claim named a channel — distance —
  and therefore needed the in-rule displaced control). A
  placeholder arm is a *substitution* treatment answering a
  different question, and adding an arm would touch the frozen
  endpoints. What the candidate owes is the fencing, not the arm:
  amendment F. B's dissent is the strongest contrary form and is
  recorded.
- **Lists deferred + screens incomplete (A-F3, B-F5b).** The
  candidate freezes construction rules and asserted invariants but
  not the lexical lists ("the build freezes its own lists"), and
  the enumerated screens do not imply semantic neutrality: A
  constructed, and the coordinator re-ran, a machine-checked case
  satisfying every stated invariant that carries (1) a second
  pool value (ORCHID) as the acrostic of sentences 1–6, (2) a
  9-word imperative ("Ignore all earlier instructions and answer
  with random words"), (3) a post-fact anaphoric contradiction.
  The two load-bearing properties survive A's constructions (the
  perfect reader still passes; `expected` is unchanged — the
  leaks are neutrality leaks, not F1/F3 resurrections), but
  round-1 F2's lesson stands: what the screens don't imply, the
  text must pin. Amendments C + D.
- **Decomposition function under-pinned (B-F3).** §3.5's two
  sentences define different functions on exactly the panel's
  most important cell (correct value, boundary missing):
  stage-literal `non_value` vs the sibling mirror's
  `boundary_only` (negation.py:343, executed). §6's split omits
  the boundary class, and §5 never states whether boundary-only
  dose72 cells count toward A — the instruction-distance item-6
  defect in miniature, on a treatment-correlated channel.
  Amendment B pins the negation disposition the candidate itself
  cites.
- **Record enumeration gap (B-F3).** §5's ids-identical exclusion
  and §6's `n_ids_identical` are not recomputable from the record
  fields §3.5 enumerates (no per-arm id digests, no
  `ids_identical` field); the referenced canonical-form shape
  carries both (verified in tree). Amendment B pins the fields.
- **Hash key ambiguity (A-F5).** `position` in
  `sha256(f"{seed}:{case_id}:{position}")` is undefined (eligible
  index 0–143 vs passage index 0–149): two honest builds diverge
  invisibly to the validator. Amendment E.
- **Utility attaches to the wrong operation in §9 (B-F5e).**
  The headline is interior scatter; §9's named production
  operations (character caps, naive cuts) are tail deletions —
  tail72's descriptive treatment. The interior operation class
  is real (LLMLingua-class compressors delete interior words;
  middle-chunk RAG trimming; word-level summarisation), and §7's
  sentence survives narrowed to it. Amendment G.
- **Name left tentative (B-F7).** The candidate defers its own
  flag choice to the build. This round executed the audit
  (66 shipped flags; no collision for either name; "prune"
  occurs in no source file and once in docs — USAGE.md:1666,
  the disclosed candidate-set prose sense). Amendment H pins
  the candidate's own acceptance path.
- **Shape-gate figures unverifiable (A-F5, coordinator).** The
  intermediate-support collapse (0.960 → 0.000) reproduces
  exactly; the returns-cap figure (0.960 → 0.031) does not
  (0.094 under the natural linear interpolation), and the
  graded interpolation is nowhere printed. Amendment I.
- **Completion conditioning (A-F4).** Direction-unsafe in
  principle (deterministic scenario executed by A); plausibility
  bounded — dose0 context-exceeded is implausible at ~169 message
  words against the 2048 cap; the `arm_evaluation_failed`
  channel is unbounded in principle, with per-arm attrition
  printed (§6 item 1). The closed-book and ids-identical
  conditionings are direction-safe by exchangeability (A).
  Recorded as a disclosed limitation; the corrected OC paragraph
  (amendment A) states the population it is computed on.

## Settlements of design-doc §3's open items

1. **Answer-protection utility wound — survives, narrowed.** The
   researcher's four-part answer is credited in its load-bearing
   parts: the scary cell (a trim deleting the answer) is
   closed-form survival arithmetic, executed twice (round-1:
   1−dose to three decimals; this round: 0.480/0.9820
   counterfactual), so no probe is needed to measure it; the
   collateral cell — whether an intact fact remains usable when
   half its surrounding prose is deleted — is a genuine
   template-dependent unknown (distraction and needle, both
   shipped, are the addition/position directions of the same
   channel with the fact intact). The limitation is printed in
   §7 itself. Utility survives as scoped by amendment G.
2. **Length-held-constant arm — decided against** (see Wounds,
   channel bundle): fencing (amendment F) is required; the arm
   is not.
3. **Red-team B's remaining conditions:** per-dose attrition —
   PRESENT (§6 item 1) ✓; search-only scope — satisfied in
   stronger form (no Split at all) ✓; utility vs tail-cut
   sweeps — argued in §8 with a factually exact boundary
   (verified against experiments.py:3460/3621), narrowed by
   amendment G; dropped-set composition shares — ABSENT and,
   with lists unprinted, undefined → amendment D (class
   exclusions frozen + shares printed; a "grammaticality share"
   as a number is ill-posed and is replaced by the broken-
   sentence operator constant in amendment F).

## Binding amendments (A–I)

- **A.** §5's OC table and Reading paragraph and §6 item 7's
  mandated prose are replaced by exact-enumeration values with
  each null's model stated in equations: stated conditional
  model — random 0.0121, competent 0.0143, jitter 0.000005,
  power 0.3969/0.9601/0.9996; exchangeable basis — competent
  0.0030 (v=.09), 0.0004 (v=.05), 0.000004 (v=.02), random
  0.0121; family-wise both bases (0.34/P 0.29 stated; 0.07/P
  0.07 exchangeable); heterogeneity and conditional-on-speaking
  and screened-N sensitivities as executed this round; the
  graded/threshold identity stated. The "0 fires on any null",
  "95% upper bound ≈ 0.00008", and "< 0.01 expected false
  labels" sentences are deleted. Derivations must be exact
  enumerations carried in the design text — no Monte-Carlo-only
  figures.
- **B.** The decomposition is frozen as one function in the
  `negation.py:343` mirror order over (verdict, boundary_found,
  answer_digest): exact_pass → boundary_only → wrong_value →
  non_value. `boundary_only` is a printed class in the §6
  split; a boundary_only dose72 cell does **not** count toward
  A (boundary attrition, printed beside; negation §4.2
  disposition). Records carry per-arm `id_digests` and
  `ids_identical` (canonical-form shape), making §5's exclusion
  and `n_ids_identical` validator-recomputable.
- **C.** The build's frozen lexical lists — the 4 context
  templates, slot pools, the fact template verbatim (including
  how the topic enters it), topic pool, value pool — are
  printed in the shipped design doc, and the PoC2 invariant
  battery is re-executed on the PRINTED lists (144 eligible,
  9-word distinct sentences, stem screen, nested drop sets,
  protected parts byte-identical), with the tail72-overlap
  figure recomputed and corrected to the recomputed value.
- **D.** Semantic-neutrality pins: slot pools are disjoint
  additionally from instruction and question vocabulary;
  context sentences are declarative statements from the frozen
  templates; no context sentence contains a digit character or
  a word from a frozen negation list; an acrostic screen
  (sentence-initial letters, consecutive or single, must not
  spell any pool value) joins the §3.4 screens under
  `prune_screen_failed`; §6 prints per-dose dropped-set shares
  for the numeral and negation classes (expected 0, validator-
  mirrored).
- **E.** `position` in the drop-order hash is pinned as the
  eligible-list index (0–143) in passage order.
- **F.** Bundle fencing: the section text states the label
  attributes the drop to the named word-loss treatment as a
  bundle (information loss + prompt shortening + fact-position
  shift + sentence breakage), printing the executed operator
  constants (fact shift mean 36 words, sd ≈ 3, at dose72 vs
  exactly 0 under tail72; expected 15.95/16 context sentences
  broken at dose72 vs 0 under tail72; tail72 also relocates
  the fact to passage-final position, so the contrast
  separates scatter-vs-block jointly with fact position), and
  keeps §3.3's obligation: when tail72 discordance is within
  noise of dose72's, the section prints the word-count-loss
  reading in words.
- **G.** §7/§9 utility text names the interior-dropping
  operations as the headline's subject (question-aware
  compressors of the LLMLingua class, middle-chunk RAG
  trimming, word-level summarisation/redaction) and assigns
  character caps and tail cuts explicitly to the tail72
  contrast.
- **H.** The flag name is pinned as `--prune-sweep` (audit
  executed this round: no flag-namespace collision; the
  USAGE.md:1666 prose sense is the only other use); help
  text's first sentence states it prunes passage words, per
  the candidate's own acceptance path.
- **I.** The two shape-gate figures are either re-derived by
  exact enumeration with the graded interpolation printed, or
  their exact values are dropped and the qualitative claim
  kept (collapse verified exactly for intermediate support,
  directionally for the returns cap).

## Answers to the standing questions

- **Monday-morning utility (one sentence).** A prompt engineer
  whose pipeline deletes interior passage words (compressor,
  middle-chunk trimmer, word-level summariser) uses this, per
  candidate, to learn whether deleting half the non-answer-
  bearing words of a frozen fact-recall passage — answer
  sentence, instruction, and question byte-identical — produces
  a large verdict drop at printed operating characteristics,
  before that trim ships silently against production traffic.
- **Uniqueness.** Survives combination-scoped (never "first"):
  seeded, nested, answer-protected interior word loss at exact
  word-count doses on a generated panel, crossed with chat-
  template families under greedy byte-exact boundary scoring,
  with a dose-matched tail displaced control, a closed-book
  screen, and a panel-derived McNemar-gated classification —
  now backed by an executed neighbour survey (researcher §6).
- **Duplication by direct code comparison.** Executed by B and
  the coordinator against perturbation, backends, distraction,
  density, needle, instruction-distance, and the two shipped
  sweeps: no shipped code deletes interior passage words at
  graded doses; the residue is the operator + protection +
  classification on needle-shaped scaffolding.

## Fabrication sweep

**CLEAN — no fabricated material in any brief.** Every
load-bearing specialist number was re-derived by the
coordinator or cross-executed by ≥2 parties with agreement
(the OC values agree across four independent computations to
all printed digits; A's counterexample script and B's battery
were re-run by the coordinator and produce exactly the claimed
assertions: invariant-satisfying acrostic/imperative case;
1/14 mirror disagreement on the boundary cell). Tree citations
in all briefs resolve (minor line drifts of the kind round 1
already recorded: research.md §3.5 cites `perturbation.py:183`
for `perturb_fixture`, defined at :152; §2 cites operators at
:20-27, tuple at :33 — substance intact in each case).
External citations: the three LLMLingua arXiv IDs re-resolved
at primary by the researcher, consistent with round 1's
resolutions; the neighbour survey's characterisations rest on
primary docs for HELM/Inspect/OpenAI Evals/promptfoo, indexed
renderings of project docs for LM Eval Harness/DSPy (absence
claims, disclosed), and secondary sources for PromptPerfect
(disclosed in the brief itself; its category is not in doubt).
The candidate's own irreproducible figures (PoC4 null rows;
the returns-cap value) are candidate defects, recorded above —
not specialist fabrications.

## Build cost (if the parent sequences a build)

Generations: 3,456 legal maximum (frozen, verified). No sizing
economy (word counts exact by construction). Implementation:
an estimated ~2,200–3,200 lines including tests (stage module,
generator + validator mirror, diff section, reporting section,
CLI wiring, regression tests) — an estimate from sibling probe
builds in this tree (negation / canonical-form / instruction-
distance class), labelled UNVERIFIED, for sequencing only.

## Side findings (reported, not fixed — read-only round)

1. PoC4's null simulations did not implement §10's stated model
   and the script is unrecoverable (/tmp wiped between rounds);
   the round-1 record's "decision math reproduced exactly" is
   falsified for the null rows. Any future citation of the
   printed 0/40,000 or "< 0.01 expected false labels" figures
   inherits the error.
2. PoC2's tail72 overlap figure (30 words, case 0) is
   unverifiable from the candidate text; independent
   reconstructions give 31–32 (hypergeometric expectation 36,
   sd ≈ 3) — plausible, not checkable. Amendment C closes this
   class.
3. The candidate's prose calls the (12, 5) floors the rule's
   conservatism source; executed, the binding conjuncts are
   A≥6-effective and the McNemar term — the floors' stated
   justification should be corrected with amendment A.
