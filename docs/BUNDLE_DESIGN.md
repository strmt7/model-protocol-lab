# Question bundling — design round (ADJUDICATED: NO-BUILD)

## Round summary (coordinator #46, design round only — nothing built)

Wave-11 slate rank 5, the last wave-11 item: question bundling
(tentatively `run --bundle`). The candidate sketch
(`~/workspace/agents_out/wave11/slate_candidates.md` Candidate 5):
generated cases of K=3 independent questions (fact recall + small
arithmetic, disjoint answer spaces); a bundled arm (all K in one
user turn, frozen convention "reply with one line per question,
`1: <answer>`", one generation, per-question verdicts parsed from
the response) against a split arm (three scripted turn pairs with
frozen assistant replies, the scored question rotated through all
three positions); falsified by any (position, candidate)
discordant pair; worst case 20 cases × (1 + 3) arms × 24
candidates = 1,920 generations (arithmetic verified correct by
all lanes).

- Tree at main `a6705d1b9607` (cycle 213 closed green, 8,442
  tests, precise coverage 95.3455%). This round was READ-ONLY
  except for this doc and the briefs; no suite was run, no push
  was made, no tag was cut or discussed.
- Specialists (blind to each other): researcher
  (`~/workspace/agents_out/bundle/research.md`), red-team A
  (`redteam_a.md`), red-team B (`redteam_b.md`), verifier/claims
  auditor (`verifier.md`). All four briefs landed; the verifier
  resolved every externally checkable claim against primary
  sources, re-ran every surviving PoC, and recomputed all
  decision math with independent code; the coordinator
  re-executed the load-bearing claims itself (§4).
  **Fabrication sweep: clean — a sixth consecutive round.** No
  invented symbol, no fabricated citation, and no unreproduced
  load-bearing number in any brief; the failures found were two
  non-load-bearing line drifts in red-team B (≤ a few lines,
  substance intact), handled in §5.
- Convergence: **unanimous NO-BUILD** — researcher, both blind
  red teams, and the verifier each reached it independently for
  the sketch as specified, fatal by fatal, by execution. The two
  red teams, working blind, converged on the same core defects
  (scorer inversion, unfrozen parser, cap censoring ordered by
  slot, the noise-firing falsifier) with independently executed
  numbers that agree. The researcher additionally specifies a
  complete redesigned instrument (`run --multi-question`,
  §3) and recommends it as a resubmission candidate; the
  verifier reproduced its decision math exactly and ruled it a
  **resubmission, not an amendment**, under this program's own
  test, and the researcher himself classifies it the same way.
  The coordinator concurs: the verdict is **NO-BUILD** — for
  the sketch, and as a build order for the redesigned instrument
  in this round. §3 records the resubmission specification in
  summary, including the seven points the fresh review must
  settle head-on.

## 1. Why NO-BUILD: the fatal defects of the candidate as specified

Each defect below is individually blocking and was established
by execution (attribution: R = researcher, A = red-team A,
B = red-team B, V = verifier, C = coordinator).

### F1 — The scorer cannot emit the headline: one byte-exact span per fixture, and the arms' reply contracts are mutually unsatisfiable

`core.evaluate` emits exactly one verdict per fixture:
`core.py:797` `exact_match = status == "completed" and answer ==
case.expected`, where `answer` is the whole decoded pre-boundary
span (`core.py:939` `answer_ids = output_ids if cut is None else
output_ids[:cut]`; `core.py:966` `answer = decoded`). There is
no per-question verdict anywhere in the tree's scoring path.
Executed through the real `evaluate` with stub backends (R poc1,
A poc1, B poc1, V re-ran; C re-ran, §4): a perfectly correct,
fully compliant bundled reply (`1: Paris\n2: 42\n3: blue`)
scores **False** against every per-question expected string and
against newline-joined bare answers; it scores True only
against the whole bundled string as `expected` — which collapses
the headline to one all-or-nothing verdict in which a
2-of-3-correct bundle and an all-wrong bundle are the same
observation, an omission, an answer swap, and all-wrong are
indistinguishable, and one trailing "." fails a fully correct
bundle. Against the whole-string expected, 8 of 9 semantically
identical formatting variants (no labels, `1.` separators, a
preamble, reordered labelled lines, a trailing period) fail —
the verdict is formatting luck. And the inversion is verbatim
DIRECTIVE F1: a disobedient model that outputs only the first
answer (`Paris`) **passes** the Q1 fixture where the compliant
bundled model fails. The split arm has the same trap in mirror
image: under the shared convention its compliant reply
`1: amber` fails a bare-answer expected and passes a labelled
one, while its bare reply does the reverse — whichever expected
string is chosen, one of two compliant behaviours fails. The
sketch never chooses, because it cannot: the two arms demand
different reply contracts under one scorer.

### F2 — The line grammar is unfrozen, honest parsers disagree on the same responses, and parse failure has no honest home

The sketch's per-question readout rests entirely on a "frozen
line grammar" that is never frozen: it states no label syntax,
no strip policy, no duplicate/missing-line rule, no rule for
answers containing `:`. Executed (A poc2, B poc3, V re-ran): two
honest parsers — positional vs by-label — return different
verdict vectors on the same responses (reordered lines:
[F,F,T] vs [T,T,T]; a preamble: [F,F,F] vs [T,T,T]). A charitable
strict line parser passes only 2 of 10 semantically perfect
bundled variants (prose preface or trailing prose, bold labels,
blank lines, glued lines, unnumbered lines, `1)` labels,
0-based labels all lose slots wholesale); one missing line
loses every slot under positional parsing, and first-wins vs
last-wins duplicate handling disagrees on self-correcting
responses. The sketch specifies no parse-failure outcome code:
folding parse failures into "wrong" makes the headline a
formatting measure, and excluding them is treatment-selected
attrition (the bundled arm produces the parse failures) —
answer-syntax F2 through the parser door. The tree's precedent
(structured's parse-vs-schema split, `structured.py:68`
OUTCOMES) exists precisely because this pooling is dishonest,
and the sketch adopts none of it.

### F3 — Digest-only records cannot carry per-slot verdicts (instruction-distance F2 verbatim)

Every sibling generated probe's records carry digests only
(`negation.py:57`: "Records carry digests only"; the verdict
identity is `verdict == (answer_digest == sha256(expected)) and
boundary_found`, recomputed by the diff validator). Executed
(A poc, B, C §4): ten distinct bundled texts — all-correct,
Q1-wrong, Q2-wrong, a swap, an omission, a parse failure —
collapse to the same record class under a whole-answer digest
(the digest differs, but nothing in it says *which* slot
failed, or whether anything parsed). Per-slot verdicts,
swap flags, and parse decompositions are underivable from
the records the lab keeps, and therefore unrecomputable by
the diff validator. The sketch specifies none of the three
tree escapes: a numform-style digest-bound answer text
(the one sanctioned exception, cycle 190), density-style
carried primitives (e.g. an ordered word-hash list), or
stage-attested booleans (the forgery surface the digest
discipline exists to remove). A probe whose headline cannot
be recomputed from its own records is not buildable in this
tree.

### F4 — There is no decision rule: the any-discordant-pair falsifier fires on noise, and the panel's cells are clustered

The sketch's falsifier is "any (position, candidate) discordant
pair" over 20 × 3 × 24 = 1,440 pairs, with no floor, no
direction, no test. Executed (R, A, B, V; C independently,
§4): P(any discordance in the panel) = **0.9439 at per-arm
jitter ε = 0.001** and **1.0000 at ε ≥ 0.005** (analytic,
Monte-Carlo confirmed by three lanes). The one apparent number
disagreement of the round — the researcher's 0.763 at
"ε = 0.001" — is a parametrisation difference, settled by the
verifier and re-derived by the coordinator: 0.763 is
P(any) when 0.001 is the *per-pair* discordance probability
(1 − 0.999^1440 = 0.7632 exactly); 0.9439 is the same quantity
when 0.001 is the *per-arm* flip probability. Both are correct
under their stated nulls; the fatal is unaffected (the rule
fires with probability 1.000 by ε = 0.005 under both). A
random labeller reaches ≥6 discordant of 60 per candidate
with probability 1.000000, so no borrowed floor rescues it
either. Worse, the panel's bundled cells are **clustered by
construction**: the three bundled slot verdicts share one
generation, so one boundary miss, truncation, or parse event
flips all three together (executed, A: a boundary miss fails
all 3 bundled slots vs 1 split slot) — any pair-counting repair
that treats the 1,440 pairs as independent is mis-calibrated
on this instrument specifically.

### F5 — Cap censoring is ordered by slot, and the budget is asymmetric by construction

The run's default cap is 64 new tokens (`cli.py:198`). The
bundled answer is K lines; the split answer is one. Executed
(A, B, R poc3; C §4): a perfect model whose bundled reply
needs 70 tokens records `completed`, truncated mid-slot-3,
boundary missed, retained as an ordinary failure — while all
three split replies pass. Survival by slot at caps 50/64 is
[T,T,F]: the cap alone manufactures the sketch's advertised
headline, "position 3 degrades first", on a model with no
bundling deficit at all. The asymmetry is constitutive, not
a tuning detail: the split arm receives K × 64 tokens of
budget across its generations against the bundled arm's 64.
Raising the bundled cap breaks comparability with the split
arm; keeping it censors the readout's own axis. The sketch
has no liveness rule (a truncated cell removing its whole
case, attrition printed) of the kind density's `_live`
discipline provides.

### F6 — Neither arm is what the claim names: the contrast bundles at least six variables, and "position" is two different variables in the two arms

Executed renders (A, B; chatml bundled 205 chars / 1 message
vs split-Q3 354 chars / 5 messages): the split arm's unscored
questions are never answered by the model at all — its history
is frozen assistant text the sketch never specifies. Both
readings of that text are already owned or confounded: answers
as history = fewshot demonstrations; `Noted.` = multiturn's
scripted arm verbatim (`multiturn.py:85` `SCRIPTED_REPLY`).
The production alternative the utility names — asking
separately, i.e. independent single-turn calls — **does not
exist in the sketch**. What the contrast actually varies,
simultaneously: turn count, scripted history presence and
content, convention load (one instruction governing K lines
vs one line, repeated), prompt length (whose sign *reverses
by position*: the bundled prompt is longer than split-Q1's
and shorter than split-Q3's), generation-cap exposure (F5),
scoring machinery (parser vs byte-exact, F1/F2), and
autoregressive exposure to the model's own earlier answers
(the contamination the sketch wants to measure is present
only in one arm — it is part of the treatment bundle, not
separable from it). No separating control can even be stated:
a length-matched control would need a different filler dose
per position, and length-padding is itself the shipped
distraction treatment. A discordance from this contrast is
attributable to "bundling" only as a named bundle of all of
the above — and the sketch's claim is written as a
single-cause packaging claim. Separately, "position" pools
two variables: in the bundled arm a slot is a fixed question
identity/type (no within-bundle permutation is specified, so
slot ≡ item), while in the split arm position is history
depth by renderer construction (scored positions 1/2/3 =
1/3/5 messages). A by-position readout compares list position
against turn depth and calls both "position".

### F7 — The disjointness the swap readout needs fails by chance, and when it fails the failure is invisible

The sketch's contamination mitigation is disjoint answer
spaces, "planted so a swapped answer is detectable as a swap".
Executed (B; C spot-checked the arithmetic, §4): two small
arithmetic answers in one case coincide with probability
≈ 0.07, and at least one collision among three with
probability ≈ 0.19–0.21 — under the natural operand
distributions, coincidence is a routine event, not an edge
case. When two expected answers are equal, a swap is
byte-identical to a perfect response: contamination becomes
invisible in exactly the cells where it occurred. The sketch
specifies no generator screen (contrast multiturn's built-in
redraw discipline), and adding one post hoc selects the
population (cases whose answers happen to collide are
systematically easier/harder — the screen is a population
choice, not a hygiene step). Without enforced-by-construction
distinctness, the swap decomposition the sketch promises
cannot be delivered; with it, the population is no longer
the sketch's.

### F8 — The selection headline is manufactured by conjunction, and forbidden on this population besides

The utility promises whether "the template they selected
survives bundling". Executed arithmetic (A, B; C re-derived,
§4): if a case passes bundled only when all 3 slots pass, then
at *identical* per-question accuracy p in both arms the
case-level gap is p − p³ = **0.171 at p = 0.9, 0.288 at
p = 0.8, 0.357 at p = 0.7** — a large "bundling effect"
manufactured by the conjunction alone, with zero treatment
effect. If instead no case-level aggregation is defined, there
is no selection readout at all: the tree's selection rule
consumes per-fixture verdicts, and a bundled fixture has three.
And the deeper bar is already settled in-tree: the numform
settlement forbids a winner/selection readout on a generated
population (`experiments.py:11049` pattern; negation v1
adopted the same bar) — the sketch's population is generated,
so its headline readout is one the program has already ruled
out for exactly this population type.

### F9 — The population cannot deliver the utility: one K, the wrong K, and "recall" that is not generatable

The utility sentence asks for "the bundle size at which
per-question verdicts start moving" for a front-end bundling
**five** questions. The instrument has exactly one K, and it
is 3. One point is not a size readout: no curve, no threshold,
no "at which" (prune-sweep F5's permutation/shape point in
miniature). The population is also undefined where it
matters: fact-recall questions with known answers are not
seed-generatable and validator-regenerable the way this
tree's generated probes require — a fact the model is asked
to recall must come from a frozen external set (a different
instrument, with provenance obligations) or be planted in
the prompt, in which case the task is extraction, not
recall, and the claim must be re-scoped to say so. The sketch
says neither. Finally, with no family × slot rotation, slot
and question family are perfectly collinear (F6), so even
the descriptive by-position readout the sketch falls back on
pools item difficulty into "position".

### F10 — The name is taken: `bundle` is the sealed-bundle term of art (blocking rename, independently of F1–F9)

Executed naming audit (R §5, A, B, V): `pack.py:1` "Sealed
result bundles (`model-protocol-lab pack` /
`verify --bundle`)", format id `mpl-bundle/1`
(`pack.py:49`), a live `verify --bundle <path>` flag
(`verify.py:645-675`, dispatch to `pack.run_verify_bundle`),
and USAGE §13.1 (`docs/USAGE.md:3320`, "Sealing a result
(`pack`, `verify --bundle`)"). "Bundle" in this tree already
names one thing, in adjacent subcommands, with a file format
attached. A `run --bundle` flag is free only as an argparse
string; under the tree's USAGE naming discipline (one word,
one meaning, the same terms in USAGE/FAQ/CLI help — repo
AGENTS.md) it fails the audit outright. Any future instrument
in this space ships under a different name (the resubmission
proposes `run --multi-question`, §3).

### Non-fatal, adjudicated

- **Generation-count arithmetic**: the sketch's 1,920 worst
  case is correct (all lanes).
- **Duplication, precisely**: no arm is byte-identical to a
  shipped probe's arm (checked by render comparison, not
  asserted). The boundary is exact per neighbour — placement
  owns question serial position within one turn; multiturn
  owns the full-vs-sharded-vs-scripted turn-packaging contrast
  (the sketch's split arm *is* multiturn's `scripted`
  construction); density owns simultaneous-demand load and
  per-serial-position fall under a byte-exact single-task
  scorer; needle owns fact position in filler; structured owns
  multi-part response parsing on user fixtures. The only
  element no shipped probe owns — K per-question verdicts from
  one generation — is exactly the element F1–F3 show the tree
  cannot currently score or record. The candidate is not a
  duplicate; it is a request for the one readout the record
  discipline forbids as specified.
- **External citation**: Cheng, Kasai & Yu, batch prompting,
  arXiv:2301.08721 — verified at the primary source by the
  verifier (the sketch's citation is accurate, and fenced: API
  models, plain/few-shot formats, per-sample accuracy, no
  template-family crossing). It supports the phenomenon's
  existence; it does not support this instrument.

## 2. Amendment or resubmission

Under the program's standing test (negation = amendment: same
population, arms, scorer, and rule shape, all already frozen
and executable; payload / answer-syntax / directive /
instruction-language / instruction-distance / history-frame /
prune-sweep = resubmission when those change), the researcher's
redesign changes **every** load-bearing element: population,
arms, scorer, decision rule, headline, and name (§3). The
sketch was never executable as a measurement — its split arm
has no specified replies (F6), its scorer cannot emit its
readout (F1), its records cannot carry its readout (F3), and
its falsifier is a noise printer (F4). This is the payload
configuration, not the negation one. The redesigned instrument
is recorded in §3 as a **resubmission candidate**. It is not a
build order; no build cycle may act on it without a fresh
blind design review.

## 3. The resubmission candidate (summary; full specification in the researcher brief §3)

`run --multi-question` (module `multiquestion.py`), specified
in full in `~/workspace/agents_out/bundle/research.md` §3:

- **Population**: generated, stage-local panels at **K=3 and
  K=5, 24 cases each**; sub-questions are planted-fact
  extraction (placement/density construction, named
  extraction, never "recall") and small arithmetic, all with
  single-normalized-word expected answers; family × slot
  rotation; generator screens enforce pairwise-distinct
  expected answers within a case (redraw discipline), so swap
  detection is a construction property.
- **Arms**: bundled (one turn, K numbered questions) vs
  split_j (one turn, only question j, **renumbered to 1**,
  no history anywhere) — the production "asking separately"
  the utility names. Length difference disclosed as
  constitutive of packaging, printed per case, never gated;
  the claim is scoped to the packaging bundle, not a single
  cause.
- **Scorer**: one total, marker-aligned parser over density's
  carried ordered word-hash list, symmetric across arms;
  per-slot outcomes `pass` / `wrong_value` (with swap flag) /
  `parse_failed` (decomposed `missing_marker` /
  `extra_words`); no answer text carried — the digest-compatible
  design is density's, so the numform exception is not invoked.
  Composite byte-exact printed as sensitivity only. Liveness:
  a case enters the headline only if the bundled cell and all
  split cells completed with boundary found; attrition printed.
- **Decision rule** (derived, never borrowed): per candidate
  per panel, case-aggregated discordance (slots within a case
  share one generation — clustering handled by construction);
  D ≥ 6 discordant cases, F > P, and exact sign test
  p ≤ 0.05 → `multiquestion_drop_observed`; D < 6 →
  `insufficient`; else `below_threshold`. Researcher's 20k-trial
  Monte Carlo (verifier reproduced the full table exactly):
  fire 0.017 random / 0.012 competent-null / ≤ 0.014 clustered
  at K=3; power 0.979 at a 25% drop (K=3), 0.999 (K=5).
  Family-wise expectation printed: 48 decisions, ≈ 0.81 false
  labels expected under the random null, P(≥1) ≈ 0.56.
- **Counts**: 240 generations per candidate; legal maximum
  **5,760**; 1,152 ground records; house 12 MiB cap.
- **Vocabulary**: frozen house pattern only — decisions
  `{insufficient, below_threshold, multiquestion_drop_observed}`,
  whole-stage fail-closed `multiquestion_unavailable`; no
  invented verdict words (verifier checked the set against the
  negation/density pattern).
- **No selection readout** (numform settlement honoured).

The fresh review must settle head-on (silence = NO-BUILD):
(a) split renumbering (to-1 vs keep-slot-number — different
treatments); (b) the decision-rule operating characteristics
re-run under a case-clustered drop model and at real
completed-case counts after attrition; (c) parse-failure in
the headline (counted as not-pass with decomposition printed,
vs conditioning on parse-ok cells — the latter is
outcome-dependent conditioning and needs its own rule);
(d) K=5 cost (5,760 legal-maximum generations) vs a K=3-only
probe with honestly narrowed utility; (e) word-rule leniency
(case-folding, punctuation-insensitive) as the headline
scorer, with byte-exact demoted to sensitivity — or the
reverse, with the rule re-derived; (f) extraction-not-recall
scoping (a genuine-recall population is a different
instrument and may not be smuggled in as an arm); and
(g) the verifier's addition: arithmetic sums landing in
{1..5} are not screened from the parser's marker words in
§3.1 as written and would break the §3.3 grammar — the
screen set must be closed before any build.

## 4. Coordinator re-execution (independent, this session)

- **Scorer (F1)**: stub backend through the real
  `core.evaluate` — a fully compliant bundled reply scores
  False against per-question expected `amber` and against
  `1: amber`, True only against the whole three-line string;
  a split reply `1: amber` scores False against expected
  `amber` and True against `1: amber`, with the bare reply
  exactly reversed. Both arms cannot be satisfied under one
  expected-string choice.
- **Records (F3)**: whole-answer digests of all-correct /
  Q1-wrong / Q2-wrong bundles are three distinct hashes with
  no slot information recoverable from any of them.
- **Falsifier (F4)**: both parametrisations reproduced
  exactly — 1 − 0.999^1440 = 0.7632 (per-pair reading,
  researcher's number) and 1 − (1 − 2·0.001·0.999)^1440 =
  0.9439 (per-arm reading, red teams' number); at ε = 0.005
  per-arm, P(any) = 1.0000.
- **Conjunction (F8)**: p − p³ re-derived — 0.171 / 0.288 /
  0.357 at p = 0.9 / 0.8 / 0.7, matching red-team B exactly.
- **Disjointness (F7)**: under a triangular two-operand sum
  distribution the coordinator obtains P(pair coincidence)
  ≈ 0.075 and P(any of three) ≈ 0.21 — same magnitude as
  red-team B's 0.068 / 0.191 under its operand range; the
  substance (coincidence is routine) is confirmed, the small
  difference is the operand-range choice, not an error.
- **Cap (F5)**: a three-line bundle of moderate answers
  exceeds the 64-token default in character proxy (71 chars
  for three 20-char answers plus labels) where each split
  line is 23 — the censoring gradient is on the readout's own
  axis, as the red teams' token-level executions show.

## 5. Errors found in the briefs (none load-bearing)

- Red-team B: two line drifts (`structured.py` OUTCOMES at
  :68, cited ~:56; the `exact_match` record site at
  `core.py:797`, cited ~:815). Substance verified at the
  correct lines by the verifier.
- The researcher's F3 number (0.763) and the red teams'
  (0.944) initially read as a disagreement; both are correct
  under their stated null parametrisations (§1 F4). No brief
  is wrong; the design doc prints both with their nulls.
- No other errors: the verifier's mechanical sweep resolved
  every remaining symbol, line, count, and external citation
  in all three briefs (catalog = 29 families executed;
  grep counts exact).

## 6. Verdict

**NO-BUILD** — unanimous (researcher, red-team A, red-team B,
verifier; coordinator concurs on its own re-execution). The
sketch's headline readout cannot be emitted by the tree's
scorer (F1), cannot be parsed by the grammar it never froze
(F2), cannot be carried by the records the lab keeps (F3),
cannot be decided by the falsifier it specifies (F4), is
manufactured by the generation cap on a perfect model (F5),
compares two arms neither of which is the thing the claim
names (F6), promises swap detection its population defeats by
chance (F7), promises a selection readout that conjunction
arithmetic fabricates and the numform settlement forbids
(F8), at a bundle size its own utility does not ask about
(F9), under a name the tree already owns (F10). The
redesigned `run --multi-question` instrument in §3 is a
resubmission candidate for a future fresh blind design
review — not a build order, and not queued ahead of any
existing item. With this round, the wave-11 slate is fully
adjudicated: ranks 1–5 all NO-BUILD, each with a resubmission
candidate recorded.
