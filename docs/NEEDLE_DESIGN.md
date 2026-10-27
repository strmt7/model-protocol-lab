# Needle-depth probe — design

Cycle 191 (wave-8 rank 3). A practitioner pasting retrieved
chunks around a question learns, per template family, at which
depth of the surrounding bulk their checkpoint stops finding a
planted fact — a per-template recall-vs-depth curve at exact
rendered-token depths — instead of a single "long context
good/bad" number measured through a vendor's own template.

The probe plants one fact sentence carrying a unique synthetic
value inside neutral filler, at five depths of the filler span
and two filler doses, and asks for the value. The independent
variable is the chat template: same checkpoint, same generated
cases, same filler — only the hand-rendered format changes.
The stage lives in `experiments.needle_stage`, the frozen
protocol and arithmetic in `model_protocol_lab/needle.py`, the
HTML section in `reporting.py`, and the full-recompute diff
validator in `diffreports.py`.

Design history: v1 of this document specified contiguous
needle-run acceptance over an unstated textual assembly.
Red-team A returned DO NOT BUILD AS SPECIFIED (run-existence is
fatal on the literal reading; the validator omitted the
verdict↔digest binding); red-team B upheld the verdict,
downgraded A's impossibility claim with a counter-assembly,
refuted the coordinator's first repair sketch (a marker-derived
head anchor with a systematic −1 bias), and supplied the
surviving measurement (the prefix-anchored difference `D`).
This v2 is the adjudicated contract: delta-based dose, `D`-
anchored offsets, digest-bound verdicts, one headline
population. The briefs are in `~/workspace/agents_out/wave8/`
(`cycle191_redteam_a.md`, `cycle191_redteam_b.md`).

## Where this sits against prior work

The depth phenomenon is published and is not claimed here:
Liu et al. (*Lost in the Middle*, arXiv:2307.03172) measured a
U-shaped recall curve over document position in API-served
models, with mid-context accuracy sometimes dropping below the
closed-book bound; NoLiMa (arXiv:2502.05167) extends
needle-in-a-haystack with minimal-lexical-overlap needles
and a proportion-based placement sweep over the haystack
(its §4.4.2; an earlier draft of this paragraph also credited
it with unique-token counting and needles inserted at
25/50/75% — neither is in the paper, corrected in cycle 216);
RULER
(arXiv:2404.06654) runs synthetic local-model tasks at
flexible length and complexity configurations, scored by
recall-based presence accuracy (its paper states depth-level
position control is not yet reported); BABILong (arXiv:2406.10149)
distributes multiple facts through book text; Chroma's
*Context Rot* report (July 2025,
https://research.trychroma.com/context-rot) varies total length
with a fixed needle and shows distractor similarity and
needle–question similarity move scores materially. Depth as an
axis, token-exact placement, literal needles, and distributed
facts all exist. What no surveyed harness reports is the
combination this probe ships: a recall-vs-depth curve **as a
function of hand-rendered chat-template family** on one local
checkpoint, with the depth offset verified on each candidate's
own rendered token sequence, a needle-absent false-recall floor
and a duplicate-needle control as arms of the same probe, and
the lab's conservative selection rule (unique winner or
abstain) re-applied per depth cell. The uniqueness pass behind
this paragraph, with per-harness coverage, is logged in
docs/RESEARCH_LOG.md (cycle 191 entry).

The boundary against the shipped distractor-dilution probe
(cycle 184) is one sentence: distract varies *how much* filler
surrounds the fixture's own question and whether a planted
wrong answer captures the model; needle varies *where* an
answer-bearing fact sits inside a fixed amount of filler.
Distract's conditions are dose × placement of filler around
existing fixtures; needle's are dose × depth of the fact
itself, over generated cases whose only question is the plant.

## Cases

Cases are generated inside the stage from the run seed (the
multiturn pattern: self-contained, no dependence on the user's
fixtures or the recorded search). Twenty cases, ids
`needle-00` … `needle-19`. The derivation is frozen exactly,
because the diff validator's mirror must reproduce it bit for
bit: one draw is
`int.from_bytes(sha256(f"{seed}|needle|{index}|{slot}|{counter}"
.encode("ascii")).digest()[:8], "big")` with `counter` starting
at 0. Slot 0 gives the code word `WORDS[draw % 64]`; slot 1
gives the number `1000 + draw % 9000`. The value is
`"{word}-{number}"`. If a case's value equals an earlier case's
value, `counter` increments and both slots redraw. Topic *i* is
`TOPICS[i]` from the frozen list of twenty in `needle.py`
("astronomy society", "beekeeping circle", …). The fact
sentence is one frozen template rendered with the corpus's own
trailing-space convention: `The {topic} code is {value}. `
The question is `What is the {topic} code?`; the expected
answer is the value, scored by the standard pipeline
(boundary-cut answer, exact match — `core.evaluate`'s literal
equality). Question and needle share the topic phrase and the
word "code" by construction — needle–question lexical match is
deliberate (Chroma's finding is that it moves scores; an
unmatched question would measure vocabulary guessing, not
position). The protocol carries `case_material_sha256`, the
SHA-256 of the canonical JSON of (fact template, topics,
words), so the frozen lists are bound by digest, and
`ground_sha256` over the canonical generated rows (multiturn's
pattern), so the seed's case set is bound as a whole.

Generation-time screens (a failure is a protocol bug, raised
before any generation, never a per-cell outcome): topics and
words are screened against the filler corpus (case-folded
substring) — the frozen lists pass and a test pins that; the
twenty values are pairwise distinct by the redraw rule; no
value occurs in the corpus. The corpus contains no digits at
all, so the numeric half of every value is unplanted by
construction everywhere except the needle.

## Arms and assembly

Fifteen cells per case per candidate. The assembly is frozen
exactly (v1's silence here was build-blocking): a case is a
single user message whose content is segment concatenation
with **no joiner anywhere** — filler and needle sentences
carry the corpus trailing space, and the question follows the
bulk directly as the final sentence.

- `dose_0` — content = needle sentence + question. The control
  arm: it proves the case is answerable when nothing surrounds
  the fact. (The wave-8 brief called the recorded native run
  the dose-0 control; the recorded search runs the user's
  fixtures with their own questions and contains no observation
  of these generated cases, so the stage-generated arm plays
  the control role — the multiturn honesty settlement, stated
  here rather than discovered later.)
- `dose_{1024,4096}_depth_{0,25,50,75,100}` — content =
  prefix filler + needle + suffix filler + question.
- `absent_{1024,4096}` — content = filler + question, no
  needle. Any pass is fabrication or guessing; the pass rate
  is the false-recall floor, reported per dose beside that
  dose's curve.
- `duplicate_{1024,4096}` — content = prefix filler + needle +
  middle filler + needle + suffix filler + question, the two
  needles at 25% and 75%. Separates position from exposure
  count, with the independence baseline of the Claims section
  keeping the readout honest.

Dose scoping: the brief's doses {1024, 4096} ship as written.
The 4096 dose needs `--max-context` above 4096 plus the
generation budget and prompt overhead; under the default 2048
cap every 4096 cell is `arm_context_exceeded` — reported per
cell, never truncated into existence. The full control set
(floor and duplicate) is live at 1024 under defaults, so the
default run still carries the falsification framing; the brief's
{1024, 2048} fallback was not taken because a 2048 filler dose
also exceeds the default cap once the needle, question, and
generation budget are added, so it buys no default-liveness
and halves the dose axis. The CLI help states the arithmetic
before the run.

## Measurement

All quantities are measured on rendered token ids under three
cached renders per (candidate, case): `Q` (question-only
content), `Z` (the dose-0 content), and the arm render itself.
`needle_len := len(Z) − len(Q)` is the needle's rendered length
in the dose-0 context for that candidate and case; a case with
`needle_len <= 0` for a candidate is degenerate and all its
cells for that candidate are `arm_unmatchable`.

- **Dose is a rendered delta** (distraction's semantics,
  stated plainly): a treated cell is accepted only when
  `len(arm) − len(Z) == dose` on the final render — the arm
  prompt is exactly `dose` rendered tokens longer than the
  same case's dose-0 prompt. The absent cell is accepted only
  when `len(arm) − len(Q) == dose` — the same equation with
  the needle term removed, not a separate statistic. The
  duplicate cell: `len(arm) − len(Z) == dose + needle_len`.
  Seam merges at the needle's edges are absorbed into the
  delta and compensated by the sizing search; the design does
  not claim the filler *text* totals the dose standalone.
- **Offset is the prefix-anchored difference `D`.**
  `D(arm, comp) := LCP(arm_ids, comp_ids) − LCP(Z_ids, Q_ids)`,
  where `comp` is the render of the arm's text up to just
  before the measured needle, with the question appended
  (needle 1: prefix + question; needle 2 in duplicate:
  prefix + needle + middle + question). The dose-0 arm is the
  anchor: its needle sits at offset 0 by construction, and the
  head-merge term — the generic BPE case, a template head
  ending in a space or newline merging with the following
  token — is present in both LCPs and cancels. Acceptance
  requires `D == depth% × dose / 100` exactly (both doses make
  every target an integer). A construction whose seams move
  `D` off target is rejected and the search continues; it is
  never relabelled. No marker-derived constant appears anywhere
  in the measurement path (red-team B executed the marker
  anchor failing outright under the repo's word tokenizer and
  biasing −1 under merging heads; `D` matched token-span ground
  truth in every executed case).
- **Joint final verification is the load-bearing invariant.**
  Both equalities (delta, and `D` for each needle) are
  evaluated on the accepted construction's own final ids
  before the cell is generated. The two-phase search below is
  an optimisation whose phase-1 measurements the suffix could
  in principle move (rank-ordered BPE reach-back); the joint
  check on the final render is what makes acceptance sound —
  every reach-back case changes the final render's LCP or
  length and is rejected there, never wrongly accepted.

## Sizing

Rule `needle-phased-exact-v1`; its full text is this section,
and the protocol carries `sizing_rule_sha256` over the frozen
rule text constant in `needle.py` (the multiturn
`template_digest` pattern — a bare version number binds
nothing). The filler stream is `NEUTRAL_CORPUS` cycled in
order, six cycles (426 sentences), sentences consumed in
stream order: a span is `span_text` over the stream (distract's
whole-sentences-plus-one-truncated-suffix construction), and
the next span continues at the sentence after the previous
span's last whole sentence (a truncated sentence counts as
consumed). One corpus pass is ≈1,278 tokens under a typical
tokenizer, so a 4096 bulk wraps the stream about 3.2 times —
each sentence appears three or four times in a 4096 bulk,
a stated property of the arm.

Phases per treated cell: (1) prefix search — candidates from
`span_candidates` (distract's closeness order; the estimate
orders, never decides) targeting the depth offset; a probe is
the candidate's arm-context render (prefix + needle +
question) plus its comparator render (prefix + question);
the first prefix whose `D` equals the target wins. Depth 0
takes the empty prefix with no search. (2) suffix search —
candidates continuing the stream, targeting the remaining
delta; a probe is the full arm render; the first suffix
satisfying the delta equality (with `D` re-verified on the
same ids) wins. Depth 100 takes the empty suffix. Duplicate
adds a middle phase between (1) and (2): candidates
continuing the stream targeting needle 2's offset, probed as
(prefix + needle + middle + needle + question) plus its
comparator; then the suffix phase targets the duplicate delta
equality, re-verifying both offsets on the final ids.

Encode budget `SIZING_ENCODE_BUDGET = 512` per cell, counting
every probe encode in the phases (a phase-1 probe is 2
encodes; a suffix/middle probe is 1, plus its comparator in
the middle phase). The cached `Q`/`Z` renders and an accepted
prefix's comparator are infrastructure, computed once per
(candidate, case[, prefix]) and not counted; the joint final
verification reuses the accepted construction's renders and
adds no encodes. The chars-per-token estimate is measured
once per stage from the corpus under the run's backend
(distract precedent); it affects search order only, so
budget-edge `arm_unmatchable` outcomes are
backend-determined — stated, not hidden. Executed feasibility
(coordinator and red-team B, independently): the full
word/char grid at both doses and all depths accepts well
inside budget (worst observed 257 encodes); merging-tokenizer
residue is reject-mode (unmatchable), not wrong-accept, once
`D` is the anchor.

One liveness limit is structural, found by executing the
built stage against a merge-dense digraph tokenizer: every
span is a prefix of its stream segment, so the rendered
delta is a single step function of span length, and where a
span's last character toggles a seam merge (a trailing space
that merges with the following text, lost as the span grows
past it) the step is +2 — the walk steps *over* the target
and no span at any length hits it for that cell. Those cells
are `arm_unmatchable` (about 30% of cells under the
adversarial digraph tokenizer in the cycle-191 tests, 0%
under word and byte tokenizers); the step-skip can only ever
remove cells, never corrupt one — every accepted cell still
satisfies both equalities on its final render, and the
readouts run over recorded cells with printed denominators.

Per-cell screening runs on the filler texts only (the needle
is the plant): filler containing the case value or the
candidate's decoded boundary text makes the cell
`arm_screened`. Stage-level, the frozen texts (fact template
with slots emptied, topics, words, corpus) are audited for
special-token literals against every candidate format before
any generation; a hit fails the whole stage closed
(`needle_screen_failed`) because it would forge turn structure
in every cell — the multiturn settlement. The screening rule
is likewise carried as `screening_rule_sha256` over its frozen
text in `needle.py`: per-span, case-folded, the value screen
and the boundary-text screen of this paragraph.

Cell detail vocabulary is frozen: `""`, `arm_unmatchable`,
`arm_screened`, `arm_context_exceeded`,
`arm_evaluation_failed`; a cell is present exactly when its
detail is `""`. Stage details: `needle_record_budget_exceeded`,
`needle_screen_failed`, `cancelled`; the payload-level
fail-closed form is `{"available": false, "detail":
"needle_unavailable"}`.

## Claims and denominators

Every rate travels with its denominator; readouts are
withheld, never zeroed. One **headline population** per dose
keeps every comparative readout like-for-like: the cases
where *every* candidate in the run passed `dose_0` with all
five depth cells at that dose recorded. (Per-candidate
populations were rejected on red-team execution: a flat
per-candidate curve could coexist with a +0.50 deficit over a
differently-conditioned subset, and per-candidate deficits
over different residues are not the comparison this probe
exists to make.)

- Curves: per dose, per depth, `n_pass / n_population / rate`
  over the headline population — one denominator per dose,
  shared by all depths and all candidates. A flat curve over
  this population makes the deficit exactly zero, by
  construction.
- `mid_depth_deficit` per dose: `(rate(0%) + rate(100%))/2 −
  rate(50%)` over the headline population — *Lost in the
  Middle*'s functional form (edge mean minus middle),
  credited, not renamed. Withheld (`None`) when the headline
  population is below `MIN_HEADLINE_POPULATION = 6`.
- `first_failure` per dose: over the headline population,
  counts of the first failing depth in 0→100 order, plus
  `n_no_failure`; withheld with the deficit.
- Dose-0 control per candidate: recorded/pass counts and rate
  over the 20 cases (the candidate's own control level; not
  part of the headline population arithmetic).
- `false_recall` per dose per candidate: evaluable count,
  false-pass count, rate from that dose's absent cell (own
  recorded cells — a within-candidate floor, denominators
  printed).
- `duplicate` per dose per candidate, over its own population
  (dose_0 pass ∧ duplicate recorded ∧ the 25% and 75% cells at
  that dose recorded): pass count/rate; the independence
  baseline `1 − (1 − r25)(1 − r75)` from the same population's
  single-needle rates, and `excess_over_independence` —
  printed as a descriptive contrast under a stated modelling
  assumption, not a test; plus mid-fail rescue counts
  (depth-50 fail ∧ duplicate pass) and the rescue rate over
  mid-fails (`None` when there are none).
- Selection per cell key (all fifteen): the conservative rule
  over candidates on that cell's completed case set, withheld
  — decision `unavailable`, detail
  `selection_withheld_mismatched_sets` — when a completed set
  is empty or the sets differ across candidates. `flip`
  compares the cell's outcome to the `dose_0` selection
  **recomputed on the same completed set** (a paired-restricted
  baseline; comparing against the full-set dose-0 outcome
  would mix set differences into the flip), and is `None`
  when either side is withheld. The `dose_0` condition entry
  reports the selection over its own completed set with
  `flip: false` when not withheld.

There is no summary classification and no latency claim of any
kind (the cycle-179 closure: depth curves are verdict curves).
The falsification reading is stated, not encoded: a candidate
whose curve is flat across depths at both doses — no depth
effect beyond its own false-recall floor — falsifies "this
template exhibits position-dependent recall" for that
candidate, and the section prints the curve either way. Two
scope limits are part of the readout, not footnotes: the
filler is topically *dissimilar* prose (Chroma showed similar
distractors hurt more, so these curves are a lower bound on
position effects under realistic retrieval bulk), and each
cell is one greedy generation — the curve is a 5-point sample
of position, with per-depth resolution set by the headline
population's denominator.

## Records and validation

Payloads carry ids, counts, and digests only — never needle
values or answer text. Each record stores the case id, topic,
the expected answer's SHA-256 digest, and `question_tokens`
(the length of the case's `Q` render for that candidate);
each cell stores verdict, boundary fields, input tokens,
answer digest, and the measured needle offsets (`[t]` treated,
`[t1, t2]` duplicate, `[0]` for `dose_0`, `[]` for absent).
There are no construction/seam fields: under delta acceptance
the seam classification of earlier probes needs a new
definition, and the offsets and deltas carry the evidence —
silence was the one wrong option (red-team B), so the fields
are dropped, not attested.

The verdict is bound, at the multiturn standard:
`verdict == (answer_digest == sha256(expected)) and
boundary_found`, with the expected digest taken from the
record and the case set mirror-regenerated by the validator
(the cycle-187 numform F2 settlement — a payload-derived case
hash would be tautological). The diff validator recomputes
everything else: the case set from the protocol seed by its
own mirror of the derivation table (fixture ids, topics, and
expected digests, in order); the protocol fingerprint;
cross-candidate fixture order; the cell identities — treated
`input_tokens − dose_0.input_tokens == dose`, duplicate
`== dose + needle_len` with `needle_len` derived as
`dose_0.input_tokens − question_tokens`, absent
`input_tokens − question_tokens == dose`; needle offsets equal
to the frozen targets; and all aggregates and per-cell
selection (including the paired-restricted flip baselines)
with independent arithmetic.

**Honesty (attested links, disclosed).** Three fields are
stage-attested, at the same standard multiturn discloses:
the needle offsets (the validator checks them against the
frozen targets; it has no renders and cannot re-derive them —
a stage that wrote the targets without measuring would pass,
which is why the stage's measurement is itself test-pinned
against scripted tokenizers); the absolute level of
`input_tokens` (every identity is a difference, and unlike
distraction — whose native counts anchor in the recorded
primary run — needle's dose-0 is stage-generated, so no
external anchor exists); and `question_tokens` in the corner
where both absent cells of a record are unrecorded (one
unobserved degree of freedom shared with the derived
`needle_len`; it feeds no aggregate). The digest-only verdict
binding proves internal consistency, not that a generation
occurred — the multiturn standard, one step below
numform-190's answer-text binding, and stated as such.
Needle cells are generations on generated fixtures, not
recorded search observations, so they are outside
`verify --determinism` scope (the multiturn settlement).

## Cost

Fifteen generations per candidate per case; 7,200 at the
caps (24 candidates × 20 cases × 15), which is also the
ground-record arithmetic: `MAX_GROUND_RECORDS = 480`,
`GENERATION_BUDGET = 7,200`. Prompts reach ~4,100 bulk tokens
at the top dose under full-prefix recompute with no KV cache;
forwards per generation scale as M·L + M(M−1)/2 in generated
length M and prompt length L. The CLI help states the counts,
the forwards formula, and the `--max-context` requirement for
the 4096 dose before the run. The run's cancel event is
threaded dispatch → wrapper → stage → evaluate, checked
between cases and between cells.
