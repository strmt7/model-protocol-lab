# Instruction-density design (cycle 204, v2 — adjudicated)

Wave-9 rank 3. Question: when one prompt carries many
simultaneous instructions, at which density does adherence
start dropping, and how does satisfaction vary by serial
position — measured per template family on a local checkpoint,
with every constraint decidable from a digest commitment and
every verdict re-derived by the diff validator. v1 was
rejected by the cycle-204 adversarial round (two FATAL
findings: no satisfiability guarantee; an uncalibrated
falsification statistic); this v2 incorporates the
adjudication in `agents_out/cycle204/coordinator_adjudication.md`.

## Utility

A practitioner stacking simultaneous instructions into one
prompt learns, per template family on their own checkpoint,
how adherence falls as simultaneous-instruction density rises
from solo to sixteen, and how satisfaction at full density
varies across early, middle, and late serial positions —
before shipping a template under which the later instructions
silently stop being followed. Position claims are scoped to
what the denominators support (see Readouts): per-position
rates at full density carry n = 20 per cell, so the headline
position readout is band-pooled.

## What this probe is not

- Not IFScale (Jaroslawicz, Whiting, Shah & Maamari,
  arXiv:2507.11538 — verified against the abstract in this
  cycle): 500 keyword-inclusion instructions on a long
  business report, 20 frontier API models, best 68% at max
  density, bias towards earlier instructions. That is the
  phenomenon at API scale on long free-form output; it does
  not vary the chat template, does not run under exact-token
  discipline, and its output imports a verbosity confound
  this design removes with short answers. The uniqueness
  here is the combination: simultaneous-instruction density
  × serial position × template family, on a local
  checkpoint, with digest-decidable constraints and a
  recomputing diff validator. The claim is the combination,
  never first.
- Not `hierarchy` (two conflicting privileged instructions),
  not `distraction` (irrelevant text carrying no demands),
  not `multiturn` (instructions spread over turns), not
  `structured` (one schema, adherence only). None varies the
  count of simultaneous compatible instructions as the
  independent variable.
- Not a judge-based instrument: no LLM scores anything.

## The digest-decidability constraint (slate adjudication)

The diff validator derives satisfaction from stored digest
data alone — never from stored satisfaction booleans. The
vocabulary is capped to what a word-level commitment can
decide, and the cap is printed here: case, sentence, and
paragraph structure constraints (all-lowercase, no-commas,
N sentences, bullet form) are NOT decidable from the
commitment and are excluded from v1 by name. One deliberate
measurement choice, stated where a reader can see it: word
identity is exact after normalization — `include_word("seven")`
scores an answer of "7" as failure; there is no digit↔word
mapping.

## Construction

CASE_COUNT = 20 cases, generated from the run seed by a
frozen generator (`density.py`); the diff validator carries
an independent mirror of the generator, the rendering, and
the checking rule.

Each case has:

- A base task from a frozen pool of short-answer task
  templates over synthetic content. The base instruction
  asks for a short answer and carries NO numeric length
  demand (length demands live only in `word_count_at_most`,
  where they are counted). Each case carries
  `expected_word_digest`: the SHA-256 of the task value's
  normalized word (multi-word values: the ordered hash
  list's digest), so task performance is measurable from
  the commitment (task-value presence) without a whole-answer
  digest — a compliant answer at high density necessarily
  differs from the canonical task answer as a string.
- A frozen ordered list of 16 constraints, positions 1..16.
  The K-density prompt for K in {2, 4, 8, 16} is the base
  task plus the first K constraints; K = 1 is the solo arm
  of constraint 1 (see Arms). A constraint's serial position
  is identical at every density that includes it.

### Constraint vocabulary (frozen) and kind allocation

Kinds (each a pure function of the word commitment):

1. `include_word(W)` — the answer's word multiset contains W.
2. `start_word(W)` — the answer's first normalized word is W.
3. `end_word(W)` — the answer's last normalized word is W.
4. `word_count_at_most(N)` — the answer's word count <= N.

Kind allocation per case is FIXED: 4 include, 4 start, 4 end,
4 count, in a kind order that rotates across cases by case
index (case c starts the repeating kind cycle at offset
c mod 4), so each position carries each kind exactly 5 times
across the 20 cases. Count parameters come from the frozen
set {8, 12, 16, 24}.

### Satisfiability invariant (generator-enforced, mirrored)

Per case: all `start_word` constraints share ONE target; all
`end_word` constraints share ONE target; the distinct word
targets across groups — 4 include targets, the start target,
the end target, the expected task word — are pairwise
distinct AFTER normalization (D <= 7 distinct words); and
min(N over the case's count constraints) >= D, which the
frozen N set guarantees (8 > 7). A satisfying answer always
exists (start target, include targets, task word, end target
in <= 7 words). The invariant is proven by test over the
pools and seed sweeps, including a pinned perfect-model test:
a constructed perfect follower scores discordance f = 0 and
joint satisfaction 20/20. Stated limit: at k16 the start and
end demands repeat (same target four times), so for
positional kinds the density treatment is partly a
repetition treatment; per-kind readouts keep this visible.

### Word commitment and normalization (frozen rule)

For each generated answer the stage records ONE commitment
object: `word_hashes`, the ordered list of SHA-256 hashes of
the answer's normalized words (multiplicity preserved), and
`word_list_digest`, SHA-256 of the list's canonical encoding
(compact JSON, frozen in the RULE_TEXT). A word is a maximal
run of alphanumeric characters, case-folded; every other
character is a separator. Derived from the list, everywhere
(stage aggregation and validator alike): the word multiset
(count the list), `word_count` (the list's length), the
first word (list head) and last word (list tail), or None
when the list is empty. The first word is the first
NORMALIZED word: leading tokens that normalize to nothing
are skipped by construction.

Satisfaction of every constraint is DERIVED from the
commitment; no satisfaction boolean is stored (the rotation
F4 pattern — the primitive determines the verdicts for all
four kinds). Records also carry `answer_digest` (SHA-256 of
the exact boundary-cut answer text, the standing pattern).

Stated limits. (1) Fidelity residue: the commitment's
internal consistency is validator-checked (digest over the
list), but its fidelity to the generated answer is
stage-attested — no digest of free text admits membership
recompute against the text itself, and the verbatim-text
alternative (the numform cycle-190 exception) was ruled out
for this probe by the slate adjudication. (2) Exposure: the
ordered hash list discloses the answer's word SEQUENCE to a
dictionary pass over the (public) frozen pools — a strictly
greater degree of disclosure than a whole-answer digest, of
the same class; accepted because the determination property
(F4) requires the ordered primitive, and the answers are
short synthetic-task outputs.

### Rendering (frozen)

Each kind's instruction template, the constraint-list joiner
and numbering, the base task templates, and the neutral
statements are frozen text in `density.py`, carried by digest
in the protocol (the needle SIZING_RULE_TEXT pattern). The
payload carries, per (case, arm), the user-turn content
digest; the validator's mirror re-renders the content and
requires digest equality — the treatment is derived, not
attested. Records carry `input_tokens` under the rotation
arm_tokens discipline.

### Draw rules (frozen; the mirror reproduces them exactly)

Slot order per case: task template, task parameters, the
sham arm's neutral statements, then per position 1..16 the
constraint kind (fixed by the allocation rotation) and its
parameter. Collision rule, with "occurs" defined as
normalized-word equality: word targets must be distinct
across groups (above) and must not occur in the case's base
task text; constraint 1's target additionally must not occur
in the case's neutral statements (the only constraint that
shares an arm with them). Redraw scope: the offending
parameter only, at its own slot counter; redrawn parameters
are re-checked against the same frozen texts. Termination:
pools are sized so the redraw always has a fresh candidate
(pool sizes frozen in density.py and stated in the protocol;
the generator test sweeps seeds to prove termination).
Pool screen (pinned property, proven by test): all pool
words are pairwise distinct after normalization, and no pool
word equals a number word after normalization.

## Screening (rotation-shaped)

Stage level: before generation, all frozen texts (task
templates, kind renderings, neutral pool, word pools) are
audited with `audit_special_token_literals` against every
candidate, and boundary-text containment is checked; failure
fails the stage closed with `density_screen_failed`.
Per case: a case whose rendered texts contain a candidate's
decoded boundary text is screened as a unit, with its own
counter. Records carry a per-record detail in
{"", arm_evaluation_failed, arm_context_exceeded}; records
with a detail, and boundary-screened cases' records, are
present in the record list (completeness below) with
`word_hashes` = [] and `word_count` = 0.

Live population: records with detail "" whose case is not
boundary-screened for that candidate. There is NO word-cap
screen: generation is bounded by the frozen max_new_tokens
(64), so every live record's commitment is complete and every
kind's verdict is decidable for it in both directions.
Zero-word live records (empty answer: first emitted token is
the boundary, or the answer normalizes to no words) carry
their own counter and are excluded from `word_count_at_most`
checked denominators (their count verdicts are vacuous);
they still fail word-target kinds, decidably.

Screen accounting per candidate, counters disjoint by the
stated order (boundary screen first, then detail, then
zero-word within live): n_generated, n_boundary_screened,
n_detail_screened, n_zero_word, n_live — all printed.

## Arms

Per case and candidate, 21 generations:

- 16 solo arms `solo_1..solo_16`: base task + only
  constraint j — the per-constraint paired baselines.
  `solo_1` IS the K=1 density arm (the v1 design generated
  it twice under two names; merged).
- 4 density arms `k2, k4, k8, k16`: base task + the first
  K constraints.
- 1 sham arm `sham`: base task + constraint 1 + neutral
  statements selected by the frozen rule below, separating
  "many instructions" from "much text".

Sham construction: block length is the user-turn word count
of the appended block. Neutrals are appended in frozen slot
order, wrapping to the start of the drawn list, until the
sham block's word count first reaches or exceeds the case's
k16 block word count; both block lengths are recorded per
record through the content commitment, and the selection is
a deterministic function the mirror recomputes. (Build-time
amendment, measured: without wrapping the sham stayed
shorter than k16 in 3,992 of 4,000 generated cases — the
length match the control exists for almost never held.
Wrapping adds no pool words: the neutrals are pinned
pool-free, so repetition cannot smuggle a target in.) The
sham readout is computed over cases live in all three arms
(solo_1, sham, k16), denominators printed.

## Records

Exactly 21 × CASE_COUNT records per candidate, in canonical
(case, arm) order; the validator rejects any other count,
order, or arm set, and rejects duplicate candidate names
(the run-level guard's clause, stated as rotation's
validator states it). Each record carries: candidate, case
index, arm id, detail, `answer_digest`, `word_hashes`,
`word_list_digest`, `word_count`, `input_tokens`, the
content digest of its arm, and the backend fingerprint via
the shared `_backend_fingerprint` guard (cycle-202 F2). The
protocol object carries: the frozen rule texts by digest
(generation, rendering, normalization+checking), pool sizes,
densities, CASE_COUNT, the paired floor and discordant-case
floor, alpha, the word cap implied by max_new_tokens, and
the record/payload caps. Caps: MAX_DENSITY_RECORDS = 10,080
(21 × 20 × 24); MAX_DENSITY_PAYLOAD_BYTES = 52 MiB (sizing:
a record's ordered list is at most 64 entries × 68 B
≈ 4.4 KB plus ≈ 0.4 KB of fields ≈ 4.8 KB; 10,080 × 4.8 KB
≈ 48.4 MiB — the derivation, printed per the house pattern).
Stage detail vocabulary: {cancelled,
density_record_budget_exceeded, density_screen_failed}.

## Readouts

Per candidate, over live records; every rate is None iff
its denominator is 0; every readout prints its denominators.

- Satisfaction by density: for each K in {1, 2, 4, 8, 16},
  the pooled satisfaction over that arm's constraints
  (K = 1 reads solo_1), printed BESIDE the same constraints'
  pooled solo rate (the composition control). The raw curve
  is labelled not composition-controlled.
- Task-value presence by arm (the F6 readout): share of
  live records whose commitment contains the expected task
  word, per arm, beside the constraint readouts.
- Position at full density: per-position satisfaction
  within k16 (16 rates, n = 20 denominators, descriptive),
  and the band-pooled headline: positions 1–5, 6–11,
  12–16 pooled rates with denominators.
- Per-kind satisfaction at solo and at k16, with
  denominators (the repetition-limit readout for positional
  kinds).
- Joint satisfaction at k16: share of live k16 records with
  all 16 constraints satisfied, with denominator (the
  IFScale headline analogue — meaningful only because the
  satisfiability invariant guarantees it is attainable).
- Sham readout: position-1 satisfaction in solo_1, sham,
  and k16 over the intersected live population, with the
  two block lengths printed.
- Discordance: per (case, position) paired solo vs k16
  satisfaction over eligible pairs (both records live; for
  count-kind pairs, neither record zero-word): f =
  solo-pass/k16-fail count, p = solo-fail/k16-pass count,
  printed pooled and per position (the per-position table
  is a density-and-position mixture — solo_j sits at
  effective position 1 — and is labelled as such).

### Falsification (pre-registered, scoped)

Claim, scoped to the contrast the test supports: for a
candidate, per-constraint satisfaction at full density
(k16) equals its solo satisfaction — independence of the
endpoint density contrast; intermediate densities are
descriptive (the satisfaction curve), and a non-monotone
intermediate dip does not falsify and is not claimed to.
Decision, a total function per candidate over the eligible
pairs: aggregate discordance per case, d_c = f_c − p_c.
If the candidate has no live records, its block is
unavailable. If the number of cases with d_c != 0 is below
6, the verdict is `insufficient` — the boolean is withheld,
never defaulted to acquittal, and the counts are printed.
Otherwise: falsified iff Σf > Σp AND the exact two-sided
sign test over the discordant cases gives p <= 0.05; the
p-value, f, p, and the discordant-case count are printed
beside the boolean. The level is per-candidate; across 24
candidates at alpha = 0.05, roughly one false falsification
is expected by chance (1 − 0.95^24 ≈ 0.71 for at least one),
and the section text says so.

## Diff validator

`diff_density` in diffreports.py with a full independent
mirror: regenerate the cases, re-render every arm's content
and require the payload's content digests to match, verify
each record's commitment (list digest), derive every
satisfaction verdict, task-presence verdict, headline, and
the falsification decision (including the sign test), and
require equality with the payload. Availability and
`insufficient` states are mirrored per candidate. The
diff-section blast radius is the standing three-element one
(test_cycle67 oracle + flip_identical anchor +
`_identical_expected`; test_cycle27 oracle + stub), verified
by executed grep at build time.

## Budget

Generations: 21 arms × 20 cases = 420 per candidate;
10,080 at the 24-candidate legal max. Solo arms are 16 of
21 arms; they earn their cost because the decision (fixed)
and the composition control both consume them — the
cycle-204 F17 sequencing condition. Construction is one
render per arm. Records and payload caps as above.

## Pin blast radius (rewritten from executed greps)

cli.py flag `--density` + dispatch + help; experiments.py
stage mirroring `rotation_stage`; reporting.py three
touchpoints (compute, TOC tuple, body concatenation);
diffreports.py section + validator; new test file
tests/test_cycle204_density.py plus the scoped test files
the gate names; test_cycle27 and test_cycle67 oracle
extensions (grep-counted at build time, rotation's
footprint: 25 / 80 placement-era hits are the model);
test_cycle179 dispatch e2e pattern; test_cycle44 mypy
source-file pin 60 → 61; test_cycle178 VALIDATION header
pin cycles 10–204 AND the live counts it pins (VALIDATION
test-count lines and the CI-scope ruff file count);
VALIDATION.md first-Evidence prose probe list gains the
density entry; README test count, capability bullet, and
BOTH section enumerations (the diff-reports sentence and
the sample-report paragraph — grep both); USAGE §8.51;
CHANGELOG entry in the same push; RESEARCH_LOG cycle-204
entry (exists — design phase already appended; the build
appends its closure half); sample-report impact verified at
build time (regenerate byte-identically twice if the
section joins it).

## Open points — dispositions

1. Solo-arm cost: PAID (16 solos kept; k1 merged into
   solo_1). The entry-density alternative covers positions
   1–8 only and changes the estimand per position; rejected
   with the reason recorded.
2. Sham position confound: position 1 is matched across
   solo_1/sham/k16 (held by both red teams); the length and
   population confounds are fixed by the frozen selection
   rule and the intersected population.
3. Fidelity residue: stands as a stated limit (above); the
   commitment's internal structure is now fully determining
   (F5 fix), so the residue covers origin only.
4. Vacuous count satisfaction: zero-word records are
   counted and excluded from count-kind denominators (F13
   fix); the residue is printed, not hidden.
5. Pooled-only boolean: replaced by the case-aggregated
   sign decision with an `insufficient` state (F2/F10
   fixes); the claim is scoped to the endpoint contrast
   (B-F5).
6. Replication control (B's alternative for the duplicated
   arm): considered and deferred — the case-level decision
   does not consume a noise-floor readout; if a future
   cycle needs one, the duplicate arm is the place to add
   it, with this paragraph as the pointer.
