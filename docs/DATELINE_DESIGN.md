# Date-line sensitivity design (cycle 188)

Status: frozen before code. This document is the contract the
implementation, the diff validator, and the test oracles are built
against. It resolves red-team A's F1–F15 and red-team B's corrections
(M1–M6) from the cycle-188 briefs in
`~/workspace/agents_out/wave8/cycle188/`; where A and B disagreed,
B's cross-checked disposition governs and is cited inline.

## Utility

If your serving stack prepends a "Today is …" line, `run
--dateline` tells you — per template family, on your own checkpoint —
how often changing only that date flips a task verdict, and whether
the template you picked still wins afterwards, so a calendar
rollover cannot silently change your eval results after you ship.

The phenomenon is published: Sanz-Guerrero, Bui, Mager, and von der
Wense, *Dating the Model: Hidden Dates in System Prompts Affect LLM
Evaluation* (arXiv:2609.36931, 2026) — across 9 recent LLMs and 6
datasets (MCQA, math reasoning, code generation, machine
translation), changing only the injected current date moves scores
by up to 6% on MCQA, 14% on math reasoning, 7% on code generation,
and 2.84 BLEU on machine translation; rankings shift; the effect
exceeds batch-size and numerical-precision variance; few-shot does
not reduce it and chain-of-thought amplifies it. That work varies
the date inside each vendor's own system-prompt rendering on served
models. No surveyed harness ships the combination frozen here: a
frozen date-line treatment × hand-rendered template families × one
local checkpoint × exact-match greedy scoring with an abstention
rule × a length-matched no-date control (uniqueness survey:
RESEARCH_LOG cycle 188; the claim is the combination, never
"first", and generic plumbing in lm-eval/promptfoo/Inspect could
approximate it by hand — what is claimed is a shipped, frozen,
recompute-validated probe).

Boundary, one sentence: `doctor --template-probe` asks whether the
checkpoint's own template code changes its *rendering* in response
to message content (render-only, no generation, no scoring), whereas
`run --dateline` asks whether an operator-supplied date string in
the message content changes the model's *scored verdicts* under each
hand-rendered template family (generation, exact-match). Note the
sharper limit from TEMPLATE_PROBE_DESIGN T4′/T2′: a day-granularity
date injection is constant across a template-probe pair and is
therefore not a template-probe finding at all; the two probes are
complementary, not mirror images.

## Construction

The lab has no system role (`core.Message` is user/assistant only),
so the date line is delivered the way hierarchy delivers privilege
blocks: folded into a user turn inside a frozen bracketed block.
Placement is the probe family's uniform choice (repetition,
distraction, leading all operate on the last user message): the
block is **prepended to the last user message's content**. The
construct is therefore named honestly everywhere (this doc, USAGE,
CLI help, section prose): **a date line immediately before the
final user message** — for single-turn fixtures that is the head of
the conversation; for multi-turn fixtures it is not, and the search
set's multi-turn share is reported beside the headlines (G9†).

- Block template (frozen literal):
  `[Context] Today's date is {DATE}. [/Context]`
- Joiner (frozen literal): `"\n\n"` between the block and the
  original content — `block + "\n\n" + content`. (Leading appends
  with `"\n"`; a prepended block takes the hierarchy-style blank
  line. The choice is frozen here, not left to the builder.)
- The arm fixture keeps the native fixture's id, so
  `core.request_seed_for` pairs request seeds across arms.

## The frozen arm set (G1†, G7)

Eight date values, each rendered in three frozen formats. Weekday
names are part of the protocol data: every arm's weekday was
computed from its own calendar date at freeze time and is pinned by
test; no weekday is ever derived at runtime (a builder templating
the anchor's weekday into every arm would silently convert five
arms into extra lie arms — the validator's weekday rule below
exists to catch exactly that).

| value key | date | weekday |
|---|---|---|
| `anchor` | 2026-10-21 | Wednesday |
| `minus_day` | 2026-10-20 | Tuesday |
| `plus_day` | 2026-10-22 | Thursday |
| `minus_year` | 2025-10-21 | Tuesday |
| `plus_year` | 2027-10-21 | Thursday |
| `leap` | 2028-02-29 | Tuesday |
| `boundary_eve` | 2026-12-31 | Thursday |
| `boundary_day` | 2027-01-01 | Friday |

Formats (frozen renderers over frozen English month/weekday tables;
no `strftime`, no locale anywhere in the module, the stage, the
reporting section, or the validator):

- `iso`: `2026-10-21`
- `long`: `21 October 2026`
- `weekday_iso`: `Wednesday, 2026-10-21` — weekday name + ISO, the
  primary paper's own hidden-prompt shape ("Today is Wednesday,
  2025-12-03"). B's correction is adopted: the weekday-long hybrid
  (`Wednesday, 21 October 2026`) appears in no cited source and
  would re-confound weekday with the long-form rendering in every
  pair; weekday-ISO keeps the date tokens identical to the ISO arms,
  so the ISO↔weekday-ISO pair per value isolates the weekday
  channel.

Arm keys: `{value}_{format}` for the 24 date arms (e.g.
`anchor_iso`, `leap_long`), plus the controls and the lie below —
28 treated arms in total.

Arm → effect mapping (the design text asserted the set "spans the
effects the primary source reports" with no mapping; here it is):
`anchor` is the reference date every contrast is anchored to;
`minus_day`/`plus_day` are the smallest calendar step (the paper's
week-of-dates variation); `minus_year`/`plus_year` span the
year-scale variation — under the weekday-bearing format they
necessarily shift the weekday too (365 ≡ 1 mod 7, no Feb 29 in
either span), so that contrast is named **year+weekday shift**,
never "±1 year" alone; `leap` is the rare-valid-date arm, frozen to
2028-02-29 (the nearer leap day, +496 days, and the next one a stack
frozen at the anchor actually rolls into; its distance from the
anchor exceeds every other arm's, so leapness is confounded with
distance and the doc says so wherever the arm is reported);
`boundary_eve`/`boundary_day` are consecutive days straddling the
anchor's next year rollover — the cleanest year-number contrast in
the set, at near-constant distance.

The anchor is named **anchor date (frozen 2026-10-21)** in the
protocol, the payload, and the docs. It is "current" only on the
freeze date; under the reproducibility rule below every later run's
anchor is a stale date *by design*, and the naming must not invite a
"the probe used yesterday's date" misreading.

### Weekday-lie arm (G1†, F8)

One lie arm: arm key `lie`. Date value: the anchor. Format:
weekday-ISO only (the lie is defined inside the weekday-bearing
format; it is never compared against ISO/long arms as a format
contrast). Direction: **+1** — literal string `Thursday, 2026-10-21`
(the calendar weekday of 2026-10-21 is Wednesday; +1 is the smaller
length step, 9→8 characters, minimising the length confound that
honest-vs-lie unavoidably carries). The lie is a single date ×
single format cell family; the design says so wherever its counts
appear. Its role is inside the date-loss truth table (below): it
separates "any date text perturbs" from "the date's content
matters".

### Reproducibility (G10)

All dates are protocol constants. Nothing in `dateline.py`, the
`experiments.py` stage, the reporting section, or the validator may
read the wall clock or the locale: no `date.today()`,
`datetime.now()`, `strftime`, or `strptime` with `%A`/`%B`. A scoped
pin test greps exactly those surfaces (`doctor.py`'s legitimate
`run_date_utc` is out of scope). A run must be byte-reproducible on
any day. Dateline cells are recorded search-side treatments, not
recorded search observations: they are **outside**
`verify --determinism`'s scope, and the docs say so rather than
implying coverage. Held-out fixtures are never date-treated, never
screened, and never enter any readout; every readout is a
search-side readout.


## The no-date control (G4†)

The control is a **sized neutral text, not a sentence** — B's
adjudication of A F4: a single grammatical sentence has one length
per candidate render and cannot match arms that provably differ in
length from each other (the design's own block shape spans 48–64
characters across arms before any tokenizer effect), and even a
per-format control is matched to at most one date value in that
format, because date values differ in length *within* a format too.
The honest construction is the precedents' one (repetition's pad,
distraction's spans): whole neutral units plus at most one
truncated suffix, sized per (candidate, fixture) against rendered
token ids, with unmatchable cells flagged and the evaluable share
reported.

Frozen choices, priced in the budget below:

- One control per format: arm keys `nodate_iso`, `nodate_long`,
  `nodate_weekday_iso` — 3 control arms.
- Each control cell is sized to match **that format's anchor arm**
  for the same (candidate, fixture): the control fixture's rendered
  token count must equal the anchor fixture's. The date-vs-control
  contrast is therefore exact at the anchor. For non-anchor date
  values the control contrast is **length-unmatched by
  construction** and is labelled as such wherever reported, with the
  per-cell `input_tokens` deltas available in the payload — the
  doc does not let the word "matched" drift onto those contrasts.
  (The fully-matched alternative — one control per date value —
  costs 24 control arms and 117,600 worst-case generations for a
  contrast the headline does not use; rejected and priced here so
  the rejection is auditable.)
- Neutral unit (frozen literal, new for this probe, temporally
  inert): `The reference panel is mounted on the north wall of the
  room.` Repetition's filler is *not* reused: it contains the
  temporal phrase "that morning", which would put temporal priming
  in the no-date control of a date probe.
- The control block keeps the envelope and replaces the sentence:
  `[Context] {NEUTRAL} [/Context]`, same joiner, same position.
- Sizing rule (frozen name):
  `whole-units-plus-one-truncated-suffix-v1` over the neutral unit,
  search budget 512 encodes per cell (repetition precedent),
  verified on rendered ids. Cells that cannot be matched carry
  `control_evaluable: False`; date-vs-control contrasts are computed
  only over matched pairs, and each control arm's evaluable share is
  a payload field, never silently absorbed into a denominator.
  Matching failure is candidate-correlated (tokenizers chunk date
  digits differently), so denominator loss is not random across the
  candidates being compared — the shares are reported per candidate
  for exactly that reason.
- Stage-start screens (fail-closed, leading/distraction
  precedent): the block template, the neutral unit, and every arm
  string are audited against the catalog's special-token literals
  (`audit_special_token_literals`), and the neutral text is screened
  against every fixture's expected answer and every candidate's
  decoded boundary text (`screen_filler` discipline). A failure
  fails the stage closed (`dateline_screen_failed`), before any
  generation.

## Eligibility screen (G5†)

A fixture whose expected answer is itself date-derived measures
copying, not sensitivity: the date line hands the model the exact
string to copy. Screened fixtures are excluded from every treated
population and form their own record class; the screened share per
candidate is a payload field.

The design's original screen — `scorerlens.rung_match` of the
expected against the arm strings — is **unsound as specified**
(A F5, executed; B §1 reproduced the battery verbatim): rung-4 is an
equality predicate with no containment component, so expecteds `21`,
`2026`, and `October` match nothing, and an expected written as a
date in a third format (`October 21, 2026`) matches nothing either.
The opposite pole — substring/containment — is equally unsound
(executed: `equivalent_claim('1', '2026-10-21')` is True, and a
containment screen voids nearly every single-digit numeric fixture,
precisely the population the source paper's math result concerns).
The frozen screen is component-wise equivalence, amended per B §1:

- **Rule name (frozen):** `dateline-components-r4-v1`.
- The expected is decomposed too (B's granularity correction: A's
  cure compared the *whole* expected against each component and so
  missed every multi-token date-derived expected — `March 3`,
  `October 21`, `21 October`). Both sides are tokenised on
  non-alphanumeric boundaries and compared as component sets.
- A fixture is screened out if any of these holds:
  1. rung-4 equality of the whole expected against any full arm
     string;
  2. rung-4 equality of any expected component against any
     **date-bearing** component of any arm string — years, month
     numbers, month names, day numbers, weekday names, and the
     frozen abbreviation aliases (`Jan`, `Feb`, `Mar`, `Apr`, `Jun`,
     `Jul`, `Aug`, `Sep`, `Sept`, `Oct`, `Nov`, `Dec`; `Mon`, `Tue`,
     `Tues`, `Wed`, `Thu`, `Thur`, `Thurs`, `Fri`, `Sat`, `Sun`).
     Carrier words (`Today's`, `date`, `is`, `Context`) are *not*
     components: B executed `rung_match('is', 'is', 4) = True`, and
     screening on envelope words would void common-word expecteds
     for a property of the envelope, not the date;
  3. a **closed-format date-parse leg**: the whole expected is tried
     against a frozen list of explicit formats (ISO `YYYY-MM-DD`;
     `D Month YYYY`; `Month D, YYYY`; `D/M/YYYY`; `M/D/YYYY`;
     `D.M.YYYY`; `YYYY/M/D`) using the protocol's own frozen English
     month table — no `strptime` `%B`/`%A`, no locale, no new
     dependency (the stdlib has no general date parser; B §1.2(c)).
     Slash dates are ambiguous by construction (`3/4/2026` is
     2026-03-04 or 2026-04-03): an expected is excluded if *either*
     reading lands on an arm date — over-exclusion, disclosed. A
     year-less month/day (`March 3`, `21 October`) whose month/day
     equals an arm date's month/day is likewise excluded.
- Rung-4's collapse of leading-zero and decimal variants (`021`,
  `21.0` rung-match the day component `21`) is a **deliberate
  over-exclusion**, disclosed here: the screen is a copy/priming
  screen over values, not a byte screen, and `leading`'s donor
  eligibility already treats rung-4-equal answers as the same
  answer. The screened share is reported so the cost stays visible.
- Scope is **fixture-level** (one flag, clean denominators; the
  arm-level alternative makes every pair's population different).
  Blank expecteds are not screened — they are their own record
  class (`expected_blank`, leading precedent).
- A fixture whose *messages* state a date turns the arm into a
  hierarchy-style conflict rather than a sensitivity measurement:
  fixtures whose message text contains a date-parse-leg hit are
  flagged (`message_date_bearing`) and excluded from the treated
  populations as their own class, reported beside the screened
  share.
- Pin battery (from A's execution and B's counterexamples, all
  become tests): `21` screened, `2026` screened, `October` screened,
  `Wednesday` screened, `October 21, 2026` screened, `February 29`
  and `October 21` screened (year-less month/day leg), `Oct`
  screened (alias), `March 3` **not** screened (no arm date falls
  on it — the screen is arm-relative, not a general date detector),
  `3/4/2026` not screened unless a reading hits an arm date,
  `021` screened (disclosed collapse), `is` **not** screened,
  `26` **not** screened (no arm component carries that value —
  substring's false positives die), while `2` **is** screened: it
  rung-matches the leap arm's zero-padded ISO month component
  `02` under the same disclosed value collapse. (Corrected during
  the build: the battery as first drafted claimed `2` unscreened,
  which the executed screen disproved — the component set is
  taken from the arm strings as rendered, padding included.)

## Records (G8, M4)

The record schema is leading's, adopted wholesale: one ground record
per (candidate, fixture) holding the native cell plus one cell per
treated arm; per-fixture classes `vacuous`, `expected_blank`,
`echo_fixture`, `screened`, `message_date_bearing`; per-cell
`input_tokens`, `boundary_found`, `boundary_text_match`, answer
digest, construction classification (`id_exact` vs `text_level`
with seam token ids — a candidate whose prefix merges into
`[Context]` must be distinguishable from a clean render), and
per-cell degradation (`arm_context_exceeded`,
`control_evaluable: False`, never a silent drop).

**M4's rule (from the numform F3 backlog): ground records carry
answer digests only.** The verdict is recomputed by the validator as
`answer_digest == expected_digest ∧ boundary_found` — there is no
stored derived answer field to forge. Any derived answer field
(parsed date, normalised value, …) is forbidden in dateline records
unless digest-bound; numform's own unbound field was closed the
same way in cycle 190 (digest-bound answer text, re-parsed by
its validator) — dateline never needed the fix, because it
never stored the unbound field in the first place.

Byte-cap arithmetic (the driver is cells per record, since the
2,400 ground-record cap already folds in the candidate
multiplication): leading carries ~3.9 KB/record at 5 cells (native +
4 treated); dateline carries 29 cells (native + 28 treated) of the
same minimal shape. The implementation must measure the real
per-record bytes on the shipped schema before fixing
`MAX_RECORDS_BYTES`, show the arithmetic in the module, and set the
cap so the legal-maximum payload (2,400 records) fits under the
32 MiB whole-file loader limit with headroom — if the measured
schema cannot fit, the schema is trimmed, never the record cap
lowered silently. (A's "~9.0 MiB at 19 cells" was an estimate, not
a measurement — B §5; the build replaces it with the measured
number.)

## Readouts (G3, G6†, G9†)

Scoring is the native exact-match + boundary conjunction, unchanged.

**Headline: per-pair discordant contrasts, partitioned by pair
type.** The design's original headline — "fraction of fixtures
whose verdict differs between any two date arms" — is a union over
C(24,2) = 276 pairs under the amended format set, of which 84 are
pure-date (same format, different value: 28 per format × 3), 24 are
pure-format (same value, different format: 3 per value × 8), and 168
change date and format together. A union over that population
saturates under generic fragility — under independent per-arm
failure probability q, P(any-pair flip) = 1 − (1−q)^k − q^k reaches
0.56 at q = 0.05, k = 16 — so a checkpoint with 5% per-arm wobble
would headline as "56% date-flip". (The independence assumption
makes that table a worst case, not an expectation — real failures
correlate across arms and lower the union — and it is quoted only
with that caveat.) The shipped readouts are therefore:

- **Pure-date pair table (the headline family):** per candidate,
  per format, per value pair (28 × 3): discordant counts in leading's
  shape (pass-A-only / pass-B-only / both / neither) with the frozen
  floors below and a total classification function.
- **Pure-format pair table (the format control):** per candidate,
  per value, per format pair (3 × 8): same shape. This is numform's
  axis, reported so a date claim can be checked against the format
  effect on the same fixtures — never merged into a date number.
- **Cross-both pairs are not computed and feed no statistic.** Any
  retained union statistic is pure-date-only, uses the
  all-arms-completed denominator (fixtures completed in every date
  arm of the union — a different, smaller set than any per-pair
  denominator, defined here because the design left it undefined),
  and prints beside the date-loss baseline below, never alone.

**Floors and classification:** `MIN_PAIRED_EVALUABLE = 6`,
`MIN_DISCORDANT = 2` (leading's frozen constants), and a total
classification over (n_paired, a_only, b_only) in leading's
vocabulary. Below the floors the readout is "insufficient" with
denominators shown — the shipped example search set has 2 fixtures,
so on the example every classification is insufficient by
construction, and the CLI help says so with the real budget (the
`--leading` help precedent).

**Date-loss subset (G6†; the design's "pure-date caving",
renamed).** Nothing is planted here for the model to adopt, so
leading's word "caving" — which in leading's payload splits by
byte-exact *adoption* of the plant — is not reused. The subset is
**date-attributable loss**, defined per date value with formats
collapsed: a fixture counts toward value v's loss set when it passes
the no-date control and fails v, over the denominator of fixtures
evaluable in the control and in v in at least one format (per-value
denominators, defined per value, not globally). The truth table per
value carries the lie cell, which is what separates the readings:
loss with the lie also failed is consistent with "any date text
perturbs"; loss with the lie passed is content-specific. The
**asymmetric counterpart** — fails the control, passes every date
arm — is reported beside the loss counts; a subset printed without
its counterpart and without the per-pair table is not a shipped
readout shape.

**Native's role (G6†, F7 as downgraded by B).** Native enters no
flip, loss, or paired population. Its only roles are the per-record
baseline and the per-arm selection-flip comparison (the
leading/distraction pattern). One sentence, frozen here because the
design listed native first among "arms" and a builder could
reasonably have included it.

**Selection and the common population (G9†).** Per-arm selection is
re-applied per arm and withheld when completed fixture sets differ
across candidates *for that condition* (withholding is per
condition across candidates; arms do not withhold each other — A's
"withheld almost always" prognosis is not adopted as a premise).
The primary cross-candidate vehicle is leading's **common-population
readout**: fixtures in every candidate's population, with the
pure-date pair table recomputed over that population. Per-arm
selection is the secondary readout.

**Falsification (G11†, M1).** The design's clause — "date arms
never flip a verdict that the no-date control doesn't also flip" —
is incoherent as written: a flip is a pairwise property and a single
control arm cannot "also flip". Restated in the shipped vocabulary:
*for a candidate, the date-content claim is falsified on this
fixture set when every pure-date pair's discordant count is zero and
every date value's loss count is zero — the section then reports
exactly that (available, zero discordants, denominators shown).*
That restated condition, not the original sentence, is what the
section prints.

## Diff validator (G8, F11)

`diff-reports` gains a `dateline` section held to the cycle-187
standard. The validator carries its **own literal copy** of the arm
table — every date string in every format, the block template, the
joiner, the neutral unit, the sizing-rule name, the lie literal — as
`_DATELINE_*` constants in `diffreports.py`, never imported from
`dateline.py` (independence is the point; the `_LEADING_TEMPLATES` /
`_numform_recompute` precedent). It rebuilds the protocol
fingerprint from those frozen scalars plus the payload's
fixture-set digest and backend fingerprint, recomputes every
aggregate above from the ground records, verifies control matching
from per-cell `input_tokens`, and enforces the record caps. Two
validator-specific rules:

- **Weekday recomputation:** the validator parses each arm string
  back to (year, month, day) with its own frozen English month
  table and recomputes the weekday via `datetime.date.weekday()`
  mapped through its own frozen English weekday table (no
  `strftime`, no locale — the validator carries a small date
  parser, deliberately). Every non-lie weekday-bearing arm's weekday
  name must equal the recomputed one; the lie arm must equal the
  frozen lie literal *and* differ from its recomputed weekday. A
  validator that derived weekdays instead of checking them would
  silently correct the lie into an honest arm.
- **Stated limit:** the eligibility screen is a property of
  expected *text*, and payloads carry digests only. The validator
  checks the screened/message-date-bearing flags' cross-candidate
  constancy (leading's reference-field pattern) but cannot re-derive
  them. This sentence also appears, in substance, in the validator's
  docstring, as leading's does.
- **Primary-pair binding (cycle 196, oversight edge-case F3):**
  `primary_decision` / `primary_winner` are bound to the run-level
  decision vocabulary (`compatible_on_fixture_set`,
  `held_out_failed`, `ambiguous`, `no_candidate`, `cancelled`,
  `failed`), with the winner required to be one of the payload's
  own candidate names exactly when the decision is winner-bearing,
  and `None` otherwise. Before the binding, the pair was only
  type-checked, so a label contradicting its own section's arm
  outcomes could validate. What the binding does *not* do is
  recompute the parent run's decision: the section does not carry
  the parent run's fixture outcomes, so within the vocabulary the
  decision remains stage-attested. The same binding applies to the
  repetition, distraction, and leading validators, which carry the
  same pair under the same mapping.
- **Oracle ground truth (cycle-188 verifier caveat, closed as
  documentation in cycle 216):** the two test oracles recompute
  the dateline arm table and weekday strings through the producer
  module (`dateline.py`) — the oracle family's standing
  producer-as-ground-truth pattern, stated in the oracles' own
  docstrings — so the oracles are not a second literal copy of
  the arm table, and a producer-side weekday drift would move a
  payload and both oracles together. The independent literal
  check is this validator: it carries its own frozen date tables
  and weekday recomputation, and rejects a drifted payload even
  when the payload's protocol and fingerprint are self-consistent
  under the drift (executed in cycle 216: a payload rebuilt
  through a producer with a swapped weekday table validates
  before the drift and is rejected after it).

Both test oracles (cycle-27 and cycle-67) mirror the validator in
the same push, per the standing oracle blast-radius rule; after
adding the section, the full `identical` conjunction is audited for
silently omitted sections (the cycle-186 lesson).

## Budget (G2†)

Priced for the **amended** protocol, not the as-written design
(whose §(d) claimed "~6–8 treated arms" for its own 18):

| component | arms | generations at the legal maximum (24 candidates × 100 fixtures) |
|---|---|---|
| date arms (8 values × 3 formats) | 24 | 57,600 |
| no-date controls (1 per format) | 3 | 7,200 |
| weekday-lie | 1 | 2,400 |
| **treated total** | **28** | **67,200** |

Native is reused at zero cost. 67,200 is 2.8× the largest budget
shipped before (distraction, 24,000) — the price of the full
value × format crossing plus per-format controls, stated here and
in the CLI help with this decomposition, before any run. The
de-crossed alternative (weekday-ISO on the anchor ±1-day pair only)
was rejected: it would leave the lie's format present for only two
values and break the pair partition's uniformity — but it is named
here so the rejection is a decision, not an oversight.

## Pin blast radius (G12†)

Same push: `test_cycle44` source-file pin 52 → 53 (`dateline.py`);
cycle-179 dispatch tables gain the `--dateline` row and the
`_run_dateline_probe` helper; cycle-31/64/84 flag audits as they
apply; reporting's three touchpoints (compute, TOC tuple, body
concatenation) with the TOC entry immediately after `numform`,
before the post-hoc block; the reporting section returns empty on a
missing `dateline` key so sample-report regeneration stays
byte-identical (the sample runs none of the post-cycle-179 probes —
B M6); README/USAGE/VALIDATION/CHANGELOG updated with the measured
test count and coverage after the gate, never before.
