# Numeric surface-form probe — design (cycle 186, wave-7 rank 7)

Status: frozen pre-build design. Sources: wave-7 research §7 as
corrected by the cycle-186 researcher refresh (Reddy et al. 2026,
arXiv:2601.15251, and Yang et al.'s NUPA, arXiv:2411.03766, were
missed by §7 and narrow the claim); red-team A (F1–F17, DO NOT
BUILD AS SPECIFIED against the §7 sketch); red-team B's executed
cross-check (BUILDABLE AS AMENDED — three of A's remedies
disproved or replaced); the verifier's blast-radius brief. Where
the briefs disagree, B's executed evidence governs.

## What it measures, in one sentence

For a user whose prompts carry figures, `run --numform` shows —
per chat-template candidate, per task kind, per numeric surface
form — whether the checkpoint copies, compares, and adds
generated integers equally well when the same values are written
plain, comma-grouped, space-grouped, or in scientific notation, so
the user can choose a normalisation for numbers in their own
prompts. Monday-morning decision: which written form to normalise
figures to before rendering, for this checkpoint, per task kind.

Scoped claim (the combination only): numeric surface form ×
hand-rendered chat-template family, on a fixed local checkpoint,
over generated fixtures with form-matched expected answers,
per-candidate paired form contrasts, and the tokenizer's digit
grouping measured at run time as a grouping variable only, was
not found in the work surveyed in the cycle-186 uniqueness pass
(LOGGED in docs/RESEARCH_LOG.md). Format variation over fixed
arithmetic is published (Reddy et al. 2026, including a
formatted-input → formatted-output condition); representation ×
task benchmarks including scientific notation are published
(NUPA); comma-enforced rechunking is published (Singh & Strouse
2024, arXiv:2402.14903). Per-form *selection* was part of the
§7 sketch and is deliberately NOT in v1 (red-team B, GB5): over
16 values per task an all-pass selection readout carries almost
no information per unit of validator surface, and "the winning
template changes under comma grouping" is exactly the overclaim
the design must not invite. If selection is ever added it takes
the `leading_analysis` form, never the hierarchy form.

## Domain and forms

Non-negative integers only. Negatives and fractions are excluded
and named (with integers only, no decimal marker exists, so a
comma can only be a grouping separator — the locale ambiguity
Reddy's F2 exposes cannot arise). Digit bands: 1–3, 4–6, 7–9,
10–12 digits. Values per task: 16 (4 drawn per band; a pair's
recorded band is its larger operand's band, so compare's
recorded per-band counts vary around 4 — read denominators
from a payload's own `by_band` counts, not from this line).
Operands never exceed 12 digits; a sum never exceeds 13.

Renderers are total functions of the integer value alone, never
of an operand string:

- plain(v) = str(v)
- comma(v): 3-3-3 grouping with "," (named 3-3-3 in the docs —
  Indian 3-2-2 grouping exists and is not this arm)
- space(v): the same grouping with U+202F (narrow no-break
  space, the SI grouping separator; ASCII space would collide
  with operand separation in the prompt text, and the choice is
  named in the CLI help)
- sci(v): 0 ↦ "0e0"; otherwise with d = str(v), exp = len(d)−1,
  mantissa digits = d with trailing zeros stripped, rendered
  first digit + ("." + rest if any) + "e" + str(exp). Lowercase
  e, no "+", mantissa in [1, 10) — NUPA's significand precedent.

Totality is proven by an exhaustive round-trip sweep (all values
0–2000 plus edge values, × 4 forms, render → parse → identity,
no scientific collisions) and by the regrouping/renormalisation
battery (999,999 + 1 → 1,000,000 / 1e6; 9.99e6 + 9.99e6 →
1.998e7; 1.25e6 + 1.25e6 → 2.5e6). Decimal appears nowhere in
generation, rendering, or parsing: its default context rounds
at 28 significant digits, its parser accepts underscores,
whitespace, NaN and non-ASCII digits, and its string forms are
not these forms (all executed by red-team B). Arithmetic is
Python int; parsing is a frozen ASCII grammar.

## Tasks and prompts

Three task kinds; each value set is drawn once per task and
rendered into all four forms (paired design). All four form arms
of one value share one fixture id (`nf-{task}-{index}`), so
`core.request_seed_for` pairs their request seeds. Frozen prompt
templates (verbatim in numform.py, SHA-256 in the protocol)
contain no digit and no worked example in any form — a single
example is a few-shot arm for one form only, and few-shot
recovery is the largest mitigation Reddy et al. measure. The
templates instruct the model to answer in the operands' form;
pass := correct value ∧ explicitly-requested form ∧ boundary
found, which is Reddy's formatted-output condition held constant
across arms — part of the task definition, not a confound that
varies with the treatment.

- copy: "Copy this number exactly, character for character:"
  + the operand on its own line. Expected: the operand string.
- add: "Add these two numbers and reply with only the sum,
  written in the same format as the numbers:" + one operand per
  line. Expected: render_form(a + b).
- compare: "Which of these two numbers is larger? Reply with
  only the word first or second." + one operand per line.
  Expected: the label `first` / `second`. The same-form rule is
  amended to exempt compare (the label is form-invariant, which
  makes compare the one arm where the scorer cannot be a form
  confound); the exemption is stated in the reporting section.
  Equal pairs are excluded by construction (a generator
  invariant, recorded); mixed-form pairs are out of scope (the
  mixed-notation cell is Yuchi et al. 2026's).

## Generation

Deterministic, SHA-256-derived (fixturefuzz pattern: parameters
from SHA-256 over (seed, task, band, slot, counter); sorted
iteration only; no `random.Random`, no `hash()`, no set order) —
the generated set is byte-identical across runs and
PYTHONHASHSEED values, proven by a two-hash-seed regeneration
test. Candidate operand pairs are drawn from the hash stream
and accepted when an independent digit-wise classifier confirms
the target stratum class (bounded retries):

- copy: one value per slot; slots 0–1 of each band are the
  round-value stratum (a power of ten; all-nines), labelled by
  the `round` stratum. The label is intrinsic — random draws
  exclude round values — so the validator can recompute it
  from the value alone. Pairs carry no round label (compare
  reorders its operands for presentation, so which operand
  was drawn round is not a record-level fact); round first
  operands in pair slots 0–1 remain a property of the
  generated set, bound by `generator_sha256`.
- compare: margin classes cycle by slot — different_length,
  same_length_early (first difference in the leading half),
  same_length_last_digit; larger-first vs larger-second
  alternates by slot parity.
- add: carry classes cycle by slot — none, some, full_chain —
  computed digit-wise with carry-in propagation.

A pair's band is the band of its larger operand's digit length.
Strata ride in every ground record; band/stratum breakdowns are
descriptive (below the contrast floor), and bands whose compared
forms are byte-identical (plain/comma/space coincide below 1000)
report `n_form_identical` and carry no headline.

## Stage mechanics

Mirror: hierarchy for synthetic-fixture *construction* only;
leading/distraction for everything else. Per-fixture
`evaluate()` (one call per candidate × task × form, ids shared
across forms), `cancel` and `max_context` threaded, per-cell
fixed-vocabulary details: `arm_context_exceeded` (rendered +
max_new_tokens > max_context), `answer_budget_exceeded`
(expected answer + boundary cannot fit max_new_tokens —
insurance at the frozen bands, stated as such),
`form_not_in_render` (an operand string does not appear
verbatim in the rendered prompt for that candidate — the check
is per candidate × cell, on the real render), and
`arm_evaluation_failed`. Stage-level: the frozen templates are
screened with `audit_special_token_literals` before any
generation (fail closed: `numform_template_screen_failed`);
cancellation returns `available: False`, detail `cancelled`.
The CLI wrapper consults neither the search entries nor the
primary decision (there is no native arm and no dependence on
the user's fixtures), honours the run's cancel event, and never
raises: any failure degrades to `numform_unavailable`. The
section is informational and never touches selection.

Cost: candidates × 3 tasks × 16 values × 4 forms generations —
4,608 at the legal maximum of 24 candidates — stated in the CLI
help in those words. Caps frozen in numform.py and carried in
the protocol: GENERATION_BUDGET 4608, MAX_GROUND_RECORDS 4608,
MAX_RECORDS_BYTES 12 MiB; exceeding the record budget fails the
stage closed (`numform_record_budget_exceeded`).

## Records and decomposition

Generated values are not user data, so records store the
expected side's canonical integer values in the clear
(operands, expected value) — and the validator does not merely
trust them: it regenerates the whole set from the protocol's
seed and compares every record's construction against the
regenerated set (cycle 187; before that, the validator hashed
whatever records were present, which vouched for a binding it
never checked). The answer side is stored differently: a cell
carries the answer's digest and the value the stage parsed
from it. The decomposition below is recomputed from that
stage-attested parsed value — the parse itself is a stage
act, pinned by the stage's own tests, not re-derivable from
a digest (the grammar deliberately accepts non-canonical
costume, so no rendering check could re-derive it either).
Storing the answer text so the validator can re-parse it is
scheduled work (see Honesty items); it is a payload-format
change, not a wording fix. A hash over a ≤12-digit expected
value would be brute-force-invertible anyway, so digest-only
expected-side storage would buy no privacy either. Records
also carry `answer_digest`, `boundary_found`, `input_tokens`,
and the expected answer's token count. Records ride in the
public results.json regardless of `--include-evidence`, like
every probe's ground records — stated here so the diff
contract is explicit; the data is synthetic.

Per-cell decomposition (recomputed by the validator from the
stage-attested parsed value):

- copy / add: `exact_pass` (verdict true); else
  `value_pass_form_fail` when the parsed answer value equals
  the expected value (reported split by boundary_found); else
  `value_fail`. For copy, value_pass_form_fail is a failure of
  the task, not a consolation — copying is form fidelity; the
  class exists there to keep "right digits, wrong costume"
  distinguishable from "wrong digits", and the section says so.
- compare: `exact_pass`; `wrong_label` (the answer is the other
  label — groundable from digests alone, since the label set
  has two known strings); `non_label` (anything else).

The answer parser is grammar-gated with digit/exponent caps
enforced BEFORE any conversion (total digits ≤ 20, exponent ≤
999 with ≤ 3 exponent digits), applied to model answers as well
as expected strings: an answer like `1e999999999` classifies
`value_fail`, never allocates (the live Decimal/DoS surface is
answer-side, red-team B GB3). The parser accepts the four form
grammars plus plain digit strings (leading zeros parse), and
for scientific a capital E or + exponent *when parsing only* —
costume differences belong in value_pass_form_fail, not in
value_fail. Wrong-width grouping (1,23,456) does not parse.

Headline contrast: per candidate × task, for each of the six
form pairs, paired discordant counts over the common-completer
population (values completed in both forms): n_paired,
n_pass_a_only, n_pass_b_only, n_both_pass, n_both_fail, McNemar
exact p over the discordants (diagnostic only, never
thresholded), and a classification from the total function of
(n_paired, b, c): paired < MIN_PAIRED_EVALUABLE (6) →
`insufficient`; b = c = 0 → `no_difference_observed`;
b ≥ MIN_DISCORDANT (2) ∧ b > c → `{form_a}_advantage`;
c ≥ 2 ∧ c > b → `{form_b}_advantage`; b = c ≥ 2 →
`mixed_discordance`; otherwise `below_threshold`. Marginal
per-form rates are reported beside the contrasts with their
denominators, never instead of them, and no rate or contrast
is ever pooled across tasks (N3, enforced structurally: the
payload has no cross-task total for the validator to certify).

## Tokenizer grouping

One run attaches one backend, so grouping takes exactly one
value per run: it is a run-level descriptive label and a
cross-run diff variable, never a within-run correlation. The
classification is measured, never inferred from a family name
(§7's "Llama/Gemma-style single-digit" was wrong for Llama 3+,
which groups by 3 — TokSuite; Reddy Table 5). A frozen,
count-based probe battery over `backend.encode`: plain
renderings of digit strings of every length 1–12 (sequential,
all-same-digit, zero-heavy) plus every form of a frozen probe
value. Decision rule (frozen): `single_digit` iff every
all-digit probe's token count equals its digit count;
`chunked_up_to_3` iff every all-digit probe of length ≥ 2 has
token count exactly ceil(len/3) and length-1 probes count 1;
`irregular_or_mixed` iff every count lies between those two
predictions without matching either rule; otherwise `unknown`
(including any encode failure). Probe evidence (string → token
count) is stored in the payload; the diff validator re-runs
the decision rule over the stored evidence — a consistency
check, labelled as such, since the validator has no tokenizer.
`unknown` never blocks the per-form rates and always blocks
grouping language in the section. §7's N2 is demoted from
invariant to measured diagnostic: the payload reports the
actual per-form operand token counts and deltas from plain;
separator merges (a comma fusing with adjacent digits) are
data, not classification errors.

## Protocol and diff

Protocol object: version, tasks, forms, bands, values_per_task,
floors (6 / 2), budgets, space codepoint (U+202F), scientific
grammar id, prompt templates + templates_sha256, renderer spec
id, `generator_sha256` over the canonical generated set
(operands and expected values per task/value/form), seed,
backend fingerprint; `fingerprint` = SHA-256 over the canonical
object. Generator drift ⇒ not comparable.

`diff_numform` recomputes from the records with independent
code (no imports from numform.py): every form rendering and
expected string (N1: all forms of a value parse to the same
value), every aggregate, the decomposition, the paired
contrasts, and the protocol fingerprints; it regenerates the
generated set from the protocol's seed with its own mirror of
the generator and compares every record's construction
against it, checks that candidates agree on the expected
answer's token count, and re-runs the grouping decision over
stored evidence (consistency) with the stored token
means/deltas held to the producer's arithmetic relation (each
delta is its form's mean minus the plain mean, up to float
summation order). Grouping
drift between runs is reported and flags the form-contrast
comparison; it does not by itself make sections incomparable.
Impossible payloads are rejected: pass > completed, rate ≠
count/denominator, duplicate (task, value, form) record keys,
a cell present iff its detail is empty, records over the caps,
malformed digests, per-record construction fields (band,
strata, operands, expected) inconsistent across candidates,
records that are not the protocol seed's generated set, and
any pooled cross-task total present at all.

## Honesty items

- Generated integer tasks, isolated operands, this exact prompt
  wording, zero-shot, this checkpoint: the section states the
  scope in the quantity's definition, and per-task readings
  only — no pooled "best form". A form × template interaction
  is a per-candidate contrast, never a pooled marginal.
- A low pass rate in a form means "cannot do the task AND
  return the answer in this form, under byte-exact scoring" —
  the decomposition is what licenses any narrower arithmetic
  reading, and only from the value_fail class.
- The pre-registered null: all paired deltas at
  no_difference_observed / below_threshold across forms — a
  publishable "normalisation does not matter here" result.
- Compare is a verbalisation-under-form measurement (Yuchi et
  al.: probed magnitude knowledge exceeds verbalised accuracy);
  grouping explains nothing by itself within a run, and
  cross-tokenizer claims require cross-run diffs of different
  checkpoints, which confound weights with tokenizer.
- The decomposition's answer side rests on the stage's
  parse (see Records and decomposition). Scheduled structural
  fix, recorded here so it is not lost: store the answer text
  on numform records and bind it with
  `answer_digest == sha256(answer_text)`, so the validator can
  re-parse the answer itself instead of trusting the stored
  parsed value. It changes the payload's exact record key set
  (stage, both diff-oracle mirrors, and the records-bytes
  accounting move together), which is why it is its own cycle
  and not part of the cycle-187 wording fix.
- Named future work, not in v1: spelled-out number words,
  locale forms (decimal-comma, 3-2-2 grouping), other space
  codepoints, mixed-form fixtures, fractional operands,
  negatives, per-form selection.
