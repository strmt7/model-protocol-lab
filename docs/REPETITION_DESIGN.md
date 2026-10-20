# Prompt repetition × template — design

`model-protocol-lab run --repeat` measures whether repeating a
request inside the turn changes verdicts when the chat template is
the variable. Leviathan, Kalman & Matias (arXiv:2512.14982) show
that prompt repetition improves non-reasoning LLMs — measured on
API models, each rendered with its provider's own chat template,
with a padding ablation showing random-token padding does not
reproduce the effect. This probe asks the lab's version of the
question: same checkpoint, hand-rendered foreign formats, exact
token sequences, a conservative abstention rule — does repetition
change who passes, and does the winner survive it?

Monday-morning use: you are about to ship a prompt that asks
twice (or are wondering whether to). The probe tells you, per
candidate template on your own checkpoint, whether repeating the
request changes verdicts once prompt length is held constant —
and whether the template you selected still wins under
repetition.

## Arms

Three arms per (candidate, fixture), all over the search split
only:

- **x1** — the recorded native search observations, reused. The
  treated arms are compared against the run that actually
  happened, not a re-execution; x1 generates zero model calls.
- **x2** — the fixture's last user message with its content
  repeated verbatim inside the same turn:
  `content + "\n" + content`. The surgery happens on the message
  content before render (the deliforge precedent): a post-render
  append would land after the assistant prefix, in the
  assistant's mouth.
- **pad** — the same fixture with neutral filler appended after
  the same joiner, sized so the rendered pad prompt has exactly
  the x2 prompt's token length for that candidate and fixture.

The headline contrast is **x2 − pad** (`repetition_effect`):
both arms are longer than x1 by the same number of tokens, so a
difference is not explained by length alone. What it does not
separate is repetition-specific attention from content position —
the design names that limit wherever the number is reported.
`pad − x1` is the `length_effect`; `x2 − x1` is the `raw_gain`,
labelled joiner-inclusive because the joiner itself (frozen as
`"\n"` across all arms) is part of what x2 adds over x1.

## Claims

- **R1′ (construction).** The x2 arm's render is the x1 render
  with the last user message's content replaced by the doubled
  content — a string-level claim, verified by rebuilding the
  render from the arm fixture, never by search-and-replace
  (identical first/last user contents make replace-based checks
  certify a wrongly-doubled first turn). Whether the *token*
  construction is exact is measured, not assumed: per
  (candidate, fixture) the payload records `id_exact` (the x2
  content region is exactly region + joiner region + region, by
  sentinel-free render affixes) or `text_level` with the seam
  token ids. A merge-y tokenizer degrades the measurement to
  text_level; it does not silently inherit an id-level claim.
- **R2′ (attribution).** A verdict change between x1 and x2 is a
  prompt-format effect on this checkpoint, attributable to the
  treated prompt as rendered — the lab's standing asymmetry
  applies: it proves the failure was prompt-fixable for fixtures
  that flip fail→pass, and proves nothing about weights for
  fixtures that never pass.
- **R3′ (length control).** The pad arm is token-length-matched
  per (candidate, fixture) on the rendered ids, by a bounded
  search (≤ 512 probes) over whole filler units plus at most one
  truncated suffix. Pairs that cannot be matched exactly —
  nonpositive delta with no positive-filler match, overshoot
  (the empty pad already renders longer than x2), search
  exhausted — are recorded `pad_unmatchable`, never shipped
  mismatched. A match whose filler contributes zero tokens is
  degenerate and does not count: the search continues.
- **R4′ (provenance).** Provenance is claimed for x1 only: the
  recorded observations, the run's protocol digest, and the
  per-fixture ground records. The treated arms rest on the
  frozen protocol (below) plus those records; the payload
  carries counts, ids, booleans and digests only — no fixture
  content, no filler text, no answer text.
- **R5′ (selection).** Each arm's winner is re-derived under the
  conservative rule over that arm's completed, non-vacuous
  fixtures — the same population as the gain denominators. The
  readout is *withheld* (`decision: "unavailable"`, flip `None`)
  whenever any candidate is arm-unavailable or the completed
  sets differ across candidates: a flip the denominators cannot
  explain is a phantom, and the scorer-lens withholding
  discipline applies. The x1 arm never reports a flip; it is the
  recorded run by construction. Run decisions are compared after
  vocabulary mapping (`compatible_on_fixture_set` and
  `held_out_failed` both map to `("selected", winner)`).

## Denominators and abstentions

Every contrast is paired: a fixture enters a contrast's
denominator only when both compared arms completed it and it is
not vacuous. `n_paired` travels with every rate; rates are
`None` (never 0.0) when nothing is paired, and
`evaluable_mismatch` flags contrasts whose arms completed
different fixture sets. Because pad sizing can fail per pair,
the payload discloses `n_pad_evaluable` and
`pad_evaluable_share` per candidate, plus the delta sign counts
— a repetition effect measured on a shrinking pad subset says
so on its face. Contrasts are also reported excluding
pre-repeated fixtures (below); that readout is a subpopulation
estimate and is labelled as such.

Vacuous fixtures (empty or whitespace-only target content) are
excluded from every denominator and selection set by content
predicate — under the frozen joiner an empty content's x2 render
differs from x1 by the joiner alone, so a render comparison
would certify a treatment that did not happen. They are counted
(`n_vacuous`), never generated.

Degradation is per fixture, never per cell: a treated arm whose
rendered length plus the generation cap exceeds the run's
context budget (`max_context`, the same value the CLI passes to
the backend constructor; equality is measured, one token over
degrades) records `arm_context_exceeded` for that fixture alone.
A backend generation failure records `arm_evaluation_failed`.
A fixture whose expected answer occurs inside the filler unit
(or whose candidate boundary text occurs in it) is
`pad_screened`: its pad arm would hand the model its answer or
its terminator, so the fixture leaves the pad denominators
visibly rather than poisoning them. If the filler unit itself
ever collided with a candidate's special-token inventory, the
whole stage would fail closed (`filler_screen_failed`); the
shipped unit is clean against the shipped catalog.

## Protocol (frozen, v1)

Joiner `"\n"`; filler unit one frozen declarative sentence (no
digits, no instruction text), screened per run as above;
pre-repetition detector v1; sizing rule
`whole-units-plus-one-truncated-suffix-v1`; at most 2400 ground
records (candidate × fixture, arms folded into one record) —
the stage returns `repetition_cell_budget_exceeded` before any
generation past the cap. The payload's protocol block carries a
SHA-256 fingerprint over the canonical protocol table including
the fixture-set hash and the backend fingerprint; any component
change is a protocol version bump, and `diff-reports` reports
protocol / fixture-set / backend drift with `comparable: false`.

The detector is exact and mechanical. Let c be the target
content after whitespace collapse (case-sensitive). Flag iff
(i) c = t + sep + t for some nonempty t and sep ∈ {"", " "}, or
(ii) c contains two immediately adjacent copies of a substring
of length ≥ 8. Consequences worth stating plainly: `"ha ha ha
ha"` is flagged (t = "ha ha"); a repeated sentence whose second
copy is identical *including its trailing period* is flagged
(t = "the cat sat.", sep = " "); a near-repeat that differs
anywhere inside a copy is **not** flagged, and neither is a
repeat whose only decomposition needs a longer separator.
Excluding-flagged contrasts are reported, but the pooled
readout remains the headline: the detector screens for
fixtures the treatment cannot change, it does not certify the
rest as repetition-free.

An echo flag marks fixtures whose expected answer occurs in the
target content: under x2 the model is handed a second copy of
the answer span, so echo-completion stays separable from
re-reading in the records even though v1 reports the flag as a
count only.

## Cost

Treated arms cost 2 generations per (candidate, fixture) with
x1 reused: at the run caps (24 candidates × 100 fixtures) that
is 4,800 generations, on top of the native search. This lab's
decoder recomputes the full prefix with no KV cache, so a
generation on an L-token prompt costs M·L + M(M−1)/2 token
forwards for M generated tokens; the stage therefore costs
roughly 2.4–4× one native search pass over the same fixtures,
depending on the content share of the prompts. The paper's
"no latency increase" claim does not transfer and is not made.

## Cancellation and gating

The run's cancel event is threaded dispatch → wrapper → stage →
`evaluate`; a cancelled stage returns
`{"available": false, "detail": "cancelled"}` with no partial
aggregates. If the run's decision is `failed` or `cancelled`,
or its search entries do not cover every candidate, the wrapper
gates the probe to zero generations
(`native_search_incomplete`): treated arms with no sound native
baseline are not measured. The probe is search-split only by
construction — held-out fixtures exist as recorded x1 for the
selected candidate alone, and a treated held-out arm would pair
it against candidates that were never selected; no held-out
fixture id appears in any ground record.

## Validation

The payload's ground records are the contract: `diff-reports`
recomputes every aggregate from them with independent code —
contrasts, mismatch flags, boundary rates, pad shares, delta
signs, the per-arm selection readout with its withholding rule,
flips against the mapped primary, think-group means in both
per-candidate and render-deduplicated form, and the protocol
fingerprint from its own components and the record fixture
ids. A payload whose numbers its records do not support does
not validate. Recompute is internal consistency: it detects
aggregates inconsistent with their records, not fabricated
records — the standing limitation of every ground layer in this
lab.
