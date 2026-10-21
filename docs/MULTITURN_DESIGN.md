# Multi-turn drift design (cycle 189)

Wave-8 rank #2. The incorporation policy below (P1–P3) is adopted
from the wave-8 research brief (`~/workspace/agents_out/wave8/
research.md` §2 #2) before any code, because cycle 179's closure
recorded the policy — not the fixture schema — as the blocker:
synthetic fixtures are built in memory on the hierarchy/numform
precedent, and `core.Fixture`'s single `expected` is enough once
the policy says which turn is scored.

This is v2 of the design. v1 was attacked by red-team A
(`~/workspace/agents_out/wave8/cycle189/redteam_a.md`, verdict
DO NOT BUILD AS SPECIFIED: four design-fatal items) and
cross-checked by red-team B (`redteam_b.md` in the same
directory), which confirmed all four fatal items, downgraded
two findings, refuted two of A's own amendments, and added one
finding A missed. The coordinator adjudication adopted B's
surviving must-fix list wholesale; §Adjudication records what
changed and why. No code was written against v1.

## Utility

A practitioner shipping a multi-turn assistant learns, per
template family, whether the winner chosen on single-exchange
fixtures still wins when the model's own earlier replies become
the history — and whether verdicts at depth 2 and depth 3
differ — instead of discovering in production that the template
only won on conversations that never actually happened. (Each
depth is its own generated task set, so a cross-depth
comparison is confounded by task length; the probe reports
per-depth verdicts, not a depth-onset curve.) Opt-in:
`model-protocol-lab run --multiturn`.

## The incorporation policy (P1–P3, frozen)

- **P1 (what a turn is).** A conversation is a frozen sequence
  of user shards. The model's reply at each intermediate turn
  is generated greedily under the candidate's own rendering,
  cut at the candidate's boundary exactly as in native
  evaluation, and the **cut text** (never the raw
  over-generation) is inserted verbatim as the assistant turn
  of the next prompt. In `core.evaluate` the cut text is
  `Observation.answer` (output up to the first boundary); the
  multiturn stage builds each next turn's `Fixture` from it
  unchanged. Scope of the claim (v2, per A9): the history is
  the cut text **re-rendered under the candidate's own
  `Format`** — the lab's construction, not a claim about what
  any vendor template or real deployment would send. For
  formats whose generation open suffix differs from the
  history assistant prefix (62 of 118 mutate candidates, 15 of
  29 base families — e.g. the thinking variants, where text
  generated inside an opened think block is re-homed outside
  it by the history render), the recorded conversation is the
  lab's uniform construction, disclosed here and in the
  section's honesty text.
- **P2 (what is scored).** Only the **final** turn's answer,
  against the fixture's single `expected`, under the unchanged
  exact-match + boundary conjunction. Intermediate replies are
  never scored; they are context, exactly as in a real
  conversation. This dissolves the "per-turn expectations"
  problem by refusing to invent them. Per-turn `Fixture`s the
  stage builds carry the fixture's final `expected`;
  `evaluate` computes `exact_match` against it on every turn
  and the stage ignores intermediate `Observation.exact_match`
  (v2, per A8 — a builder must not invent per-turn
  expectations).
- **P3 (selection).** The probe is informational and never
  touches the native selection. Per-depth selection readouts
  (for the `full` and `sharded` arms; the scripted arm is a
  control and gets none) re-apply the conservative rule to
  final-turn verdicts and are withheld (null) when any
  candidate's completed set at that depth differs from the
  defined fixture set or completed sets differ across
  candidates (the dateline wording, v2 per A13).

## Construction (synthetic, generated, numform discipline)

Fixtures are generated inside the stage from the run seed; no
user fixtures are read and no file schema is involved. Values
derive from SHA-256, never `random.Random`, so the same seed
regenerates the identical fixture set across runs and
`PYTHONHASHSEED` values.

**Derivation (frozen, v2 per A8).**
`_derive(seed, family, depth, index, slot, counter)` returns
`int.from_bytes(sha256(material)[:8], "big")` where
`material = "multiturn-v1|{seed}|{family}|{depth}|{index}|{slot}|{counter}"`
(the numform `_derive` construction, with depth deliberately
in the material: depths are independent task sets and no
nesting or lineage is claimed between them).

Two task families, each a deterministic function of all shards.
The question **rides the final shard** (appended to it), so
depth is the number of entries/shelves — there is no
replace-reading and no own-turn reading (v2 per A8/S4).

- **ledger** — shard *i* (non-final) is the text
  `Ledger entry: {v}.`; the final shard is
  `Ledger entry: {v}. What is the ledger total?` Addend for
  slot *i*: `u = _derive(seed, "ledger", depth, index, i,
  counter)`; magnitude `1 + u % 999`; value `+magnitude` when
  `(u >> 32) % 2 == 0`, else `-magnitude` (nonzero, ±1..±999).
  Entries render as `str(int)` — entry style and expected
  style coincide by construction (v2 per B's A8 correction: a
  `+42` entry style against a `str(int)` expected would
  manufacture the surface-form confound numform exists to
  isolate). The expected answer is `str(sum)`. **Entry-equality
  screen (A14 as rescoped by B):** if `str(sum)` equals any
  stated entry string in the same fixture, the whole fixture
  is redrawn with `counter + 1` until it does not. (Substring
  screening is *not* used: it is the wrong predicate — a shelf
  label is a substring of its own shard text in 100% of
  fixtures.)
- **shelf** — shard *i* (non-final) is
  `Shelf {label} holds {n} units.`; the final shard is
  `Shelf {label} holds {n} units. Which shelf holds the most
  units? Answer with the shelf label.` Labels are `A`, `B`,
  `C` in shard order. Value for slot *i*:
  `1 + _derive(seed, "shelf", depth, index, i, counter) % 99`
  (range 1..99). Values are pairwise distinct per fixture: if
  a drawn value equals an earlier slot's value in the same
  fixture, that slot is redrawn with `counter + 1` until
  distinct, so the argmax is unique. The expected answer is
  the argmax label. Shelf needs no leakage screen (B).

Depths are 2 and 3 (the shard count). Fixtures per family per
depth: 12, so 48 fixtures total. Fixture ids are
`mt-{family}-{depth}-{index:02d}`. Every `Fixture` the stage
builds (per-turn included) carries group `"multiturn"`. The
`full` arm joins shards with the frozen joiner `"\n\n"` (the
dateline `JOINER` precedent). The scripted reply is the frozen
text `Noted.`. Shard templates, the question strings, the
joiner, and the scripted reply are frozen protocol text,
screened before any generation (the numform template screen:
`audit_special_token_literals` against the candidate formats,
no boundary-text containment).

Ground records store shard texts and generated integers in
the clear — synthetic data, and the diff validator must be
able to recompute from them (numform precedent).

## Arms

- **full** — all shards concatenated into one user turn with
  the frozen joiner (single-turn control; Laban's Full
  condition). One generation.
- **sharded** at depth d — the same shards delivered one per
  user turn, with the model's own boundary-cut replies as the
  assistant turns between them (P1). d generations; only the
  last is scored (P2).
- **scripted** at depth d — the same sharding, but every
  intermediate assistant turn is the one frozen scripted reply
  (`Noted.`), identical for every candidate, fixture, and
  turn. One generation (the final turn). The sharded-vs-scripted
  contrast measures a **bundle** — authorship, content, and
  history length move together, because a self-generated reply
  carries computed state and premature answers (rule 6) at
  whatever length the model produced, and content/length are
  the mechanism of self-history, not confounders separable
  from it (v2 per A4 as reduced by B: no length-matched arm —
  sizing the control to each candidate's own replies would
  candidate-depend the control and change the estimand). The
  design, CLI help, and readouts therefore never say the
  contrast "isolates" authorship; they report the discordant
  pair (§Readouts) so churn and silence are distinguishable.

## Stage mechanics (pre-registered edge rules)

The stage is a sequential loop — the first stage in the lab
where generation k's output builds generation k+1's prompt.
Rules, fixed before code:

1. **Unbounded intermediate.** If an intermediate generation
   finds no boundary, the full generated text (decoded,
   budget-capped — exactly `Observation.answer` when
   `boundary_found` is false) becomes the history turn, and
   the cell is flagged `intermediate_unbounded`. Truncating
   or dropping the turn would manufacture a different
   conversation per candidate.
2. **Empty intermediate.** If an intermediate cut text is the
   empty string, it is inserted as an empty assistant turn
   (legal in every catalog format — verified) and the cell is
   flagged `intermediate_empty` (v2 per A9/S1). An empty turn
   is a no-reply turn, a distinct phenomenon from an unbounded
   one, and its rendering is family-shaped (e.g. Mistral's
   empty assistant turn contributes only its turn end), so it
   is named rather than left derivable-only.
3. **History screen (v2 per A5).** Before an intermediate
   reply is incorporated as history, its text is screened
   with `audit_special_token_literals` against the candidate
   format's own special-token inventory (the audit the shipped
   stages apply to frozen text). Generated text can contain
   special-token literals that re-encode to real special-token
   ids — a forged turn boundary inside history is token-real,
   not inert text. On a screen hit the sharded cell ends with
   detail `arm_screened` (the dateline detail precedent), the
   cell leaves the paired set, and screened cells are counted
   per candidate (§Readouts). The final-turn answer is never
   incorporated, so it is not screened; the scripted history
   is frozen text, screened once with the protocol text.
4. **Context growth.** Before each turn, if the rendered
   prompt's token count plus `max_new_tokens` exceeds
   `max_context`, that arm's cell for that fixture ends with
   detail `arm_context_exceeded` and leaves the completed set
   at that depth. A completed-set event, reported — never a
   crash, never a silent truncation of the conversation.
5. **Cancellation, with precedence (v2 per S3).** The run's
   cancel event is threaded into every `evaluate` call and is
   checked **immediately after every `evaluate` return,
   before the evaluation-failure mapping (rule 6)** — a cancel
   observed during a turn must not be mislabelled an
   evaluation failure. A cancelled stage returns the
   fail-closed payload (`available: False`,
   `detail: "cancelled"`), never partial candidates.
6. **Evaluation failure.** A turn whose `evaluate` report is
   not aggregate-eligible ends the cell with detail
   `arm_evaluation_failed`.
7. **Request seeds (v2 pins per A12).** Each (fixture, arm,
   turn) evaluates under a distinct case id —
   `{fixture_id}-full`, `{fixture_id}-scripted`, and
   `{fixture_id}-sharded-t{k}` for k = 1..depth (turn `depth`
   is the final turn) — so `core.request_seed_for` gives every
   generation its own deterministic request seed. Final-turn
   seeds are deliberately unpaired across arms (contrast
   numform's paired discipline); the real backend does not use
   the seed for selection, and tests must not assume pairing.
8. **Premature answers are signal.** An intermediate reply may
   contain the final answer early (Laban's anchoring
   mechanism). Nothing detects, penalises, or repairs this;
   P2 scores only the final turn.

Attrition is arm-asymmetric by construction (a sharded cell
must survive d sequential evaluations; the others one), and
the exiting cells are treatment-correlated (long or unbounded
intermediates), so every paired denominator is reported beside
per-arm completion counts with a detail breakdown (v2 per
A11).

## Records

One record per (candidate, fixture), carrying everything the
diff validator recomputes from:

- `fixture_id`, `family`, `depth`, `shards` (texts), `values`
  (the generated integers), `expected`.
- Per arm (`full`, `sharded`, `scripted`): a cell or a
  `detail` string from the house vocabulary
  (`arm_context_exceeded`, `arm_evaluation_failed`,
  `arm_screened`). A cell is `verdict`, `boundary_found`,
  `boundary_text_match` (v2 per A6 — the only diagnostic
  separating "model failed" from "boundary text emitted under
  a different tokenisation"), `input_tokens` (final turn),
  `answer_digest` (SHA-256 of the final answer text), and for
  `sharded` also `intermediate_unbounded`,
  `intermediate_empty`, and `turns`: one entry per turn
  (length == depth), each `input_tokens`, `answer_digest`,
  `boundary_found`. The last `turns` entry **is** the final
  turn and must agree with the cell on digest, boundary, and
  `input_tokens` (validator-checked identity, v2 per A6);
  `intermediate_unbounded` ≡ any intermediate turn with
  `boundary_found` false, and `intermediate_empty` ≡ any
  intermediate turn whose digest is SHA-256 of `""`
  (validator-checked identities, v2 per A6). The scripted arm
  records no `turns` (its intermediates are frozen text;
  there is nothing to attest).
- `ground_sha256` over the canonical generated fixture rows
  binds the payload to the generator (numform
  `generator_sha256` precedent).

The protocol object carries: version, families, depths,
fixtures per family per depth, the shard templates, question
strings, joiner, and scripted reply (fingerprinted via a
template digest, numform precedent), the floors,
`GENERATION_BUDGET`, `MAX_GROUND_RECORDS`,
`MAX_RECORDS_BYTES`, the seed, and the backend fingerprint.
Protocol drift in any of these makes two payloads not
comparable.

## Readouts

Per candidate, computed by `multiturn.aggregate_candidate`
and recomputed by the diff validator. Rates always travel
with denominators; counts are shipped, never rates alone.

Per depth d ∈ {2, 3}:

- **Paired denominator** = fixtures of depth d completed in
  all three arms; per-arm final-turn pass rates over that
  denominator. Per-arm completion counts with detail
  breakdowns (including `n_screened`) are reported beside it.
- **Delivery 2×2** (`full` vs `sharded`) over the paired set:
  drift (pass `full` ∧ fail `sharded`), gains (fail `full` ∧
  pass `sharded`), both-pass, both-fail; McNemar exact p as a
  diagnostic only. The `full` vs `scripted` 2×2 likewise
  (drift_scripted, gains_scripted, both-pass, both-fail).
- **Self-history pair** over the paired full-passers:
  `sh_only` (pass `scripted` ∧ fail `sharded`), `sc_only`
  (fail `scripted` ∧ pass `sharded`), both-treated-pass,
  both-treated-fail. The pair is the headline; `net_excess =
  sh_only - sc_only` is reported beside it, never alone, with
  a McNemar exact p as a diagnostic (v2 per A3 — net-only
  reporting is the cycle-185 acquittal construction: churn
  (5, 5) and silence (0, 0) have the same net).
- **Per-family counts beside pooled** (v2 per B-NEW-1):
  drift, gains, drift_scripted, `sh_only`, `sc_only` per
  family per depth. The families have different answer
  spaces and baseline difficulties; pooling is stated here
  explicitly, per-family counts ship beside every pooled
  count, and the classification is pooled (its inputs are
  visible per family).
- **Selection readouts** (P3, §policy) for `full` and
  `sharded` at each depth, or null under the withholding rule.

**Classification** per candidate — a total function of the
per-depth counts with floors `MIN_PAIRED_EVALUABLE = 6` and
`MIN_DISCORDANT = 2` (house constants). A depth is *evaluable*
when its paired denominator ≥ 6. In order (v2 per A2 — the
sibling classifiers' below-threshold outcome restored, and
the strong label gated on at-floor paired evidence at a
single depth):

1. `insufficient_evidence` — no depth is evaluable.
2. `self_history_drift` — at some evaluable depth,
   `sh_only ≥ 2` and `sh_only > sc_only`.
3. `framing_only` — not (2), and at some evaluable depth
   max(drift, gains, drift_scripted, gains_scripted) ≥ 2:
   delivery/framing discordance at floor with no
   self-history signal at floor.
4. `below_threshold` — not (2) or (3), and any discordant
   count (drift, gains, drift_scripted, gains_scripted,
   `sh_only`, `sc_only`) > 0 at any evaluable depth.
5. `no_drift` — no discordance in either direction at any
   evaluable depth.

There is **no cross-depth recovery counter** (v1's was a
category error — A1: the depths are different generated
tasks, so passing at depth 3 after failing at depth 2 is not
recovery from anything, and its numerator was not a subset of
its stated denominator). Cross-depth reporting is the two
per-depth blocks, full stop.

## Falsifiable claims (pre-registered)

- **F1 (per candidate).** If at every evaluable depth the
  self-history pair is (0, 0) while scripted-arm discordance
  exists, self-authorship adds nothing beyond the framing
  bundle for this candidate, and the claim "self-generated
  history is the active ingredient" fails for this candidate
  and fixture set. The classification concludes per candidate
  (`framing_only` or `below_threshold`), never pooled across
  candidates (v2 quantifier fix per B).
- **F2.** If a candidate shows drift = gains = drift_scripted
  = gains_scripted = 0 at every depth, the claim "multi-turn
  delivery is a verdict-relevant surface under this template"
  is false for this candidate and set; the section says
  exactly that, with denominators (v2 per A10 — two-directional:
  drift 0 with gains 5 is a verdict-relevant surface and must
  not satisfy the null).
- **F3 (instrument).** Every derived number in the section is
  recomputable from the ground records. The binding that
  makes this true is stated, not implied: for every arm cell,
  `verdict ≡ (answer_digest == sha256(expected)) ∧
  boundary_found`, with `expected` from the mirror-regenerated
  fixture (v2 per A6). One honest exception, disclosed rather
  than papered over: the correspondence between the recorded
  intermediate digests and the history that was actually
  rendered is **stage-attested** (the DISTRACTION_DESIGN
  honesty pattern) — the records bind each turn's digest and
  the validator checks every identity among recorded fields,
  but no record can prove the stage incorporated the text it
  recorded. A payload whose aggregates disagree with a
  recomputation is rejected by `diff-reports`.

## Diff validator

`diff_multiturn` recomputes everything (the wave-7 F6
standard): it regenerates the fixture set from the protocol
seed through an independent stdlib mirror of the generator
(the cycle-187 numform fix — `generator_sha256` alone would
be tautological), checks every record's shards/values/
expected against the regenerated set, re-derives every cell
verdict from its digest identity, checks the `turns`
identities (§Records), recomputes pass rates, the 2×2s, the
self-history pair, per-family counts, selection readouts,
and the classification from the cells, and rejects on any
disagreement. Protocol drift (version, seed-derived fixture
set, scripted reply, templates) makes two payloads not
comparable.

## Determinism scope

The chain is greedy, so a conversation is a deterministic
function of (checkpoint, candidate, fixture). Multiturn cells
are **outside** the determinism self-audit's scope: `verify
--determinism` replays recorded *search* observations only,
and this probe's payload says so rather than implying
coverage.

## Budget

Per candidate: depth-2 fixtures (24) cost 1 + 2 + 1 = 4
generations each, depth-3 fixtures (24) cost 1 + 3 + 1 = 5,
so 216 generations per candidate; at the tree's 24-candidate
cap (`run_experiment`: "Supply between 1 and 24 candidates"),
`GENERATION_BUDGET = 5184` (v2 per A7 — v1's 25-candidate /
5,400 figure was wrong). `MAX_GROUND_RECORDS = 1152`
(24 candidates × 48 records — generations and records are
different quantities here and get separate constants, v2 per
S2) and `MAX_RECORDS_BYTES = 12 * 1024 * 1024` (measured need
≈ 1.2 MB at the legal maximum); caps fail closed before
generation, as in numform. The CLI help states the worst case
with the forwards formula before the run: generations =
candidates × (24×4 + 24×5), and prompt cost grows every turn
because each sharded turn re-renders the whole history, so
token cost is superlinear in depth even though the generation
count is linear.

## Honesty

The generated tasks are a probe instrument, not a benchmark:
they measure whether verdicts survive sharded delivery on
tasks whose entire information content fits in three short
shards. Nothing here estimates performance on real user
conversations, which are underspecified in ways a frozen
shard sequence is not (Laban's simulations use a user
simulator for exactly that reason). Two further disclosures
travel with the section: the P1 history construction is the
lab's re-rendering, with the open-suffix asymmetry for
thinking formats (§policy); and the sharded-vs-scripted
contrast is a bundle (authorship + content + length) over a
minimal neutral history, reported as a discordant pair, not
an isolated authorship effect. The probe's claim is the
lab's claim throughout: holding weights fixed, the *template*
decides whether the winner survives the conversation shape.

## Adjudication (v1 → v2)

Red-team A: DO NOT BUILD AS SPECIFIED (A1–A14). Red-team B:
confirmed A1, A2, A3, A5, A7, A8, A9 (strengthened), A10,
A11, A12, A13; A4 partially (downgraded HIGH → MEDIUM, its
length-matched-arm amendment refuted — adopted: wording-only
fix, bundle named, "isolates" struck); A6 confirmed-
downgraded (HIGH → MEDIUM — the verdict is re-derivable from
v1's specified fields; adopted: the identity stated, lineage
disclosed stage-attested, `boundary_text_match` added);
A14 partially (phenomenon confirmed; its substring screen
refuted — it rejects the shelf family by construction;
adopted as rescoped: ledger-only entry-equality screen).
Supplementary items S1 (empty-turn render artefacts),
S2 (separate budget/record constants), S3 (cancel precedence
pin), S4 (third budget reading — closed by the
question-placement decision) all adopted. B-NEW-1 (family
pooling unstated) adopted: per-family counts beside pooled.
Fatal four fixed as specified by A and confirmed by B:
recovery deleted (A1), classification rebuilt with
`below_threshold` and single-depth gating (A2), history
screen pre-registered (A5), generator fully specified
(A8: question placement, derive material with depth, both
value mappings, `str(int)` entries per B's correction,
joiner, group, caps, protocol object, house vocabulary).

## Pin blast radius (coordinator checklist)

New pure module `multiturn.py` (54th source file →
test_cycle44 mypy pin in the same push); one stage in
`experiments.py`; one CLI flag `--multiturn` (cycle-179
dispatch e2e, cycle-31/64/84 audits, cycle-131 Namespace
contract); reporting.py's three touchpoints; one diff section
(test_cycle67's three elements + test_cycle27's stub); USAGE
§8.44 (next free — verify at build); README/VALIDATION/
CHANGELOG currency; sample report regenerates byte-identical
(the section renders nothing when its key is absent).
