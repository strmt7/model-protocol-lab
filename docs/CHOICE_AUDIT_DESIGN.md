# Choice-scoring audit design (`run --choice-audit`, cycle 195)

Status: frozen contract, red-team cross-checked before code (research
wave-8 rank #6). The swarm record is in `docs/RESEARCH_LOG.md`
(cycle 195): researcher brief, red-team A (16 findings, verdict
BUILD WITH AMENDMENTS), red-team B (every executed claim of A's
re-run bit-identically; 6 findings confirmed, 10 partially
confirmed, 9 sub-claims refuted, 5 B-NEW findings). This document
is the contract the build implements; where the original wave-8
brief and this document disagree, this document governs.

## 1. Utility

Monday morning: a practitioner deciding whether to trust the lab's
greedy-generation winner — or an lm-eval-style log-likelihood
ranking they got elsewhere — learns, per template family, how
often teacher-forced choice scoring and free generation pick
different answers on the same fixtures, and whether the two
methods' top candidates agree. The audit is a **second opinion**:
it never selects, never feeds the selection rule, and its payload
contains no `decision` about the run. (Vocabulary rule in §10.)

## 2. What is measured

Post-hoc over a completed run, teacher-forced only. The audit adds
**no generations** beyond the native search it audits; its cost is
forward passes (§9). For each search fixture, a four-choice set is
constructed from the run's own search fixtures (§3). Each choice
is teacher-forced under each candidate over that candidate's
recorded search prompt for the fixture, using the existing
`scoring.py` engine. Three rankings of the same cells are reported
(§4). Headlines per candidate: choice accuracy under the total
ranking, the generation × choice discordance table (§7), and
difficulty strata (§7). Method level: whether the choice method's
top candidate agrees with the run's primary winner, computed only
over the common evaluable population (§6).

Two scope sentences that ship in the docs and CLI help with the
feature:

- The generation side is the lab's **greedy** generation. Nothing
  here speaks to sampled decoding.
- The choices are scored in separate forward passes over the same
  recorded prompt; they are never presented to the model as a
  list. There is no position or primacy confound and no
  permutation arm.

## 3. Choice-set construction (frozen order of operations)

Construction is a pure function of the search split. The order is
frozen (red-team B, B-NEW-5): construct → screen → dedup → score
→ non-finite demotion → evaluability → argmax and tie
classification, per ranking. No step may be reordered by an
implementer; the producer and the diff validator walk the same
order.

1. **Exclusions.** A fixture whose expected answer normalises
   to empty at rung 1 of the scorer-lens ladder is excluded
   from every choice population, as a target and as a donor,
   with the exclusion counted (scorer-lens / leading
   precedent; rung 1 begins with a trim, so strip-empty is the
   same set — one exclusion, one reason, `blank_expected`).
   A blank true choice would score `ok_empty` with total 0.0 —
   the maximum attainable — and win its set by construction;
   that is a fixture defect, not a measurement.
2. **Donor rule** `smallest-eligible-id-first-three-v1`
   (probe-local name; modelled on leading's donor rule, extended
   from one donor to three — leading's machinery itself yields
   exactly one plant per fixture and is *not* reused). For target
   fixture T, the donor pool is every non-excluded search fixture
   whose group differs from T's group (donors may share a group
   among themselves). A pool fixture is eligible when its
   expected is non-equivalent to **every** expected in T's group
   under the screen in step 3. Donors are taken in ascending
   fixture-id order.
3. **Screen** `scorerlens-r4-plus-containment-v1` — literally
   leading's `equivalent_claim`: rung-4 equivalence under
   `scorerlens.rung_match`, or folded containment either way
   (NFC + casefold + strip). What the screen does **not** prove:
   paraphrases outside the ladder's grammar still pass — `"1/2"`
   against `"0.5"` is the named example (the frozen numeric
   grammar is ASCII-decimal; fraction notation is outside it).
   The screen also has a stated cost: containment removes the
   best legitimate distractors (donor `"Paris, Texas"` against
   true `"Paris"` is a genuinely different, maximally plausible
   choice). The screen is adopted anyway — ambiguity in the
   correctness definition is worse than an easier set, and the
   resulting upward bias is disclosed by the strata in §7.
4. **Dedup.** Donors are deduplicated by byte identity, then by
   rung-4 equivalence class, in pool order: a donor whose class
   is already represented in the set is skipped. The realised
   set is the true choice plus the first three distinct-class
   eligible donors.
5. **Shortfall.** If fewer than three distinct-class eligible
   donors exist, the fixture is **not constructible**: it enters
   no choice population for any candidate, and the shortfall is
   counted in the per-fixture eligibility record. There is **no
   numeric-perturbation fallback** (leading's digit surgery is
   rejected on the record: a fabricated string is not a plausible
   *choice* the way it is a plausible *claim*, and the fallback
   would relieve shortfall selectively for numeric fixtures — a
   difficulty-correlated denominator distortion).

The construction is recorded per fixture in the payload
(`fixtures` block: target id and group, exclusion reason if any,
the chosen donors' fixture ids and groups, a SHA-256 digest of
every choice text, the realised distinct-choice count,
constructible or not) and bound by `donor_map_sha256` over the
whole block (leading's precedent: editing fixture B's expected
changes fixture A's choice set, so the map is part of the
treatment and is digested as such). Choice order is frozen —
true choice first, then donors in ascending fixture-id order —
and is presentational only, since each choice is scored in its
own forward pass.

## 4. Rankings (three, one headline)

Per fixture and candidate, each choice carries the engine's
reported values: float32-rounded total log-likelihood,
float32-rounded mean per scored token, scored token count, and
target byte length (`TargetScore` already reports all four).

- **Total** (headline): argmax over reported totals. This is
  lm-eval's `acc` semantics exactly (EleutherAI,
  https://blog.eleuther.ai/multiple-choice-normalization/).
- **Per-byte**: score defined as reported total ÷ target byte
  length, computed from the reported values so producer and
  validator agree bit-for-bit. This is lm-eval's `acc_norm`
  semantics (byte-length-normalised).
- **Per-token mean**: argmax over reported means. This is the
  token-length-normalised quantity the harness deliberately does
  not report (it is not tokenization-agnostic). It is reported
  because the engine already produces it and the comparison here
  is intra-checkpoint (one tokenizer), but it must **never be
  labelled `acc_norm`** in any surface — payload, report, docs,
  or CLI help.

The reported mean is taken from the engine's reported means
directly. It is **not** derivable from the reported total (the
engine rounds the mean from the *unrounded* total), and no code
— producer, validator, or test oracle — may exact-check
`mean == total / count` or `total == sum(per_token)`; either
check manufactures phantom drift on honest payloads. Per-token
values are not stored at all.

The audit measures total-vs-normalised disagreement; it applies
no PMI or calibration correction (Holtzman et al., arXiv:
2104.08315, surface-form competition, is the cited reason such
corrections exist). A discordant cell is evidence about the
scoring rule, not a repaired score.

## 5. Cells, evaluability, ties

- **Cell.** One (candidate, fixture, choice) teacher-forced score.
  The prompt span is alignment-checked once per (candidate,
  fixture) (`check_alignment` over the prompt with an empty
  target): a prompt-side failure (`decode_failed`,
  `encode_failed`, `prompt_roundtrip_mismatch` from that
  pre-check) voids all four cells with that reason, is counted
  once per fixture in `n_prompt_failures`, and reserves the
  per-cell reasons for target-side causes (target roundtrip,
  boundary merge, provider and context codes — including
  `context_limit_exceeded`, a first-class counted reason).
- **Non-finite demotion.** A cell whose reported total or mean is
  non-finite (reachable with status `ok`: float32 saturation of
  an extreme finite logit sum — the cycle-76 pin) is stored as
  `unavailable` with the fixed reason `non_finite_score`, in
  producer and validator vocabularies alike. A verbatim `-inf`
  would crash the `allow_nan=False` report write; the shipped
  logprob diagnostic's per-fixture path carries this latent
  exposure today and the audit does not inherit it.
- **Evaluability.** A fixture has a choice verdict under a
  candidate iff it is constructible **and** every constructed
  cell has status exactly `ok` with finite reported values.
  (`ok_empty` is a distinct engine status, not `ok`: strip-empty
  choices never reach scoring because construction excludes
  them, so an `ok_empty` cell appearing anyway is a protocol
  violation, counted under its own reason, never folded into
  alignment unavailability.) Anything else is
  `choice_unavailable` for that candidate, with per-reason cell
  counts beside it. Partial choice sets are never argmaxed:
  unavailability correlates with the choice's surface form and
  with the candidate's rendering, so a surviving-subset argmax
  is an easier, different task per fixture.
- **Ties.** Per fixture and ranking, the verdict is `correct`
  (true choice uniquely at the reported maximum), `tied` (true
  choice at the maximum with at least one other choice), or
  `incorrect` (otherwise). Ties are counted, never broken, and
  never counted as correct. All argmax comparisons — producer
  and validator — operate on the reported (float32-rounded)
  values, so a tie is a property of the published numbers, not
  of hidden precision.

## 6. Populations, denominators, the method-level headline

- Per-candidate accuracies are reported **beside their own
  denominators** (evaluable fixtures for that candidate). A
  candidate with zero evaluable fixtures reports accuracy
  `None` with a fixed detail — never 0.0, never a crash
  (scorer-lens precedent).
- The method-level comparison is computed **only over the
  common evaluable population**: fixtures with a choice verdict
  under *every* candidate. `n_common` is an always-reported
  readout, next to the floor outcome — never a silent absence.
- The floor is the probe-local constant
  `MIN_COMMON_EVALUABLE = 6`. Below it, the agreement *claim* is
  withheld with the fixed detail
  `choiceaudit_common_population_below_floor`; the counts still
  print. The floor gates the claim, never the reporting. Note
  the compounded arithmetic (B-NEW-3): all-cells evaluability
  intersected across up to 24 candidates makes a withheld
  headline a routine, expected outcome on boundary-hostile
  tokenizers — a known property of the instrument, stated here
  so it is never mistaken for a defect.
- **Native comparator.** The run's primary outcome is mapped
  exactly as leading maps it: `compatible_on_fixture_set` and
  `held_out_failed` map to `(selected, selected_candidate)`;
  any other decision passes through with its winner (usually
  `None`). The audit is gated on a complete search before any
  scoring: decision in {failed, cancelled}, a missing search
  list, or a search list shorter than the candidate list yields
  the whole payload `{"available": False, "detail":
  "native_search_incomplete"}` (leading's wrapper gate, same
  detail string).
- **Agreement** is multi-state, never a bare boolean, with
  frozen priority: `withheld` (below floor) →
  `native_abstain` (mapped primary decision is not a selection)
  → `choice_tied` (choice accuracies tie at the top over the
  common population) → `agree` / `disagree`.

## 7. Discordance and strata

- **Generation verdict** per (candidate, fixture), from the
  recorded observation's stored facts: `pass` iff the
  observation is present with status `completed` and
  `exact_match` and `boundary_found`; `fail` iff completed and
  not both; `missing` otherwise (no observation, or not
  completed), counted as `n_generation_missing`. (In genuine
  runs `exact_match` is construction-gated on completion at
  `core.py`'s single observation site; the status check is
  kept anyway so a foreign payload cannot convert a missing
  measurement into a generation fail.) Records store the facts
  (status, exact_match, boundary_found); the verdict is
  derived identically by producer and validator.
- **The 2×2** per candidate, headline (total) ranking, over
  fixtures with a generation verdict and a choice verdict of
  `correct` or `incorrect`: all four cells printed
  (gen-pass/choice-correct, gen-pass/choice-incorrect,
  gen-fail/choice-correct, gen-fail/choice-incorrect). Tied
  choice verdicts are counted beside the table, never forced
  into it. The generation-fail row is split by `boundary_found`
  into labelled sub-counts. The word "boundary" names two
  different facts in this payload — generation `boundary_found`
  (the response terminator was found) and the scoring reason
  `boundary_merge_mismatch` (a token straddles the prompt/answer
  seam) — and the report glosses the seam reason once as
  tokenisation alignment so the two are never read as one.
  Boundary-caused discordance is a named component of
  discordance, reported, not discounted: termination behaviour
  is part of what free generation does and teacher forcing does
  not.
- **Strata** per candidate (total ranking, zero extra forwards):
  accuracy split by numeric vs non-numeric true answer (the
  ladder's numeric grammar decides), accuracy on the
  token-length-matched subset (every donor within ±1 scored
  tokens of the true answer's scored token count), and the
  per-ranking verdict-agreement counts (total vs per-byte,
  total vs per-token mean). Donors are *unmatched* — answers to
  unrelated fixtures, screened for equivalence only, not for
  plausibility, type, or length — so choice accuracy is an
  upper bound on item-matched choice accuracy, and the design
  says so wherever the number is introduced.

## 8. Ground records and the diff validator

The payload is self-contained (the default report view strips
observations, so nothing may be joined back at diff time):
protocol object (version, donor rule, screen name, K, ranking
names, `donor_map_sha256`, backend fingerprint), the `fixtures`
construction block (§3), and per-candidate entries carrying
status/detail, the aggregates of §6–§7, and one record per
search fixture: fixture id, the generation facts, the prompt
reason if any, and the four cells (status, reason, reported
total, reported mean, scored token count, target byte length —
no per-token values).

The diff validator (`diff_choiceaudit`) recomputes **every**
derived value from these ground records with its own mirrored
implementation (diffreports loads standalone; it imports nothing
from the producer): the donor-map digest over the stored block,
donor distinctness and cross-group membership from the stored
ids/groups, per-ranking verdicts and tie classification from the
reported values, accuracies, the 2×2 and its boundary split, the
strata, the common population, and the method-level agreement
including its withholding. It asserts every stored float is
float32-representable (mirrored struct check) — recompute from
reported values is only meaningful if the stored values *are*
the reported values. It never derives a mean from a total.

What the validator does **not** verify, stated plainly: the
equivalence screen and donor provenance against the fixture
*texts* (records carry digests, not texts — the same binding
standard as the shipped scorer-lens and leading validators,
which verify binding and arithmetic over producer-asserted
ground records). Protocol, donor-map, or backend drift between
two payloads marks the comparison not comparable, with the
drift still listed.

## 9. Cost, caps, cancellation

- Cost: **4 forward passes per candidate per fixture (9,600 at
  the legal maximum** of 24 candidates × 100 search fixtures),
  stated in the CLI help in the tree's cost-sentence form. A
  scoring forward is one pass; this is one to two orders of
  magnitude cheaper than a generation probe's budget at the
  same caps, and cheaper than the native search the run already
  paid for.
- Caps, checked **before** any scoring (the house pattern):
  records = candidates × search fixtures ≤ `MAX_RECORDS =
  2400` (fits the legal maximum exactly — no legal scale is
  made unavailable) and a 12 MiB records byte cap; breach fails
  closed with `choiceaudit_cell_budget_exceeded`.
- Cancellation is checked at each fixture boundary (per-cell
  checks are permitted, not required); a cancelled audit is
  `{"available": False, "detail": "cancelled"}` with no partial
  aggregates.
- Backends without `forward_logits` (everything except
  `LocalHFBackend` today) yield the whole audit unavailable
  with the engine's `logits_not_supported` detail — honest
  degradation, and the dispatch tests are built for it.

## 10. Vocabulary and boundaries

Second-opinion vocabulary, frozen: ranking keys live under the
`choice_audit` payload; the native comparator is always named
`primary_*`; the HTML section is titled "Choice-scoring audit
(second opinion)". The tree's discipline is namespacing, not
word bans (scorer-lens emits `winner_by_rung` inside its own
payload; leading emits per-condition `decision`/`winner` beside
`primary_*`) — the audit follows it. Two boundary sentences,
identical in this doc, USAGE, and CLI help:

- Against `--logprob-diagnostic`: the diagnostic teacher-forces
  only each fixture's expected answer and bands its mean logprob
  beside the native verdicts; it constructs no choice set, ranks
  nothing, and cannot express that another scoring method would
  have picked a different answer or template.
- Against `--scorer-lens`: the lens re-grades the *recorded
  generations* under normalisation rungs and re-derives a winner
  from the same answers; the audit scores *answers the model
  never generated* and compares the two methods' winners.

## 11. Uniqueness (research pass, cycle 195)

Log-likelihood choice scoring is established prior art — it is
lm-eval's native multiple-choice mode (`acc` over summed
log-probability; `acc_norm` over byte-length-normalised
log-probability; EleutherAI,
https://blog.eleuther.ai/multiple-choice-normalization/), and
HELM showed that the choice between likelihood scoring
(*separate*) and generating the answer (*joint*) can swing
accuracy by tens of points across models under a standardised
prompt (Liang et al., arXiv:2211.09110). What no surveyed tool
does is run both scoring ontologies over the *same*
hand-rendered chat-template candidates on a single local
checkpoint — greedy generation judged by exact match plus
response boundary under the lab's conservative
unique-winner/abstention rule on one side, and teacher-forced
choice scoring over donor-screened choice sets built from the
run's own fixtures on the other — and audit their disagreement
per fixture (the full generation × choice 2×2) and at the
method level (whether the two methods' top candidates agree,
ties reported as ties). The combination — donor-screened choice
sets × template family × generation-winner discordance audit —
is the claim. Nearest neighbours, each missing at least two
limbs: HELM's adaptation comparison (across models, dataset-
native choice sets, no abstention winner), Molfese et al.
(arXiv:2503.14996 — its "Logprobs" is the answer *letter*
token's probability, not full-string scoring), and the MCQ vs
free-generation comparison of arXiv:2403.17752 (a study, not a
tool; no template candidates, no winner audited).

## 12. Falsification

Zero discordance on completed, fully-evaluable cells under the
total-loglikelihood ranking — over the reported completed-cell
denominator, for greedy generation, this checkpoint, and this
fixture set — falsifies "the scoring method matters for this
checkpoint/fixture set": a publishable null that cross-validates
the native winner. An audit whose 2×2 population is empty claims
no falsification; it reports its denominators.
