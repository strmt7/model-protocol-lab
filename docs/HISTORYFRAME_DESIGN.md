# History-frame probe — design of record

## Status

BUILT in cycle 219 as `run --history-frame`, under the fresh
review's BUILD-as-amended verdict (coordinator #61, Part III
below): amendments A1–A10 are incorporated into Part I, which
is the instrument as built. Part II is the round-1
adjudication (coordinator #44, NO-BUILD) whose §3
resubmission specification the fresh review judged; the
candidate text of record is research §6
(`~/workspace/agents_out/historyframe/research.md`), of which
Part I is the as-amended form. Endpoints (the four arms, the
single k=6 prepend, the frozen chatml/llama3 donor pair), the
population (the run's search fixtures preceded by the frozen
synthetic history), and the headline scorer (byte-exact ∧
boundary) are frozen — an amendment touching any of them
would have been a resubmission, not this build.

# Part I — The instrument as built (candidate text as amended)

## 1. What exactly varies — the treatment dimension

The *framing strings of the prepended history segment only*:
per-turn role prefixes, per-role turn terminators, and the
history segment's internal repeat_bos behaviour, taken from a
Format's fields. Message contents are byte-identical across
arms by construction; the final segment (the fixture's own
messages), the prompt-initial bos, and the open suffix are
always the candidate's own. Token length, seam shape, and
marker encoding are *consequences* of the framing strings,
reported per case — the claim subject is the named frozen
framing pair, never "history" or "migration" in the abstract.
Arm-level batch composition and numerics channels correlated
with the foreign prompts' length are named residuals inside
the string-scoped claim (amendment A8, §10 below).

## 2. Population and frozen texts

- The run's **search fixtures only** (≤100), post-hoc stage
  on the recorded run; held-out fixtures are never treated
  and the primary decision/winner/exit code are never
  touched.
- Every search fixture is prepended with the shipped frozen
  history: `synthhistory.build_history_fixtures(cases,
  condition="history_6")`, i.e. `SYNTHETIC_TURNS` (6 turns,
  3 exchanges), reused, never copied or re-authored. The
  prepended segment is the reframed object (k = 6). The
  fixture's own messages — including any native history they
  already carry — are the tail and are always framed by the
  candidate's own family. Construction pin (amendment A5):
  the stage takes the built fixtures' *messages* and
  re-wraps them under each fixture's **original id**, its
  original `expected`, and the module's frozen group label
  `history_frame`; the built fixtures' namespaced ids
  (`synthhistory/history_6/<id>`) are never stage fixture
  ids, because ids feed request seeds (§3).
- The reframed history is *scripted and benign and
  task-free*, so the claim is scoped to this frozen
  transcript — exactly the scoping synthhistory already
  ships — and says nothing about task-bearing or
  self-generated histories. Screen: none beyond §4 (a
  closed-book screen is meaningless here — the history
  carries no task content that could be removed; the
  fixture's task is in the tail, held constant across arms).

## 3. Hybrid-render semantics (frozen formal definition)

For history Format A, final Format B, message tuple `msgs`
(the prepended fixture's messages), split point k = 6:

```
seg(F, m, i)  = (F.bos if i > 0 and F.repeat_bos else "")
              + (F.user_prefix if m.role == "user" else F.assistant_prefix)
              + m.content
              + (per-role end of F for m.role, falling back to F.turn_end)
open(B)       = (B.bos if B.repeat_bos else "")
              + (B.open_suffix if B.open_suffix is not None else B.assistant_prefix)
hybrid(A,B)   = B.bos
              + concat(seg(A, msgs[i], i) for i < k)
              + concat(seg(B, msgs[i], i) for i >= k)
              + open(B)
verbatim(A,B) = A.bos + (hybrid with only the leading B.bos removed)
```

Pins, each load-bearing:

- **BOS is B's** in every headline arm. A's bos appears in
  no headline prompt. The verbatim arm differs from its
  headline twin *only* in the leading bos string, so
  headline-vs-verbatim discordance is the BOS-substitution
  share, printed descriptively, never a decision input.
  The removal is **leading-only** (amendment A1): a
  repeat_bos candidate's internal bos occurrences stay.
- **A's open suffix never appears.** The history segment
  ends at the last history turn's A terminator; the next
  bytes are the tail's first B segment. No separator is
  inserted at the seam, ever.
- **repeat_bos follows the segment's family**, with global
  message indices (so the tail's first message, at index 6,
  takes B's repeat when B.repeat_bos). Under A == B this
  reproduces `Format.render` exactly (verified by the
  review at 354/354 over the candidate Formats and by the
  build's property test over the catalog plus local
  repairs).
- **Assistant-history turns use A's `assistant_prefix`**
  (with its think scaffolds where the family defines them),
  never A's `open_suffix`; the generation prompt is B's
  `open(B)`. Behaviours no Format field represents
  (auto-injected system blocks, reasoning-content history)
  are out of scope for every arm equally and are named as
  such in the section.
- Evaluation path: one `core.evaluate` call per (candidate,
  arm, fixture) through a carrier `Format(B.name, "", "",
  "", "")` holding the hybrid text as a single user message,
  with B's resolved boundary ids, following the
  `_canonical_form_cell` pattern: ids precomputed by
  `backend.encode(hybrid_text)`, context check before
  generation, aggregate-eligible report → cell, else the
  per-cell detail. The stage fixture carries the fixture's
  **original id** in every arm (request-seed identity across
  arms, `core.py:371`). Records key the fixture by that id.

## 4. Arms, controls, and pre-generation screens

| Arm | History framing | Role |
|---|---|---|
| `own` | B (bos-native hybrid with A = B) | headline baseline; byte-identical to the synthhistory `history_6` render of the same messages (screen S1) |
| `foreign_chatml` | chatml | headline contrast 1 |
| `foreign_llama3` | llama3 | headline contrast 2 |
| `verbatim_chatml` | chatml, bos = chatml's | descriptive only: BOS-substitution decomposition vs `foreign_chatml` |

Amendment A10 corrects the own arm's claim: the byte
identity to synthhistory's `history_6` render **holds**;
verdict-identity with the synthhistory *stage* is **not**
claimed — that stage runs namespaced fixture ids, hence
different request seeds. This stage's baseline is its own
`own` arm.

Controls and what each separates:

- `own` vs foreign: **history-frame effect**, current-turn
  format held constant by construction (the tail and open
  suffix are B's in every arm) — no extra arm is needed or
  honest for the current-turn axis.
- `verbatim_chatml` vs `foreign_chatml`: **BOS artefact**
  share, exactly. The verbatim arm has a single donor by
  design: there is no `verbatim_llama3`, and that asymmetry
  is printed as scope, not an omission.
- Render/ids divergence accounting (§5): separates "the
  frame changed the prompt" from "the frame changed the
  verdicts"; render-identical and ids-identical cases never
  enter a headline population.
- History *content/authorship/length* are frozen constants
  of the instrument (one transcript, scripted, one dose),
  so they cannot confound the within-panel contrasts; they
  bound the claim's scope instead.

Screens, all pre-generation, all fail-closed with counts:

- **S1 per-arm slice-swap identity (amendment A9,
  strengthened beyond the candidate's diagonal-only
  form).** For every candidate, fixture, and headline arm:
  the hybrid must equal the candidate's native render with
  only the history-segment slice exchanged; the verbatim
  arm must satisfy its own identity (the same slice
  exchange under its defining leading-bos substitution).
  The diagonal identity `hybrid(B,B) == B.render(...)` is
  the corollary the round-1 screen checked alone — the
  review executed a seam mutant (a `\n` inserted at the
  seam) that passed the diagonal-only screen for 51
  candidates and is killed by the per-arm form. Any
  failure → whole stage `historyframe_screen_failed`,
  zero generations spent.
- **S2 frozen-text audit:** `audit_special_token_literals`
  (`fixtures.py:471`) over the frozen history texts,
  audited against the special-token inventory derived from
  the framing fields of chatml/llama3/every candidate
  family. A literal collision inside the *frozen* material
  → `historyframe_screen_failed`. User fixture contents
  are **not** screened: they are byte-identical across
  arms, so they cannot manufacture a frame contrast, and
  screening them would silently change the population
  relative to the primary run. This asymmetry is
  deliberate and stated in the section.
- **S3 budget:** record/generation caps of §8 checked
  before generation → `historyframe_record_budget_exceeded`.

## 5. Eligibility, divergence, and the encoding stratum

Per (candidate, headline contrast), per fixture:

- **Render-divergent** iff the A history segment bytes
  differ from the B history segment bytes for that fixture
  (the segment depends only on the frozen turns, so this is
  constant across fixtures within a candidate — but it is
  *recorded* per fixture, as segment digests, so the
  validator recomputes the same population).
  Render-identical fixtures are excluded from the contrast,
  counted as `n_render_identical`; a contrast
  render-identical on every fixture is **inapplicable for
  that candidate**: no classification is emitted for it, and
  the section prints the inapplicable share, never a
  robustness reading.
- **Ids-identical** (byte-divergent renders encoding to the
  same id sequence under the run's backend) cases are
  likewise excluded from the headline population and
  counted; a same-ids verdict flip is recorded as a
  determinism finding, never as frame fragility
  (canonical-form discipline). The gate is **digest-derived**
  (amendment A4): records carry per-arm token-id digests
  (the negation `digest_ids` pattern), and both the
  producer and the validator derive ids-identity from
  digest equality — the candidate text's carried divergence
  flags are not used, because the review executed a label
  flip through them. Per contrast the section prints the
  `id_length_delta` list (own ids − foreign ids, fixture
  order) and `max_abs_id_delta`.
- **Verbatim applicability (amendment A1):** when the
  candidate's bos equals the chatml donor's bos, the
  verbatim arm is byte-identical to its headline twin
  (55/118 candidate Formats in the review's census, chatml's
  bos being the empty string) — the readout is recorded
  **inapplicable** for that candidate: counted, printed,
  its discordance never pooled, and never read as a BOS
  share of zero. Applicability is likewise derived from the
  recorded bos digests.
- **Encoding stratum (run-level, per foreign family):** for
  each distinct donor history framing literal, `token_real`
  iff `backend.encode(literal)` yields exactly one id (the
  deliforge operational definition; the stage has only the
  Backend protocol, so registration/id-consistency in
  `core.template_tokenizer_compatibility`'s sense is *not*
  claimed). Stratum = `all_single` / `all_fragmented` /
  `mixed`, with the per-literal table printed when `mixed`.
  The stratum is constant within a run (one tokenizer), is
  printed beside every contrast's classification, and is a
  compared field in `diff-reports` so cross-checkpoint
  comparisons never pool strata silently. No checkpoint
  fact is asserted in this design: which stratum a given
  checkpoint lands in is a run-time measurement.

## 6. Scorer, decomposition, decision rule, selection

- **Cell verdict:** `exact_match and boundary_found` from
  the carrier evaluation (§3), i.e. `core.py:797` byte-exact
  conjoined with boundary presence. No stripping, parsing,
  or normalisation is added anywhere on the headline path.
- **Decomposition** of every discordant pair (negation §4.2
  pattern, whose `boundary_only` disposition the parent
  accepted for this tree): `boundary_only` when the arms'
  answer digests are equal (the discordance is a boundary
  event, never a frame effect on the answer), `exact_only`
  when digests differ and both boundaries were found,
  `mixed` otherwise. All three counts printed per contrast;
  the classification counts verdict discordance as recorded
  (boundary-only discordance is *printed separately* so a
  reader can see how much of a classification is
  boundary-carried — it is not silently removed from the
  counts, because the tree's verdict is the conjunct).
- **Classification (per candidate, per headline contrast),**
  over paired completed render-divergent ids-divergent
  cases: n = paired count, a = own-only passes, b =
  foreign-only passes, d = a + b. Total function: n < 10 →
  `insufficient`; d = 0 → `no_discordance_observed`; d < 6 →
  `below_threshold`; a == b → `discordant_balanced`
  (descriptive); else McNemar two-sided exact p
  (`confidentiality.py:293`) < 0.05 →
  `discordant_asymmetric`, else `below_threshold`. The
  discordant floor of 6 is forced by α = 0.05 (a unanimous
  5–0 split gives two-sided p = 0.0625; 6–0 gives 0.03125)
  — derived, not borrowed. The n-floor of 10 stands as a
  frozen **conservative constant** (amendment A3): the
  candidate text's derivation was false — the smallest
  interior-attainable n for a 6–0 split is 8 — and the
  operating-characteristics table, derived with the floor
  in place, is the floor's justification. **Falsification
  boolean := classification == `discordant_asymmetric`**
  (direction read from a vs b), with the exact
  random-labeller fire rate at the run's own n printed
  beside every classification and the family-wise
  expectation **computed** from those rates over the run's
  decisions (amendment A2: at uniform n = 100 over 48
  decisions the exact expectation is 1.76 false labels with
  P(≥1) ≈ 0.83 — the candidate text's printed 1.44 / 0.77
  was 48 × a rounded rate and is corrected everywhere it
  rendered). Under symmetric two-arm jitter the fire rate
  is 0.0004 / 0.0048 / 0.0213 at ε = 0.01 / 0.02 / 0.05
  (D-invariant), printed in the section (amendment A6);
  one-sided jitter variants are arm-asymmetric
  perturbations — inside the alternative, not the null.
  The review's PoC5 correlated column is **dropped**
  (amendment A7: no reproducible reading); the printed
  bounds stand — random-labeller fire ≤ 3.6 percent and
  competent-null fire ≤ 3.1 percent at every tabulated N.
  Discordance is never pooled across candidates or
  contrasts.
- **Selection readout:** per arm, the conservative
  selection (exactly one candidate passing every fixture
  in that arm wins) recomputed on the common completed set
  across candidates for that arm; withheld with the
  canonical-form detail `selection_withheld_mismatched_sets`
  when the restricted completed sets differ or are empty.
  Winner headline per headline contrast via
  `canonicalform.winner_headline` semantics: `winner_changed`
  only when both arms select and winners differ;
  selection-vs-abstention is `winner_abstention_asymmetric`,
  never a change; withheld on either side →
  `winner_undefined`. Flips against the *primary* run are
  printed separately, labelled as pooling history presence
  with frame (the synthhistory `flips` pattern, with its
  verdict mapping), and are never the headline.
- **Per-arm attrition** (`n_context_exceeded`,
  `n_evaluation_failed`, `n_render_identical`,
  `n_ids_identical`) printed beside every paired rate;
  ungated discordance (all paired completed cases, before
  divergence exclusions) printed as a sensitivity wherever
  gating applies.

## 7. Fail-closed vocabulary (frozen; no new verdict words)

Following the sibling pattern exactly (negation §4.3):

- Stage details exactly: `{cancelled,
  historyframe_record_budget_exceeded,
  historyframe_screen_failed}`, the latter bound to screens
  S1/S2 of §4.
- Per-cell details exactly: `{"", arm_context_exceeded,
  arm_evaluation_failed}`.
- Whole-stage failure (any renderer/serializer exception,
  non-UTF-8 payload): `{"available": False, "detail":
  "historyframe_unavailable"}` — the stage never fails the
  run (the CLI `_run_*_probe` except-pattern).
- Section classifications exactly: `insufficient`,
  `no_discordance_observed`, `discordant_balanced`,
  `discordant_asymmetric`, `below_threshold`; winner
  headlines exactly `winner_changed`, `winner_stable`,
  `winner_abstention_asymmetric`, `winner_undefined`. The
  only new strings are the module-prefixed stage details,
  which is the established pattern, not a new verdict word.
- Model parse failure is scored wrong, never a detail.

## 8. Counts (frozen, with derivations)

- Generating arms: 4 (`own`, `foreign_chatml`,
  `foreign_llama3`, `verbatim_chatml` descriptive). Legal
  maximum: 4 × 100 search fixtures × 24 candidates =
  **9,600 generations**; headline arms only: 3 × 100 × 24 =
  7,200. (A run's actual candidate count is one family's
  variants — typically ≤ 8 — so the legal maximum is a cap,
  not an expectation; it is stated as a cap, per tree
  convention.)
- Ground records: 24 × 100 = **2,400** multi-cell records
  (one per candidate × fixture, one cell per arm — the
  canonical-form record shape), `MAX_RECORDS_BYTES` at the
  house 12 MiB pattern. Caps are frozen constants in the
  module with derivations in CLI help, enforced by the
  stage and re-derived by the validator.
- Context reality check: the prepended native render adds
  the frozen history's tokens to every fixture; long
  foreign framings will push some (candidate, fixture)
  cells over `--max-context` into `arm_context_exceeded`.
  That attrition is arm-ordered by construction and is
  exactly why §6 recomputes selection on common completed
  sets and prints attrition beside every rate; it is
  disclosed, not hidden.

## 9. Records and validator mirror

Records carry digests only: per record, the fixture id, the
expected-answer digest, the candidate/donor bos digests,
each arm's history-segment digest, hybrid-text digest, and
token count, the per-arm token-id digests (amendment A4),
the four cells (verdict, boundary_found,
boundary_text_match, input_tokens, answer_digest — the
canonical-form cell shape), and cell_details. The payload's
protocol object binds the donor pair, k = 6, the fixture
ids, the frozen history material's digest, the segment
rule's digest, the run's seed-bearing backend fingerprint,
and the frozen floors and budgets. The diff validator
mirror re-derives — never imports — the eligibility
accounting, the classification, the restricted-set
selection, the winner headlines, the primary flips, and
the family-wise expectation from the records, and rejects
flag/count inconsistencies and any bos-pair inconsistency
across a candidate's records. (The mirror cannot re-render
the hybrids: the population is the run's own fixtures,
which are not regenerable from the payload — the render is
pinned at stage time by screen S1 and in the protocol by
the segment-rule digest. Everything decidable from the
records is recomputed; the review's decidability executions
re-proved exactly that set.) Reports render counts/rates
only, no prompt/answer text, every payload-derived string
through `_escape_text`, never bare `escape()`.

## 10. The claim (only falsification-class sentence)

"On this candidate, on this run's search fixtures preceded
by the frozen synthetic history, rendering that history's
turns in the frozen chatml [resp. llama3] framing instead
of the candidate's own framing produced verdict discordance
at or above this panel's derived floor, asymmetric by an
exact McNemar test — a history-frame discordance for these
named framings and this frozen history, on this checkpoint."
Every word of scope in that sentence is load-bearing; no
sentence in the report may generalise it to histories,
migrations, or templates in general. Named residuals inside
that scope (amendment A8): token length and marker encoding
are consequences of the framing strings and move with the
treatment by construction; arm-level directional channels —
batch composition and numerics correlated with the foreign
prompts' length — are not separated by this instrument and
are printed beside the claim so the label is never read as
a mechanism certificate.

## 11. Naming audit (binding build condition, executed)

The fresh review required the build's USAGE naming audit to
either adopt a prepend-carrying flag name or keep
`--history-frame` only with first-sentence disclosure of
the prepend scope. Audit (cycle 219 build log): zero
collisions for `--history-frame` / `history_frame` in
cli.py, USAGE, or the package; the name's plain reading
("the frame of the history") understates the construction —
the probe *prepends* a frozen synthetic history and reframes
only that segment — so the disclosure form was adopted:
the CLI help's first sentence, the USAGE section's first
sentence, and the report section's first sentence all state
that the probe prepends the frozen six-turn synthetic
history to every search fixture and reframes only that
prepended segment, and that the fixtures' own messages are
never reframed.

## 12. Uniqueness pass (primary sources)

The review round re-surveyed the neighbouring instruments
against primary sources (LM Evaluation Harness, HELM,
OpenAI Evals, DSPy — see the research log entry for cycle
219): none ships segment-level cross-format reframing of a
prepended conversation history with paired verdict
discordance, divergence-gated eligibility, and per-frame
selection recomputation. Within the tree, the neighbours
are synthhistory (history *presence/depth*, native framing
only), canonical-form (Unicode normalisation of the whole
prompt), and template robustness (whole-template swaps).
The differentiation claimed is the combination — a frozen
history held byte-identical in content while only its
framing strings are exchanged for a foreign family's, under
the run's own fixtures and scorer — never a claim of being
first at any single element.

# Part II — Round-1 adjudication (coordinator #44, NO-BUILD)

## Round summary (coordinator #44, design round only — nothing built)

Wave-11 slate rank 3: cross-format history (tentatively
`run --history-frame`). The candidate sketch
(`~/workspace/agents_out/wave11/slate_candidates.md` Candidate 3):
hold every message's text byte-identical, render the history turns
in a foreign chat-template family A while the final turn stays in
the candidate's own family B, and test whether per-candidate
verdicts and the selection are invariant to that reframing.

The round ran the full blind team (briefs in
`~/workspace/agents_out/historyframe/`: `research.md`,
`redteam_a.md`, `redteam_b.md`, `verifier.md`). The verdict is
**unanimous NO-BUILD for the sketch as specified** — researcher,
both blind red teams, and the verifier's blind interim audit each
reached it independently, fatal by fatal, by execution. The
fabrication sweep is **clean — a fourth consecutive round**: every
cited tree symbol resolves, and both external arXiv identifiers
resolve to the cited titles, authors, and substance. The errors
found are counting/criterion slips, corrected in §5; none is
load-bearing for the verdict.

The researcher additionally specified a complete repaired
instrument (research §6: bos-native field splice over
synthhistory's frozen prepended history, carrier-Format
evaluation, two frozen donor families, a derived McNemar-gated
classification). The verifier upholds, and the coordinator
concurs, that this instrument is a **resubmission, not an
amendment**, under this program's standing test (§2): it changes
the population mechanism, replaces the headline BOS semantics,
changes the headline population, and adds the decision rule the
sketch lacks. It is recorded in §3 as the resubmission candidate.
No build cycle may act on it without a fresh blind design review.

## 1. Why NO-BUILD: the fatal defects of the candidate as specified

### F1 — The population is undefined: "history" has three incompatible readings, and the shipped fixtures are half-vacuous

The sketch never says where the history turns come from, and the
tree offers three mutually exclusive sources — the fixture's own
messages, synthhistory's scripted turns, self-generated replies —
each a different instrument with a different construct and budget
(RT-A F1, RT-B F4, R §2 F1, V F-POPULATION). Executed on the
shipped example set (`examples/fixtures.json`, verified by all
four lanes): search = `search-copy-blue` (1 message, no history)
and `search-latest-green` (3 messages); held-out likewise 1 of 2.
For a single-turn fixture the foreign condition is byte-identical
to control by construction — guaranteed-concordant filler in every
denominator. The loader admits 1-message fixtures
(`datasets.py:53-72`; `MAX_MESSAGES = 21` at `datasets.py:14`), so
no dataset guarantees the sketch's "100 fixtures" of history.
Under the fixture-own reading, history length (dose) and content
are per-fixture and uncontrolled — in the shipped set the history
assistant turns are task answers ("RED", "NORTH") and the final
turns refer to them, so discordance would pool framing with
history-content leverage (RT-A F12, RT-B F4). Under the
self-generated reading the budget is false: each foreign cell
needs one generation per history exchange before the scored
generation — up to 28,800 generations at history_6 depth against
the sketch's 7,200 (RT-A F1, arithmetic verified).

### F2 — The hybrid renderer does not exist, its seam semantics are unspecified, and the sketch's seam paragraph is wrong in both particulars

`Format.render` (`core.py:76-115`) renders a whole message tuple
in one framing and requires a final user turn. Executed by all
four lanes: `render(history_alone)` raises `ValueError` for
29/29 catalog families — a history prefix, which ends with an
assistant turn, is not a renderable object. The sketch's literal
recipe (render the segments, concatenate) fails its own
pre-registered control check: concatenating a history render
(which needs a dummy closing user turn to satisfy `render` at
all) with a final-turn render duplicates BOS, strands family A's
open suffix mid-prompt, and inserts a dummy turn (RT-A F2:
182 vs 132 chars chatml, 362 vs 246 llama3, control condition
included). The sketch's seam note — A's render "may carry its
own BOS and closing suffix mid-prompt", to be kept "verbatim
(recommended)" — is executed-wrong twice (RT-B F1, verifier
V-2.2): in a prefix-truncated A render there is **no** A closing
suffix at the seam (the open suffix is appended only after the
final turn), and A's BOS lands at position 0, *replacing* B's,
not mid-prompt. The decisions that determine the prompt's bytes
are all unmade in the sketch, and each is load-bearing:

- **BOS ownership.** Coordinator execution: the donor-BOS and
  final-BOS splice readings produce different prompts on
  658/812 ordered family pairs. Only the choice is frozen in
  §3's resubmission; the sketch freezes none.
- **`repeat_bos` cadence.** No catalog base sets it, but shipped
  candidates do (`llama3/toggle-bos` via
  `catalog.candidates(mutate=True)`). A per-segment cadence
  restart produces 1 BOS where the control has 4 (RT-A F2).
- **None-vs-empty fallbacks.** `core.py:57-73` honours
  empty-string overrides as distinct from None; a hybrid written
  with the natural `or` fallback fails control identity for
  `mistral_large` alone (open_suffix `""` vs assistant_prefix
  `" "`), so a casual control test on chatml passes while the
  renderer is wrong (RT-A F2).
- **Open suffix is not history framing.** `open_suffix !=
  assistant_prefix` for 15/29 families (verifier-confirmed list),
  9 of them opening a think block; the suffix is the generation
  prompt for the next reply, and any recipe that keeps a history
  render's tail injects it, sometimes unclosed, at the seam.

The control-identity check the sketch pre-registers is a *test*,
not a *specification*: three independent executions this round
(coordinator: 118 candidate Formats × 2 conversations; verifier;
researcher: 944/944 renders) prove a faithful field-splice
semantics *can* reproduce `B.render(full)` byte-identically when
A == B — by a recipe the sketch does not contain.

### F3 — There is no decision rule: the any-difference falsifier fires on noise with probability ≈ 1, and the selection readout is a one-flip tripwire

The sketch's falsifier is verbatim the one this program has
killed twice (DIRECTIVE_DESIGN F2; INSTRUCTIONDISTANCE_DESIGN
F3): "any fixture whose verdict differs … or a selection flip in
any foreign-frame condition." No floor, no discordant-pair rule,
no fail-closed vocabulary. Executed (RT-A F3, RT-B F3, V F-NOISE):
at per-cell jitter ε = 0.001, P(any discordance in the panel) =
0.909–0.992; at ε ≥ 0.005, ≈ 1.000. `batchinv.py` exists
precisely because identical token sequences need not reproduce
across batch composition, so ε = 0 cannot be assumed. The
selection half is worse: `_select_winner`
(`experiments.py:3847-3863`) requires the winner to pass *every*
search fixture, so one flipped cell manufactures a "selection
flip"; conditioned on a native winner existing, P(it loses
all-pass status in ≥ 1 of 2 foreign conditions) = 0.633 / 0.866
/ 0.982 at ε = 0.005 / 0.01 / 0.02 (RT-B F3, verifier
re-evaluated). Nor can a builder patch in a familiar floor: any
(6,·)-shaped discordant floor fires ≈ 1.000 at this panel's
N = 100, because the null discordance mean is 50 (random) or
≈ 18 (competent) — the sibling floors were derived for panels
of 16 generated cases, and this panel's N is the user's fixture
count (RT-B F3). As specified, the instrument can essentially
only report "not invariant" — a verdict decided by the rule,
not measured from the model.

### F4 — Attribution: the treatment is a bundle of at least five simultaneous changes, and the claim names one cause

Control vs foreign differ at once in (RT-A F4, RT-B F2, R §2 F4):

- **Marker identity** — the named treatment.
- **BOS identity/presence** (F2): the prompt's first tokens
  change with A. The tree treats BOS as a verdict-moving axis in
  its own right (`core.local_repairs`, `core.py:118-143`, ships
  toggle-bos/no-bos candidates to measure it).
- **History token length.** Same six synthetic turns framed:
  341 chars (mistral) to 989 chars (kimi3) (verifier-confirmed);
  per-exchange prompt-length deltas are nonzero for 766/812
  ordered pairs, range [−213, +213] chars under a bos-inclusive
  splice (segment-only splice: 770/812, ±216 — both executions
  correct under their splice definitions, and the divergence
  between two honest executions is itself F2's point). Length
  and final-turn position are shipped axes (needle, distraction,
  placement).
- **Think scaffolding.** Nine families carry think material
  inside their *assistant history prefix* itself (`glm45`,
  `glm53`, `laguna`, `ling`, `nanbeige42`, `kimi3`, `k2horizon`,
  `hy4`, `minimax3` — verifier-executed count; §5 corrects
  RT-A's five). Foreign history in these families injects a
  foreign reasoning-channel scaffold the control lacks — an
  output-priming component, not a delimiter relabelling.
- **Closure conventions.** `deepseek` history has no user-turn
  terminator (`user_turn_end=""`); some families' turns run
  together. "Foreign framing" here includes a segmentation
  change.

The sketch's claim sentence attributes any discordance to
framing-as-such. What the contrast can honestly estimate is
discordance between two named, frozen prompt strings per
(candidate, fixture) — the payload settlement. The sketch's
Monday-morning framing ("whether history must be re-rendered on
migration or can be carried over as-is") additionally fails on
the artifact: the hybrid (A's BOS at the head of a B session,
B's final turn) is neither a carried-over transcript nor a
re-render — no migration produces it (RT-B F2).

### F5 — The pair rule is broken: unnamed pairs, a disjointness criterion that excludes the realistic migrations, and a family/candidate unit mismatch

The ordered pair space is 29 × 28 = 812; the sketch freezes 2
(0.25%), names neither, and gives no selection rule beyond
"disjoint marker sets" (RT-B F5). Executed by all lanes:

- **Aliased pairs collapse to control.** Under the
  history-segment criterion (bos excluded), **36 ordered pairs**
  have a foreign history segment byte-identical to control:
  classes {chatml, qwen3, qwen38, kolibri, lfm25, nemotron35},
  {gemma, gemma3}, {deepseek, deepseek41},
  {muse_glimmer, muse_spark} — 21 distinct history segments
  across 29 families. Including BOS, **24 ordered pairs** are
  byte-identical outright. Any such cell is a guaranteed null
  pooled as invariance — the canonical-form ids-identical trap.
- **Disjointness selects against the utility.** The aliased
  ChatML-shaped families are precisely the plausible real
  migrations (8 families' `user_prefix` contains `<|im_start|>`;
  verifier-confirmed). A disjoint-marker rule excludes every one
  and selects maximally distant pairs, where F4's bundle is
  largest. Marker disjointness is in any case the wrong
  criterion: the operative check is rendered-segment equality
  (331–355 of 406 unordered pairs are textually disjoint under
  either lane's extraction, including pairs whose segments are
  identical).
- **Global vs per-candidate pairs dilemma.** Fixed global pairs
  collapse for candidates B ∈ {A1, A2} (their "foreign"
  condition *is* the control); per-B pairs make the
  per-condition selection compare candidates who faced different
  treatments — the restricted-common-set defect. The sketch's
  selection headline is coherent under neither reading (RT-B
  F5).
- **Unit mismatch.** Selection in this tree is over candidates,
  not families: a run's candidates are one family's base plus
  local-repair/thinking variants (`catalog.candidates`,
  `catalog.py:914ff`; 118 candidate Formats across the catalog,
  89 beyond the 29 bases — §5 corrects RT-B's 107). "Final-turn
  family B" and "foreign family A" are undefined for variant
  candidates whose framing differs from base exactly in the
  fields the splice moves (RT-A F10, RT-B F5).

### F6 — The boundary mechanism: a correct answer terminating like its history fails as a wrong answer, and the sketch has no decomposition

Stage verdicts conjoin `exact_match` and `boundary_found` with
the boundary resolved per candidate — i.e. per final-turn family
B (`experiments.py` `_resolve_boundaries`). The hybrid's history
demonstrates A's terminator after every assistant turn. Executed
through the real `core.evaluate` (RT-B F6; reproduced exactly by
the verifier and by the coordinator with independent stubs): a
stub answering `GREEN` and terminating with B's `<|im_end|>` →
`exact_match=True, boundary_found=True`; the same stub answering
`GREEN` and terminating with A's `<|eot_id|>` →
`exact_match=False, boundary_found=False`, answer recorded as
`'GREEN<|eot_id|>'` — with no boundary cut, the terminator text
joins the scored span, so the failure is double-counted as a
wrong answer. The sketch never states which boundary governs a
hybrid cell and provides no exact-only vs boundary-only
decomposition (the payload/negation settlements made that
decomposition mandatory). Under the any-difference falsifier,
termination-style discordance is indistinguishable from answer
discordance, and the migration advice differs completely between
the two. Ordered attrition rides along: foreign framings run
longer, so `arm_context_exceeded` censoring is ordered by arm
(the DIRECTIVE F6(a) defect), with no per-condition attrition
reporting specified.

### F7 — The token-encoding mitigation is incoherent: "token-real in both frames" selects the headline population by checkpoint, and its fallback contradicts the headline

The sketch concedes the deliforge axis leaks in and proposes the
headline population be "cells where the history's markers are
token-real in both frames, **or** the confound is disclosed per
cell, never pooled silently." Executed/adjudicated (RT-A F7,
RT-B F7, R §2 F6, V F-TOKENREAL): token-reality is a property of
(checkpoint tokenizer × literal), not of a frame — under HF's
default `split_special_tokens=False` a literal becomes a real
control token iff the checkpoint's tokenizer reserves it
(`control_tokens.py:1-14`; backend encode is a plain
`tokenizer.encode(text, add_special_tokens=False, ...)` call,
`backends.py:1192-1200`, `:1284`). On a given checkpoint,
typically only its native family's markers are reserved, so the
headline population varies per checkpoint, is selected on
exactly the axis being measured, and can be empty — with no
eligibility pre-check and no fail-closed vocabulary for that
outcome (contrast deliforge's `find_encoding_pair`,
`deliforge.py:238-281`, which fails closed). "In both frames" is
undefined for families whose markers are ordinary text. And the
OR-fallback contradicts the pooled selection headline: a
per-condition selection *is* a pooled object. Separately, the
single-encode seam re-tokenizes the novel A-terminator/B-prefix
adjacency; the tree's own precedent refuses exactly this where a
seam matters (prefill joins at the token-id level,
`prefill.py:19-24`), and the sketch records no seam-canonicality
per cell.

### F8 — The build surface is misstated and the utility overclaims: "scoring untouched / size M" is false, and scripted history is not a migrated transcript

`core.evaluate` renders through the candidate alone
(`prompt = candidate.render(case.messages)`, `core.py:846`), and
no `Format` renders the hybrid (F2), so the probe needs a new
evaluation surface — a plan-keyed backend wrapper (the
`DeliforgeBackend` precedent, `deliforge.py:492-547`), an
evaluate variant, or the researcher's carrier-Format route
(§3) — plus a payload schema from which the diff validator can
recompute each cell, plus configuration-digest binding of pair,
recipe, and history source (the cycle-207 obligation,
`_require_native_match`, `experiments.py:3940`; RT-B F9). None
of this is in the sketch's size-M estimate (RT-A F5). On the
claim side: whichever population reading is chosen, the
sketch's utility sentence describes *generated* history — a live
transcript written by a model under A — while the cheap readings
measure frozen scripted strings. Multiturn already settled that
scripted and self-generated history are different treatments and
must not be conflated (`multiturn.py:16-46`, including its
special-token screening discipline, which foreign-framed genuine
history violates by design). Compounding it, for the
think-scaffolding families the catalog's own comments record
that real templates treat assistant history differently from
the lab's flattened `Format` hypothesis (reasoning preserved,
dropped, or required, by family and branch): a field splice
measures the lab's `Format` hypotheses' strings, a scoping the
sketch never makes (RT-B F8).

### Non-fatals, adjudicated (what passed)

- **Duplication is not a standalone fatal.** The residue —
  genuine history turns, framed in A, preceding a final turn in
  B, benign content, verdict readout — is narrowly real
  (verifier §A7 map, upheld): synthhistory owns history presence
  over prepended turns rendered own-frame; multiturn owns
  sharded delivery and scripted-vs-self history, always
  own-frame; faketurn/deliforge own foreign delimiters *inside
  content* and the encoding axis; hierarchy owns single-turn
  privilege position; batch-invariance owns no part of this
  design but is the standing proof behind F3. The burden the
  sketch fails is not novelty — it is that the residue is
  unmeasurable as specified (F1–F8), and that a repaired version
  must justify itself as a new instrument rather than an arm on
  `synthhistory_stage` (RT-A F9).
- **Splice control identity is achievable** (F2): three
  independent executions reproduce `B.render(full)` under a
  faithful field splice. Credited to the sketch's instinct, not
  its text.
- **The single-encode premise is tree-true**: backends encode
  rendered text with `add_special_tokens=False`
  (`backends.py:1200`, `:1284`), so a concatenated string is a
  legal prompt input.
- **Budget arithmetic is consistent** with the tree's caps
  (≤24 candidates, ≤100 search fixtures): 3 × 100 × 24 = 7,200
  is an honest worst case *for the sketch's three conditions*;
  its defect is F1's, not arithmetic.
- **The name is free as a string**: `--history-frame` /
  `history_frame` / `historyframe` return zero hits in the
  package, `docs/USAGE.md`, and `README.md` (all lanes). Whether
  it overclaims is a resubmission naming-audit question (F8),
  not a fatal.

## 2. Why the repaired instrument is a resubmission, not an amendment

The standing test (payload, answer-syntax, directive,
instruction-language, instruction-distance rounds; contrast
negation): BUILD-as-amended is available when the sketch's
population, arms, and scorer were already frozen and executable
before the round, and the corrections fix the decision rule and
claim language *of that same instrument*. Here nothing
load-bearing was frozen: the population source is unspecified
(F1), the renderer and its seam semantics do not exist (F2),
there is no decision rule (F3), the pairs are unnamed and the
pair criterion is wrong (F5), the headline population is
incoherent (F7), and the scorer cannot score the treatment
(F8). The researcher's §6 repairs all of it — by *choosing* a
population mechanism (synthhistory's prepended frozen history on
the run's search fixtures), *replacing* the headline BOS
semantics (bos-native headline plus a descriptive verbatim arm,
against the sketch's recommended verbatim headline), *changing*
the headline population (render/ids-divergent gated, against
token-real gated), and *adding* a derived decision rule. That
is the payload configuration: a new instrument, reviewable only
as a resubmission through a fresh blind design review. Any
deviation by a future build from this settlement voids it.

## 3. Resubmission specification (NOT a build order)

The resubmission candidate is the researcher's §6
(`~/workspace/agents_out/historyframe/research.md`), incorporated
by reference. Its frozen items, in summary:

1. **Treatment (one dimension).** The framing strings of the
  prepended history segment only — per-turn role prefixes,
  per-role terminators, the segment's internal repeat_bos
  behaviour — from a Format's fields. Message contents are
  byte-identical across arms by construction; the final segment,
  the prompt-initial bos, and the open suffix are always the
  candidate's own. Token length, seam shape, and marker encoding
  are consequences, reported per case; the claim subject is the
  named frozen framing pair.
2. **Population.** The run's search fixtures only (≤100),
  post-hoc stage; held-out never treated; primary
  decision/winner/exit code never touched. Every search fixture
  is prepended with `synthhistory.build_history_fixtures(cases,
  condition="history_6")` (`synthhistory.py:105-131`): the
  frozen benign task-free `SYNTHETIC_TURNS`, reused, never
  re-authored. The fixture's own messages are the tail, always
  framed by the candidate's own family. The claim is scoped to
  this frozen transcript and says nothing about task-bearing or
  self-generated histories.
3. **Renderer (frozen formal semantics).** A segment function
  over Format fields with `is not None` fallback semantics
  copied from `core.py:76-115`, global-index repeat_bos cadence
  following the segment's family, B's bos at the head, A's open
  suffix never appearing, no separator inserted at the seam.
  Under A == B it reproduces `Format.render` exactly (research
  PoC1: 944/944 renders; arithmetic verifier-confirmed:
  118 candidate Formats × 4 message sets × 2 BOS readings).
4. **Evaluation path.** One `core.evaluate` call per (candidate,
  arm, fixture) through a carrier `Format(B.name, "", "", "", "")`
  holding the hybrid text as a single user message (verified
  this round: the carrier renders its single message verbatim),
  with B's resolved boundary ids and the fixture's *original id*
  in every arm (request-seed identity, `core.py:371`), following
  the `_canonical_form_cell` pattern (`experiments.py:9473`).
  The own arm is byte- and verdict-identical to the synthhistory
  `history_6` condition through the new path.
5. **Arms.** `own` (headline baseline) / `foreign_chatml` /
  `foreign_llama3` (headline contrasts; the pair is
  string-disjoint against each other and the obvious neighbours,
  executed) / `verbatim_chatml` (descriptive only: differs from
  its headline twin *only* in the leading bos string, so its
  discordance against `foreign_chatml` is the BOS-substitution
  share). Selection readout: per-arm conservative selection
  recomputed on the common completed set, withheld on mismatched
  or empty sets, winner headlines in canonical-form's vocabulary
  (`winner_changed` / `winner_stable` /
  `winner_abstention_asymmetric` / `winner_undefined`); flips
  against the primary run printed separately and labelled as
  pooling history presence with frame, never the headline.
6. **Eligibility and divergence accounting.** Per (candidate,
  contrast): render-identical fixtures excluded and counted as
  `n_render_identical` (a contrast render-identical on every
  fixture is inapplicable for that candidate — no classification
  is emitted, and the inapplicable share is printed, never read
  as robustness); ids-identical cases likewise excluded and
  counted, a same-ids verdict flip recorded as a determinism
  finding; per-contrast `id_length_delta` list and
  `max_abs_id_delta` printed. The encoding stratum is a
  *run-level* label per foreign family (per-literal
  `token_real` iff `backend.encode(literal)` yields exactly one
  id; stratum `all_single` / `all_fragmented` / `mixed`, table
  printed when mixed), printed beside every classification and
  compared in diff-reports; it never gates the headline. No
  checkpoint fact is asserted: all token-real evidence this
  round is stub-level machinery definition.
7. **Decision rule (derived, with printed characteristics).**
  Per candidate per headline contrast over paired completed
  render-divergent ids-divergent cases (n paired, a = own-only
  passes, b = foreign-only passes, d = a + b): n < 10 →
  `insufficient`; d = 0 → `no_discordance_observed`; d < 6 →
  `below_threshold`; a == b → `discordant_balanced`
  (descriptive only); else two-sided exact McNemar p < 0.05
  (`confidentiality.py:293` `mcnemar_exact_p`) →
  `discordant_asymmetric`, else `below_threshold`. The
  falsification boolean is `discordant_asymmetric` alone. The
  d ≥ 6 floor is forced by α = 0.05 (5–0 gives two-sided
  p = 0.0625; 6–0 gives 0.03125). Operating characteristics
  (research PoC5, verifier spot-reproduced within Monte Carlo
  error): asymmetric fire ≤ 3.6% on the random-labeller null,
  ≤ 3.1% on competent nulls, ≈ 0 under ε = 0.01 jitter, at
  every N tabulated (16–100); power at a 25-point drop is
  0.99 at N = 100 but a coin flip (0.50–0.53) at N ≤ 30 — the
  exact random-labeller fire rate at the run's own n and the
  family-wise expectation (≈ 1.44 false labels per 48-decision
  panel under the random null) are printed beside the
  classifications, and `below_threshold` is labelled weak
  evidence in the section.
8. **Decomposition.** Every discordant pair decomposed as
  `boundary_only` (answer digests equal), `exact_only` (digests
  differ, both boundaries found), or `mixed`; all three counts
  printed per contrast. Per-arm attrition (`n_context_exceeded`,
  `n_evaluation_failed`, `n_render_identical`,
  `n_ids_identical`) printed beside every paired rate.
9. **Vocabulary (frozen).** Classifications exactly
  canonical-form's: `insufficient`,
  `no_discordance_observed`, `discordant_balanced`,
  `discordant_asymmetric`, `below_threshold`. Stage details
  exactly `{cancelled, historyframe_record_budget_exceeded,
  historyframe_screen_failed}` (the last bound to the
  pre-generation screens: S1 per-candidate control identity —
  any mismatch fails the whole stage before any generation;
  S2 frozen-text audit via `audit_special_token_literals`,
  `fixtures.py:471`; S3 budget caps). Per-cell details exactly
  `{"", arm_context_exceeded, arm_evaluation_failed}`.
  Whole-stage failure: `{"available": False, "detail":
  "historyframe_unavailable"}`; the stage never fails the run.
10. **Counts.** 4 generating arms × ≤100 search fixtures ×
    ≤24 candidates = **9,600 generations** legal maximum
    (headline arms only: 7,200); 2,400 multi-cell ground
    records; 12 MiB records cap (house pattern). A run's actual
    candidate count is one family's variants (typically ≤ 8):
    the legal maximum is a cap, not an expectation.
11. **Records and validator.** Digest-only records (hybrid-text
    digest and token count per arm, divergence flags,
    `id_length_delta`, canonical-form-shaped cells). The diff
    validator re-implements the segment renderer, hybrid,
    eligibility accounting, classification, and restricted-set
    selection — never imports them — and recomputes every
    aggregate. Report text passes through `_escape_text`
    (`reporting.py:28`) only.
12. **Claim (the only falsification-class sentence).** "On this
    candidate, on this run's search fixtures preceded by the
    frozen synthetic history, rendering that history's turns in
    the frozen chatml [resp. llama3] framing instead of the
    candidate's own framing produced verdict discordance at or
    above this panel's derived floor, asymmetric by an exact
    McNemar test — a history-frame discordance for these named
    framings and this frozen history, on this checkpoint." No
    sentence in any report may generalise it to histories,
    migrations, or templates in general.
13. **Utility (resubmission, scoped).** A team migrating a live
    product between chat-template families — or importing a
    transcript one client rendered into a session another
    serves — uses this, per candidate, to decide whether a
    carried-over history must be re-rendered into the new
    family's framing before the final turn is trusted, on their
    own search fixtures, before a silent history-frame flip
    changes the template winner they already picked. (The
    sketch's migration sentence is removed for any
    scripted-history instrument, §5 item 6; this scoped version
    is the fresh review's to accept or reject under item 14.)
14. **Name.** `--history-frame` is the resubmission's proposal,
    free as a string (§1 non-fatals); it ships only if a build
    cycle's USAGE naming audit passes, with help text stating
    the prepend in its first sentence so no user reads it as
    reframing their fixtures' native histories.

### Fresh-review agenda (silence on any item = NO-BUILD)

The fresh review must settle, head-on (research §7, verifier
§A4–A5, coordinator additions):

1. Scripted-only history: accept the §3.12 string-scoped claim
   as worth the build surface, or require a self-generated
   variant with its own population discipline and screening —
   it may not be smuggled in as an arm.
2. One dose (k = 6 turns) vs a frozen dose ladder with its own
   decision rule.
3. Native-history reframing (fixtures' own multi-turn
   histories): v1 scope accepted as the first instrument, or a
   deferral of the real one.
4. Verbatim arm: keep at 2,400 generations' legal-maximum cost,
   or cut to three arms and print the BOS bundle as a named
   limitation. Degeneracy (verifier §A4.2): for candidates
   whose own bos is empty, `verbatim_chatml` ≡ `foreign_chatml`
   byte-identically — the arm must be counted inapplicable
   there, never pooled.
5. Small-panel power: at N ≤ 30 the rule is coin-flip powered
   at a 25-point drop (research PoC5); the shipped example set
   (2 search fixtures) classifies `insufficient`, correctly.
   Accept with at-N characteristics printed, or pair with a
   generated panel (re-opening the selection-over-generated-sets
   question) — argued, not defaulted.
6. Boundary-only counting: §3.8 counts verdict discordance with
   the decomposition printed beside it; requiring
   exact-only-counting is permitted only with re-derived
   operating characteristics.
7. The rule's high-jitter behaviour (coordinator execution,
   §4): the McNemar discordant-count rule's null behaviour at
   ε = 0.05 with high pass rates (D = 80 of 100) was not in
   research PoC5's grid; the coordinator's rate-based (Fisher)
   variant held ≤ 1.2% across the ε ≤ 0.05 grid in the same
   simulation harness. The fresh review must print the chosen
   rule's characteristics on the ε ∈ {0.01, 0.02, 0.05} ×
   D ∈ {20, 50, 80} grid before freezing it.
8. Diagonal contrast (RT-A F4): not required for §3.12's
   named-string claim (verifier §A4.3); required if any future
   wording claims mixing-specific harm.
9. Configuration-digest binding (RT-B F9): discharged in §3.11
   via per-arm hybrid-text digests plus the validator mirror —
   the fresh review confirms explicitly.
10. Instrument vs arm (RT-A F9): the fresh review must
    explicitly accept the resubmission as a new instrument
    rather than a `synthhistory` condition, with the §1
    duplication map as the record.
11. Neighbour-tool re-survey (LM Eval Harness, HELM,
    PromptPerfect, OpenAI evals, DSPy) is UNEXECUTED this
    round; the uniqueness combination (§6) stays scoped to
    surveyed sources until a fresh review re-runs it.

## 4. Adjudication evidence (coordinator's own execution)

- **Render refusal / control identity.** `Format.render` on
  `SYNTHETIC_TURNS` raises `ValueError` (final-user contract).
  A generalised per-message-format loop reproducing
  `core.py:76-115` semantics is byte-identical to the native
  render under A == B for all 118 candidate Formats
  (`candidates(family, mutate=True)` summed over 29 families)
  × 2 conversations — and diverges from it under every naive
  segment-restart or `or`-fallback variant tried by the lanes.
- **BOS ownership.** Donor-BOS vs final-BOS splices differ on
  658/812 ordered pairs; char deltas (foreign − control) span
  [−645, +645] over the 6-turn history.
- **Aliased pairs.** Segment criterion (bos excluded): 36
  ordered pairs byte-identical, classes {chatml, qwen3, qwen38,
  kolibri, lfm25, nemotron35}, {gemma, gemma3}, {deepseek,
  deepseek41}, {muse_glimmer, muse_spark} → 21 distinct history
  segments. Segment+bos criterion: 24 ordered pairs. (The two
  counts carry different criteria; §5 item 10.)
- **Boundary mechanism.** Independent stub through the real
  `core.evaluate` (byte backend, chatml boundary): `GREEN` +
  `<|im_end|>` → pass; `GREEN` + `<|eot_id|>` →
  `exact_match=False`, `boundary_found=False`, answer
  `'GREEN<|eot_id|>'`. Reproduces RT-B F6 exactly.
- **Population prevalence.** `examples/fixtures.json`: search
  and held-out each 1 of 2 single-turn.
- **Carrier route.** `Format(B.name, "", "", "", "")` renders a
  single user message byte-verbatim (executed on text containing
  foreign markers and seams) — the resubmission's evaluation
  path is mechanically real.
- **Pair disjointness.** Non-empty framing literals of chatml
  vs llama3/gemma/mistral pairwise disjoint (executed, ∅
  intersections).
- **Decision-rule simulations.** Coordinator Monte Carlo
  (40,000/30,000 trials per cell, seed 20261027): the sketch's
  any-difference falsifier and any (6,·) floor fire ≈ 1.000 at
  N = 100 under the nulls (agreeing with RT-B F3); a
  rate-based Fisher variant of the discordance rule holds
  ≤ 1.2% null fire across ε ∈ {0.01, 0.02, 0.05} at
  D ∈ {20, 50, 80}, with power 0.78–0.95 at a 20-point drop
  for D ∈ {30, 50} — recorded for fresh-review item 7, not
  substituted into §3.7 (the researcher's McNemar rule is the
  candidate; its grid is the gap).
- **Counts.** `candidates(mutate=True)` totals 118 Formats
  (29 bases + 89 variants); `local_repairs` outputs total 107
  *including* bases. The verifier's interim "136 formats"
  enumeration (29 + 107) double-counts the bases; the control
  identity it supported was verified on the correct sets by
  three independent executions, so the fact stands and the
  number does not enter this doc.

## 5. Removals applied (verifier §A6 — none of these enter any build doc)

From the sketch:

1. The any-difference / any-selection-flip falsifier, in any
   form.
2. "Kept verbatim (recommended)" seam semantics and the "BOS
   and closing suffix mid-prompt" description (wrong in both
   particulars).
3. "Disjoint marker sets" as a pair-selection rule (render
   identity is the operative criterion).
4. "Headline population = token-real in both frames, or
   disclosed per cell" (incoherent; contradicts the pooled
   selection headline).
5. "Scoring untouched" and "Size: M (hybrid render + condition
   plumbing)".
6. The migration-decision utility sentence, for any
   scripted-history instrument.
7. "Every message's text byte-identical" as a prompt-level
   claim (true only of the message list).

From the briefs (verifier corrections, coordinator-confirmed):

8. RT-A's think-scaffold count "five families" → **nine**
   (verifier-executed; §1 F4 lists them).
9. RT-B's "107 local-repair variants" → **118 candidate Formats
   total, 89 beyond the 29 bases** (107 is the `local_repairs`
   output count including bases).
10. Either aliased-pair count (36 / 24) quoted without its
    criterion (segment / segment+bos).
11. RT-A's "only BOS-of-the-history-family reproduces the
    control when A == B" (any single-BOS-at-head policy does;
    the substantive divergence for A ≠ B stands).
12. Any upgrade of the uniqueness sentence (§6) beyond
    "combination, scoped to surveyed sources" while the
    neighbour-tool re-survey is unexecuted.
13. Any token-real statement about a real checkpoint (all
    token-real evidence this round is stub-level).
14. Research PoC2's absolute hybrid totals quoted without their
    tail fixture (the verified quantities are the deltas).

## 6. Sources the resubmission may safely cite

- Wei, Zhao, Gong, Chen, Xiang, Zhu, arXiv:2405.20234 ("Hidden
  in Plain Sight") — verified at the primary listing this round
  (title, authors, attack substance, 97% ChatGPT figure in the
  abstract). Fencing: adversarial injected history, attack
  success readout, model varied against attacker token sets;
  already the in-tree citation of `faketurn.py:7-14`, whose
  Table-4 figures are carried there and were not re-derived
  this round.
- Jiang, Xu, Niu, Lin, Poovendran, arXiv:2406.12935 ("ChatBug")
  — verified at the primary listing this round (format-mismatch
  + message-overflow attacks, eight aligned LLMs,
  adversarial-training mitigation at a performance cost).
  Fencing: whole-prompt, adversarial, safety readout; never
  isolates the history segment, never holds content benign and
  identical, no selection readout.
- Anil et al., *Many-shot Jailbreaking* (Anthropic, Apr 2024) —
  in-tree anchor via `synthhistory.py:9-11` and USAGE §8.20;
  NOT re-resolved this round and not load-bearing here. A fresh
  review wanting it load-bearing resolves it primary-first.
- In-tree precedents: `synthhistory.py` (population and
  per-condition selection), `multiturn.py:16-46` (authorship
  settlement and screening), `canonicalform.py` (divergence
  gating, restricted-set selection, winner vocabulary),
  `negation.py` (derived floors, decomposition, frozen
  vocabulary), `confidentiality.py:293` (McNemar),
  `deliforge.py:238-281` (encoding-pair operationalisation),
  `prefill.py:19-24` (seam tokenisation discipline).

The uniqueness statement, if the resubmission is ever built, is
the combination: segment-level hybrid rendering with a
byte-identical own-frame control, paired per-fixture verdict
discordance under a panel-derived classification, conservative
selection re-applied per frame condition on a common completed
set, and token-encoding stratum plus render/ids divergence
accounting on every contrast — scoped to the sources surveyed
above, never "first".

## 7. What a build cycle must NOT do on the basis of this doc

- Do not build the circulated sketch: its falsifier, verbatim
  seam semantics, native/unspecified history population,
  token-real headline population, or disjoint-marker pair rule.
- Do not build the §3 resubmission on this doc's authority: it
  enters only through a fresh blind design review (researcher +
  red-team pair + verifier) that settles every item of the
  fresh-review agenda in §3, silence on any item being
  NO-BUILD.
- Do not ship a flag under `--history-frame` or any other name
  until that review returns BUILD and the build cycle's USAGE
  naming audit passes.
- Do not quote any figure from this doc without its criterion,
  panel N, or tail fixture, as flagged in §5.

# Part III — Fresh-review adjudication (coordinator #61)

# Fresh-review adjudication — history-frame §3/§6 resubmission (coordinator #61, 2026-10-28)

**Verdict: NO-BUILD as written. BUILD-as-amended**, under the parent's
amendment test: every required amendment below is text, pins, reporting,
record-binding, or screen-strengthening inside the frozen semantics;
none touches the endpoints (arms `own` / `foreign_chatml` /
`foreign_llama3` / `verbatim_chatml`; the single k=6 prepend; the frozen
chatml/llama3 donor pair), the population (the run's search fixtures
preceded by synthhistory `history_6`, divergence/completion gating as
designed), or the headline scorer (byte-exact ∧ boundary cell verdict;
the McNemar-gated classification with floors n≥10, d≥6 and
`discordant_asymmetric` as the sole falsification boolean). A build
that omits any of amendments A1–A10 is not this verdict's build. If
BUILD: do not build in this round — the parent sequences build cycles.

Candidate of record: research.md §6 (`~/workspace/agents_out/
historyframe/`), summarised as §3 of this design doc. Briefs +
coordinator verification in `~/workspace/agents_out/historyframe2/`
(researcher.md, redteam_a.md, redteam_b.md,
coordinator_verification.md). Every load-bearing claim below is the
coordinator's own execution or was cross-executed by two parties.

## What survived blind attack (executed)

- **Splice machinery is real.** The §6.3 segment/hybrid semantics,
  implemented independently from the formal text, reproduce
  `Format.render` byte-identically under A == B: coordinator 354/354
  (118 candidate Formats × 3 prepended message sets); red team A
  1062/1062 (× 9 sets incl. repeat_bos, open_suffix, adversarial
  contents). The carrier-Format evaluation path verifies through the
  real `core.evaluate` (coordinator + round 1): carrier renders the
  hybrid verbatim; the own arm's verdict and `input_text` equal a
  native evaluation exactly; a foreign arm flips a scripted stub;
  the foreign-terminator decomposition inputs are the carried cell
  fields. Request-seed identity across arms holds under the §6.3
  original-id pin and breaks under arm-suffixed ids (executed).
- **Population is named decisively.** §6.2: search fixtures only,
  prepended with `build_history_fixtures(condition="history_6")`;
  every fixture evaluable regardless of native turn count (round-1
  F1 answered by construction). Frozen turns executed clean: no
  digits, no special-token literals, none of the example fixtures'
  expected answers. On the shipped example set (2 search fixtures)
  every contrast classifies `insufficient` — the honest behaviour,
  executed.
- **Decision rule OC confirmed by exact enumeration.** Random-null
  asymmetric fire: n=16 0.0196, 24 0.0243, 30 0.0278, 50 0.0296,
  100 0.0367 (PoC5's MC rows confirmed; three parties agree).
  Competent-independent: ≤0.0291 at every N. Power: 25-point drop
  0.52 @ N=30, 0.988 @ N=100; 50-point 0.983/1.000. The d≥6 floor's
  derivation is correct (p(5,0)=0.0625, p(6,0)=0.03125, via the
  tree's `mcnemar_exact_p`, confidentiality.py:293).
- **Agenda item 7 (jitter grid) settled FOR the frozen rule.**
  Symmetric two-arm jitter, N=100: fire 0.0004 / 0.0048 / 0.0213 at
  ε = 0.01 / 0.02 / 0.05 (coordinator MC + red team B exact
  trinomial; researcher conditional-on-D enumeration ≤0.0414 and MC
  worst cell 0.0219 agree). D-invariant under the symmetric model —
  the model matching this stage, where both arms are freshly
  generated in one run. No Fisher substitution is needed.
- **Duplication kills fail.** Token-depth
  (`experiments.py:3621 prompt_token_depth_sweep`) truncates
  trailing token ids in a single framing — a different axis by
  direct code comparison. Synthhistory's stage renders prepended
  fixtures whole via `candidate.render` and has no slot for a
  foreign segment; the residue (bos-native field splice over a
  genuine history segment + paired cross-frame classification +
  per-frame restricted-set selection) is a new instrument
  (agenda item 10 accepted). Length collinearity within a contrast
  is real (the segment is fixture-invariant) and is fenced by
  §6.1/§6.10: length and encoding are named consequences, the
  claim subject is the named framing pair of strings.
- **Neighbour re-survey executed** (researcher; agenda item 11):
  LM Eval Harness, HELM, OpenAI Evals, DSPy — none ships
  segment-level cross-format history reframing with paired verdict
  discordance + per-frame selection. PromptPerfect UNRESOLVED
  (primary fetch blocked; never cited). Wei arXiv:2405.20234 and
  Jiang arXiv:2406.12935 re-resolved at the primary listings;
  fencing holds. Uniqueness = the combination, scoped to surveyed
  sources, never "first".

## Agenda settlements (§3 tail items 1–11)

1. Scripted-only history: ACCEPTED — §6.10's claim is scoped to the
   frozen transcript; a self-generated variant is a different
   instrument (multiturn's authorship settlement), not an arm.
2. One dose (k=6): ACCEPTED — the claim carries no dose language;
   the single dose is printed scope.
3. Native-history reframing: DEFERRED by design — v1's decisiveness
   is exactly its refusal of round-1 F1 heterogeneity.
4. Verbatim arm: KEPT, with amendment A1 (below).
5. Small-panel power: ACCEPTED with at-N characteristics printed
   (§6.6 freezes this; executed on the shipped example set).
6. Boundary-only counting: ACCEPTED as written — the verdict is the
   tree's conjunct, the decomposition is printed beside every
   classification, and the OC was derived for this counting.
7. Jitter grid: SETTLED for the frozen McNemar rule (above);
   printed per amendment A6.
8. Diagonal contrast: NOT required for §6.10's named-string claim.
9. Configuration-digest binding: discharged in substance (per-arm
   hybrid-text digests + validator mirror), strengthened by
   amendment A4.
10. Instrument vs arm: ACCEPTED as a new instrument (above).
11. Neighbour re-survey: EXECUTED (above); uniqueness stays scoped.

## Defects in the candidate as written (each executed; all amendment-class)

1. **Family-wise constant wrong (RT-B F1; coordinator exact).**
   §6.6 prints "≈1.44 false labels per 48-decision panel, P(≥1)
   ≈0.77" — that is 48 × a rounded 0.0300. At the tabulated
   random-null rate (exact 0.0367 @ n=100) the expectation is
   **1.76** and P(≥1) **0.83** (researcher's corrected enumeration
   concurs). The wrong constant is the one §6.6 mandates printing.
   → **A2.**
2. **n-floor derivation false (RT-B F2; coordinator brute force).**
   A 6–0 split with both arms' pass rates strictly interior is
   first attainable at **n=8** (concordant passes = 1), not 10.
   The floor value 10 is conservative and every printed OC figure
   is derived with it in place; only the derivation sentence is
   false. → **A3.**
3. **Verbatim arm unaccounted on its degenerate half (RT-A F2,
   researcher concession, coordinator — three executions).**
   chatml's bos is `""`, so `verbatim_chatml` ≡ `foreign_chatml`
   byte-identically for **55/118 candidate Formats** (46.6%),
   exactly the bos-equality set. §6.5's render/ids-identical
   accounting covers headline contrasts only; the verbatim pair
   has none, so a build would print a "BOS-substitution share"
   that is jitter by construction for those candidates. The
   §6.3 formula "hybrid with B.bos removed" also fails to pin
   leading-only removal (17 repeat_bos candidates carry internal
   B.bos occurrences). → **A1.**
4. **Ids-gate not validator-decidable (RT-A F1, sustained in
   part; coordinator E7).** §6.9's records carry the ids-divergence
   only as a flag plus per-arm token counts; equal counts do not
   imply equal id sequences, so the headline-population gate
   (ids-identical exclusion) rests on an unverifiable carried
   flag — RT-A's executed forgery (flipping the flag on a (7,0)
   set) moves `discordant_asymmetric` to
   `no_discordance_observed`, validator-consistent. The shipped
   discipline exists for exactly this (negation.py:172
   `digest_ids`; sibling records carry per-arm id digests).
   Classification/decomposition/selection inputs are otherwise
   fully carried or recomputable (segments from in-tree
   constants; cells as primitives — the canonical-form posture).
   → **A4.**
5. **Own-arm verdict-identity overclaim + construction pins
   (RT-A F4; coordinator E4).** Design §3 item 4 / research §6.4
   assert the own arm is "byte- and verdict-identical to the
   synthhistory history_6 condition". Byte-identity holds;
   verdict-identity does not: the shipped synthhistory stage runs
   namespaced ids (`synthhistory/history_6/<id>`), hence different
   request seeds — a seed-sensitive stub flips verdicts on
   identical text (executed). §6.3's original-id pin is the right
   choice for cross-arm identity; the cross-stage claim is what
   fails. The stage fixture's `expected`/`group` are also unpinned
   in §6 (empty group raises, executed). Separately, §6.2's
   "task-free" overstates the frozen turns' discipline (they
   carry conversational-commitment phrasing, common-mode across
   arms; synthhistory's own scoping is answer-vocabulary /
   task-content freedom). → **A5, A10.**
6. **S1 screen is diagonal-only (RT-A F3, executed).** A
   seam-normalising renderer mutant (insert `"\n"` at the seam
   iff the history segment lacks one) corrupts foreign renders
   while S1 passes for 51 candidates — S1 as specified checks
   only `hybrid(B,B)`. → **A9** (per-arm slice-swap identity:
   every arm's hybrid must equal the native render with only the
   segment slice exchanged — checkable pre-generation from the
   S1 native render + the frozen segment renders; assembly-level
   seam corruption then fails the stage for every arm).
7. **Correlated-null column under-specified (RT-B F4;
   coordinator two readings, RT-B a third).** PoC5's "mixture
   0.97/0.5" states no weights or sharing structure; readings
   spread 5× at N=30 (0.0003–0.0302 across parties). Every reading
   keeps competent fire ≤0.031 at every N, so §6.6's printed
   bound holds; the column itself is irreproducible as printed.
   → **A7.**
8. **Non-exchangeable channels unnamed (RT-B F5, adjudicated).**
   A fixed directional arm penalty is the design's *alternative*
   under §6.10's string-scoped claim, not a null — rejected as a
   null. The stochastic arm-level batch shock is a real run-level
   mechanism whose effect lies inside the string-scoped claim;
   like instruction-distance's proximity channel, it must be
   named in the section's residuals so the label is never read
   as a mechanism certificate. → **A8.**

**Not sustained:** RT-B F6c (selection headline has no n-floor) —
canonical-form's `arm_selection`/`winner_headline`, read directly,
withhold only on empty/mismatched restricted sets; §6.6 transplants
the shipped discipline verbatim, and falsification rests on the
floored classification alone. The observation is recorded as a
side finding about the shipped precedent. RT-A F5 (fragmented-
stratum seam merges) — real as executed, but the single-encode
construction is the faithful one for a text-level carried
transcript, and the encoding stratum is printed beside every
classification precisely to disclose it; recorded as a side
finding, not a defect. RT-B F7's composition argument proves
feasibility from shipped parts + the §6.3 spec (as did round 1's
PoCs and this round's E1/E2) — feasibility was never in dispute
and is not duplication of a shipped measurement.

## Binding amendments (A1–A10)

- **A1.** Verbatim accounting: per candidate, if the donor bos
  equals the candidate's bos, `verbatim_chatml` is byte-identical
  to its headline twin — the arm is recorded inapplicable for
  that candidate (counted, printed), and its discordance is
  never pooled or read as a BOS share of zero. The verbatim
  construction removes only the *leading* B.bos. The
  single-donor asymmetry (no `verbatim_llama3`) is printed as
  scope.
- **A2.** The section's family-wise expectation is computed from
  the exact at-n random-labeller fire rate it already prints,
  times the run's actual decision count (at n=100, 48 decisions:
  ≈1.76, P(≥1) ≈0.83 under the random null) — never the frozen
  1.44 constant.
- **A3.** §6.6's n-floor sentence corrected: floor 10 stands as a
  frozen conservative constant (smallest interior-attainable n
  is 8); the printed OC table, derived with the floor in place,
  is its justification.
- **A4.** Records carry per-arm token-id digests (negation's
  `digest_ids` pattern); the validator derives ids-identical
  from digest equality and rejects flag/count/`id_length_delta`
  inconsistencies. The payload carries the sibling-pattern
  protocol (seed, backend fingerprint, donor pair, k, stratum
  table) so every compared field is present.
- **A5.** Construction pins stated in §6.2/§6.3: the stage takes
  `build_history_fixtures`' messages and re-wraps them under
  each fixture's original id, original `expected`, and the
  module's frozen group label; the built fixtures' namespaced
  ids are not used as stage fixture ids. §6.2's history
  description scoped to synthhistory's discipline (free of
  answer vocabulary and task content; conversational phrasing
  is common-mode across arms).
- **A6.** The section's OC paragraph prints the agenda-7 jitter
  grid (symmetric model: ≈0.0004 / 0.0048 / 0.0213 at
  ε = 0.01 / 0.02 / 0.05, D-invariant; one-sided variants
  bounded and named as arm-asymmetric perturbations, i.e.
  inside the alternative).
- **A7.** The correlated-null model is stated in the design doc
  with weights and sharing structure and its column re-derived,
  or the column is dropped; the printed bounds (random ≤3.6%,
  competent ≤3.1%) stand under every executed reading.
- **A8.** The section's residuals name the arm-level directional
  channels (batch composition / numerics correlated with the
  foreign prompts' length) beside §6.1's length/encoding
  consequences: the label certifies string-level discordance on
  this run, not a mechanism.
- **A9.** S1 strengthened to the per-arm slice-swap identity
  (defect 6); its rationale sentence scoped to what the
  strengthened screen plus the validator mirror jointly cover.
- **A10.** The own-arm claim corrected in §6.4 and design §3
  item 4: byte-identical render to synthhistory's `history_6`
  render of the same messages; verdict-identity with the
  synthhistory *stage* is not claimed (namespaced ids ⇒
  different request seeds); the stage's baseline is its own
  `own` arm.

**Naming (binding condition, from §3 item 14 + RT-B F8).** The
name is not settled by this review. `--history-frame` is free as
a string (zero collisions, executed), but RT-B's objection is on
record: the bare name strips the prepend scope that
`--synth-history` carries, and the instrument reframes only the
prepended synthetic segment. A build cycle's USAGE naming audit
must either adopt a prepend-carrying name or pass the bare name
with the first-sentence prepend disclosure in help and section
text; shipping the bare name on this verdict alone is not
authorised.

## Answers to the standing questions

- **Monday-morning utility (one sentence).** A team carrying a
  transcript across chat-template families uses this, per
  candidate on its own search fixtures preceded by the frozen
  synthetic history, to decide whether a history rendered in the
  old family's framing can be trusted before the final turn or
  must be re-rendered first — before a silent history-frame flip
  changes the template winner it already picked.
- **Uniqueness.** The combination — bos-native segment splice
  with a byte-identical own-frame control, paired per-fixture
  verdict discordance under a panel-derived McNemar
  classification, conservative selection re-applied per frame
  condition on a common completed set, and render/ids divergence
  plus encoding-stratum accounting on every contrast — scoped
  to the sources surveyed this round, never "first".
- **Duplication by direct code comparison.** Executed against
  synthhistory (+ its stage), token-depth
  (experiments.py:3621), multiturn, canonical-form: no shipped
  machinery measures a foreign framing of a genuine history
  segment; the composition fragments are shipped, the
  measurement is not.

## Fabrication sweep

**CLEAN for this round's briefs**, with one recorded episode: the
researcher's first delivered summary carried rule-enumeration
numbers that disagreed with its own script's genuine completion;
the researcher caught the disagreement by hand-check, repaired
the brief on disk, and re-delivered with the verified values —
which match the coordinator's independent exact enumeration
(random 0.0367 @ N=100; competent 0.0291; family-wise 1.76 /
0.83). Every load-bearing number in every brief that entered
this adjudication was re-executed by the coordinator or
cross-executed by two parties (verbatim degeneracy 55/118 ×3;
jitter grid ×3; control identity ×2; segment classes ×2). The
one irreproducible figure in the record is the ROUND-1 PoC5
correlated column — a candidate-document under-specification
(amendment A7), not a this-round fabrication. No invented
symbol or citation found in any brief; PromptPerfect is
reported unresolved by the researcher and is cited nowhere.

## Build cost (if the parent sequences a build; estimate, UNVERIFIED, for sequencing only)

Sibling-scale: a new `historyframe` module (splice renderer +
screens), experiments stage on the carrier pattern, CLI flag +
pin audits, reporting section (three touchpoints), diffreports
validator mirror + blast radius (test_cycle67 oracle/flip
anchor/`_identical_expected`, test_cycle27 stub, test_cycle44
pin), design-doc consolidation — ≈2,500–3,500 lines including
tests, by analogy with the cycle-218 instruction-distance build.
Generations: 9,600 legal maximum as frozen (headline arms
7,200); records 2,400 under the 12 MiB house cap.

## Side findings (reported, not fixed — read-only round)

- Fragmented-stratum cross-seam token merges are real on actual
  seam strings (RT-A F5, executed); disclosed by the stratum
  label, but §6 records no per-case seam-canonicality field —
  a build may add one descriptively without touching the
  headline.
- Canonical-form's winner headlines (shipped) are
  single-flip-sensitive at tiny common completed sets (RT-B
  F6c); a precedent-level observation about the shipped
  instrument, not a §6 deviation.
- Design §3 item 4's summary sentence carries the same
  verdict-identity overclaim as research §6.4 (amendment A10
  covers both texts).
- The instrume
...[truncated 10273 chars]
