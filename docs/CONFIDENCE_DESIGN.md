# Verbalized-confidence × template (`run --confidence`) — design

Cycle 199 (wave-8 rank #8, the last wave-8 item). This document
is the shipped contract. It folds the wave-8 research brief
(docs/RESEARCH_LOG.md, wave-8 round), the cycle-199 researcher
pass (primary sources re-verified), red-team A's ten findings
(F1–F10) and nine amendments (A1–A9), red-team B's disproofs of
A's positive amendments (its own amendments B1–B7), and a
verifier pass that re-executed both red teams' probes and
confirmed every load-bearing claim (V1–V12). Where the brief
and the shipped contract differ, this document wins, and the
differences are the adjudicated amendments, recorded below.

## What this is, honestly framed

`--confidence` appends **one frozen suffix** to each search
fixture's last user message, asking the model to end its reply
with a line of the form `Confidence: <integer 0–100>`, and
re-evaluates every candidate template on the search split. The
elicited text is parsed under a **frozen grammar**; the stated
number is then set against the exact-match verdict the lab
already trusts, per candidate: how often the model complies,
what it states when it is right vs wrong, and what the suffix
itself costs (verdict flips against the recorded native run).
A **generated pair** runs alongside: planted-fact cases whose
answer the prompt states (knowable) vs the same cases with the
payload binding a *different* key (donor-key control), testing
whether the stated number drops when the answer is not
knowable from the prompt.

Monday-morning utility: a practitioner serving a checkpoint
behind a template learns whether the confidence number the
model states under that template tracks whether it was right —
and whether it drops when the answer is not in the prompt —
before exposing that number to anyone downstream.

A stated confidence is **model output text, not a
probability**, and nothing in this probe turns it into one.
The readouts describe *agreement* between the stated number
and the measured verdict under each template. No readout is a
calibration error; the words "calibrated", "overconfident",
and "ECE" appear nowhere as findings (the verifier confirmed
no shipped section uses them). What would change the picture:
a stated number that separates correct from incorrect
compliant answers *and* drops on the donor-key control is
evidence the number carries answer-dependent signal under this
instrument; a number that does neither is evidence it is
theatre under this instrument. Both are claims about this
checkpoint, this fixture set, this suffix, and this seed —
nothing more is claimed.

## Prior art and the scoped uniqueness claim

Verbalized confidence is a studied elicitation method; the
lab's contribution is not the method but **the combination**:
(1) per-template candidate crossing — prompt format as the
independent variable, the stated number compared across the
lab's candidate templates on identical fixtures; (2) the
confidence line **separated from the scored span** under a
frozen grammar, so elicited generations are scored by the
lab's ordinary exact-match verdict on the span (published
studies score generated text with a judge model, or score
log-probabilities instead); (3) **verdict flips against the
recorded native run** — the elicitation cost of asking, which
no surveyed study reports; (4) a **donor-key knowability
control** — a paired generated control in which nothing false
is stated and the asked fact is simply absent; and (5) the
result shipped as a **fail-closed diff-validated section** of
the lab's deterministic report, recomputed from digest-bound
ground records. No surveyed tool or study ships this
combination (researcher pass, cycle 199: lm-evaluation-harness,
HELM, Inspect AI, OpenAI Evals, promptfoo, DSPy — Inspect's
primitives could assemble something similar, but primitives are
not a shipped measurement).

The neighbours, named so a reviewer does not have to discover
them, with the researcher's corrections applied:

- Kadavath et al., "Language Models (Mostly) Know What They
  Know" (arXiv:2207.05221): logit-based self-evaluation
  (P(True)/P(IK)), **not** verbalized confidence; cited only
  for the logit line of work, never as a verbalized precedent.
- Lin et al., "Teaching Models to Express Their Uncertainty in
  Words" (arXiv:2205.14334): the origin of "verbalized
  probability", and a **fine-tuning** result (GPT-3 trained on
  CalibratedMath). Zero-shot untrained calibration is not
  shown there and is not claimed here.
- Tian et al., "Just Ask for Calibration" (arXiv:2305.14975):
  the method ancestor (answer-then-confidence elicitation),
  with two differences that forbid numeric comparison: their
  scale is a float 0.0–1.0, and their correctness is GPT-judged,
  not exact-match. Their closing open question — how sensitive
  the method is to the prompt — is part of what a
  per-template crossing speaks to.
- Xiong et al., "Can LLMs Express Their Uncertainty?"
  (arXiv:2306.13063): the lab's exact surface form (integer
  0–100, "Answer and Confidence (0-100)") and the documented
  shape to expect: stated values cluster at multiples of 5 in
  the 80–100 range with a peak at 95, and vanilla elicitation
  is heavily overconfident by their (judge-scored) measure.
  Top-bin-dominated, sparse low bins are the pre-registered
  expected shape here, not a surprise.
- Seo et al., "ADVICE: Answer-Dependent Verbalized Confidence
  Estimation" (arXiv:2510.10913): a **primary fine-tuning
  study** (an earlier brief miscited it as a survey; the
  researcher corrected the genre and the verifier confirmed it
  against the paper). Its §7 answer-masking experiment —
  replace answer tokens with padding and watch stated
  confidence — is the nearest published neighbour of this
  probe's knowability control, and is cited by name for that
  reason: the donor-key control differs by manipulating the
  *prompt's evidence* instead of the model's own answer tokens,
  and by needing no fine-tuned model. ADVICE's five-format
  taxonomy includes ScorePercent; the lab's surface form is
  literature-standard, claimed as such, not as an invention.
- Damani et al., "Beyond Binary Rewards" (arXiv:2507.16806):
  RLCR, the RL-with-calibration-rewards training method
  (verbalized-confidence ECE 0.37 → 0.03 on HotpotQA after
  training). An earlier brief attributed that number to a
  different paper (arXiv:2604.26152, a single-author
  observability survey); the researcher refuted the
  attribution and the verifier confirmed Damani et al. as the
  primary source against the full text. Training a model to
  calibrate is the opposite direction from measuring an
  untrained checkpoint, and no number from it is comparable
  to anything this probe prints.
- Yang, Tsai & Yamada, "On Verbalized Confidence Scores for
  LLMs" (arXiv:2412.14737): the strongest published
  prompt-method-sensitivity result for verbalized confidence —
  reliability "heavily depends on how we ask for it", with
  method rankings inverting between small and large models.
  Named here because it is the published reason a
  per-template crossing is a measurement and not a formality.

Cross-study numeric comparison to published calibration
numbers is **invalid** and the docs must not invite it: the
correctness ground truths differ (exact match here, judge
models or human labels elsewhere).

## The instrument

**Suffix and surgery.** The suffix is a frozen constant
(`confidence.SUFFIX`): "Answer the question, then state your
confidence that your answer is correct. Put the confidence on
its own final line as a whole number from 0 to 100, in exactly
this form:\nConfidence: <number>". It is appended to the last
user message only, as `content + "\n\n" + SUFFIX` (dateline's
joiner and surgery pattern; id, group, and expected are
unchanged, so the per-fixture seed is the same as the native
run's). The placeholder line in the suffix is deliberately not
a grammar line: a model echoing the instruction verbatim is
`malformed_line`, never compliant. Before any generation, the
suffix is screened against every candidate's special-token
inventory (`fixtures.audit_special_token_literals`, the
repetition precedent): a collision fails the stage closed
with `confidence_screen_failed`, because a framing that turns
the suffix text into a special token would forge the
elicitation.

**The grammar and the parser.** A compliant final line matches
`^Confidence: (0|[1-9][0-9]?|100)$` exactly — ASCII digits, no
sign, no leading zeros, no surrounding whitespace. The regex
is load-bearing, not pedantry (executed in the red-team
passes): `int()` accepts `+85`, `8_5`, ` 85`, and non-ASCII
digits, and an unflagged `\d` matches fullwidth digits, so any
strip-and-int parser manufactures compliance the grammar does
not grant. The parser is a **total function** over all
strings, pure `str` (it never encodes, so a lone surrogate
cannot raise), with five outcome codes and a **final-line-first**
precedence, adjudicated in cycle 199 over red-team A's
count-first alternative:

- lines come from `split("\n")` with exactly one trailing empty
  element dropped (`splitlines()` is never used — it also
  splits on U+2028/U+0085 and would manufacture a final line);
- a grammar-matching final line: `ok` if it is the only grammar
  line in the answer, `multiple_lines` if any earlier line also
  matches;
- else a final line starting with `Confidence:` (exact) or
  `confidence:` (case-insensitive): `malformed_line` — even
  when an earlier line matched (the model's last act decides);
- else any grammar line elsewhere: `confidence_not_final`;
- else `missing_line`.

The **span** is the answer with the compliant line removed for
`ok` cells, and the whole answer for every other code. A cell's
**verdict** is one definition everywhere: `span_digest ==
expected_digest and boundary_found`. Compliance and correctness
are separate axes and the joint table keeps them apart: a
correct answer that never stated a number is correct *and*
non-compliant; the verdicts recorded on suffixed generations
by the raw evaluator are meaningless (the suffix text is part
of the answer), which is why every verdict here is re-derived
from the span.

**Screens.** A fixture is screened out of every elicitation
population — its record carries the class and no cell, for
every candidate identically — when: its expected answer is
blank under `strip()` (`expected_blank`; a whitespace-only
expected is "correctly" answered by whitespace, the vacuous
pole); its expected answer already contains a grammar line
**anywhere** (`expected_confidence_shaped`; red-team B's
executed counterexample showed a final-line-only screen lets
an honest generation gain a second grammar line through no
act of the model's); its last user message is vacuous
(`vacuous_last_user`; the suffix would be the whole turn,
measuring suffix echo, not elicitation — red-team B forced
the exclusion over red-team A's keep-or-exclude ambiguity);
or it has no user turn (`no_user_turn`). Screens are computed
by the stage and recorded in the protocol; the validator
checks protocol/record consistency (see Residue).

**The generated pair.** 24 planted-fact cases, derived from
the run seed: each binds a synthetic key (`key-<8 hex>`) to a
6-hex value, pairwise distinct by construction. The knowable
arm's payload states the asked key's true binding; the
control arm's payload states the **next case's** key and
value. The control is red-team B's donor-key construction,
and what it replaced matters: red-team A's original repair
replaced the stated value with a false one, and B executed a
definitionally payload-faithful backend against it — the
faithful model reads the only stated value, answers it at
Confidence 95, and the falsification boolean convicts correct
behaviour of theatre (95 ≥ 95 fires backwards). Donor-key
states **nothing false about the asked key**; the asked
binding is simply absent. Pre-registered expectation: a model
whose stated number tracks knowability states a lower number
on the control arm; a model confidently answering the donor
value there is copying the only value it can see — genuine
answer-independence, the ADVICE mechanism, measured from the
prompt side. Stated values are unguessable (16⁻⁶) and
unmemorizable (seeded hex), so the knowable arm measures
reading and the control its absence. The generated cases are
**instrument cases**, labelled as such in the report; they are
not operator fixtures.

## The readouts

Per candidate, over the operator population (unscreened search
fixtures): completed and compliant counts with the compliance
rate; the compliant × correct joint table; mean stated
confidence over compliant cells, split by verdict, and their
**separation gap** — descriptive only, printed beside the
compliance rate and the joint table, and **withheld with a
named reason** (`no_compliant_cells`, `no_correct_compliant`,
`no_incorrect_compliant`) when a class is empty, because the
three situations the bare `None` would conflate are
themselves findings. The frozen four-bin table
(`[0,24] [25,49] [50,74] [75,100]`, edges in the protocol
fingerprint) counts compliant cells per bin with per-bin
accuracy (withheld at n = 0). Elicitation cost: verdict flips
against the recorded native run, split by direction
(`pass_to_fail` / `fail_to_pass`), over fixtures with a native
verdict and a completed elicited cell.

Per candidate, over the generated population, each arm gets
the same block. Between the arms, the **knowability
readout**: over cases where *both* arms are compliant, the
paired means and `falsified = control_mean >= knowable_mean`.
The boolean is withheld (`insufficient_paired_compliant`)
below **6 paired compliant cases** (placement's
`MIN_PAIRED_EVALUABLE` floor precedent: floors gate readouts,
never counts). Two scope statements ship with it, in the
report text and here: the boolean conditions on joint
compliance — it asks only "among cases where the model stated
a number both times, did the number drop?" — and a control arm
whose compliance collapses is *not* folded into the boolean;
it is reported as its own per-arm compliance count beside it.
The word "elicited" never names a population mean in this
design (red-team B showed the brief's `elicited_mean` was
ambiguous between operator and generated arms); the boolean
compares the two generated arms only.

## Payload and diff validation

Ground records are digest-bound: one record per (candidate,
fixture) and one per (candidate, generated case). A cell
carries `answer_digest`, `span_digest`, `boundary_found`,
`code`, `confidence`, `final_line`, `final_line_length`,
`input_tokens`, and the derived `verdict`. Red-team A's
original record shape carried the confidence as a bare
attested integer, and red-team B **executed the forgery**:
editing only the integers moves the gap +5 → ±100 and flips
the falsification boolean while passing every consistency
check A proposed, because the residue was the probe's headline
axis. The shipped binding (B2) closes it:

- an `ok` cell carries its final line verbatim. The line is one
  of exactly 101 strings and a bijection with the scalar
  already in the record, so carrying it discloses zero
  additional information — and the validator **re-runs the
  grammar over it and derives the scalar**, rejecting any
  record whose scalar disagrees. Stage/validator parser
  divergence and internally inconsistent records die with it.
- a `malformed_line` cell carries the first 32 characters of
  the final line plus the full line length — exactly what
  re-verifying the class takes (the classification prefix is
  11 characters; a grammar line is at most 15).
- every other code's final line is ordinary answer text and is
  never carried.
- digest relations are code-dependent and unconditional:
  `span_digest == answer_digest` for every non-`ok` code,
  `!=` for every `ok` cell — **with no exception**: red-team
  A's draft excepted the answer that is exactly one grammar
  line, and B and the verifier both executed the disproof
  (sha256("") ≠ sha256("Confidence: 50")).
- scalar typing is strict: `type(confidence) is int` — a JSON
  `true` must not validate as confidence 1 (the tree's F2
  discipline).

The generated case set is **regenerated from the protocol
seed** by the diff validator's own mirror and must match the
protocol exactly (ids, key digests, expected digests, both
payload digests), so the control arm's construction is bound,
not attested. Caps are enforced before any recompute (2,400
operator records, 576 generated records, 12 MiB combined — the
combined-bytes measure is the stage's own construction,
repeated verbatim by the validator). Every derived value is
recomputed from the records by the validator's independent
mirror and compared type-strictly; the diff section reports
per-candidate compliance/gap/flip drift, knowable and control
mean drift, and falsification drift, with protocol or backend
drift marking the comparison not comparable.

**Residue, named.** What remains stage-attested: that the
carried line was the generation's *final* line and the *only*
grammar line (finality and multiplicity have no textual
artefact that may be carried — the final lines of
`missing_line`/`confidence_not_final`/`multiple_lines` cells
are ordinary answer text); the span split itself; the screen
classes (the fixture texts are not in the payload, so screens
are checked for protocol/record consistency, not recomputed);
and the digest↔text relation every digest-bound probe in the
tree shares. The diff validator's standard is the
answer-collision standard: everything with a checkable
artefact is derived, and what cannot be is named here rather
than silently trusted.

## Cost

One full extra search evaluation (2,400 generations at the
legal maximum: 24 candidates × 100 fixtures) plus the
generated paired control (2 arms × 24 cases × 24 candidates =
1,152) — **3,552 extra generations** in total, each on a
suffix-lengthened prompt. (Red-team A's draft priced the
generated arms as 576 *records*; generations and records
carry different constants — placement.py's own comment — and
B's recount, confirmed by the verifier, is the figure above.)
No second checkpoint load, no teacher forcing. Defence in
depth: the records caps above are also the budget the stage
enforces before evaluating (`confidence_cell_budget_exceeded`).

## Guards and failure vocabulary

The CLI wrapper adopts the choice-audit wrapper gate verbatim
(the verifier confirmed the shipped wrappers' conditions):
a failed, cancelled, or truncated native search gates the
probe to zero forwards with `native_search_incomplete`. Stage
unavailability details: `confidence_screen_failed`,
`confidence_cell_budget_exceeded`,
`confidence_records_budget_exceeded`,
`native_search_incomplete`, `cancelled`. Per candidate, a
failed evaluation degrades that candidate's cells alone
(absent cells, never fabricated ones); a candidate with no
completed cell anywhere is listed in
`unavailable_candidates`, a set the validator re-derives
exactly. Scope: the search split only — held-out fixtures are
never treated; the probe never touches the decision, the
selected candidate, or the exit code; and **confidence cells
are outside `verify --determinism` scope** — the determinism
certificate covers the recorded native search observations,
and the report section states so in the same three places the
multiturn/needle/placement/dateline sections state it (design
doc, USAGE, section text).

## Tests

`tests/test_cycle199_confidence.py`: the parser battery (24
named shapes, including every int()/regex trap and both
contested precedence cases); a **mutation gate** over the
parser — eight mutants (sign, leading zeros, non-ASCII digits,
any-line, multiple-threshold, splitlines, loose-regex-plus-int,
count-first precedence), each killed by a named battery case,
with the unmutated control passing the whole battery; screens
and surgery; case-generation distinctness/determinism/donor
pairing; stage behaviour over scripted backends (faithful,
theatrical, per-candidate failure, budget, cancel, suffix
screen); a one-break-at-a-time validator tamper battery (19
breaks, including the scalar edit the carried line exists to
catch); diff-section behaviour; the HTML section including a
hostile candidate name through the C0/surrogate-stripping
escape; the CLI wrapper gate; and one tiny-checkpoint
end-to-end run. The diff-section blast radius is covered in
`test_cycle67_diffreports_audit.py` (independent oracle trio,
a fuzz builder that runs the real stage over scripted
backends with hostile mutations and a coherent foreign
fingerprint, the `_identical_expected` key set, and the
flip-identical gate anchor extended in both its original and
mutant texts — the same pass also repaired two sections the
oracle's identical conjunction had silently omitted,
placement and precision) and in
`test_cycle27_diffreports_verify.py` (availability oracle
recomputing through the producer's aggregation). CLI pins:
`test_cycle179_cli_dispatch.py` (flag + helper-dispatch
entries) and `test_cycle44_validation_audit.py` (source-file
count 58 → 59).
