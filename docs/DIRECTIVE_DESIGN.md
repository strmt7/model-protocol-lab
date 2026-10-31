# Response-directive ladder — design round (ADJUDICATED: NO-BUILD)

## Round summary (coordinator #41, design round only — nothing built)

- Tree at main `f31e72e1ace4` (cycle 212 closed green, 8,439 tests,
  precise coverage 95.3455%). This round was READ-ONLY except for
  this doc and the briefs; no suite was run, no push was made, no
  tag was cut or discussed.
- Specialists (blind to each other): researcher
  (`~/workspace/agents_out/directive/research.md`), red-team A
  (`redteam_a.md`), red-team B (`redteam_b.md`), verifier/claims
  auditor (`verifier.md`). All four briefs landed; the verifier
  resolved every externally checkable claim against primary
  sources, re-ran every surviving PoC, and recomputed all decision
  math with independent code; the coordinator re-executed the
  load-bearing claims itself (§4). **Fabrication sweep: clean** —
  no invented symbol, no fabricated citation, and no unreproduced
  load-bearing number in any brief (the rank-5 clean result
  repeats); the failures found were line drift (all ≤9 lines,
  substance intact), one vacuous statistic, and one wrong author
  attribution, all handled in §5.
- Convergence and divergence, in one paragraph: all three blind
  briefs return NO-BUILD for the candidate as specified in the
  circulated sketch (three directive arms appended to the last
  user message, verdicts through `core.evaluate` unchanged,
  selection re-applied per arm, falsified by any pass-set
  difference or selection flip), and the verifier upholds that
  unanimously, fatal by fatal, by execution — four independent
  stub backends through the real `core.evaluate` agree on the
  central defect. The researcher additionally specifies a
  redesigned instrument (categorical panel of named strings,
  a frozen final-line contract in every prose arm, a span-verdict
  headline with byte-exact demoted to an elicitation-cost readout,
  a panel-derived decision rule, screens, the name
  `--response-directive`) and recommends it as a resubmission
  candidate; the verifier reproduced its decision math exactly
  and ruled it a **resubmission, not an amendment**, under this
  program's own test. The coordinator concurs: the verdict is
  **NO-BUILD** — for the sketch, and as a build order for the
  redesigned instrument in this round. §3 records the
  resubmission specification in full, including the three points
  no brief fully closed (budget discipline at the recorded cap,
  the directive-bearing-fixture screen, the absent neutral-suffix
  control).

## 1. Why NO-BUILD: the fatal defects of the candidate as specified

Each defect below is individually blocking and was established by
execution (attribution: R = researcher, A = red-team A, B =
red-team B, V = verifier, C = coordinator).

### F1 — The byte-exact scorer × prose arms: compliance is punished, disobedience rewarded (the load-bearing fatal)

`core.evaluate` has no answer extraction: the scored answer is
the whole decoded pre-boundary span (`core.py:939`
`answer_ids = output_ids if cut is None else output_ids[:cut]`;
`core.py:966` `answer = decoded`), and the verdict is
`core.py:797`: `exact_match = status == "completed" and answer ==
case.expected`. A brief/steps-compliant response is by the arm's
own demand a strict superstring of the expected answer, so
P(pass | full compliance, brief/steps) = 0 by construction.
Executed through the real `evaluate` with stub backends (R poc1,
A scorer_stub, B scorer_stub, V scorer_poc — four independent
implementations; C re-ran A's, §4): a perfectly compliant model
passes brief **0/4** and steps **0/4** — including the most
charitable variant, the correct final answer alone on its own
final line — while a model that ignores the directives entirely
passes **4/4 in every arm**. The manufactured share of the arms'
discordance equals the candidate's compliance rate, which the
instrument never measures. The sketch's conceded mitigation
(a boundary-failure taxonomy printed beside flips) does not
cover the mode: the compliant failures have `boundary_found=True`
and land in `non_label` under the tree's decomposition —
indistinguishable from genuinely wrong answers. No shipped
machinery rescues it: confidence survives its suffix only because
the suffix demands a severable grammar line and its verdict is
re-derived on the span (`confidence.py:400-410` `cell_verdict`;
CONFIDENCE_DESIGN records that raw evaluator verdicts on suffixed
generations are "meaningless"), and an explanation is not
severable — executed: `parse_verbalized_confidence` on a brief
answer returns the whole text as the span (`missing_line`), and
`scorerlens.rung_match` recovers a compliant brief/steps answer
at no rung 0–4 (the ladder has no label-strip operation,
`scorerlens.py:53,133`). Negation survived this trap by
construction: a byte-identical answer-only reply contract in
every arm (`negation.py:125-137`). The sketch's arms have
different reply contracts, two of them unsatisfiable under its
own scorer. This is rank-4's F3 shape (a perfect XML answer
scoring `schema_failed` 10/10) arriving through the verdict path.

### F2 — There is no decision rule: the any-difference falsifier fires on noise with probability 1

The sketch's falsification is "any candidate whose pass set
differs between arms, or a selection flip in any arm" over
7,200 cells. Executed (A, B, V; C independently, §4): under a
null of per-cell verdict jitter ε, P(any pass-set difference in
the panel) = **1.000 at every ε from 0.001 to 0.05** (analytic
E[discordant pairs] = 71.6 at ε=0.005, matching Monte Carlo
exactly), and conditioned on a native winner existing — the
only case where a selection flip is observable —
P(winner loses all-pass status in ≥1 of 3 arms) = 0.951 at
ε=0.01 (analytic 1−(1−p)^300, Monte-Carlo confirmed). A rule
that fires with certainty under any nonzero jitter is a
discordance printer wearing a falsification label (the payload
/ answer-syntax F2 pattern), and its two headline tests are
degenerate in opposite directions: pass-set difference fires
always, while at typical pass rates there is usually no native
winner to flip at all. Every surviving probe in this tree pairs
discordance with a derived floor and printed operating
characteristics (negation (8,5), leading (6,2)); the sketch has
none, and none can be borrowed — its population is the user's
fixtures, whose pass distribution is data-dependent (the
borrowed-floor trap of answer-syntax F5 / instruction-language
F4; the researcher's poc3 confirms the borrowed (6,2) row fires
0.94–1.00 on this panel's nulls too).

### F3 — There is no ladder: the arms are incomparable bundles, and native is not a rung

By construction from the arm texts (R §2, A F3, B F3, V P5 —
unanimous): on demanded pre-answer prose, native = direct (0) <
brief (one sentence) < steps (unbounded); on output
restrictiveness, steps (none) < brief (one sentence) < direct
("only"); on suffix length, direct (34 chars) < steps (47) <
brief (unfrozen, see F4). Brief and steps swap rung order
depending on the property measured; direct is a sign flip on
prose (forbidding it), not a smaller dose of it; native is a
control condition, not rung 0 — under native a model may emit
more prose than brief's cap allows. No single named dimension
orders the arms, so "ladder" and any rung/trend language
attribute a bundle (contract strictness, prose budget, prose
function, length) to a dose — payload F3 in miniature. The
tree's one real ladder, scorerlens, is a pipeline in which each
rung consumes the previous rung's output and monotonicity is a
theorem; nothing here is cumulative.

### F4 — The brief arm is not even a frozen string

The sketch quotes direct (34 chars) and steps (47 chars)
verbatim and gives brief only as a paraphrase ("a one-sentence
explanation, then the final answer"). The three briefs each had
to invent a different brief string (64, 60, and 55 chars) — the
demonstration itself. An arm that exists only as a paraphrase
cannot be audited against the tree's frozen-instrument
discipline (special-token screen, length accounting, material
digest): the instrument as specified is not buildable without
inventing load-bearing text.

### F5 — Token-cap censoring is ordered by arm, and counting it does not fix it

The run's default cap is 64 tokens (`cli.py:198`), and the
recorded native run was generated at that cap. The arms place
the answer at treatment-dependent positions — token 0 for
native/direct, after all demanded prose for brief/steps — so
the effective answer budget under one shared cap is ordered
steps < brief < direct = native. Executed (A, V re-run;
codepoint stub): a compliant steps output needing 78 tokens
before its boundary records `answer_len=64`,
`boundary_found=False`, status still `completed` — censored,
not measured, and retained in the "common completed" population
as an ordinary failure. Raising the cap for the treated arms
breaks comparability with the recorded native run, which cannot
be re-generated at a new cap; keeping it manufactures steps
failures in proportion to explanation length. The verbosity
convention the sketch echoes (a capped length is a lower bound,
reported as a share) is a descriptive label over recorded runs;
here the unobserved quantity is the verdict itself.

### F6 — Conditioning is outcome-dependent, the native arm is asymmetric, and the population scope is unstated

(a) `core.evaluate` aborts the whole report at the first
failed/cancelled cell: executed (A, V re-run), one arm-only
backend error (`context_limit_exceeded` — exactly the error a
longer treated input invites) at fixture 1 of 4 yields a failed
report with 2/4 observations, `aggregate_eligible=False`. The
"common completed set" after such an error is a prefix selected
by the treatment (answer-syntax F2 verbatim), while truncated
cells (F5) do the opposite damage and stay in as failures — the
population split is decided by backend error mechanics, not by
the design. (b) The native arm is the recorded run: held-out
fixtures are evaluated in `run_experiment` only for the selected
candidate, so a directive-arm "selection flip" can never be
followed through to a held-out verdict for the new winner
within the stated budget — the flip readout is a search-only
hypothetical and the sketch never labels it so. (c) The sketch
says "100 fixtures" without stating search-only scope; every
sibling stage is search-only, and silence on population scope
in a selection probe is a defect, not a detail — treated
held-out fixtures with re-applied selection would be held-out
contamination of exactly the kind the Split discipline exists
to prevent. (d) The sketch also never states the same-run /
configuration-digest obligation for the reused native baseline
(the cycle-207 `_require_native_match` / `_configuration_digest`
machinery, `experiments.py:3884` / `:3940`, exists because a
baseline from another configuration was silently compared
before) — a spec gap, recorded as such, not an executed defect.

### F7 — The tree's only concrete population already carries the treatment's contradiction

Executed (B fixture_conflicts; C re-ran it and read the file
directly, §4): all 4 shipped example fixtures
(`examples/fixtures.json`, 2 search + 2 held-out) carry an
answer-only directive in the last user message — "Reply with
exactly BLUE and nothing else.", "Now reply with exactly GREEN
and nothing else.", "Return only the numeral 7.", "Replace that
answer with the single word SOUTH." On such fixtures direct is
a near-duplicate paraphrase of the incumbent instruction
(treatment ≈ no-op), and brief/steps *contradict* it: the model
must disobey one instruction to obey the other, so discordance
measures instruction-conflict resolution — hierarchy's
construct — unscreened, uncounted, and varying by fixture.
Confidence's screens (`confidence.py:225-247`) cover blank /
confidence-shaped expected answers and vacuous turns, not
directive-bearing turns. The shipped example set is small (2
search fixtures), but it is the lab's own fixture style, and
any realistic answer-only fixture set is affected in proportion
to how well its fixtures are written. No screen is specified.

### F8 — No neutral-suffix control: discordance cannot be attributed to directive content

Appending anything to the last user message changes the token
sequence and can flip a greedy path. The sketch's only contrast
is native vs directive-suffix, so its discordance pools
(a) directive semantics, (b) suffix presence/length
(+34/+47/+60 chars), and (c) recency effects. The tree already
treats the control as mandatory: `repetition.py` ships a `pad`
arm matched per (candidate, fixture) on rendered token length
precisely so the repetition effect is separable from the length
effect (`repetition.py:66-71`: ARMS `("x1", "x2", "pad")`,
CONTRASTS `length_effect` / `repetition_effect`). Without a
length-matched neutral arm, even a scorer-fixed headline could
not say the *directive* did it.

### F9 — Single-phrasing collinearity kills the class-level claim (instruction-language F1 pattern)

One frozen string per directive makes treatment and rendering
perfectly collinear: every discordance is an effect of these
strings, with this joiner (which the sketch does not even
specify). The claim that survives scoping — "appending these
exact strings changed these candidates' pass sets on these
fixtures in this run" — does not clear the Monday-morning bar
the sketch sets itself (a prompt engineer choosing *a directive*
for their prompt learns nothing about step-by-step directives as
a class, and nothing about whether steps improves the answer,
the decision-relevant quantity F1 makes unmeasurable). String
scoping is the negation/instruction-language settlement and the
resubmission adopts it (§3); the sketch's claim is written at
class level, and at class level it is false advertising.

### Non-fatal, adjudicated (the sketch's defence, and what passed)

- **Duplication.** The brainstorm's defence — "confidence never
  re-applies selection per arm" — is literally true of confidence
  (V §1 S7: `confidence_stage` copies `primary_decision` /
  `primary_winner` from the native result) and **refuted as a
  novelty claim about the tree**: prefill re-applies selection
  per condition (`experiments.py:2585` `prefill_adherence`;
  prefill's pre-registered claim 1 is a winner-under-condition
  contrast), canonical-form re-applies it per arm over a
  restricted shared population (`canonicalform.py:599`
  `arm_selection`, `:641` `winner_headline`), and confidence
  already ships the elicitation-cost flips readout
  (`pass_to_fail` / `fail_to_pass` vs the recorded native run).
  Surgery, native reuse, flips, and per-arm selection are all
  shipped measurements; the only unshipped element is the
  treatment strings — and F1 makes the treatment unmeasurable
  for two of its three arms. Sustained against the defence; not
  independently fatal to a future instrument whose novelty is
  argued on treatment alone (researcher §9 concedes exactly
  this, and §3 item 6(e) puts the judgment to the fresh review).
- **Selection re-application on user fixtures is legitimate**
  (R §4, upheld by V as dispute D4): the numform settlement
  (`experiments.py:11049-11051`) bars winners over *synthetic*
  sets; this population is the user's own search fixtures — the
  same contest — with prefill/canonical-form as shipped
  precedents, under the conditions of §3 item 2 (restricted-set
  recompute, search-only labelling, primary decision untouched).
- **Passed checks (not findings):** the suffix lands inside the
  last user turn for every catalog family (executed renders,
  chatml/llama3 checked explicitly; `Format.render`,
  `core.py:76`); the special-token audit over all 29 catalog
  candidates returns 0 findings for all three suffix strings;
  per-fixture seeds are id-derived (`core.py:371`
  `request_seed_for`), so id-preserving surgery keeps arm seeds
  identical to native — an obligation the design must pin (the
  prefill id-prefixing pattern would silently break it), not a
  defect found; the 7,200-generation arithmetic is consistent
  with the tree's caps (24 candidates × 100 search fixtures,
  `determinism.py:32-38`, `experiments.py:4014`).
- **Naming.** `directive` has zero hits in the package,
  `docs/USAGE.md`, `README.md`, and `docs/FAQ.md` (R, V, and C
  greps concur); the only hits anywhere are prose "owner
  directive" in `docs/RESEARCH_LOG.md:3814`, `:5596`, `:5755`.
  `--directive` is therefore free as a string but over-broad as
  a name (it names any instruction — hierarchy's whole
  territory), the payload §8.46 failure shape; the resubmission
  proposes `--response-directive` (§3 item 5).

## 2. Why the redesigned instrument is a resubmission, not an amendment

The negation round's BUILD verdict rested on a stated test:
every defect was corrected by amendment to the *same* instrument
— same population, same arms, same scorer, same joint-rule
shape — whose population and arms were already frozen and
executable before the round began. The payload and answer-syntax
rounds rest on the complementary test: where the amendments
replace an undefined or unexecutable instrument with a different
one, the result is a resubmission and must be reviewed as a new
design before it is built.

Measured against that test, the researcher's redesigned
instrument changes: the **arms** (new texts carrying a frozen
final-line contract; the panel is categorical and the "ladder"
framing is dropped); the **headline scorer** (span verdict —
final line equals expected, conjoined with boundary — with
byte-exact demoted to a beside-the-headline elicitation-cost
readout); the **population discipline** (fixture screens, the
restricted common-completed-set selection recompute, search-only
labelling); the **decision rule** (none in the sketch → a
derived joint rule, §3 item 3); and the **name**
(`--directive` → `--response-directive`). The sketch's instrument
was never executable as a measurement in the first place: its
brief arm has no string (F4), its scorer inverts its headline
(F1), and its falsifier is a tautology (F2). This is the payload
configuration, not the negation one.

The verdict is therefore NO-BUILD for this round. This is
explicitly **not** a finding that the redesigned instrument is
unsound: the verifier reproduced every one of its operating
characteristics by independent exact enumeration (and the
coordinator re-ran the enumeration, §4), it answers F1, F2, F3,
and F9 by construction rather than by disclosure, and its
author states its central risk himself (§3 item 6(e)). What it
has never faced is a blind red team of its own: its novel
machinery — the span parser over prose arms, the frozen
decomposition classes, the per-run-N operating-characteristics
printing, the not-yet-designed directive screen of F7, the
not-yet-settled budget discipline of F5 — was claim-verified
and math-verified in this round, not adversarially attacked.
The program does not build an instrument on verification alone.

Two alternative residues were also recorded by the red teams
and are noted here so the fresh review can weigh them against
the main candidate, not silently lose them: (a) red-team B's
directive-*phrasing* probe — every arm demands the same
answer-only reply contract under different frozen phrasings,
plus a length-matched neutral control, scoreable under
`core.evaluate` unchanged (a different, smaller instrument
measuring phrasing sensitivity, not production-style effects);
(b) red-team A's residue — a confidence-style descriptive flip
readout for the direct arm alone, the only arm whose compliant
behaviour the byte-exact scorer can pass, as an extension of an
existing instrument family rather than a new probe.

## 3. Resubmission specification (NOT a build order)

A resubmission is reviewable as a NEW design only if it
satisfies all of the following in full. Partial compliance is
still NO-BUILD. The researcher's redesigned instrument
(`~/workspace/agents_out/directive/research.md` §5–§11) is the
resubmission candidate and is incorporated by reference,
subject to items 1–5 below; the fresh review's job is to attack
the frozen machinery itself (the arm texts, the span parser, the
decomposition, the screens, the rule at realistic N), not to
re-litigate the sketch, which is dead.

1. **Corrections applied before review.** Tree line ranges as
   corrected in §5 (JOINER `confidence.py:78`; parser def
   `confidence.py:177`, body to :205; SUFFIX `confidence.py:85-89`;
   answer span `core.py:939` / `:966`); the first author of
   arXiv:2511.04108 is Srivastava (Saurabh Srivastava et al.;
   no author named Qiu appears in the current arXiv record)
   and the paper is dropped
   as load-bearing in any case (§5); the RESEARCH_LOG naming
   claim is three prose hits (`:3814`, `:5596`, `:5755`), not
   two; the rule table's "0.0000" per-decision rate at N=100 is
   ≈2.4e-5 and must never be quoted as zero.
2. **Frozen items that carry verbatim in substance:** the
   population (the run's search fixtures only, post-hoc stage,
   native arm = the recorded run never re-executed, held-out
   never treated; the selection-flip readout labelled
   search-only wherever it appears); the arms (`terse` /
   `brief` / `steps` with the researcher's §5 frozen texts and
   the `"\n\n"` joiner, every prose arm mandating the final
   answer on its own final line; special-token screen over every
   suffix for every family before generation); the headline
   scorer (span verdict) with the byte-exact verdict printed
   beside it as elicitation cost, never conflated; the frozen
   decomposition (`span_pass` / `answer_line_not_final` /
   `answer_embedded` / `answer_absent`, boundary-only discordance
   decomposed out first, `preanswer_present` counts printed);
   the decision rule of item 3; the selection readout
   (canonical-form `winner_headline` transplanted per two-arm
   contrast, selections recomputed on the restricted common
   completed set, withheld when sets differ or are empty,
   selection-vs-abstention never a winner change, the primary
   run's decision/winner/exit code never touched); the
   fail-closed vocabulary of research §8 (stage details
   `{cancelled, directive_record_budget_exceeded,
   directive_screen_failed}`, per-cell `{"",
   arm_context_exceeded, arm_evaluation_failed}`, whole-stage
   `directive_unavailable`, section labels
   `directive_drop_observed` / `below_threshold` /
   `insufficient`); token-ID divergence reported, not gated
   (negation disposition; same-ids = rendering anomaly, same-ids
   flip = determinism finding); digest-only records with the
   span split named as stage-attested residue (no carried final
   line — it *is* the answer); the validator mirror obligation
   (suffixes, joiner, screens, span parser, decomposition, rule,
   restricted-set selection — re-implemented, never imported).
3. **Decision rule (as derived in research §7, verified twice
   independently):** per candidate, per arm-vs-native contrast,
   over the common completed set — D = native passes, J = of D,
   arm span-verdict fails, R = reverse discordants printed beside
   J; fire `directive_drop_observed` iff D ≥ 10 and J ≥ 5 and
   J ≥ D/4 and McNemar exact one-sided p(J vs J+R) < 0.05;
   D < 10 → `insufficient`; otherwise `below_threshold`.
   Operating characteristics (exact enumeration; researcher,
   verifier, and coordinator agree cell-for-cell): random
   labeller fires 0.0377 (N=100) / 0.0304 (N=30); no-effect
   competent rows 0.0000–0.0071; power at a true 25% drop is
   **0.5027 (N=100) / 0.3441 (N=30)** — a coin flip; at a 50%
   drop, 1.0000 / 0.9490. Because the panel size is the user's
   own fixture count, the section must print the operating
   characteristics **at the run's own N** (no single frozen
   paragraph), and the family-wise expectation with them (at
   N=30, ≈0.51 expected false labels per 72 decisions,
   P(≥1) ≈ 0.40; at N=100, P(≥1) = 0.002). `below_threshold` is
   weak evidence and must say so; small user sets (the shipped
   example has 2 search fixtures) read `insufficient` —
   correctly.
4. **Monday-morning utility (resubmission, string-scoped):** a
   prompt engineer who has picked a template family and is about
   to ship a response-style directive in the user turn uses this,
   per candidate, to decide whether that exact directive string
   can ride along without changing the lab's winner on their own
   fixtures — before the choice silently flips a selection they
   already made. Where the verdict is `insufficient` or
   `below_threshold`, the honest deliverable is "this panel
   cannot distinguish them; choose on other grounds", and the
   design must say so in those words' substance.
5. **Name:** `--response-directive` (module
   `responsedirective.py`), per research §10's in-design audit —
   `--directive` is free as a string but names any instruction
   and overclaims the scope; "ladder" is dropped entirely. The
   name ships only after the build cycle's USAGE naming audit
   (the rank-2 §8.46 precedent) regardless.
6. **What the fresh review must settle head-on** (silence on
   any of these is NO-BUILD):
   a. **Budget discipline at the recorded cap (F5).** At the
      recorded `--max-new-tokens` the steps arm's span verdict
      still grades a censored population (truncated cells are
      completed, answer never emitted). The resubmission's
      per-arm `n_limit` counting and decomposition label the
      censoring; the fresh review must decide whether labelling
      suffices, whether truncated cells leave the headline
      population under a pre-registered rule (and what that does
      to the conditioning, F6), or whether the instrument needs
      a different budget regime — argued against the native
      comparability constraint, not defaulted.
   b. **The directive-bearing-fixture screen (F7).** A frozen,
      decidable screen for fixtures whose messages already carry
      an answer-style directive (in any turn, not only the last —
      the shipped example carries it in the last turn, but
      nothing in the fixture format confines it there), with
      screened fixtures counted and classed pre-generation,
      never silently dropped. The researcher's §4 screen list
      (multiline / blank / vacuous / no-user) does not include
      this class; without it the span verdict on such fixtures
      measures conflict resolution, not directive sensitivity.
   c. **The neutral-suffix control (F8).** Include a
      length-matched neutral arm (the `repetition.py` pad
      precedent), or argue — against F8, not around it — why the
      span-verdict design attributes discordance to directive
      content without one. Suffix-presence is a real channel:
      any appended text can flip a greedy path.
   d. **Panel power at realistic N (F2/item 3).** The rule is
      coin-flip powered at a 25% drop even at N=100, and most
      real user sets are far smaller. The fresh review must
      either accept this with the at-N operating characteristics
      printed beside every verdict, or change the panel/rule —
      and argue the choice, not default to it.
   e. **The confidence-overlap judgment.** Machinery reuse from
      confidence is total (surgery, span verdict, flips,
      wrapper gate); the treatment difference is the whole claim
      to novelty (research §9, conceded by its author). The
      fresh review decides whether that difference justifies a
      new probe, an extension of confidence, or nothing — with
      red-team B's phrasing variant and red-team A's direct-arm
      residue (§2) in front of it as the competing smaller
      instruments.
   f. **Neighbour re-survey.** The uniqueness pass's neighbour
      set (LM Eval Harness, HELM, PromptPerfect, OpenAI evals,
      DSPy) was NOT re-surveyed this round (research §11,
      UNEXECUTED; the cycle-199 survey covers the confidence
      combination). The fresh review re-runs it before any
      build; the claim stays combination-scoped to surveyed
      sources until then.
7. **Counts (as frozen in the brief):** 3 treated arms × ≤100
   search fixtures × ≤24 candidates = 7,200 generations maximum
   (native reused, 0); ground records = 24 × 100 = 2,400
   multi-cell records; caps frozen in the module with
   derivations (confidence's 2,400-record precedent). The
   7,200 figure must never be quoted without its search-only
   and recorded-cap qualifications (§5).
8. **Sources** are limited to §6 of this doc, with their
   fencing. No "first" is claimed anywhere; the contribution,
   if the resubmission is ever built, is the combination stated
   in research §11 (per-template crossing of frozen
   response-directive strings on the user's own fixtures,
   span-verdict discordance plus byte-exact elicitation cost,
   conservative selection re-applied per arm on a common
   completed set), scoped to surveyed sources.

## 4. Adjudication evidence (coordinator's own execution)

All snippets read-only; no suite was run.

- **Scorer line and answer span.** `core.py:797` read directly:
  `exact_match=status == "completed" and answer == case.expected`;
  `core.py:939` `answer_ids = output_ids if cut is None else
  output_ids[:cut]`; `core.py:966` `answer = decoded`. No
  stripping, parsing, or extraction anywhere on that path.
- **Scorer inversion (F1).** Re-ran red-team A's
  `/tmp/redteama/scorer_stub.py` (stub backend through the real
  `core.evaluate`): compliant native 4/4, direct 4/4, brief 0/4
  (answer `'The question asks for a known fact. 42'`), steps 0/4
  — including the final-answer-on-its-own-line variant, 0/4;
  ignoring model 4/4 in all four arms;
  `parse_verbalized_confidence` on the brief answer returns
  `missing_line` with the whole answer as span;
  `scorerlens.rung_match` recovers neither the brief answer nor
  a bare label-prefix answer at any rung.
- **Population (F7).** Re-ran red-team B's
  `/tmp/redteamb/fixture_conflicts.py` and read
  `examples/fixtures.json` directly: 2 search + 2 held-out
  fixtures, all four last user messages carrying an answer-only
  directive (texts in §1 F7); catalog = 29 formats; special-token
  audit 0 findings for all three suffix strings; suffix lands
  inside the user turn in chatml and llama3 renders.
- **Noise floor (F2).** Coordinator Monte Carlo (300 trials,
  jitter null): P(any pass-set difference in the 24×100×3 panel)
  = 1.000 at ε = 0.001, 0.005, and 0.01; analytic E[discordant
  pairs] = 14.4 / 71.6 / 142.6. Selection, conditioned on a
  native winner (analytic): P(winner loses all-pass) = 0.394 /
  0.634 / 0.867 for one arm at p = 0.005 / 0.01 / 0.02, and
  0.778 / 0.951 / 0.998 over three arms.
- **Resubmission rule (item 3).** Re-ran the researcher's
  `/tmp/researcher/poc3b_mc.py` (exact enumeration): every cell
  of the research §7 table reproduced — random 0.0377 / 0.0304,
  drop-25% power 0.5027 / 0.3441, drop-50% 1.0000 / 0.9490,
  family-wise N=100 P(≥1) = 0.002, N=30 expected 0.51,
  P(≥1) = 0.402. This is the third independent reproduction
  (researcher, verifier, coordinator).
- **Tree settlements verified by reading (clean reads this
  round):** `cli.py:198` (`--max-new-tokens` default 64);
  `cli.py:827` (`"--confidence",`); `confidence.py:400-410`
  (`cell_verdict`: span-digest equality plus `boundary_found`);
  `canonicalform.py:599` (`arm_selection`);
  `density.py:131-138` (STAGE_DETAILS / RECORD_DETAILS pattern);
  `repetition.py:66-71` (pad arm and length/repetition
  contrasts).
- **Verifier reproductions accepted:** the abort-prefix
  execution (F6a), the cap-64 truncation execution (F5), the
  conditioned selection-noise table (B's `selection_noise.py`),
  the sketch-claim corrections of §5, and the external-source
  resolutions of §6. The coordinator did not re-run those in
  full; they are credited to the verifier, not claimed as
  coordinator executions.

## 5. Removals applied (verifier §9 — none of these enter any build doc)

- **Sketch:** `cli.py:769` / `cli.py:2175` as anchors (true:
  flag `cli.py:827`, `_run_confidence_probe` def `cli.py:2319`,
  call site `cli.py:4181-4182`); the "reporting.py:894 pattern"
  (the confidence analogue is `_confidence_section`,
  `reporting.py:8521ff`); the unqualified "confidence's readout
  is the stated number, not the verdict" (confidence ships
  elicitation-cost flips and a span-derived verdict);
  arXiv:2511.04108 as evidence that prompt framing changes
  reasoning length (it is a batching study — §6); the
  any-difference falsifier; "ladder" and all rung/trend language
  for these arms; the 7,200-generation figure without its
  search-only and recorded-cap qualifications.
- **Researcher:** "Srivastava et al." for arXiv:2511.04108 →
  Srivastava et al. (the current arXiv record's author list;
  an earlier printing named "Qiu & Srivastava", corrected in
  cycle 226; dropped as load-bearing regardless);
  `confidence.py:76` JOINER → `:78`; parser range 168–207 →
  def `:177`, body to `:205`; SUFFIX range 85–90 → 85–89;
  "only hits" in RESEARCH_LOG → three hits (`:3814`, `:5596`,
  `:5755`); the per-decision "0.0000" at N=100 / r=0.10 must
  not be quoted as zero (≈2.4e-5).
- **Red-team A:** the `noise_floor` "P(winner-set differs) ≈
  0.000–0.001" figure — REFUTED as evidence (vacuous
  parameterisation: at its base pass rate a native all-pass
  winner almost never exists, so there is no selection to flip;
  the conditioned figure is 0.951 at ε=0.01 and is the one used
  in §1); `core.py:942-978` → answer span at `:939` / `:966`;
  the `core.py:340s` invariant cite → the raise at `:323`,
  `aggregate_eligible` at `:343-344`.
- **Red-team B:** `core.py:954` → `:939`; `confidence.py:93`
  grammar → `_GRAMMAR_RE` at `:96`; SUFFIX cite 80–86 → 85–89;
  `primary_decision` cites `12871-12872` → `:12861-12862` /
  `:12986-12987`; the 60-char brief length presented as a sketch
  property (it is B's reconstruction of an unfrozen arm — F4);
  B's unconditioned selection-flip column (same vacuity as A's;
  B's brief itself correctly relies on its conditioned
  `selection_noise` instead). B's early-draft import of a
  nonexistent `catalog_formats` was self-corrected to `CATALOG`
  before the brief landed and is relied on by no claim; it is
  recorded here for the sweep, not as a removal.

## 6. Sources the resubmission may safely cite

All resolved against their primary sources this round by the
researcher and independently by the verifier (abstract/HTML
level; fencing must travel with each):

- Wei et al., arXiv:2201.11903 — chain-of-thought *via few-shot
  exemplars* improves arithmetic/commonsense/symbolic reasoning
  at scale. Fencing: the treatment is demonstrations, not an
  appended zero-shot directive; the sketch's `steps` suffix is
  Kojima-shaped, not Wei-shaped. Citable only for "intermediate
  reasoning steps can change answers", never for suffix
  directives or templates.
- Sprague et al., arXiv:2409.12183 — meta-analysis of 100+
  papers plus own evaluations (20 datasets, 14 models): CoT's
  benefit is concentrated in math/logic/symbolic tasks; on MMLU
  direct answering is almost identical to CoT absent an equals
  sign. Fencing: benchmark accuracy; no chat-template crossing,
  no selection rule, no byte-exact boundary scorer. Its
  load-bearing use here is predictive: the steps arm's *answer*
  effect on the lab's mostly short-answer fixtures is expected
  small — which is why the contract/truncation channels (F1, F5)
  dominate the sketch and must be designed out, not disclosed
  away.
- Srivastava et al., arXiv:2511.04108 — batching multiple queries
  into one prompt suppresses overthinking in large reasoning
  models. NOT directive evidence; the researcher drops it as
  load-bearing and a resubmission should too.
- Kojima et al., arXiv:2205.11916 — **NOT SAFE**: unresolved at
  primary source this round (seen only in secondary
  bibliographies; the researcher labelled it UNRESOLVED and
  non-load-bearing, and the verifier concurs). Resolve
  primary-first before any use.
- In-tree precedents a resubmission doc may cite:
  `docs/CONFIDENCE_DESIGN.md` (span-verdict precedent; raw
  verdicts on suffixed generations are meaningless);
  `negation.py` OPERATING_CHARACTERISTICS / `decompose_cell`
  (joint-rule and decomposition precedents);
  `canonicalform.py` `arm_selection` / `winner_headline`
  (restricted-set selection precedent); `repetition.py` pad arm
  (length-matched control precedent); `confidence.py`
  `screen_fixture` (screen precedent — whose classes do not
  cover directive-bearing fixtures, §3 item 6(b)).

No "first" is claimed anywhere in this doc. The contribution of
any future resubmission is the combination stated in §3 item 8,
never priority.

## 7. What a build cycle must NOT do on the basis of this doc

- Do not build the circulated sketch, any "ladder" framing or
  rung/trend readout, the any-difference falsifier, or brief/
  steps arms scored through `core.evaluate` unchanged.
- Do not build the redesigned instrument on this doc's
  authority: it enters only through a fresh design review
  (blind red-team pair + verifier) of a resubmission satisfying
  §3 in full, and only on a BUILD verdict from that review.
- Do not ship a flag under `--directive`; the resubmission's
  name is `--response-directive`, and even it ships only after
  the build cycle's USAGE naming audit.
- Do not re-apply selection over held-out fixtures, quote the
  7,200-generation figure without its search-only and
  recorded-cap qualifications, or carry any figure listed in §5
  into a doc or help text.
- Do not present the span verdict and the byte-exact
  elicitation-cost readout as one number, or let the
  decomposition rescue a discordant cell into a pass.

## 8. Fresh review of the §3 resubmission (2026-10-28, coordinator #58)

Blind round: researcher (strongest honest FOR case), two
independent red teams, coordinator adjudication by own execution
only. Briefs and scripts: `~/workspace/agents_out/directive2/`
(`research.md`, `redteam_a.md`, `redteam_b.md`); the full
adjudication is copied to
`~/workspace/agents_out/directive2/ADJUDICATION.md`. READ-ONLY
round: no suite run, no build. Main at `18e4b4ca2e4f` (cycle 217
closed green).

**Verdict: NO-BUILD as written.** The resubmission's core
mechanism is real — the F1 cure verifies through the real
`core.evaluate` path (below) — and this is the closest
resubmission in the queue. It fails on its own terms: §3's
preamble makes partial compliance and silence on any item-6
decision a NO-BUILD, and item 6(b) is unsettled by the
candidate's own concession, while items 3 and 4 do not survive
execution as written. The cures for the decisive items each
touch the population, the headline statistic, or the decision
rule's stated basis, so this is not BUILD-as-amended. A restated
candidate (§8.7) may re-enter through a fresh review; it does
not jump the queue.

### 8.1 What verifies clean

- **F1 cure (executed, coordinator, real `core.evaluate` with a
  character-codec stub).** Brief-compliant ("one sentence of
  prose, final line = expected") and steps-compliant responses
  PASS the frozen span verdict where byte-exact `exact_match`
  fails every compliant shape; a trailing single newline passes;
  the decomposition classes behave as designed (`span_pass`,
  `answer_line_not_final`, `answer_embedded`, `answer_absent`,
  boundary-only separated). A model that ignores the directive
  and answers bare also passes the span verdict — disclosed in
  the design (the contract grades form, not intent) and priced
  by the byte-exact elicitation-cost readout printed beside it.
  The scorer×compliance inversion of round 1 is cured for
  compliant shapes.
- **Name (executed twice independently).** Zero hits for
  `response-directive` / `response_directive` /
  `responsedirective` / bare `directive` in the package, USAGE,
  README, FAQ; no colliding flag among the 68 parsed from
  `cli.py`. `--response-directive` stays cleared; §7's naming
  audit at build time still applies.
- **Kojima et al., arXiv:2205.11916 — RESOLVED.** §6 listed this
  as NOT SAFE (unresolved). The researcher resolved it at the
  primary source and the coordinator re-fetched the arXiv
  abstract page firsthand: "Large Language Models are Zero-Shot
  Reasoners", Kojima/Gu/Reid/Matsuo/Iwasawa; "Let's think step
  by step"; MultiArith 17.7→78.7%, GSM8K 10.4→40.7%
  (text-davinci-002; also PaLM 540B). Citable WITH fencing:
  Kojima's string is not §3's frozen `steps` string; the models
  are API-scale; the outcome is benchmark accuracy; the citation
  is non-load-bearing for this instrument either way.
- **Neighbour re-survey (item 6(f), researcher-executed).** The
  combination in §3 item 8 survives against LM Eval Harness,
  HELM, PromptPerfect, OpenAI Evals, DSPy (near-miss: its
  Predict/ChainOfThought split is a treatment vocabulary, but
  DSPy optimizes and abstracts the template away) and promptfoo
  / Inspect AI (near-misses: matrix evaluation without a
  final-line contract, paired D/J/R rule, or restricted-set
  winner readout). The coordinator firsthand-verified only the
  Kojima resolution; the survey is recorded as
  researcher-executed and is not load-bearing for this verdict
  (a missed neighbour could only add a duplication ground).

### 8.2 Decisive fatal — item 6(b): the directive screen does not exist

Executed three times independently (coordinator, researcher,
red team A): `examples/fixtures.json` is directive-bearing in
4/4 fixtures' last user turns — both search fixtures
(`search-copy-blue`, `search-latest-green`), i.e. the entire
treated population of the shipped demo. `examples/fixtures.csv`
carries the identical set, and earlier turns are
directive-bearing too ("Reply with RED.", "Say NORTH."). The
frozen screen classes (§3 item 2; `confidence.py:109-114`
`SCREEN_CLASSES`) contain no directive class; §3 item 6(b)
concedes the gap and the researcher's §4 list omits it. On the
shipped population the span verdict would measure conflict
resolution between a fixture's own answer-only directive and
the appended arm directive — not directive sensitivity. A
decidable screen over free prose (with a stated false-positive
discipline: what counts as a directive, in which turns) is
itself undesigned, and designing it changes the treated
population. Under §3's own preamble this item alone returns
NO-BUILD; under the amendment test it cannot be cured inside
this candidate.

### 8.3 Item 3 — the printed operating characteristics are exact only under a model the text never states

Coordinator exact enumeration (factorised binomial form of the
frozen rule; D ~ Bin(N, p), J | D ~ Bin(D, q), R | N−D ~
Bin(N−D, r), McNemar exact one-sided):

- Random labeller: fire = **0.0377 (N=100) / 0.0304 (N=30)** —
  the printed values CONFIRMED exactly, under every reading.
- Under the natural reading of research §7's rows ("loses x% of
  passes, reverse x%" — both conditional on their pools), the
  printed table does NOT reproduce: drop-25% power = 0.5040
  (N=100) / 0.5030 (N=30) against printed 0.5027 / 0.3441; the
  competent row (lose 10%, reverse 5%) at N=30 fires 0.0132
  against printed 0.0071. The round-1 verifier's independent
  spot value at N=100 was also 0.5040 (§4 of this doc records
  0.5027 from the researcher's script; the disagreement predates
  this round and was never adjudicated).
- Red team B reverse-engineered the generating model and the
  coordinator verified it cell-for-cell (12/12 cells): the
  printed table is exact iff p_nat = 0.9 AND "reverse x%" is an
  ABSOLUTE share of all N cells — i.e. at p_nat = 0.9, "reverse
  5%" means the arm rescues 50% of native fails, "reverse 9%"
  means 90%. Neither text states this. The printed numbers are
  therefore not wrong; they are unreproducible from the printed
  design, which fails the reproducibility standard §3 item 3
  itself invokes ("agree cell-for-cell" was agreement on a lost
  script's outputs, not on a stated model). The coordinator's
  interim conclusion that the N=30 column was irreproducible is
  corrected here: it is reproducible, under the unstated model.
- The unstated model is not innocent: its "no-effect" competent
  row silently fixes the scorer-leniency channel of §8.5 at a
  large value (half of native fails rescued) without naming it,
  and the family-wise paragraph inherits the ambiguity. Under
  the natural reading, the N=30 family-wise story (72 decisions)
  is expected ≈0.95 false labels / P(≥1) ≈0.62, not the printed
  ≈0.51 / ≈0.40.

### 8.4 Item 4 — the decision claim fails the document-syntax standard at its headline effect

§3 item 4 prints a decision claim ("decide whether that exact
directive string can ride along without changing the lab's
winner"), carried by the gated label. Label mix at a true 25%
drop, exact enumeration: under the decoded model, fire =
0.5027 at (p_nat 0.9, N=100) — modal by 0.0054, a knife-edge —
and 0.3441 at N=30 (escape-modal); under the natural reading,
0.5040 / 0.5030 at p_nat 0.9. At p_nat 0.7 / 0.5 the escape
sentence is modal at both N under the decoded model (fire
≤0.4815 / ≤0.3544 at N=100); at p_nat 0.3, N=30, even a 50%
drop is modal-`insufficient` (0.589) — the D≥10 gate, not the
effect, decides. So at the headline 25% effect the label's
modal outcome is its own "choose on other grounds" sentence in
every cell except the single most favourable corner, and which
model governs that corner is itself unstated (§8.3). By the
standard the document-syntax round set, the gated label cannot
carry item 4's decision claim. The per-arm `winner_headline`
readout is not power-gated and survives as a DESCRIPTIVE claim;
item 4 as written ties the decision to the label, so the item
fails as written.

### 8.5 Cross-scorer asymmetry — structural, executed, uncontrolled (red teams A-1c and B-K4, coordinator executions concur)

D is frozen from the recorded native run and is BYTE-EXACT
(`core.py:797`: `exact_match = status == "completed" and
answer == case.expected` — no boundary conjunct). J and R are
SPAN-verdict (final line == expected ∧ boundary_found). One
contrast, two scorers; only the `terse` arm is exempt. Executed
consequences:

- Native byte-fail / arm span-pass on identical content
  (coordinator: native span "BLUE\n" → byte-exact False, span
  verdict True). R therefore contains a scorer-manufactured
  component of unknown size; it is printed beside J as "reverse
  discordants" and gates the headline through McNemar. Red team
  B's labelled model puts the cost concretely: at byte-exact
  pass rate 0.5 with a leniency differential λ=0.2 (span
  rescues 20% of byte-fails), drop-25% power collapses
  0.314→0.033 (N=30) and 0.505→0.085 (N=100). λ is nowhere
  measured, estimated, or printed in the candidate.
- Native byte-pass / arm span-fail with zero answer change is
  also constructible: a boundary-free bare native answer is a
  recorded pass (no boundary conjunct in `core.py:797`); the
  identical arm generation span-fails on the boundary conjunct.
  The boundary-only decomposition absorbs this cell ONLY under
  the exclusion reading of J (§8.6) — which the candidate's
  text does not unambiguously state.
- The winner readout inherits the channel: native selection is
  recomputed from recorded byte-exact verdicts, arm selection
  from span verdicts, so `winner_changed` is manufacturable by
  scorer substitution alone (a candidate whose native failures
  are form-failures with correct final-line content is forgiven
  only on the arm side). The candidate never states whether the
  native side of any contrast is the recorded byte-exact
  verdict or a span regrade of the recorded spans — the two
  readings give different D, different R, and potentially
  different winners.
- No control exists in the frozen spec. A scorer-only control
  (span-regrade the recorded native spans; zero generations)
  would bound the channel, and a symmetric span verdict on both
  sides would remove it — but the symmetric cure redefines D,
  i.e. the headline statistic, so it belongs to a restated
  candidate, not to an amendment of this one.

### 8.6 Specification seams in the headline scorer and rule (red team A, partially sustained)

- **Extraction rule unpinned.** Neither candidate text states
  how the final line is extracted. The named mirror,
  `parse_verbalized_confidence`, is a confidence-GRAMMAR parser
  (executed: it returns `missing_line` on a compliant brief
  answer); the tree's actual line convention lives in the
  unnamed helper `confidence._final_line` (`confidence.py`
  ~:162-175: split on "\n", drop exactly one trailing empty
  element). Plausible extractors disagree on executed shapes:
  CRLF prose passes under splitlines, fails under `_final_line`
  ('BLUE\r' welded to the final line); a trailing blank line
  fails under all of them. A build would freeze the convention
  by fiat — i.e. §3 did not freeze its only novel component.
  Sustained as a freeze gap; not independently decisive.
- **J's boundary-only membership is textually ambiguous.**
  Item 3 defines J as "of D, arm span-verdict fails" (the span
  verdict includes the boundary conjunct); the decomposition
  text says boundary-only discordance is "decomposed out first"
  and is "never a directive effect". Two faithful readings
  compute different J and, at the margin, different labels.
  This review settles it by the tree's own precedent
  (`negation.py`: `boundary_only` is its own class, never
  pooled into the discordance count) toward EXCLUSION — but a
  candidate whose rule needs a review to disambiguate it is not
  frozen as claimed. A build under any future BUILD verdict
  must pin the membership by test either way.

### 8.7 Settlements of §3 item 6, as instructed

- **6(a) budget/censoring — SETTLED for the candidate, with
  conditions.** Red team A's execution is adopted in full:
  20 fixtures, native bare answers all passing, compliant
  `steps` prose of 234 tokens against the recorded cap of 64
  (`cli.py:198` default verified) → every arm cell truncated,
  completed, span-fail, classed `answer_absent`; D=20, J=20,
  R=0, label fires at p≈1e-6 with zero answer changes. The
  settlement nevertheless goes the candidate's way, because:
  (i) the cap is a frozen protocol field of the recorded run —
  the tree's native-match discipline forbids varying it per
  arm, and raising it breaks the recorded-native comparability
  item 2 freezes; (ii) excluding truncated cells conditions on
  a treatment-caused mediator — the F6 error this design
  exists to avoid; (iii) under the frozen string-and-
  configuration scoping, the label in A's world is literally
  true and decision-correct (at the recorded configuration,
  shipping that string destroys those passes), and the
  decomposition plus per-arm `n_limit` print the channel beside
  it. CONDITION, binding on any future build: no rendered
  sentence may gloss `directive_drop_observed` as the directive
  changing the model's answers; the gloss is passes lost at the
  recorded configuration. If that discipline ever fails, this
  settlement reverses and A's kill becomes fatal.
- **6(b) screen — NOT SETTLED (§8.2). Decisive.**
- **6(c) neutral control — SETTLED for the candidate,
  conditionally.** Red team A executed the strongest form: a
  length-matched neutral suffix built from
  `repetition.FILLER_UNIT` (69 chars, zero directive content)
  produces a firing label indistinguishable from the `steps`
  world's (minimal form D=20, J=5, R=0, p=0.03125). Under §3's
  frozen string-scoped claims this is a true-but-coarse label,
  not a false one: suffix-presence is a constituent of the
  exact string being decided on, and the design never prints a
  mechanism claim. The settlement holds only under the same
  claims discipline as 6(a) — the moment any sentence
  attributes the drop to directive CONTENT, A's counterexample
  is decisive. The residual is recorded honestly: the
  instrument cannot separate presence from content, and its
  claims must never imply it can.
- **6(d) panel power — NOT settled as the candidate
  intended.** The at-N printing cure is accepted in principle
  and is insufficient in fact: the numbers to be printed are
  model-unstated (§8.3) and the decision claim they support
  fails at the headline effect (§8.4). Family-wise, the printed
  paragraph is additionally a point-mass plug-in: fire is
  strongly convex in the lose propensity near the J≥D/4
  threshold (coordinator enumeration at N=30: 0.0071 at lose
  0.10, 0.0384 at 0.14, 0.1684 at 0.20), so modest candidate
  heterogeneity multiplies the expectation — red team B's
  deterministic quadrature at frailty σ=0.5 gives per-decision
  0.0328, expected ≈2.36, P(≥1) ≈0.85 against the printed
  0.0071 / 0.51 / 0.40 (coordinator's crude two-point mixture
  check: 0.0194, 2.7×, same direction).
- **6(e) confidence overlap — settled conditionally for the
  candidate; moot under this verdict.** Red team B's component
  map is verified: surgery, protocol, screens, cell verdict,
  flips, selection, McNemar, and vocabulary are all transplants
  (confidence / canonicalform / confidentiality); the genuine
  residue is three string constants, one small span
  parser/decomposition, and one small rule. The treatment and
  estimand nevertheless differ from confidence's (user-turn
  response-style directive vs confidence elicitation; a gated
  drop rule vs an elicitation readout), and the tree's
  precedent ships probes on thinner residues than this. B's
  sharpest observation is recorded for the restated candidate:
  a confidence EXTENSION could span-regrade both sides inside
  existing machinery and cure §8.5 by construction — the
  standalone candidate's frozen D forbids exactly that cure.
- **6(f) neighbour re-survey — EXECUTED (§8.1); the
  combination survives.**

### 8.8 Fabrication sweep

- **Researcher: CLEAN.** The Kojima resolution was verified
  firsthand by the coordinator at the primary source (§8.1).
  Wei/Sprague fencing stands on round 1's dual resolution
  (§6) and is non-load-bearing. The build-surface estimate in
  the brief is labelled an estimate and UNEXECUTED, as
  required. The brief's two honest concessions (6(b) missing;
  6(a) settled in substance but not in the candidate's text)
  are adopted.
- **Red team A: executions verified; one overstatement
  corrected.** Its span, fixture, CSV, and screen findings were
  independently reproduced by the coordinator or grepped/read
  against the tree. Corrected: the extraction convention is
  recoverable from the tree (the unnamed `_final_line` helper)
  — the defect is that the candidate never names it (§8.6),
  not that no convention exists. Not adopted: its 6(a)
  settlement (decided against, §8.7, with its execution
  adopted in full).
- **Red team B: verified, with a self-correction on record.**
  Its decoded OC model was confirmed cell-for-cell by the
  coordinator's independent enumeration (12/12 cells). Its
  first frailty figures (600-draw Monte Carlo) disagreed with
  a later 4,000-draw run; B resolved the disagreement by
  deterministic quadrature before delivery, flagged the
  correction itself, and the coordinator's independent
  convexity check is consistent with the corrected values.
  Its K4 power figures are a labelled model over unmeasured
  parameters (λ, byte-exact pass rate) — the mechanism beneath
  them is executed (§8.5); the magnitudes are recorded as
  model outputs, not measurements.
- **Coordinator's own interim errors, corrected in this
  section:** the N=30 OC column is reproducible (§8.3) — under
  a model the text never states; and the 0.5027-vs-0.5040
  discrepancy at N=100 is the same unstated-model ambiguity,
  not an arithmetic error by the original round.

### 8.9 The restated candidate must settle, in its own text

1. A frozen, decidable directive-bearing-fixture screen
   (definition of a directive in free prose, turn scope,
   false-positive discipline), with the consequence stated:
   the shipped example population screens out entirely, so the
   text must say what population the instrument is for.
2. The OC model stated in symbols — one reading of "reverse",
   named as conditional or absolute — with the table and the
   family-wise paragraph recomputed under it, and the
   family-wise computed under candidate heterogeneity or
   honestly bounded instead of plugged in at a point mass.
3. The native-side scorer decision: a symmetric span verdict
   on both sides from the recorded spans (which redefines D
   and must be argued against the recorded-selection meaning
   of item 4), OR a printed scorer-only control bounding the
   leniency channel (span-regrade of recorded native spans,
   λ reported per contrast). The winner readout's baseline
   defined under the same choice.
4. Item 4's claim re-scoped to what the statistics carry:
   either the decision rests on the descriptive winner readout
   with the gated label demoted to a sufficient-pattern flag,
   or the headline effect / panel is restated (e.g. 50% drop,
   or a panel whose size is derived in the text) with the
   label mix printed for the candidate's own expected
   (p_nat, N) range — not only at p_nat = 0.9.
5. The final-line extraction convention named by file and
   line (`confidence._final_line`) as binding on stage and
   validator mirror alike, and J's boundary-only membership
   stated in one sentence.
6. The 6(a) population sentence pre-registered in the design
   text (truncated cells stay in the headline population; the
   label's gloss pinned to passes at the recorded
   configuration), and the 6(c) string-scope defence written
   into the claims themselves — or the neutral `pad` arm
   added.
7. Whether the instrument is a new probe or a confidence
   extension, decided in the text against §8.7's 6(e) record —
   the extension shape is the one that cures item 3 above by
   construction.

### 8.10 Build surface, if a future review ever returns BUILD (not a build order)

Researcher's labelled estimate, coordinator-checked against
the tree's recorded blast-radius lessons: new
`responsedirective.py` (~450–600 lines: surgery, protocol,
screens — including the §8.9.1 screen, currently undesigned and
uncounted — span parser/decomposition, rule, at-N OC printer);
experiments stage + native gate; CLI flag with the subcommand
pin audits (test_cycle31/64/84 class); reporting section (three
touchpoints); diffreports section with the full blast radius
(test_cycle67 fuzz oracle + flip-identical anchor in original
AND mutant text + `_identical_expected`; test_cycle27 oracle +
payload stub; test_cycle44 mypy source-file pin); README /
VALIDATION / USAGE counts measured, not derived; ~1,600–2,200
lines of tests. Total ≈2,500–4,000 lines. Nothing in this
section authorizes building any of it.
