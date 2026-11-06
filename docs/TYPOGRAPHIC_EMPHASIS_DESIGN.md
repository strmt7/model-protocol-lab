# Typographic emphasis — design (cycle 237)

`run --typographic-emphasis`: render a frozen fact-recall
panel whose fact sentence's topic span carries one
typographic treatment — verbatim control, bold
(`**span**`), italic (`*span*`), monospace (`` `span` ``),
or the span uppercased — with exactly one treated
occurrence per case, and measure, per candidate (template
family) and per arm, behind an encode-time token-id gate
and a completion guard, whether the treatment alone moves
the model's answers to wrong in-grammar values (the fire
rule), or whether the candidate earns a level-controlled
certificate that the treatment's marginal drop is below
δ = 0.34 (the certification rule). A positive-control
panel that asks which of two facts' topics is written in
bold gates every certificate. This is the falsification
track's predicted-null instrument: it exists so that a
null result on this axis is a certified, level-controlled
statement, not a failure to reject.

This document is the build contract. It consolidates the
wave-12 falsification-track candidate with the fresh
review's amendments A1–A6 folded in (adjudication:
BUILD-as-amended, coordinator #105, 2026-11-04; the A1
derivation debt discharged the same day; built in cycle
237). Where an amendment changed the candidate's text,
the amended text is what appears below; the amendment
list follows so a reader can see what moved.

## Amendments folded in (A1–A6)

- **A1 (the level is completed, δ re-registered).** All
  "dependence-free" sentences are withdrawn (§6.2's
  heading and bound, the δ* bullet, §7's certification
  sentence, §12's TOST bullet). The level is restated
  over the heterogeneous class: per-pair joint laws are
  arbitrary, and the null is a pooled marginal drop of
  at least δ. The extremal computation over the
  completed class (per-pair laws ranging over the full
  vertex set, the event computed exactly) moves the
  bound at the old δ = 0.28 above α, so δ is
  re-registered at **0.34**: the completed extremal
  bound there is **0.013826 per comparison** at
  α = 0.02, and that is the figure §7 and VALIDATION
  print. The restricted-vertex bound at δ = 0.34
  (0.012253), the homogeneous uniform bound (0.005414),
  and the anchored probability at the old δ (0.012232)
  are three different quantities that all print near
  0.0122; every occurrence in this document and in the
  report carries its full name. Family-wise over the
  96 labelled comparisons the certification price is
  E = 1.3273 expected false certificates,
  P(≥ 1) = 0.7373 (independence convention).
- **A2 (the M-gate's derivation).** §6.4's M-gate
  paragraph is replaced by the worst-case-completion
  inheritance argument: a guarded certificate at domain
  size M implies the full-data event E at size 24 under
  the blanket completion, by componentwise monotonicity
  of (D, H, S); the shipped gate requires M = 24 as a
  design decision on that argument, not as a derived
  necessity (at δ = 0.34 the completed bound at M = 23
  is 0.017934, which also meets α; M = 22, at 0.023163,
  does not). The label function is unchanged.
- **A3 (the PC premise stated, the parameter named
  correctly).** §6.3 states the deaf-world cross-arm
  independence premise in words and prints the
  dependence range. The parameter is x, the
  **discordant-cell probability** (x = pB = pH; both-pass
  and both-fail carry 0.5 − x each) — not a joint
  both-pass probability; read as a joint, the curve
  inverts. Values: x = 0.25 → 0.0147 (the independence
  point), 0.30 → 0.0235, 0.35 → 0.0332, 0.50 → 0.0758.
  The PC licence is read as independence-conditioned.
- **A4 (the PC panel is reproducible from text).** §6.3
  writes the PC content template into the text: two
  facts wrapped in the main panel's per-case neutrals,
  the PC target values being the main panel's per-case
  values (value sharing, stated). The build's PC
  generator pair is written fresh on the main panel's
  A/B pattern; the designer script's PC pair was an
  adjudicated tautology (its A/B arms differed only by
  an assert) and is not reused.
- **A5 (the PC completion guard is built and named).**
  The PC gate carries its completion guard
  (k_pc ≤ 6 and B − (H_pc + k_pc) ≥ 8) as an implemented,
  named battery in the build's tests; it existed in no
  designer script.
- **A6 (eight text repairs).** (i) "No Monte Carlo
  anywhere" is scoped: the guard worlds and the §5 fuzz
  are Monte Carlo; the exact DPs reprice the guard
  worlds with no expensive-direction understatement.
  (ii) Certification-curve rows computed at a rounded
  grid Δ are labelled with exact PASTA magnitudes only
  beside the row's own computation Δ (the 0.1791 row:
  grid value 0.0725, exact-at-label 0.0724). (iii) The
  family-wise P(≥ 1) figures state the independence
  convention. (iv) §1's liveness gloss is corrected:
  Tier 1 proves bold-selector use per candidate; the
  other arms' liveness rests on the M = 24
  identity-divergence argument; the PASTA study used
  single-asterisk emphasis and curly quotes, never
  `**bold**`, and §1 does not imply otherwise. (v) §5's
  witnesses are 20 entries, 18 distinct full tuples.
  (vi) The two 0.0179 quantities are disambiguated
  (the anchored Pc-sweep maximum at pc 0.79 versus the
  uniform sup 0.017885). (vii) "Planted gap 0.40 by
  construction" is restated: the panel plants the
  information asymmetry; the 0.40 gap is the sensitive
  world's assumed behaviour. (viii) The "t grid step
  0.001" wording is corrected to the executed bisection
  steps where the two differ.

## 1. Monday-morning utility

A template author deciding whether house style may
bold, italicise, monospace, or capitalise key terms in
local-checkpoint prompts gets, per candidate (template
family) and per treatment arm, a guarded verdict: the
treatment fires (it moved answers to wrong values), or
a level-controlled certificate excludes marginal drops
of δ = 0.34 or more, or the instrument abstains and
says why. Liveness, stated exactly (A6 (iv)): the
positive control proves, per candidate, that the
checkpoint can use a bold selector at all (Tier 1);
the other arms' liveness rests on their treated renders
being token-id-divergent from control on the full
domain (M = 24 by the certification gate) — an arm
whose renders tokenise identically to control is
reported instrument-inapplicable, never as passing.
The published motivation is the PASTA study's finding
that prompt-side emphasis moves task performance
(asterisk emphasis among its treatments; the study
used single-asterisk forms and curly quotes, not the
double-asterisk bold of this panel) — this instrument
measures the same axis on the lab's own terms, with a
certificate where the literature offers an average.

## 2. Arms, span, and mark placement (frozen in code points)

Five main arms, differing only in the treatment of the
fact sentence's topic span:

| arm | span rendering | mark characters |
| --- | --- | --- |
| control | verbatim | none |
| bold | `**span**` | 4 × U+002A |
| italic | `*span*` | 2 × U+002A |
| monospace | `` `span` `` | 2 × U+0060 |
| caps | span uppercased | none |

Bold and italic differ only in asterisk **dose** — the
same character, four times versus twice — which is
what makes the bold/italic contrast a dose reading
(§8). Caps is the delimiter-free internal control: it
changes the span's bytes without adding any mark
character, so a caps-only effect cannot be attributed
to mark parsing. The span is the topic word's
occurrence in the fact sentence `The {span} inspection
code is {value}.`; it begins at code-point offset 4 of
the sentence, the marks are immediately adjacent, and
exactly one occurrence per case is treated — the
question's topic occurrence stays verbatim in every
main arm. Excluded, by axis: quote marks (the
punctuation axis, a different instrument's question);
highlight and colour (no plain-text representation —
the PASTA study's +0.302 / +0.262 effects there are
scoped out of §7's claim in words); strikethrough
(not in the slate).

Two further record arms complete the panel. The
positive control (§6.3) renders each case twice
(`pc_plain`, `pc_info`). The PC-B arm renders the main
case with the topic wrapped in `***...***` at **both**
its occurrences (fact and question): a dose/site-count
probe, reported descriptively, never gating and never
certifying.

## 3. Frozen generated material (complete in text)

The panel is 24 cases with the sibling shape. Case i:
topic = TOPICS[i]; value = WORDS[draw] + "-" + number,
number ∈ [1000, 9999], pairwise distinct across the
panel by redraw; two distinct neutral statements;
content = `{neutral1} {fact} {neutral2} {question}`;
expected = value. TOPICS is the sibling 24
(harbor…birch, identical across invisiblechars,
confusableskeleton, and compatibilityform); WORDS is
canonicalform's 24 (alder…yew); NEUTRAL_STATEMENTS is
density's 24. The module carries its own copies; the
pool contents are pinned by SHA-256 digests of their
canonical serialisation (data and digests, never a
host or library version). The scorer pin is the tree's
own: `core.evaluate`'s byte-exact `answer == expected`
conjoined with boundary presence.

Draws are frozen: the draw material is
`f"{seed}|typographicemphasis|{index}|{slot}|{counter}"`
and the draw is the first 8 bytes of its SHA-256,
big-endian. Slot 0 draws the value's word, slot 1 its
number; slots 2 and 3 draw the neutrals (the second
advances +1 mod 24 on collision); a value colliding
with an earlier case's value redraws slots 0 and 1 at
counter+1; give-up past 1,000 redraws fails the stage
closed (`typographic_emphasis_screen_failed`). The
namespace `typographicemphasis` is new.

The PC panel's draws are part of the frozen material
(A4): slot 4 draws the distractor topic index
(draw % 24, +1 mod 24 when equal to the case index);
slots 5 and 6 draw the distractor value (redrawn at
counter+1 while equal to the case's own value, bound
1,000); slot 7 is the position bit (0 = the target
fact is printed first).

## 4. The panel validator (content-addressing, enforced)

Every generated case passes a frozen battery before
any generation; a failure fails the whole stage
closed. The battery pins: the exact field set and
shapes; the topic's identity with TOPICS[index]; the
value under the frozen grammar (§5) with the expected
value ASCII; two distinct in-range neutrals; the
distractor drawn off the case's own topic and value;
and the generated-text invariants executed against the
builders — exactly one treated span occurrence per
main arm, the question and neutrals byte-identical
across main arms, the PC plain arm carrying no mark
characters anywhere, the PC info arm's only marks
being the target topic's wrap, and the PC target value
being the main case's value.

## 5. Decomposition (ordered decision list; total; executed)

Each completed cell is classified by the first
matching class:

1. `pass` — the verdict passed.
2. `boundary_only` — the answer digest equals the
   expected digest and the boundary alone failed.
3. `wrong_value` — the answer parses under the frozen
   grammar and differs from the expected value. The
   grammar is applied with **fullmatch semantics**:
   `[a-z]+-[0-9]{4}` over the whole answer, the word
   in WORDS, the number in [1000, 9999]. A trailing
   newline or trailing space is out of grammar (the
   cycle-231 F1 rule: `$`-anchored matching classifies
   "value + newline" as in-grammar, a scorer artefact
   that manufactures discordance). This is the only
   class counting toward J.
4. `echo_<arm>` — the answer is in the arm's echo set:
   the whole-value form (marks around the whole value;
   the whole value uppercased for caps) or the
   word-half form (marks around the word part; the word
   part uppercased for caps — for caps the two forms
   coincide). The control's echo set is empty.
5. `form_only` — the answer differs from the expected
   value, is not class 4, and stripping the arm's marks
   (asterisks for bold/italic, backticks for monospace,
   lowercasing for caps, identity for control) yields
   exactly the expected value; unreachable for control
   by construction.
6. `non_answer` — everything else, including a marked
   topic offered as the answer (`**harbor**`) and marks
   under the caps arm (`**ALDER**-1234`).

A markdown-stripped echo is indistinguishable from the
value itself and lands in `pass` by construction;
disclosed here, not hidden. The pair decomposition is
the sibling total partition: equal answer digests are
a `boundary_only` pair; differing digests with both
boundaries found are `exact_only`; anything else is
`mixed` (reported, never attributed). The executed
evidence is the designer's battery (20 entries, 18
distinct full tuples — A6 (v)) plus this build's
fullmatch witnesses (trailing newline, trailing space)
and a 10,000-case totality fuzz in the build's tests.

## 6. Headline statistics, floors, guard, operating characteristics

Population, per (candidate, treated arm): the cases
completed in both control and the arm **and**
token-id-divergent for the arm (the regime gate is
measured at encode time on the full render ids against
control: n_ids_identical, n_ids_divergent, the per-case
id_length_delta list, max_abs_id_delta, and
n_same_ids_verdict_flips are all reported). D = control
passes in the population; J = of those, the arm cells
decomposed `wrong_value`; n_reverse (control-fail /
arm-pass pairs) is printed beside every label and never
pooled into J. M = the divergent-domain size,
k = the domain's incomplete count, N = M − k.

### 6.1 The fire rule (sibling shape, transferred unchanged)

Labels: `insufficient` (D < 10, or the guard withholds);
`no_discordance_observed` (D ≥ 10, J = 0, guard passes);
`discordant_asymmetric` (D ≥ 10, J ≥ 6, guard passes —
the arm **fires**); `below_threshold` (D ≥ 10,
0 < J < 6, guard passes); and §6.2's
`equivalence_certified`. The floors (MIN_D = 10,
MIN_J = 6) are the sibling pair, transferred; their
grid is reproduced digit-for-digit by the designer's
exact enumeration: competent-null fire 0.017274
independent and 0.004442 paired (Gaussian copula
ρ = 0.5, exact by quadrature); weak panel (control
rate 0.80) 0.1743; random labeller 0.550306; power at
base 0.90 is 0.2542 (10-point drop), 0.8159 (25-point,
paired 0.698078), 0.9990 (50-point); at base 0.85 the
25-point power is 0.8761. The PASTA-magnitude rows,
computed at the stated grid drops with the exact
magnitudes printed beside them (A6 (ii)): grid
Δ = 0.1791 → fire 0.5831; Δ = 0.2514 → 0.8193;
Δ = 0.4145 → 0.9909; Δ = 0.5544 → 0.9998. Family-wise
over the 96 labelled comparisons (4 arms × 24
candidates): competent fire expectation 1.6583,
P(≥ 1) = 0.8123 (independence convention — A6 (iii)).

### 6.2 The equivalence certification (this round's reason to exist)

The certification statistic runs over **all**
completed pairs in the id-divergent domain: H =
control-pass / arm-fail pairs over **every** failure
class, R = control-fail / arm-pass pairs, S = H − R.
The event is **E = {D ≥ 10 and H ≤ 5 and S ≤ 0}**
(δ-independent; E and fire are mutually exclusive by
construction, since H ≤ 5 implies J ≤ 5). The label
`equivalence_certified` issues only when **all three
gates** pass: (1) the statistic gate — E holds and
**M = 24**; (2) the guard gate — E is stable under the
worst-case completion of the domain's incomplete pairs
(H + k ≤ 5 and S + k ≤ 0); (3) the positive-control
gate (§6.3) passed for that candidate in that run. On
any gate's failure the arm keeps its sibling label and
the blocking gates are printed by name
(`statistic_event`, `domain_size`, `completion_guard`,
`positive_control`). The label never prints without its
companions: δ, α, M, N, H, R, S, and the PC status.

The level (A1): over the heterogeneous class — per-pair
joint laws arbitrary, the null a pooled marginal drop
of at least δ — the completed extremal bound at
δ = 0.34 is **0.013826 per comparison** at α = 0.02.
Under the anchored coupling (independence at control
rate 0.90) a true-null candidate certifies with
probability 0.5974 (the null yield), and
P(certify | Δ = 0.34, pc = 0.90) = 0.0034; the anchored
curve at Δ = 0.05 / 0.10 / 0.15 / 0.20 / 0.25 / 0.28 /
0.30 / 0.4145 is 0.3796 / 0.2165 / 0.1115 / 0.0519 /
0.0217 / 0.0122 / 0.0081 / 0.0005 (grid values; the
0.28 row is the anchored probability at the old δ,
0.012232 unrounded — not the licence). The anchored
δ* is 0.2545 and the uniform δ* is 0.2739. The
homogeneous uniform sup at Δ = 0.34 is 0.005414
(context only, never the licence). A real-table
counterexample world certifies at 0.022662 under the
restricted class at the old δ, which is what forced
the completed class. Two further 0.0179 quantities are
distinct (A6 (vi)): the anchored Pc-sweep maximum at
pc 0.79, and the uniform sup 0.017885; neither is
printed as a headline figure. The certification
operating characteristics under the paired copula
(ρ = 0.5) are 0.6139 at Δ = 0 and 0.0044 at Δ = 0.28.
The combined label function is a total function of
(D, J, H, R, k, M, pc_passed); its exhaustive audit
(the designer's, re-executed in the build's tests)
finds 0 manufactured-fire, 0 unstable-acquittal, and
0 unstable-certification violations, 1,225
certification states, and 245 states where a
certificate issues while the sibling acquittal would
withhold (J = 0, k > 0, guard stable) — a disclosed
inversion: the certificate is the stronger claim and
carries the completion guard as its own stability
condition, so the inversion is a property of the two
claims, not a hole. One Monte Carlo note (A6 (i)): the
guard worlds of §6.4 and the §5 fuzz are simulations;
every other figure in this section is an exact
enumeration or an exact quadrature, and the exact DPs
reprice the guard worlds with no expensive-direction
understatement.

### 6.3 The positive control (gate for certification)

Per candidate, over the PC panel's completed pairs:
B = plain-fail / info-pass pairs, H_pc = plain-pass /
info-fail pairs, info_passes = pc_info passes,
k_pc = incomplete PC pairs. The gate passes iff
info_passes ≥ 10 **and** B − H_pc ≥ 8 **and** the A5
completion guard holds (k_pc ≤ 6 and
B − (H_pc + k_pc) ≥ 8; info_passes is unchanged under
that completion by construction). The PC content
template (A4): content = `{n1} {factA} {factB} {n2}
{question}` with the main case's own neutrals n1/n2;
factA/factB are the target fact (the main case's topic
and **the main case's own value** — value sharing) and
the distractor fact, in the order the position bit
fixes; the question, byte-identical in both PC arms,
is `What is the inspection code of the topic written
in bold?` pc_plain carries no marks anywhere (no
correct answer by construction); pc_info is identical
bytes except the target topic's fact occurrence is
bold-wrapped. Expected = the target value in both
arms.

The floors are derived under the PC's own null — the
deaf guesser, for whom the marks channel carries
nothing and each arm passes at 0.50 — by an exact
four-category multinomial: at margin thresholds
5 / 6 / 7 / 8 the deaf gate rates are 0.0966 / 0.0557 /
0.0297 / 0.0147, so 8 is the smallest threshold
meeting the licence bound, with the info-pass floor
at 10. The gate's operating characteristics: sensitive
world (info 0.90, plain 0.50 — the panel plants the
information asymmetry; the 0.40 gap is that world's
assumed behaviour, A6 (vii)) 0.7709; partial worlds
0.4668 (info 0.80) and 0.3250 (info 0.75); a candidate
good on both arms (0.90 / 0.90) 0.0002; the
deaf-abstainer 0.0000 (blocked by the info-pass floor
by construction). The deaf rate is derived under a
cross-arm independence premise, stated here in words
(A3): the two PC arms' per-case outcomes are
independent given the candidate. The premise is a
modelling choice, and the dependence range is printed
with it: parameterised by x, the discordant-cell
probability (x = pB = pH, both-pass and both-fail
carrying 0.5 − x each), the deaf gate rate is 0.0147
at x = 0.25 (the independence point), 0.0235 at
x = 0.30, 0.0332 at x = 0.35, and 0.0758 at x = 0.50 —
so the PC licence is read as independence-conditioned.

### 6.4 Completion guard + incompletion perimeter (both rules)

The completion domain is exactly the id-divergent
incomplete cases — an encode-time property, decidable
whether or not the generation completed. Hidden slots
(completed cells whose span is unparseable) are inside
the perimeter as `non_answer`: they count as failures
for H and the rates, never toward J, and n_non_answer
is printed per arm. Fire uses observed counts only;
`no_discordance_observed` issues only at k = 0;
`below_threshold` survives only while J + k < 6; the
certification guard is §6.2's gate (2). The cap:
k > 6 of 24 withholds **all** labels (`insufficient`) —
6 is the largest k leaving at least 18 paired cases.
The M-gate (A2): the certification statistic gate
requires M = 24 because a guarded certificate at any M
implies the full-data event E at size 24 under the
blanket worst-case completion, by componentwise
monotonicity of (D, H, S) — the certificate therefore
speaks about the full panel or not at all; the label
function is unchanged by the argument, and the
heterogeneous completed bounds at M = 23 (0.017934,
meets α) and M = 22 (0.023163, does not) are printed
in §6.2's licence paragraph so the decision's price is
visible. No sign test appears anywhere; McNemar's exact
test (`confidentiality.mcnemar_exact_p`) is descriptive
only, never a decision input. The executed attack
worlds (10,000 trials each at the designer's seed,
every case id-divergent — the pinned domain; the
build's battery adds an ids-dimension variant at
divergence probability 0.8): world A (zero-effect
manufacture) certifies guarded at 0.0016 and fires at
0.0047; world B at δ (unguarded certification 0.2709)
certifies guarded at 0.0127, and on the PASTA profile
at 0.0006; world C (a true 25-point drop) fires
guarded at 0.6318.

## 7. Pre-registered claims (falsifiable, guard-qualified, scoped)

The fire claim is falsified exactly when a guarded
label `discordant_asymmetric` issues for a (candidate,
arm) whose true in-grammar wrong-value discordance is
absent. The certification claim — true marginal drop
below δ = 0.34 — is asserted **only** when
`equivalence_certified` issues, at the §6.2 licence
(completed extremal bound 0.013826 per comparison,
α = 0.02, family-wise price printed beside it).
Scope exclusions, in words: the certificate is silent
below 0.34 and does **not** exclude the PASTA-study
average effects 0.1791 / 0.2514; no highlight or colour
treatments; no marks-as-selectors use; no churn among
cases both arms fail; no ids-identical cases; no quote
marks, strikethrough, other spans, other doses, or
treatments of instruction-segment spans. Every claim
is per candidate and per arm; there is no pooled
headline and no winner readout.

## 8. Cross-arm readings and isolation

Frozen readings, stated in advance: a bold fire beside
an italic null (or the reverse) is an **asterisk-dose
response** — the arms differ only in dose of the same
character — never two independent findings. A
monospace-only fire reads as "a code-span treatment
moves this candidate"; the emphasis reading is withheld
(code spans carry their own semantics). A caps fire
with the mark arms quiet is a case-channel effect. All
four arms firing is surface fragility, not four
findings. The PC-B arm's rate beside bold's is a dose
and site-count signal for a future design, never a
contradiction of the bold label.

## 9. Risks and pre-registered readings

The predicted result is the null: the published
magnitudes on this axis are averages over task suites,
and this panel's certificate is deliberately coarse
(δ = 0.34) — the instrument trades resolution for a
level it can state honestly over arbitrary per-pair
dependence. A certificate therefore does not read as
"emphasis is safe"; it reads as "a drop of 0.34 or
more is excluded at the stated level, for this span,
this candidate, this checkpoint". The coarse δ is the
price of the heterogeneous class: the restricted-class
bound would have licensed the old δ = 0.28, and the
real-table counterexample (0.022662) is the executed
proof that the restriction was load-bearing. The
remaining risks are the sibling ones: a checkpoint
whose tokenizer makes a treated render ids-identical
shrinks the domain (printed as M, never silently); a
candidate whose PC gate fails for capability reasons
unrelated to the main panel loses certificates, not
fires (the demotion is printed by name); and
incompletion inside the divergent domain is priced by
the guard everywhere a label issues.

## 10. Fail-closed vocabulary, record shape, naming

Whole-stage payload: `{"available": False, "detail":
"typographic_emphasis_unavailable"}`. Stage details:
`cancelled`,
`typographic_emphasis_record_budget_exceeded`,
`typographic_emphasis_screen_failed`. Per-cell details:
`""`, `arm_context_exceeded`, `arm_evaluation_failed`.
Rates are None only when their denominator is 0.
Report strings pass through the reporting layer's
escaping. Budget: 24 × 5 main + 24 × 2 PC + 24 × 1
PC-B = 192 generations per candidate; 4,608 at the
24-candidate maximum; 576 ground records; a 12 MiB
serialised-records cap. The record per (candidate,
case) carries: the fixture id, the topic, the expected
digest, the per-arm content digests (all eight record
arms), the per-arm token counts and render-id digests,
ids_identical and id_length_delta per treated arm, and
the cells with their cell details per arm. The
analysis payload carries, per (candidate, arm), the
certification inputs (M, N, k, H, R, S) and the PC
gate's counts (B, H_pc, info_passes, k_pc); the
diff-reports mirror recomputes every label — including
`equivalence_certified` — from those committed counts,
so a certificate cannot be asserted by a record whose
counts do not earn it.

## 11. Duplication check (verified against the tree)

The fire rule, guard, decomposition shape, and stage
machinery are the sibling instruments' (invisiblechars,
confusableskeleton, compatibilityform,
enumerationconvention), transferred deliberately; what
is new is the axis (typographic treatment of a carrier
span, with a delimiter-free case arm), the
certification statistic and its completed-class level,
and the bold-detection positive control that gates
certification. No existing probe renders emphasis
treatments, certifies an equivalence at a stated level,
or runs a marks-channel positive control. The
uniqueness pass for the combination is recorded in
docs/RESEARCH_LOG.md (cycle 237 entry): behavioural
probing of a checkpoint's own rendering pipeline with
a level-controlled certificate and an abstention
vocabulary was found in no surveyed neighbour.

## 12. Rejected alternatives

A TOST procedure on the paired difference at the old
δ (rejected: its level statement needs the dependence
assumption A1 withdrew). A sign test or McNemar gate
(rejected: wrong estimand — the certificate is about a
marginal drop, and both tests answer a symmetry
question). Highlight/colour arms (rejected: no
plain-text representation; a markup-language rendering
would change the axis). Quote-mark arms (rejected:
punctuation axis). Treating every topic occurrence
(rejected for the main panel: it confounds dose with
site count; the PC-B arm carries that contrast
descriptively instead). A pooled cross-arm headline
(rejected: the arms answer different readings, §8).
Certification without the positive control (rejected:
a certificate from a candidate that cannot use the
marks channel at all would be a statement about the
instrument, not the treatment).

## 13. Build debts

(1) The guard battery gains the ids dimension and
re-executes worlds A / B-δ / B-PASTA / C under the
pinned id-divergent domain. (2) Generator asserts and
freeze tests: expected ASCII; exactly one treated span
occurrence; question and neutrals byte-identical across
main arms; PC distractor value ≠ target value and
distractor topic ≠ target topic. (3) Forced-collision
freeze tests (main slots 0+1 redraw; PC slots 5+6
likewise). (4) An OC/certification test reproducing
every §6 figure from its own code (the paired
quadrature 0.004442 / 0.698078, the uniform sup table,
both δ* bisections, the PC floor derivation); the
heterogeneous completed-class sup is pinned against
named single-assignment vertices computed by an exact
count DP (the RTB fixed assignment reproducing
0.022662 among them), the full backward DP being an
archived coordinator derivation whose state space does
not fit a unit test. (5) The decomposition battery
(§5's witnesses plus precedence, and the build's new
trailing-newline and trailing-space witnesses). (6)
The diff mirror recomputing the combined label from
committed counts, with the named test that a
certificate cannot be asserted by a record whose
counts do not earn it. (7) The pre-build census re-run
(long options 85 → 86, this section as USAGE §8.64,
the source-file pin 73 → 74). (8) USAGE §8.64 and the
VALIDATION entry carrying the δ/α licence and the
family-wise prices.

## 14. Sources and provenance

The design consolidates: the wave-12 falsification-
track candidate and its designer evidence scripts
(exact fire grid, certification enumerations, PC gate
enumeration, guard audit and worlds, decomposition
battery and fuzz — all re-derived digit-for-digit by
the readiness lane before the build); the adjudication
amendments A1–A6 (coordinator #105, 2026-11-04) with
the A1 derivation record (the completed-class
extremal computation, the M-row table, the RTB
counterexample check); the sibling instruments'
shipped designs for every transferred shape; and the
PASTA study's emphasis magnitudes as the axis's
published motivation (scoped as in §1: its asterisk
treatments are single-asterisk forms, and its
highlight/colour effects sit outside this panel's
claim). The competitive-research program's uniqueness
framing (combination claims only, primary-sourced)
governs §11.
