# Multi-question probe — fresh design review of the bundle resubmission candidate (ADJUDICATED: NO-BUILD)

## Round summary (coordinator #47, design round only — nothing built)

Fresh review of the resubmission candidate recorded in
`docs/BUNDLE_DESIGN.md` §3: `run --multi-question`, specified in
`~/workspace/agents_out/bundle/research.md` §3. Under the program's
standing settlement a resubmission is reviewed as a NEW design:
this review could return BUILD (a build order for the frozen
instrument) or NO-BUILD (permanent closure of this candidate);
silence on any of the seven agenda points is NO-BUILD.

- Tree at main `a6705d1b9607` (cycle 213 closed green, 8,442
  tests, precise coverage 95.3455%). This round was READ-ONLY
  except for this doc and the briefs; no suite was run, no push
  was made, no tag was cut or discussed.
- Specialists (blind to each other): researcher
  (`~/workspace/agents_out/multiquestion/research.md`), red-team
  A (`redteam_a.md`), red-team B (`redteam_b.md`),
  verifier/claims auditor (`verifier.md`). All four briefs
  landed; the coordinator re-executed the load-bearing claims
  itself (§4).
- Convergence: **unanimous NO-BUILD for the candidate as
  literally written** in bundle research.md §3 — both red teams,
  the verifier, and the researcher himself (whose BUILD verdict
  attaches only to his restated instrument, §3 of this doc, and
  who states plainly that without his amendments the candidate
  is NO-BUILD). The coordinator's verdict is NO-BUILD for the
  candidate in both forms, for the reasons in §1: the restated
  instrument cures the grammar, screen, and rotation defects
  (closures the verifier verified and the coordinator concurs
  are negation-configuration completions, §2), but three
  load-bearing defects survive the restatement untouched —
  the decision rule's mechanism-impure firing condition, the
  liveness rule's post-treatment population selection, and
  two-family cancellation inside the case aggregate — and every
  cure for those changes the rule, the headline, or the
  liveness discipline, which this review may not mint.
- **Fabrication sweep: clean — a seventh consecutive round.**
  The verifier resolved every tree citation in the candidate
  spec exactly (one range overrun and one flag/dispatch range
  conflation, both non-load-bearing), reproduced the rank-5 OC
  table's independent-world rows exactly, and re-ran the fresh
  briefs' load-bearing PoCs (bit-for-bit where the author's own
  code was re-run). The one false statement found this round is
  in the candidate spec itself (§3.4's clustered-drop caveat),
  not in any fresh brief. Non-load-bearing imprecisions are
  recorded in §5.

## 1. Why NO-BUILD: the fatal defects

Each defect below is individually blocking. Attribution:
R = researcher, A = red-team A, B = red-team B, V = verifier,
C = coordinator.

### F1 — The rule's firing condition is mechanism-impure, and its calibration never simulates the instrument's own parser (A (g); C reproduced)

The decision rule counts a bundled slot as not-pass for
`wrong_value` **or** `parse_failed` alike (§3.3 of the
candidate), so the label `multiquestion_drop_observed` — under
a utility sentence about whether bundling "moves per-question
verdicts" — can be fired by formatting events alone, in worlds
with zero per-question knowledge difference. Executed:

- **Symmetric per-line slip null** (each answer line acquires
  an extra word with probability s, identical behaviour in both
  arms; knowledge pass 0.9 given parse): fire **0.023 / 0.062 /
  0.194 at s = 0.02 / 0.05 / 0.10, K=5** (A; C reproduced
  0.024 / 0.061 / 0.189 with independent code). The mechanism
  is structural: a slip voids the whole bundled cell (all K
  slots not-pass) with probability 1−(1−s)^K, while a split
  cell's slip voids one slot. The candidate's §3.3 sentence —
  the word rule "applies identically to both arms, so it cannot
  manufacture discordance by asymmetry" — is executed false at
  cell level: the code is symmetric, the failure surface is not
  (V concurs, §6.2 of its brief).
- **Formatting-collapse worlds** (bundled generations collapse
  to preamble/prose with probability q, per-question knowledge
  identical): the drop label fires **0.11 / 0.28 / 0.64 at
  q = 0.15 / 0.25 / 0.40** (A; C's whole-case-collapse variant
  fires 0.054–0.181 at c = 0.10–0.20). Every discordant slot in
  these worlds is a `parse_failed`, none a `wrong_value`.
- The printed operating characteristics (fire ≤ 0.017 under
  every simulated null; family-wise ≈ 0.57–0.81 false labels
  per run) are all **verdict-level** nulls: none of them
  simulates the parser the headline runs through. Under the
  instrument's own mechanisms the family-wise expectation at a
  10% per-line slip rate is ~9 false labels per run at K=5, an
  order of magnitude above the printed figure (A G3).

The house standard is negation's: its joint rule counts only
`wrong_label`, putting the echo/boundary decompositions
*inside* the firing condition so a non-target mechanism cannot
fire the label. Here the decomposition is merely printed
beside a label that parse failures alone can fire — and the
candidate's own parser section states that a parse-failed slot
has "no slot verdict claimed", while the rule counts it as a
verdict that moved. The researcher's defence (parse failure is
production-faithful: the front-end gets no usable answer) is
undercut on its own terms: the frozen parser is stricter than
the production behaviour invoked — it voids five correct,
correctly labelled answers over a one-line preamble (A G2).
Curing F1 requires either a mechanism-pure firing condition
(the rule changes) or a re-scoped headline — usable-answer
loss under the frozen convention, with parser-in-the-loop
nulls as the headline OC (the claim changes). Both are
load-bearing elements; under this review's settlement neither
may be minted here. **F1 stands against the restated
instrument as much as against the literal spec**: the
restatement (§3) keeps parse-failure inside the firing
condition by design and its fresh OC grid still contains no
parser-in-the-loop null.

### F2 — The headline population is selected by the treatment: liveness conditions on the bundled cell completing (A (d), B §D; unanswered by the restatement)

A case enters the headline iff the bundled cell **and** all K
split cells completed with `boundary_found`. The bundled cell
is the treated generation: a candidate whose bundled behaviour
is worst (rambling, restating, boundary loss — behaviours that
grow with K) loses exactly those cases, and its decision is
computed over the terse residue. Executed magnitudes under a
stated correlated-attrition model (30% of cases "hard", true
drop 0.6 there, bundled-side boundary loss on hard cases): fire
collapses 1849/2000 → 305/2000 at K=3 with `insufficient`
becoming the modal outcome (B §D). Two candidates' decisions
are computed over different case sets, selected by each
candidate's own bundled behaviour; printing the attrition
counts does not unselect the population. This is the same
defect class the candidate itself invokes to reject
conditioning the headline on parse-ok cells (its §6(c):
conditioning on a post-treatment outcome, answer-syntax F2) —
applied to liveness, it is nowhere answered, in the spec or in
the restatement. The restatement answers the *magnitude*
question honestly (attrition degrades the decision to
`insufficient`, never to a drop label — a manufactured null,
not a manufactured drop) but not the selection question.

Compounding inconsistency (A (d) point 2): a cap-truncated
bundled reply and a parse-failed bundled reply are
production-identical outcomes — the front-end gets no usable
answer from either — yet the first is excluded from the
headline as attrition while the second is counted as
not-pass. The candidate's production-faithfulness principle is
applied to one mechanism and not the other. Every cure
re-opens a load-bearing element: counting truncated cells as
not-pass revives slot-ordered cap censoring (rank-5 F5) inside
the headline; conditioning liveness on anything else changes
the population definition.

What is NOT defective here (conceded by both red teams, C
verified): slot-ordered censoring of correct replies is
impossible by construction under the single-word screens — the
worst legal correct K=5 reply is ≤ 49 bytes (R §2(d), V
re-verified the bound), far under the 64-token cap
(`cli.py:198`); the sketch's F5 manufacture does not survive.

### F3 — Two-family pooling cancels inside the case aggregate (B §C; A (c))

The panel pools two families with different failure physics —
planted-fact extraction (alphabetic pool words) and small
arithmetic (digit strings adjacent to the marker space) — into
one per-panel decision, aggregated as d_c = f_c − p_c summed
across families **within a case** before the sign test.
Executed (B §C, A's (c) in agreement):

- An arithmetic-only drop (50% on split-pass arithmetic slots,
  extraction drop zero) fires the unqualified pooled label in
  1982/2000 (K=3) and 2000/2000 (K=5) panels.
- **Opposite family effects cancel**: extraction drops 45
  points while arithmetic gains comparably → the pooled
  decision is `below_threshold` in ~1990/2000 panels at both
  K. Two large, real, opposite packaging effects are both
  invisible — not diluted, cancelled — by the aggregation.
  Density's case-aggregation settlement was derived over one
  task family; this population has two, and cases differ in
  family exposure (2F1A vs 1F2A alternating), so a family effect
  enters d_c with case-varying weight the sign test discards.

The original spec froze no per-family readout at all (A (c)
fatal as written). The restatement mandates per-family f/p,
pass rates, and parse decomposition in print, and requires the
section text to name a family-concentrated drop — that cures
the *reading*, on the negation per-stratum precedent. It does
not cure the decision's cancellation blindness: the label
itself can be manufactured by one family and annulled by
opposite effects, and no frozen rule states what the section
may claim when the families disagree in sign. Per-family
decisions would need their own derived rule and family-wise
accounting (96 decisions) — a rule change, not a print.

### F4 — The literal spec is not executable as frozen: open screen, non-total grammar, unfrozen generator (A (a)/(b), B §A/§B, V §2.4/§3; C reproduced)

Individually fatal to the candidate **as written**, each
executed by at least three lanes:

- **Marker/sum collision (charter point (g)).** Markers are
  m_j = `density.digest_text(str(j))`; §3.1 screens fact values
  away from "1".."5" but places no screen on arithmetic sums.
  A correct bundled reply whose arithmetic sum lands in {1..K}
  parse-fails the whole cell while its split cells pass
  (C: `1: ochre\n2: 3\n3: cobalt` → `extra_words`, all slots
  parse_failed, f=3, d_c=+3 from one correct case). The
  breakage is arm-asymmetric and sign-aligned: it manufactures
  only drop evidence, never gain evidence (V §3.2, incl. the
  sum=1 case manufacturing f = K−1 via cell voiding). At
  population scale, a *perfect* model draws
  `multiquestion_drop_observed` in **69% (B) to 79% (V) of K=5
  panels** under the natural single-digit operand reading —
  and §3.1 freezes no operand distribution, so the design
  cannot bound its own false-fire rate within two orders of
  magnitude (A G6: 0.70 at U{1..9}, 0.00 at U{1..99}).
- **The class system is not total, nor disjoint.** Under
  §3.3's literal class definitions, a segment holding exactly
  one marker-valued word satisfies none of `ok` /
  `missing_marker` / `extra_words` (V §2.4 executed the literal
  predicates; C's independent implementation had to invent the
  residual clause to classify `1: 1` at all), and the literal
  `extra_words` second clause overlaps `missing_marker` with
  no precedence frozen. The "total parser" the scorer rests on
  does not exist in the frozen text.
- **The generator is described, not frozen.** K=5 rotation is
  "analogously" (two honest builders produce different panels;
  B §E exhibits a 15/9-skewed natural reading); the operand
  distribution, topic/value draw discipline, and seed/counter
  derivation are unstated — against the NEGATION standard,
  where the instrument's generator was shipped, cited code.

The closures for all of F4 exist, were specified exactly, and
were verified by execution this round (V §3.3, R §2(a)/(e)):
exclude sums {1,2,3,4,5} in both panels (the researcher's
operand floor a,b ∈ [3,99] ⇒ sums ∈ [6,198] is one valid
freeze: 0/9,409 grid sums in the marker set; 1,891 generated
correct cases parse `ok` bundled and split, 0 failures,
re-run bit-for-bit by V); the residual-class clause (a segment
word ∈ M ⇒ `extra_words`; `missing_marker` otherwise when the
in-order scan fails) making the grammar total and disjoint;
the K=5 pattern pair frozen as F,A,F,A,F / A,F,A,F,A by case
parity (re-counted: every slot exactly 12F/12A, totals 60/60).
**The coordinator concurs with the verifier that these are
negation-configuration completions of unfrozen parameters, not
the ground of this NO-BUILD** (§2). They are moot for the
verdict because F1–F3 stand against the restated instrument
that already contains them; they are recorded frozen in §3
for the next candidate.

### F5 — The candidate's operating-characteristics evidence is materially incomplete in both directions (all lanes; C reproduced)

- §3.4's table reproduces **exactly** under its stated
  verdict-level worlds (V 100k trials/cell; A G0; C spot MC:
  random 0.0157, competent-null 0.0110, independent drop-0.25
  power 0.9772/0.9991 — all within seed noise of the printed
  0.0169 / 0.0118 / 0.9787 / 0.9993). The arithmetic is honest.
- Its clustered-drop caveat — "a case-clustered drop is
  *easier* for this rule, so the power rows are not
  overstated" — is **false**. At the same marginal 25% drop,
  case-clustered power is **0.4165 (K=3) / 0.3401 (K=5)**
  (V, marginal-matched model), falling to 0.18–0.28 at
  attrition-reduced N=18–20; under the researcher's
  collapse-rate-0.25 model it is **0.5803 / 0.5860** at N=24
  (R, analytic cross-check 0.5767/0.5778 matching). The two
  figures differ by drop model; under every executed clustered
  model, power at the advertised drop is roughly half or less
  of the printed independent row. The mechanism: the sign test
  reads only the signs of d_c and the D≥6 gate only the count
  of discordant cases; clustering concentrates the same
  marginal drop into fewer cases.
- No at-N characteristics exist in the candidate as written
  (the charter's point (b) from BUNDLE_DESIGN §3): power at
  live N=8 falls to 0.16 (A G4), while §3.3 makes attrition a
  designed outcome of the instrument. The restatement's fresh
  grid (§3 of the researcher brief) supplies at-N and
  clustered rows and is adopted into §3 of this doc as the
  next candidate's mandatory print — with F1's
  parser-in-the-loop rows still missing from every grid
  produced this round except the red teams'.

### F6 — Record binding: the two carried commitments are unbound (A (h); V (d) condition)

The headline quantities recompute from the carried word-hash
lists alone (A's attack failed there, conceded; V concurs) —
the density discipline's core holds. But the record carries
the hash list **plus** a raw `answer_digest`, neither derivable
from the other, with no `commitment_digest` over the list —
the binding density itself uses (`density.commitment_digest`)
and the candidate claims to follow. Executed consequence
(A H1/H2): a record pairing a passing answer's hash list with
an all-wrong generation's digest passes every recompute the
spec defines, so the byte-exact sensitivity — the reader's
only check that the lenient headline is not a formatting
artefact — can be made to disagree with the headline
undetectably. Additionally `id_length_delta`, the
bundle-disclosure evidence, is stage-attested only: §3.3/§3.5
carry no render-id digests or lengths (negation §3.4 required
them carried). Fixable in the record shape; recorded as frozen
requirements for the next candidate (§3). Not independently
fatal; it compounds F1, whose printed decomposition is the
mitigation this binding protects.

### Non-fatal, adjudicated

- **Rotation parity-collinearity** (A (e), V §6.2): under the
  two-pattern freeze, family is a deterministic function of
  (case parity, slot parity); slots of the same parity carry
  the same family in 24/24 cases. Marginal balance — the
  design's stated claim — holds exactly (C counted 12/12 per
  slot per family at both K). The joint-structure residue is
  real and must be disclosed or designed out (richer pattern
  set) in the next candidate; it does not defeat the marginal
  claim.
- **Cap pinning unstated** (B §D.3): the candidate does not say
  whether the probe pins its own generation cap (density pins
  `MAX_NEW_TOKENS=64`) or inherits the run's `--max-new-tokens`
  (negation inherits). A freeze item for the next candidate,
  with the ≤49-byte correct-reply bound stated beside it.
- **`wrong_value` is a new closed-field label** (V §1.2.4): no
  tree occurrence; pattern-consistent with numform's
  `wrong_label` lineage and not a decision/detail/stage word.
  It goes through the build cycle's naming audit; not a
  vocabulary violation.
- **Class names misdescribe two events** (A (b), B, V):
  out-of-order replies classify `missing_marker`, and an empty
  segment classifies `extra_words`; the printed decomposition
  inherits the misdescription. Cosmetic, headline-neutral
  (both are `parse_failed`), recorded for the next candidate's
  grammar text.
- **Undisclosed leniency** (A via V §6.2): `-7` passes for
  expected `7` (the minus sign is a word separator under
  `normalize_words`). Symmetric across arms; a disclosure line
  for the next candidate's leniency paragraph.

## 2. Agenda settlements (the seven chartered points)

(a) **Parser grammar in full, incl. the sum/marker screen —
OPEN as written (F4); closure specified and verified.** The
grammar's classes are non-total as written and the screen set
is not closed; the verified closure (uniform sum screen
{1..5} / operand floor [3,99], residual-class clause, frozen
scan convention) is recorded in §3. Silence would have been
NO-BUILD; the point was answered by execution on every lane,
and the answer is that the frozen text does not contain a
total parser.

(b) **Parser behaviour on swaps, omissions, duplicated slots,
out-of-order replies — SETTLED.** Three lanes' executions agree
cell-for-cell (C's 25-case battery included): swaps under `ok`
surface as `wrong_value` with the swap flag (the other slot's
expected hash, well-defined under the distinctness screen);
omission (either numbering), duplication, out-of-order, and
preamble all void the whole cell to `parse_failed` with no
slot misattribution — the rank-5 F6 positional-parser defect
is cured by marker alignment. Marker-valued swaps are
invisible, but that is F4's subdomain, closed by F4's screen.

(c) **One population or two strata — NOT SETTLED.** The
population definition itself is settled (one generated
population per panel; extraction and arithmetic are rotating
strata of it; the claim is scoped to the mixed panel; genuine
recall is a different instrument and is not smuggled in — R
§2(c), V §5(c)). The headline's behaviour over the strata is
not: F3's cancellation inside d_c means the pooled decision
can be annulled by opposite family effects, and no frozen rule
governs family sign-disagreement. A print does not settle a
decision.

(d) **K=5 cap discipline — NOT SETTLED.** The F5 sub-point is
settled by construction (correct replies ≤ 49 bytes; whole-case
liveness removes slot-ordered censoring; both red teams
concede the sketch's manufacture does not survive). The
liveness discipline itself is F2: post-treatment population
selection with executed magnitudes, an inconsistent
truncated-vs-parse-failed discipline, and (as written) no
at-N characteristics. The budget asymmetry (bundled 1×64 vs
split K×64 across generations) is constitutive and disclosed;
it is not the defect — the selection is.

(e) **Family×slot rotation completeness — SETTLED as
completed.** K=3 was frozen and exact in the candidate; K=5
was "analogously" (open as written). The frozen completion
(F,A,F,A,F / A,F,A,F,A by case parity) is verified by counting
on three lanes: every slot 12F/12A at both K, panel totals
36/36 (K=3) and 60/60 (K=5), imbalance zero. Parity-
collinearity (§1 non-fatals) is the recorded residue.

(f) **No-selection-readout discipline — SETTLED, clean.**
Red-team A's attack failed by grep + full read; B and V
concur; C concurs. No winner, selection, flip-against-primary,
cross-K trend, or survival readout exists anywhere in the
candidate or the restatement; every readout is per-candidate,
per-panel. The numform settlement (`experiments.py:11049`;
repeated at `:10018` for negation) is honoured.

(g) **Naming audit for `--multi-question` — SETTLED, clean.**
Executed greps (R, B, V, C): `multi-question` / `multiquestion`
/ `multi_question` occur nowhere in code, USAGE, FAQ, CLI help,
or module names — the only tree occurrences are in
`docs/BUNDLE_DESIGN.md` itself. No `--question-*` flag exists.
The sole near-neighbour is `--multiturn` (`cli.py:565`, module
`multiturn.py`, USAGE §8.44): a different construct (one task
sharded across turns vs K independent questions in one turn),
a disclosure obligation for any future build's USAGE text, not
a collision. The rejected name's collision is confirmed
independently: `pack.py:1`, format id `mpl-bundle/1`
(`pack.py:49`), `verify --bundle` (`verify.py:670-673`, flag
at `:750`), USAGE §13.1 (`docs/USAGE.md:3320`).

## 3. The amended instrument, recorded as a NEW candidate (not a build order)

The fresh researcher's restated instrument
(`~/workspace/agents_out/multiquestion/research.md` §1) —
bundle §3 plus the operand floor a,b ∈ [3,99] (sums ∈ [6,198],
screens S1–S5 fail-closed), the frozen K=5 rotation, the
total residual-class grammar, mandatory per-family printing,
and the corrected OC grid with clustered and at-N rows — is
the base text of the next candidate, together with the frozen
requirements this round adds:

1. **Commitment binding**: each cell record carries
   `density.commitment_digest` of its word-hash list; the raw
   `answer_digest` is dropped or bound; render-id digests and
   lengths are carried so `id_length_delta` is recomputable
   (negation §3.4 pattern).
2. **Cap discipline stated**: pin the probe's generation cap
   or inherit the run's, in the protocol text, with the
   ≤49-byte correct-reply bound beside it; liveness mirrors
   density `_live` fully (`detail == "" ∧ boundary_found` per
   cell).
3. **Rotation**: a pattern set richer than the two alternating
   strings, with marginal balance preserved and verified by
   counting — or the parity-collinearity disclosed in the
   section text as a named limitation.
4. **Grammar text**: the scan convention frozen (greedy
   sequential first-occurrence), class names that describe
   their events, the `-7`-class leniencies disclosed.

And the three open decisions the next candidate must settle in
its own text **before** its fresh review — they are the
substance of F1–F3, and a review that finds them still open
returns NO-BUILD on sight:

- **Rule purity vs claim re-scope (F1).** Pick one and freeze
  it: either the firing condition counts `wrong_value`
  discordance only, with parse-failure discordance as its own
  non-firing readout (any decision label for it needs its own
  derived rule), or the headline is re-scoped in words to
  usable-answer loss under the frozen reply convention, with
  parser-in-the-loop operating characteristics (per-line slip
  and formatting-collapse nulls) printed as the headline OC —
  not verdict-level nulls alone.
- **Liveness (F2).** A headline population that is not
  conditioned on the treated arm's completion alone, or a
  frozen argument — with executed magnitudes — for why the
  conditioning cannot manufacture a null for the candidates it
  most affects; the truncated/parse-failed discipline made
  consistent under one stated principle; at-N operating
  characteristics printed at the candidate's live N.
- **Family aggregation (F3).** Per-family D/F/P printed beside
  the pooled decision with a frozen claim rule for
  sign-disagreement between families, or per-family decisions
  with a re-derived rule and family-wise accounting over the
  doubled decision count.

This candidate is **not queued ahead of any existing item**
and is not a build order. It enters only through a fresh blind
design review of its own.

## 4. Coordinator re-execution (independent, this session)

All snippets read-only; no suite was run.

- **Parser + hole.** Implemented the §3.3 grammar from the
  tree's real `density.word_hashes` / `digest_text`: baseline
  correct replies parse `ok` at K=3, K=5, and split; correct
  bundled replies with sums 3 (slot 1), 2 (slot 2), 1 (slot 3),
  4 (K=5 slot 4) all break to `extra_words` with no slot
  verdicts, while the same answers pass split — except sum=1,
  which breaks the split grammar too (answer = m_1). The
  arm-asymmetry and its sign are as the lanes report. Sum=6 is
  safe in both arms. The 25-case behaviour battery (swaps,
  omissions, duplicates, out-of-order, preamble, empty,
  markers-only, two-word answer, self-correction, glued lines,
  trailing prose) matches the lanes cell-for-cell under the
  residual-class closure — which the coordinator's
  implementation had to supply, confirming F4's totality gap
  from the implementer's side.
- **Rotation.** Counted from §3.1's patterns: exactly 12F/12A
  per slot at K=3 and K=5 under the alternating completion;
  panel totals 36/36 and 60/60.
- **Cap.** A generous K=5 composite (10-char answers) is 69
  chars / 10 words; the researcher's frozen-format worst case
  (49 bytes) is the tighter, correct bound. Compliant replies
  cannot approach the 64-token cap.
- **Operating characteristics.** Independent Monte Carlo
  (20,000 trials/cell, own code, the §3.4 rule exactly):
  random fire 0.0157 (K=3) / 0.0140 (K=5); competent-null
  0.0110 / 0.0121; independent drop-0.25 power 0.9772 /
  0.9991 — the printed table's independent rows confirmed.
- **Slip null (F1).** Independent implementation: K=5 fire
  0.024 / 0.061 / 0.189 at per-line slip s = 0.02 / 0.05 /
  0.10 (zero knowledge difference); K=3: 0.016 / 0.028 /
  0.071. Whole-case-collapse variant: 0.054–0.181 at collapse
  rates 0.10–0.20. A's G3/G2 magnitudes confirmed in kind.
- **Naming.** Grep over package + docs: `multi-question` /
  `multiquestion` occur only in `docs/BUNDLE_DESIGN.md`; the
  only `multi*` flag in `cli.py` is `--multiturn` (:565).

## 5. Errors and imprecisions found in the briefs (none load-bearing)

- Researcher: the §2(d) worst-case witness repeats `umbrella`
  three times, which its own without-replacement draw + S1
  distinctness forbids in a real case (V §6.1); the 49-byte
  bound's conclusion survives (every fact word ≤ 8 chars). The
  185/4,860 figure's prose omits its K=3 / single-arithmetic-
  slot scope; the rate is correct for that simulation. The
  on-disk MC code's clustered worlds used definitional fudges
  (V §6.1); the printed §3.2 definitions are clean, analytic-
  matched, and bracketed by the verifier's independent rows.
- Verifier: its competent-null K=5 rate differs from the
  rank-5 table by 0.0034 (~4σ at the stated trial counts) —
  recorded by the verifier itself; both values are ≪ 0.05 and
  no conclusion turns on it. Its fabrication sweep ran while
  the fresh briefs were still drafts; the pending sections are
  named unswept in its brief. The coordinator read the final
  briefs in full and reproduced the load-bearing late numbers
  itself (§4), so no load-bearing claim in the final briefs
  rests on the sweep alone.
- Red-team B's 69% vs the verifier's 79% perfect-model
  panel-fire rate: a generator-detail difference (slot family
  counts / distinctness handling); the verifier rules B's claim
  if anything understated. The coordinator's own executions
  confirm the mechanism and the magnitude class.
- Clustered-drop power is reported by two models (marginal-
  matched: 0.4165/0.3401; collapse-rate-0.25: 0.5803/0.5860).
  Both are printed with their models named wherever this doc
  cites them; the falsity of the candidate's caveat is
  unanimous and model-independent.
- No other errors: the verifier's mechanical sweep resolved
  every remaining symbol, line, count, and citation in the
  candidate spec and the landed brief material (catalog = 29
  families executed; the 5,760 / 1,152 / 12 MiB counts
  re-derived exactly).

## 6. Verdict

**NO-BUILD** — permanent closure of this candidate. The
candidate as written is unanimously NO-BUILD: its screen does
not close the marker set its own grammar creates (F4: a perfect
model is labelled a bundling drop in ~7 of 10 K=5 panels under
the natural operand reading), its parser classes are not total
in the frozen text, its generator is described rather than
frozen, and its operating-characteristics evidence carries a
false clustered-drop caveat (F5). The researcher's restated
instrument cures F4 and F5 — with closures this review
verified and adopts into the record (§3) — but it does not
touch F1 (a falsification label that formatting-only worlds
fire at 0.19–0.64, calibrated against nulls that omit the
instrument's own parser), F2 (a headline population selected
by the treated arm's completion, under a conditioning
principle the candidate itself forbids elsewhere), or F3
(opposite family effects cancelling inside the decision's own
aggregate). Each cure changes the rule, the headline, or the
liveness discipline — load-bearing elements a fresh review
records for a new candidate; it does not mint them as a build
order. The amended instrument in §3 is that new candidate. It
is not a build order, and no build cycle may act on it without
a fresh blind design review that finds its three open
decisions already settled in its text.
