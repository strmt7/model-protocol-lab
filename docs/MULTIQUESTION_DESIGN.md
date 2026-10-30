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

---

## Appendix: second fresh-review adjudication (coordinator #70)

Appended in cycle 223, as ordered when the line was parked: the
second restated candidate was reviewed fresh and adjudicated
NO-BUILD as written; the settlement obligation below (absorb
bundled incompletion into the guarded perimeter, or condition
the claim on it) is the standing gate for any future
restatement.

## Fresh-review adjudication — SECOND restated `--multi-question` candidate (coordinator #70, 2026-10-29)

**Verdict: NO-BUILD as written.** The restatement's two design
repairs are real and verified to a standard no previous round in
this line reached: the completion guard is *sound* — its lemma
survived five independent executions including exhaustive
completion enumeration — and the Q-marker grammar closes the
marker-valued-wrong-answer channel by construction. The
candidate fails at the boundary its own §5 draws around the
guard. The guard ranges over **hidden** slots (cells that
completed but did not parse). A generation that produces no
boundary-delimited span at all is an **incomplete** outcome
whose slots are **absent** — outside every completion, by §5's
definition ("not attrition"). Through that gap, both directions
the guard was built to close reopen, executed below: bundled
incompletion concentrated on would-be-pw cases manufactures the
drop label at 0.54/0.72 (K=3) and 0.92/0.90 (K=5) in a
zero-value-effect world, and bundled incompletion concentrated
on would-be-fw cases acquits a true licensed-size drop
(`below_threshold` 0.026 → 0.144 at K=5). The §7 claim sentence
conditions only on *split-cell* incompletion; §5's two
direction sentences about bundled incompletion ("converts the
blindness into withholding"; "its direction is withholding")
are false as written. Every cure — naming the bundled-side
assumption on the claim, conditioning the acquittal licence,
or bringing absent slots inside the perimeter — changes the
headline claim or the rule, so BUILD-as-amended is not
available under the amendment test (rule, population,
headline scorer, and the headline claim untouched). The
generator freeze is also still incomplete in two executed
places (§1's arithmetic redraw; the 67/72 evidence computed on
a string the text does not freeze), which standing alone would
have been amendment-class by the record-notation precedent —
recorded as obligations. **Parked, not closed**: the mechanism
core is verified and the remaining defect has a defined
settlement shape; the caution at the end of this document
governs a third restatement.

Candidate of record:
`~/workspace/agents_out/multiquestion_restated2/candidate.md`.
Round: researcher (BUILD-as-amended), red team A (guard sound
as a lemma, broken as a decision rule), red team B (one
population-class freeze defect, rest amendment-class),
verifier (reproduction strong). Briefs in this directory.
All load-bearing claims adjudicated by the coordinator's own
execution (`coord/` in this directory: `coord_engine.py`,
`coord_lemma.py`, `coord_lemma2.py`, plus re-runs and grid
checks logged at the end).

### The decisive defect — the incompletion perimeter

§5's frozen principle sorts every generation into *outcome*
(span produced, however malformed) or *incomplete* (no
boundary-delimited span). Parse failure is an outcome: its
slots are hidden and §6's guard ranges over them. Bundled
incompletion is not: the case stays in the §5 population, is
counted in `n_bundled_incomplete`, and "contributes no slot
outcomes and no visibility counts — its slots are **absent**,
not hidden". §5 then walls the channel off from the attrition
analysis by definition: "not attrition (the case is not
removed)". The wall is definitional, not arithmetical. A
bundled-incomplete case contributes a (0,0) row to every
quantity the decision reads (D, F, P, npos, nneg); deleting
the row instead changes none of them. Coordinator-verified
identity: on 3,000 random panels, zeroing one half's rows and
removing that half's rows give identical decisions, 3,000/3,000.
Absent ≡ removed, for the decision. The split side of this
exact arithmetic is the channel obligation 4 forced onto the
record — named as §5's load-bearing assumption and priced in
§6. The bundled side is the same arithmetic with no name and
no price.

**Manufacture (coordinator's execution, `coord_engine.py`
row `atk_latent10`; parser-in-the-loop, own implementation of
§§2/3/5/6 from the text).** The candidate's own mirror
population — half the cases pw-prone (split pass 0.55,
bundled 0.95), half fw-prone (the mirror); delivered marginal
rates identical across arms (0.75/0.75), zero value effect by
construction — with one change: the pw-prone half's bundled
cells never complete (their slots are absent, all 24 cases
"live", mean bundled-incomplete 12.00). Guarded fire:
**0.5405 (F) / 0.7198 (A) at K=3; 0.9230 / 0.9047 at K=5**
(10,000 trials each). Controls, same engine: the identical
mirror-trait population with incompletion rate **0.0** fires
at the null (**0.0018 / 0.0124** at K=3; **0.0018 / 0.0016**
at K=5; 10,000 trials each), and
flat bundled incompletion at rate 0.15 on the homogeneous
competent population fires at the competent null
(0.0077/0.0061 at K=3; 0.0130/0.0152 at K=5) — volume does
nothing, concentration does everything. For scale: the
candidate's printed sensitivity for the *split* side of this
arithmetic, at half the removal (q = 0.5), is 0.3541/0.8779.
The bundled side at full concentration is worse, and it is
nowhere in §6.

**Acquittal (red team A, `absent_acquittal.py`; coordinator
concurs in the mechanism and verified the arithmetic).** In a
spread extraction drop of 0.25 — the licence's own row — with
bundled incompletion at 0.5 concentrated on cases realising
≥1 fw slot: K=5 fire falls 0.9665 → 0.4133 and
**`below_threshold` rises 0.0259 → 0.1442** (8,000 trials);
the displaced mass is not withholding, it is acquittal of a
true drop at 5.6× the baseline rate. The deterministic form:
18 absent cases whose would-be bundled values are all wrong
(the §5 blindness scenario verbatim) plus 6 visible pw cases
give D_MAX = 6, F_MAX = 0, P_MAX = 6 → `below_threshold`,
while the completed panel fires under the candidate's own
base rule (rotation-legal form verified by the coordinator:
F = 36, P = 6, D = 24, one-sided p(18, 6) = 0.0113 ≤ 0.05;
red team A's printed panel uses the author's decide-demo
FFF patterns, disclosed in its brief — the mechanism is
pattern-independent). A's own hard-case control records
honestly that in the candidate's printed hard-case world the
dominant direction *is* withholding; the falsified text is
the categorical claim, and it fails exactly in the scenario
§5 itself names ("fail to complete precisely on the cases
where bundling would have hurt it").

**Why this is not amendment-class.** The claim sentence (§7)
licenses the drop "under §5's named assumption that
*split-cell* incompletion is independent of a case's would-be
discordance sign". The executed channel needs the same
independence for *bundled-cell* incompletion, which no
sentence in the candidate states, prices, or attaches to any
licence; the header's "no formatting event can fire or
acquit" and §5/§6's direction sentences assert the contrary.
The cures are: (a) a unified incompletion assumption named
on the claim sentence with worst-case prices for both arms —
a change to the headline claim's conditions, the class the
previous round already ruled restatement material (its
defect 1 and obligation 4); (b) conditioning
`below_threshold` on bundled-incompletion exposure — the
same class for the acquittal licence; (c) bringing absent
slots inside a completion perimeter or out of the population
— a rule/population change, and §5's own text argues (c)'s
premise away ("no span exists whose completion the guard
could range over"). A reviewer who believes (a) is a mere
wording fix should note what the wording would have to say:
that the label is certified only if a second, untestable
independence assumption holds — which is precisely the
settlement the previous round demanded be made *in a
restated candidate's own text*, not appended by a verdict.

### Obligation by obligation

**1. Error-visibility channel — SETTLED for parse failure;
NOT SETTLED for generation incompletion (decisive, above).**
For the channel the obligation named — value-correlated
*parse* failure — the completion guard is the genuine
settlement the first restatement lacked, and it is verified
beyond this tree's usual standard:

- *Lemma.* The author's monotone-path walk (60,000 panels,
  13,131,607 steps) re-ran byte-identically (verifier;
  coordinator's re-run of the sibling scripts identical).
  Red team A: exhaustive completion enumeration over two
  lanes — 3,277 panels (145,326,876 completions) and 1,002
  engineered panels (103 MIN-firing; 69,366,975 completions)
  — plus 899,531 upward-walk steps with 0 fire→nonfire
  transitions: 0 violations of claims A/B. (Counts as
  corrected by red team A after its first delivery: the
  figures in its original brief came from background-delivery
  texts that proved to be phantom, and were replaced by
  re-runs read from on-disk logs — the same failure family
  as the corrections to this document's own log, below.)
  Red team B: 268 panels exhaustively — 0 violations, and the
  `insufficient` basis dichotomy exact. Red team A also
  verified the α..ε attainability under cell-granular hiding,
  including joint cross-family corners the text does not
  require. Coordinator: exhaustive (npos, nneg) grid —
  the one-sided tail is non-increasing under all three step
  kinds on the full 0..60 × 0..60 grid (0 violations), and
  no (npos ≤ 5, D ≥ 6) panel fires on a grid to D = 200 (the
  firing ⇒ npos ≥ 6 lemma the D-gate argument needs); plus
  endpoint enumeration (1,405 panels, 24 MIN-firing,
  8,804,693 completions): claims A and B held on every
  completion, per-case df extrema equal to the MIN/MAX
  corner rows in every panel.
- *Behaviour.* The author's OC rows were diffed line-by-line
  against the shipped outputs (red team B: every figure
  matched) and re-executed on shipped seeds (verifier, venv
  interpreter: part-1 rows within |Δ| ≤ 0.0020, the
  manufacture rows exact). Red team A's independent
  guard-vs-truth engine: guarded FIRE with true base ≠ fire
  = 0 and guarded BELOW with true base = fire = 0 in every
  cell (manufacture, symmetric, masking at dress 0.5/1.0).
  Coordinator's independent engine reproduces the competent
  nulls (0.0093/0.0101 at K=3, 0.0193/0.0196 at K=5 vs
  printed 0.0112/0.0092, 0.0184/0.0207) and the masking
  endpoint (fire 0.0000 at r = 1.0, K=3, 5,000 trials; a
  below-verdict discrepancy in the coordinator's own
  masking construction is recorded in the execution log
  and is not counted against the candidate).
  Manufacture by *parse* dressing is dead (guarded fire flat
  in r at the null); masking by parse dressing never acquits
  (withholding 1.0000 at r = 1.0). §6's replacement
  properties (i)–(iii) are true — of hidden slots.

**2. Marker-valued wrong answers — SETTLED.** Q-markers are
disjoint from both value spaces as strings, hence (the
tree's standing digest assumption) as hashes; executed
disjointness over pools, topics, and decimals 0..9999 at
K ∈ {1, 3, 5}; the wrong-value battery the previous battery
omitted is frozen and passes under both conventions
(coordinator's re-run of `grammar_check2.py` byte-identical);
the §6 marker sweep prints the digit grammar's collapse
(1.0000 → 0.1059 → 0.0000 at K=5) beside Q-marker invariance,
diffed clean by red team B and spot-reproduced by the
verifier. The label-word-as-content residue is correctly
classified in the text as a formatting event inside the
guard's perimeter. The S3 floor's restated status (population
definition, not grammar safety) is honest and its retention
reasoning is the right shape.

**3. Claim inside the printed power — SETTLED at the
licence's headline rows; two scoping defects sustained
(amendment-class, obligations below).** The 25-point licence
figures are correct roundings of *both* families' executed
spread rows (F 0.7493/0.9681; A 0.7474/0.9677); the header
utility names the spread rows and disclaims clustered
25-point drops; slip×drop and collapse×drop dilution rows
are in the OC and behave as printed (dilution converts to
withholding, `below_threshold` 0.0048/0.0001). But: §6's
50-point rows and all clustered rows are **extraction-only
executions** printed without a family qualifier, and §7's
licence quotes "(0.9988/1.0000 for 50 points)" and the
clustered "detected at 0.45/0.48" unscoped (red team B,
source audit of the grid — coordinator concurs from the
same read: the grid contains spread-A 0.25 and no A 0.50 or
clustered-A world at any size). And the §7 licence, which
"licenses exactly:", quotes the full-panel powers with **no
live-N condition**, while §6's own at-N rows put the
25-point spread power at 0.3155/0.7592 at live N = 16 —
the acquittal's evidential weight is overstated ≈2.2× at
K=3 for a reduced-N panel (red team A, executed at N = 16:
0.3389/0.7669, matching §6's rows).

**4. §5 conditioning and disclosure — split side SETTLED as
specified; its printed price understates the stated
assumption's worst case; bundled side NOT SETTLED
(decisive).** The manufacture sentence is conditioned on the
named assumption, the assumption is labelled load-bearing
and record-untestable in §5, §6, and the claim sentence,
and the claim is scoped to "the live cases of this generated
panel" — obligation 4's letter, met. But §6 presents one
construction (latent halves, per-cell q) as "its executed
sensitivity" / "the sensitivity price", and that
construction is not the assumption's literal violation.
Conditioning removal on the **realised** discordance sign —
exactly what the stated assumption forbids dependence on —
dominates the printed rows at every matched budget (red team
A; deterministic floor verified by the coordinator: 12 fw +
12 pw cases, removing the pw cases with 12 single-cell
misses, turns base `below` (F = P = 12) into a guaranteed
fire at D = 12, p = 0.000244). Calibration note: red team
A's replication figures for the candidate's own construction
(0.2061/0.6174 at q = 0.3) equal the F/A *means* of the
printed rows (0.1356/0.2636; 0.6470/0.5950) to Monte Carlo
noise — their comparison is internally consistent but their
baseline is the family mean, not a family row; the
qualitative break stands on the deterministic floor and the
matched-budget argument, not on their headline margin. The
placement breakdown that was to distinguish *where* the
blindness sits is half vacuous (below, defect 5). The
invisibility disclosure's inc = 0.0 baseline rows themselves
(0.0406/0.1232; 0.0016/0.0070 at inc 0.5) are executed and
check out.

**5. §4/F4 completions — NOT complete (two executed gaps;
amendment-class by precedent, obligations below).**
- *Arithmetic redraw (red team B, `gen_freeze.py`).* §1: a
  case-level screen "rejects the later slot's conflicting
  item", redrawn "at its kind counter + 1". For an arithmetic
  slot the conflicting item is the sum, drawn from two kinds
  (`opa`, `opb`); the text never says which operand redraws.
  Executed: 12 raw S1 collisions in 192 cases (seeds 0–3,
  both panels); redraw-`opa` vs redraw-`opb` populations
  differ in 4 of 192 cases. §4's "regenerates the identical
  population from the seed alone" is false as written, in
  obligation 5's original defect class.
- *The 67/72 evidence (red team B; coordinator-verified).*
  §1 prints that two conforming mappings "disagree on the
  drawn topic for 67 of 72 case-slots", reproduced in
  `grammar_check2.py`. The script builds its material
  *without* the `kind` field (`…|{index}|{slot}|0`). On §1's
  own frozen material (kind = `topic`) the coordinator
  computes **72/72**, not 67/72. The qualitative point
  survives strengthened; the printed figure does not
  describe the frozen generator, and the shipped script
  tests a different string than the text freezes.
- Minor companions (red team B): S2 is vacuous by
  construction (alphabetic values vs decimal sums — a
  conflict is impossible by type) and the text presents it
  as a live screen without disclosing this; no give-up
  condition is frozen for when redraws end and
  `multiquestion_screen_failed` fires, while §1's screens
  header and draws paragraph pull in opposite directions.
  The researcher adds: the material's `{index}` field is
  nowhere defined in §1 (the author's own scripts read it as
  case index within panel, as does the `_derive` citation; a
  one-line pin removes the ambiguity). The remaining F4
  completions are done and verified: commitment-verification
  rejection row frozen and executed both directions
  (`decide2.py`, coordinator's re-run identical);
  `render_id_digest` dropped; swap scope frozen to the
  bundled cell; `_live` cited with all three conjuncts
  (coordinator-verified at `density.py:788` in the current
  source, the third vacuous here as argued); §7 aggregation
  sentence corrected; isolation rows regenerated under the
  independent-arms null (non-target fire 0.009–0.021, its
  own null rate); pattern recurrence corrected to four;
  magnitude-blindness disclosure printed with its executed
  pair.

### The one-sided deviation — adjudicated, accepted

**Counterexample: verified.** Coordinator's own panel
through the tree's `density.sign_test_p`: discordant cases
at (npos, nneg) = (6, 17) with F = 18 > P = 17 carried by
slot magnitudes fires two-sided at **p = 0.0347**; one
completion step moving a negative case to zero (F = 18,
P = 16, D = 22 — every other firing component intact) raises
it to **p = 0.0525** and extinguishes the fire. The doubled
smaller tail is not completion-monotone; the two-sided form
is unsound under the guard, exactly as §6 argues. (The
verifier and red team A reproduce both p-values; red team
B confirms them against the tree function. Provenance
note, verifier: §6 attributes the panel to
`guard_lemma.py`'s "first run", which is not in the shipped
directory — the script carries the p-values in a docstring
only, so the author's original panel is unconfirmable; a
panel with these totals is arithmetically forced, and the
lanes' constructed witnesses agree.)

**Upward closure of the one-sided rule: verified
independently** (coordinator's grid exhaustion and endpoint
enumeration, §1 above; the author's walk, red team A's
exhaustive and walk lanes, and red team B's enumeration
concur — five executions, three methods).

**The printed price: accepted for this tree.** One-sided
family-wise accounting (coordinator's re-run of
`familywise2.py` byte-identical; arithmetic independently
recomputed): random E = 2.95, P(≥1) = 0.95, per-candidate
0.1174; competent E = 1.43, P = 0.76, per-candidate
**0.0582**; paired ≈ 0. The previous round recorded
per-candidate 0.031/0.045/0.053 (two-sided) as inside shipped
tolerances, citing field-order's ≈1.2 expected false labels
per run. The one-sided competent figure (0.0582) sits
marginally above that range's top; the random-null figure
(0.1174) is ~2.2× it. Against the excess: the price is
printed in §6 beside the deviation argument and again in
the family-wise paragraph, which calls the rates "visibly
worse" and frames the label as comparative evidence at a
stated base rate, never a verdict on a model; the
decision-relevant nulls for candidate comparison are the
competent and paired ones (E = 1.43 is inside the
field-order envelope; paired is ≈ 0 at the executed
resolution); and the alternative is not a cheaper test but
no guard — the two-sided price is unavailable at any
printing. A falsifier whose false-label base rate is
printed where its licence lives meets this tree's standard.
**Not a defect; do not re-litigate in a third restatement.**

### What verified — the settlements that hold

- **Guard lemma and extremal completions** (five executions,
  above), including per-case corner attainability and the
  joint cross-family corners.
- **Guard behaviour vs truth** (red team A): 0 false fires,
  0 false acquittals against completed truth across
  manufacture/symmetric/masking cells, parse channels only.
- **Base-rule counting purity** (carried from the previous
  round, re-verified in this round's engines): the rule
  counts `wrong_value` discordance only.
- **Grammar**: total, disjoint, value-blind by construction;
  battery (old shapes + the wrong-value shapes) cell-for-cell;
  120,000-list totality; renderer/parser agreement through
  the real `density.word_hashes` (coordinator's re-run
  byte-identical).
- **Rotation**: every slot 12F/12A at both K; totals 36/60;
  four recurring K=5 patterns (AAAFF, AAFAF, FFAFA, FFFAA).
- **Power honesty at the executed rows**: the full §6 table
  diffs clean against the shipped outputs (red team B);
  sampled re-executions concur (verifier; red team A;
  coordinator's nulls). The drop-parameter convention
  (conditional flip, rev 0.10) is the line's established
  convention and its direction is conservative for the
  licence (a true marginal 25-point drop is detected at
  *higher* power than printed).
- **Decidability core**: `decide2.py` — every §4 row
  recomputed from the carried fields; commitment tampering
  rejected in both directions; deterministic manufacture /
  masking / no-over-withholding / magnitude panels behave
  exactly as §6/§7 describe (coordinator's re-run identical).
- **Counts and budget**: 240 generations per candidate,
  5,760 legal maximum, 1,152 ground records; candidate cap
  24 at `experiments.py:4013–4014` (verifier; line moved +1
  since the previous citation, value unchanged); attrition
  closed form 24(1−q)^K reproduces 20.6/18.6/17.5/14.2;
  worst-case correct K=5 reply 54 bytes / 10 words under the
  64-token cap (longest pool word `umbrella`, 8 chars).
- **Density citations**: exact at current source (one
  range start one line early — `_derive` cited 407–414,
  def at :408; verifier/researcher note, immaterial) —
  `_derive` `density.py:407–414`, `_draw_word` :417,
  `normalize_words` :701, `word_hashes` :720,
  `commitment_digest` :726, `sign_test_p` :770, `_live` :788
  (coordinator-verified).
- **Naming**: `--multi-question` is free in the *current*
  parser — the tree has since shipped `--record-notation`;
  the only "multi" string is `--multiturn`;
  `allow_abbrev=False` on the root and all subparsers
  (coordinator-instantiated). §10's printed census (80)
  equals the distinct long-option count on the current tree
  (coordinator: 80 distinct; red team B: 114 occurrences,
  81 `cli.py` source literals — three methods, three
  numbers; the candidate's method is unstated and its
  narrative omits the record-notation ship: a reporting
  note, defect 9 below, not a naming failure).
- **Uniqueness** (researcher, primary sources opened): the
  residue stands — no shipped probe varies the number of
  independent questions in one turn against asking them
  separately; Cheng (arXiv:2301.08721) and HELM
  (arXiv:2211.09110) fencing is accurate at the operative
  scope (HELM does define multi-question instances inside
  scenarios; it nowhere runs the packaging-vs-separation
  contrast the candidate fences). The claim is the
  combination; no "first" claimed.

### Sustained non-fatal defects (obligations for a restatement)

1. **§7 licence unconditioned on live N** (red team A):
   quote the at-N powers or condition the licence on the
   printed live count.
2. **50-point and clustered rows unscoped** (red team B):
   scope them to extraction in §6/§7, or execute the
   arithmetic rows and print both families.
3. **Attrition price is not the worst case** (red team A):
   print the realised-sign sensitivity (or a bound) beside
   the latent construction, for the split side — and, per
   the decisive defect, for the bundled side.
4. **False provenance in §5/§6** (red team B;
   coordinator-verified): the first restatement's header
   printed **0.0018–0.0063** — the invisibility world at 50%
   bundled incompletion (`grep -c "0.035"` over its text:
   0). The 0.035/0.111 figures were the previous
   adjudication's executed inc = 0.0 baseline, and the
   recorded direction under incompletion was a fall. §5's
   retelling ("printed a hard-case world firing at
   0.035/0.111 with a further rise under bundled
   incompletion") inverts the record it cites. The new
   candidate's own rows are unaffected; its settlement
   narrative is false and must be corrected.
5. **Placement breakdown half vacuous** (red team B;
   coordinator-verified from the rotation definition): at
   both K, 0 of 24 patterns are "balanced" (K is odd) and
   24 of 24 contain at least one slot of each family, so
   the per-family counts are identically
   `n_bundled_incomplete` on every panel — they distinguish
   nothing. Only the majority leg carries information.
   Specify a breakdown that does the job (per-pattern, or
   majority-only honestly labelled).
6. **`id_length_delta` aggregation unfrozen** (red team B):
   §4's row admits per-j deltas; `decide2.py` implements
   per-case bundled − Σ_j split; §7 readout 4 prints a
   per-candidate quantity whose aggregation (sum? list?
   cf. `historyframe.py`'s by-case list under the same
   name) is nowhere stated. Pin the function.
7. **Readout 3's denominator unfrozen** (red team B): a
   cell with `boundary_found` false has no delimited span,
   so its parser class is undefined by §4's table, and the
   text never says whether readout 3 counts incomplete
   cells or non-live cases' cells. Freeze the population
   of the readout.
8. **D_MIN/D_MAX gloss** (red team B; coordinator-verified
   arithmetically): the names invite an extremal reading
   the quantities do not bear — 6 cases of (fw=0, pw=0,
   α=1) give D at the MIN completion = 6 and at the MAX
   completion = 0. They are D evaluated at the two
   completions; rename or gloss precisely. (The verdict
   logic depending on D_MAX is sound.)
9. **§10 census method unstated; §8 header overclaims**
   (red team B; researcher; coordinator): state the census
   method and refresh the narrative for shipped flags;
   `sequence_broken`/`segment_malformed` occur nowhere in
   the tree — they are new stage-local coinages, so §8's
   "(frozen; tree words only; unchanged)" is false as a
   provenance claim (no harmful collision; the decision and
   detail words are tree words).

### Fabrication sweep

**Candidate text: two hits, both in the candidate's own
evidence narrative — the previous round's pattern.** (i)
The 67/72 mapping-divergence figure was computed on a
material string the frozen text does not use (frozen
string: 72/72; coordinator-executed both). (ii) The §5/§6
provenance retelling of the first restatement's hard-case
numbers is false and direction-inverted
(coordinator-verified by direct read). Every other printed
number the coordinator or a lane re-derived matched:
the full §6 OC table (line-by-line diff, red team B;
sampled re-execution, verifier + red team A + coordinator),
the family-wise conversions (exact), the grammar battery
and counts (byte-identical re-runs), the counterexample
p-values (three lanes), the attrition closed form, the
54-byte bound. No invented symbol or citation in the
candidate: density citations spot-verified at source by
three lanes; Cheng/HELM resolved by the researcher against
primary sources.

**Specialists: two briefs carry errors, recorded.** The
researcher's brief asserts `docs/USAGE.md` does not exist
(it does — coordinator-verified by direct listing; the
probe list lives there and in `docs/reference/cli.md`) and
raises a multiturn-budget staleness claim (5,184 vs a
"computed 10,272") that the coordinator could not
substantiate against the source (`multiturn.py:117` =
5184, agreeing with the docs) — immaterial to the candidate,
which cites neither, but the brief's two FAILS verdicts
resting on the USAGE premise collapse. Red team A's
attrition-replication figures are family means mislabelled
as a replication of the printed rows (calibration note in
obligation 4 above); its two breaks stand on their own
executions. Red team A also corrected its own lemma-lane
counts after delivery — its first figures came from
phantom background-delivery texts; the re-run totals read
from on-disk logs (≈214M completions, 0 violations) are
the ones cited in obligation 1, and its corrected
one-sided control (0.9947) agrees with the coordinator's
own computation. The verifier's first `oc_guard.py` attempt ran
under the system Python 3.14 (a dataclass-evaluation crash)
and was re-executed under the tree venv, self-reported in
its brief; its venv numbers are the ones relied on. No
fabricated citation, symbol, or tree fact with load-bearing
weight in any brief.

### Restated candidate — must settle in its own text

1. **The incompletion channel, both arms, both directions.**
   Either (a) one named assumption covering *any* generation
   incompletion — split-cell or bundled-cell — whose
   occurrence correlates with a case's would-be discordance
   sign, attached to the claim sentence and to the
   `below_threshold` licence, with executed worst-case
   (realised-sign) sensitivity prices for both arms and both
   directions (manufacture and acquittal); or (b) a
   construction that brings bundled-incomplete cases inside
   a decision perimeter (guard-ranged, population-excluded,
   or verdict-capped by their count) with its operating
   characteristics re-derived. The §5/§6 direction sentences
   must be replaced by the executed truth in either path.
2. **The licence conditions**: `below_threshold` conditioned
   on live N and on bundled-incompletion exposure; the
   50-point and clustered rows scoped to the families
   actually executed (or executed for both).
3. **The freeze, completed and demonstrated**: the
   arithmetic redraw operand pinned; a redraw give-up
   condition frozen; `{index}` defined; S2's vacuity
   disclosed; the mapping-divergence evidence recomputed on
   the frozen material string (72/72) with the script fixed.
   A restatement should demonstrate the freeze the way the
   previous round's B-3 test did — two independent
   implementations of §1 generating identical populations —
   not assert it.
4. **The reporting repairs**: placement breakdown that
   distinguishes (defect 5); `id_length_delta` aggregation
   pinned (6); readout-3 denominator frozen (7); D_MIN/
   D_MAX renamed or glossed (8); §5/§6 provenance corrected
   (4); §10 census method stated (9); §8 header's provenance
   claim corrected (9).

**Parked, not closed** — behind the DeepSource remediation
in queue order; not a build order. Caution for the next
round, stated once: this is the line's third NO-BUILD, and
each round's decisive defect was a channel adjacent to the
previous settlement (counting purity → eligibility → the
guard's perimeter). The verified core (guard, one-sided
test, Q-markers) is not in question. But a third restatement
that settles obligation 1 by assumption alone — a second
untestable independence clause carrying the headline claim —
should expect the review to weigh whether the instrument's
claim is now conditioned on more of the world than it
measures; the by-construction path (1b), or a candid
re-scoping of the utility to what survives both channels,
is the stronger submission.

### Standing questions

- **Monday-morning utility (one sentence, as scoped):** a
  form-filling front-end operator deciding, per template
  candidate and per question family, whether to bundle three
  or five short questions into one user message — as a
  per-family large-drop falsifier whose *parse*-channel
  soundness is now proven, and whose licence still
  overclaims wherever generation incompletion, rather than
  parse failure, is the channel that correlates with value.
  The user and the decision are real.
- **Uniqueness:** verified (above). The residue is real:
  the completion guard itself — a verdict that must hold
  under the extremal completions of its hidden evidence —
  is a genuinely new mechanism in this tree, and it works.
- **Duplication:** none; the stage reuses density's frozen
  topics/words/primitives by citation, the tree's own reuse
  discipline.

### Build cost (if a future review returns BUILD)

Generations 5,760 legal maximum (frozen, correct).
Implementation estimated ~2,500–3,500 lines including tests
by sibling-probe analogy — an estimate for sequencing only,
labelled UNEXECUTED.

### Coordinator's execution log (own work, this directory)

- Parser census (venv, `cli.build_parser()` instantiated):
  80 distinct long option strings incl. `--help`;
  `--multi-question` absent; `--record-notation`,
  `--history-frame`, `--prune-sweep` present; only
  "multi" string `--multiturn`; `allow_abbrev=False` on
  root and all 11 subparsers.
- Counterexample: own panel (6×(3,0) + 17×(0,1)) through
  tree `density.sign_test_p` → fires at 0.0347; after one
  neg→zero step → 0.0525, no fire; F > P and D ≥ 6
  throughout. One-sided p1(6,17) = 0.9947.
- Lemma grids: p1 non-increasing under all three step kinds,
  full 0..60 × 0..60 grid, 0 violations; no (npos ≤ 5,
  D ≥ 6) firing panel for D ≤ 200, 0 violations.
- Endpoint enumeration (`coord_lemma2.py`): 1,405 panels
  (24 MIN-firing), 8,804,693 completions — claims A/B held
  on every completion; per-case df extrema = MIN/MAX
  corners in every panel. (v1's generator skipped
  3,999/4,000 panels for >16 hidden slots and tested 1;
  superseded by v2, recorded here rather than hidden.)
- Script re-runs (venv): `decide2.py`, `familywise2.py`,
  `grammar_check2.py` — all exit 0, outputs byte-identical
  to the author's. Family-wise arithmetic recomputed
  independently: E and per-candidate figures exact.
- Independent engine (`coord_engine.py`, written from the
  candidate text; 20,000 trials unless noted): competent
  null 0.0093/0.0101 (K=3), 0.0193/0.0196 (K=5) vs printed
  0.0112/0.0092, 0.0184/0.0207 — agreement; masking
  (drop 0.50, hide_bund r = 1.0) fire 0.0000 at K=3
  (5,000 trials).
- Decisive attack (`coord_engine.py`, 10,000 trials):
  mirror population, bundled incompletion at rate 1.0 on
  the pw-prone half → guarded fire 0.5405/0.7198 (K=3),
  0.9230/0.9047 (K=5), mean live 24.0, mean
  bundled-incomplete 12.00. Trait control (identical
  population, incompletion rate 0.0): 0.0018/0.0124 (K=3),
  0.0018/0.0016 (K=5) — the null at both K. Flat-incompletion control (0.15, homogeneous
  competent population): 0.0077/0.0061 (K=3), 0.0130/0.0152
  (K=5) — the null rates.
- Discrepancy recorded, not counted: the coordinator's own
  masking row (drop 0.50, hide_bund r = 1.0, 5,000 trials)
  produced fire 0.0000 at both K but `below_threshold` at
  0.0002 (F) / 0.0042 (A) at K=3 (0.0000 / 0.0000 at K=5),
  where the author's rows (20,000 trials) print 0.0000 and
  red team A's guard-vs-truth found 0 acquittals in masking
  cells. The two verified engines
  agree with each other and with the proven lemma; the
  deviation localizes to the coordinator's corner/masking
  construction and is recorded here rather than charged to
  the candidate. The masking property relied on in
  obligation 1 is the one those engines carry.
- Identity check: absent (0,0) rows ≡ removed rows for the
  decision, 3,000/3,000 random panels.
- Fabrication checks: mapping divergence 67/72 on the
  author's script string vs **72/72** on §1's frozen string
  (own computation); `grep -c "0.035"` over the first
  restatement = 0, its header prints 0.0018–0.0063 (line 29);
  p1(18, 6) = 0.0113; p1(12, 0) = 0.000244; density
  citations verified at `density.py` :408/:417/:701/:720/
  :726/:770/:788; `arm_context_exceeded` /
  `arm_evaluation_failed` confirmed as tree vocabulary;
  `sequence_broken` / `segment_malformed` confirmed absent
  from the tree.
- Load note: all coordinator batches ran single-threaded
  at nice 10–15 alongside cycle 221's gate (PID 1844);
  no repo test suites were run and the tree was not
  modified. No simulation result used above showed
  contention sensitivity (all decisive rows are large-margin
  or exact).
