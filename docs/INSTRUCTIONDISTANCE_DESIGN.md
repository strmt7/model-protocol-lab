# Instruction-distance probe — design of record

## Status

**BUILT — cycle 218** (BUILD-as-amended, fresh review by
coordinator #59; the queue's first build verdict). This document
is the complete design of record, consolidated in the build
push:

- **Part I** is the instrument as built: the resubmission
  candidate (round-1 researcher's amended instrument, transcribed
  as §3 of the round-1 round) **with the fresh review's binding
  amendments A–J incorporated into its text**, not merely listed.
  Where an amendment corrected a number or a sentence, Part I
  carries the corrected form only.
- **Part II** is the round-1 adjudication (coordinator #43,
  NO-BUILD), reproduced from the surviving working-tree copy.
  That copy truncates mid-F6(c); the truncation is marked where
  it occurs and nothing past it has been reconstructed except
  the one code expression F6(c) was quoting, which is verifiable
  from the tree it cites. The round-1 design document's own
  §§2–7 (the circulated sketch and its transcription) do not
  survive anywhere; the round-1 briefs
  (`~/workspace/agents_out/instructiondistance/` —
  `research.md`, `redteam_a.md`, `redteam_b.md`, `verifier.md`)
  are the surviving record, and this consolidation says so
  rather than fabricating replacements.
- **Part III** is the fresh-review adjudication (coordinator
  #59) verbatim, including the verdict, settlements (a)–(c),
  and amendments A–J in their binding form.

The frozen errors the amendments corrected appear in this
document **only** inside the Part III adjudication narrative
that corrects them (and in Part II's round summary, which
records the round-1 verifier catching one of them).

---

# Part I — The instrument as built (candidate text as amended)

## 1. What exactly varies — the distance definition

### 1.1 A dose is a verified rendered-token delta, per candidate

Transplant of needle's dose settlement, in substance
(`model_protocol_lab/needle.py` docstring and `SIZING_RULE_TEXT`):

- For one case and one candidate, let `adj_ids` be the rendered
  token ids of the adjacent arm (§3) under that candidate's own
  template/backend. A separated or displaced arm at dose `d` is
  a construction whose rendered ids satisfy
  `len(ids) − len(adj_ids) == d` **exactly**, verified on the
  accepted construction's final ids before generation.
- The filler span is built by the sizing discipline of §1.4
  (whole sentences from the cycled neutral corpus stream plus
  one truncated suffix, probed in closeness order, encode
  budget 512 per cell) until the delta is exact. If no span in
  budget hits the delta exactly, the cell is `arm_unmatchable`
  — never approximated, never silently sized in characters.
- Therefore "distance" is **not** a property of the filler
  text, not characters, and not a cross-candidate constant:
  the same filler string tokenizes differently per candidate,
  so each candidate gets a *different filler text* sized to
  the same verified delta. What is frozen across candidates
  is the dose (the delta), the corpus, the stream order, and
  the sizing rule (bound by digest in the protocol object,
  needle pattern).

### 1.2 The measured separation, stated honestly

The dose is the number of *added tokens*: in the separated
arms the added tokens sit **between** the directive and the
block; in the displaced arms the same number of added tokens
sits **after** the block (the displaced arm's between-distance
is the adjacent baseline; its dose is length). The adjacent
render itself has a small nonzero separation (the fact
sentence sits between directive and question); that baseline
is a per-case, per-candidate quantity, reported per candidate
as the per-arm rendered-token counts in the records, never as
a single printed token figure in prose.

### 1.3 Doses

`DOSES = (256, 1024)` rendered tokens.

- Derivation, not borrowing: 256 and 1024 are the two largest
  neutral levels of the distraction factorial
  (`distraction.py` `NEUTRAL_LEVELS = (64, 256, 1024)`), and
  1024 is needle's smaller dose. The *reasons* they are right
  here are local: (i) the largest dose must fit the default
  context cap with the whole prompt — `--max-context` default
  is 2048 (`cli.py`) and under it a 1024-token filler plus the
  adjacent prompt plus template head plus generation budget
  is the largest power-of-four dose that is a real measurement
  rather than a guaranteed `arm_context_exceeded` column
  (executed in the fresh review: worst 1024-arm total 1,395
  tokens against the 2,048 cap across every catalog family);
  (ii) two doses give three same-turn layouts at two doses
  (adjacent, 256, 1024) — three points, printed with
  denominators, from which no shape is claimed (§2.2).
- The sketch's 128/512 pair is rejected: 128 is below every
  shipped neutral dose except 64 and maximises the short-dose
  null risk the sketch itself names; nothing in the tree
  supports it.
- Cells that exceed the run's `--max-context` are
  `arm_context_exceeded`, counted per arm, never truncated
  into existence.

### 1.4 The sizing rule (frozen text; amendment J)

The protocol carries the SHA-256 of the sizing rule text
(`instructiondistance.SIZING_RULE_TEXT`,
"instructiondistance-single-phase-exact-v1"): spans are whole
sentences plus one truncated suffix (`span_text`) over the
cycled neutral corpus stream, every sized cell searching from
the stream head; candidates are probed in `span_candidates`
closeness order against the dose, each probe rendering the
full arm content under the candidate's own template,
accepting the first whose rendered token count minus the
case's adjacent-arm rendered token count equals the dose
exactly, verified on the accepted construction's final ids
(the probe render *is* the arm render, so acceptance and
verification are one encode). One encode per probe; the
per-cell encode budget is 512.

This is needle's sizing discipline with one named omission
(amendment J / A-K5c): needle's rule carries an in-span
**offset-verification phase** (needle.py's `SIZING_RULE_TEXT`),
because needle measures where inside the filler a fact sits.
This instrument has no offset variable — the filler sits
wholly between the directive and the block, or wholly after
it — and the "\n" joiner insulates the span from the block it
precedes or follows, so delta-exactness on the final render
is the only acceptance condition. The transplant is therefore
**not** "verbatim in substance"; it is the needle discipline
minus the phase that has no analogue here (executed in the
fresh review: 12/12 cells exact under the single-phase rule).

## 2. The claim and the ordering

### 2.1 Claim (frozen, one-directional, per candidate, as
re-scoped by amendment F)

> For one candidate, over the cases it completes in all three
> arms of a dose contrast (adjacent, separated, displaced)
> whose adjacent cell is `compliant_correct` (§5.2), let D be
> that population, S the cases in D whose separated cell is
> noncompliant, and T the cases in D whose dose-matched
> displaced cell is noncompliant. The label
> `instruction_distance_drop_observed` fires exactly under
> the rule in §6. No other sentence is a headline.

The falsifiable statement behind it, with its conditioning
and completion scope carried in the sentence (amendment F):
*per-candidate compliance with the frozen directive (§3.2),
conditioned on the cases the candidate completes in the
adjacent, separated, and displaced arms of the dose and
answers compliant-correct when adjacent, is identical at
added distance 0 and at added distance d.* Proved wrong, per
dose, only by the §6 pattern — discordant pairs (compliant
adjacent, noncompliant separated) in excess of both the
derived floor and the dose-matched displaced control.
Reverse discordants (noncompliant adjacent, compliant
separated) are counted and printed beside every decision
(`n_reverse`), never pooled away.

The label is an **excess-of-between-losses label**, never a
distance effect simpliciter (amendment F, estimand
restatements): a pure constraint-to-generation effect yields
S = T → `below_threshold` even at a 24/24 → 0/24 collapse,
and a genuine between-effect exactly masked by the control's
own trailing-filler effect (S = 10, T = 10) also yields
`below_threshold`. Both are the frozen estimand operating as
defined, with S and T printed.

### 2.2 What the claim is NOT — no ladder, no shape, no
cross-turn rung

- **Not monotone decay, not a threshold.** The readout prints
  the three same-turn compliance rates with denominators;
  per-dose labels can invert the raw dose order through
  control noise alone, so no sentence describes a curve, a
  threshold distance, or decay. The literature's "rapidly
  degrades" (LongIns, §10) is a claim about exam-task
  performance at 4K–16K tokens on API models; it is not a
  design input and no shape is pre-registered.
- **The cross-turn arm is excluded, by name.** A scripted turn
  pair bundles turn boundary + assistant-turn presence + role
  alternation + token distance; it cannot sit on the
  token-dose dimension. Multiturn owns cross-turn delivery.
- **The adjacent-after arm is a control, not a rung** (§4):
  distance 0, order reversed. It is reported descriptively
  and never enters a dose contrast.

## 3. Population and frozen texts

### 3.1 Generated cases, stage-local, never the Split

Cases are generated from the run seed by a frozen SHA-256
derivation (needle pattern), live only in the stage, and
never enter the user's search/held-out Split.

Case shape: a fact block B that is byte-identical and
internally adjacent in every arm —

```
The {topic} number is {value}.
What is the {topic} number?
```

- `{value}`: a 3-digit integer (100–999) from the seed
  derivation; pairwise distinct across cases (needle redraw
  discipline). 3 digits keeps the answer a single short digit
  string whose compliant form is unambiguous, and makes a
  closed-book guess a 1-in-900 event before any model prior.
- `{topic}`: drawn from a frozen topic list of 24 single
  words (harbor, summit, quarry, tundra, granite, willow,
  copper, saddle, walnut, velvet, cobalt, marble, cinder,
  falcon, garnet, saffron, indigo, juniper, lagoon, magnet,
  nickel, onyx, pepper, quartz), verified absent from the
  neutral corpus as case-folded substrings and pinned by
  test. (The first candidate list had four collisions —
  meadow, orchard, timber, hazel occur in the corpus — and
  was corrected before freezing, needle's own freeze
  discipline.)
- The fact is *inside* B, after the directive in every
  directive-first arm, so the fact↔question distance is
  **constant across arms by construction**. What moves is
  only directive↔B separation and, in the displaced control,
  B's distance to the generation point (§4).

Case count: **24** (`CASE_COUNT = 24`, placement's count;
amendment A — the candidate's §12 verdict sentence printed a
stale case count, corrected in Part III). The derivation is the
decision rule's, §6: the frozen rule at N=24 reaches power
0.9088 at a 50% conditional drop while at N=20 it reaches
only 0.813, and N=16 (negation's count) cannot support the
rule's D≥10 floor at realistic adjacent pass rates.

Candidates: at most 24 (run_experiment's operative cap,
`experiments.py`: "Supply between 1 and 24 candidates";
amendment A — the candidate cited an audit budget in
determinism.py, which is not the operative cap).

### 3.2 The frozen directive (single draw, scoped)

```
Reply with only the number: no units, no sentence, nothing else.
```

(64 chars — amendment A; the candidate printed a wrong
character count, an error the round-1 verifier had already
flagged.) Byte-
identical in every arm that carries it; only its position
and its distance to B vary. The claim is scoped to this
string. A second phrasing is excluded from v1: two phrasings
× the arm set doubles the panel and re-opens the
single-rendering collinearity question for zero decision
value — the instrument's subject is distance, and phrasing
is held constant precisely so distance can vary alone.

### 3.3 What the cases do NOT inherit

The tree's shipped/generated populations already carry
answer-only directives (the example fixtures, numform's
templates, negation's contract). On user fixtures the frozen
directive would duplicate or contradict an incumbent
directive (hierarchy's construct), unscreened. On generated
cases the frozen directive is the *only* directive in the
prompt, by construction and by the §7 screen. That is why
this instrument generates its own cases instead of treating
user fixtures.

## 4. Arms and the attribution contract

Six arms per case. Layouts are within one user turn; joiner
"\n" (distraction `JOINER`); B and the directive are
byte-identical wherever they appear.

| Arm | Layout | Role |
|---|---|---|
| `adj` | C + B | headline reference (added distance 0) |
| `sep_256` | C + filler(256) + B | headline dose 256 |
| `sep_1024` | C + filler(1024) + B | headline dose 1024 |
| `disp_256` | C + B + filler(256) | displaced control, dose-matched |
| `disp_1024` | C + B + filler(1024) | displaced control, dose-matched |
| `adj_after` | B + C | order/recency control, descriptive |

### 4.1 Why the sketch's control set was insufficient

The sketch claimed the cross-turn arm and `adj_after` "bound"
the filler confound. They do not: neither contains filler, so
in the sketch every separated cell differed from every
control cell in two ways at once — filler presence (hence
total prompt length) and between-ness. (Part II, F1, executes
this; §4.2 is the cure.)

### 4.2 The displaced control is the missing arm

`disp_d` carries filler sized to the same verified delta d
over the same case's adjacent render, in the same turn, but
**after** B, so:

- total rendered prompt length: equal to `sep_d` (both are
  adj + d tokens, verified per candidate — the sizing rule
  sizes each arm's span to its own exact delta, so seam
  tokenization differences are absorbed by the search);
- filler presence/content class: same corpus, same dose.
  The two arms' spans are drawn from the same stream prefix
  before sizing; they are **not** the same span — per-arm
  sizing accepts different (whole, suffix) constructions
  (executed in the fresh review: one asymmetric seam merge
  gave a 284-char separated span vs a 285-char displaced
  span at dose 256, both delta-exact). A description of the
  two arms as sharing one span would be false; per-arm
  sizing is what makes the doses equal, not span identity.
- directive↔B distance: the adjacent distance (the filler is
  not between them);
- what *does* differ from `adj`: B's distance to the
  generation point grows by d, and total length grows by d.

The headline contrast is therefore an *excess of losses*:
on the headline population, S counts `sep_d` compliance
losses and T counts `disp_d` compliance losses on the same
cases, and the rule (§6) fires only when S exceeds T by a
derived margin. A pure question-recency effect fires on
`disp_d`, not `sep_d`, and works against the headline. A
pure filler-presence/length effect fires on both — how
completely the margin differences it out depends on how
case-correlated the effect is, and §6 prints both bounds
honestly (a perfectly case-correlated length effect gives
S = T and can never fire; the independent-cells bound is
printed as a number, 0.0788, with its premise, §6).

Residual confounds, printed beside every decision, never
argued away: (i) **value-mediation** — in `sep_d` the filler
is also between the directive and the *fact*, but
fact↔question adjacency is held (§3.1), and a compliance
loss with the value kept (`noncompliant_correct_value`,
§5.2) cannot be a fact-loss story; the S split by
value-kept/value-lost is printed so a fact-mediated pattern
is visible as such; (ii) **filler-proximity** (amendment G)
— on the directive-first arms, filler immediately preceding
B is coextensive with between-ness, so a mechanism that
fails iff filler immediately precedes the governed block
(the directive never consulted; executed in the fresh
review as a deterministic mock firing S = 12, T = 0) fires
this label, and the separating arm is not among the six;
the value-kept/value-lost split mitigates but does not
gate; (iii) filler *content* is one frozen corpus (§4.3) —
the claim is scoped to neutral prose filler, and a
topically-related or instruction-like filler is a different
treatment, excluded.

### 4.3 Filler

`distraction.NEUTRAL_CORPUS` (71 sentences, digest
`NEUTRAL_CORPUS_SHA256`), reused by needle. Executed hygiene:
71 sentences, **no digit character of any kind** — filler
cannot contain the answer value, a competing number, or a
digit that primes a numeric format. Spans are stream
prefixes in stream order. Screening per §7.

## 5. Scorer

### 5.1 The span and the two verdicts

The judged span is exactly `core.evaluate`'s answer: the
decoded pre-boundary span. Two verdicts are derived from the
record per cell, never conflated:

- **Byte-exact verdict** (descriptive, beside the headline):
  `exact_match` conjoined with `boundary_found` in the cell
  (sibling pattern). This is the lab's verdict; the probe
  does not redefine it.
- **Compliance** (headline): a total function of the span
  under the frozen grammar §5.2, derived in the stage from
  the recorded span and recomputed by the diff validator
  from the carried primitives (§5.4).

The reply contract is answer-only and byte-identical in
every arm, so a fully compliant response *is* `expected`,
byte for byte — there is no prose arm whose compliant output
is a superstring of the expected answer; prose output is,
under this instrument's frozen contract, noncompliance
itself, in every arm equally (the negation settlement:
byte-identical reply contract in every arm).

Cap censoring: the default generation cap is 64 tokens. A
compliant answer is ≤3 digits plus boundary — three orders
under the cap — and no arm demands pre-answer prose, so the
answer's position in the output does not move with the
treatment. The remaining channel is input-side: longer arms
invite `arm_context_exceeded` attrition (§7), printed beside
every rate, and headline populations require completion in
all three arms of the contrast (§5.3).

### 5.2 Frozen compliance grammar and decomposition
(amendments B1–B3, C)

Grammar (total, no stripping, no casefolding, no parsing
beyond the stated extraction): a completed cell is
**compliant** iff its span matches `\A[0-9]+\Z` — fullmatch
semantics (amendment B2: Python's `$` anchor matches before
a trailing newline, so `re.match(r"^[0-9]+$", "417\n")` is
True while fullmatch is False; the frozen semantics is
fullmatch). The digit alphabet is ASCII [0-9] everywhere
(amendment B3): `digit_count` and first-run extraction use
[0-9] only — `str.isdigit()` and `\d` disagree with each
other on spans like "4²7", and Unicode digits (٤١٧, ４１７)
are non-digits under this grammar.

Decomposition of every completed cell into exactly one
class (expected `417`):

| Class | Definition | Example |
|---|---|---|
| `compliant_correct` | compliant ∧ span == expected | `"417"` |
| `compliant_wrong` | compliant ∧ span != expected | `"418"`, `"0417"` |
| `noncompliant_correct_value` | ¬compliant ∧ first maximal `[0-9]+` run == expected | `"The answer is 417."`, `"417 crates"` |
| `noncompliant_wrong` | ¬compliant ∧ (no digit run ∨ first run != expected) | `"1,417"` (first run `1`), `"I cannot find that number."`, `""` |
| `boundary_only` | ¬boundary_found ∧ the span is compliant-correct | `"417"` with the boundary missed |

Frozen rulings: the value extraction is the **first**
maximal ASCII digit run — `"1,417"` is
`noncompliant_wrong`, not correct-value (grouping is a
format violation and its first run is not the value);
leading zeros are compliant but wrong (`"0417"`); a value
embedded in a longer digit string is compliant-wrong
(`"14170"`). Empty span is `noncompliant_wrong`.

**Boundary disposition (amendment B1, per negation §4.2 as
shipped).** Boundary-missing cells are grammar-classified
over the whole-output span (the span `core.evaluate`
defines when no boundary is found). A boundary-missing cell
whose span is compliant-correct decomposes as
`boundary_only` — boundary attrition only: it cannot count
toward D (D requires the adjacent cell boundary-found,
which `compliant_correct` already implies) and never toward
S or T. Boundary-missing cells with any other span classify
normally (a boundary-missing `"418"` is `compliant_wrong`;
a boundary-missing `"The answer is 417."` is
`noncompliant_correct_value` and *does* count as a loss in
S/T). Completion = a span was produced. This disposition is
exactly negation's shipped `boundary_only` rule; the
candidate's blanket "boundary-missing cells are decomposed
out first" was broader than the precedent it cited and is
superseded (Part III, item 6, executes the three readings
the blanket text admitted, yielding three different labels
on identical generations).

### 5.3 Conditioning

D = cases whose adjacent cell is `compliant_correct`,
inside the population of cases completed in the adjacent,
separated, and displaced arms of the dose. Correctness
alone is the wrong baseline property: a case answered
correctly-but-noncompliantly when adjacent was never
compliant, and its separated noncompliance is not a
distance loss; a case compliant-but-wrong when adjacent
*can* show a compliance loss but is not evidence the
candidate both knew and formatted the answer at distance 0.
The conditioning is on the **reference arm only** (exactly
negation's D), so the treated arm's outcome never excludes
a case from D; it only classifies it. The full per-arm 2×2
(compliant × correct) is printed for every arm (§8), so a
correctness drop at distance is visible as its own readout
— it is simply not the headline, because correctness at
distance is needle/distraction's axis.

### 5.4 Records: validator-recomputable primitives
(amendment C)

Per cell the record carries: `verdict`, `boundary_found`,
`boundary_text_match`, `input_tokens`, `answer_digest`
(SHA-256 of the span), `first_digit_run_digest` (SHA-256 of
the first maximal ASCII digit run; **"" when the span has
no run** — amendment B3), `nondigit_projection_digest`
(SHA-256 of the span with every ASCII digit removed — the
field amendment C adds), and the descriptive integers
`answer_len` (characters) and `digit_count` (total ASCII
digit characters).

Every §5.2 class is a function of the digests plus the
case's expected digest, and **only** of the digests:
compliant ⇔ `answer_digest != sha256("")` ∧
`nondigit_projection_digest == sha256("")`; correct ⇔
answer digest equals expected digest; correct-value ⇔
first-run digest equals expected digest. `answer_len` /
`digit_count` are descriptive only and never a verdict
input: under the candidate's original bare-integer
derivation (compliant ⇔ `answer_len == digit_count`), an
executed forgery — identical digests, `digit_count` 3 vs
10 — flipped `noncompliant_correct_value` ↔
`compliant_wrong` undetectably (Part III, item 9). The diff
validator recomputes the classes from the record alone; the
stage classifies spans by digesting them into exactly these
inputs, so stage and validator can never disagree (the
cycle-190 digest-binding discipline; decidability executed
on 15 spans by three parties, Part III). No raw answer
text, no shape string, no prose is carried.

Per record the stage also carries `arm_tokens` and
`id_digests` per arm. An arm that was never constructed
(the sized arms when unmatchable, or context-exceeded
before sizing) carries neither — there is no render to
measure — and every constructed sized arm's token count
must sit exactly its dose above the adjacent arm's, which
the validator checks on the stored counts.

## 6. Decision rule — derived for this panel, executed

Rule, per candidate, per dose d ∈ {256, 1024}, over cases
completed in `adj`, `sep_d`, and `disp_d`:

- D = # cases with `adj` cell `compliant_correct`.
- S = # cases in D with `sep_d` noncompliant.
- T = # cases in D with `disp_d` noncompliant.
- `n_reverse` = # cases with `adj` noncompliant ∧ `sep_d`
  compliant — printed beside, never in the rule.
- Fire `instruction_distance_drop_observed` iff **D ≥ 10 ∧
  S ≥ 6 ∧ S − T ≥ 5**. Otherwise: D < 10 → `insufficient`;
  D ≥ 10 without the pattern → `below_threshold` (an
  observed null at this panel, weak evidence, never
  invariance).

Operating characteristics (exact enumeration, four
independent derivations agreeing; corrected per amendment
E):

| Model | Fire rate, frozen rule, N=24 |
|---|---|
| random labeller | 0.0013 |
| no-effect competent | 0.0058 |
| drop q = 0.25 | 0.2460 |
| drop q = 0.50 | 0.9088 |
| drop q = 0.75 | 0.9993 |
| pure length/presence (independent cells) | 0.0788 |
| pure length/presence (fully case-correlated) | 0.000 (by construction: S = T) |

**The pure-length bound carries its premise (amendment E).**
The independent-cells figure is the worst case **only under
corr(sep, disp) ≥ 0**. Under negatively paired nulls the
bound is breached, executed by two independent methods that
agree: antithetic Monte Carlo — 0.1085 at ρ = −0.3, 0.1333
at ρ = −0.6; Gaussian copula — 0.1102 at ρ = −0.5, 0.1327
at ρ = −0.8. The corrected statement: the pure-length rate
is at most **0.133 at ρ = −0.8** in every executed model,
while the no-effect rate stays **≤ 0.0077 in every paired
model** executed by any party (positive-ρ paired nulls only
improve the rule: no-effect 0.0023 / 0.0004 / ~0 at ρ =
+0.3 / +0.6 / +0.9, power 0.918–0.938). No single "the
control rate is X" sentence may be printed.

**Family-wise (amendment E).** 2 dose decisions × 24
candidates = 48 decisions per run. At the no-effect rate,
the exact expectation is **0.28 false labels** per panel
with **P(≥1) = 0.24**. Under candidate-level frailty
(Beta-distributed baseline rates, κ = 50/20/10) the
no-effect expectation rises to **E = 0.40–0.59,
P(≥1) = 0.33–0.44**. At the printed pure-length bound
(0.0788 per decision): **E ≈ 3.78 labels, P(≥1) ≈ 0.98**.
All three conversions print beside the verdicts (§8).

**Verdict availability (amendment E).** The D ≥ 10 floor is
reached with probability **0.511** at an adjacent
compliant-correct rate of 0.40 and **0.153** at 0.30
(exact), so `insufficient` is the common verdict for weak
candidates by construction; §8 states this in degree, not
only in kind.

**Utility scope (amendment F).** The panel is powered for
large drops (≥0.5 conditional) and weak for a 0.25 drop
(0.2460 exact; 0.78–0.93 power range at q = 0.50 across
every executed perturbation — pairing, frailty conditioning,
even N=20: 0.813). The instrument is a per-dose large-drop
falsifier; the 0.25 row is labelled **below resolution**
wherever the operating characteristics print.

Withholding semantics (frozen): `insufficient` = the panel
cannot speak (withheld, never acquittal); `below_threshold`
= the pattern was looked for at this panel's stated power
and not found (weak evidence).

## 7. Fail-closed vocabulary (amendment D)

No invented words beyond the section-label pattern:

- Stage details: `{cancelled,
  instruction_distance_record_budget_exceeded,
  instruction_distance_screen_failed}`.
  `…_screen_failed` is bound to the stage-level screens:
  the special-token-literal audit over the frozen directive,
  fact/question templates, topics, and corpus against every
  candidate format, and the boundary-text containment check
  over the generated texts — either hit fails the whole
  stage closed before any generation.
- Per-cell details: `{"", arm_context_exceeded,
  arm_evaluation_failed, arm_unmatchable, arm_screened}` —
  the needle cell set including `arm_screened` (amendment
  D: the candidate's §7 claimed "the needle cell set" while
  omitting it). Per-cell screening is the needle rule: a
  constructed filler span containing the case value
  (case-folded) or the candidate's decoded boundary text is
  screened — the cell is `arm_screened`, counted per arm in
  §8's attrition, and the stage continues. (Read literally
  with a stage-level binding, one screened span would have
  failed the whole stage; the shipped behaviour is
  needle's per-cell disposition, and the screening rule
  text in the protocol says so.)
- Whole-stage failure (generator, renderer, serializer,
  non-UTF-8): `{"available": False, "detail":
  "instruction_distance_unavailable"}`.
- Section labels only: `instruction_distance_drop_observed`
  / `below_threshold` / `insufficient`.
- Model noncompliance is scored, never a detail. No
  `arm_parse_failed`, no per-arm "not applicable".
- A construction that cannot be verified exact is never
  shipped approximate: sizing failure inside the budget is
  the per-cell `arm_unmatchable` (§1.1), which is the
  verification discipline operating, not an exception to it.

## 8. Report shape

Per candidate, per dose, in order: (1) per-arm 2×2 counts
(compliant × correct) for all six arms with per-arm
attrition (`n_context_exceeded`, `n_evaluation_failed`,
`n_unmatchable`, `n_screened` — amendment D) beside each;
(2) D, S, T, `n_reverse`, the decision, and S split by
value-kept (`noncompliant_correct_value`) vs value-lost,
with S − T beside it; (3) the `adj_after` compliance rate
(order control) and the closed-book floor (§8.1); (4)
per-case `id_length_delta` per arm against the adjacent
render and `max_abs_id_delta` — token-ID divergence is
reported, not gated (arms differ in text by construction;
same-ids between `adj` and another arm is a rendering
anomaly, and a same-ids compliance flip is a determinism
finding); (5) the operating-characteristics paragraph (§6,
in substance, at this panel's N=24 — printed, not
footnoted), carrying amendment E's corrected conversions
and amendment F's scope sentences.

### 8.1 Closed-book floor (descriptive, not an arm)

No extra generation: the value is a seed-derived 3-digit
number stated only in the fact sentence, so a candidate
that never saw the fact can be compliant-correct only by
guessing the number. The section prints this derivation
(900 equiprobable values under the generator; model priors
over round numbers are a stated caveat, not a hidden one)
instead of spending an arm on it.

## 9. Overlap boundary — per shipped probe, with the
duplication disclosure (amendment H)

| Probe | Its owned axis | The line |
|---|---|---|
| needle (191) | where an answer-bearing **fact** sits in fixed filler; recall vs depth | needle moves the *fact*; here the fact↔question geometry never moves (§3.1); the *directive* moves relative to a fixed block, and the score is form compliance, not recall. |
| placement (193) | question vs **payload** order inside one turn | placement moves the question around content; here question and content are one fixed block B and what moves is a directive around B, at verified token doses placement never uses. |
| distraction (184) | irrelevant filler **presence/dose** spliced into a fixture's own message, verdicts re-measured | distraction's filler is the treatment and its readout is the fixture verdict. Here filler is the *spacing medium*, held present-and-equal in the displaced control so presence is differenced out; the readout is compliance with a directive the filler separates from its question. |
| position | which **option position** holds the correct MC answer | no options, no permutation. |
| multiturn drift (189) | information sharded across **turns** | all arms are single-turn; cross-turn delivery is excluded by name (§2.2). |
| density (204) | **count** of simultaneous constraints at distance ~0 | exactly one directive in every arm; density varies how many, this varies how far. |
| hierarchy | **conflicting** instruction's privilege position | no conflict, no privilege marking, one directive. |
| structured | JSON parse/schema adherence of the **answer document** | the derivation pattern is borrowed (adherence from the recorded span, independent of `exact_match`); the treatment and population are structured's, not used here. |

**Duplication disclosure (amendment H, executed in the
fresh review).** The displaced arm's content is
byte-identical to a shipped distraction after-cell on the
candidate fixture (`distraction.build_cell_fixture(...,
"after")`: adjacent content + joiner + span), and the
separated arm is a constant prefix over a shipped
before-cell. The residue this probe adds is therefore a
contrast and a scorer, not an arm set: the mid-message
insertion point between a directive and the block it
governs, the differencing of the displaced control inside
the decision rule, and the compliance grammar scored
separately from correctness. No shipped probe differences a
between-placement against its length-matched displaced twin
inside a decision rule over a compliance grammar, and no
shipped surgery inserts mid-message between a directive and
the block it governs.

## 10. Uniqueness pass — primary sources, fenced
(amendment I)

Combination framing only; "first" is claimed nowhere.

- **LongIns**, arXiv:2406.17588, §4.3 (LIST-vs-GIST
  ablation): repeating the instruction before each question
  (LIST) beats giving it once globally (GIST); "most models
  are sensitive to the distance of instruction dependency…
  when instructions and inputs are separated by a greater
  distance, the performance of most models rapidly
  degrades." Fencing: exam-style task *performance* (not
  format compliance), concatenated question sets at
  long-context lengths, API/hosted models — no exact-token
  dose verification, no compliance/correctness
  decomposition, and LIST-vs-GIST also changes instruction
  *repetition count*, which this instrument holds at
  exactly one.
- **Instructional distractions**, arXiv:2502.04362, §5.2:
  as QA input length grows, distraction rises; the authors'
  own explanation is that it "**may be due to** the fact
  that as the passage lengthens, the distance between the
  instruction and the question grows" (the "may be due to"
  hedge is part of the quotation — amendment I restores
  it). Fencing: the distance claim is a post-hoc explanation
  of a *length* manipulation in the presence of
  instruction-like distractors — the confound (§4) this
  instrument's displaced control exists to remove; no dose
  is isolated, no compliance grammar, no local checkpoint.
- **LIFBench**, arXiv:2411.07037 (Wu et al., ACL 2025,
  Anthology 2025.acl-long.803, pp. 16445–16468; name
  collision: LIFEbench, arXiv:2505.16234). Fenced per the
  fresh review's settlement (c): §3.1 places the instruction
  at the **end** of the prompt, six intervals to 128k; §5.3
  reports performance declining "particularly beyond 16k or
  32k tokens", **but** "'Format' performance remains
  relatively stable across all input lengths in most models,
  suggesting that tasks related to formatting are less
  sensitive to longer contexts". **Licensed:** decline is
  published prior art at 4k–128k with instruction-at-end.
  **Forbidden:** citing LIFBench as support for a
  format-compliance drop at doses 256/1024, or for
  directive-at-head geometry — its §5.3 cuts against the
  former and its layout is the latter's opposite. It is
  never cited as load-bearing for any claim of this
  instrument beyond "long-context instruction following
  declines with length is published prior art".
- **Neighbour tools** (surveyed in the design round, scoped
  to the sources fetched): LM Evaluation Harness (task/metric
  framework; no instruction-distance instrument in its
  surveyed surface); HELM Capabilities / IFEval
  (verifiable-instruction following at fixed prompt layouts,
  no distance dose); OpenAI Evals (grader-scored task
  framework); PromptPerfect (prompt *optimizer*);
  DSPy (prompt-program *compiler/optimizer*). None of the
  five doses a directive's separation from its question on
  a local checkpoint crossed with chat-template families
  under a derived decision rule. This survey is a
  search-level pass over the named neighbours, stated as
  such; it does not prove a universal negative.

The combination that is this instrument's: **verified
rendered-token doses of separation between one frozen
format directive and a fixed fact+question block, on a
local checkpoint crossed with the lab's template families,
filler presence differenced out by a dose-matched displaced
control, compliance derived from the recorded span under a
frozen grammar and reported in a 2×2 against correctness,
with a panel-derived decision rule.** Scoped to the sources
above.

## 11. Naming audit

- `--instruction-distance` / `instruction_distance`: zero
  hits in the package, `docs/USAGE.md`, `docs/FAQ.md`,
  `README.md` at design time. The flag string is free.
- The word "distance" is **not** free of a reserved sense in
  the tree: surface-noise owns edit distance (Myers distance
  between texts/verdicts). Mitigation (frozen): the USAGE
  section and CLI help define the term on first use as
  "rendered tokens between the frozen directive and the
  question block, verified per candidate" and never use bare
  "distance" for the edit-distance concept in this probe's
  text.
- Collision with needle's vocabulary: needle owns "depth"
  (position of a fact *inside* filler) and "dose" (filler
  amount). This probe uses "dose" in needle's exact sense
  (verified rendered-token delta — a deliberate shared
  convention, stated in the help) and **never** uses "depth"
  — nothing here is buried at a depth; the directive stays
  at the prompt's head in every headline arm.
- Alternatives considered and rejected:
  `--directive-distance` (inherits DIRECTIVE's over-broad
  "directive"); `--instruction-separation` (invents a
  synonym where the literature — LongIns §4.3 — already says
  "distance of instruction dependency").

## 12. Counts, caps, and the utility (amendments A, F, J)

- Arms per case: 6 (§4). Cases: 24 (§3.1). Candidates: ≤24
  (§3.1).
- **Generations: 6 × 24 × 24 = 3,456 maximum.**
- Ground records: 24 × 24 = 576 multi-cell records.
  Records-bytes bound 12 MiB (house bound).
- **Sizing economy (amendment J):** sized cells are 4 per
  candidate-case (sep/disp × 2 doses) = 2,304 panel-wide.
  Expected ≈56K tokenizer encodes at the shipped
  `span_candidates` estimator's executed mean (24.5
  probes/cell, executed range 2–98 by backend); worst case
  2,304 × 512 = **1,179,648 encodes** at the frozen
  per-cell budget; plus per-generation encodes (≥3,456).
  §1.4's rule text and the CLI help both carry this line.
- Generation cap: `--max-new-tokens` as recorded (default
  64); a compliant answer needs ≤3 digits + boundary, so
  the cap is never the binding constraint on a compliant
  cell (§5.1). Context cap: `--max-context` as recorded
  (default 2048); the 1024-dose arms are sized to be
  measurable under it (§1.3), and any cell over it is
  `arm_context_exceeded`, counted.
- No winner/selection readout: generated stage-local cases
  are not the user's contest. Per-candidate decisions only;
  nothing pooled across candidates; the primary run's
  decision is never touched. Informational stage; never
  fails the run.

### Monday-morning utility (one sentence, as re-scoped by
amendment F)

For a maintainer laying out a prompt, the panel states —
per dose, at printed operating characteristics — whether
placing 256 or 1024 tokens of filler between a format
directive and the question it governs produces a large
(≥50-point conditional) format-compliance collapse beyond
what the same filler after the question produces, for this
checkpoint family; a non-fire excludes large
separation-driven collapse and claims nothing smaller.
Where the verdict is `insufficient` or `below_threshold`,
the honest deliverable is "this panel cannot distinguish
them; choose on other grounds", and the section says so in
substance. (The candidate's threshold-presupposing layout
instruction — "sets their prompt layout inside the distance
that holds" — is deleted: per-dose labels can invert the
raw dose order through control noise alone, so no layout
instruction may presuppose a threshold.)

---
# Part II — Round-1 adjudication (coordinator #43, NO-BUILD)

Reproduced verbatim from the surviving working-tree copy,
minus its original title line (superseded by this
consolidation's status header). The copy truncates
mid-F6(c), at the point marked below; the round-1 design
document's own §§2–7 do not survive (see the status
header), and the round-1 briefs are the surviving record.


## Round summary (coordinator #43, design round only — nothing built)

- Tree at main `a6705d1b9607` (cycle 213 closed green, 8,442 tests,
  precise coverage 95.3455%). This round was READ-ONLY except for
  this doc and the briefs; no suite was run, no push was made, no
  tag was cut or discussed.
- Specialists (blind to each other): researcher
  (`~/workspace/agents_out/instructiondistance/research.md`),
  red-team A (`redteam_a.md`), red-team B (`redteam_b.md`),
  verifier/claims auditor (`verifier.md`). All four briefs landed;
  the verifier resolved every externally checkable claim against
  primary sources, re-ran every surviving PoC, and recomputed all
  decision math with its own exact enumeration; the coordinator
  re-executed the load-bearing claims itself (§4). **Fabrication
  sweep: clean** — no invented symbol and no fabricated citation
  in any brief (the third consecutive clean round); the failures
  found were one wrong number (64, not 61), one wrong file
  attribution, one mislabelled table column, stale draft residues
  in the researcher's verdict text, and line drift ≤13 lines, all
  handled in §5.
- Convergence and divergence, in one paragraph: all three blind
  briefs and the verifier return NO-BUILD for the candidate as
  specified in the circulated sketch (generated fact cases, one
  frozen digits constraint, adjacent-before / adjacent-after /
  filler-dosed / cross-turn arms, digits-only compliance derived
  from records, any-discordant-pair falsifier with a borrowed
  floor, headline conditioned on adjacent-correct), and the
  verifier upholds that unanimously, fatal by fatal, by
  execution. The researcher additionally specifies an amended
  instrument (needle's verified rendered-delta doses at
  (256, 1024), the cross-turn arm dropped, dose-matched
  *displaced* controls whose losses enter the decision rule as a
  subtraction term, reference-arm conditioning on adjacent
  compliant-correct, a frozen grammar with validator-recomputable
  carried primitives, a panel-derived rule) and recommends BUILD;
  the verifier reproduced its operating characteristics exactly
  and ruled it a **resubmission, not an amendment**, under this
  program's own test, and the researcher himself flags that call
  in his §12. The coordinator concurs: the verdict is
  **NO-BUILD** — for the sketch, and as a build order for the
  amended instrument in this round. §3 records the amended
  instrument as the resubmission candidate in full, including the
  points the fresh review must settle head-on.

## 1. Why NO-BUILD: the fatal defects of the candidate as specified

Each defect below is individually blocking and was established by
execution (attribution: R = researcher, A = red-team A, B =
red-team B, V = verifier, C = coordinator).

### F1 — Attribution: distance, filler presence, and total length are one number on every sketch arm, and the separating control does not exist

By the sketch's own construction, on every same-turn arm
`distance_tokens == filler_tokens == added_total_length_tokens`
(adjacent 0/0/0; separated d/d/d — A poc1, V re-ran). Three causal
worlds — compliance fails iff between-distance > 0, iff any
neutral filler is present anywhere, iff total prompt length
exceeds a threshold — predict identical observations `[1,1,0,0,0]`
on all five sketch arms (A; B poc2 executes the length-only vs
distance-only version: identical on every arm, diverging only on
the missing control). The one arm that separates them — filler at
the same dose placed *not between* constraint and question —
does not exist in the sketch. Neither claimed control bounds it:
adjacent-after carries no filler (it is an order/recency control
only), and the cross-turn arm carries no filler either (it bundles
turn boundary + scripted `"Noted."` history + prior-turn
placement). The sketch's conceded mitigation (b) is therefore
false by construction. The tree precedents the control twice —
`repetition.py:66-71` (the `pad` arm and the `length_effect`
contrast) and `density.py:121` (`sham` in `ARM_KEYS`) — and
`docs/DISTRACTION_DESIGN.md` exists precisely because neutral
filler at these doses is a verdict-moving treatment in this tree.
Two further identities ride along (A, B): in the constraint-first
layout, distance-to-question and distance-to-generation-point
differ only by the constant question length, so "distance" and
"recency of the constraint" are affine-identical on these arms;
and the sketch never states which side the separated filler sits
on (B: separated-before and separated-after have the same
distance-between and opposite distance-to-generation — a build
would have to invent the load-bearing choice). No discordance
this instrument can produce is attributable to distance.

### F2 — The scorer relocates the byte-exact trap, and the promised verdict cannot be derived from the tree's records

(a) **The 2×2 is degenerate.** The sketch promises a
(compliant × correct) readout with correctness through the
tree's only correctness axis, byte-exact (`core.py:797`:
`exact_match = status == "completed" and answer == case.expected`,
answer = the pre-boundary span, `core.py:939`/`:966`). Executed
(B poc1, V re-ran): with a digit-string expected, exact ⇒
byte-equal ⇒ digits-only, so the (correct ∧ non-compliant) cell
is **structurally empty** (0/12 rows), and prose-correct vs
prose-wrong are indistinguishable on both axes — no value
extractor is specified. DIRECTIVE F1 relocated, not escaped.
(b) **The grammar is unfrozen.** "Digits-only match" states none
of: strip policy, sign, leading zeros, decimal, grouping, ASCII
vs Unicode digits — each executed to flip verdicts (A poc4,
B poc1: byte-exact correct compliant answers `"-42"`, `"42.5"`,
`"1,234"` score digits-only False — 3 of the 4 number shapes the
tree itself admits). The tree's standard for a frozen digits
grammar exists (`confidence.py:91-96`: ASCII only, no sign, no
leading zeros, the `int()`/`\d` traps named) and the sketch meets
none of it. (c) **Digest-only records cannot carry the verdict.**
Every sibling generated probe's records carry digests only
(`negation.py:57`: "Records carry digests only"; the
needle/placement verdict identity is `verdict == (answer_digest
== sha256(expected)) and boundary_found`). Executed (A poc4,
B poc1 §E): compliant-wrong `999` and non-compliant `42 kg` are
the same record class (digest ≠ expected, boundary found). A
digits verdict therefore needs answer text in the records
(breaks the digest discipline), a stage-attested boolean (the
forgery surface the discipline exists to remove), or a frozen
carried primitive (density's word-hash pattern) — the sketch
specifies none of the three. `structured.classify_response`
(`structured.py:568`) does not transfer: it derives adherence
from the full decoded text of *user-fixture observations*, a
record shape the generated probes deliberately do not use.
(d) **The population is silently restricted** to plain
non-negative integers — perfect word, grouped, and negative
answers all score compliance False (B poc1 §F) — a restriction
the sketch never states and which makes its "fact question"
population a numform-style lookup set, not the RAG population
its utility paragraph promises.

### F3 — The borrowed floor, minus the test it was borrowed from, is a noise printer

What density and field-order actually freeze is a conjunction:
floor **and** directional majority **and** an exact sign test
(`density.py:1056-1066`; `fieldorder.py:708-720`, floor constant
`fieldorder.py:99` / `density.py:128`). The sketch borrows the
floor (6) and mentions neither direction, nor the sign test, nor
α. Executed under both jitter models (the sketch never states
whether discordance is counted per pair or per case-across-arms
— the ambiguity is itself part of the defect): the floor-only
any-arm rule fires **1.000** on a random labeller under both
models (exact anchor: a single pair's P(≥6 discordant of 20) =
0.9793 — the floor sits *below* the null's expected discordance
of 10); on a competent no-effect candidate it fires **0.8912**
(A's per-case-across-arms model, exact; ≈21.4 of 24 candidates
per panel) and **0.3532** (B's per-pair model, exact DP; ≈8.48
false candidates per panel) — V's independent enumeration
confirms both models' numbers. The rule is also direction-blind
(A, deterministic): a candidate whose compliance *improves* with
distance "falsifies" identically. 96 unadjusted decisions per
panel (24 candidates × 4 separated contrasts), no family-wise
expectation anywhere. This is the negation round's borrowed-(6,2)
failure at larger scale — that rule at least demanded a joint
pattern.

### F4 — Conditioning on adjacent-correct deletes exactly the cases the treatment moves (answer-syntax F2, executed)

Deterministic worlds (A poc7, B poc4 — agreeing): 20 cases,
compliance 1.00 when adjacent; at distance the 10
adjacent-wrong-but-compliant cases lose the format too, so
unconditional compliance falls **1.00 → 0.50** — a 50-point
format-only effect that is the claim's subject — while the
conditioned headline (pool = the 10 adjacent-correct cases)
reports discordant 0, "identical at every distance". The mirror
world hides gains symmetrically. And verdict availability is
outcome-dependent (A world C, B §B): a candidate with ≤5
adjacent-correct cases can never reach a floor of 6 inside its
conditioned pool, so the weakest candidates — the ones the
utility is about — are unfalsifiable by arithmetic. The 2×2
printed *beside* the headline does not cure this: the
falsification claim and its floor operate only on the
conditioned slice, so the claim as written ("compliance is
identical at every distance") is never tested.

### F5 — The falsifier cannot express "decay", the claim the slate title makes

The rule is a function of pairwise inequality only — no
direction, dose-order, or monotonicity term. Executed (A poc3,
B poc3 §B; C independently, §4): monotone decay, recovery
(compliance *improving* with distance), a threshold, a U, and a
single-dose dip all return the identical verdict
(`('falsified', 8)` under C's deterministic rendering). The
tree's own position prior art is non-monotone (`needle.py:25-34`
credits Lost in the Middle's edge-minus-middle functional form);
a design that presupposes monotone decay contradicts it, and
this rule could not detect the contradiction if it occurred.
Per-dose rates printed descriptively could show a shape; the
sketch's headline — its only decision-bearing output — cannot.
Blocking for any build carrying the word "decay" or a
curve-based layout recommendation.

### F6 — The rungs are not one dimension, the doses are undefined, and the top dose is unmeasurable at defaults

(a) **Bundle (DIRECTIVE F3 verbatim).** The cross-turn arm's
rendered gap between constraint and question is a few dozen
framing tokens — executed renders: 67 / 115 / 75 chars for
chatml / llama3 / gemma (B poc2; A's independent renders agree) —
*below* the smallest same-turn dose, while adding a turn
boundary, an assistant turn, and per-family framing. Ordered by
tokens it sits next to adjacent; ordered by turns it is last.
The arms swap order by property, so "distance" names a bundle.
(b) **"F tokens" is undefined.** A fixed filler text encodes to
wildly different token counts under different tokenizers
(A poc5: 200 word-tokens vs 1,140 char-tokens; B poc2: 192 /
959 / 479 under three toy encoders), and a merged token can span
the filler/question seam (B, executed). The settlement that
makes a dose meaningful exists — needle's: dose as a
rendered-token delta over the case's own dose-0 render, per
candidate, exact-hit or `arm_unmatchable`, frozen encode budget
(`needle.py` docstring; `distraction.py` sizing discipline) —
and the sketch adopts none of it: no baseline, no sizing rule,
no budget, no seam policy. Here the omission bites harder than
in needle: candidates are template families, so a text-level
dose is a different dose for every row of the readout.
(c) **The 2048 rung is a constant censoring class.**
`cli.py:198-199`: `--max-new-tokens` default 64,
`--max-context` default 2048; the stages' context check is
`len(ids) + max_new 
...[truncated 17489 chars]

---


[The surviving round-1 text truncates here, mid-sentence,
inside F6(c). The check F6(c) was citing is verifiable from
the tree it references: the stages' context check is
`len(ids) + max_new_tokens > max_context` (e.g. the
negation stage's cell construction). The remainder of
F6(c) is lost; the round-1 red team B brief
(`~/workspace/agents_out/instructiondistance/redteam_b.md`)
carries the surviving form of the 2048-rung argument.]

---

# Part III — Fresh-review adjudication (coordinator #59)

# Fresh-review adjudication — instruction-distance §3 resubmission (coordinator #59, 2026-10-28)

**Verdict: NO-BUILD as written. BUILD-as-amended**, under the parent's
amendment test: every required amendment below is text, pins, reporting,
or record-binding; none touches the endpoints (doses 256/1024, six arms,
label vocabulary), the population (24 generated cases, D-conditioning as
designed in §5.3), or the headline scorer (the §5.2 compliance grammar
and decomposition, the rule D≥10 ∧ S≥6 ∧ S−T≥5). A build that omits any
of amendments A–J is not this verdict's build. If BUILD: do not build in
this round — the parent sequences build cycles.

**Document-integrity note (side finding, reported not fixed).** The
working-tree copy of `docs/INSTRUCTIONDISTANCE_DESIGN.md` is truncated
mid-F6(c) at line 214, ending "`len(ids) + max_new "; the file is
untracked and no fuller copy exists anywhere in the tree (searched).
The MkDocs site render and search index were built from the same
truncated source. The candidate text of record for this review is
therefore `~/workspace/agents_out/instructiondistance/research.md`
§§1–12, which the round summary and the parent brief agree is what §3
transcribed. This adjudication is appended to the truncated doc as the
new final section.

## What survived blind attack (executed, three parties independently)

- **Inherited OC CONFIRMED.** Exact enumeration (coordinator, independent
  convention) reproduces §6: random 0.0013; no-effect competent **0.0058**;
  power q=0.25 → 0.2460; q=0.50 → **0.9088**; q=0.75 → 0.9993; pure-length
  independent **0.0788**. Red team B's exact enumeration and red team A's
  recomputation agree to rounding on every row; the round-1 verifier
  makes four independent derivations. Neighbour grid confirms §6's knee.
- **Paired/correlated nulls, positive ρ: the rule survives.** Coordinator
  MC (200k, seed 20261028): ρ=+0.3/0.6/0.9 → no-effect 0.0023/0.0004/~0,
  pure-length 0.0448/0.0132/0.0001, power 0.918–0.938. Red team B's
  Gaussian-copula and shared-beta models agree in direction. The
  independent-cells convention is the worst case for ρ≥0, as §6 claims.
  Power under S–T pairing: 0.89–0.93 (B, executed).
- **F1 cure is real.** Executed geometry: sep vs disp have identical
  total length, identical constraint position (head), identical
  constraint-to-generation distance; only the governed block's position
  differs. The dose-matched displaced control exists and differences out
  length and constraint-to-generation distance by construction.
- **F2(c) cure is real.** §5.4's digest derivation is decidable:
  coordinator executed stage-class == validator-class on 15 spans
  including every frozen ruling ("1,417"→noncompliant_wrong;
  "0417"/"14170"→compliant_wrong; ""→noncompliant_wrong); red team B
  11/11 on an independent implementation. Under the ASCII pin
  (amendment B3) there are no divergences.
- **Context budget at dose 1024: attack FAILED-TO-KILL.** Red team A
  executed renders over every catalog family plus local-repair variants:
  worst adjacent render 307 tokens (byte tokenizer), worst 1024-arm
  total 1,395 vs the 2,048 cap (headroom ≥653), sep/disp totals
  identical — no arm-ordered attrition between the headline arms is
  possible at a dose.
- **Duplication kill FAILS; residue is real but narrower than §9
  claims.** No shipped probe differences a between-placement against its
  length-matched displaced twin inside a decision rule over a compliance
  grammar, and no shipped surgery inserts mid-message between a
  directive and the block it governs. But red team B executed:
  `disp_d` is byte-identical to a shipped distraction after-cell on the
  candidate fixture (`distraction.build_cell_fixture(..., "after")`,
  distraction.py:409–422), and `sep_d` is a constant prefix over a
  shipped before-cell. The residue is a contrast and a scorer, not an
  arm set.

## Settlement (a) — the displaced gate under blind attack

The estimand survives: S−T identifies the excess of between-placement
losses over displaced-placement losses on D, with total length and
constraint-to-generation distance held identical by construction. Three
attacks landed as fencing/naming items, none as a kill:

- **A-K1.1 (landed, amendment G).** A deterministic mock that fails iff
  filler immediately precedes the governed block (fact-proximity; the
  constraint never consulted — constraint-deleted and dose-8
  counterfactuals fail identically) fires S=12, T=0. On
  constraint-first arms, filler-before-B proximity is coextensive with
  between-ness, and the separating arm is not among the six. §4.2's
  residual list names the value-mediation channel only; the
  pure-proximity channel must be named beside it. §4.2's printed
  value-kept/value-lost split mitigates but does not gate.
- **A-K1.2/K1.3 (estimand restatements, amendment F).** A pure
  constraint-to-generation mechanism yields S=T → below_threshold even
  at a 24/24→0/24 collapse; a genuine between-effect masked by the
  control's own trailing-filler effect (S=10, T=10) also yields
  below_threshold. Both are the frozen excess estimand operating as
  defined (§2.1/§4.2), with S and T printed — but §§1.2/2.1/12 must stop
  describing the label as a distance effect simpliciter.
- **B-K6 (landed, amendment F).** Per-dose labels can invert the raw
  dose order through control noise alone (executed patterns: identical
  S at both doses, opposite labels, driven by T). §2.2 disavows
  curve/threshold claims; §1.3's "three-point curve" sentence and
  §12's "inside the distance that holds" utility presuppose exactly
  that shape. The utility sentence is incoherent in the
  fire-at-256-only world. Claim text must change; the labels themselves
  are exactly as strong as §2.1 says and no stronger.

## Settlement (b) — sizing cost, recomputed at the candidate's specification

- §12's **3,456 maximum generations is correct** (6 arms × 24 cases ×
  24 candidates) and its 576-record / 12 MiB bounds check out. It omits
  the sizing economy entirely.
- Sized cells: 4 per candidate-case (sep/disp × 2 doses) = 2,304.
  Worst case at the frozen 512-encode per-cell budget:
  2,304 × 512 = **1,179,648 tokenizer encodes** (coordinator and red
  team A agree exactly; this is encodes, not generations — the parent
  brief's figure confirmed as encodes). Expected case, executing the
  shipped `span_candidates` estimator (red team A): 2–98 probes per
  cell by backend, mean 24.5 → **≈56,448 encodes panel-wide**, plus
  core.evaluate's per-generation encodes (≥3,456). The shipped needle
  precedent (GENERATION_BUDGET 7,200, same per-cell encode budget over
  more cells) carries this economy in its protocol text; §12 must too
  (amendment J).

## Settlement (c) — LIFBench fencing (primary source resolved mechanically)

arXiv:2411.07037 (Wu et al., ACL 2025, Anthology 2025.acl-long.803,
pp. 16445–16468; v3 23 Jul 2025). §5.3 "Effects of longer inputs":
performance declines "particularly beyond 16k or 32k tokens", BUT
"'Format' performance remains relatively stable across all input
lengths in most models, suggesting that tasks related to formatting are
less sensitive to longer contexts". §3.1 places the instruction at the
END of the prompt; six intervals to 128k. **Licensed:** decline is
published prior art at 4k–128k with instruction-at-end. **Forbidden:**
citing LIFBench as support for a format-compliance drop at doses
256/1024, or for directive-at-head geometry — its §5.3 cuts against the
former and its layout is the latter's opposite. The citation must carry
the arXiv/Anthology ID (name collision: LIFEbench, arXiv:2505.16234).
Candidate §10 prints only the abstract-level decline (amendment I).

## Defects in the candidate as written (each executed; classification)

Frozen-text defects — the text contradicts itself or prints falsehoods:

1. **Case-count contradiction (I-1 / B-K1).** §3.1/§6/§8 and §12's own
   counts say 24 (3,456 = 6×24×24); §12's frozen verdict sentence says
   "**20 seed-generated cases**" (research.md:812) in the same sentence
   as 3,456. 6×20×24 = 2,880 ≠ 3,456; records would be 480, not 576.
   Exact OC at N=20 (coordinator + B agree): power 0.813 at q=0.50,
   0.152 at q=0.25 — every printed OC sentence is false for the N=20
   panel. §12 declares any deviation from a frozen item voiding, so the
   frozen text cannot be built without voiding itself. The operative
   population is unambiguously 24 (§3.1 derives it from the power
   difference the text itself prints); the "20" is a stale token.
   Amendment A.
2. **Constraint length wrong (I-2).** §3.2 prints "(61 chars)"; executed
   `len("Reply with only the number: no units, no sentence, nothing
   else.")` = **64**. Flagged by the round-1 verifier; uncorrected in
   the surviving candidate text. Amendment A.
3. **§6's printed bound is false as stated (B-K4).** §6/§12 print the
   independent convention as "the worst case" and the pure-length truth
   as lying "between 0.000 and 0.079". That is a theorem only under
   corr(sep, disp) ≥ 0, a premise nowhere stated. Under negatively
   paired nulls the bound is breached by two independent executions
   that agree: coordinator antithetic MC — pure-length 0.1085 (ρ=−0.3),
   0.1333 (ρ=−0.6); red team B Gaussian copula — 0.1102 (ρ=−0.5),
   0.1327 (ρ=−0.8). No-effect stays ≤0.0077 in every paired model run
   by any party, and the corrected confound bound (≈0.13) does not
   invert the instrument's usefulness — but a frozen section may not
   print a false bound. Amendment E.
4. **Family-wise paragraph is selective (A-K4 / B-K5).** §6 converts
   only the friendly no-effect rate. Exact family-wise at the printed
   rates over 48 decisions: no-effect E=0.279, P(≥1)=0.244 (candidate
   prints 0.24/0.21 from the rounded 0.005 — correction stands).
   Candidate-level frailty (B, executed: Beta-distributed baseline
   rates, κ=50/20/10) raises the no-effect expectation to E=0.40–0.59,
   P(≥1)=0.33–0.44. At the candidate's own printed pure-length bound
   (0.0788/decision): E≈3.78 labels, P(≥1)≈0.98. §8 mandates the OC
   paragraph beside every verdict; as frozen it understates the panel
   by ~2× under unremarkable heterogeneity, and is silent on the
   confound row. Amendment E.
5. **§7 vocabulary omits `arm_screened` (I-3).** §7 claims "the needle
   cell set" but the shipped set includes `arm_screened`
   (needle.py:234–240 CELL_DETAILS, verified). Read literally with
   §4.2's stage-level binding, one screened span fails the whole stage
   as `instruction_distance_screen_failed`; §8's attrition list has no
   screened count. Amendment D.

Verdict-membership underspecification — S is not a function of the
frozen text:

6. **Boundary-missing membership (A-K2.4 / B-K3-C).** §5.2 decomposes
   boundary-missing cells out "first… never a compliance class", while
   §6's population text supports three readings with three different
   labels on identical generations (count-as-loss → fires; exclude
   case → D′=6 → insufficient; stays in D, not in S →
   below_threshold). The channel is treatment-correlated: prose that
   never emits the terminator within the cap is plausibly the
   canonical failure shape for this contract. Worse, the candidate
   misstates the precedent it cites: shipped negation
   (negation.py:347–365) defines `boundary_only` as the
   correct-text/boundary-missed class that "cannot count toward J" —
   boundary-missing cells with other text are still classified and
   stay in the population. The blanket attrition is broader than
   negation §4.2. The precedent itself supplies the disposition
   (amendment B1): grammar-classify boundary-missing cells over the
   whole-output span (the span core.evaluate defines); a
   boundary-missing cell whose span is compliant-correct decomposes
   as boundary attrition only; D requires the adjacent cell
   boundary-found; completion = a span was produced. Population,
   scorer, and endpoints unchanged — the amendment states the
   disposition the candidate's own citation already mandates.
7. **Grammar anchor unfrozen (A-K2.1).** Python `re.match(r"^[0-9]+$",
   "417\n")` is True ($ matches before a trailing newline); fullmatch
   is False. §5.4's mirror derivation agrees only with fullmatch; the
   candidate never freezes the semantics. Executed divergence on a
   one-token span. Amendment B2: freeze `\A[0-9]+\Z` / fullmatch.
8. **Digit alphabet unpinned (I-4 / A-K2.2 / B Attack 5).** `digit_count`
   is "total digit characters" with no alphabet. Executed: under a
   Unicode-aware count, ٤١٧, ４１７, ٤١٧417, ١٢٣ flip class between
   stage and validator (`str.isdigit()` and `\d` also disagree with
   each other on "4²7"). The shipped confidence grammar exists
   precisely to preclude this (confidence.py:91–96, verified).
   Amendment B3: pin ASCII [0-9] for digit_count and first-run
   extraction; state the no-run field of `first_digit_run_digest`
   ("" vs digest-of-"" is currently unstated, I-4b).
9. **Unattested integers in §5.4 (A-K2.3).** The compliance half of the
   derivation rides on two bare integers (`answer_len`,
   `digit_count`) bound by no digest. Executed forgery: identical
   digests, digit_count 3 vs 10 flips `noncompliant_correct_value` ↔
   `compliant_wrong`, undetectable by the validator. The shipped
   density precedent binds its verdict inputs by content (ordered
   word hashes, density.py:27–29, verified). Amendment C: derive
   compliant from digests — compliant ⇔ answer_digest ≠ sha256("")
   ∧ non-digit-projection digest == sha256("") — adding one record
   field and demoting `answer_len`/`digit_count` to descriptive.
   Scorer unchanged; classes identical on every executed span.

Claim-scope overreach — the instrument is sounder than its sentences:

10. **§2.1's falsifiable statement overclaims (I-5 / B-K2).** The
    statement claims identity of compliance "over this generated
    population"; the test is D-conditioned and requires completion in
    three arms. Executed residues of the F4 pattern at claim level:
    World A (effect confined to the compliant-wrong stratum:
    unconditional compliance 1.00 → 0.50, rule reads below_threshold)
    and World B (rule fires while population compliance rises
    0.583 → 0.667; the 10 reverse discordants sit outside D by
    design). Both are the disclosed estimand operating as defined —
    the defect is the sentence, which must carry the conditioning and
    completion scope (amendment F). Verdict availability is also a
    monotone function of baseline competence (B-K2 World E, exact:
    P(D≥10|N=24) = 0.511 at adjacent-CC rate 0.40, 0.153 at 0.30);
    the `insufficient` vocabulary discloses this in kind; §8's OC
    paragraph should state it in degree.
11. **Utility concedes to a large-drop falsifier (researcher §6,
    conceded; B-K5 power fragility).** At a true conditional 25%
    drop, power is 0.246 exact — and 0.06–0.17 under
    competence-correlated marginal presentation (B, executed) — so in
    ~3/4 of such worlds the deliverable is "choose on other grounds".
    LIFBench §5.3's format-stability result makes the low-power region
    the likely region. The instrument survives honestly only as a
    per-dose large-drop falsifier: power at q=0.50 is 0.909 exact and
    0.78–0.93 across every executed perturbation (pairing, frailty
    conditioning, even N=20: 0.813). Amendment F takes the fork's
    first branch (re-scope the §12 utility text); the second branch
    (enlarge the panel with re-derived OC) changes the population and
    would be a resubmission, not an amendment.
12. **Minor textual items.** §1.2's dose noun denotes a quantity the
    displaced arm does not possess (its between-distance is baseline;
    its dose is length) — §4.2 states it correctly, §1.2 is what
    USAGE inherits (A-K5a). "Same filler span" is literally false
    under one asymmetric seam merge (sep span 284 chars vs disp 285
    at dose 256, both delta-exact; A-K5b) — per-arm sizing already
    absorbs this; the phrase must go. The sizing transplant dropped
    needle's in-span offset-verification phase (needle.py:254
    SIZING_RULE_TEXT, verified) while claiming "verbatim in
    substance" — no executed in-layout bite (12/12 cells exact; the
    "\n" joiner insulates), but the claim must be corrected to name
    the omission and why it has no analogue here (A-K5c). §12's cap
    citation (determinism.py:32–38) is an audit budget; the operative
    candidate cap is experiments.py:4014.

## Binding amendments (A–J)

- **A.** §12 "20 seed-generated cases" → 24; §3.2 "(61 chars)" →
  "(64 chars)"; §12 cap citation → experiments.py:4014.
- **B.** Verdict membership pinned: (B1) boundary disposition per
  negation §4.2 as shipped (item 6 above); (B2) grammar frozen as
  fullmatch `\A[0-9]+\Z`; (B3) digit_count and first-digit-run
  extraction pinned to ASCII [0-9]; no-run `first_digit_run_digest`
  field = "".
- **C.** §5.4 compliance derived from digests via one added field
  (non-digit-projection digest); `answer_len`/`digit_count` demoted
  to descriptive (item 9).
- **D.** §7 per-cell vocabulary gains `arm_screened` (or the
  needle-parity claim is dropped and stage-level screening stated as
  a deliberate deviation); §8 prints the per-cell screened count.
- **E.** §6 reporting corrected: the pure-length bound carries its
  premise (corr(sep, disp) ≥ 0) and a negative-ρ row (≤0.133 at
  ρ=−0.8, executed); the family-wise paragraph converts both bounds
  (no-effect exact E=0.28 / P=0.24; frailty range E=0.40–0.59 /
  P=0.33–0.44; pure-length E≈3.78 / P≈0.98 at the printed bound).
- **F.** Claim sentences re-scoped: §2.1's falsifiable statement
  carries the D-conditioning and three-arm completion; §1.2's dose
  defined as added tokens (between in sep, after in disp); §1.3's
  "curve" sentence → three same-turn layouts at two doses; §12's
  utility rewritten as a per-dose large-drop falsifier with
  conditional power stated as such and the 0.25 row labelled below
  resolution; the threshold-presupposing layout instruction deleted.
- **G.** §4.2 residuals gain the filler-proximity channel by name
  (settlement (a), A-K1.1), beside the value-mediation channel.
- **H.** §9 discloses that the disp arm string is byte-identical to a
  shipped distraction after-cell on the candidate fixture and sep is
  a constant prefix over a shipped before-cell; the residue is the
  mid-message insertion point, the in-rule differencing, and the
  compliance scorer.
- **I.** §10 LIFBench fenced per settlement (c): §5.3
  format-stability sentence added; the arXiv:2502.04362 quotation's
  "may be due to" hedge restored; arXiv/Anthology IDs carried.
- **J.** §12 budget gains the sizing line: expected ≈56K tokenizer
  encodes at the shipped estimator's executed mean (24.5 probes/cell),
  worst case 1,179,648, plus per-generation encodes; generations
  3,456 and records 576 stand.

## Answers to the standing questions

- **Monday-morning utility (one sentence).** For a maintainer laying
  out a prompt, the panel states — per dose, at printed operating
  characteristics — whether placing 256 or 1024 tokens of filler
  between a format directive and the question it governs produces a
  large (≥50-point conditional) format-compliance collapse beyond
  what the same filler after the question produces, for this
  checkpoint family; a non-fire excludes large separation-driven
  collapse and claims nothing smaller.
- **Uniqueness.** The residue is real: dose-response with a displaced
  control over a directive, differenced inside the decision rule,
  scored by a correctness-separated compliance grammar on generated
  cases. It is a contrast and a scorer, not an arm set (settlement
  (a)/duplication above).
- **Duplication by direct code comparison.** Executed by coordinator
  and red team B against the shipped modules: the candidate adjacent
  arm is byte-identical to no placement arm; distraction splices
  spans into fixture messages and re-reads fixture verdicts; the only
  shared object is NEUTRAL_CORPUS (`needle.NEUTRAL_CORPUS is
  distraction.NEUTRAL_CORPUS` → True; disclosed in candidate §4.3).

## Fabrication sweep

Specialist output was treated as untested input; every load-bearing
claim was re-derived by the coordinator's own execution or
cross-executed by two specialists independently. **No fabricated
material found in any brief.**

- Researcher: citations spot-verified against the tree —
  determinism.py:32–38 (audit budget, as he reported),
  experiments.py:4014 (operative 1–24 candidate cap),
  needle.py:234–240 (CELL_DETAILS includes arm_screened),
  cli.py:604–607 (needle 4096 help; candidate cited 604–606, trivial
  drift). His I-1/I-2 line quotes (research.md:812, :212) agree with
  both red teams' independent quotes.
- Red team A: confidence.py:91–96 verified (the ASCII-grammar comment
  says what he claimed); density.py:27–29 word-hash binding verified;
  needle.py:254 SIZING_RULE_TEXT verified, including the offset phase
  the candidate dropped. His worst-case encode figure (1,179,648)
  matches the coordinator's exactly; his antithetic pure-length
  values (0.1085/0.1333) match red team B's copula values
  (0.1102/0.1327) under independent methods.
- Red team B: placement.py:69 (CASE_COUNT = 24), distraction.py:73–74
  (JOINER "\n"; NEUTRAL_LEVELS (64, 256, 1024)), and
  distraction.py:409–422 (build_cell_fixture) all verified. His
  Attack-0 exact table matches the coordinator's enumeration to
  rounding on all eight rows.
- Coordinator's own scripts: `coord_oc.py`, `coord_sizing.py`,
  `coord_decide.py` in `~/workspace/agents_out/instructiondistance2/`
  (coord_decide used `re.fullmatch`, so its 15/15 decidability result
  holds under the fullmatch reading amendment B2 freezes).

## Build cost (if the parent sequences a build)

Generations: 3,456 maximum (frozen, correct). Sizing: ≈56K tokenizer
encodes expected, 1,179,648 worst case, plus ≥3,456 evaluation
encodes. Implementation size: an estimated ~2,500–3,500 lines
including tests (stage module, diff-validator mirror, reporting
section, CLI wiring, regression tests) — an estimate from sibling
probe builds in this tree, labelled UNVERIFIED, for sequencing only.

## Side findings (reported, not fixed)

1. **Design-doc truncation** (above): `docs/INSTRUCTIONDISTANCE_DESIGN.md`
   ends mid-F6(c); its §§2–7 including any transcribed §3 are lost;
   the docs site was built from the truncated source.
2. **Round-1 verifier corrections never applied.** The round-1
   verifier required five corrections before resubmission (61→64
   chars; §12 "20 cases"; stale "J" terminology; LIFBench §5.3
   omission; exact family-wise 0.28/0.24). The surviving candidate
   text carries none of them — the resubmission was assembled from
   the uncorrected round-1 research text.
3. NEUTRAL_CORPUS verified: 71 sentences, zero digits of any kind —
   the closed-book and sizing constructions' shared premise holds.
4. The frozen constraint is 64 characters (item 2) — any future text
   quoting "61" inherits the error.
