# Enumeration convention — design (cycle 233)

`run --enumeration-convention`: render a frozen nine-item
list under six enumeration conventions (arabic `1.`,
upper alpha `A.`, lower alpha `a.`, Roman `I.`,
parenthesised wrapper `(1)`, and the index-free bullet
`•`), with the count never stated anywhere in the
content, and measure — per candidate (template family)
and per arm, behind an encode-time token-id gate and a
completion guard — whether the marker convention alone
moves the model's answers to wrong in-grammar values.

This document is the build contract. It consolidates
the wave-12 round-4 candidate with the fresh review's
amendments A1–A9 folded in (adjudication:
BUILD-as-amended, coordinator #81, 2026-10-31; built in
cycle 233). Where an amendment changed the candidate's
text, the amended text is what appears below; the
amendment list follows so a reader can see what moved.

## Amendments folded in (A1–A9)

- **A1 (the scorer reaches the stated statistic).**
  §5's class 3 applies to the answer **and** to the
  answer after removing exactly one marker-shaped
  token (leading or trailing, with its single
  separating space): if the remainder parses under the
  frozen value grammar and is not the expected value,
  the class is `wrong_value`. Precedence is pinned in
  the ordered list (the stripped test sits inside
  class 3, before echo/form_only; a remainder equal to
  the expected value continues to classes 4/5 exactly
  as before; at most one token is ever stripped — a
  doubled marker's remainder does not parse and stays
  `non_answer`; non-marker prefixes never strip, so
  the article witness stands). Class 6's enumeration
  names the prefixed family explicitly. §9 R4's
  masking sentence is restated: marker-prefixed wrong
  values count toward J; the residual `non_answer`
  reading governs genuinely unparsed discordance. No
  floor, OC figure, or §7 sentence changes — §6's
  convention sentence now describes the mechanism as
  built.
- **A2 (totality; glyphs ASCII-pinned).** §5's marker
  definition pins glyphs to ASCII (`[0-9]`, `[A-Za-z]`,
  canonical ASCII Roman); the implementation must not
  use Unicode-lenient digit tests (`isdigit()`,
  `int()`, regex `\d`) anywhere in glyph parsing — the
  same ASCII-pinned set governs A1's strip. The
  build's fuzz includes non-ASCII digit heads
  (U+00B2, U+0663, U+FF13) and asserts totality (no
  raise) and `non_answer` for those prefixes.
- **A3 (echo identity: one rule, all arms).** §5
  states echo is *string* identity with the arm's own
  marker (the marker table's form, or its all-lower
  Roman variant) in every arm; index-equality parsings
  (`03.`, `(03)`) classify `form_only`. The battery
  pins both witnesses.
- **A4 (validator glyph set completed).** V3's
  bare-glyph set gains the lower-case Roman strings
  for 1..9 and every remaining glyph the scorer parses
  (`x`/`X`, `l`/`L`, `c`/`C`, `d`/`D`, `m`/`M`), so
  V2/V3's independent enforcement matches §4's stated
  property; the build pins V2/V3 standalone, including
  the F4 witnesses as battery items.
- **A5 (length reporting at the confound's
  granularity).** §6/§10 print mean `id_length_delta`
  by target position beside the by-position D/J; §8
  gains the length-pattern reading (vi): a fire whose
  discordance concentrates at the long-marker
  positions is reported with the length explanation
  attached, including the non-monotone internal
  control (VIII. at t=8 longest, IX. at t=9 short — a
  pure length effect predicts t=8 ≫ t=9
  concentration).
- **A6 (required-N corrected).** §3's sentence is
  restated: the smallest N under the stated
  re-searched-floors criterion is **117** (floors
  (0, 18), null 0.0172, power10 0.8023), the power
  curve is non-monotone in N (117: 0.8023; 120:
  0.7657; 126: 0.8336), and the cost is 16,848
  generations (≈ 4.9×); the grid figures at 120/132
  stand as grid values. N = 24 is frozen regardless;
  §12's rejection of the enlarged panel stands on the
  corrected cost.
- **A7 (Sclar characterisation completed).** (i) §3,
  §14, and R7 disclose task190's criterion —
  probability-ranking accuracy, few-shot, on a
  classification-style task — wherever the 25-point
  target is derived, noting the source's own
  criterion-sensitivity (24% of atomic changes ≥ 5
  points under exact prefix matching vs 11% under
  probability ranking) as the byte-exact-side context.
  (ii) §2/§14 state F_item2's full six-member set,
  including the Unicode Roman-numeral series, and
  state that this design's Roman arm covers the ASCII
  upper/lower forms only — a deliberate restriction,
  with its reason (ASCII-valued panel; the Unicode
  forms are compatibility-foldable code points and
  belong to candidate 3's axis — shipped as
  `model_protocol_lab/compatibilityform.py`, whose
  Roman class is that axis's owner). (iii) §5's set
  quotation is corrected to `ROMAN[x].upper()` /
  `ROMAN[x].lower()`.
- **A8 (freeze completed in the consolidated text).**
  This doc pins: the WORDS, NEUTRAL_STATEMENTS, and
  TOPICS pools by content digest (the readiness lane's
  extension of the amendment to TOPICS); the neutral
  slot-order clause (slot 2 → first neutral, slot 3 →
  second); the material index as the 0-based case
  number; the redraw semantics (panel traversal in
  case-then-position order; the item's word and
  number slots share one redraw counter; give-up
  raises at counter > 1000); §6's plateau evaluation
  range; and the naming audit's rejection rationale
  extended explicitly to `--enumeration-style`. §13's
  forced-collision freeze test discharges FG-4's
  latency in the build.
- **A9 (evidence completeness).** The build's OC tests
  reproduce every §6 figure from their own code,
  including the exact paired column and the corrected
  §3 required-N values at the printed grid points; the
  build's guard battery gains the ids dimension and
  re-executes worlds A1/A2/B/C under the pinned
  domain; the §5 battery and fuzz ship as tests
  including A1's prefixed witnesses and A2's Unicode
  heads.

## 1. Monday-morning utility

A template author choosing one house list style for
prompts rendered across many chat templates learns, per
template family and per enumeration arm, whether
changing only the list markers on identical list
content moves the checkpoint's answers at the frozen
floors — standardising on the incumbent arabic style
unless an arm fires. The user and the decision are
real; the probe measures the marker convention on this
generated population, not a best style, not
multiple-choice option labels, not content or order.

## 2. The axis and the arms (the novel point)

**The axis is Sclar et al.'s enumeration grammar,
decomposed as their §4.3 distributional analysis
decomposes it.** In Sclar's grammar (arXiv:2310.11324,
§3.1 and Appendix A.2, opened), an itemizer is a
composition of two independent functions: `F_item2`,
the *indexing system*, and `F_item1`, the *wrapper*
(`(x)`, `x.`, `x)`, `x␣)`, `[x]`, `<x>`). `F_item2`'s
full six-member set, verbatim from Appendix A.2, is
{x+1, 'A'+x, 'a'+x, 0x215F+x+1, ROMAN[x].lower(),
ROMAN[x].upper()} — arabic, upper letters, lower
letters, the Unicode Roman-numeral series
(`0x215F+x+1`), and Roman in both cases. Their §4.3
analysis (500 formats × 250 samples × 31 tasks,
1-shot LLaMA-2-7B) finds the two dimensions behave
entirely differently:

| dimension | median spread | weakly different | strongly different |
|---|---|---|---|
| `F_item1` (wrapper) | 0.176 | **0%** | **0%** |
| `F_item2` (indexing) | 0.173 | **45%** | **10%** |

The wrapper has a large median spread yet *never*
independently predicts a performance difference; the
indexing system is one of the two strongest
independent factors in the grammar.

**This design's Roman arm covers the ASCII upper and
lower forms only** — a deliberate restriction
(amendment A7(ii)). The panel is ASCII-valued, and
the Unicode Roman-numeral series members are
compatibility-foldable code points: they belong to
candidate 3's axis, shipped in cycle 231 as the
compatibility-form probe
(`model_protocol_lab/compatibilityform.py`), whose
frozen population includes a Roman class. Nothing in
this probe renders a compatibility character.

**Arms (6, fully crossed).** Every arm renders the
same frozen list (§3); only the per-item marker
differs.

| arm key | system | role | pre-registered expectation |
|---|---|---|---|
| `control_arabic` | arabic `1.` | control | — |
| `alpha_upper` | letters `A.` | treated (indexing) | direction-free |
| `alpha_lower` | letters `a.` | treated (indexing) | direction-free |
| `roman` | Roman `I.` | treated (indexing) | direction-free |
| `wrapper` | wrapper `(1)` over arabic | **expected-null** (secondary) | no fire beyond null rates: `F_item1` is 0% weak / 0% strong in Sclar §4.3 |
| `bullet` | glyph `•`, index-free | **expected-null** (gap-filling) | no fire beyond null rates: no opened support anywhere for a bullet effect; the arm exists to close the gap, labelled and counted like every other arm |

**Per-arm hypotheses are direction-free, for every
arm.** Sclar's distributional analysis licenses the
*existence* of indexing effects (45% weak / 10%
strong task separation) but not their *direction* for
any specific system on any specific task — and
Sclar's own cross-model analysis (their Table 2 /
Appendix B.2) finds formats are not inherently good or
bad (ordering preservation across models < 0.62). The
headline rule (§6) is therefore the sibling
one-directional D-conditioned discordance rule, and no
arm is pre-registered as "should help" or "should
hurt". A fire on an expected-null arm is not a design
failure; it is the finding those arms exist to be able
to produce, and §8's reading governs it.

**The bullet glyph: `•` (U+2022 BULLET), not `-`.**
(1) `-` (U+002D HYPHEN-MINUS) is the frozen value
grammar's separator (`word-NNNN`, §5): a dash-bulleted
arm would put the grammar's own separator in the
marker position, inviting exactly the decomposition
non-uniformity round 1's adjudication recorded for its
visible-hyphen arm. The scorer of §5 must be uniform
across arms; `•` has no role anywhere else in the
panel, the grammar, or any marker system. (2) `•` is
the canonical bullet of house style guides — the arm a
template author actually chooses when they choose
"bullets" — while a dash list is a distinct convention
with its own readability semantics. (3) Sclar's
grammar contains no bullet at all (Appendix A.2): the
bullet arm is avowedly gap-filling, and choosing the
glyph with no confound against the instrument's own
grammar is the conservative fill. The bullet marker is
**the same glyph at every position** — the bullet
system carries no index — which is itself part of the
treatment. Its byte cost is disclosed: `•` is 3 UTF-8
bytes vs 1–2 for the ASCII markers; per-arm id-length
deltas are reported by the §6 regime gate, as for
every arm.

**List length K = 9, decided by the collisions it
makes live.** `K = len(INDICES) = 9` items per list,
indices 1..9. (a) At K = 9 the alpha systems reach
`I`/`i`, so the surface token `I.` exists in *two*
arms with *different* denotations (Roman 1 vs
alpha-upper 9) — the §5 scorer trap is exercised
inside the panel, not hypothesised outside it; (b)
Roman markers exercise *both* subtractive forms in
range (IV at 4, IX at 9); (c) 9 items is still a small
house-style list, and the target item's position is
drawn per case (§3), so no position is structurally
favoured. Smaller K leaves the `I.` collision
hypothetical and IX unexercised; larger K turns the
panel into a needle/depth instrument (§11 adjacency)
and lengthens every render for no additional marker
system.

## 3. Frozen generated material

The panel is 24 cases, generated from the run seed
inside the stage; there is no dependence on the user's
fixtures or the primary run. The draw is frozen:

```
draw(seed, index, slot, counter) = int.from_bytes(
  sha256(f"{seed}|enumerationconvention|{index}|{slot}|{counter}")
  .digest()[:8], "big")
```

The material **index is the 0-based case number**
(amendment A8). Per case, in draw order:

1. **Neutrals** (slots 2 and 3, counter 0): neutral
   indices = draw % len(NEUTRAL_STATEMENTS). **Slot 2
   is the first neutral, slot 3 the second** (A8); the
   second advances +1 mod len on collision.
2. **Target position** (slot 20, counter 0):
   `1 + draw % len(INDICES)`.
3. **List topics**: the stride-5 base sequence
   `B_j = TOPICS[(index + 5*j) % len(TOPICS)]` for
   j = 0..8, with the target `B_0` placed at the drawn
   position and `B_1..B_8` in order at the remaining
   positions. (Value *words* may repeat across items
   within a case; full *values* never do — the pinned
   property is panel-global value distinctness.)
4. **Item values**, per position t = 1..9 in order
   (**panel traversal in case-then-position order**,
   A8): word slot `30 + (t-1)` = `WORDS[draw %
   len(WORDS)]`; number slot `40 + (t-1)` =
   `1000 + draw % 9000`. A value colliding with any
   earlier value in the panel redraws **that item's
   two slots together — the item's word and number
   slots share one redraw counter** — and give-up
   raises when the counter exceeds 1000 (the stage
   fails closed with
   `enumeration_convention_screen_failed`).

The pools are pinned by content digest (sha256 of
`"\n".join(pool)`, UTF-8; amendment A8, extended to
TOPICS by the cycle-232 readiness lane):

- TOPICS (24): `71d122676c1b84aa96fc7e690e2308360ccbf3448d4be75213551e3d81da236e`
- WORDS (24): `26b9f09bc49ee096aa3d04a3951ef739acd617819af04c8be1bbbe0a5097f1b1`
- NEUTRAL_STATEMENTS (24): `b2afc3e88afabd8efea564d4e5eb54e94ddc1a71c355263d8ccfc7f831e6db07`

The frozen marker table (index t = 1..9):

| arm | markers |
|---|---|
| `control_arabic` | `1.` `2.` `3.` `4.` `5.` `6.` `7.` `8.` `9.` |
| `alpha_upper` | `A.` `B.` `C.` `D.` `E.` `F.` `G.` `H.` `I.` |
| `alpha_lower` | `a.` `b.` `c.` `d.` `e.` `f.` `g.` `h.` `i.` |
| `roman` | `I.` `II.` `III.` `IV.` `V.` `VI.` `VII.` `VIII.` `IX.` |
| `wrapper` | `(1)` `(2)` `(3)` `(4)` `(5)` `(6)` `(7)` `(8)` `(9)` |
| `bullet` | `•` at every position |

Content shape (frozen): `{neutral 1} Inspection
codes:\n{marker} The {topic} inspection code is
{value}. (×9, \n-joined) {neutral 2} What is the
{target topic} inspection code?` The count is never
stated anywhere in the content — not in the intro, not
in the question, not in any carrier — so the answer
can only be found by enumerating the list.

**Panel sizing and the corrected required-N
(amendments A6, A7(i)).** The panel is frozen at the
sibling size N = 24 and priced for a **25-point target
effect**: power 0.8159 at base .90 (§6). The target's
scale comes from Sclar's task190 atomic change
(0.360 → 0.614, +25.4 points) — which is
**probability-ranking accuracy, few-shot, on a
classification-style task** (Table 1's own caption),
not a byte-exact generative score; the source itself
shows the criterion is not inert (24% of atomic
changes move ≥ 5 points under exact prefix matching
vs 11% under probability ranking), and the
byte-exact-side figure is the context in which this
instrument spends the scale. The instrument's
blindness is printed, not hidden: a 10-point drop
fires 0.2542 of the time, a 5-point drop 0.0947 (§6,
§7, §9 R1). The required-N derivation, corrected by
amendment A6: scanning every integer N under the
identical re-searched-floors criterion (null ≤ 0.02,
power at drop-10 ≥ 0.80), the **smallest N is 117**
(floors (0, 18), null 0.0172, power10 0.8023; the
N = 116 control reaches only 0.7905); the power curve
is **non-monotone** in N (117: 0.8023; 120: 0.7657;
126: 0.8336; 132: 0.8325 — the 120/132 figures stand
as grid values), and the true cost of the few-points
shape is 117 × 6 × 24 = **16,848 generations
(≈ 4.9×)**, not the coarse grid's 19,008. At N = 117
the optimal pair drops the D floor entirely
((0, 18)), so the larger instrument differs in shape,
not only in size. N = 24 is frozen regardless.

## 4. Questions and the panel validator (content-addressing, enforced)

A question is admissible iff all four rules hold; the
sub-rules are separate functions so the build's tests
pin each independently:

- **V1 (template):** the question is exactly
  `What is the {target topic} inspection code?`
- **V2 (blocklist):** no ordinal or positional word
  (first…ninth, last, next, previous, preceding,
  following, above, below), no `item \d`, `option
  [A-Za-z0-9]`, or `line \d` reference, no word
  "marker", no bracket or bullet character.
- **V3 (marker disjointness):** no whitespace-delimited
  token of the question is a marker rendering of any
  arm at any index, or a bare marker glyph. The
  bare-glyph set is the digits 1–9, the letters A–I /
  a–i, the Roman strings for 1..9 in both cases, and
  every remaining glyph the §5 scorer parses — `x`/`X`,
  `l`/`L`, `c`/`C`, `d`/`D`, `m`/`M` (amendment A4).
- **V4 (answer invariance):** the expected value
  matches the frozen value grammar (fullmatch:
  `[a-z]+-[0-9]{4}`, word ∈ WORDS, number ∈
  [1000, 9999]).

The per-case battery additionally checks the frozen
shapes, the stride topic construction with the target
at the drawn position, within-case value
distinctness, pure-ASCII expected values, pairwise
distinctness of the six arm contents, and — by line
order, never by search (the bullet marker is the same
at every position) — that replacing each arm's markers
with the control markers reproduces the control
content exactly: only the marker differs across arms.

## 5. Decomposition (ordered decision list; total)

Marker interpretation is **arm-relative**: a glyph is
parsed in one named system (arabic digits; wrapper
digits; alpha upper; alpha lower; Roman, subtractive,
canonical form). The same surface token therefore
classifies differently per arm — `I.` is echo at Roman
t=1 and alpha-upper t=9, form_only in the crossed
pairs; `C.` is echo at alpha-upper t=3 and form_only
at Roman t=3 (it parses 100). Sclar's F_item2 carries
both `ROMAN[x].upper()` and `ROMAN[x].lower()`
(amendment A7(iii) corrects the candidate's
quotation), so the Roman parse is case-insensitive
between the all-upper and all-lower canonical forms;
mixed case is not a canonical rendering.

**Glyphs are pinned to ASCII (amendment A2).** A glyph
is marker-shaped iff it is arabic digits (1–2, ASCII),
a single ASCII letter, or a canonical Roman numeral
(all-upper or all-lower ASCII; `IIII` and `VX` are not
canonical, hence not Roman at all). No Unicode-lenient
digit test (`isdigit()`, `int()`, regex `\d`) appears
anywhere in glyph parsing, in the producer or the
mirror; a non-ASCII digit head (U+00B2, U+0663,
U+FF13) is not marker-shaped, never strips, and never
raises — the decision list is total: exactly one class
per completed cell.

The value grammar is the frozen one, **fullmatch**
semantics: `^[a-z]+-[0-9]{4}$` with word ∈ WORDS and
number ∈ [1000, 9999]. (Fullmatch, not `$`: Python's
`$` matches before a trailing newline, so a
trailing-newline answer must classify `non_answer`,
never as an in-grammar value — the cycle-231 audit F1
rule, with trailing-newline and trailing-space battery
cases pinned.)

Exactly one class per completed cell, in order:

1. **pass** — the verdict passed.
2. **boundary_only** — the answer digest equals the
   expected digest; the boundary conjunct alone
   failed.
3. **wrong_value** — the answer, **or the answer after
   removing exactly one marker-shaped token (leading
   or trailing, with its single separating space)**,
   parses under the frozen value grammar and is not
   the expected value (amendment A1: a wrong value
   wearing a marker prefix counts toward J exactly as
   a bare one does). A stripped remainder equal to the
   expected value continues to classes 4/5; at most
   one token is ever stripped; non-marker prefixes
   never strip (the article witness `A alder-1234`
   stays `non_answer`; a doubled marker stays
   `non_answer`).
4. **echo** — removing exactly one leading marker
   token yields exactly the expected value, and the
   token is the arm's **own** marker for the target
   index, **string-identical** (the marker table's
   form, or — Roman only — its all-lower variant;
   amendment A3). Index-equality parsings such as
   `03.` or `(03)` are never echo.
5. **form_only** — removing exactly one leading or
   trailing marker-shaped token yields exactly the
   expected value, but class 4 does not apply.
6. **non_answer** — everything else, including empty
   answers, bare markers, bare counts, article
   prefixes, doubled markers, no-space fusions,
   non-canonical Roman prefixes, non-ASCII digit
   heads, **marker-prefixed strings whose stripped
   remainder does not parse** (the prefixed family is
   named here per amendment A1: it reaches class 3
   when the remainder parses, and lands here when it
   does not), and out-of-grammar value changes.

The sibling pair partition is total: digests equal →
`boundary_only`; both boundaries found → `exact_only`;
else `mixed`.

Pinned witnesses (build battery): `IX. cedar-4159`
(Roman t=9), `I. cedar-4159`, `3. cedar-4159`
(control), `• cedar-4159` (bullet), `(9) cedar-4159`
(wrapper), and trailing `cedar-4159 3.` all classify
`wrong_value` (A1); `². cedar-4159`, `٣. cedar-4159`,
`３. cedar-4159` classify `non_answer` without raising
(A2); `03. iris-1423` at control t=3 and `(03)
iris-1423` at wrapper t=3 classify `form_only` (A3);
`ix. iris-1423` at Roman t=9 classifies `echo` (the
all-lower Roman variant).

## 6. Headline statistic, decision vocabulary, floors, guard

Per (candidate, treated arm): the population is the
**id-divergent paired domain** — cases whose treated
render's token ids differ from control's and which
both arms completed. D = control passes in the
population, J = arm cells classified `wrong_value`
among control-passing cases, k = incomplete cases in
the divergent domain. The divergent-domain size M is
printed beside every label. Floors: **MIN_D = 10,
MIN_J = 6**; cap: **MAX_INCOMPLETE = 6** over the
divergent domain only. The guarded label is a total
function of (D, J, k):

- k > 6 or D < 10 → `insufficient`
- J ≥ 6 → `discordant_asymmetric`
- J = 0 → `no_discordance_observed` iff k = 0, else
  `insufficient`
- 0 < J < 6 → `below_threshold` iff J + k < 6, else
  `insufficient`

Exhaustive audit over the reachable state space
(D + k ≤ 24: 2,925 states): 0 manufactured-fire
violations, 0 unstable-acquittal violations, exactly
264 cap-suppressed completion-stable fires. §8's
readings and every report print, per treated arm, the
**mean `id_length_delta` by target position beside the
by-position D and J** (amendment A5) — the by-position
join over the paired population — so a length
explanation is checkable at the granularity where the
confound acts.

**Operating characteristics** (exact enumeration under
per-case independence; the convention takes every arm
failure conservatively to be an in-grammar wrong
value — the statistic §5 as amended now implements;
the paired column couples the two outcomes within each
case at correlation 0.5 and is computed exactly):

| quantity | value |
|---|---|
| competent-null fire (independent) | 0.0173 |
| competent-null fire (paired, ρ = 0.5) | 0.0044 |
| weak-panel fire (base .80) | 0.1743 |
| random-labeller fire | 0.5503 |
| power, drop-5 | 0.0947 |
| power, drop-10 | 0.2542 |
| power, drop-25 (target) | 0.8159 |
| power, drop-25 at base .85 | 0.8761 |
| power, drop-25 (paired) | 0.6981 |
| power, drop-50 | 0.9990 |
| sensitivity M = 18: drop-25 / competent | 0.5214 / 0.0038 |
| sensitivity M = 12: drop-25 / competent | 0.1379 / 0.0003 |
| transplant (rejected two-directional) fire | 0.8897 |

Floor frontier: exactly (1..6, 6) — MIN_J = 6 is set
by the search (no MIN_J ≤ 5 pair meets the null
bound); MIN_D = 10 is a plateau selection, disclosed
as such. The plateau is printed [1, 14] and exact
[1, 6]; the printed evaluation range is the executed
script's MIN_D sweep 1..19 against the floor search's
grid 1..16 (amendment A8 records the range; no figure
is affected). The exact power-side gap over the
plateau is 2.796189e-11.

Family-wise, over the **120 labelled comparisons**
(5 treated arms × 24 candidates): competent
expectation 2.0729 false labels, P(≥1) 0.8764; paired
expectation 0.5331, P 0.4139; random-labeller
expectation 66.0368, P 1.0000. Guard worlds (10,000
trials, seed 20261030, ids dimension included per
amendment A9): on the zero-effect manufacture world
(A1) the guarded rule fires 0.0055 against the
unguarded 0.0186 and withholds 0.9923 as insufficient;
on the control-side concentration world (A2) guarded
and unguarded fire equally (0.0173) — D and J are
untouched by construction there — with guarded
insufficient 0.2897; on the true-drop acquittal
attack (B) the unguarded rule reads below_threshold
0.7277 while the guarded rule fires 0.1731, reads
below_threshold 0.0117, and withholds 0.8152 (the
seed-777 replication reads guarded below_threshold
0.0110); under flat incompletion (C) the guarded rule
fires 0.8893 against the unguarded 0.9399.

Budget: 6 generations per candidate per case — 3,456
at the legal maximum (24 candidates × 24 cases × 6);
576 ground records; serialised records capped at
12 MiB, degrading the whole stage to the budget detail
(§10).

## 7. Pre-registered falsifiable claim

> Over generated fact-recall cases (§3) whose control and treated
> renders are **id-divergent** (§6 gate), for a given candidate
> and enumeration arm, the arm's verdicts on control-passing cases
> show no in-grammar value-changing discordance at or above the
> frozen floor: conditioning on the id-divergent paired population,
> under the §6 guard, J < 6 whenever D ≥ 10.

- **Falsified** for a (candidate, arm) exactly when its guarded
  label is `discordant_asymmetric`. That is the whole falsification
  condition; a fire proves only that *this* arm's marker convention
  moved *this* checkpoint's answers to *wrong in-grammar values* on
  *this* generated population — nothing about weights, other list
  contents, other list lengths, or multiple-choice option labels.
- **Not falsified ≠ robust, and the resolution is part of the
  claim.** `below_threshold` and `no_discordance_observed` acquit
  only at the printed OC: a 25-point marginal drop fires 0.8159 of
  the time, a 10-point drop only 0.2542, a 5-point drop only
  0.0947. An acquittal from this instrument is *never* evidence
  that a few-point enumeration effect is absent — the instrument
  cannot see one (§3's sizing derivation). Ids-identical arms are
  instrument-inapplicable for that family, reported as such, never
  as passes.
- **A null result is a legitimate, pre-registered outcome.** If no
  arm fires for a family, the Monday-morning decision (§1) is
  served exactly as well as by a fire: the house style is free *at
  this instrument's resolution* for that family. Given Sclar's
  task-level heterogeneity (45% of tasks separate weakly, which
  entails most do not separate strongly), family-level nulls are
  the expected texture, not a failed probe. No arm, dose, or list
  length may be escalated in response to a null inside this probe
  (§9).
- The claim is per-arm and per-candidate. **No pooled headline and
  no winner readout in v1** (sibling precedent: the house-style
  decision is per checkpoint, and ranking arms by their labels
  would be abstention-dominated at these rates — and direction-free
  by §2 in any case).

## 8. Isolation rows

- The two expected-null arms are generated and classified under
  the identical rule and **receive their own labels** (they are
  counted in the §6 family-wise table for exactly that reason).
  Their null behaviour *is* an isolation row: under the null
  process each fires at the §6 table's rates, printed per candidate
  in the report.
- **Pre-registered isolation readings.** (i) A treated indexing
  arm fires *and neither expected-null arm fires* for the same
  candidate: the reading is indexing-system-specific — the active
  dimension is the one Sclar's `F_item2` identifies. (ii) The
  **wrapper** arm fires (with or without treated arms): the reading
  contradicts the `F_item1` prior on this population and is
  reported as such — wrapper sensitivity on chat-template renders
  at a scale Sclar's few-shot distributional analysis does not
  predict — never silently pooled with indexing fires. (iii) The
  **bullet** arm fires *alone*: the reading is that removing the
  index itself (not changing its system) disrupts this candidate —
  an index-presence finding, distinct from both dimensions above.
  (iv) Treated arms fire *and* an expected-null arm also fires:
  the reading is "list-marker surface change at these sites
  disrupts this candidate" (a general marker/tokenisation
  fragility finding), and the treated arms add no
  system-specific evidence beyond it. (v) If the bullet arm fires
  and no indexed arm does, the panel has measured disruption that
  *having* an index avoids — also decision-relevant for §1,
  printed as such. (vi) **Length-pattern reading (amendment
  A5):** the Roman markers' character lengths by position are
  2, 3, 4, 3, 2, 3, 4, 5, 3 (VIII. at t=8 longest, IX. at t=9
  short — Roman length is non-monotone in position), against
  constant arabic 2, alpha 2, wrapper 3, bullet 1. A fire whose
  discordance concentrates at the long-marker positions (t ∈
  {3, 7, 8}) with a null at t=9 is reported with the length
  explanation attached; a pure length effect predicts t=8 ≫ t=9
  concentration, while an indexing effect need not track marker
  length at all. The by-position join (§6) prints the mean
  id-length delta beside the by-position D/J for exactly this
  check.
- **Length isolation:** every arm's `id_length_delta` distribution
  is printed beside its label (§6 gate). A Roman-arm fire whose
  per-case deltas are large is read with that disclosure attached;
  the design does not claim to separate "Roman-ness" from "longer
  markers" beyond this reporting — no length-matched Roman arm
  exists, and inventing one (e.g. padding arabic markers) would
  change the convention being measured.
- Non-target arms/families under the null process: every arm
  notionally rendered with a null convention behaves as the
  competent/paired rows; the report's per-arm table (5 labelled
  arms × candidates, each with label, D, J, `n_reverse`,
  `n_non_answer`, and the by-position descriptive breakdown) is
  itself the isolation grid. Cross-arm comparisons use only the
  uniformly-defined classes (`wrong_value`, `non_answer`,
  `boundary_only`); echo/form classes are reported per-arm as
  descriptive — their *rates* are not comparable across arms,
  because what counts as an own-system echo differs by system
  (§5); their *existence* per arm is the scorer-artefact evidence.

## 9. Risks and pre-registered readings

- **R1 — Small-effect blindness, priced, not hidden.** §3/§6: a
  10-point drop fires at 0.2542, a 5-point drop at 0.0947. If the
  true enumeration effect on small local checkpoints is a few
  points — plausible, since Sclar's evidence is few-shot LLaMA-2
  at larger scale and task-averaged — this probe will mostly
  acquit or abstain. That is a stated property of the frozen
  instrument, and §7 forbids reading its acquittals as few-point
  nulls. The corrected required-N derivation (§3, amendment A6:
  smallest N = 117 at ≈ 4.9× cost) is the honest price of the
  alternative, recorded for a future candidate.
- **R2 — Ordinal leakage.** Closed by construction and by machine:
  the question is content-addressed (§4), the expected answer is
  arm-invariant, and the validator (V1–V4) rejects any question
  whose answer could depend on marker identity. The residual risk
  is subtler: the *model* may use marker identity internally (e.g.
  attend differently to `IX.` than to `9.`) — but that is not
  leakage; that **is** the treatment. Leakage would be the
  *question* changing meaning across arms, and the validator is
  the proof it cannot.
- **R3 — Token-length confound.** Markers differ in character and
  token length (Roman longest, bullet non-ASCII). Mitigation is
  reporting, not a rule change: per-arm id deltas beside every
  label (§6/§8), the by-position length join (amendment A5), and
  the ids gate itself. A reader who judges length the whole story
  has the numbers to say so from the report.
- **R4 — Echo asymmetry across systems (restated by amendment
  A1).** Models may echo markers more readily in some systems (a
  Roman prefix is more salient than an arabic one). Without §5
  that asymmetry *is* the result — the "Flaw or Artifact?" failure
  mode. With §5 as amended, echoes land in `echo`/`form_only`
  symmetrically *by rule* (the rule is arm-relative but identical
  in shape for every arm), and **marker-prefixed wrong values
  count toward J exactly as bare ones do** — the echo channel
  cannot launder a value change out of the headline. The residual
  risk is masking via silence (`non_answer`): an arm whose
  discordance is dominated by genuinely unparsed `non_answer` is
  reported as "the convention disrupts output form", a distinct
  finding — never silently acquitted, never pooled into J.
- **R5 — Harder panel than the siblings'.** A 9-item list is a
  harder recall task than a single fact: control pass rates may
  sit nearer the weak-null row (0.1743 spurious-fire) than the
  competent row. The design's defence is the sibling one: every
  label prints beside its candidate's control rate, D < 10
  abstains, and the by-position breakdown (§6) shows whether
  control failures concentrate at late positions (a depth effect,
  §8.45's territory) rather than spreading evenly. No floor
  adjustment is licensed by a low observed control rate — the
  floors are frozen under the stated null.
- **R6 — Fully specified panel.** Per Pecher et al.
  (arXiv:2602.04297), underspecification inflates apparent
  sensitivity; the panel's instruction/question/contract strings
  are frozen verbatim (§3) so the instrument measures the marker
  convention, not ambiguity. The list intro line
  (`Inspection codes:`) is identical in every arm.
- **R7 — Sclar's setting ≠ this setting.** Sclar measures
  few-shot, non-chat-template formats on LLaMA-2-7B/13B with a
  blended grammar and a spread estimator; this probe measures
  zero-shot payload lists on chat-template renders with paired
  classification. The citation licenses the *axis* and its
  decomposition (and the atomic magnitude scale for the target
  effect), not an expectation that any particular arm fires here.
  The transfer gaps are four, all disclosed (amendment A7(i)):
  few-shot vs zero-shot, model scale, the estimator (spread vs
  paired classification), and the **scoring criterion** —
  task190's figures are probability-ranking accuracies on a
  classification-style task, while this instrument scores
  byte-exact generations; the source's own split (24% of atomic
  changes move ≥ 5 points under exact prefix matching vs 11%
  under probability ranking) is the byte-exact-side context.
  Seleznyov et al. (arXiv:2508.11383) corroborate only the class
  claim — punctuation/format perturbations are behaviourally real
  across 8 models and 52 Natural Instructions tasks — and do
  **not** isolate list markers; nothing in this design cites them
  as if they did.

## 10. Fail-closed vocabulary, record shape, naming

- Whole-stage failure: `{"available": False, "detail":
  "enumeration_convention_unavailable"}`. Stage details:
  `cancelled`, `enumeration_convention_record_budget_exceeded`,
  `enumeration_convention_screen_failed` (sibling pattern).
  Per-cell details: `""`, `arm_context_exceeded`,
  `arm_evaluation_failed`. Rates are None only when the
  denominator is 0. Reports render counts and rates only, every
  payload-derived string through `_escape_text`.
- Record per (candidate, case): fixture id, target topic, target
  position, expected digest, per-arm content digests (6), per-arm
  token counts and render-id digests, `ids_identical` per arm vs
  control, `id_length_delta` per arm, cells and cell details per
  arm. Content digests per arm are mandatory: the diff validator's
  mirror regenerates all six arm contents from the seed and checks
  each digest — including the full marker table applied at the
  frozen positions — so the treatment itself, not just token
  counts, is committed. The mirror recomputes the §5 decomposition
  (with the answer text bound by digest) and the §4 validator,
  bit-exactly. Aggregate blocks: `arms`, `ids`, `positions`
  (the §6 by-position join), `headline`,
  `operating_characteristics`, `n_cases`, `n_incomplete`.
- **Naming audit (executed at design time; re-executed at build
  integration):** census via `cli.build_parser()` walked
  recursively; `--enumeration-convention`, `--list-marker`,
  `--enumeration`, `--itemizer`, `--list-style`, and
  `--enumeration-style` were all **FREE** at design time;
  `allow_abbrev=False` on all 17 parsers walked. Shipped probe
  flags name the axis population in hyphenated noun form
  (`--canonical-form`, `--invisible-chars`,
  `--confusable-skeleton`, `--record-notation`).
  **Chosen: `--enumeration-convention`** — it names the axis (the
  convention by which list items are enumerated), matching the
  slate candidate's own title and the shipped register; it avoids
  `--itemizer` (Sclar's internal grammar term — jargon no other
  flag uses, and it would name a component, not the convention),
  `--list-marker` (names the glyph token, and undersells that the
  treated dimension is the indexing *system*, not the mark), and
  `--enumeration` / `--list-style` / `--enumeration-style` (bare
  or style-guide-vague — each could as easily name a content
  operation; the rejection rationale is extended explicitly to
  `--enumeration-style` per amendment A8). USAGE section: **§8.63**
  (assigned at integration; §8.62 compatibility-form was the last
  shipped section at build start). Module:
  `model_protocol_lab/enumerationconvention.py`; group
  `enumeration_convention`; stage details as above. Build census
  correction: the design-time census read 82 distinct long
  options (tree mid-cycle-225); at cycle-233 integration the
  census reads **84 → 85** with this flag added — the delta is
  the two probes built since (canonical-form's successors in
  cycles 231–232), recorded here so the design-time figure is
  never quoted as current.

## 11. Duplication check (verified against the tree)

Read from `docs/USAGE.md` §8.1–§8.62 (62 probes at build
integration): no shipped probe varies the list-marker convention
on a frozen payload list. Nearest neighbours, and why they are
not this: `position` (§8.22) owns option *order*/primacy — this
probe never reorders (content and order byte-identical across
arms, executed in §3); `fewshot` (§8.27) varies demonstration
order × template in few-shot renders — this is a zero-shot
payload list; `placement` (§8.46) owns question↔payload order —
the question's position never moves; `needle` (§8.45) owns
target depth in distractor noise — here the list *is* the
payload, every item is a fact of the same form, and target
position is a drawn nuisance variable reported descriptively,
never a readout; `token-depth-sweep` (§8.53) owns token-count
effects — §6's id-length reporting exists precisely to keep
this probe from re-measuring it; `choice_audit` (§8.47) is
teacher-forced scoring over answer texts for labelled options —
this is generative recall with no labelled options, which is
also why the rejected standalone option-label candidate (§12)
stays rejected. The sibling wave-12 probes are disjoint by
axis: invisible-chars (§8.60) inserts code points *inside
words*; confusable-skeleton (§8.61) substitutes letters
*inside words*; compatibility-form (§8.62) substitutes
compatibility characters *inside a carrier sentence* —
including a Roman-numeral class, which is the Unicode axis
this probe deliberately does not touch (§2, amendment
A7(ii)); neither sibling touches the markers *between* list
items, and this probe never alters a word. (Evidence restated
in current form at build: the design-time claim "no module
mentions roman" is dead — `compatibilityform.py` ships a ROMAN
axis since cycle 231; the duplication claim itself stands,
because that axis substitutes characters inside text while
this one varies the marker system between items.) Parked set
(document-syntax, instruction-language, response-directive,
multi-question, answer-syntax) shares no axis with marker
conventions; multi-question's contribution to this design is
its *settlement obligation* (the §6 guard), not its subject
matter. The slate's dedicated sweep found no published study
isolating list markers as the sole variable; that is stated as
a negative finding, not a priority claim. Uniqueness is
claimed for the **combination only** — frozen indexing-system
arms (with expected-null wrapper and bullet arms) ×
content-addressed list recall with a mechanical validator ×
arm-relative marker-echo decomposition × paired id-divergent
classification with own-null derived floors ×
completion-guarded labels × digest-committed treatments, on
local checkpoints with exact token sequences — never "first".

## 12. Rejected alternatives

- **Which-item questions** ("which item mentions X?", expected =
  a marker or index): the expected answer's denotation changes
  across arms by definition (item 3 is `3.` / `C.` / `III.` /
  `(3)` / unmarked), so the question violates content-addressing
  at the level of the *answer*, not just the wording; scoring it
  would require cross-system answer translation, making the
  scorer part of the treatment. Rejected for v1; the §4 validator
  exists in part to make this rejection mechanical for any future
  question type.
- **Count questions** ("how many items mention Y?", expected = a
  bare digit): the expected answer collides with the arabic
  marker glyphs — an answer `3` is simultaneously a plausible
  count and a plausible stripped marker — and counts are
  insensitive to *which* system is used in a way that conflates
  counting ability with marker form. §5's battery pins the
  bare-digit answer as `non_answer` under the shipped question
  type; a count arm would need its own decomposition and is a
  separate candidate if ever.
- **The slate main-text arm list** (`1.` / `A.` / `(1)` / `-` /
  `•` / `→` as co-equal arms): superseded by Addendum 2's
  correction — it centres the wrapper glyph (0% weak / 0% strong)
  and omits two of the three indexing systems that carry the
  evidence (`a.`, Roman). The `→` arrow was dropped entirely:
  no opened source supports it, it is not an indexing system,
  wrapper, or bullet in any grammar opened, and adding an
  unsupported arm pays family-wise price for decoration.
- **The `-` bullet:** §2 — the value grammar's separator cannot
  also be a marker without breaking §5's uniformity (round-1
  visible-hyphen precedent).
- **A bracket wrapper arm `[1]`:** Sclar's `F_item1` contains it,
  but one wrapper representative suffices for the expected-null
  wrapper row; a second wrapper arm pays family-wise price to
  re-measure a dimension the primary source already prices at
  0%/0%. `[…]` remains in §5 as a *foreign* echo form, where it
  belongs.
- **Enlarging the panel for a 10-point target:** derived honestly
  in §3 as corrected by amendment A6 (smallest N = 117, cost
  16,848 generations ≈ 4.9×, floors (0, 18)) and rejected for
  *this* candidate — it breaks the sibling 24-case family the
  tasking freezes, to chase an effect size no opened source
  exhibits for this axis (Sclar's atomic example is 25.4 points).
  Recorded as the costed shape of a future few-points candidate.
- **Two-directional canonical classification:** rejected by
  execution in round 1 (0.8897 transplant fire, §6).
- **Borrowed floors:** forbidden by the slate's method note; §6's
  floors are derived under this candidate's own null and happen
  to land on the sibling pair, with the plateau disclosed.
- **Sign-test headline:** two-sided, not completion-monotone;
  this design needs no test (§6).
- **Winner readout / best-style ranking:** §7 — not consumed by
  the utility, direction-free by §2, abstention-dominated at
  these rates.
- **User-fixture arm:** shipped fixtures carry no frozen K-item
  lists in this grammar; generated-only is the sibling discipline.

## 13. Build debts

The build cycle owes, and cycle 233 discharges: the guard
battery with the ids dimension, re-executing worlds A1/A2/B/C
under the pinned domain (§6 caveat); the §5 battery and fuzz as
shipped tests, including every collision witness, A1's
prefixed witnesses, and A2's Unicode heads; the §4 validator
sub-rules pinned independently (V2/V3 standalone, including
A4's F4 witnesses); a forced-collision freeze test exercising
the §3 redraw path (the shipped evidence seeds exercise it zero
times — values are distinct on first draw on all five seeds,
so the redraw semantics are text-pinned but evidence-latent,
round-2 A9 pattern; the test discharges FG-4 by forcing
collisions through the draw function); the OC reproduction
recomputing every §6 figure from its own code, including the
exact paired column and the corrected required-N values at the
printed grid points (amendment A9).

## 14. Sources (every entry opened in the design session)

- Sclar, Choi, Tsvetkov, Suhr, "Quantifying Language Models'
  Sensitivity to Spurious Features in Prompt Design",
  arXiv:2310.11324 (v2) — opened at arxiv.org/abs/2310.11324 and
  arxiv.org/html/2310.11324v2. Used, all verified in the opened
  text: abstract (format spread up to 76 points on LLaMA-2-13B,
  ~10 points average over 50+ tasks); §3.1 + Appendix A.2 (the
  enumeration grammar: `F_item` = `F_item1` wrapper ∘ `F_item2`
  indexing; `F_item1` = {(x), x., x), x␣), [x], <x>}; `F_item2` =
  {x+1, 'A'+x, 'a'+x, 0x215F+x+1, ROMAN[x].lower(),
  ROMAN[x].upper()} — six members, the Unicode series included
  (amendment A7(ii)); enumerations indexed from 1); §4.3 Table 6
  (500 formats, 250 samples, 31 tasks, 1-shot LLaMA-2-7B:
  `F_item1` median spread 0.176, 0% weak / 0% strong; `F_item2`
  median spread 0.173, 45% weak / 10% strong; `S_1` and
  `F_item2` named the most individually impactful); §4.3
  Table 1 (task190 atomic change: `Sentence[I]- {}` /
  `Sentence[II]- {}` → `Sentence[A]- {}` / `Sentence[B]- {}` =
  0.360 → 0.614, diff 0.254 — the +25.4-point figure, which is
  an indexing change *inside* a bracket wrapper, stated as such
  wherever the design uses it as the target-effect scale; the
  figures are **probability-ranking accuracies** per Table 1's
  own caption — few-shot, classification-style — with the
  source's own criterion split at 24% vs 11% for ≥5-point moves
  under exact prefix matching vs probability ranking
  (amendment A7(i))); Appendix B.2 / Table 2 (formats not
  inherently good or bad; cross-model ordering preservation
  < 0.62 — the direction-free discipline of §2).
- Seleznyov, Chaichuk, Ershov, Panchenko, Tutubalina, Somov,
  "When Punctuation Matters: A Large-Scale Comparison of Prompt
  Robustness Methods for LLMs", arXiv:2508.11383 — opened at
  arxiv.org/abs/2508.11383 (abstract): 5 robustness methods in a
  unified framework, 8 models from the Llama/Qwen/Gemma families,
  52 Natural Instructions tasks, format/punctuation distribution
  shifts. Cited for the class claim only (§9 R7); the abstract
  isolates no list-marker effect and the design claims none from
  it.
- Hua, Tang, Gu, Gu, Wong, Qin, "Flaw or Artifact? Rethinking
  Prompt Sensitivity in Evaluating LLMs", arXiv:2509.01790 —
  opened at arxiv.org/abs/2509.01790 (abstract): 7 LLMs, 6
  benchmarks, 12 prompt templates; much reported sensitivity
  stems from heuristic evaluation including log-likelihood
  scoring and **rigid answer matching**; LLM-as-judge evaluation
  substantially reduces the variance. The §5 decomposition
  obligation — discharged here by execution, not by adopting
  judge scoring (the lab's scorer stays byte-exact; the
  decomposition is what makes its discordance interpretable).
- Pecher, Spiegel, Belanec, Cegin, "Revisiting Prompt
  Sensitivity in Large Language Models for Text Classification:
  The Role of Prompt Underspecification", arXiv:2602.04297 —
  opened at arxiv.org/abs/2602.04297 (abstract): underspecified
  prompts exhibit higher performance variance and lower
  performance; instruction prompts suffer less. The §9 R6
  fully-specified-panel obligation.
- Tree (anchored to symbols, not line numbers): `core.evaluate`'s
  `exact_match` conjunction (byte-exact scorer); the TOPICS pool
  shared with `confusableskeleton` / `compatibilityform` (§3
  digests); `density.NEUTRAL_STATEMENTS`; `canonicalform.TOPICS`
  is a *different*, diacritic pool and is not used. Census at
  build integration: 84 distinct long options pre-build, 85 with
  this flag (§10).
