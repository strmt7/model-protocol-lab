# Negation probe — design round (ADJUDICATED: BUILD, with binding amendments)

## Round summary (coordinator #36, design round only — nothing built)

- Tree at main `960b70dcde87` (cycle 211 closed green, 8,402 tests, per the
  round brief). This round was READ-ONLY except for this doc and the briefs;
  no suite was run, no push was made, no tag was cut or discussed.
- Specialists (blind to each other, then audited): researcher
  (`~/workspace/agents_out/negation/research.md`), red-team A
  (`redteam_a.md`), red-team B (`redteam_b.md`), verifier/claims auditor
  (`verifier.md`). All four briefs landed.
- Convergence and divergence, in one paragraph: both red teams
  independently returned NO-BUILD for the loose sketch, and the verifier
  sustained NO-BUILD for the researcher's narrowed instrument *as the
  researcher specified it* — its §1 "iff" is false, its §3.4 operating
  characteristics story is backwards, its winner readout must be dropped
  rather than demoted, and its floors, echo decomposition, and scope
  justification were unfrozen. The coordinator concurs with every one of
  those judgments about the brief *as specified*. The verdict below is
  nevertheless BUILD, for a precise reason stated in §1: every defect
  the verifier names is corrected *inside this doc*, by amendment to the
  same instrument — same population, same arms, same scorer, same
  joint-rule shape — with the one genuinely quantitative correction (the
  decision threshold) derived by the coordinator's own execution in §7
  and frozen here. The build cycle implements THIS doc, not the research
  brief. Any deviation from a frozen item in this doc (arms, rule,
  threshold, vocabulary, echo set, claim language) is not an
  implementation detail: it voids this verdict and requires fresh review.
  This is the distinction from the payload round's F6 rule: there the
  amendments replaced an undefined instrument with a different one
  (undefined payload → frozen tables, causal → descriptive); here they
  correct the decision rule and the claim language of an instrument
  whose population and arms were already frozen and executable.

## Verdict: BUILD (as amended in this doc)

- **Verdict: BUILD** — the narrowed instrument specified in §2–§6,
  under the binding amendments in §1.1.
- What the verdict is NOT: it is not a build order for the research
  brief, not for the loose "negation sensitivity" sketch, not for any
  winner headline, and not for the borrowed 6/2 floors. All four are
  NO-BUILD and are removed in §8.

### 1.1 Binding amendments (each incorporated below, none left open)

1. **Claim narrowed and made one-directional.** No "iff", no
   "negation-invariance is falsified", no "negation sensitivity" as a
   headline, no causal "negation changes verdicts" sentence. The
   section's only falsification-class label is
   `negation_drop_observed`, defined exactly in §2.4 as a sufficient
   pattern detector, with its named rival patterns printed beside it.
2. **Decision threshold re-derived from this panel's own operating
   characteristics (§7.3), not borrowed.** The rule is D ≥ 8 and
   J ≥ 5 (definitions in §2.4). The borrowed placement/canonical-form
   floors (6/2) are REMOVED: the coordinator reproduced the verifier's
   finding that (6,2) fires on a random labeller 58% of the time.
   At (8,5) the random-labeller fire rate is ~4%, the no-effect
   competent-candidate fire rate is 0.7%, power at a 0.5 drop rate is
   84%, and power at a 0.25 drop rate is only ~21% — the panel is
   powered for large drop rates and weak for small ones, and the
   section must print exactly that (§5).
3. **Winner readout DROPPED in v1, not demoted.** No per-arm selection,
   no winner headline, no winner-vs-abstention readout. Per-arm pass
   rates and the joint pattern are the only readouts. Grounds: the
   numform settlement (experiments stage docstring, §7.1) forbids
   selection over the synthetic set, canonical-form's `winner_headline`
   is a total function of *two* arms with no three-arm extension, and
   NUMFORM_DESIGN itself records that over 16 values an all-pass
   selection readout "carries almost no information per unit of
   validator surface". Any future selection claim must first overturn
   that settlement with argument; it may not be added at build time.
4. **Frozen `negation_echo` decomposition (§4.2).** The echo diagnostic
   is enumerated, validator-mirrored, and printed beside — never
   inside — the byte-exact headline. Without it the negated-arm rates
   are uninterpretable (a perfect comprehender answering "not second"
   scores 0/16 on the negated arm); with it, that pattern is labelled
   as echo, not as negation-drop.
5. **Antonym bundle disclosed in the claim itself (§2.2).** The negated
   arm differs from direct in two lexical moves (negation token +
   larger→smaller antonym swap). The claim is therefore
   "antonym-negated definite-description discordance", and the section
   states the bundle wherever the headline appears. The flipped arm is
   a *necessary competence gate*, not a control: flipped-pass is
   necessary, not sufficient, for attributing a negated failure to
   negation processing, and wording × position interaction remains an
   unexcluded rival, printed as such.
6. **Scope justification (§6).** Semantic instruction variation is
   deliberately excluded scope for surface-noise (see §6); this probe
   re-enters that scope for one frozen, derivable-ground-truth
   criterion variation, for the reasons in §6 — not for paraphrase
   generally.
7. **Open points decided, not deferred (§2.5, §3, §4):** the 2×2
   factorial completion is v2; the descriptive alt-negated arm is kept
   as a non-decision sensitivity only; the record shape is canonical
   multi-cell (one record per candidate × case, one cell per arm);
   the stage screen detail is `negation_screen_failed`, bound to the
   pre-generation audit screen defined in §4.3.

## 1. Why the loose sketch was NO-BUILD (fatals, each executed)

### F1 — No single arm pair diagnoses negation

Red-team B's executed truth table over the seed-0 generated compare
set: a larger-default candidate (always answers the larger label)
passes direct *and* negated 16/16 with zero direct–negated discordance,
and is indistinguishable from a smaller-blind-lexical candidate on
every per-arm pass count and every pair discordance. An always-`first`
candidate yields maximal direct–flipped discordance (16/16) with zero
negation content (red-team A E5). The direct–negated pair therefore
cannot be the headline, and the direct–flipped pair cannot be a
control: it measures antonym/criterion following, position bias, and
base-task weakness jointly. Only the per-case *triple* pattern
(pass direct, wrong-label on negated, pass flipped on the same case)
diagnoses the negation-drop pattern — and only as a sufficient
detector, never as an "iff" (see §2.4: larger-default is an
*undiagnosed rival pattern*, not an acquittal; it is visible in the
flipped pass rate printed beside the decision, never hidden).

### F2 — Byte-exact scorer × negated answer form (reporting fatal)

The verdict is byte-exact (`core.py:797`: `exact_match = status ==
"completed" and answer == case.expected`), conjoined with boundary
presence in stage cells. A semantically correct negated answer
("not second" for expected "first") therefore fails at every
scorer-lens rung — the ladder has no label-strip operation
(`scorerlens.py`: "label_prefix": "no label-strip operation exists")
— and numform's own decomposition buckets it as `non_label`, the same
class as gibberish (coordinator execution, §7.2). Wrong-label-only
counting in the joint rule neutralises this for the *boolean* (echoes
cannot fire it), but without a frozen `negation_echo` subclass the
negated-arm *rates* stay uninterpretable and the perfect comprehender
still scores 0/16 on the arm. That is why amendment 4 is binding and
why the section must never print a negated-arm pass rate without its
echo share beside it.

### F3 — The negated arm bundles treatments; length has no control

Direct (79 chars) vs flipped (80) vs "not smaller" (84) vs "not the
smaller one" (92) vs "not less than the other" (96) — red-team B's
executed character deltas. The negated arm is longer by construction,
and no length-matched negated control exists that is not itself a
treatment (verbosity territory). Length is therefore a *reported*
rival, not a controlled one: per-case `id_length_delta` is printed,
the descriptive alt-negated arm (longer still) provides a
length-gradient check, and a systematic *wrong-label* pattern (not
mere failure) is the only pattern the section may attribute — with
the length rival printed beside it, never argued away. Token-count
delta is UNEXECUTED in this round (the tree has no frozen local
tokenizer; only `backend.encode` exists) and must not be claimed in
the build doc; character delta and render-id length delta are the
reported quantities.

### F4 — Winner re-derivation over this panel is forbidden by settlement

See amendment 3. The three-arm winner rule in the research brief is
not a total function over three arms, and the settlement it would
violate is explicit in the tree (experiments numform stage docstring:
"No selection, no winner, no flip against the primary run: the
synthetic set is a different contest from the user's fixtures, and a
winner over 16 generated values per task would invite exactly the
headline the design forbids"). Dropped, not demoted.

### F5 — Borrowed floors fire on noise (new fatal for the narrowed design)

Coordinator reproduction (§7.3): the researcher's 6/2 rule fires on a
random labeller (independent 0.5 labels per arm) 58% of the time, and
the researcher's §3.4 operating-characteristics story is backwards
(it claims the rule "fires only on a fairly lopsided pattern").
A falsification rule that fires on noise more often than not is not
a conservative label; it is a broken instrument. The derived (8,5)
rule in §2.4 replaces it in full. Borrowed floors are forbidden for
every rule in this panel.

### Non-fatals, adjudicated

- **Conditional equivalence survives.** "Not the smaller" ≡ "larger"
  given a ≠ b, by the pair-closed complement rule (§2.3). The article
  form beats the bare form (the verifier's §6.1: the bare form's own
  ¬(x<y) ⟺ x≥y concedes the equality ambiguity). Ties are not a
  population defect: the generator enforces a ≠ b (`numform.py:550`)
  and the validator re-derives it; an equal pair fails the stage
  closed (§4.3), never gets tie-broken. Refuted attacks (red-team B
  R1–R5): no-equivalent-phrasing, operand-order scoping change,
  context-capacity threat (max delta +17 chars), single-digit
  breakage, ties-as-population-defect — all refuted by execution and
  not carried forward.
- **Scope (red-team A M3) stands as a requirement, not a fatal.**
  Addressed by §6.
- **Form-crossing (red-team A M1) is settled, not open.** Plain form
  only in v1; form-crossing is a v2 panel with its own claim. Budget
  arithmetic follows from the frozen record shape (§3.3).
- **Degenerate populations (red-team A M4) are pinned, not open.**
  Distinct-only, direct-pass conditioning, per-stratum/per-band
  reporting (§3).

## 2. The instrument (frozen)

### 2.1 Home and frozen base

The home is the numform compare task, plain form only. Its frozen
template is `model_protocol_lab/numform.py:121-127` (compare entry):

```
"Which of these two numbers is larger? Reply with only the word first or second.\n{a}\n{b}"
```

Compare is the one numform task whose expected answer is a
form-invariant label (`numform.py:596-603`, `expected_answer`:
compare returns `expected_label`, never a rendered number). The
generator constructs compare pairs with `a != b` enforced
(`numform.py:550`), records which side is larger (`larger_first`,
lines 572–576), and decomposes a compare cell into `exact_pass` /
`wrong_label` (the other label) / `non_label`
(`numform.py:840-863`, `cell_decomposition`). That decomposition is
load-bearing: `wrong_label` on the negated arm is the negation-drop
signature, `non_label` is a different failure, and `negation_echo`
(§4.2) is a frozen subclass of `non_label` — never pooled into
`wrong_label`, never rescuing a flip into a pass.

### 2.2 Frozen instruction strings (v1 arms)

Operands are rendered in plain form only and substituted for `{a}`
(first line) and `{b}` (second line), exactly as numform does. The
reply contract — "Reply with only the word first or second." — is
byte-identical in every arm, so the answer vocabulary never varies.
Operand order is identical across the arms of one case — only the
instruction sentence changes.

| Arm | Frozen instruction (operands appended, one per line) | Expected answer |
|---|---|---|
| `direct` | `Which of these two numbers is larger? Reply with only the word first or second.` | label of the larger operand |
| `negated` (headline) | `Which of these two numbers is not the smaller? Reply with only the word first or second.` | label of the larger operand — the same label as `direct` |
| `flipped` | `Which of these two numbers is smaller? Reply with only the word first or second.` | label of the smaller operand — the opposite label to `direct` |
| `negated_alt` (descriptive only, §2.5) | `Which of these two numbers is not smaller than the other? Reply with only the word first or second.` | label of the larger operand — never a decision input |

Worked examples (from the shipped generator at seed 0, verified by
the coordinator, §7.2): operands `[100, 47]`,
`larger_first=True`, `expected_label="first"` — direct → `first`,
negated → `first`, flipped → `second`. Operands `[199, 999]`,
`larger_first=False`, `expected_label="second"` — direct →
`second`, negated → `second`, flipped → `first`. The label flips
with presentation order, so a position-biased candidate cannot pass
all three headline arms by position alone, and the generator's
`larger_first` alternation (slot parity, `numform.py:572`) keeps the
direct expected-label balance exact by construction (8/8 at seed 0).

**Antonym-bundle disclosure (binding).** The `negated` arm differs
from `direct` in two lexical moves: insertion of the negation token
("not the") *and* the antonym swap (larger→smaller). Negating the
direct predicate alone ("not larger") *is* the flipped arm. The
claim in §2.4 is therefore about the antonym-negated definite
description as a whole, and every headline rendering of it states
the bundle. No sentence in the report may attribute a discordance
to "negation" simpliciter.

### 2.3 Expected-answer derivation rule (and the distinctness condition)

Let the pair be (x, y), x ≠ y, presented as (first, second). Define
L = the label of max(x, y), S = the label of min(x, y); L ≠ S.

- direct: "larger" denotes max(x, y) → L.
- negated: "the smaller" is a definite description denoting
  min(x, y) — it has a unique referent *only because* x ≠ y. "Not
  the smaller", read as complement within the presented pair,
  denotes the pair member that is not min(x, y), i.e. max(x, y) → L.
  The derivation is the pair-closed complement rule, not a general
  English equivalence. It is conditional, and the section states
  the condition wherever the derivation appears.
- flipped: "smaller" denotes min(x, y) → S = the opposite label.
- negated_alt: same derivation as negated, for its own phrasing;
  its scope profile (comparative + "the other" anaphora) differs
  slightly, which is one reason it is descriptive only.

**Distinctness condition (load-bearing, validator-enforced).** If
x = y, "the smaller" has no unique referent and "not the smaller"
is undefined — not "either", not "both", not a third label. The
generator asserts `a != b` (already present, `numform.py:550`); the
diff validator independently re-derives `a != b` for every case and
rejects the record set if any pair is equal; an equal pair is never
scored, never silently dropped, and never "resolved" by a tie rule.
Equal values are not a v1 population at all — a tie arm would need
its own instruction ("neither" is not in the frozen answer
vocabulary) and is excluded. Margin classes emitted by the
generator are exactly `different_length`, `same_length_early`,
`same_length_last_digit` (`numform.py:320-342`); `same_length_middle`
is a real class the classifier reports but the generator never
emits, so a record carrying it fails validation, as does `equal`.

### 2.4 Falsifiable claim (pre-registered, one-directional)

For one candidate, over the cases it completes in all three headline
arms, let D be the cases it passes on the direct arm, and let J be
the cases in D where, on the *same* case, the negated cell's
decomposition is `wrong_label` (the smaller label — not any failure,
not an echo, not a boundary miss) *and* the flipped cell passes.
Then, per candidate:

- D < 8 → `insufficient` — withheld, never defaulted to acquittal
  and never to a robustness claim.
- D ≥ 8 and J ≥ 5 → `negation_drop_observed` — the negation-drop
  pattern is observed on this candidate, on this frozen phrasing,
  at this panel size. This is a sufficient-pattern detector, not a
  biconditional: it does not establish "negation-invariance" when
  it does not fire, and it does not diagnose every negation-related
  defect when it does.
- D ≥ 8 and J < 5 → `below_threshold` — an observed null at this
  panel size, not proof of invariance. The section prints the
  operating characteristics in §5 beside every `below_threshold`.

**Named rival patterns (printed beside every decision, never hidden):**

- *Larger-default / instruction-ignoring:* passes direct and negated,
  fails flipped broadly. Visible as a low flipped pass rate; it is
  undiagnosed by J (J = 0 for it by construction) and must be
  labelled "negation-drop undiagnosed, not absent" whenever the
  flipped pass rate on D is below 0.5. It is not acquitted as
  invariant.
- *Echo-comprehension:* passes direct and flipped, answers negated
  with a negated label form (§4.2). Visible as a high `negation_echo`
  share; it is not negation-drop and must not be reported as such.
- *Wording × position interaction:* a wrong-label pattern that is
  concentrated in one presentation order (larger-first vs
  larger-second) is labelled as order-concentrated, not as general
  negation-drop. The section prints J split by presentation order.

**What the claim is not.** No causal claim about weights, training,
or "negation understanding" as a latent ability follows from any
outcome. The instrument measures verdict discordance under greedy
decoding on one frozen phrasing, not a mechanism. Generalisation to
open-world negation is an extrapolation the report must not make:
in an open-world task the complement of the negated description is
not a singleton, and the expected answer is underdetermined.

### 2.5 Variants: kept, descriptive, excluded (all decided)

| Variant | Status in v1 | Reason |
|---|---|---|
| Negated-alt phrasing "not smaller than the other" | Descriptive arm only, never a decision input, never pooled | Bounds phrasing awkwardness: a joint pattern reproducing on two negated phrasings is harder to dismiss as one phrasing's artefact; a pattern appearing on only one phrasing is labelled phrasing-dependent. Its wrong-label count on D and its alt-joint count are printed descriptively. It is frozen byte-for-byte in §2.2 or it does not ship. |
| Negated-flipped "not the larger" (expected = smaller) | EXCLUDED from v1; named as the v2 2×2 factorial completion | Fourth cell of criterion {larger, smaller} × polarity {affirmative, negated}. Its expected answer coincides with the flipped arm's, inviting pooling error, and combining two negated contrasts needs its own pre-registered rule. A v2 factorial requires a rewritten claim and fresh review, not a column added at build time. |
| Negated answer option ("reply with the word that is not first") | EXCLUDED | Changes the answer vocabulary and the scorer's target set at once; failure could be vocabulary confusion, not negation processing. |
| Imperative negation ("Do not pick the smaller…") | EXCLUDED from v1 | Different speech act (prohibition vs interrogative definite description) and a different scope; a genuine second probe, closer to hierarchy territory, not a variant of this one. |
| Scope variants "not larger" vs "not the larger" (bare vs article) | EXCLUDED as arms; REQUIRED as a design note | Without the article, "not larger" in open English means ≥, which coincides with the article reading on distinct pairs and diverges on equal pairs — precisely why §2.3's distinctness assert is load-bearing. The article is not decoration; it is the scope pin. |
| Double negation, affixal/implicit negation ("un-", "less", "never", "the other one") | EXCLUDED from v1 | Compositional and lexical negation are different mechanisms with their own ambiguity profiles. "The other one" is not a negation at all (anaphora) and must not enter a negation headline. |
| Negated operand statement ("The first number is not the smaller…") | EXCLUDED | That is a claim arm (leading territory), not a criterion arm. |

## 3. Population and controls

### 3.1 Generated cases only, stage-local, never the user Split

Cases are generated by the shipped numform compare generator (same
seed, same value set) and live only in the stage; they never enter
the user's search/held-out Split. This is the canonical-form
invariant, and it is forced here for a stronger reason than in
canonical-form: variant fixtures (direct vs negated vs flipped
renderings of one pair) are three fixtures about one pair, and any
future Split-adjacent analysis that groups by pair would inherit a
triple-counting trap. Stage-local generation avoids the question
entirely, exactly as canonical-form does
(`canonicalform.py:17-22` in its module docstring: generated cases
only, stage-local, never the Split). Note the Split mechanics for
the record: Split compares message inputs by NFC key and rejects
identical message inputs across the split (`core.py:169`, class
`Split`; uniqueness at line 204, identical-message rejection at
line 214) — variant fixtures are not identical messages, so they
would not collide, but the triple-counting trap above is the reason
to stay stage-local regardless.

### 3.2 Value ranges, strata, form, and presentation

- **Values:** reuse numform's bands verbatim —
  `numform.py:99`: `BANDS = ((1, 3), (4, 6), (7, 9), (10, 12))`
  digit lengths, 4 values per band per task, giving 16 compare
  cases per candidate. Margin strata cycle by slot —
  `different_length`, `same_length_early`, `same_length_last_digit`
  — and the headline is reported per stratum descriptively, never
  pooled across strata into a single "negation rate" that hides a
  stratum effect (e.g. a last-digit-only effect would be a
  digit-discrimination story, not a negation story). Seed-0 shape
  (verifier-confirmed): strata 8/4/4, bands 3/4/5/4, 8/16
  length-decidable (`different_length` — the larger is necessarily
  the longer string, so length alone decides those cases).
- **Form:** plain only. Crossing negation with form would be a
  3 × 4 (4 × 4 with the descriptive arm) panel whose headline could
  not attribute a flip to either axis — the payload F3 bundle
  again. Plain is the form where the operands are bare digit
  strings and the only moving part is the instruction, which is
  the point. Form-crossing is a v2 panel with its own claim.
- **Presentation order:** the shipped `larger_first` alternation
  (`numform.py:572-573`) is kept byte-for-byte, so direct expected
  labels are balanced first/second by construction and no new
  order bias is introduced.
- **Answer contract and decoding:** byte-identical reply contract
  in every arm (§2.2); greedy decoding, same `--max-new-tokens`,
  same boundary ids, same seed, same template family per
  candidate. Templates are not touched: negation lives inside the
  user message content only.
- **Closed-book screen: none, and canonical-form is the precedent.**
  Placement's screen (`placement.py:370-381`, `screened()`: a
  record is screened when its closed-book cell completed and
  passed) does **not** transfer, for a structural reason: in
  placement the payload is a prose fact passage that can be
  removed while leaving a well-formed question; in compare, the
  operands *are* the task content, and removing them leaves a
  question with no ground truth. The screen's function — excluding
  cases the treatment cannot move — is already served by
  conditioning on D: a case the candidate fails directly is a
  case negation-drop cannot be shown to move, and it is excluded
  from the headline population by construction. Canonical-form
  explicitly takes the same position (`canonicalform.py:556-557`:
  "There is no closed-book screen in this probe").

### 3.3 Counts (frozen, with derivations)

- Headline generations: 16 cases × 3 arms × 24 candidates
  (run_experiment's cap, per the placement comment on its
  24-candidate panel) = 1,152.
- With the descriptive alt arm: 16 × 4 × 24 = 1,536 generations.
- Ground records: 24 candidates × 16 cases = 384 records, in the
  canonical multi-cell shape (one record per candidate × case,
  one cell per arm — the canonical-form record shape, not
  numform's one-record-per-generation shape; this choice is what
  makes the arithmetic above the record arithmetic, and the diff
  validator regenerates arms from the same shape).
- The build cycle states the byte/record cap derivation in CLI help
  from this shape (the sibling pattern: a frozen cap constant with
  its derivation, enforced by the stage and re-derived by the
  validator), and prints these numbers with derivations, not as
  bare constants.

### 3.4 Token-ID divergence: reported, not gated

Canonical-form's headline population is ID-divergent paired cases
only (`canonicalform.py:399`, `_pair_block_divergent`), because
NFC/NFD renderings *can* collapse to identical id sequences, in
which case the instrument is inapplicable rather than the model
robust. For negation the situation is inverted: the arms differ in
instruction *text* by construction, so their full render id
sequences will diverge on any ordinary tokenizer, and
divergence-gating would be vacuous — it would gate on a condition
that is true by construction and hide the one informative event, a
same-ids case, which indicates a tokenizer/rendering pathology.
(The research brief's universal "will diverge on any sane
tokenizer" is REMOVED: it is an unexecuted universal. The design
reports divergence and treats same-ids as an anomaly, which covers
both outcomes without asserting either.)

Required machinery (canonical-form shape, different gating):

- Per case, per arm, record the full render id sequence digest and
  length, following the canonical-form stage pattern
  (`experiments.py:9639-9647`: `ids_identical` and `id_length_delta`
  computed from the arms' render ids at construction).
- Report, per candidate: `n_ids_identical` across arm pairs
  (expected 0), the per-case `id_length_delta` list in case order
  for direct-vs-negated (expected positive: the negated instruction
  is longer), and `max_abs_id_delta`.
- Headline population = cases completed in all three headline arms,
  **not** id-divergent. A same-ids case, if one ever occurs, is
  excluded from the headline population *and* counted separately
  as a rendering anomaly, never as a pass or a fail, and a
  same-ids verdict flip is recorded as a determinism finding,
  never as negation fragility (payload-round rule).
- Token-count delta is NOT reported as a measured quantity (see
  F3): the tree has no frozen local tokenizer. Character delta and
  render-id length delta are the reported quantities.

## 4. Scorer pins and fail-closed vocabulary (binding)

### 4.1 Headline scorer

`core.evaluate`'s byte-exact verdict (`core.py:797`:
`exact_match = status == "completed" and answer == case.expected`),
conjoined with boundary presence in the stage cell (the
canonical-form cell pattern: cell verdict = `exact_match and
boundary_found`). No stripping, no parsing, no normalisation is
added to the headline path. The scorer-lens ladder
(`scorerlens.py:137`, `RUNGS = (0, 1, 2, 3, 4)`) may be reported
as a sensitivity beside the headline — in particular, a
negated-arm failure that recovers at R1/R3 (whitespace/fence) is
a scorer artefact, not a negation discordance — but it never
rescues a flip into a pass, and the ladder has no label-strip
operation, so "Answer: first" does not recover against "first"
at any rung.

### 4.2 Decomposition and the frozen `negation_echo` subclass

Every completed cell is decomposed by numform's
`cell_decomposition` (`numform.py:840-863`) into `exact_pass` /
`wrong_label` / `non_label`, where for compare `wrong_label` means
the answer digest equals the digest of the other label and
`non_label` means anything else. On top of that, the build
defines and the validator mirrors exactly one frozen subclass:

- `negation_echo`: a completed cell whose raw answer is, exactly
  (byte-for-byte, no stripping, no casefolding), one of the
  negated forms of the *other* label for that arm — that is, for
  an arm whose expected label is E with other label O, the answer
  is `"not " + O` or `"not the " + O`. Concretely: expected
  `first` → echo answers `"not second"`, `"not the second"`;
  expected `second` → echo answers `"not first"`,
  `"not the first"`. Any other negated form ("not the second
  one", "not second.", "NOT second") is `non_label` (other), not
  echo — the set is frozen, and enlarging it at build time voids
  this verdict.

Reporting rules: per arm, the section prints the `exact_pass` /
`wrong_label` / `negation_echo` / `non_label` (other) split. The
`negation_echo` share is printed beside every negated-arm rate,
never pooled into `wrong_label` and never into a pass. The joint
rule (§2.4) counts only `wrong_label`, so echoes cannot fire it;
a candidate whose negated failures are predominantly echo is
reported as echo-dominant, not as negation-drop, and the section
must say so in words, not merely in a table column.

- **Boundary decomposition (mandatory):** every discordance is
  decomposed into exact-only vs boundary-only. A boundary-only
  discordance (answer text equals expected, boundary missed) is
  a boundary effect, never a negation effect, and cannot count
  toward J. The cell already carries `boundary_found`, so this
  decomposition costs no new machinery — but it must be printed,
  not merely available.
- **Expected values are the two ASCII labels** `first` /
  `second` (canonical-form SCORER PIN transplant: decomposable
  characters live in carrier prose only; here there is no carrier
  prose at all, so the pin is satisfied trivially and the scorer
  cannot be a form/encoding confound).
- **Per-arm attrition** (`n_context_exceeded`,
  `n_evaluation_failed`) is printed beside every paired rate;
  ungated discordance is printed as a sensitivity wherever any
  gating applies.

### 4.3 Fail-closed vocabulary (frozen before building)

Following the sibling pattern (e.g. density's stage details
`("cancelled", "density_record_budget_exceeded",
"density_screen_failed")` at `density.py:131-136`, and per-record
details `("", "arm_evaluation_failed", "arm_context_exceeded")`):

- Stage details exactly: `{cancelled,
  negation_record_budget_exceeded, negation_screen_failed}`.
  `negation_screen_failed` is bound to one real screen: the
  pre-generation audit screen over the frozen instruction strings
  and the generated set — special-token-literal audit over every
  frozen instruction string for every family (the
  `fixtures.py:471` `audit_special_token_literals` pattern),
  distinctness re-derivation (`a != b` for every case), and
  margin-class validity (only the three emitted classes).
  Failure of that screen makes the stage unavailable with that
  detail. (Red-team A's `negation_template_screen_failed` was
  considered and not adopted: the screen covers the whole
  generated set, not the template alone.)
- Per-cell details exactly: `{"", arm_context_exceeded,
  arm_evaluation_failed}`.
- Whole-stage failure: any generator failure, renderer failure,
  serializer exception, or non-UTF-8 payload degrades the whole
  stage to `{"available": False, "detail":
  "negation_unavailable"}` — never to a per-cell verdict. A
  validator rejection of the record set (including an equal pair)
  is likewise whole-stage `negation_unavailable`, not a
  per-cell drop.
- Model parse failure is scored wrong, never a detail. No new
  verdict words. In particular: no `arm_parse_failed`, no
  per-arm "not applicable", no `negation_too_large`.
- Section decision labels (`negation_drop_observed`,
  `below_threshold`, `insufficient`) are section labels only,
  not cell or stage details, and are defined solely by §2.4.

### 4.4 Pre-generation audit (counts, never silent drops)

Cases failing the §4.3 audit screen are excluded pre-generation
with counts (`n_audit_excluded`, by reason), never silently
dropped. For the shipped generator the expected audit-excluded
count is 0 (distinctness is enforced by construction); a nonzero
count is itself a finding about the generator and must be printed,
not normalised away. Screened cases are never scored.

## 5. Report shape and operating characteristics (printed, not footnoted)

Per candidate, the section prints, in this order:

1. Per-arm pass rates (direct / negated / flipped / negated_alt
   descriptive) with per-arm attrition beside each.
2. The decomposition split per arm (§4.2), with the
   `negation_echo` share separated on the negated arms.
3. D, J, and the decision (§2.4), with J split by presentation
   order (larger-first / larger-second) and by stratum.
4. The rival-pattern readouts: flipped pass rate on D (with the
   gate-weak qualifier when below 0.5), echo-dominant flag, and
   order-concentration flag.
5. `id_length_delta` list (direct-vs-negated, case order) and
   `max_abs_id_delta`; `n_ids_identical` (expected 0; any nonzero
   count is a determinism finding, §3.4).
6. Boundary-only vs exact-only discordance decomposition.
7. The operating-characteristics paragraph, verbatim in substance:

> "At D = N, this panel (16 cases) falsifies only negation-drop
> patterns at least as lopsided as J ≥ 5 at D ≥ 8. It does not
> establish invariance. Measured operating characteristics of
> the (8,5) rule: a random labeller (independent 0.5 labels per
> arm) fires ~4% of the time; a competent candidate with no
> negation effect (direct/flipped pass 0.9, negated wrong-label
> rate 0.1) fires 0.7%; power is 84% at a 0.5 drop rate
> (direct/flipped pass 0.9) and 99%+ at 0.75, but only ~21% at a
> 0.25 drop rate. Small effects are therefore usually missed;
> a `below_threshold` on this panel is weak evidence."

Records carry digests only (per-arm content digests, token
counts, `ids_identical`, `id_length_delta`, cells, details —
canonical-form record shape). Reports render counts/rates only,
no payload/question/answer text, every payload-derived string
through `_escape_text` (`reporting.py:28`), never bare
`escape()`.

**No winner headline, no pooled discordance.** Discordance is
never pooled across candidates/templates: per-candidate contrasts
only. The per-case joint classification is the primary readout;
per-arm rates are descriptive context beside it.

## 6. Scope justification (red-team A M3, required section)

Surface-noise deliberately excludes semantic instruction variation:
its `not_measured` list includes "semantic paraphrase of
instructions" (`surfacenoise.py`, listed twice in the module's
scope declarations), and its perturbation axis is surface-only.
This probe re-enters that scope for exactly one variation, on
three grounds that paraphrase generally does not meet:

1. **Derivable ground truth.** The negated arm's expected answer
   is derivable by construction from the pair (pair-closed
   complement, §2.3), without labelling, judging, or a gold set.
   General paraphrase has no such derivation: a paraphrase's
   expected answer is "the same as the original" by stipulation,
   which is exactly the payload round's F2 defect (a false
   "same content" claim).
2. **One frozen treatment, not a family.** The variation is one
   frozen string (§2.2), not "paraphrase" as an open class. Its
   scope is the definite-description mechanism, stated in the
   claim (§2.4), not "instruction meaning" in general.
3. **A sibling precedent for meaning-bearing instruction variation
   with controls.** Leading varies a stated claim and hierarchy
   varies instruction privilege, both meaning-bearing, both with
   joint per-case disciplines — the lab's scope exclusion is
   specific to surface-noise's axis, not a lab-wide ban on
   semantic treatments. What surface-noise excludes is *uncontrolled*
   semantic variation; this probe's variation is controlled by
   construction (§2.3, §3, §4).

If the build cycle widens the variation beyond the frozen string
(any second negated phrasing as a decision input, any paraphrase
arm, any imperative arm), it exits this justification and voids
this verdict.

## 7. Adjudication evidence (coordinator's own execution)

All snippets read-only; no suite was run.

- **Seed-0 compare population.** `build_fixtures(0)` gives 16
  plain compare cases; the first two match the research brief's
  worked examples exactly (`nf-compare-00`: "…larger?…
  \n100\n47", expected `first`; `nf-compare-01`: "\n199\n999",
  expected `second`). `larger_first` alternation and the
  `a != b` assert are as cited (`numform.py:550`, `572-576`).
- **Echo decomposition.** Constructed records through
  `numform.cell_decomposition` (`numform.py:840-863`): answer
  `"not second"` against expected `"first"` → `non_label`;
  answer `"second"` → `wrong_label`. The echo fatal is therefore
  real for *rates* (a perfect echo-comprehender scores 0/16 on
  the negated arm) and neutralised for the *boolean* (echoes
  cannot satisfy the joint rule's `wrong_label` condition) —
  exactly the verifier's §6.3 split, reproduced here.
- **Larger-default.** By construction (not simulation): a
  larger-default candidate passes direct (expected = larger) and
  negated (expected = larger) on every case and fails flipped
  (expected = smaller) on every case, so J = 0 for it. It is an
  undiagnosed rival pattern, visible only in the flipped pass
  rate — hence the gate-weak qualifier in §2.4/§5. The
  always-`first` simulation (red-team A E5) and the truth table
  (red-team B §F1, reproduced row-for-row by the verifier) are
  accepted on the verifier's reproduction.
- **Operating characteristics (§5 table).** Coordinator Monte
  Carlo, independent labels per arm, 40,000 trials per cell,
  16 cases: (6,2) random-labeller fire rate 0.581 (reproducing
  the verifier's 57.6% within seed noise); (8,5) random 0.036–0.039
  across seeds; no-effect competent (0.9, 0.9, 0.1) 0.0073;
  partial drop (0.9, 0.9, 0.5) 0.84; moderate drop (0.9, 0.9,
  0.75) 0.99+; small effect (0.9, 0.9, 0.25) 0.21. The (8,5)
  rule is the smallest (Dmin, K) pair in the explored grid
  with random fire below 5% and partial-drop power above 80%;
  (10,5) fires 0.023 with power 0.85 and is an acceptable
  stricter variant, but (8,5) is frozen here so the build does
  not re-derive it. Method note: "random labeller" is a
  noise-floor null, not a model of a real checkpoint; the
  no-effect competent row is the realistic false-fire row.
  Both are printed in §5.
- **Tree settlements verified by reading (line numbers as
  observed in this round):** `core.py:797` exact-match line;
  `numform.py:550` distinctness assert; `numform.py:572-576`
  larger-first construction; `numform.py:840-863`
  decomposition; experiments numform stage docstring ("No
  selection, no winner, no flip against the primary run");
  `canonicalform.py` `winner_headline` as a total function of
  *two* arms; `density.py:131-136` stage/per-record detail
  pattern; `surfacenoise.py` `not_measured` including
  "semantic paraphrase of instructions"; `scorerlens.py`
  "label_prefix": "no label-strip operation exists";
  NUMFORM_DESIGN's per-form selection exclusion ("over 16
  values per task an all-pass selection readout carries almost
  no information per unit of validator surface").
- **Verifier reproductions accepted:** red-team B's Monte Carlo
  (17,215/2,763/22) bit-for-bit, its truth table row-for-row,
  red-team A's scorer tables exactly (verifier §4). The
  coordinator did not re-run those reproductions; they are
  credited to the verifier, not claimed as coordinator
  executions.

## 8. Removals applied (verifier §7.2 — none of these enter any build doc)

- Research §1 "iff" biconditional and the phrase
  "negation-invariance is falsified" → replaced by
  one-directional "negation-drop discordance observed" (§2.4).
- Research §3.4 operating-characteristics story in full →
  replaced by §5/§7.3 (backwards: the 6/2 rule fires on noise).
- Borrowed 6/2 floors for every rule in the panel → replaced by
  the derived (8,5) rule.
- The winner readout in any form (secondary or headline) →
  dropped (§1.1 amendment 3).
- Research R7 word counts 11/9/9 → correct counts are
  question-only 9/7/7, full 17/15/15 (verifier §7.2); no word
  counts are load-bearing in this doc, and the §3.6 "3-word
  instruction delta" implication is removed (character deltas
  are reported instead, §1 F3).
- The "Negation Neglect survey" citation and Zhou §4
  compositional pointer → REMOVED (unresolved).
- Rezaei & Blanco as a load-bearing source → REMOVED (real
  paper, never cited in the body).
- Kassner & Schütze Table 2 digits (ρ > 85%, 20.1%, 54.2%,
  31.3%, p < 0.01) → UNVERIFIED pending primary re-resolution;
  not declared false, not used. Kassner is not load-bearing
  in this doc.
- Vrabcová §§2–3 synthesis as evidence → REMOVED (abstract only
  is used, §9).
- Research's "will diverge on any sane tokenizer" universal →
  REMOVED (replaced by report-and-anomaly, §3.4).
- Any "negation sensitivity" / "negation-blind" / "negation
  understanding" causal wording for a pair-only or pooled readout
  → replaced by the §2.4 language.
- The payload "F1–F5" count → the payload round has F1–F6.
- Red-team A M1's "exactly at the cap" gloss → REMOVED (budget
  arithmetic follows from the frozen record shape, §3.3).
- Red-team B's bare-phrasing freeze as the default → the
  article form is adopted (§2.2).
- Red-team A `negation_template_screen_failed` → considered,
  not adopted (§4.3).
- Wave-10 Elkins & Chun figures (77%/100% under negation) →
  NOT carried into this doc (not re-verified in this round).
- Incorrect line ranges corrected, not carried forward (verifier
  §2.2; substance survives in each case).

## 9. Sources the build may safely cite

Only verifier-§3.1-confirmed figures, plus the combination framing
scoped to fetched sources. No "first" is claimed anywhere in this
doc. The contribution is the combination stated in §2–§5, never
priority.

- Truong et al., arXiv:2306.08189, §3 / Table 5 (via verifier
  §3.1): negation NLI benchmarks below random for GPT-J-6B
  zero-shot (e.g. SNLI-neg 0.316, MNLI-neg 0.275, RTE-neg 0.211,
  NaN-NLI 0.298), MoNLI at chance (0.500, not below random);
  InstructGPT improves classification (RTE-neg 0.767, NaN-NLI
  0.647) with MoNLI at 0.470; Finding 4's warning that the
  InstructGPT gain could reflect similar patterns in training
  data. Fencing: curated datasets, not generated pairs; no
  template crossing, no selection rule, no flipped control.
- Ravichander et al. (CondaQA), arXiv:2211.00295, abstract and
  results section (verifier-corrected pair): UnifiedQA-v2-3B
  accuracy 73.26% vs human 91.94%; consistency 42.18% vs human
  81.58%; 14,182 pairs, 200+ negation cues. Fencing:
  human-labelled Wikipedia passages with open answers, not
  generated pairs; consistency across passage edits, not
  instruction phrasings for one fixed task.
- Vrabcová et al., arXiv:2503.22395v2, abstract only: paired
  entailment datasets differing in negation; size may improve
  negation handling; accuracy and robustness are
  language-dependent. Fencing: NLI entailment, not a generated
  binary-choice task; §§2–3 synthesis not used as evidence.
- Zhou et al., arXiv:2605.03052v1, §3.1 / Table 1 / Table 2 kept
  separate (verifier §3.1): 162 unique questions × 4 templates
  = 648 entries; positive accuracy 90.7–95.5% vs negative
  accuracy 45.4–58.3% across six open-weight models, sensitivity
  95.1–98.0% (§4.1, Table 1); late-layer attention ablation
  raising negative accuracy (e.g. Llama-3.1 50.5% → 67.8%,
  §4.2, Table 2). Fencing: logit-difference accuracy on
  single-token answers, not greedy byte-exact generation under
  a chat template with a boundary; no selection rule; no
  flipped-criterion control; its mechanistic conclusion is not
  testable through the lab's Backend interface and must not be
  claimed by this probe.
- Literature direction (not digits, except where listed above):
  negation failure is well-documented across cloze, NLI, and
  contrastive-consistency paradigms, and no fetched source
  measures it on this lab's template-family × greedy × boundary
  instrument — the combination framing, scoped to fetched
  sources. Anything beyond this paragraph needs fresh primary
  resolution before it enters a build doc.

## 10. What a build cycle must NOT do on the basis of this doc

- Do not build the research brief's §1 rule (6/2, "iff"), the
  loose "negation sensitivity" sketch, or any winner headline.
- Do not add a decision-input arm (negated-flipped, imperative,
  double negation, paraphrase) without fresh review.
- Do not enlarge the echo set, re-derive the (8,5) threshold, or
  pool discordance across candidates/templates.
- Do not write a causal "negation changes verdicts" headline, a
  prose-control framing, or any new verdict word.
- Do not transfer placement/canonical-form or field-order floors
  to this panel; its floors are §2.4 and §5 only.
- Do not ship a `run --negation` flag until the naming audit
  against USAGE passes (precedent: the payload round's reserved
  "payload" sense; a grep of USAGE finds no existing "negation"
  use, but the audit itself is a build-cycle check, not a
  conclusion of this doc).
- Building, if it happens, is a separate cycle, on this doc —
  and only on this doc's frozen items. A build that deviates
  from any frozen item needs fresh design review first.
