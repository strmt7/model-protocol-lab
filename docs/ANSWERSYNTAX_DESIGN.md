# Answer-syntax probe — design round (ADJUDICATED: NO-BUILD)

## Round summary (coordinator #39, design round only — nothing built)

- Tree at main `f31e72e1ace4` (cycle 212 closed green, 8,439 tests,
  precise coverage 95.3455%). This round was READ-ONLY except for
  this doc and the briefs; no suite was run, no push was made, no
  tag was cut or discussed.
- Specialists (blind to each other): researcher
  (`~/workspace/agents_out/answersyntax/research.md`), red-team A
  (`redteam_a.md`), red-team B (`redteam_b.md`), verifier/claims
  auditor (`verifier.md`). All four briefs landed; the verifier
  resolved every externally checkable claim against primary
  sources, re-ran every surviving PoC script, and recomputed all
  decision math with its own simulator. The coordinator
  re-executed the load-bearing claims itself (§4).
- Convergence and divergence, in one paragraph: all three blind
  briefs return NO-BUILD for the candidate as specified in the
  circulated sketch (three syntax arms, JSON-vs-YAML endpoint
  contrast, field-order's decision rule transplanted unchanged),
  and the verifier upholds that unanimously. The researcher
  additionally specifies an amended instrument (JSON-vs-XML
  headline endpoints, YAML descriptive only, neutral schema
  description, re-derived 13-spec paired pool, parsed-value
  headline scorer) and recommends BUILD for it; the verifier
  upholds that instrument as-amended, subject to corrections
  (§5), and flags that the amendments change the endpoint pair,
  the paired population, and the headline scorer. The
  coordinator's verdict is **NO-BUILD** — for the sketch, and as
  a build order for the amended instrument in this round — for
  the procedural reason stated in §2: under this program's own
  adjudicated distinction (payload round vs negation round), the
  amended instrument is a *resubmission*, and a resubmission is
  reviewed as a new design before it is built. §3 records it as
  the resubmission candidate in full. No fabricated citation or
  symbol was found in any brief this round; the failures found
  were one wrong p-value, several unreproducible secondary
  figures, and sloppy line ranges, all handled in §5.

## 1. Why NO-BUILD: the fatal defects of the candidate as specified

Each defect below is individually blocking and was established by
execution (attribution: A = red-team A, B = red-team B, V =
verifier, C = coordinator).

### F1 — Population and scorer are both undefined (rank-2 F1/F4, inherited)

The claim names "the shipped flat structured-output spec pool"
(9 specs, `structured.py:75-223`); the budget names field-order's
pool (15 specs, 10 paired). The claim, the budget, and the
decision rule therefore have three different denominators. Worse,
"the answer field's exact-value accuracy" has two readings in the
tree — byte-exact (`core.py:797`: `exact_match = status ==
"completed" and answer == case.expected`) and field-order's
parsed-value digest comparison — and they give opposite verdicts
on identical behaviour (B, executed with stub policies: a
JSON-only policy with always-correct values scores endpoint
10/10 vs 0/10 under byte-exact, 10/10 vs 10/10 under
parsed-value; a perfect policy with reversed YAML key order
scores 0/10 byte-exact). A design whose headline flips with an
unmade scorer choice does not have a headline.

### F2 — The conditioning makes the adherence half of the claim unfalsifiable (new fatal)

The sketch's only decision rule conditions the value contrast on
specs adherent under both endpoint arms, yet adherence-rate
equality is half of the claim sentence — and no rule anywhere in
the sketch tests the adherence contrast. Executed (B,
deterministic): a candidate with JSON adherence 10/10 and YAML
adherence 5/10 — a 50-point adherence gap — leaves a conditioned
pool of 5, below the floor of 6, so the verdict is `insufficient`
by construction; Monte Carlo over adherence gaps of 0.95 vs 0.55
falsifies 0.000–0.013 of the time and returns `insufficient`
0.977–1.000 of the time. The conditioning also biases the value
contrast it does test toward the null (C, executed: with a true
0.60 value gap on 40% of a 10-spec pool, mean discordant specs
fall from 3.72 unconditioned to 1.74 conditioned — the floor of
6 is unreachable in that world). Outcome-dependent exclusion of
exactly the cases the treatment moves is the payload round's F4
lesson, here at the level of the claim itself.

### F3 — The XML arm can never pass the tree's schema checker (new fatal)

XML parsers produce strings for every scalar. The tree's checker
is type-exact (`structured.check_schema`, `structured.py:500`;
`_field_value_ok`, `structured.py:529-565`: an integer field
requires `type(value) is int`). Executed (B; C independently):
all 10 of field-order's paired specs contain at least one
non-string field, so the only document an XML parser can produce
for a *perfect* model answer returns `schema_failed` on 10/10
paired specs. Scoring XML at all requires a schema-directed
coercion layer that does not exist in the tree; the sketch
neither names nor freezes one, and its choices (whitespace,
`"36.0"`, `"yes"`) would be the headline. That is rank-2 F2
arriving through the scorer: "same value" across syntaxes is
exactly what has to be stipulated, and the sketch does not
stipulate it.

### F4 — The YAML endpoint rests on ground the rank-2 round already buried (rank-2 F6)

Zero runtime dependencies (`pyproject.toml:28`,
`dependencies = []`); no stdlib YAML parser exists. PyYAML
imports in this VM only as an environment accident — executed
(C): version 6.0.3 in the project venv and 6.0.1 in the system
interpreter, two different accidents, neither declared. Both
parser choices break a claim (B, executed): JSON is a subset of
YAML flow style, so a JSON echo is adherent and correct under a
real YAML parser and `parse_failed` under a block-only subset
parser — the "YAML adherence rate" is a property of the parser
choice, not of the model. And a real parser imports YAML 1.1's
coercion folklore into the measurement (executed by A, B, and C:
`yes`/`on` → `True`; `no` in a *string* field → `False`;
`0123` → 83 (octal); `0x24` → 36 (hex); `500,000` → the string
`"500,000"`; duplicate keys silently last-wins, with no hook to
express field-order's duplicate-key pairing condition). The
sketch supplies no reckoning with any of this; rank-2 F6 already
held that a hand-rolled mirror parser is new untrusted code
whose choices become hidden treatments, and that holding was
not answered.

### F5 — The transplanted decision rule is borrowed, and near-inert on its own panel

The binding constraint of this program is that floors are
derived for the panel, never borrowed (the negation round
removed borrowed (6,2) floors that fired on a random labeller
58% of the time). The sketch transplants field-order's
(6, sign-test) rule unchanged and prints none of this panel's
operating characteristics. Executed (A, B, C, V — independent
code, reconciled by V): the rule's false-fire rate is fine
(exact 0.9% on a random labeller, 10-spec pool, adherence 1.0 —
C), but its power at the canonical gap field-order itself
advertises (0.90 vs 0.60) is 3.2% through the full conditioned
pipeline (A), 5.1% at adherence 0.95 (B), 7.4–7.7% at adherence
1.0 (B, C, V). On the shipped pool alone (9 specs, 4 paired
under field-order's flags) the floor of 6 discordant specs is
unreachable by construction (C): the instrument can only ever
answer `insufficient`. An instrument that returns its withheld
label 82–97% of the time across the plausible world will be
read — whatever its docs say — as an acquittal machine.

### F6 — The congruence dilemma: the schema description bundles the treatment (rank-2 F3)

`structured.describe_schema` (`structured.py:403-443`) renders a
JSON-shaped description — executed verbatim (V):
`{"age": integer (>= 0); "name": string} (required: age, name;
no other keys allowed)`. Hold it constant across arms and every
non-JSON arm is depiction-incongruent (a JSON-shaped schema
inside a YAML/XML demand); re-render it per arm and the
instruction's shape, length, and escaping move together with
the treatment. The sketch mentions neither horn. The arm-length
bundle is likewise real and systematic (C, executed on the
shipped 9 specs with an `<answer>`-root XML renderer: expected
documents run YAML 0.65–0.79× and XML 1.57–2.12× the JSON
length; the researcher's `<response>`-root renderer gives XML
1.49–2.35× across the 15 — quote lengths only with the
renderer named, §5 R-11). A paired contrast over these arms
attributes a flip to "syntax" only if the claim is worded as
discordance between named frozen prompt+rendering pairs; the
sketch's claim is worded as syntax sameness in the abstract.

### F7 — The XML surface is unfrozen, and its parser surface is unsafe as shipped defaults

Attribute-form and element-form documents encoding identical
content score `{}` under each other's parsers (A and B,
executed); namespaced answers yield `{urn:x}`-style keys and
fail schema; no attribute-vs-element ruling, namespace policy,
whitespace policy, DTD policy, or size cap is named in the
sketch. The stdlib parser the build would reach for,
`xml.etree.ElementTree`, *expands* internal entities on
untrusted model output (C, executed: a depth-4 nested-entity
document expands to 10,000 chars; A, executed and
V-reproduced: expanded lengths 7,290 / 65,610 / 590,490 chars
at increasing depth), while external entities raise
`ParseError`. ET alone is therefore not a safety story for
parsing model output; a frozen pre-parse screen is required,
and even behind such a screen the ET call needs a bandit
disposition at build time (B314 fires regardless — V, §5).

### Non-fatals, adjudicated

- **Multiplicity.** Per-candidate decisions at 24 candidates
  compound: B executed P(≥1 falsified candidate) ≈ 0.197 on
  pure noise for the sketch's panel (and ≈0.48 if all three
  pairs are read as decisions). Any successor design must
  print its panel-wise expectation beside the verdicts; the
  amended instrument does (0.36 expected false falsifications
  per 24-candidate panel at its executed random rate).
- **Naming.** The sketch performs no naming audit; "syntax
  adherence" is already owned by the tool-call probe
  (`toolcall.py:1`; `cli.py:374`). The researcher's in-design
  audit (§3, item 8) resolves this and is carried into the
  resubmission specification.
- **Overlap.** `structured` owns JSON adherence, `fieldorder`
  owns value contrasts within JSON, `toolcall` owns its call
  notation. The residue — demanded document syntax as a
  treatment — is real but thin relative to the slate's largest
  build; that is a value judgment the fresh review of §3 must
  make with the power question (§3, item 6a) in front of it,
  not a fatal in itself.
- **ID-divergence discipline** is unstated in the sketch; the
  amended instrument states it (reported, not gated, with a
  rendering-anomaly exclusion) and the resubmission inherits
  that text.

## 2. Why the amended instrument is a resubmission, not an amendment

The negation round's BUILD verdict rested on a stated test:
every defect was corrected *inside the adjudication doc*, by
amendment to the *same* instrument — same population, same
arms, same scorer, same joint-rule shape — whose population and
arms were already frozen and executable before the round began.
The payload round's NO-BUILD rested on the complementary test:
where the amendments replace an undefined instrument with a
different one, the result is a resubmission and "must be
reviewed as a new design, not built as 'this probe with
amendments'".

Measured against that test, the researcher's amended instrument
changes: the endpoint pair (JSON–YAML → JSON–XML); the YAML
arm's role (headline endpoint → descriptive only, behind a new
schema-directed subset decoder); the paired population (10 →
13, under a new pairing derivation); the headline scorer
(byte-exact/adherence conjunct → parsed answer-field value
accuracy, with adherence demoted to descriptive rates); and the
prompt's schema text (the shipped `describe_schema` → a neutral
field list). The sketch's instrument was never frozen and
executable in the first place — its YAML arm has no parser, its
scorer is ambiguous between opposite-verdict readings (F1), and
its population is self-contradictory (F1). This is the payload
configuration, not the negation one.

The verdict is therefore NO-BUILD for this round. This is
explicitly **not** a finding that the amended instrument is
unsound: the verifier upheld it as-amended (subject to §5's
corrections), it answers every fatal in §1 by construction
rather than by disclosure alone, and measured against red-team
B's nine resubmission conditions it satisfies 1, 2, 4, 5, 6, 7,
8 and the naming half of 9 in full. What it has never faced is
a blind red team of its own: its novel machinery — the §3.4
coercion table, the neutral description renderer, the YAML
subset grammar, the XML screen-and-decoder rules, the
13-spec pairing derivation — was claim-verified and
math-verified in this round, not adversarially attacked. The
program does not build its largest instrument on verification
alone.

**Express narrowing accepted by the coordinator** (B's
condition 3, decision half): the candidate's claim is narrowed
to answer-field *value* invariance between the JSON and XML
endpoints. Adherence is reported as per-arm descriptive rates
with denominators and carries **no** decision rule in the
resubmission. Any future design that reintroduces an adherence
falsification claim must supply an *unconditioned* adherence
decision rule with its own derived operating characteristics;
it may not reattach adherence to the conditioned value rule.

## 3. Resubmission specification (NOT a build order)

A resubmission is reviewable as a NEW design only if it
satisfies all of the following in full. Partial compliance is
still NO-BUILD. The researcher's amended instrument
(`~/workspace/agents_out/answersyntax/research.md` §1–§6, with
the frozen items of its §11) is the resubmission candidate and
is incorporated by reference, subject to items 1–5 below; the
fresh review's job is to attack the frozen machinery itself
(the coercion table, the neutral renderer, the subset grammar,
the decoder rules, the pairing derivation), not to re-litigate
the sketch, which is dead.

1. **Corrections applied before review.** The XML length ratio
   is 1.49–2.35× across the 15 specs (§5 R-6), quoted only with
   its renderer named (R-11); the YAML decoder reading is the
   brief's poc2 reading — line grammar plus schema-directed
   coercion, with the plain-safe grammar binding the *emitter*
   only (R-8); tree line ranges as corrected in §5 (R-7, R-9,
   R-10).
2. **Ratifications recorded.** (i) The `document_kind` rename
   (`object` → `document`) for the two new arms is accepted as
   a disclosed vocabulary adaptation; the JSON arm keeps the
   shipped set. (ii) The adherence narrowing of §2 is accepted
   expressly. (iii) The build must carry a bandit disposition
   for the `ElementTree.fromstring` call (it is flagged even
   behind the frozen `<!` screen): an in-code justification
   following the tree's `# nosec` precedent, or an equivalent
   restructure; the screen, not the parser, is the safety
   mechanism and must be described as such. (iv) The neutral
   field list is named identically in the design's prompt
   template and its worked examples.
3. **The voiding rule carries verbatim in substance:** the
   resubmission's frozen items (arms and roles, prompt template
   and neutral description, population and pairing, renderers
   and decoders and coercion table, the three readouts with
   parsed answer-field accuracy as sole headline, the decision
   rule and its printed operating characteristics, the §6.3
   vocabulary of the brief, counts and caps, digest-only
   records, the validator mirror) are frozen; deviation from
   any one voids the design and requires fresh review.
4. **Monday-morning utility (narrowed instrument):** a team
   exposing a prompt-only extraction endpoint uses this, per
   candidate (checkpoint × chat template), to decide whether to
   standardise the endpoint's answer document on JSON or XML —
   and to see YAML subset adherence descriptively before
   considering it — instead of choosing JSON by habit. Where
   the verdict is `insufficient` or `not_falsified`, the honest
   deliverable is "this panel cannot distinguish them; choose
   on other grounds", and the design must say so in those
   words' substance.
5. **Name:** `--document-syntax` (module `documentsyntax.py`,
   ref marker `documentsyntax/<id>`), per the brief's in-design
   naming audit — `toolcall` owns "syntax adherence"
   (`toolcall.py:1`, `cli.py:374`) and `--answer-syntax`
   collides with it and with numform's answer forms. The name
   ships only after the build cycle's USAGE naming audit
   (the rank-2 §8.46 precedent) regardless.
6. **What the fresh review must settle head-on** (silence on
   any of these is NO-BUILD):
   a. **Panel power.** The amended panel (13 paired,
      adherence-gated) falsifies a true 0.90-vs-0.60 gap only
      ~14% of the time (researcher E3: 0.136; verifier's
      independent simulator: 0.1385) and a 0.90-vs-0.50 gap
      ~30%. The resubmission must either accept this with the
      operating characteristics printed beside every verdict,
      enlarge the entailed pool with generated specs (each
      carrying a recorded reader-entailment derivation, per
      the FIELDORDER_DESIGN F1 discipline), or add replicated
      generations per (spec, arm) — and argue the choice
      against the §3.4 decision, not default to it.
   b. **The YAML descriptive arm's cost.** It is a third
      decoder plus a full validator mirror, bought for a
      subset-adherence rate that must be labelled "YAML
      (frozen block subset)" wherever it appears and can never
      be quoted as YAML adherence generally. Keep it, or cut
      v1 to the two headline endpoints; either is reviewable.
   c. **Family-wise expectation.** Printed beside the
      per-candidate verdicts: at the amended panel's executed
      random-labeller rate (0.0151), a full 24-candidate panel
      expects ≈0.36 false falsifications, P(≥1) ≈ 0.31. No
      pooling of discordance across candidates, ever.
7. **Counts (as frozen in the brief):** 15 specs × 3 arms × 24
   candidates = 1,080 generations maximum (720 on headline
   endpoints); records are digests and decoder commitments
   only; the diff validator re-implements — never imports —
   the neutral renderer, all renderers, all decoders, and the
   coercion table, and recomputes every aggregate.
8. **Sources** are limited to §6 of this doc, with their
   fencing. No "first" is claimed anywhere; the contribution,
   if the resubmission is ever built, is the combination
   stated in the brief's §7.

## 4. Adjudication evidence (coordinator's own execution)

All snippets read-only; no suite was run. Scripts in
`/tmp/asyn/` during the round.

- Byte-exact verdict: `core.py:797` confirmed in the
  Observation construction (`exact_match = status ==
  "completed" and answer == case.expected`); `check_schema`
  type-exactness read at `structured.py:500-526` and
  `_field_value_ok` at `529-565`.
- Paired-pool type audit: all 10 `paired: True` specs in
  `fieldorder.FIELD_ORDER_SPECS` contain at least one
  non-string field (integer or boolean) — a perfect XML answer
  fails the shipped checker on 10/10. Field types across all
  15 specs: string / integer / boolean only (no `number`, no
  `null`), confirming the brief's eligibility premise.
- Arm rendering lengths (shipped 9 specs; canonical JSON vs a
  block-YAML rendering vs an `<answer>`-root XML rendering):
  YAML 0.65–0.79× JSON, XML 1.57–2.12× JSON, monotone in every
  spec; no two arms' expected bytes coincide on any spec.
- XML parser behaviour (stdlib `xml.etree.ElementTree`):
  every scalar parses as `str` (`"36"`, `"true"`, `"500000"`);
  a depth-4 nested-entity document expands to 10,000 chars
  (entity expansion is live); undefined external entity and
  malformed input raise `ParseError`.
- YAML behaviour (PyYAML, present only as an undeclared
  accident — 6.0.3 in the project venv, 6.0.1 in the system
  interpreter): `ack_required: yes`/`on` → `True`;
  `language: no` → `False` (a string field silently becomes a
  boolean); `population: 500,000` → the string `"500,000"`;
  `priority: 03` → 3; `age: 0x24` → 36.
- Decision-rule operating characteristics (exact enumeration
  plus 20,000-trial Monte Carlo, 10-spec pool, adherence 1.0,
  unconditioned — the most generous reading of the sketch's
  rule): random-labeller fire rate 0.90% (exact 0.8986%);
  power at 0.90 vs 0.60 = 7.6%; at 0.90 vs 0.30 = 57.7%. On a
  4-spec paired pool (the shipped specs under field-order's
  flags) the floor of 6 is unreachable.
- Conditioning dilution (4,000-candidate simulation): with a
  true 0.60 value gap confined to the 4 hard specs of a 10-spec
  pool and YAML adherence 0.25 on hard specs, mean discordant
  specs fall from 3.72 (unconditioned) to 1.74 (conditioned on
  both-endpoint adherence); conditioned pool size 7.01 of 10.
- Floor derivation check (the tree's own
  `fieldorder.sign_test_p`): unanimous two-sided p is 0.0625
  at 5 discordant and 0.03125 at 6 — a discordant floor of 6
  is forced by α = 0.05 for any sign-test rule on any panel,
  so the amended instrument's floor is derived, not borrowed,
  even though the number coincides with field-order's.
- Schema-description evidence: `describe_schema` output for
  extract-person printed verbatim (§1 F6); keeping it while
  changing only the demand sentence moves prompt length by
  +1/+2 chars — the confound is depiction shape, not length.
- Naming: `document-syntax` / `document_syntax` greps empty
  across the package and `docs/USAGE.md`; `--answer-syntax`
  absent from the CLI flag set; `toolcall.py:1` and `cli.py:374`
  own "syntax adherence".

## 5. Removals applied (verifier §5 — none of these enter any build doc)

- **R-1** (red-team A): "6-1 gives p = 0.219" is REFUTED; the
  tree's `sign_test_p(6, 1)` = 0.125. The substance (critical
  splits are extreme) survives on corrected numbers.
- **R-2** (red-team A): collider/brittle-arm Monte Carlo
  magnitudes (0.533; 0.625→0.833; E[cond] = 7.50) REMOVED as
  executed evidence — no generating model in the brief. The
  qualitative bias survives on red-team B's deterministic PoC
  and the coordinator's simulation (§4).
- **R-3** (red-team A): the "YAML description 32 chars" figure
  and the "cross-arm prompt delta ≤ 8 chars" sentence REMOVED
  — renderers/phrasing absent from the brief. The 84-char JSON
  description is tree-exact and is used in §1.
- **R-4** (red-team A): "defusedxml NOT available" corrected:
  it is importable in the project venv as an undeclared
  accident, exactly like PyYAML; the zero-dependency substance
  (it may not be relied on) survives.
- **R-5** (red-team A): entity-expansion *ratios* quoted
  template-free REMOVED; the expanded lengths (7,290 / 65,610
  / 590,490 chars) are confirmed and used in §1.
- **R-6** (researcher): "XML is 1.7–2.6× the JSON length"
  REFUTED by the brief's own PoC table; the correct range
  across the 15 specs is 1.49–2.35×.
- **R-7** (researcher): `build_structured_fixtures` citation
  corrected to `structured.py:466-488`, `expected=` at :484.
- **R-8** (researcher): the §3.3 decoder sentence subjecting
  plain scalars to the emitter's plain-safe grammar is
  internally inconsistent with the brief's own §2.3/§3.4/E2;
  the poc2 reading is frozen in its place (§3, item 1).
- **R-9** (red-team B): `STRUCTURED_SPECS` range corrected to
  `structured.py:75-223`.
- **R-10** (red-team B): `_field_value_ok` range corrected to
  `structured.py:529-565`.
- **R-11** (red-team B): XML length figures are renderer-bound
  (root `<answer>` vs `<response>` changes the same spec's
  length, 139 vs 143); lengths may be quoted only with the
  renderer named.
- **R-12** (inherited, no brief at fault): field-order's
  printed "~15% power at 0.90-vs-0.60" is UNREPRODUCED under
  the natural independent model at its own panel size
  (verifier: 0.074–0.077) and is retired as a transferable
  figure. Every design in this line prints its own panel's
  derived operating characteristics or none.

## 6. Sources the resubmission may safely cite

All five were resolved against their primary sources this
round by the researcher and independently by the verifier
(abstract level, title and authors confirmed; fencing must
travel with each):

- He et al., arXiv:2411.10541 — *input* prompt-template format
  (plain text / Markdown / JSON / YAML) moves GPT-3.5-turbo by
  up to 40% on code translation, Azure API models only. Not an
  answer-document syntax contrast; no chat-template crossing.
- Tam et al., arXiv:2408.02442 — format *restrictions* degrade
  reasoning vs free-form answers; §4.1's emitted key-order
  observation is constrained-JSON-mode only and is fenced off
  (emitted order is not demanded syntax). Reasoning-task scope:
  a null on this probe's extraction specs neither replicates
  nor contradicts it.
- Sclar et al., arXiv:2310.11324 — abstract only (up to 76
  accuracy points on LLaMA-2-13B; weak cross-model
  correlation). The §3.1 grammar detail and any venue tag
  remain REMOVED per the rank-2 adjudication.
- Zhang et al. (POML), arXiv:2508.13948 — abstract only here;
  §7 Table 3 digits are carried only as verified in the rank-2
  doc §5. POML's styles bundle table syntax with other styling
  dimensions; input tables, not answer documents.
- Geng et al. (JSONSchemaBench), arXiv:2501.10868 —
  structured-output adherence under *constrained-decoding*
  frameworks, JSON only. This candidate is prompt-only; its
  JSON arm is the baseline those frameworks' papers omit.

## 7. What a build cycle must NOT do on the basis of this doc

- Do not build the circulated sketch, any YAML headline
  endpoint, or the transplanted field-order decision rule.
- Do not build the amended instrument on this doc's authority:
  it enters only through a fresh design review (blind red-team
  pair + verifier) of a resubmission satisfying §3 in full, and
  only on a BUILD verdict from that review.
- Do not reintroduce an adherence falsification claim without
  its own unconditioned decision rule and derived operating
  characteristics (§2).
- Do not carry field-order's "~15% power" figure, the removed
  figures of §5, or any arm-length figure without its renderer
  named, into any doc or help text.
- Do not ship the flag under `--answer-syntax`; the
  resubmission's name is `--document-syntax`, and even it
  ships only after the build cycle's USAGE naming audit.

## 8. Fresh review of the §3 resubmission (coordinator #56, 2026-10-28 — ADJUDICATED: NO-BUILD as restated)

Round shape: blind researcher + two blind red teams + coordinator
adjudication by own execution, per the round protocol. Briefs and
the coordinator's verification notes are in
`~/workspace/agents_out/documentsyntax/` (researcher.md,
redteam_a.md, redteam_b.md, coordinator_verification.md).
Researcher: BUILD as written. Red team A (frozen machinery):
NO-BUILD as frozen. Red team B (population/power/naming):
NO-BUILD. Coordinator verdict: **NO-BUILD as restated** — on the
statistics-for-the-decision (§3 item 6a, settled head-on in §8.1),
not on the machinery. The machinery is the closest any
resubmission in this slate has come (§8.2), and every machinery
gap found is a completion a resubmission can fold in (§8.4). The
decisive defect admits no cure inside the amendment taxonomy:
enlarging the panel changes the population, and re-scoping the
utility changes what the instrument claims to decide. Either is
a resubmission, not an amendment — the payload-round test (§7.3
of PAYLOADSERIALIZATION_DESIGN states the same rule).

### 8.1 The three inherited open items, settled with numbers

**(a) Panel power — honest, and insufficient for the decision
§3 item 4 names.** The brief's §5.3 figures were reproduced
exactly by three independent computations (coordinator's exact
trinomial enumeration; red team B's exact enumeration and MC;
researcher's MC): random labeller 0.0154 (exact; brief MC
0.0151), competent no-effect 0.0015, power at adherence 0.95 of
0.1377 at 0.90-vs-0.60 and 0.3024 at 0.90-vs-0.50. The rule is
calibrated and nothing is misadvertised. What the brief did not
print is the full label mix at the headline gap (coordinator,
exact, 0.90-vs-0.60, adherence 0.95): **falsified 0.138 /
insufficient 0.636 / not_falsified 0.227**. Facing a real
30-point gap — a gap that, if known, decides the standardisation
question — the instrument returns silence 64% of the time and
the acquittal-shaped label 23% of the time; the honest
deliverable sentence of §3 item 4 ("this panel cannot
distinguish them; choose on other grounds") is the deliverable
in ≈86% of the worlds where the choice matters. Both cures
named in §3 item 6a fail or exit the taxonomy. Replicated
generations are illusory under the lab's protocol: generation
is greedy (`fieldorder.py:115`), so a replicate is the same
generation, not new information; even red team B's idealised
majority-of-3 model reaches only 0.160. Enlargement works —
exact power at 0.90-vs-0.60, adherence 0.95: n=20 → 0.37–0.40,
n=26 → 0.53–0.56, n=39 → 0.77–0.80 (ranges span the per-arm and
both-adherent readings of adherence; §8.5) — but a ≥26-spec
entailed panel is a new population with new entailment
derivations, i.e. a different instrument. The shipped
field-order precedent does not carry the accept branch: its own
panel power at the same gap is 0.0759 (exact, this round; the
R-12 figure), and field-order is a warning instrument whose
utility never asks the user to choose between two contracts on
its verdict. Every sibling resubmission candidate in the queue
operates at ≥0.40 power at its headline gap (directive §3:
0.5027 at a 25-point drop; prune-sweep §3: 0.40 at 25 points).
**B1 upheld: FATAL to the accept branch.**

Concurring (red team B's B2, upheld in part): the adherence
gate is not a neutral caveat on the decision. At identical
per-spec accuracies (5 hard specs at 0.90-vs-0.50, 8 easy at
0.95/0.95), coupling adherence to hardness cuts the fire rate
≈3× under the brief's both-adherent reading (coordinator MC:
0.0072 → 0.0024) and ≈7× under the per-arm reading (red team
B: 0.0053 → 0.0008) — the gate preferentially deletes exactly
the specs where the treatment bites. The §5.1 claim sentence
names the conditioned population, so the *claim* is true; but
§3 item 4's *decision* cannot be carried by the conditioned
value verdict alone. The mitigation (per-arm adherence rates
and pairing attrition inside the verdict block, deliverable
worded as a joint reading) is an amendment — and it concedes
the point: read jointly, the practical deliverable reduces to
the descriptive adherence block in most decision-relevant
worlds, while the build cost sits in the falsification
machinery (red team B's measured analogues: field-order, the
closest shipped sibling, is ≈2,660 lines before tests;
document-syntax exceeds it on every axis — three decoder
front-ends, a coercion table, an 8-block report, a full
validator mirror — the largest single-probe build in its
class, for the slate's weakest headline statistic).

**(b) YAML descriptive-arm cost — budget trivial, code real,
framing survivable only as a consolidated text.** Counts
verified by all parties: 360 of 1,080 generations (33%); the
headline never touches YAML. The cost is one of three decoders
plus its full diff-validator mirror, buying a rate that can
never enter a decision. The parser problem is real but
confined: the brief's §3.3 literal decoder sentence (plain
scalars subject to the plain-safe grammar) rejects the
canonical emitter output on 12/15 specs — the R-8 poc2 reading
frozen in §3 item 1 is the only consistent reading, and it
lives outside the brief's own text (red team A-N7); the frozen
line grammar scores a single trailing newline, the commonest
generation artifact, as `parse_failed` while both headline
arms tolerate it (executed by the coordinator and red team A —
the span `core.evaluate` returns is unstripped, so the case is
reachable); the quoted-scalar pipeline, the single-quote rule,
and the ` #` comment check each need one ruling (A-N8/N10/N11;
the emitter produces `name: "a #b"`, which its own decoder
rejects — off-pool, but the grammar is self-inconsistent as
written). None of this touches the headline; it is recorded
here because §3 item 6b required the judgement: in any
resubmission, cut YAML from v1 (red team B's recommendation)
or justify the decoder+mirror cost against a named decision
it informs. Not independently decisive for this verdict.

**(c) Family-wise expectation — derived, and the brief's
figures stand.** From the exact per-candidate random-labeller
rate 0.01538: expectation over a 24-candidate panel is
24 × 0.01538 = **0.369** false falsifications, P(≥1) = **0.310**
(the brief's ≈0.36 / ≈0.31 used its MC rate 0.0151 → 0.362 /
0.306; immaterial). Print-don't-correct is defensible against
the shipped precedent — field-order prints ≈1.2 expected at
nominal α with P(≥1) ≈ 0.71 (FIELDORDER_DESIGN.md:385-386)
and shipped — and correction is not a real alternative at this
panel (red team B, executed: Bonferroni power 0.0010 at the
headline gap). The surviving defect is placement: the
family-wise sentence lives in the OC paragraph, not attached
to an individual falsified verdict, which is the result a
reader acts on. Amendment-class; recorded in §8.4. **Not
fatal.**

### 8.2 What survived (executed, not argued)

- **Round-trip 45/45.** The coordinator implemented the brief's
  §3.2/§3.3/§3.4 machinery from the text and fed all 15 specs ×
  3 arms' canonical renderings through the decoders +
  reconstruction via the shipped `fieldorder.schema_ok` /
  `structured.check_schema` and `fieldorder.field_correct`:
  every spec reconstructs with `schema_ok` and every field
  correct. A perfect XML answer passes the tree's type-exact
  checker through the coercion layer — the original round's F3
  does not recur. Red team A (15/15 × 3) and the researcher
  independently replicate. Expected renderings are pairwise
  byte-distinct on all 15 specs; brief E1 lengths reproduce
  exactly (35/26/59 chars; XML/JSON ratio 1.49–2.35).
- **Neutral description is information-equal** to shipped
  `describe_schema` on all 15 schemas (coordinator; red team A
  reproduces the §1.3 worked example byte-identically). The
  brief's frozen prompts differ across arms only in the demand
  phrase (executed).
- **Population and pairing are exactly derived.** Pool field
  types ⊆ {string, integer, boolean} on all 15; the paired 13
  are the pool minus FIELDORDER_DESIGN's two non-entailed specs
  (route-alert, route-order); the three 1-field specs' order
  degeneracy does not transfer across syntaxes.
- **Budget is the cheapest in its class:** 1,080 generations
  maximum, 720 headline — below shipped field-order's 1,440.
- **Naming audit clean** (coordinator and red team B,
  independently): `document-syntax`/`documentsyntax` occur
  nowhere in the package or USAGE; no `--document-*` incumbent;
  toolcall owns "syntax adherence" as a different phrase.
  Near-collisions (the `classify-document` spec id; `--structured`
  as semantic neighbour) are help-text matters, recorded for any
  future build's USAGE audit.
- **The residue is real** (all parties): no shipped probe varies
  the demanded answer document's syntax. Duplication is not a
  finding against this candidate.

### 8.3 Machinery findings (red team A, adjudicated)

Three undecided-or-unrepresentable cases in the headline XML
decoder, all adherence-moving, all confirmed by the
coordinator's own independent implementation (found before
A's brief was read):

- **A-F1 (mixed content).** §3.2 step (4) mandates
  `schema_failed` for non-whitespace root text / child tails,
  but the frozen commitment shape {kind, field_values,
  duplicate_keys, document_digest} has no slot that records
  them; a from-the-text implementation returns `adherent`
  (executed by A; the coordinator's implementation needed an
  out-of-shape side channel to honour the mandate). The stage
  and the §6.4 validator mirror can silently diverge.
- **A-F2 (child attributes).** The text freezes attributes at
  the root only; `<age unit="years">36</age>` decodes adherent
  under the silent reading. Undecided by the frozen text.
- **A-F3 (empty elements).** ElementTree gives `text=None` for
  `<name/>`; the brief never rules on it. Readings diverge
  (`schema_failed` vs adherent-with-wrong-value) on string
  fields — the answer field on all 15 specs.

Plus twelve non-fatal completions (A-N1–N12): the `<!` screen
also kills CDATA and does not touch processing instructions
(undisclosed both ways); §3.4's `int()` raises on >4,300-digit
lexicals, so the coercion is not total as written (coordinator
re-executed: ValueError, grammar full-matches; cap-bounded
live, but the mirror decodes off the live path); duplicate
winner unspecified in both new decoders (JSON's shipped rule
is last-wins); coercion-failure encoding in `field_values`
unspecified (verdicts coincide, `document_digest` diverges);
the neutral renderer's `additionalProperties: True` sentence
and enum/bounds annotation order unfrozen (off-pool); the
parse-vs-schema decomposition asymmetry across arms for
identical lexicals (outcome-symmetric, undisclosed); the YAML
items of §8.1(b); XML character references expand before
"verbatim" text is read (undisclosed; the mirror must use ET
to reproduce it). And red team B's B9: the §2.2(b)/§5.1
pairing sentences omit the live-record condition (boundary /
detail / screening) that field-order's pairing cascade
carries — the §4.3 transplant implies it; the frozen pairing
text must state it. B4: route-complaint's entailment is a
routing convention (its context names front-line support, not
technical), to be recorded as such, and a 1-field-stratum-
confined effect is undetectable by construction (at most 3
discordants against a floor of 6; executed fire 0.013 at a
90-point stratum-confined gap) — the §5.5 split must print
that limit. B3: the shape-correlation calibration caveat
field-order's design prints (FIELDORDER_DESIGN.md:380-383)
is omitted here, and §5.3's "well under α" is not
unconditional — at full within-shape direction correlation the
random-labeller fire is 0.161 (B) / 0.200 (coordinator's crude
construction), ≈10× the printed iid rate.

**Every item in this section is a completion that touches
neither the endpoints, the population, nor the headline
scorer** — amendment-class in isolation, and §8.4 lists them
as obligations for any resubmission. They are not the grounds
of this verdict: §8.1(a) is.

### 8.4 Obligations folded into any resubmission of this instrument

1. A-F1's commitment encoding for root-text/tail violations;
   A-F2's child-attribute ruling; A-F3's `text is None` ruling
   per type (red team A's recommendation — no value ⇒
   `schema_failed` — matches the integer/boolean readings).
2. The one-line completions: integer digit cap / ValueError ⇒
   coercion failure; last-wins duplicates in both new
   decoders; coercion-failure encoding (the raw-string
   canonical, mirroring JSON's `{"age": "36"}`); PI/declaration
   acceptance, CDATA collateral, and character-reference
   expansion stated in §3.2; the neutral renderer's
   additional-properties-True sentence and annotation order;
   the parse/schema asymmetry note.
3. The YAML decoder as one self-consistent text in the brief
   itself (poc2 folded in; quoted-scalar pipeline ruled;
   single-quote rule stated as a line-grammar rule or dropped;
   ` #` scoped outside quoted scalars; trailing-newline rule
   stated either way) — or the arm cut per §8.1(b).
4. The live-record condition stated in the pairing sentences;
   route-complaint's entailment recorded as a routing
   convention.
5. The shape-correlation caveat and the 1-field stratum
   detectability limit printed beside the operating
   characteristics; the family-wise sentence attached to any
   individual falsified verdict block, not only to the panel
   paragraph.
6. Per-arm adherence rates and pairing attrition carried
   inside the verdict block, with the deliverable worded as a
   joint adherence+value reading (B2's cure (i)).

### 8.5 Fabrication sweep — CLEAN

- All five §6 arXiv sources re-resolved by the coordinator
  against the arXiv abstract pages on 2026-10-28 and
  independently by the researcher: 2411.10541, 2408.02442,
  2310.11324, 2508.13948, 2501.10868 — authors, titles, and
  fenced magnitudes match the abstracts. No invented citation.
- Tree citations spot-checked against the current tree by the
  coordinator: core.py:797; structured.py:68/336/403–443/500/
  529–565/568/466–488 (R-7 governs the brief's stale range);
  fieldorder.py:73/88/95/99/170/571/635/649/694 and :115
  (greedy); FIELDORDER_DESIGN.md:380–386 (both precedent
  passages red team B cites); diffreports.py:13559 (mirror
  span start); toolcall.py:1; pyproject.toml zero dependencies.
  No invented symbol.
- Researcher's brief: every load-bearing tree claim
  self-verified in its §2 table; its two declared UNVERIFIED
  items (brief E2 magnitudes, §5.3 non-headline cells) rest on
  the original round's coordinator executions (§4) and this
  round's E3 cells respectively.
- Red team A: A-N2 re-executed by the coordinator (confirmed);
  A-F1/F2/F3 match the coordinator's independently found gaps;
  A's round-trip and renderer results replicate the
  coordinator's.
- Red team B: n=13 statistics exact under the brief's model.
  Its off-panel figures (enlargement, n=12, coupled worlds)
  are exact under a per-arm reading of "adherence 0.95" —
  identified by the coordinator: at both-adherent 0.9025 the
  coordinator's enumeration reproduces B's 0.532/0.370/0.772/
  0.1057 to four decimals. A presentational inconsistency in
  the brief (its n=13 runs use the brief's both-adherent
  reading); immaterial — under the brief's reading the
  enlargement helps slightly more, and the coupling direction
  holds under both readings. No fabrication.

### 8.6 The restated candidate (NOT a build order; NOT queued ahead of anything)

A future resubmission of this instrument must settle, in its
own text:

1. **The panel question (§8.1a).** Either an enlarged
   reader-entailed paired panel — ≥26 specs, each with a
   recorded entailment derivation in the FIELDORDER_DESIGN F1
   discipline, budget and operating characteristics
   re-derived at the new panel — or an explicitly re-scoped
   utility (e.g. screening for very large discordance) with
   the build cost re-argued against the smaller claim. The
   13-spec panel with the standardisation decision as stated
   does not return.
2. The verdict-block joint reading (§8.4 item 6).
3. Every obligation in §8.4, folded into the candidate's own
   text (not incorporated by reference to this section).
4. The YAML arm kept only with its consolidated decoder text
   and a named decision it informs, or cut.

### 8.7 What a build cycle must NOT do on the basis of this section

- Do not build §3 as restated, on this section's authority or
  the researcher's BUILD recommendation.
- Do not treat §8.4's completions as a buildable amendment
  set: they complete an instrument whose panel/utility
  question (§8.6 item 1) is unsettled, and building it would
  freeze a 0.138-power headline behind the slate's largest
  build.
- Do not enlarge the panel by re-using the two non-entailed
  specs, relaxing the entailment discipline, or borrowing
  specs whose derivations are conventions without recording
  them as such.
- Do not reintroduce an adherence falsification claim, a YAML
  headline endpoint, or the `--answer-syntax` name (§7 stands).
