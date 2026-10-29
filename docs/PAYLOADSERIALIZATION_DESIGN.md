# Payload-serialization probe — design round (ADJUDICATED: NO-BUILD)

## Round summary (coordinator #35, design round only — nothing built)

- Tree at main `960b70dcde87` (cycle 211 closed green, 8,402 tests, per the
  round brief). This round was READ-ONLY except for this doc and the briefs;
  no suite was run, no push was made, no tag was cut or discussed.
- Specialists (blind to each other): researcher
  (`~/workspace/agents_out/payloadformat/research.md`), red-team A
  (`redteam_a.md`), red-team B (`redteam_b.md`), verifier/claims auditor
  (`verifier.md`). All four briefs landed; the verifier resolved every
  externally checkable claim against primary sources and spot-checked the
  load-bearing tree citations.
- Convergence: both red teams independently returned NO-BUILD, and the
  verifier upheld both verdicts. The coordinator re-executed the
  load-bearing claims itself (see "Adjudication evidence" below) and
  concurs.
- **Verdict: NO-BUILD** — for the candidate as tentatively framed
  (`run --payload-format`: JSON vs plain text vs other encodings of "the
  same content", verdict flips as headline) AND for the researcher's
  design as written. The reasons are not implementation gaps; they are
  defects in the defining claim (see §1). §2 specifies what a
  resubmission would have to satisfy in full. A resubmission satisfying
  §2 is a DIFFERENT, narrower instrument and must be reviewed as a new
  design, not built as "this probe with amendments". Accepting the
  amendments while keeping the original causal claim does not count as
  accepting them.

## 1. Why NO-BUILD: the fatal defects (each individually blocking)

### F1 — "The task payload" is undefined, and the tree gives it incompatible meanings

Placement's payload is a prose fact passage (`placement.py:267-283`
`arm_content`); a structured spec's context is prose
(`structured.py:445-458` `render_prompt`); a structured spec's answer
document is canonical JSON (`structured.py:463-480`
`build_structured_fixtures`); shipped user fixtures are instructions, not
data (`examples/fixtures.json`). No module defines a serializable
task-data record type. A design that leaves the choice to the builder
does not have a population, a scorer, or a headline — underspecification
at this level is itself a build-blocker.

### F2 — "The same content" is false across the named encodings

Typing, loss, and parser-dependence change the content before any model
runs. Executed in adjudication: `json.loads('{"a": 1, "a": 2}')` gives
`{'a': 2}` (last-wins); JSON `349` is `int` while plain-text `349` is
`str`; JSON `true`/`null` are boolean/null while plain-text `true`/`null`
are strings; CSV flattens nesting; XML must choose attributes vs
elements. There is no canonical-equivalence theorem for JSON-vs-prose
comparable to canonical-form's UAX #15 basis, and prose has no parser to
be equivalent under. A content-preserving claim needs, per arm, a frozen
byte-exact renderer, a frozen key order, and a lossless value
restriction — which the candidate as framed does not supply.

### F3 — "Serialization" bundles at least six simultaneous treatments

Token length, delimiters/punctuation, quoting/escaping, key order,
whitespace, and output priming all move together. Key order is already
a shipped probe (`run --field-order`); whitespace is `ws_canonical`'s
territory; length/position are needle/distraction/density/placement
axes. No paired contrast over such arms attributes a flip to
"serialization" rather than to the bundle. The only supportable claim
is descriptive pairwise discordance between named frozen renderings,
with length and priming stated as bundled, never a causal
"serialization changes verdicts" headline.

### F4 — The byte-exact scorer × output-format priming manufactures and hides flips

The verdict is byte-exact with no answer stripping:
`core.py:797` (`exact_match = status == "completed" and answer ==
case.expected`), conjoined with boundary presence in stage cells
(`experiments.py:9515` pattern). A model answering correctly in the
input arm's idiom (`{"code": "amber-1234"}` on the JSON arm) scores
False against a bare expected value; a model failing to parse the JSON
arm but guessing the bare code scores True on both. The fixes are
closed: parse-then-compare redefines "verdict" relative to every sibling
probe and the diff validator's digest identity (the fix canonical-form
rejected); outcome-dependent exclusion deflates discordance exactly as
field-order's F6 lesson discloses; the scorer-lens ladder has no
label-strip operation and cannot rescue JSON-echo answers. The
researcher's echo diagnostic beside the headline is necessary but does
not cure this: it explains flips, it never rescues them, and the
headline still bundles comprehension with priming.

### F5 — Degenerate populations and arms

Single-field cases collapse to punctuation differences (executed:
scalar JSON `42` is byte-identical to plain `42`; single-field JSON
differs from key-value lines only by braces, colon, and quotes).
Multi-field populations are already partly owned: field-order owns key
order over 15 specs, structured owns output adherence over 9 specs
(1–2 fields only — not 1–4, as red-team A misstated; see §4).
Whitespace-only JSON variants (compact vs spaced vs pretty) are a
whitespace probe under a serialization name. The researcher's
multi-record table population is the right *shape* of answer to F5, but
it does not cure F1–F4, and its arm set and headline do not survive
(see §2).

### F6 — The researcher's design as written does not survive its own amendments

Credit where due — the researcher already supplies a frozen multi-record
population with unique keys and distinct values, frozen field order, a
value alphabet plus a four-way byte-divergence assert, and an echo
diagnostic beside the headline. What still kills it as written:

- The YAML arm must go in v1: zero runtime dependencies
  (`pyproject.toml:28`, `dependencies = []`); no stdlib YAML parser;
  YAML 6.0.1 importing in this VM is an environment accident, not a
  declared dependency; a hand-rolled mirror parser is new untrusted
  code whose quoting choices become hidden treatments. TOML has no
  stdlib writer (`tomllib` has no `dumps`, executed); XML escaping
  rewrites visible text. V1 arms must be stdlib-serializable frozen
  renderings only.
- The falsification headline must be downgraded from causal to
  descriptive pairwise discordance with ONE pre-registered primary pair;
  the researcher's prose-vs-rest framing plus all-pairs matrix, with
  placement's floors borrowed without operating characteristics for a
  four-arm panel, does not survive the k(k−1)/2 multiplicity problem
  (10 pairs at 5 arms).
- The prose "control" framing must go: prose is one encoding among
  others. The researcher says this, then still computes prose-vs-rest
  first. The decision rule must pick one framing before building.
- Missing mandatory machinery: a closed-book screen (placement pattern;
  canonical-form's lack of one is a gap to avoid copying, not a
  precedent); a boundary-only vs exact-only discordance decomposition;
  the full selection-vocabulary transplant; per-arm attrition
  reporting; a frozen fail-closed vocabulary with renderer failure
  failing the whole stage closed.

## 2. Resubmission specification (NOT a build order)

A resubmission is reviewable as a NEW design only if it satisfies ALL
of the following in full. Partial compliance is still NO-BUILD. The
resubmission's working name must reflect the narrowed claim (a
descriptive frozen-renderer discordance probe); do NOT ship
`--payload-format` until the naming audit against USAGE §8.46 passes,
since placement's "payload" sense is reserved.

### Monday-morning utility (narrowed instrument)

A prompt-assembly owner, deciding whether to pin one frozen rendering
(prose or canonical JSON) in their prompt assembly and re-run template
selection under that pinned rendering — before a routine upstream
format change silently flips eval verdicts. The readout answers "these
two frozen renderings discord on N paired cases on this checkpoint",
never "serialization is (in)ert in general", and never anything about
weights.

### Arms (frozen, byte-exact, stdlib-only)

- Exactly one pre-registered primary pair, recommended: prose vs
  canonical JSON as defined by `structured._canonical_json`
  (`structured.py:336-343`: `json.dumps(document, sort_keys=True,
  separators=(", ", ": "), allow_nan=False)`; default
  `ensure_ascii=True`). Executed in adjudication: that rendering is
  byte-unequal to compact `json.dumps` (`'{"age": 36, "name":
  "Ada Lovelace"}'` vs `'{"age":36,"name":"Ada Lovelace"}'`), so the
  canonical form must be pinned by name and by test, never assumed.
- Every arm is a named total function with a byte-exact specification
  and a worked example in the resubmission doc, pinned by a test that
  its output is byte-identical to the doc's example. "Plain text"
  without such a function is not an arm name; it is a placeholder.
  Pretty JSON, flow YAML, Markdown lists, and single-record tables are
  excluded as duplicates, each with the duplicate it duplicates named.
- CSV stays deferred (loss analysis: CSV flattens nesting and drops
  types). XML stays deferred (escaping rewrites visible text; no
  canonical-serialization convention in the tree). YAML stays out of
  v1 entirely (see F6); it may return only as a separately justified
  v2 with a declared dependency or a frozen stdlib-only renderer the
  validator can mirror. No other arms are decision inputs; any further
  arms are descriptive context only.

### Population and controls (the attribution contract)

- Generated cases only, stage-local, never entering the user's
  search/held-out Split (canonical-form invariant 3 pattern; Split
  compares message inputs by NFC key, `core.py:193-214`, so variant
  fixtures would collide — no fixture arm).
- Fresh multi-record tables (4–6 records × 3–4 fields), one key field
  (unique per table), one answer field, 1–2 distractor fields whose
  values are distinct from the answer value in every row. Cases whose
  answer value appears in more than one row, or whose key field is not
  unique, are excluded pre-generation with counts, never silently
  dropped.
- Value restriction (v1, strictest form): flat, string-only, ASCII,
  stem-disjoint, reader-entailed tables. Answer field is a plain
  `word-number`-style code in every arm. No null, no nesting, no
  duplicate keys, no floats, no non-string keys, no YAML-reserved
  words (`yes`, `no`, `on`, `off`, `null`, `~`), no date-like strings,
  no leading-digit strings. Non-answer fields are string-shaped too.
  Integers/booleans are a v2 question with a frozen per-arm
  type-rendering table, not a v1 entitlement.
- Held constant across arms (each a generator/validator assert):
  the same record multiset, field values, key field, and answer field;
  one frozen field order (alphabetical, the
  `structured.describe_schema` convention, `structured.py:403-443`,
  which sorts properties) in JSON keys, YAML keys (v2), and table
  columns alike, and the same sentence field order in prose; a
  byte-identical question and instruction string in the same position
  relative to the table (question after the table in all arms); a
  byte-identical output contract ("Answer with ONLY the value." or
  equivalent, frozen); greedy decoding, the same `--max-new-tokens`,
  the same boundary ids, the same seed, the same template family per
  candidate. Templates are not touched: serialization lives inside
  the user message content only.
- No format label in the instruction unless the label choice itself
  is frozen identically across arms and its confound is stated.
- Closed-book screen arm (placement pattern, `placement.py:374-380`
  `screened()`): a case the candidate passes closed-book is excluded
  from the live populations, with `n_screened` and
  `n_closed_book_incomplete` reported. Without it the headline is
  dilutable by cases serialization cannot move.
- Answer-embedding audit: no field name, key, topic token, or neutral
  sentence shares a 4-character prefix with, or is a prefix of, any
  expected value (placement's stem-disjointness pin, extended to
  keys). Reader-entailment audit per case: a reader given question +
  table must derive the answer uniquely, or the case is
  descriptive-only. If a resubmission instead reuses shipped specs,
  it inherits field-order's `paired: False` flags verbatim — all
  five: route-alert, route-order, classify-sentiment,
  classify-urgency, classify-language (verified by grep: 5
  `paired: False` entries, 10 `paired: True`, 15 specs total).
- Byte-divergence assert across arms per case, with excluded counts;
  boundary-containment screen plus `audit_special_token_literals`
  (`fixtures.py:471`) over every frozen renderer output for every
  family before generation. Screened cases are excluded
  pre-generation with counts, never silently dropped.
- Generation/record/payload caps and the generation count at the
  legal maximum of 24 candidates are frozen in the module with
  derivations and stated in CLI help (field-order precedent: 15
  specs × 4 arms × 24 candidates = 1,440; canonical-form: 2 × 24 ×
  24 = 1,152; placement: 4 × 24 × 24 = 2,304).

### Scorer pins (binding)

- Headline scorer, named before building: byte-exact verdict per
  `core.evaluate` (`core.py:797`). No stripping, no parsing, no
  normalization is added to the headline path for this probe.
- Beside the headline, never rescuing flips into passes: a
  parsed-value / echo-substring diagnostic (per arm, the share of
  byte-exact failures whose decoded answer contains the expected
  value as a substring or as a parsed field value — the echo rate);
  scorer-lens R1 (whitespace trim/collapse, `scorerlens.py:146-153`)
  and fence-strip (`:176-187`) regrade rates as sensitivities; and
  the exact-only vs boundary-only discordance decomposition (a
  boundary-only discordance is a boundary effect, never a
  serialization effect, and cannot falsify any claim).
- Expected values are pure ASCII (canonical-form SCORER PIN
  transplant): decomposable characters live in carrier prose only.
- If byte-exact and parsed-value scorings disagree, the section
  prints both and says which question each answers; it must not pick
  the flattering one post hoc.

### Fail-closed vocabulary (frozen before building)

- Stage details exactly: `{cancelled,
  payload_format_record_budget_exceeded,
  payload_format_screen_failed}`. (The `payload_format_` prefix is
  retained for the stage-detail pattern only; the CLI flag name is
  still subject to the naming audit above.)
- Per-cell details exactly: `{"", arm_context_exceeded,
  arm_evaluation_failed}`.
- Model parse outcomes exactly field-order's `document_kind` set
  (`object` / `non_object` / `unparseable` / `""`), where
  `unparseable` means JSON parse failure of the ANSWER only.
- Model parse failure is scored wrong, never a detail. Any renderer
  failure, serializer exception, or non-UTF-8 payload degrades the
  whole stage to `{"available": False, "detail":
  "payload_format_unavailable"}` — never to a per-cell verdict.
- No new verdict words. In particular: no `arm_parse_failed`, no
  `unparseable` as a cell detail for stage-produced inputs, no
  `payload_too_large`, no per-arm "not applicable".
- Per-arm attrition (`n_context_exceeded`,
  `n_evaluation_failed`) printed beside every paired rate; ungated
  discordance printed as a sensitivity wherever any gating applies.

### Report shape and decision rule

- Headline population: both arms completed AND full-render id
  sequences diverged (canonical-form invariants 9–10). Three-way
  report per pair: ids-identical / divergent-concordant /
  divergent-discordant, with the per-case `id_length_delta` list (in
  case order) and `max_abs_id_delta`. Same-ids verdict flips are
  recorded as determinism findings (`n_same_ids_verdict_flips`),
  never as serialization fragility. A run landing 100% in the
  ids-identical bucket reads "instrument inapplicable to this
  tokenizer", never "the model is robust to serialization".
- Per-candidate classification over the id-divergent paired
  population, with the resubmission's OWN operating
  characteristics printed with the verdict (critical splits, power
  at a stated gap, expected false falsifications across
  candidates for its panel size). Borrowed floors are forbidden:
  placement/canonical-form floors (`MIN_PAIRED_EVALUABLE = 6`,
  `MIN_DISCORDANT = 2`, `placement.py:172-173`) and field-order
  floors (6 discordant specs + sign test) may not be transferred
  to a payload panel without that argument. Below the discordant
  floor the verdict is `insufficient` — withheld, never defaulted
  to acquittal.
- Selection vocabulary transplanted verbatim from canonical-form
  (`canonicalform.py:566-641`): a candidate "passes" an arm iff it
  passes every completed case on that arm; `winner_changed` only
  when both arms select and winners differ;
  `winner_abstention_asymmetric` for exactly one side selecting;
  `winner_undefined` when neither selects or selection is
  withheld (completed sets differ across candidates or are empty).
  Selection-vs-abstention is NEVER a winner change. Per-arm
  accuracy rates sit beside every selection readout; per-arm
  completion and withheld rates are printed. The per-case
  discordance classification is the primary readout; the winner
  headline is secondary and labelled as such. Discordance is never
  pooled across candidates/templates: per-candidate contrasts only.
- Records carry digests only (per-arm content digests, token
  counts, `ids_identical`, `id_length_delta`, cells, details —
  canonical-form record shape). The diff validator regenerates
  every arm's rendering from its own literal renderer copies and
  checks each digest plus pairwise digest inequality across arms,
  plus `ids_identical` implies equal arm token counts. Reports
  render counts/rates only, no payload/question/answer text, every
  payload-derived string through `_escape_text` (`reporting.py:28`),
  never bare `escape()`.

## 3. Adjudication evidence (coordinator's own execution)

All snippets read-only; no suite was run.

- `core.py:797`: `exact_match = status == "completed" and answer ==
  case.expected` — verified exact line; no answer stripping in the
  verdict path (only strip hits are field validation and a
  prefix-trim repair variant, per the verifier's grep).
- `structured._canonical_json` at `structured.py:336-343`:
  `json.dumps(document, sort_keys=True, separators=(", ", ": "),
  allow_nan=False)` — byte-unequal to compact `json.dumps`
  (executed: `'{"age": 36, "name": "Ada Lovelace"}'` vs
  `'{"age":36,"name":"Ada Lovelace"}'`).
- Duplicate-key parse: `json.loads('{"a": 1, "a": 2}')` →
  `{'a': 2}` (last-wins) — executed.
- `tomllib` has no `dumps` (executed: `hasattr(tomllib, 'dumps')`
  is False); YAML 6.0.1 imports in this VM while
  `pyproject.toml:28` declares `dependencies = []` — executed.
- Catalog: 29 `Format(` entries (`catalog.py:26`) — executed.
- Placement floors: `MIN_PAIRED_EVALUABLE = 6`,
  `MIN_DISCORDANT = 2` at `placement.py:172-173` — verified.
- Field-order spec inventory: 15 specs, 10 `paired: True`, 5
  `paired: False` (route-alert, route-order, classify-sentiment,
  classify-urgency, classify-language) — verified by grep over
  `fieldorder.py`.
- Placement `arm_content` at `placement.py:267-283` (prose payload,
  question-position arms) — verified.

## 4. Removals applied (verifier §6 — none of these enter any build doc)

- Researcher S2 §3.1 grammar sentence (descriptors, separators,
  casing functions, itemizers; equivalence by shared rule
  application): REMOVED until re-resolved from the Sclar primary
  text. The Sclar headline (up to 76 accuracy points on
  LLaMA-2-13B, weak cross-model correlation) is verified from the
  abstract and survives; the grammar detail does not.
- Researcher S2 "(ICLR 2024)" venue tag: REMOVED until re-resolved
  (the abstract page does not state it).
- Incorrect line ranges corrected, not carried forward:
  structured canonical JSON 337–343 → 336–343 with
  `allow_nan=False` added; placement 251–279 → 250–283; placement
  50–51 → 50–52; cli 177–926 → 174–938; red-team A's
  `experiments.py:9515-1520` range deleted (cell function
  `_canonical_form_cell` at 9473–9522, verdict line 9515).
- POML Mistral second format corrected to "HTML (Ugly) (0.348)",
  not plain HTML (Table 3 values otherwise verified).
- Red-team A F3/M1/M6 length figures (66/69/76/86/59/99/93) and
  the derived "68% longer" / "25% longer" sentences REMOVED as
  executed evidence: the brief never states the renderers, so the
  numbers are not reproducible from the brief. The length-confound
  argument survives on red-team B's reproducible case (canonical
  JSON 35 chars vs one plain rendering 27 chars, a 29.6% delta)
  and on canonical-form's per-case `id_length_delta` discipline.
- Red-team A F4 "structured spec set … with 1–4 fields" CORRECTED:
  the shipped structured set is 1–2 fields (9 specs); 3–4-field
  specs are field-order's six additions.
- Red-team A M4 prose-selection rendering `'winner': 'A'→ none`
  REMOVED; correct form is `{'decision': 'no_candidate',
  'winner': None}`.
- Red-team B M4 paired-flag sentence CORRECTED to all five
  `paired: False` specs (see §2); B's m1 already handles the
  single-field three, so the combined text must list five.
- Design-level removals: the YAML arm in v1; any causal headline
  of the form "serialization changes verdicts"; prose as "the
  control" / "baseline truth" / "natural format"; any new verdict
  word (see §2); the attribution of `input_tokens` to
  `placement.py` (it lives in `experiments.py` stage cells);
  any borrowed floors without the resubmission's own operating
  characteristics.

## 5. Sources the resubmission may safely cite

- He, Rungta, Koleczek, Sekhon, Wang, Hasan,
  arXiv:2411.10541v1: Abstract (plain text, Markdown, JSON, YAML
  templates; GPT-3.5-turbo varies by up to 40% on code
  translation); §2.2 (five prompt components, content of each
  placeholder stays the same, only structure/syntax differ);
  §2.3 (OpenAI GPT-3.5/GPT-4 via Azure only); Table 1 / §3 (MMLU
  59.7 JSON vs 50.0 Markdown on GPT-35-turbo-0613); Figure 1
  caption ("Accuracy goes up by 42% for JSON compared to
  Markdown" on MMLU international-law questions). Fencing that
  must travel with it: He et al. bundle the whole prompt
  template into one "format" treatment, cross no chat-template
  family, run no local checkpoint under exact-token discipline,
  and re-select no winner.
- Sclar, Choi, Tsvetkov, Suhr, arXiv:2310.11324v2: Abstract only
  (up to 76 accuracy points on LLaMA-2-13B; format performance
  only weakly correlates between models; meaning-preserving
  formatting in few-shot settings on open models). The §3.1
  grammar detail and the ICLR venue tag are REMOVED (see §4).
- Zhang, Chen, Xu, Yang (POML), arXiv:2508.13948v1, §7 Table 3
  (100 sampled styles, WikiTQ subset): Claude 3 Haiku CSV
  (0.449) / Markdown (0.424); DeepSeek V3 HTML (0.801) / JSON
  (0.800); Gemini 2.0 Flash XML (0.791) / HTML (0.788); Mistral
  AI 8x7B XML (0.365) / HTML (Ugly) (0.348); §7 text on
  model-specific optimal table formats. Fencing: POML's 100
  styles bundle table syntax with other styling dimensions; no
  chat-template crossing or winner re-selection.
- Tam et al., arXiv:2408.02442, §4.1 (via the scoping already in
  docs/FIELDORDER_DESIGN.md): emitted key order under
  constrained JSON mode only (100% of GPT-3.5 JSON-mode
  Last-Letter responses placed the "answer" key first). Fenced
  off: emitted-order effects are not input-serialization effects.
- JSONSchemaBench, arXiv:2501.10868v2 (via the citation in
  `structured.py:1-60`): output adherence only, fenced off on
  the same ground.
- No "first" is claimed anywhere in this doc. The contribution
  of any future resubmission is the combination stated in §2,
  never priority. No fabricated citation was found in this round
  (unlike the wave-9 round); the failures found were one
  unresolved detail, one dropped qualifier, and sloppy line
  ranges, all handled above by removal or correction.
- Two 2026 preprints the researcher saw only as search snippets
  remain uncited and unrelied on; a resubmission wanting them
  must verify them from primary text first.

## 6. What a build cycle must NOT do on the basis of this doc

- Do not build the researcher's four-arm probe, the tentative
  `run --payload-format` sketch, or any YAML arm in v1.
- Do not write a causal "serialization changes verdicts"
  headline, a prose-control framing, or any new verdict word.
- Do not transfer placement/canonical-form or field-order floors
  to a new panel without printing that panel's own operating
  characteristics.
- Building, if it ever happens, is a separate cycle, on a fresh
  design review of a resubmission satisfying §2 in full — and
  only on a BUILD verdict from that review.

## 7. Fresh review of the §2 resubmission (ADJUDICATED: NO-BUILD, coordinator #55)

Round: three blind specialists (researcher, red teams A and B —
`~/workspace/agents_out/payloadserial2/`), coordinator adjudication
by its own execution (`verification.md` in the same directory).
Tree at main `18e4b4ca2e4f` (cycle 217 closed green). READ-ONLY
round: nothing built, nothing pushed.

**Verdict: NO-BUILD for §2 as written.** §2 was required to
satisfy ALL of itself in full ("Partial compliance is still
NO-BUILD" — its own preamble), and it does not: it freezes no
case count, its prose arm is not lossless over its own legal
population, its mandated falsification classification has no
stated null and no sane instantiation at its own panel, and its
two mandated transplants (closed-book screen, canonical-form
selection) are mutually incompatible as written. Even the
researcher's FOR case concedes §2 needs a further document
closing the case count (its D1) and the prose arm's literal text
(its D2) before any build verdict. BUILD-as-amended is not
available: the cures change the population (value alphabet), the
headline machinery (classification), or the frozen vocabulary —
the endpoints §2 exists to narrow. A restated candidate is
recorded in §7.3; it is not a build order and is not queued ahead
of anything.

### 7.1 Fatal items (each individually blocking; coordinator-executed)

- **G1 — The prose arm is not lossless over the §2-legal
  population (red-team A F1, CONFIRMED).** §2's value restrictions
  ban non-ASCII, null/nesting/duplicate keys, floats, non-string
  keys, YAML-reserved words, date-like and leading-digit strings —
  but not commas, spaces, or field-label phrases inside values,
  and they freeze no value alphabet for non-answer fields. Two
  distinct §2-legal tables whose distractor values contain a
  separator plus the next field's label render byte-identically
  in prose and distinctly in canonical JSON — executed by the
  coordinator under two independent renderer templates (red-team
  A's `", ".join` and the researcher's own Tree-3 ", and"
  template, with the collision pair adapted to each; all crafted
  values pass every §2 ban mechanically). The reader-entailment
  property §2 requires is therefore not guaranteed by §2's
  generator constraints: any unquoted prose template is
  non-injective over a value set admitting separator + label.
  Quoting/escaping cures injectivity only by making "prose" a
  quoted key-value encoding. Cure: a frozen single-token value
  alphabet for ALL fields — a population change.
- **G2 — The mandated per-candidate classification has no
  stated null and no sane rule at §2's panel (red-team B F1,
  red-team A F11, and the researcher's own OC table; ALL numbers
  reproduced exactly by the coordinator's independent exact
  enumeration).** §2 mandates a classification printed with
  "expected false falsifications" but states neither the rule
  nor the claim/null being falsified. Executed at n_paired = 24
  (the largest panel §2's precedents suggest): the canonical-form
  transplant fires on a competent .9/.9 null with probability
  0.8897 per candidate (family-wise 1.0000 over 24 candidates);
  the researcher's derived Rule A (discordant ≥ 4) fires 0.650
  at per-case discordance δ = 0.18 and, even under its
  friendliest noise model (δ = 0.05), yields expected false
  falsifications 0.71 per panel with P(≥ 1) = 0.516, rising to
  5.14 expected at δ = 0.10; the family-wise-controlled sign
  rule has power 0.0542 at a 25-point gap; McNemar power is
  0.260 at a 20-point gap and 0.053 at a 10-point gap. The
  descriptive core (discordance counts/rates, decomposition,
  per-arm accuracies) needs no classification; the falsification
  classification §2 mandates cannot be instantiated honestly at
  this panel. Cure: drop the falsification classification for a
  purely descriptive report, or a much larger panel — either
  changes §2's headline machinery or population.
- **G3 — §2 freezes no case count, so its population and budget
  never close (red-team B F2, conceded by the researcher as
  D1).** §2's only numbers are the table dimensions (4–6 × 3–4),
  24 candidates, and other probes' precedents. Generations at
  the legal maximum are 72·N for an unspecified N (2 headline
  arms + 1 screen arm per candidate-case); the record and
  payload caps are likewise underivable. The sibling standard
  §2 invokes freezes the count in the design document
  (`canonicalform.py` CASE_COUNT = 24; the negation design
  froze its own arithmetic in-doc). Deferring the population
  size to the builder is the underspecification §1's F1 called
  a build-blocker.
- **G4 — The screen and selection transplants are mutually
  incompatible as written (red-team A F4, CONFIRMED).**
  §2 mandates placement's closed-book screen AND canonical-form's
  selection vocabulary "transplanted verbatim". But
  canonical-form's shared-set machinery explicitly rests on
  there being NO screen (`canonicalform.py:556`: "There is no
  closed-book screen in this probe, so…"), while placement's
  shared set is union-screened (`placement.py`
  `shared_live_fixtures`). §2 never names which shared set its
  selection readout runs over; red-team A executed constructed
  records on which the two §2-compliant readings give opposite
  winner headlines (`winner_changed` vs no winner on either
  arm). The winner headline's population is undefined.

### 7.2 Serious items (confirmed; a restated candidate must cure all)

- **S1 — `document_kind` transplant is semantically void on
  §2's answers** (both red teams; researcher concedes as D3).
  Executed with the shipped `fieldorder.parse_document`: every
  correct bare-code answer is `unparseable`; the only `object`
  answers are JSON echoes — the §1 F4 failure mode. The frozen
  set labels ~100% of correct cells with a failure word and is
  a pure re-encoding of the echo variable. Cure changes §2's
  frozen vocabulary.
- **S2 — Scorer pin names the wrong function** (red-team A F5).
  §2 pins the verdict to `core.py:797` alone; every transplanted
  stage cell uses `exact_match and boundary_found`
  (`experiments.py:9515`, verified), and §2's own decomposition
  presupposes both components. The headline scorer §2 exists to
  pin is left ambiguous between two functions.
- **S3 — The JSON arm is pinned as a function, not as a
  document** (red-team A F7). List vs keyed vs wrapped shapes of
  the same table are byte-distinct under `_canonical_json`
  (executed), the keyed shape reorders records via `sort_keys`,
  and §2 holds only the record *multiset* constant — record
  sequence (a position treatment) is unconstrained. §2-legal
  values containing `"`, `\`, or tab render visibly differently
  across arms (escaping), so "same content" fails at the byte
  level for admitted values.
- **S4 — The mandated diagnostics are uncomputable from §2's
  record shape** (red-team A F10). Canonical-form cells carry an
  answer digest only (verified: verdict, boundary_found,
  boundary_text_match, input_tokens, answer_digest); §2's echo,
  parsed-value, R1 and fence-strip diagnostics are functions of
  the answer TEXT and cannot be recomputed from digests by the
  diff validator. Field-order's cure — per-cell parsed
  commitments (`experiments.py:10638ff`, verified) — is absent
  from §2's enumerated shape.
- **S5 — The exact/boundary decomposition is not total**
  (red-team A F9). The scored answer is the pre-boundary segment
  (`core.py:939`, verified), so a boundary miss manufactures an
  exact failure; the both-components-differ discordance — the
  common boundary-miss mode — has no §2 label and no
  falsification rule.
- **S6 — The id-divergence gate is vacuous for this pair**
  (red-team B F6 = red-team A F8). The tree's HF backend accepts
  only gpt2/llama byte-level BPE (`backends.py`, verified) whose
  decode is lossless, so byte-distinct renderings are
  id-divergent with probability 1: the ids-identical bucket and
  the "instrument inapplicable" reading are unreachable, and the
  three-way report collapses. The negation round already
  adjudicated this shape: `docs/NEGATION_DESIGN.md` §3.4,
  "Token-ID divergence: reported, not gated". §2 mandates the
  gate without answering §3.4. (No estimand bias arises —
  divergence is treatment-side; this is machinery misfit.)
- **S7 — The transplanted selection headline is near-inert on
  §2's population** (red-team B F7; researcher concedes as D4).
  All-pass over 24 multi-row lookups: P(passes an arm) =
  0.95²⁴ ≈ 0.292, 0.90²⁴ ≈ 0.080, 0.80²⁴ ≈ 0.005 (executed
  arithmetic, all parties agree). The winner readout §2's
  utility paragraph leans on ("re-run template selection") is
  `winner_undefined`/abstention in the large majority of runs.
- **S8 — Duplication residue is thin** (red-team B F5, ownership
  inventory verified). After removing shipped parts (canonical-form's
  three-way report, selection, record and validator skeleton;
  placement's screen; `structured`'s renderer and field-order
  convention; scorer-lens regrades; the field-order aggregate
  already computes pairwise frozen-renderer discordance decisions
  over structured specs), the genuinely new residue is one table
  generator, one prose renderer, and the echo wiring — and G1
  shows the residue's own renderer component fails over §2's
  population. Duplication alone was not fatal in sibling rounds;
  it is recorded here as context, not as an independent fatal.

### 7.3 Refuted or downgraded items (adjudicated by execution)

- Red-team A F3 (frozen stage vocabulary self-contradictory):
  **REFUTED as fatal.** The tree's own convention is two-layer —
  `negation.py:101-112` freezes STAGE_DETAILS as exactly
  {cancelled, negation_record_budget_exceeded,
  negation_screen_failed}, excluding both `""` (the stage success
  detail, returned by every sibling stage) and
  `negation_unavailable` (the wrapper/analysis fallback,
  `cli.py:2076`, `experiments.py:10027`). §2's two bullets mirror
  that pattern. Downgraded to a wording imprecision: §2 should
  name the wrapper layer for `payload_format_unavailable`.
- Red-team B F4 (naming gate unsatisfiable): **DOWNGRADED** from
  fatal to friction. The collision table is real (68 CLI flags;
  the Format/document/render senses are shipped), but §2 defers
  the flag name to an audit and does not forbid coinage; sibling
  rounds coined audited names (`--response-directive`;
  `--multi-question` settled clean). Counted as one more deferred
  load-bearing choice under G3's completeness pattern, not an
  independent fatal.
- Red-team B F10 (validator overstatement): upheld as a wording
  point only — the diff validator can check arithmetic
  consistency of stage-asserted `ids_identical`/token counts,
  not re-derive them (diffreports has no tokenizer). The mirror
  machinery itself is precedented (`diffreports.py:10401`,
  `:11201`, `:11341` — all refs verified).
- Red-team A F2 (priming manufactures/hides discordance): the
  descriptive re-scope blunts the "manufactures" half (a priming
  flip IS a verdict discordance between the two renderings,
  which is all the descriptive headline claims); the "hides"
  half and the classification interaction stand jointly with
  G2 — on the parsed-value view the same executed run flips its
  classification, and §2 never states which view its
  falsification language consumes, because it never states the
  falsified claim.
- Screen cost (both red teams; researcher D5): facts confirmed
  (the screen arm is 1 of 3 arms — 576 of 1,728 generations at
  N = 24 — with expected exclusions ≈ 0 for seed-generated
  codes). Hygiene, not a defect of principle; a restated
  candidate must state the cost and justify or drop the arm.

### 7.4 Fabrication sweep: CLEAN

Every load-bearing number and citation was re-derived by the
coordinator: 11 operating-characteristic values across three
independent computations reproduce exactly (0.8897 / 0.9959 /
0.0298 / 0.2143 / 0.6500 / 0.8850 / FWER 0.5160 / McNemar 0.0534,
0.2598, 0.7994 / Bonferroni-sign 0.0542); parse kinds, renderer
collisions, selection arithmetic, 68 CLI flags, the 20,132-line
diffreports with its mirror line refs, USAGE §8.54/:2713 and
§8.55/:2762, the 1,819-line / 76-test cycle-209 file — all exact.
External sources: the coordinator fetched arXiv:2411.10541 and
arXiv:2310.11324 primary texts; titles, authors, and the fenced
figures match §5 exactly (POML Table 3 stands on round 1's
verification plus this round's researcher re-verification; it
was not independently re-fetched here, recorded honestly). The
only errors found in any brief are the two interpretive
overstatements adjudicated in §7.3 — no fabricated citation,
symbol, or figure was found in any of the three briefs.

### 7.5 What survives, and the restated candidate (NOT a build order)

Survives intact: the narrowed *question* — input-side
serialisation of the same generated multi-record table inside
the user message, template family held fixed per candidate, is
owned by no shipped probe (rotation varies block order,
placement question position, canonical-form sub-lexical
normalisation, field-order schema presentation order with a JSON
*output* document, structured output adherence). The uniqueness
residue is real as a contrast; §2's instrument for it is what
fails. The §5 sources and the §2 utility sentence survive
unchanged.

A restated candidate must, in its own text, before any fresh
review: (1) freeze the case count N and derive its budget
arithmetic including the screen arm; (2) freeze a single-token
value alphabet for ALL fields (no spaces or punctuation in any
value) and re-prove prose injectivity under it; (3) print the
prose arm as a literal compositional total function over sorted
keys with a worked example; (4) pin the JSON document shape
(a list of record objects) and hold record SEQUENCE, not
multiset, constant across arms; (5) state the cell verdict
formula (the sibling `exact_match and boundary_found` pattern)
and make the decomposition total over the four (exact,
boundary) cells, with the both-differ class named; (6) drop the
falsification classification — report discordance counts/rates,
decomposition, and per-arm accuracies descriptively — OR state
the falsified claim and its null and derive a rule whose
executed operating characteristics at the frozen N are sane on
both sides; (7) drop the `document_kind` transplant (the echo
diagnostic carries the parse information, committed per cell in
the record shape on the field-order pattern); (8) name the
selection population composition (union-screened shared set, the
placement pattern) or drop the winner readout in v1 (the
negation precedent); (9) report token-ID divergence, not gate
on it (negation §3.4); (10) coin the flag name and defend it
against the whole of USAGE. Open decisions the restated text
must settle itself: classification-vs-pure-description (item 6),
screen retention at +50% budget for ≈0 expected exclusions, and
selection in v1 vs dropped. This candidate is not queued ahead
of anything; it enters only through a fresh blind design review
returning BUILD.

### 7.6 Side findings (reported, not fixed)

- JSONSchemaBench currency (researcher): arXiv now lists a v3
  (27 Feb 2025) as latest; the tree pins v2 deliberately
  (`structured.py:5`). Not an error — a resubmission or docs
  cycle should keep the v2 pin consciously or re-verify against
  v3, never drift silently.
- `diffreports.py` stands at 20,132 lines; each additional
  probe's validator mirror costs ≥ 140 lines there. A growing
  structural cost, noted for the parent's planning, not a
  defect of this candidate.

## 8. Restated candidate as amended (cycle-221 build text of record)

The text below is the restated candidate authored by
coordinator #64 (`payloadserial_restated/candidate.md` in the
agent workspace), incorporated AS AMENDED by the fresh
review's eight binding amendments (§9 below): the RNG
call sequence frozen in §2.3 (A); the instantiated-key
overlap accounted in §2.3 and printed in the report (B); the
§4.4 parsed/echo subset claim corrected (C); the §5.2 mirror
given the input-token binding and the aggregate-eligible
invariant, with §5.3's token row corrected to match (D); the
§6.1 power figures given their marginal bases, with the
base-.90 figures printed beside the mixed-base ones (E); the
resolution printed beside the §6 discordance counts (F); the
§8 margin claim corrected (G); the §9 flag census re-derived
by parser instantiation (H). Amendment sites are marked in
the text. One further condition from the review's researcher
(condition C): the scope phrase "for this checkpoint and
this candidate's template family" appears in the §1 utility,
the CLI help, the USAGE section, and the report section.
Everything not marked is the candidate's text verbatim,
including its frozen numbers — every one of which the build
re-derived by execution before relying on it (the worked
example is byte-identical at 317/286 chars; the worst-case
lengths are 653/594; the operating-characteristics table
reproduces exactly under the candidate's own `oc_tables.py`
enumeration method).

## 0. Settlement map (what this restatement decides, and where)

The ten items §7.5 of the design doc requires a restated
candidate to settle in its own text:

| # | Item | Settlement |
|---|---|---|
| 1 | Freeze N; budget incl. screen arm | N = 24 frozen (§2); full arithmetic (§8): 1,728 generations at the legal maximum, screen arm's 576 (+50%) inside the total |
| 2 | Single-token value alphabet for ALL fields; re-prove prose injectivity | Frozen alphabet (§2); injectivity theorem + exhibited inverse + executed round-trip over 100,000 legal tables (§3) |
| 3 | Prose arm as a literal compositional total function over sorted keys, with worked example | Literal function (§3) + byte-exact worked example (§3.3) |
| 4 | JSON shape pinned; record SEQUENCE held constant | List of record objects; sequence, not multiset, is the held constant (§3.1) |
| 5 | Cell verdict formula stated; decomposition total, both-differ class named | Formula (§4.1); total decomposition with the `mixed` class (§4.3) |
| 6 | Drop the falsification classification, or state claim+null and derive a sane rule | DROPPED. Purely descriptive report (§6); the rejected rules' executed operating characteristics under independent, paired, and δ-noise nulls are printed there |
| 7 | Drop the `document_kind` transplant | Dropped (§4.4); the echo diagnostic carries the parse information |
| 8 | Name the selection population, or drop the winner readout in v1 | Winner readout DROPPED in v1 (§6.3), negation precedent |
| 9 | Token-ID divergence reported, not gated | Reported, not gated (§6.2), negation §3.4 disposition verbatim in effect |
| 10 | Coin the flag name; defend against the whole of USAGE | `--record-notation` coined and defended (§9) |

The three open decisions §7.5 names are settled as follows:
classification-vs-pure-description → pure description (§6);
screen retention at +50% budget for ≈0 expected exclusions →
RETAINED, cost stated and justified (§2.4, §8); selection in v1
vs dropped → dropped (§6.3). The §7.2 serious items are cured at:
S1 §4.4, S2 §4.1, S3 §2.2+§3.1, S4 §5, S5 §4.3, S6 §6.2,
S7 §6.3, S8 §10 (residue stated, not hidden).

## 1. Utility and claim scope

**Monday-morning utility (one sentence).** A prompt-assembly
owner deciding whether to switch the frozen notation their
prompt assembly uses for generated data tables — prose sentences
or canonical JSON — learns, for this checkpoint, on how many of
24 paired generated tables the two notations' byte-exact verdicts
discord, in which direction, and whether the discordance is
exact-only, boundary-only, or mixed, before making that switch
in production prompts.

**The claim is descriptive pairwise discordance between two
named frozen renderings, and nothing else.** It is never a causal
claim ("serialization changes verdicts"), never a claim about
weights, never a claim about notations not frozen here. The
treatment is the two renderings *as wholes*: token length,
delimiters and punctuation, lexical framing, field order as a
reading sequence in prose (§3.4), and output priming (§4.4) all
move together and are named as the bundle, never attributed
individually. Component attribution belongs to the shipped
single-axis probes (field-order owns key order; the whitespace
scan owns whitespace).

**What would prove this design's own load-bearing statements
wrong.** (a) Injectivity (§3.2): exhibit two distinct legal
tables (§2) with byte-identical prose renderings. (b)
Decidability (§5.3): exhibit a quantity printed in §6 that the
diff validator cannot recompute from the record shape in §5.1
plus the regenerated tables. (c) The descriptive-only decision
(§6.1): exhibit a classification rule whose executed operating
characteristics at N = 24 are sane on both sides under the null
families in §6.1 — the authoring execution found none.

## 2. Population (frozen)

### 2.1 Cases

`CASE_COUNT = 24` generated cases per candidate, stage-local,
never entering the user's search/held-out Split (canonical-form
invariant-3 pattern; the Split compares message inputs by NFC
key, `core.py:193-214`, so form-twin fixtures would collide — no
fixture arm). One table per case. Table shape is a frozen
function of the case index `i` (0-based):

- records: `n_records = 4 + (i mod 3)` ∈ {4, 5, 6};
- fields: 3 if `i` is even, 4 if `i` is odd.

Over `i mod 6` the shapes are (4,3), (5,4), (6,3), (4,4), (5,3),
(6,4) — every combination exactly once per six cases, exactly
four times per panel.

### 2.2 Fields and the value alphabet (all fields, no exceptions)

Field universe, alphabetical (the `structured.describe_schema`
convention, `structured.py:403-443`): `code`, `label`, `mark`,
`zone`. A 3-field table uses {code, label, zone}; a 4-field table
adds `mark`. The answer field is always `code`; the key field is
always `zone`.

Every value in every field — answer, key, and distractor alike —
matches `^[a-z]+-[0-9]{4}$`, its word part drawn from the frozen
24-word list

amber, birch, cedar, copper, coral, falcon, garnet, harbor,
indigo, juniper, lagoon, magnet, marble, nickel, onyx, pepper,
quartz, saddle, saffron, summit, tundra, velvet, walnut, willow

and its number part in [1000, 9999]: a code space of 24 × 9000 =
216,000 values. All cell values within a table are pairwise
distinct (sampled without replacement; the largest table needs
24 values of 216,000). Consequences, all intended: no value
contains a space, a comma, a period, a quote, a backslash, a
control character, or a non-ASCII character, so JSON escaping is
unreachable over the legal population (executed: §3.2) and no
value can contain a separator or a field-label phrase — the two
holes the §2 population died on (G1, S3). No value is a
YAML-reserved word, a date-like string, or a bare number. The
population is a population of *code tables*: this is the stated
price of a lossless prose arm, paid openly (§3.2), not a hidden
narrowing.

### 2.3 Generator (a total function of (seed, case index))

For case `i`: `rng = random.Random(seed * 1000 + i)`. Draw
(word, number) pairs until `n_records × n_fields` distinct values
are collected (duplicate draws are consumed and discarded — part
of the frozen stream); fill the table row-major, fields in
alphabetical order. The queried record index is the next draw,
`rng.randrange(n_records)`. The question names that record's
`zone` value; the expected answer is its `code` value. The diff
validator regenerates every table with its own literal copy of
this generator (§5.2).

**Amendment A (binding, fresh review): the RNG call sequence
is frozen, not paraphrased.** The paragraph above admits
faithful readings that diverge on 24/24 tables (a paired draw
is not a call sequence). The frozen sequence, bound into the
protocol fingerprint through the case material, is: per value
draw, in order, `word = WORDS[rng.randrange(24)]`, then
`number = rng.randrange(1000, 10000)`; a duplicate value is
consumed and discarded; the table fills row-major, fields in
alphabetical order; the queried record index is the next draw,
`rng.randrange(n_records)`, after the table is complete. The
generator in `recordnotation.py`, the validator's literal
mirror, and the test oracle all execute exactly this sequence;
a payload whose tables were generated under any other reading
fails regeneration.

Pre-generation audits, all with counts in the stage's audit
block; any failure fails the whole stage closed
(`record_notation_screen_failed`), never a silent per-case drop:

- alphabet conformance of every value; within-table
  distinctness; key (`zone`) uniqueness (implied by
  distinctness, asserted anyway);
- the answer value appears in exactly one row (implied by
  distinctness, asserted anyway);
- byte-divergence of the two arm renderings per case (assert;
  §3.2 makes failure a construction impossibility, so a failure
  is a generator defect, which is what the assert is for);
- boundary-containment plus `audit_special_token_literals`
  (`fixtures.py:471`) over both renderings, the question, and
  the contract, for every candidate family;
- answer-embedding audit: no field name, question token, or
  contract token shares a 4-character prefix with, or is a
  prefix of, any value word. Executed over the frozen lists by
  the authoring coordinator: 0 violations (`audit_alphabet.py`).
  The stage re-runs it over its frozen lists at construction.
- instantiated-key accounting (**amendment B, binding**): the
  audit above covers template tokens only. The instantiated
  key is a drawn value, and the answer's word equals the key's
  word in about 4.17 percent of cases (analytic 8999/215999
  over ordered distinct value pairs). The overlap is printed
  in the report beside the screen readout. It is
  arm-symmetric — the question is byte-identical across arms,
  so it cannot manufacture discordance — and it is NOT
  redrawn: a redraw rule would change the population and is
  not authorised.

Reader-entailment is not a per-case audit in this design: it is
*discharged by the injectivity theorem* (§3.2). A reader given
the question and either rendering recovers the identical table;
`zone` is unique in it; the answer is therefore uniquely
derivable from either arm. This is strictly stronger than §2's
per-case entailment check, and it is the reason the alphabet is
frozen as it is.

### 2.4 Closed-book screen arm (retained, cost stated)

Per candidate-case, a third generation: the question and
contract with no table (§3.3 shows the exact content). Screening
is per candidate (placement pattern, `placement.py:374-380`): a
case a candidate passes closed-book (cell verdict true, §4.1)
is excluded from *that candidate's* live populations;
`n_screened` and `n_closed_book_incomplete` (screen cells that
did not complete — never silently treated as screened or as
passed) are printed per candidate. Expected exclusions are ≈ 0
by construction: a closed-book pass requires guessing one value
of 216,000 (executed P = 4.63e-06 per case). The arm is retained
as fail-closed hygiene against generator or code-space drift —
its pass count is itself a printed leak readout — at a stated
cost of 576 of 1,728 generations (+50% over the headline arms,
§8). The headline readouts are also printed over the unscreened
paired set as a sensitivity (§6), so the screen's effect on any
number is visible.

## 3. Arms (frozen, literal, byte-exact)

Exactly two headline arms. No other arm is a decision input;
there are no descriptive extra arms in v1. Templates are not
touched: the notation lives inside the user message content
only. Held constant across arms: the record *sequence* (not
multiset — §3.1), all values, the key and answer fields,
alphabetical field order, the question (byte-identical, after
the table in both arms), the output contract (byte-identical),
greedy decoding, `--max-new-tokens`, boundary ids, the seed,
and the template family per candidate.

### 3.1 The canonical-JSON arm

`structured._canonical_json` (`structured.py:336-343`:
`json.dumps(document, sort_keys=True, separators=(", ", ": "),
allow_nan=False)`, default `ensure_ascii=True`) applied to the
**list of record objects, in record sequence order**.
`sort_keys` orders keys *inside* each record object; it never
reorders the list — so record sequence, a position treatment if
left free (S3), is held constant by construction and asserted
by the validator's regeneration.

### 3.2 The prose arm, and the injectivity theorem

The prose arm is this literal total function of the record
sequence (fields in sorted-key order within each record; one
sentence per record; sentences joined by a single space):

- 1 field: `The <f> is <v>.`
- 2 fields: `The <f1> is <v1> and the <f2> is <v2>.`
- k ≥ 3 fields: `The <f1> is <v1>, the <f2> is <v2>, …, and the
  <fk> is <vk>.`

**Theorem (injectivity over the legal population).** Two
distinct legal tables never have the same prose rendering.
Proof: the function has an exhibited inverse — split the
rendering into sentences at `. ` boundaries and the final `.`;
within a sentence, strip the leading `The `, normalise the
final-join forms (`, and the ` / ` and the `) to the separator
`, the `, and split; each resulting segment has the unique form
`the <field> is <value>`. Segmentation is unique because every
separator contains a space, a comma, a period, or an uppercase
letter, and **no value and no field name contains any of those
characters** (values match `[a-z0-9-]+`, field names match
`[a-z]+`); the field names are pairwise distinct and none is a
prefix of another. The inverse is implemented in
`renderers.py:parse_prose` and executed by the authoring
coordinator (`proof_injectivity.py`): `parse(render(t)) == t`
on 100,000 random legal tables across all six shapes, and on
3,000 further tables of 1–6 fields *outside* the population
(the function is total, not fitted to the population). The same
execution confirms `json.loads(render_json(t)) == t` on every
sample, that the JSON rendering contains **no backslash** on
every sample (escaping unreachable under §2.2), and that the two
renderings are byte-distinct on every sample. The G1 collision
pair from the §2 review is not a counterexample: all four of its
crafted values are illegal under §2.2 (executed rejection).
Quoting or escaping values in prose — the only cure available
to §2's population — is rejected here for §7's stated reason:
it would make "prose" a quoted key-value encoding and change
the treatment.

### 3.3 Worked example (byte-exact; executed, `proof_injectivity.py`)

Table (4 records × {code, label, zone}); queried zone
`garnet-6006`; expected answer `copper-4004`:

Prose rendering (317 chars):

```
The code is amber-1001, the label is birch-2002, and the zone is cedar-3003. The code is copper-4004, the label is falcon-5005, and the zone is garnet-6006. The code is harbor-7007, the label is indigo-8008, and the zone is juniper-9009. The code is lagoon-1010, the label is magnet-2020, and the zone is nickel-3030.
```

Canonical-JSON rendering (286 chars):

```
[{"code": "amber-1001", "label": "birch-2002", "zone": "cedar-3003"}, {"code": "copper-4004", "label": "falcon-5005", "zone": "garnet-6006"}, {"code": "harbor-7007", "label": "indigo-8008", "zone": "juniper-9009"}, {"code": "lagoon-1010", "label": "magnet-2020", "zone": "nickel-3030"}]
```

Frozen message composition, both arms:
`content = rendering + "\n\n" + question + " " + contract`,
where the question is `What is the code of the record whose
zone is garnet-6006?` and the frozen contract is `Answer with
ONLY the value.` The closed-book (screen) content is
`What is the code of the record whose zone is garnet-6006? Answer with ONLY the value.`
A build pins every string in this section byte-identically by
test, including this example (the §2 D2 discipline).

### 3.4 Disclosed asymmetries (part of the named bundle)

- *Length.* The renderings differ in length per case (worked
  example: prose 317 vs JSON 286 chars; worst case at 6×4:
  prose 653, JSON 594 — executed). No length matching is claimed
  or attempted; `id_length_delta` is printed per case (§6.2).
  Both worst cases sit far below any context cap, so
  arm-asymmetric `arm_context_exceeded` attrition is not
  expected at these sizes; per-arm attrition is printed
  regardless (§6).
- *Order in prose.* Alphabetical order inside a JSON object is
  presentational; inside a prose sentence it is a reading
  sequence (the answer field is sentence-initial, the key field
  sentence-final, in every record sentence). The assert holds
  the *label order* constant, not the reading process (red-team
  A F12 on §2). This asymmetry is part of the bundled contrast
  the descriptive claim is about; it is disclosed here so no
  reader mistakes the claim for a pure-delimiter contrast.

## 4. Scorer and decomposition

### 4.1 Cell verdict (one formula, stated — the S2 cure)

A cell's verdict is

```
verdict = (status == "completed") and (answer == expected) and boundary_found
```

i.e. `core.evaluate`'s `exact_match` (`core.py:797`) conjoined
with `boundary_found` — the sibling stage-cell pattern
(`experiments.py:9515`), not `core.py:797` alone. The scored
answer is the decoded pre-boundary segment (`core.py:939`).
`exact_match` and `boundary_found` are stored separately per
cell (§5.1), so every readout below is a function of stored
components. Model parse failure is a verdict of false, never a
cell detail (§7).

### 4.2 Per-cell decomposition (completed cells only)

A missing cell is attrition with a cell detail (§7), never a
class. Every completed cell is exactly one of:

- **pass** — verdict true;
- **boundary_only** — answer digest equals the expected digest
  and `boundary_found` is false (negation's class, same
  meaning: the answer text was right, the boundary was missed;
  a boundary effect, never a notation effect);
- **other failure** — counted, and characterised only through
  the §4.4 diagnostics; no class word is coined for it.

Counts of all three are printed per arm per candidate.

### 4.3 Pair decomposition (total — the S5 cure)

Over discordant pairs in the headline population (§6), the
negation construction (`negation.py:543-568`), which is total:

- verdicts differ and answer digests are equal →
  **boundary_only**;
- verdicts differ, digests differ, and both cells found their
  boundary → **exact_only**;
- verdicts differ otherwise → **mixed** (reported, never
  attributed).

Totality: if verdicts differ and digests are equal, the exact
components are equal (same answer text, same expected value),
so the verdict difference is the boundary component alone;
if both boundaries were found, the verdict reduces to the
exact component on both sides, so a verdict difference is an
exact difference; the remaining discordant pairs are exactly
`mixed`. The three classes partition the discordant pairs.
`mixed` is the both-components-differ class §2 lacked a name
for; its expected shape is a boundary miss inflating the
pre-boundary segment (`core.py:939` coupling), stated here so
nobody reads `mixed` as two independent failures. A
boundary-only discordance is a boundary effect: it is reported
in its own count and cannot support any notation reading.

### 4.4 Diagnostics (beside the headline, never rescuing)

The `document_kind` transplant is **dropped** (item 7): on this
instrument's bare-code answers it labels every correct cell
`unparseable` and is a pure re-encoding of the echo variable
(S1). The parse information is carried by the echo diagnostic
instead, committed per cell in the record (§5) so the validator
recomputes it:

- **Echo rate** per arm: the share of the arm's completed
  byte-exact failures whose answer contains the expected value
  as a substring.
- **Parsed-value rate** per arm: the share whose answer parses
  via `json.loads` as an object whose value at the answer-field
  key (`code`) equals the expected value. (**Amendment C,
  binding correction:** an earlier draft of this section
  claimed a parsed-value hit is always a substring hit, making
  the parsed rate a subset of the echo rate. That claim is
  FALSE, executed counterexample: an answer whose `code` value
  carries the escaped hyphen — `\u002d` as six literal
  characters — parses to the expected value while the raw text
  does not contain it as a substring; conversely a prose
  answer can contain the value without parsing. The two rates
  are independent diagnostics, printed beside each other,
  neither a subset of the other.)
- **Regrade sensitivities**: pair discordance recomputed with
  each answer replaced by `scorerlens._norm_r1(answer)`
  (whitespace trim/collapse, `scorerlens.py:146-153`) and,
  separately, by `scorerlens._fence_strip(answer)`
  (`scorerlens.py:176-187`), the boundary conjunct unchanged.

Priming (a model answering in the input arm's idiom) moves the
byte-exact headline and is *measured* by the echo and
parsed-value rates; it is never corrected away, and — the
structural cure for red-team A F2 — **there is no classification
for it to contaminate** (§6.1): no downstream readout consumes
one scoring view over another, and where the byte-exact and
parsed-value views disagree on a case's concordance, the counts
for both views are what the report prints.

## 5. Records and decidability

### 5.1 Record shape (one record per case per candidate)

- `fixture` (frozen case id), `n_records`, `n_fields`,
  `expected_digest` (SHA-256 of the expected code);
- per headline arm: `content_digests` (SHA-256 of the arm's
  full user-message content), `arm_tokens` (full-render id
  length), `id_digests` (SHA-256 of the full-render id
  sequence);
- `ids_identical` (bool), `id_length_delta` (canonical-JSON id
  length minus prose id length);
- `cells`: per arm (both headline arms and `closed_book`) a
  cell or null, plus `cell_details` per arm (§7);
- a headline-arm cell is `{verdict, boundary_found,
  boundary_text_match, input_tokens, answer_digest,
  answer_text}`; a screen-arm cell omits `answer_text`.

`answer_text` is the decoded pre-boundary answer string,
bound by `answer_digest = SHA-256(answer_text)` — the
cycle-190 pattern (numform's digest-bound carried text). This
is a deliberate, stated deviation from §2's digest-only record
shape: §2's own mandated diagnostics are functions of the
answer text and are uncomputable from digests (S4), so §2's
shape could not carry §2's report. Reports still render
counts and rates only; answer text never reaches a report.

### 5.2 Validator mirror duties

The diff validator regenerates every table from (seed, case
index) with literal copies of the generator and both renderers;
checks each content digest and pairwise digest inequality
across arms; checks the expected digest; verifies each
headline cell's `answer_digest` against its carried
`answer_text`, then recomputes verdicts, the §4.2/§4.3
decompositions, all §4.4 diagnostics, and every count, rate,
and split in §6 from the records alone. **Amendment D
(binding):** the mirror also verifies every completed cell's
`input_tokens` against the record's `arm_tokens[arm]` — the
cycle-218/219 binding this candidate's draft omitted, under
which a payload with fabricated token fields passes. The
stage measures the arm token count at encode time; the
cell's `input_tokens` come from the evaluation's own
observation, so a backend whose encode disagreed with itself
fails validation. Cells exist only from aggregate-eligible
reports, so a non-null cell certifies a completed evaluation
and the verdict's status conjunct is discharged structurally;
that invariant is part of the mirror's contract, stated here
and in its docstring.

### 5.3 Decidability, class by class

| Printed quantity | Decidable from | Validator path |
|---|---|---|
| Cell verdict | carried answer_text + expected_digest + boundary_found | digest check, then §4.1 recomputed |
| boundary_only cell class | answer_digest, expected_digest, boundary_found | digest equality (§4.2) |
| Pair decomposition | both cells' verdicts, digests, boundary flags | §4.3 construction |
| Echo / parsed-value rates | carried answer_text + expected value (regenerated) | substring test; `json.loads` |
| R1 / fence regrades | carried answer_text | literal mirrors of `scorerlens` functions |
| Screen verdict | screen cell digest + boundary flag (no text carried; no text-derived quantity is reported for the screen arm) | digest equality (§4.1) |
| Content/expected digests | (seed, case index) | literal regeneration (§5.2) |
| ids_identical, id_length_delta, arm token counts | recorded token-id digests + stage encode counts + each evaluation's own input_tokens | ids_identical re-derived from the recorded digests; id_length_delta must equal the canonical-JSON minus prose count difference; every completed cell's input_tokens bound to the record's arm_tokens (§5.2, amendment D). The mirror has no tokenizer and does not re-tokenize; the two independent token measurements the stage records are cross-bound instead (the §5.3 draft's "arithmetic consistency only" understated this and is corrected) |

## 6. Report shape — descriptive only

Per candidate, in order: audit counts (`n_screened`,
`n_closed_book_incomplete`); per-arm completed counts, pass
counts and rates over completed cells, §4.2 class counts, and
per-arm attrition (`n_context_exceeded`, `n_evaluation_failed`)
beside every rate; the headline population and its readouts;
the §4.4 diagnostics; the id report (§6.2); the sensitivities
(discordance over the unscreened paired set; regrade
discordance counts). Rates over an empty denominator print as
the reporting layer's existing `n/a`. Discordance is never
pooled across candidates or template families: per-candidate
contrasts only.

The **headline population** is: live cases (§2.4) with both
headline-arm cells completed, minus any same-ids anomaly
(§6.2). Readouts over it: `n_paired`; discordant count `d`, with the
resolution `100 / n_paired` points per discordant case printed
beside the count (**amendment F, binding** — a count without
its resolution invites a finer reading than the panel
supports);
directional split `n_prose_only` / `n_canonical_json_only`
(cases passing on exactly that arm — symmetric by construction;
this instrument has **no baseline arm**, and no arm is a
control); decomposition counts `n_exact_only`,
`n_boundary_only`, `n_mixed` (§4.3); per-arm pass rates over
the paired population beside the per-arm rates over all
completed cells. The estimand, stated once and printed with
the section: *discordance between these two frozen renderings,
over live cases both arms completed, for this candidate's
template family on this checkpoint.*

### 6.1 No falsification classification (item 6, decided)

V1 prints no per-candidate classification and falsifies no
claim. The claim that was considered — "the two notations are
verdict-equivalent for this candidate", falsified by a
discordance rule — cannot be instantiated honestly at N = 24.
Executed by the authoring coordinator (exact multinomial
enumeration over per-case outcomes, `oc_tables.py`), at
n_paired = 24, 24 candidates:

| Rule | Competent null .9/.9 (indep.) | .9/.9 paired φ=0.5 | δ=0.05 fair-direction noise | Power, 25-pt gap | Power, 20-pt gap |
|---|---|---|---|---|---|
| Canonical-form transplant (n≥6, max-dir ≥2) | fires 0.8897; FWER E = 21.35, P(≥1) = 1.0000 | fires 0.5088 | fires 0.2276 | — (fires on noise) | — |
| Rule A (d ≥ 4, §2 review's derived rule) | fires 0.6500 | fires 0.1648 | fires 0.0298; FWER E = 0.71, P(≥1) = 0.516 | — | — |
| McNemar (d ≥ 6, sign p ≤ 0.05) | fires 0.0061 | fires 0.0005 | fires 0.0000 | 0.3547 | 0.2598 |
| Sign, Bonferroni (α = 0.05/24) | fires 0.0000 | fires 0.0000 | fires 0.0000 | **0.0542** | 0.0242 |

**Amendment E (binding): the power columns carry their
marginal bases.** The 25-point column runs on marginals
.85/.60 and the 20-point column on .90/.70; the two columns
are not on a common base and must not be read as one curve.
At the table's own competent base .90, a 25-point gap
(.90/.65) gives McNemar power 0.4000 (not a figure in the
printed columns). The N = 96 Bonferroni power below is
0.7452 on the .85/.60 marginals and 0.8329 at the .90 base;
both are printed wherever the panel figure is cited.

Every count-based rule is a false-falsification machine on
distant surfaces under every null family executed (independent,
paired/correlated, and fair-direction δ-noise — δ is the
per-case flip probability under any token perturbation, and
nothing in this instrument bounds δ for a prose-vs-JSON pair);
every family-wise-controlled rule is near-inert at the gaps
that matter. The larger-panel branch was executed, not assumed:
at N = 96 the Bonferroni sign rule reaches false-fire 0.0008
with power 0.7452 at a 25-point gap on marginals .85/.60
(0.8329 at the .90 base, amendment E) — sane, at 6,912
generations (4× this candidate's budget, §8), to license a
*directional-superiority* claim this instrument's utility (§1)
does not consume. V1 therefore reports the discordance counts
the utility actually reads. A future v2 that wants a
classification must freeze the larger panel and its rule in
its own text, with this table as its starting evidence.

### 6.2 Token-ID divergence: reported, not gated (item 9)

The headline population is **not** conditioned on id
divergence (negation §3.4: gating would gate on a condition
true by construction and hide the one informative event).
Reported per candidate: `n_ids_identical`, the per-case
`id_length_delta` list in case order, `max_abs_id_delta`, and
`n_same_ids_verdict_flips`. Under the tree's HF backends
(gpt2/llama byte-level BPE, lossless decode) byte-distinct
renderings are id-divergent, so `n_ids_identical` is expected
to be 0; a same-ids case, if one ever occurs, is excluded from
the headline population and counted as a rendering anomaly,
and a same-ids verdict flip is recorded as a determinism
finding — never as notation fragility, never as robustness.
A run landing 100% ids-identical would read "instrument
inapplicable to this tokenizer", never "the model is robust
to notation".

### 6.3 No winner readout in v1 (item 8)

The canonical-form selection transplant is **not** carried.
Executed arithmetic (agreed by all parties in two rounds): a
candidate passes an arm only by passing every completed case,
so P(passes an arm) = 0.95²⁴ ≈ 0.292, 0.90²⁴ ≈ 0.080,
0.80²⁴ ≈ 0.005 over 24 multi-row lookups — the winner headline
would be undefined or abstention in the large majority of runs
(S7), and §2's pairing of the selection transplant with the
closed-book screen left its population undefined (G4).
Dropping selection dissolves both defects; the negation
precedent ("no winner readout in v1") governs. Per-arm
accuracies are printed (§6) as the selection-adjacent readout.
A v2 wanting selection must name its population composition
(union-screened shared set, the placement pattern) in its own
text.

## 7. Fail-closed vocabulary (frozen, two layers named)

Following the tree's two-layer convention (`negation.py:101-112`
and the wrapper pattern), named layer by layer so neither layer
contradicts the other:

- Stage details (the stage's own returns), exactly:
  `{cancelled, record_notation_record_budget_exceeded,
  record_notation_screen_failed}`.
- Per-cell details, exactly: `{"", arm_context_exceeded,
  arm_evaluation_failed}`.
- Whole-stage fallback at the wrapper/analysis layer: any
  renderer failure, serializer exception, generator failure,
  or non-UTF-8 payload degrades the whole stage to
  `{"available": False, "detail": "record_notation_unavailable"}`
  — never to a per-cell verdict.

No other words. In particular: no classification labels (§6.1),
no `document_kind` (§4.4), no `arm_parse_failed`, no
`payload_too_large`, no per-arm "not applicable", no new
verdict words of any kind.

## 8. Budget (frozen, with derivations)

- `MAX_CANDIDATES = 24` (the tree's legal maximum).
- `CASE_COUNT = 24` (§2.1).
- `MAX_GROUND_RECORDS = 24 × 24 = 576` (sibling derivation:
  `placement.py:179`, `canonicalform.py:64`).
- `GENERATION_BUDGET = 3 arms × 576 = 1,728` — 1,152 headline
  (2 arms) + 576 screen (1 arm). The screen arm's +50% over the
  headline count is stated here, in the CLI help, and in §2.4,
  per the field-order precedent that generation counts at the
  legal maximum are stated in CLI help.
- `MAX_RECORDS_BYTES = 12 MiB` (sibling constant:
  `placement.py:183`, `canonicalform.py:69`). (**Amendment G,
  binding correction:** the draft's "4.0× margin" claim was
  false. 2,048 chars IS the tree's 64-token answer-text bound
  (`fieldorder.py`), not "far beyond a 64-token decode"; and
  measured on the built §5.1 record shape — which carries
  fields the draft's 5,442-byte synthetic record omitted —
  the ASCII worst case is 5,507 bytes per record → 3.03 MiB
  per panel, a headroom of 3.96×, not 4.0×. A
  non-ASCII-stressed panel (CJK answer texts at the cap,
  25,987 bytes per record under `ensure_ascii` escaping)
  measures about 14.28 MiB, OVER the cap: such a panel fails
  closed with `record_notation_record_budget_exceeded`. The
  fail-closed budget behaviour is unchanged; only the printed
  claim is corrected. A typical record (bare-code answers)
  measures about 0.79 MiB per 24-candidate panel.)

## 9. Naming (item 10)

The flag is **`--record-notation`**; the module is
`recordnotation.py`; the stage-detail prefix is
`record_notation_` (sibling convention: flag → module →
prefix). The name is a coinage, defended against the whole of
USAGE and the CLI namespace (executed, `namespace_audit.py`):

- Not taken: (**amendment H, binding correction:** the
  draft's "69 flags" census was wrong under every counting
  method). By parser instantiation at review time the tree
  carried 77 distinct long options including `--help` (76
  excluding); at build time (post-cycle-220) it carried 79
  including `--help` (78 excluding), 62 of them on `run`
  including `--help`; `--record-notation` is not among them,
  no existing option contains the stem "record", and
  "record notation" / `record_notation` occur nowhere in the
  package or docs. Post-build census: 80 distinct long
  options, 63 on `run`. The coinage was free under every
  method; the audit is recorded in the cycle-221 build log
  and the research log.
- "Notation" has no shipped sense: `cli.py` contains no
  occurrence; USAGE's only hit is the substring in "an
  annotation" (`docs/USAGE.md:2693`). The word is used in its
  ordinary sense — the written form in which the records are
  expressed — which is exactly the treatment: identical records,
  two notations.
- Every stem red-team B's naming table killed is absent
  (executed check): not `--payload-*` (placement's prose
  payload sense is reserved, USAGE §8.46 — and the stage prefix
  is renamed to match the coinage, so the reserved word is not
  spent in protocol vocabulary either, curing the §2 breach
  red-team B noted); not `*-format` (`Format` is the catalog
  chat-template class, `core.py:33`); not `*serializ*` (the
  bundled causal word §1 of the design doc disowns, and the
  tree's output-serialisation sense); not `--document-*`
  (documents are the model's *output* JSON in this tree); not
  `--table-*` (tables are report/portability artefacts in
  USAGE, and the prose arm is not a table rendering); not
  `--render*` (`Format.render` owns template rendering — the
  arm outputs are called renderings in prose, as sibling
  designs do, but the flag does not spend the stem); not
  `--encoding-*` (deliforge/fake-turn token-encoding arms);
  not `--prose-json` (hard-codes the pair and reinstates a
  named-first baseline — this instrument has no baseline arm,
  §6, and a v2 notation joins under the same flag without
  renaming).
- "Syntax" was considered and rejected: it already carries the
  tool-call syntax-adherence sense in CLI help (`cli.py:374`)
  and the parked `--document-syntax` coinage spends it for
  output documents.
- The nearest parked coinages (`--document-syntax`,
  `--response-directive`, `--multi-question`) do not collide.
  The USAGE section number is assigned at build time (the next
  free §8 section), as in sibling cycles.

## 10. Uniqueness — the combination, and the honest residue

What each shipped neighbour measures, and why none is this
instrument (inventory verified in two review rounds): rotation
varies block *order* in prose; placement varies *question
position* around a prose payload; distraction/needle vary
filler *dose* / fact *depth*; canonical-form varies *sub-lexical*
Unicode normalisation (NFC/NFD) of the same prose; negation
varies instruction *wording*; field-order varies *schema
presentation order* — its input is context prose plus a schema
description and its answer is a JSON document, never a
serialised data table in two notations; structured measures
*output* JSON adherence. The surviving residue (§7.5): the
**input-side serialisation of the same generated multi-record
table inside the user message, template family held fixed per
candidate, is owned by no shipped probe**. The contribution is
that combination — local checkpoint × chat-template families ×
frozen-renderer input contrast × paired byte-exact verdicts ×
digest-verified renderings with a validator mirror — stated as
a combination, with no priority claim.

The residue's size is stated as plainly (S8): the genuinely
new components are one table generator, one prose renderer,
and the diagnostic wiring, built on canonical-form's stage /
validator / reporting skeleton. The renderer is precisely the
component that failed in §2 (G1); §3.2 is this restatement's
answer, and a fresh review should attack it first.

## 11. Sources (as adjudicated in design-doc §5; fenced)

- He et al., arXiv:2411.10541v1 — hosted-GPT prompt-template
  format sensitivity (MMLU 59.7 JSON vs 50.0 Markdown, Table 1).
  Fencing: bundles the whole prompt template; no chat-template
  crossing, no local checkpoint, no winner re-selection.
- Sclar et al., arXiv:2310.11324v2 — Abstract only: up to 76
  accuracy points on LLaMA-2-13B from meaning-preserving
  formatting; weak cross-model correlation. (The §3.1 grammar
  detail and venue tag stay removed per design-doc §4.)
- Zhang, Chen, Xu, Yang (POML), arXiv:2508.13948v1, §7 Table 3
  — model-specific optimal table formats on hosted models.
  Fencing: 100 styles bundle table syntax with other styling.
- Tam et al., arXiv:2408.02442, §4.1 — emitted key order under
  constrained JSON mode; fenced off (emitted-order ≠
  input-serialisation).
- JSONSchemaBench, arXiv:2501.10868v2 — output adherence only,
  fenced off. The tree pins v2 deliberately (`structured.py:5`);
  a v3 exists (27 Feb 2025) — this candidate keeps the tree's
  pin consciously and does not drift it.

No "first" is claimed anywhere in this text.

## 12. Scope limits of v1 (stated, not discovered later)

No classification or falsification claim (§6.1). No winner
readout (§6.3). No typed values — integers/booleans are a v2
question requiring a frozen per-arm type-rendering table. No
CSV, XML, or YAML arms (CSV flattens and drops types; XML
escaping rewrites visible text; YAML has no declared dependency
or stdlib parser). No length-matched or component-level
attribution (§1, §3.4). No generalisation beyond the two frozen
renderings, the generated code-table population (§2.2), and a
candidate's own template family. The readout's resolution is
one case in 24 (≈4.2 points of discordance rate); the count is
the readout, and no finer claim is printed.


## 9. Fresh-review adjudication (coordinator #65) — BUILD-as-amended

The text below is the fresh review's adjudication
(`recordnotation/ADJUDICATION.md` in the agent workspace),
reproduced verbatim as far as the archived file extends.
**Archival note (stated, not papered over):** the archived
file is truncated — it ends mid-sentence at defect 6 with a
literal truncation marker, at 10,589 bytes. Defects 1–5 and
the verdict survive verbatim. Defects 6–8 and the formal
amendment texts F–H are completed after the verbatim text
from the round's surviving record (the specialists' briefs
in the same directory, the parent agent's cycle-221 brief,
and the long-term memory entry written the same evening,
which agree on all three).

# Fresh-review adjudication — payload-serialization restated candidate `--record-notation` (coordinator #65, 2026-10-28)

**Verdict: NO-BUILD as written. BUILD-as-amended**, under the parent's
amendment test: every required amendment below is text, pins, reporting,
or record-binding; none touches the endpoints (the two frozen renderings
— prose sentences vs canonical JSON — of the same generated table), the
population (N = 24 generated cases, the frozen single-token alphabet,
the 4–6 × 3–4 shape schedule, the closed-book screen), or the headline
scorer (§4.1: `status == "completed"` ∧ `answer == expected` ∧
`boundary_found`). A build that omits any of amendments A–H is not this
verdict's build. If BUILD: do not build in this round — the parent
sequences build cycles (this would be cycle 221, behind 219/220).

Round: three blind specialists (researcher, red teams A and B —
briefs in this directory), coordinator adjudication by its own
execution (scripts in /tmp during the round; every load-bearing figure
below was re-derived by the coordinator or cross-executed by two
parties independently). Tree: main `18e4b4ca2e4f` plus cycle 218's
working-tree changes. READ-ONLY round: nothing built, nothing pushed,
no design-doc append (parallelism guard — the parent reconciles the
append when the tree is quiet). Candidate of record:
`~/workspace/agents_out/payloadserial_restated/candidate.md`.

## What survived blind attack (executed)

- **Injectivity — the previous round's fatal G1 is genuinely cured.**
  Four independent implementations agree: the drafter's exhibited
  inverse (100,000 tables), red team A's own parser/generator (120,000
  random legal tables, 0 failures, 0 collisions; exhaustive small
  domains; injectivity survives *without* within-table distinctness —
  4,352 exhaustive duplicate-allowed tables — and with scaffolding-word
  values and prefix-related field names), the researcher's sampler
  (~44,000 round-trips, 20,000-table collision search, 0 collisions),
  and the coordinator's renderer+parser written from §3.2's text alone
  (18,000 population tables + 2,800 totality tables incl. prefix field
  names + 2,000 duplicate-value/record tables; seed differs from the
  drafter's). The worked example (§3.3) reproduces **byte-identically
  from the text alone** (prose 317 / JSON 286 chars — the coordinator's
  independent implementation of the printed function produces the
  printed bytes), and the §3.4 worst case is exact (prose 653 / JSON
  594 at 6×4). G1's four crafted collision values are rejected by the
  frozen alphabet (all parties). No backslash appears in any JSON
  rendering over the legal population (escaping unreachable — S3's
  visible-text divergence is cured as a side effect).
- **The OC table (§6.1) is exact.** Every printed figure reproduces
  under three independent exact enumerations (drafter, red team B by
  an independent DP method, coordinator; researcher makes four):
  transplant fire 0.8897 with FWER E = 21.35 / P(≥1) = 1.0000; Rule A
  0.6500 (independent), 0.1648 (paired φ=0.5), 0.0298 at δ=0.05 with
  FWER E = 0.71 / P = 0.516; McNemar fire 0.0061 / 0.0005; the N=96
  branch (Bonferroni false-fire 0.0008, 6,912 generations). The
  descriptive-only drop and the winner drop (§6.3 arithmetic
  0.95²⁴ ≈ 0.292, 0.90²⁴ ≈ 0.080, 0.80²⁴ ≈ 0.005 — verified by all
  parties) are forced by these numbers, not chosen.
- **Decomposition totality (§4.2/§4.3).** The construction is negation's
  shipped one (`negation.py:543-568`, read directly); red team A's
  brute force over concrete discordant combinations found 10 combos,
  0 double-counted / unlabelled / mislabelled. The both-differ class
  §2 lacked is named (`mixed`, reported, never attributed) — S5 cured.
- **Decidability (§5.3) holds, with two pins (amendments D).** The
  verdict's status conjunct is discharged structurally: stage cells
  are recorded only from `aggregate_eligible` reports (the sibling
  gate at `experiments.py` ~:9515, cited by §4.1), and
  `aggregate_eligible` requires a completed report, whose invariants
  (`core.py:313-330`, `:343-345`) force every recorded observation's
  status to "completed" — so digest-equality ∧ `boundary_found`
  recomputes the verdict exactly for every recordable cell. Red team
  A executed the edge the text glosses: a raw `core.evaluate`
  observation CAN carry answer == expected ∧ boundary_found ∧
  status = cancelled — such an observation never becomes a cell under
  the gate, but the candidate never states the invariant its §5.3 row
  rests on. The digest-bound `answer_text` (cycle-190 pattern) makes
  every §4.4 diagnostic genuinely recomputable; the screen row is
  decidable from digest + boundary flag (needle's digest-only binding
  precedent). The id/token row states the correct weaker claim —
  arithmetic consistency only, diffreports has no tokenizer — matching
  the shipped canonical-form mirror's actual standard
  (`diffreports.py:11300-11331`: token counts are stage-asserted,
  type-checked, and bound by `ids_identical` ⇒ equal counts and
  `id_length_delta` == difference).
- **Screen arm (§2.4).** The guess probability is exactly 1/216,000
  per case for ANY guessing strategy (the expected value is uniform
  over the code space by generator construction — red team A's
  execution: round-number priors buy nothing; reaching E = 1 exclusion
  per panel needs a 375× inflation). Coordinator's arithmetic:
  expected exclusions 0.0027 per panel, P(≥1) ≈ 0.27% panel-wide,
  1.11e-04 per candidate. The arm's effect on printed numbers is
  disclosed (§6 prints `n_screened` / `n_closed_book_incomplete` first
  and the unscreened paired discordance as a sensitivity — red team
  B's disclosure sub-attack failed on the text). Retention follows
  design-doc §2's mandate and §7.3's adjudication ("hygiene, not a
  defect of principle; state the cost and justify or drop") — the cost
  is stated in §2.4, §8, and (at build) CLI help, and the justification
  (fail-closed guard against generator/code-space drift; pass count as
  leak readout) is the one §7.3 accepted.
- **Uniqueness (§10).** Verified by direct reading (researcher) and
  coordinator spot-execution: field-order's aggregate computes paired
  discordance between schema-*presentation orders* over structured
  specs whose answers are JSON documents — it never serialises a data
  table in two notations as input; structured owns output adherence;
  canonical-form varies sub-lexical normalisation. The residue is as
  thin as §10 itself states (one generator, one prose renderer,
  diagnostic wiring on canonical-form's skeleton) and is stated, not
  hidden (S8's disposition).

## Defects in the candidate as written (each executed; classification)

1. **§2.3's generator is not a total function of the text (red team A
   kill 1; coordinator-confirmed).** "Draw (word, number) pairs" names
   no RNG calls and no within-pair draw order. Faithful readings
   diverge: word-first vs number-first vs single-`randrange(216000)`
   differ on 24/24 tables at seeds 0/1/42 (A); the coordinator's
   word-first vs number-first implementations produce different tables
   at seed 0, case 0 (`marble-7890…` vs `nickel-7311…`), and the
   queried index differs across readings on most cases. The population
   *distribution* is fully pinned (alphabet, shapes, without-replacement
   distinctness, row-major fill, query as next draw); the seed→table
   mapping that §5.2's validator regeneration and §1(b)'s decidability
   claim rest on is not. Since stage and validator mirror are separate
   literal implementations, the build would have to invent the mapping
   — exactly the looseness-at-the-novel-point pattern. **Amendment A.**
2. **§2.3's answer-embedding audit overstates its guard set (red team
   A kill 2; coordinator-verified against `audit_alphabet.py`).** The
   executed audit replaces `{key}` with "x" in the question template,
   so the printed "0 violations" silently drops design §2's mandated
   subject — the instantiated key. The instantiated question embeds
   the key value, which shares a 4-char prefix with its own word for
   24/24 possible key words, and the answer's word equals the key's
   word in ≈4.17% of cases (analytic 8999/215999; A executed
   2,064/48,000 = 4.30%). The phenomenon is arm-symmetric (the question
   is byte-identical across arms) and cannot manufacture discordance;
   the defect is the audit sentence claiming a guard the execution
   didn't run. **Amendment B** (reporting cure only — a redraw rule
   would change the population and is NOT authorised by this verdict).
3. **§4.4's parsed ⊆ echo claim is false (red teams A and B,
   independently; coordinator-confirmed).** Executed counterexample:
   the answer `{"code": "amber\u002d1001"}` parses via `json.loads` to
   the expected value (a parsed-value hit) while the literal expected
   string never appears in the answer text (no substring hit). JSON
   string escapes break the claimed subset relation; the two rates are
   independent diagnostics. One printed sentence is wrong.
   **Amendment C.**
4. **§5.2/§5.3 omit one sibling binding and one invariant (red team B
   partial kill, sustained in narrow form; coordinator-verified in
   source).** Every shipped mirror binds each cell's `input_tokens`
   to that arm's carried token count (`diffreports.py:10507`
   placement, `:11329` canonical-form); the candidate carries both
   fields but never enumerates the equality, and B's executed forgery
   (token fields fabricated, all §5.3 checks green) passes where the
   sibling standard rejects. The candidate also never states the
   aggregate-eligible-only recording invariant (item under
   "Decidability" above). Both are record-binding text.
   **Amendment D.**
5. **§6.1's power columns state no model (red team A kill 3;
   coordinator-verified against `oc_tables.py`).** The null columns
   are exact (above), but "Power, 25-pt gap" = 0.3547 is exact only
   at marginals .85/.60 and "Power, 20-pt gap" = 0.2598 only at
   .90/.70 — two different, unstated bases; at the table's own
   competent base (.90), a 25-pt gap gives McNemar 0.4000 and the
   N=96 Bonferroni power is 0.8329, not the printed 0.7452 (A's exact
   enumeration). Gap alone does not determine power. The
   descriptive-only decision survives every model (all parties), so
   the defect is confined to the table's labelling. **Amendment E.**
6. **§6 prints no resolution beside its counts (defect
   completed from the round's record).** The discordance
   count is the headline readout, but the section as written
   prints the count without the panel's resolution — at
   N = 24 one discordant case is 4.17 points of discordance
   rate, and a reader given only the count will read it more
   finely than the panel supports. **Amendment F** (reporting
   cure: print the resolution beside every discordant count,
   in the design text, the report section, and the CLI/USAGE
   surfaces).
7. **§8's "4.0× margin" claim is false (defect completed
   from the round's record).** The 2,048-character answer
   cap the budget stresses to IS the tree's 64-token bound
   (`fieldorder.py`), not "far beyond a 64-token decode";
   and a non-ASCII-stressed panel measures about 14.2 MiB
   (about 25,856–25,987 bytes per record) — over the 12 MiB
   cap. The panel fails closed via the frozen budget detail
   (`record_notation_record_budget_exceeded`), so the
   behaviour is safe; the printed claim is wrong (4.0× holds
   only for ASCII, and measures 3.96× on the built record
   shape). **Amendment G** (correct the claim everywhere it
   renders; keep the fail-closed budget behaviour).
8. **§9's "69 flags" census is wrong (defect completed from
   the round's record).** By parser instantiation the tree
   at review time carried 77 distinct long options including
   `--help` (76 excluding). The coinage itself was free under
   every counting method — the defect is the printed census,
   not the name. **Amendment H** (re-audit the flag name
   against the post-cycle-220 namespace by parser
   instantiation at build time and record the audit in the
   build log and the research log).

The eight amendments, as built in cycle 221: **A** freeze
the RNG call sequence in text and code (§2.3). **B** print
and account for the instantiated-key overlap (about 4.17
percent, analytic 8999/215999) — reporting only, no redraw
(§2.3). **C** correct the §4.4 parsed/echo claim — the two
rates are independent diagnostics (§4.4). **D** bind
`input_tokens == arm_tokens[arm]` in the validator mirror
and state the aggregate-eligible-only recording invariant
(§5.2/§5.3). **E** state the §6.1 power columns' marginal
bases and print the base-.90 figures beside them (§6.1).
**F** print the resolution beside the discordance counts
(§6). **G** correct the §8 margin claim; keep the fail-closed
budget (§8). **H** re-derive the flag census by parser
instantiation (§9). Plus the researcher's condition C: the
scope phrase "for this checkpoint and this candidate's
template family" in the §1 utility, the CLI help, USAGE,
and the report section. The fabrication sweep for the round
was CLEAN. Build sequenced as cycle 221, behind 219/220;
built and closed in cycle 221 (see the research log).

