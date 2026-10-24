# Schema field-order probe — design (cycle 205, wave-9 rank 4)

Status: v2, adjudicated after the adversarial round
and shipped in cycle 205
(`agents_out/cycle205/{redteam_a,redteam_b,verifier,
coordinator_adjudication}.md`: A filed 2 FATAL / 9 MAJOR /
4 MINOR, the verifier reproduced every load-bearing number,
B confirmed both FATALs with corrections and added NB1–NB6).
v1's defects that shaped this rewrite: answer labels the
contexts do not entail (F1), a self-contradictory non-finite
sentinel with an undefined canonical form (F2), overstated arm
distinctness (F3), unfrozen new specs (F4), and a decision rule
whose printed operating characteristics understated its
weakness (F5/F6).

## Utility

A practitioner defining a JSON schema for extraction learns, per
template family, whether putting the main answer field first vs
last in the schema changes the values the model produces for it —
before a cosmetic schema reorder in a code review silently changes
extraction accuracy in production.

## Question and claim

For a fixed candidate (chat template) and a fixed set of flat JSON
extraction/classification/routing specs, is the value produced for
a spec's designated answer field invariant under the field's
position in the presented schema?

Pre-registered claim: **per-field value accuracy is invariant
under schema field order.** Proved wrong, per candidate, if the
designated field's paired accuracy differs between the
answer-first and answer-last arms on specs adherent in both arms,
under the decision rule below. Intermediate arms (canonical,
reversed) are descriptive context, never decision inputs.

Motivation, stated precisely: Tam et al. (arXiv:2408.02442)
establish that format restrictions degrade reasoning; their
abstract makes no key-order claim. Their §4.1 does report direct
key-order evidence of a different kind: 100% of GPT-3.5 JSON-mode
Last-Letter responses placed the "answer" key before the "reason"
key, "resulting in zero-shot direct answering", and the authors
conclude that "the order of keys in structured outputs … emerge
as important factors". That is **emitted** key order under
constrained JSON mode, on a reasoning task — not the order in
which a prompt presents a schema's fields. The
presentation-order hypothesis — a field the schema presents
earlier is also emitted earlier, with less of the object's
content in context — is the lab's own reasoning, built on Tam et
al.'s two verified results. JSONSchemaBench (arXiv:2501.10868,
cited by `structured.py`) varies constrained-decoding frameworks
and models, not prompt-only field order crossed with chat
templates. The combination — prompt-only field order × template
family × per-field exact scoring — is the contribution; no
"first" is claimed.

## Spec set

The shipped `structured` spec set (9 specs) is reused unchanged,
and six further specs in the same flat dialect are added in
`fieldorder.py` (the `structured` set itself is not modified).
Reason for the addition, stated plainly: the shipped schemas have
at most 2 fields, and three have exactly 1, so on the shipped set
alone the arms collapse to at most two distinct orders per spec
and the reversed arm is rarely a distinct treatment. The new
specs carry 3–4 fields.

Canonical order is the order the shipped instrument presents:
**alphabetical by field name** (`structured.describe_schema`
sorts properties and the required list). Each spec designates one
**answer field** and a frozen **paired** flag. Pairing rule
(derived, from frozen data): a spec enters a candidate's paired
population iff `paired` is yes, its first/last orders differ, both
endpoint records are adherent, and neither endpoint record has
duplicate keys.

**Reader-entailment rule (F1):** a spec may be `paired: yes` only
if a reader given the question, context, and schema — but not the
answer table — can derive the answer field's value uniquely.
Two shipped specs fail it and are `paired: no` (kept as
adherence/descriptive specs; the shipped set cannot be edited):
route-alert (`channel = pager` — nothing maps 97% disk usage to
pager over sms/email) and route-order (`warehouse = south` —
nothing entails it; Porto is in northern Portugal). Non-answer
fields of shipped specs may carry frozen judgement values
(route-ticket's `priority = 3`); they feed descriptive per-field
readouts only and are named here so the noise is not discovered
later.

The six new specs, frozen in full (schema dialect exactly as in
`structured.py`; every answer passes `check_schema` against its
own schema, verified at module load in tests):

- **extract-product** (extraction, paired yes). Q: "Extract the
  product's brand, model, price and stock status." Context: "The
  catalogue lists it as: brand Acme, model Rotor X2. It costs 349
  dollars and is currently in stock." Schema: brand (string),
  model (string), price (integer ≥ 0), in_stock (boolean).
  Answer: {brand "Acme", model "Rotor X2", price 349,
  in_stock true}. Answer field: model. (The catalogue split is
  stated verbatim in the context, so the answer field is
  determined, not convention-dependent.)
- **extract-event** (extraction, paired yes). Q: "Extract the
  event venue, day and attendance." Context: "The workshop meets
  at Harbor Hall on Friday with 40 attendees." Schema: venue
  (string), day (string, enum monday…sunday, all lowercase),
  attendance (integer ≥ 0). Answer: {venue "Harbor Hall",
  day "friday", attendance 40}. Answer field: venue.
- **route-complaint** (routing, paired yes). Q: "Route the
  complaint and record whether a defect was confirmed and
  whether data was lost." Context: "Complaint: 'The update
  deleted my saved files.' Front-line support confirmed the
  update itself is defective." Schema: team (string, enum
  [sales, support, technical]), defect_confirmed (boolean),
  data_lost (boolean). Answer: {team "technical",
  defect_confirmed true, data_lost true}. Answer field: team.
  (Every field is entailed: files deleted, defect confirmed by
  support, and a confirmed product defect routes past front-line
  support to technical.)
- **classify-document** (classification, paired yes). Q:
  "Classify the document's type, language and confidentiality."
  Context: "Document: a German-language internal memo marked
  confidential." Schema: doc_type (string, enum [memo, report,
  invoice]), language (string, enum [english, german, french]),
  confidential (boolean). Answer: {doc_type "memo",
  language "german", confidential true}. Answer field: doc_type.
- **extract-vehicle** (extraction, paired yes). Q: "Extract the
  vehicle make, year, fuel and seat count." Context: "The 2019
  Volvo wagon runs on diesel and seats five." Schema: make
  (string), year (integer 1900–2100), fuel (string, enum
  [petrol, diesel, electric]), seats (integer ≥ 1). Answer:
  {make "Volvo", year 2019, fuel "diesel", seats 5}. Answer
  field: make.
- **route-delivery** (routing, paired yes). Q: "Route the
  delivery and flag its requirements." Context: "Delivery:
  vaccines to a clinic in the south, sent by express courier for
  tomorrow morning; a signature is required and the box must stay
  refrigerated." Schema: region (string, enum [north, south,
  east, west]), speed (string, enum [standard, express]),
  signature_required (boolean), refrigerated (boolean). Answer:
  {region "south", speed "express", signature_required true,
  refrigerated true}. Answer field: region.

Designation table (canonical = alphabetical order; position of
the answer field within it):

| spec | canonical order | answer field (position) | paired |
|---|---|---|---|
| extract-person | age, name | name (2/2) | yes |
| extract-city | city, population | city (1/2) | yes |
| extract-book | title, year | title (1/2) | yes |
| classify-sentiment | label | label (1/1) | no (one field) |
| classify-urgency | level | level (1/1) | no (one field) |
| classify-language | language | language (1/1) | no (one field) |
| route-ticket | department, priority | department (1/2) | yes |
| route-alert | ack_required, channel | channel (2/2) | **no (unentailed)** |
| route-order | express, warehouse | warehouse (2/2) | **no (unentailed)** |
| extract-product | brand, in_stock, model, price | model (3/4) | yes |
| extract-event | attendance, day, venue | venue (3/3) | yes |
| route-complaint | data_lost, defect_confirmed, team | team (3/3) | yes |
| classify-document | confidential, doc_type, language | doc_type (2/3) | yes |
| extract-vehicle | fuel, make, seats, year | make (2/4) | yes |
| route-delivery | refrigerated, region, signature_required, speed | region (2/4) | yes |

Paired pool: **10 specs** (extraction 6, routing 3,
classification 1). Across them the answer field sits first in 3
canonical orders, last in 3, and in a middle position in 4, so
the endpoint contrast is not confounded with "the answer field
is always the alphabetically-first field".

## Arms

Per spec, four renderings of the same prompt, differing **only**
in the field enumeration order inside the schema description
(the property list and the required list both follow the arm
order):

- `canonical`: alphabetical (the shipped presentation).
- `answer_first`: the answer field moved to the front; the
  remaining fields keep canonical relative order.
- `answer_last`: the answer field moved to the end; the
  remaining fields keep canonical relative order.
- `reversed`: canonical order reversed (generic-reordering
  control).

**Distinctness, stated exactly (F3):** the four orders are all
distinct only when the answer field sits in a middle position
(4 specs). When it sits at a canonical endpoint with ≥ 3 fields,
there are 3 distinct orders (canonical coincides with that
endpoint arm; 2 specs). With 2 fields there are 2 distinct
orders (6 specs); with 1 field, 1 (3 specs). Canonical therefore
coincides with an endpoint arm on 6 of the 10 paired specs, and
its readouts there are replicates of that endpoint arm under
greedy decoding — labelled as such wherever canonical numbers
are printed next to endpoint numbers. Per candidate the stage
runs 60 arm-generations covering **37 distinct prompts**; the
duplicate generations are kept (uniform code path and record
shape), and the cost is printed as 60, not 37. The decision is
unaffected: answer-first ≠ answer-last on all 10 paired specs.

The prompt template is `structured.render_prompt`'s, with the
schema description rendered in the arm's order by an
order-parametrised describe function whose canonical output is
byte-identical to `structured.describe_schema` (pinned by a
test). Shipped specs keep the `(ref: structured/<id>)` marker;
new specs use `(ref: fieldorder/<id>)`. The schema presentation
and the implied output key order move together — that is the
production change being tested; no separate output-order
instruction exists in the shipped prompt, and none is added.

## Records and recompute (residue stated, §Records)

Generation mirrors the density/rotation stages: one record per
(candidate, spec, arm), exactly 15 × 4 = 60 per candidate in
canonical (spec, arm) order with unique (spec, arm) keys — the
validator rejects any other record set for a candidate (NB1).
Records come from `core.evaluate` at MAX_NEW_TOKENS = 64, fixed
by the protocol rather than threaded from the CLI: the record
caps below are derived from it (NB4). Budget cap: 24 candidates
→ MAX_FIELD_ORDER_RECORDS = 1,440. Screening mirrors density:
the stage-level special-token audit runs over the frozen text
inventory (questions, contexts, templates, description
components), and a per-spec boundary-containment screen runs
over the rendered prompts (NB5). Stage details {cancelled,
field_order_record_budget_exceeded, field_order_screen_failed};
per-record detail ∈ {"", arm_evaluation_failed,
arm_context_exceeded}.

Record shape (frozen):

```
{candidate, spec_id, arm, detail, case_screened, boundary_found,
 duplicate_keys, answer_digest, document_kind, document_digest,
 field_values, input_tokens, content_digest}
```

Digest preimages (NB2): `content_digest` = SHA-256 of the arm's
rendered user content (the prompt string), recomputed by the
validator from its literal spec table. `answer_digest` = SHA-256
of the decoded answer text, or None when there was no generation.

- `document_kind`: "object" | "non_object" | "unparseable" | ""
  ("" exactly when there was no generation). `unparseable` means
  JSON parse failure **only**.
- `duplicate_keys`: a stage-observed bool — the raw answer
  contained a repeated object key. Scoring consequences
  (derived, below): a duplicate-key document is never adherent
  and never enters pairing, because its extraction is
  parser-dependent (Python's `json.loads` keeps the last
  occurrence; a production parser may keep the first). Per-field
  readouts score the last-wins parse, with duplicate counts
  printed beside them.
- `field_values`: for "object" documents, a dict mapping every
  produced key to the **canonical JSON string** of its parsed
  value (canonical form pinned below), or to None when the
  produced value has no canonical form — a non-finite number
  (NaN/±Infinity, which `json.loads` accepts as a Python
  extension) or a value beyond the int↔str digit limit. A produced JSON
  null is the string "null", never None — the sentinel is
  unambiguous. {} for every other kind, and for degraded
  records (detail set or screened: kind "", digest None,
  field_values {}).
- `document_digest`: the digest of the document's canonical
  form, stored **only** where the validator recomputes it:
  non-None ⟺ kind is "object" and no `field_values` entry is
  None; then it must equal the digest of the reconstructed
  document. Otherwise None.

**Canonical JSON, pinned (F2c):** `structured._canonical_json`
exactly — `json.dumps(value, sort_keys=True,
separators=(", ", ": "), ensure_ascii=True, allow_nan=False)`.
Worked example: the document {"name": "Ada Lovelace", "age": 36}
canonicalises to the string `{"age": 36, "name": "Ada Lovelace"}`
(note the spaced separators — this probe's canonical form is the
`structured` convention, not density's compact one).

Everything scored is **derived** from the record fields, by the
producer's aggregator and the diff validator's mirror alike:

- A field's produced value is **correct** iff its key is present
  in `field_values` with a string whose digest equals the
  expected value's canonical digest (expected values come from
  the frozen spec table). A None (non-finite) value is incorrect
  for every field, since no expected value is a non-finite
  number. Correctness is exact canonical equality; every numeric
  field in the spec set is `integer` (F11), and `number`-typed
  fields are out of scope for answer-field designation.
- The schema verdict is recomputed by re-running the flat schema
  check over the reconstructed document (each `field_values`
  string parsed back; the validator requires every stored string
  to be exactly the canonical form of its parse, and requires
  `document_digest` consistency as above — so the stored values
  are bound, not attested). A None value fails its key's field
  check **before** reconstruction, so a null-typed field can
  never be passed by the sentinel (no current spec has a null
  field; the rule is frozen anyway). "non_object" and
  "unparseable" documents are schema-failed by kind.
- **Validator totality (F10):** any exception while parsing or
  canonicalising a stored string — ValueError (including the
  int↔str digit limit), RecursionError, TypeError — makes the
  section invalid, never a crash. Both mirrors use a safe
  canonicaliser with that catch (`classify_response`'s
  RecursionError handling is the in-tree precedent).
- Adherence per record: parse_ok iff kind ∈ {object,
  non_object}; adherent iff parse_ok, the recomputed schema
  verdict is ok, and not `duplicate_keys` — `structured`'s
  parse/schema decomposition, kept, so a value effect cannot
  hide inside an adherence effect.

**Caps (F8, derived from enforced premises):** per record, the
summed length of `field_values` strings is at most 4,096 chars,
validator-enforced. Derivation: MAX_NEW_TOKENS = 64 and at most
32 decoded chars per generated token for the tokenizers this
lab targets bound an answer at 2,048 chars; the cap is 2× that.
Producer rule: a live record whose values exceed the cap is
stored with detail `arm_evaluation_failed` and empty
`field_values` (counted under n_detail_screened). Payload cap
MAX_FIELD_ORDER_PAYLOAD_BYTES = 8 MiB: 1,440 records × (4,096 +
≈ 1.4 KB of other fields and JSON overhead) ≈ 7.9 MB.

**Stated residue (F9):** the link between `answer_digest` and
(document_kind, field_values) is stage-attested — the validator
recomputes every verdict from the record fields, but it cannot
prove the fields came from the answer the digest names (the
same origin-only residue density discloses). `duplicate_keys`
and `boundary_found` are stage-observed facts of the same
status; every consequence drawn from them is recomputed.

## Aggregation and decision

The aggregate candidate block always carries `detail` ("" or
"field_order_unavailable") plus every readout key below, so a
zero-live candidate prints None rates and an `insufficient`
decision with n_paired = 0 rather than vanishing (NB6). Live =
not screened, no detail, boundary found.

- **Adherence by arm**: n, parse/schema/adherent rates over live
  records (structured's decomposition; rates None iff n = 0).
- **Per-field accuracy by arm**: for every field of every spec,
  correct / live-in-arm (missing counts incorrect; denominators
  printed). Plus whole-document accuracy by arm — all spec
  fields correct **and** no extra keys in `field_values` —
  over live records (F15).
- **Answer-field accuracy by arm** over all live records, and
  over the paired population below.
- **Pairing and attrition**: per candidate, the paired
  population per the rule above; attrition printed per arm
  (specs lost to non-adherence, to duplicate keys, to the
  paired flag / order rule). Beside the paired counts, the
  **ungated discordant counts** — all `paired: yes` specs whose
  endpoint orders differ, scoring non-adherent records' answer
  fields by their `field_values` — are printed as a sensitivity,
  because adherence gating is outcome-dependent selection: a
  generation failure that makes the answer field wrong in one
  arm often makes that record non-adherent and removes the spec
  from pairing, deflating measured discordance (F6; the
  deflation is bounded in this instrument — most incorrect
  values here are adherent-shaped, e.g. a wrong enum member —
  but the direction is disclosed, not hidden).
- **Decision**: per paired spec, answer-field correct in first
  / correct in last; discordant counts n_first_only,
  n_last_only; exact two-sided sign test over discordant specs.
  `falsified` iff discordant ≥ MIN_DISCORDANT_SPECS = 6 and
  p ≤ 0.05 (direction recorded); `insufficient` iff discordant
  < 6 — the boolean is withheld, never defaulted to acquittal;
  else `not_falsified`. Printed sensitivities: the same decision
  recomputed on the extraction-only paired specs (6), and the
  paired counts split by field count (2-field vs ≥ 3-field —
  on 2-field specs the answer field's promotion *is* the
  companion's demotion, F13).
- **Operating characteristics, printed wherever the verdict is
  (F5/F12):** critical splits — at 6–8 discordant specs only a
  unanimous split falsifies; at 9, 8–1 (p = 0.0391); at 10,
  9–1 (p = 0.0215). Power: a true 0.90-vs-0.60 endpoint accuracy
  gap yields `falsified` only ≈ 15% of the time (simulated,
  independent specs); most such runs print `insufficient` or
  `not_falsified`. The sign test's α is exact under independent
  discordance directions across specs; specs are a fixed panel,
  not a sample, and within-shape direction correlation degrades
  the calibration (at the extreme of one direction coin per
  shape, conditioned on full discordance, the false-falsification
  rate reaches ≈ 50%; unconditional rates stayed ≤ ≈ 4.4% across
  the round's parameter sweep). Per-candidate α = 0.05 gives
  ≈ 1.2 expected false falsifications across 24 candidates at
  the nominal rate (probability of at least one ≈ 0.71); the
  realised per-candidate rate is below nominal by discreteness.

Protocol object: {seed, backend fingerprint, spec/arm inventory,
and digests of three frozen rule texts — GENERATION (spec set,
designations, paired flags, arm derivation), RENDERING (prompt
construction), CHECKING (canonicalisation, sentinel, duplicate
rule, correctness, schema recompute, pairing, decision)} —
mirror-recomputed by the validator from its own literal copies,
as in cycles 203/204.

## Surfaces

- Pure module `fieldorder.py` (spec table, arm derivation,
  rendering, checking, aggregation, protocol).
- `experiments.fieldorder_stage` / `fieldorder_analysis`
  mirroring `density_stage`; CLI `--field-order` with the
  utility sentence and the 1,440-generation cost in help.
- Reporting section "Schema field order"; diff section
  `field_order` whose validator recomputes everything above
  from the records (full literal-table mirror in diffreports).
- USAGE §8.52; RESEARCH_LOG cycle-205 entry; CHANGELOG;
  VALIDATION counts by measurement.
- Pin blast radius (NB3), all in the same push: test_cycle44
  mypy source-file pin 61 → 62 (fieldorder.py is the 62nd) and
  its VALIDATION.md line; test_cycle178 header pin "cycles
  10–204" → "cycles 10–205"; test_cycle27 + test_cycle67 oracle
  elements for the new diff section (oracle trio, identical key
  set, summary literal, flip-gate anchor original + mutant);
  README capability bullet + sample-report and diff-section
  enumerations; VALIDATION probe list.

## Known limits (printed in the report)

- Flat schemas of 1–4 fields; endpoint placement only — nothing
  is measured about middle positions as a treatment.
- Canonical coincides with an endpoint arm on 6 of 10 paired
  specs; single-field specs cannot enter the pairing; the
  reversed arm coincides with an endpoint on 2-field specs.
- Two shipped specs are excluded from pairing because their
  frozen answer labels are not entailed by their contexts
  (route-alert, route-order); one shipped non-answer field is a
  frozen judgement value (route-ticket priority).
- Adherence gating selects specs the candidate already handles,
  and selects against discordance; the paired claim is
  conditional on adherence in both endpoint arms, with the
  ungated counts printed as sensitivity.
- Duplicate-key documents are scored by their last occurrence
  in per-field readouts and excluded from adherence/pairing.
- 10 paired specs and a sign test: only very large endpoint
  effects can falsify, and they usually will not (≈ 15% power
  at a 30-point gap). The verdict's printed characteristics
  say so.
