# Additional validated subsystem findings

**Source:** `d8a1346343e1b50b28bf567ab6b193b38fc79608`.
**Checked:** 2026-11-10. **Status:** findings and recommendations only.

Four completed bounded specialists supplied exact-path/range notes and safe
synthetic reproductions. The coordinator read their reports and probe sources,
then re-executed all four probe groups successfully. Results are preserved in
[evidence.json](evidence.json); review depth is in [inventory.csv](inventory.csv).
Neither synthetic backends nor control-flow spies constitute actual model,
network-stack, accelerator or operating-system signal certification.

IDs retain the specialist namespace so root causes remain traceable. RTI-01
is already RR-10 in [REPORT.md](REPORT.md), not a second distinct finding.
Priorities describe readiness risk, not vulnerability scores.

## Finding and work-package map

| ID | Priority | Finding | Package |
| --- | --- | --- | --- |
| SA-01 | P2 | Template evaluator misses result validation | W2 |
| SA-02 | P2 | Boundary exclusions do not conserve counts | W6 |
| SA-03 | P2 | Duplicate tool arguments get full credit | W2/W6 |
| SA-04 | P2 | Custom numeric schema can overflow | W2/W6 |
| SA-05 | P2 | Custom enums conflate bool and number | W2/W6 |
| SCI-B-01 | P1 | Built-in semantic answer key is ambiguous | W6 |
| SCI-B-02 | P2 | Prefix sharing is called exact duplication | W6 |
| SCI-B-03 | P2 | Few-shot comparison mixes fixture cohorts | W6 |
| SCI-B-04 | P2 | Cache profile uses quadratic storage | W4/W6 |
| RTI-02 | P2 | Pack produces over-limit bundles | W5 |
| RTI-03 | P2 | Invalid inputs cause expensive work first | W2/W4 |
| RTI-04 | P2 | Serving artifact reads are weak/unbounded | W2/W4 |
| RTI-05 | P2 | Aborted cache runs skip teardown | W4 |
| RTI-06 | P2 | Invalid MCP envelopes execute successfully | W2 |
| RTI-07 | P3 | Debug save exceeds its final byte cap | W4 |
| RTI-08 | P3 | HTTP error-read faults escape classification | W4 |
| ORCH-01 | P1 | Default JSON retains private evidence fields | W2 |
| ORCH-02 | P1 | Optional stages disconnect cancellation | W4 |
| ORCH-03 | P2 | Transfers retain multiple loaded backends | W4 |
| ORCH-04 | P2 | Unavailable cells display measured negatives | W2 |
| ORCH-D01 | P2 | Export lacks version/schema admission | W2 |
| ORCH-D02 | P2 | Report pairs lack a crash completion boundary | W4/W5 |

## Privacy, cancellation and backend lifecycle

### ORCH-01 — default export does not redact every optional section

`reporting._public_view` / `_strip_raw_diagnostics` (lines 298–363) remove
primary observations but retain `transfer.checkpoints[].path`. The ordinary
`run --transfer-model` branch places operator-supplied checkpoint paths there
(`cli.py:4557–4565`). A synthetic sentinel survives default JSON with
`raw_evidence_included=false`; HTML does not expose that path.

Separately, composing `serving_backends_analysis` output into `save_report`
retains `records[].prompt_ids` and `returned_ids` under the same false flag.
Those are real producer fields (`servingbackends.py:1478–1501`), not arbitrary
invented keys. Standard `run` does not automatically create that serving
section. Raw execution artifacts intentionally retain exact IDs for replay;
this finding concerns the distinct evidence-excluded public projection.

**Fix and acceptance:** specify a public/evidence projection for every section;
default JSON/HTML must exclude synthetic private paths and reconstructable raw
streams, while retaining permitted boundary IDs, digests, counts and labels.
Opt-in evidence must be explicit and must not mutate caller-owned input. A new
evidence-bearing section without a projection must fail a completeness check.

### ORCH-02 — cancellation does not span the complete run

`cli.main` restores shared-event signal handlers immediately after the primary
experiment (`4237–4282`), before optional compute. `batch_stage`
(`experiments.py:4234–4336`) creates separate `Event()` objects rather than
receiving the owner event. Healing/confidentiality have related legacy paths.
In the real batch stage, setting an owner event during the first generation
still permits all three generation calls; each receives an unset private event.
This proves event disconnection, not a measured real-signal integration failure.

**Fix and acceptance:** maintain one cancellation lifetime through optional
work and report preparation. A pre-cancelled run must make zero generation or
secondary-load calls; cancellation during the first solo generation must
prevent subsequent solo/batch work. Exercise actual CLI SIGINT/SIGTERM in a
bounded synthetic integration lane, preserve the frozen scientific decision,
and persist explicitly unattempted/cancelled optional cells.

### ORCH-03 — transfer model residency is cumulative

`cli.py:4511–4549` constructs up to four `LocalHFBackend` objects into a retained
dictionary before transfer evaluation. Constructor code (`backends.py:1178–1208`)
eagerly loads and places model parameters; these are not lazy descriptors.
Executing only those loader statements with a recording constructor produces
`load, load, load, evaluate`, with three secondary objects retained. No model
allocation or OOM was attempted, and the primary remains referenced too.

**Fix and acceptance:** lazy factories, at most one secondary live backend,
explicit cleanup before loading the next, and no further loads after cancel.
Use weak-reference lifecycle tests plus a separate approved, bounded real-model
peak-memory measurement; do not mistake the synthetic trace for hardware proof.

### ORCH-04 — unavailable is rendered as an observed negative

`serving_backends_analysis` (`experiments.py:13364–13385`) sets `fired=false`
when no winner exists, even for `capability_absent` / `available=false` cells.
`reporting._serving_backends_section` (`10734–10760`) renders that as “no fire”.
The fidelity verdict still says capability absent, so agreement aggregation is
not claimed to have become a false success.

**Fix and acceptance:** use an explicit null/unobservable state and unavailable
label. Preserve real measured “no fire” and positive-stop cases; JSON and HTML
must distinguish absent, cancelled, unsupported and observed-negative evidence.

## Scientific correctness and measurement populations

### SCI-B-01 — a default position question has two correct answers

`position.POSITION_SPECS[0]` (`position.py:61–67`) asks for a European capital,
includes Lisbon and Oslo, but accepts only Oslo. `build_position_fixtures`
(`217–244`) therefore marks a correct Lisbon answer wrong. This is reachable
in the default fixture set, without malformed input. The
[University of Lisbon's city page][lisbon] independently identifies Lisbon as
Portugal's capital; checked 2026-11-10.

**Fix and acceptance:** replace the ambiguous distractor or narrow the question;
independently review every frozen semantic answer key, not only fixture shape.
Test every position with an independently justified unique correct answer.

### SCI-B-03 — few-shot comparisons mix different fixture cohorts

`fewshot.change_rate` (`fewshot.py:238–263`) intersects each order separately.
`experiments.fewshot_gain` (`2337–2374`) counts the common three-arm population
but computes order rates on different, larger populations. A two-fixture stage
with one unchanged complete fixture and a second fixture missing only order B
yields `n_evaluable=1`, gain 0.25 and sensitivity 0.25. On the one common case,
both statistics are 0.0. The public aggregation function accepts this stage.

**Fix and acceptance:** freeze a common baseline/A/B population for comparative
headlines and common candidates for cross-family comparisons. Keep arm-specific
rates separate with their own denominators. Permute missing arms; dropping a
cell outside the common population must not change its headline. Genuine
complete-case differences must remain detectable.

### SCI-B-02 and SCI-B-04 — cache identity and quadratic space

`cacheprofile._prefix_counts` (`97–104`) stores every `prompt[:end]` tuple.
For one unique length-L prompt this retains L(L+1)/2 token references.
Bounded lengths 256, 512 and 1,024 retained 32,896, 131,328 and 524,800
references respectively. Peak traced allocations grew approximately fourfold
on each doubling. No large-context allocation was attempted.

`candidate_cache_profile` (`156–174`) also uses prefix multiplicity for exact
prompt duplicates. Distinct prompts `(1,)` and `(1,2)` incorrectly give one
duplicate occurrence instead of zero. A six-order independent cold-prefix
oracle still matches the achievable-hit formula; that formula is not alleged
to be wrong by this reproduction.

**Fix and acceptance:** use a compact linear-storage prefix representation and
a separate exact-prompt multiplicity count. Preserve token/block hit formulas
against the independent oracle. Measure scaling, cap aggregate work, and
explicitly refuse over-budget analysis rather than silently truncate prompts.

### SA-02 — boundary-audit exclusion counts do not conserve observations

`boundary_audit.audit_search_observations` (`140–163`) appends unusable checks
on tokenizer failures, then subtracts all appended checks instead of usable
checks. A single decode failure yields `n_observations=1`, `n_usable=0`,
`n_unusable=0`. The usable-only rate filtering itself remains correct.

**Fix and acceptance:** enforce observations = usable + unusable for every
mixture of missing prompt and encode/decode failure, including per-candidate
counts where exposed. Preserve unavailable output for zero usable checks.

### SA-01 — template robustness bypasses generation validation

`surfacenoise._evaluate_template_case` (`2068–2083`) dereferences returns without
`core._valid_generation`, unlike `_evaluate_one` (`403–407`). A synthetic `None`
return raises `AttributeError`; a short `reason=limit` return is credited as a
pass for a four-token cap while the neighboring evaluator correctly abstains.
This requires an adapter-contract violation; no real adapter occurrence is
claimed.

**Fix and acceptance:** share the generation contract, not duplicated guesses.
Exercise None/wrong object/reason/ID/cap/error combinations and valid exact-cap
controls through the actual stage; preserve interrupt and cancellation behavior.

### SA-03 — duplicate tool arguments are graded after information loss

`toolcall._parse_emitted` (`407–412`) uses ordinary `json.loads`, so duplicate
members collapse before argument grading (`499–501`). Inserting a conflicting
null argument before the expected argument in a shipped query still produces
`adherent`, with argument/selection/relevance flags true. Structured/field-order
siblings already handle duplicate ambiguity explicitly.

**Fix and acceptance:** retain/reject duplicate member names before grading;
test both orders, identical duplicates, nested argument objects and valid calls.
Publish the parse taxonomy. Preserve intentional mathematical int/float equality,
not bool/number conflation.

### SA-04 and SA-05 — custom structured schemas have numeric-domain holes

`structured._is_finite_number` (`229–233`) applies float finiteness operations
to arbitrary-precision integers. A valid 401-digit integer under a supported
custom `number` field raises `OverflowError`. Default shipped specs currently
use integer/string/boolean fields; this is not claimed as a default-schema crash.

`structured._field_value_ok` (`555–557`) uses Python membership for enums.
An integer field with enum `[true]` accepts `1`; a boolean field with enum `[1]`
accepts `true`. Its differential test oracle repeats that Python-equality rule.
[JSON Schema core section 4.2.2][equality] distinguishes booleans from numbers;
mathematically equal integer/float JSON numbers remain equivalent.

**Fix and acceptance:** handle finite exact integers without float conversion;
validate huge bounds deliberately; use JSON-type-aware scalar equality. Add
independently derived cross-type enum cases and full-stage numeric tests.
Do not infer correctness from the existing mirror agreeing with production.

## Runtime artifacts and protocol boundaries

### RTI-02 — a successfully packed bundle can fail its own container verifier

`pack.collect_pack_files` / `pack_bundle` (`270–357`) lack producer-side
aggregate entry/byte ceilings. `_preparse_container` (`395–444`) rejects more
than 1,024 entries including the manifest. A sample results file plus 1,023 tiny
files packs successfully into 1,025 archive entries, then verification raises
`bundle_too_many_entries`. The bounded ZIP was 411,681 bytes.

**Fix and acceptance:** reserve manifest/container overhead and enforce the
same limits before buffering/writing. A 1,024-entry total must round-trip; one
more must fail before output creation. Test cumulative-size limits with reduced
budgets, not large allocations; retain strict verifier limits.

### RTI-03 — cheap invalid input causes expensive work before rejection

`mcp_server.tool_probe` (`266–282`) constructs a backend before checking cap/seed
and acquiring the sweep lock. A cap-zero request calls a recording constructor
once before returning `-32602`. `run_exec_backends` (`2183–2297`) loads/probes
before validating several settings: invalid cap/stops combined with a failing
probe returns exit 0 and writes an unavailable artifact. No real model or HTTP
request was made in these control-flow probes.

**Fix and acceptance:** complete cheap scalar/list/URL/path-shape validation
before optional imports, loading, hashing or network access. Reject invalid
caps/seeds/stops/cache inputs and nonfinite/nonpositive timeouts with zero
expensive calls or output files. Genuine valid-input capability failures remain
explicit unavailable results. Guard model construction against concurrency too.

### RTI-04 — serving artifact ingestion bypasses strict bounded readers

`servingbackends.load_records_artifact` (`1860–1935`) uses unbounded
`read_text` / permissive JSON parsing, then unvalidated values as hash keys.
Two small frames, a header with an empty plan followed by a record whose
candidate is a list, raise unhandled `TypeError`. The `--results` branch
(`2261–2278`) independently uses permissive unbounded ingestion.

**Fix and acceptance:** regular-file/bounded strict ingestion, shape checks
before key formation, and fixed public error classes. Test duplicate/nonfinite
JSON, invalid UTF-8, depth/size budgets, list/dict keys and non-regular files.
Preserve valid interrupted prefixes without converting corruption to empty
success. FIFO hangs and large allocations were not attempted in this audit.

### RTI-05 — aborted cache sequences skip end teardown

`run_cache_sequence` (`servingbackends.py:1734–1768`) erases/rechecks identity at
the end only on normal completion. A controlled persistence callback raising
`ExecutionCancelled` leaves only the start erase and identity check executed.
The production persistence budget supplies a real early-exit path. Subsequent
runs erase at their own start; universal future corruption is not alleged.

**Fix and acceptance:** exception-safe teardown after acquiring the sequence;
preserve the original failure while recording cleanup uncertainty. Cancel or
fail after each arm; every acquired sequence must attempt end erase. Do not
promote an incomplete prefix to a successful observation.

### RTI-06 — invalid JSON-RPC envelopes and argument shapes execute

`mcp_server.handle_request` (`502–570`) does not validate version/ID types;
`arguments = ... or {}` coerces falsey non-objects to objects. An actual stdio
subprocess accepts a `jsonrpc: "1.0"` ping with an object ID and accepts
`arguments: []` for protocol info. A following valid ping also succeeds.
This is a local protocol contract defect, not an authentication bypass.

**Fix and acceptance:** validate envelopes before dispatch and distinguish
absent defaults from explicit invalid values. Wrong version/method/ID must not
invoke handlers; list/bool/number/string arguments must return `-32602`.
Retain bounded strict framing, notification policy and next-frame recovery.
The [JSON-RPC specification][jsonrpc] was checked on 2026-11-10.

### RTI-07 and RTI-08 — bounded save and transport-error details

`debug.DebugShell._cmd_save` (`721–749`) separately checks existing bytes and
transcript characters, not final UTF-8 bytes. A 4,999,999-byte scratch file grows
to 5,000,764 bytes with no command error despite its five-million-byte cap.
Enforce the opened regular file's resulting byte size; test exact/over-limit
and multibyte cases without disallowing intentional operator-selected paths.

`LlamacppHttpBackend._call` (`servingbackends.py:820–834`) reads an HTTP error
body inside an exception suite. A body-read `OSError` escapes its sibling
transport handler instead of becoming `_StackUnreachable`. The tested streams
closed; a persistent leak hypothesis was rejected. Normalize bounded response
reads on both paths and test read failure/oversize/closure. A guaranteed CLI
traceback is not claimed because its outer layer catches `OSError`.

## Export admission and crash recovery: design debt, not existing promises

**ORCH-D01:** `save_report` accepts an unsupported `schema_version=999`; it has
no envelope-version/section admission gate. This does not show that the normal
producer emits the wrong version. Define valid primary, partial, failed,
cancelled and standalone-serving envelopes and reject unsupported contracts
before output creation.

**ORCH-D02:** `save_report` (`14528–14558`) writes the final JSON then final
HTML. A reader can observe only the first file before the second open.
The injected second-open Python failure correctly cleans both artifacts;
that protection is worth preserving. A hard crash was not tested. Add an
explicit same-filesystem staging/completion boundary and bounded crash/recovery
acceptance, distinguishing atomic visibility from durability requirements.

## Evidence strength and rejected overclaims

- Actual pure functions, pack/verify, report serialization, the artifact reader
  and MCP subprocess were exercised. Models, real serving sockets, large
  allocations, real signals and hard process crashes were not.
- Sign-test enumeration matched 65 valid count pairs; six cache-order checks
  matched the achievable-hit formula. These support specific formulas, not all
  statistics. Typographic equivalence-certificate calibration remains open.
- Search chooses before held-out evaluation in the sampled pipeline. A held-out
  failure preserves the selected candidate and changes only the outcome.
- Nonfinite JSON is rejected by `save_report`; Python-level second-file cleanup
  works. Unsigned digests do not authenticate origin, and the code says so.
- Known limitations of catalog fidelity, frozen Unicode tables, real tokenizers
  and every unreviewed range remain in the plan; AST mapping is not semantic
  proof. No model's competence or failure frequency is inferred from spies.

[lisbon]: https://www.ulisboa.pt/en/info/city-lisbon
[equality]: https://json-schema.org/draft/2020-12/json-schema-core#section-4.2.2
[jsonrpc]: https://www.jsonrpc.org/specification
