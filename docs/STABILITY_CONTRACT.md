# Stability contract: capability inventory, contract table, boundary invariants

**Status: baseline record for the stable-release program. The baseline is
NOT release-ready.** This document freezes what the product surface is at
one exact source, what each part's contract chain looks like, and which
boundary invariants are established in code today versus still to be
established by a named work package. It is a census, not a certification:
a row that says "established" points at the code or test that establishes
it; a row that says "gap" is an open finding with an owner.

**Baseline source:** `1ffdb64005f87d7b95a9ff98d462241c9054b52f` (`main`,
2026-11-10). The audit baseline was
`d8a1346343e1b50b28bf567ab6b193b38fc79608`; the two commits between them
are documentation-only (see the [stable-release ledger](STABLE_RELEASE_LEDGER.md)
for the manifest reconciliation). Produced by work package W0 of the
[2026-11-10 audit plan](audit/2026-11-10/PLAN.md).

## 1. Capability inventory

### 1.1 Surfaces

| Surface | Identity | Class | Evidence |
| --- | --- | --- | --- |
| Console script | `model-protocol-lab` → `model_protocol_lab.cli:main` | stable candidate | `pyproject.toml` `[project.scripts]`; 12 commands (§1.2) |
| Console script | `model-protocol-lab-mcp` → `model_protocol_lab.mcp_server:serve` | experimental | same; MCP server identity below |
| Python package | `model_protocol_lab` (75 modules + `py.typed`) | mixed — per §2 | package tree at baseline |
| MCP server | name `mpl-surface-noise-probe`, version `1.0.0`, 4 tools (§1.3) | experimental | `mcp_server.SERVER_INFO`, `mcp_server.TOOLS` |
| Docs site | MkDocs Material, published by workflow from `main` | stable candidate | `mkdocs.yml`, Docs workflow |

"Stable candidate" means: the surface is exercised end-to-end by CI on the
baseline and has a written contract in this document. It does **not** mean
release-ready; the ledger lists the findings that still block that claim.

### 1.2 CLI commands (12 top-level, enumerated from the built parser)

| Command | Purpose (one line) | Class |
| --- | --- | --- |
| `catalog` | List the 29 bounded text-only protocol hypotheses as JSON | stable candidate |
| `inspect` | Validate a fixture file without loading a model | stable candidate |
| `doctor` | Pre-flight check of a local checkpoint; never loads weights | stable candidate |
| `run` | Evaluate a local checkpoint and save a report (the measurement core) | stable candidate (core); opt-in probe sections experimental |
| `fixtures build` / `fixtures fuzz` | Author and fuzz fixture files; never loads a model | stable candidate |
| `surface-noise probe` / `diff` / `protocol` | Prompt surface-noise stability probe and its file differ | experimental |
| `verify` | Independently re-derive a recorded decision and diagnostics; `--bundle` re-checks a sealed bundle | stable candidate |
| `pack` | Seal a report directory into a hash-bound bundle | stable candidate |
| `diff-reports` | Compare two experiment result directories or `results.json` files | stable candidate |
| `diff-backends` | Differential-test emulated backend stop semantics | experimental |
| `exec-backends` | Run the serving-backend protocol against a real stack (protocol `exec-backends-protocol-1`) | experimental |
| `debug` | Interactive protocol debugger REPL (offline, tokenizer-only) | experimental |

Exit-code vocabulary is uniform and documented per command in
`docs/USAGE.md` §6: 0 success or positive verdict; 2 negative verdict or
usage error; 1 failure before a report could be produced; 130 cancelled.
(The audit's R4 adjudication record documents three places where the §6
table itself is stale relative to code; correcting the table is W2/W7
documentation work, tracked in the ledger under RR-16.)

### 1.3 MCP tools (4, enumerated from `mcp_server.TOOLS`)

| Tool | Class | Notes |
| --- | --- | --- |
| `probe_surface_noise_stability` | experimental | Bounded by `MAX_REQUEST_CHARS` = 1 MiB and per-fixture caps (§3) |
| `surface_noise_hardening_notes` | experimental | Invalid-input verdict is the frozen `-32602` since cycle 244 |
| `surface_noise_protocol_info` | experimental | — |
| `surface_noise_controls` | experimental | — |

Transport is stdio JSON-RPC with strict framing (duplicate members,
non-finite constants and surrogates are rejected at the frame layer).
Envelope-shape validation gaps found by the audit are RTI-06 (ledger).

### 1.4 Backends

| Backend | Where | Class | Support |
| --- | --- | --- | --- |
| `LocalHFBackend` | `backends.py` | supported via the `hf` extra | Linux proven in CI; Windows install currently broken (RR-07) |
| Prompt-transform wrappers (`TruncatedPromptBackend`, `TokenTruncatedPromptBackend`, `TokenHealingBackend`) | `backends.py` | experimental | wrap any core backend; used by mutation/healing arms |
| `LlamacppHttpBackend` | `servingbackends.py` (`ExecutionBackend` protocol) | experimental | one pinned stack build (llama.cpp b12182), loopback only, API key required; proven by the cycle-243 probes |
| vLLM (any route) | — | **explicitly unsupported** | no adapter ships; the PyPI wheel is CUDA-only and the CPU from-source route is unproven on this host; a W6 work item, not a capability |
| Synthetic backends in tests/fixtures | tests, `examples/` | not a product capability | deterministic stand-ins for tests; never presented as measurement evidence |

The template catalog holds **29 format families** (`catalog.CATALOG`),
each a hand-rendered hypothesis with a primary source; catalog entries
are hypotheses to measure per checkpoint, never scoring keys.

### 1.5 Platforms, interpreters, extras

| Cell | State at baseline | Evidence |
| --- | --- | --- |
| CPython 3.11 / 3.12 / 3.13 on Linux, with `hf` extra | **proven** | CI test legs on every push (9,676 tests at the audit baseline) |
| CPython 3.12 on macOS / Windows, base install | **smoke-proven only** | Cross-OS smoke workflow: CLI help and base commands; no `hf` extra, no full suite |
| Windows, full install from the checked-in locks | **broken** | RR-07: the cocoindex and dev-hf closures contain an unmarked Linux-only wheel (`nvidia-cufile`); W3 owns the repair |
| Windows, pinned mypy | **failing** | audit: 34 errors in 3 files (POSIX-only attributes); W3 owns the repair |
| Interpreters outside 3.11–3.13 | **explicitly unsupported** | `requires-python = ">=3.11"` is a floor statement, not a tested promise above 3.13 |
| Extras | `hf` (torch/transformers/safetensors/jinja2), `dev` (ruff/mypy/coverage/hypothesis/pytest) | `pyproject.toml` `[project.optional-dependencies]` |

No other platform cell is claimed. In particular, nothing in this
document asserts Windows or macOS support for the measurement path
beyond the smoke cell above.

### 1.6 Schemas, versions, artifacts

| Artifact | Version identity | Producer → consumer |
| --- | --- | --- |
| Experiment report (`results.json` + `report.html`) | `schema_version: 1`; per-instrument `PROTOCOL_VERSION` = 1 (int) or `"<instrument>/1"` | `run` → `verify`, `diff-reports`, `pack` |
| Surface-noise probe file (`probe.json`) | surfacenoise protocol, versioned in-module | `surface-noise probe` → `surface-noise diff`, MCP tools |
| Sealed bundle | hash-bound directory seal; verifier limits in §3 | `pack` → `verify --bundle` |
| Serving records artifact | protocol `exec-backends-protocol-1` | `exec-backends` → `diff-reports` (serving section mirror) |
| Sample report | committed example under `docs/`/sample data, regenerated byte-identically by tests | docs + installed-artifact smoke |
| SBOM / attestations | PEP 740 attestations exist on the release path only | release workflow (publication is owner-gated; W1/W8 fence it) |

Provenance rule (established, see §3): report digests are *provenance*
(they bind a record to the exact inputs that produced it), never
*attestation* (the bundle seal is unsigned; nothing in the default
surface claims third-party-verifiable origin).

## 2. Contract table — persisted instruments and sections

The persisted report carries a core block plus **62 named sections**:
**53** are dispatched by `diff-reports` (the dispatched set is pinned
byte-exactly by the cycle-67 oracle's expected diff), and **9** have no
diff dispatch (§2.2). The core block (`decision`, `selected_candidate`,
`search`, `held_out`, configuration and boundary IDs, digests) is
produced by `experiments.run_experiment`, validated by `verify`'s
independent re-derivation, rendered by `reporting`, and compared by
`diff-reports`; it is the stable-candidate contract.

Column key — **Validator**: the admission validator in `diffreports.py`
that gates a stored section before comparison. **Comparator**: whether
`diff-reports` dispatches the section. **Renderer**: a dedicated HTML
section renderer in `reporting.py`. **Independent verifier**: a shipped
component, independent of the producer, that re-derives the section's
claims — for every section except the core block the honest entry is
"test-side oracle mirrors only" (the cycle-27 and cycle-67 test oracles
recompute sections independently, but they are tests, not a shipped
verifier; `verify` re-derives the core decision and diagnostics only).
**Installed-artifact test**: whether the section is exercised from the
installed wheel rather than the checkout — today that is true only of
the sample report's sections via the release smoke, which RR-11/RR-12
show is partial; the entry is therefore "sample smoke (partial)" for
every section and is a W5 gap, stated once here instead of 62 times.

### 2.1 Sections with a complete contract chain (53)

Producer is `experiments.py` unless a module is named. Validator names
are the `diffreports.py` function suffixes.

| Section | Producer | Validator | Comparator | Renderer | Documentation |
| --- | --- | --- | --- | --- | --- |
| ablation | experiments | `_valid_ablation` | yes | yes | USAGE §8 |
| aliasing | experiments | `_valid_aliasing` | yes | yes | ALIASING_AUDIT_DESIGN.md |
| answer_collision | experiments (collision) | `_valid_answer_collision` | yes | yes | USAGE §8 |
| batch_invariance | experiments | `_valid_batchinv` | yes | yes (`_batchinv_section`) | BATCH_PATH_DESIGN.md |
| boundary_audit | boundary_audit | `_valid_boundary_audit` | yes | yes | USAGE §8 |
| budget_frontier | experiments | `_valid_frontier` | yes | yes (`_frontier_section`) | USAGE §8 |
| cache_profile | experiments (cacheprofile) | `_valid_cache_profile` | yes | yes | USAGE §8 |
| canonical_form | experiments (canonicalform) | `_valid_canonicalform` | yes | yes | CANONICALFORM_DESIGN.md |
| choice_audit | experiments (choiceaudit) | `_valid_choiceaudit` | yes | yes (`_choiceaudit_section`) | CHOICE_AUDIT_DESIGN.md |
| compatibility_form | experiments (compatibilityform) | `_valid_compatibility_form` | yes | yes | COMPATIBILITYFORM_DESIGN.md |
| confidence | experiments (confidence) | `_valid_confidence` | yes | yes | CONFIDENCE_DESIGN.md |
| confidentiality | experiments (confidentiality) | `_valid_confidentiality` | yes | yes | USAGE §8 |
| confusable_skeleton | experiments (confusableskeleton) | `_valid_confusable_skeleton` | yes | yes | CONFUSABLESKELETON_DESIGN.md |
| contamination | contamination | `_valid_contamination` | yes | **none** (summary-only) | USAGE §8 |
| dateline | experiments (dateline) | `_valid_dateline` | yes | yes | DATELINE_DESIGN.md |
| deliforge | experiments (deliforge) | `_valid_deliforge` | yes | yes | USAGE §8 |
| density | experiments (density) | `_valid_density` | yes | yes | DENSITY_DESIGN.md |
| distraction | experiments (distraction) | `_valid_distraction` | yes | yes | DISTRACTION_DESIGN.md |
| enumeration_convention | experiments (enumerationconvention) | `_valid_enumeration_convention` | yes | yes | ENUMERATION_DESIGN.md |
| faketurn | experiments (faketurn) | `_valid_faketurn` | yes | yes | USAGE §8 |
| fewshot | experiments (fewshot) | `_valid_fewshot` | yes | yes | USAGE §8 |
| field_order | experiments (fieldorder) | `_valid_field_order` | yes | yes | FIELDORDER_DESIGN.md |
| held_out_budget_floor | experiments | `_valid_held_out_floor` | yes | yes (`_held_out_section` family) | USAGE §8 |
| hierarchy | experiments (hierarchy) | `_valid_hierarchy` | yes | yes | USAGE §8 |
| history_frame | experiments (historyframe) | `_valid_history_frame` | yes | yes | HISTORYFRAME_DESIGN.md |
| instruction_distance | experiments (instructiondistance) | `_valid_instruction_distance` | yes | yes | INSTRUCTIONDISTANCE_DESIGN.md |
| invisible_chars | experiments (invisiblechars) | `_valid_invisible_chars` | yes | yes | INVISIBLECHARS_DESIGN.md |
| leading | experiments (leading) | `_valid_leading` | yes | yes | LEADING_DESIGN.md |
| multiturn | experiments (multiturn) | `_valid_multiturn` | yes | yes | MULTITURN_DESIGN.md |
| needle | experiments (needle) | `_valid_needle` | yes | yes | NEEDLE_DESIGN.md |
| numform | experiments (numform) | `_valid_numform` | yes | yes | NUMFORM_DESIGN.md |
| perturbation | experiments | `_valid_perturbation` | yes | yes | USAGE §8 |
| placement | experiments (placement) | `_valid_placement` | yes | yes | PLACEMENT_DESIGN.md |
| portfolio | experiments (portfolio) | `_valid_portfolio` | yes | yes | USAGE §8 |
| position | experiments (position) | `_valid_position` | yes | yes | USAGE §8 |
| precision | experiments (precision) | `_valid_precision` (recompute) | yes | yes | PRECISION_DESIGN.md |
| prefill | experiments | `_valid_prefill` | yes | yes | USAGE §8 |
| prune_sweep | experiments (prunesweep) | `_valid_prune_sweep` | yes | yes | PRUNESWEEP_DESIGN.md |
| record_notation | experiments (recordnotation) | `_valid_record_notation` | yes | yes | PAYLOADSERIALIZATION_DESIGN.md |
| refusals | experiments (refusals) | `_valid_refusals` | yes | yes | USAGE §8 |
| repetition | experiments (repetition) | `_valid_repetition` | yes | yes | REPETITION_DESIGN.md |
| rotation | experiments (rotation) | `_valid_rotation` | yes | yes | ROTATION_DESIGN.md |
| scorerlens | experiments (scorerlens) | `_valid_scorerlens` | yes | yes | SCORER_LENS_DESIGN.md |
| selection_robustness | experiments | `_valid_selection_robustness` | yes | yes | USAGE §8 |
| serving_backends | experiments (servingbackends) | `_valid_serving_backends` | yes | yes | SERVING_BACKENDS_DESIGN.md |
| stop_cause | experiments (stopcause) | `_valid_stop_cause` | yes | yes (`_stopcause_section`) | USAGE §8 |
| structured | experiments (structured) | `_valid_structured` | yes | yes | PAYLOADSERIALIZATION_DESIGN.md |
| synthhistory | experiments (synthhistory) | `_valid_synthhistory` | yes | yes | USAGE §8 |
| template_robustness | experiments (surfacenoise) | `_valid_template_robustness` | yes | yes | TEMPLATE_ROBUSTNESS_DESIGN.md |
| token_healing | experiments (healing) | `_valid_token_healing` | yes | yes | USAGE §8 |
| toolcall | experiments (toolcall) | `_valid_toolcall` | yes | yes | USAGE §8 |
| transfer | experiments | `_valid_transfer` | yes | yes | USAGE §8 |
| typographic_emphasis | experiments (typographicemphasis) | `_valid_typographic_emphasis` | yes | yes | TYPOGRAPHIC_EMPHASIS_DESIGN.md |
| verbosity | experiments (verbosity) | `_valid_verbosity` | yes | yes | USAGE §8 |
| ws_canonical | experiments (wscanonical) | `_valid_ws_canonical` | yes | yes (`_wscanonical_section`) | USAGE §8 |

Known chain weaknesses inside this table (all in the ledger): the
projection rule set covers only 4 keys, so most sections reach the
default export by producer discipline alone (ORCH-01); two invalid
sections can compare as `identical` with no invalid-data diagnostic
(RR-09); precision's recompute equality conflates JSON `false` with
`0.0` (RR-04).

### 2.2 Sections with no diff dispatch (9) — gaps and stated reasons

| Section | Producer | Validator | Renderer | Why there is no comparator / exposure |
| --- | --- | --- | --- | --- |
| negation | experiments (negation) | **none** | yes (`_negation_section`) | **Defect (RR-03).** The section is produced and rendered but never compared: a 16/16 → 0/16 producer change diffs as `identical: true`. No documented reason exists. W2 adds the comparison or an explicit opt-out. |
| sensitivity_sweep | experiments | **none** | yes (`_sweep_section`) | No documented reason found in code or docs. Recorded as a gap; the W2 contract-completeness census adjudicates: add dispatch or record an explicit opt-out. |
| sensitivity_sweep_tokens | experiments | **none** | yes (`_token_depth_section`) | Same as above. |
| stop_matrix | experiments via `exec-backends` composition | **none** | yes (`_stop_matrix_section`) | Companion section of the serving/stop-cause surface; no documented reason for the missing dispatch. Gap; W2 adjudicates (the serving section itself is dispatched). |
| logprob_diagnostic | experiments | **none** | yes (`_logprob_section`) | Diagnostic section (numbers + fixed-vocabulary interpretation prose). No documented reason. Gap; W2 adjudicates. |
| template_conformance | doctor, composed by the CLI | **none** | yes (`_conformance_section`) | Doctor pre-flight diagnostic: per-run environment evidence, not a measurement axis. No *documented* opt-out exists today; W2 records one explicitly or adds dispatch. |
| template_firewall | doctor | **none** | yes (`_template_firewall_section`) | Same as above. |
| marker_isolation | doctor | **none** | yes (`_marker_isolation_section`) | Same as above. |
| backend_divergences | diffbackends, composed by the CLI | **none** | yes (`_backend_divergence_section`) | Emulated-backend differential diagnostic. Same status as the doctor rows: no documented opt-out; W2 adjudicates. |

No other persisted section lacks dispatch. A *new* section without a
dispatch entry is exactly the failure RR-03 shipped; W2's completeness
check (a registry-driven census) is the structural fix, and until it
exists this table is the registry of record, maintained by hand.

### 2.3 Comparators and verifiers (surface level)

| Comparator / verifier | Compares | Class |
| --- | --- | --- |
| `diff-reports` | two reports: decision, winner, and the 53 dispatched sections | stable candidate |
| `diff-backends` | two emulated-backend probe files | experimental |
| `surface-noise diff` | two probe files | experimental |
| `verify` | one report against its own recorded inputs (independent re-derivation of decision + diagnostics); `--bundle` re-checks a seal | stable candidate |
| `verify` (serving records) | A8 artifact structural re-read | experimental |

## 3. Boundary invariants

"Established" means the invariant is enforced in code at the baseline
and named here with its constant or vocabulary. "To be established"
means no uniform invariant exists today; the owning work package is
named and the ledger carries the finding. Inventing a number here
would be a false claim, so gaps are stated as gaps.

### 3.1 Established in code

| Boundary | Invariant |
| --- | --- |
| Fixture/dataset admission (`datasets.py`) | exact JSON types; `MAX_BYTES` = 2,000,000; `MAX_CASES_PER_SPLIT` = 100; `MAX_MESSAGES` = 21; `MAX_TEXT` = 16,384 chars; search/held-out split separation is structural |
| JSON export (`reporting.save_report`) | finite values only (non-finite constants rejected); records carry answer **digests**, with the sanctioned `answer_text` exceptions named in §3.2 |
| Stored-report admission (`diffreports.py`) | per-section validators gate shape and type before any consumption; stored magnitudes are admitted by float-convertibility (conversion tested, never a literal bound — cycle 244); a section that fails admission degrades to a documented unavailable state instead of raising |
| Integer magnitude | CPython's int↔str digit limit (4,300) bounds loadable integers; conversion-tested admission (above) is the overflow discipline |
| MCP framing (`mcp_server.py`) | request cap `MAX_REQUEST_CHARS` = 1 MiB; single text block `MAX_CONTENT_BYTES` = 64 KiB; duplicate members, non-finite constants and lone surrogates rejected at the frame layer; tool errors use the JSON-RPC vocabulary (`-32602` invalid params, `-32603` internal) |
| Sealed bundles (`pack.py`) | producer caps: `MAX_BUNDLE_ZIP_BYTES` = 288 MiB, per-file 64 MiB, total 256 MiB; verifier additionally rejects > 1,024 entries including the manifest — producer and verifier limits currently disagree (RTI-02) |
| Serving records (`servingbackends.py`) | `MAX_RECORD_BYTES` = 16,777,216 per record; HTTP response cap 16,777,216 bytes; GGUF metadata caps (100,000 keys; 1,048,576-byte strings); stack errors use the frozen `serving_backends_*` code vocabulary |
| Debugger (`debug.py`) | save cap `_MAX_SAVE_BYTES` = 5,000,000 (RTI-07: enforced on components, not on the final encoded bytes — gap in the ledger); script cap 1,000,000 bytes; nested-read depth 5 |
| Selection | conservative unique-winner rule: ties, failures and incomplete evidence abstain (`no_candidate`); denominators and abstentions are reported, never silent; held-out data is read only behind a search winner |
| Completion vocabulary | report `decision` values are a frozen set (select / `no_candidate` with reason); CLI exit codes per §1.2; a cancelled run exits 130 and cannot produce a winner |
| Provenance vs attestation | digests bind records to inputs (provenance); the bundle seal is unsigned and claims no attestation; PEP 740 attestations exist only on the owner-gated release path |
| Default redaction | the default export is evidence-excluded (`raw_evidence_included=false`): `search`/`held_out` observations are stripped and `template_conformance`/`backend_divergences` raw carriers are withheld by `_public_view` — but the rule set covers only 4 keys; §3.2 lists what still passes through (ORCH-01) |

### 3.2 Pass-through exceptions the projection does not yet govern (ORCH-01)

Execution-proven by the projection census (sentinel values, twice) at
this baseline; full detail in the ledger under ORCH-01:

- `transfer.checkpoints[].path` — operator-supplied filesystem paths
  reach default JSON.
- `serving_backends.records[].prompt_ids` / `returned_ids` —
  reconstructable token streams reach default JSON.
- `answer_text` in 8 sections (`numform`, `record_notation`,
  `prune_sweep`, `invisible_chars`, `confusable_skeleton`,
  `compatibility_form`, `enumeration_convention`,
  `typographic_emphasis`) — raw model answer text reaches default
  JSON. For `numform` and `record_notation` this is a previously
  sanctioned, digest-bound design exception; the other six inherited
  the shape without a projection decision. W2 adjudicates the family
  explicitly (evidence-only vs named whitelist exception); two of the
  diff mirrors recompute digests *from* `answer_text`, so the choice
  has a documented diff-degradation consequence either way.
- Latent: `serving_backends.identity.config` passes through verbatim
  (digests and counts today). An allowlist projection must land
  before or with any new serving stack (W6 vLLM item), or a future
  stack can leak endpoint/path fields on day one.
- HTML and the terminal summary are clean for every flagged field
  today — by renderer discipline, not by projection (HTML is rendered
  from the full result in both modes). The completeness check W2
  builds must therefore test the projection, not the renderers.

### 3.3 To be established (no uniform invariant today)

| Invariant | State | Owner |
| --- | --- | --- |
| Uniform report depth/item ceilings across all section validators | per-validator only; no global ceiling | W2 |
| Compared-vs-omitted axis disclosure in diff output (absent / invalid / unsupported / unavailable / valid as distinct states) | two invalid sections can read as `identical` today (RR-09) | W2 |
| Public/evidence projection registry covering every section | 4 of 62 sections governed (ORCH-01) | W2 |
| Run-level budget model (base + panels + transfers + retries) and durable partial-run recovery | per-stage caps only (RR-14) | W4 |
| Portable hostile-render containment (worker process with a real deadline on every supported platform) | platform-dependent fallbacks documented in code (RR-10) | W4 |
| Producer/verifier bundle-limit agreement | limits disagree (RTI-02) | W5 |
| Installed-artifact verification independent of the checkout | smoke can import the checkout (RR-11); sdist test scope not self-contained (RR-12) | W5 |
| One readiness entry point emitting a structured per-lane verdict | does not exist; designed in the ledger (W0 deliverable) | W1/W8 |

## 4. Reading this contract

A capability not listed here is not part of the supported surface.
A class of "experimental" means the contract chain exists but the
evidence behind it is thinner (single stack, single host class, or
diagnostics-grade); it never means "broken silently" — every known
defect is a ledger row in the [stable-release ledger](STABLE_RELEASE_LEDGER.md).
