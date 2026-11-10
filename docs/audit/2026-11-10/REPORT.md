# Current-state audit report

## Verdict and scope

The source at `d8a1346343e1b50b28bf567ab6b193b38fc79608` is a substantial,
working research tool, but it does **not** satisfy a stable-release gate.
Passing Linux tests coexist with a disabled secret-detection rule set,
public report-comparison failures, an omitted instrument, unavailable analyzer
results, and a publication path that does not enforce the ordinary quality gates.
These are specific defects and evidence gaps, not a judgment about the model
that originally wrote the repository.

This audit did not implement any remedy. The source revision above is the
baseline for every finding unless explicitly labeled historical. Exact source
paths, symbols, reproduction recipes, the evidence ledger and the per-file
inventory make the assessment followable from a fresh checkout.

### Review coverage

- All **579 tracked files** were byte-inventoried and classified. This includes
  source, tests, tooling, locks, workflows, docs, examples, vendored data,
  generated site files and generated package metadata.
- All **341 Python files**, containing **362,433 physical lines**, were parsed
  into ASTs. The package contains **75 Python files** and **116,124 physical
  lines**. These are physical-line counts, not executable-line counts.
- Structural checks also parsed 3 TOML files, 4 JSON files and 51 HTML files;
  identified 74 Markdown files and 19 YAML paths; and inventoried 2 binary
  assets. YAML path inventory alone is not YAML semantic validation.
- Manual review and execution prioritize public input and serialization
  boundaries, measurement populations, orchestration, optional backends,
  provenance, release tooling and CI. Large-module AST maps supplement
  targeted source reads. This is **not** a claim that every line received an
  equally deep manual review or every possible input was exercised.
- The inventory describes the original 579 files. Newly added audit documents
  are deliverables, not silently included in that baseline denominator.

Eight original specialist attempts failed before completing their reports
because the provider exhausted credits. Their traces and intermediate maps
were recovered, but they are not counted as completed independent reviews.
The resumed review used bounded specialists and parent-side reproduction;
only cross-checked claims enter the findings below.

### What changed during the audit

The initial snapshot was `27bf1a2b863148d946d41374d389d1cfd583bcd4`.
A fresh fetch found `d8a1346`, with exactly three changed files: the changelog,
security record and CocoIndex requirement lock. No Python source changed.
The new filelock/litellm pins closed the original 13 Dependabot alerts.
A fresh, paginated GitHub read reports **0 open Dependabot alerts**, **0 open
CodeQL alerts**, and **3 open Scorecard posture alerts**. DeepSource is a
separate, unavailable signal; its count is unknown.

## Verified strengths and current execution evidence

The repository already has useful safeguards worth preserving:

- The core has no mandatory third-party runtime dependencies. The optional HF
  adapter is separate. Token-ID boundaries, exactly-N generation, conservative
  unique-winner selection and explicit abstentions are coherent product choices.
- The audited head's [CI run 42835070179][ci] passed all three Linux test legs:
  Python 3.11, 3.12 and 3.13 each ran **9,676 tests**, with **3 skips**.
  The ordinary lint/typecheck, dependency/security and packaging jobs passed.
- Linux Python 3.12's coverage row reports 44,461 statements, 1,557 missed
  statements, 21,676 branches and a displayed **95%** aggregate. Because
  branch measurement is enabled, that displayed percentage is not simply
  covered statements divided by statements. The latter is approximately
  **96.4981%**. Do not conflate the two metrics or a rounded gate with complete
  coverage.
- Fresh pinned local checks passed Ruff over package/tests/examples/tools/fuzz,
  Ruff formatting over its CI scope, and Bandit over the package. Bandit emits
  suppression/comment warnings; exit 0 is not evidence that exclusions do not
  exist. The audit environment's installed dependency audit passed; that scope
  excludes the uninstalled HF and semantic-search stacks.
- `tools/release_build.py` built the wheel and normalized sdist twice with
  identical artifacts. The initial build and a final documentation-tree
  double build both passed; both final distributions also passed Twine.
  [evidence.json](evidence.json) records their measured hashes and scope.
- A clean wheel-only environment outside the checkout, with isolated imports,
  passed version/import checks, the 29-family catalog, fixture inspection,
  MCP self-test, all 12 top-level commands' help, and sample self-comparison.
  This is an offline smoke test, not model inference or proof of every command.
- A pinned strict MkDocs build passed to an external output directory.
  The sample report and generated HTML were inspected structurally, not
  browser-rendered. Browser rendering and accessibility remain unverified.
- Default-rule Gitleaks scanning of the current package found no findings.
  A whole-checkout default-rule scan found two known synthetic fixture matches
  in tests, not evidence of live credentials. Their literal values are omitted.
  The existing configured scanner's green result is not reliable; see RR-01.

A dev-only Windows collection attempt found 8,936 tests and 15 collection
errors; a full-suite attempt exited 2 without a usable final unittest summary.
Missing optional ML dependencies and POSIX-only imports were present. This is
**not** the Linux HF-enabled CI configuration, and no coverage or final pass/fail
count is inferred from that incomplete local run. A separate exact pinned
Windows mypy run reports **34 errors in 3 files**. The plan separates these
portability and environment-contract problems rather than calling Linux CI red.

## Finding register

Priority **P1** blocks a stable release of the affected advertised surface.
**P2** is required readiness work or a material verification/maintenance gap.
**P3** is lower-risk cleanup. A configuration blocker is not mislabeled as an
exploitable source vulnerability. No newly demonstrated critical/P0 exploit is
claimed.

| ID | Priority | Kind | Finding | Package |
| --- | --- | --- | --- | --- |
| RR-01 | P1 | Defect | Secret scanner has no rules | W1 |
| RR-02 | P2 | Defect | Malformed reports crash | W2 |
| RR-03 | P2 | Defect | Negation diff is missing | W2 |
| RR-04 | P2 | Defect | Precision accepts wrong types | W2 |
| RR-05 | P1 | External | DeepSource unavailable | W1 |
| RR-06 | P1 | Gap | Publish bypasses quality gates | W1/W8 |
| RR-07 | P1 | Defect | Windows lock cannot install | W3 |
| RR-08 | P2 | Gap | Dev setup and typing fail | W3 |
| RR-09 | P2 | Gap | Identical omits evidence drift | W2 |
| RR-10 | P1 | Known | Unbounded render fallbacks | W4 |
| RR-11 | P2 | Gap | Smoke can import checkout | W5 |
| RR-12 | P2 | Gap | Sdist tests lack dependencies | W5 |
| RR-13 | P2 | Gap | No real positive-path evidence | W6 |
| RR-14 | P2 | Gap | Budget/recovery incomplete | W4 |
| RR-15 | P2 | Risk | Duplicated contracts and gates | W7 |
| RR-16 | P2 | Debt | Docs/generated metadata drift | W7 |
| RR-17 | P2 | Gap | Exclusion policy not enforced | W1/W7 |
| RR-18 | P2 | Gap | Support matrix evidence thin | W3/W8 |

### RR-01 — the configured secret scan is effectively empty

**Location:** `.gitleaks.toml`; `.github/workflows/gitleaks.yml`, scan-history
step. The custom file contains an allowlist but no `rules` and no default-rule
extension. The workflow explicitly supplies that file. Upstream Gitleaks 8.30.1
loads a supplied configuration instead of implicitly merging default rules.

**Execution:** the exact pinned, checksum-verified scanner returned 0 and
“no leaks found” on a scratch synthetic provider-shaped token with the repository
configuration. The same binary and input returned 1 and detected the canary
with the default rules. No real credential was used. This is a sensitivity
failure, not an assertion that the repository currently contains live secrets.

**Remedy:** restore the upstream rule set explicitly, remove reliance on a
zero-rule configuration, and add a canary control that proves detection in the
same invocation CI uses. Reconcile existing synthetic matches professionally at
source without new suppressions. Historical fixtures must not trigger an
unauthorized history rewrite, blanket exclusion, or dismissal.

### RR-02 — validation is incomplete at the consumption boundary

**Locations:** `diffreports._valid_frontier`, `diff_frontiers.decision_points`,
`_valid_held_out_floor`, `diff_held_out_floor`, `render_summary`, and
`_precision_recompute`.

A JSON-loadable frontier with `decision_steps: 1` survives the outer frontier
admission and raises `TypeError` when iterated. A loadable held-out floor with
`budget_robust: []` reaches a summary lookup that raises an unhashable-type
`TypeError`. A precision section with a mixed-type `unavailable_candidates`
list is sorted before element types are validated and raises `TypeError`.

These are reachable through `diff-reports`, not merely exotic non-JSON direct
Python calls. The CLI catches reader errors but performs comparison and summary
rendering outside that catch. One exercised case writes valid JSON to stdout
and then fails rendering stderr, so consumers can receive a plausible artifact
from a failed command.

A bounded matrix exercised 56 section differs on 12 primitive values each,
then 4,092 one-field substitutions over 24 admitted control shapes. The latter
found 14 exception cases in the frontier/floor families. That is an audit
sample, not all schema states and not 14 independent root causes.

**Remedy:** validate every consumed field before iteration, sorting, hashing,
conversion or indexing. Make validators total over bounded JSON values. Keep
section-local invalidity explicit, build both outputs before emitting either,
and test public CLI exit/stdout/stderr together. A blanket exception that hides
corrupted evidence is not a sufficient fix.

### RR-03 — a shipped measurement can change without any reported difference

**Locations:** `cli._run_negation_probe`, `experiments.negation_stage`,
`negation_analysis`, and `diffreports.diff_reports`.

The CLI produces `result["negation"]`, but the diff dispatcher does not consume
it. An exact source routing census identified that missing CLI-stage key.
Execution with two deterministic arithmetic backends then produced genuine
probe sections: each ran 64 generations over 16 generated cases. The negated
arm changed from 16/16 passes to 0/16 and the headline changed from
`below_threshold` to `negation_drop_observed`. `diff_reports` still returned
`summary.identical: true` and no negation output. A present-versus-absent
negation section was also invisible.

**Remedy:** add a versioned negation comparison/validation contract and explicit
presence asymmetry; register it in dispatch, JSON summary and human summary.
The human summary explicitly says identical on every **compared** axis, not
complete-report equivalence. Add a mechanical census tying every persisted
instrument to a comparator or a conspicuous, intentional exclusion. Do not
infer completeness from separate unit tests of the producer and differ.

### RR-04 — “type-strict” validation does not consistently distinguish types

**Locations:** `diffreports._recomputed_equal` and `_precision_recompute`.
The equality helper is strict when the expected value is a bool or int but
falls back to Python equality for a float. Thus `False` validates as expected
`0.0`, including nested fields. Precision ground records also lack an explicit
bool/None type gate for `diverged`, and its candidate arm pass fields are not
uniformly type-gated before recomputation.

An independently assembled one-pair precision control was admitted. Replacing
its zero divergence rate or log-probability delta with `false` was also admitted
and compared unchanged. Replacing a ground divergence flag with a list or
string was admitted. A bool substituted into an integer total was correctly
rejected: the bug is inconsistent domain checking, not universal laxness.

**Remedy:** specify JSON domains field by field, reject bools in numeric fields,
reject containers/strings in boolean flags, and validate attested fields even
where their provenance cannot be independently reconstructed. Preserve valid
JSON integer/float interoperability only where the contract explicitly allows
it. Re-run all mirror validators with an independent invalid-type table.

### RR-05 and RR-06 — missing security and publication gates

**Evidence:** [Findings guard run 42835070138][guard] on the audited head failed
with `DeepSource NOT-CONFIGURED: no DEEPSOURCE_TOKEN or CLI login`.
Its `deepsource_total` is null. The app's check is skipped. Neither state is a
clean analysis. GitHub reports no required status checks or required PR review
on the current main protection endpoint; force pushes are disabled and admin
enforcement is off. These settings were read, not changed.

`.github/workflows/release.yml` accepts `v*` tag pushes. Its build job binds the
tag version to package metadata and checks artifact reproducibility, but it
neither runs the full quality gate nor verifies successful check results for
the exact tagged commit. The publish job needs only the build job. It does not
prove main ancestry, fresh security inventories, successful model-contract
checks, or an immutable readiness receipt. Environment protections may add an
external human gate, but the workflow source itself does not supply these
software gates; PyPI trusted-publisher registration was not independently
verified.

**Remedy:** restore analyzer access through the approved secret/credential
channel without exposing credentials. Preserve fail-closed behavior. Add a
non-publishing exact-SHA readiness workflow and make publishing consume only
that verified artifact/evidence identity. Test missing, failed, skipped,
cancelled and stale checks as rejection cases. Keep tag/release creation under
separate owner authorization. Do not change repository protections as part of
this documentation audit.

### RR-07 and RR-08 — “portable” setup is not verified at its advertised boundary

The latest exact `tools/requirements/cocoindex.txt` install and the exact
`dev-hf.txt` resolution fail on Windows because an unmarked
`nvidia-cufile==1.17.1.22` requirement has only Linux wheels. The mandated
semantic search therefore cannot be provisioned on this host using its own
supported launcher. Pins and hashes do not make a Linux resolution portable.
The filelock/litellm fix does not address that independent problem.

Dev-only test collection also imports torch/numpy and POSIX `resource`
unconditionally in some modules. The README's optional unittest invocation and
the developer setup descriptions do not clearly distinguish a base install,
a dev-only install and the required HF-enabled full gate. Pinned mypy reports
34 Windows errors in `fixturefuzz.py`, `doctor.py`, and `mcp_server.py` for
POSIX-only attributes.

**Remedy:** generate and actually resolve platform-aware closures; separate
optional integration layers from base/core contracts; define the supported
contributor matrix; and type-check platform abstractions on their real targets.
Do not use forced installations, ignored dependency conflicts, pretend HF
modules, blanket skips, or weakened typing to manufacture green results.

### RR-09 — “identical” is narrower than artifact equivalence

Two invalid, present structured sections with different out-of-domain rates
are both reduced to `available: neither`; the overall result can be
`identical: true` with no invalid-data diagnostic. Separately, changing a
held-out planned/pass count from 1/1 to 50/50 with unchanged pass status yielded
no configuration drift and `identical: true` in a digest-only comparison.

The differ intentionally compares selected axes, not every byte, so neither
observation alone establishes a broken selection rule. They do establish a
misinterpretation risk for an evidence tool: invalid evidence and materially
different evidence strength are not the same as unchanged, validated evidence.

**Remedy:** publish compared/omitted axes and section states
(absent/invalid/unsupported/unavailable/valid); report denominator drift even
without raw observations; and separate `no_detected_difference` from
`equivalent_on_all_supported_axes`. Keep digest-only privacy and do not demand
raw fixtures merely to compare planned counts.

### RR-10 and RR-14 — containment and recoverability need explicit guarantees

The repository already documents that Jinja's SIGALRM wall-clock guard is not
available on non-Unix platforms or off the main thread, and that the memory
limit has platform-dependent fallbacks. MCP timeout handling has similar
limitations. A source/output cap checked after rendering is not a compute or
peak-memory limit. These are known residuals, not newly invented RCE claims.

The experiment suite also has many independently capped optional panels. The
original development plan's actual-work budget ledger, reservations, retries,
partial-run recovery and provenance are not established merely by input/output
token counts. Full-prefix decoding repeats work. Historical real-model runs
lost progress when the host was interrupted; that history is evidence of a
recoverability requirement, not proof of a specific OS or hardware fault.

**Remedy:** isolate untrusted rendering/probe work behind an owned worker
process with portable deadlines and clean cancellation, or explicitly refuse
unsupported hostile-input execution. Add preflight cost estimates, enforceable
run-level budgets, durable partial records and resumable identity checks.
Distinguish deterministic measurement artifacts from nondeterministic operational
receipts; do not pollute the deterministic report with wall-clock timestamps.

### RR-11 and RR-12 — distribution checks do not yet prove the advertised artifact

The release smoke installs a wheel into a fresh venv but executes console
scripts while still in the checkout, without asserting the imported package's
origin. Console-script startup does not automatically imply checkout shadowing;
a broken wheel hidden by that smoke was **not** demonstrated. This is an
evidence-isolation gap. The audit's own smoke ran outside the checkout with
isolated imports and verified the installed origin.

`MANIFEST.in` intentionally includes tests and one test-data file while
excluding docs/examples. Tests import example generators, inspect workflow and
AGENTS files, and require other checkout content. A source recipient therefore
cannot assume the shipped test suite is a self-contained verification route.
Choose and test a supported sdist contract: ship its dependencies, separate
checkout-only tests, or clearly declare that artifact test scope is narrower.

The release path has provenance attestations, but no complete readiness receipt
binding scanner results, support matrix and installed-artifact tests to the
published hashes. A double build on one host proves a useful, narrower property;
it does not establish cross-OS byte reproducibility or interpreter independence.

### RR-13 — real-model evidence is almost entirely a negative control

`docs/REAL_CHECKPOINT_VALIDATION.md` records seven completed rows, 324
generations and four checkpoints. Every candidate scored 0/12 on the exact-answer
and boundary conjunction, every decision was `no_candidate`, and held-out
recommendation verification was never entered. TinyLlama's native Zephyr-style
format is not in the tested catalog rows. Other rows and GPU parity are
explicitly unverified.

Those failures are legitimate findings, not tests to weaken. They demonstrate
abstention and failure reporting, not a successful real-model recommendation,
held-out confirmation, or deployment-parity workflow. Re-running a mismatched
negative row alone does not close the positive-path gap.

**Remedy:** use a properly licensed, locally provisioned, supported checkpoint
and a native-format positive control. Freeze fixtures and boundaries before
looking at held-out results. Demonstrate one successful recommendation path and
intentional tie/failure/cancellation controls, or keep the capability labeled
experimental and scope the release accordingly. Never optimize on the same
held-out outcomes to force a success.

### RR-15 through RR-18 — prevent the next local repair from leaving siblings broken

Measured architecture concentration is material: `diffreports.py` has 31,587
lines, `experiments.py` 16,388, and `reporting.py` 14,559. Together they account
for **62,534 physical lines**, approximately **53.85%** of package Python
source. There are 247 runtime functions over 100 lines. The largest test oracle
function spans 9,507 lines; the cycle-67 test module spans 27,553 lines.

Size alone is not a correctness defect. The demonstrated missing dispatch and
validation holes show why duplicated producer/mirror/renderer/test contracts
need an explicit inventory and consistent boundary tests. Mirrored code can
agree on the same wrong assumption. Preserve independently derived oracles,
not textual mutation anchors as architectural constraints.

A specific verification defect is also reproducible:
`tests/test_cycle124_margin_trajectory_audit.py`,
`TestMutationGate.test_mutants_are_killed`, counts any nonzero child return code
as a killed mutant. On a disposable copy, injecting child exit 2 with a
collection-failure diagnostic and zero executed semantic tests made all four
iterations count as kills and the actual gate method return successfully.
This does not invalidate every existing mutation result, but it disproves the
gate's ability to distinguish behavioral detection from harness failure.
Require a passing unmutated baseline and parsed, non-empty intended test
outcomes; collection/import/syntax/timeout failures are inconclusive, not kills.

Other measured debt:

- `site/` and `model_protocol_lab.egg-info/` contribute 107 tracked generated
  files. The egg metadata says 0.1.0 while `pyproject.toml` and the installed
  wheel say 0.2.0. Generated metadata should not be a competing authority.
- The living architecture page omits the newer `exec-backends` interface and
  describes a no-network measurement path without integrating the loopback
  serving backend. `docs/SECURITY.md` still contains broad “no subprocess / HTTP
  clients anywhere” claims contradicted by the newer serving implementation.
  Historical audit sections should remain historical; living summaries must
  identify which contract they describe.
- Current docs mix historical and current test counts, line versus combined
  coverage, and older release-readiness assertions. The README's broad
  production-hardened wording is stronger than this audit supports.
- The AST census found 50 runtime functions without docstrings, including
  nested functions; 22 names do not begin with an underscore. This is not a
  count of public API defects. Existing generic docstrings also do not
  necessarily describe inputs/outputs or invariants.
- A lexical inventory found 706 Python comment lines with scanner-exclusion
  markers, 300 in the package. This is not 706 active vulnerabilities or 706
  distinct suppressions: a line can contain several markers and some are
  examples. It is evidence that the absolute no-suppression policy is not
  mechanically universal. Review each actual exclusion and repair source
  professionally rather than silently inheriting it.
- Linux full-suite CI and Python 3.12 base-command smoke on three operating
  systems do not prove all optional extras, newer admitted interpreters,
  hostile-input timeouts, accelerator behavior or real serving-stack parity.
  These gaps must appear in a capability matrix, not be implied away by
  `Operating System :: OS Independent` or `requires-python >=3.11`.

Refactor by bounded responsibility after regression contracts exist. Prefer a
small typed core, explicit instrument registration and independent conformance
harnesses over a whole-repository rewrite. The supported surface should become
smaller and more precise before adding more catalog families or probes.

## Additional independently checked subsystem findings

[SUBSYSTEMS.md](SUBSYSTEMS.md) is part of this report, not optional background.
It records default-export privacy leakage, disconnected optional cancellation,
ambiguous built-in position scoring, unequal few-shot populations, quadratic
cache storage, pack/verify limit disagreement, MCP and serving input holes,
transfer residency, and related diagnostics/export design debt. Each item has
its own severity, reachability, source location and acceptance criteria. The
coordinator re-executed all four completed specialist probe groups; no real
models, servers, large allocations or private fixtures were needed.

An independent fifth reviewer reproduced scanner, negation, malformed-report
and precision findings. Its refutation narrowed the scope/severity of foreign
input failures and rejected a claim that console-script cwd alone proves
checkout shadowing. Those corrections are reflected above.

## Rejected claims and limits

- The 13 filelock alerts are **not still open** at the audited head. They were
  fixed by the intervening commit and the remote state was re-read.
- The package does build, install and execute offline commands. An empty
  “nothing works” conclusion would contradict both CI and the isolated smoke.
- A selected winner is not required on every dataset. Abstention on ties or
  failure is intentional. No test or acceptance gate may force a winner.
- Optional sections absent from the sample HTML are not automatically missing
  features; they can be correctly conditional on available input.
- A parser accepting a deeply nested JSON value did not by itself prove a
  crash: the exercised depths through 1,600 completed their targeted comparison.
  No generic recursion vulnerability is asserted from nesting alone.
- The field-order large-`number` hypothesis was rejected after inspecting the
  actual frozen specs: that probe's schemas contain no `number` field. A
  hypothetical branch is not a reachable release defect.
- Gitleaks default rules found no current-package credential. The scanner
  configuration defect is proven independently of any live leak claim.
- DeepSource and Codecov/other app-suite states are distinct from GitHub
  Actions job success. No skipped or queued vendor suite is called clean.
- No live model download, large model/GPU run, browser render, PyPI upload,
  release dispatch or system configuration change was performed in this audit.
  These unexecuted paths remain explicit acceptance work.

## How to use the plan

Fix the confirmed failures first, with a failing regression before the repair.
Keep missing evidence visible, and independently re-derive scientific expected
values. Pass each work package's gates before structural refactoring. Stop at
release readiness: creating a tag, a draft release or uploading a package needs
separate owner authorization. Never turn this report into a claim that the
remedies have already landed.

[ci]: https://github.com/strmt7/model-protocol-lab/actions/runs/42835070179
[guard]: https://github.com/strmt7/model-protocol-lab/actions/runs/42835070138
