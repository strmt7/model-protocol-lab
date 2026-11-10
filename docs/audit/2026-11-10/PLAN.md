# Stable-release implementation plan

**Baseline:** `d8a1346343e1b50b28bf567ab6b193b38fc79608`.
**Status:** planned, not implemented. **Publication:** prohibited without a new,
explicit owner instruction.

This is the active readiness backlog derived from [the audit](REPORT.md).
Older development plans and release-readiness reports remain useful historical
records, not evidence that this backlog is closed. A later agent must compare
each finding with the newest main commit before changing anything.

## Operating contract

1. Work from current `main`, preserve other work, and use one integration writer.
   Read-only specialists may review disjoint areas. Only one source-mutating
   test/mutation lane may use a checkout at a time; use disposable copies for
   mutators and never run them over a tree that another worker is reading.
2. Record a minimal failing test before fixing a demonstrated defect. Derive
   expected outcomes from the written definition or primary reference, not from
   the implementation's output. Include a non-empty valid control.
3. Separate correctness fixes, schema migrations and structural refactors into
   different verified changes. Do not expand catalog/probe breadth during
   stabilization. Preserve exact-token semantics and conservative abstention.
4. No suppressions, dismissals, ignore-rule expansion, dependency forcing,
   monkey-patches, or lower coverage/security thresholds to obtain green checks.
   Unsolved findings remain open with evidence. History rewrites and repository
   settings changes require their own authorization.
5. All shipped code, tests and instructions must be host- and installation-
   independent. Use caller-provided paths, standard temporary-directory APIs,
   capability detection and supported dependencies. No personal account data,
   local model paths, credentials, machine names or raw private observations
   belong in commits or retained public reports.
6. Commit only through the authenticated official GitHub connector/API, without
   author or committer overrides. Read back the exact commit, changed paths,
   identity and remote status. Do not create tags, draft releases or publish.
7. A work-package checklist is not proof of stability. Every acceptance item
   below must point to a command/result, source SHA and artifact/input identity.
   Missing, skipped, cancelled, stale or unavailable evidence stays non-passing.

## Dependency order

```text
W0 baseline and contract census
  -> W1 security controls and non-publishing release fence
  -> W2 report correctness and schema admission
  -> W3 portable setup and test layers
  -> W4 containment, budgets and recovery
  -> W5 artifact-only verification
  -> W6 scientific and real-backend conformance
  -> W7 bounded architecture and documentation convergence
  -> W8 exact-SHA release-readiness rehearsal (no publication)
```

W2 and W3 can be investigated independently after W0. W4's threat-model work
can begin in parallel, but integration and full-suite execution stay serialized.
The dependency chain orders evidence integration, not needless delay: the
small semantic-key and other W6 regression fixes can start after W0 alongside
W2. [SUBSYSTEMS.md](SUBSYSTEMS.md) adds the exact SA/SCI-B/RTI/ORCH acceptance
items and their work-package mapping. Every mapped item is in scope.

W7 must not become a prerequisite for the small confirmed source fixes: obtain
regression protection first, then remove duplication in bounded slices.

## W0 — freeze a reproducible baseline and define the supported surface

**Addresses:** every finding's provenance; RR-18. **Prerequisites:** none.
**Owner role:** integration coordinator with independent verifier.

### W0 Changes to make

- Fetch newest main, record its full SHA and complete tracked-file manifest,
  and compare it with this audit's baseline. Mark each finding open, fixed,
  superseded or not reproduced, with evidence. Do not blindly reapply old work.
- Inventory every top-level CLI command, MCP tool, persisted instrument,
  backend capability, report schema/version, output artifact and comparator.
  For each, specify supported platforms/interpreters/extras and whether it is
  stable, experimental or explicitly unsupported.
- Write one contract table linking producer, admission validator, independent
  verifier, renderer, comparator, documentation and installed-artifact tests.
  An intentional missing component needs an explicit user-visible reason.
- Define boundary invariants: exact JSON types; maximum bytes/depth/items;
  finite/undefined values; populations/denominators; provenance versus
  attestation; completion/abstention; default redaction; and stable error codes.
- Design an executable readiness entry point that mirrors the full CI surface
  and emits a structured per-lane verdict. It is **proposed work**: no existing
  `tools/local_ci.py` or equivalent should be assumed to exist in this repo.

### W0 Acceptance

- The manifest's deduplicated count equals `git ls-files`; no omitted tracked
  path, silently truncated pagination, or uncategorized instrument remains.
- Two clean environments can run the documented setup recipe. Tool versions
  come from the checked-in locks, not ambient executables.
- A cold verifier can identify every required check and every excluded
  capability without the previous agent's scratch directory or conversation.
- The baseline is recorded as not release-ready, not retrospectively green.

**Suggested commit:** `docs: define stable-release capability and contract matrix`.

## W1 — restore effective security controls and fence publication

**Addresses:** RR-01, RR-05, RR-06, RR-17. **Depends on:** W0.
**Owner role:** security/CI maintainer; credential custodian for external access.

### W1 Changes to make

1. Correct Gitleaks configuration so the pinned default rules are loaded.
   Replace synthetic fixture strings that unnecessarily resemble credentials
   with test data that preserves the sanitization test's actual invariant.
   Remove new reliance on scanner allowlists. Do not print token-shaped values
   in audit output or replicate them into a findings document.
2. Add a scanner-sensitivity test in a disposable directory/repository. Its
   never-issued canary must be detected by the exact configured CI command;
   a benign negative control must not be detected. Run both directory and
   history paths if CI promises full-history scanning. Remove the canary.
3. Resolve historical synthetic matches without hiding them or rewriting
   history without approval. If the standing no-suppression policy and the
   immutable history gate conflict, document the precise conflict and obtain
   the required narrow decision; do not quietly weaken detection.
4. Restore DeepSource inventory access using the approved credential channel.
   Run `tools/deepsource_inventory.py` for the exact new head; verify complete
   pagination, requested/returned source identity, enabled analyzer set and
   per-occurrence records. Fix source findings or leave them open and listed.
5. Inventory all genuine Ruff/Bandit/other scanner exclusions. Remove them
   through professional source changes in bounded waves; preserve deliberate
   sanitization boundaries and tests, not their suppression syntax.
6. Design a non-publishing readiness workflow and a release-policy verifier.
   It must bind successful CI, effective secret scanning, dependency/security
   audit, fresh required analyzer evidence and artifact-only tests to one SHA.
   The tag path must reject a stale/missing receipt or the wrong source.
7. Keep ordinary evidence per SHA so superseding pushes do not cancel the only
   acceptable result for a release candidate. Review permission minimization,
   immutable action pins and artifact origin on the actual workflow graph.

### W1 Acceptance

- The old Gitleaks configuration fails the sensitivity test; the repaired
  configuration detects the canary. An empty rule set cannot return a passing
  readiness receipt. Whole-tree and history results are separately enumerated.
- DeepSource's exact-head inventory is complete and healthy. Missing access,
  skipped analysis or a stale inventory remains a blocking state. A green
  vendor badge alone is insufficient.
- GitHub's full current alert inventory is reconciled. The original filelock
  alerts remain fixed on latest main; no obsolete downgrade is applied.
- Unit tests reject absent, failed, cancelled, skipped and wrong-SHA evidence,
  incorrect artifact hashes, wrong package version and unauthorized ancestry.
- A dry run can produce evidence but has no publication credentials or write
  path to PyPI/releases. A publishing job cannot consume arbitrary artifacts.
- Scanner findings are not dismissed. If a required external credential is
  unavailable, the limitation is explicit and readiness remains blocked.

**Suggested commits:** separate scanner repair, analyzer-source remediation and
release-fence changes; do not combine them with report refactoring.

## W2 — make every report boundary total, honest and complete

**Addresses:** RR-02, RR-03, RR-04, RR-09. **Depends on:** W0.
**Owner role:** report-contract engineer and independent oracle reviewer.

### First correctness wave

- Pin public reproductions for scalar frontier `decision_steps`, malformed
  frontier `flip_ranges`, invalid `budget_robust`, and mixed-type precision
  `unavailable_candidates`. Test the valid neighboring values in both input
  orders. Assert exit code, both output streams, and no traceback/partial
  success artifact together.
- Validate container/element shapes before sorting, iteration, hashing,
  conversions and subscripting. Bound nested work before allocation. Preserve
  existing valid payloads and documented section-local degradation.
- Add negation section admission and comparison, including protocol drift,
  candidate changes, numerical readouts, headline states, missing/invalid
  sections and summary coverage. Reproduce the genuine 16/16 versus 0/16
  producer-pair change; do not test only arbitrary hand-built top-level keys.
- Tighten precision ground-record boolean/None fields, candidate pass flags and
  numeric-domain checks. Correct float recompute equality so a JSON bool is
  never a number. Check nested structures and every consumer of that helper.

### Additional W2 boundary contracts

- **ORCH-01:** repair default-public versus opt-in evidence projections for all
  optional sections, including transfer paths and composed serving token
  streams. Assert sentinel absence and preserve caller-owned input.
- **RTI-03/04/06:** validate cheap arguments before expensive work, admit
  strict bounded serving records/results, and reject invalid MCP envelopes
  and falsey non-object argument values before handler dispatch.
- **SA-01/03/04/05:** use core generation validation, preserve duplicate-member
  ambiguity, avoid float conversion of arbitrary finite integers and compare
  enums with the JSON value model. Include the documented valid controls.
- **ORCH-04/D01:** distinguish unobservable stop cells from measured no-fire
  results and validate supported envelope/section contracts before export.

### Contract-completeness wave

- Make instrument registration drive an explicit coverage census, not hidden
  duplicate lists. Every persisted stage must have a tested diff policy or a
  documented opt-out; unsupported new sections must not disappear silently.
- Distinguish absent, invalid, unsupported, unavailable and valid evidence.
  Expose which axes were compared. Do not present two invalid sections as
  validated equivalence. Keep the stable public vocabulary compatible through
  additive metadata or version/migration policy rather than casual breakage.
- Include held-out planned/complete counts and other evidence-strength drift
  in digest-only comparisons. Keep raw fixture/evidence export opt-in.
- Construct the complete machine and human outputs before writing either;
  never emit a success-looking result and then fail formatting the summary.
- Specify report versioning and migration expectations across the older
  sample, existing optional sections and serving-backend artifacts.

### W2 Acceptance

- Each demonstrated defect has a regression that fails on the audited tree
  and passes after the repair. A distinct valid control still passes.
- A bounded cross-family property matrix covers null/bool/int/float/string/
  list/object substitutions, missing keys, wrong element shapes, huge finite
  integers, overflowing exponents, non-finite constants, duplicate fields,
  invalid UTF-8, nesting limits and excessive item counts.
- For admitted bounded JSON, validators return a documented result instead of
  throwing. CLI rejection/degradation is deterministic and traceback-free.
- All producer-generated optional sections have dispatch coverage. Asymmetric
  availability is not lost; raw observations are never required to report
  ordinary denominator or protocol drift.
- Independent expected values come from definitions/reference vectors.
  Changing production code and an oracle to agree is not the evidence.
- A malicious HTML/terminal-value matrix proves escaping, control-character
  handling, default redaction, CSP and no unintended network assets.

**Suggested commits:** input boundary repair; negation coverage; exact-type
admission; then explicit comparison-evidence states.

## W3 — make setup, typing and test layers portable

**Addresses:** RR-07, RR-08, RR-18. **Depends on:** W0; security pins from W1.
**Owner role:** packaging/toolchain maintainer and platform verifier.

### W3 Changes to make

- Generate locks from actual supported target environments. Add environment
  markers only when justified by upstream metadata and real resolution; avoid
  a single Linux freeze masquerading as a universal dependency closure.
- Verify every direct/transitive constraint and `Requires-Python` pairing.
  Resolve base, dev, HF, semantic-search, documentation, auditing and build
  environments independently. Do not arbitrarily bump incompatible families.
- Split dependency-free tests, optional HF integration tests, POSIX-specific
  behavior and provisioned-agent-tool tests into explicit layers. Core suite
  collection must not import torch merely to find tests. Optional-layer skips
  are valid only in the base layer; the full HF gate must execute that layer.
- Provide a platform abstraction for optional resource/signal APIs that mypy
  can verify on Windows as well as Linux, without blanket ignores.
- Document exactly which setup installs Bandit and other tools used by the
  gate. Reconcile `.[dev]`, `.[dev,hf]`, locked CI installs and contributor docs.
- Extend Python/OS coverage according to the explicit support matrix rather
  than an untested unbounded metadata promise. Newer interpreter lanes may be
  exploratory until supported upstream dependencies are verified.

### W3 Acceptance

- Fresh hash-verified installs and `pip check` succeed for each supported
  lock/OS/interpreter combination. Unsupported combinations are refused with
  a precise message before a large download, not left as resolver surprises.
- The semantic launcher can install, index and search on its declared targets;
  a version marker alone does not prove its environment matches the lock.
- Base/core test collection succeeds with ML imports intentionally unavailable;
  the HF lane collects and executes its complete integration suite.
- Pinned mypy passes on actual supported hosts; the audited Windows 34-error
  repro is closed without suppression or a pretended Linux environment.
- No test uses a live credential store, private model directory, display,
  external account or uncontrolled network as an implicit prerequisite.

## W4 — establish resource, privacy and failure guarantees

**Addresses:** RR-10, RR-14 and relevant backend findings. **Depends on:** W2, W3.
**Owner role:** runtime engineer and adversarial isolation reviewer.

### W4 Changes to make

- Execute untrusted template rendering in an owned worker process with
  portable wall-clock cancellation and bounded output/memory/work. If a
  supported platform cannot enforce the needed guarantee, refuse the hostile
  operation explicitly rather than rendering unbounded and calling it safe.
- Preserve fixed public error vocabulary and redact arbitrary backend/template
  exceptions. Never replace a failed measurement with fabricated observations.
- Implement a run-level budget model for base search, held-out, each requested
  panel, transfers, scoring, retries and full-prefix decode work. Estimate
  costs before expensive imports/loads, then reconcile actual charges.
- Add durable per-cell progress and partial evidence outside the public final
  report. Resume only after matching model/tokenizer/protocol/dataset/source
  identities, seeds, selected candidate and frozen held-out policy.
- Define cleanup for cancellation and all failure phases: workers, HTTP
  responses, sockets, temp files, model references, output claims and logs.
  No detached worker may remain after a failed timeout.
- Audit output permissions/ACL behavior and redaction on each supported OS.
  POSIX `0o600` does not by itself prove a private Windows ACL.

### Additional W4 lifecycle and resource contracts

- **ORCH-02/03:** keep cancellation active across every optional stage; use one
  owner event and lazy sequential transfer residency, including the primary
  in admission. Weak-reference lifecycle checks precede approved real memory
  measurements; real signal handling needs its own bounded integration test.
- **RTI-05/08:** always attempt acquired-cache teardown and normalize both
  success/error HTTP-body failures without masking the original cause.
- **SCI-B-04 / RTI-07:** make cache prefix storage approximately linear and
  enforce the final encoded debugger-save byte limit on the opened file.
- **ORCH-D02:** define a report-pair completion boundary and interrupted-write
  recovery; retain current exclusive creation and Python-exception cleanup.
  Do not silently inflate that existing guarantee into crash durability.

### W4 Acceptance

- Adversarial templates time out from main and worker threads on every
  supported platform; the parent remains responsive and child resources are
  reclaimed. A post-render byte cap is not accepted as the timeout proof.
- Cancelling before startup, during generation and while writing produces
  documented incomplete evidence; incomplete candidates cannot win.
- Repeated aborted runs leave no tool-owned workers, open listeners, secret
  files or claimed output directories. Cleanup is scoped, never a blanket
  kill of user processes or an OS/driver/VM reset.
- An interrupted tiny run resumes to the same deterministic result as an
  uninterrupted run; incompatible identities and held-out retuning are refused.
- Private sentinels are absent from default JSON/HTML/terminal output, and only
  the explicitly permitted opt-in evidence output can contain them.

## W5 — prove artifacts independently of the checkout

**Addresses:** RR-11, RR-12. **Depends on:** W1 through W4 as applicable.
**Owner role:** distribution verifier, independent of the builder.

### W5 Changes to make

- Run wheel smoke from an empty external working directory, with checkout and
  ambient `PYTHONPATH` excluded. Assert the actual imported file belongs to the
  installed artifact, not merely that an import succeeded.
- Install and test the canonical sdist in another clean environment. Make its
  declared test/verification scope self-contained; otherwise separate and name
  checkout-only tests instead of shipping a misleading incomplete harness.
- Enumerate archive members. Wheel content must be intentional package code,
  typing marker, license and package data; no tests/private reports/model
  weights/caches/personal paths. Validate license/provenance inputs for data
  actually distributed.
- Build twice from independent clean inputs using the pinned backend and
  canonical epoch; compare hashes and normalized archive metadata. Rebuild a
  wheel from the distributed sdist and explain any scoped differences.
- Produce a manifest of artifact hashes, package version, source identity,
  dependency inventory/SBOM and readiness evidence. Verify attestations bind
  those exact hashes. No package upload is needed for this exercise.

### Additional W5 artifact contract

**RTI-02:** make the pack producer enforce verifier entry/aggregate-size limits
including manifest overhead. Every successful emitted bundle must pass its own
container limits. Exercise exact/one-over limits using tiny files and reduced
byte budgets; refuse before output creation, not after retaining all input.

### W5 Acceptance

- Every top-level command prints help from the installed wheel; offline
  catalog, inspect, fixtures round trip, sample diff, pack/verify and MCP
  protocol operations work without the checkout. Non-empty shape assertions
  prevent two empty outputs from masquerading as deterministic equivalence.
- The from-sdist verification recipe executes exactly the documented scope.
- Two builds match under the declared reproducibility conditions; cross-OS
  equality is claimed only if actually measured.
- Wrong-version, unsafe-link, accidental private file, malformed metadata and
  altered artifact controls are rejected before any publish-capable job.

## W6 — validate scientific meaning and real backend behavior

**Addresses:** RR-13, RR-18 and scientific follow-ups. **Depends on:** W2–W5.
**Owner role:** method reviewer plus real-backend verifier.

### W6 Changes to make

- For every probe, state the estimand, experimental unit, included population,
  exclusion reasons, missingness, paired controls, stopping rule, thresholds,
  uncertainty method and falsification conditions. Treat stage-attested claims
  separately from facts recoverable from digest-only records.
- Re-derive statistical primitives and representative full panels from primary
  definitions. Test zero/one/all-pass/all-fail, ties, incomplete populations,
  duplicates, correlated groups, empty strata, undefined aggregates, and
  multiple-comparison behavior. Never renormalize away missing inputs.
- Select one legally usable, already-provisioned supported checkpoint with a
  genuine native-format positive control. Freeze input revisions, tokenizer,
  exact boundary IDs, fixtures, budget and selection policy before execution.
  Keep previously completed negative rows as negative evidence.
- Demonstrate the complete search -> freeze -> held-out -> report -> pack ->
  verify workflow on a positive case plus tie/no-candidate/failure controls.
  If no real positive case can be established, retain experimental labeling
  rather than weaken the rule.
- Compare the lab with the real pinned serving stack: encoded IDs must match.
  Check boundary/EOS behavior, completion counts, cache isolation and fidelity.
  Faithful protocol fakes and actual binaries are separate evidence layers.
- Validate intended accelerator/precision behavior with a small compute smoke
  first, resource-bounded sequential model residency and approved supported
  runtime versions. Do not change drivers/system configuration or infer device
  placement from only the first parameter. Never run large loads merely to
  satisfy a nominal coverage target.

### Additional W6 scientific corrections

- **SCI-B-01:** independently review and repair ambiguous frozen position
  keys; include all positions and primary semantic references.
- **SCI-B-03 / SA-02:** use common fixture/candidate cohorts for comparative
  headlines and conserved usable/unusable counts. Permute missing arms and
  preserve genuine complete-case effects.
- **SCI-B-02/04:** separate exact prompt duplicates from shared prefixes; keep
  the independently verified cold-prefix hit formula while reducing storage.
- **SA-03/04/05:** re-derive duplicate JSON, huge finite integer and cross-type
  enum expectations rather than copying the flawed differential oracle.
- Independently audit frozen Unicode mappings and catalog render fidelity,
  and derive/calibrate the typographic equivalence certificate before making
  a stable statistical-certification claim. These are evidence gaps, not
  demonstrated defects in every corresponding formula.

### W6 Acceptance

- Expected values are independently derived and cited; test/mirror agreement
  alone is not the oracle. A planted arithmetic/protocol defect is detected.
- Populations, denominators and abstentions reconcile across producer, JSON,
  HTML, independent verifier and diff output for every supported instrument.
- A real positive-path record and deliberate negative controls are retained
  with synthetic/non-private fixtures and sufficient provenance to reproduce.
- Unsupported architectures and stack versions refuse cleanly. Emulated
  semantics are not mislabeled as measured production-stack equivalence.
- The legal/license decision and private asset provisioning are explicit. If
  authorization or hardware access is unavailable, that cell remains blocked,
  not simulated and not claimed green.

## W7 — reduce structural debt without rewriting the product wholesale

**Addresses:** RR-15, RR-16, RR-17. **Depends on:** regression contracts from W2–W6.
**Owner role:** architecture maintainer with independent behavior reviewer.

### W7 Changes to make

- Split report admission, schema primitives, per-instrument comparison and
  human rendering into bounded modules. Extract orchestration by responsibility
  and lifecycle, not by historical cycle number. Keep public import/CLI
  compatibility intentionally tested.
- Introduce a typed instrument registry describing producer, validator,
  comparator, rendering and evidence policy. Do not share one implementation
  between both sides of a claimed independent correctness oracle.
- Replace text-replacement mutation anchors as architectural constraints with
  reliable isolated mutation tooling or tested semantic seams. A test must
  fail for the intended mutant, not an unrelated import/parse failure. Require
  a passing unmutated baseline and parsed non-empty intended test outcomes;
  child exit 2, timeout, collection/import or syntax failure is inconclusive.
- Organize tests by contract/layer and retain regression traceability. Account
  for collected test deltas; do not delete hard tests to reduce runtimes.
- Remove tracked build metadata/generated site output if they are not release
  inputs; if anything remains versioned, give it a deterministic generator and
  freshness gate. Keep one version authority.
- Reconcile living README, architecture, security and usage docs with serving
  backends, exact current metrics and the support matrix. Keep historical audit
  records explicitly historical. Add truthful input/output/invariant docstrings
  rather than mechanically inserting vacuous descriptions.

### W7 Acceptance

- Every extraction preserves valid deterministic artifacts and public commands
  under independent cross-checks. Deliberate schema changes have migration tests.
- There is one complete instrument census, with no missing negation sibling.
- Lint/type/security checks cover tools/fuzz as well as package code where
  applicable. Documentation checks cover newly added files, not only tracked
  modifications. Genuine scanner issues remain fixed/open, never hidden.
- Full tests and meaningful mutation/property checks remain sensitive. Coverage
  is reported as statement and branch metrics separately; floors are not lowered.
- A new agent can run the complete gate without private historical work logs.

## W8 — rehearse a stable release without releasing

**Addresses:** all release blockers. **Depends on:** W1–W7.
**Owner role:** independent readiness reviewer, not the implementation writer.

### Required evidence packet

- Full source SHA, clean-tree identity, supported capability matrix and complete
  test/check inventory. Any concurrent source change invalidates the packet.
- Exact locked setup versions; full supported test matrix, all expected test
  counts and explicit skip reasons; Ruff check and format; strict mypy; Bandit;
  dependency audit; effective full-history secret scan; CodeQL; complete fresh
  DeepSource inventory; docs build and artifact-only verification.
- Confirmed fixes for each open P1 finding and resolution/re-scope for every
  release-required P2 acceptance item. Explicit open risk records for anything
  excluded from the supported release surface, never a silent omission.
- Reproducible wheel/sdist, metadata checks, checksums, SBOM/license inventory,
  provenance verification and clean installed functional tests.
- Real-backend evidence for the declared stable measurement contracts, with
  privacy review of every public artifact.
- A dry-run workflow result with no release/tag creation, no PyPI upload and
  no publishing permissions. Test its rejection paths, not only its happy path.

### Existing baseline commands to preserve

These are existing commands, not proof that they pass on an arbitrary setup:

```sh
python tools/agent_context.py startup
ruff check model_protocol_lab tests examples tools fuzz
ruff format --check model_protocol_lab tests examples
ruff format --check .
mypy model_protocol_lab
bandit -r model_protocol_lab -q
coverage run -m unittest discover -s tests -t .
coverage report --fail-under=95
python -m pytest --collect-only -q -p no:cacheprovider
mkdocs build --strict
```

The final gate must additionally run the new regression/property/installed-
artifact/security-sensitivity work above and every actual CI lane. No single
command in that block currently supplies the whole readiness decision.

### Final acceptance and stopping rule

1. Read the complete exact-head GitHub check matrix, not a selected tail or
   only the main CI workflow. Reconcile every required lane and external
   analyzer. Queued/skipped/missing results do not count as passing.
2. Have an independent reviewer attempt to invalidate the evidence packet,
   including stale artifacts, wrong imports, empty comparisons, timeout leaks,
   weakened scanners and circular oracles. Repair and rerun affected gates.
3. Commit the final readiness report through the official GitHub connector,
   verify GitHub-assigned identity and read back the exact changed files.
4. **Stop.** Report ready/not-ready and the remaining scoped risks. Do not
   create any tag, draft/public release, package upload or publishing dispatch.
   The owner decides publication separately.

## Human-only decisions versus agent-executable work

Agent-executable: source/test repairs, portable lock generation with ordinary
public dependencies, contract inventories, artifact builds, safe fixtures,
non-publishing workflows and read-only verification.

Potentially external: missing analyzer credentials, private/gated checkpoint
license/access decisions, publishing-identity/environment registration and any
proposed history/governance change. Ask for only the missing authorization or
secure credential connection; never request secrets in chat. Continue the
independent source work while a genuine external dependency is pending.

The absence of a credential does not authorize inventing a result, disabling a
gate, accepting a license for the owner, or declaring the release stable.
