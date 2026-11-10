# Development handover

## Read first

This handover is for `strmt7/model-protocol-lab`, not a sibling repository.
Read [the audit entry point](audit/2026-11-10/README.md),
[REPORT.md](audit/2026-11-10/REPORT.md),
[SUBSYSTEMS.md](audit/2026-11-10/SUBSYSTEMS.md), and
[PLAN.md](audit/2026-11-10/PLAN.md) before development.

**Audit source:** `d8a1346343e1b50b28bf567ab6b193b38fc79608`.
**Verdict:** not ready for a stable release. **Remedies implemented:** none.
The commit adding this package is documentation/evidence only. It neither
creates a release nor certifies that future changes will be bug-free.

## Verified state and evidence limits

- All 579 original tracked paths were inventoried; all 341 Python files were
  AST-parsed. Review depth and exact source ranges are recorded per file in
  [inventory.csv](audit/2026-11-10/inventory.csv). Inventory is not exhaustive
  manual review. Unlisted semantic coverage remains open.
- Exact-source Linux CI run **42835070179** passed Python 3.11/3.12/3.13;
  each test leg ran 9,676 tests with 3 skips. The displayed aggregate coverage
  was 95%; it is branch-aware, not simply the statement ratio.
- Fresh Ruff/format/Bandit checks passed. Windows pinned mypy produced 34
  errors in 3 files. Dev-only collection produced 15 errors; that incomplete
  environment's results are not the complete HF-enabled suite verdict.
- Wheel/sdist reproducibility, Twine, strict docs build and a clean installed
  offline wheel smoke were exercised. Model inference, real serving sockets,
  hostile rendering on all OSes, real signals and accelerator parity were not.
- Fresh remote inventories at the audit source had 0 open Dependabot alerts,
  0 open CodeQL alerts and 3 open Scorecard posture alerts. The initial 13
  dependency alerts were fixed by an intervening upstream commit, not this
  audit. Do not reintroduce an obsolete dependency plan.
- Findings guard **42835070138** fails because DeepSource access is missing.
  The analyzer's count is unknown, not zero. A skipped app check is not clean.
- Four bounded subsystem reviews and one independent refuter completed after
  the initial eight attempts failed. The coordinator reran their safe probes.
  No private scratch path or credential is required to understand the record.

The [machine-readable ledger](audit/2026-11-10/evidence.json) distinguishes
source checks, historical build evidence, diagnostic failures, fresh remote
state and unexecuted acceptance cells. The latest post-commit remote result
must be read independently; do not infer it from a saved baseline receipt.

## Root causes to address, not another feature cycle

1. **Incomplete boundary contracts:** malformed reports reach consumers;
   exact JSON types drift; negation has no diff axis; MCP envelopes and
   serving artifacts bypass complete validation; private optional-section
   fields escape the default export projection.
2. **Measurement semantics are not globally enforced:** one built-in semantic
   key is ambiguous, few-shot comparisons mix populations, unusable counts
   fail conservation, duplicate tool arguments lose information, and schema
   enums share the implementation's Python-equality defect with their oracle.
3. **Lifecycles are shorter than the whole operation:** cancellation ends
   before optional stages, transfer models remain resident together, cache
   aborts skip teardown, prefix analysis uses quadratic storage, and budgets
   do not account for all work or interrupted-run recovery.
4. **Evidence gates can be green without proving their promise:** Gitleaks
   has no detection rules, one mutation gate accepts collection failures as
   kills, packaging smoke lacks an explicit installed-origin assertion,
   and publishing has no enforced full exact-SHA quality receipt.
5. **Support/documentation debt accumulates:** Linux-only lock entries break
   Windows setup; source typing assumes POSIX; tracked generated metadata and
   living documentation drift from current serving capabilities.

## Start here

1. Fetch newest main and compare its full SHA/tree with the audited source.
   Reproduce or close each applicable finding; do not blindly reapply old work.
2. Execute W0's support/contract census, then W1's effective security gates.
   Independently begin regression tests for default-export privacy and the
   invalid-report/protocol paths. Fix small confirmed defects before refactoring.
3. Correct the scientific P1 semantic-key defect early, without waiting for
   a large real-model campaign. Keep the rest of W6's evidence requirements.
4. Implement one coherent, reviewable wave at a time. Do not add probes or
   catalog families during stabilization. One integration writer, one mutating
   suite per checkout; isolate mutation work from other source readers.
5. Use the exact work-package acceptance tests and rejection controls in the
   plan. Update the finding's status only after fresh evidence for the actual
   new source. Preserve negative controls and meaningful abstentions.

## Existing commands and missing gate work

Start with `python tools/agent_context.py startup`. It loads the repo's
mandatory skills. The audited semantic-search lock cannot install on Windows;
retain that evidence and use exact paths/AST navigation until W3 supplies a
supported setup. Never force an incompatible installation to fake compliance.

Copy actual CI commands from `.github/workflows/ci.yml`, including Ruff check
and format, strict mypy, Bandit/dependency auditing, the coverage/unittest lane,
package build and installed smoke. The broader plan also requires docs,
security sensitivity, fresh external analysis, platform and real-backend work.
There is **no existing complete `tools/local_ci.py`** in this repo. W0 proposes
a machine-checkable entry point; never write that its verdict was run before
it exists. A selected green subset is not the complete release decision.

Do not run the expensive whole suite while another process edits the same
source. Also run `ruff format --check .`: the cycle-44 toolchain test checks
the whole repository, including Python fences in Markdown under pinned Ruff
0.17.3. The static CI job's narrower path list does not cover that test's full
scope. The first audit commit's three Linux legs each ran 9,676 tests and
failed only on formatting in the new reproduction document; its examples were
reformatted without changing their parsed Python ASTs or reproduced outcomes.
The direct whole-tree formatter is separately executable on Windows; the
existing cycle-44 test helper assumes extensionless executable filenames and
fails before tool launch there. Do not add an executable shim to hide that
platform limitation. Read current exact-head GitHub results for the corrective
documentation commit rather than declaring that failed historical run green.

Several existing mutation gates overwrite source and clear bytecode;
use a disposable checkout and verify source blobs after an interrupted run.
A nonzero mutation subprocess exit caused by collection/import/timeout failure
is inconclusive, not evidence that a behavioral mutant was detected.

## Commit and privacy discipline

- Work on main; preserve unrelated work. Commit through the authenticated
  official GitHub connector/API only. Do not use a local manually authored
  commit, supply author/committer fields or change identity configuration.
- Git-data recipe: read current ref and parent tree; create reviewed UTF-8 LF
  blobs; create a tree based on that exact parent; create a commit with only
  message/tree/parents; verify GitHub-assigned author and committer before
  advancing the ref. Update without force. If main changed, reconcile first.
- Read back the exact commit, message, parent, identity and changed-path set.
  Verify every returned blob against the intended content. Remote creation
  does not mean local HEAD was advanced; sync only after verifying the remote
  and protecting any unrelated local changes.
- Run local gates after the final edit, inspect results, then publish as a
  separate action. After publication, read the complete exact-head checks and
  downstream analyzer state. Missing/queued/skipped is not passing. Never
  suppress findings, weaken checks, or edit product code under audit-only scope.
- Keep private account IDs, emails, host/model paths, raw observations and
  credentials out of documents, fixtures, logs and commit messages. Use
  synthetic fixtures with reserved names and inspect all public artifacts.
- Requirements changes need their complete transitive closure/hashes rebuilt
  for the declared targets and checked for compatibility. Re-record any
  intentionally affected provenance/hash pins in the same verified wave;
  do not edit captured historical result identities or manufacture new ones.

## Human-only versus agent-executable remaining work

**Agent-executable:** all ordinary source/test fixes, contract census,
portable public-dependency locks, installed-artifact tests, safe protocol
fixtures, non-publishing readiness workflows, documentation and exact-head
read-back verification. Continue these autonomously after implementation is
authorized; do not stop just because an unrelated external cell is pending.

**Potential external dependency:** approved analyzer credential access;
private/gated checkpoint access and license decisions; publishing identity and
environment registration; any proposed history/protection change. Check the
current supported authentication mechanism before claiming a credential is
required. Use the secure connector/credential channel, never ask for a secret
in chat. Do not accept licenses or change governance without authorization.

## Stop condition

Finish W8 with a complete, independently challenged exact-source evidence
packet. Report ready/not-ready and remaining scoped risks. **Do not tag,
draft a release, upload to PyPI or trigger publishing.** Publication remains a
separate owner decision. Do not promise a mathematically perfect release;
prove the declared supported contracts and keep residual uncertainty explicit.

## Copyable instruction for the next development agent

> Work on the latest main of strmt7/model-protocol-lab. Read AGENTS.md,
> docs/HANDOVER.md and the complete docs/audit/2026-11-10 package. Reconcile the
> audit's d8a1346 source with current main before acting. Implement PLAN.md in
> verified waves: regression first, small correctness/privacy/security fixes
> before structural refactoring, independent scientific oracles, bounded
> resources and complete supported-platform evidence. Fix or explicitly keep
> every finding open; no suppressions, weakened gates, fabricated results or
> personal data. Commit only through the official authenticated GitHub
> connector/API with no manual author/committer fields; verify exact remote
> files and all checks. Continue through release-readiness verification, but
> do not create tags/releases or publish packages.
