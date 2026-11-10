# Release-readiness audit

**Audited source:** `d8a1346343e1b50b28bf567ab6b193b38fc79608` (`main`).
**Audit date:** 2026-11-10. **Decision: not ready for a stable release.**

This is an assessment and implementation plan, not a release certification.
The audit commit changes documentation and evidence only. No product fix,
workflow repair, dependency update, tag, release, or publication is authorized
by this document.

## Read this package

1. [Audit report](REPORT.md): verified state, findings, scope and limitations.
2. [Implementation plan](PLAN.md): ordered work packages and measurable exit gates.
3. [Reproduction recipes](REPRODUCTIONS.md): small checks for the principal defects.
4. [Evidence ledger](evidence.json): sanitized measurements and exact-head CI references.
5. [File inventory](inventory.csv): one row for each of 579 tracked files.
6. [Agent handover](../../HANDOVER.md): continuation and publication rules.
7. [Subsystem findings](SUBSYSTEMS.md): additional validated boundary defects.

The inventory is exhaustive over tracked paths at the audited commit. Review
intensity is not uniform: byte inventory and structural scans cover the whole
tree; manual review and adversarial execution concentrate on trust boundaries,
measurement logic, orchestration, persisted reports, and release machinery.
Neither file coverage nor a passing test suite proves absence of defects.

## Important snapshot correction

The first pass inspected `27bf1a2b863148d946d41374d389d1cfd583bcd4`.
Before resuming, the audit fetched current `main` and inspected the actual delta:
only `CHANGELOG.md`, `docs/SECURITY.md`, and
`tools/requirements/cocoindex.txt` changed. The latest commit updates filelock
and litellm. GitHub now reports **zero open Dependabot alerts**. The earlier
13-alert count is historical and is not a current release blocker.

The findings guard still fails on the audited head because DeepSource is not
configured in its runner. That is an unavailable security signal, not evidence
of zero findings. Do not disable the guard to obtain a green badge.

## Meaning of stable

A stable release needs reproducible, source-bound evidence that the supported
product contracts work, invalid inputs fail safely, measurement results remain
honest, artifacts run independently of a checkout, and publication cannot bypass
the release gates. “Perfect” or “no mistakes” cannot be established by a finite
audit. The plan therefore uses explicit acceptance tests, negative controls,
remaining-risk records, and a no-publication dry run instead of that promise.
