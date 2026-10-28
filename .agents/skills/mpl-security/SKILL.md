---
name: mpl-security
description: Investigate model-protocol-lab security findings with source-bound evidence; retrieve the complete DeepSource inventory and respect the findings guard thresholds.
origin: adapted pattern from strmt7/FlitzZip .agents/skills/flitzzip-security/SKILL.md; content is model-protocol-lab specific
---

# MPL Security

Follow `AGENTS.md` and `docs/SECURITY.md`. Those documents own
authorization boundaries, source admission, and the release gate.
A finding is a diagnostic until it is traced to a demonstrated
defect in this tree; a passing policy test, a dashboard dismissal,
or a green unrelated workflow does not establish a source repair.

## Investigation

1. Reuse a complete finding inventory only after matching its
   source, scanner, and configuration identities. Read new or
   changed reports in full. Keep individual finding evidence;
   never collapse distinct findings into a count.
2. Trace the trust boundary and failure propagation through direct
   callers and sibling implementations. Preserve the original
   failure and an independent oracle where practical.
3. Repair the production mechanism (or the test, when the test is
   the defect — see the cycle-169 rule in `docs/RESEARCH_LOG.md`:
   when a test contradicts the design doc's fail-closed semantics,
   the test is wrong). Apply the engineering learning loop: record
   the root cause as a rule for the next run.
4. Run the actual pinned detectors for the changed area and inspect
   raw output. A guard that never fired proves nothing.

## DeepSource backlog

For the complete DeepSource backlog, use the paginated inventory
procedure in `docs/SECURITY.md` ("DeepSource analysis"):
`tools/deepsource_inventory.py --repository strmt7/model-protocol-lab
--commit FULL_MAIN_SHA --output <new directory>`. Require its
complete, source-matched manifest; a green commit check or a capped
CLI export does not establish repository-wide disposition. The
connection is the `deepsource-io` GitHub App with no repo config
file; read app presence from GitHub check suites, never from check
runs alone. A missing credential is NOT-CONFIGURED, never zero
findings. The enabled analyzer set is read from the repo's own
DeepSource configuration (`configJson` in the inventory manifest);
never copy another repository's set.

## Findings guard

The findings guard (`.github/workflows/findings-guard.yml`) fails
when open CodeQL code findings exceed two, or when any DeepSource
finding exists for the audited commit. CodeQL findings are counted
by tool name only — Scorecard posture checks are not code findings.
Keep findings and unresolved evidence visible; do not turn this
skill into a count-based approval ledger.
