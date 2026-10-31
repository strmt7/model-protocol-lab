# Security policy

The authoritative security policy — trust boundaries, the dated audit
log with findings and explicitly accepted residuals, supported
versions, and how to verify a release — lives in
[`docs/SECURITY.md`](docs/SECURITY.md). This root file exists so the
policy is visible at the top level, where security reviewers look
first; it is a pointer, not a second copy, so the two can never drift.

**Reporting a vulnerability:** use GitHub's private vulnerability
reporting for this repository (Security tab → Report a
vulnerability, or
https://github.com/strmt7/model-protocol-lab/security/advisories/new).
Reports are visible only to the maintainer until a fix ships. Never
file security defects as public issues.

**Supported versions, in one line:** security fixes land on the latest
release line only; older releases receive no backports. The full
statement is in `docs/SECURITY.md` ("Supported versions").

**What to expect:** one maintainer reads every report. Acknowledgement
is best-effort — there is no guaranteed response time and no bug
bounty programme. That is stated here, on the page GitHub renders,
rather than only behind the pointer.

Do not edit the policy here — edit `docs/SECURITY.md`.
