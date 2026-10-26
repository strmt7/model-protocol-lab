# Support

## Where to ask

Open an issue using one of the templates in `.github/ISSUE_TEMPLATE/`.
Before filing, check `docs/FAQ.md` and `docs/USAGE.md` — most setup and
output questions are already answered there.

## What to expect

This project is maintained by one person, on a best-effort basis.
There is no SLA and no guaranteed response time: issues are triaged as
time allows, and security reports take priority over everything else.
That is stated plainly here rather than implied — if you need a
guaranteed response, this is not that project yet.

## What a good report contains

Copy the checklist from `CONTRIBUTING.md` ("Reporting issues"): the
exact command, the full error output, your Python version, and how the
package was installed. For adapter problems, add the `doctor` output
for the checkpoint. A minimal synthetic reproduction beats a long
description, and it must never contain credentials, proprietary
prompts, private model files, or personally identifying data.

## Security reports

Do not file security defects as public issues. Use GitHub's private
vulnerability reporting for this repository (Security tab → Report a
vulnerability), as documented in `docs/SECURITY.md`. Reports there are
visible only to the maintainer until a fix ships.

## Supported versions

See the supported-versions policy in `docs/SECURITY.md` (latest
release line only; Python versions per `requires-python` and CI).
