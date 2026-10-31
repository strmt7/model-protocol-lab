# Contributing

Thank you for helping improve Model Protocol Lab. This document is the
complete route from a fresh checkout to a mergeable change: setup, the
verification gate, the conventions the project actually enforces, and
how to report problems. The maintainer reviews everything; the public
CLI is stable, so anything that would rename or remove a command or
flag is discussed in an issue first. By contributing, you agree that
your contributions are licensed under the project's MIT license.

## What belongs here

Wanted: bug fixes with a reproducing test; documentation corrections;
new format families backed by a primary source for the format; new
probes that start from a design document (see "Documentation
obligations"). Not wanted, by design: anything that downloads models
or executes remote code, hosted-service or network features (the lab
is local-only), and benchmark leaderboards or rankings (the selection
rule abstains on ties rather than ranking — that is the point of the
tool). For anything larger than a trivial fix, open an issue first so
the direction is agreed before the work is done.

## Setup

Python 3.11 or newer is required (CI tests 3.11, 3.12 and 3.13). Use a
dedicated virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"      # lint, types, tests, coverage
python -m pip install -e ".[dev,hf]"   # additionally: the local HF adapter stack
```

The `[hf]` extra pulls in PyTorch; on a CPU-only Linux machine, install
the CPU wheel first (`python -m pip install torch --index-url
https://download.pytorch.org/whl/cpu`) so the CUDA build is not
downloaded. The package itself has zero runtime dependencies — every
extra is development- or adapter-only.

Optional: the repository ships a pre-commit configuration
(`.pre-commit-config.yaml`, ruff hooks) if you want the lint and format
checks to run on every commit.

## The verification gate

Run the full verification from the repository root before opening a
pull request. This is the same gate CI enforces; all of it must be
clean:

```sh
ruff check model_protocol_lab tests examples
ruff format --check model_protocol_lab tests examples
mypy model_protocol_lab
bandit -r model_protocol_lab -q
coverage run -m unittest discover -s tests -t .
coverage report        # the gate is 95% line coverage
```

`python -m pytest -q` runs the same suite and is fine for iteration,
but the coverage invocation above is the exact CI command and the one
whose verdict counts. The full suite is thorough rather than quick
(expect roughly 20–55 minutes depending on hardware).

If you touched `model_protocol_lab/reporting.py`, also run the report
validator tests:

```sh
python -m pytest tests/test_sample_report.py
```

If you changed the sample report inputs, regenerate it — the golden
test requires byte-identical output:

```sh
python examples/make_sample_report.py
```

## A first contribution

You do not need the full suite to start. For a first change, run just
the test file you touched, plus lint on the files you changed:

```sh
python -m pytest tests/test_<area>.py
ruff check <changed files>
ruff format --check <changed files>
```

Then run the full verification above before opening the pull request.

## Conventions

- **Evidence over claims.** Do not claim a behavior you have not
  executed. If you fix a bug, add a failing-then-passing test that
  reproduces it first.
- **Security guarantees are load-bearing.** The adapter must never
  download models, never execute remote model code, and never load
  non-safetensors weights. Any change near `backends.py` needs a test
  proving the guarantee still holds.
- **Public CLI stays stable.** Renaming or removing a command or flag
  is a breaking change; discuss it in an issue first.
- **Determinism.** Report output must be byte-identical for identical
  inputs: no timestamps, no random IDs, no dict-ordering assumptions.
- **Doc honesty.** `VALIDATION.md` keeps an explicit "not established"
  list. Update it when your change moves something across the line —
  in either direction.
- Keep lines within 100 columns (`ruff format` enforces this).

## Documentation obligations

The docs are part of the change, and several of their invariants are
enforced by tests, so a code change that skips them fails the gate:

- **New or changed CLI surface** (commands, flags, output contracts):
  document it in `docs/USAGE.md`, name new `run` probe flags in the
  README's Capabilities section, and add a `CHANGELOG.md` entry under
  `## [Unreleased]` using the per-cycle heading form
  `### Added (cycle NNN — short title, YYYY-MM-DD)` (or the matching
  Keep-a-Changelog category). Every subcommand and flag of the live
  CLI must be mentioned in the changelog — a test measures this.
- **New measurements or probes** are designed before they are built:
  the design, its sources, and its falsification conditions go into a
  `docs/<NAME>_DESIGN.md` and an entry in `docs/RESEARCH_LOG.md`
  first. The research log is a chronological journal — append, never
  rewrite dated entries.
- **Counts in docs are measured, never remembered.** Test counts,
  family counts, file counts and coverage figures in `README.md` and
  `VALIDATION.md` are pinned to the live tree by tests; re-measure
  with the discovery command when you touch them.
- **Validation evidence:** when a change alters what the project can
  claim, update the top evidence section of `VALIDATION.md`; older
  sections are historical snapshots and keep their own era's numbers.

## What happens after you open a pull request

One maintainer reviews everything, on a best-effort basis — there is
no review SLA (see `SUPPORT.md`). Small, single-purpose pull requests
with a green gate are reviewed first; a polite ping on the pull
request is welcome if a week passes without a response. Draft pull
requests are a fine way to get early direction on a larger change.

## Project governance

This is a single-maintainer project, stated plainly rather than
dressed up: the maintainer makes the decisions, in the open, in
issues and pull requests, and this document — like every other file
here — changes by pull request. There is no succession process and no
committee; if that ever changes, this section changes with it.
Complaints about the maintainer go to GitHub's own reporting tools,
which do not terminate at the maintainer (see `CODE_OF_CONDUCT.md`).

## Reporting issues

Use the issue templates in `.github/ISSUE_TEMPLATE/`. Include the exact
command, the full error output, your Python version
(`python --version`), and how the package was installed. For adapter
problems, include the `doctor` report (redact paths if needed) —
never attach model weights. Security defects never go in public
issues; use the private route in `SECURITY.md`.
