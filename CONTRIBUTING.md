# Contributing

## Setup

Python 3.11+ is required. Use a dedicated virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"      # lint, types, tests, coverage
python -m pip install -e ".[dev,hf]"   # additionally: the local HF adapter stack
```

## Before opening a pull request

Run the full verification from the repository root. All four must be clean:

```sh
python -m unittest discover -s tests -t .
ruff check model_protocol_lab tests examples
ruff format --check model_protocol_lab tests examples
mypy model_protocol_lab
```

If you touched `model_protocol_lab/reporting.py`, also run the report
validator tests:

```sh
python -m pytest tests/test_sample_report.py
```

If you changed the sample report inputs, regenerate it — the golden test
requires byte-identical output:

```sh
python examples/make_sample_report.py
```

## Conventions

- **Evidence over claims.** Do not claim a behavior you have not executed. If
  you fix a bug, add a failing-then-passing test that reproduces it first.
- **Security guarantees are load-bearing.** The adapter must never download
  models, never execute remote model code, and never load non-safetensors
  weights. Any change near `backends.py` needs a test proving the guarantee
  still holds.
- **Public CLI stays stable.** Renaming or removing a command or flag is a
  breaking change; discuss it in an issue first.
- **Determinism.** Report output must be byte-identical for identical inputs:
  no timestamps, no random IDs, no dict-ordering assumptions.
- **Doc honesty.** `VALIDATION.md` keeps an explicit "not established" list.
  Update it when your change moves something across the line — in either
  direction.
- Keep lines within 100 columns (`ruff format` enforces this).

## Reporting issues

Use the issue templates in `.github/ISSUE_TEMPLATE/`. Include the exact
command, the full error output, your Python version (`python --version`), and
how the package was installed. For adapter problems, include the `doctor`
report (redact paths if needed) — never attach model weights.
