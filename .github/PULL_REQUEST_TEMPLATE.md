## What this changes

<!-- One paragraph: behavior, not implementation. -->

## Verification

- [ ] `coverage run -m unittest discover -s tests -t .` passes (the exact CI command)
- [ ] `ruff check` and `ruff format --check` pass on `model_protocol_lab tests examples`
- [ ] `mypy model_protocol_lab` passes
- [ ] `bandit -r model_protocol_lab -q` is clean
- [ ] New behavior has a regression test (or explains why not)

## Documentation

- [ ] `docs/USAGE.md` updated for any CLI or output-contract change
- [ ] README Capabilities names any new `run` probe flag
- [ ] `CHANGELOG.md` entry added under Unreleased (per-cycle heading form)
- [ ] `VALIDATION.md` updated if the change moves anything across the evidence boundary

## Checklist

- [ ] No model downloads, no remote code, no weakened adapter guarantees
- [ ] Public CLI behavior unchanged (or the breaking change is called out above)
- [ ] Reports remain byte-deterministic; no new external assets in HTML output
- [ ] No secrets, weights, or private prompts committed
