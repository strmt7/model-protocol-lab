## What this changes

<!-- One paragraph: behavior, not implementation. -->

## Verification

- [ ] `python -m unittest discover -s tests -t .` passes
- [ ] `ruff check` and `ruff format --check` pass on `model_protocol_lab tests examples`
- [ ] `mypy model_protocol_lab` passes
- [ ] New behavior has a regression test (or explains why not)
- [ ] `VALIDATION.md` updated if the change moves anything across the evidence boundary

## Checklist

- [ ] No model downloads, no remote code, no weakened adapter guarantees
- [ ] Public CLI behavior unchanged (or the breaking change is called out above)
- [ ] Reports remain byte-deterministic; no new external assets in HTML output
- [ ] No secrets, weights, or private prompts committed
