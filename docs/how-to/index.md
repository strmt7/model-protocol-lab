# How-to guides

Each guide here is a route: a concrete task, and the documentation
section that carries the authoritative procedure. The procedures
themselves live in the [usage reference](../USAGE.md) and the
[FAQ](../FAQ.md); this page exists so you can find the right one
from the task you have.

## Set up and pre-flight

- **Validate a checkpoint before running anything** — `doctor`
  checks the asset manifest, model config, tokenizer structure and
  fixture schema without loading weights: [USAGE §2](../USAGE.md).
- **Find the right `--boundary-ids`** — from the checkpoint's own
  tokenizer, never from another model; verify each ID against the
  vocabulary and confirm the tokenization is stable:
  [FAQ](../FAQ.md) and [USAGE §4](../USAGE.md).
- **Check a format against a second checkpoint's tokenizer** — the
  portability matrix screens which catalog formats a second
  tokenizer can even encode as atomic special tokens:
  [USAGE §2](../USAGE.md).

## Author and check fixtures

- **Author fixtures in a spreadsheet** — one CSV row per fixture,
  compiled to the strict schema with the search/held-out discipline
  enforced at build time: [USAGE §9](../USAGE.md).
- **Fuzz the fixture loaders** — `fixtures fuzz` runs the
  grammar-structured hostile-input campaign against the loaders:
  [USAGE §9](../USAGE.md).
- **Understand a contamination warning** — `inspect` compares every
  held-out fixture against the search partition and reports
  duplicate / near-duplicate / paraphrase-suspect pairs:
  [USAGE §1](../USAGE.md).

## Run and compare

- **Compare two completed runs** — `diff-reports` reports what
  changed between two `results.json` files, section by section:
  [USAGE §12](../USAGE.md).
- **Compare backend stop behaviour** — `diff-backends` differentially
  tests how HF, vLLM and SGLang stop-string handling treats the same
  prompt/output pair: [USAGE §11](../USAGE.md).
- **Explore a tokenization interactively** — the `debug` REPL renders
  and steps through a conversation against a dataset or a live
  checkpoint: [USAGE §10](../USAGE.md).

## Verify and seal results

- **Re-derive a result independently** — `verify` replays the
  selection decision from a report directory: [USAGE §13](../USAGE.md).
- **Seal a result bundle** — `model-protocol-lab pack` binds results,
  report and inputs into a bundle with a per-file SHA-256 manifest;
  `verify --bundle` re-hashes offline and replays the decision as an
  independent second layer: [USAGE §13.1](../USAGE.md). The seal is
  unsigned — it proves bit-integrity since packing only against a
  separately kept digest.
- **Audit determinism** — `verify --determinism` re-runs the
  report pipeline and compares bytes: [USAGE §13.2](../USAGE.md).

## Troubleshoot

- **A run failed or a verdict looks wrong** — the
  [FAQ](../FAQ.md) answers the recurring failure modes
  (`local_context_limit_exceeds_declared_window`,
  `context_limit_exceeded`, boundary text/token mismatches) from the
  documented behaviour, each pointing at its authoritative section.
