# Release notes — model-protocol-lab 0.1.0

**Status: DRAFT notes.** Prepared 2026-10-16 by the release-readiness
audit worker. These notes are pending the owner's release tag; no tag
has been created.

## What this is

Model Protocol Lab measures whether a local language model fails because
of its weights or because its conversations are formatted incorrectly. It
renders a 29-family catalog of text-only chat-format hypotheses (plus
bounded BOS, prefix and thinking-mode mutations) as exact token
sequences, sends them to a local checkpoint, and compares task answers
and response boundaries token-for-token. A strict search/held-out
fixture discipline selects a single winning format or abstains — no
leaderboards.

## What's new in 0.1.0

This is the first runnable release (the `0.1.0a0` label marked a source
snapshot that was never executed). Since that snapshot:

- **Catalog 3 → 29 families**, each hand-rendered from primary sources
  (official `chat_template.jinja` / `tokenizer_config.json`, preserved
  byte-identical under `docs/evidence/`): ChatML; Llama 3/4; Gemma 2/3/4;
  Qwen3/3.8; DeepSeek V3/R1/V4.1; Mistral v0.3/Large; Phi-4; GLM-4/4.5/5.3;
  Kimi K2/K3; Kolibri-1; LFM2.5; Nemotron 3.5; Muse Glimmer; Muse Spark;
  Laguna 2.1; Ling-3.1; K2-Horizon; Hy4; MiniMax-M3; Nanbeige 4.2.
  Out-of-scope branches (tool calls, reasoning content, auto-injected
  system blocks) are documented, not modeled.
- **`model-protocol-lab debug`**: an interactive, offline protocol-debugger REPL —
  render with per-segment char/token maps, stepped boundary detection,
  fixture auditing; weights never loaded.
- **`model-protocol-lab diff-backends`**: differential testing of emulated backend stop
  semantics (HF StopStringCriteria, vLLM detokenizer strip/keep,
  EOS-appended, SGLang) over one recorded token sequence, with a
  classified divergence taxonomy.
- **`model-protocol-lab diff-reports`**: deterministic comparison of two experiment
  result directories for the iterate-and-compare operator loop, with a
  config-drift warning section.
- **`doctor --portability-matrix`**: static template↔tokenizer
  portability check over 118 mutate candidates, fail-closed
  (`missing_or_unsafe_asset` on unreadable assets).
- **Stop-semantics fidelity**: line-faithful ports of upstream
  `transformers.StopStringCriteria` and vLLM's `check_stop_strings`,
  verified against the real implementations over thousands of
  adversarial differential trials.

See `CHANGELOG.md` (`[0.1.0]`) for the full per-cycle record,
including every real bug found and fixed during the audit cycles.

## Post-0.1.0 snapshot additions (2026-10-16 → 2026-10-18, on `main`)

Development continued after the 0.1.0 snapshot; these additions are on
`main` and covered by `CHANGELOG.md`'s `[Unreleased]` entries:

- **Deterministic inline-SVG token-budget frontier chart** in `report.html`,
  with a **decision ribbon** coloring every budget span by post-hoc outcome
  (cycles 121–122).
- **Selection-margin trajectory chart** under the frontier chart, showing the
  per-budget leader and lead margin with a safe-budget-floor dashed line
  (cycle 124).
- **Safe budget floor readout**: the first budget from which the native
  decision never flips again, as a first-class HTML callout and
  `diff-reports` comparison (cycle 126).
- **Per-flip-span attribution**: the exact (candidate, fixture) pass-status
  changes driving each decision flip, recorded in the payload and compared
  by `diff-reports` (cycles 127–128).
- **Selection-rule consistency audit**: proved the frontier's post-hoc
  `decision_at` is exactly the selection rule the run would have produced
  (cycle 129).
- **`model-protocol-lab verify`**: N-version independent results-integrity
  replay — re-derives decision, winner, frontier, and held-out floor from a
  stored `results.json` through independent code paths (cycle 131).
- **Held-out verification-budget floor**: the first budget from which the
  held-out verdict holds unchanged, as an HTML callout and `diff-reports`
  comparison (cycle 130).
- **Holdout contamination detector**: measures search↔held-out fixture
  similarity (deterministic `difflib` + trigram Jaccard tiers) and warns in
  `inspect`/`fixtures build`/`doctor --dataset`/`run` (cycle 132).
- **Security red-team hardening** of the HTML report renderer against
  hostile payloads (strict type guards, fail-closed restore, symlinked
  report-dir rejection) (cycle 132).
- **Prompt-boundary canonicality audit** (`doctor --boundary-audit`):
  flags prompt templates whose tokenization is fragile at the
  prompt/response seam, as an HTML section and `diff-reports` comparison
  (cycle 134).
- **Chat-marker isolation conformance**: verifies that chat markers
  (e.g. `<|im_start|>`) survive tokenization as single tokens rather
  than splitting into pieces (cycle 135).
- **Template conformance firewall**: static AST scan of the checkpoint's
  Jinja `chat_template` denying RCE primitives (dunder access,
  `|attr` sandbox escape, dangerous globals) before any render; the
  render itself stays inside `ImmutableSandboxedEnvironment`
  (cycle 136).
- **Selection robustness under stop semantics**: pure post-hoc
  re-application of the selection rule under each emulated stop semantic
  (native, HF, vLLM, EOS-appended) — the serving-stack migration signal,
  as an HTML section and `diff-reports` comparison (cycle 137).
- **Cross-checkpoint transfer experiment** (`run --transfer-model`):
  re-evaluates every search candidate against up to 4 local checkpoints
  with per-checkpoint selection verdicts, as an HTML transfer-matrix
  section (cycle 138).
- **Transfer comparison in `diff-reports`** (`diff_transfer`): verdict
  drift, per-checkpoint and per-cell drift, checkpoints added/removed
  (cycle 140).
- **Hostile-input crash hardening**: three `TypeError` crashes on
  non-string dict keys fixed fail-closed (cycle 141).
- **Contamination comparison in `diff-reports`**
  (`diff_contamination`): worst-tier drift, pairs added/removed, tier
  escalations — the fixture-hygiene regression signal (cycle 143).
- **Fixture-perturbation robustness experiment** (`run --perturb N`):
  re-runs the search evaluation on N deterministically perturbed fixture
  variants (typo, case flip, whitespace, Unicode variant, punctuation
  strip) and re-applies the selection rule per variant — answering
  whether the winner survives phrasing perturbations. New HTML section
  and `diff-reports` comparison (`diff_perturbation`); completes the
  winner-reliability trilogy with cycles 137 and 138 (cycle 144).

## Breaking changes

**None.** `0.1.0a0` was a source snapshot that never executed, so there
is no prior released behavior to break. This is the first runnable
release. The CLI surface (9 subcommands), the fixture schema, and the
report formats have been stable across the development cycles; the exit
code contract (0/1/2/130) is pinned by tests.

## Install

Requires Python 3.11+.

From source (verified path):

```sh
python -m venv .venv
# Activate with your OS's usual command, e.g. source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m model_protocol_lab catalog
```

To run a model you also need the optional HF adapter and a trusted,
already-downloaded checkpoint (the adapter never downloads models):

```sh
python -m pip install -e ".[hf]"
python -m model_protocol_lab doctor --model path/to/checkpoint --dataset examples/fixtures.json
```

A `model-protocol-lab` console script is installed as an equivalent
entry point. PyPI publication has not been done yet; installing from
source as above is the supported path. Upgrade from a source checkout:
`git pull` then re-run `python -m pip install -e ".[dev]"`.

## Verified for this release (2026-10-16)

The checks below are a historical snapshot, verified on 2026-10-16
against the 0.1.0 tree. Numbers on current `main` have moved (4699 tests
on 2026-10-18, both runners agree) — the "Post-0.1.0 snapshot additions"
section above lists what changed since.

- `ruff check` clean; `ruff format --check` clean (135 files);
  `mypy model_protocol_lab` clean (15 files);
  `bandit -r model_protocol_lab` — no issues identified.
- Suite: 6063 tests collected; `coverage report --fail-under=95` passes
  (98% total line coverage).
- `python -m build` wheel verified: 21 files, no `__pycache__` /
  `.pytest_cache` / tests shipped; installs in a network-isolated fresh
  venv; all 9 subcommands' `--help` exit 0 there.
- Zero real TODOs in shipped code (the only TODO-shaped strings are
  quotes of upstream SGLang's `# TODO(lmzheng)` in docstrings/research
  notes).

## Honest limits — what is NOT established

- **Real-model runs beyond the synthetic checkpoint**: the HF adapter is
  integration-tested against a locally generated tiny GPT-2 checkpoint
  only; no real-model experiment runs are recorded.
- **Browser rendering checks**: report HTML was not visually verified in
  a browser.
- **Beyond CI's matrix**: no PyPy/Windows/macOS runs, no real
  SGLang/vLLM/TGI servers — their stop semantics are verified against
  source code and local library implementations only.
- The test suite is hermetic (no network).
