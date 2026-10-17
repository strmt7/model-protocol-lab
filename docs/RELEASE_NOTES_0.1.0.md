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
- **`mpl debug`**: an interactive, offline protocol-debugger REPL —
  render with per-segment char/token maps, stepped boundary detection,
  fixture auditing; weights never loaded.
- **`mpl diff-backends`**: differential testing of emulated backend stop
  semantics (HF StopStringCriteria, vLLM detokenizer strip/keep,
  EOS-appended, SGLang) over one recorded token sequence, with a
  classified divergence taxonomy.
- **`mpl diff-reports`**: deterministic comparison of two experiment
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

## Breaking changes

**None.** `0.1.0a0` was a source snapshot that never executed, so there
is no prior released behavior to break. This is the first runnable
release. The CLI surface (8 subcommands), the fixture schema, and the
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

- `ruff check` clean; `ruff format --check` clean (164 files);
  `mypy model_protocol_lab` clean (16 files);
  `bandit -r model_protocol_lab` — no issues identified.
- Suite: 4391 tests collected; `coverage report --fail-under=95` passes
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
