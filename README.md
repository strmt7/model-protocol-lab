# Model Protocol Lab

[![CI](https://github.com/strmt7/model-protocol-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/strmt7/model-protocol-lab/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-informational)](CITATION.cff)

**Is a local language model failing because of its weights, or because its conversations are formatted incorrectly?**

Model Protocol Lab answers that question with measurements, not guesses. It renders chat-format hypotheses — a 29-family catalog (ChatML; Llama 3 and 4; Gemma 2, 3 and 4; Qwen3 and 3.8; DeepSeek V3/R1 and V4.1; Mistral v0.3 and Large; Phi-4; GLM-4, 4.5 and 5.3; Kimi K2 and K3; Kolibri-1; LFM2.5; Nemotron 3.5; Muse Glimmer; Muse Spark; Laguna 2.1; Ling-3.1; K2-Horizon; Hy4; MiniMax-M3; Nanbeige 4.2), plus bounded BOS, prefix and thinking-mode mutations — as exact token sequences, sends them to a local checkpoint, and compares task answers and response boundaries token-for-token. A strict search/held-out fixture discipline selects a single winning format or abstains; nothing is ranked, nothing is downloaded, and every result ships as a byte-deterministic local report.

What sets it apart: it measures checkpoints on token sequences their own chat templates would never produce; it detects response boundaries as token-ID subsequences rather than string stop-sequences; it generates exactly N tokens with no implicit EOS stop; and it abstains on ties instead of crowning a leaderboard winner. The differences are verifiable design decisions, documented with sources in [How this differs](#how-this-differs-from-nearby-tools) — not marketing claims.

## Status

**Production-hardened and tested.** The package installs cleanly, the full test suite passes (5473 tests, 95% line coverage), `ruff` and `mypy` are clean, and the optional Hugging Face adapter is integration-tested against a locally generated tiny GPT-2 checkpoint. See [VALIDATION.md](VALIDATION.md) for the exact commands, counts, dates and environment — and for what is still *not* established (real-model runs beyond the synthetic checkpoint, browser rendering checks). No benchmark-superiority or unique-invention claim is made.

## Quick start

Python 3.11 or newer is required.

```sh
python -m venv .venv
# Activate the environment with your operating system's usual command, e.g.:
#   source .venv/bin/activate        # POSIX
#   .venv\Scripts\activate           # Windows
python -m pip install -e ".[dev]"
python -m model_protocol_lab catalog
python -m model_protocol_lab inspect examples/fixtures.json
python -m unittest discover -s tests -t .
```

(A `model-protocol-lab` console script is installed as an equivalent entry point.)

To run a model you also need the optional dependencies and a trusted,
already-downloaded checkpoint directory (the adapter never downloads models):

```sh
python -m pip install -e ".[hf]"
python -m model_protocol_lab doctor --model path/to/checkpoint --dataset examples/fixtures.json
```

`doctor` validates the checkpoint assets and fixture file before any run and
never loads model weights. It even suggests the exact `run` command to use
next, with limits fitted to the checkpoint's declared context window. Read
[docs/USAGE.md](docs/USAGE.md) before running a model — boundary token IDs must
be established from the **actual checkpoint tokenizer**.

No checkpoint handy? Generate the synthetic sample report (a scripted backend
stands in for a model; clearly labeled, deterministic):

```sh
python examples/make_sample_report.py
# then open examples/sample-report/report.html in a browser
```

Prefer authoring fixtures in a spreadsheet? Write one row per fixture as CSV
and compile it to the strict schema with the search/held-out discipline
enforced at build time:

```sh
python -m model_protocol_lab fixtures build examples/fixtures.csv --output /tmp/mpl-fixtures.json
# (a fresh path: fixtures build never overwrites an existing file)
```

## What the report looks like

`examples/sample-report/report.html` is a committed, byte-deterministic example
produced from `examples/fixtures.json`: a decision banner with a failure-mode
diagnosis, run configuration, a color-coded candidate heatmap with pass-rate
bars, a per-fixture outcome grid, decisive fixtures, token-budget robustness
curves, and the held-out check. It is one self-contained file — a single
tiny inline table-sorting script (hash-allowlisted in its
Content-Security-Policy), no fonts or remote assets — and it never embeds
raw prompts or generations.

## Capabilities

**Exact-token measurement.**
- Strict, bounded JSON fixture loading with search/held-out separation checks (IDs, groups and message inputs cannot leak across the split; NFC-normalized).
- Text-only serialization hypotheses across a 29-family catalog — ChatML; Llama 3/4; Gemma 2/3/4; Qwen3/3.8; DeepSeek V3/R1/V4.1; Mistral v0.3/Large; Phi-4; GLM-4/4.5/5.3; Kimi K2/K3; Kolibri-1; LFM2.5; Nemotron 3.5; Muse Glimmer; Muse Spark; Laguna 2.1; Ling-3.1; K2-Horizon; Hy4; MiniMax-M3; Nanbeige 4.2 — plus bounded BOS, prefix and thinking-mode mutations (`catalog` lists every family with its framing fields; primary sources are cited in the catalog code comments).
- Raw token evaluation (`add_special_tokens=False`): explicit token-ID boundary detection with a text-level mismatch diagnostic, cooperative cancellation, and sanitized failure records.
- Per-candidate response boundaries (`run --boundary-map`): each format hypothesis is measured with its own end-of-turn marker.

**Disciplined selection.**
- A conservative rule: exactly one candidate must pass every search fixture; only that candidate reaches held-out evaluation. Ties and failures abstain — there is no leaderboard.
- Failure-mode taxonomy with per-candidate counts, cited in the decision banner (`no-boundary-in-N`, `boundary-text-mismatch`, `eos-without-boundary`, `held-out-divergence`, `backend-error`).

**Second-opinion diagnostics.**
- Template-conformance diffs: each hand-rendered hypothesis compared token-for-token against the checkpoint's own chat-template rendering.
- Stop-behavior isolation matrix: the winner re-measured under emulated HF stop-string, vLLM detokenizer, and EOS-appended semantics, so backend stop behavior never silently decides the outcome.
- Token-budget robustness frontier: the search decision recomputed at every smaller token budget, post-hoc and exact.
- Cross-tokenizer boundary portability check (`doctor --reference-model`).
- Prompt-boundary sensitivity sweep (`run --sensitivity-sweep`): measures whether truncating the prompt mid-token flips the selection.

**Operator experience.**
- `doctor` pre-flight validation of checkpoint assets and fixtures without loading weights, with actionable next-command suggestions.
- `fixtures build`: CSV fixture authoring that makes search/held-out leakage structurally unbuildable.
- `debug`: an interactive, offline protocol-debugger REPL over one dataset and one hypothesis — render with a per-segment char-span (and token-ID) map, step response-boundary detection one token at a time, and audit fixtures for special-token literals; `--model` attaches a checkpoint tokenizer only, weights never loaded. Try `python -m model_protocol_lab debug --dataset examples/fixtures.json`, then `.render`.
- `diff-backends`: differential testing of emulated backend stop semantics (HF StopStringCriteria, vLLM detokenizer strip/keep, EOS-appended) on one recorded token sequence — prints a JSON verdict (`agree`/`diverge`) with the per-semantic cells and any divergence findings, so a surprising run verdict can be checked against backend stop behavior. Try `python -m model_protocol_lab diff-backends --model path/to/checkpoint --prompt-text "the answer is" --output-text "42 STOP" --stops "STOP"`.
- Local JSON output and a static, color-coded HTML comparison report with no external assets.
- A committed synthetic sample report that regenerates byte-identically.
- An optional local Hugging Face adapter restricted to supported GPT-2 and Llama causal-language-model configurations with standalone safetensors weights.

The format catalog and model-loader support are different things. A Gemma-format hypothesis does **not** mean this adapter loads Gemma model architectures. The report is a static heatmap, not a live training or model-internals dashboard.

## How this differs from nearby tools

This is not a benchmark suite and not an inference server. The distinctions
below are verifiable design differences, grounded in the referenced sources
(checked 2026-10-13; research log in `docs/RESEARCH_LOG.md`), not superiority claims.

**Exact token sequences in, exact token sequences out.** General harnesses take
text strings and apply the checkpoint's own chat template: lm-eval renders via
`tokenizer.apply_chat_template`
([huggingface.py](https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/models/huggingface.py));
vLLM and llama.cpp render the model's Jinja template. This lab renders format
hypotheses by hand and encodes with `add_special_tokens=False` — the workflow
the HF docs bless for avoiding duplicated special tokens
([chat_templating](https://huggingface.co/docs/transformers/main/en/chat_templating)) —
so a checkpoint is measured on token sequences its own template would never
produce. The question "weights or formatting?" cannot be answered by a tool that
always applies the model's own formatting.

**Token-ID boundaries instead of string stop-sequences.** lm-eval, vLLM,
llama.cpp and HF `generate()` all detect stops at the string level, each
shipping mitigations for string↔tokenization mismatches (e.g. lm-eval's
`MultiTokenEOSCriteria` "tokenizers suck" lookback,
[vLLM's stop-buffer holdback](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/engine/detokenizer.py),
HF's
[`StopStringCriteria` overhang matching](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/generation/stopping_criteria.py)).
This lab matches an explicit token-ID subsequence and records a text-level
cross-check (`boundary_text_match`) as a diagnostic, so a tokenization mismatch
shows up as evidence instead of silently changing the result.

**No implicit EOS stop.** lm-eval appends EOS to its stop sequences (when the
tokenizer declares one); vLLM and llama.cpp offer opt-in `ignore_eos`. This lab
generates exactly N tokens and observes where the boundary fell, separating
backend stop behavior from boundary measurement.

**Abstention instead of ranking.** Exactly one candidate passing every search
fixture is selected and frozen before held-out evaluation; ties and failures
abstain and the held-out set is never used to break them. No counterpart to
this selection rule was found in lm-eval's evaluator; there is no leaderboard
here.

**What this lab does not do:** benchmark general model quality, recover a
model's training template, serve inference, or train anything. The bundled
fixtures demonstrate the schema; they are not a scientifically sufficient
benchmark.

## How to interpret a result

`compatible_on_fixture_set` means the sole search winner also passed the supplied held-out cases under this exact run configuration. It does not recover a model's historical training template or establish general compatibility. `ambiguous` means several candidates passed; `no_candidate` means none passed. A partial, failed or cancelled evaluation is not a valid aggregate comparison.

The current adapter uses greedy full-prefix recomputation, float32 weights, no KV cache and no implicit EOS stopping. The seed is recorded and validated but does not alter greedy decoding. It is a correctness-oriented starting point, not an efficient inference engine. Token counts in reports are retained-input/output counts, not equal-compute accounting.

## Repository layout

```text
model_protocol_lab/       Renderer, evaluator, catalog, loader, experiment, reports, CLI
tests/                    Core, workflow, adapter-integration, e2e and sample-report tests
examples/                 Fixture files (JSON + CSV) and a byte-deterministic sample report
docs/                     Usage, FAQ, research log, and security guidance
.github/workflows/       CI: lint, typecheck, tests on 3.11-3.13, packaging
DEVELOPMENT_PLAN.md       Architecture, milestones, test matrices and release gates
VALIDATION.md             Verification evidence and limitations
CHANGELOG.md              What changed and when
```

## Privacy

Reports stay on the local filesystem. JSON excludes raw observations by default; `--include-evidence` explicitly includes potentially sensitive prompts as token IDs, generated text and traces. Summary metadata, candidate names, format markers and fingerprints remain in ordinary JSON. Inspect all reports before sharing. No report files, model weights, credentials or personal configuration belong in source control.

See [docs/SECURITY.md](docs/SECURITY.md) and [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under MIT — see [LICENSE](LICENSE). If you use this software, please cite it via [CITATION.cff](CITATION.cff).
