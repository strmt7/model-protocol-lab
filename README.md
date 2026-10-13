# Model Protocol Lab

[![CI](https://github.com/strmt7/model-protocol-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/strmt7/model-protocol-lab/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-informational)](CITATION.cff)

**Is a local language model failing because of its weights, or because its conversations are formatted incorrectly?**

Model Protocol Lab is a bounded experiment lab: render several conversation formats as exact token sequences, send them to a local checkpoint, and compare task answers and response boundaries. A separate held-out fixture set checks the single candidate selected during search.

## Status

**Hardened and tested; not a validated release.** The package installs cleanly,
the full test suite passes (213 tests, 97% line coverage), `ruff` and `mypy`
are clean, and the optional Hugging Face adapter is integration-tested against a
locally generated tiny GPT-2 checkpoint. See [VALIDATION.md](VALIDATION.md) for
the exact commands, counts, dates and environment — and for what is still *not*
established (real-model runs beyond the synthetic checkpoint, browser rendering
checks, mutation testing). No benchmark-superiority or unique-invention claim is
made.

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
```

To run a model you also need the optional dependencies and a trusted,
already-downloaded checkpoint directory (the adapter never downloads models):

```sh
python -m pip install -e ".[hf]"
python -m model_protocol_lab doctor --model path/to/checkpoint --dataset examples/fixtures.json
```

`doctor` validates the checkpoint assets and fixture file before any run and
never loads model weights. Read [docs/USAGE.md](docs/USAGE.md) before running a
model — boundary token IDs must be established from the **actual checkpoint
tokenizer**.

No checkpoint handy? Generate the synthetic sample report (a scripted backend
stands in for a model; clearly labeled, deterministic):

```sh
python examples/make_sample_report.py
# then open examples/sample-report/report.html in a browser
```

## What the report looks like

`examples/sample-report/report.html` is a committed, byte-deterministic example
produced by the command above from `examples/fixtures.json`: a decision banner,
run configuration, a color-coded candidate heatmap with pass-rate bars, and the
held-out check. It is one self-contained file — no scripts, fonts or remote
assets — and it never embeds raw prompts or generations.

## How this differs from nearby tools

This is not a benchmark suite and not an inference server. The distinctions
below are verifiable design differences, grounded in the referenced sources
(checked 2026-10-12), not superiority claims.

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

## Included source

- Strict, bounded JSON fixture loading with search/held-out separation checks.
- Text-only ChatML, Llama 3 and Gemma serialization hypotheses, plus bounded BOS and prefix mutations.
- Raw token evaluation, explicit boundary detection with a text-level mismatch diagnostic, cancellation and sanitized failure records.
- A conservative selection rule: exactly one candidate must pass every search fixture; only that candidate reaches held-out evaluation. Ties and failures abstain.
- A `doctor` pre-flight command that validates checkpoint assets and fixtures without loading weights.
- An optional local Hugging Face adapter restricted to supported GPT-2 and Llama causal-language-model configurations with standalone safetensors weights.
- Local JSON output and a static, color-coded HTML comparison report with no external assets.
- A committed synthetic sample report that regenerates byte-identically.

The format catalog and model-loader support are different things. A Gemma-format hypothesis does **not** mean this adapter loads Gemma model architectures. The report is a static heatmap, not a live training or model-internals dashboard.

## How to interpret a result

`compatible_on_fixture_set` means the sole search winner also passed the supplied held-out cases under this exact run configuration. It does not recover a model's historical training template or establish general compatibility. `ambiguous` means several candidates passed; `no_candidate` means none passed. A partial, failed or cancelled evaluation is not a valid aggregate comparison.

The current adapter uses greedy full-prefix recomputation, float32 weights, no KV cache and no implicit EOS stopping. The seed is recorded and validated but does not alter greedy decoding. It is a correctness-oriented starting point, not an efficient inference engine. Token counts in reports are retained-input/output counts, not equal-compute accounting.

## Repository layout

```text
model_protocol_lab/       Renderer, evaluator, catalog, loader, experiment, reports, CLI
tests/                    Core, workflow, adapter-integration, e2e and sample-report tests
examples/                 Small synthetic fixture file and a byte-deterministic sample report
docs/                     Usage and security guidance
.github/workflows/       CI: lint, typecheck, tests on 3.11-3.13, packaging
DEVELOPMENT_PLAN.md       Architecture, milestones, test matrices and release gates
VALIDATION.md             Verification evidence and limitations
CHANGELOG.md              What changed and when
```

## Privacy

Reports stay on the local filesystem. JSON excludes raw observations by default; `--include-evidence` explicitly includes potentially sensitive prompts as token IDs, generated text and traces. Summary metadata, candidate names, format markers and fingerprints remain in ordinary JSON. Inspect all reports before sharing. No report files, model weights, credentials or personal configuration belong in source control.

See [docs/SECURITY.md](docs/SECURITY.md) and [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under MIT — see [LICENSE](LICENSE). If you use this software, please cite it via [CITATION.cff](CITATION.cff).
