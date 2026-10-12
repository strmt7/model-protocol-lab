# Model Protocol Lab

**Is a local language model failing because of its weights, or because its conversations are formatted incorrectly?**

Model Protocol Lab is an early source implementation of a bounded experiment: render several conversation formats, send their exact token sequences to a local checkpoint, and compare task answers and response boundaries. A separate held-out fixture set checks the single candidate selected during search.

**Status: unexecuted alpha source.** The source has received static review; the application, dependency installation and tests have not been run. See [VALIDATION.md](VALIDATION.md) for the evidence boundary and [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for the staged engineering plan. No production-readiness, benchmark superiority or unique-invention claim is made.

## Included source

- Strict, bounded JSON fixture loading with search/held-out separation checks.
- Text-only ChatML, Llama 3 and Gemma serialization hypotheses, plus bounded BOS and prefix mutations.
- Raw token evaluation, explicit boundary detection, cancellation and sanitized failure records.
- A conservative selection rule: exactly one candidate must pass every search fixture; only that candidate reaches held-out evaluation. Ties and failures abstain.
- An optional local Hugging Face adapter restricted to supported GPT-2 and Llama causal-language-model configurations with standalone safetensors weights.
- Local JSON output and a static, color-coded HTML comparison report with no external assets.
- Unit-test source and a detailed development roadmap.

The format catalog and model-loader support are different things. A Gemma-format hypothesis does **not** mean this adapter loads Gemma model architectures. The report is a static heatmap, not a live training or model-internals dashboard.

## Quick start

Python 3.11 or newer is required. These are intended usage commands, not a record of successful execution.

```sh
python -m venv .venv
# Activate the environment using your operating system's usual command.
python -m pip install -e .
python -m model_protocol_lab catalog
python -m model_protocol_lab inspect examples/fixtures.json
python -m unittest discover -s tests -t . -v
```

Model execution additionally requires the optional dependencies:

```sh
python -m pip install -e ".[hf]"
```

Dependency ranges are provisional, not a tested compatibility matrix or lockfile. Use a dedicated environment. Supply an already-downloaded, trusted, standalone checkpoint directory; the adapter does not download models or execute remote model code.

Read [docs/USAGE.md](docs/USAGE.md) before running a model. Boundary token IDs must be established from the **actual checkpoint tokenizer**. The bundled fixtures demonstrate the schema; they are not a scientifically sufficient benchmark.

## How to interpret a result

`compatible_on_fixture_set` means the sole search winner also passed the supplied held-out cases under this exact run configuration. It does not recover a model's historical training template or establish general compatibility. `ambiguous` means several candidates passed; `no_candidate` means none passed. A partial, failed or cancelled evaluation is not a valid aggregate comparison.

The current adapter uses greedy full-prefix recomputation, float32 weights, no KV cache and no implicit EOS stopping. The seed is recorded and validated but does not alter greedy decoding. It is a correctness-oriented starting point, not an efficient inference engine. Token counts in reports are retained-input/output counts, not equal-compute accounting.

## Repository layout

```text
model_protocol_lab/   Renderer, evaluator, catalog, loader, experiment, reports, CLI
tests/                Core and workflow test source
examples/             Small synthetic fixture file
docs/                 Usage and security guidance
DEVELOPMENT_PLAN.md   Architecture, milestones, test matrices and release gates
VALIDATION.md         Current verification status and limitations
UPLOAD_TO_GITHUB.md   Instructions for publishing the extracted source
```

## Privacy

Reports stay on the local filesystem. JSON excludes raw observations by default; `--include-evidence` explicitly includes potentially sensitive prompts as token IDs, generated text and traces. Summary metadata, candidate names, format markers and fingerprints remain in ordinary JSON. Inspect all reports before sharing. No report files, model weights, credentials or personal configuration belong in source control.

See [docs/SECURITY.md](docs/SECURITY.md). Licensed under MIT.
