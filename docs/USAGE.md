# Usage

## 1. Inspect the fixture schema

Each dataset has exactly two arrays: `search` and `held_out`. Every fixture has `id`, `group`, `messages` and `expected`. Messages have exactly `role` and `content`; roles alternate user/assistant, beginning and ending with user. System messages, tools and assistant prefills are not supported.

IDs are globally unique. Groups must not overlap across partitions. Identical message sequences cannot appear in both partitions. These checks do not detect paraphrases or other semantic leakage. Deliberately separate task families and freeze the held-out set before search.

Limits include a 2 MB UTF-8 file, 100 fixtures per partition, 21 messages per fixture and 16,384 characters per text field. Inspect the loader for the authoritative contract.

```sh
python -m model_protocol_lab inspect examples/fixtures.json
python -m model_protocol_lab catalog
```

## 2. Prepare a local model

Use trusted local GPT-2 or Llama causal-LM assets supported by the adapter. Required files include `config.json`, a self-contained `tokenizer.json`, and either `model.safetensors` or `model.safetensors.index.json` with its flat-named shards. Optional `generation_config.json` is fingerprinted but does not control generation.

The adapter rejects remote-code mappings, quantization configurations, adapter-only directories, encoder-decoder configurations, cross-attention configurations, conflicting weight layouts, asset symlinks and incomplete weight loading. No automatic downloads, pickle weights, custom model code, model-specific tokenizer Python code, quantized loading or LoRA merging are provided.

The tokenizer is loaded directly from tokenizer.json. Encoding disables automatic special-token insertion. Verify its behavior against the checkpoint's intended tokenizer before trusting results. Literal special-token markers may not behave as expected for every checkpoint.

The device is `cpu` or `cuda` (CUDA device 0). Weights and computation are requested in float32, with eager attention and no cache. Context limits are checked against model configuration; practical RAM/VRAM limits remain your responsibility.

## 3. Establish the response boundary

`--boundary-ids` is a comma-separated token-ID sequence expected to end an assistant answer. It is not automatically inferred and is not assumed to equal the checkpoint's EOS token. Determine it from the actual tokenizer and intended protocol. Do not copy token IDs from another model.

The same explicit sequence is evaluated for every candidate in a run. The current CLI therefore cannot fairly compare candidates requiring different boundary sequences; use separate clearly labeled runs until per-candidate boundary support exists.

## 4. Run a bounded experiment

Replace every uppercase placeholder below. This is an invocation template, not a validated command for a specific checkpoint.

```sh
python -m model_protocol_lab run --model LOCAL_MODEL_DIRECTORY --dataset YOUR_FIXTURES.json --family chatml --mutate --boundary-ids TOKEN_ID --max-new-tokens 64 --max-context 2048 --device cpu --output NEW_REPORT_DIRECTORY
```

The local adapter emits the full requested number of tokens unless cancelled or failed. It does not stop automatically on EOS or the supplied boundary. The evaluator finds the first boundary sequence, decodes preceding tokens, and checks the exact expected answer. This separates boundary observation from backend stop behavior.

With `--mutate`, the bounded candidate set adds BOS and assistant-prefix variants. Search does not modify weights. A unique candidate passing all search cases is frozen before held-out evaluation. Ties cause abstention; the held-out set is not used to break them.

Ctrl+C requests cooperative cancellation after model loading. Interrupted evaluation can retain a partial report. Model loading and individual backend operations are not guaranteed to be promptly interruptible.

## 5. Read the outputs

- `results.json`: structured decision, summaries, candidate formats and fingerprint.
- `report.html`: offline static comparison table with color and text labels.

Open the HTML file in a browser. It loads no scripts, fonts or other remote assets. Raw evidence is never embedded in this HTML view. Add `--include-evidence` only when you deliberately want raw observations in JSON and understand that token IDs can reveal private prompts.

Output requires a new directory and will not overwrite an existing one. Inspect a partial directory after any write failure. Do not assume that a file's existence proves a complete run.

Exit status: 0 for compatibility on the supplied fixture set; 2 for completed negative or ambiguous decisions; 130 for cancellation; 1 for recorded evaluator failure or caught initialization, loading or filesystem errors. Argument-parser errors also return 2. Inspect the decision field rather than interpreting exit 2 alone.

## 6. Validate the source

From the repository root, in a disposable environment:

```sh
python -m unittest discover -s tests -t . -v
```

The core tests use fake backends and do not require model weights. They do not establish Hugging Face adapter compatibility, CPU/GPU parity or real-model effectiveness. Follow the staged plan for those checks and preserve exact dependency versions, model hashes and logs.
