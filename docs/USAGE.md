# Usage

## 1. Inspect the fixture schema

Each dataset has exactly two arrays: `search` and `held_out`. Every fixture has `id`, `group`, `messages` and `expected`. Messages have exactly `role` and `content`; roles alternate user/assistant, beginning and ending with user. System messages, tools and assistant prefills are not supported.

IDs are globally unique. Groups must not overlap across partitions. Identical message sequences cannot appear in both partitions. These checks do not detect paraphrases or other semantic leakage. Deliberately separate task families and freeze the held-out set before search. ID-uniqueness, group-overlap and identical-message comparisons use NFC-normalized text, so NFD/NFC variants of the same visual string cannot evade them; the stored fixture data itself is left byte-exact.

Limits include a 2 MB UTF-8 file, 100 fixtures per partition, 21 messages per fixture and 16,384 characters per text field. The dataset path must be a regular file — character/block devices, named FIFOs, sockets and directories are refused (a FIFO with no writer would block forever). Inspect the loader for the authoritative contract.

Empty `expected` answers are allowed but pass vacuously: any output that emits only the boundary token satisfies them. `inspect` and `run` warn on stderr naming such fixtures so the measurement gap is visible.

```sh
python -m model_protocol_lab inspect examples/fixtures.json
python -m model_protocol_lab catalog
```

## 2. Pre-flight check with `doctor`

Before any model run, validate the checkpoint directory and fixture file without
loading weights:

```sh
python -m model_protocol_lab doctor --model path/to/checkpoint --dataset YOUR_FIXTURES.json
```

`doctor` checks the asset manifest (required files, no symlinks, size caps),
the model config (supported GPT-2/Llama causal-LM settings only), the
tokenizer document structure, and the fixture schema. With `--boundary-ids` it
additionally verifies each ID against the tokenizer vocabulary (loading the
tokenizer only, never the weights). Exit 0 means the run inputs look sane;
exit 1 lists the failing checks. It reports the tokenizer's declared
`added_tokens` with their IDs to help choose `--boundary-ids`.

## 3. Prepare a local model

Use trusted local GPT-2 or Llama causal-LM assets supported by the adapter. Required files include `config.json`, a self-contained `tokenizer.json`, and either `model.safetensors` or `model.safetensors.index.json` with its flat-named shards. Optional `generation_config.json` is fingerprinted but does not control generation.

The adapter rejects remote-code mappings, quantization configurations, adapter-only directories, encoder-decoder configurations, cross-attention configurations, conflicting weight layouts, asset symlinks and incomplete weight loading. No automatic downloads, pickle weights, custom model code, model-specific tokenizer Python code, quantized loading or LoRA merging are provided.

The tokenizer is loaded directly from tokenizer.json. Encoding disables automatic special-token insertion. Verify its behavior against the checkpoint's intended tokenizer before trusting results. Literal special-token markers may not behave as expected for every checkpoint.

The device is `cpu` or `cuda` (CUDA device 0). Weights and computation are requested in float32, with eager attention and no cache. Context limits are checked against model configuration; practical RAM/VRAM limits remain your responsibility. Stock GPT-2 checkpoints declare `n_positions=1024`, below the CLI default `--max-context 2048`: pass `--max-context 1024` (or less) for them, or backend initialization fails fast with the distinct code `local_context_limit_exceeds_declared_window` instead of the generic init failure.

## 4. Establish the response boundary

`--boundary-ids` is a comma-separated token-ID sequence expected to end an assistant answer. It is not automatically inferred and is not assumed to equal the checkpoint's EOS token. Determine it from the actual tokenizer and intended protocol. Do not copy token IDs from another model.

The boundary match is an exact token-ID subsequence match by design. It can miss
when the model emits the boundary *text* with a different tokenization than the
configured IDs (one string can be many token sequences, e.g. `["\n", "\n"]` vs
`["\n\n"]`; see HF transformers' `StopStringCriteria`). Each completed
observation therefore also records `boundary_text_match`: whether the decoded
output ends with the decoded boundary text. A `boundary_found=false` with
`boundary_text_match=true` flags that false-negative mode for investigation; it
never changes the pass criterion. Prefer boundary sequences whose tokenization
is stable in the checkpoint's tokenizer (verify with `doctor --boundary-ids`).

The same explicit sequence is evaluated for every candidate in a run. The current CLI therefore cannot fairly compare candidates requiring different boundary sequences; use separate clearly labeled runs until per-candidate boundary support exists.

## 5. Run a bounded experiment

Replace every uppercase placeholder below. This is an invocation template, not a validated command for a specific checkpoint.

```sh
python -m model_protocol_lab run --model LOCAL_MODEL_DIRECTORY --dataset YOUR_FIXTURES.json --family chatml --mutate --boundary-ids TOKEN_ID --max-new-tokens 64 --max-context 2048 --device cpu --output NEW_REPORT_DIRECTORY
```

The local adapter emits the full requested number of tokens unless cancelled or failed. It does not stop automatically on EOS or the supplied boundary. The evaluator finds the first boundary sequence, decodes preceding tokens, and checks the exact expected answer. This separates boundary observation from backend stop behavior.

With `--mutate`, the bounded candidate set adds BOS and assistant-prefix variants. The prefix mutations explore the same space as assistant-prefill controls in chat APIs (HF's `add_generation_prompt` / `continue_final_message`, vLLM's guided prefill): `prefix-newline` and `prefix-trim` vary how the assistant turn is opened, `toggle-bos` and `no-bos` vary BOS placement. Search does not modify weights. A unique candidate passing all search cases is frozen before held-out evaluation. Ties cause abstention; the held-out set is not used to break them.

Ctrl+C requests cooperative cancellation after model loading. Interrupted evaluation can retain a partial report. Model loading and individual backend operations are not guaranteed to be promptly interruptible.

## 6. Read the outputs

- `results.json`: structured decision, summaries, candidate formats and fingerprint. Each observation carries `boundary_text_match` (see §4).
- `report.html`: offline static comparison table with color and text labels.

Open the HTML file in a browser. It loads no scripts, fonts or other remote assets. Raw evidence is never embedded in this HTML view. Add `--include-evidence` only when you deliberately want raw observations in JSON and understand that token IDs can reveal private prompts.

Output requires a new directory and will not overwrite an existing one. Inspect a partial directory after any write failure. Do not assume that a file's existence proves a complete run.

Exit status: 0 for compatibility on the supplied fixture set; 2 for completed negative or ambiguous decisions; 130 for cancellation; 1 for recorded evaluator failure or caught initialization, loading or filesystem errors. Argument-parser errors also return 2. Inspect the decision field rather than interpreting exit 2 alone.

## 7. Validate the source

From the repository root, in a disposable environment:

```sh
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -t .
ruff check model_protocol_lab tests examples
ruff format --check model_protocol_lab tests examples
mypy model_protocol_lab
```

The core tests use fake backends and do not require model weights. They do not establish Hugging Face adapter compatibility, CPU/GPU parity or real-model effectiveness. Follow the staged plan for those checks and preserve exact dependency versions, model hashes and logs.
