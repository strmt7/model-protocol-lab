# Usage

For troubleshooting and interpretation questions, see [docs/FAQ.md](FAQ.md).

## 1. Inspect the fixture schema

Each dataset has exactly two arrays: `search` and `held_out`. Every fixture has `id`, `group`, `messages` and `expected`. Messages have exactly `role` and `content`; roles alternate user/assistant, beginning and ending with user. System messages, tools and assistant prefills are not supported.

IDs are globally unique. Groups must not overlap across partitions. Identical message sequences cannot appear in both partitions. These checks do not detect paraphrases or other semantic leakage. Deliberately separate task families and freeze the held-out set before search. ID-uniqueness, group-overlap and identical-message comparisons use NFC-normalized text, so NFD/NFC variants of the same visual string cannot evade them; the stored fixture data itself is left byte-exact.

Limits include a 2 MB UTF-8 file, 100 fixtures per partition, 21 messages per fixture and 16,384 characters per text field. The dataset path must be a regular file — character/block devices, named FIFOs, sockets and directories are refused (a FIFO with no writer would block forever). Inspect the loader for the authoritative contract. To author fixtures without hand-writing this JSON, see `fixtures build` in §9.

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

When every check passes and both `--dataset` and `--boundary-ids` were
supplied, the doctor report carries a `suggestion` with the exact `run`
invocation implied by the pre-flight result: `--max-context` is filled from
the checkpoint's declared window, `--max-new-tokens` is sized so the longest
fixture prompt (rendered with every catalog candidate and encoded with the
checkpoint tokenizer) still fits that window, paths are shell-quoted, and
the format family stays an explicit `FAMILY` placeholder for you to choose
(see `catalog`). If even the longest fixture prompt fills the window, the
suggestion keeps `--max-new-tokens 1` and warns you to shorten the fixtures
or use a larger-window checkpoint. The suggestion is omitted whenever it
could mislead (failed verdict, no dataset, no boundary IDs, or no declared
window).

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

### Per-candidate boundaries (`run --boundary-map`)

`--boundary-ids` is the default boundary for every candidate, and the
repeatable `--boundary-map NAME=ID,ID` flag overrides it for individual
candidates:

```sh
python -m model_protocol_lab run ... --family chatml --mutate \
    --boundary-ids 128001 \
    --boundary-map chatml/prefix-newline=128001 \
    --boundary-map chatml/prefix-trim=128002
```

This is the scientifically honest choice, not just convenience: each
candidate is a format hypothesis, and real formats terminate the assistant
turn with different markers (ChatML `<|im_end|>`, Llama 3 `<|eot_id|>`,
Gemma `<end_of_turn>`). Judging every hypothesis by a single sequence would
mis-measure the formats whose own terminator differs. Each candidate is
therefore measured with its own boundary -- the one its format's protocol
actually uses -- and the report records every candidate's boundary
(`boundary_token_ids` per search entry, per-candidate lines in the run
configuration, a boundary column in the candidate table).

The selection rule is unchanged: exactly one candidate must pass every
search fixture, each under its own boundary; ties and failures still
abstain, and the held-out set still only sees the frozen winner (measured
with the winner's boundary). The stop matrix and the budget frontier use
each candidate's own boundary as well. A mistyped candidate name fails fast
(the valid names are listed); `--mutate` variants are named
`<family>/toggle-bos`, `<family>/no-bos`, `<family>/prefix-newline`,
`<family>/prefix-trim` (some collapse when the family has no BOS marker --
see `catalog` output via `local_repairs` dedup).

Discipline: use each format's genuine end-of-turn marker, verified against
the checkpoint tokenizer with `doctor --boundary-ids` (or `doctor
--boundary-map NAME=ID,ID`, which checks each mapping). A boundary chosen
to make a candidate pass is not a measurement -- it is gaming the operator's
own experiment.

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
- `report.html`: offline static comparison -- decision banner, candidate table with per-candidate boundary IDs, per-fixture outcome grid, decisive fixtures, template conformance, stop matrix, token-budget frontier with per-candidate pass curves, sensitivity sweep and held-out cards. No scripts, fonts or remote assets.

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

## 8. Experiment diagnostics

Every `run` attaches diagnostic sections to `results.json` and `report.html`.
None of them changes the selection rule or the pass criterion; the decision
banner cites the failure-mode taxonomy whenever the run did not select
cleanly.

### 8.1 Template conformance

`run` diffs each candidate's hand-rendered prompt (encoded with
`add_special_tokens=False`) against the checkpoint tokenizer's own
`apply_chat_template` output for the same messages, as exact token-ID
sequences. Each divergent segment lists the token-ID subsequences that
differ and their positions; a byte-identical rendering reports an empty
diff. If `tokenizer_config.json` declares no chat template, the section
reports "no template declared" and the run continues. This makes
"sequences its own template would never produce" mechanically checkable
instead of a slogan.

### 8.2 Stop-behavior isolation matrix

For the frozen winner, `run` re-measures every fixture under emulated
competitor stop semantics and reports which semantics flip the verdict:

- **HF StopStringCriteria**: emulates the overhang matching from
  `transformers`' `stopping_criteria.py` on the decoded boundary text --
  generation stops after the token that completes the stop string, even
  when the stop string never appears as a literal suffix of the decoded
  text (token-boundary overhangs). For byte-level and byte-fallback
  decoders the match runs in byte space, mirroring current transformers
  (which builds the match table from raw token bytes and UTF-8-encodes the
  stop string); tokens the byte table cannot represent fall back to the
  isolated decode encoded as UTF-8.
- **vLLM detokenizer**: emulates `vllm/v1/engine/detokenizer.py`
  `check_stop_strings` -- the earliest-completing stop string in the
  decoded text truncates the output before it (exclude semantics, the
  vLLM default). The `stop_buffer_length` holdback only withholds streamed
  deltas and does not change the final truncation.
- **EOS-appended**: truncates before the first EOS token, emulating
  competitors that stop on EOS (lm-eval appends EOS to its stops; vLLM and
  llama.cpp stop on EOS by default). The lab itself generates through EOS.

Each cell recomputes the lab's pass rule (exact answer and token-ID
boundary) on the truncated output. Emulations are post-hoc truncations of
the recorded greedy N-token outputs -- exactly what early stopping would
have produced under greedy decoding; an approximation for non-greedy
backends. A cell is marked unavailable (never a failure) when the emulation
cannot run, e.g. the boundary text cannot be decoded.

The HF emulation ports the tensor implementation's matching rule, including
stop strings that end inside the final token (verified differentially
against `transformers`' own `StopStringCriteria`). Known approximations
remaining: on the classic string path, token texts come from per-token
`decode`, while real HF matches on cleaned vocab strings, so exotic spacing
edge cases can diverge; on the byte path, tokens the byte table cannot
represent (e.g. special tokens) fall back to the isolated decode encoded as
UTF-8, approximating HF's cleaned-vocab-string fallback for them.
The vLLM emulation likewise approximates incremental `DecodeStream`
detokenization with per-prefix batch decodes, and assumes `min_tokens=0`
(the lab never sets it).

**Why no llama.cpp cell.** Verified against llama.cpp's current sources
(`tools/server/server-context.cpp`, `common/common.h`), a llama.cpp
stop-semantics cell would duplicate the vLLM cell on every realistic input:
its full-stop rule is the same exclude-semantics "truncate before the
completed stop word" (`find_stopping_strings`: windowed `text.find` of each
antiprompt word on the unsent text tail, earliest start wins, the stop word
excluded from the final text). The differences do not survive post-hoc
emulation: partial stops only withhold *streamed* deltas and are never
erased from the final text; the multi-word earliest-start vs
earliest-completion selection is inapplicable to the matrix's single stop
string; the 1-character search-window offset sits below the emulation's
incremental-decode approximation floor; and EOS handling is already covered
-- llama.cpp stops on any end-of-generation token by default (the
EOS-appended cell), while `--ignore-eos` means "ignore end of stream token
and continue generating (implies --logit-bias EOS-inf)", i.e. the lab's own
native generate-through-EOS behavior. Adding the cell would double the
decode-approximation stack for zero new information, so it stays out by
design.

### 8.3 Failure-mode taxonomy

Every non-passing observation is classified into a closed, deterministic
taxonomy, with per-candidate counts in `results.json`:

- `no-boundary-in-N`: no boundary-delimited expected answer within the
  token cap (covers an absent boundary and a wrong answer with a found
  boundary; the per-observation records keep the finer distinction).
- `boundary-text-mismatch`: no token-ID boundary, but the decoded output
  ends with the decoded boundary text (the tokenization-mismatch
  false-negative mode).
- `eos-without-boundary`: the model emitted EOS without the boundary
  (only classified when an EOS id is known, e.g. from the checkpoint
  tokenizer).
- `held-out-divergence`: the search winner failed held-out evaluation.
- `backend-error`: the backend failed, reported an error, or the
  observation was cancelled.

### 8.4 Boundary portability (`doctor --reference-model`)

```sh
python -m model_protocol_lab doctor --model PRIMARY --reference-model REFERENCE \
    --boundary-ids 1,2,3
```

`doctor` decodes `--boundary-ids` with the primary checkpoint's tokenizer
and re-encodes the marker text with the reference checkpoint's tokenizer
(tokenizer only, never weights), reporting portable/non-portable with both
ID lists. A divergent encoding means the boundary IDs are
tokenizer-specific and will not transfer. Non-portability is reported as
`mismatch` -- informational, not a verdict failure, because boundary IDs
are per-checkpoint by design. Without `--reference-model` (or without
`--boundary-ids`) the check is skipped.

### 8.5 Prompt-boundary sensitivity sweep (`run --sensitivity-sweep`)

```sh
python -m model_protocol_lab run ... --sensitivity-sweep --max-drop-bytes 3
```

For each depth in 1..`--max-drop-bytes`, every prompt is truncated by that
many trailing bytes and re-encoded -- truncating the prompt in the middle
of its final token -- and the full search re-runs. The report lists the
depths where the selected winner or decision changed, measuring boundary
sensitivity as a variable. This multiplies search compute by the number of
depths, so it is opt-in. Depths that would split a UTF-8 codepoint record
a failed decision for that depth instead of silently skipping.

Related work: this measures the prompt-boundary tokenization bias that
*token healing* (guidance-ai, 2023) exists to mitigate -- when a prompt ends
mid-token, the next-token distribution is skewed because the final token
could have been encoded differently. Serving stacks that heal the boundary
(back up and constrain the first token) hide this effect; the sweep
quantifies it instead, telling you how much your verdict depends on the
exact byte where the prompt ends.

### 8.6 Token-budget robustness frontier

Every `run` recomputes the search-stage decision at every smaller token
budget from the recorded outputs and reports it as `budget_frontier` in
`results.json` and the HTML report -- answering "was the token cap
load-bearing for this verdict?" with no extra model calls.

The derivation is exact, not sampled: for an observation that passed
natively with the boundary first occurring at token offset `cut`, it passes
at every budget `b >= cut + len(boundary)` (the boundary occurrence and the
answer region are unchanged) and at no smaller budget (`cut` is the first
occurrence, so no complete boundary fits in a shorter prefix). An
observation that did not pass natively cannot pass at any smaller budget.
Each candidate is summarized by its per-fixture minimal passing budgets, the
pass-count step curve, and the budget ranges where the decision or winner
would have differed from native (`flip_ranges`); `budget_robust` is true
when the decision is identical at every budget from 1 to `--max-new-tokens`.
Like the stop matrix, this is a post-hoc diagnostic: it never changes the
selection rule, and it degrades to an "unavailable" marker (never a run
failure) when the search did not complete.

## 9. Authoring fixtures (`fixtures build`)

Hand-writing the strict fixture JSON is verbose, and the search/held-out
discipline (ID uniqueness, group non-overlap across partitions, no identical
message sequences across partitions) is easy to violate. `fixtures build`
compiles a compact authoring CSV -- one row per fixture -- into the strict
JSON, enforcing the discipline at build time with row-numbered errors:

```sh
python -m model_protocol_lab fixtures build cases.csv --output fixtures.json
```

The CSV header must be `id,group,partition,expected` followed by message
columns in `user_1,assistant_1,user_2,...` order:

```csv
id,group,partition,expected,user_1,assistant_1,user_2
search-copy-blue,copy-colors,search,BLUE,Reply with exactly BLUE and nothing else.,,
held-number,held-integer,held_out,7,Return only the numeral 7.,,
```

`partition` is exactly `search` or `held_out` -- the split assignment is
explicit per row, so leakage is caught while authoring instead of at
`inspect` time. The conversation ends at the first empty message cell; an
empty cell followed by content is an error (empty-string message content is
not expressible here -- use JSON for that). Quoted cells may contain
newlines. Cells are used verbatim: no whitespace stripping, no formula
evaluation. Limits mirror the runtime schema (2 MB source, 100 fixtures per
partition, 21 messages, 16,384 characters per cell).

The builder refuses to write a violating file: group overlap, duplicate IDs
and identical message sequences across partitions raise before any output
exists, and the emitted JSON is round-tripped through the strict loader
before it is returned -- a built file satisfies the runtime schema by
construction. The output path is never overwritten and is written with
owner-only permissions (fixture prompts can be sensitive). Empty `expected`
answers warn on stderr exactly like `inspect`, since they pass vacuously.
See `examples/fixtures.csv`, which builds to the equivalent of
`examples/fixtures.json`.
