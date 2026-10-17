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

### Template↔tokenizer portability matrix (`doctor --portability-matrix`)

```sh
python -m model_protocol_lab doctor --model PRIMARY --portability-matrix SECOND_CHECKPOINT
```

For every catalog candidate (each base format plus its bounded local
repairs -- the same hypothesis space `run --mutate` measures), `doctor`
checks the candidate's special-token inventory against the second
checkpoint's tokenizer: each token must be registered in the checkpoint's
added/special tokens, encode as exactly one token id, and encode to its
registered id. Tokenizer only -- weights are never loaded -- so this costs
about as much as the other doctor checks (one tokenizer load plus a few
encodes per candidate).

**Asymmetry warning: `incompatible` means the format cannot be the second
checkpoint's native format -- re-running the experiment there is
scientifically pointless, and you learned that for one tokenizer load.
`compatible` means the format's control tokens are merely encodable as
atomic special tokens by that tokenizer; it predicts NOTHING about
experimental success, which depends on the checkpoint's weights.** The
matrix never transfers a selection verdict across checkpoints and never
emits a portability score: it is a static exclusion screen, not a
portability measurement (research basis:
`docs/RESEARCH_PORTABILITY_2026-10-15.md`).

Details: formats with an empty `bos` (chatml/qwen3-style) skip the BOS
requirement -- a missing BOS token is not an incompatibility for them, and
each report row carries `bos_skipped` to document the skip. A format whose
special-token inventory is empty reports `unknown`, never `compatible`;
an encode failure degrades to `unknown` the same way. The matrix never
fails the doctor verdict: an unusable second checkpoint yields an
`unavailable` section with a fixed-vocabulary detail. The JSON report
carries the full matrix under `portability_matrix` (per-candidate rows
with per-token checks); the CLI also prints a readable table to stderr so
stdout stays parseable JSON.

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
see `catalog` output via `local_repairs` dedup), plus
`kolibri/thinking-off` (the Kolibri-1 thinking-disabled generation-prompt
branch), `nemotron35/thinking-off` (the Nemotron 3.5 Lightning
thinking-disabled generation-prompt branch), and `laguna/thinking-on` (the
Laguna S-2.1 thinking-enabled generation-prompt branch).

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

`--seed` accepts an unsigned 64-bit integer (default 0). It is validated up front and recorded in `results.json` for provenance, but it does not change greedy selection: the local adapter decodes greedily, so the seed has no effect on outputs. It exists so future sampling backends can share the flag.

With `--mutate`, the bounded candidate set adds BOS and assistant-prefix variants. The prefix mutations explore the same space as assistant-prefill controls in chat APIs (HF's `add_generation_prompt` / `continue_final_message`, vLLM's guided prefill): `prefix-newline` and `prefix-trim` vary how the assistant turn is opened, `toggle-bos` and `no-bos` vary BOS placement; `kolibri` and `nemotron35` additionally yield a `thinking-off` variant (the same template's generation prompt with an empty `<think>` block prefilled, for thinking-disabled requests), and `laguna` additionally yields a `thinking-on` variant (the S-2.1 thinking-enabled generation prompt). Search does not modify weights. A unique candidate passing all search cases is frozen before held-out evaluation. Ties cause abstention; the held-out set is not used to break them.

Ctrl+C requests cooperative cancellation after model loading. Interrupted evaluation can retain a partial report. Model loading and individual backend operations are not guaranteed to be promptly interruptible.

## 6. Read the outputs

- `results.json`: structured decision, summaries, candidate formats and fingerprint. Each observation carries `boundary_text_match` (see §4).
- `report.html`: offline static comparison -- decision banner (with selection margin, see §8.7), section table of contents, candidate table with per-candidate boundary IDs, per-fixture outcome grid, decisive fixtures, template conformance, stop matrix, token-budget frontier with per-candidate pass curves, token-overhead card (see §8.8), sensitivity sweep, log-likelihood diagnostic
(see §8.9) and held-out cards. Tables sort on header click via one small inline script (hash-allowlisted in the report's Content-Security-Policy); no fonts or remote assets. The report reads identically with scripting disabled.

Open the HTML file in a browser. It loads no fonts or other remote assets. The only script is a small inline table sorter; the content is unchanged with scripting disabled. Raw evidence is never embedded in this HTML view. Add `--include-evidence` only when you deliberately want raw observations in JSON and understand that token IDs can reveal private prompts.

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
`safe_budget_floor` is the first budget from which the native decision never
flips again (1 when nothing flips; absent when a flip reaches the sweep
maximum) -- the HTML report calls it out and marks it on the frontier and
margin charts, and `mpl diff-reports` compares it between runs.
`flip_attribution` lists, for each flip span, the (candidate, fixture)
observations whose minimal passing budget falls inside the span -- exactly
the pass-status changes the span brackets, i.e. the observations to
inspect when asking why the decision flips there (a span with none
records an empty list: the decision already differed at its start).
Structural fact: the per-budget full-passer set only grows with the
budget, so `flip_ranges` is always empty or a single prefix span
`[1, K]`; the floor is then simply `K + 1`.
Like the stop matrix, this is a post-hoc diagnostic: it never changes the
selection rule, and it degrades to an "unavailable" marker (never a run
failure) when the search did not complete.

The held-out stage gets the same treatment on a smaller scale:
`held_out_budget_floor` in `results.json` records the per-fixture minimal
passing budgets for the winner's held-out observations and
`safe_budget_floor` -- the first token budget from which the verification
verdict holds unchanged (the maximum of those budgets; absent when the
held-out failed natively, since no budget achieves a passing
verification). The held-out verdict is monotone in the budget, so this
single number is the whole story. The HTML held-out section calls it out
("verification floor ≥ K tokens", or a robustness/failure notice), and
`mpl diff-reports` compares it between runs.

### 8.7 Selection margin (decision fragility)

Every completed `run` reports `selection_margin` in `results.json` and one
line in the decision banner -- answering "is any fixture load-bearing for
this verdict?" with no extra model calls. The margin is a closed-form
integer over the recorded search outcomes: for a unique winner, the minimum
over runners-up of their failing-fixture counts (flips to a tie); for
`no_candidate`, the minimum over candidates (flips to a winner); for
`ambiguous`, the tied-candidate count minus one (flips to uniqueness).
`failed`/`cancelled` runs report no margin. The banner names the closest
runner-up and the decisive fixtures. The margin is descriptive; it never
changes the decision. A margin of 1 on a unique winner means the verdict
hinges on a single borderline fixture -- treat it as barely more
informative than a tie and consider authoring more discriminating fixtures
(see §9).

### 8.8 Protocol token-overhead accounting

Every `run` with an attached checkpoint tokenizer reports `token_overhead`
in `results.json` and a "Token overhead" card in the HTML report: per
candidate, mean/max prompt tokens (exact `tokenizer.encode` counts of the
rendered prompts, `add_special_tokens=False`) and mean output tokens,
the cheapest candidate, and the winner's token tax versus the cheapest
passing candidate as a percentage. Token counts in reports are
retained-input/output counts, not equal-compute accounting. Decision
utility: context-window budgeting and serving cost when two candidates
both pass, or when the winner is token-hungry -- formats differ
substantially in control-token overhead.

### 8.9 Teacher-forced log-likelihood diagnostic (`run --logprob-diagnostic`)

```sh
python -m model_protocol_lab run ... --logprob-diagnostic
```

Opt-in because it costs one extra forward pass per candidate x fixture.
After search completes, each candidate's *expected answer* is teacher-forced
through the checkpoint after the recorded prompt
(`model_protocol_lab/scoring.py`: one forward pass over
`prompt_ids + encode(expected)`; prompt positions masked, answer tokens
gathered with the causal shift). `results.json` carries `logprob_diagnostic`
and the HTML report renders a "Log-likelihood diagnostic" section with
per-candidate total/mean logprob (nats, per scored token) plus a
plain-language reading that separates the two failure attributions:

- **Greedy derailed**: the weights assign the expected answer *high*
  teacher-forced likelihood, yet greedy generation did not reproduce it --
  a decoding/search problem, not missing knowledge.
- **Weights don't know**: even teacher-forced, the weights assign the
  expected answer *low* likelihood under this candidate's framing.

The reading bands are declared heuristics recorded in the payload
(`thresholds`): mean >= -2.0 nats/token reads as derailment, mean < -5.0 as
missing knowledge; between the bands the diagnostic withholds judgment.
When the alignment check refuses a fixture (e.g. a token straddling the
prompt/answer boundary) the score is `unavailable` with a fixed reason --
reported, never guessed. Diagnostic only: it never changes the selection
rule or the pass criterion.

Related work: this is the lab's own DEVELOPMENT_PLAN.md §9.3-§9.5 design,
corroborated by lm-eval's `loglikelihood` request type and the HELM
adapter's token-probability measurements, which likewise score a fixed
target string by teacher forcing rather than by sampling.

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

## 10. Interactive protocol debugger (`debug`)

`debug` is an offline, stdlib-only REPL for poking at format hypotheses the
way the experiments do: render a catalog hypothesis over a fixture's
messages, inspect the per-segment char spans (and exact token IDs when a
tokenizer is attached), step through response-boundary detection one token
at a time, and audit fixtures for special-token literals -- all in one
terminal session, scriptable for repeatability. It is a local tool that
displays the operator's own fixture content by design; do not paste
`.save` transcripts into public issues.

```sh
python -m model_protocol_lab debug --dataset examples/fixtures.json
python -m model_protocol_lab debug --dataset examples/fixtures.json --model path/to/checkpoint
python -m model_protocol_lab debug --dataset examples/fixtures.json --script session.txt
```

- `--dataset` (required): fixture JSON file, loaded through the same
  bounded loader as `inspect` and `run`.
- `--model` (optional): local checkpoint directory. This attaches **the
  checkpoint's tokenizer only** -- it reuses `doctor`'s tokenizer path,
  building a `PreTrainedTokenizerFast` from the checkpoint's
  `tokenizer.json` (with the same 128 MB pre-load size cap). Model weights
  are never loaded, no model code is executed, and nothing touches the
  network. Without `--model`, `.tokens` and the per-segment token IDs print
  a reminder instead of IDs; everything else works.
- `--script` (optional): run a `.read` script non-interactively and exit
  (no banner); deterministic output, suitable for checked-in operator
  sessions.

A live session starts with a banner naming the dataset, the fixture count,
the active hypothesis family (starts at `chatml`) and the tokenizer state,
then prompts `mpl:<family> fixture-<n> >`.

Dot-commands (sqlite3-style: the dot namespace, unambiguous abbreviations
like `.help ren`, `.help PATTERN` search, `#` comments):

| Command | What it does |
|---|---|
| `.help [COMMAND\|PATTERN]` | List every dot-command, or full help for one |
| `.use <family>` | Switch the format hypothesis (validated against the catalog) |
| `.families` | List the 29 catalog families |
| `.fixtures` | List loaded fixture ids (search, then held-out); current one marked `*` |
| `.fixture <n\|id>` | Select the active fixture by number or id |
| `.render` | Render the hypothesis over the active messages, with a per-segment char-span map (plus exact token IDs per segment when `--model` is given) |
| `.tokens` | Exact token IDs of the full render (needs `--model`) |
| `.boundary ID,ID,...` | Set the boundary token-ID sequence (1-16 ASCII IDs, same grammar as `run --boundary-ids`) |
| `.stream ID,ID,...` | Record a token stream (up to 4096 IDs) for `.step` to walk |
| `.step [ID,ID,...]` | Walk boundary detection one token at a time; prints the running overhang (how much of the boundary currently matches as a stream suffix) and the first full-boundary position |
| `.audit` | Flag special-token literals (e.g. `<|im_start|>`) embedded in fixture text that could forge a role boundary |
| `.say` | Enter ad-hoc messages multiline (one per line; an optional `user:`/`assistant:` prefix sets the role, otherwise roles alternate starting with user); a lone `.` ends entry, and `.render` then uses the buffered messages instead of the fixture's |
| `.save <path>` | Append the session transcript (command echoes plus output) to a file |
| `.read <path>` | Execute a scripted command file (bounded: refused over 1 MB or 10k lines; nesting capped at 5) |
| `.quit`, `.q` | Exit (the literal line `EOF` also exits, e.g. end of piped input) |

Example session (real output; the segment map names every char span of the
render, so the "extra" control tokens stop being invisible):

```text
$ python -m model_protocol_lab debug --dataset examples/fixtures.json
model-protocol-lab debug 0.1.0
dataset: examples/fixtures.json (4 fixtures) | family: chatml | tokenizer: none (rerun with --model PATH)
Enter .help for usage hints.
mpl:chatml fixture-1 > .fixture 1
selected fixture 1: search-copy-blue
mpl:chatml fixture-1 > .render
render: family=chatml fixture=search-copy-blue messages=1 chars=91
<|im_start|>user
Reply with exactly BLUE and nothing else.<|im_end|>
<|im_start|>assistant

segments:
  [0] bos              chars 0:0
  [1] user_prefix      chars 0:17
  [2] user_content     chars 17:58
  [3] user_turn_end    chars 58:69
  [4] open_suffix      chars 69:91
mpl:chatml fixture-1 > .boundary 8
boundary set: [8]
mpl:chatml fixture-1 > .step 1,2,8,3
...
mpl:chatml fixture-1 > .quit
```

## 11. Differential backend stop testing (`diff-backends`)

When a run verdict surprises you, the suspect is sometimes not the weights
or the format hypothesis but the backend's stop behavior. `diff-backends`
runs the lab's stop emulations -- HF `StopStringCriteria`
(`hf_stop_string`), vLLM's detokenizer holdback with stop-string stripping
(`vllm_holdback_strip`) and keeping (`vllm_holdback_keep`), SGLang's
str-based finish (`sglang_stop`), and an EOS-appended truncation
(`eos_appended`) -- over **one recorded token sequence** and reports where
the emulated backends disagree:

```sh
python -m model_protocol_lab diff-backends \
  --model path/to/checkpoint \
  --prompt-text "the answer is" \
  --output-text "42 STOP hello" \
  --stops "STOP"
```

- `--model` (required): local checkpoint directory; tokenizer only --
  weights are never loaded.
- `--prompt-text`, `--output-text` (required): prompt and recorded output
  text, each encoded with `add_special_tokens=False`.
- `--stops` (required): comma-separated stop texts, e.g. `"STOP,</answer>"`.
- `--eos-id` (default: none): EOS token id for the `eos_appended` cell.
- `--min-tokens` (default 0): vLLM `min_tokens` gating applied to the
  holdback cells. SGLang has no `min_tokens` equivalent (its
  `min_new_tokens` only clips the length budget), so the `sglang_stop`
  cell is never gated -- that absence is measured by
  `NO_MIN_TOKENS_GATE`.

The SGLang cell (`sglang_stop`) ports `Req._check_str_based_finish`:
the first output step whose decoded text contains the stop string fires,
and the returned text is cut at the leftmost stop occurrence
(`DetokenizerManager.trim_matched_stop`) -- stripped by default like
vLLM, unlike HF. Its multi-stop rule is first-listed-wins, a hard
divergence from vLLM's earliest-completion rule (vLLM PR #49391).

The divergence classes, with one-line meanings (each drawn from the
vendors' own bug trackers; research log in `docs/RESEARCH_LOG.md`):

- `KEPT_VS_STRIPPED` -- HF keeps the stop string in the returned text
  while vLLM's default strips it (or vice versa); the same generation
  scores differently depending on the backend.
- `FIRING_STEP_DIFFERS` -- the emulated semantics disagree on *which*
  output step completes the stop (or whether it fires at all).
- `MIN_TOKENS_GATED` -- a stop that fires with `min_tokens=0` does not
  fire under the given `--min-tokens`.
- `EOS_PRECEDENCE` -- EOS fires at or before the stop-string firing step,
  so stop-string vs EOS precedence changes the outcome.
- `HARNESS_NORMALIZATION_RELIED` -- a strip-disposition cell's returned
  text still contains the stop string (the strip failed), so correct
  scoring would depend on the harness re-stripping.
- `MULTI_STOP_SELECTION` -- SGLang's multi-stop check selects the
  first-listed stop while vLLM selects the earliest-completing one, so
  the two engines stop on different stop strings for the same output.
- `NO_MIN_TOKENS_GATE` -- a stop that vLLM's `--min-tokens` gate blocks
  still fires under `sglang_stop` (SGLang has no `min_tokens`
  equivalent).
- `STR_OVER_EOS_PRECEDENCE` -- stop string and EOS fire in the same
  step, so precedence decides the outcome and the engines disagree:
  SGLang's shipped fix for #28802 gives the stop string the win, while
  TGI's documented precedence (length > EOS > stop-sequence) gives EOS
  the win.

Reading the JSON output: the top level carries `prompt_tokens`,
`output_tokens`, `min_tokens`, `eos_id`, `stop_texts`, `cells`,
`divergences` and the `verdict` (`"agree"` or `"diverge"`). `cells` holds
one record per stop text, with a `semantics` map of the five emulated
cells; each cell records `available` (false when the emulation could not
run -- recorded but excluded from the divergence checks), `fired`,
`firing_step`, `truncated_ids`, `returned_text` (set only by the
strip/keep-style cells) and `disposition` (`"kept"`, `"stripped"`, or null for
`eos_appended`). Each entry in `divergences` names the `class`, the
`stop_text` and a human-readable `detail`. A compact real example:

```sh
$ python -m model_protocol_lab diff-backends --model path/to/checkpoint \
    --prompt-text "the answer is" --output-text "42 STOP hello world" --stops "STOP"
{
  ...
  "divergences": [
    {
      "class": "KEPT_VS_STRIPPED",
      "stop_text": "STOP",
      "detail": "vllm_holdback_strip returned '42 ' while hf_stop_string kept '42 STOP'"
    }
  ],
  "verdict": "diverge"
}
```

Scope limit: one stop string is emulated per stop text per run, but
multi-stop *selection* is compared across the run's stop texts
(`MULTI_STOP_SELECTION`): the SGLang winner is the fired stop with the
smallest firing step, ties broken by list order, versus vLLM's
earliest-completion rule. With 1 token/step the earlier-completing stop
still fires at its own step under both rules; list order decides only
when two or more stops match within one check. Streaming vs
non-streaming divergence and match-position leftmost/rightmost rules are
likewise not emulated; the port follows the vendors' documented
non-streaming final-truncation paths. No semantics are invented: every cell
is one of the existing emulations in the `backends` module, run with fixed
inputs.

Deliberate non-cells: there is no TGI column -- every TGI stop rule is
covered by composition of the existing cells (TGI keeps stop sequences,
like the `hf_stop_string` cell; TGI multi-stop is first-listed, like the
`sglang_stop` selection rule; TGI has no `min_tokens` equivalent, like
`sglang_stop`'s ungated path; TGI's EOS-beats-stop precedence is the
documented contrast in `STR_OVER_EOS_PRECEDENCE`). SGLang's `stop_regex`
(`MAX_STOP_REGEX_COUNT=32`, `MAX_STOP_REGEX_LEN=256`) is outside
stop-string scope and gets this note only: neither vLLM nor HF has an
equivalent, and regex stops follow different matching code paths
upstream, so they are not part of the stop-string differential test.

Input bounds (never silent truncation): `--prompt-text`/`--output-text`
are capped at 1 MB each (argparse error, exit 2); encoded output
(generated) token sequences at 4096 ids (`output_ids_too_long`),
prompts at 131072 ids (`prompt_ids_too_long`).
The output cap is the load-bearing one: the emulations do one
`backend.decode(prefix)` per output step, so cost is quadratic in output
length -- at the cap a single stop completes in ~2-3 s on the repo's
linear reference stub, versus tens of minutes at the old 131072 cap
(measured 2026-10-15). It also matches `mpl run --max-new-tokens`' own
4096 cap, so the lab can never record a longer generated output. Each
stop text is capped at 1024 characters (`stop_text_too_long`) and a run
accepts at most 16 stop texts (`too_many_stop_texts`): a stop is a needle,
not a document, and each stop adds five emulated cells, so the count
multiplies the quadratic bound. Over-cap recorded observations in the
`run` diagnosis path are skipped (the section is omitted) rather than
burning quadratic time. The *count* of recorded observations the diagnosis
processes is also bounded: at most 512 (`MAX_DIVERGENCE_OBSERVATIONS`) --
legitimate runs hand over at most 200 (the experiment fixture cap is 100
search + 100 held-out fixtures, and only the winner's observations are
used), so real runs are never affected. When a report records more, the
first 512 are differential-tested in input order and the section records
the fixed-vocabulary `observations_truncated: true` with the recorded
count (`observations_total`) -- the run degrades by omission, never by
exception, and the HTML report names the bound explicitly.

## 12. Comparing two runs (`diff-reports`)

The core operator loop is iterate-and-compare: tweak a template or a
fixture, re-run, and see exactly what changed. `diff-reports` replaces
eyeballing two HTML reports with a precise, deterministic comparison of
two `results.json` files (or the report directories holding them). No
model is loaded, no network is used, and the same two inputs always
produce byte-identical output:

```sh
python -m model_protocol_lab diff-reports run-baseline run-fixed-template
```

Each argument is a report directory (containing `results.json`) or a
`results.json` path directly. The JSON comparison goes to stdout; a
human-readable summary goes to stderr. A real example -- the baseline run
selected no candidate, the fixed run selected `llama3`:

```
$ python -m model_protocol_lab diff-reports run-baseline run-fixed-template
diff-reports: run-baseline vs run-fixed-template
WARNING: run configurations differ -- results may not be comparable:
  - fixtures:held_out: [] -> ["h1", "h2"]
Decision: no_candidate -> compatible_on_fixture_set
Winner: (none) -> llama3
Candidate verdict changes (2):
  search/llama3: fail -> pass
  held_out/llama3: absent -> pass
Fixture outcome diffs (4 changed):
  held_out/llama3/h1: absent -> pass
  held_out/llama3/h2: absent -> pass
  search/llama3/f1: fail -> pass
  search/llama3/f2: fail -> pass
Summary: decision CHANGED, winner CHANGED, frontier CHANGED, 1 candidate(s) changed verdict, 2 fixture(s) flipped, 1 config drift warning(s).
```

The stdout JSON carries the same information in machine form: `decision`
and `winner` (old/new/changed), `frontier` (the token-budget frontier
comparison: sweep-range drift, budget-robustness flip, the exact
budget spans where the post-hoc decision, winner, or per-budget leader
differ, the safe budget floor, and per-flip-span attribution of the
driving (candidate, fixture) pass-status changes), `config_drift` (the comparability warnings), `candidate_verdicts`
(per-candidate pass/fail flips on search and held-out), `fixture_diffs`
(per-fixture old/new outcomes per candidate), `evidence` (whether each
side carried `--include-evidence` observations), and a `summary` with
counts (`fixtures_flipped`, `candidates_changed_status`,
`config_drift_warnings`, `frontier_changed`, `identical`).

What it compares, exactly:

- **Decision and winner**: old decision -> new decision, and the
  `selected_candidate` change if any. Exact string comparison, no fuzzy
  matching.
- **Candidate verdicts**: which candidates changed pass/fail status on
  search and on held-out (held-out verdicts exist only for each run's
  winner; every other candidate is `absent` there).
- **Fixture outcomes**: for each fixture and candidate, old vs new outcome
  (`pass` requires both the exact answer and the boundary token match --
  the same criterion `run` uses). Only *changed* fixtures are listed by
  default; `--all` lists every compared fixture.
- **Config drift**: the classic operator mistake -- comparing runs that
  were never comparable. The tool warns loudly when the runs used
  different candidate families (derived from candidate base names; the
  `--family` flag is not stored in `results.json`), different candidate
  sets, different per-candidate boundary token IDs, different
  `--max-new-tokens`, different backend fingerprints (model identity), or
  different fixture sets.
- **Token-budget frontier**: when both runs recorded one, the tool
  compares the post-hoc robustness story -- sweep-range drift, whether
  the selection stayed budget-robust, the safe budget floor, and the
  exact budget spans where the post-hoc decision, post-hoc winner, or
  per-budget leader differ (e.g. `budgets 3–4: alpha -> beta`). It also
  compares each flip span's attribution: the (candidate, fixture)
  pass-status changes driving the flip, so a template tweak that moves
  the flip from one fixture to another is reported even when the spans
  themselves are unchanged. A run without a recorded frontier
  is reported plainly instead of compared.
- **Held-out budget floor**: when both runs recorded one, the tool
  compares the verification's budget story -- the floor from which the
  held-out verification passes unchanged, and whether it was
  budget-robust. A run without a recorded floor is reported plainly
  instead of compared.

Per-fixture comparison needs per-fixture evidence: `results.json` carries
`observations` only when the run used `--include-evidence`. When either
side lacks it, the tool says so plainly on stderr and still compares
everything else (decisions, verdicts, config) instead of guessing.

Exit code is 0 whenever the comparison succeeds -- differences are the
tool's useful output, not a failure. Exit 1 means an input problem, with a
fixed error code on stderr and no traceback. Input bounds (all fixed
codes): at most 32 MB per `results.json` (`results_too_large`); the path
must resolve to a regular file -- a directory is accepted when it holds
`results.json`, while FIFOs, devices and sockets are refused
(`results_not_regular_file`); duplicate JSON fields and non-finite
numbers are rejected (`results_duplicate_field`, `results_nonfinite_number`);
operator-visible identifiers (fixture IDs, candidate names) are stripped
of control characters before they reach stderr.

Deliberate non-features: the comparison is exact -- there is no fuzzy
fixture matching and no verdict inference from missing summaries (a
malformed summary degrades to the verdict `unknown`, shown as such). HTML
reports are not compared; the structured `results.json` is the source of
truth.
