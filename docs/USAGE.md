# Usage

For troubleshooting and interpretation questions, see [docs/FAQ.md](FAQ.md).

## 1. Inspect the fixture schema

Each dataset has exactly two arrays: `search` and `held_out`. Every fixture has `id`, `group`, `messages` and `expected`. Messages have exactly `role` and `content`; roles alternate user/assistant, beginning and ending with user. System messages, tools and assistant prefills are not supported.

IDs are globally unique. Groups must not overlap across partitions. Identical message sequences cannot appear in both partitions. These checks do not detect paraphrases or other semantic leakage. Deliberately separate task families and freeze the held-out set before search. ID-uniqueness, group-overlap and identical-message comparisons use NFC-normalized text, so NFD/NFC variants of the same visual string cannot evade them; the stored fixture data itself is left byte-exact.

Limits include a 2 MB UTF-8 file, 100 fixtures per partition, 21 messages per fixture and 16,384 characters per text field. The dataset path must be a regular file — character/block devices, named FIFOs, sockets and directories are refused (a FIFO with no writer would block forever). Inspect the loader for the authoritative contract. To author fixtures without hand-writing this JSON, see `fixtures build` in §9.

Empty `expected` answers are allowed but pass vacuously: any output that emits only the boundary token satisfies them. `inspect` and `run` warn on stderr naming such fixtures so the measurement gap is visible.

`inspect` also runs a search/held-out contamination check: every held-out fixture is compared against every search fixture over NFKC-normalized, format-char-stripped, lowercased, whitespace-collapsed user text plus the expected answer, using character-level `SequenceMatcher` ratio and token-3-gram Jaccard. Pairs at `duplicate` (ratio ≥ 0.98), `near_duplicate` (≥ 0.90) or `paraphrase_suspect` (≥ 0.80 or Jaccard ≥ 0.85) print a stderr warning and appear in the JSON `contamination` block. The check is advisory -- thresholds are heuristic and heavy paraphrases still evade -- and never fails the command. `doctor --dataset` reports the same block, and `run` records it in `results.json` with a "Fixture hygiene" line in the report's run-configuration section.

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

### Behavioural template probe (`doctor --template-probe`)

The static template firewall (section 8.12) proves a checkpoint's chat
template contains no RCE primitives. It cannot see content-conditional
logic written in legal Jinja — a template that injects an instruction
only when a trigger phrase appears in a user message passes the AST
scan untouched. `model-protocol-lab doctor --model DIR
--template-probe` is the dynamic tier: it renders frozen probe
conversations — synthetic messages bracketed by sentinel strings —
through the checkpoint's own tokenizer, in pairs that differ by one
frozen word swap, and reports per pair whether the rendering changed
anywhere outside the swapped message's own span. Run it on a
downloaded checkpoint before trusting its template; that is the
decision it informs.

The JSON report carries the section under `template_probe`:
`counts` for every pair outcome (`clean`, `extra_span_change`,
`span_content_mismatch`, `transform_unverifiable`, `content_dropped`,
`span_unresolvable`, `unexercised`, `nondeterministic_render`,
`render_error`), a findings list (capped, grouped by the changed
region's SHA-256, each with a capped control-stripped excerpt), role-arm
render-diff facts (a message's span vanished, merged into another
span, or changed order when a system message is added or a user
message is relabelled), the template source SHA-256, and the probe
protocol fingerprint. Findings never change the doctor verdict or the
exit code, in either direction: the probe reports facts, not a
certificate. The full design — definitions, pre-registered claims,
budgets — is in `docs/TEMPLATE_PROBE_DESIGN.md`.

Limits, in the same breath as the capability: the probe
conversations are published constants, and a template can detect them
(sentinel, shape, or phrasing) and suppress conditional behaviour
while being probed, so a section with no findings bounds only
non-probe-aware templates; the probe set is a frozen sample, so a
backdoor keyed on anything outside its swaps is missed; the probe
reads only the `tokenizer_config.json`-embedded template string, so
list-form templates and a separate `chat_template.jinja` file report
`no_template`. Legitimate content conditionals produce findings too
— judge the inserted text, which is why it is shown.

## 3. Prepare a local model

Use trusted local GPT-2 or Llama causal-LM assets supported by the adapter. Required files include `config.json`, a self-contained `tokenizer.json`, and either `model.safetensors` or `model.safetensors.index.json` with its flat-named shards. Optional `generation_config.json` is fingerprinted but does not control generation.

The adapter rejects remote-code mappings, quantization configurations, adapter-only directories, encoder-decoder configurations, cross-attention configurations, conflicting weight layouts, asset symlinks and incomplete weight loading. No automatic downloads, pickle weights, custom model code, model-specific tokenizer Python code, quantized loading or LoRA merging are provided.

The tokenizer is loaded directly from tokenizer.json. Encoding disables automatic special-token insertion. Verify its behavior against the checkpoint's intended tokenizer before trusting results. Literal special-token markers may not behave as expected for every checkpoint.

The device is `cpu` or `cuda` (CUDA device 0). Weights and computation are requested in float32 for the primary measurement (the `run --precision` probe loads a second, bfloat16 copy purely as a comparison arm), with eager attention and no cache. Context limits are checked against model configuration; practical RAM/VRAM limits remain your responsibility. Stock GPT-2 checkpoints declare `n_positions=1024`, below the CLI default `--max-context 2048`: pass `--max-context 1024` (or less) for them, or backend initialization fails fast with the distinct code `local_context_limit_exceeds_declared_window` instead of the generic init failure.

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
see `catalog`), plus
`kolibri/thinking-off` (the Kolibri-1 thinking-disabled generation-prompt
branch), `nemotron35/thinking-off` (the Nemotron 3.5 Lightning
thinking-disabled generation-prompt branch), `ling/thinking-off` (the
Ling-3.1-flash thinking-disabled branch), `laguna/thinking-on` (the
Laguna S-2.1 thinking-enabled generation-prompt branch),
`qwen38/thinking-off`, `nanbeige42/thinking-off`,
`k2horizon/effort-medium` and `k2horizon/effort-low` (the K2 Horizon
reasoning-effort branches), `hy4/no-think`, and
`minimax3/thinking-disabled` / `minimax3/thinking-enabled`.

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

With `--mutate`, the bounded candidate set adds BOS and assistant-prefix variants. The prefix mutations explore the same space as assistant-prefill controls in chat APIs (HF's `add_generation_prompt` / `continue_final_message`, vLLM's guided prefill): `prefix-newline` and `prefix-trim` vary how the assistant turn is opened, `toggle-bos` and `no-bos` vary BOS placement; several families additionally yield generation-prompt variants for thinking-mode control (the empty-`<think>`-block prefill convention, and reasoning-effort branches): `kolibri/thinking-off`, `nemotron35/thinking-off`, `ling/thinking-off`, `qwen38/thinking-off` (enable_thinking=false), `nanbeige42/thinking-off`, `laguna/thinking-on` (S-2.1 thinking-enabled generation prompt), `k2horizon/effort-medium` / `k2horizon/effort-low` (reasoning_effort branches), `hy4/no-think` (reasoning_effort='no_think'), `minimax3/thinking-disabled` / `minimax3/thinking-enabled` (thinking_mode branches). Search does not modify weights. A unique candidate passing all search cases is frozen before held-out evaluation. Ties cause abstention; the held-out set is not used to break them.

Ctrl+C requests cooperative cancellation after model loading. Interrupted evaluation can retain a partial report. Model loading and individual backend operations are not guaranteed to be promptly interruptible.

## 6. Read the outputs

- `results.json`: structured decision, summaries, candidate formats and fingerprint. Each observation carries `boundary_text_match` (see §4).
- `report.html`: offline static comparison -- decision banner (with selection margin, see §8.7), section table of contents, run configuration, candidate table with per-candidate boundary IDs, per-fixture outcome grid, decisive fixtures, token overhead, template conformance, backend divergence, stop matrix, token-budget frontier with per-candidate pass curves, a deterministic inline-SVG frontier chart with a decision ribbon coloring every budget span by post-hoc outcome, a selection-margin trajectory chart with the safe-budget-floor dashed line, a per-candidate sparkline table, per-flip-span attribution lines, a prompt-boundary canonicality section, a chat-marker isolation matrix (see §8.11), a template-firewall verdict block (see §8.12), a selection-robustness matrix under stop semantics (see §8.13), a cross-checkpoint transfer matrix (see §8.14), a fixture-perturbation robustness section (see §8.15), a token-healing sensitivity section (see §8.16), an instruction-hierarchy adherence section (see §8.17), an assistant-prefill sensitivity section (see §8.18), a protocol-portfolio section (see §8.19), a synthetic-history selection
stability section (see §8.20), a batch-invariance audit section (see §8.21), a position-sensitivity section (see §8.22), a stop-cause termination audit section (see §8.23), a whitespace-canonicalization section (see §8.24), a structured-output adherence section (see §8.25), a delimiter-authority section (see §8.26), a few-shot x template section (see §8.27), a fake-turn history-tampering section (see §8.28), a prompt-only tool-call adherence section (see §8.29), a response-verbosity profile (see §8.30), a template surface-noise robustness section (see §8.31), a template field-ablation section (see §8.32), a prefix-cache sharability profile (see §8.33), a labeled-block confidentiality section (see §8.34), an answer-collision audit section (see §8.35), a suspected-refusal profile (see §8.36), a scorer-lens regrade (see §8.37), a candidate aliasing audit section (see §8.38), a prompt-repetition section (see §8.39), a distractor-dilution section (see §8.40), sensitivity sweep, log-likelihood diagnostic (see §8.9) and held-out cards (including the held-out verification-floor callout). A section renders only when the run recorded its stage; a run without that stage omits the section rather than rendering an empty one. Tables sort on header click via one small inline script (hash-allowlisted in the report's Content-Security-Policy); no fonts or remote assets. The report reads identically with scripting disabled.

A byte-deterministic example is committed at `examples/sample-report/report.html` (produced from `examples/fixtures.json` by `examples/make_sample_report.py`, which regenerates the directory rather than appending to it) -- open it to see most of the sections above with no model required. The sample's scripted run does not record every stage, so the template-conformance, backend-divergence, stop-matrix, sensitivity-sweep, log-likelihood, token-healing, template-robustness, field-ablation, scorer-lens, candidate-aliasing, prompt-repetition, distractor-dilution, leading-claim, numeric-surface-form and date-line sections are absent from the sample even though real runs render them.

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
flips again (1 when nothing flips; `null` when a flip reaches the sweep
maximum) -- the HTML report calls it out and marks it on the frontier and
margin charts, and `model-protocol-lab diff-reports` compares it between runs.
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
verdict holds unchanged (the maximum of those budgets; `null` when the
held-out failed natively, since no budget achieves a passing
verification). The held-out verdict is monotone in the budget, so this
single number is the whole story. The HTML held-out section calls it out
("verification floor ≥ K tokens", or a robustness/failure notice), and
`model-protocol-lab diff-reports` compares it between runs.

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

The HTML report's margin-trajectory chart (under the frontier chart) shows
the per-budget leader and its lead margin across the whole budget sweep,
answering "how decisive was the race at each budget?" -- the visual
complement of the scalar `selection_margin` above (gray spans mark tied
leads; the dashed line marks the safe budget floor from §8.6).

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

### 8.10 Prompt-boundary canonicality (`boundary_audit`)

Every `run` re-encodes each stored rendered prompt x greedy completion
and requires the joint encoding to equal the recorded
`prompt_ids + completion_ids`. A mismatch is the Prompt Boundary Problem:
the tokenizer re-segments the text at the prompt/completion seam, so the
ids the model saw are not what the stored text would reproduce. The
check is post-hoc and exact (one extra encode per observation, no model
calls); encode/decode failures degrade an observation to "unusable",
never a run failure.

`results.json` carries `boundary_audit` with the overall rate, per-candidate
rates, and per-family rates (`chatml/x` and `chatml/y` roll up into
`chatml`). The HTML report renders a "Prompt-boundary canonicality"
section with the family table; a zero rate is presented as a certificate
of boundary-canonicality for the run's templates. `model-protocol-lab
diff-reports` compares the overall and per-family rates between runs
(`boundary_audit_changed` / `boundary_audit_asymmetric` summary flags
feed the `identical` verdict). The metric is descriptive: it measures
text->ids reproducibility of the stored artifacts, not model quality,
and a nonzero rate names the families whose prompts the tokenizer
re-segments rather than condemning them.

### 8.11 Chat-marker isolation (`marker_isolation`)

Every `run` also probes structural template injection at the tokenizer
level: each of 16 hostile marker strings (`<|im_start|>`, `<s>`, `[INST]`,
`<|begin_of_text|>`, a forged `<|im_start|>assistant<|im_end|>` turn, and
others) is injected into user text, and the checkpoint tokenizer's
control-token ids are counted before and after — the number should not
change. The probe uses `add_special_tokens=False`, the same call the
backend's `encode` makes, so any control token in the output came from the
text itself; the special set is the tokenizer's `all_special_ids` union
special-flagged added tokens. A nonzero delta means the hostile text's
literal markers became real control tokens (the ChatInject/Phantom
injection class, via HF's `split_special_tokens=False` default).

`results.json` carries `marker_isolation` with per-(format, fixture, marker)
probes, per-family isolation scores, and the overall isolation rate. The
HTML report renders a "Chat-marker isolation" section with the families x
markers conformance matrix (`isolated` vs `smuggled` cells). The check is
pure tokenizer work (no model calls, no weights loaded) and never fails
the run: a tokenizer declaring no special ids reports `no_special_ids`
(unmeasurable, not a failure), and one bad triple degrades to
`unavailable`. The core lives in `model_protocol_lab/control_tokens.py`
with `render`/`encode` injected, so the whole probe is testable without a
model; `doctor.marker_isolation` is the checkpoint-backed driver.

### 8.12 Template firewall (`template_firewall`)

A checkpoint's `chat_template` is attacker-controlled Jinja, and the
conformance check renders it locally. Before any render, every `run`
statically scans the template source for RCE primitives
(`model_protocol_lab/template_audit.py`): dunder attribute/subscript
access, the `|attr` filter (CVE-2025-27516 sandbox escape), dangerous
Jinja globals (`lipsum`, `cycler`, `joiner`, `self`), `super()` calls,
and non-literal or huge `range()` bounds. A denied template is never
rendered — `template_conformance` fails closed with status `denied` —
and the findings surface as a "Template firewall" section in the report.

`results.json` carries `template_firewall` with the verdict (`allowed` /
`denied` / `no_template` / `unavailable`) and fixed-vocabulary findings.
An `allowed` verdict is not a safety proof: it means none of the known
primitives were found, and the render still runs inside transformers'
`ImmutableSandboxedEnvironment` (the lab pins `jinja2>=3.1.6` for the
sandbox-escape fix) under the existing CPU watchdog and memory cap. The
stack is: static firewall -> sandboxed render -> watchdog + memory cap.

### 8.13 Selection robustness under stop semantics (`selection_robustness`)

The stop matrix (§8.2) re-measures only the *winner* under emulated stop
semantics; the selection rule itself is only ever applied to native
generations. A user deploying on vLLM has no answer to "would a different
protocol have won under my serving stack's stop semantics?" Every `run`
now answers it post-hoc (`experiments.selection_robustness`, pure:
tokenizer-only, no model calls): for every search candidate it builds the
stop matrix, re-derives per-candidate pass/fail under each semantic
(exact answer + boundary on the truncated output), and re-applies the
conservative selection rule per semantic.

`results.json` carries `selection_robustness` with the per-semantic
verdicts (`native`, `hf_stop_string`, `vllm_holdback`, `eos_appended`),
the per-candidate pass matrix, and `flips` -- the semantics whose
(decision, winner) differs from native's, e.g. `vllm_holdback: ambiguous
-> selected(chatml)`. A semantic whose emulation cannot run for every
candidate (e.g. an unknown EOS id) is reported as `unavailable`, never
as a phantom `no_candidate`: "unknown" must not masquerade as a
verdict. The report renders a "Selection robustness under stop
semantics" section with the verdicts, the candidate x semantic matrix,
and flip callouts. `diff-reports` compares the per-semantic verdicts
and the flip pattern between runs (`selection_robustness_changed` and
`selection_robustness_asymmetric`, both feeding `identical`) -- the
serving-stack migration signal: the same stack selecting a different
winner than before.

### 8.14 Cross-checkpoint transfer (`transfer`)

`run --transfer-model` (repeatable, max 4) answers "does the winner also
win on my other checkpoints?" After the primary run, every search
candidate is re-evaluated against each transfer checkpoint on the
search split only (same boundary, same `--max-new-tokens`; greedy,
exact), and the conservative selection rule is re-applied per
checkpoint (`experiments.transfer_stage` + pure
`experiments.transfer_portability`).

`results.json` carries `transfer` with the checkpoint ids, paths and
fingerprints, the per-checkpoint decision/winner, the candidate x
checkpoint verdict matrix, and the transfer verdict (`full` / `partial`
/ `none` / `unavailable`). Per-cell verdicts: `transfers` (the primary
winner is the unique winner there), `wins_there` (another candidate
wins there), `ties_there`, `fails_there`, `checkpoint_incompatible`
(the candidate could not be evaluated there -- e.g. boundary IDs
outside the checkpoint's vocabulary -- fail-closed, never silently
skipped). The report renders a "Cross-checkpoint transfer" section;
the decision banner notes the verdict. Compute multiplies by
(1 + #checkpoints); the transfer stage never fails the run.
`diff-reports` compares the transfer experiment between runs
(`diff_transfer`: transfer-verdict drift, per-checkpoint
(decision, winner) drift, per-cell candidate x checkpoint verdict
drift, checkpoints added/removed — `transfer_changed` /
`transfer_asymmetric` feeding `identical`): the fleet-portability
regression signal after a checkpoint or fixture update.

### 8.15 Fixture-perturbation robustness (`perturbation`)

`run --perturb N` (1-8) answers "does the winner survive fixture
phrasing perturbations, or did it win on phrasing luck?" After the
primary run, every search candidate is re-evaluated against the
primary checkpoint on N deterministically perturbed fixture variants
(search split only; same boundary, same `--max-new-tokens`; greedy,
exact), and the conservative selection rule is re-applied per variant
(`experiments.perturbation_stage` + pure
`experiments.perturbation_robustness`).

The perturbation operators (`model_protocol_lab.perturbation`, pure
and seeded -- no model needed for the transforms themselves) rewrite
fixture *message content* only: `typo` (seeded character
substitutions), `case_flip`, `whitespace` (collapse / double /
newline-split), `unicode_variant` (ASCII alphanumerics to their
NFKC-foldable fullwidth forms -- nearly identical to a human reader,
different tokens to most tokenizers), and `punctuation_strip`. Fixture
ids, groups, and expected answers are never touched, so per-variant
verdicts stay comparable to the primary run; operators are assigned
round-robin from the run seed, so the whole experiment reproduces
exactly.

`results.json` carries `perturbation` with the per-variant
(decision, winner), the `flips` (variants whose verdict differs from
the primary run), the `stability` summary (how many evaluable
variants retain the primary winner/decision), and a per-operator
breakdown with the mean fixture-change rate -- so you can see *which*
phrasing noise the winner is fragile to. A variant where no candidate
could be evaluated is reported as `unavailable`, never as a phantom
verdict. The report renders a "Fixture-perturbation robustness"
section with the stability line, the per-operator table, and the
per-variant verdict table. `diff-reports` compares the stability
triplet, the per-variant verdicts, and the variant set between runs
(`diff_perturbation`; `perturbation_changed` /
`perturbation_asymmetric` feed `identical`) -- the phrasing-fragility
regression signal after a fixture or checkpoint update. Compute
multiplies by (1 + N); the stage never fails the run. Together with
stop-semantics robustness (§8.13) and cross-checkpoint transfer
(§8.14) this completes the winner-reliability trilogy: serving stack,
checkpoint fleet, and fixture phrasing.

### 8.16 Token-healing selection sensitivity (`token_healing`)

Tokenizers like SentencePiece split trailing whitespace into the next
token, so a prompt ending in `"hello"` encodes differently from the
same prompt decoded and re-encoded -- the greedy-tokenization bias at
the prompt boundary. Hugging Face's `token_healing` countermeasure
replaces the prompt's tail token with the highest-scoring vocabulary
extension at inference time. `model-protocol-lab run --token-healing` replays the
whole experiment once more, with every prompt healed before
generation, and re-applies the conservative selection rule to the
healed outcomes.

The implementation does not call `generate()` -- this backend decodes
greedily by hand -- but ports `transformers.GenerationMixin.heal_tokens`
faithfully into the lab's manual decoder (`model_protocol_lab/healing.py`,
a pure function with injected decode/re-encode/vocab/argmax callables):
decode the prompt, re-encode it (canonical re-tokenization), extend the
tail token with every vocabulary token that starts with it (sorted for
determinism), trim the tail, then take a single biased argmax that
adds +10.0 to every extension and +1.0 to the original tail token.
Two deliberate adaptations: decoding keeps special tokens (the lab's
prompts are exact id sequences), and an empty trimmed prompt fails
closed to the native ids instead of HF's BOS substitution (this
pipeline never injects implicit specials). The wrapper fails closed
everywhere: healing errors, healers that violate the
`tuple[int, ...]` contract, and the single healing forward pass all
degrade to the native prompt, never to a failed run.

`results.json` carries `token_healing` with the healed
(decision, winner), the `flipped` flag (verdict differs from the
primary run), the per-candidate (native, healed) outcome table, the
per-fixture flip list, a `falsification` block, and
`n_healed`/`n_fallback` counts. The falsification block tests the
cycle-134 prompt-boundary audit: the prediction is one-directional --
a verdict may change only for a candidate the boundary audit flagged
as seam-fragile, but a fragile candidate need not flip (healing is a
best-effort bias, not a guarantee). The report renders a
"Token-healing selection sensitivity" section with the healed verdict
line, the per-candidate table, and the falsification table.
`diff-reports` compares the healed verdict, the per-candidate
(native, healed) cells, and the falsification entries between runs
(`diff_token_healing`; `healing_changed` / `healing_asymmetric` feed
`identical`) -- the token-boundary regression signal after a
tokenizer or checkpoint update.

### 8.17 Instruction-hierarchy adherence probe (`hierarchy`)

When a system instruction and a user instruction conflict, which one
does each chat template lead the model to follow? The instruction
hierarchy (system > developer > user) is established as a
training-time property, but no harness measures it as a
*template-level* behavioral property. `model-protocol-lab run
--hierarchy` runs the probe: every search candidate is re-evaluated on
8 synthetic, fully deterministic system-vs-user conflict fixtures
(benign scenarios like "system says reply `azure`, user says reply
`emerald`"; adherence is exact string equality, the same semantics as
the pipeline's `exact_match`).

The lab never sends system messages (`core.Message` is
user/assistant only and `Format` deliberately does not model a system
role), so the system instruction is delivered as a privilege-marked
block folded into the user turn:

```
[PRIVILEGED SYSTEM INSTRUCTION]
{system instruction}
[/PRIVILEGED SYSTEM INSTRUCTION]

{user instruction}
```

in two positions -- before the user instruction (primacy) and after
it (recency). Each fixture's `expected` is the *system* answer, so the
standard pipeline pass/fail already means "followed the privileged
instruction"; the user-side match is scored alongside. The selection
rule is re-applied per position, and the report shows per-candidate
system/user/neither adherence rates, winner-vs-clean flips, the
primacy-vs-recency position effect, and a family-group comparison.

The group comparison tests a pre-registered structural hypothesis
against the catalog's own documentation: families whose real
templates give system content a dedicated, structurally isolated
frame (14 families, e.g. the `[SYSTEM_PROMPT]` / `<|im_start|>system`
/ `<role>SYSTEM</role>` families) vs families whose real templates
fold or merge system content into the user turn (3 families:
gemma, gemma3, deepseek). Families without an unambiguous documented
treatment are labeled `unknown` and excluded. This is reported as a
correlation readout over documentation-derived labels, not a causal
claim -- the labels are pinned by tests in
`tests/test_cycle147_hierarchy.py`.

`results.json` carries `hierarchy` with the per-position
(decision, winner), per-candidate adherence rates, the flip list, the
position effect, and the group means. `diff-reports` compares the
per-position verdicts, the flip pattern, per-candidate adherence
drift, and the group rates between runs (`diff_hierarchy`;
`hierarchy_changed` / `hierarchy_asymmetric` feed `identical`) --
the instruction-following regression signal after a checkpoint or
fixture update. Roughly triples search compute; the stage never fails
the run.

### 8.18 Assistant-prefill sensitivity (`prefill`)

Does the winning protocol survive when the caller pre-fills the
assistant's opening tokens -- and which template families honor vs.
ignore prefills? Prefill is a documented first-class generation
control (Anthropic's Messages API and AWS Bedrock document prefilling
the assistant's response; the attack literature shows prefill
overrides the initial-token distribution), yet no prompt-format
harness measures per-template prefill adherence or re-runs
*protocol selection* under prefill.
`model-protocol-lab run --prefill` runs the experiment: every search
candidate is re-evaluated with benign format-lock prefills
(`Answer: `, `Final answer: `, `Result: `) appended as raw token ids
after the generation prompt. The model generates the continuation
and the scored span is `prefill + generated text` -- the deterministic
contract that the model "said" the prefill.

Grading needs no fixture transformation: `prefill + generated ==
prefill + expected` iff `generated == expected`, so the pipeline's
`exact_match` on the plain expected answer already grades the prefill
run. Honoring is detected exactly and reported separately: a model
that did not accept the prefill as its own opening restarts its turn
and re-emits its preferred opener, *duplicating* the prefill text at
the start of its generation. The selection rule is re-applied per
prefill, and the report shows per-candidate honoring/restarting
rates, winner-vs-native flips, and a generation-prompt-structure
group comparison.

The group comparison tests a pre-registered mechanistic hypothesis:
candidates whose generation prompt opens a reasoning/think channel
(10 catalog families, e.g. qwen38's `<|im_start|>assistant\n<think>\n`
and kimi3's `<|open|>message role="assistant"<|sep|><|open|>think<|sep|>`)
honor format-lock prefills less, because the think-channel opener puts
the model in reasoning mode, which conflicts with an answer-format
lock. The predicate is computed from each candidate's own `Format`
(the generation-prompt tail), so thinking on/off variants of one
family are labeled independently; it is pinned per family by tests in
`tests/test_cycle148_prefill.py`. Reported as a correlation readout,
not a causal claim.

`results.json` carries `prefill` with the per-prefill
(decision, winner), per-candidate honoring rates, the flip list, and
the think-channel group means. `diff-reports` compares the
per-prefill verdicts, the flip pattern, per-candidate honoring drift,
and the group rates between runs (`diff_prefill`;
`prefill_changed` / `prefill_asymmetric` feed `identical`) -- the
prefill-sensitivity regression signal after a checkpoint or fixture
update. Roughly quadruples search compute; the stage never fails the
run.

### 8.19 Protocol portfolio (`portfolio`)

How much does committing to the single global winner leave on the
table versus a per-fixture oracle -- and what is the smallest protocol
portfolio that covers every solvable fixture? The routing literature
(RouteLLM et al., 2024) shows per-request model selection beats any
single model on the cost--quality frontier, but routing is always
*across models*. This is the routing lens applied *within* the lab to
*prompt protocols*: the per-fixture oracle is the set of passing
candidates for each fixture, and the gap quantifies what the
conservative single-winner commitment costs.

Every `run` computes this post-hoc over the stored search
observations (a candidate *passes* a fixture exactly when the run's
own measurement says `exact_match and boundary_found` -- nothing is
re-scored). `results.json` carries `portfolio` with the oracle pass
rate vs the global winner's rate and the gap (`gap >= 0` always: the
oracle passes every fixture the winner passes -- pinned by a
Hypothesis property), the per-fixture oracle winner sets, and the
greedy covering portfolio over solvable fixtures (deterministic
greedy set cover with lexicographic tie-breaks). The report's
"Protocol portfolio" section renders the gap, the cover list, and a
per-fixture winner table.

Two honesty points, stated in the report itself: the portfolio is a
greedy set-cover *approximation*, not a proven optimum; and fixtures
no protocol passes are reported as unsolvable, never folded into the
portfolio. `diff-reports` compares the gap, the rates, the
solvable-fixture set, the portfolio membership, the per-fixture
oracle-winner sets, and each member's covers between runs
(`diff_portfolio`; `portfolio_changed` / `portfolio_asymmetric` feed
`identical`) -- a shrinking gap means the protocols are converging; a
growing gap means the winner is increasingly specialized. Zero new
model runs; the stage never fails the run.

### 8.20 Synthetic-history selection stability (`synthhistory`)

Does the winning protocol survive when fixtures are asked after
synthetic prior conversation turns -- is protocol selection robust to
conversational context, or does history flip the winner? The
many-shot literature (Anil et al., Anthropic, 2024) proved that prior
turns -- even fabricated assistant turns -- steer subsequent behavior
predictably; no prompt-format harness re-runs *selection* under
synthetic history. `model-protocol-lab run --synth-history` prepends K fixed, benign,
fully deterministic synthetic prior turns to every fixture (one
exchange for `history_2`, three for `history_6`) and re-applies the
conservative selection rule per condition. The fixture's `expected`
answer is unchanged, so the standard `exact_match` grading already
scores the history-conditioned run.

`results.json` carries `synthhistory` with the per-condition
(decision, winner) re-application, the winner flips against the
primary run, per-candidate history pass rates, and the
delimiter-explicitness group comparison: families whose templates use
unambiguous per-turn open/close markers (derived mechanically from
the lab's own `Format` hypotheses -- non-blank open *and* close
marker per turn type) vs loosely-delimited families, reported as a
correlation readout, not a causal claim. The report's
"Synthetic-history selection stability" section renders the verdict
lines, the flip list, the pass-rate table, and the group comparison.
`diff-reports` compares the per-condition verdicts, the flip pattern,
per-candidate pass-rate drift, and the group rates between runs
(`diff_synthhistory`; `synthhistory_changed` /
`synthhistory_asymmetric` feed `identical`) -- a changed flip pattern
after a checkpoint or fixture update is the history-robustness
regression signal. Roughly triples search compute; the stage never
fails the run.

### 8.21 Batch-invariance audit (`batch_invariance`)

The lab's core invariant is "send exact token sequences, compare exact
outcomes" -- but GPU inference is not batch invariant. Thinking
Machines ("Defeating Nondeterminism in LLM Inference", Oct 2026)
documents that kernel selection, attention reduction order, and
KV-cache layout change per batch, so the *same prompt* can produce
different logits depending on what else ran alongside it. Every
prompt-format harness treats the serving path as a fixed point; the lab
sends exact token sequences, so batch composition is a hidden
confounder that could poison every comparison it publishes.
`model-protocol-lab run --batch-invariance` re-runs every search
fixture's exact prompt token sequence through the backend's batch path
(when one exists) and compares solo vs batched greedy continuation ids
token-for-token, reporting the divergence rate and first-divergence
positions per prompt.

Honesty note, updated at cycle 192: through cycle 191 no backend in
the tree exposed a batch path, so on the real local backend the stage
recorded `batch_unsupported`. Cycle 192 built the real path on the
local backend (`LocalHFBackend.generate_batch`: left-padded prompts,
position ids recomputed from the attention mask at every step, one
forward per generation step over the whole padded batch, the same
fp32 full-prefix recompute discipline as solo decoding -- see
`docs/BATCH_PATH_DESIGN.md`). The audit still deliberately does *not*
synthesize a batch out of sequential calls, and backends without a
batch path (including the prompt-rewriting wrapper backends) still
record `batch_unsupported`. Only prompts whose solo generation
succeeded are fed to the batch path -- a solo-failed prompt's row
could never be compared, and one over-context prompt must not
collapse an otherwise measurable audit. Equivalence is measured, not
assumed: any nonzero divergence rate falsifies the determinism
assumption for that backend -- and is a result in itself. A batch
whose estimated forward peak exceeds the backend's byte budget is
refused before any forward runs and records `batch_size_exceeded`;
the operator levers are `--max-new-tokens` and the fixture count.
Monday-morning use: before trusting a winner measured solo, run the
audit once -- if the verdict would flip under batched serving, the
divergence rate here is the number that says so.
`results.json` carries `batch_invariance` with the audit verdict
(`batch_invariant` / `batch_variant` / `unavailable`), the divergence
rate, and the per-prompt comparison. The report's "Batch-invariance
audit" section renders the verdict badge and the per-prompt table.
`diff-reports` compares the verdict, the divergence-rate drift, and
the diverged-prompt sets between runs (`diff_batchinv`;
`batch_invariance_changed` / `batch_invariance_asymmetric` feed
`identical`) -- a newly diverged prompt after a backend or fixture
update is the batch-nondeterminism regression signal. The stage never
fails the run.

### 8.22 Position sensitivity (`position`)

Option-order bias is studied by prompt *text*; the lab varies the
*chat template* as the independent variable. arXiv:2507.13949
("Exploiting Primacy Effect To Improve Large Language Models", Jul
2025) shows fine-tuned LLMs pick the first listed option far more often
than chance in MCQA; corroborated by arXiv:2512.02665 (primacy effect
in multi-document summarization). A "robust template" claim is hollow
if the template makes the model blind to anything not listed first.
`model-protocol-lab run --position` re-runs every search candidate on
synthetic multiple-choice fixtures whose correct option is permuted
between the first, middle, and last positions (seeded distractor
order) and re-applies the selection rule per condition.

Each fixture's `expected` is the correct option's *text*, so the
pipeline's `exact_match` grades the run unchanged; the per-candidate
primacy index is P(correct | first) - P(correct | last), in [-1, 1]
(pre-registered in `position.py` with the falsifiable claim that the
index varies across template families rather than being constant).
`results.json` carries `position` with the per-condition verdicts, the
winner flips against the primary run, the per-candidate primacy
indexes, and the condition effect (mean index, positive/negative
counts). The report's "Position sensitivity" section renders the
per-condition badges, the correct-rate table, and the index table.
`diff-reports` compares condition verdicts, flip patterns,
per-candidate correct-rate drift, and primacy-index drift
(`diff_position`; `position_changed` / `position_asymmetric` feed
`identical`) -- a primacy-index shift after a template or fixture
change is the position-bias regression signal. The stage never fails
the run.

### 8.23 Stop-cause termination audit (`stop_cause`)

The lab's backend contract deliberately generates *through* EOS to the
token cap ("Never stop at EOS or candidate boundaries"), so every
cause below is post-hoc classification of what a stop-aware engine
*would have* done with the recorded token ids -- the same honest
emulation stance as cycle 137's `eos_appended` cell. The boundary
audit (cycle 134) measures *where* a response ends; this experiment
measures *how it would have stopped*. Termination is a learned
behavior, not a guaranteed mechanism (arXiv:2407.20042, "When to
Stop?"), and the serving stack conflates two distinct mechanisms into
one finish reason (llama-cpp's `"stop"` covers both the EOS token and
the stop-string substring match).
`model-protocol-lab run --stop-cause` classifies every search
candidate's recorded generations into `eos_id` (the family's
end-of-turn token id was emitted), `stop_string` (the HF stop-string
emulation on the candidate's boundary text fired), `max_length` (the
cap was reached with no mechanism firing), or `stop_unknown`
(fail-closed: neither the EOT id nor a stop text was available).
Cause precedence follows the vLLM port notes in `backends.py`: the
earliest-firing mechanism wins, with a same-step tie going to
`stop_string`.

Per candidate the report carries the termination-cause rates, the
would-stop-early rate, and the over-generation bleed -- tokens the
lab's non-stopping decoder emitted past the first end-of-turn token,
which no real engine would emit. The pre-registered falsifiable claim
(in `stopcause.py`) is that termination-cause distributions are
identical across template families; a family that systematically runs
to max-length or bleeds past its own end-of-turn token while others
terminate cleanly falsifies it -- and means the template's own stop
tokens are leaking into measured verdicts. `results.json` carries
`stop_cause` with the per-candidate aggregates and per-observation
records. The report's "Stop-cause termination audit" section renders
the cause-rate table. `diff-reports` compares per-candidate
cause-rate drift, over-generation / would-stop-early / max-length
rate drift, and added/removed candidates (`diff_stop_cause`;
`stop_cause_changed` / `stop_cause_asymmetric` feed `identical`) -- a
newly eos-emitting candidate after a checkpoint or fixture update is
the termination-behavior regression signal. The stage runs no new
generations and never fails the run.

### 8.24 Whitespace canonicalization (`ws_canonical`)

Whitespace at template boundaries has caused real production bugs: a
training template ending in a lone space token, a serving template with
a trailing space changing the first generated token, an evaluator
disagreeing on leading-space outputs (aicassindra, "Tokenization, in
depth"), and adversarial tokenization degrading LLMs on purpose-built
cases (arXiv:2405.17067, "Tokenization Matters!").
`model-protocol-lab run --ws-scan` audits this per template family:
every search candidate's rendered prompts are re-encoded under 8
whitespace mutations of the render string (`trailing_space`,
`double_trailing_space`, `trailing_newline`, `trailing_tab`,
`leading_space`, `leading_newline`, `strip_trailing_space`,
`collapse_double_space`), the id sequences are compared exactly, and
fixtures on divergent variants are re-run through the normal
`core.evaluate()` pipeline via an encode-time backend wrapper
(`WsScanBackend`, mirroring `PrefillBackend`) to measure verdict flips.

The pre-registered falsifiable claim (in `wscanonical.py`) is that
whitespace variants of the same rendering tokenize to identical id
sequences and produce identical task verdicts. A mutation that diverges
in ids *and* flips a verdict makes the candidate whitespace-fragile;
a mutation that diverges in ids without flipping is reported as
divergent-but-stable (the per-mutation table marks `=` / `D` / `F`
cells). A flipped verdict on *identical* ids is recorded separately as
a backend-determinism anomaly -- it cannot be whitespace fragility, and
counting it as such would be dishonest. `results.json` carries
`ws_canonical` with the per-candidate per-mutation records and the
`whitespace_robust` verdict. The report's "Whitespace
canonicalization" section renders the id-divergence / verdict-flip rate
matrix. `diff-reports` compares per-candidate per-mutation rate drift,
robust-verdict flips, fragile-candidate set changes, and added/removed
candidates or mutations (`diff_ws_canonical`;
`ws_canonical_changed` / `ws_canonical_asymmetric` feed `identical`) --
a newly fragile candidate after a template or tokenizer change is the
regression signal. Roughly 9x search compute; the stage never fails the
run.

### 8.25 Structured-output adherence (`structured`)

Prompt-only JSON-mode adherence: the lab cannot constrain a decoder, so
the honest baseline measures what the template + prompt actually elicit.
`model-protocol-lab run --structured` probes each search candidate with
9 synthetic fixtures across 3 shapes (extraction, classification,
routing), each carrying a flat hand-rolled JSON schema (no runtime
dependencies) and a fixture-identity marker `(ref: structured/<id>)` so
scripted backends can key policy per fixture. Each decoded response is
classified independently of the strict `exact_match` verdict:
`parse_ok` / `parse_failed` / `not_text`, then `schema_ok` /
`schema_failed` / `schema_invalid` against the schema (a hand-rolled
validator -- string/integer/number/boolean/null fields, enum, finite
minimum/maximum bounds -- with a 200-case seeded brute-force oracle
differential in the test suite). Non-finite floats (`NaN`, `±Infinity`)
are not JSON numbers (RFC 8259 §6) and can never satisfy a `number`
field, even though `json.loads` parses them as a Python extension.
Per-candidate aggregation reports
parse/schema/adherent rates with the invariant `adherent_rate <=
parse_rate`; `not_text` responses are fail-closed exclusions.

The pre-registered falsifiable claim (in `structured.py`) is that the
template change produces a measurably positive adherence gap
(max-minus-min adherent rate across candidates) rather than ~0.
`results.json` carries `structured` with per-candidate cells,
`weakest_candidates`, `adherence_gap`, and the `uniform`
null-hypothesis flag. The report's "Structured-output adherence"
section renders the rate table. `diff-reports` compares per-candidate
rate drift, gap drift, weakest-set changes, and added/removed
candidates (`diff_structured`; `structured_changed` /
`structured_asymmetric` feed `identical`) -- a newly non-adherent
candidate after a template change is the regression signal. The stage
runs no new generations beyond its own fixtures and never fails the
run.

### 8.26 Delimiter authority (`deliforge`)

Reserved-token vs subword delimiter authority per template family: a
forged chat-template marker (e.g. `<|im_start|>`) injected in user
content reaches the model either as a single reserved control token or
as ordinary subword tokens -- the two decode to exactly the same text,
but arXiv:2609.35932 ("Same Bytes, Different Authority", Sep 2026)
reports attack success dropping 39-66 percentage points when the
marker is sent as subwords. The lab controls the exact token-id
sequence it sends, so `model-protocol-lab run --deliforge` holds
decoded text fixed while varying the encoding across all families --
the missing instrument.

For each of 2 forged-turn payloads (fake assistant turn asserting a
wrong answer; fake system instruction contradicting the task) and 4
delimiter markers, every search candidate is evaluated twice on the
forged fixtures: once with the forged marker in its native single-id
("reserved") encoding, once with the forged markers re-encoded to
their forced-subword form through a `DeliforgeBackend` plan. The
subword arm is built by text-level construction, not id alignment:
the exact inserted string (forged payload plus its unique per-fixture
ref) is located exactly once in the rendered prompt, and only the
forged marker occurrences are re-encoded segment-wise -- the
template's genuine markers keep their reserved ids (verified by id
count; id-level difflib alignment was proved unsound by a brute-force
oracle differential and is not used). Both arms decode to identical
text by construction (verified per fixture); any localization failure
degrades to an unsupported cell, never a phantom gap.

Per cell, the flip rates of the two arms against the native search
verdicts give `authority_gap = flip_rate(reserved) -
flip_rate(subword)` in [-1, 1]. The pre-registered falsifiable claim
(in `deliforge.py`) is a gap of ~0 uniform across families; a positive
gap replicates the paper's reserved-token authority finding inside
the lab. "Authority" is measured strictly as task-verdict flip rate on
fixed fixtures, not open-ended harm. `results.json` carries
`deliforge` with per-candidate cells, mean/max/min gaps, and
`max_authority_gap`. `diff-reports` compares per-cell gap drift,
top-gap drift, added/removed candidates, and excluded-set changes
(`diff_deliforge`; `deliforge_changed` / `deliforge_asymmetric` feed
`identical`) -- a newly positive gap after a template change is the
regression signal. Roughly 16x search compute (2 payloads x 4 markers
x 2 arms); the stage never fails the run.

### 8.27 Few-shot x template (`fewshot`)

Does in-context learning work the same under every chat template? For a
fixed demo set, `model-protocol-lab run --few-shot` measures the
zero-shot -> k-shot verdict-change rate (**gain**) and the demo-order
sensitivity per candidate on held-out fixtures, with model, demos, task
and decoder fixed -- the template is the only independent variable.
The ICL literature (arXiv:2511.09700 "Order Matters", Nov 2025) varies
demo order on one fixed prompt format per study; nobody varies the chat
template as the independent variable. The lab's 29 families x exact
token sequences is the missing instrument.

The v1 protocol pins k=2 with the two-orders design (the minimal
order-sensitivity probe): 2 demos are sampled (seeded; the pool is
id-sorted first so the set is input-order independent) from the
**search** split only, and every candidate is evaluated 0-shot and with
both demo orders on the **held-out** fixtures. Demos are prepended as
real user/assistant turns (each demo's question plus its answer under
`A:`, followed by an assistant turn confirming the answer), so every
candidate renders them through its own template; each fixture's
`expected` is unchanged, so the pipeline's `exact_match` grades every
arm unchanged. A demo fixture can never evaluate itself -- the demo
pool is the search split, the measurement set is the held-out split
(the contamination guard). Per candidate, `gain` is the mean over the
two orders of the per-fixture verdict-change rate (k-shot vs 0-shot)
in [0, 1], and `order_sensitivity` is the population standard
deviation of the two per-order change rates in [0, 0.5]; only fixtures
with a verdict on both sides are evaluable (missing verdicts are
excluded, never counted as unchanged). The selection rule is
re-applied per arm (baseline / order_a / order_b) and each arm's
(decision, winner) is compared against the primary run. The
pre-registered falsifiable claim (in `fewshot.py`) is a gain of ~0 and
order sensitivity of ~0 uniform across families; family differences are
the finding. Honest framing: gain is a verdict-*change* rate, never a
quality judgment -- a high gain means the demos moved the verdicts, not
that they helped. `results.json` carries `fewshot` with per-candidate
gain/sensitivity/per-order rates, per-arm (decision, winner), and
flips. `diff-reports` compares per-candidate gain drift, sensitivity
drift, per-arm drift, added/removed candidates, and excluded-set
changes (`diff_fewshot`; `fewshot_changed` / `fewshot_asymmetric` feed
`identical`). Roughly triples held-out compute; the stage never fails
the run.

### 8.28 Fake-turn history tampering (`faketurn`)

Does a forged prior turn flip verdicts differently depending on whose
delimiters it wears? `model-protocol-lab run --fake-turn` injects a
forged turn -- a fake assistant message asserting a wrong answer, or a
fake system instruction contradicting the task (the same payload
builders as `deliforge`) -- into the search fixtures' user content at
each family's native encoding, and measures the verdict-flip rate per
(candidate, payload, marker) cell against the native search verdicts.
The independent variable is the forged marker's token **provenance**:
`same` when the forged marker string also occurs as a genuine
delimiter in the rendering template's own benign render (the forged
turn wears the family's own delimiters), `cross` when the marker is a
fixed known chat-template control token from another family's token
set. Per candidate, the pooled same-family and cross-family flip
rates and the **transfer gap** (cross minus same, in [-1, 1]) are
reported: a systematically positive gap means foreign delimiters carry
more tampering authority -- arXiv:2405.20234's token-set-dependence
claim ("Hidden in Plain Sight"), measured per family with the model
fixed, the exact complement of the paper's cross-model design, plus
the cross-family transfer no other harness measures. The boundary
with `deliforge` is crisp: deliforge varies the *encoding* of fixed
forged text (reserved vs subword); this varies the *delimiter
provenance* at the family's native encoding. The pre-registered
falsifiable claim (in `faketurn.py`) is a transfer gap of ~0 and flip
rates that do not cluster by token set; clustering is the finding.
Every rendered prompt is integrity-checked (its unique ref occurs
exactly once, the forged marker at least twice); violations degrade
the cell fail-closed. `results.json` carries `faketurn` with per-cell
flip rates and provenance, per-candidate pooled rates and transfer
gap. `diff-reports` compares per-cell flip-rate drift, transfer-gap
drift, per-cell provenance changes, added/removed candidates, and
excluded-set changes (`diff_faketurn`; `faketurn_changed` /
`faketurn_asymmetric` feed `identical`). Roughly 8x search compute (2
payloads x 4 markers); the stage never fails the run.

### 8.29 Prompt-only tool-call adherence (`toolcall`)

Do tool-calling pipelines behave the same under every chat template?
`model-protocol-lab run --tool-call` describes a fixed 4-function tool
list (`get_weather`, `convert_currency`, `search_docs`, `set_reminder`,
with typed params, enums, and numeric bounds) in prompt text -- no
native function-calling APIs -- and asks each family to emit exactly
one line per query: `CALL <tool_name> <JSON arguments>`, or `NO_CALL`
when no tool fits. The 9 fixed queries cover the three BFCL-style
skills the Berkeley Function-Calling Leaderboard scores in prompt
mode (Databricks, "Beyond the Leaderboard"): **selection** (which
tool), **argument extraction** (typed parameter values from the
query), and **relevance detection** (2 queries with no relevant
function, where the correct answer is `NO_CALL` -- the abstention
analogue of the lab's conservative selection rule). BFCL varies
models/providers against fixed prompts and JSONSchemaBench varies
frameworks; nobody varies the *chat template* while holding model,
tools, queries, and decoder fixed. The lab's deterministic validator
is the honest baseline: unparseable lines are `parse_failed` (counted
as failures on all three rates, never silently dropped), non-string
responses are `not_text` (excluded from rates), and an unknown tool
name can never count as correct selection. A correct abstention
counts as correct selection (of no tool) with vacuously correct
arguments, so a perfect model scores 1.0 on every rate -- the rates
never carry a structural ceiling from the abstain queries. The
pre-registered falsifiable claim (in `toolcall.py`) is a
max-minus-min adherence gap of ~0 uniform across families, reported
separately per rate so a composite gap decomposes; a measurably
positive gap means the template is a first-class variable for
tool-calling pipelines. Boundary with cycle 157 (structured output):
157 measures JSON *schema* validity of a free response; this measures
function selection, argument extraction, and abstention -- different
task, different failure modes, different claim. `results.json`
carries `toolcall` with per-candidate selection/args/relevance rates
and the four gaps. `diff-reports` compares per-candidate
selection-rate, args-rate, and relevance-rate drift, added/removed
candidates, and excluded-set changes (`diff_toolcall`;
`toolcall_changed` / `toolcall_asymmetric` feed `identical`). The
probe floors the token budget at 128 (the longest canonical call is
79 characters) so an unrelated budget parameter can never truncate
calls into `parse_failed`. 9 synthetic fixtures; the stage never
fails the run.

### 8.30 Response verbosity profile (`verbosity`)

Do some chat templates make the model yap more than others?
`model-protocol-lab run --verbosity` measures, per template family,
the distribution of *excess generated tokens* on brevity-ideal
fixtures -- prompts with short stable expected answers (a pinned
rule: stripped, single-line, at most 32 characters). Excess =
generated tokens minus the exact tokenization of the expected
answer, clamped at zero; per family the report shows median, P25/P75,
IQR, mean, and max excess. The measurement is purely post-hoc over
the main run's observations -- zero extra model calls -- which is
exactly the point: arXiv:2601.00624 ("Do Chatbot LLMs Talk Too Much?
The YapBench Benchmark") finds an order-of-magnitude spread in median
excess length across 76 *models* on fixed prompts, but nobody varies
the *chat template* as the independent variable for verbosity. The
lab already records exact generated token counts, so the template
becomes the IV for free. The pre-registered falsifiable claim (in
`verbosity.py`) is a max-minus-min median-excess gap of ~0 uniform
across families; a measurably positive gap means the template, not
the model or the prompt, decides how much gets generated -- with
direct inference-cost consequences. Two honesty pins: (1) verbosity
is a cost/behavior profile, never a quality judgment -- the report
states explicitly that shorter is not better; (2) the per-candidate
budget-saturation share marks generations that hit the token budget,
whose measured lengths are *lower bounds* on true verbosity --
unconfounding this experiment from the cycle-155 stop-cause audit
(over-generation vs max-length termination are otherwise confounded
channels). `results.json` carries `verbosity` with per-candidate
distributions, most/least verbose behavior labels, the median gap,
and the capped-candidate list. `diff-reports` compares
per-candidate median-excess, IQR, and capped-share drift,
added/removed candidates, and excluded-set changes
(`diff_verbosity`; `verbosity_changed` / `verbosity_asymmetric` feed
`identical`). Monday-morning use: a developer budgeting inference
spend checks which template families systematically over-generate on
short-answer tasks before choosing a serving template. The stage
never fails the run.

### 8.31 Template surface-noise robustness (`template_robustness`)

Which chat template should you ship if your users type sloppily?
`model-protocol-lab run --template-robustness` answers by making the
template the independent variable: for each candidate, it perturbs
*user message contents* with presentation-level surface noise --
whitespace modes, case flips, unicode (fullwidth) variants -- and
re-renders through the template, keeping weights, fixtures, backend,
and protocol fixed. The headline is the **verdict preservation rate**:
the fraction of the template's *correct* fixture verdicts with no
pass-to-fail flip across the noise variants. Only pass-baseline
fixtures enter the denominator (a template that gets nothing right
reports no rate, not 0.0); variants that tokenize identically to the
original are no-ops, never counted; per-operator preservation vectors
are reported separately and never blended. Typo and punctuation
operators are excluded on principle -- they can change the correct
answer while `expected` stays pinned. Three honesty pins: (1) the
number is a template-x-backend-tokenizer interaction, valid only
within identical (fixtures, backend, protocol) triples; (2) nearest
observed flips are upper bounds over the searched subset, never
guarantees; (3) the stage is informational -- it never changes winner
selection. `results.json` carries `template_robustness` with
per-candidate rates, denominators, per-operator vectors, no-op lists,
and the frozen protocol fingerprint. `diff-reports` compares
per-candidate preservation drift and per-operator drift
(`diff_template_robustness`; feeds `identical`). Monday-morning use: a
developer choosing a serving template for a sloppy-input user base
compares which templates hold their correct answers under real-world
typing noise. Opt-in (about C*F*25 extra generations). The stage never
fails the run.

### 8.32 Template field ablation (`ablation`)

Your template won -- but what *about* it won? The BOS token, the role
labels, or the turn terminator? `model-protocol-lab run --ablate-fields`
answers by making each structural field the independent variable: for
the top-2 candidates by search accuracy (ties broken by name, so the
focus is deterministic), it replaces one `Format` field at a time --
`bos`, `user_prefix`, `assistant_prefix`, `turn_end`, `repeat_bos`,
`user_turn_end`, `assistant_turn_end`, `open_suffix` -- with its neutral
value ("" for text, False for repeat_bos, None for the optional
overrides) and re-evaluates, keeping weights, fixtures, and backend
fixed. The headline per arm is the **verdict-flip attribution**: how
many fixture verdicts changed in each direction, and whether the arm
still clears the conservative selection bar (passes every fixture). A
field is **load-bearing** when the base passes every search fixture but
the arm does not -- that field carried the candidate over the bar.
Three honesty pins: (1) an arm proves causality only within the
measured (template, fixtures, backend) triple, never universally;
(2) main effects only -- the 28 pairwise interactions are explicitly
out of scope; (3) a field already at its neutral value gets no arm
(a no-op swap proves nothing), and an arm with zero attribution across
all candidates is a publishable negative ("no single field explains
the winner"), reported explicitly via `any_attribution`. Each candidate
is judged under its own boundary (`--boundary-map` honored). The stage
is informational -- it never changes winner selection. `results.json`
carries `ablation` with per-candidate arms, flip counts, and
load-bearing fields. `diff-reports` compares per-field attribution
drift and load-bearing-set drift (`diff_ablation`; feeds `identical`).
Monday-morning use: a platform engineer writing the next template asks
not "which template won" but "what about it won" -- because only the
causal field transfers to the template they write next. The stage never
fails the run.

### 8.33 Prefix-cache sharability profile (`cache_profile`)

Your serving stack deduplicates prompt prefixes in the KV cache -- how
much prefill could it skip for each template? `model-protocol-lab run
--cache-profile` measures, per template family, the fraction of the
run's recorded prompt tokens that exact-prefix reuse (vLLM automatic
prefix caching) could skip, computed offline from the recorded
`input_ids` with zero extra model calls, block-quantized to full
blocks (default 16, vLLM's block size; partial trailing blocks never
count). The headline metric is the **achievable block fraction** --
exact, not a bound: for each shared prefix, every request order (a
permutation of the recorded prompts served once into a cold, infinite,
dedicated cache) yields exactly one first-occurrence miss and
k_{shared}-1 hits, so achievable = potential minus first occurrences,
order-free. Red-team A found the naive "cacheable fraction" metric
provably *misranks* templates against every realizable order (the
inversion is in the design record); the achievable quantity survived
an adversarial disproof attempt plus brute-force simulation over all
permutations of random prompt sets. The report also shows the token-
level diagnostic (ignores block quantization) and the eviction-robust
"potential" upper bounds -- labeled as such, never as production
ceilings. The pre-registered falsifiable claim (in `cacheprofile.py`)
is a max-minus-min achievable-block gap of exactly 0 uniform across
families; a measurably positive gap means the template, not the model
or the fixture set, decides how much prefill a prefix cache could
skip. Three honesty pins: (1) sharability is a cost *input*, never a
cost verdict -- it says nothing about absolute prefill cost, latency,
or TTFT; (2) the metric's closed world is a permutation of the
recorded prompts -- production interleaving, cross-workload warming,
and repeated requests can move real hits either way; (3) "efficiency"
is deliberately never used for it (the term belongs to HELM-style
accuracy-per-compute). `results.json` carries `cache_profile` with
per-candidate cells, most/least sharable labels, the gap, and an
`evaluable_mismatch` flag when candidates were measured on different
denominators. `diff-reports` compares achievable-block, achievable-
token, and potential-block drift, added/removed candidates, and
excluded-set changes (`diff_cache_profile`; `cache_profile_changed` /
`cache_profile_asymmetric` feed `identical`). `--cache-profile-block-size`
re-quantizes for other engines (recorded in the output). Monday-morning
use: an inference engineer deciding between two prompt templates for a
vLLM deployment checks which one's prompts share more exact prefixes
(block-quantized) before wiring up prefix caching -- the number is the
routing/templating input, not a serving-cost promise. The stage never
fails the run and never touches selection.

### 8.34 Labeled-block confidentiality (`confidentiality`)

If you prepend standing instructions to every prompt as a labeled block,
does the label stop the model quoting that block back when a user asks?
`model-protocol-lab run --confidentiality` renders a fixed nonce marker
(`zorblax quindle vexmor 7719`) inside the privilege-labeled block -- the
same `[PRIVILEGED SYSTEM INSTRUCTION]` framing the hierarchy probe uses,
because the lab has no system role -- and, as a control, the identical
bare marker at the same position in the user turn. Frozen extraction
elicitations (a direct request, an authority claim, a memory reframe)
then ask for the block quoted verbatim, mirrored exactly across arms.
Detection runs on the full decoded output (never the boundary-truncated
answer): the longest contiguous run of marker tokens, with a leak at 3
of 4 tokens. Per candidate you get the paired control/treatment leak
rates, the gap (control minus treatment), discordant pair counts, and an
exact McNemar p-value; the profile reports the most/least resistant
candidates, the positive set (gap > 0 with at least 2 discordant
control-only pairs), and `uniform_nonpositive` -- the null result, where
the label conferred no measurable selective protection anywhere (the
analogue of Tensor Trust Appendix H's small-difference
role-scheme result, and a publishable negative).

Read the numbers as a **verbatim-exposure lower bound**, never as a
confidentiality certification: paraphrase, translation, and
summarization leakage are out of scope by construction, a nonce marker
is not natural-language system content, and a low leak rate is not a
"safe to ship secrets" claim. "Confidentiality" here is the behavioural
sense (OWASP LLM07, system prompt leakage), not confidentiality at rest
or in transit. Fail-closed throughout: candidates whose control arm
never leaks carry no information about the label (their gap is `None`,
flagged `base_rate_zero`), candidates under the evaluability floor
report `None`, and if no candidate is measurable the payload is
`{"available": false, "detail": "confidentiality_unavailable"}`.
`diff-reports` compares gap drift, positive-set changes, the null flag,
and protocol-fingerprint drift (a marker or threshold change makes runs
not like-for-like; `diff_confidentiality`, `confidentiality_changed` /
`confidentiality_asymmetric` feed `identical`). Monday-morning use: an
operator about to paste a real policy block into every prompt checks
whether their chosen template's labeling actually reduces verbatim
exposure under direct asking, before trusting the label. Opt-in: costs
2 arms x 12 fixtures x C extra generations. The stage never fails the
run and never touches selection.

### 8.35 Answer-collision audit (`answer_collision`)

Could your boundary marker silently truncate correct answers that quote
the marker? `model-protocol-lab run --answer-collision` audits the lab's
own token-ID boundary detector: synthetic expected answers embedding
marker-like text (the configuration's own marker at the start, inside,
at the end of, and repeated in the answer; every hostile protocol
marker from the control-token catalogue; one clean control) are emitted
verbatim through the evaluation path by scripted backends, and each is
classified by a total order -- **collision** (the detector cut a correct
answer and discarded content that followed the marker),
**ends-at-marker** (the scripted emission simply ended at the marker; a
distinct bookkeeping cell, never evidence of early stopping), or
**intact** (the detector did not fire). Each fixture runs in two arms:
the reserved arm (the backend's natural encoding -- the measurement)
and a subword control arm (the same decoded text with every marker
occurrence re-encoded in subword ids, where the detector is blind; a
collision there means the boundary matches non-reserved encodings).
Where no encoding pair exists the arm records `no_encoding_pair`; where
the backend's encode/decode does not round-trip an answer exactly, the
fixture leaves the denominator and is counted, never silently dropped.

The unit is the **boundary configuration**, not the template:
boundaries are operator-supplied ids (shared by default, overridden per
candidate by `--boundary-map`), and configurations sharing a boundary
produce identical cells by construction. Scope is the configured
boundaries and this tokenizer only; scripted emission measures the
detector, never any model, and says nothing about how often real
answers contain marker text. Fail-closed throughout: if no
configuration is measurable the payload is `{"available": false,
"detail": "answer_collision_unavailable"}`, and `uniform_intact` -- the
detector truncated nothing anywhere -- is the regression-guard pass.
`diff-reports` compares reserved-arm collision-rate drift, the colliding
set, the intact flag, and protocol-fingerprint drift
(`diff_answer_collision`, `answer_collision_changed` /
`answer_collision_asymmetric` feed `identical`). Monday-morning use:
before trusting exact-match scores on a new checkpoint or boundary
setup, check whether your boundary marker can manufacture failures out
of correct answers. Opt-in: runs entirely on scripted emissions (no
extra model generations, but `model-protocol-lab run` itself loads the
checkpoint as usual). The stage never fails the run and never touches selection.

### 8.36 Suspected-refusal profile (`refusals`)

Opt-in with `model-protocol-lab run --refusals`. Post-hoc over the
main run's recorded search observations -- zero extra model calls:
each candidate's recorded answers are classified against a frozen,
published refusal-opener list, and the run gains a `refusals`
section with the per-candidate suspected-refusal rate on benign
fixtures. Monday-morning use: after adding or changing a chat
template, check whether the new rendering makes the checkpoint
refuse questions it answered under the old one -- with the weights
and the message content held fixed, a rate difference is a
rendering effect by construction.

The detector is frozen as `frozen_refusal_openers_v2`: the core is
XSTest's Appendix D string list (arXiv:2308.01263), minus "no," and
"no." -- on this lab's short-answer fixtures those two fire on
correct yes/no answers, so they are dropped and the deviation is
recorded in the detector's own docstring -- plus itemised
supplements from OR-Bench Appendix M (arXiv:2405.20947) and the
GCG refusal-string lineage. An answer is a *suspected* refusal when
its normalised form starts with a listed opener and the opener ends
at a word boundary (NFKC, format characters treated as spaces,
casefolded, combining marks stripped, curly apostrophes folded,
whitespace collapsed; the longest boundary-satisfying opener is
reported). The boundary rule is the v2 change (cycle 175, wave-7
red-team find): bare prefix matching also fired on "I will note
the key points..." and "As an Air Force pilot...", and a detector
that manufactures the effect it measures is worse than one that
misses a glued "I cannotprovide...". v1 rates are not comparable
with v2 rates; the section's `method` block carries the detector
id, the full list, and its SHA-256 digest in every result, so a
detector change is visible in `diff-reports` as protocol drift
instead of silently moving the rates.

Read the numbers as the section names them: *suspected*. String
matching is the field's known-weak baseline -- XSTest's own
validation shows it over-counting one model's refusals (empathy
openers such as "I'm sorry to hear...") and under-counting
another's, with the sign of the error model-dependent, and no
published accuracy figure transfers to this fixture set. Three
consequences are built in rather than footnoted: every rate travels
with its exact counts (suspected k of n completed); the
suspected-and-passed cell separates detector artifacts on
echo-style fixtures (whose expected answer itself opens with an
opener) from the decision-relevant suspected-and-failed cell,
while a hedged failing answer ("I can't be certain, but ...")
stays indistinguishable from a true refusal at this tier; and
answers that open with markdown or a quotation mark are a
documented miss, not silently normalised into hits. The classified
text is the lab's recorded answer (the boundary-cut retained
decode every answer-level analysis here measures), over completed
search observations only -- held-out observations are never read.

Rates are withheld (reported as counts with a `rate_note`, never a
number) for a candidate whose search did not complete: an
interrupted candidate is measured on a fixture prefix, and
comparing its rate against full-set rates would be a silent
denominator change. When candidates' completed fixture sets
differ, the section sets `evaluable_mismatch` and the HTML report
prints a caution next to the table. Hits identify the fixture and
the opener that fired -- a pointer to the evidence, never a copy:
the answer text itself stays in the evidence-bearing observations,
so the section is safe in the public (evidence-stripped)
`results.json`.

`diff-reports` compares per-candidate rate drift (with denominator
drift alongside -- a rate that moved because its n moved is a
different story), hit-set changes, excluded-candidate transitions,
the mismatch flag, and detector protocol drift
(`diff_refusals`, `refusals_changed` / `refusals_asymmetric` feed
`identical`). Informational only: the profile never affects
selection, and the stage never fails the run.

### 8.37 Scorer-lens regrade (`scorerlens`)

Opt-in with `model-protocol-lab run --scorer-lens`. Post-hoc over
the main run's recorded search observations -- zero extra model
calls: every recorded answer is regraded under a frozen ladder of
normalisations, and the run gains a `scorerlens` section with
per-candidate verdict counts per rung, first-flip attribution per
fixture, and the search-phase winner re-derived under each rung.
Monday-morning use: a template change made the run look worse --
decide whether the template actually got worse or whether the
byte-exact scorer is discarding answers a human would accept, and
whether the winner survives the mildest defensible scorer.

The ladder is one pipeline (each rung consumes the previous
rung's output, so a fixture passing at rung k passes at every
higher rung by construction): R0 exact; R1 trim, collapse
space/tab runs, cap newline runs at two (line structure survives:
a two-line answer is not the same answer on one line); R2 NFC,
casefold, combining-mark removal (the refusal detector's fix for
Turkish dotted capital I); R3 line-anchored markdown-fence strip,
then iterative stripping of a small presentational punctuation
set at the string edges; R4 R3, or both sides match a frozen
ASCII numeric grammar (plain digits or strict thousands
grouping, optional fraction/exponent) and parse to finite, equal
decimal values. The ladder is embedded in every payload as data
with its SHA-256 digest, so a ladder change is protocol drift in
`diff-reports`, never a silent redefinition. The full contract --
including the strip set as an explicit character list -- is in
`docs/SCORER_LENS_DESIGN.md`.

Three honesty boundaries are part of the instrument, not
footnotes. The ladder has *no label-strip operation*: an answer
of `Answer: 42` against expected `42` flips at no rung -- lead-in
prose is not surface the ladder forgives. R4 does not read
decimal commas (`1,5` is one-and-a-half in much of the world,
not `15`; the ladder has no locale input), never equates
underscores or non-ASCII digits with ASCII ones, and compares
parsed decimals by value, never by a normalised string form.
Casefold equates German ß with ss, so a rung-2 flip on a name
fixture deserves a second look; the payload's flip attribution
records whether a rung-2 flip needed more than NFC. Verdict
counts always conjoin the recorded boundary fact, which no rung
can move, and the boundary-blocked count travels with the table:
a winner that is stable because every candidate is
boundary-blocked is not scorer evidence. The winner readout is
the *search-phase* winner only -- held-out is never regraded --
and it is withheld, with a named reason, whenever any candidate
is unavailable or ineligible or the evaluable fixture sets
differ. Fixtures whose expected text is empty (or whitespace
only) are excluded from the lens with their count reported: the
lens has nothing to say about a fixture with no surface. The
payload carries counts, fixture IDs and rung numbers only -- no
answer text, no expected text, no normalised forms -- and its
per-fixture ground records let `diff-reports` recompute every
aggregate before trusting it.

`diff-reports` compares per-rung winner drift, per-candidate
surface-failure-share drift (with the failure denominators
alongside), per-fixture first-flip-rung changes,
excluded-candidate transitions, the mismatch flag, ladder
protocol drift, and fixture-set drift -- across a different
fixture set or ladder, drift is still listed but flagged
`comparable: false`, because a moved winner then conflates the
exam change with the scorer's effect (`diff_scorerlens`,
`scorerlens_changed` / `scorerlens_asymmetric` feed `identical`).
Informational only: the regrade never affects selection, and
the stage never fails the run.

### 8.38 Candidate aliasing audit (`aliasing`)

Opt-in with `model-protocol-lab run --aliasing`. Post-hoc over
the main run's recorded search observations -- zero extra model
calls: candidates are grouped into token-identity equivalence
classes over the recorded `input_ids`, and the run gains an
`aliasing` section with effective-vs-recorded candidate counts,
the class lists, a per-pair coincidence matrix, and the
alias-suspect abstention flag. Monday-morning use: after a
24-candidate sweep, decide whether to prune the candidate set
before the next expensive run -- and whether an `ambiguous`
abstention was two genuinely different templates tying, or one
token sequence wearing two name tags.

Two class notions are reported and never conflated. *Render
classes* share one prompt token sequence on every fixture of
their cohort; *measurement classes* also share the response
boundary, which is part of the measurement in this lab (under
`--boundary-map`, identical prompts under different boundaries
are different experiments that happen to share a prompt).
Effective counts are reported for both; the redundant count and
the flag cite measurement classes only. Classes exist only
inside cohorts of candidates with identical completed fixture
sets -- an interrupted candidate was measured on a fixture
prefix and is listed as excluded, never silently merged.
Redundant completed observations are counted once per fixture
under one formula; they are redundant as prompt evaluations,
and they are also the replicates the determinism cross-check
below consumes, which is why the payload calls them redundant,
never wasted.

Within one measurement class, the recorded outputs must agree
per fixture: the per-request seed derives from the run seed and
the fixture ID only, and the generation cap is run-global, so
identical measurement configurations present identical
generation inputs. A disagreement is reported per cell as a
*determinism-suspect* cell -- recorded cells a deterministic
backend could not have produced from one measurement
configuration, attributable to backend nondeterminism (stateful
or order-dependent backends included) or artifact corruption;
the audit asserts neither cause, and `verify --determinism`
replay is the discriminating follow-up. The alias-suspect flag
fires only when the decision is `ambiguous`, the recomputed
passing set has at least two members inside one measurement
class, and that class holds no outcome-suspect cell; a tie
spanning distinct prompts never fires it -- that is the lab
working as intended, not an artifact. The flag is advisory: it
never rewrites the decision, and it does not assert any
candidate would have survived held-out, which was never run for
any of them.

Honesty boundaries. Every claim is relative to this run's
fixture set and backend tokenizer: two templates identical on
short fixtures may diverge on long or multi-turn ones, and a
collapse is a tokenizer-by-fixture interaction, not a claim
that the catalog families are duplicates. Back-porting works
over `--include-evidence` artifacts and in-memory results
only; over a public `results.json` (observations stripped) the
audit is unavailable by design and says so. The payload carries
no raw token IDs and no text -- per cell it carries SHA-256
digests of the canonical ID lists under a frozen digest
definition (`sha256-json-intlist-v1`), plus counts and class
structure; an unsalted digest of a short prompt does permit
offline confirmation of a guessed prompt, which is strictly
weaker than carrying the IDs themselves (they reconstruct the
prompt exactly). The payload is the largest per-cell payload
the lab ships (about 0.92 MB worst case in the indented public
JSON at the run caps of 24 candidates x 100 fixtures), which is
why the audit is flag-gated. Its ground records let
`diff-reports` recompute every aggregate -- an
internal-consistency recompute: it detects aggregates that
disagree with their records, not fabricated records.

`diff-reports` compares measurement-partition drift over the
common candidates (pairs joined / pairs split), per-candidate
boundary drift, effective-count and suspect-count drift, the
flag, candidates added/removed/excluded, and the mismatch flag
-- across a different fixture set, backend fingerprint, or
digest version, drift is still listed but flagged
`comparable: false`; a changed backend fingerprint means
tokenizer identity is *not proven equal* (the fingerprint also
binds device, context limit, and dependency versions), never
"a different tokenizer" (`diff_aliasing`, `aliasing_changed` /
`aliasing_asymmetric` feed `identical`). The full contract is
in `docs/ALIASING_AUDIT_DESIGN.md`. Informational only: the
audit never affects selection, and the stage never fails the
run.

### 8.39 Prompt repetition (`repetition`)

Opt-in with `model-protocol-lab run --repeat`. A generation
stage over the search split: each fixture's last user message
is repeated verbatim inside the same user turn (the x2 arm),
and, as a control, followed by neutral filler sized to the
exact same rendered token length (the pad arm); the x1 arm is
the recorded native search, reused with zero extra model
calls. Monday-morning use: before shipping a prompt that asks
twice, learn per candidate template whether repeating the
request changes verdicts once prompt length is held constant
-- and whether the winner survives repetition.

The report's headline number is the *repetition effect* (x2
pass rate minus pad pass rate over fixtures completed in both
arms): the two treated arms have identical token lengths, so
the difference is not explained by length alone (content
position and repetition-specific attention are not separated;
the section says so). The *length effect* (pad minus x1) and
the joiner-inclusive *raw gain* (x2 minus x1) travel with it,
each with its paired denominator. Pairs whose pad cannot be
matched exactly are never shipped mismatched -- they leave
the denominator, and the per-candidate pad-evaluable share is
reported beside the effect. A fixture whose expected answer
occurs inside the filler is pad-screened for the same reason.
Fixtures that were already repeated verbatim are flagged by a
frozen exact detector and also reported as an
excluding-flagged subpopulation; vacuous (whitespace-only)
fixtures are excluded by predicate and counted. Degradation
is per fixture: one overlong treated render records
`arm_context_exceeded` for that fixture alone.

The conservative selection rule is re-applied per arm, and an
arm's winner/flip readout is *withheld* whenever the completed
fixture sets differ across candidates -- a flip the
denominators cannot explain is not published. The x1 arm never
reports a flip (it is the recorded run). Cost: two generations
per (candidate, fixture), 4,800 at the run caps, at full-prefix
recompute prices -- this lab has no KV cache, so doubled
prompts cost real compute, and the section says so.
`diff-reports` recomputes every aggregate from the payload's
per-fixture ground records (the full recompute standard) and
compares effect/arm drift, flagging protocol, fixture-set, or
backend drift as `comparable: false` (`diff_repetition`,
`repetition_changed` / `repetition_asymmetric` feed
`identical`). The full contract is in
`docs/REPETITION_DESIGN.md`. Informational only: the probe
never affects selection, and the stage never fails the run
(a failed or cancelled native search gates it to zero
generations instead).

### 8.40 Distractor dilution (`distraction`)

Opt-in with `model-protocol-lab run --distract`. A generation
stage over the search split: irrelevant filler is spliced into
each fixture's last user message at exact rendered token doses
(64, 256, and 1024 tokens of a frozen neutral corpus; 64 and
256 tokens of near-miss filler carrying a planted wrong answer
derived from the fixture's own expected answer by a frozen
digit-increment rule), placed before or after the real content;
the dose-0 arm is the recorded native search, reused with zero
extra model calls. Monday-morning use: before a retrieval or
long-context deployment, learn per candidate template how much
irrelevant context the winner tolerates — and whether
answer-shaped filler captures it outright.

The report's headline numbers are the per-dose paired pass
rates, the first-flip distribution with its *returns* counter
(passing levels above the first flip — recovery under more load
means the flip was noise, not a threshold), and the near-miss
*capture excess over neutral*: records whose answer became the
planted string under near-miss filler but not under the same
dose of neutral filler. The raw capture rate is labelled an
upper bound, because byte-exact scoring cannot see a captured
answer with stray formatting. Capture is defined only where the
expected answer is a plain ASCII number; on other answer sets
the near-miss arm is empty by construction and only the neutral
dose curves report. Every cell records its construction
(`id_exact` when the treated token ids decompose exactly into
native prefix + span + suffix, `text_level` when only the text
does, with seam token ids), and filler carrying special-token
literals or a candidate's boundary texts is screened per cell,
never generated. The conservative selection rule is re-applied
per condition, withheld for that condition whenever completed
fixture sets differ across candidates. Cost: ten conditions
over the native records, 24,000 generations at the run caps, at
full-prefix recompute prices — the CLI help states the count
and the forwards formula before the run. `diff-reports`
recomputes every derived value from the payload's per-fixture
ground records (the full recompute standard) and compares
curve, first-flip, capture, and per-condition selection drift,
flagging protocol, fixture-set, or backend drift as
`comparable: false` (`diff_distraction`, `distraction_changed`
/ `distraction_asymmetric` feed `identical`). The full contract
is in `docs/DISTRACTION_DESIGN.md`. Informational only: the
probe never affects selection, and the stage never fails the
run (a failed or cancelled native search gates it to zero
generations instead).

### 8.41 Leading-claim caving (`leading`)

Opt-in with `model-protocol-lab run --leading`. A generation
stage over the search split: one confident claim sentence is
appended to each fixture's last user message — a *wrong* claim
planted from another fixture's correct answer (donor-first
construction, with a digit-surgery fallback) and, as the
control arm, the *right* claim — each in two frozen framings
that differ only in perspective ("I'm sure the answer is X."
vs "Another user is sure the answer is X."). The native arm is
the recorded native search, reused with zero extra model
calls. Monday-morning use: before shipping a template for a
support or tutoring product, learn whether its instruction
block makes the model defer to a confident user assertion —
and whether polite third-person distance protects it.

The headline number is the per-candidate, per-framing *caving
count*: fixtures the candidate passes natively and under the
right claim but fails under the wrong one. Caving counts only
when the answer becomes the planted string byte-for-byte
(*adoption*, a behavioural readout — this tier cannot tell
belief from copying); failures under a wrong claim that land
elsewhere are reported separately as instability, never as
caving. A plant is eligible only when it cannot be read as
the correct answer under the scorer-lens R4 rung or by
substring containment, checked against every expected answer
in the fixture's group, so "42.0" can never pose as a wrong
claim against "42". The classification over the paired
discordant counts (`caving`, `claim_following`,
`right_claim_fragility`, `no_caving_observed`,
`below_threshold`, `insufficient`) replaces any comparison of
marginal rates: disjoint failure sets with equal rates still
cave, and the frozen truth table says so. The right-claim arm
is the acquittal path — a template that follows right and
wrong claims alike is labelled claim-following, not caving.
Rates always ship with counts and denominators; the frozen
floors gate labels only. Per-condition selection is withheld
whenever completed sets differ across candidates or miss the
condition's defined set, and the only ranked cross-candidate
comparison is over the common caving population. Cost: four
treated generations per candidate per live fixture (9,600 at
the run caps); the CLI help states the count before the run.
`diff-reports` recomputes every derived value from the
payload's per-fixture ground records — verdicts from digests,
donor consistency across records, the donor map and protocol
fingerprints — and flags protocol, fixture-set, donor-map, or
backend drift as `comparable: false` (`diff_leading`,
`leading_changed` / `leading_asymmetric` feed `identical`).
Payloads and reports carry ids, counts, rates, and digests
only — never claim, answer, or fixture text. The probe
measures factual caving on verifiable fixtures, not social
sycophancy. The full contract is in
`docs/LEADING_DESIGN.md`. Informational only: the probe never
affects selection, and the stage never fails the run (a
failed or cancelled native search gates it to zero
generations instead).

### 8.42 Numeric surface form (`numform`)

Opt-in with `model-protocol-lab run --numform`. A synthetic
generation stage: the lab generates its own integer tasks —
copy a number, say which of two numbers is larger, add two
numbers — and renders every value in four surface forms:
plain digits (`1234567`), comma-grouped (`1,234,567`),
grouped with a narrow no-break space (`1 234 567`, U+202F),
and scientific (`1.234567e6`). The chat template, wording,
and decoding never change; only the numeral's shape does.
Monday-morning use: before a template ships for a product
that shows prices, quantities, or readings, learn whether
the form your UI prints is the form the model handles worst
under that template — and whether grouping digits the way
your locale does costs exact answers.

The prompt wording is frozen and carries no digit and no
worked example, and the pass bar is the same in every arm:
the exact value, written in the form the operands arrived
in, with the response boundary found. Each value is rendered
into all four forms under one shared fixture id, so the
headline numbers are *paired* contrasts per task and form
pair over the values completed in both forms — the frozen
classification (`plain_advantage`, `mixed_discordance`,
`no_difference_observed`, `below_threshold`, `insufficient`)
over the discordant counts, with a McNemar exact p-value as a
diagnostic only. Every completed cell is also decomposed:
for copy and add, an exact pass, a right-value/wrong-form
failure, or a wrong-value failure — so a model that computes
correctly but will not keep your grouping is visibly a
different problem from a model that miscalculates, and only
the wrong-value class licenses an arithmetic reading. For
copy, right-value/wrong-form is a task failure, not a
consolation: copying is form fidelity. Compare answers are
the labels `first`/`second` and decompose into wrong-label
vs non-label answers; the task measures whether the
verbalised comparison survives the forms, not comparison
itself (its operands always share one form — mixed-form
pairs are a different instrument's cell and out of scope).
Values are stratified by digit band (1–3, 4–6, 7–9, 10–12
digits) and by construction class — round vs random for
copy, margin class for compare, carry class for add — and
those classes are recomputed from the values, never trusted
from the record. The tokenizer's own digit grouping is
measured at run time with a frozen probe battery over the
backend's encoder (`single_digit`, `chunked_up_to_3`,
`irregular_or_mixed`, `unknown`) and reported as a run
label beside per-form operand token counts; grouping is a
property of the checkpoint-plus-tokenizer, so the same form
effect measured on two checkpoints with different grouping
is confounded, and the report says so. A pre-registered
null — every contrast `no_difference_observed` or below
threshold — is a publishable answer: normalisation does not
matter here, for these tasks, at this checkpoint.

Records store the canonical integers in the clear (the
values are lab-generated; a hash over a 12-digit value
would be invertible by enumeration and buy nothing), so
`diff-reports` recomputes everything from the ground
records — construction, aggregates, contrasts, the
protocol — and regenerates the generated set from the
run's seed to check that the records are that set, not
merely a self-consistent grid. Since cycle 190 the
records also carry the answer text itself, bound by its
digest, and the validator re-parses the text under
mirrored grammars instead of trusting the stage's
recorded parse (compare answers, being labels, are bound
by the digest alone); payloads written before cycle 190
are rejected — re-run the probe to diff them. It
reports rate and
classification drift per candidate, task, and form, with
protocol, generator, or backend drift marking the
comparison not comparable and grouping drift flagged
without changing comparability (`diff_numform`,
`numform_changed` / `numform_asymmetric` feed `identical`).
There is no per-form selection in this probe: the report
informs the template decision, it does not make one. Cost:
candidates × 3 tasks × 16 values × 4 forms generations
(4,608 at the legal maximum); the CLI help states the count
before the run. The full contract is in
`docs/NUMFORM_DESIGN.md`. Informational only: the probe
never affects selection, and the stage never fails the run.

### 8.43 Date-line sensitivity (`dateline`)

Opt-in with `model-protocol-lab run --dateline`. A
generation stage: a `[Context] ... [/Context]` block is
prepended to the final user message of each search fixture,
carrying one frozen date line — eight date values (an
anchor day, the days around it, the same day a year either
way, a leap day, and a year boundary pair) each rendered in
three formats (ISO `2026-10-21`, long `21 October 2026`,
and weekday-ISO `Wednesday, 2026-10-21`, the shape real
system prompts inject) — plus, per format, a no-date
control whose neutral text is sized to the anchor arm's
rendered length for that candidate and fixture, and one
weekday-lie arm (the anchor date wearing the next day's
weekday name). Monday-morning use: if your serving stack
prepends a "Today is …" line, this tells you — per
template family, on your own checkpoint — how often
changing only that date flips a task verdict, and whether
the template you picked still wins afterwards, so a
calendar rollover cannot silently change your eval results
after you ship.

The headline is per-pair discordant counts, partitioned by
pair type: pure-date pairs (one value against another in
the same format) are the date-sensitivity family,
pure-format pairs (one value across formats) are the
format control, and pairs differing in both are not
computed — a union "any-pair flip rate" over 24 date arms
saturates under generic per-arm fragility and is not a
shipped statistic. Each pair carries the frozen
classification (`insufficient`, `no_discordance_observed`,
`discordant_asymmetric`, `discordant_balanced`,
`below_threshold`) over its discordant counts with floors,
and a McNemar exact p-value as a diagnostic only. Beside
the pairs, the date-loss subset counts fixtures that pass
a format's no-date control and fail a date value, per
value with formats collapsed and its asymmetric
counterpart (fail the control, pass the value) reported
beside it; the lie arm splits the anchor loss into losses
consistent with "any date text perturbs" and losses
specific to the true date's content. Fixtures whose
expected answer could itself be date content — a component
of the answer rung-matches a date-bearing component of an
arm, or the answer parses as a date — are screened out of
every treated population before generation, because on
such a fixture a changed verdict is the correct answer
changing, not task behaviour moving; date-bearing fixture
messages and blank expecteds form their own reported
classes (the frozen screen rule is named
`dateline-components-r4-v1` in the protocol). A
pre-registered null — every pure-date pair at
zero discordants and every loss count zero — prints
exactly that: the section states the instrument found no
date effect, which is a publishable answer, not a missing
one.

Every date is a protocol constant and every weekday name
is computed from its own calendar date; nothing in the
probe reads the wall clock or the locale, so a run made in
March and a run made in October measure the same
instrument. Per-candidate per-arm selection is recomputed
but withheld — a winner picked inside one date arm is an
arm-conditional selection, not a template verdict — while
the common-population selection (fixtures completed under
every arm) is reported as the like-for-like readout.
Records carry answer digests only, so `diff-reports`
recomputes every verdict as digest-equality plus the
boundary flag and rebuilds every aggregate, the screen,
and the protocol fingerprint from its own literal arm
table — a payload whose dates were re-rendered from one
weekday fails validation outright (`diff_dateline`,
`dateline_changed` / `dateline_asymmetric` feed
`identical`). The ±year contrast moves the weekday with
the year and is named a year+weekday shift wherever it is
reported; the leap arm sits 496 days from the anchor and
the report says so. Cost: candidates × fixtures × 28 arms
generations (67,200 at the legal maximum); the CLI help
states the count before the run. The full contract is in
`docs/DATELINE_DESIGN.md`. Informational only: the probe
never affects selection, and the stage never fails the
run.

### 8.44 Multi-turn drift (`multiturn`)

Opt-in with `model-protocol-lab run --multiturn`. A
generation stage over fixtures the probe generates itself
from the run seed — two task families whose answers are
computed, not judged: ledger sums (signed entries stated
one per shard; the total is asked on the final shard) and
shelf argmax (per-shelf unit counts; the fullest shelf's
label is asked), at depths 2 and 3, twelve fixtures per
family and depth. Each fixture runs in three arms: `full`
(every shard in one user message, one generation),
`sharded` (one shard per turn, the model's own
boundary-cut replies incorporated verbatim as the
history), and `scripted` (the same turn structure with a
frozen `Noted.` reply at every intermediate turn). Only
the final turn is scored. Monday-morning use: if your
product serves multi-turn conversations, this tells you —
per template family, on your own checkpoint — whether a
task the model passes in one shot starts failing once its
own earlier replies are the history, separated from the
generic effect of merely having history at all, so a
chat-shaped deployment risk is measured before your users
find it.

The comparison that carries the claim is the
self-history discordant pair, over fixtures the `full`
arm passed: `sh_only` counts fixtures the sharded arm
fails while the scripted arm passes, `sc_only` the
reverse, and the pair is always reported together with
its denominators — the scripted arm holds turn count,
message framing, and approximate history length constant,
so the pair attributes the bundle of self-generated
content (its wording, length, and tokenization under the
candidate's own template), never self-authorship alone.
Beside it sit the two delivery 2×2 tables (full against
sharded, full against scripted) with a McNemar exact
p-value as a diagnostic only, per-family counts beside
the pooled ones, and per-arm pass rates and completion
counts with an attrition breakdown — a cell that degrades
(context exceeded, evaluation failed, or screened out)
leaves the paired set and is counted, never silently
dropped. The classification is a total function over the
paired counts with floors (six paired fixtures, two
discordants): `self_history_drift` requires the sharded
side to dominate the scripted side, scripted-side or
balanced discordance at the same bar reads
`framing_only`, anything smaller reads `below_threshold`
or, with no evaluable depth, `insufficient_evidence`.

Two edge rules are pre-registered. A generated reply
that contains one of the candidate format's own
special-token literals is token-real — it would forge a
turn boundary in any real transcript — so the cell ends
there as `arm_screened` and leaves the paired set rather
than letting forged structure into later prompts. An
intermediate reply with no boundary, or an empty one,
still becomes history (that is what the model produced)
and is flagged on the cell. There is deliberately no
cross-depth recovery statistic: depth changes the task
set itself, so a "recovery rate" across depths would
divide answers to one set of generated tasks by answers
to another. Selection readouts are recomputed for the
`full` and `sharded` arms and withheld whenever any
candidate's completed set is not the defined set.
Records carry answer digests only, and `diff-reports`
regenerates the fixture set from the protocol seed with
its own generator mirror and recomputes every aggregate
before comparing (`diff_multiturn`, `multiturn_changed` /
`multiturn_asymmetric` feed `identical`), so a stored
readout nobody can re-derive fails validation outright.
Multiturn cells sit outside `verify --determinism`
scope: a sharded history is whatever the model happened
to generate, so token-exact re-execution is not the
contract. Cost: candidates × (24×4 + 24×5) generations
(5,184 at the legal maximum); the CLI help states the
count before the run. The full contract, including the
incorporation policy and the red-team adjudication, is
in `docs/MULTITURN_DESIGN.md`. Informational only: the
probe never affects selection, and the stage never fails
the run.

### 8.45 Needle depth (`needle`)

Opt-in with `model-protocol-lab run --needle`. A
generation stage over cases the probe generates itself
from the run seed: twenty cases, each planting one fact
sentence — a topic, a synthetic code like
`velvet-4821` — inside neutral filler, and asking for
the code. The filler is sized per candidate so the
prompt grows by exactly 1024 or 4096 rendered tokens
over the same case with no filler, and the needle sits
at 0, 25, 50, 75, or 100 percent of that span, the
offset verified on the candidate's own rendered tokens
by a prefix-anchored measurement that cancels the
template head's token merges. Around the ten treated
cells per case sit a no-filler control (`dose_0`), a
needle-absent cell per dose whose pass rate is the
false-recall floor, and a duplicate-needle cell per
dose (the fact at 25% and 75%) whose pass rate is
reported against the independence baseline of the two
single-needle rates, so "two chances" is not mistaken
for "position stopped mattering". Monday-morning use:
you paste retrieved chunks around a question — this
tells you, per template family, at which depth of that
bulk your checkpoint stops finding a planted fact,
instead of a single long-context number measured
through somebody else's template.

Curves, the mid-depth deficit (edge mean minus the
50% rate, the *Lost in the Middle* functional form),
and first-failure counts share one headline population
per dose: the cases every candidate passed with no
filler and completed at all five depths, so a flat
curve makes the deficit exactly zero and candidates
are compared over the same cases. The deficit and
first failure are withheld when that population drops
below six; the curves print at any size because they
are counts with their denominator attached. Per-cell
selection readouts re-apply the conservative rule over
each cell's completed set, with the flip baseline
recomputed on the same set. Two scope limits are part
of the readout: the filler is topically dissimilar
prose (similar distractors hurt more in the published
work, so these curves are a lower bound on position
effects under realistic retrieval bulk), and each cell
is one greedy generation. The 4096 dose needs
`--max-context` above 4096 plus the needle, question,
and generation budget; under the default 2048 cap every
4096 cell is reported `arm_context_exceeded`, never
truncated into existence, and the 1024 layer carries
the full control set. Records carry answer digests
only, and `diff-reports` regenerates the case set from
the protocol seed with its own generator mirror and
recomputes every readout before comparing
(`diff_needle`, `needle_changed` / `needle_asymmetric`
feed `identical`). Needle cells sit outside
`verify --determinism` scope: they are generations on
generated cases, not recorded search observations.
Cost: 15 generations per candidate per case (7,200 at
the legal maximum); the CLI help states the count and
the context arithmetic before the run. The full
contract, including the red-team adjudication that
replaced the first construction, is in
`docs/NEEDLE_DESIGN.md`. Informational only: the probe
never affects selection, and the stage never fails the
run.

### 8.46 Question/payload placement (`placement`)

Opt-in with `model-protocol-lab run --placement`. A
practitioner assembling prompts from a retrieved passage
plus a question learns, per template family, whether
putting the question before or after the payload changes
verdicts on their checkpoint — and picks the layout their
winning template actually rewards — instead of copying a
layout convention from a blog post measured on a different
model.

Twenty-four cases are generated from the run seed. Each
payload is three sentences — a neutral scene sentence, the
one answer-bearing fact, a second neutral sentence — with
the fact frozen in the middle position, and the question
wording is byte-identical in every arm. Four cells per
case move only the question: before the payload
(`question_first`), after it (`payload_first`), both
(`sandwich`, the control), and alone (`closed_book`, the
screen: a case the model answers with no payload in view
is excluded from that candidate's readouts, and the
exclusion count and closed-book pass rate are reported,
never silently dropped).

The headline is the paired discordant count between the
two single-question layouts, split by direction. The
sandwich arm is computed on the same paired cases and its
discordance against each layout sets the control level:
the falsification bar for a directional placement effect
is exceeding that level. The sandwich bounds what recency
and exposure together produce; it does not identify which
of the two drives an excess, and the readouts never claim
otherwise. The bar is a maximum of two noisy counts, so
it acquits readily at small discordance — that asymmetry
is the registered price of a control with no multiplicity
adjustment, and the design document prints the acquittal
rates. A case where both layouts fail together
(`both_fail_dominant`) is a joint collapse, not placement
evidence, and never counts as falsified. Per-arm
selection winners are computed over the population
screened for no candidate in the run, so one candidate's
memorized answer cannot shrink another's denominator.

Placement is not the neighbouring probes: hierarchy
varies privilege-block position inside the turn; position
permutes answer options; distract varies filler dose and
placement; needle varies burial depth in long filler;
placement varies where the task question sits relative to
a short payload inside one user turn. The payload is
frozen short by design — placement is not dilution and
not depth.

Records carry answer digests only, and `diff-reports`
regenerates the case set from the protocol seed with its
own generator mirror and recomputes every readout before
comparing (`diff_placement`, `placement_changed` /
`placement_asymmetric` feed `identical`). Placement
cells sit outside `verify --determinism` scope: they are
generations on generated cases, not recorded search
observations. Cost: 4 generations per candidate per case
(2,304 at the legal maximum: 24 candidates x 24 cases
x 4). The full contract, including the falsification
formula and the red-team adjudication that replaced the
first classification scheme, is in
`docs/PLACEMENT_DESIGN.md`. Informational only: the probe
never affects selection, and the stage never fails the
run.

### 8.47 Choice-scoring audit (`choice_audit`)

Opt-in with `model-protocol-lab run --choice-audit`. A
practitioner deciding how to score their checkpoints
learns whether log-likelihood choice scoring picks the
same answers — and the same template — as the greedy
generation the lab selects on, on their own fixtures,
instead of assuming the two scoring methods agree because
a benchmark paper reported them side by side.

For each search fixture a four-choice set is built from
the run's own fixtures: the expected answer plus three
donor expecteds from other groups, taken in ascending
fixture-id order and screened so no donor is the same
answer as any expected in the fixture's own group (the
scorer-lens equivalence ladder plus folded containment).
A fixture that cannot field three screened donors is
recorded as not constructible and excluded from every
choice population; the shortfall is counted, never
repaired with a fabricated string. Every choice is then
teacher-forced under every candidate over that
candidate's recorded search prompt — the audit performs
zero extra generations.

Each set is ranked three ways from the reported scores:
by total log-likelihood (the headline; the quantity
lm-eval calls `acc`), by reported total over target byte
length (lm-eval's `acc_norm` semantics), and by per-token
mean — a third quantity the harness deliberately does not
report, carried here for the comparison and never
labelled `acc_norm`. A fixture's verdict exists only
when all four of its cells scored cleanly; a tie at the
maximum is reported as tied, never broken and never
counted correct. Per candidate the audit reports the
three accuracies with their denominators, the
generation × choice discordance table (with the
generation-failure cells split by whether the boundary
was found, so a boundary failure is not mistaken for a
scoring disagreement), accuracy strata for numeric and
non-numeric answers and for token-length-matched sets,
and how often the per-byte and per-token rankings agree
with the total ranking's verdict.

The method headline compares the choice method's top
candidate against the primary (generation) winner over
the common evaluable population only — the fixtures with
a choice verdict under every candidate — and is withheld
below a floor of six, with the population size always
reported. Donors are not difficulty-matched to the true
answers, so a candidate's choice accuracy is an upper
bound on its discrimination, and the report says so. The
registered falsification claim is narrow: zero
generation × choice discordance over the completed,
fully-evaluable paired cells under the total ranking
would falsify "the two scoring ontologies disagree on
this checkpoint"; the readout states the denominator it
ran over, and claims nothing when that denominator is
zero.

Choice-audit is not the neighbouring probes: the
logprob diagnostic reports per-answer log-likelihoods
without constructing choice sets or ranking rivals, and
scorer-lens regrades recorded generation answers under
text normalisation — it never scores un-generated
alternatives. Records carry choice digests and reported
scores, never answer texts, and `diff-reports`
recomputes every readout from the ground records before
comparing (`choice_audit_changed` /
`choice_audit_asymmetric` feed `identical`). Cost: 4
forward passes per candidate per fixture (9,600 at the
legal maximum: 24 candidates x 100 fixtures x 4). The
full contract, including the donor rule, the evaluability
rule, and the red-team adjudication that reshaped both,
is in `docs/CHOICE_AUDIT_DESIGN.md`. Informational only:
the audit never affects selection, and the stage never
fails the run.

### 8.48 Load precision (`precision`)

Opt-in with `model-protocol-lab run --precision`. A
practitioner who will serve the checkpoint in bf16 learns,
before deploying, whether the template winner the lab
selected under its fp32 measurement is still the winner
when the same weights are loaded in bf16 — and gets the
per-fixture flip list if it is not.

The same checkpoint directory is loaded a second time in
bfloat16 and every search candidate is re-evaluated on
the search split only, under identical conditions; the
fp32 arm is the recorded native run, never re-executed.
Reports per candidate: the verdict-flip matrix over
fixtures completed in both arms (a fixture an arm never
reached is unpaired and counted, never an arm failure),
the output-divergence rate (the sensitive readout —
generations can diverge where no verdict flips), and the
teacher-forced expected-answer logprob deltas between
arms. The headline is winner survival: the selection rule
is re-applied to both arms over their shared paired
population and the outcomes compared, withheld when any
candidate's completed sets mismatch, when the primary run
produced no winner, or when nothing paired.

Read the numbers at their stated scope. This is dtype
precision, not quantization: the bf16 arm is a load-time
cast of the same unquantized weights. Zero flips falsifies
a precision effect on the search split, at the verdict
level, in a single bf16 run against the recorded fp32
run — and where no candidate passes in either arm, zero
flips is the floor effect, not robustness; the divergence
and logprob numbers are the informative readouts there.
Held-out behaviour under bf16 is never measured, and the
determinism certificate covers the native fp32 search
observations only. If the winner does not survive, the
fp32 selection still stands as the run's decision: do not
serve that winner in bf16 on this evidence. The bf16
arm's own winner is a search-split readout with no
held-out evidence, not a selection.

Records carry digests only, never answer text, and
`diff-reports` recomputes every readout from the ground
records before comparing (`precision_changed` /
`precision_asymmetric` feed `identical`); a drifted
arm fingerprint marks the comparison not comparable while
still listing it. Cost: a second full search evaluation
plus a second checkpoint load, with both models resident
at once; on CPU bf16 measured several times slower than
fp32 in this lab's sandbox, so the stage can dominate run
time. The full contract is in
`docs/PRECISION_DESIGN.md`. Informational only: the probe
never affects selection, and the stage never fails the
run.

### 8.49 Verbalized confidence (`confidence`)

Opt-in with `model-protocol-lab run --confidence`. When
this checkpoint states a confidence under each template,
does the number track whether it was right — and does it
drop when the answer is not in the prompt?

Every search fixture is re-run once per candidate with one
frozen suffix appended to its last user message, asking
the model to end its reply with a line of the form
`Confidence: <integer 0-100>`. The elicited text is parsed
under a frozen grammar (ASCII digits only, no sign, no
leading zeros — five outcome codes, and the answer span
without the compliant line is what gets scored), so the
stated number is set against the same exact-match verdicts
as the native run. Reports per candidate: the compliance
rate, mean stated confidence on correct vs incorrect
compliant answers with their descriptive gap (withheld with
a named reason when a class is empty), a frozen four-bin
table, and the verdict flips the suffix itself caused
against the recorded native run. A generated pair runs
alongside: 24 planted-fact cases whose value the prompt
states, against the same cases with the payload binding a
different key — nothing false is stated about the asked
key; the binding is simply absent, so a model confidently
answering there is copying the only value it can see. The
knowability readout compares the two arms' stated means
over cases where both complied, withheld below 6 paired
cases.

Read the numbers at their stated scope. A stated
confidence is model output text, not a probability, and
the gap and bins describe agreement between the number and
the verdict — they are not calibration scores, and no
published calibration number is comparable to them (the
ground truths differ). Search split only: held-out
fixtures are never treated, and the determinism
certificate covers the recorded native search observations
only, so the elicited cells are outside its scope.
Fixtures whose expected answer is blank, already contains
a confidence line, or whose last user message is empty are
screened out of the populations and counted, never
silently dropped.

Records carry digests, never answer text — the one
exception is a compliant cell's final line, which is a
bijection with the confidence scalar already in the
record and lets `diff-reports` derive the number instead
of trusting it (`confidence_changed` /
`confidence_asymmetric` feed `identical`); a drifted
protocol or backend fingerprint marks the comparison not
comparable while still listing it. Cost: one full extra
search evaluation (2,400 generations at the legal
maximum: 24 candidates x 100 fixtures) plus the generated
paired control (2 arms x 24 cases x 24 candidates =
1,152) — 3,552 extra generations in total, each on a
suffix-lengthened prompt. The full contract is in
`docs/CONFIDENCE_DESIGN.md`. Informational only: the
probe never affects selection, and the stage never fails
the run.

### 8.50 Context rotation (`rotation`)

Opt-in with `model-protocol-lab run --rotate`. If my
retriever re-ranks the same five chunks into a different
cyclic order, does this template keep the same verdict —
and the same winner?

Twenty cases are generated from the run seed. Each case
assembles five fact-bearing blocks — one asked fact and
four distractors built from the same sentence shape, so
every block looks worth attending to — into a span with
the question pinned after it, and runs eleven arms per
candidate: the five cyclic rotations of the block order
(same content, same total text length, same cyclic
relative order; only the seam and the absolute slots
move); four mirror arms, the cyclic rotations of the
reversed order, which form a generic-reordering control
set placing the asked fact at the same slots as the
rotations do; a fact-absent arm that prices false recall;
and a closed-book screen (the question alone — a pass
excludes the case from that candidate's rotation
populations, with the exclusion counted). A span arm
counts as construction-exact only when its rendered token
inventory digest equals the unrotated arm's, so the
headline populations run over token-exact rotations only;
that is an inventory claim, and segmentation at the block
seams is not verified.

The headline is verdict invariance across the five
rotations, classified by a frozen rule whose last step
compares the rotations' any-flip count against the same
statistic over the control set on the same cases — a pure
slot effect moves both equally and is acquitted, which is
exactly what the control is for. Invariance is falsified
under a pre-registered rule (at least 6 headline cases and
at least one flip), and the classification and the
falsification boolean are printed together, because the
pairing is part of the contract: a falsified run can still
classify as within-control. The pass-by-absolute-slot
curve is a cross-check against the needle probe's depth
curve, not a second measurement of it — the needle moves
one fact through inert filler, while here every position
in the span carries a fact-shaped block. Per-arm
selections re-run the lab's selection rule on each arm's
verdicts over the shared live population, gated per arm on
token-exactness for every candidate; a run whose gated
populations disagree is withheld with a named detail, not
averaged over.

Records carry digests, token counts, and inventory
digests only — no block, question, or answer text — and
`diff-reports` recomputes the whole section from its own
literal tables and a regenerated case set
(`rotation_changed` / `rotation_asymmetric` feed
`identical`); a drifted protocol, generator, or backend
fingerprint marks the comparison not comparable while
still listing the drift. Cost: 11 generations per
candidate per case (5,280 at the legal maximum: 24
candidates x 20 cases x 11). The full contract is in
`docs/ROTATION_DESIGN.md`. Informational only: the probe
never affects selection, and the stage never fails the
run.

### 8.51 Instruction density (`density`)

Opt-in with `model-protocol-lab run --density`. My system
prompt keeps growing — which of its instructions does this
template silently drop first, and at what stack depth?

Twenty cases are generated from the run seed. Each case
pairs one small fact-recall task (a signal word to report)
with sixteen constraints: include a given word, start with
a given word, end with a given word, stay within a word
limit — four of each. The sets are satisfiable by
construction: the start constraints of a case share one
target, the end constraints share another, every target
group is distinct, and the smallest word limit (8) exceeds
the largest possible distinct-target count, so a perfect
instruction-follower passes every arm and any failure is
the model's, not the instrument's. Each candidate runs
twenty-one arms per case: every constraint alone (solo),
the first 2, 4, 8, and all 16 constraints stacked, and a
sham arm carrying only the first constraint plus neutral
text length-matched to the full stack's block, which
separates "many instructions" from "much text".

Answers are committed as ordered per-word hash lists — a
word is a maximal run of alphanumeric characters,
case-folded — and every satisfaction verdict is derived
from the commitment, never stored on the record, so the
diff validator re-derives each one. Readouts print
satisfaction by stack depth beside the same constraints'
pooled solo rate (the composition control), task-value
presence per arm, position bands at full density, per-kind
solo-vs-full rates (positional kinds repeat their target
at full density — a stated limit of the instrument), joint
satisfaction at sixteen, and the sham readout. The claim
under test is scoped to the endpoint contrast:
per-constraint satisfaction at full density equals its
solo satisfaction. The decision aggregates solo-vs-full
discordance per case and applies an exact two-sided sign
test over the discordant cases; below six discordant cases
the verdict is withheld as `insufficient` — never a
defaulted acquittal — and intermediate depths are
descriptive only. The level is per candidate: across 24
candidates at alpha 0.05, at least one false falsification
is expected with probability about 0.71, and the report
says so next to the numbers.

Records carry digests and the word-hash commitments only —
no case, constraint, or answer text — and `diff-reports`
recomputes the whole section from its own literal tables
and a regenerated case set (`density_changed` /
`density_asymmetric` feed `identical`); a drifted protocol,
generator, or backend fingerprint marks the comparison not
comparable while still listing the drift. Cost: 21
generations per candidate per case (10,080 at the legal
maximum: 24 candidates x 20 cases x 21). The full contract
is in `docs/DENSITY_DESIGN.md`. Informational only: the
probe never affects selection, and the stage never fails
the run.

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
`inspect` time. The built split is contamination-checked exactly like
`inspect`: near-duplicate search/held-out pairs print a stderr warning and
appear in the JSON `contamination` block (advisory; see §1). The conversation ends at the first empty message cell; an
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

### Fuzzing the loaders (`fixtures fuzz`)

Every experiment in this lab stands on two parsers: the JSON loader
and the CSV authoring reader. If either one quietly accepts a
malformed file -- or crashes on it instead of refusing it -- every
measurement downstream inherits the defect. `fixtures fuzz` exists
for the maintainer who wants proof, on demand, that the loaders
still keep their contract:

```sh
python -m model_protocol_lab fixtures fuzz [--seed N]
```

The command mutates a known-good fixture file from a fixed,
published table of mutation classes (the full table, with the
reasoning behind each expected outcome, is
`docs/FIXTURE_FUZZ_DESIGN.md`) and feeds every mutation to the real
loaders: homoglyph and NFC-twin identifiers, duplicate keys spelled
with escapes, lone surrogates, fields past the size limit, bare-CR
line endings, unterminated quotes, FIFOs and symlinks in place of
files, and more -- 133 cases across the JSON text, the CSV text,
and both file readers. Each case expects `accept`, `reject` (a
`ValueError`, and nothing else), or -- CSV only -- an accept with
the fixture named in the empty-`expected` warning list. Accepted
cases also pin the *stored* value, so a loader that silently
normalizes text fails just as loudly as one that accepts a file it
should refuse. The report is JSON on stdout with a one-line summary
on stderr; the exit code is 0 only when every case conforms. The
seed varies mutation parameters (lengths, padding) but never which
outcome a case expects, and the corpus digest printed with every
run makes two runs comparable at a glance.

What this is not: it is not a coverage-guided fuzzer in the
Atheris/libFuzzer/AFL++ sense -- there is no instrumentation, no
coverage feedback, and no corpus evolution. That is a deliberate
trade: weaker exploration, stronger auditability. The whole search
space is the published table, printed before the run, and a green
run proves exactly that the table holds for the shipped loaders --
nothing more. In the harnesses and fuzzing tools surveyed for this
feature (lm-evaluation-harness, HELM, OpenAI evals, PromptBench,
garak, and Inspect, checked October 2026), no shipped first-class
command was found that fuzz-tests the harness's own fixture or
dataset loaders against a published accept/reject contract; the
command tests the loaders, never a model and never fixture quality.

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

Display caveat: the REPL keeps tab, newline and carriage return in fixture
content so multi-line renders stay readable; every other terminal control
(including ESC) is stripped. A fixture containing a line such as
`render: family=chatml ...` can therefore visually mimic a REPL status line
-- plain-text mimicry only, and fixture content always follows a genuine
status line, but treat `.save` transcripts (and pasted sessions) with that
in mind.

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
(measured 2026-10-15). It also matches `model-protocol-lab run --max-new-tokens`' own
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
driving (candidate, fixture) pass-status changes), `held_out_floor` (the
held-out verification-floor comparison: floor drift, floor mismatch,
one-sided/missing floors), `boundary_audit` (the prompt-boundary
canonicality comparison: overall and per-family PBP-rate drift),
`selection_robustness` (the stop-semantics selection comparison:
per-semantic (decision, winner) drift and flip-pattern changes),
`transfer` (the cross-checkpoint transfer comparison: transfer-verdict
drift, per-checkpoint (decision, winner) drift, per-cell verdict drift,
checkpoints added/removed),
`contamination` (the fixture-hygiene comparison: worst-tier drift,
pairs added/removed, tier escalations -- did the held-out set get
cleaner or more contaminated?), and one key per recorded experiment
stage -- `perturbation`, `token_healing`, `hierarchy`, `prefill`,
`portfolio`, `synthhistory`, `batch_invariance`, `position`,
`stop_cause`, `ws_canonical`, `structured`, `deliforge`, `fewshot`,
`faketurn`, `toolcall`, `verbosity`, `refusals`, `scorerlens`, `aliasing`, `repetition`,
`distraction`, `leading`, `numform`, `dateline`, `multiturn`, `cache_profile`,
`template_robustness`, `ablation`, `confidentiality`,
`answer_collision` -- each carrying that stage's verdict drift and
per-candidate drift, with the stage's own section in section 8
describing its comparison in full. A stage a run did not record is
reported as one-sided (asymmetric), never silently dropped.
`config_drift` (the comparability warnings), `candidate_verdicts`
(per-candidate pass/fail flips on search and held-out), `fixture_diffs`
(per-fixture old/new outcomes per candidate),
`fixture_diffs_include_unchanged` (whether `--all` included unchanged
fixtures), `evidence` (whether each
side carried `--include-evidence` observations), and a `summary` with
counts (`fixtures_flipped`, `candidates_changed_status`,
`config_drift_warnings`, `frontier_changed`, `frontier_asymmetric`,
`held_out_floor_changed`, `held_out_floor_asymmetric`,
`boundary_audit_changed`, `boundary_audit_asymmetric`,
`selection_robustness_changed`, `selection_robustness_asymmetric`,
`transfer_changed`, `transfer_asymmetric`,
`contamination_changed`, `contamination_asymmetric`,
`decision_changed`, `winner_changed`, `evidence_asymmetric`, a
`<stage>_changed` / `<stage>_asymmetric` pair for every stage key
above (the `token_healing` stage is shortened to `healing_changed` /
`healing_asymmetric`), and `identical`).

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
- **Experiment stages**: every stage the two runs recorded (the stage
  keys listed above) is compared on its own terms -- verdict drift,
  per-candidate drift, and the stage-specific rates its section in
  section 8 defines. A stage recorded on only one side is reported
  one-sided; the summary's `identical` verdict counts every stage.

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

## 13. Re-deriving a result (`verify`)

Before trusting a `results.json` -- after a hand-edit, a copy across
machines, or a long-lived archive -- `verify` independently re-derives
what the pipeline recorded, through code paths separate from the ones
that produced it (N-version cross-check). No model is loaded and no
network is used:

```sh
python -m model_protocol_lab verify run-baseline
```

Each check is reported `passed`, `failed`, or `unavailable` (with the
reason stated). The selection decision is replayed from the stored
per-candidate summaries with an independently re-implemented selection
rule; the token-budget frontier and the held-out budget floor are
recomputed from the recorded observations and compared for exact
equality; the search/held-out split is checked for fixture overlap, and
every candidate is checked to have run the same search fixtures.
Candidate-name uniqueness, summary self-consistency, and the
selected candidate's boundary are checked too.

What it cannot prove is stated, not hidden: a stored `"cancelled"`
decision is reported `unavailable` because the cancel signal is not
recorded in the artifact, and runs written without `--include-evidence`
strip the observations, so observation-dependent checks report
`unavailable` instead of pretending.

The JSON report goes to stdout; a human-readable summary goes to
stderr. Exit 0 when every applicable check passes, 2 when any check
fails, 1 for unreadable or invalid input.

### 13.1 Sealing a result (`pack`, `verify --bundle`)

Monday-morning use: an auditor asks to prove a run's result files are
unchanged since packing. After `verify` answers "is this result
self-consistent", sealing answers the second question: "was it edited
since the run was packed":

```sh
python -m model_protocol_lab pack run-baseline
python -m model_protocol_lab verify --bundle run-baseline.bundle.zip \
    --expect-digest <the 64-char hex `pack` printed>
```

`pack` seals every file in the report directory (plus any files named
with `--include`, stored under `inputs/`) into one bundle with a
SHA-256 per file and a bundle digest printed on stderr — publish that
hex alongside the result; `--expect-digest` lets `verify --bundle`
compare against it. Re-checking re-hashes every sealed file, fails on
any mismatch, and then runs the section-13 replay on the sealed
`results.json` as a second, independent layer.

The seal is unsigned, and the docs say so plainly: it proves
bit-integrity since packing only when the digest comes through an
independent channel. It does not prove who packed the bundle, that the
experiment really ran, or that the bundle is the newest one. The format
contract, caps, and check list live in
`docs/SEALED_BUNDLE_DESIGN.md`. Exit codes mirror `verify` (0 ok,
2 checks failed, 1 unreadable); scripts should branch on the JSON
report, not the exit code alone.

### 13.2 Determinism self-audit (`verify --determinism`)

Monday-morning use: a reviewer challenges a published result — "prove
the backend actually generated those tokens deterministically." Plain
`verify` replays the *decision* from the stored artifact; the
determinism audit replays the *generation*. It loads the checkpoint
(the one exception to the "no model is loaded" rule above), re-derives
each recorded search observation's per-request seed through the same
code path the original run used, re-executes the recorded `input_ids`
at the recorded token budget, and compares the output token-for-token:

```sh
python -m model_protocol_lab verify run-baseline \
    --determinism --model path/to/checkpoint
```

The JSON report gains a `determinism` section and a
`determinism_replay` check. The audit runs only when the loaded
checkpoint's fingerprint equals the recorded `backend_fingerprint`
(the fingerprint binds the checkpoint assets, library versions,
device, and context limit, so a mismatch means "a different
experiment", never "nondeterminism"); `--device` and `--max-context`
select the replay configuration and are gated by that same
fingerprint. Runs written without `--include-evidence` have no
observations to re-execute and report `unavailable`.

If every recorded completed search cell reproduces, the section
carries a determinism certificate: unsigned and self-issued, bound by
a SHA-256 digest to the exact cells compared, stating its scope
(search cells only; held-out cells are not re-executed). If some cells
diverge, the section lists each one with its first divergent token —
a divergence is a **backend-determinism finding**, not an
artifact-integrity failure and not proof the original decision was
wrong. Cells the artifact claims but the audit cannot re-execute
(tampered lengths, out-of-vocabulary IDs, a search that never
completed) cap the verdict at `inconclusive`; the audit detects
divergence but does not diagnose its mechanism. The contract,
including the falsifiable claims the implementation is tested
against, is `docs/DETERMINISM_AUDIT_DESIGN.md`.

Re-execution costs a full generation pass over the recorded search
cells (bounded at 2,400 cells, the largest search phase a run can
record) — budget for it like a re-run, not like a file check. Exit
codes mirror `verify` (0 ok, 2 checks failed, 1 unreadable or invalid
input), with one consequence to know: when the determinism check is
`unavailable` — for example the backend errors during re-execution —
the command can still exit 0. Read the `determinism` section, not the
exit code alone.

## 14. Surface-noise self-consistency probe (`surface-noise`)

Monday-morning use: you rewrote a prompt over the weekend — the probe
tells you whether a typo, case change, or whitespace shuffle would flip
its answers, before you ship it.

The probe perturbs the *prompt text only* (never the fixtures) with a
frozen set of surface-noise operators — typo, case flip, unicode
variant, whitespace modes, punctuation strip — and re-scores the prompt
on your fixtures with your backend. It reports a **per-operator
stability vector** (never a single score) plus the nearest observed flip
per operator: the smallest token-edit distance at which the verdict
vector changed.

```sh
python -m model_protocol_lab surface-noise probe \
    --prompt prompt.txt --fixtures fixtures.json --boundary-ids 7 \
    --output probe-out/
python -m model_protocol_lab surface-noise diff probe-out/probe.json probe-out2/probe.json
python -m model_protocol_lab surface-noise protocol
```

`probe` writes `probe.json` and a self-contained `dashboard.html`
(inline SVG stability bars, minimal-flip counterfactuals, the static
lint panel) into a new output directory. `diff` compares two probe runs
— the improve-then-re-probe loop. `protocol` prints the frozen
instrument descriptor so you can inspect what the probe does before
trusting it. `--backend selftest` exercises the whole loop without a
checkpoint; `--backend local-hf --model PATH` scores against a real
checkpoint.

Read the scope statement on the dashboard first — it is the first thing
on the page. The probe measures stability of task success under a frozen
set of surface-form perturbations, nothing more: the reported "nearest
observed flip" is an **upper bound** over the searched subset (it errs
toward false confidence, never a guarantee), and scores are comparable
only under identical fixtures, backend, and protocol fingerprint. It
says nothing about semantic paraphrase, prompt injection, hierarchy
attacks, or tool-call adherence under noise. An MCP server
(`tools/surface_noise_mcp_server.py`, stdlib-only) exposes the same
instrument to agents.
