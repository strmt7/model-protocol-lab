# Frequently asked questions

Troubleshooting and interpretation questions, answered from the documented
behavior. Each answer points at the authoritative section in
[docs/USAGE.md](USAGE.md).

## How do I find the right `--boundary-ids`?

From the checkpoint's actual tokenizer, never from another model. Run
`doctor --model DIR --boundary-ids ...`: it reports the tokenizer's declared
`added_tokens` with their IDs, which is where boundary markers usually live,
and verifies each ID against the vocabulary (see USAGE §2, §4). Then confirm
the tokenization is stable: decode the IDs and re-encode the text — if the
IDs change, the boundary text has multiple tokenizations and the exact
token-ID match can miss (USAGE §4).

## `run` fails with `local_context_limit_exceeds_declared_window` — what now?

Your `--max-context` exceeds the checkpoint's declared window. Stock GPT-2
checkpoints declare `n_positions=1024`, below the CLI default of 2048: pass
`--max-context 1024` (or less). After a clean `doctor`, the suggested next
command already fills in the right value (USAGE §2, §3).

## `run` fails with `context_limit_exceeded` — what now?

Prompt tokens plus `--max-new-tokens` exceeded the checkpoint's declared
context window: the backend refused generation rather than silently
truncating. Retry with a smaller `--max-new-tokens`. The `doctor`
suggestion already sizes it so the longest fixture prompt fits the window;
if you raised it afterwards, that is the cause. The report's decision
banner names the failure in a "Failure diagnosis" line, and each candidate
summary in `results.json` carries the sanitized `failure_code`.

## `boundary_found=false` but `boundary_text_match=true` — is my run broken?

Not broken — this is the tokenization-mismatch diagnostic doing its job. The
model emitted the boundary *text* with a different tokenization than your
configured IDs (one string can be many token sequences). The pass criterion
is unchanged (token-ID match only); treat the observation as a prompt to pick
boundary IDs whose tokenization is stable in this checkpoint's tokenizer, and
verify with `doctor --boundary-ids` (USAGE §4).

## Why does `inspect` warn about empty `expected` answers?

An empty expected answer passes vacuously: any output that emits only the
boundary token satisfies it. The warning names the fixtures so the
measurement gap is visible; consider giving them real expected answers or
removing them (USAGE §1).

## What do `ambiguous` and `no_candidate` mean? Can I use held-out to break ties?

`ambiguous` means several candidates passed every search fixture;
`no_candidate` means none did. The lab abstains in both cases by design, and
the held-out set is never used to break ties — it only validates the single
search winner. This is the conservative selection rule (USAGE §5); there is
no flag to change it.

## The stop matrix disagrees with real vLLM / HF — which is right?

The matrix cells are emulations with documented approximations, not
reproductions. The HF cell mirrors `StopStringCriteria`'s matching rule
including its byte-space tables for byte-level tokenizers, but token texts
for unmappable tokens fall back to isolated decodes, and the truncation is
post-hoc (exact only for greedy decoding). The vLLM cell approximates
incremental `DecodeStream` detokenization with per-prefix batch decodes,
strips the stop string from the returned text by default
(`include_stop_str_in_output=False`, the upstream default -- HF always
keeps it), and models `min_tokens` gating (default 0). Disagreements are
expected at these edges; the matrix answers "which semantics *could* flip
this verdict", not "what vLLM would print" (USAGE §8.2).

## The lab generates straight through EOS — how do I match that in HF / vLLM?

The lab's decoding discipline is "exactly N tokens, no implicit EOS stop":
generation runs to the token cap even when EOS appears. Both competitors
stop on EOS by default -- HF's `generate()` auto-adds `EosTokenCriteria`,
and vLLM defaults to `ignore_eos=False` -- so neither matches the lab out
of the box. To make each backend behave like the lab:

- HF: pass `eos_token_id=None` to `generate()` (disables the auto-added
  `EosTokenCriteria`; no other stop source is on unless you pass stop
  strings).
- vLLM: pass `ignore_eos=True`, `stop=[]`, `stop_token_ids=[]` and
  `min_tokens=0` (every stop source explicitly disabled).

Sources: transformers `main` `stopping_criteria.py`
(https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/generation/stopping_criteria.py),
vLLM `main` `v1/engine/detokenizer.py`
(https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/engine/detokenizer.py),
vLLM PR #22014 (`min_tokens` gating). See docs/RESEARCH_LOG.md "R2" for the
full divergence write-up.

## What happens if I interrupt a run with Ctrl+C?

Cancellation is cooperative: the run stops after the current backend
operation, keeps a partial report (exit 130), and the output directory stays
claimed so a retry needs a fresh path. Model loading itself may not be
promptly interruptible (USAGE §5, §6).

## Can I share the report? What's in it?

`results.json` excludes raw observations unless you pass
`--include-evidence`; the HTML view never embeds raw prompts or generations.
With `--include-evidence`, token IDs can reveal private prompts — inspect the
report before sharing. Summary metadata, candidate names, format markers and
fingerprints are always present (USAGE §6, README "Privacy").

## Can I compare candidates that need different boundary IDs?

Yes: `run --boundary-map NAME=ID,ID` (repeatable) overrides `--boundary-ids`
per candidate, so each format hypothesis is measured with its own
end-of-turn marker. The selection rule is unchanged -- one candidate must
pass every search fixture under its own boundary. See USAGE §4.

## The sensitivity sweep is slow / reports failed depths — normal?

Yes. The sweep re-runs the whole search at every depth in
1..`--max-drop-bytes`, so it multiplies compute by that number; it is opt-in
for exactly this reason. A depth that would split a UTF-8 codepoint records
a failed decision for that depth instead of silently skipping it (USAGE
§8.5).

## `doctor` says ok but `run` fails — what did I miss?

`doctor` validates assets, config, tokenizer structure and fixtures without
loading weights. It does not check RAM/VRAM, CUDA availability beyond the
flag, or that the checkpoint's weights match its tokenizer's expectations
beyond the manifest checks. When the decision is `failed`, the report's
decision banner carries a "Failure diagnosis" line naming the sanitized
failure code (e.g. `context_limit_exceeded`) with what to do about it --
read that before anything else. `local_backend_initialization_failed` means
the checkpoint failed the adapter's strict loading checks (USAGE §3).

## How do I read the token-budget frontier?

It recomputes the search decision at every smaller token budget from the
recorded outputs (exact under greedy decoding, no extra model calls). "Stable
across 1–N" means the verdict never depended on the cap you chose; flip
ranges like "budgets 1–9: no_candidate" mean the winner only emerges once it
gets enough tokens — the verdict is budget-contingent and a larger
`--max-new-tokens` (or a cheaper winner) deserves a thought (USAGE §8.6).

## Where is the evidence that any of this was actually verified?

[VALIDATION.md](../VALIDATION.md): exact commands, counts, dates and
environment for every verification round — including what is still *not*
established.
