# Research: template portability analysis without doubling compute (2026-10-15)

Research subagent report. Question: is there a principled,
constraint-compatible design for template portability analysis that does
NOT re-run the experiment on a second checkpoint (which doubles compute)
and does NOT weaken the per-checkpoint philosophy?

## Short answer

Yes — but only as a **static exclusion screen**, not as a portability
measurement. There is no established methodology anywhere (lm-eval, HELM,
2025–2026 literature) for "template X works on models A, B, C" that the
lab could adopt, and the honest limits of a static check must be stated
up front: it can *rule out* formats on a second checkpoint, it can never
*predict* that the winning format would win there. That asymmetry is the
whole design.

## 1. What the ecosystem does (and does not do)

**lm-eval (lm-evaluation-harness).** Renders each checkpoint's *own* chat
template via `tokenizer.apply_chat_template`; task prompts are fixed Jinja
templates in YAML configs. There is no template-portability concept: one
template per task, one rendering per model, and per-model scores are
compared. Recent movement toward multi-prompt evaluation: PR #2520 added
a `--examples` argument as "the first step towards efficient multi-prompt
evaluation with PromptEval"
(source: EleutherAI/lm-evaluation-harness v0.4.9 release notes,
https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.9).

**PromptEval (Polo et al., NeurIPS 2024, arXiv 2405.17202).** Estimates the
*performance distribution* across many prompt variants, borrowing strength
across prompts and examples; can estimate quantiles over 100 prompt
templates on MMLU with a budget equivalent to ~2 single-prompt evaluations
(source: https://arxiv.org/abs/2405.17202). Also used for best-prompt
identification via best-arm identification (Shi et al., 2024). This is the
closest thing to "which prompt works" methodology — but it (a) measures
the distribution *per model*, never portability of one template across
models, and (b) is statistical estimation, fundamentally incompatible with
this lab's exact-token-sequence / no-estimation discipline
(`not_estimated`). Not adoptable.

**Prompt-sensitivity literature.** Sclar et al. 2024 ("Does Prompt
Formatting Have Any Impact on LLM Performance?", arXiv 2411.10541):
matched-pairs t-tests between best/worst formats per model and a CMD
dispersion metric — per-model format sensitivity, never cross-checkpoint
portability of a specific template. The 2025 follow-up "Flaw or Artifact?
Rethinking Prompt Sensitivity in Evaluating LLMs" (arXiv 2509.01790) shows
heuristic evaluation *exaggerates* sensitivity (Gemma-2.0 ARC-Challenge
0.25–0.90 under heuristics vs. far tighter under LLM-as-judge; model
rank correlation 0.31 → 0.92) — a cautionary note: any "portability"
claim built on accuracy deltas is partly measuring the *evaluator*, not
the template. The lab's RESEARCH_LOG (Cycle 3) already surveyed
arXiv 2604.18389, 2608.18539, 2509.13680 and found every paper measures
*semantic or surface* prompt variants; none does token-level
template×checkpoint compatibility. The niche is still empty.

**HELM / lighteval / Unitxt.** Standardized prompts, task accuracy;
no template-portability axis at all.

**Conclusion for Q1:** no established "template X works on models A, B, C"
methodology exists. Every existing approach either (a) renders each model's
own template (sidestepping the question), (b) estimates per-model prompt
distributions (PromptEval — estimation, not exactness), or (c) measures
format sensitivity per model (not portability). The lab would be defining
the methodology, not adopting one.

## 2. The cheap static signal (the actual proposal)

**Idea: template↔tokenizer compatibility matrix, computed from the
tokenizer alone — no model forward passes.**

For each candidate `Format`, the lab already extracts a special-token
inventory (`fixtures._extract_special_tokens`: token-shaped substrings of
bos, user_prefix, assistant_prefix, turn_end, per-role terminators, open
suffix — e.g. `<|im_start|>`, `<|eot_id|>`, `<end_of_turn>`, `[INST]`).
Against a second checkpoint's tokenizer (loaded exactly the way
`doctor` already loads tokenizers — `PreTrainedTokenizerFast` from the
local `tokenizer.json`, never weights), check per inventory token:

1. **Registered** — the literal is in the tokenizer's added/special token
   map (`added_tokens`, `bos_token`/`eos_token`/etc.). The doctor already
   collects these dicts (`special_tokens`, `added_tokens` in
   `examine_model`).
2. **Single-token encodable** — `encode(literal, add_special_tokens=False)`
   returns exactly one token id. If `<|im_end|>` fragments into subword
   pieces under checkpoint B's tokenizer, the model cannot be seeing the
   atomic control token the format assumes.
3. **ID-consistent** — when registered *and* single-token, the encoded id
   equals the registered special-token id (catches vocab collisions where
   the string happens to be one ordinary token).

Verdict per (candidate, checkpoint): `compatible` / `incompatible:<reason>`
/ `unknown` (degraded, never fails the caller — the lab's established
fail-closed convention).

**Why this is principled, not a heuristic.** There is direct 2026
literature grounding for the registered-vs-fragmented distinction:
"Same Bytes, Different Authority: Reserved-Token Representations in
Chat-Template Prompt Injection" (arXiv 2609.35932) establishes that "in
every tokenizer we use, each marker string is either an added token or
several ordinary tokens" and shows reserved/added marker vectors carry
different representational authority than subword-mean surrogates
(source: https://arxiv.org/html/2609.35932). A format whose markers are
not atomic special tokens in checkpoint B's vocabulary is, with high
confidence, not a format B was trained with. Conversely the practitioner
literature is unanimous that a wrong template produces garbage/hallucinated
turns (e.g. the "When Chat Templates Go Wrong" analysis of Jinja-engine
mismatch and hardcoded-template substitution,
https://medium.com/@pielouNobodyWho/when-chat-templates-go-wrong-8a437c62490e).

**Logical status (must be documented honestly).** The check is a
*necessary-condition* gate:
- `incompatible` ⇒ the format cannot be B's native format ⇒ re-running the
  experiment on B is scientifically pointless ⇒ **portability answer
  obtained at ~zero compute** (tokenizer load + a few encodes).
- `compatible` ⇒ the markers *could* be B's control tokens ⇒ **no
  prediction** about whether the experiment would select it on B. That
  depends on weights; claiming otherwise would violate the lab's
  exactness discipline.

So the feature answers "can I rule out checkpoint B without running it?"
— the exact question the deferred "cross-checkpoint portability
re-measurement" item was gesturing at — at a cost comparable to `doctor`
(a tokenizer load), i.e. ~0% of a re-run.

**Fit with the exact-token-sequence discipline.** Perfect fit: the matrix
makes statements about *token sequences* (how the rendered prompt encodes
under tokenizer B), never about estimated scores. No stochasticity, no
greedy-decoding assumption, no sampling. It is also the natural
generalization of the existing `core.check_boundary_portability`
(cross-tokenizer re-encoding of the boundary string, P4) from boundary text
to full format framing — same cost class, same fail-closed conventions.

**Suggested shape (for the coordinator):** `doctor --portability-matrix
<second-checkpoint-path>` (or a `core.template_tokenizer_compatibility`
function + doctor wiring), emitting per-candidate rows:
`{candidate, token, registered, single_token, id_match, verdict}`. A
natural second row of the same matrix: run it against the *primary*
checkpoint too — a `compatible` row there plus `incompatible` on the
second checkpoint is itself evidence about *why* the winner won (it was
the only natively-encodable format), which is exactly the lab's "weights
or formatting?" question.

**What could go wrong (must be in the docs):**
1. **False-compatible via token pollution.** A checkpoint can register
   ChatML-style added tokens without being trained with them (merges,
   sloppy conversions). `compatible` is explicitly *not* a prediction —
   the docs must say so, loudly, or users will misread the matrix.
2. **Added-token encode behavior.** `tokenizers` only splits on added
   tokens when they are registered as such in the loaded tokenizer; a
   hand-rolled `PreTrainedTokenizerFast(tokenizer_file=...)` may not carry
   the `added_tokens` metadata the same way the full `from_pretrained`
   path does. The lab's doctor already builds tokenizers this minimal way
   (`doctor._build_primary_tokenizer`) — the matrix must be validated
   against `AutoTokenizer.from_pretrained` behavior on real checkpoints
   before shipping, or it inherits a silent-encode trap.
3. **BOS-less formats.** chatml/qwen3-style formats have empty `bos`; the
   matrix must skip the BOS requirement for them (a naive "bos missing ⇒
   incompatible" would be wrong).
4. **Shape-based inventory limits.** `_extract_special_tokens` is
   regex-shape-based; a format whose control tokens don't match the shapes
   (unlikely for the catalog, possible for user formats) yields an empty
   inventory ⇒ `unknown`, not `compatible`.
5. **It does not replace the re-run.** If both checkpoints report
   `compatible` for the winner, the only honest statement is "not ruled
   out" — the deferred re-measurement item stays deferred for that case.
   The matrix *shrinks* the set of checkpoints worth re-running; it does
   not eliminate re-running.

**Cost.** One tokenizer load per extra checkpoint + O(candidates × tokens)
encodes. No weights, no forward passes, no GPU. Effectively free next to
a `run`.

## 3. Template fingerprinting / inference from behavior

**Not worth building.** The literature on fingerprinting is about *model
identity/lineage* (under-trained-token fingerprints, arXiv 2410.12318;
TokenPrint token-space provenance, arXiv 2608.08139; API token-count
fingerprinting, arXiv 2608.29930) — not about inferring which chat
template a checkpoint expects. No established method exists for
"which template is this checkpoint trained for" beyond the white-box
answer: read `tokenizer_config.json`'s `chat_template` (which
`doctor.template_conformance` already does). Black-box behavioral probing
(feed candidate markers, watch for format adherence) would require model
inference — the expensive thing the static matrix avoids — and would be
speculative science, not a measurement instrument. The static
compatibility matrix in §2 already answers the actionable version of the
question ("is format F encodable as control tokens by checkpoint B?")
cheaper and exactly.

## Recommendation

**Build the static template↔tokenizer compatibility matrix** as a
`doctor` extension (tokenizer-only, fail-closed, exact token-ID
statements). Frame it in docs/USAGE as a *necessary-condition exclusion
screen*: it can rule out checkpoints without running them; it can never
transfer a selection verdict. Explicitly reject: a "portability score",
any claim that `compatible` predicts experimental success on the second
checkpoint, and any behavioral template-inference feature. This keeps the
per-checkpoint philosophy intact (the experiment still runs on exactly
one checkpoint; the matrix is pre-flight triage, like `doctor` itself),
costs ~nothing, and fills a methodology gap no competitor or paper
addresses.
