# Model Protocol Lab

[![CI](https://github.com/strmt7/model-protocol-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/strmt7/model-protocol-lab/actions/workflows/ci.yml)
[![Codecov](https://img.shields.io/codecov/c/github/strmt7/model-protocol-lab?label=Codecov&logo=codecov)](https://codecov.io/gh/strmt7/model-protocol-lab)
[![cocoindex-code](https://img.shields.io/static/v1?label=&message=cocoindex-code&color=555&logo=github&logoColor=white)](https://github.com/cocoindex-io/cocoindex-code)
[![caveman](https://img.shields.io/static/v1?label=&message=caveman&color=555&logo=github&logoColor=white)](https://github.com/JuliusBrussee/caveman)
[![crawl4ai](https://img.shields.io/static/v1?label=&message=crawl4ai&color=555&logo=github&logoColor=white)](https://github.com/unclecode/crawl4ai)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-informational)](CITATION.cff)

**Is a local language model failing because of its weights, or because its conversations are formatted incorrectly?**

**Who it's for:** developers shipping prompts to local LLMs, and researchers studying how chat formatting changes model behavior. If you need to know whether a bad answer comes from the model or from how you talked to it — and prove the answer with numbers, not vibes — this is the tool.

Model Protocol Lab answers that question with measurements, not guesses. It renders chat-format hypotheses — a 29-family catalog (ChatML; Llama 3 and 4; Gemma 2, 3 and 4; Qwen3 and 3.8; DeepSeek V3/R1 and V4.1; Mistral v0.3 and Large; Phi-4; GLM-4, 4.5 and 5.3; Kimi K2 and K3; Kolibri-1; LFM2.5; Nemotron 3.5; Muse Glimmer; Muse Spark; Laguna 2.1; Ling-3.1; K2-Horizon; Hy4; MiniMax-M3; Nanbeige 4.2), plus bounded BOS, prefix and thinking-mode mutations — as exact token sequences, sends them to a local checkpoint, and compares task answers and response boundaries token-for-token. A strict search/held-out fixture discipline selects a single winning format or abstains; nothing is ranked, nothing is downloaded, and every result ships as a byte-deterministic local report.

What sets it apart: it measures checkpoints on token sequences their own chat templates would never produce; it detects response boundaries as token-ID subsequences rather than string stop-sequences; it generates exactly N tokens with no implicit EOS stop; and it abstains on ties instead of crowning a leaderboard winner. The differences are verifiable design decisions, documented with sources in [How this differs](#how-this-differs-from-nearby-tools) — not marketing claims.

## Status

**Production-hardened and tested.** The package installs cleanly, the full test suite passes (6983 tests, 95% line coverage — CI gate: 95%), `ruff` and `mypy` are clean, and the optional Hugging Face adapter is integration-tested against a locally generated tiny GPT-2 checkpoint. See [VALIDATION.md](VALIDATION.md) for the exact commands, counts, dates and environment — and for what is still *not* established (real-model runs beyond the synthetic checkpoint, browser rendering checks). No benchmark-superiority or unique-invention claim is made.

## Quick start

Python 3.11 or newer is required.

```sh
python -m venv .venv
# Activate the environment with your operating system's usual command, e.g.:
#   source .venv/bin/activate        # POSIX
#   .venv\Scripts\activate           # Windows
python -m pip install -e ".[dev,hf]"
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
produced from `examples/fixtures.json`: a decision banner (the sample run
passes cleanly; a failure-mode diagnosis joins the banner when a run fails
or abstains), a section table of contents, run configuration, a color-coded
candidate heatmap with pass-rate bars, a per-fixture outcome grid, decisive
fixtures, a token-overhead table, a selection-robustness matrix under stop
semantics, token-budget robustness curves with a deterministic SVG frontier
chart, decision ribbon, selection-margin trajectory chart, safe budget floor
readout and per-flip-span attribution, a prompt-boundary canonicality
section, a chat-marker isolation matrix, a template-firewall verdict block,
a cross-checkpoint transfer matrix, a fixture-perturbation robustness
section, and sections for instruction hierarchy, assistant prefill, the
protocol portfolio, synthetic history, batch invariance, position
sensitivity, stop cause, whitespace canonicalization, structured-output
adherence, delimiter authority, few-shot, fake-turn history tampering,
tool-call adherence, verbosity, suspected refusals, prefix-cache
sharability, confidentiality, the answer-collision audit, and the held-out
stage with its verification-floor callout. Sections whose inputs a run does
not produce (template conformance, backend divergence, the stop matrix,
sensitivity sweep, the log-likelihood diagnostic, token healing, template
robustness, field ablation, scorer lens, candidate aliasing) render only when the data exists and are absent
from the sample. It is one self-contained file — a single
tiny inline table-sorting script (hash-allowlisted in its
Content-Security-Policy), no fonts or remote assets — and it never embeds
raw prompts or generations.

## Capabilities

**Exact-token measurement.**
- Strict, bounded JSON fixture loading with search/held-out separation checks (IDs, groups and message inputs cannot leak across the split; NFC-normalized).
- Text-only serialization hypotheses across a 29-family catalog — ChatML; Llama 3/4; Gemma 2/3/4; Qwen3/3.8; DeepSeek V3/R1/V4.1; Mistral v0.3/Large; Phi-4; GLM-4/4.5/5.3; Kimi K2/K3; Kolibri-1; LFM2.5; Nemotron 3.5; Muse Glimmer; Muse Spark; Laguna 2.1; Ling-3.1; K2-Horizon; Hy4; MiniMax-M3; Nanbeige 4.2 — plus bounded BOS, prefix and thinking-mode mutations (`catalog` lists every family with its framing fields; primary sources are cited in the catalog code comments; `run --mutate` adds the bounded mutation variants of a family to the candidate set).
- Raw token evaluation (`add_special_tokens=False`): explicit token-ID boundary detection with a text-level mismatch diagnostic, cooperative cancellation, and sanitized failure records.
- Per-candidate response boundaries (`run --boundary-map`): each format hypothesis is measured with its own end-of-turn marker.

**Disciplined selection.**
- A conservative rule: exactly one candidate must pass every search fixture; only that candidate reaches held-out evaluation. Ties and failures abstain — there is no leaderboard.
- Failure-mode taxonomy with per-candidate counts, cited in the decision banner (`no-boundary-in-N`, `boundary-text-mismatch`, `eos-without-boundary`, `held-out-divergence`, `backend-error`).

**Second-opinion diagnostics.**
- Template-conformance diffs: each hand-rendered hypothesis compared token-for-token against the checkpoint's own chat-template rendering.
- Stop-behavior isolation matrix: the winner re-measured under emulated HF stop-string, vLLM detokenizer, and EOS-appended semantics, so backend stop behavior never silently decides the outcome.
- Held-out verification-budget floor: the first token budget from which the held-out verification verdict holds unchanged (exact, monotone, post-hoc).
- Prompt-boundary canonicality: the fraction of stored prompt x completion pairs whose concatenated text does *not* re-encode to the recorded ids, per template family (the Prompt Boundary Problem rate — a pair that re-encodes exactly is canonical, so the rate counts the failures).
- Chat-marker isolation: hostile marker strings (`<|im_start|>`, `<s>`, `[INST]`, …) injected into user text must not become real control tokens — a per-family conformance matrix against the checkpoint tokenizer.
- Template firewall: the checkpoint's Jinja `chat_template` is statically scanned for RCE primitives (dunder access, `|attr`, dangerous globals, unbounded `range()`) before any render; denied templates are never rendered (fail closed).
- Selection robustness under stop semantics: the conservative selection rule re-applied to every search candidate under each emulated stop semantic (HF, vLLM, EOS) — does the winner survive your serving stack? — with flip callouts and `diff-reports` migration analysis.
- Cross-checkpoint transfer (`run --transfer-model`): every search candidate re-evaluated against up to 4 additional local checkpoints (search split only); the selection rule re-applied per checkpoint to report whether the primary winner transfers — the fleet-portability question.
- Fixture-perturbation robustness (`run --perturb N`): every search candidate re-evaluated on N deterministically perturbed fixture variants (typos, case flips, whitespace, Unicode variants, punctuation stripping; search split only); the selection rule re-applied per variant to report whether the primary winner survives phrasing noise — the phrasing-luck question. With stop-semantics robustness and cross-checkpoint transfer this completes the winner-reliability trilogy.
- Token-healing selection sensitivity (`run --token-healing`): the whole experiment replayed with every prompt healed before generation (a faithful port of HF `GenerationMixin.heal_tokens` into the lab's greedy decoder: re-tokenize the prompt, extend the tail token with every vocabulary prefix-extension, one biased argmax) — does the winner survive the greedy-tokenization boundary fix? Includes a falsification block testing the prompt-boundary audit's own prediction: verdicts may change only for seam-fragile candidates.
- Instruction-hierarchy adherence probe (`run --hierarchy`): every search candidate re-evaluated on synthetic system-vs-user conflict fixtures (the system instruction delivered as a privilege-marked block folded into the user turn, in primacy and recency positions) — which instruction does each template lead the model to follow, and does the winner change under conflict? Includes the documentation-derived dedicated-frame vs folded family comparison, reported as correlation.
- Assistant-prefill sensitivity (`run --prefill`): every search candidate re-evaluated with benign format-lock prefills (`Answer: `, `Final answer: `, `Result: `) appended as token ids after the generation prompt — does the winning protocol survive prefill, and which template families honor vs ignore it? Honoring is detected exactly (a model that restarts its turn duplicates the prefill); the selection rule is re-applied per prefill, with a think-channel group comparison.
- Protocol portfolio (`run`, always on): pure post-hoc over the stored search observations — how much the single global winner leaves on the table versus a per-fixture oracle, and the greedy covering protocol portfolio over solvable fixtures (deterministic; reported as an approximation, not an optimum).
- Token-budget robustness frontier: the search decision recomputed at every smaller token budget, post-hoc and exact — plus a deterministic inline-SVG frontier chart with a decision ribbon and a per-budget selection-margin trajectory chart.
- Cross-tokenizer boundary portability check (`doctor --reference-model`).
- Prompt-boundary sensitivity sweep (`run --sensitivity-sweep`): measures whether truncating the prompt mid-token flips the selection (the per-depth byte cap is `--max-drop-bytes`).
- Raw evidence, opt-in only (`run --include-evidence`): adds the recorded answer/token traces to `results.json` for debugging. Off by default — the public results file carries verdicts, counts and IDs, not raw generations, unless you ask for them.
- Whitespace canonicalization (`run --ws-scan`): every search candidate's rendered prompts re-encoded under 8 whitespace mutations of the render string (trailing/leading space, newline, tab, stripping, collapsing) — id sequences compared exactly, fixtures on divergent variants re-run to measure verdict flips. Per-family whitespace fragility with the honest split: divergent ids + flipped verdict is fragility, flipped verdict on identical ids is a backend-determinism finding.
- Synthetic-history selection stability (`run --synth-history`): every search candidate re-evaluated with fabricated prior turns prepended to each fixture — does the winner survive a conversation history it never had, and do delimiter styles differ in how much planted history moves the answer?
- Batch-invariance audit (`run --batch-invariance`): checks whether verdicts would survive batched serving; the local backend exposes no batch path, so the audit records `batch_unsupported` rather than inventing a result — an honest gap marker, revisited if a batch-capable backend lands.
- Position sensitivity (`run --position`): answer options are permuted per fixture and the verdict re-measured — a per-candidate primacy index in [-1, 1] says whether a template's winner only wins when the right answer sits in its favourite slot.
- Stop-cause termination audit (`run --stop-cause`): a counterfactual over the recorded generations — under each emulated serving semantic, which stop cause (boundary, EOS, budget) would have ended each answer, and would the verdict have changed? The lab's own decoder never stops early, so every divergence is attributed to the serving layer, not the model.
- Structured-output adherence (`run --structured`): fixtures whose expected answer is a JSON value checked against a per-fixture schema — which templates keep the model emitting parseable, schema-valid JSON instead of prose around it?
- Delimiter authority (`run --deliforge`): forged turn delimiters are presented twice — once encoded as reserved special tokens, once as plain subword text — to measure which encoding each template actually obeys.
- Few-shot × template (`run --few-shot`): exemplars rendered inside each candidate's own framing — does the winner survive demonstrations, and which framing learns the task format from examples instead of fighting them?
- Fake-turn history tampering (`run --fake-turn`): forged user/assistant turns are injected into the conversation history — which templates let planted history move the answer, and by how much?
- Prompt-only tool-call adherence (`run --tool-call`): the prompt asks, in plain text, for a tool call in a declared format — which templates yield calls that parse, with the right tool name and arguments?
- Response verbosity profile (`run --verbosity`): post-hoc over the recorded generations, zero extra model calls — how much text each template elicits per correct answer. A cost and behavior profile, not a quality score; capped generations count as lower bounds.
- Surface-noise robustness per template (`run --template-robustness`): the fraction of each candidate's *correct* verdicts that survive presentation-level noise (whitespace, case, Unicode variants) applied to the user messages — measured on correct verdicts only, so a template cannot look robust by failing consistently.
- Template field ablation (`run --ablate-fields`): one structural field of the format (BOS, role markers, separators, suffix) at a time is replaced by a neutral value and the conservative selection re-applied per arm — which fields are actually load-bearing for the winner?
- Prefix-cache sharability (`run --cache-profile`): post-hoc over the recorded token IDs, zero model calls — how much prompt prefix each template shares across fixtures, quantized to cache blocks (`--cache-profile-block-size`) — the KV-cache-friendliness of a format, before you serve it.
- Labeled-block confidentiality (`run --confidentiality`): a privilege-marked block folded into the user turn carries a marker the model is told to keep internal; frozen elicitation probes then measure, per template, whether the marker leaks verbatim or in fragments.
- Answer-collision audit (`run --answer-collision`): the lab audits its own token-ID boundary detector — can a different answer produce the same boundary configuration and be misread as the recorded one? The unit of measurement is the boundary configuration, not the template.
- Suspected-refusal profile (`run --refusals`): post-hoc over the recorded answers, zero extra model calls — a frozen refusal-opener detector estimates each template's spurious-refusal rate on benign fixtures. Rates are *suspected* (string matching is the field's weak baseline), travel with exact counts, and are withheld for candidates whose search did not complete.
- Scorer-lens regrade (`run --scorer-lens`): post-hoc over the recorded answers, zero extra model calls — every recorded answer is regraded under a frozen normalisation ladder (exact; whitespace; case/Unicode; edge punctuation and markdown fences; numeric value), reporting how many of each template's failures each rung recovers, which fixture flips first at which rung, and whether the search-phase winner changes under a more forgiving scorer. The ladder has no label-strip operation (`Answer: 42` flips at no rung), never reads decimal commas as grouping, and reports counts and fixture IDs only — no answer text.
- Candidate aliasing audit (`run --aliasing`): post-hoc over the recorded token IDs, zero extra model calls — candidates are grouped into token-identity equivalence classes on the run's fixture set: render classes (same prompt token sequences) and measurement classes (same prompts *and* same response boundary), reporting effective vs recorded candidate counts, redundant completed observations, per-pair coincidence, and determinism-suspect cells inside a class. When an `ambiguous` abstention's passing set sits inside one measurement class, the audit flags an alias-suspect abstention — the tie was one measurement wearing two name tags. Classes are relative to the fixture set and tokenizer; the payload carries per-cell SHA-256 digests and counts only — no token IDs, no text.
- Teacher-forced log-likelihood diagnostic (`run --logprob-diagnostic`): the log-probability of each fixture's expected answer under every candidate, reported as a banded heuristic next to the exact verdicts — where the greedy answer and the model's own scoring disagree, the band says so instead of hiding it.

**Operator experience.**
- `doctor` pre-flight validation of checkpoint assets and fixtures without loading weights, with actionable next-command suggestions; `--portability-matrix` renders the cross-tokenizer boundary-portability matrix for every attached checkpoint in one pass.
- `inspect`: validates a fixture file against the strict schema and prints its split hygiene (counts, groups, vacuous-answer warnings) without running anything.
- `fixtures build`: CSV fixture authoring that makes search/held-out leakage structurally unbuildable.
- `fixtures fuzz`: a seeded contract fuzzer for the fixture loaders — a fixed, published 133-case mutator table (JSON text, CSV text, whole-file mutations) checked against the loaders' documented accept/reject contract; the same seed reproduces the report byte-for-byte, and the report carries counts and case IDs only, never fixture text. It is a fixed corpus, not a coverage-guided fuzzer, and says so.
- `model-protocol-lab pack`: seals a report directory into one bundle — a per-file SHA-256 manifest binding the results, the report, any `--include`d inputs, the catalog digest, and the package version. The seal is unsigned: it proves bit-integrity since packing only when the bundle digest is kept separately (`verify --expect-digest`), never attestation or tamper-proofing.
- `debug`: an interactive, offline protocol-debugger REPL over one dataset and one hypothesis — render with a per-segment char-span (and token-ID) map, step response-boundary detection one token at a time, and audit fixtures for special-token literals; `--model` attaches a checkpoint tokenizer only, weights never loaded. Try `python -m model_protocol_lab debug --dataset examples/fixtures.json`, then `.render`.
- `diff-backends`: differential testing of emulated backend stop semantics (HF StopStringCriteria, vLLM detokenizer strip/keep, EOS-appended) on one recorded token sequence — prints a JSON verdict (`agree`/`diverge`) with the per-semantic cells and any divergence findings, so a surprising run verdict can be checked against backend stop behavior. Try `python -m model_protocol_lab diff-backends --model path/to/checkpoint --prompt-text "the answer is" --output-text "42 STOP" --stops "STOP"`.
- `verify`: N-version independent results-integrity replay — re-derives the decision, winner, frontier, and held-out floor from a stored `results.json` through independent code paths, so a corrupted or hand-edited result file is caught instead of trusted. With `--determinism --model <checkpoint>` it also re-executes the recorded search observations and audits token-exact reproduction, issuing an unsigned determinism certificate or a per-cell divergence table. With `--bundle` it first re-hashes a sealed bundle offline (see `pack`), then runs the same replay on the sealed results as an independent second layer. Try `python -m model_protocol_lab verify OLD_RESULTS/results.json`.
- `surface-noise`: a self-consistency probe for a single prompt — perturbs the prompt text with a frozen set of surface-noise operators (typo, case flip, unicode variant, whitespace modes, punctuation strip), re-scores on your fixtures, and reports a per-operator stability vector with minimal-flip counterfactuals plus a self-contained HTML dashboard. Monday-morning use: you rewrote a prompt over the weekend — the probe tells you whether a typo would flip its answers, before you ship it. The reported "nearest observed flip" is an upper bound over the searched subset, never a guarantee; scores compare only under identical fixtures, backend, and protocol fingerprint. An stdlib-only MCP server (`tools/surface_noise_mcp_server.py`) exposes the same instrument to agents, and `surface-noise diff` compares two probe runs under the same protocol discipline as `diff-reports`. Try `python -m model_protocol_lab surface-noise protocol` to inspect the instrument before using it.
- `diff-reports`: deterministic comparison of two experiment result directories for the iterate-and-compare operator loop— reports the decision flip, winner change, per-candidate verdict flips on search and held-out, per-fixture outcome diffs (changed-only by default), the token-budget frontier comparison (exact budget spans where the post-hoc decision, winner, or per-budget leader differ, per-flip-span attribution, and the safe budget floor), the held-out verification-floor comparison, the prompt-boundary audit comparison, the stop-semantics selection-robustness comparison (per-semantic verdict drift and flip pattern), the cross-checkpoint transfer comparison (verdict, per-checkpoint and per-cell drift), the contamination comparison (worst-tier drift, pair changes, tier escalations), the fixture-perturbation comparison (stability drift, per-variant drift), the token-healing comparison (healed-verdict drift, per-candidate (native, healed) drift, falsification drift), the instruction-hierarchy comparison (per-position verdict drift, flip pattern, adherence drift, group-rate drift), the assistant-prefill comparison (per-prefill verdict drift, flip pattern, honoring drift, think-channel group drift), the protocol-portfolio comparison (gap drift, solvable-set drift, membership drift, per-fixture oracle-winner drift, per-member cover drift), the synthetic-history comparison (condition verdict drift, flip pattern, pass-rate drift, delimiter-group drift), the batch-invariance comparison (verdict drift, divergence-rate drift, diverged-prompt sets), the position-sensitivity comparison (condition verdict drift, flip pattern, correct-rate drift, primacy-index drift), the stop-cause comparison (cause-rate drift, aggregate-rate drift, candidate changes), the whitespace-canonicalization comparison (id-divergence / verdict-flip rate drift, robust-verdict flips, fragile-candidate changes), the structured-output comparison (adherence drift, per-field drift), the delimiter-authority comparison (authority-gap drift), the few-shot comparison (gain drift, per-shot-count drift), the fake-turn comparison (tamper-susceptibility drift), the tool-call comparison (parse-rate drift, per-tool drift), the verbosity comparison (per-candidate length drift), the refusal comparison (suspected-rate drift, detector-protocol drift), the cache-profile comparison (shared-prefix drift), the template-robustness comparison (preservation-rate drift), the field-ablation comparison (attribution drift, load-bearing drift), the confidentiality comparison (leak-rate drift, McNemar drift), the answer-collision comparison (collision-rate drift), and a loud config-drift warning when the runs used different candidates, boundaries, or fixture sets. Try `python -m model_protocol_lab diff-reports OLD_RESULTS NEW_RESULTS`.
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

**Dynamic prompt probing next to static prompt scoring.** ctxray
([github.com/ctxray/ctxray](https://github.com/ctxray/ctxray), MIT, checked
2026-10-19) scores prompts deterministically from 30+ static features in
<50ms with zero LLM calls — the static tier. This lab's `surface-noise`
probe is the dynamic tier: it *executes* the prompt under a frozen set of
surface-noise perturbations and reports a per-operator verdict-stability
vector with minimal-flip counterfactuals — a measurement ctxray explicitly
does not do. The reported "nearest observed flip" is an upper bound over
the searched subset, never a guarantee; the academic precedent is the
CoIPO/NoisyPromptBench "decoding radius" (arXiv:2603.03314) and the 2026
PCE minimal-flip counterfactuals work (arXiv:2601.03156), both cited, not
claimed over.

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
