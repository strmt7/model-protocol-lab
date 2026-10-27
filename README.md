# Model Protocol Lab

[![CI](https://github.com/strmt7/model-protocol-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/strmt7/model-protocol-lab/actions/workflows/ci.yml)
[![Codecov](https://img.shields.io/codecov/c/github/strmt7/model-protocol-lab?label=Codecov&logo=codecov)](https://codecov.io/gh/strmt7/model-protocol-lab)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/strmt7/model-protocol-lab/badge)](https://scorecard.dev/viewer/?uri=github.com/strmt7/model-protocol-lab)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-informational)](CITATION.cff)

**Is a local language model failing because of its weights, or because its conversations are formatted incorrectly?**

**Who it's for:** developers shipping prompts to local LLMs, and researchers studying how chat formatting changes model behavior. If you need to know whether a bad answer comes from the model or from how you talked to it — and prove the answer with numbers, not vibes — this is the tool.

Model Protocol Lab answers that question with measurements, not guesses. It renders chat-format hypotheses — a 29-family catalog (ChatML; Llama 3 and 4; Gemma 2, 3 and 4; Qwen3 and 3.8; DeepSeek V3/R1 and V4.1; Mistral v0.3 and Large; Phi-4; GLM-4, 4.5 and 5.3; Kimi K2 and K3; Kolibri-1; LFM2.5; Nemotron 3.5; Muse Glimmer; Muse Spark; Laguna 2.1; Ling-3.1; K2-Horizon; Hy4; MiniMax-M3; Nanbeige 4.2), plus bounded BOS, prefix and thinking-mode mutations — as exact token sequences, sends them to a local checkpoint, and compares task answers and response boundaries token-for-token. A strict search/held-out fixture discipline selects a single winning format or abstains; nothing is ranked, nothing is downloaded, and every result ships as a byte-deterministic local report.

What sets it apart: it measures checkpoints on token sequences their own chat templates would never produce; it detects response boundaries as token-ID subsequences rather than string stop-sequences; it generates exactly N tokens with no implicit EOS stop; and it abstains on ties instead of crowning a leaderboard winner. The differences are verifiable design decisions, documented with sources in [How this differs](#how-this-differs-from-nearby-tools) — not marketing claims.

## Status

**Production-hardened and tested.** The package installs cleanly, the full test suite passes (8477 tests, 95.3% line coverage — CI gate: 95%), `ruff` and `mypy` are clean, and the optional Hugging Face adapter is integration-tested against a locally generated tiny GPT-2 checkpoint. See [VALIDATION.md](VALIDATION.md) for the exact commands, counts, dates and environment — and for what is still *not* established (real-model runs beyond the synthetic checkpoint, browser rendering checks). No benchmark-superiority or unique-invention claim is made. CI tests Linux x86-64 for the full suite; a separate cross-OS smoke workflow verifies install plus the offline CLI on macOS and Windows, while the Jinja render watchdog, the MCP server's SIGALRM budget, and doctor's resource fallbacks still degrade there.

## Contents

- [Quick start](#quick-start)
- [Documentation](#documentation)
- [What the report looks like](#what-the-report-looks-like)
- [Capabilities](#capabilities)
- [How this differs from nearby tools](#how-this-differs-from-nearby-tools)
- [How to interpret a result](#how-to-interpret-a-result)
- [Repository layout](#repository-layout)
- [Privacy](#privacy)

## Quick start

Python 3.11 or newer is required.

```sh
python -m venv .venv
# Activate the environment with your operating system's usual command, e.g.:
#   source .venv/bin/activate        # POSIX
#   .venv\Scripts\activate           # Windows
python -m pip install -e .
python -m model_protocol_lab catalog
python -m model_protocol_lab inspect examples/fixtures.json
```

Expected output — `catalog` prints the 29-family catalog as JSON
(first entry shown; 28 more follow), and `inspect` prints one summary
line for the fixture file:

```json
[
  {
    "name": "chatml",
    "bos": "",
    "user_prefix": "<|im_start|>user\n",
    "assistant_prefix": "<|im_start|>assistant\n",
    "turn_end": "<|im_end|>\n",
    "repeat_bos": false,
    "user_turn_end": null,
    "assistant_turn_end": null,
    "open_suffix": null
  }
]
```

```json
{"search_cases": 2, "held_out_cases": 2, "valid": true, "contamination": {"worst_tier": "clean", "pairs": []}}
```

(A `model-protocol-lab` console script is installed as an equivalent entry point.)

Two install notes. The `hf` extra pulls in PyTorch: on Linux the default
PyPI wheel is the CUDA build and downloads several GB of NVIDIA libraries
with it. For a CPU-only machine, install the CPU wheel first
(`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`),
then the command above reuses it. And the full test suite is a separate,
optional verification step — it is thorough rather than quick:

```sh
python -m unittest discover -s tests -t .   # optional; expect roughly 20–55 minutes
```

To run a model you also need the optional dependencies and a trusted,
already-downloaded checkpoint directory (the adapter never downloads models):

```sh
python -m pip install -e ".[hf]"
python -m model_protocol_lab doctor --model path/to/checkpoint --dataset examples/fixtures.json
```

Contributors: the dev toolchain (`.[dev]`, for lint, types, tests, and coverage) is a separate install — see [CONTRIBUTING.md](CONTRIBUTING.md). It is not needed to run the tool.

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

## Documentation

The documentation is organised in four parts — tutorials, how-to
guides, reference and explanation — and published as a site at
https://strmt7.github.io/model-protocol-lab/. The same pages live in
this repository under [docs/](docs/index.md), starting from the
[documentation index](docs/index.md); [docs/USAGE.md](docs/USAGE.md)
remains the complete command reference. To build the site locally:
`python -m pip install -r requirements-docs.txt`, then
`mkdocs build --strict`. The build tooling is documentation-only —
the package itself keeps zero runtime dependencies.

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
robustness, field ablation, scorer lens, candidate aliasing, prompt repetition, distractor dilution, leading-claim caving, numeric surface form, date-line sensitivity, multi-turn drift, needle depth, question/payload placement, context rotation, instruction density, schema field order, choice-scoring audit) render only when the data exists and are absent
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
- Behavioural template probe (`doctor --template-probe`): frozen sentinel-bracketed probe conversations rendered in word-swap pairs through the checkpoint's own tokenizer, reporting per pair whether the rendering changed anywhere outside the swapped message's span — the signature of content-conditional template logic (an instruction that appears only for some user messages) that the static template firewall cannot see. Findings only, never a verdict; a section with no findings bounds only templates that cannot detect the published probe conversations.
- Prompt-boundary sensitivity sweep (`run --sensitivity-sweep`): measures whether truncating the prompt mid-token flips the selection (the per-depth byte cap is `--max-drop-bytes`).
- Token-depth sensitivity sweep (`run --token-depth-sweep`): drops whole trailing tokens instead of bytes (`--max-drop-tokens`, default 4, cap 8 provisional) — the ladder that can remove a multi-token assistant turn-closing header cleanly, where the byte sweep can only mangle its tail. Depths too long for the shortest prompt are reported as inapplicable, never counted as flips.
- Raw evidence, opt-in only (`run --include-evidence`): adds the recorded answer/token traces to `results.json` for debugging. Off by default — the public results file carries verdicts, counts and IDs, not raw generations, unless you ask for them. One stated exception: the numform probe's ground records carry their synthetic answer text (never your fixtures' text), because the diff validator re-derives every stored parse from it — see `docs/NUMFORM_DESIGN.md`.
- Whitespace canonicalization (`run --ws-scan`): every search candidate's rendered prompts re-encoded under 8 whitespace mutations of the render string (trailing/leading space, newline, tab, stripping, collapsing) — id sequences compared exactly, fixtures on divergent variants re-run to measure verdict flips. Per-family whitespace fragility with the honest split: divergent ids + flipped verdict is fragility, flipped verdict on identical ids is a backend-determinism finding.
- Synthetic-history selection stability (`run --synth-history`): every search candidate re-evaluated with fabricated prior turns prepended to each fixture — does the winner survive a conversation history it never had, and do delimiter styles differ in how much planted history moves the answer?
- Batch-invariance audit (`run --batch-invariance`): checks whether verdicts would survive batched serving by re-running every search prompt through the local backend's real batched path (left-padded, one forward per step over the whole batch, same fp32 full-prefix discipline as solo decoding) and comparing continuations token-for-token. Equivalence is measured, never assumed: any divergence is reported as a `batch_variant` finding. Oversize batches degrade to a labelled `batch_size_exceeded` refusal instead of an OOM; backends without a batch path still record `batch_unsupported` rather than inventing a result.
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
- Prompt repetition (`run --repeat`): each search fixture's last user message is repeated verbatim inside the same turn (x2), against a token-length-matched filler control (pad) and the recorded native run (x1, reused — no extra calls for the baseline). The headline contrast is x2 minus pad at identical prompt length, so a difference is not explained by length alone; pairs that cannot be length-matched exactly leave the denominator and the pad-evaluable share is reported beside the effect. The winner is re-derived per arm and the readout is withheld when the completed fixture sets differ across candidates. Doubled prompts cost real compute here (full-prefix recompute, no KV cache) — the "no latency increase" claim from the repetition literature does not transfer and is not made.
- Distractor dilution (`run --distract`): irrelevant filler is spliced into each search fixture's last user message at exact rendered token doses (64/256/1024 tokens of a frozen neutral corpus, before and after the real content; 64/256 of near-miss filler carrying a planted wrong answer derived from the fixture's own expected number by a frozen digit-increment rule), against the recorded native run as dose 0. Per-dose pass curves, a first-flip distribution with a returns counter (recovery under more load marks the flip as noise), and the near-miss capture excess over neutral — answers that became the planted string under near-miss but not under neutral filler — separate dilution from active capture; the raw capture rate is labelled an upper bound. Capture is defined only for plain ASCII numeric answers. Every cell records whether its token ids decompose exactly at the splice (id-exact vs text-level, with seam ids), filler is screened per cell for special tokens and boundary texts, and per-condition selection is withheld when completed fixture sets differ. The 24,000-generation worst case is stated in the help before the run.
- Leading-claim caving (`run --leading`): one confident claim sentence is appended to each search fixture's last user message — a wrong claim planted from another fixture's correct answer (donor-first, eligibility-screened so it cannot be read as the right answer in another surface form) and the right claim as control, each in first-person and third-person framings that differ only in perspective — against the recorded native run, reused at zero extra calls. The headline is the caving count: fixtures passed natively and under the right claim but failed under the wrong one, with byte-exact adoption of the planted string reported beside plain instability. A template that follows right and wrong claims alike is labelled claim-following, not caving; the classification is a frozen truth table over the discordant pair, never a comparison of marginal rates. Payloads carry ids, counts, and digests only — no claim or answer text.
- Numeric surface form (`run --numform`): the lab generates its own integer tasks — copy a number, say which of two numbers is larger, add two numbers — and renders every value in four surface forms (plain digits, comma-grouped, narrow-no-break-space-grouped, scientific) under a shared fixture id, with the template and frozen digit-free wording held constant. Answers must match the value in the form the operands arrived in; per-cell decomposition separates wrong values from right-value/wrong-form failures, paired contrasts per task and form pair carry the headline, and the tokenizer's own digit grouping is measured at run time and reported as a run label, since the same form effect means different things under different grouping. For a product that prints prices or readings, this is how you learn whether the form your UI uses is the form the model handles worst under a template — before the template ships. Records store the generated integers in the clear, so `diff-reports` recomputes the whole section from its ground records.
- Multi-turn drift (`run --multiturn`): tasks the probe generates itself from the run seed — ledger sums and shelf argmax over shards, at depths 2 and 3 — each run three ways: all shards in one message (`full`), one shard per turn with the model's own replies as the history (`sharded`), and the same turn structure with a frozen `Noted.` history (`scripted`). Only the final turn is scored. The headline is the self-history discordant pair over single-shot passers — sharded-only failures against scripted-only failures, always with denominators — which separates "its own words in the history hurt" from "any history hurts". A reply containing the template's own special-token literals ends its cell as screened, because forged turn structure is token-real; unbounded or empty intermediate replies become history and are flagged. There is no cross-depth recovery statistic, because depth changes the task set itself. Motivated by Laban et al. 2025 (arXiv:2505.06120), who found models lose 39% on average when the same task arrives sharded across turns instead of in one message.
- Needle depth (`run --needle`): cases the probe generates itself from the run seed — one fact sentence with a unique synthetic code planted inside neutral filler at five depths (0/25/50/75/100%) of two filler doses (1024 and 4096 rendered tokens) — with a no-filler control, a needle-absent false-recall floor, and a duplicate-needle arm reported against its independence baseline. The prompt grows by exactly the dose in rendered tokens over the control render and each depth offset is verified on the candidate's own rendered tokens, so the recall-vs-depth curve compares template families, not tokenizers. Curves, the mid-depth deficit (edge mean minus middle, the *Lost in the Middle* functional form), and first-failure counts share one headline population per dose — the cases every candidate passed with no filler — and the filler is topically dissimilar prose, so the curves are a lower bound on position effects under realistic retrieval bulk.
- Question/payload placement (`run --placement`): 24 generated cases, each a three-sentence payload with the one answer-bearing fact frozen in the middle sentence, asked four ways per template family — question before the payload, after it, twice around it (the sandwich control), and alone (the closed-book screen: a pass without the payload excludes the case for that candidate, with the exclusion counted). The headline is the directional split of discordant verdicts between the two single-question layouts; a directional placement effect is falsified when that discordance stays at or under the sandwich control's discordance on the same cases — a deliberately conservative bar that bounds what recency and exposure together produce without claiming to identify which one. Cases both layouts fail together are reported as joint collapse, never as placement evidence.
- Context rotation (`run --rotate`): 20 generated cases, each a span of five fact-bearing blocks (one asked fact, four fact-shaped distractors) with the question pinned after the span, run in eleven arms per template family — the five cyclic rotations of the block order, four mirror arms (rotations of the reversed order: a generic-reordering control set that places the asked fact at the same slots as the rotations), a fact-absent arm (the false-recall floor), and a closed-book screen (question alone; a pass excludes the case for that candidate). A rotation only counts toward the headline when its rendered token inventory matches the unrotated arm's exactly, so the paired readouts run over token-exact rotations. The headline is verdict invariance across the five rotations under a pre-registered falsification rule; the pass-by-absolute-slot curve is printed as a cross-check against the needle probe's depth curve, not as a second measurement of it. If your retriever re-ranks the same five chunks into a different cyclic order, this is how you learn whether the template keeps the same verdict — and the same winner.
- Instruction density (`run --density`): 20 generated cases, each pairing a small fact-recall task with 16 simultaneously-satisfiable constraints (include a word, start with a word, end with a word, stay within a word limit — four of each, with shared start/end targets and word limits above the largest possible distinct-target count, so a perfect follower passes every arm by construction), run in 21 arms per template family — each constraint solo, the first 2/4/8/16 stacked, and a length-matched sham carrying only the first constraint. Answers are committed as ordered per-word hash lists and every satisfaction verdict is derived from the commitment, never stored. The headline is the solo-vs-full-density contrast, decided per case by an exact sign test with the verdict withheld as insufficient below six discordant cases. If your system prompt keeps growing, this is how you learn which of its instructions the template silently drops first — and at what stack depth.

- Schema field order (`run --field-order`): 15 frozen flat-JSON specs — the shipped structured-output set plus six 3–4-field specs — each rendered in four field orders (canonical alphabetical, the designated answer field first, the answer field last, fully reversed), over the 10 specs whose answer is entailed by question and context. Answers are committed as per-field canonical values under the document digest, and every verdict — per-field correctness, schema adherence, the pairing — is derived from the commitment, never stored. The headline is the endpoint contrast on the answer field's accuracy, decided per candidate by an exact sign test over specs adherent in both endpoint arms, with the verdict withheld as insufficient below six discordant specs and the instrument's low power printed beside the numbers. If your structured-output schema puts the field you actually read last, this tells you whether that costs you accuracy.
- Date-line sensitivity (`run --dateline`): a frozen `[Context] Today's date is … [/Context]` block is prepended to each search fixture's last user message — eight protocol-constant date values (an anchor day, ±1 day, ±1 year, a leap day, a year-boundary pair) in three formats (ISO, long, and weekday-ISO, the shape real system prompts inject), a per-format no-date control sized to the anchor arm's rendered length, and one weekday-lie arm wearing the next day's weekday name. The headline is per-pair discordant counts partitioned pure-date vs pure-format — a union flip rate over 24 date arms saturates and is not shipped — with date-loss subsets (pass the control, fail a date value) and their counterparts beside them. Fixtures whose expected answer is itself date content are screened out before generation, because there the correct answer moves with the date. Every date is a protocol constant and every weekday is computed from its own date; nothing reads the wall clock, so a run in March and a run in October measure the same instrument. Records carry answer digests only, so `diff-reports` recomputes the whole section against its own literal arm table. Motivated by Sanz-Guerrero et al. 2026 (arXiv:2609.36931), who showed injected system-prompt dates move benchmark scores by up to 14% and reorder rankings.
- Choice-scoring audit (`run --choice-audit`): a second opinion on the generation winner, post-hoc over the recorded search run at zero extra generations. For each search fixture a four-choice set is built from the run's own fixtures — the expected answer plus three donor expecteds from other groups, screened so no donor is the same answer twice — and every choice is teacher-forced under every candidate over the recorded search prompt. Reports per candidate: choice accuracy under the total log-likelihood ranking (with per-byte and per-token-mean rankings beside it), the generation × choice discordance table, and whether the choice method's top candidate agrees with the primary winner over the common evaluable population. Monday morning: does log-likelihood choice scoring pick the same answers and the same template as greedy generation on this checkpoint? The audit is informational only — it never selects and never feeds the selection rule; where the two methods disagree, the disagreement itself is the finding.
- Load precision (`run --precision`): loads the same checkpoint a second time in bfloat16 and re-evaluates every search candidate on the search split against the recorded fp32 native run — the per-candidate verdict-flip matrix over paired fixtures, the output-divergence rate, teacher-forced expected-answer logprob deltas, and whether the fp32 winner survives as the selection under bf16 (withheld on mismatched completed sets). This is dtype precision, not quantization: the bf16 arm is a load-time cast of the same unquantized weights. Monday morning: you will serve this checkpoint in bf16 — is the template winner you picked under fp32 still the winner there? Search split only; held-out behaviour under bf16 is not measured.
- Verbalized confidence (`run --confidence`): appends one frozen suffix to each search fixture's last user message asking the model to end its reply with `Confidence: <integer 0-100>`, parsed under a frozen grammar and scored against the same exact-match verdicts — per candidate: the compliance rate, mean stated confidence on correct vs incorrect compliant answers, a four-bin table, and the verdict flips the suffix itself caused. A generated donor-key control (planted-fact cases whose payload binds a different key, so the asked fact is simply absent) tests whether the stated number drops when the answer is not knowable from the prompt. A stated confidence is model output text, not a probability, and no readout is a calibration score. Monday morning: when this checkpoint states a confidence under each template, does the number track whether it was right — and does it drop when the answer is not in the prompt? Search split only; held-out fixtures are never treated.
- Teacher-forced log-likelihood diagnostic (`run --logprob-diagnostic`): the log-probability of each fixture's expected answer under every candidate, reported as a banded heuristic next to the exact verdicts — where the greedy answer and the model's own scoring disagree, the band says so instead of hiding it.

**Operator experience.**
- `doctor` pre-flight validation of checkpoint assets and fixtures without loading weights, with actionable next-command suggestions; `--portability-matrix` renders the cross-tokenizer boundary-portability matrix for every attached checkpoint in one pass; `--template-probe` behaviourally probes the checkpoint's own chat template (see Capabilities).
- `inspect`: validates a fixture file against the strict schema and prints its split hygiene (counts, groups, vacuous-answer warnings) without running anything.
- `fixtures build`: CSV fixture authoring that makes search/held-out leakage structurally unbuildable.
- `fixtures fuzz`: a seeded contract fuzzer for the fixture loaders — a fixed, published 133-case mutator table (JSON text, CSV text, whole-file mutations) checked against the loaders' documented accept/reject contract; the same seed reproduces the report byte-for-byte, and the report carries counts and case IDs only, never fixture text. It is a fixed corpus, not a coverage-guided fuzzer, and says so.
- `model-protocol-lab pack`: seals a report directory into one bundle — a per-file SHA-256 manifest binding the results, the report, any `--include`d inputs, the catalog digest, and the package version. The seal is unsigned: it proves bit-integrity since packing only when the bundle digest is kept separately (`verify --expect-digest`), never attestation or tamper-proofing.
- `debug`: an interactive, offline protocol-debugger REPL over one dataset and one hypothesis — render with a per-segment char-span (and token-ID) map, step response-boundary detection one token at a time, and audit fixtures for special-token literals; `--model` attaches a checkpoint tokenizer only, weights never loaded. Try `python -m model_protocol_lab debug --dataset examples/fixtures.json`, then `.render`.
- `diff-backends`: differential testing of emulated backend stop semantics (HF StopStringCriteria, vLLM detokenizer strip/keep, EOS-appended) on one recorded token sequence — prints a JSON verdict (`agree`/`diverge`) with the per-semantic cells and any divergence findings, so a surprising run verdict can be checked against backend stop behavior. Try `python -m model_protocol_lab diff-backends --model path/to/checkpoint --prompt-text "the answer is" --output-text "42 STOP" --stops "STOP"`.
- `verify`: N-version independent results-integrity replay — re-derives the decision, winner, frontier, and held-out floor from a stored `results.json` through independent code paths, so a corrupted or hand-edited result file is caught instead of trusted. With `--determinism --model <checkpoint>` it also re-executes the recorded search observations and audits token-exact reproduction, issuing an unsigned determinism certificate or a per-cell divergence table. With `--bundle` it first re-hashes a sealed bundle offline (see `pack`), then runs the same replay on the sealed results as an independent second layer. Try `python -m model_protocol_lab verify OLD_RESULTS/results.json`.
- `surface-noise`: a self-consistency probe for a single prompt — perturbs the prompt text with a frozen set of surface-noise operators (typo, case flip, unicode variant, whitespace modes, punctuation strip), re-scores on your fixtures, and reports a per-operator stability vector with minimal-flip counterfactuals plus a self-contained HTML dashboard. Monday-morning use: you rewrote a prompt over the weekend — the probe tells you whether a typo would flip its answers, before you ship it. The reported "nearest observed flip" is an upper bound over the searched subset, never a guarantee; scores compare only under identical fixtures, backend, and protocol fingerprint. An stdlib-only MCP server (`model-protocol-lab-mcp`, shipped in the package; a `tools/surface_noise_mcp_server.py` launcher remains for checkouts) exposes the same instrument to agents, and `surface-noise diff` compares two probe runs under the same protocol discipline as `diff-reports`. Try `python -m model_protocol_lab surface-noise protocol` to inspect the instrument before using it.
- `diff-reports`: deterministic comparison of two experiment result directories for the iterate-and-compare operator loop— reports the decision flip, winner change, per-candidate verdict flips on search and held-out, per-fixture outcome diffs (changed-only by default), the token-budget frontier comparison (exact budget spans where the post-hoc decision, winner, or per-budget leader differ, per-flip-span attribution, and the safe budget floor), the held-out verification-floor comparison, the prompt-boundary audit comparison, the stop-semantics selection-robustness comparison (per-semantic verdict drift and flip pattern), the cross-checkpoint transfer comparison (verdict, per-checkpoint and per-cell drift), the contamination comparison (worst-tier drift, pair changes, tier escalations), the fixture-perturbation comparison (stability drift, per-variant drift), the token-healing comparison (healed-verdict drift, per-candidate (native, healed) drift, falsification drift), the instruction-hierarchy comparison (per-position verdict drift, flip pattern, adherence drift, group-rate drift), the assistant-prefill comparison (per-prefill verdict drift, flip pattern, honoring drift, think-channel group drift), the protocol-portfolio comparison (gap drift, solvable-set drift, membership drift, per-fixture oracle-winner drift, per-member cover drift), the synthetic-history comparison (condition verdict drift, flip pattern, pass-rate drift, delimiter-group drift), the batch-invariance comparison (verdict drift, divergence-rate drift, diverged-prompt sets), the position-sensitivity comparison (condition verdict drift, flip pattern, correct-rate drift, primacy-index drift), the stop-cause comparison (cause-rate drift, aggregate-rate drift, candidate changes), the whitespace-canonicalization comparison (id-divergence / verdict-flip rate drift, robust-verdict flips, fragile-candidate changes), the structured-output comparison (adherence drift, per-field drift), the delimiter-authority comparison (authority-gap drift), the few-shot comparison (gain drift, per-shot-count drift), the fake-turn comparison (tamper-susceptibility drift), the tool-call comparison (parse-rate drift, per-tool drift), the verbosity comparison (per-candidate length drift), the refusal comparison (suspected-rate drift, detector-protocol drift), the scorer-lens comparison (per-rung verdict-count drift, winner-per-rung drift), the candidate-aliasing comparison (coincidence-matrix drift, alias-suspect drift), the prompt-repetition comparison (per-arm verdict drift, paired-contrast drift), the distractor-dilution comparison (per-dose pass-curve drift, capture-excess drift), the leading-claim comparison (caving-rate drift, donor-map drift), the numeric-surface-form comparison (per-form exact-rate drift, paired-contrast drift), the date-line comparison (date-loss and falsification drift, pure-date classification drift), the multi-turn comparison (classification drift, self-history pair drift), the needle-depth comparison (deficit and false-recall drift), the placement comparison (classification and falsification drift, directional discordant counts), the rotation comparison (classification and falsification drift, flip and mirror-control counts, reversal discordance), the density comparison (endpoint verdict and falsification drift, discordance counts, joint-satisfaction drift), the field-order comparison (endpoint verdict and direction drift, discordance counts, paired-rate and per-arm answer-rate drift), the choice-scoring comparison (choice-accuracy and discordance drift, method-agreement drift), the cache-profile comparison (shared-prefix drift), the template-robustness comparison (preservation-rate drift), the field-ablation comparison (attribution drift, load-bearing drift), the confidentiality comparison (leak-rate drift, McNemar drift), the answer-collision comparison (collision-rate drift), and a loud config-drift warning when the runs used different candidates, boundaries, or fixture sets. Try `python -m model_protocol_lab diff-reports OLD_RESULTS NEW_RESULTS`.
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

The current adapter uses greedy full-prefix recomputation, float32 weights for the primary measurement (a bfloat16 load exists only as the `run --precision` comparison arm), no KV cache and no implicit EOS stopping. The seed is recorded and validated but does not alter greedy decoding. It is a correctness-oriented starting point, not an efficient inference engine. Token counts in reports are retained-input/output counts, not equal-compute accounting.

## Repository layout

```text
model_protocol_lab/       Renderer, evaluator, catalog, loader, experiment, reports, CLI
tests/                    Core, workflow, adapter-integration, e2e and sample-report tests
examples/                 Fixture files (JSON + CSV) and a byte-deterministic sample report
docs/                     Usage, FAQ, research log, security guidance, site source
.github/workflows/       CI: lint, typecheck, tests on 3.11-3.13, packaging
DEVELOPMENT_PLAN.md       Architecture, milestones, test matrices and release gates
VALIDATION.md             Verification evidence and limitations
CHANGELOG.md              What changed and when
```

## Privacy

Reports stay on the local filesystem. JSON excludes raw observations of your fixtures by default (the numform probe's synthetic answer records are the one designed exception, above); `--include-evidence` explicitly includes potentially sensitive prompts as token IDs, generated text and traces. Summary metadata, candidate names, format markers and fingerprints remain in ordinary JSON. Inspect all reports before sharing. No report files, model weights, credentials or personal configuration belong in source control.

See [docs/SECURITY.md](docs/SECURITY.md) and [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under MIT — see [LICENSE](LICENSE). If you use this software, please cite it via [CITATION.cff](CITATION.cff).
