# Model Protocol Lab Development Plan

Model Protocol Lab is a proposed local tool for inspecting and repairing model conversation formats through raw token-ID probes. The intended workflow is to inspect a model and tokenizer, choose a documented format family, evaluate a bounded set of explicit mutations, inspect exact boundary and continuation evidence, test a frozen choice on independent held-out fixtures, and export a format only when its behavior can be reproduced.

This document is an implementation and validation plan for contributors. It does not report a completed application, a tested model adapter, measured repair accuracy, or a competitive benchmark result. Its central release requirement is that every recommendation remain traceable to recorded evidence, including limitations and reasons to abstain.

## 1. Status and evidence vocabulary

Use the following labels consistently in documentation, reports, issue tracking, and the interface:

- **PRESENT SOURCE** means an implementation or test draft exists in the source bundle. It does not mean that the code imports, runs, passes tests, or satisfies its stated contract.
- **VERIFIED** means a specific claim is supported by a reproducible validation record identifying the source revision, environment, procedure, inputs, and result. Verification applies to that claim and configuration only.
- **PLANNED** means the behavior described here is a requirement or proposed design that still needs implementation or validation.
- **UNSUPPORTED** means the current release deliberately refuses the capability or configuration. Unsupported cases must not quietly fall back to a less trustworthy mode.

### 1.1 Present source baseline

`model_protocol_lab/core.py` contains design-level definitions for messages, a simple format renderer, a bounded local-repair generator, grouped fixtures, search and held-out partitions, a backend protocol, raw-generation records, token-boundary matching, and a sequential evaluator. Source inspection shows intended safeguards for invalid backend returns, fixed public error codes, cancellation, fixture-derived seeds, and incomplete-run exclusion.

`tests/test_core.py` contains 45 proposed unittest test methods. They describe checks for rendering, mutations, split validation, boundary matching, generation contracts, failures, seeds, cancellation, and report integrity. Additional workflow tests are in `tests/test_workflows.py`; asset-layout and fingerprint helper tests are in `tests/test_backend_assets.py`. Test-method count is an inventory fact, not coverage evidence or proof of correctness. These helper tests do not exercise actual model loading or generation.

The expanded source bundle has a flat `model_protocol_lab` package containing `__init__.py`, `__main__.py`, `core.py`, `catalog.py`, `datasets.py`, `experiments.py`, `reporting.py`, `cli.py`, and `backends.py`. These files remain PRESENT SOURCE until corresponding verification records exist. Target module names later in this document describe responsibilities; they do not assert that the finer-grained architecture is already implemented.

The current draft scope is deliberately narrower than the target design:

- `catalog.py` supplies text-only ChatML, Llama 3, and Gemma format hypotheses and bounded same-family mutations. A catalog entry is not verified checkpoint support.
- `datasets.py` validates the current fixture representation. It is not a full sealed-dataset, provenance, or near-duplicate analysis system.
- `experiments.py` searches the supplied candidates and selects only a unique candidate that passes every search fixture on exact answer plus boundary presence. No winner or multiple winners produces no selection. It then evaluates exactly one selected candidate on held-out fixtures. The narrow success label is `compatible_on_fixture_set`; this is not a statistical improvement claim or an equal-compute comparison. Selection is frozen by the draft's control flow, without the complete immutable freeze/provenance system planned below.
- `backends.py` proposes local safetensors loading for built-in GPT-2 and Llama architecture types, a self-contained fast `tokenizer.json`, fp32 eager execution, and greedy full-prefix recomputation without cache or automatic stops. It validates the seed but does not use it to change greedy selection. Architecture loading support and conversation-format catalog entries are separate concepts. Gemma in the catalog does not make Gemma checkpoint loading supported.
- `cli.py` proposes `catalog`, `inspect`, and `run` commands. A run takes a local model, dataset, family, optional bounded mutations, explicit boundary IDs, output cap, context limit, seed, device, and report destination. Raw answer/token evidence is an explicit export option.
- `reporting.py` writes a JSON result and escaped static HTML heatmap. This is a read-only report draft, not a live run-control interface or a conversation graph.

The draft's encoded-input and retained-output counts are not actual inference-compute receipts. Full-prefix recomputation processes the growing prompt again on every generation step. Statistical uncertainty, likelihood scoring, strict total compute limits, immutable split/freeze provenance, and target-runtime export parity remain planned.

### 1.2 Verification baseline

No runtime validation is established by this source-draft delivery. No successful import, compilation, test execution, real-model inference, browser interaction, security scan, benchmark, or export round trip is claimed here. In particular:

- A scripted backend cannot demonstrate that a model understands a format.
- A completed evaluation is not automatically an accurate evaluation.
- Exact answer match and boundary presence are separate observations.
- A good result on a small fixture set cannot identify a unique training template.
- An adapter file is not evidence that its loading, cancellation, stopping, or scoring behavior is correct.
- An HTML file is not evidence of a working application or accessible interface.

All stages and gates below are open until explicit evidence closes them. There are no promised completion dates, throughput figures, accuracy figures, or hardware requirements derived from measurements.

## 2. Product scope and success criteria

### 2.1 Intended first useful release

The first useful release should let a technically competent user:

1. Select a previously downloaded, explicitly supported local checkpoint and tokenizer without executing checkpoint-provided Python code.
2. Inspect model/tokenizer fingerprints, role delimiters, special-token metadata, context limits, and the applicable format-family catalog entry.
3. Load a small, versioned fixture set and inspect its groups and split manifest before running inference.
4. Preview rendered prompt bytes, token IDs, and assistant-prefix placement for each proposed candidate.
5. Set separate search and held-out budgets covering actual prefill, decode, scoring, and retries.
6. Evaluate the baseline and a finite set of approved local mutations with a controlled raw-token backend.
7. Inspect the full generated continuation, exact matched boundary spans, suffix behavior, answer scores, incomplete attempts, and resource charges.
8. Freeze a candidate, thresholds, seeds, and analysis policy before opening the held-out phase.
9. Receive either a scoped recommendation, an explicit no-change result, or an abstention with actionable reasons.
10. Export a canonical format artifact, reload it in the target runtime, and see a parity report before relying on it.

### 2.2 Definition of a successful recommendation

A recommendation must identify the exact model revision, tokenizer revision, format specification, workload scope, and export target. It must pass structural and adapter-conformance checks; meet preregistered held-out decision criteria; have no unresolved critical boundary, continuation, or parity defect; and fit within the actual budget ledger. It must preserve the baseline as an available outcome.

The word “repair” refers to a supported behavioral improvement under the measured workload and runtime. It does not establish recovery of the original training template, safety against prompt injection, general model intelligence, or correctness for all conversational features.

### 2.3 Explicit non-goals for the first release

- Unbounded prompt search, arbitrary template synthesis, or recovery of unknown protocols from scratch.
- Training, fine-tuning, model-weight modification, or tokenizer-vocabulary modification.
- Hosted inference, account integration, model downloading, or credential management as prerequisites.
- Execution of remote model code, user-defined Python templates, or arbitrary shell commands from imported artifacts.
- Tool calls, multimodal content, parallel function calls, or model-specific reasoning channels without separately specified support.
- Universal system-message support, assistant prefills, or continuation of partial assistant turns in the initial simple renderer.
- A universal template that works across model families or runtimes.
- A security certification or guarantee that role delimiters prevent malicious instructions.
- A leaderboard that combines incompatible tasks, runtime configurations, or tokenization policies into one score.
- Claims of novelty or superiority over ChatLint, llama.cpp autoparser, gbench, Ingot, FormatSpread, or simpler baselines without a documented comparison.

## 3. Design invariants

These invariants are release-blocking requirements rather than optional implementation preferences.

1. **Raw evidence is authoritative.** Preserve continuation-only token IDs before any stop handling, trimming, normalization, display formatting, or answer extraction.
2. **Prompt construction is explicit.** Record rendered bytes, tokenizer options, and resulting input IDs. Do not permit an adapter to add an implicit chat wrapper or silently insert special tokens.
3. **Boundaries are token sequences.** Detect exact token-ID subsequences. Displayed text is a convenience view and cannot substitute for token-level evidence.
4. **Generation and interpretation are separate.** A detected boundary must not terminate diagnostic generation. The controlled raw mode continues to the approved cap unless cancelled or failed.
5. **Budgets account for actual work.** Repeated prefill, retries, scoring forwards, warmups, and post-boundary tokens must not disappear from the ledger.
6. **The held-out split is a decision boundary.** Candidate generation and search selection cannot read held-out outcomes. Any adjustment after exposure requires a new independent confirmation set.
7. **Incomplete evidence cannot win.** Cancelled, failed, budget-exhausted, or insufficiently covered runs remain visible but are ineligible for comparative recommendation.
8. **Uncertainty can block a result.** Ties, conflicting metrics, inadequate coverage, unsupported behavior, or wide uncertainty should produce abstention rather than a decorative confidence number.
9. **Exports are executable specifications.** Every supported target needs byte/token construction parity and boundary-semantics parity checks after reload.
10. **Local means local by default.** Model data, prompts, evidence, error logs, and reports stay local unless the user deliberately exports them. The default workflow makes no telemetry or network request.
11. **A capability is scoped.** Support is attached to a tested model, tokenizer, adapter, device, precision, decoding configuration, and export target combination.
12. **The source release is clean.** Public archives contain source, synthetic examples, documentation, and approved manifests; they exclude credentials, private runs, model weights, caches, and environment-specific files.

## 4. Target architecture and responsibility boundaries

Keep the core independent of heavyweight inference libraries and UI frameworks. The backend adapter should be replaceable without changing the experimental policy. The viewer should consume validated reports without being able to change a finished run.

### 4.1 Proposed package responsibilities

| Component | Responsibility | Must not own |
| --- | --- | --- |
| `domain` | Immutable messages, formats, fixtures, candidates, attempts, and decisions | Model loading or UI state |
| `rendering` | Restricted format rendering, exact byte preservation, tokenization plans | Candidate ranking |
| `catalog` | Known-family entries, provenance, supported roles, bounded mutation policies | Automatic trust in checkpoint metadata |
| `splits` | Group-level partitioning, leakage checks, sealed manifests | Reading held-out outcomes during search |
| `backends` | Capability negotiation, raw generation, token scoring, cancellation, actual counters | Silent wrapping or answer cleanup |
| `budget` | Reservations, charges, receipts, reconciliation, budget termination | Invented estimates presented as actuals |
| `probes` | Versioned diagnostic fixture families and scoring requirements | Dynamically changing the test after results |
| `evaluation` | Attempt scheduling, immutable evidence, answer extraction, structural checks | Unlogged retries or candidate retuning |
| `selection` | Search-only ranking, ties, freeze records, abstention policy | Held-out optimization |
| `statistics` | Group-aware estimates, paired comparisons, uncertainty methods | Causal or broad claims from small samples |
| `provenance` | Canonical serialization, fingerprints, run ancestry, event records | Exporting machine identity or secrets |
| `reporting` | Validated JSON/JSONL, human summaries, redaction-aware views | Recomputing convenient scores differently |
| `exporters` | Target-specific formats, compatibility declarations, parity harnesses | Guessing unsupported target semantics |
| `cli` | Configuration, validation, orchestration, exit statuses, local artifacts | Hidden policy defaults that alter evidence |
| `local_api` | Loopback-only job control and report access, if added | Arbitrary file reads or command execution |
| `viewer` | Read-only evidence browsing and carefully separated run controls | Trusting imported HTML or modifying scores |

The optional Hugging Face adapter should be isolated behind an installation extra and lazy imports. Reading a report or validating a format must not require loading a checkpoint or importing the inference stack.

### 4.2 Data flow

The intended flow is:

`validated local inputs -> immutable run manifest -> baseline and mutation catalog -> search attempts -> frozen decision -> held-out confirmation -> decision record -> export -> reload parity report`.

Every transition writes a versioned artifact or event linking its input hashes. The held-out evaluator accepts a freeze record rather than a mutable candidate list. The exporter accepts a fully specified format and evidence references; it must not infer missing delimiters from answer text.

### 4.3 Extension policy

Third-party adapters and catalog entries are data or installed packages selected explicitly by the operator. Merely opening a report or template must never import a plugin. A capability declaration is checked by conformance tests before a backend becomes eligible for recommendation runs.

Catalog entries need maintainers, evidence references, revisions, and a clear distinction between documented upstream behavior and empirically tested local behavior. An ambiguous checkpoint family stays unsupported until resolved.

## 5. Versioned data contracts

Schemas should reject unknown required semantics, invalid enum values, non-finite numbers, invalid token IDs, excessive nesting, and oversized arrays before model work begins. Compatible additions use explicit optional fields; breaking semantic changes require a schema version change and migration tests.

### 5.1 Core records

| Record | Required information | Important invariants |
| --- | --- | --- |
| `ModelIdentity` | Local artifact digest, upstream identifier when known, immutable revision, architecture, weight format, precision, quantization | A mutable model name is insufficient identity |
| `TokenizerIdentity` | Artifact digests, revision, tokenizer class, vocabulary size, special-token map, normalization and special-token options | Identity covers behavior-affecting configuration, not just vocabulary |
| `BackendCapabilities` | Raw generation, explicit token input, special-token-preserving decode, scoring, cancellation, counter quality, cache policy | Unsupported capabilities are declared before scheduling |
| `FormatSpec` | Schema version, family, typed literal segments, role rules, BOS policy, turn-end policy, open-assistant policy | Text delimiters and token delimiters cannot be silently interchanged |
| `CandidateSpec` | Stable ID, parent ID, mutation IDs, canonical format hash, order, approval policy version | Names are display labels; identity uses canonical semantics |
| `Fixture` | ID, group ID, role/content messages, expected result, task type, scorer version, provenance, sensitivity class | Fixture and group IDs are stable within a sealed dataset |
| `SplitManifest` | Dataset hash, grouped partition assignment, split seed, duplicate-check version, search/held-out counts | Related groups and exact inputs cannot cross partitions |
| `RunManifest` | Code revision, model/tokenizer/backend IDs, configuration, seeds, budgets, split hash, candidate policy, decision policy | Immutable before work starts; changes create a new run |
| `Attempt` | Run/phase/candidate/fixture IDs, attempt number, input IDs, continuation IDs, timestamps, state, backend receipt | Each inference or scoring call has one auditable identity |
| `BoundaryEvidence` | Candidate boundary sequences, all exact match spans, chosen span, ambiguity, trailing tokens | Offsets refer to raw continuation token indexes |
| `ScoreEvidence` | Scorer version, exact target span, per-token values where supported, numerator, denominator, exclusions | Score identity includes alignment and normalization policy |
| `BudgetReceipt` | Reserved and actual prefill/decode/scoring work, cache effects, retry linkage, counter source | Unknown counters remain unknown; they never become zero |
| `FreezeRecord` | Chosen candidate or shortlist, search report hash, tie policy, thresholds, seeds, analysis policy | Created before held-out outcomes are read |
| `Decision` | Recommend/no-change/abstain, scoped claims, evidence IDs, uncertainty, reasons, restrictions | Recommendation cannot outlive incompatible model identity |
| `ExportManifest` | Format hash, target runtime/version, files and digests, compatibility, redactions, parity result | An unverified export must be labeled as such |

### 5.2 Serialization and identity rules

- Specify canonical UTF-8 serialization and canonical key ordering for hashes. Distinguish absent fields, `null`, empty strings, empty lists, and zero.
- Store token IDs as bounded nonnegative integers; reject booleans and floats masquerading as integers. Validate vocabulary bounds when a tokenizer identity is available.
- Use explicit index conventions: token spans are zero-based half-open intervals; display line numbers may be one-based but must be labeled.
- Preserve line endings, spaces, tabs, and Unicode code points in prompt and delimiter text. No normalization is allowed without an explicit, versioned policy.
- Reject invalid Unicode at the public JSON/text boundary, or encode it through a deliberately documented lossless representation. The backbone's surrogate-tolerant seed encoding is an internal draft detail, not a public portability guarantee.
- Keep operational paths outside portable scientific identities. Portable exports use logical artifact IDs and digests, not a machine's directory structure.
- Keep raw data and redacted summaries distinct. A redacted report cannot claim to contain sufficient evidence for full replay when it does not.
- Hashing sensitive prompts does not necessarily anonymize them; short or predictable text can be guessed. Public manifests must avoid unnecessary prompt hashes as well as prompt text.

### 5.3 Run state machine

Define an explicit sequence of states: `created`, `validated`, `searching`, `search_complete`, `frozen`, `confirming`, `confirmed`, and `exported`. Terminal non-success states include `cancelled`, `failed`, `budget_exhausted`, and `abstained` where applicable. A completed evidence collection can still yield an abstained decision.

Record transitions as append-only events with a monotonic sequence number, event type, phase, timestamp, triggering actor category, and fixed public reason code. Resume creates a linked continuation record and cannot erase the earlier failure or cancellation.

The original backbone has the narrower report statuses `completed`, `cancelled`, and `failed`. Migration must preserve those meanings rather than retroactively treating every completed report as a confirmed result.

## 6. Rendering and bounded candidate generation

### 6.1 Initial renderer scope

The current backbone describes alternating user/assistant messages starting with a user and ending with a user, followed by an open assistant prefix. Implement and verify this subset first. Reject unsupported roles and message order instead of silently reordering messages or folding system content into a user turn.

The renderer must make the following policies visible:

- Whether BOS is present once, before every message, or absent.
- Whether repeated BOS includes the final open assistant prefix.
- Which exact bytes precede each role's content.
- Which delimiters follow user and assistant turns.
- Whether an assistant prefix is open or already terminated.
- Whether a separator belongs to the preceding turn or the next role.
- Whether special-token spellings are ordinary text or recognized special tokens under the tokenizer's settings.

The backbone has one shared `turn_end` field. Families that require different user and assistant endings must remain unsupported by that representation until a richer typed format is implemented and tested.

### 6.2 Candidate policy

Start from a known-family specification or an explicitly supplied baseline. Generate only catalog-approved changes, such as a BOS policy change, the removal of an unintended BOS, one specified whitespace correction, or a documented open-assistant-prefix variant. The current backbone's small mutation list is a starting design, not a complete family catalog.

Each mutation has a stable ID, an applicability predicate, a clear semantic description, and a maximum expansion count. Combination depth and total candidate count are bounded before execution. Do not use held-out scores to add mutations.

Deduplicate in two stages:

1. Canonical format-semantic equivalence, including no-op BOS toggles when BOS is absent.
2. Rendered-token equivalence across the sealed search fixtures, retaining a record of aliases and recognizing that equivalence on those fixtures is not universal equivalence.

Use a deterministic order with the baseline first. If the budget cannot complete the planned fair comparison, reduce the candidate set before starting under a declared rule, or abstain. Do not award a winner to whichever candidate happened to finish first.

### 6.3 Prompt audit

Before inference, expose a preview containing raw string escapes, byte count, token count, token IDs, special-token labels, and the message-to-prompt mapping. For tokenizers whose token pieces do not decode independently, do not present concatenated per-token strings as exact text. Mark token-to-text alignment as unavailable or approximate when necessary.

## 7. Raw generation, boundaries, and continuation probes

### 7.1 Adapter contract

The raw diagnostic adapter accepts explicit input IDs and returns continuation-only IDs. Automatic chat templates, implicit BOS/EOS insertion, backend stop strings, default EOS termination, and response-cleanup behavior must be disabled or proven absent. Generation reaches the approved token cap unless cancellation or an error prevents it.

Adapter conformance must establish the actual runtime behavior. Passing an option named “ignore EOS” or setting an empty stop list is insufficient evidence if the backend still terminates internally. A backend that cannot expose raw continuation or bypass stop processing is unsupported for raw boundary diagnosis; it may later be offered as a separately labeled compatibility mode.

### 7.2 Boundary specification

A `BoundarySpec` should identify one or more permitted token sequences and their meanings, such as assistant turn end or conversation end. The sequence must be verified in the relevant tokenization context. Encoding a delimiter string by itself can produce different IDs from encoding it adjacent to content.

For every boundary probe, preserve:

- The exact boundary token sequence and how it was obtained.
- All match spans, including overlaps and repeated occurrences.
- Whether the sequence appeared at token zero, near the cap, or only as a partial suffix.
- The extraction rule used when multiple boundary types match.
- The complete prefix, matched span, and suffix token IDs.
- Whether expected answer content itself can contain the same sequence.

The initial `first_boundary` helper finds the first exact sequence. A production report must not lose later matches or ambiguity merely because answer extraction uses the first one.

### 7.3 Continuation evidence

Diagnostic generation should retain a fixed approved continuation opportunity after an observed boundary whenever the cap and context permit. A boundary found at the final generated token cannot establish what the model would have emitted next; mark that observation as insufficient continuation coverage.

Classify continuation conservatively using predefined, token-aware criteria: next-role delimiter, repeated assistant start, repeated end marker, additional ordinary content, unexpected control token, or unclassified. These are diagnostic categories, not assertions of security or model intent.

Never make the candidate's chosen stop string both the mechanism that hides the suffix and the evidence that the suffix is correct. Operational stopping for exported production templates is tested separately against the diagnostic raw trace.

### 7.4 Cancellation and failure provenance

Capture cancellation-request time, observed time, stage, any work in flight, partial token IDs, counter certainty, and a terminal event. Record backend errors separately from user cancellation and budget exhaustion. Define deterministic precedence when they race; preserve all observed events even if one reason is primary.

Public reports use fixed error codes. Private diagnostic logs are opt-in, local, redacted, and excluded from normal exports. Cancellation must not turn the last partially completed fixture into a successful complete run. Resume must not secretly replace a failed attempt with a successful one.

## 8. Actual compute budget accounting

### 8.1 Budget dimensions

A generation cap alone is not a compute budget. Record and enforce separate limits for:

- Candidate count and fixture count.
- Actual prefill token positions processed.
- Actual decode steps and emitted continuation tokens.
- Teacher-forced scoring token positions.
- Total model-forward invocations where measurable.
- Context length and KV-cache occupancy estimates.
- Optional wall-clock, resident-memory, and device-memory limits.
- Search, held-out confirmation, parity validation, and diagnostic overhead.

Token counts are workload counters, not hardware-independent FLOPs or cost estimates. Report hardware-specific elapsed time separately. Never equate equal generated-token caps with equal compute when prompt lengths or caching policies differ.

### 8.2 Reservation and charge protocol

Before each model call, reserve a conservative upper bound from the relevant phase budget. The reservation uses tokenized prompt length, requested decode cap, scoring span, known context limit, and cache policy. Reject or defer the call if that bound does not fit.

After the call, settle the reservation with an adapter receipt reporting actual work. Refund only work known not to have happened. If a failure prevents exact measurement, retain the conservative charge or mark the interval as uncertain; do not charge zero. Uncertain accounting can block a strict-budget claim.

Retries receive new attempt IDs and new charges linked to the original failure. Automatic retry policy is fixed in the manifest and bounded. Retrying to obtain a favorable answer is prohibited.

### 8.3 Prefill and decode definitions

Define prefill as token positions actually passed through the model to establish or rebuild context. Define decode computation separately from emitted tokens. With an ordinary cached autoregressive loop, the initial prefill predicts the first continuation token, and subsequent decode forwards process earlier generated tokens to predict later ones. Therefore the number of emitted tokens and one-token decode-forward calls may differ by one. Record both rather than assuming equality.

If the adapter recomputes the whole sequence at each generation step, charge all processed positions as actual forward work and label the no-cache policy. For an uninterrupted greedy call with prompt length `L` and `K` emitted tokens, the current draft's full-prefix strategy submits sequence lengths `L, L + 1, ..., L + K - 1`: a total of `K * L + K * (K - 1) / 2` forward token positions. This is an analytic workload count for that algorithm, not a measurement of FLOPs or elapsed time. Partial calls and failures need receipts recording the forwards actually performed. Do not report the prompt as prefilling only once. Prefix-cache hits require a verifiable reused-prefix length, cache identity, and invalidation policy; a nominal hit is not enough to claim saved work.

For a run, reconcile at minimum:

- `actual_prefill_positions = sum(receipt.prefill_positions)`.
- `actual_decode_positions = sum(receipt.decode_positions)`.
- `emitted_continuation_tokens = sum(length(raw_output_ids))`.
- `actual_scoring_positions = sum(receipt.scoring_positions)`.
- `actual_forward_positions = sum(all disjoint charged forward-position categories)`.

The categories must be disjoint by definition so that scoring or full-context recomputation is neither omitted nor counted twice. If a backend cannot expose a fine-grained split, retain a coarser actual counter and label the unavailable breakdown.

### 8.4 Overheads and interrupted calls

Warmup, conformance probes, token-alignment checks that require inference, and export parity inference consume resources. Either include them in the operator's overall cap or expose a separately approved overhead budget before they start. CPU tokenization and report rendering may have their own time and memory counters but must not be mislabeled as model inference.

When batching or speculative decoding is introduced, record padding work, accepted and rejected draft tokens, batch size, and the counter method. The first release should prefer a simpler auditable path over an unmeasured optimization. A cancellation that returns only some output may still have consumed additional device work; reconcile through synchronization/counters where possible and mark the remainder uncertain.

### 8.5 Budget acceptance gate

Use an instrumented deterministic backend to enumerate every call and reconcile the ledger exactly. Inject failures before prefill, during prefill, during decode, during scoring, after a backend return, and during receipt persistence. Demonstrate no overscheduling under concurrent reservations. Repeat on each real adapter and document which counters are exact, measured, bounded, or unavailable.

## 9. Scoring and alignment

### 9.1 Separate evidence dimensions

Report at least four independent dimensions:

1. Task behavior, such as exact match on a narrowly defined fixture.
2. Boundary behavior, including exact boundary presence, placement, and ambiguity.
3. Continuation behavior after the boundary.
4. Runtime and export fidelity, including raw adapter conformance and reload parity.

A candidate can answer correctly with the wrong termination behavior, or terminate cleanly while answering incorrectly. Do not collapse those cases into a single success flag. If a composite search score is introduced, publish its formula, weights, exclusions, and failure floors before running it, and show its components alongside it.

### 9.2 Answer extraction and normalization

Preserve the original backbone's strict exact-match scorer as one clearly labeled scorer. It compares the decoded pre-boundary answer without trimming or Unicode normalization. Additional scorers need explicit versions and task-specific rules. Never silently change the displayed answer or scoring denominator to make a candidate look better.

Specify behavior for no boundary, an empty answer, multiple boundaries, a boundary at token zero, a partial boundary at the cap, non-text output, and decode failure. An invalid or unavailable score is distinct from a legitimate score of zero.

### 9.3 Likelihood alignment

For teacher-forced target scoring, build a single explicit full token sequence and identify which token positions belong to the answer. In a causal language model, the logits at position `t - 1` predict the token at position `t`. Shift labels once and mask prompt, padding, and any non-scored control tokens according to the declared policy.

Do not assume that `encode(prompt) + encode(answer)` equals `encode(prompt + answer)`. Boundary-crossing merges can invalidate that assumption. Choose and record a supported alignment strategy:

- An exact token-boundary construction justified by the format and tokenizer.
- An offset-mapping strategy with validated byte/code-point semantics and no ambiguous cross-boundary token.
- A refusal to produce likelihood for the fixture when neither method yields a defensible span.

If one token straddles prompt and answer text, do not silently assign all of it to whichever side is convenient. Reject the alignment or apply a separately justified policy with an explicit comparability warning. Empty targets, tokenizer-added tokens, and target terminators require dedicated cases.

### 9.4 Score comparability

Record whether a score is a total log-likelihood, mean negative log-likelihood per scored token, per-byte measure, or task accuracy. These quantities answer different questions. Across candidates, token counts can differ even for the same visible answer. Report target byte length, scored token count, and the exact span with any normalized likelihood.

Avoid cross-tokenizer likelihood comparisons unless a method is specified and justified. A model/tokenizer pair is the natural unit for within-run candidate comparisons. Never include prompt-token likelihood in an answer-only metric by accident. Detect NaN, infinity, underflow-sensitive conversions, empty denominators, and impossible vocabulary indexes before aggregation.

### 9.5 Alignment acceptance gate

Construct tiny deterministic logit tensors for which every target token's probability is known by hand. Test one-token and multi-token prompts and answers, attention masks, padding on both sides where supported, special tokens, empty targets, and merge-sensitive boundaries. Cross-check adapter scores against a simple independent reference calculation rather than another wrapper around the same scoring function.

## 10. Split isolation and statistical decisions

### 10.1 Fixture grouping and sealing

Group paraphrases, variants of the same template, shared source examples, and near-duplicate tasks before partitioning. Keep exact duplicate detection as a guard, not a complete leakage detector. Add documented near-duplicate review and group-level dataset provenance.

Seal the dataset and split assignment with a manifest before candidate search. Store held-out expected answers and outcomes behind a phase-specific interface so search code cannot access them. Tokenization checks that inspect held-out content must be limited to preregistered validity checks and must not feed adaptive candidate creation.

A deterministic split seed makes a split reproducible; it does not make it independent if the fixtures were selected after seeing outcomes. The manifest must describe how the dataset was created and whether it was used in earlier development.

### 10.2 Search and freeze

Use search fixtures for candidate ranking and diagnostics. Freeze the selected candidate or predefined shortlist, baseline, score definitions, uncertainty method, minimum coverage, decision thresholds, retry policy, and confirmation seed schedule before the held-out run.

If multiple candidates enter confirmation, predeclare how multiplicity and final choice are handled. The simplest first-release design confirms one chosen candidate against the unchanged baseline. Keep a no-change outcome available when the baseline is adequate or improvements are uncertain.

### 10.3 Paired comparisons and uncertainty

Pair candidate and baseline observations on the same fixtures and seed schedule. Treat related fixtures as clustered observations. A group-aware bootstrap or another justified method should resample independent groups, not pretend that all paraphrases are independent.

Publish the estimand, confidence level, sample size, group count, seed count, and interval method. Explain the limited interpretation of intervals on very small or selected samples. Separate variability across fixtures, seeds, models, and hardware; repeated seeds do not create new independent task groups.

Do not choose a decision threshold after seeing confirmation results. Do not manufacture a “confidence percentage” from a score gap. If the result is underpowered, report the uncertainty and abstain or request more independent data under a new manifest.

### 10.4 Abstention conditions

At minimum, abstain when:

- No candidate completes the required fair coverage.
- The adapter cannot establish raw generation or actual budget behavior.
- Held-out leakage or post-exposure tuning invalidates confirmation.
- Improvement over baseline is below the predeclared materiality criterion or uncertainty overlaps unacceptable regression.
- Task, boundary, and continuation evidence disagree under the decision policy.
- Boundary identity or likelihood alignment remains ambiguous.
- The model/runtime configuration is outside the verified support matrix.
- Export parity fails or cannot express required semantics.
- The budget is exhausted before the decision requirements are met.

Keep “no change needed” distinct from “insufficient evidence.” Both can be useful outcomes, but they imply different next actions.

## 11. Seeds and reproducibility

Use a documented hierarchy: dataset/split seed, candidate-order seed if any, fixture-generation seed, run seed, request seed, and statistical-resampling seed. Derive request seeds from canonical identifiers using a versioned hash scheme. Record the full derivation specification and the effective seed width accepted by each backend.

The backbone currently accepts an unsigned 64-bit run seed and derives a 32-bit request seed from the run seed and fixture ID. Preserve compatibility in regression tests, document the possibility of collisions, and decide whether the production adapter contract needs a wider derived seed. A change in derivation requires a version change.

Keep random number generation request-local when sampling is supported. Candidate order, unrelated requests, and fixture scheduling must not perturb a fixture's random stream. Shared seeds are useful for pairing but do not guarantee identical random consumption across different prompts or runtimes. In the current greedy adapter draft, the seed is validated and recorded but is not used for token selection; repeating greedy runs with different seeds does not constitute a multi-seed sampling experiment. Model initialization can have separate global-RNG effects that need to be documented and tested.

Record library versions, backend build, device category, precision, quantization, attention implementation, cache mode, sampling parameters, deterministic settings, batch size, and relevant environment flags. Do not publish machine usernames, hostnames, serial numbers, or unfiltered environment dumps.

Declare the expected reproducibility level separately:

- Exact prompt bytes and input token IDs.
- Exact format and report serialization.
- Exact output IDs on a fixed deterministic reference stack.
- Numerically close scores under a stated tolerance.
- Statistically comparable behavior across devices where exact generation is not guaranteed.

## 12. Implementation stages and acceptance gates

Task IDs are stable references for issues and pull requests. A task is complete only when its deliverable and validation evidence are available. Stages express dependencies, not calendar estimates.

### Stage 0 Establish the source and evidence baseline

- **BASE-01 Source inventory.** Enumerate archive files, licensing status, entry points, dependencies, optional extras, and each PRESENT SOURCE capability. Deliverable: source manifest and truthful README status. Dependency: none.
- **BASE-02 Claim registry.** Link every documented working feature to a verification record or label it unverified. Deliverable: claim-to-evidence checklist. Dependency: BASE-01.
- **BASE-03 Clean packaging.** Establish an allowlisted source archive layout, ignore rules, public synthetic fixtures, and secret/privacy review procedure. Deliverable: packaging specification and release exclusion list. Dependency: BASE-01.
- **BASE-04 Preserve regressions.** Keep the original backbone/test draft available as an auditable reference while moving behavior into the package. Deliverable: migration mapping from original symbols/tests to package equivalents. Dependency: BASE-01.

**Gate G0:** The source bundle is self-describing, contains no claimed runtime evidence that does not exist, and identifies every optional dependency and unsupported feature. A later archive build must verify cleanliness on the exact bytes being distributed.

### Stage 1 Define contracts and pure core behavior

- **CORE-01 Domain schemas.** Implement the versioned records in Section 5, strict validation, canonical serialization, and schema migration policy. Deliverable: schema files and validation tests. Dependency: G0.
- **CORE-02 Renderer parity.** Port and test exact single/multi-turn rendering, repeated BOS semantics, role-order validation, literal-content preservation, and unsupported-role errors. Deliverable: golden prompt-byte fixtures. Dependency: CORE-01.
- **CORE-03 Candidate catalog.** Add bounded mutations, applicability rules, stable IDs, deterministic ordering, and semantic deduplication. Deliverable: catalog schema and a small reviewed initial catalog. Dependency: CORE-02.
- **CORE-04 Boundary engine.** Support exact sequences, all match spans, overlap handling, ambiguity flags, and lossless suffix evidence. Deliverable: boundary conformance corpus. Dependency: CORE-01.
- **CORE-05 State integrity.** Implement complete/incomplete eligibility, stage-specific fixed errors, transition validation, and immutable event records. Deliverable: state-machine tests and report-validator tests. Dependency: CORE-01.

**Gate G1:** Pure-core tests run without inference dependencies. Golden inputs produce exact expected bytes and IDs through the scripted reference backend. Invalid states and unsupported semantics fail closed. These results establish software behavior only.

### Stage 2 Build a safe local inference adapter

- **ADAPT-01 Capability interface.** Separate tokenization, raw generation, scoring, cancellation, and accounting capabilities. Deliverable: backend contract and reusable conformance harness. Dependency: G1.
- **ADAPT-02 Local loading.** Resolve approved local artifacts, restrict weight formats, disable remote code and downloads, enforce path/size rules, and fingerprint loaded files. Deliverable: loader plus denial tests. Dependency: ADAPT-01.
- **ADAPT-03 Raw decode loop.** Implement explicit sampling with preserved token IDs, disabled implicit wrapping/stopping, context-limit checks, and request-local RNG. Deliverable: adapter source and raw-trace fixtures. Dependency: ADAPT-02.
- **ADAPT-04 Cancellation.** Check cancellation at documented inference boundaries and preserve partial results. Deliverable: interruption tests and measured cancellation behavior for each verified backend/device. Dependency: ADAPT-03.
- **ADAPT-05 Backend compatibility.** Pin an initially supported library range and reject untested behavior changes conservatively. Deliverable: environment lock and adapter support matrix. Dependency: ADAPT-03.

**Gate G2:** At least one explicitly identified local model/tokenizer configuration passes the adapter conformance harness. Loading makes no network request, no checkpoint code executes, raw generation reaches its cap when expected, and cancellation/failure traces remain honest. Hardware-specific measurements are reported only after they exist.

### Stage 3 Make resource use auditable

- **BUDGET-01 Ledger and receipts.** Implement reservations, actual charges, conservative uncertain charges, phase budgets, and reconciliation. Deliverable: budget schema and exact instrumented-backend tests. Dependency: CORE-05 and ADAPT-01.
- **BUDGET-02 Adapter counters.** Count actual prefill/decode/scoring forwards, emitted tokens, cache effects, and failures. Deliverable: adapter receipts and counter-quality declaration. Dependency: BUDGET-01 and ADAPT-03.
- **BUDGET-03 Fair scheduling.** Validate a comparison plan before execution and prevent incomplete candidate coverage from becoming a winner. Deliverable: scheduler and budget-exhaustion tests. Dependency: BUDGET-01 and CORE-03.
- **BUDGET-04 Crash reconciliation.** Persist reservations and receipts atomically enough to recover a conservative ledger after interruption. Deliverable: crash/restart test cases and recovery rules. Dependency: BUDGET-02.

**Gate G3:** All inference and scoring work is represented in the ledger, including unsuccessful attempts. Concurrent scheduling cannot overspend the declared reservable cap. Reports distinguish exact actuals from bounds and unknowns.

### Stage 4 Establish valid probes and scores

- **PROBE-01 Fixture schema.** Add task/scorer versions, group provenance, expected answer representation, and sensitivity labels. Deliverable: small synthetic public corpus plus fixture authoring guide. Dependency: CORE-01.
- **PROBE-02 Diagnostic families.** Implement the boundary, continuation, role-order, whitespace, Unicode, and multi-turn probes in the test matrix. Deliverable: probe catalog with what each probe can and cannot establish. Dependency: PROBE-01 and CORE-04.
- **SCORE-01 Exact and structural scores.** Keep answer accuracy, boundary evidence, continuation categories, and operational failures separate. Deliverable: scorer definitions and independent oracle tests. Dependency: PROBE-02.
- **SCORE-02 Likelihood alignment.** Implement exact target masking and refusal on unsupported tokenization boundaries. Deliverable: per-token score traces and analytic-reference tests. Dependency: ADAPT-01 and PROBE-01.
- **SCORE-03 Numerical validation.** Verify dtype behavior, finite values, tolerances, and aggregation denominators. Deliverable: adversarial numerical tests and documented tolerances. Dependency: SCORE-02.

**Gate G4:** Every score is reproducible from its recorded inputs, mask, and scorer version. Alignment failures and missing scores are visible. No score uses prompt tokens, stripped suffixes, or normalized answers contrary to its definition.

### Stage 5 Separate search from confirmation

- **EXP-01 Grouped splits.** Implement exact duplicate guards, group-aware partitioning, near-duplicate review hooks, and sealed split manifests. Deliverable: leakage regression suite. Dependency: PROBE-01.
- **EXP-02 Freeze protocol.** Create immutable search results and a freeze record before confirmation access. Deliverable: search/held-out API separation and tamper tests. Dependency: EXP-01 and BUDGET-03.
- **EXP-03 Selection policy.** Implement baseline comparison, deterministic tie handling, materiality thresholds, and no-change/abstain outcomes. Deliverable: decision-policy specification and simulations with known outcomes. Dependency: SCORE-01 and EXP-02.
- **EXP-04 Uncertainty.** Add paired, group-aware estimates and preregistered interval/multiplicity methods. Deliverable: statistical method notes and reference-dataset tests. Dependency: EXP-03.
- **EXP-05 Confirmation lock.** Prevent tuning after held-out exposure within the same confirmation claim. Deliverable: run ancestry rules and UI/CLI enforcement tests. Dependency: EXP-04.

**Gate G5:** Search cannot read held-out labels/outcomes. A result can be reconstructed from the frozen policy and evidence. Known null, tie, regression, insufficient-coverage, and contradictory-evidence scenarios do not produce spurious recommendations.

### Stage 6 Make evidence inspectable

- **REPORT-01 Portable reports.** Implement validated JSON summaries and JSONL attempt/event records with linked digests. Deliverable: schema examples and malformed-report corpus. Dependency: CORE-05, G3, and G4.
- **REPORT-02 Human-readable summaries.** Render limitations, scope, actual budgets, incomplete attempts, and abstention reasons prominently. Deliverable: CLI summary snapshots and report examples. Dependency: REPORT-01 and G5.
- **VIEW-01 Read-only viewer.** Load bounded local reports, display tokens and spans safely, support filtering without altering denominators, and expose raw/redacted status. Deliverable: local HTML viewer and UI test fixtures. Dependency: REPORT-01.
- **VIEW-02 Accessibility.** Add keyboard operation, focus management, semantic structure, text alternatives, contrast checks, and reduced-motion behavior. Deliverable: accessibility audit and documented manual checks. Dependency: VIEW-01.
- **VIEW-03 Optional run controls.** If a local API is introduced, separate it from the report viewer and add explicit job state, cancellation, and stale-result protection. Deliverable: loopback API threat model and integration tests. Dependency: VIEW-01, G2, and G3.

**Gate G6:** The viewer accurately represents complete and incomplete reports, escapes hostile content, remains usable by keyboard, and never converts an imported report into executable behavior. The UI's figures match the canonical report rather than separately computed approximations.

### Stage 7 Export and verify target parity

- **EXPORT-01 Canonical format artifact.** Define a versioned data-only format with complete role, BOS, separator, assistant-prefix, and boundary semantics. Deliverable: canonical format schema and round-trip tests. Dependency: G1.
- **EXPORT-02 First target adapter.** Choose one target format that can express the initial supported subset without loss. Deliverable: exporter, importer or target-runtime loader, and explicit compatibility limits. Dependency: EXPORT-01 and G2.
- **EXPORT-03 Golden prompt parity.** Compare source and reloaded target prompt bytes and token IDs across golden conversations. Deliverable: parity corpus and exact diff output. Dependency: EXPORT-02.
- **EXPORT-04 Stop and continuation parity.** Compare diagnostic raw traces with the target's operational termination and retained answer. Deliverable: termination-semantics tests and an honest parity report. Dependency: EXPORT-03 and G4.
- **EXPORT-05 Failure behavior.** Reject lossy exports, version mismatch, unsupported roles, ambiguous stop strings, and missing tokenizer identity. Deliverable: negative export tests. Dependency: EXPORT-02.

**Gate G7:** A supported export reloads to equivalent prompt IDs and documented termination behavior for the supported fixture corpus. Any generation comparison uses a fixed compatible runtime or a declared numerical/stochastic tolerance. Unsupported targets remain marked unverified or unavailable.

### Stage 8 Run independent real-model validation

- **MODEL-01 Support registry.** Select licensed local checkpoints covering the initial protocol families and record immutable model/tokenizer revisions. Deliverable: model support registry. Dependency: G2 and G4.
- **MODEL-02 Blind confirmation corpus.** Assemble independent grouped fixtures not used to implement the candidate policy. Deliverable: sealed corpus and provenance review. Dependency: G5.
- **MODEL-03 Cross-configuration tests.** Run supported precision/device/cache configurations and record differences without generalizing beyond tested combinations. Deliverable: reproducibility report. Dependency: MODEL-01 and G3.
- **MODEL-04 Failure analysis.** Review false positives, false negatives, abstentions, regressions, boundary failures, and incompatible exports. Deliverable: categorized error analysis with representative public-safe examples. Dependency: MODEL-02 and MODEL-03.
- **MODEL-05 Support promotion.** Mark only passing cells VERIFIED and attach evidence. Deliverable: published support table and known limitations. Dependency: MODEL-04 and G7.

**Gate G8:** At least the advertised configurations have complete evidence for adapter, task, boundary, continuation, budget, and export behavior. The release does not infer support for untested variants sharing a model-family name.

### Stage 9 Compare against simpler methods

- **BENCH-01 Related-work map.** Verify primary sources and pin the versions and applicable features of ChatLint, llama.cpp autoparser, gbench, Ingot, and FormatSpread. Deliverable: citation-backed scope comparison with no unsupported novelty claim. Dependency: G0.
- **BENCH-02 Benchmark protocol.** Preregister workloads, candidate limits, budgets, baselines, metrics, seeds, hardware, statistical analysis, and exclusion rules. Deliverable: benchmark manifest. Dependency: G5 and MODEL-01.
- **BENCH-03 Controlled comparisons.** Run native-template, metadata-only, no-change, simple delimiter/BOS correction, and bounded-search baselines under comparable conditions. Deliverable: raw reports and resource receipts. Dependency: BENCH-02 and G8.
- **BENCH-04 Ablations.** Remove selected components one at a time to test their contribution. Deliverable: ablation report with uncertainty and failures. Dependency: BENCH-03.
- **BENCH-05 Reproduction.** Repeat the procedure from a clean environment using published public artifacts or clear local-weight instructions. Deliverable: independent reproduction record. Dependency: BENCH-04.

**Gate G9:** Comparative statements cite measured results and their exact scope. The report includes no-change wins, abstentions, regressions, incompatibilities, and total resource use. If a named tool cannot be compared fairly, explain that limitation rather than assigning it a fabricated score.

### Stage 10 Release and maintain

- **REL-01 CI tiers.** Implement the CI policy in Section 19 with offline default tests and opt-in model tests. Deliverable: CI configuration and reproducible environments. Dependency: G1 onward, completed before release.
- **REL-02 Security review.** Test loading, imports, local API boundaries, viewer input handling, exports, dependencies, and archive cleanliness. Deliverable: threat-model review and remediated findings. Dependency: G6 and G7.
- **REL-03 Source release.** Produce an allowlisted archive, checksums, license notices, installation instructions, support matrix, and verification status. Deliverable: clean release artifact. Dependency: G8 and REL-02.
- **REL-04 Maintenance policy.** Define version compatibility, catalog review, fixture rotation, issue templates, deprecation, and regression response. Deliverable: contribution and maintenance documentation. Dependency: REL-03.

**Gate G10:** A clean-machine source installation and documented workflow pass for the advertised release tier. The exact distributed archive passes integrity and privacy checks. Research comparisons may remain experimental if the release makes no unsupported performance claim.

### 12.1 Critical path and parallel work

The critical path is G0 -> G1 -> G2/G3/G4 -> G5 -> G7/G8 -> G10. Reporting and the read-only viewer can progress against validated synthetic reports after schemas stabilize. Security and accessibility work begins during design rather than waiting for the final gate. Related-work verification can proceed independently but must finish before any comparative claim.

Do not begin real-model recommendation runs until raw-adapter conformance, accounting, and scoring alignment are sufficiently established to interpret them. A visually complete viewer is not a substitute for those dependencies.

## 13. Detailed test matrix

Tests below are planned validation requirements. Each test record should link its requirement ID, fixture/input, expected outcome, actual outcome, source revision, and environment. The original 45 test methods are an initial regression draft and do not replace the broader matrix.

### 13.1 Pure unit and contract tests

| ID | Area | Required cases and expected behavior |
| --- | --- | --- |
| UNIT-01 | Message validation | Accept literal user/assistant strings; reject unsupported roles, non-string content, malformed containers, and invalid order |
| UNIT-02 | Format validation | Reject blank names, wrong types, ambiguous schema versions, unsupported fields, and invalid role policies |
| UNIT-03 | Exact rendering | Golden bytes for single-turn, multi-turn, empty content, tabs, trailing spaces, CRLF, and Unicode |
| UNIT-04 | BOS semantics | Once/repeated/absent behavior, including the final open assistant prefix, remains explicit |
| UNIT-05 | Content preservation | Delimiter-looking content, braces, markup, and escape sequences remain literal and cannot execute |
| UNIT-06 | Mutation bounds | Maximum count/depth, deterministic order, no baseline mutation, applicability predicates, and deduplication |
| UNIT-07 | Boundary matching | Absent, first-position, last-position, repeated, overlapping, multi-token, longer-than-output, and partial-suffix sequences |
| UNIT-08 | Answer extraction | Correct half-open spans; raw output and suffix retained even when extracted answer is empty |
| UNIT-09 | Token validation | Reject negative IDs, booleans, floats, huge values, wrong containers, and vocabulary-out-of-range IDs |
| UNIT-10 | Generation contracts | Exact-cap success; short/long success invalid; cancelled/error partial outputs retained consistently |
| UNIT-11 | Error sanitization | Exceptions containing path, prompt, or secret-like strings cannot enter public error fields |
| UNIT-12 | Report integrity | Planned/attempted/completed counts reconcile; last-attempt failure never becomes complete |
| UNIT-13 | Split guards | Duplicate IDs, cross-group overlap, exact duplicate prompts, and malformed partitions rejected |
| UNIT-14 | Seed derivation | Boundaries, Unicode IDs, fixed golden seeds, order independence, seed-width mapping, and version changes |
| UNIT-15 | Serialization | Canonical hashes, stable ordering, null/empty distinctions, strict finite numbers, and migration errors |
| UNIT-16 | Scorer semantics | Exact match without hidden trimming; missing/invalid score distinct from zero |
| UNIT-17 | Score alignment | Hand-calculated causal shift, masks, target terminator inclusion, and boundary-crossing token refusal |
| UNIT-18 | Budget receipts | Reservations, settlement, uncertain charges, retries, refunds, and phase totals reconcile |
| UNIT-19 | Decision policy | Baseline wins, tie, candidate win, regression, contradictory metrics, and every abstention code |
| UNIT-20 | State transitions | Reject impossible transitions, duplicate terminal events, mutable freeze records, and replayed stale events |

### 13.2 Property and metamorphic tests

| ID | Property | Required invariant |
| --- | --- | --- |
| PROP-01 | Render determinism | Identical immutable inputs produce identical bytes without modifying messages or format |
| PROP-02 | Boundary oracle | Optimized matcher equals a simple independent exhaustive matcher on generated token sequences |
| PROP-03 | Candidate aliases | Deduplication preserves at least one representative and does not change baseline semantics |
| PROP-04 | Seed order independence | Permuting fixture execution changes report order only, not per-fixture request seeds |
| PROP-05 | Split disjointness | Arbitrary grouped datasets cannot place one group into both partitions |
| PROP-06 | Evidence preservation | Extracting or displaying an answer cannot change raw IDs, counts, or hashes |
| PROP-07 | Budget conservation | Charges plus remaining reservations and available budget follow the declared accounting identity |
| PROP-08 | Cancellation monotonicity | Once cancellation is observed, no new inference call starts in that run |
| PROP-09 | Canonical round trip | Supported records serialize and reload without semantic change |
| PROP-10 | Export equivalence | Generated supported conversations produce identical input IDs before and after export reload |
| PROP-11 | Viewer filtering | Filtering/sorting never changes canonical scores, denominators, status, or raw evidence |
| PROP-12 | Isolation | Mutating held-out labels cannot change search candidate selection or search report bytes |

Use fixed reproduction seeds for property-test failures and retain minimized public-safe counterexamples. Do not call a property established merely because a finite randomized suite passed; state its tested bounds.

### 13.3 Mutation testing

Introduce deliberate faults that a trustworthy suite must detect:

- **MUT-01:** Remove the duplicate-input or group-overlap split check.
- **MUT-02:** Strip trailing whitespace before exact match.
- **MUT-03:** Drop special tokens in decode.
- **MUT-04:** Stop raw generation at the candidate boundary or EOS.
- **MUT-05:** Accept a short `limit` result as complete.
- **MUT-06:** Include prompt IDs in continuation IDs.
- **MUT-07:** Shift likelihood labels by zero or two positions rather than one.
- **MUT-08:** Include prompt or padding tokens in the score denominator.
- **MUT-09:** Ignore repeated prefill or retry work in the budget ledger.
- **MUT-10:** Permit a cancelled final attempt to satisfy complete coverage.
- **MUT-11:** Reuse global RNG state across requests.
- **MUT-12:** Select a candidate using held-out scores before freezing.
- **MUT-13:** Export one separator differently from the in-memory renderer.
- **MUT-14:** Render imported model output as HTML.
- **MUT-15:** Forward a backend exception string into a public report.

Review surviving mutations individually. Do not chase a mutation percentage by adding assertions unrelated to real invariants. Critical faults above must be caught before their associated feature is promoted.

### 13.4 Integration and fault injection

| ID | Scenario | Required outcome |
| --- | --- | --- |
| INT-01 | Clean core installation | Report validation and pure-core tests work without inference extras |
| INT-02 | Optional adapter installation | Explicitly compatible dependency set loads only when requested |
| INT-03 | Offline execution | Block network access and complete the supported local workflow without hidden fetches |
| INT-04 | Missing or unsafe weights | Refuse absent artifacts, unsupported serialization, and remote-code requirements with useful fixed errors |
| INT-05 | Context overflow | Reject before inference; never truncate silently |
| INT-06 | Cancellation checkpoints | Before render, after encode, during prefill, during decode, after return, during scoring, and during save |
| INT-07 | Failure precedence | Simultaneous error/cancel/budget events retain provenance and the declared primary status |
| INT-08 | Persistence interruption | Kill between reservation/call/receipt/save and recover conservatively without duplicate evidence |
| INT-09 | Retry policy | Transient failures obey the fixed retry limit and charge every attempt |
| INT-10 | Search isolation | Instrument held-out access and fail any pre-freeze read by selection code |
| INT-11 | CLI report flow | Validate -> run -> inspect -> export -> reload uses matching hashes and statuses |
| INT-12 | Viewer report flow | Complete, failed, partial, redacted, old-version, malformed, and oversized reports display or reject correctly |
| INT-13 | Local API isolation | Loopback binding, origin validation, CSRF defenses, file scope, and cancellation authorization |
| INT-14 | Cross-platform files | Portable paths, atomic writes, Unicode filenames, and newline handling where supported |
| INT-15 | Export failures | Unsupported target semantics produce explicit refusal, never a plausible-looking lossy template |
| INT-16 | Artifact mismatch | Wrong tokenizer, changed weights, changed scorer, or changed split invalidates replay or support claims |

### 13.5 Real-model protocol tests

Use multiple licensed model/tokenizer pairs selected for protocol diversity, not a favorable aggregate score. The minimum investigation should include:

- A tokenizer with ordinary text separators and one with dedicated special tokens.
- A family with one initial BOS and a family whose documented format uses another BOS policy.
- A format with multi-token delimiters or context-sensitive delimiter tokenization.
- Single-turn and multi-turn conversations with preserved earlier assistant content.
- Known-good native templates and intentionally perturbed versions of those same templates.
- Models that follow tasks but produce an unexpected boundary, and models that terminate correctly but fail tasks.
- Long prompts near the supported context limit, short prompts, empty content, and delimiters embedded in quoted user text.
- Deterministic decoding and a separately declared sampling configuration with multiple seeds.
- Supported precision/quantization variants treated as distinct matrix cells.

Record failures and abstentions as first-class results. Do not remove hard models after outcomes are visible unless the preregistered exclusion rule applies, and then report the exclusion and reason.

## 14. Supported model and capability matrix

The first published matrix should be conservative. Until runtime evidence exists, every proposed model configuration remains UNVERIFIED. Do not populate a table with checkmarks based solely on model-family familiarity or package import success.

### 14.1 Initial scope matrix

| Capability | Initial target | Status before validation |
| --- | --- | --- |
| Local causal text models | Draft whitelist: built-in GPT-2/Llama types, local safetensors, self-contained fast tokenizer | PRESENT SOURCE adapter draft; runtime unverified |
| User/assistant alternation | Conversations ending in a user turn with an open assistant prefix | PRESENT SOURCE draft; runtime unverified |
| Known-family bounded mutations | ChatML/Llama 3/Gemma text hypotheses and bounded local BOS/prefix changes | PRESENT SOURCE draft subset; no checkpoint compatibility established |
| Raw capped generation | No implicit wrapper or automatic EOS/stop handling | Adapter contract/draft only; unverified |
| Exact token-boundary evidence | Nonempty explicit token sequences | PRESENT SOURCE draft; runtime unverified |
| Teacher-forced likelihood | Exact answer-token alignment | PLANNED |
| Search and held-out workflow | Unique all-search-pass choice followed by one selected-candidate held-out evaluation | PRESENT SOURCE draft; no confidence or comparative-improvement guarantee |
| Results viewer | Escaped static HTML heatmap generated from a result | PRESENT SOURCE draft; runtime and browser behavior unverified |
| Sampling | Request-local seeded non-greedy decoding | PLANNED; current adapter is greedy |
| System messages | Family-specific typed semantics | UNSUPPORTED until separately implemented |
| Assistant prefill | Partial assistant content with correct continuation semantics | UNSUPPORTED initially |
| Tool calls and structured channels | Full role/channel/argument semantics | UNSUPPORTED initially |
| Multimodal inputs | Image/audio/video token and payload semantics | UNSUPPORTED initially |
| Remote-code checkpoints | Checkpoint-supplied executable implementation | UNSUPPORTED |
| Hosted APIs with hidden preprocessing | Unknown wrapper/tokenization/stopping | UNSUPPORTED for raw-protocol diagnosis |
| Cross-runtime export | Only named target/runtime versions with parity evidence | PLANNED |

### 14.2 Per-configuration registry fields

For each promoted cell, publish the model identifier and immutable revision, tokenizer digest, architecture, parameter/quantization description, license reference, backend and dependency versions, device/precision/cache configuration, supported role subset, boundary token policy, conformance result, real-model fixture result, export target/version, parity result, known limitations, and evidence artifact digest.

A change to weights, tokenizer, special-token map, backend stopping behavior, cache semantics, precision, or export target can invalidate a cell. Support promotion must be explicit. “Same family” is not an automatic compatibility rule.

## 15. Local security and privacy

### 15.1 Threat model

Treat checkpoints, tokenizer metadata, format catalogs, imported reports, fixture text, generated output, file names, and browser content as untrusted data. Protect the local filesystem, process environment, model integrity, report integrity, and any private prompts. A local application can still expose data through a network fetch, malicious HTML, permissive API binding, or an unsafe loader.

### 15.2 Safe model and tokenizer loading

- Require explicitly selected local artifacts and enforce offline/local-only loading in the supported adapter.
- Disable remote code execution and reject checkpoints or tokenizers that require it.
- Prefer safe, non-executable weight formats; do not silently fall back to pickle-based loading.
- Validate resolved paths, symlink behavior, file types, artifact sizes, and expected manifests before loading.
- Never search arbitrary directories for credentials or automatically use credentials from the environment.
- Treat tokenizer regexes, metadata, and large vocabularies as potential resource-exhaustion inputs; enforce limits and isolate work where practical.
- Pin dependencies and record their versions. Safe weight serialization does not eliminate vulnerabilities in parsers, native libraries, or the surrounding inference stack.

### 15.3 Local API and browser controls

If the product has a server, bind only to loopback by default. Validate origin and host headers, prevent cross-site request forgery and DNS-rebinding-style access, and require a per-session access mechanism for state-changing operations. Avoid permissive CORS. Imported reports must not choose file paths or instruct the server to open arbitrary resources.

Render all prompt/model/report text as text, with explicit escaping. Do not use untrusted HTML, script evaluation, inline event handlers, or unsanitized URL schemes. Prefer bundled static assets, restrictive content security policy, and no third-party fonts, analytics, CDNs, or remote images in the default viewer.

### 15.4 Data retention and export controls

Default to local, user-controlled storage with clear retention options. Separate raw evidence from public summaries and diagnostic logs. Before an export, show whether it includes fixture text, answers, raw token IDs, model identifiers, artifact hashes, private paths, or error details.

Raw token IDs can reveal text and must be treated as sensitive when the source text is sensitive. A report with hidden text but retained token IDs is not necessarily redacted. Provide deliberate export profiles and verify that redaction removes all equivalent representations and attached files.

### 15.5 Secret prevention and public-source hygiene

Maintain an allowlist for release contents and an exclusion list for local configuration, environment files, tokens, keys, caches, downloaded models, private fixture sets, run outputs, logs, editor state, operating-system metadata, and repository history. Run secret scanning and manual privacy review on the exact staged source and exact release archive.

Use clearly synthetic placeholders in examples. Do not include a real secret to prove a scanner works; use test patterns recognized by the scanner. Public test fixtures must not contain copied private conversations or personal identifiers. Hashes, screenshots, HTML reports, and filenames also need review.

## 16. Interface correctness and accessibility

### 16.1 Views

The UI should have a small set of distinct views:

1. Model and tokenizer identity, capabilities, and support status.
2. Format preview and candidate diffs.
3. Fixture groups and sealed split summary.
4. Run configuration, phase budgets, and actual consumption.
5. Attempt evidence with raw token IDs and exact boundary spans.
6. Frozen decision and held-out confirmation.
7. Export compatibility and reload parity.

The HTML results viewer can implement the read-only evidence views before interactive orchestration exists. Its title and status banner must make that scope clear.

### 16.2 Honest presentation

- Show PRESENT SOURCE, VERIFIED, PLANNED, and unsupported capability states in text, not color alone.
- Put incomplete status, budget exhaustion, alignment failure, redaction, and abstention near the result they qualify.
- Show planned, attempted, completed, failed, and excluded counts with the correct denominator.
- Keep search and held-out results visually distinct and explain when a candidate was frozen.
- Preserve raw output; provide answer-only and token views as selectable projections.
- Distinguish an exact token match from a visually similar delimiter string.
- Display the selected boundary span and trailing continuation, including when there is not enough suffix coverage.
- Distinguish estimated/reserved/actual/uncertain resource figures.
- Never label a progress bar “complete” merely because all scheduled calls returned if the decision or parity gate failed.

### 16.3 Interaction and accessibility tests

Support full keyboard navigation, visible focus, properly associated labels, meaningful button names, predictable tab order, screen-reader summaries of status changes, and sufficient contrast. Do not require hover or color to understand token evidence. Respect reduced motion and zoom/reflow.

Long token streams need accessible summaries and bounded rendering. Virtualization must not make essential evidence unreachable to assistive technology. Provide a plain-text or downloadable evidence alternative. Test empty tables, very long lines, Unicode, bidirectional text, malicious markup, small screens, and high zoom.

For interactive runs, prevent duplicate submissions, stale report replacement, accidental held-out reruns after tuning, and a cancellation button that only changes the display. Reflect actual backend state and show when cancellation has been requested but not yet observed.

## 17. Export formats and round-trip parity

### 17.1 Export tiers

- **Canonical data format:** the authoritative versioned `FormatSpec`, capability restrictions, model/tokenizer identity, and boundary semantics.
- **Evidence report:** JSON summary plus optional JSONL attempts/events and raw evidence, subject to deliberate privacy choices.
- **Human-readable report:** local HTML or text summary generated from canonical data with no different scoring rules.
- **Target-runtime template:** a target-specific artifact, such as a compatible template/configuration representation, only for a verified subset and named runtime version.

The first target should be chosen for faithful expressiveness and testability. Do not promise every Jinja chat template, GGUF metadata representation, or serving-runtime stop API can express every canonical format. Preserve “unsupported” as an exporter result.

### 17.2 Required parity checks

1. Serialize and reload the canonical format without semantic change.
2. Render golden conversations with both the source renderer and target runtime.
3. Compare exact bytes where the target representation is text-based.
4. Compare input token IDs under the same tokenizer and explicit special-token options.
5. Verify open-assistant prefix, BOS placement, per-role endings, and context accounting.
6. Verify target stop behavior against raw diagnostic output, including overlapping/multi-token boundaries and suffix preservation in the diagnostic report.
7. Check supported multi-turn cases, empty content, embedded delimiter spellings, Unicode, and whitespace edge cases.
8. Run a controlled generation comparison only where runtime and determinism assumptions justify it; otherwise report the limited parity established.
9. Reject or clearly mark a target artifact when required semantics are dropped or changed.

Byte equality and token-ID equality establish prompt-construction parity, not necessarily generation equality across different kernels or runtimes. The report must state which layer passed.

### 17.3 Round-trip artifact contents

Include schema version, source format hash, target runtime/version, tokenizer identity, supported role subset, boundary definitions, files and checksums, privacy profile, verification procedure, and parity results. Exclude local absolute paths and credentials. Re-import rejects incompatible versions rather than guessing a migration.

## 18. Reproducible benchmarking and related work

### 18.1 Related-work treatment

ChatLint, llama.cpp autoparser, gbench, Ingot, and FormatSpread are comparison and design-review candidates. Their current capabilities, assumptions, licenses, and relevant publications/repositories must be checked against primary sources under BENCH-01. This plan does not attribute unverified features to them or claim that Model Protocol Lab is the first tool to address any of these problems.

The comparison should answer practical questions: which layer a tool evaluates, whether it exposes raw IDs, how it handles boundaries and stopping, whether it searches formats, how it prevents evaluation leakage, and what artifacts it exports. A scope difference is not a measured advantage.

### 18.2 Baselines

At minimum compare against:

- The unchanged baseline format.
- The checkpoint's documented/native template, when safely available and compatible.
- A metadata-only catalog lookup with no model-guided repair.
- A small deterministic correction rule set, such as an applicable documented BOS or separator correction.
- A bounded exhaustive evaluation of the same candidate set without adaptive selection, where relevant.
- Applicable named external tools only when their task and output can be compared fairly.

An oracle using held-out outcomes may be reported only as a clearly labeled retrospective upper-bound diagnostic. It cannot be presented as a deployable baseline or used to tune the published method on that held-out set.

### 18.3 Benchmark procedure

1. Register exact code, dependency, model, tokenizer, catalog, fixture, and split revisions.
2. Declare the supported workload, candidate corruption model, corruption severity, and expected recovery criterion.
3. Seal grouped search and held-out data and document prior use of every dataset.
4. Fix hardware, precision, sampling, cache behavior, context limits, and seed schedule.
5. Set comparable total resource budgets; report actual prefill, decode, scoring, retries, and overhead for every method.
6. Run conformance and alignment checks before collecting candidate results.
7. Execute search, freeze decisions, then evaluate the frozen method and baselines on held-out fixtures.
8. Report task behavior, boundary/continuation failures, repair/no-change/abstention rates, coverage, export parity, latency, and resource use separately.
9. Use paired, group-aware uncertainty and publish the preregistered aggregation rules.
10. Include failed loads, unsupported formats, cancelled runs, budget exhaustion, and exclusions in the accounting.
11. Publish public-safe raw evidence or a sufficient reproducibility package with explicit restrictions for licensed weights or private data.
12. Reproduce the run from a clean environment and record any divergence.

Do not average incompatible model families or tasks into a headline score without a declared weighting scheme and per-cell results. A method with lower coverage must not appear better merely because difficult cases were omitted.

### 18.4 Ablations and error analysis

Test the contribution of raw boundary evidence, suffix probes, bounded mutations, group-aware held-out confirmation, explicit abstention, and export parity checks. Use the same underlying data and resource accounting where possible. Explain when an ablation changes the scientific question or available information.

Manually review public-safe examples of false repair recommendations, missed repairs, correct no-change results, and justified/unjustified abstentions. Analyze whether a task gain masks boundary damage or whether a clean delimiter merely truncates a wrong answer. Error analysis should drive new development fixtures; a fresh sealed confirmation set is required for subsequent claims.

## 19. Continuous integration release and maintenance

### 19.1 CI tiers

- **Tier A, every change:** formatting/linting, type checks, pure unit tests, schema validation, deterministic golden tests, public-fixture checks, and secret scanning. These tests should run offline without model weights.
- **Tier B, contract changes:** property tests, mutation tests, adapter mocks, fault injection, package build/install tests, report/viewer integration, export round trips, and dependency audits.
- **Tier C, adapter or runtime changes:** opt-in local-model conformance and scoring tests against licensed test checkpoints in controlled environments. Missing access must be reported as not run, not passed.
- **Tier D, release candidates:** full advertised support matrix, security/accessibility review, clean-environment reproduction, exact-archive inspection, and release evidence review.

CI must not print private prompts, secrets, environment dumps, or local authentication files. Downloading model weights in CI requires an explicit license/access policy and must not become a hidden dependency of the ordinary contributor workflow.

### 19.2 Release checklist

- Version package, report schemas, catalog semantics, seed derivation, and export formats consistently.
- Build a source archive from an allowlist; exclude generated runs, caches, weight files, private configuration, and repository history.
- Verify file inventory, checksums, licenses, third-party notices, README commands, and optional dependency boundaries.
- Test installation and the documented workflow in a clean supported environment.
- Attach verification records for advertised capabilities and identify all tests not run.
- Publish supported model/runtime cells, known limitations, security reporting instructions, and migration notes.
- Inspect the exact archive for secret-like material, personal data, symlinks, unexpected binaries, hidden files, and accidental nested archives.
- Ensure example reports are synthetic and labeled, or explicitly sourced from approved public-safe experiments.
- Never promote an unverified source draft by changing only its README status.

### 19.3 Maintenance policy

Treat backend/tokenizer changes as possible semantic changes, not routine dependency bumps. Re-run affected conformance, alignment, budget, and export tests. Keep known-good locked environments for reproducing older reports. Retain schema readers for a documented compatibility window or provide explicit migrations with tests.

Catalog contributions require provenance, applicability rules, a bounded mutation set, golden prompts, and passing negative tests. New model support requires an evidence-bearing registry cell. New fixture sets need provenance and grouping review. Public issue templates should ask for sanitized manifests and fixed error codes rather than private prompts or credentials.

When a release regression is found, document the affected configurations and claims, mark support cells accordingly, and provide a reproducible regression fixture. Do not silently replace old benchmark artifacts or erase failing results.

## 20. Risks and mitigations

| Risk | Consequence | Mitigation and release blocker |
| --- | --- | --- |
| Hidden backend wrapping or stopping | False evidence of correct format or termination | Raw conformance gate; unsupported backend refusal |
| Context-sensitive tokenization | Wrong prompt or boundary interpretation | Full-context token audit; exact alignment or abstention |
| Likelihood off-by-one or masking error | Candidate rankings based on invalid scores | Analytic oracle and independent reference calculation |
| Compute undercounting | Unfair comparison or unexpected resource use | Reservation/receipt ledger including retries and recomputation |
| Fixture leakage or overfitting | Inflated repair claims | Grouped sealed splits, freeze protocol, fresh confirmation data |
| Small or correlated samples | Misleading confidence | Group-aware uncertainty and explicit insufficient-evidence outcomes |
| Boundary as ordinary content | Incorrect answer truncation | Ambiguity-aware probes and preserved full continuation |
| Unsupported role semantics | Plausible but incorrect exports | Capability matrix and strict rejection of unsupported roles |
| Cross-runtime drift | Export works differently from evaluation | Target-version-specific reload parity |
| Nondeterministic kernels | Irreproducible output traces | Reproducibility tiers and configuration-specific evidence |
| Unsafe model/report import | Local code execution or data exposure | Data-only formats, safe loading, escaping, offline defaults |
| Private evidence in a source release | Accidental disclosure | Allowlisted packaging, secret scan, privacy review of exact archive |
| UI hides failures or denominators | Users overtrust a result | Canonical report-driven UI and negative-state tests |
| Broad claims from narrow success | Misleading product positioning | Claim registry and evidence-scoped documentation |
| Expanding scope too early | Many draft features without one reliable workflow | Finish one supported end-to-end path before adding role/runtime breadth |

## 21. Decisions to resolve before implementation expands

Record these choices as short architecture decisions with rationale and rejected alternatives:

- **DEC-01:** Which exact local model/tokenizer pair becomes the first reference configuration?
- **DEC-02:** Which restricted canonical format representation can express the first supported families without arbitrary code?
- **DEC-03:** Which target runtime is the first export destination, and which semantics must it preserve?
- **DEC-04:** Which actual backend counters are available, and what conservative policy applies when a call fails before a receipt?
- **DEC-05:** Which likelihood-alignment method is valid for the selected tokenizer, and which cases will abstain?
- **DEC-06:** Which grouped fixture corpus and preregistered decision thresholds support the first useful recommendation claim?
- **DEC-07:** Is a static read-only viewer sufficient for the first release, or is a local control API necessary?
- **DEC-08:** What redaction profiles and retention defaults keep private evidence from entering shareable artifacts?
- **DEC-09:** Which configuration changes invalidate a support cell or require a new schema/catalog version?
- **DEC-10:** Which related-work comparisons are scientifically comparable, and which remain descriptive only?

These decisions should be resolved with evidence and explicit scope rather than by making the initial release appear broader than its validation supports.

## 22. Completion standard

The project reaches a usable first release when one documented local workflow passes the relevant gates end to end: safe loading, exact rendering, bounded candidate generation, raw probing, honest accounting, valid scoring, isolated confirmation, uncertainty-aware decisions, inspectable evidence, and verified export. The source archive and documentation must say precisely which configurations and capabilities meet that standard.

Until then, the bundle is a development starting point. Its value is the explicit contracts, testable invariants, and a concrete path to trustworthy results; runtime reliability and repair quality remain to be established.
