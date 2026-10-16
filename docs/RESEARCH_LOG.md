# Research → Plan → Develop Cycle Log

How this product is built: repeating cycles of (1) online research from
reputable primary sources, (2) planning based on findings, (3) development —
then back to (1) with new questions raised by what was built. There is no
iteration cap: thoroughness over speed. Cycling continues until
**consecutive** cycles yield nothing significant — a single quiet round does
not end the process. Sources are official
docs, specs, and top-tier open-source projects; never marketing copy or
unverified blogs. Each round below records what was researched, the sources
used, what changed in the plan, and what was built.

Every cycle additionally ends with a **differentiation review**: for each
major competitor, write down exactly what they do, then state the concretely
different answer this lab gives — verifiable in code, never marketing fluff —
and ask: is this substantially different from everything else out there, and
would it make someone star the repo? Creative additions must be genuinely
novel *and* working: new experiment modes, novel report visualizations,
developer-experience tooling (the `doctor` concept extended), fixture
generation, analysis views nobody else offers. Every addition is real,
tested, documented, and secure — no gimmicks.

## Prior rounds (2026-10-12 → 2026-10-13, summarized)

- **Hardening rounds 1–2.** Full audit, test suite grown to 224 tests (97%
  coverage), ruff/mypy/bandit/pip-audit clean, two adversarial red-team
  rounds with all findings fixed and regression-tested. Evidence in
  VALIDATION.md; methodology in docs/SECURITY.md.
- **Round A — competitive research (2026-10-13).** Primary-source study of
  lm-evaluation-harness (`huggingface.py`, `utils_hf.py`), vLLM v1
  `detokenizer.py`, llama.cpp server, HF transformers docs
  (`chat_templating`, `stopping_criteria.py`), lighteval, HELM, Unitxt,
  Guidance/Outlines/llguidance. Findings: every competitor renders the
  model's *own* chat template; all detect stops at string level with
  mitigations that hide tokenization mismatch; lm-eval auto-appends EOS to
  stop lists; none has a selection/abstention discipline; none verifies
  boundary portability across tokenizers. 10 ideas killed (already covered
  by competitors). Proposed P1–P5 (below), ranked by value/effort.
- **Round B — hostile audit (2026-10-13).** 5 bugs found and fixed
  (doctor report shape, codec/parser text leaks, fingerprint-less backend
  AttributeError, NFC evasion gap in `evaluate`, chmod through symlinks),
  each with regression tests. 231 tests green, 97% coverage held.
- **Round C — differentiation features (2026-10-13).** Implemented all five
  proposals: P1 template-conformance diff (`core.template_diff`,
  `doctor.template_conformance`), P2 stop-behavior isolation matrix
  (HF overhang / vLLM holdback / EOS-appended emulations, verified against
  real `transformers.StopStringCriteria` as an oracle — 0 mismatches on 9
  docstring cases), P3 prompt-boundary sensitivity sweep
  (`core.truncate_text_tail`, opt-in `--sensitivity-sweep`), P4
  cross-tokenizer boundary portability (`core.check_boundary_portability`,
  `doctor --reference-model`), P5 failure-mode taxonomy
  (`experiments.classify_observation`, cited in the decision banner).
  352 tests, 98% coverage, full battery green.
- **Round D — red-team round 3 (2026-10-13, in progress).** Adversarial
  review of the new P1–P5 surface plus full verification battery.

## Cycle 1 — research (2026-10-13)

**Emulation fidelity re-check.**
- HF transformers: PR #46530 (fix for issue #46519, merged ~123 days ago,
  present in the installed transformers 5.19.0) changed `StopStringCriteria`
  table construction to **byte space** for byte-level and byte-fallback
  decoders (`_get_stop_string_matching_mode`, `_token_to_bytes`,
  `_get_stop_strings_for_matching`; `__call__` runtime path unchanged).
  Source: installed
  `transformers/generation/stopping_criteria.py` (5.19.0, read locally) and
  https://github.com/huggingface/transformers/pull/46530 (byte-space fix) /
  https://github.com/huggingface/transformers/issues/46519 (the CJK
  byte-fragment repro). The lab's `_hf_overhang_match` ports the matching
  rule correctly but feeds it per-token `decode` strings, so on UTF-8
  fragment tokens real HF now fires where the emulation does not -- the
  residual documented in `docs/USAGE.md` §8.2 is now a known upstream-fixed
  divergence, not just an approximation. (The round-3 1500/1500 differential
  trials still pass because they were ASCII-only: byte-space and
  string-space agree when no fragment token appears.)
- vLLM: `check_stop_strings` earliest-completing selection (PR #49391,
  ~82 days ago) is already what the lab emulates; `stop_buffer_length =
  max(len(s)) - 1` confirmed unchanged in docs.vllm.ai v0.16.0
  (`vllm/v1/engine/detokenizer.py`). Source:
  https://github.com/vllm-project/vllm/pull/49391,
  https://docs.vllm.ai/en/v0.16.0/api/vllm/v1/engine/detokenizer/. No
  material change since the 2026-10-13 verification.

**Prior art check.** No tool found doing any of: token-ID-level diffing of
hand-rendered format hypotheses against a checkpoint's own chat template
(Ferric gates its *own renderer's* fidelity vs HF `apply_chat_template` --
a serving-correctness check, different question:
https://github.com/dcharlot-physicalai-bmi/ferric/commit/b0b4448917712125a5780e839460c77482ec23d1);
cross-tokenizer boundary portability checks; mid-token prompt-boundary
sensitivity measurement; stop-semantics isolation matrices. Academic prompt
sensitivity work (arXiv 2509.01790 "Flaw or Artifact?", ProSA
findings-emnlp 2024, arXiv 2509.13680) measures *semantic* template
variants, never sub-character token-boundary perturbations. The niche is
still empty.

**Creative territories.** (a) The agent-eval-autopsy master plan states the
methodological principle this cycle adopts: report the sensitivity curve
before the effect size -- "step budget: too high, and no ablation can
change the outcome" (https://github.com/kush1311/agent-eval-autopsy/blob/HEAD/agent-autopsy-master-plan.md).
No eval harness found that derives pass-rate-vs-token-budget from already
recorded outputs. (b) Doctor-style DX: the recurring best-practice pattern
is GUIDE -- actionable next steps after a clean scan (specific commands,
re-scan to verify). (c) Packaging, grounded in specific repos: ruff's
README "Show Your Support" badge block and docs/ hub layout
(https://github.com/astral-sh/ruff); sortitionfoundation's
docs/{index,quickstart,concepts,adapters,advanced}.md structure. Honest
gaps here: no FAQ/troubleshooting doc, no guided next step after `doctor`.

**Plan (selected).**
1. **HF byte-space stop matching** -- close the upstream-fixed divergence:
   generalize `_hf_overhang_match` to str|bytes, add an optional
   per-token byte mapper on the backend (port of upstream's decoder-mode
   detection + `_token_to_bytes`), thread it through `emulate_hf_stop_string`
   / `stop_matrix` / CLI. Differentially verified against the real
   `StopStringCriteria`, including the issue-#46519 fragment shape.
2. **Token-budget robustness frontier** -- new `experiments.budget_frontier`:
   from the recorded search observations, the exact pass/budget curve per
   candidate (pass(b) ⟺ native pass ∧ b ≥ cut+len(boundary)), the budgets
   where the selection decision/winner would flip, surfaced in results.json
   and the HTML report. Post-hoc, no extra model calls; directly answers
   "was N=64 load-bearing for this verdict?".
3. **`doctor` next-command suggestion** -- on a clean `doctor` with
   `--dataset` and `--boundary-ids`, emit the exact `run` invocation with
   `--max-context` filled from the checkpoint's declared window (the value
   users get wrong, per the round-2 GPT-2 finding). Machine-readable in the
   doctor JSON.
4. **`docs/FAQ.md`** -- troubleshooting grounded in documented behaviors
   (boundary IDs, context window, tokenization-mismatch diagnostic, empty
   expected, interrupted runs, report privacy); linked from README/USAGE.

**Rejected.** llama.cpp stop-semantics cell (needs primary-source
verification of its current matching semantics -- next-cycle candidate);
`init-fixtures` scaffold (marginal over `examples/fixtures.json`);
per-fixture boundary-position histogram (subsumed by the frontier's
min-budget data); confidence intervals (violates the `not_estimated`
discipline); inline-SVG charts (XSS review cost; text bars suffice);
non-greedy backends (out of scope; post-hoc exactness rests on greedy).

## Cycle 1 — built (2026-10-13)

1. **HF byte-space stop matching.** `backends._hf_overhang_match` generalized
   to str|bytes (TypeVar; the overlap algorithm is encoding-agnostic).
   New `backends._hf_byte_token_mapper(tokenizer)`: ports upstream's
   `_get_stop_string_matching_mode` + `_token_to_bytes` (byte_level via
   `bytes_to_unicode`, byte_fallback via `<0xNN>`), returning a per-token-id
   byte mapping or None (classic string path). `LocalHFBackend` exposes it as
   `hf_stop_token_bytes` (construction failure degrades to None, never fails
   init); `TruncatedPromptBackend` forwards it; `emulate_hf_stop_string` /
   `stop_matrix` / CLI thread an optional `token_bytes` through, with
   per-token fallback to isolated-decode→UTF-8 and OOB→`b""` (mirrors HF's
   clamp-to-dummy). Differentially verified against the real
   `StopStringCriteria`: the issue-#46519 fragment shape (old path silent,
   oracle fires, new path fires) and 200 randomized ASCII trials where
   classic path, byte path and incremental oracle all agree.
2. **Token-budget robustness frontier.** `experiments.budget_frontier`
   derives the exact per-candidate pass/budget curve from recorded search
   observations (pass(b) ⟺ native pass ∧ b ≥ cut+len(boundary); proven via a
   300-case brute-force truncation-lemma test), plus per-budget selection
   decisions, `flip_ranges`, and `budget_robust`. Always-on `run`
   diagnostic (degrades to "unavailable", never fails the run); new
   `report.html` section with winner pass curve and flip notices; stored in
   `results.json`.
3. **`doctor` next-command suggestion.** `examine_model` reports
   `model.context_window` (declared `n_positions`/`max_position_embeddings`);
   clean `doctor` with `--dataset` + `--boundary-ids` emits a `suggestion`
   with the exact `run` command (`--max-context` from the declared window,
   shell-quoted paths, `FAMILY` placeholder + catalog families in the note).
4. **`docs/FAQ.md`.** 12 troubleshooting/interpretation Q&As grounded in
   documented behavior; linked from README and USAGE.

Tests: 47 new in `tests/test_cycle1.py` (361 → 408 total). Docs: USAGE
§2 (suggestion), §8.2 (byte-space HF bullet, residual note rewritten), new
§8.6 (frontier); CHANGELOG `[Unreleased]`; README feature list (+1 line).

## Cycle 1 — differentiation review (2026-10-13)

For each major competitor: what they do → this lab's concretely different
answer (verifiable in code) → would it make someone star the repo?

- **lm-eval (lm-evaluation-harness).** Renders the checkpoint's *own* chat
  template via `tokenizer.apply_chat_template`; string-level stops with a
  "tokenizers suck" lookback (`MultiTokenEOSCriteria`); appends EOS to stop
  sequences; ranks candidates with no abstention discipline. This lab:
  hand-renders format hypotheses and encodes with `add_special_tokens=False`
  (`core.Format.render`), then diffs each hypothesis as an exact token-ID
  sequence against the checkpoint's own template rendering
  (`core.template_diff`, `doctor.template_conformance`) -- measuring the
  model on sequences its own template would never produce; token-ID
  boundary match with a text cross-check (`core._boundary_text_match`);
  unique-all-search-pass selection or abstention
  (`experiments.run_experiment`). Star-worthy for the narrow audience
  asking "weights or formatting?": no other tool asks that question at the
  token-ID level.
- **vLLM.** Inference server: renders the model's Jinja template, detects
  stops in the detokenizer (`check_stop_strings`, earliest-completing
  wins), stops on EOS by default. This lab: not a server -- it emulates
  vLLM's stop semantics post-hoc as one cell of an isolation matrix
  (`backends.emulate_vllm_holdback`, `experiments.stop_matrix`) to ask
  whether the verdict *survives* under competitor semantics, a question
  vLLM cannot ask about itself. Nobody else ships a stop-semantics
  isolation matrix.
- **llama.cpp.** Server with its own template rendering and string-level
  stops; EOS stopping by default. Same structural differentiation as vLLM.
  Honest gap: no llama.cpp stop cell yet -- its current matching semantics
  need primary-source verification first (deferred to a later cycle, not
  claimed).
- **lighteval / HELM.** Benchmark suites measuring task accuracy (HELM adds
  fairness/calibration/efficiency) with standardized prompts. This lab is
  not a benchmark: it contributes measurement-integrity instruments none of
  them has -- the token-budget frontier (`experiments.budget_frontier`:
  the exact pass/budget curve and decision-flip ranges from recorded
  outputs, no extra model calls), the stop matrix, the failure taxonomy,
  the mid-token sensitivity sweep. The frontier in particular answers the
  "report the sensitivity curve before the effect size" principle no
  harness implements for token budgets.
- **Unitxt.** Dataset/task preparation recipes. No overlap with the novel
  features; the lab's fixture schema is a strict local contract
  (`datasets.parse_dataset`), not a preparation framework.

Cycle-1 additions against this landscape: (1) byte-space HF stop matching
tracks a *merged upstream transformers change* (issue #46519 / PR #46530)
that string-level tools never needed to care about -- the emulation now
fires where real HF fires on byte-fragment tokens, differentially verified
against the real `StopStringCriteria`; (2) the budget frontier is a new
analysis view (no prior art found); (3) the doctor next-command suggestion
closes the most common operator error (wrong `--max-context`) at the
moment it matters. Star-worthy? For the niche -- local checkpoint
debuggers -- the honest answer is "these are real, verifiable differences,
not marketing"; the FAQ and doctor DX lower the floor for everyone else.

## Cycle 1 — verification (2026-10-13, Python 3.12, Linux x86-64)

Full battery, all runs by this cycle (commands from the repository root):

- **Fresh venv reinstall**: `python -m venv` + `pip install -U pip setuptools`,
  then `pip install -e .` (base: `catalog` smoke OK) and
  `pip install -e ".[dev,hf]"` (CPU index for torch). Installed set matches
  the main venv: torch 2.14.1+cpu, transformers 5.19.0, safetensors 0.8.0,
  tokenizers 0.23.3. (First attempt stalled unpacking PyPI's CUDA torch
  bundle; retried via the PyTorch CPU index.)
- **Tests**: `python -m unittest discover -s tests -t .` — **410 tests, all
  passing** (361 carried over + 49 new), across **3 consecutive green
  executions in the fresh venv** (16.5 s / 9.6 s / 9.7 s) and repeated
  greens in the main venv. `[dev]`-only path (transformers/tokenizers/torch
  blocked): **OK (skipped=76)** — the byte-space mapper no longer needs the
  HF stack (the `bytes_to_unicode` table is ported, cross-checked against
  transformers when present), so only the true oracle tests skip.
- **Lint/format**: `ruff check` and `ruff format --check` clean on
  `model_protocol_lab tests examples` (22 files) in both venvs.
- **Types**: `mypy model_protocol_lab` clean (10 source files) in both venvs.
- **Coverage**: **98% total line coverage** (1753 statements) in both venvs,
  above the 97% bar; every new-code branch is covered (remaining misses are
  the pre-existing defensive lines).
- **Security**: `bandit -r model_protocol_lab` — 0 issues; `pip-audit` — no
  known vulnerabilities (torch `2.14.1+cpu` local tag invisible to the OSV
  DB, the accepted residual).
- **Differential evidence**: byte-space emulation vs real
  `StopStringCriteria` — the issue-#46519 fragment shape: old path silent,
  oracle fires, new path fires (exact agreement); 200 randomized ASCII
  trials: classic path, byte path and incremental oracle agree on all 200.
  (One test-authoring trap found and fixed along the way: the oracle must be
  consulted per prefix, as `generate()` does — a final-state-only oracle
  disagrees with the real criterion on mid-sequence completions.)
- **CLI smoke** (tiny locally-generated GPT-2 checkpoint, no downloads, in
  the fresh venv): `catalog`, `inspect`, `doctor` (suggestion present with
  `--max-context 64`, the checkpoint's declared window), `run --mutate
  --sensitivity-sweep --max-drop-bytes 2` (exit 2 `no_candidate` as expected
  with random weights; `budget_frontier` and `sensitivity_sweep` sections
  present in `results.json`; frontier section in `report.html`).
  Byte-level backend smoke: `LocalHFBackend.hf_stop_token_bytes` maps
  correctly on a real byte-level checkpoint and generation still works.
- **Golden test**: `examples/make_sample_report.py` regenerates
  byte-identically in the fresh venv (also covered by the suite).
- **Config validation**: CI YAML, pre-commit YAML, Dependabot YAML,
  `CITATION.cff` all parse; title/authors present (unchanged this cycle).
- **Residuals, bluntly**: (1) The byte path's fallback for unmappable tokens
  (isolated decode → UTF-8) approximates HF's cleaned-vocab-string fallback
  — documented, untestable without HF's internals. (2) The frontier's
  exactness rests on greedy decoding, like the stop matrix. (3) No llama.cpp
  stop cell yet. (4) The doctor suggestion's `--family` is a placeholder the
  operator must fill — deliberate, not inferred.

## Cycle 2 — research (2026-10-13)

**llama.cpp stop cell — primary-source verification, then killed.**
Read the current llama.cpp sources directly (master, 2026-10-13):
`tools/server/server-context.cpp` `server_slot::find_stopping_strings`
(L820-848) and `server_slot::process_token` (L2591-2765),
`common/common.h` `string_find_partial_stop`, `common/sampling.cpp`, the
server README's sampling-params table, and PR #13785 (partial-stop
streaming fix). Findings:

- Full stop: for each antiprompt word, `text.find(word, max(0, size -
  word.size() - last_token_size))` on the *unsent* text tail; the
  earliest-starting word wins; on fire the generated text is truncated at
  the stop start -- **exclude semantics**, the stop word never appears in
  the final text.
- Partial stops only withhold *streamed* deltas; per PR #13785 "partial
  stop words aren't erased by server_context::process_token, but full stop
  words are" -- they have no final-text effect and there is nothing to
  emulate post-hoc.
- EOS: `llama_vocab_is_eog(vocab, result.tok)` stops generation
  (STOP_TYPE_EOS) with no text truncation; `--ignore-eos` is documented as
  "ignore end of stream token and continue generating (implies --logit-bias
  EOS-inf)" -- i.e. exactly the lab's native generate-through-EOS behavior.
- Verdict for the matrix (single stop string = decoded boundary text,
  recorded greedy outputs, post-hoc truncation): llama.cpp's rule is the
  same "truncate before the completed stop word" as the vLLM cell. The
  deltas do not survive emulation -- multi-word earliest-start vs
  earliest-completion selection is inapplicable (one stop string), the
  1-char search-window offset sits below the emulation's incremental-decode
  approximation floor, partial stops are streaming-only, and EOS is covered
  (EOS-appended cell for stop-on-EOS; native behavior for --ignore-eos).
  A cell would duplicate the vLLM cell on every realistic input while
  doubling the decode-approximation stack. Killed with reasons; the
  analysis is recorded in `docs/USAGE.md` §8.2 so the decision is checkable.

**Per-candidate boundary support — scientifically justified, built.**
USAGE §4 documented the limitation ("the current CLI therefore cannot
fairly compare candidates requiring different boundary sequences") and
promised per-candidate support. The justification is not convenience: each
candidate is a format hypothesis, and real formats terminate the assistant
turn with different markers (ChatML `<|im_end|>`, Llama 3 `<|eot_id|>`,
Gemma `<end_of_turn>`), so one boundary for all candidates is the
unscientific choice. Design: `--boundary-ids` stays the default, repeatable
`--boundary-map NAME=ID,ID` overrides per candidate name (validated against
the actual candidate set -- unknown/duplicate names fail fast before any
compute); `run_experiment(..., boundary_map=...)` resolves per candidate;
every search entry records its `boundary_token_ids`; held-out, stop matrix
(via the winner's boundary), budget frontier (per-entry boundaries) and
sensitivity sweep all follow. The selection rule is untouched: exactly one
candidate passing every search fixture *under its own boundary*; ties and
failures abstain. `doctor --boundary-map` checks each mapping against the
checkpoint vocabulary.

**Fixture-generation UX — surveyed, built.**
lighteval: Python `prompt_fn` converting a dataset line to a Doc, plus
declarative `LightevalTaskConfig` (docs: adding-a-custom-task).
promptfoo: YAML `tests:` with vars/assert, CSV column-as-vars expansion
with the `__expected` assertion column, `file://` globs, `promptfoo init`
scaffolding (promptfoo.dev docs). Unitxt: Python Task/templates over data,
`create_dataset`, `unitxt-explore` UI (IBM/unitxt README). The pattern
across all three: a more compact authoring surface than the runtime schema.
None of them enforces a search/held-out separation discipline at authoring
time -- promptfoo has no held-out concept; lighteval has splits but no
group-overlap or identical-message leakage checks. The lab's answer:
`fixtures build` compiles a one-row-per-fixture CSV
(`id,group,partition,expected,user_1,assistant_1,...`) to the strict JSON,
with the partition explicit per row; group overlap, duplicate IDs and
identical message sequences across partitions raise with row numbers
*before any file is written*, and the emitted JSON is round-tripped
through the strict loader before it is returned -- discipline by
construction, not by post-hoc inspection. CSV chosen over YAML (stdlib
only, promptfoo-consistent columns-as-vars DX); empty-string message
content stays JSON-only (documented).

**Richer report views — built, scriptless.**
Surveyed best-in-class static reports (pytest-html's filterable tables need
JS -- rejected under the lab's scriptless CSP; criterion.rs's self-contained
inline SVG; coverage.py's annotated views). Cycle 1's inline-SVG rejection
stands (text achieves the value with no new review surface). Built, all
post-hoc from recorded observations, all values escaped, zero new assets:
(1) per-fixture x candidate outcome grid (✓/✗/–, ★ on the selected
candidate) -- the debugging view the report lacked; (2) decisive fixtures
(fixtures passed by exactly one candidate while all others failed -- the
evidence the verdict actually rests on; empty with an honest message when
no fixture separates the field); (3) per-candidate token-budget pass-curve
sparklines (block elements, aria-labeled, ≤48 sampled budgets) next to the
winner's exact table; (4) a boundary-IDs column in the candidate table and
per-candidate boundary lines in the run configuration.

**Re-verification against latest upstream — no drift.**
transformers main `stopping_criteria.py`: `_get_stop_string_matching_mode`,
`_token_to_bytes`, `_get_stop_strings_for_matching`,
`_stop_string_get_matching_positions` and `__call__` are identical to the
ported 5.19.0 logic (byte-space table build for byte-level/byte-fallback
decoders intact). vLLM main `vllm/v1/engine/detokenizer.py`:
`check_stop_strings` earliest-completing selection with tie-break by
stop-list order, `stop_buffer_length = max(len(s)) - 1`,
`include_stop_str_in_output=False` default, `min_tokens` gating -- all
unchanged since the Cycle-1 verification; the lab's documented `min_tokens=0`
assumption still holds.

**Plan (selected).**
1. Per-candidate boundaries (`--boundary-map`, experiments/CLI/doctor/
   reporting threading, USAGE §4 + FAQ updates).
2. `fixtures build` CSV authoring (`model_protocol_lab/fixtures.py`,
   `examples/fixtures.csv`, USAGE §9).
3. Report views: outcome grid, decisive fixtures, frontier sparklines,
   boundary column (reporting.py, sample report regenerated).
4. llama.cpp kill documentation (USAGE §8.2).

**Rejected.** llama.cpp stop-semantics cell (verified duplication of the
vLLM cell -- see above); inline-SVG charts (Cycle-1 rejection stands);
a YAML authoring format (CSV needs no new dependency); confidence intervals
(still violates `not_estimated`); non-greedy backends (still out of scope).

## Cycle 2 — built (2026-10-13)

1. **Per-candidate boundaries.** `experiments._resolve_boundaries`
   (default + map resolution, validated: unknown names, non-dict maps and
   bad ID tuples rejected); `run_experiment(..., boundary_map=None)` --
   each search entry records `boundary_token_ids`, held-out uses the
   winner's boundary (`selected_boundary_token_ids`); `budget_frontier`
   reads per-entry boundaries with fallback to the single-boundary
   argument (backward compatible); `prompt_sensitivity_sweep` threads the
   map through. CLI: repeatable `--boundary-map NAME=ID,ID` on `run`
   (parsed by `parse_boundary_map`, validated against the real candidate
   set via `_resolve_boundary_map` -- duplicates/unknown names exit 2
   before any compute) and on `doctor` (duplicates only; per-name vocab
   checks in `examine_model`); the stop matrix uses the winner's boundary;
   the doctor suggestion note mentions `--boundary-map`. Reporting: search
   table gains a boundary-IDs column; the run configuration lists
   per-candidate boundaries when they differ (single line when uniform).
   Docs: USAGE §4 rewritten (+ `### Per-candidate boundaries`), FAQ
   updated, CHANGELOG, README (+1 line).
2. **`fixtures build`.** New `model_protocol_lab/fixtures.py`:
   `read_authoring_file` (O_NONBLOCK + fstat regular-file gate, 2 MB cap,
   UTF-8), `parse_fixture_csv` (strict header `id,group,partition,expected`
   + `user_1,assistant_1,...` pairs, row-numbered errors, verbatim cells,
   quoted newlines, blank-line tolerance, short-row padding; the `Split`
   constructor enforces the separation discipline before any write),
   `split_to_json_text` (round-trips through `parse_dataset` -- by
   construction), `write_fixture_file` (O_EXCL, 0o600, parent must exist,
   partial-write cleanup). CLI `fixtures build SOURCE --output OUT`
   prints a JSON summary, warns on empty `expected` like `inspect`, and
   exits 1 with the reason on authoring errors. `examples/fixtures.csv`
   builds to the exact equivalent of `examples/fixtures.json` (verified).
   Docs: USAGE §9 + §1 cross-reference, CHANGELOG, README (+1 line).
3. **Report views.** `reporting._search_outcomes` (tolerant post-hoc
   reader), `_outcome_grid_section`, `_decisive_section`, `_sparkline` +
   `_frontier_spark_table` (exact counts from recorded minimal budgets,
   ≤48 sampled budgets), boundary column + per-candidate config lines.
   New CSS classes only (glyph colors, spark); CSP unchanged, no scripts.
   Committed sample report regenerated (byte-identical re-runs verified).
4. **llama.cpp kill note** in USAGE §8.2 with the primary-source analysis.

Tests: `tests/test_cycle2.py`, 81 tests -- boundary resolution and
selection semantics under the map (FakeBackend scripted generations prove
each candidate is measured with its own boundary, and that the map flips
ambiguous/selected outcomes exactly as designed), frontier per-entry
boundaries, sweep passthrough, CLI flag parsing/validation (exit-2 paths),
a full `run --boundary-map` on the tiny GPT-2 checkpoint (per-candidate
boundaries recorded in results.json and rendered in report.html), doctor
per-name checks (including the no-transformers degradation path),
`fixtures build` unit + CLI tests (row-numbered errors, discipline
refusals, 0o600/O_EXCL behavior, control-char stripping), and report-view
tests (glyphs, escaping of hostile fixture IDs, decisive logic, sparkline
scaling). Suite: 410 -> 491 tests.

## Cycle 2 — differentiation review (2026-10-13)

- **lm-eval.** Renders the checkpoint's own template and measures with one
  stop configuration for every task. This lab now additionally measures
  each *format hypothesis* with its own protocol terminator
  (`--boundary-map`) -- the "weights or formatting?" question asked with
  per-format end-of-turn markers, which a single-boundary harness cannot
  express without run multiplication. Verifiable: per-entry
  `boundary_token_ids` in results.json.
- **vLLM / llama.cpp.** Servers with one stop-word list per request. The
  lab's answer to "which stop semantics would flip my verdict" remains the
  isolation matrix; Cycle 2 verified from primary sources that a llama.cpp
  cell would add nothing (documented in USAGE §8.2) -- an honest
  non-duplication competitors cannot claim, since they never ask the
  question.
- **lighteval / HELM.** No per-format boundary discipline, no
  decisive-fixture analysis, no token-budget frontier. The new decisive
  fixtures view ("the evidence the verdict rests on") and the frontier
  sparklines are measurement-integrity instruments neither suite has.
- **Unitxt.** Dataset preparation recipes with no search/held-out leakage
  discipline. `fixtures build` is the only authoring tool found that makes
  split leakage structurally unbuildable (explicit partition column +
  build-time refusal + loader round-trip by construction).
- **promptfoo.** The closest DX analogue (CSV test cases, `__expected`
  column). promptfoo has no held-out concept and no leakage checks; the
  lab's CSV keeps the strict schema and the separation discipline, and its
  report stays a single zero-asset file while promptfoo needs its viewer.

Star-worthy? The per-candidate boundary support turns the lab's central
question ("weights or formatting?") from a single-terminator approximation
into a per-format measurement -- the feature the docs had been promising.
`fixtures build` removes the most tedious step between the user and a
disciplined experiment. The report views answer "what decided this?" at a
glance. All three are real, tested, and unlike anything in the surveyed
tools. The honest limit stands: the niche is local-checkpoint debugging,
not general benchmarking.

## Cycle 2 — verification (2026-10-13, Python 3.12, Linux x86-64)

Fresh venv (`pip install -e ".[dev,hf]"`, CPU torch index):
- `python -m unittest discover -s tests -t .`: 491 tests OK, three
  consecutive greens (10.9s / 10.1s / 10.1s).
- Dev-only venv (`pip install -e ".[dev]"`, no hf): 491 tests OK,
  78 skipped (HF-dependent tests skip cleanly, including the new
  Cycle 2 ones via the established HAS_HF guard).
- `ruff check` + `ruff format --check`: clean (24 files).
- `mypy model_protocol_lab`: no issues, 11 files.
- coverage: 98% total (fixtures.py 100%, experiments.py 100%,
  reporting.py 98%; remaining misses are pre-existing defensive
  branches) -- bar is 97%.
- Main venv: `bandit -r model_protocol_lab`: 0 issues; `pip-audit`:
  no known vulnerabilities (torch's `2.14.1+cpu` local tag is invisible
  to the OSV database -- accepted residual, same as Cycle 1; everything
  auditable is clean).
- CLI smoke: `catalog`, `inspect`, `fixtures build examples/fixtures.csv`
  (-> byte-equivalent of fixtures.json, `inspect` says valid),
  `run --help` shows the new `--boundary-map` flag, `fixtures --help`
  lists `build`.
- Sample-report golden test: green in every run (regenerates
  byte-identically); committed report contains the new outcome grid,
  decisive-fixtures section and boundary column, no http/https strings,
  no `<script>`.
- Config YAML validation: ci.yml, pre-commit-config.yaml, dependabot.yml,
  CITATION.cff all parse.

Known/accepted: (1) torch CPU wheel's local version tag is not auditable
by pip-audit. (2) Exactness still rests on greedy decoding, as documented.
(3) The doctor suggestion's `--family` remains a deliberate placeholder.
(4) No llama.cpp stop cell (verified duplication -- see research section).
(5) The sample report shows the single uniform "Boundary token IDs" line
rather than per-candidate lines because its boundaries are uniform -- the
intended display behavior, covered by test.

## Cycle 3 — research (2026-10-13)

**Fresh upstream sweep — no material drift.**
- transformers main `src/transformers/generation/stopping_criteria.py`
  re-fetched 2026-10-13 and compared against installed 5.19.0 and the lab's
  port: `_token_to_bytes`, `_get_stop_string_matching_mode`,
  `_get_stop_strings_for_matching`, `_stop_string_get_matching_positions`
  and `__call__` are identical on every line the emulation ports. New in
  main since 5.19.0: `STOP_STRING_EMBEDDING_CACHE` +
  `clean_and_embed_tokens_with_cache` (a table-construction perf refactor --
  same matching semantics) and a `new_token_length` parameter on
  `EosTokenCriteria` (not emulated, not relevant). The lab's
  `_hf_byte_token_mapper` matches both byte-for-byte on the ported logic.
- vLLM main `v1/engine/detokenizer.py`: `check_stop_strings`
  earliest-completion selection with stop-list-order tie-break,
  `stop_buffer_length = max(len(s)) - 1`, `include_stop_str_in_output`
  default False, `min_tokens` gating -- all unchanged since the Cycle-2
  verification. New: `_protected_step` overflow/invalid-prefix recovery in
  the fast detokenizer (error handling, not matching semantics). No drift
  affecting the emulation.
- transformers #47975 (deferred/async stopping check) and #48814
  (attention-mask elision) touch generate-loop timing/perf, not the
  criterion matching rule the lab ports; the lab's backend is its own
  greedy loop, so there is nothing to track. llama.cpp stays killed with
  the Cycle-2 primary-source documentation; lm-eval is comparison-only
  (not emulated), so no re-check was needed.

**Recent literature — nothing adoptable, one related-work grounding.**
Searched 2025–2026 work on prompt/chat-format sensitivity and
tokenizer-boundary effects: arXiv 2604.18389 ("Understanding the Prompt
Sensitivity", 7 modification types incl. token misalignment and
orthographic errors), arXiv 2608.18539 (interaction-based prompt
sensitivity), arXiv 2411.10541 (prompt-formatting impact, NAACL 2025),
arXiv 2609.35869 ("The Price of Token Boundaries", compression cost of
pre-tokenization, Sept 2026), and the token-healing literature
(guidance-ai 2023; measured effects in arXiv 2402.01035 §Table 8). Every
paper measures *semantic or surface* prompt variants; none does
sub-character token-boundary perturbation of format hypotheses against a
checkpoint's own template -- the lab's niche is still empty. Token healing
is the serving-side mitigation for exactly the bias the lab's sensitivity
sweep measures (a prompt ending mid-token skews the next-token
distribution); it is now cited as related work in USAGE §8.5. No paper
suggests a method the lab should adopt.

**Mutation testing — feasible with cosmic-ray, infeasible with mutmut;
found one real gap.**
- mutmut 3.8.0 is unusable for this repo: its config loader crashes on the
  documented `tests_dir` string shape (TypeError in its own code), its
  `mutants/` working-dir model drops `examples/` so the baseline suite
  fails (tests resolve fixtures via `__file__`), and its pytest-only
  runner does not match the project's unittest invocation.
- cosmic-ray 8.7.0 works: it runs the project's own test command in-tree.
  Scoped trial on `model_protocol_lab/fixtures.py` (261 lines, 100% line
  coverage): 416 mutants, ~15 s/mutant sequentially. Interim result
  27/32 killed (84%); the 5 survivors split into 3 real gaps and 2
  equivalent mutants. The real gaps, all fixed with regression tests in
  this cycle: (1) `len(content) > MAX_BYTES` vs `>=` survived -- no test
  pinned that a file of exactly MAX_BYTES bytes is accepted;
  (2) `len(cases) > MAX_CASES_PER_SPLIT` vs `>=` survived -- same boundary
  pattern for the per-split fixture cap; (3) `not record or all(...)` vs
  `and` survived -- a row of only empty cells (`,,,`) was silently
  tolerated by luck of no test, and the mutant turned it into a downstream
  validation error instead of a skipped blank line. The 2 equivalent
  mutants are unkillable by construction (`os.O_RDONLY | os.O_NONBLOCK`
  with O_RDONLY=0, so `>>` is identical; `type(x) is not str` vs `!=`,
  identical for type objects); one further survivor was cosmetic
  (`json.dumps` indent 2 -> 3, no behavioral difference -- not pinned,
  documented here).
- CI practicality: ~1.7 h for one small module at sequential speed; ~11
  modules extrapolate to ~10–15 h for the whole codebase. Not practical as
  a per-commit CI gate; conceivable as a weekly scheduled job with parallel
  workers. The honest-limits item ("no mutation-testing run") is now a
  documented trial with numbers instead of a blank.

**First-run UX walk — two genuine rough edges, both fixed.**
Walked install → catalog → inspect → fixtures build → doctor → run →
report against a tiny locally-generated GPT-2 checkpoint (n_positions=64):
1. The doctor's suggested `run` command used a fixed `--max-new-tokens
   64`; with an 18-token prompt in a 64-token window the *suggested
   command itself* failed with `context_limit_exceeded`. The suggestion now
   sizes `--max-new-tokens` so the longest fixture prompt (rendered with
   every catalog candidate, encoded with the checkpoint tokenizer) fits
   the declared window, and warns when even the longest prompt fills it.
2. A failed run's report banner said only "backend-error ×1"; the
   sanitized cause was recorded on the observation but stripped from the
   default evidence-excluded report. `evaluate` now passes the built-in
   backend's fixed-vocabulary codes through an allowlist (arbitrary
   backend text still collapses to `backend_reported_error`), summaries
   carry `failure_code`, and the banner prints a "Failure diagnosis" line
   with operator guidance.
- FAQ gaps closed: no entry existed for runtime `context_limit_exceeded`
  (only the init-time `local_context_limit_exceeds_declared_window`), and
  the "doctor says ok but run fails" entry did not mention the diagnosis
  banner. Both fixed.

**Deferred list revisited — nothing newly feasible.**
Cycle 1/2 rejections all stand: the llama.cpp cell (killed with
primary-source documentation in Cycle 2), the `init-fixtures` scaffold
(superseded by `fixtures build`), the boundary-position histogram
(subsumed by the frontier), confidence intervals (violates
`not_estimated`), inline-SVG charts (XSS review cost), non-greedy backends
(out of scope). No deferred item became feasible or more valuable.

**Plan (selected).**
1. Doctor suggestion window-fit (`doctor._longest_prompt_tokens`,
   `_suggestion_tokenizer`, `_build_primary_tokenizer` extraction).
2. Failed-run diagnosis (`failure_code` in `experiments._summary`,
   `_FAILURE_GUIDANCE` + `_failure_diagnosis` in `reporting`,
   allowlisted backend-code pass-through in `core.evaluate`).
3. Regression tests for the three mutation-found gaps (authoring
   size-limit boundary, per-split case-count boundary, all-blank-cell row
   tolerance).
4. Docs: FAQ entries, USAGE §2 (suggestion sizing) and §8.5 (token-healing
   related work), CHANGELOG, README test-count correction (361 → 491).

**Rejected.** Upstream emulation changes (no material drift found);
literature-driven new methods (none adoptable); mutmut in CI (infeasible
for this repo -- documented above); a full mutation-testing CI gate
(impractical per-commit at ~15 s/mutant -- documented above).

## Cycle 3 — built (2026-10-13)

1. **Doctor suggestion window-fit.** `doctor._build_primary_tokenizer`
   extracts the tokenizer construction (same 128 MB cap, same
   fail-closed/skipped semantics in `examine_model`). New
   `doctor._longest_prompt_tokens` renders every fixture with every
   candidate of every family (both mutate states) and encodes with the
   checkpoint tokenizer -- mirroring the backend's `encode`
   (`add_special_tokens=False`) -- so the budget fits whichever family the
   operator picks; any measurement failure degrades to the default 64,
   never fails doctor. `_next_command` emits
   `--max-new-tokens = max(1, min(64, window - longest))` and the note
   states the sizing; when the longest prompt already fills the window it
   keeps 1 and warns to shorten fixtures or use a larger-window
   checkpoint. Verified on the tiny GPT-2 checkpoint: suggestion now says
   `--max-new-tokens 10` (64 - 54) and the suggested command completes
   (`no_candidate` on random weights) instead of failing.
2. **Failed-run diagnosis.** `core.evaluate` passes through only
   allowlisted fixed-vocabulary backend codes
   (`_BACKEND_ERROR_CODES`); everything else -- None, empty, arbitrary
   text -- still collapses to `backend_reported_error`, so the
   sanitization boundary holds (existing PRIVATE-marker tests unchanged).
   `experiments._summary` records `failure_code` (None when nothing
   failed); `reporting._failure_diagnosis` renders a "Failure diagnosis"
   banner line mapping known codes to operator guidance
   (`_FAILURE_GUIDANCE`), generic guidance for unknown codes, all
   HTML-escaped. The code survives the evidence-excluded public view, so
   the default `results.json` names the cause.
3. **Mutation-found boundary test.** `read_authoring_file` accepts exactly
   MAX_BYTES (the `>` vs `>=` survivor) -- pinned by
   `test_exactly_at_size_limit_is_accepted`.
4. **Docs.** FAQ: new `context_limit_exceeded` entry + diagnosis-banner
   pointer; USAGE §2 documents the sizing, §8.5 cites token healing as
   related work; CHANGELOG `[Unreleased]`; README status count corrected.

Tests: `tests/test_cycle3.py`, 14 tests -- suggestion sizing on a 64- and
a 2048-window checkpoint, the fill-warning, tokenizer-less degradation,
measurement-failure degradation, allowlist pass-through vs collapse,
`failure_code` in summary and banner, diagnosis emptiness rules, and the
three mutation-found boundary/tolerance cases. Suite: 491 -> 505 tests.

## Cycle 3 — differentiation review (2026-10-13)

- **lm-eval / vLLM / llama.cpp.** No new emulation surface was needed
  (verified no drift); the honest non-duplication stance from Cycle 2
  stands. The doctor suggestion and failure diagnosis are DX refinements,
  not measurement science -- no competitor was found with a pre-flight
  that sizes the next command to the checkpoint's declared window, but
  this is operator convenience, not a new question asked.
- **lighteval / HELM / Unitxt / promptfoo.** Unaffected by this cycle's
  changes; the Cycle-2 differentiation (per-candidate boundaries,
  decisive fixtures, frontier, leakage-proof authoring) is unchanged.
- The substantive verification advance is methodological, not
  competitive: the first real mutation-testing data on the suite
  (cosmic-ray, 84% interim kill rate on the trial module, three real gaps
  found and fixed, two equivalent mutants identified) and a documented
  feasibility verdict for the honest-limits list.

Star-worthy? Honestly: this cycle's user-facing changes are small,
correct refinements to the first-run path -- worth having, not worth a
star on their own. The value is in the research verdicts: upstreams
stable, literature empty of adoptable methods, mutation testing trialed
with numbers, deferred list exhausted. That is a quiet round by the
cycle's own definition, and it is documented as one.

## Cycle 3 — verification (2026-10-13, Python 3.12, Linux x86-64)

Full battery, all runs by this cycle (commands from the repository root):

- **Fresh venv reinstall**: `python -m venv` + `pip install -U pip setuptools`,
  then `pip install -e ".[dev,hf]"` (CPU index for torch). Installed set
  matches the main venv: torch 2.14.1+cpu, transformers 5.19.0, safetensors
  0.8.0, tokenizers 0.23.3. (Note: /tmp is a 512 MB tmpfs -- too small for
  the torch wheel, so the trial venvs lived on the main filesystem and were
  removed afterwards.)
- **Tests**: `python -m unittest discover -s tests -t .` — **505 tests, all
  passing** (491 carried over + 14 new), across **3 consecutive green
  executions in the main venv** (17.5 s / 15.4 s / 16.1 s) and **3
  consecutive greens in the fresh venv** (18.1 s / 20.6 s / 21.8 s, at 503
  tests -- the last 2 tests were added after; all 14 cycle-3 tests pass in
  the fresh venv individually).
- **Dev-only venv** (`pip install -e ".[dev]"`, no hf): 503 tests OK,
  81 skipped (78 carried + 3 new HF-gated doctor-sizing tests skip
  cleanly via the established HAS_HF guard).
- **Lint/format**: `ruff check` and `ruff format --check` clean on
  `model_protocol_lab tests examples` (25 files) in both venvs.
- **Types**: `mypy model_protocol_lab` clean (11 source files) in both venvs.
- **Coverage**: **98% total line coverage** (2171 statements) in the main
  venv, above the 97% bar; fixtures.py 100%, experiments.py 100%,
  core.py 99%, doctor.py 98%, reporting.py 97%. Remaining misses are
  pre-existing defensive branches (specials loop, boundary_map TypeError
  path, no-transformers paths) -- every new-code branch is covered.
- **Security**: `bandit -r model_protocol_lab` — 0 issues (one B615
  false positive on the local `load_dataset` name needed the established
  `nosec` annotation, as in cli.py); `pip-audit` — no known
  vulnerabilities (torch `2.14.1+cpu` local tag invisible to the OSV DB,
  the accepted residual, same as Cycles 1-2).
- **Differential/UX evidence**: on the tiny locally-generated GPT-2
  checkpoint (n_positions=64): clean `doctor` suggests
  `--max-new-tokens 10` (64 - 54 longest prompt) and the suggested command
  completes (`no_candidate` on random weights) instead of failing; a
  deliberately oversized `--max-new-tokens 64` run fails with decision
  `failed`, `failure_code: context_limit_exceeded` in results.json, and the
  banner line "Failure diagnosis: context_limit_exceeded: Prompt tokens
  plus --max-new-tokens exceeded the checkpoint's declared context window.
  Retry with a smaller --max-new-tokens (the doctor suggestion sizes it to
  fit the fixtures)."
- **CLI smoke** (main venv): `catalog`, `inspect`, `fixtures build`
  (-> `inspect`-valid), `doctor` (suggestion present and sized), `run`.
- **Golden test**: `examples/make_sample_report.py` regenerated
  (results.json gains `"failure_code": null` in summaries -- the committed
  files were regenerated through the documented process) and regenerates
  byte-identically; report has no `<script>` and no http/https strings.
- **Config validation**: CI YAML, pre-commit YAML, Dependabot YAML,
  `CITATION.cff` all parse.

**Residuals, bluntly**: (1) The mutation trial was still running at
41/416 mutants (35 killed, 6 survived: 3 real gaps fixed, 2 equivalent,
1 cosmetic) -- the full-module kill rate is not yet known; the interim
84% is labeled as interim. (2) The fresh-venv 3-green ran at 503 tests,
before the final 2 tests were added (all 14 pass there individually; the
main venv did 3 greens at the final 505). (3) Exactness still rests on
greedy decoding, as documented. (4) The doctor suggestion's `--family`
remains a deliberate placeholder. (5) No llama.cpp stop cell (verified
duplication -- see Cycle 2 research).

## Cycle 3 — honest assessment: QUIET

The research genuinely found no substantial new work: both emulated
upstreams are unchanged on the ported logic (verified line-by-line
against current main), the 2025–2026 literature has no adoptable method
for this lab's question, and the deferred list is exhausted. What got
built is real but small: a window-fitted doctor suggestion, a named
failure diagnosis, three mutation-found boundary tests, and doc fixes --
all verified, none of it new measurement science. Per the cycle's own
definition, a round that finds nothing substantial is a valid quiet
round, not a failure; this one found two genuine UX rough edges and the
first mutation-testing data, fixed all of it, and stopped there instead
of inventing work.

## Cycle 4 — research (2026-10-13)

**Upstream re-check — no material drift (hours after the Cycle-3 check).**
- transformers main `src/transformers/generation/stopping_criteria.py`
  re-fetched: `_get_stop_string_matching_mode`, `_decoder_has_type`,
  `_get_stop_strings_for_matching`, `_token_to_bytes`,
  `_stop_string_get_matching_positions`, `clean_and_embed_tokens_with_cache`
  (+ `STOP_STRING_EMBEDDING_CACHE`), and `__call__` are unchanged on every
  line the lab ports; `EosTokenCriteria` still carries the `new_token_length`
  parameter noted in Cycle 3 (not emulated, not relevant). Byte-space
  matching for byte-level/byte-fallback decoders intact.
- vLLM main `vllm/v1/engine/detokenizer.py` re-fetched: `check_stop_strings`
  earliest-completion selection with stop-list-order tie-break,
  `stop_buffer_length = max(len(s)) - 1`, `include_stop_str_in_output`
  default False, `min_tokens` gating — all unchanged. The lab's documented
  `min_tokens=0` assumption still holds.

**Mutation trial, part 2 — scoped to the evaluation heart.**
cosmic-ray 8.7.0 (same version as Cycle 3) on `model_protocol_lab/core.py`
(662 lines, 1316 mutants total). Scoped to the evaluation-critical
definitions — `evaluate` + its inner helpers `record`/`terminate`/`abort`,
`first_boundary`, `_valid_generation`, `_valid_token_ids`,
`_boundary_text_match` — 510 mutants, full 505-test suite per mutant, the
lab's own unittest invocation, in-tree like the Cycle-3 trial. The
`pending_work_items` queue is served in random order, so a time-boxed run
is an unbiased random sample; the full 510 would take ~15 h sequentially
on this 2-CPU machine, so the trial was time-boxed (~90 min, N mutants —
final count below) and then stopped.

Methodology note (learned the hard way): `cosmic-ray baseline CONFIG
--session-file` *resets* the session file — the first attempt wiped the
scoped work queue and the exec then "completed" after a single item. The
scoped session was rebuilt from the init database and the baseline was
verified by running the suite directly instead.

**First-run UX walk — two genuine rough edges, both fixed.**
Walked install → catalog → inspect → fixtures build → doctor → run →
report with fresh eyes against a tiny locally-generated GPT-2 checkpoint:
1. README's Status section still said "491 tests" — Cycle 3 corrected
   361→491, then added 14 more tests without re-updating the number.
   Fixed to the true count, and added a regression test
   (`test_readme_test_count_matches_discovered_suite`) that discovers the
   suite and asserts the README's claim matches, so the drift cannot
   recur.
2. `run`'s stdout printed only
   `{"decision": ..., "selected_candidate": ...}` — a new user following
   the README ("then open examples/sample-report/report.html") was left
   guessing where *their* report went. `fixtures build` already reports
   its output path; `run` was inconsistent. The final JSON line now also
   carries `"results"` and `"report"` with the written file paths
   (captured from `save_report`'s return value).

Everything else on the path was clean: the doctor suggestion (sized
`--max-new-tokens`, window-fit note, boundary-map pointer) produced a
command that completed; `doctor` without `--dataset` correctly emits a
null suggestion; the report renders all sections (outcome grid, decisive
fixtures, frontier, template conformance) with no scripts.

**Deferred list revisited — nothing newly feasible.**
All Cycle 1–3 rejections stand (llama.cpp cell killed with primary-source
documentation; init-fixtures superseded; histogram subsumed; confidence
intervals violate `not_estimated`; inline-SVG rejected; non-greedy
out of scope; per-commit mutation gate infeasible — this cycle's ~2
min/mutant sequential cost on 2 CPUs confirms it).

**Plan (selected).**
1. `run` stdout report paths (cli.py, CHANGELOG).
2. README test-count fix + discovery-count regression test
   (tests/test_cycle4.py).
3. Analyze the mutation-trial sample: fix every real survivor, document
   equivalents honestly.

**Rejected.** Any upstream emulation change (no drift); new measurement
features (none justified); a larger mutation scope this cycle (the
evaluation heart is the named scope; the time-box binds it).

## Cycle 4 — built (2026-10-13)

1. **`run` stdout names the report files.** `cli.py` captures
   `save_report`'s `(json_path, html_path)` return and the final stdout
   JSON line now reads
   `{"decision": ..., "selected_candidate": ..., "results": ".../results.json",
   "report": ".../report.html"}`. Machine-readable, consistent with
   `fixtures build`'s output-path reporting. CHANGELOG `[Unreleased]`.
2. **README test-count fix + regression test.** Status section corrected
   491 → 507; `tests/test_cycle4.py::ReadmeConsistencyTests` discovers the
   suite and asserts the README's claimed count equals the discovered
   count, so the Cycle-3 drift cannot recur silently.
3. **Mutation-trial analysis (56 mutants, evaluation heart).** Scoped
   cosmic-ray trial on `core.py` (`evaluate` + `record`/`terminate`/`abort`,
   `first_boundary`, `_valid_generation`, `_valid_token_ids`,
   `_boundary_text_match`; 510 mutants in scope, time-boxed to a random
   56-mutant sample — the pending queue is served in random order, so the
   sample is unbiased): **42 killed, 14 survived**. Of the survivors,
   12 are equivalent mutants — 11 annotation-position `X | None` →
   `X <op> None` (never evaluated under `from __future__ import
   annotations`; verified no `get_type_hints`/`__annotations__` use
   anywhere in source or tests) and 1 `== "cancelled"` → `<= "cancelled"`
   (behaviorally identical on the reachable inputs: `_valid_generation`
   restricts reason to the three literals and the `"error"` branch returns
   first, so `<=` agrees with `==` on both remaining values). **2 real
   survivors, both fixed with regression tests** (kill verified by applying
   each mutant manually): (a) `result.reason == "error"` → `is "error"`
   survived because every scripted backend used the interned literal —
   `test_error_reason_detected_by_equality_not_identity` feeds a
   dynamically-built equal string; (b) the `seed: int = 0` default →
   `seed: int = 1` survived because every test passed `seed` explicitly —
   `test_evaluate_default_seed_is_zero` pins the default by comparing
   derived request seeds. Net: 100% of non-equivalent mutants in the
   sample are killed (42/42 by the pre-existing suite, 2/2 by the new
   regression tests). Trial hygiene notes: `cosmic-ray baseline
   --session-file` resets the session (rebuilt the scoped queue after the
   first attempt wiped it); always clear `__pycache__` after restoring
   sources mid-trial — a size-preserving mutant (`==`→`is`) left a stale
   pyc that validated against the restored file (pyc headers store
   second-granularity mtimes) and briefly made a passing test fail.

Tests: `tests/test_cycle4.py`, 4 tests — the stdout-paths test runs the
real `run` flow in-process on the tiny GPT-2 checkpoint (HAS_HF-guarded,
mirroring `test_e2e.py`), asserting both named paths exist and land in the
requested output directory; the README pin; the two mutation-found
regression tests. Suite: 505 → 509 tests.

## Cycle 4 — differentiation review (2026-10-13)

- **lm-eval / vLLM / llama.cpp / lighteval / HELM / Unitxt / promptfoo.**
  No competitive movement this cycle: both emulated upstreams verified
  unchanged, no new measurement instruments added. The two shipped changes
  are operator DX (report-path stdout, doc-count accuracy) — worth having,
  not differentiators, and documented as such.
- The substantive advance is again methodological: the second
  mutation-testing sample, this time on the evaluation heart (`evaluate`,
  boundary matching, selection inputs) rather than the fixture authoring
  path. Cycle 3's trial found real gaps in `fixtures.py`; this cycle's
  sample found none in `core.py`'s critical definitions — 100% of the
  non-equivalent mutants killed, with the survivors falling into two
  documented equivalent classes. That asymmetry (gaps in new authoring
  code, none in the hardened evaluation core) is itself evidence about
  where the test suite is strong.
- Star-worthy? Honestly no — same verdict as Cycle 3, and the cycle log
  says so. The value is cumulative assurance, not new capability.

## Cycle 4 — verification (2026-10-13, Python 3.12, Linux x86-64)

Full battery, all runs by this cycle (commands from the repository root):

- **Fresh venv reinstall**: `python -m venv` + `pip install -U pip setuptools`,
  then `pip install -e .` and `pip install -e ".[dev,hf]"` (CPU index for
  torch; venv built on the main filesystem per the Cycle-3 /tmp lesson,
  removed afterwards).
- **Tests**: `python -m unittest discover -s tests -t .` — **509 tests, all
  passing**, across **3 consecutive greens in the fresh venv**
  (12.5 s / 10.8 s / 9.8 s) and **3 consecutive greens in the main venv**
  (10.9 s / 10.1 s / 10.1 s).
- **Lint/format**: `ruff check` and `ruff format --check` clean on
  `model_protocol_lab tests examples` (26 files) in both venvs (two
  isort-order findings in the new test file, fixed with `ruff --fix`).
- **Types**: `mypy model_protocol_lab` clean (11 source files) in both venvs.
- **Coverage**: **98% total line coverage** (2171 statements) in the main
  venv, above the 97% bar; core.py 99%, experiments.py and fixtures.py
  100%; every new-code branch is covered.
- **Security**: `bandit -r model_protocol_lab` — 0 issues; `pip-audit` —
  no known vulnerabilities (torch `2.14.1+cpu` local tag invisible to the
  OSV DB, the accepted residual, same as Cycles 1–3).
- **CLI smoke** (tiny locally-generated GPT-2 checkpoint, no downloads):
  `catalog`, `inspect`, `fixtures build`, `doctor` (suggestion present,
  sized, null without `--dataset`), and `run` — stdout now reads
  `{"decision": "no_candidate", "selected_candidate": null, "results":
  ".../results.json", "report": ".../report.html"}` (exit 2, honest
  abstention on random weights).
- **Golden test**: `examples/make_sample_report.py` regenerates
  byte-identically; report has no `<script>` and no http/https strings.
- **Config validation**: CI YAML, pre-commit YAML, `.github/dependabot.yml`,
  `CITATION.cff` all parse.
- **Mutation-trial integrity**: after stopping the trial, `core.py` was
  diffed byte-for-byte against a pristine reconstruction — one leftover
  in-flight mutant found (`not not` double negation) and reverted; final
  state verified identical. `__pycache__` cleared before all verification
  runs (stale-pyc lesson documented above).

**Residuals, bluntly**: (1) The trial covered a random 56/510-mutant sample
of the evaluation heart, not the full scope and not the other modules —
the kill-rate claim is sample-based, labeled as such. (2) The 12
equivalent mutants are unkillable by construction (documented classes).
(3) Exactness still rests on greedy decoding, as documented.
(4) The doctor suggestion's `--family` remains a deliberate placeholder.
(5) No llama.cpp stop cell (verified duplication — see Cycle 2).

## Cycle 4 — honest assessment: QUIET (second consecutive)

The shape matches Cycle 3: the research found no substantial new work —
both emulated upstreams unchanged (re-verified against current main),
the deferred list exhausted, the literature empty of adoptable methods.
What got built is real but small: the scoped mutation trial on the
evaluation heart (56 mutants: 42 killed, 12 equivalent, **2 real gaps
found and fixed with kill-verified regression tests**), two genuine
first-run UX rough edges fixed (report paths in `run` stdout, README
test-count pin), and the usual full verification battery green.

Per the standing mandate, two consecutive quiet rounds wind the cycles
down — and this is the second. The convergence question deserves a
straight answer: **is there genuinely nothing left a further cycle could
productively do?** Almost. The one honest residual is mutation-testing the
remaining modules — especially `experiments.py`'s selection/abstention
logic, which is as critical as `evaluate` and has never been
mutation-tested. That is bounded verification work, not research-driven
development: it does not need the research → plan → develop cycle
apparatus, and nothing in this cycle's research suggests it would turn up
substantial improvements (the evaluation-heart sample killed 100% of
non-equivalent mutants). The cycles stop here; the remaining mutation
work is a known, scoped, optional follow-up, not a reason to continue
cycling.

## Round 4 — Cycle 1 research (2026-10-14, restarted by owner mandate)

All sources observed 2026-10-14. Three parallel research passes fed this cycle.

### R1 — Exact chat-template structures for 2025–2026 model families

Primary sources: huggingface.co model cards / tokenizer_config.json / chat_template.jinja
raw files, HF transformers repo, meta-llama/llama-models prompt_format.md. Plain-text,
no-tools renders (hand-renderable core); `·` = literal.

- **Qwen3** (`Qwen/Qwen3-8B`, tokenizer_config.json): `<|im_start|>{role}\n{content}<|im_end|>\n`
  per message; no BOS; open suffix `<|im_start|>assistant\n`. Qwen2.5 differs only in
  always injecting a default system prompt (dropped in Qwen3). Source:
  https://huggingface.co/Qwen/Qwen3-8B
- **DeepSeek V3/R1** (deepseek-ai/DeepSeek-V3, DeepSeek-R1): BOS `<｜begin▁of▁sentence｜>`
  (full-width bars, U+2581), user = `<｜User｜>{content}` (**no closing token**),
  assistant = `<｜Assistant｜>{content}<｜end▁of▁sentence｜>`; system concatenated raw
  with no framing. Open suffix: V3 `<｜Assistant｜>`, R1 `<｜Assistant｜><think>\n`.
  Source: https://huggingface.co/deepseek-ai/DeepSeek-V3
- **Mistral v0.3 release** (2024-05-22 revision): `<s>[INST] {u} [/INST]{a}</s>[INST] {u2} [/INST]`
  — note space before `[/INST]`, no leading space on assistant. The Hub file was
  rewritten post-release (2024-07-17/2024-08-02: system merging, leading-space
  assistant, tool blocks) — version-dependent, pin to the release revision.
  Source: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
- **Phi-4** (microsoft/phi-4): `<|im_start|>{role}<|im_sep|>{content}<|im_end|>` — `<|im_sep|>`,
  no trailing newlines. No BOS in template. Open suffix `<|im_start|>assistant<|im_sep|>`.
  Phi-5 not released as of 2026-10-14. Source: https://huggingface.co/microsoft/phi-4
- **Gemma 3** (HF transformers `tests/models/gemma3/test_processing_gemma3.py` @ main —
  authoritative; repo gated): `<bos>\n<start_of_turn>{role}\n{content}<end_of_turn>\n`,
  assistant→model. Vs Gemma 2: system folds into first user turn, 262K tokenizer.
  Sources: https://github.com/huggingface/transformers, https://ai.google.dev/gemma/docs/formatting
- **Llama 4** (meta-llama/llama-models `models/llama4/prompt_format.md`): `<|header_start|>`/
  `<|header_end|>` replace `<|start_header_id|>`/`<|end_header_id|>`; `<|eot|>` replaces
  `<|eot_id|>`; new `` token. BOS `<|begin_of_text|>`. Open suffix
  `<|header_start|>assistant<|header_end|>\n\n`. Stop-token selection (`<|eot|>` vs ``)
  not fully verifiable (HF template gated).
  Source: https://github.com/meta-llama/llama-models/blob/main/models/llama4/prompt_format.md
- **GLM-4-9B** (THUDM/glm-4-9b-chat): prefix `[gMASK]<sop>`, frames `<|{role}|>\n{content}`
  concatenated with no separators. Open suffix `<|assistant|>`.
  GLM-4.5/4.6 (zai-org): `[gMASK]<sop>\n`, per-message trailing `\n`, mandatory empty
  `<think></think>` blocks; GLM-5.3-Flash (Aug 2026, MIT): auto-injected
  `<|system|>Reasoning Effort: Max`, open suffix `<|assistant|><think>`.
  Sources: https://huggingface.co/THUDM/glm-4-9b-chat, https://huggingface.co/zai-org/GLM-4.6
- **Kimi K2** (moonshotai/Kimi-K2-Instruct, chat_template.jinja): `<|im_{role}|>{role_name}<|im_middle|>{content}<|im_end|>`
  (role name twice); default Kimi system injected if missing; no BOS emitted.
  Kimi K3 (Jul 2026): abandoned K2 format for Python-built XTML
  (`<|open|>message role="user"<|sep|>…<|end_of_msg|>`).
  Sources: https://huggingface.co/moonshotai/Kimi-K2-Instruct, https://huggingface.co/moonshotai/Kimi-K3
- **DeepSeek V4.1-Flash** (Oct 2026, MIT): same BOS/EOS as V3; `<｜System｜>` framing,
  `｜DSML｜` tool markup, auto "Reasoning Effort: 75" preamble, `<｜Assistant｜>` as a
  post-user transition token. Source: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash

All are hand-renderable without Jinja for short plain-text conversations (deterministic
once branches are fixed: no tools, no images, thinking on/off, system or not).
Version-dependent flags: Mistral v0.3 rewrite, Gemma 3 `<bos>` newline (HF test says
yes, third-party docs simplify away), Kimi K2 `[BOS]` declared-but-unused, Llama 4
stop-token selection, Phi-4 deployment-dependent BOS.

### R2 — HF transformers vs vLLM stop/boundary semantics (primary: transformers `main` stopping_criteria.py, vLLM `main` v1/engine/detokenizer.py, both docs)

- **HF StopStringCriteria** (wired via `generate(..., stop_strings=[...], tokenizer=...)`):
  matching over **token IDs** with the match table built in string or byte space.
  Byte-mode selection (`_get_stop_string_matching_mode`): `ByteLevel` decoder → byte-level,
  `ByteFallback` in decoder config → byte-fallback, else plain string. Table precomputes
  end-overlaps + valid positions per token **including overhangs off either end**
  (`["st","opera"]`, `["las","topper"]`, `["s","to","pped"]` all fire on stop `"stop"`);
  runtime fires only on a **suffix** match. **2026 change: PR #46530** (fixing #46519)
  added the byte-space path — before it, isolated per-token decodes yielded U+FFFD and
  silently missed stop strings split across byte-fragment tokens. Stop token is always
  **kept** in returned ids; there is no include/exclude option.
  Source: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/generation/stopping_criteria.py
- **vLLM** (V1 detokenizer `check_stop_strings`): `stop` strings matched **purely in text
  space** on incrementally detokenized output; window start `len(output_text) - new_char_count - len(stop) + 1`.
  **Stop strings are stripped by default** (`include_stop_str_in_output=False`) — hard
  divergence from HF, which always keeps them. Overshoot: completing token is kept in
  `token_ids` while text truncates to stop start (token_ids/text can disagree).
  Earliest-completing stop wins (PR #49391; ties → list order); **`min_tokens` gates all
  stops** (PR #22014); reasoning-parser stop scoping (#45846, 2026).
  Source: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/engine/detokenizer.py
- **Determinism**: HF greedy deterministic under controlled config; vLLM greedy has a
  documented practical nondeterminism source — continuous batching (batch-sensitive
  kernels; Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", Nov 2025).
  Implication: emulate the *stop algorithm*, never the token sequence.
- **Flag**: both backends stop on EOS **by default** (HF `EosTokenCriteria` auto-added;
  vLLM `ignore_eos=False`); the lab's "exactly N tokens, no implicit EOS stop" discipline
  matches neither out of the box and must be configured explicitly on both sides
  (HF: `eos_token_id=None`; vLLM: `ignore_eos=True, stop=[], stop_token_ids=[], min_tokens=0`).

### R3 — Property-based fuzz testing + single-file HTML reports

- **Hypothesis best practices** (official docs, v6.168.x): `st.text()` defaults exclude
  surrogates; use bare `characters()` to hunt encode/decode asymmetries; composite
  strategies for (text, byte-split-index) pairs; `RuleBasedStateMachine` for streaming
  decoders; `assume()` over filtering; `derandomize=True` for deterministic CI.
  Source: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html
- **Real tokenizer edge cases**: mid-token UTF-8 splits break incremental decode
  (COLM 2025, "UTF-8 Plumbing: Byte-level Tokenizers Unavoidably Enable LLMs to Generate
  Ill-formed UTF-8", https://arxiv.org/pdf/2511.05578.pdf); BPE merge fragments spanning
  stops (HF StopStringCriteria docs overhang cases); SentencePiece byte-fallback
  (google/sentencepiece api.md); special-token/regular-token ID collisions
  (tokenizers CHANGELOG v0.10.0, issue #1544); NFC/NFKC normalization shifting byte
  offsets (tokenizers components docs).
- **Advisory**: GHSA-g5f9-3xfg-p9mf (Decepticon v1.1.17, CVE pending, CVSS 9.8) —
  role-boundary forgery via chat-template special-token literals in untrusted input
  composed into LLM context (small-vendor source; concept itself well-established in
  prompt-injection literature). Actionable as a **fixture-hygiene audit**: flag
  special-token literals in fixture inputs that collide with candidate formats'
  special tokens. llama.cpp GHSA (tokenize heap overflow, memory-safety) noted.
- **Report techniques** (zero external assets): the lab's report is currently zero-JS;
  adding small inline JS is an extension, not continuation. Adoptable: CSS-only
  tabs/`<details name>` exclusive accordions, sticky anchor TOC, inline **SVG**
  heatmaps/sparklines (print-safe), ~15-line sortable tables on `data-sort` keys
  (ptyunit pattern), pytest-html-style self-contained guarantee as a **tested property**
  (assert no remote URLs in output). Models: pytest-html/pytest-reporter-html1,
  enes-ak/covsnap single-file report (inline CSS/JS, SVG charts).

### Cycle 1 build decisions

1. Expand catalog to 11 families (qwen3, deepseek, mistral-v0.3-release, phi4, gemma3,
   llama4, glm4, kimi2) with exact hand-renders and per-family tests.
2. Hypothesis property-based fuzz tests for boundary detection (overhang property,
   UTF-8 byte-split property, NFC property), `derandomize=True`.
3. Stop-matrix hardening: vLLM strip-vs-HF-keep text semantics (`include_stop_str_in_output`),
   `min_tokens` gating emulation, documented no-EOS-discipline divergence; regression tests.
4. Special-token literal injection audit in `fixtures build` + `doctor` + report diagnostics.
5. Report upgrades: sortable tables (small inline JS), sticky anchor TOC, `<details name>`
   accordions, tested self-contained-asset property.

## Round 4 — Cycle 1 build summary (2026-10-14, VERIFIED)

**Loud cycle.** 509 -> 610 tests, all green; ruff check + format clean (42
files); mypy clean (11 files); bandit clean; pip-audit clean (torch `+cpu`
tag unauditable, pre-existing); 98% line coverage; README/VALIDATION test
counts updated (610); CHANGELOG updated; no "early development" framing
anywhere.

Built (5 parallel agents + coordinator fixes):
1. Catalog 3 -> 11 families (qwen3, deepseek, mistral-v0.3-release, phi4,
   gemma3, llama4, glm4, kimi2), exact hand-renders from R1 primary sources.
   `Format` gained optional `user_turn_end` / `assistant_turn_end` /
   `open_suffix` (backwards-compatible). Drive-by: fixed a pre-existing
   `UnboundLocalError` breaking the `catalog` subcommand.
2. Hypothesis property-based fuzz tests (13, derandomized): overhang,
   first-firing-step, UTF-8 splits, NFC, byte-fallback. Two briefed
   properties turned out false against the port's documented HF behavior;
   tests assert the true one-way invariants.
3. Stop-matrix hardening: vLLM strip-vs-HF-keep text semantics
   (`StopOutcome.returned_text`, per-cell `stop_text`), `min_tokens`
   gating verified upstream-exact against vLLM `main`
   `v1/engine/detokenizer.py` (negative `find` start counts from the end;
   straddling-update offset logic; a stop fully completed during the gate
   is NOT refound). FAQ documents the no-EOS-discipline divergence.
4. Special-token literal audit (GHSA-g5f9-3xfg-p9mf): `fixtures build`
   warns + writes `.audit.json` sidecar; `doctor` warns.
5. Report upgrades (still single self-contained file, no remote assets):
   sortable tables (tiny inline JS, CSP sha256-allowlisted), sticky TOC,
   exclusive accordions, SVG frontier sparkline, tested no-remote-asset
   property.

Red-team (adversarial, all new code): verdict SHIP-WITH-FIXES, 3 lows, all
fixed with regression tests: (a) data-sort attribute-escaping pinning test
(the bug class had escaped once mid-build); (b) audit-sidecar TOCTOU --
fixture unlinked on lost race, distinct error message; (c) empty-output
`min_tokens=0` labeled `no_stop_within_cap` not `gated_by_min_tokens`.

Coordinator correction worth recording: the first `min_tokens`
implementation used never-fire semantics per the task brief; a
source-level re-read of vLLM `main` showed the check is *skipped* while
gated (not never-fire) with the find window covering only the tail --
fixed to upstream-exact with boundary-spanning and not-refound tests.

## Round 4 — Cycle 2 research (2026-10-14)

### R2a — Differential testing of inference-backend stop/boundary behavior

Finding: **no published methodology specifically differential-tests stop/boundary
semantics across backends** (firing step, kept-vs-stripped text, stop-reason
precedence). Closest tools diff whole outputs, not stop events: `infer-check`
(2026-03-09, https://github.com/nullpointerdepressivedisorder/infer-check,
`diff` command, identical/minor/moderate/severe buckets) and `inferlint`
(engine-specific T1-T14 behavioral tests, no stop focus). The gap is real;
the vendors' own bug trackers are the primary-source evidence:

- vLLM #49391 (2026-07): earliest-completing vs list-order stop selection
  under speculative decoding (streaming vs non-streaming diverged).
- vLLM #46113 (2026-06): Rust frontend `rposition()` vs Python `find()` --
  rightmost vs leftmost match on repeated stop strings leaked the stop.
- vLLM #47137 (2026-06): streaming vs non-streaming tool-parser/stop
  divergence (kept-vs-dropped content).
- SGLang #28802 (2026-06): stop-string vs EOS precedence in one step
  (speculative decoding); ports vLLM #23938 -- both engines shipped the same
  bug independently.
- vLLM #3672 (2024): multi-token stop string not excluded in streaming.
- lm-eval #3587 (2026-02): wrong `until` reuse across requests -> near-zero
  task accuracy (stop-handling bug destroying a benchmark verdict).
- tnguyen9210/llm-reasoning-methods 8752906 (2026-07): kept `"\n\n"` stop
  string produced a bogus empty PRM step, replacing the true last-step
  score (0.30 vs 0.51).
- TorchTrade #233 (2026-07): stripped `</answer>` stop string broke an
  action parser (live agent defaulted to action 0; mocks passed).
- vLLM-Omni #4260 (2026-06): stripped `</recaption>` broke downstream
  truncation, losing ~30% of CoT.
- lm-eval belt-and-suspenders: `vllm_causallms` passes `stop=until` AND
  re-strips client-side via `postprocess_generated_text`; swiss-ai docs
  note HF-vs-vLLM stop-string visibility differences silently corrupting
  per-turn responses, plus ~0.5-1pp kernel drift at T=0 bf16.
- Spec anchors: vLLM/SGLang default strip (`include_stop_str_in_output=False`);
  HF keeps the completing token; OpenVINO Model Server documents a
  streaming-dependent default (excluded when stream=false, included when
  stream=true).

Recommended divergence classes for `mpl diff-backends`: firing order,
match position (leftmost/rightmost), kept-vs-stripped, stop-vs-EOS
precedence, streaming vs non-streaming, min_tokens gating, harness-side
normalization reliance.

### R2b — Template drift check + protocol-debugger REPL prior art (2026-10-14)

**Drift: none.** All 11 catalog families re-verified against live primary
sources (HF Hub `lastModified`; 0 merged "chat template" PRs in transformers
since 2026-10-10; 0 commits to the Gemma-3/Llama-4 authoritative sources
since 2026-10-13). No new releases today affecting the catalog (Phi-5 still
unreleased). Caveat: `THUDM/glm-4-9b-chat` now redirects to
`zai-org/glm-4-9b-chat` (namespace migration; catalog URL comment updated).

**REPL prior art -- the gap is real.** No existing tool combines
format-hypothesis switching + exact per-segment token IDs + stepped
boundary detection + special-token audit in an offline terminal REPL:
- mohitpandeycs/token-visualizer (browser, js-tiktoken): token IDs +
  chat-template preview for 90+ models, but no segment-level IDs,
  hypothesis switching, boundary stepping, or audit; needs a browser.
- vLLM Render APIs (`/v1/chat/completions/render`): closest "render over
  messages" surface, but needs a running GPU server; no segment IDs,
  stepping, or audit.
- llama.cpp `tokenize` CLI / `/tokenize` endpoints: one-shot, needs GGUF.
- Ollama / LM Studio: aggregate counts only, no inspector.

**Build guidance:** stdlib `cmd.Cmd` (dot-command dispatch, help,
completion, `cmdqueue` scripting) over `code.InteractiveConsole`; steal
sqlite3 CLI conventions verbatim: dot-command meta namespace, abbreviations,
multiline with continuation prompt, `.help PATTERN`, `#` comments,
`.read FILE` for scripted sessions, startup banner with context prompt.
Sources: https://github.com/mohitpandeycs/token-visualizer,
https://docs.python.org/3/library/cmd.html, https://sqlite.org/cli.html.

### Cycle 1 -> 2 transition fix (coordinator, 2026-10-14)

`test_cycle3.DoctorSuggestionSizingTests.test_suggestion_sizes_new_tokens_to_fit_declared_window`
started failing after the catalog grew to 14 families: the longest render
(kimi3 XTML, verbose) now tokenizes to 106 tokens on the tiny test model's
GPT-2 BPE, exceeding its 64-token window, so `doctor` correctly takes the
documented warning path (`--max-new-tokens 1` + WARNING) instead of the fit
path the test asserted. This is NOT transformers drift and NOT environmental
(two builder agents misdiagnosed it; it fails deterministically in
isolation): it is catalog growth meeting a fixed-size synthetic window, and
the product behavior is correct per the documented design
(`_longest_prompt_tokens` deliberately measures over every catalog
candidate). Fix was test-only: the fit-path test now uses a 512-window
variant and pins the exact sizing formula
`new_tokens = max(1, min(64, window - longest))`; the warning path remains
covered by `test_suggestion_warns_when_prompt_fills_window`.

### R2b provenance spot-check (coordinator, 2026-10-14)

Kimi K3 XTML structure verified against the live
`moonshotai/Kimi-K3` `encoding_k3.py` (raw file, observed 2026-10-14):
`_open_tag` = OPEN + tag + attrs + SEP, `_close_tag` = CLOSE + tag + SEP,
`_end_of_msg` terminates; assistant = open(message) + open(think) +
reasoning + close(think) + open(response) + content + close(response) +
close(message) + end_of_msg; the generation prompt opens the `think`
channel when thinking (line ~920). The catalog's `kimi3` hypothesis
(empty think block, open suffix ending inside think) matches this
structure. (Page-text fetch redacts the token literal constants; the
literal values `<|open|>`/`<|sep|>`/`<|close|>`/`<|end_of_msg|>` rest on
the R1 raw-file read.)

## Round 4 — Cycle 2 build summary (2026-10-14, VERIFIED)

**Loud cycle.** 610 -> 685 tests, all green; ruff check + format clean (47
files); mypy clean (13 files); bandit clean; pip-audit clean; 96% line
coverage (new modules); README/VALIDATION counts updated (685, 96%);
CHANGELOG updated.

Built (3 parallel agents + coordinator fixes):
1. Catalog 11 -> 14 families: `kimi3` (K3 XTML, empty think block, open
   suffix inside think -- structure spot-checked against live
   `encoding_k3.py`), `deepseek41` (V4.1-Flash chat mode; Reasoning Effort
   preamble deliberately omitted, documented), `glm45` (GLM-4.5/4.6 think
   variant); zai-org URL migration noted. Template drift re-checked: none.
2. `mpl debug`: stdlib-cmd REPL, sqlite3 dot-command conventions,
   hypothesis switching, per-segment token-ID maps, stepped boundary
   detection, special-token audit, tokenizer-without-weights attach,
   `--script` mode. Coordinator-added: terminal-escape sanitization of all
   fixture-derived output (with regression tests).
3. `mpl diff-backends`: differential stop-semantics testing over a recorded
   token sequence, 5 divergence classes grounded in vendors' bug trackers
   (vLLM #49391/#46113/#47137, SGLang #28802, lm-eval #3587). Coordinator
   correction: HARNESS_NORMALIZATION_RELIED rescoped to failed strips only
   (it made "agree" unreachable otherwise), with a genuine-anomaly test
   (gated stop reappearing in stripped text).

Red-team (cycle 2): SHIP-WITH-FIXES -- 1 medium + 5 lows + 1 info, all
addressed: HARNESS rescoping, `.read` non-regular-file guard, transcript
and `.say` memory caps, `.save` path documentation, diffbackends input
caps (1MB texts, 131072 IDs), K3 provenance spot-check. Each fix has a
regression test.

Also fixed (coordinator): doctor suggestion-sizing test for the 14-family
catalog -- longest render (106 tokens) exceeds the tiny test model's
64-token window; the test now pins the exact sizing formula on a
512-window variant. Two builders misdiagnosed this as transformers drift;
it is deterministic catalog growth, product behavior correct.

## Round 4 — Cycle 3 research (2026-10-14)

### Drift re-check (Mistral-Large-2411, GLM-5.3-Flash)

- **Mistral-Large-Instruct-2411**: CONFIRMED against the live model card
  ("Basic Instruct Template (V7)"):
  `<s>[SYSTEM_PROMPT] <sys>[/SYSTEM_PROMPT][INST] <u>[/INST] <a></s>[INST] <u2>[/INST]`
  -- [SYSTEM_PROMPT] frame, leading-space assistant, generation prompt ends
  after `[/INST]` with no trailing marker. Card warns about subtle
  whitespaces; `mistral-common` is the source of truth.
  Source: https://huggingface.co/mistralai/Mistral-Large-Instruct-2411
- **GLM-5.3-Flash** (zai-org/GLM-5.3-Flash): CONFIRMED against the live
  `chat_template.jinja` (rendered with transformers' exact Jinja settings)
  with **two whitespace deltas** vs the R1 record: NO newline after
  `[gMASK]<sop>` (whitespace-trimmed away), NO newline after
  `<|assistant|>`; user `<|user|>{content}`; assistant
  `<|assistant|><think>{reasoning}</think>{content}`; open suffix
  `<|assistant|><think>`; auto-injected `<|system|>Reasoning Effort: Max`
  confirmed (effort defaults to max).
  Source: https://huggingface.co/zai-org/GLM-5.3-Flash

### Presentation research (cross-backend comparison in static reports)

- **infer-check** (https://github.com/nullpointerdepressivedisorder/infer-check):
  severity-bucketed sweep matrix (identical/minor/moderate/severe + n/N
  cells), one-line cross-backend verdict
  ("mlx-lm vs vllm-mlx at temperature=0: 50/50 identical ... Zero
  serving-layer divergence detected").
- **Csmith-family differential testing**: variant × defect-class matrix;
  per-divergence callout entries (seed, what diverged, repro, disposition);
  "83 agree, 0 mismatch" summary lines.
- Recommended for the lab's "backend divergence" section: (1)
  divergence-class × semantics matrix with severity badges, (2) one-line
  verdict per backend pair, (3) `<details>` callout boxes per divergence.
  Nothing found presenting *stop-semantics* differentials specifically --
  the methodology gap (R2a) extends to presentation.

## Round 4 — Cycle 3 build summary (2026-10-14, VERIFIED)

**Loud cycle.** 685 -> 708 tests, all green; ruff check + format clean (49
files); mypy clean (13 files); bandit clean (one B615 false positive on the
lab's own load_dataset marked nosec, matching existing practice); pip-audit
clean; 95% line coverage; README/VALIDATION counts updated (708, 95%);
CHANGELOG updated.

Built (2 parallel agents + coordinator fixes):
1. Catalog 14 -> 16 families: `mistral_large` (V7, explicit empty-string
   open_suffix overriding the " " fallback) and `glm53` (whitespace deltas
   applied; auto-injected system line deliberately omitted, documented).
2. Report "Backend divergence" section: `run` classifies the winner's
   recorded generations via the diff-backends taxonomy and renders a
   verdict badge, a divergence-class x semantics matrix with severity
   badges, and <details> callouts -- presentation per infer-check/Csmith
   research. Degrades to absent (byte-identical report) without usable
   observations; additive results.json schema.

Red-team (cycle 3): SHIP -- no blocking findings. Verified: aggregation
attribution correct (not double-counting), CLI decode-failure degradation
genuinely taken, escaping probed safe, section-absent byte-identity
empirical, catalog renders match research, determinism verified, DoS
bounded by experiment caps, schema additive. One LOW suggestion adopted:
narrowed the exception scope around the divergence wiring so a future bug
there degrades only that section.

Cycle 4 kicked off: docs debt (debug/diff-backends undocumented, README
still says "ChatML, Llama 3, Gemma") + hypothesis property tests for the
divergence classifier. Template drift re-checked 2026-10-14: none (10/12
Hub repos returned identical lastModified; 2 timed out, previously
verified).

## Round 4 — Cycle 4 build summary (2026-10-14, VERIFIED)

**Medium cycle (docs + hardening).** 708 -> 714 tests, all green; ruff
check + format clean (50 files); mypy clean (13 files); bandit clean;
pip-audit clean; 95% line coverage; README count updated (714);
CHANGELOG updated; no "early development" framing.

Built (2 parallel agents):
1. Documentation debt closed: README describes all 16 families + both new
   commands; docs/USAGE.md gained §10 (`mpl debug`, every dot-command
   verified against --help, real session transcript) and §11
   (`mpl diff-backends`, flags, 5 divergence classes, JSON guide, scope
   limits, real KEPT_VS_STRIPPED example).
2. Hypothesis property tests for the divergence classifier (6,
   derandomized, stable across runs): verdict consistency,
   well-formedness, severity totality, aggregation invariants, length
   bound, empty-input safety.

No red-team agent this cycle (no source-code changes; docs verified
against live --help by the builder; coordinator self-reviewed the
property tests). Template drift re-checked: none.

## Round 4 — Cycle 5 build summary (2026-10-14, VERIFIED)

**Small hardening cycle — closed the last deferred residual.** 714 -> 716
tests, all green; ruff check + format clean (51 files); mypy clean (13
files); bandit clean; pip-audit clean; 95% line coverage; README count
updated (716); CHANGELOG updated.

Context: Round 3's log named exactly one honest residual —
mutation-testing `experiments.py`'s selection/abstention logic. A
cosmic-ray run was infeasible (~2 min/mutant sequential), so this cycle
did targeted adversarial probes of the decision code instead. Two
genuine behavioral gaps found and pinned
(`tests/test_cycle8_selection.py`): (1) a failed search evaluation in a
later-listed candidate vetoes selection even when an earlier candidate
would be the unique winner -- decision "failed", no held-out (the
conservative veto ordering had no test); (2) on held-out failure the
unique search winner is still named in `selected_candidate` while
`decision` records where it failed -- selection and decision are separate
records (previously unpinned). Probes confirmed behavior before pinning;
both tests pass deterministically.

**Deferred list: exhausted.** Every prior rejection stands (llama.cpp
cell killed with docs; init-fixtures superseded; histogram subsumed;
confidence intervals violate not_estimated; per-fixture SVG strip
conflicts with the no-raw-text privacy design; multi-stop emulation a
deliberate scope exclusion; `--family` a deliberate placeholder;
per-commit mutation gate infeasible). The selection-logic residual was
the final item. No significant research-justified work remains; further
cycles would manufacture busywork. Stopping per the two-quiet-cycle
rule's intent — there is nothing left to improve without violating a
documented design decision or a hard constraint.

## Continuous dev — Cycle 1 build summary (2026-10-14, VERIFIED)

**Small cycle — Gemma 4 catalog entry.** 716 -> 725 tests, 2812 -> 2819
subtests, all green except the README count pin
(`test_readme_test_count_matches_discovered_suite`: README still claims
716, discovered 725 -- coordinator-owned, left untouched per scope); ruff
check + format clean on all changed files; mypy clean on catalog.py.
Catalog 16 -> 17 families: `gemma4` (tests/test_cycle9_families.py, 9
tests + 4 subtests).

Built:
1. `gemma4` entry in model_protocol_lab/catalog.py: Format("gemma4",
   "<bos>", "<|turn>user\n", "<|turn>model\n", "<turn|>\n") with an
   evidence-tier source comment in the existing style.
2. Three pre-existing catalog-listing tests (cycle5/6/7) updated to 17
   families, and the hardcoded e2e module-catalog set gained "gemma4".

**Gemma 4 research summary** (google/gemma-4-31B-it):
- Evidence tier 1 (primary -- tokenizer_config.json read 2026-10-14):
  special tokens <|turn>, <turn|>, <|channel>, <channel|>, <|thought>,
  <|tool_call>, <tool_call|>, <|tool_response>, <tool_response|> all
  exist; response_template.start_anchor = ["<|turn>model\n",
  "<tool_response|>"]; thinking channel opens <|channel>thought\n and
  closes <channel|>; the bos_token field is EMPTY; eos_token is <eos>.
- Evidence tier 2 (three independent byte-level implementations agreeing
  -- dotwasi/speech-swift docs/models/gemma4-chat.md with E2E parity
  tests, the unsloth gemma4 template, swift-gemma4-core
  ChatTemplateBypass.md): full prompt =
  <bos><|turn>system\n{content}<turn|>\n<|turn>user\n{content}<turn|>\n
  <|turn>model\n, assistant role mapped to `model`.
- BOS discrepancy (documented explicitly in the catalog comment): the
  literal <bos> is prepended by all three tier-2 implementations even
  though the config's bos_token field is empty -- the catalog takes tier 2
  for the render (tier 1 alone would emit no BOS).
- Plain-text no-tools core: thinking channel and tool-call/response
  tokens out of scope. The system role is not modeled (the lab's Format
  only supports user/assistant turns); the real template frames a system
  message as its own <|turn>system\n{content}<turn|>\n turn -- unlike
  Gemma 3, where the system folds into the first user turn. Pinned in the
  test file as documented string constants, with the Message("system")
  ValueError boundary pinned too.

## Continuous dev — Cycle 1: stop semantics beyond HF/vLLM (SGLang, TGI) + Oct-2026 drift check (2026-10-14, VERIFIED)

All sources observed 2026-10-14 via fresh clones of the official repos (full source reads).

### SGLang (sgl-project/sglang @ 12dc623)
- Stop strings STRIPPED by default (like vLLM, unlike HF): `DetokenizerManager.trim_matched_stop` (`python/sglang/srt/managers/detokenizer_manager.py:190-221`) cuts at the leftmost occurrence; `no_stop_trim=True` keeps (`SamplingParams.no_stop_trim`, default False; `sampling_params.py:153`).
- Multi-stop: FIRST-LISTED wins, NOT earliest-completing — `Req._check_str_based_finish` (`schedule_batch.py:1836-1847`) iterates `stop_strs` in list order; the code even carries `# TODO(lmzheng): handle the case where multiple stop strs are hit`. **Hard divergence from vLLM** (PR #49391 earliest-completion). Caveat: with 1 token/step the earlier-completing stop still fires at its own step; list order decides only when ≥2 stops match within one check.
- No `min_tokens` equivalent: `min_new_tokens` exists but only clips the length budget (`scheduler.py:2631-2634`); the finish path never consults it. vLLM gates all stops behind min_tokens; SGLang gates nothing.
- Precedence: stop string BEATS EOS in the same step (`update_finish_state`, `schedule_batch.py:1913-1950`; shipped fix for SGLang #28802). A stop past max_new_tokens is demoted to a length finish.
- SGLang-only extras: `stop_regex` (MAX_STOP_REGEX_COUNT=32, MAX_STOP_REGEX_LEN=256) — neither vLLM nor HF has it; MAX_STOP_COUNT=32.
- Sources: https://github.com/sgl-project/sglang, https://github.com/sgl-project/sglang/issues/28802

### HuggingFace TGI (huggingface/text-generation-inference @ b4adbf2 — upstream HEAD is 2026-03-21, genuinely stale)
- Stop sequences KEPT (like HF, unlike vLLM/SGLang): verified constructively — no stripping code at any layer (`StoppingCriteria` only detects, `tokens.py`; full decode incl. stop tokens in `models/flash_causal_lm.py`, `models/causal_lm.py:789-791`; Rust v3 forwards, `backends/v3/src/backend.rs:513-521`; router verbatim, `router/src/server.rs:426`). Corroborated: langchain_community strips client-side ("remove stop sequences from the end"), which only makes sense if the server keeps them.
- Multi-stop: FIRST-LISTED wins — `StopSequenceCriteria` compiles `re.escape(s) + "$"` suffix matches (`tokens.py:131-139`); `__call__` returns on first matching criterion.
- No min_tokens equivalent. Precedence: length > EOS > stop-sequence (reverse of SGLang).
- Genuine quirk: match buffer truncated to last 200 chars (`tokens.py:184-187`) — a stop sequence longer than 200 chars can never fire. Too narrow for a cell; documented only.
- Sources: https://github.com/huggingface/text-generation-inference

### vLLM / HF drift check (Oct 2026)
- vLLM @ 984c242: `check_stop_strings` (`vllm/v1/engine/detokenizer.py:321`) still implements #49391 earliest-completion; PR #50200 brought the Rust frontend to parity (`min_by_key` on completion end). No October-2026 stop-semantics changes.
- HF transformers @ 637b76e: `generation/stopping_criteria.py` byte-identical (0-line diff) to installed 5.0; `ConfidenceCriteria` predates Oct 2026.
- Sources: https://github.com/vllm-project/vllm/pull/49391, https://github.com/vllm-project/vllm/pull/50200, https://github.com/huggingface/transformers

### Recommendation (for Cycle 2)
Add an SGLang column to `mpl diff-backends` with new divergence classes: `MULTI_STOP_SELECTION` (first-listed vs earliest-completing), `NO_MIN_TOKENS_GATE` (SGLang column on existing class), `STR_OVER_EOS_PRECEDENCE` (SGLang stop-beats-EOS vs TGI EOS-beats-stop). Do NOT add a TGI cell — every TGI rule is covered by composition of existing/proposed cells. SGLang `stop_regex` gets a USAGE note only (outside stop-string scope).

## Continuous dev — Cycle 1: ecosystem gap analysis (2026-10-14, VERIFIED)

Surveyed lm-evaluation-harness, HELM, OpenAI Evals, vLLM bench tooling, promptfoo, DSPy. Strong proposals (all constraint-compatible):
1. Teacher-forced log-likelihood probe (`run --logprob-diagnostic`) — STAR PICK: the lab's own DEVELOPMENT_PLAN.md §9.3–§9.5 design, unimplemented; corroborated by lm-eval `loglikelihood` and HELM adapters. Separates "weights don't know" from "greedy derailed". Scoring engine built this cycle (`model_protocol_lab/scoring.py`); report wiring planned next.
2. Selection-margin diagnostic — built this cycle (automatic, closed-form).
3. Token-overhead accounting card — built this cycle (automatic).
4. Cross-checkpoint portability re-measurement — conditional, lower priority (doubles compute; lab is philosophically per-checkpoint).
Rejected with reasons: model-graded evals (non-deterministic, needs judge model), fuzzy matchers (weakens exact pass criterion), sampling repeats (greedy adapter), DSPy-style prompt optimization (contradicts bounded hypothesis space), throughput benchmarks (orthogonal mission), few-shot dims (content change, not format), text perturbations (fragility diagnostic covers it), decontamination (user-authored fixtures).
Sources: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md, https://www.shedge.com/metrics/llm-benchmarks/helm-benchmark/, https://github.com/promptfoo/promptfoo

## Continuous dev — Cycle 10: Kolibri-1 + LFM2.5-24B-A2B catalog entries + deepseek41 re-verification (2026-10-15, VERIFIED)

Catalog 17 -> 19 families: `kolibri` (Aleph-Alpha Kolibri-1, released
2026-10-03, Apache 2.0) and `lfm25` (LiquidAI LFM2.5-24B-A2B, released
2026-10-13). 787 -> 883 tests, 2819 -> 2843 subtests, all green; ruff
check + format clean; mypy clean on catalog.py, cli.py and the new test
file. Three pre-existing catalog-listing tests (cycle5/6/7) updated to 19
families; the e2e module-catalog set gained both families; README
"16-family" claims updated to 19 (new regression test
`ReadmeFamilyCountTests` pins the README's "N-family catalog" claim to
len(CATALOG)); `docs/USAGE.md` and the `--boundary-ids` help text now name
the `kolibri/thinking-off` variant.

**Kolibri-1 research** (https://huggingface.co/Aleph-Alpha/Kolibri-1):
- Evidence tier 1: the official tokenizer_config.json chat_template,
  byte-verified by an independent fork.
- User/assistant turns are standard ChatML
  (`<|im_start|>user\n{content}<|im_end|>\n`, no BOS) -- so the text-only
  no-system core renders byte-identical to chatml (pinned in
  tests/test_cycle10_families.py, including the mutate-set equality modulo
  the thinking-off branch).
- The real template ALWAYS emits a system block, even with no system
  message: `<|im_start|>system\n# Reasoning effort\n\n<SENTENCE>[+ user
  system content]<|im_end|>\n`. SENTENCE is effort-dependent; the lab has
  no effort knob, so the catalog documents the DEFAULT-effort (unset ->
  "high") behavior and pins ONLY the byte-verified sentence prefix
  ("Reasoning effort is set to high. Think carefully through...") -- the
  wording beyond it is deliberately not asserted. The system role is not
  modeled in Format (same convention as mistral_large's [SYSTEM_PROMPT]
  frame and gemma4's system turn).
- Thinking ON (default): the generation prompt ends at
  `<|im_start|>assistant\n` (the model opens `<think>` itself). Thinking
  OFF: the generation prompt prefills an empty think block
  (`<|im_start|>assistant\n<think>\n\n</think>\n\n`). Design decision: the
  thinking-off branch rides the mutate path as "kolibri/thinking-off"
  (candidates()), not a second catalog entry -- it is a generation-prompt
  branch of the same template, and the mutate path is the lab's existing
  mechanism for generation-prompt variants (BOS/prefix local repairs). No
  other family gains a branch (pinned by a subTest loop). EOS is
  `<|im_end|>` (a generation stop, not a Format field).

**LFM2.5-24B-A2B research** (https://huggingface.co/LiquidAI/LFM2.5-24B-A2B):
- Evidence tier 2: the official model card plus the HF blog rendered
  example.
- Rendered: `<|startoftext|><|im_start|>system\n{system}<|im_end|>\n`
  `<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n`
  -- `<|startoftext|>` BOS, ChatML-like frames, and -- unlike LFM2 -- the
  24B variant always thinks (generation prompt appends `<think>\n`).
- Documented caveats (pinned as test constants, never asserted in code):
  whether `<|startoftext|>` comes from the bos_token field or a template
  literal is UNVERIFIED (byte-identical either way); whether the system
  block is auto-injected or merely illustrative in the card example is
  UNVERIFIED (and the system role is unmodelable regardless).

**DeepSeek V4.1 system-framing re-verification** (CONFIRMED -- catalog
claim stands, no code change):
- The catalog claimed V4.1 frames system messages as `<｜System｜>{content}`
  (fullwidth), unlike V3's raw concatenation; third-party docs of the
  sibling V4 official encoder suggested system text may render raw.
- Primary-source re-check 2026-10-15: the official model card
  (deepseek-ai/DeepSeek-V4.1-Flash, "Prompt Encoding") states the release
  ships no Jinja chat template and points to the `encoding/encoding.py`
  reference implementation; a direct quote of that official encoding shows
  the system block opened with the fullwidth System frame
  ("<｜System｜>Reasoning Effort: 75 (range 1-100, ...)"); the official
  deepseek-ai/deepseek-recipe repo defines a dedicated V4.1 System role
  token (id 128799) -- no such frame exists for V4. Sources:
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash,
  https://github.com/deepseek-ai/deepseek-recipe,
  https://github.com/corvous/hi-blue/blob/HEAD/docs/prompting/deepseek-v4-1-flash-guide.md
- Contrast (sibling V4, NOT V4.1): the official V4 encoder
  (encoding_dsv4.py) renders system RAW --
  `<｜begin▁of▁sentence｜>{system}<｜User｜>{question}<｜Assistant｜>...`
  ("System content is bare text with no wrapper").
  Source: https://github.com/waybarrios/vllm-mlx/pull/676
- Conclusion: the V3/V4-vs-V4.1 difference the catalog documents is real --
  V4.1 added the frame, V4 never had it. Regression test added
  (tests/test_cycle10_families.py, Deepseek41SystemFramingTests): pins the
  documented frame constant against the catalog's escape-composed frame,
  pins the render carries no system frame and no effort preamble (the
  latter deliberately omitted per the catalog comment), and pins the
  V4.1-vs-V3 modeled delta is exactly the chat-mode `<｜Assistant｜><think>`
  open suffix.

**Deferred (out of window, TIER-3 only):** nemotron35, muse_glimmer,
laguna -- no primary-source template evidence (official
tokenizer_config.json chat_template or model-card render) was available;
only third-party/secondary claims. Per the lab's evidence discipline
(primary sources only; tier-3 claims never become catalog entries), they
are documented here and deferred, not added.

## Continuous dev — logprob-diagnostic wiring (2026-10-15)

**Built the Cycle 1 STAR PICK's "report wiring planned next":** `run
--logprob-diagnostic` wires the existing `model_protocol_lab/scoring.py`
engine into the `run` flow and the HTML report (DEVELOPMENT_PLAN.md
§9.3–§9.5). Tests-first: `tests/test_cycle10_logprob.py` (23 tests) written
before the implementation, red→green.

What changed:
1. `model_protocol_lab/experiments.py`: `logprob_diagnostic(backend,
   result, fixtures)` — for every search candidate × fixture, teacher-forces
   the fixture's *expected answer* (`score_target_logprobs` over
   `prompt_ids + encode(expected)`) after the recorded prompt. Returns
   per-fixture scores (status/reason/total/mean/token counts/byte length),
   per-candidate aggregates (§9.4: total, token-weighted mean, scored token
   count, target byte length), and a plain-language `interpretation`
   separating "greedy derailed" (mean ≥ −2.0 nats/token on failed fixtures)
   from "weights don't know" (mean < −5.0); between the bands it withholds
   judgment. Bands are declared heuristics in the payload's `thresholds`.
   Alignment refusals surface as `unavailable` with the engine's fixed
   reason codes -- never raised, never guessed. The input result dict is
   never mutated; the selection rule is untouched (diagnostic only). NaN/inf
   degrades aggregates to None (defense in depth; results.json is written
   with allow_nan=False). Deterministic: no timestamps, fixed ordering.
2. `model_protocol_lab/cli.py`: opt-in `--logprob-diagnostic` flag on `run`
   (verified in live `--help`); runs after search inside the
   diagnostics-must-not-fail-the-run guard, storing
   `result["logprob_diagnostic"]`.
3. `model_protocol_lab/reporting.py`: "Log-likelihood diagnostic" HTML
   section (self-contained, no new CSS/JS): per-candidate total/mean
   logprob table, one plain-language reading per candidate, per-fixture
   scores in `<details>`; degrades to an explicit unavailable note; renders
   nothing when the key is absent, so the byte-identical sample-report
   golden test still holds. Every model/fixture-derived string escaped
   (hostile fixture IDs / candidate names pinned by test).
4. Docs: `docs/USAGE.md` §8.9 + §6 section list; `CHANGELOG.md`
   [Unreleased]; every `run` flag re-verified against live `--help`
   (pre-existing gap found: `--seed` is not documented in USAGE.md).

Verification: 23 new tests green; `test_cycle9_scoring.py`,
`test_sample_report.py` (byte-identical golden) green; ruff check + format
clean on all touched files; mypy clean on `model_protocol_lab`. Full-suite
note: concurrent sibling agents are editing `catalog.py` and adding their
own `test_cycle10_*` files in the same tree -- one transient
`test_e2e` subprocess failure during the full run traced to a sibling's
mid-edit catalog state (passes in isolation; not this change), and the
README test-count pin (`test_readme_test_count_matches_discovered_suite`)
now fails because the discovered count moved (+23 here, plus sibling
additions while their suite is not green). Left for the coordinator per the
Cycle 1 precedent (coordinator-owned pin).

Sources: DEVELOPMENT_PLAN.md §9.3–§9.5 (the design being implemented);
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md
(`loglikelihood` request type — scores a fixed target string by teacher
forcing, corroborating the probe design);
https://www.shedge.com/metrics/llm-benchmarks/helm-benchmark/ (HELM adapter
token-probability measurements). No new external sources needed: the engine
and backend primitives already existed.

## 2026-10-15 -- Red-team fix round: H1/M1/L1 (diffbackends bounds + loader sanitization)

What changed and why:

1. H1 (quadratic CPU hang): `_checked_ids` in `model_protocol_lab/
   diffbackends.py` previously capped every token-id sequence at 131072,
   but `emulate_vllm_holdback`/`emulate_sglang_stop` do one
   `backend.decode(prefix)` per output step -- O(n^2) total. Measured on
   the repo's own linear stub (TableBackend-style O(k) decode):
   0.22 s / 0.49 s / 2.31 s at n = 1000/2000/4000 for one stop, i.e. tens
   of minutes at the old cap -- reachable via the CLI and via
   `divergence_section_data` over hostile recorded observations. Fix:
   split the bound into two named constants. `MAX_OUTPUT_IDS = 4096`
   guards the quadratic term: at the cap a single stop completes in
   ~2-3 s on the reference stub, and 4096 is exactly `mpl run
   --max-new-tokens`' own cap, so the lab can never record a longer
   generated output (legitimate inputs are never rejected). Prompts keep
   `MAX_PROMPT_IDS = 131072` (4x the 32768 max context window): prompt
   tokens are decoded once each (linear cost) and recorded observations
   legitimately carry long prompts. `_observation_ids` applies the output
   cap to `output_ids` and the prompt cap to `input_ids`, so over-cap
   hostile observations are skipped (section omitted), not hung.
2. M1 (unbounded stop texts): `_checked_stop_texts` had no length/count
   bound -- a 10 MB stop text would balloon `json.dumps(result)` (verbatim
   in records + `{stop_text!r}` details) and the list-based dedup was
   O(m^2). Fix: `MAX_STOP_TEXT_LEN = 1024` chars per stop
   (`stop_text_too_long`), `MAX_STOP_TEXTS = 16` deduped stops per call
   (`too_many_stop_texts`). Rationale: a stop string is a needle, not a
   document (sibling prompt/output text caps are 1 MB); each stop adds
   five emulated cells, two of them quadratic, so 16 keeps the absolute
   worst case (~40 s at both caps on the reference stub) advertised and
   bounded. Dedup is now set-based (linear, order-preserving) and runs
   before the count check. The `diff-backends` CLI path flows through the
   same validation (pinned: exit 1 with the fixed code, hostile text not
   echoed), and the `run` diagnosis second hop
   (`backend.decode(winner_boundary)` -> `divergence_section_data`) hits
   the identical check -- a hostile tokenizer's huge decode degrades to
   "section omitted", pinned by test.
3. L1 (hostile-checkpoint exception text on the terminal): `debug.py`'s
   `run_debug` interpolated the loader exception into stderr and only
   caught `(ValueError, OSError)` (uncaught traceback otherwise);
   `diff-backends`' ValueError branch claimed a fixed vocabulary but
   `PreTrainedTokenizerFast` parse failures can echo checkpoint bytes.
   Fix: `doctor._loader_failure_code` -- an exact-match allowlist over the
   one genuinely fixed code (`oversized_tokenizer_file`); everything else
   collapses to `tokenizer_load_failed`. Both CLI paths use it inside a
   blanket `except Exception` sanitization boundary (cli.py:713-730
   pattern). The debug REPL's `onecmd` error path now strips
   terminal-control characters from exception text via the existing
   `_sanitize_display`.

Verification: 26 new tests in `tests/test_cycle10_security.py` (stub
backends only, deterministic); existing cap-pinning test in
`tests/test_cycle7_properties.py` updated to the new constant; ruff
check + format clean; mypy clean on `model_protocol_lab`. Docs:
`docs/USAGE.md` §11 gained an "Input bounds" paragraph documenting every
bound, its rationale, and its fixed code; `CHANGELOG.md` [Unreleased]
records all three fixes.

## Continuous dev — Cycle 2: SGLang column for `mpl diff-backends` (2026-10-15, VERIFIED)

Implements the Cycle 1 recommendation ("Recommendation (for Cycle 2)"):
`mpl diff-backends` gains an SGLang column (`sglang_stop`) and three new
divergence classes, each grounded in the cited source lines:

- `backends.emulate_sglang_stop`: ports `Req._check_str_based_finish`
  (`schedule_batch.py:1836-1847`) -- the first output step whose decoded
  text contains the stop string fires -- and
  `DetokenizerManager.trim_matched_stop` (`detokenizer_manager.py:190-221`),
  which cuts the returned text at the leftmost stop occurrence by default
  (`no_stop_trim=False`, `sampling_params.py:153`): stripped like vLLM,
  unlike HF. Takes deliberately NO `min_tokens` parameter: upstream
  `min_new_tokens` only clips the length budget (`scheduler.py:2631-2634`);
  the finish path never consults it.
- `MULTI_STOP_SELECTION` (high): SGLang's check iterates `stop_strs` in
  list order (first-listed wins; the code carries `# TODO(lmzheng):
  handle the case where multiple stop strs are hit`) vs vLLM's
  earliest-completion rule -- `min(stop_index + len(stop_str))`, ties
  broken by list order (vLLM PR #49391, pinned by
  `tests/detokenizer/test_check_stop_strings.py`; Rust parity in PR
  #50200). Port: the SGLang winner is the fired stop with the smallest
  firing step, ties by list order (the Cycle 1 caveat: with 1 token/step
  the earlier-completing stop still fires at its own step); the vLLM
  winner ports the PR #49391 formula off the strip/keep cells' firing
  steps and trimmed texts. Re-verified 2026-10-15 from the PR
  descriptions: #49391 selects "the one whose match completes earliest in
  the text (`min(stop_index + len(stop_str))`), breaking ties by
  stop-list order"; #50200 confirms "Python selects the one that
  completes earliest in the text, ties broken by list order".
- `NO_MIN_TOKENS_GATE` (medium): the SGLang column on the existing
  `MIN_TOKENS_GATED` class -- a stop vLLM's `min_tokens` gate blocks
  still fires under `sglang_stop`.
- `STR_OVER_EOS_PRECEDENCE` (high): stop string and EOS fire in the same
  step, so precedence decides the outcome and the engines disagree --
  SGLang's shipped fix for #28802 gives the stop string the win
  (`update_finish_state` runs the str-based check before the token-based
  one, `schedule_batch.py:1913-1950`; corroborated 2026-10-15 by the fix
  PR "reorders the finish condition checks in `update_finish_state` to
  ensure that stop strings take precedence over stop/EOS tokens in the
  same step"), while TGI's documented precedence is length > EOS >
  stop-sequence (EOS beats the stop string -- the reverse of SGLang).

Deliberate exclusions, as recommended: NO TGI cell -- every TGI stop rule
is covered by composition of the existing cells (kept sequences = the
`hf_stop_string` cell; first-listed multi-stop = the `sglang_stop`
selection rule; no `min_tokens` = the ungated `sglang_stop` path;
EOS-beats-stop = the documented contrast in `STR_OVER_EOS_PRECEDENCE`);
SGLang `stop_regex` gets a USAGE.md note only (outside stop-string
scope).

Verification: 27 new tests in `tests/test_cycle10_sglang.py` (each
SGLang rule vs vLLM/HF behavior, all three new classes, the no-TGI-cell
rationale, hostile-string escaping in the report, determinism); existing
severity/count pins in `tests/test_cycle7_divergence.py` and
`tests/test_cycle7_properties.py` updated from 5 to 8 classes; the
`pairs_verdict` one-liner now reads "HF vs vLLM vs SGLang stop
semantics: …". ruff check + format clean; mypy clean. Docs: `USAGE.md`
§11 (new column, new classes, flags re-verified against live `--help`,
no-TGI-cell rationale, `stop_regex` note), `CHANGELOG.md` [Unreleased].

## 2026-10-15 -- Residual fix: observation-count bound on `divergence_section_data`

What changed and why:

The H1/M1/L1 red-team fix cycle bounded each observation's ids
(`MAX_OUTPUT_IDS`/`MAX_PROMPT_IDS`) and the stop texts
(`MAX_STOP_TEXT_LEN`/`MAX_STOP_TEXTS`), but left one residual: the *count*
of observations `divergence_section_data` processes was unbounded. The
`run` diagnosis path hands it the winner's recorded observations, and each
observation costs one full `diff_backends` emulation (~2-3 s worst case
per stop text at the output cap on the reference stub) -- so a hostile
`results.json` with millions of recorded observations would burn
open-ended hours, even though every individual observation was "valid".

Fix: named constant `MAX_DIVERGENCE_OBSERVATIONS = 512`. Rationale, all
from the lab's own caps: the section consumes only the winner's recorded
observations (search + held-out fixture outputs), and experiments.py
rejects fixture splits above 100 search / 100 held-out fixtures, so a
legitimate run hands over at most 200 observations. 512 is comfortably
above that maximum (2.56x headroom) but far below DoS territory. No
legitimate run is affected; the bound can only engage on inputs no real
experiment can produce.

Contract (deliberate): degrade by omission, never fail the run. Over-count
input processes the first 512 observations in input order and records the
fixed-vocabulary `observations_truncated: true` plus `observations_total`
(the offered count) and `observations_processed` (the bound) in the
section dict; the HTML report renders an explicit "Observation bound
reached" note naming both numbers. No exception is raised (a raise would
be swallowed by cli.py's diagnosis try/except and silently drop the
*whole* section; omission keeps the first-512 verdicts). Per-observation
skip semantics (non-dicts, invalid ids) are unchanged, including inside
the first-512 window. Deterministic: the same first-N prefix truncates
every run. Boundary N is accepted fully; N+1 truncates.

Verification: 8 new tests in `tests/test_cycle11_bounds.py` (stub backend
only, deterministic): named-constant pin, full processing at N,
truncation flag/counts at N+1 with first-N order, hostile-scale input
finishes fast without raising, per-observation skip contract preserved
under truncation, same-input determinism, all-unusable first-N still
degrades to an omitted section, tuple input handled. ruff check + format
clean; mypy clean. Docs: `docs/USAGE.md` §11 "Input bounds" paragraph,
`CHANGELOG.md` [Unreleased] (new `### Fixed` block), this entry.

## Drift check 2026-10-15 (all backends NO CHANGE)

Re-verified vLLM, HF transformers, SGLang, TGI stop/decoding semantics against 2026-10-01..2026-10-15 upstream. All pinned behaviors stand (vLLM #49391 earliest-completion, HF stop-kept, SGLang first-listed + stop-beats-EOS #28802, TGI stale @b4adbf2). No divergence class or emulation cell invalidated.
WATCH: sgl-project/sglang#42466 (open, unmerged, CI failing) proposes earliest-*start* multi-stop selection — would change MULTI_STOP_SELECTION if merged; differs from both current SGLang (first-listed) and vLLM (earliest-completion). Re-check on next drift sweep.

## Continuous dev — Cycle 11: Nemotron 3.5 Lightning + Muse Glimmer 30B + Poolside Laguna 2.1 catalog entries (2026-10-15, VERIFIED)

Catalog 19 -> 22 families: `nemotron35` (NVIDIA Nemotron 3.5 Lightning,
released 2026-08-11), `muse_glimmer` (Meta Muse Glimmer 30B, released
2026-08-10, Apache 2.0), `laguna` (Poolside Laguna 2.1, Jul 2026,
OpenMDW-1.1). 891 -> 915 tests (24 new in tests/test_cycle11_families.py);
ruff check + format clean; mypy clean on catalog.py and the new test file.
Three pre-existing catalog-listing tests (cycle5/6/7) updated to 22
families; the e2e module-catalog set gained all three families; the
cycle-10 thinking-off cross-catalog invariant gained the nemotron35
exception; the `<|im_start|>` audit collision counts went 5 -> 6
(+nemotron35); README "19-family" claims updated to 22 and the test-count
claim to 915 (new regression test pins the naming); CHANGELOG [Unreleased]
extended. Full suite green except pre-existing environmental failures in
this sandbox (torch/hypothesis/transformers not installed: 5 loader
errors, the diff-backends tiny-tokenizer test, and the README-count test
whose discovery is shrunken by those same missing modules) -- all
unrelated to this change.

**Tier promotion (revises the Cycle 10 deferral).** Cycle 10 deferred these
three as TIER-3 (only secondary claims, no primary-source template
evidence). This cycle re-examined them against newly verified evidence and
promotes all three to TIER-2 (official sources + independent byte-level
corroboration):
- laguna: the renderers library's LagunaXS21Renderer/LagunaS21Renderer are
  token-exact parity-verified against `apply_chat_template` on the
  official chat_template.jinja (upstream rev 575f0f28); the PR texts (#99,
  #111) were re-read directly this cycle and corroborate every modeled
  string (hugging tags, thinking gates, `</assistant>` terminator, gen
  prompts, empty-system opt-out).
- nemotron35: the official HF model card + NeMo SFT docs + the community
  osaurus byte-level renderer (PR #2360) agree on the frames and thinking
  tails; the BOS question stays explicitly UNVERIFIED/conflicting at
  official level and is pinned as such (no BOS token asserted).
- muse_glimmer: the official meta-oss-cookbook quickstart + HF transformers
  model docs + HF blog walkthrough agree on the frame protocol; the gen
  prompt is community-refined (the simpler `<|start|>assistant` alternative
  was superseded) and is marked tier-2/community-corroborated.

**Nemotron 3.5 Lightning research**
(https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16):
- Per-turn frames are ChatML (`<|im_start|>{role}\n{content}<|im_end|>\n`)
  -- so the text-only core renders byte-identical to chatml except the
  generation tail (pinned in tests/test_cycle11_families.py, including the
  mutate-set equality modulo the thinking-off branch).
- Thinking ON (default -- enable_thinking defaults true): the generation
  prompt ends at `<|im_start|>assistant\n<think>\n` (the base entry's
  open_suffix). Thinking OFF: the generation prompt prefills an EMPTY think
  block, `<|im_start|>assistant\n<think></think>` -- with NO trailing
  newlines, unlike kolibri/thinking-off's "<think>\n\n</think>\n\n" tail
  (the byte difference is pinned).
- Design decision: the base entry is the thinking-on DEFAULT branch; the
  thinking-off branch rides the mutate path as "nemotron35/thinking-off"
  (candidates()), a generation-prompt branch of the same template, not a
  second catalog entry (kolibri/thinking-off convention).
- CRITICAL: BOS behavior is UNVERIFIED/conflicting at official level. The
  catalog models bos="" as an ABSENCE-OF-EVIDENCE placeholder and does NOT
  assert any BOS token -- a test pins both the modeled value and the
  source-level UNVERIFIED marker in the catalog comment, and asserts no
  BOS-like token appears in the render.
- Sources: the HF model card above,
  https://github.com/nvidia-nemo/nemotron/blob/HEAD/docs/nemotron/lightning35/sft.md,
  https://github.com/osaurus-ai/osaurus/pull/2360

**Muse Glimmer 30B research**
(https://github.com/meta-models/meta-oss-cookbook/blob/HEAD/quickstart/README.md):
- Frame: `<|begin_of_text|><|start|>{role}<|message|>{content}<|eot|>` --
  no newline anywhere. Assistant turns use recipient channels: the
  reasoning trace is `<|start|>assistant to=self<|message|>...` and the
  user-facing reply `<|start|>assistant to=user<|message|>...<|eot|>` --
  the lab's plain-text no-reasoning hypothesis models the assistant turn as
  to=user and does NOT render the to=self trace (same class of omission as
  kimi3's empty think block and glm45/glm53's empty <think></think>: the
  reasoned-content branch is not a hypothesis).
- Generation prompt: `<|start|>assistant to=user<|message|>` --
  COMMUNITY-REFINED (tier 2/community-corroborated); the simpler
  `<|start|>assistant` alternative was superseded (pinned: the open_suffix
  equals the refined form and not the simple form).
- Documented, NOT modeled: system messages render the caller text with
  "\nReasoning strength: high.\n# Valid recipients: \"self\", \"user\"."
  APPENDED after the caller content (single newlines -- not a standalone
  preamble block); the system role is not modeled in Format (same
  convention as mistral_large/gemma4/kolibri). EOS is `<|end_of_text|>` (id
  200001);  is the end-of-MESSAGE marker, NOT a stop token. All three
  are pinned as documented test constants only.
- Closest-family diff (pinned): llama3 shares the `<|begin_of_text|>` BOS,
  but every frame differs.
- Sources: the quickstart README above,
  http://huggingface.co/docs/transformers/main/model_doc/muse_glimmer,
  https://github.com/huggingface/blog/blob/main/muse-glimmer.md

**Poolside Laguna 2.1 research** (https://huggingface.co/poolside/Laguna-XS-2.1):
- Frames ("hugging tags", no inner newlines): `<user>{content}</user>`,
  `<tool_response>{content}</tool_response>`, `<system>{content}</system>`.
- Reasoning gated on enable_thinking -- on: every assistant turn renders
  `<think>{reasoning}</think>` VERBATIM (empty included); off: turns open
  with a bare `</think>` and reasoning is dropped. Gen prompt:
  `<assistant>` + (`<think>` | `</think>`), no trailing newline. Completed
  assistant turns terminate with `</assistant>` (tier-2 via the renderers
  PR text: "the two opener tokens are unsampled scaffold; everything
  through the sampled </assistant> is the model's emission").
- Design decision: the base entry is XS-2.1 DEFAULTS (enable_thinking
  FALSE -- documented choice): gen prompt `<assistant></think>`, history
  assistant turns `<assistant></think>{content}</assistant>`. S-2.1
  (enable_thinking defaults TRUE) is the "laguna/thinking-on" mutate-path
  variant -- the kolibri/thinking-off convention: a generation-prompt
  branch of the same template, not a second catalog entry. Its history
  assistant turns render an EMPTY think block verbatim
  (`<assistant><think></think>{content}</assistant>`) -- the lab's
  deterministic no-reasoning hypothesis (the reasoned-content branch is not
  a hypothesis here, kimi3 convention).
- EOS is `〈|EOS|〉` -- DECODE-ONLY, never rendered in the prompt
  (documented, not modeled; pinned as a constant that must not leak into
  any render). BOS: no BOS token appears in any tier-2 render; modeled as
  empty with the absence-of-evidence caveat (contrast nemotron35, whose BOS
  is explicitly conflicting).
- System role: not modeled (lab convention). The upstream template ships a
  default Poolside system prompt, but an empty caller system message
  suppresses the `<system>` block entirely -- and the lab never sends
  system messages, so the no-system render modeled here is byte-identical
  either way; the default-prompt wording is deliberately NOT asserted.
- Sources: the HF model card above,
  https://github.com/PrimeIntellect-ai/renderers/pull/99,
  https://github.com/PrimeIntellect-ai/renderers/pull/111

**Watched, NOT shipped:** Mistral Large 4 and Reflection Beam are not
released -- deliberately not catalog entries. Tracked as watched; a future
cycle may add them once primary-source template evidence exists (the
tier-3 bar from Cycle 10 still applies: tier-3 claims never become catalog
entries).

**Strings deliberately not pinned:** nemotron35's BOS token (UNVERIFIED /
conflicting at official level -- see the catalog comment); the wording of
Laguna's default Poolside system prompt (suppressed in the lab's no-system
render either way); the muse_glimmer `to=self` reasoning-trace content and
the system-appendage beyond the documented suffix (unmodelable roles /
generation stops, documented only); the laguna reasoned-content branch
(non-deterministic, kimi3 convention).

## Continuous dev — Cycle 12: static template↔tokenizer portability matrix (2026-10-15, VERIFIED)

Builds the research recommendation from
docs/RESEARCH_PORTABILITY_2026-10-15.md (kept as-is): `doctor
--portability-matrix SECOND_CHECKPOINT` checks every catalog candidate's
special-token inventory (`fixtures._extract_special_tokens`) against the
second checkpoint's tokenizer -- registered in added/special tokens,
encodes as exactly one token id, encoded id == registered id -- via the new
`core.template_tokenizer_compatibility`, a deliberate generalization of
`core.check_boundary_portability` (same probe-must-not-fail-caller,
fixed-vocabulary, exact-token-ID discipline). Tokenizer only, never
weights: one tokenizer load + O(candidates × tokens) encodes, reusing
doctor's `_build_primary_tokenizer` path, the 128 MB `tokenizer.json` cap,
and the `_loader_failure_code` sanitization boundary (hostile checkpoint
bytes never reach the terminal).

**The asymmetry is the design and is stated loudly everywhere** (report
section, CLI table, docs/USAGE.md §2): `incompatible` ⇒ the format cannot
be that checkpoint's native format ⇒ re-running the experiment there is
scientifically pointless, answered at ~zero compute. `compatible` ⇒ the
markers are merely encodable ⇒ NO prediction about experimental success
(weights decide). No portability score is emitted anywhere; a recursive
key-scan test pins that no score/prediction/transfer key can appear in the
section. BOS-less formats (8 catalog families, bos="") skip the BOS
requirement by design (`bos_skipped` per row -- the inventory simply holds
no BOS literal); empty inventories and encode failures report `unknown`,
never `compatible`. The matrix is triage, never a gate: it cannot fail the
doctor verdict, and an unusable second checkpoint yields `unavailable`
with a fixed-vocabulary detail.

Verification: 38 new tests in tests/test_cycle12_portability.py --
incompatible pinned for all three failure modes (unregistered / fragments /
id_mismatch), unknown for empty inventory and degraded encodes, first-
failure-in-inventory-order, degrade-beats-incompatible, BOS-less skip both
ways, hostile dirs (missing / oversized-sparse / malformed-JSON with
hostile marker bytes / non-dict doc / unloadable tokenizer) all collapse
to fixed codes, determinism (two runs byte-identical), CLI flag wiring
(stdout stays parseable JSON, table + ASYMMETRY WARNING on stderr), and a
no-transfer-claim scan over the whole section. 915 -> 953 tests (README
count updated; the pinning test passes); ruff check + format clean; mypy
clean on all 14 source files.

**Deliberate exclusions:** no primary-checkpoint matrix row (the research's
"natural second row" -- deferred; the experiment's own checkpoint is
covered by template_conformance, and the flag's question is strictly
"can I rule out checkpoint B without running it?"); no behavioral
template-inference feature (rejected by the research as speculative
science); no AutoTokenizer-vs-minimal-loader validation on real
checkpoints yet -- the encode path is the same Rust tokenizer the file
defines, and behavior is pinned against `tokenizers`-built synthetic
checkpoints; operators should still verify encode behavior against the
checkpoint's intended tokenizer before trusting results (USAGE.md §3
already carries that caution for the primary path).

## Continuous dev — Cycle 13: Ling-3.1-flash catalog entry (2026-10-15, VERIFIED)

Catalog 22 -> 23 families: `ling` (inclusionAI/Ling-3.1-flash, released
2026-10-15, MIT; https://huggingface.co/inclusionAI/Ling-3.1-flash).
Evidence TIER-1: the official `chat_template.jinja` ("Bailing V3"),
read and derived line-by-line by hand (preserved at
docs/evidence/ling31-chat-template.jinja); the tokenizer_config.json
`bos_token` field was verified live against the HF repo (read 2026-10-15):
`<|startoftext|>`.

What the template says (plain-text, no-tools core): an always-emitted
system block (with system message X: `<role>SYSTEM</role>X\ndetailed
thinking on<|role_end|>`; without: `<role>SYSTEM</role>detailed thinking
on<|role_end|>`); user `<role>HUMAN</role>{content}<|role_end|>`;
assistant history `<role>ASSISTANT</role>\n<think></think>{content}
<|role_end|>`; generation prompt `<role>ASSISTANT</role>\n<think>`
(thinking defaults ON, jinja line 12). The catalog models the default
thinking-on render; thinking-off prefills an empty think block and rides
the `--mutate` path as `ling/thinking-off` (kolibri/thinking-off
convention). The system block is documented, not modeled (lab
convention); tool-call branches, reasoning-content branches, and the
multi-step tool scan are out of scope for the plain-text core.

**EOS correction (verified, not assumed):** the official eos_token field
is `<|role_end|>` -- NOT `<|endoftext|>`, which is the pad_token and a
special added token. Documented in the catalog comment; Format asserts no
EOS string.

**BOS caveat (documented):** the jinja template itself emits no BOS
literal and add_bos_token is false; the entry takes the config's
bos_token (lfm25 convention), so a raw apply_chat_template call would
not emit it -- noted in the catalog comment.

954 -> 964 tests (10 new in tests/test_cycle13_families.py: byte-exact
render pins, thinking-off mutation pin, dedup-key no-collision pin,
diff-vs-all-other-22-families pin, README naming/count pin); the
cycle-10/cycle-11 thinking-branch invariants and the cycle-11 README
count pin updated for the new family; README "22-family" claims updated
to 23 and the test-count claim to 964; CHANGELOG [Unreleased] extended.

## Continuous dev — Cycle 13: `mpl diff-reports` iterate-and-compare tool (2026-10-15)

Research recommendation, no external research needed (pure local tooling):
the core operator workflow is iterate-and-compare -- tweak a template or
fixture, re-run, see what changed -- and eyeballing two HTML reports is
imprecise. `mpl diff-reports OLD_RESULTS NEW_RESULTS` (new module
`model_protocol_lab/diffreports.py`, wired into the CLI at the two ANCHOR
sites) compares two `results.json` files deterministically: same inputs →
byte-identical output, no model, no network, exact comparisons only (no
fuzzy matching).

Compared axes: decision flip (old → new), winner change, per-candidate
pass/fail verdict flips on search and held-out (held-out verdicts exist
only for each run's winner; others report `absent`), per-fixture outcome
diffs per candidate (pass requires exact answer + boundary token match,
the `run` criterion; changed-only by default, `--all` lists everything),
and a config-drift warning section -- the classic operator mistake of
comparing runs that were never comparable. Drift flags different
candidate families (base names; `results.json` does not store the
`--family` flag, so this is the exact family signal available), candidate
sets, per-candidate boundary token IDs, `--max-new-tokens`, backend
fingerprints (model identity), and fixture sets (when evidence exists).
Per-fixture comparison needs `--include-evidence` on both runs and
degrades loudly otherwise; exit 0 on success even when differences are
found (differences are the output, not a failure).

Security posture mirrors the repo's bounded-reader discipline: 32 MB cap
(`results_too_large`), regular-file-only reads via the
`os.open(O_NONBLOCK)` + `fstat` pattern from `datasets.load_dataset`,
duplicate-field and non-finite rejection with fixed codes, shape
validation of only the compared axes (everything else degrades to
`unknown` rather than failing), and control-character stripping of
operator-visible identifiers before they reach stderr. All failures map
to fixed `ValueError` codes -- no tracebacks, no file content in error
text.

Verification: 30 new tests in tests/test_cycle13_diffreports.py --
decision/winner flips, verdict flips (search, held-out, absent, malformed
summary → `unknown`), per-fixture diffs (changed-only default, `--all`,
deterministic (stage, candidate, fixture) ordering), config drift for all
six fields with fixed field order, identical-run → "no differences",
missing-evidence degradation, hostile inputs (malformed JSON, duplicate
keys, NaN, non-UTF-8, oversized, FIFO, missing paths, deep nesting,
directory resolution) all collapsing to fixed codes with no tracebacks,
stderr sanitization of hostile identifiers, byte-identical determinism,
and CLI wiring (JSON on stdout, summary on stderr, exit 0 with
differences). ruff check + format clean; mypy clean on all 15 source
files.

**Deliberate exclusions:** no fuzzy fixture matching (exact fixture-ID
comparison only -- cross-fixture alignment is a research question, not a
diff feature); no HTML report comparison (`results.json` is the source of
truth); no verdict inference from missing summaries (degrades to
`unknown`, shown as such); the `--family` flag is not recovered (not
stored in `results.json` -- candidate base names are the exact available
signal, documented as such).

## 2026-10-15 -- Red-team fix round: diffreports H1/M1-M3/L1 + finding 6 documentation

What changed and why:

A red-team pass over the cycle-13 `mpl diff-reports` code found six
findings; findings 1-5 were fixed (each pinned by a regression test in
`tests/test_cycle13_diffreports.py`), finding 6 documented in a code
comment with no behavior change.

- **H1 (CI-blocking):** the README test-count pin test shells out to
  `python -m pytest --collect-only`, but `pytest` was not in the
  `pyproject.toml` dev extras or installed by `.github/workflows/ci.yml`
  -- the pin test would fail deterministically in CI. Fix: added
  `pytest>=8.0` to the `dev` extras (pytest is the suite's actual
  runner; it belongs there). Verified from the workflow file that both
  jobs installing the suite (`static` installs `.[dev]`, `test` installs
  `.[dev,hf]`) pull the dev extra, so pytest is present wherever the
  suite runs. No skip guard added -- the conditional did not trigger.
  Pin test verified passing in the repo venv (README count updated
  994 -> 1000 for the 6 new tests).

- **M1:** `render_summary` interpolated `decision["old"]`/`["new"]`
  (hostile results.json strings) and the source path labels raw into
  stderr -- ANSI escape / OSC sequence injection. Fix: both now pass
  through the existing `_display_id` sanitizer, like every other
  operator-visible identifier.

- **M2:** `_config_snapshot`'s `name not in names` list-scan dedup was
  O(n^2) and hung ~1.5h on 800k candidate names. Fix: set-based dedup
  preserving first-occurrence order. Regression test holds 50k
  distinct names to <10s (measured well under 1s).

- **M3:** `_search_verdicts` dropped per-candidate `planned` counts, so
  planned 2 vs 50 (no evidence) reported `identical: True` -- a "pass"
  verdict means "all *planned* fixtures passed", making the comparison
  a lie of omission. Fix: `_search_verdicts` now also returns the
  per-candidate planned counts (first-entry wins, `None` for
  malformed/missing summaries), and the diff emits `planned:<candidate>`
  config-drift entries (fixed order, after the five `_drift` fields)
  which render in the stderr drift section.

- **L1:** `identical: True` plus "identical on every compared axis" was
  reported when evidence existed on only one side -- a whole axis was
  never compared. Fix: evidence-presence asymmetry now sinks
  `identical` (with a new machine-readable
  `summary["evidence_asymmetric"]` flag), and the identical-run stderr
  line names the axes actually compared (decision, winner, candidate
  verdicts, configuration, plus per-fixture evidence only when both
  sides carry it).

- **Finding 6 (theoretical):** duplicate fixture IDs inside one
  candidate's observations are last-wins, while duplicate candidate
  names in "search" are first-wins. Documented in a comment at
  `_fixture_outcomes`; behavior deliberately unchanged -- `mpl run`
  never emits either duplicate, so reconciling the conventions would
  change only hostile-input behavior.

Verification: 6 new tests in `tests/test_cycle13_diffreports.py`
(hostile decision/label sanitization with ANSI + OSC payloads,
config-snapshot scaling, planned-count drift, evidence-asymmetry
sinks identical, identical summary names compared axes) plus one
assertion added to the existing missing-evidence test. Full suite:
964 unittest-discovered tests OK (README pin: 1000 pytest-collected).
ruff check + format clean; mypy clean on all 15 source files.
Docs: `CHANGELOG.md` [Unreleased] `### Fixed`, this entry.

## 2026-10-15 -- Red-team fix round (follow-up): diffreports drift-field sanitization + README pin test debuggability

A follow-up red-team pass over the cycle-13 `mpl diff-reports` code produced
two small genuine low-severity findings. Both are fixed and regression-pinned.

- **Finding 1 (Low):** `render_summary` interpolated the config-drift
  `field` string raw into stderr. The field is built as
  `f"planned:{name}"` with the raw candidate name (`diff_reports`, ~line
  434), and was rendered via `f"  - {entry['field']}: ..."` (~line 504),
  so a hostile candidate name carrying ANSI/OSC escapes reached stderr
  raw (confirmed live). Fix: the field now passes through the existing
  `_display_id` sanitizer at the render loop. Sanitizing at render --
  rather than at construction -- was chosen deliberately because it
  covers every drift field type: `planned:<name>` (hostile), the fixed
  config fields (`families`, `candidates`, `max_new_tokens`,
  `boundaries`, `backend_fingerprint`), and `fixtures:search/held_out`
  (fixed stage suffixes). The `_compact` JSON rendering of drift values
  was already ASCII-safe (`ensure_ascii=True`), and the JSON stdout
  payload keeps the raw field for machine consumers -- sanitization
  applies only to the human-readable stderr summary. Pinned by a new
  test in `tests/test_cycle13_diffreports.py`: hostile candidate name in
  a planned-drift entry reaches the payload (`planned:<evil>` present)
  but the rendered summary contains no raw `\x1b`/`\x07` bytes, while
  the readable text ("PWNED") and the `planned:` prefix survive.

- **Finding 5 (Low):** the README test-count pin test in
  `tests/test_cycle4.py` never inspected `proc.returncode`/`proc.stderr`,
  so a pytest collection error surfaced as the misleading "could not
  parse pytest collection count", hiding the real failure. Fix: the test
  now asserts `proc.returncode == 0` first, and both failure messages
  include the last ~2000 characters of stderr/stdout, so CI collection
  failures are debuggable from the test output alone.

Verification: new regression test passes; `tests/test_cycle13_diffreports.py`
37 passed; the hardened pin test correctly surfaced a real collection
error (missing `torch` in a throwaway venv) instead of the misleading
parse message, then passed in the real environment after the README
count was bumped 1000 -> 1001 for the new test. Full suite: 1001
passed + 3142 subtests, ruff check + format clean, mypy clean on all
source files.
Docs: `CHANGELOG.md` [Unreleased] `### Fixed`, this entry.

## 2026-10-15 (cycle 13 upstream drift)

Five-item upstream-drift sweep, all primary sources re-read this round.
No code changed; recommendations only.

**1. SGLang PR #42466 (multi-stop selection) — STILL OPEN, NOT MERGED.**
Re-read https://github.com/sgl-project/sglang/pull/42466 (page text,
2026-10-15): title "[Fix] Stop at the earliest matching stop_str when
several match" by untrall; CI still failing on all three listed runs
(Latest PR Test Base / Extra / AMD ROCm 10 all red). Independently
confirmed absent from main via GitHub code search of sgl-project/sglang:
the `# TODO(lmzheng): handle the case where multiple stop strs are hit`
comment is still present in
`python/sglang/srt/managers/detokenizer_manager.py`, and the PR's new
test file `test/srt/test_multi_stop_str.py` does not exist in the repo.
Proposed behavior (unchanged from the earlier drift note, quoted from
the PR): "Collect every matched stop string in `_check_str_based_finish`
(`schedule_batch.py`) and record the one with the earliest occurrence in
the decoded text (ties keep list order; single-match behavior unchanged)"
— i.e. within one decode step, the *leftmost* occurrence wins instead of
list order. WATCH: if it merges, the repo's SGLang multi-stop tie-break
(currently smallest firing step, ties broken by list order — see
`emulate_sglang_stop` docstring in `model_protocol_lab/backends.py`)
must become: same firing step -> earliest occurrence position in decoded
text, ties keep list order. No repo change until merged.

**2. Mistral Large 4 / Reflection Beam weights — NOT SHIPPED.**
- Mistral Large 4: API preview live since 2026-10-06; the HuggingFace
  pages https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B and
  https://huggingface.co/mistralai/Mistral-Large-4.0-1T05-A52B both show
  "Upcoming release" / "Expected release: October 31, 2026. Current ETA".
  No downloadable weights.
- Reflection Beam: announced 2026-10-05 (501B/23B-active MoE, 1M
  context); weights promised "later in October 2026" under Apache 2.0
  after final red-teaming, early access via waitlist only
  (https://www.cryptopolitan.com/reflection-ai-beam-early-access-open-weights/,
   https://aistockwire.com/blog/reflection-ai-beam-open-weight-model-nvidia-nvda-kimi-glm-benchmarks-october-2026).
  No downloadable weights.
Catalog impact: NONE. The `mistral` (Mistral-7B-Instruct-v0.3) and
`mistral_large` (Mistral-Large-Instruct-2411) entries remain accurate for
all released Mistral models. Recommendation for the next cycle: when the
weights land, add NEW family entries (do not edit the existing ones) —
Large 4 is natively multimodal with a new template/tokenizer, Beam is a
new 501B lab/family.

**3. New open-weight releases since 2026-10-01.**
- DeepSeek-V4.1-Flash: weights went live 2026-10-06 on Alibaba's
  ModelScope hub and the Qwen platform (552B, MIT)
  (https://abhs.in/blog/deepseek-v4-1-flash-552b-moe-1m-context-mit-pricing-benchmarks-2026).
  ALREADY IN CATALOG as `deepseek41` (sourced from
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash). No gap.
- Decision 3.0 (vllm-sr/d3, d3-flash, d3-mini, d3-nano, d3-lite,
  d3-edge): uploaded 2026-10-10, Apache-2.0, safetensors on HuggingFace
  (verified via the HF model cards, e.g.
  https://huggingface.co/vllm-sr/d3). BUT they are non-generative
  decision models — "returns a probability for every answer, without
  generating text" — so they have no chat template; out of scope for the
  chat-template catalog. Watchlist only.
- AutoTrust AI JEV-27B-VL / GEV-26B-Decide (announced 2026-10-08,
  open-weight decision/vision models): same verdict — notable, non-chat,
  no catalog entry.
  (https://www.morningstar.com/news/pr-newswire/20261008cn66859/...)
- EmbeddingGemma 2 (2026-10-06, Apache 2.0): embedding model, no chat
  template, out of scope.
- Out-of-window backlog for a future research cycle (all before
  2026-10-01, all missing from the catalog): IFM/MBZUAI **K2-Horizon**
  fleet (6 models 0.9B–375B, Apache 2.0, released ~2026-09-03, on HF with
  day-zero vLLM/SGLang support — genuine missing chat families);
  Tencent **Hy4 preview** (2026-08-28, Apache 2.0, 770B/49B); MiniMax-M3
  (2026-06, MiniMax Community License); Nanbeige4.2-3B (release date
  uncertain, likely ~Sept 2026).

**4. vLLM stop-string/min_tokens semantics (releases since 2026-09-01:
v0.28.0, v0.29.0, v0.30.0) — NO CHANGE to modeled semantics.**
Read the official release notes at
https://github.com/vllm-project/vllm/releases/tag/v0.28.0,
/v0.29.0, and /v0.30.0. Stop-adjacent items found:
- v0.28.0 #49227/#50595: request stop tokens masked in xgrammar until
  the grammar terminates (structured-output path only, not the core
  stop-string holdback the lab models); #51447: new `VLLM_MAX_STOP_STRINGS`
  cap (default 4) on stop strings per request — a request-level limit, not
  a selection-semantics change; #51997/#51556: Anthropic/Cohere frontend
  stop-sequence bounding/reporting.
- v0.29.0 #54218 (read the full PR): "terminal grammars stop under
  `min_tokens`" — fixes the min_tokens+structured-output terminal-state
  hang, does NOT change stop-string/EOS min_tokens holdback; #53750:
  applies the stop-string count limit to Cohere requests (frontend
  validation).
- v0.30.0 #55325: Anthropic `stop_sequence` reported in `message_delta`
  (streaming frontend); #51450: invalid structured-output requests no
  longer stop the engine.
None alters vLLM's core semantics pinned by the repo (all stops gated
behind `min_tokens`; multi-stop earliest-completion selection per
vLLM #49391). `emulate_vllm_stop` and the diffbackends cells stand.

**5. transformers stop_string semantics since 5.19.0 — NO CHANGE.**
The releases page https://github.com/huggingface/transformers/releases
tops out at v5.19.0 (no v5.19.1, no v5.20): the repo's pinned knowledge
is the current upstream. No `generate()` `stop_string` handling change
to report. (v5.19.0 notes read this round: MoE router-logits breaking
change, CB cache/attention work, no generation stopping-criteria items.)

Net: no repo change warranted on any item this round. Keep the SGLang
#42466 WATCH and add Mistral Large 4 / Reflection Beam weights to the
watch list for the next drift sweep.

## 2026-10-15 (cycle 14 K2-Horizon family research)

Catalog 23 -> 24 families: `k2horizon` (IFM K2-Horizon fleet, Institute of
Foundation Models launched by MBZUAI, released 2026-09-03, Apache 2.0).
Evidence TIER-1: the official `chat_template.jinja`, read and derived by
hand (preserved at docs/evidence/k2horizon-chat-template.jinja); the
tokenizer_config.json `bos_token`/`eos_token` fields verified live
against the HF repos (read 2026-10-15).

**Org + fleet verification (primary: HF API + model cards).** The HF org
is `IFM` (not "IFM/MBZUAI" as the backlog note guessed). All six fleet
repos verified present, ungated, with download counts, via the HF API
author listing: IFM/K2-Horizon-375B-A23B, IFM/K2-Horizon-MoVA-36B-A4B
(note: "MoVA" is in the repo name), IFM/K2-Horizon-32B, IFM/K2-Horizon-7B,
IFM/K2-Horizon-3.7B, IFM/K2-Horizon-0.9B. Safetensors weights publicly
downloadable: the 375B-A23B repo file listing shows all 61
model-*-of-00061.safetensors shards. License Apache 2.0 and 2026-09-03
release confirmed on the official model card
(https://huggingface.co/IFM/K2-Horizon-375B-A23B).

**Primary sources read this round (URL + what was read):**
- https://huggingface.co/IFM/K2-Horizon-375B-A23B -- model card: fleet
  composition, Apache-2.0 license, vLLM/SGLang day-zero recipes,
  reasoning_effort high/medium/low via chat_template_kwargs, k2_horizon
  reasoning/tool-call parsers.
- https://huggingface.co/IFM/K2-Horizon-375B-A23B/raw/main/tokenizer_config.json
  -- bos_token `<|ifm|begin_of_text|>`, eos_token `<|ifm|endoftext|>`,
  PreTrainedTokenizerFast.
- https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/raw/main/tokenizer_config.json
  -- byte-identical bos/eos fields (fleet shares the tokenizer config).
- https://huggingface.co/IFM/K2-Horizon-375B-A23B/raw/main/chat_template.jinja
  -- the full official template (1014 lines), downloaded and preserved at
  docs/evidence/k2horizon-chat-template.jinja. md5-verified byte-identical
  across the 375B-A23B, 7B and MoVA-36B-A4B repos
  (150b0fd46f7babfc6bfd8b769ad0475c) -- one template for the fleet.
- https://huggingface.co/IFM/K2-Horizon-375B-A23B/raw/main/generation_config.json
  -- eos_token_id [1, 250019] (250019 = `<|ifm|im_end|>` per the 7B
  added_tokens_decoder).
- https://huggingface.co/IFM/K2-Horizon-375B-A23B/raw/main/modeling_k2_horizon.py
  -- confirms no chat-template handling in modeling code (the template is
  the whole story).

**Template decisions (plain-text, no-tools core; line numbers in the
preserved 1014-line file):** BOS `{{- bos_token }}` (line 1) =
`<|ifm|begin_of_text|>`; user `<|ifm|im_start|>user\n{content}<|ifm|im_end|>`
(line 959, no trailing newline); assistant history
`<|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>{content}<|ifm|im_end|>`
(lines 983-997) -- the template REQUIRES a thinking field on assistant
messages (think/think_fast/think_faster/reasoning_content/reasoning as a
string, lines 962-976; a content-only assistant message raises), so the
lab's deterministic no-reasoning hypothesis is an EMPTY think field
(line 989), kimi3/laguna/ling convention; generation prompt
`<|ifm|im_start|>assistant\n<ifm|think>\n` for the default
reasoning_effort='high' (lines 1005-1006), `<ifm|think_fast>` for medium
(line 1008), `<ifm|think_faster>` for low (line 1010) -- the medium/low
branches ride the `--mutate` path as `k2horizon/effort-medium` /
`k2horizon/effort-low` (kolibri/thinking-off convention); system block
`<|ifm|im_start|>system\n{content}<|ifm|im_end|>` (line 948) documented,
not modeled. The catalog entry's render was verified byte-identical
against the official template rendered with a no-op Jinja extension for
the custom tag (1-turn, 2-turn, effort variants, system block).

**{% generation %} caveat (verified, not assumed).** The official
template wraps the assistant's inner content in {% generation %} /
{% endgeneration %} block tags (lines 984, 998), which are NOT valid
stock Jinja2 -- verified TemplateSyntaxError ("unknown tag
'generation'") with jinja2 3.1.6 AND the exact
ImmutableSandboxedEnvironment(trim_blocks/lstrip_blocks) transformers
5.19.0 uses for apply_chat_template. They emit no bytes; serving stacks
must supply the extension. The \n after "assistant" in the history frame
is the source line break inside the {% generation %} block under
standard Jinja block-extension semantics -- the only render consistent
with the template's own explicit generation prompt
(`<|ifm|im_start|>assistant\n<ifm|think>\n`, line 1006). Community
renderers that strip the tags textually emit the frame WITHOUT that \n
-- a known, documented ecosystem divergence. The template was last
touched 2026-10-14 ("Support OpenAI-style text content lists (#8)"),
so the md5 above is the revision pin.

**Fetch warning for future cycles.** browser.open's text extraction
MANGLES this template (it dropped `</...>` closing-tag fragments and
whole think-tag literals, showing 1830 phantom lines vs the real 1014):
for chat_template.jinja files, always curl the raw bytes and derive from
the file, never from the browser text view.

**Triage of the other backlog models (all pinnable from primary sources
today; deferred to a later cycle -- this cycle's test file is
k2horizon-scoped):**
- Tencent Hy4 preview (2026-08-28, Apache 2.0): HF org `tencent`, repo
  tencent/Hy4-preview; chat_template.jinja publicly readable (309 lines,
  compiles under stock Jinja). Plain core: frames
  `<｜hy_start:opensource｜>{role}<｜hy_middle:opensource｜>{content}<｜hy_end:opensource｜>`
  (｜ = U+FF5C fullwidth bar, DeepSeek-style); history assistant
  `<｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜><think:opensource></think:opensource>{content}<｜hy_end:opensource｜>`
  (empty think for plain messages); generation prompt
  `...<think:opensource>` for the default reasoning_effort='high',
  `...<think:opensource></think:opensource>` for `no_think`; a system
  block is ALWAYS synthesized when no leading system message
  (`<｜hy_start:opensource｜>system<｜hy_middle:opensource｜><｜reasoning_mode:opensource｜>reasoning_effort:{effort}<｜hy_end:opensource｜>`)
  -- documented-not-modeled, kolibri convention. Config bos_token
  `<｜hy_start:opensource｜>`, eos `<｜hy_end:opensource｜>`; the template
  emits no BOS literal. Template + tokenizer_config.json both read
  2026-10-15. READY TO ADD.
- MiniMax-M3 (2026-06, MiniMax Community License): HF org `MiniMaxAI`,
  repo MiniMaxAI/MiniMax-M3; chat_template.jinja publicly readable (247
  lines). Plain core: template-local tokens bod `]~!b[`, bos-var `]~b]`,
  eos-var `[e~[`; ALWAYS-emitted system super-prompt
  (`]~!b[]~b]system\n` + default identity + `<thinking_instructions>`
  block) and developer super-prompt (`]~b]developer\nYou are a helpful
  assistant.`) -- both documented-not-modeled; user `]~b]user\n{content}
  [e~[\n`; history assistant `]~b]ai\n</mm:think>{content}[e~[\n`
  (think_end prefix when no reasoning); generation prompt `]~b]ai\n`
  (thinking_mode undefined/adaptive), `]~b]ai\n</mm:think>` (disabled),
  `]~b]ai\n<mm:think>` (enabled). Template read 2026-10-15. READY TO
  ADD.
- Nanbeige4.2-3B (Nanbeige LLM Lab / Boss Zhipin; paper 2026-07-27):
  HF org `Nanbeige`, repo Nanbeige/Nanbeige4.2-3B; no standalone
  chat_template.jinja (404) but tokenizer_config.json embeds a full
  `chat_template` (primary source, read 2026-10-15). Plain core:
  ChatML-like with mandatory think blocks -- ALWAYS-emitted default
  system prompt `<|im_start|>system\n` + Chinese Nanbeige identity line +
  `<|im_end|>\n` (documented-not-modeled); user
  `<|im_start|>user\n{content}<|im_end|>\n`; history assistant
  `<|im_start|>assistant\n<think>\n\n</think>\n\n{content}<|im_end|>\n`;
  generation prompt `<|im_start|>assistant\n<think>\n` (default),
  `<|im_start|>assistant\n<think>\n\n</think>\n\n` with
  enable_thinking=false (mutate-path branch). Config bos `<|im_start|>`,
  eos `<|im_end|>`; template emits no BOS literal. READY TO ADD.

**Built this cycle.** `k2horizon` catalog entry + `k2horizon/effort-medium`
+ `k2horizon/effort-low` mutate variants in model_protocol_lab/catalog.py;
tests/test_cycle14_k2horizon_families.py (12 tests: byte-exact 1-turn and
2-turn renders, character-exact marker pins, system-block/EOS
documentation pins, effort-branch mutation + dedup pins,
diff-vs-all-23-others pin, README naming pin); docs/evidence/
k2horizon-chat-template.jinja preserved. ruff check + format clean, mypy
clean. Full suite: 1109 passed, 10 failed, 1 skipped, 5994 subtests
passed. The 10 failures are all count/README pins for the coordinator:
test_cycle10/11 README count pins (23 vs 24), test_cycle13 ling diff pin
(22 vs 23 others) + README count pin (23 vs 24), test_cycle4 README
test-count pin (1108 vs 1120 discovered), test_cycle5/6/7 twenty-two
pins (pre-existing -- catalog was already at 23 before this cycle),
test_e2e catalog-set pin (missing "k2horizon"), and the new
ReadmeK2HorizonTests pin (README lacks "K2-Horizon"). No README/
CHANGELOG/VALIDATION edits made (coordinator handles).

## 2026-10-15 (cycle 15 Hy4/MiniMax-M3/Nanbeige family research)

Catalog 24 -> 27 families: `hy4` (Tencent Hy4 preview), `minimax3`
(MiniMax-M3), `nanbeige42` (Nanbeige 4.2-3B). All three were triaged as
pinnable in the cycle-14 research; this round re-verified every template
fact against the primary source (raw HF URLs, fetched with curl per the
cycle-14 fetch warning -- browser.open mangles chat_template.jinja text;
small text files only, no weights downloaded). All three repos are
ungated (HTTP 200 on the raw files). All three templates VERIFIED
pinnable -- none skipped.

**Primary sources read this round (URL + what was read):**
- https://huggingface.co/tencent/Hy4-preview/raw/main/chat_template.jinja
  -- the official template, 309 lines, md5
  a0b7e28386487d804009340155e7be2b (preserved byte-identical at
  docs/evidence/hy4-chat-template.jinja). Compiles under stock Jinja2:
  `raise_exception` (lines 36/38) is render-time only. Fullwidth bar
  ｜ = U+FF5C verified at the byte level (0x3c 0xff5c 0x68 ...).
- https://huggingface.co/tencent/Hy4-preview/raw/main/tokenizer_config.json
  -- bos_token `<｜hy_start:opensource｜>`, eos_token
  `<｜hy_end:opensource｜>`, pad_token `<｜hy_pad:opensource｜>`.
- https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/main/chat_template.jinja
  -- the official template, 247 lines, md5
  455403f9cba16ea73388aeb8898da752 (preserved byte-identical at
  docs/evidence/minimax-chat-template.jinja). `raise_exception` (line
  203) is render-time only.
- https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/main/tokenizer_config.json
  -- bos_token `]~b]`, eos_token `[e~[`, pad_token `]!p~[`,
  unk_token `[e~[`, PreTrainedTokenizerFast.
- https://huggingface.co/Nanbeige/Nanbeige4.2-3B/raw/main/tokenizer_config.json
  -- 103 lines; no standalone chat_template.jinja (verified 404). The
  embedded `chat_template` (7961 chars) is the primary source, preserved
  at docs/evidence/nanbeige-chat-template.jinja (195 lines, exact
  extraction of the JSON string). Config: bos_token `<|im_start|>`,
  eos_token `<|im_end|>`, pad_token `<unk>`, tokenizer_class
  LlamaTokenizer.
- https://huggingface.co/Nanbeige/Nanbeige4.2-3B/raw/main/chat_template.jinja
  -- fetched only to confirm the 404 (no standalone template).

**Template decisions (plain-text, no-tools core; line numbers in the
preserved files):**

hy4 (token suffix ':opensource'): user
`<｜hy_start:opensource｜>user<｜hy_middle:opensource｜>{content}<｜hy_end:opensource｜>`
(line 227); history assistant with EMPTY think tags
`<｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜><think:opensource></think:opensource>{content}<｜hy_end:opensource｜>`
(line 243, the lab's deterministic no-reasoning hypothesis; the
reasoned-content branch, line 241, is not a hypothesis); generation
prompt `...<think:opensource>` for the default reasoning_effort='high'
(line 306; default set line 33) and `...<think:opensource></think:opensource>`
for 'no_think' (line 304 -- rides the mutate path as `hy4/no-think`).
The template ALWAYS synthesizes a system block when messages[0] is not
system (lines 196-204:
`<｜hy_start:opensource｜>system<｜hy_middle:opensource｜><｜reasoning_mode:opensource｜>reasoning_effort:{effort}<｜hy_end:opensource｜>`)
-- documented-not-modeled (kolibri convention). BOS: the template emits
no BOS literal, so the entry's bos is "" (phi4/qwen3 convention); the
config bos_token is deployment-dependent and not modeled. Out of scope:
the tool prompt/render macros (lines 74-99), tool_call/tool_response
blocks, raw_last_assistant prefill, fallback_strategy retry.

minimax3 (template-local tokens bod `]~!b[`, bos `]~b[`, eos `[e~[`;
think `<mm:think>...</mm:think>`, lines 2-11): user `]~b]user\n{content}[e~[\n`
(line 228); history assistant with the think_end-prefix branch
`]~b]ai\n</mm:think>{content}[e~[\n` (lines 167-168, the lab's
deterministic no-reasoning hypothesis; the reasoned-content branch,
line 164, is not a hypothesis); generation prompt `]~b]ai\n` with NO
prefix for thinking_mode undefined/adaptive (the default, lines
234-241), `]~b]ai\n</mm:think>` for 'disabled' (line 238),
`]~b]ai\n<mm:think>` for 'enabled' (line 242) -- the disabled/enabled
branches ride the mutate path as `minimax3/thinking-disabled` /
`minimax3/thinking-enabled`. The template ALWAYS emits a system
super-prompt (`]~!b[]~b]system\n` + default identity + the
`<thinking_instructions>` block + `[e~[\n`, lines 113-117) and a
developer super-prompt (`]~b]developer\nYou are a helpful assistant.
[e~[\n`, lines 120-121) -- both documented-not-modeled. BOS: the
template emits no BOS literal (the `]~b[` frame prefixes are message
prefixes, not a BOS; the bod token prefixes only the system
super-prompt), so the entry's bos is "" (phi4/qwen3 convention). Out of
scope: the XML tool-call renderer, tool-role turns, the thinking-mode
instruction text.

nanbeige42 (ChatML-like with mandatory think blocks): user
`<|im_start|>user\n{content}<|im_end|>\n` (line 150); history assistant
`<|im_start|>assistant\n<think>\n\n</think>\n\n{content}<|im_end|>\n`
(line 99, the empty-think branch -- the lab's deterministic no-reasoning
hypothesis; the reasoned-content branch, line 101, is not a hypothesis);
generation prompt `<|im_start|>assistant\n<think>\n` (default, line 187)
and `<|im_start|>assistant\n<think>\n\n</think>\n\n` with
enable_thinking=false (line 185 -- rides the mutate path as
`nanbeige42/thinking-off`, kolibri convention). The default Chinese
Nanbeige identity system prompt (`<|im_start|>system\n` + identity line
+ `<|im_end|>\n`, lines 59-62) is ALWAYS emitted without a leading
system message -- documented-not-modeled. BOS: the template emits no
BOS literal, so the entry's bos is "" (phi4/qwen3 convention). Out of
scope: the tools system prompt (lines 34-51), tool_call/tool_response
rendering, preserve_thinking compaction, the multimodal reminder.

**Render cross-check (this round).** Every catalog entry and every
mutate variant was rendered with stock Jinja2 (+ a raise_exception shim)
against the official templates and found BYTE-IDENTICAL on the modeled
core (1-turn, 2-turn, and each thinking-branch variant; the
documented-not-modeled super-prompts/system blocks stripped before
comparison). Note: hy4's synthesized system block is
reasoning_effort-dependent (the no_think variant's block reads
`reasoning_effort:no_think`) -- verified per-variant.

**Built this cycle.** Three catalog entries + four mutate branches
(`hy4/no-think`, `minimax3/thinking-disabled`,
`minimax3/thinking-enabled`, `nanbeige42/thinking-off`) in
model_protocol_lab/catalog.py with tier-1 comment blocks (primary-source
URLs, jinja line numbers, caveats); docs/evidence/hy4-chat-template.jinja,
docs/evidence/minimax-chat-template.jinja,
docs/evidence/nanbeige-chat-template.jinja preserved; tests/
test_cycle15_hy4_minimax_nanbeige_families.py (26 tests: byte-exact
1-turn/2-turn renders, character-exact marker pins including the U+FF5C
pin, system-block/super-prompt/EOS documentation pins, branch mutation
+ dedup pins, per-family diff-vs-all-others pins with dynamic
len(CATALOG)-1 counts; deliberately no README pin -- coordinator owns
README counts/naming). Pin updates for the new total: the sorted key
lists in test_cycle5/6/7 (new keys land `hy4` after `glm53`, `minimax3`
after `llama4`, `nanbeige42` after `muse_glimmer` -- verified with
sorted()), `len(CATALOG) == 27` in test_cycle11/13, the thinking-off
invariant sets in test_cycle10/11 (+ `nanbeige42`), the ling/k2horizon
diff pins made dynamic, the e2e catalog-set pin, and the cycle5 audit
pins (`<|im_start|>` now collides with 7 families -- nanbeige42 added).
ruff check + format clean on the whole repo; mypy clean on all touched
files (20 pre-existing mypy errors in untouched files remain). Full
suite: 1140 passed, 5 failed, 1 skipped, 6110 subtests passed -- all 5
failures are README count/naming pins (test_cycle10/11/13/14 README
tests + test_cycle4 README test-count pin) because the README still
claims the 24-family catalog; README/CHANGELOG/VALIDATION untouched
(coordinator handles). No pushes made.

## 2026-10-15 (cycle 25 drift re-check)

SGLang PR #42466 re-read this round (https://github.com/sgl-project/sglang/pull/42466):
still OPEN, not merged; all three CI runs still failing (Base/Extra/AMD ROCm 10).
No change to the repo's SGLang multi-stop modeling; stays on the WATCH list.

## 2026-10-15 (cycle 28 drift re-check)

SGLang PR #42466 (multi-stop earliest-occurrence): re-check attempted via web
search this round -- no direct hit on the PR page, so status could NOT be
re-verified this round. Last confirmed state: OPEN, CI still failing
(checked 2026-10-15 ~12:30Z). Remains on the WATCH list; if merged, the
`emulate_sglang_stop` tie-break becomes: same firing step -> earliest
occurrence in decoded text -> list order.

Mistral Large 4 / Reflection Beam weights: still unshipped as of
2026-10-15 (sources: https://trendintech.com/ai-open-source-bet-beam-mistral-2026/,
https://blog.codercops.com/blog/mistral-large-4-open-weight-preview-2026,
https://beam.ai/agentic-insights/mistral-large-4-open-weights -- weights
promised for end of October 2026 / reported date October 27, 2026; both
still API previews). Remains on the WATCH list.

## 2026-10-15 (cycle 29: fixtures.py differential verification)

Differential verification of `model_protocol_lab/fixtures.py` against an
independent oracle written from the docstrings (tests/
test_cycle29_fixtures_verify.py, 139 tests): NO functional bug found.
Mutation sensitivity: 16/16 mutants caught on hermetic /tmp copies.
Two docstring claims corrected in fixtures.py (behavior unchanged):
(1) `parse_fixture_csv` no longer claims row numbers on *all* authoring
errors -- only per-row errors carry a row number; (2) `_extract_special_tokens`
parenthetical reworded (None optional fields are skipped, covered by the
already-scanned base field). README test count updated to 1644.

## 2026-10-15 (cycle 30: datasets.py differential verification)

Differential verification of `model_protocol_lab/datasets.py` against an
independent oracle written from the docstrings (tests/
test_cycle30_datasets_verify.py, 779 passed + 1 skipped): NO functional
bug found. Mutation sensitivity: 20/20 mutants caught on hermetic /tmp
copies. Doc discrepancies D1/D2 (str-subclass size-limit message; lone
surrogate UnicodeEncodeError vocabulary) are cosmetic error-message
quirks, pinned by tests -- no source change made. README test count
updated to 2424.

## 2026-10-15 (cycle 31: cli.py dispatch verification)

Verification of `model_protocol_lab/cli.py` + `__main__.py` (tests/
test_cycle31_cli_verify.py, 82 tests + 76 subtests): NO functional bug
found. Covers all 8 subcommands' parsing/validation/dispatch, exit-code
mapping, hostile argv, `allow_abbrev=False` option-abbreviation
rejection, determinism, and decision->exit-code mapping with a fake
backend. Mutation sensitivity: 9/9 mutants caught on hermetic /tmp
copies. No help-text discrepancies (4 consistency notes pinned in the
test-file docstring). README test count updated to 2506.

## 2026-10-15 (cycle 32: core.py Message/Format differential verification)

Differential verification of the `Message`/`Format` data layer against an
independent oracle written from the docstrings (tests/
test_cycle32_core_verify.py, 37 tests + 178 subtests): NO functional bug
found. Renders all 25 real CATALOG families through both oracle and
implementation (byte-exact); exact-type vs equality-based validation
asymmetries pinned (str-subclass roles accepted, str-subclass content
rejected). Mutation sensitivity: 15/15 mutants caught on hermetic /tmp
copies. No doc contradictions (code enforces more than docstrings
state; documented in the test-file docstring D1-D5). README test count
updated to 2543.

## 2026-10-15 (cycle 33: reporting.py internals verification)

Verification of `model_protocol_lab/reporting.py` report internals (tests/
test_cycle33_reporting_verify.py, 11 tests + 342 subtests): NO functional
bug found. Covers all 12 section renderers, byte-determinism of HTML/JSON
(across PYTHONHASHSEED via subprocess), self-contained-HTML property
(zero external refs, CSP-allowlist script hash), fixed-vocabulary
markers, summary computations, JSON payload shape, O_EXCL + 0o600/0o700
permissions, hostile-input escaping. Mutation sensitivity: 20/20 mutants
caught on hermetic /tmp copies. Two documentation fixes in reporting.py:
(1) matrix row-order comment clarified (hardcoded class_order; unknown
classes get no matrix row); (2) `_held_out_section` docstring added,
noting absent/malformed conflation. README test count updated to 2554.

## 2026-10-15 (cycle 34: experiments.py internals differential verification)

Differential verification of `model_protocol_lab/experiments.py`
internals against independent oracles written from the docstrings (tests/
test_cycle34_experiments_verify.py, 59 tests + 2718 subtests): NO
functional bug found. Covers classify_observation/failure_taxonomy,
budget_frontier, _resolve_boundaries/candidate_fingerprint,
_stop_cell/stop_matrix, logprob internals, prompt_sensitivity_sweep, and
seed handling (one seed reaches every generate call; per-fixture request
seeds equal the independently recomputed sha256 values). Mutation
sensitivity: 15/15 mutants caught on hermetic /tmp copies. One docstring
fix: `_stop_cell` now documents that unavailable outcomes omit the
`stop_text` key entirely (was: claimed None). README test count updated
to 2613.

## 2026-10-15 (cycle 35: backends.py verification)

Verification of `model_protocol_lab/backends.py` (tests/
test_cycle35_backends_verify.py, 184 tests): NO functional bug found.
Covers the Backend protocol contract (seeded fake), asset loading
(traversal/symlink rejection, 128 MiB boundary, sharded layouts),
LocalHFBackend fail-closed init paths, TruncatedPromptBackend,
StopOutcome, byte machinery, and all four stop ports vs independent
oracles (seeded fuzz). Mutation sensitivity: 20/20 mutants caught on
hermetic /tmp copies. 7 doc-vs-code discrepancies (D1-D7) pinned in the
test-file docstring; deliberately no source changes: D1
(`_directory(Path(""))` resolving to cwd) is a pathlib normalization
quirk -- `Path("")` IS `Path(".")`, indistinguishable from an explicit
cwd, and rejecting it would break the legitimate case; D2-D7 are
behavioral nuances, not doc contradictions. README test count updated
to 2797.

## 2026-10-15 (cycle 36: debug.py REPL verification + 1 real bug fixed)

Verification of the `mpl debug` REPL command loop (tests/
test_cycle36_debug_repl_verify.py, 140 tests + 10 subtests): found ONE
REAL BUG -- truncated transcripts could never be saved (truncation kept
_MAX_SAVE_BYTES of content then prepended the 29-char marker, pushing
the total back over the bound so .save always refused). FIXED in
debug.py: the marker is now budgeted inside the truncation
(remaining = _MAX_SAVE_BYTES - len(marker)); regression test
TestTranscript.test_truncated_transcript_save_succeeds pins the fix.
Also fixed two doc stalenesses in docs/USAGE.md: `.families` row said
"16 catalog families" (catalog has 27); `.quit` row now documents the
literal `EOF` line. Mutation sensitivity: 16/16 mutants caught on
hermetic /tmp copies. Security: no leaks beyond the documented
USAGE.md section 10 contract (operator content by design). README test
count updated to 2937.

## 2026-10-15 (cycle 37: diffbackends.py classification internals)

Differential verification of the divergence classification internals
(tests/test_cycle37_diffbackends_verify.py, 39 tests + 1996 subtests):
NO implementation bug found. Mutation sensitivity: 7/7 mutants caught on
hermetic /tmp copies. Two docstring fixes in diffbackends.py
(behavior-preserving): (1) the STR_OVER_EOS_PRECEDENCE "co-fires with
EOS_PRECEDENCE on the tie" claim was false in general -- co-firing holds
only when no other stop cell fires strictly before the tie step
(counterexample pinned); module docstring and code comment corrected;
(2) the diff_backends return-key list now names prompt_tokens/
output_tokens. README test count updated to 2976.

## 2026-10-15 (cycle 38: scoring.py internals verification)

Verification of `model_protocol_lab/scoring.py` internals (excluding
the cycle-12-verified `score_target_logprobs` engine; tests/
test_cycle38_scoring_verify.py, 86 tests + 500 subtests): NO functional
bug found. Covers _f32 IEEE-754 semantics vs struct oracle,
score_span_from_logits (8 rejection codes + precedence, masking
discipline, float64 accumulation of unrounded values),
_safe_decode/_safe_encode, check_alignment, _provider_code,
_unavailable reason vocabulary, _require_torch. Mutation sensitivity:
12/12 mutants caught on hermetic /tmp copies. One docstring fix:
`score_span_from_logits` now documents that non-tensor logits raise
TypeError("logits_must_be_tensor") (was: claimed all rejections are
ValueError). README test count updated to 3062.

## 2026-10-15 (cycle 39: catalog integrity cross-check + 1 robustness fix)

Whole-catalog integrity sweep (tests/test_cycle39_catalog_integrity.py,
44 tests + 5803 subtests): NO catalog data bugs. Covers the exact
27-key pin, byte-identical renders (1/3/5-turn), an independent
from-first-principles Format.render oracle (80 seeded conversations x
27 families), mutate-branch validity/dedup/rebuild equivalence,
fallback semantics, hostile lookups, red-team contents. Mutation
sensitivity: 9/9 mutants caught on hermetic /tmp copies. One robustness
FIX in catalog.py: `candidates()` now type-gates
(`type(family) is not str`) before the CATALOG lookup, so the
documented ValueError("Unknown format family") holds for unhashable and
non-str keys instead of leaking TypeError. README test count updated to
3106.

## 2026-10-15 (cycle 40: doctor.py examine_model internals verification)

Behavioral verification of `doctor.py`'s `examine_model` internals
(excluding the cycle-25/26-verified asset validation and weight_headers
checks; tests/test_cycle40_doctor_verify.py, 101 tests + 171 subtests):
NO bug found. Covers report assembly, verdict logic (failed iff any
check failed), per-asset checks (config, tokenizer_file dropout
strictness, tokenizer_deep fail-closed, added/special token maps, asset
manifest), portability machinery (110-row matrix, bos_skipped flags,
asymmetry warning), error paths (examine_dataset fixed vocabulary,
template_conformance validation, doctor() key assembly). Mutation
sensitivity: 10/10 mutants caught on hermetic /tmp copies. Doc notes
(no bugs): _portability_check "Never raises" holds even for non-path
inputs; template_conformance no-raise starts after arg validation;
context_window bool exclusion; unreachable empty-inventory branch. README
test count updated to 3207.

## 2026-10-15 (cycle 41: public API surface + documented-example accuracy)

Executed every runnable USAGE.md/README CLI example and compared against
documentation (tests/test_cycle41_api_surface_verify.py, 56 tests + 56
subtests; model-backed tests use a tiny synthetic GPT-2 checkpoint built
locally). Fixed FOUR doc errors (docs only, no behavior changes):
(1) README Quick start `fixtures build -o` -> `--output` (nonexistent
flag), and the example target changed to /tmp/mpl-fixtures.json since
fixtures build never overwrites the shipped examples/fixtures.json
(verified: builds verbatim, semantically identical to shipped file);
(2) USAGE.md section 10 `.boundary` "same grammar as `run --boundary`"
-> `run --boundary-ids`; (3) README 27-family parenthetical listed 26
names -- added Gemma 4 (two spots); (4) README "`catalog` lists every
family with its source" -> framing fields + code-comment sources
(`mpl catalog` emits no source field). Public API surface: __init__.py
exports nothing but __version__ (0.1.0a0, consistent); the CLI is the
public interface. Also installed the package editable in the venv so
console-script tests run locally as in CI. README test count updated to
3263.

## 2026-10-15 (cycle 42: core.py remaining helpers verification)

Differential verification of the six core.py pure helpers not covered by
prior cycles (truncate_text_tail, first_boundary, local_repairs,
template_diff, check_boundary_portability,
template_tokenizer_compatibility; tests/
test_cycle42_core_helpers_verify.py, 40 tests + 2129 subtests): NO
functional bug found. Mutation sensitivity: 16/16 mutants caught on
hermetic /tmp copies. One documentation gap closed: `first_boundary`
had no docstring at all -- added one stating the pinned contract (first
contiguous-subsequence index, falsy stop raises ValueError, no type
validation). Remaining D2-D7 notes are pinned behavioral details, not
doc errors. README test count updated to 3303.

## 2026-10-15 (cycle 43: second canary evidence-leak sweep)

Second canary sweep with marker CANARY43-LEAKSWEEP-QZ7 planted as
fixture content, expected answers, generated text and token-ID arrays
(tests/test_cycle43_leak_sweep.py, 35 tests + 11 subtests): NO LEAK
FOUND. Covers save_report JSON+HTML with include_evidence False/True
(backend_divergences, template_conformance, stop-matrix,
budget-frontier, logprob sections), divergence_section_data end to end,
diff-reports/diff-backends (function + CLI), debug REPL surfaces
(by-design echoes vs must-stay-clean pinned per USAGE.md section 10),
doctor + portability matrix, CLI inspect/fixtures build, and the
committed examples/sample-report. Sensitivity: 3/3 reintroduced-leak
mutants caught on hermetic /tmp copies (stripped _strip_raw_diagnostics,
HTML stop_text render, _public_view keeping observations). README test
count updated to 3338.

## 2026-10-15 (cycle 44: VALIDATION.md accuracy audit)

Accuracy audit of VALIDATION.md (tests/test_cycle44_validation_audit.py,
36 tests): refreshed the stale top-section figures (cycles 10-13 ->
cycles 10-44): test count 1001 -> 3374 (+43319 subtests), ruff format
66 -> 101 files, bandit "1 B105 false positive" -> "no issues
identified" (targeted nosec suppressions landed), families 23 -> 27,
red-team follow-ups "8 fixed" -> "7 fixed + 1 documented-only". The
cycle-1 FAQ line (12 Q&As) is an archival claim about the creation
state, left as-is. Environment, CLI surface, CI YAML, CITATION.cff,
USAGE.md technical sections, and the "Not established" section all
verified accurate. README test count updated to 3374.

## 2026-10-15 (cycle 45: run_experiment core-loop differential re-verification)

Differential re-audit of the run_experiment core loop
(tests/test_cycle45_run_experiment_audit.py, 13 tests): hand-rolled
independent oracle (own subsequence search, summary math, taxonomy,
result assembly) vs the real loop, 200-iteration fuzz over fixtures,
budgets, families, boundaries, eos ids, seeds. Verified: conservative
unique-winner rule (2-way/3-way/all-pass ties -> ambiguous, zero
held-out calls), search/held-out separation under adversarial layouts,
exact budget accounting (no off-by-one, cap=1 edge), determinism
(byte-identical JSON on same seed), boundary handling incl.
classify_observation unit contract, failure paths with allowlisted
error codes. No real bugs found (zero oracle divergences). Mutation
gate: 6/6 caught (search budget +1, ties-select-first, held-out gating
removed, winner held-out with default boundary -> decode_failed
cascade, lax >=1 winner, inverted held-out verdict); unmutated control
clean. Source untouched (sha256 61dcc509... verified before/after).
README test count updated to 3387.

## 2026-10-15 (cycle 46: docs command-example accuracy audit)

End-to-end audit of every runnable command example in docs/USAGE.md
(ss1-12) and the README command sections
(tests/test_cycle46_docs_accuracy_audit.py, 54 tests + 12 subtests):
command names exist, every documented flag parses and round-trips,
representative invocations run against a scripted char<->ord fake
backend or synthetic tokenizer-only checkpoints (run, doctor incl.
--portability-matrix/--boundary-ids/--reference-model, fixtures
build with the documented CSV, verbatim debug session from s10,
diff-backends reproducing KEPT_VS_STRIPPED byte-for-byte,
diff-reports JSON/summary shapes + exit codes). No material
inaccuracies; one genuine precision fix applied: USAGE.md s11 claimed
all input bounds are "enforced with fixed error codes" -- the 1 MB
--prompt-text/--output-text caps go through argparse (exit 2), so the
wording was corrected (behavior already pinned in tests). Not run
locally: pip installs (offline), full unittest discover (over budget),
make_sample_report.py (would rewrite committed example files),
real-model runs (no weights), exit-130 signal path, s8.2/8.5/8.9
literature prose, local_context_limit_exceeds_declared_window (checked
statically: backends.py raises the exact code). README test count
updated to 3441.

## 2026-10-15 (cycle 47: sample-report pipeline end-to-end audit)

End-to-end audit of examples/make_sample_report.py
(tests/test_cycle47_sample_report_audit.py, 5 tests + 10 subtests):
regeneration into /tmp via the script's own build_sample_report() with
its scripted DemoBackend (decision compatible_on_fixture_set, selected
llama3); HTML self-contained (zero external URLs, inline-only script,
default-src 'none' CSP); exact 19-key top-level JSON schema pinned
(schema_version 1, config echoes, nested search/held-out/
selection_margin/token_overhead/failure_taxonomy schemas);
byte-identical determinism across two fresh regenerations; and the
regenerated output is byte-identical to the committed
examples/sample-report/ (no schema drift). No real bugs. Deliberately
not verified: actual browser rendering, CSP script-src hash
re-derivation, pip-audit/CI. README test count updated to 3446.

## 2026-10-15 (cycle 48: third canary evidence-leak sweep)

Third canary sweep of all public report surfaces
(tests/test_cycle48_evidence_leak_sweep3.py, 14 tests + 2 subtests):
unique canary CANARY48-LEAKSWEEP3-ZK2 planted in raw text, expected
answers, and token sequences; every public surface swept with
include_evidence=False (run JSON via save_report, run HTML, CLI
run --output end-to-end against a toy GPT-2 checkpoint incl. stdout/
stderr, diff-reports function+CLI+render_summary, debug saved
transcript, doctor function+CLI, committed examples/sample-report/).
Verdict: CLEAN -- all carriers stripped (observations popped,
hand_ids/template_ids gone, stop texts gone, detail withheld,
raw_evidence_included false). Controls prove the canaries were planted
(evidence=True runs contain them; debug .render echo by design).
Mutation gate 3/3 caught on a /tmp package copy (_strip_raw_diagnostics
bypass, _cmd_save token-ID leak, diff_reports answer_probe leak);
repo source sha256-verified untouched. README test count updated
to 3460.

## 2026-10-15 (cycle 49: logprob scoring numerics differential)

Numerical differential of score_target_logprobs + teacher-forced
logprob diagnostic against an independent pure-Python math.log oracle
(tests/test_cycle49_logprob_numerics_audit.py, 17 tests): 300 seeded
fuzz configs (vocab 2-512, prompt 1-8, targets 0-64, fp32/fp64, logit
scales 1/5/30, spiky rows with dynamic range 60) plus directed and
end-to-end cases -- 6145 field comparisons with MAX abs/rel divergence
exactly 0.0: the engine's float64 path agrees with the oracle
bit-for-bit after float32 rounding, including extreme +-30 logits.
Pinned: tolerances |diff| <= max(2e-5, 4*ulp32) and rel <= 1e-6,
exact float32-representability, no NaN/inf, fsum(per_token) within
(n+1) ulp32 of the reported total, causal-shift spike test. No real
numerical bugs. Mutation gate 6/6 caught on /tmp copies (mean-as-total,
off-by-one gather, log2-vs-ln normalizer, drop-last-target, skip
float32 rounding, mean denominator n+1); repo source sha256-verified
untouched. Not in scope: _aggregate_fixture_scores (plain summation),
real-HF-backend forward numerics (needs a checkpoint), diagnostic
interpretation thresholds (policy, not numerics). README test count
updated to 3477.

## 2026-10-15 (cycle 50: adversarial HF stop-emulation differential)

Adversarial differential of emulate_hf_stop_string vs the real
transformers 5.19.0 StopStringCriteria
(tests/test_cycle50_stop_emulation_adversarial.py, 58 tests + 493
subtests): real tokenizers-built PreTrainedTokenizerFast on all three
upstream matching modes (ByteLevel, ByteFallback, classic/Metaspace),
427 seeded adversarial cases + ~50 directed edge cases. 4 REAL
divergences found, all documented with reproducers; coordinator
decision: DOCUMENT, do not change behavior --
(1) empty stop: lab invalid_emulation_input (unreachable by design:
diff_backends raises invalid_stop_texts; stop_matrix degrades an
undecodable boundary to an unavailable cell; upstream's fire-at-step-1
is pathological);
(2) multi-stop window: lab composes per-stop len(stop) windows
(deliberate, mirrors the single-stop tensor walk); upstream's joint
global-window bound is an artifact reachable only with exotic
empty-string vocab entries;
(3a) classic path WordPiece: stop "ab" over ["a","b"] false-positives
here vs true "a b" (disclosed approximation; no catalog family uses
non-concatenative decoders);
(3b) upstream fires a step early on its own cleaning artifact (lab
fires at the true position -- lab is more correct);
(4) non-contiguous vocab ids: degenerate (real checkpoints are
0..N-1). Disclosure added to emulate_hf_stop_string's docstring
("Known divergences") + one sentence in diff_backends on deliberate
per-stop composition. Tie-break verified: transformers has literally
no selection code (torch.any); earliest-completing wins by
construction, same-step ties outcome-identical. Mutation gate 6/6
caught on /tmp copies. README test count updated to 3535.

## 2026-10-15 (cycle 51: upstream drift re-check)

Re-checked all standing watch items against primary sources
(github.com PR/release pages read in full, huggingface.co model
pages, GitHub release-tag API for publish dates):
- SGLang PR #42466 (multi-stop earliest-occurrence): STILL OPEN, CI
  still red (PR Test Base/Extra/AMD ROCm all failing). No action;
  keep watch. Proposed semantics if merged: earliest occurrence in
  decoded text wins, ties keep list order.
- vLLM PR #49391 equivalent check: NO FIDELITY GAP -- the repo
  already models and pins earliest-completing selection with
  list-order ties (emulate_vllm_holdback docstring;
  tests/test_cycle10_sglang.py, test_cycle15_divergence_verify.py,
  test_cycle37_diffbackends_verify.py). Released vLLM v0.27.0+
  behavior matches the emulation.
- Mistral Large 4 (mistralai/Mistral-Large-4-1T-A52B): still
  "Upcoming release", ETA October 31, 2026. Reflection Beam: no
  weights (org lists zero models; "later in October"). No action.
- transformers v5.20.0 published 2026-10-15T18:05:36Z: stopping_criteria.py
  byte-identical to v5.19.0 (34,389 bytes each; last change 2026-09-18
  docstring-only). No stop-semantics drift. No action.
- vLLM latest v0.31.0 (2026-10-05), SGLang latest v0.5.21 (2026-10-02):
  no stop-string/detokenizer changes in the window. No action.
- NEW FAMILY LEAD: **Muse Spark Open** (Meta) -- weights released
  2026-10-15 (816B total / 42B active MoE, 1M context, multimodal,
  BF16 ~1.5TB / NVFP4 ~530GB, Muse Model License). Repo has
  muse_glimmer but not muse_spark. Top of the catalog research queue;
  chat template must be verified from the HF model card before adding
  (cycle 52). Also flagged: repo `ling` entry vs Ling 3.0's official
  template (native transformers support landed today); Qwen3.8 as a
  known out-of-window catalog gap.

## 2026-10-15 (cycle 52: Muse Spark Open catalog family)

Added the 28th catalog family `muse_spark` (Meta Muse Spark Open,
meta-models/Muse-Spark-Open-816B-A42B, weights released 2026-10-15;
816B total / 42B active MoE, 1M context, Muse Model License).
Template verified byte-exact against the primary sources this round:
the official chat_template.jinja (230 lines) from
hf-internal-testing/tiny-muse-spark (HF-official test mirror of the
gated release tokenizer; reference template in transformers PR
#49644), cross-checked by rendering under stock Jinja2. Modeled
deterministic core: BOS <|begin_of_text|>, user
<|start|>user<|message|>{content}<|eot|>, assistant history
<|start|>assistant to=user<|message|>{content}<|eot|>, generation
prompt <|start|>assistant (bare); the always-emitted system block,
tool/reasoning branches, multimodal placeholders and per-message
timestamps are documented-not-modeled (lab convention); <|eot|> is the
end-of-message marker, not a stop token (EOS <|end_of_text|> not
modeled; Format has no EOS field). 15 byte-exact tests
(tests/test_cycle52_muse_spark.py). Count pins updated for 27->28
families / 110->114 doctor candidates: test_cycle39 (key set, sorted
order, open-suffix pin), test_cycle5/6/7 (sorted family lists),
test_cycle11/13 (README N-family claim), test_cycle36 (REPL count),
test_cycle40 (portability matrix 114), test_cycle41 (EXPECTED_FAMILIES
+ renames), test_cycle44 (28-not-23), test_cycle46 (catalog/REPL
counts); README enumerations and USAGE.md `.families` row updated to
28. README test count updated to 3550.

## 2026-10-15 (cycle 53: Ling 3.0/3.1 template audit)

Verified the repo's `ling` family against the official templates from
primary sources (tests/test_cycle53_ling_template_audit.py, 7 tests):
chat_template.jinja downloaded byte-exact from inclusionAI/Ling-3.0-flash
(md5 23b682c460649dbbc03e9d2bc9609071) and inclusionAI/Ling-3.1-flash
(md5 24b514df8b812d0bb0e9c061d0bb164c -- byte-identical to the repo's
docs/evidence/ling31-chat-template.jinja); transformers PR #49446 adds
model code only, no template docs. VERDICT: MATCH -- Ling 3.0 vs 3.1
templates render byte-identically across 10 conversation/thinking-mode
combinations (differences are jinja whitespace-control only), and every
modeled repo string is byte-identical to the official 3.0 plain-text
core (BOS <|startoftext|>, <role>HUMAN</role> user frame,
<role>ASSISTANT</role>\n<think></think> assistant frame, <|role_end|>
terminator, thinking-on/off gen prompts). Noted artifact (not a
divergence): the official template leaks a literal newline+4 spaces
after a system block (jinja indentation); Format never renders system
blocks. No catalog change needed. README test count updated to 3557.

## 2026-10-15 (cycle 54: Qwen3.8 catalog family)

Added the 29th catalog family `qwen38` (Qwen/Qwen3.8-27B, released Aug
2026). Template verified byte-exact against the primary source: the
official chat_template.jinja at main (verified commit 412f8b6). The
template DIFFERS from qwen3: thinking ON (the default) opens the
generation prompt with <|im_start|>assistant\n<think>\n (qwen3 ends at
<|im_start|>assistant\n); thinking OFF prefills an empty think block
<|im_start|>assistant\n<think>\n\n</think>\n\n (same tail as
kolibri/nanbeige42, unlike nemotron35's "<think></think>") -- modeled
as the qwen38/thinking-off mutate-path variant. Per-turn frames are
ChatML, no BOS -- byte-identical to qwen3. Deliberately not modeled:
the auto-injected system reasoning-effort block, assistant
reasoning-content turns (preserve_thinking), tool branches. 17
byte-exact tests (tests/test_cycle54_qwen38.py). Notable: qwen38's base
framing is byte-identical to nemotron35's (both thinking-by-default
ChatML models) -- documented as the second duplicate group in
test_cycle39 (DISCREPANCY-3), alongside chatml/kolibri/qwen3. Count pins
migrated 28->29 families / 114->118 doctor candidates (qwen38 adds base
+ thinking-off + prefix-newline + prefix-trim): test_cycle39 (key set,
branches, duplicate groups), test_cycle5/6/7/e2e (family lists),
test_cycle10/11 (thinking-off invariant), test_cycle10/11/13/14/15
(README N-family claims), test_cycle36/41/44/46 (family counts),
test_cycle40 (118 candidates); README enumerations (+Qwen3.8),
USAGE.md `.families` row, VALIDATION.md updated to 29. README test
count updated to 3574.

## 2026-10-15 (cycle 55: core.evaluate() differential audit)

Differential re-verification of core.evaluate() and the observation
pipeline against a hand-rolled independent oracle
(tests/test_cycle55_evaluate_audit.py, 53 tests): 400 randomized
trials x 1-3 fixtures (exact matches, near-misses, empty outputs,
multi-byte boundaries, boundary-at-0, over-long boundaries, overlapping
boundary/answer, decode failures, invalid generations, encode/render
failures, mid-run cancellation), field-by-field Observation comparison
+ report status/counts, seed derivation (sha256 seed\0case.id, 4-byte
big-endian), determinism, secret-leak scan, error-code sanitization,
diagnostic boundary_text_match modes, abort-vs-terminate input-id
retention. No real bugs in source (one divergence found was an oracle
bug -- the post-encode cancellation path uses terminate(), retaining
input_ids; oracle fixed, source confirmed correct). Mutation gate 6/6
caught on /tmp copies (case-insensitive match, boundary off-by-one,
invalid->completed, dropped last token, decode_error taxonomy, inverted
endswith); repo source sha256-verified untouched. Not in scope:
check_boundary_portability, candidate ranking/selection, real backends,
thread-safety/perf, first_boundary's no-validation paths. README test
count updated to 3627.

## 2026-10-15 (cycle 56: prompt-sensitivity sweep audit)

Adversarial re-verification of the prompt-sensitivity sweep machinery
(core.truncate_text_tail, backends.TruncatedPromptBackend,
experiments.prompt_sensitivity_sweep;
tests/test_cycle56_sensitivity_sweep_audit.py, 30 tests + 9640
subtests): exact output pins on adversarial inputs (empty, 1-byte,
combining marks, emoji, 100k-char, length caps); invariants (byte
length, strict whole-character prefix, None-iff-codepoint-split via an
independent backward-continuation-byte oracle, no U+FFFD fabrication,
determinism); 1205-prompt x drops-1..8 seeded fuzz; operator algebra
(documented non-idempotency, non-composition of depths); wrapper
contract (prompt_truncation_splits_codepoint code, passthrough,
fingerprint, constructor validation); sweep aggregation truth table
incl. winner-same/decision-differs branch, per-depth independence via a
recording backend, max_drop_bytes validation. Scope correction: the
sweep has exactly ONE perturbation operator (tail byte-truncation);
unicode enters only through the granularity rule. No real bugs. One
documented edge (not a bug): truncate_text_tail on lone surrogates
raises UnicodeEncodeError; downstream evaluate collapses it to
encode_failed and the sweep records the failed decision (documented
fallback). Mutation gate 6/6 caught on /tmp copies; repo source
sha256-verified untouched. README test count updated to 3657.

## 2026-10-15 (cycle 57: budget_frontier adversarial audit)

Adversarial re-verification of experiments.budget_frontier
(tests/test_cycle57_budget_frontier_audit.py, 14 tests): 11
hand-pinned adversarial cases (step functions, flat regions,
flip-at-budget-1 as initial point, multi-flip non-monotonic decisions,
singleton ranges, min-budget-beyond-max, first-occurrence counting,
malformed outputs, per-candidate boundary_token_ids overrides,
flip-flip-back range merging, held_out_failed native comparison) plus a
60-seed differential fuzz against an independent reference (naive
subsequence scan, per-budget truncation re-evaluation). Invariants
pinned: sorted/strictly-increasing/minimal decision steps, exact flip
convention at N>1 vs N-1, completeness, maximally-merged disjoint flip
ranges, budget_robust iff no ranges, determinism. Sharp edge pinned:
budget 0 is never evaluated (sweep is 1..max_new_tokens), so a frontier
point at budget 1 is an initial point, not a flip vs budget 0. No real
bugs (4 initially mis-derived hand expectations were oracle arithmetic
errors -- implementation and independent reference agreed). Mutation
gate 6/6 caught on /tmp copies (off-by-one min budget, drop first/last
decision step, inverted pass-step detection, inverted flip_ranges
native comparison, off-by-one flip-range end); repo source
sha256-verified untouched. Not in scope: HTML/markdown rendering,
exhaustive validation paths, max-scale perf, the truncation-equals-
regeneration modeling assumption. README test count updated to 3671.

## 2026-10-15 (cycle 58: HTML escaping / XSS audit)

Adversarial audit of HTML escaping in the report renderers
(tests/test_cycle58_html_escaping_audit.py, 9 tests + 14 subtests):
reporting.render_html (all 12 sections), save_report, the sample-report
generator (shares the renderer), and diffreports.render_summary
(verified: no HTML entry point, terminal-only with ANSI sanitization).
14 adversarial payloads (script tags, <img onerror>, javascript: URLs,
</textarea>/</title> breakouts, HTML comments, CSS expression(),
dotless-i <scrıpt>, fullwidth tags, attribute-injection quotes,
mixed-case tags, U+2028) injected into every renderer-visible field:
every payload rendered only in escaped form; an HTMLParser-based audit
found exactly one <script> (the renderer's own sorter), no on* attrs,
no javascript: in URL attributes, no img/iframe/textarea elements;
JSON round-trip integrity and legitimate-content fidelity (unicode,
<|special|> tokens, code with <>&) verified; CSP meta with hash-locked
script-src confirmed present. No real escaping holes. Mutation gate 3/3
caught on /tmp copies (dropped escaping, escape-<-only, double-escape);
repo untouched. Not verified: live-browser execution (no live browser
in this environment; structural audit + raw-substring checks used
instead). README test count updated to 3680.

## 2026-10-15 (cycle 59: boundary-portability checker adversarial audit)

Adversarial differential audit of check_boundary_portability
(core.py:413) and its generalization template_tokenizer_compatibility
(core.py:462) (tests/test_cycle59_boundary_portability_audit.py, 33 tests
+ 398 subtests): hand-pinned exact-dict contract tests for every branch,
an independent hand-rolled oracle (no shared code with the
implementation) differential-fuzzed over 266 randomized cases (19 encode
behaviours x 14 output shapes) + 60 invalid-argument cases, and 14
adversarial edge tests for the generalization (row schemas, precedence,
duplicate inventory, guard tables). Mutation gate 6/6 caught on /tmp
copies (==->!=, inverted validity guard, dropped invalid-output guard,
swapped identical/divergent_ids, fragments misclassification,
degraded-membership narrowing). No genuine bugs — CLEAN. One contract
gray zone pinned: empty-tuple secondary encode classifies as
divergent_ids (sibling probe treats empty encodes as encode_invalid);
any change is now deliberate. Deliberately not verified:
doctor._portability_check / doctor.portability_matrix (require real
checkpoints; thin wrappers, existing tests). README test count updated
to 3713.

## 2026-10-16 (cycle 60: test-harness concurrency robustness)

Flaky suite failures (1-3 per run) investigated and fixed. Reproduced
deterministically by running two full suites concurrently; all failures
were in test scaffolding, never product code:
- cycle37 mutant test: FileExistsError on the FIXED /tmp scratch dir
  (both processes staged into /tmp/cycle37_diffbackends_mutants).
- cycle33 mutant test: ModuleNotFoundError 'm12_overhead_stat' (one
  process's setUpClass rmtree deleted the other's in-use
  /tmp/cycle33_mutants packages mid-import).
- cycle35 128 MiB fixture writes: ENOSPC on the 512 MiB /tmp tmpfs.
Fixes (tests only, product code untouched): PID-scoped mutant scratch
roots in test_cycle19/20/32/33/37/42; scratch-dir cleanup after the
cycle32/37/42/19 sensitivity runs and cycle20 staged files (stops /tmp
accumulation that starved the fixture tests); sparse-file oversized
fixture in cycle35 (identical byte count on read, ~zero blocks);
honest skip of the exactly-at-cap test when <256 MiB scratch is free.
Regression tests: tests/test_cycle60_concurrency_robustness.py (9 tests)
pins PID-scoping at all six sites, runs two parallel mutant-sensitivity
batteries, pins the sparse rejection path, and pins the disk-space skip
via a mocked disk_usage. README test count updated to 3722.

## 2026-10-16 (cycle 61: upstream drift re-check)

All seven tracked items re-verified against primary sources; no drift:
1. SGLang PR #42466: still open, CI red on all three lanes (Base/Extra/AMD
   ROCm 10); intended earliest-occurrence tie-break unchanged.
2. transformers: v5.20.0 still latest; stopping_criteria.py untouched since
   the v5.19.0/v5.20.0 byte-identical baseline (only docstring/loop-hoist
   PRs predate the freeze).
3. vLLM: v0.31.0 still latest; no stop/detokenize changes.
4. SGLang: v0.5.21 still latest.
5. Mistral Large 4 (mistralai/Mistral-Large-4-1T-A52B): still "Upcoming
   release", ETA October 31, 2026; no weights.
6. Reflection Beam: no public weights (waitlist only); 501B/23B-active MoE,
   23.8T tokens, 1M context per current coverage.
7. Muse Spark Open weights (meta-models/Muse-Spark-Open-816B-A42B, released
   2026-10-15, 816B/42B-active MoE, 1M context): already cataloged as the
   repo's `muse_spark` family in cycle 52 (byte-exact from the official
   chat_template.jinja via hf-internal-testing/tiny-muse-spark and
   transformers PR #49644; system block, reasoning-effort mapping
   low->8/medium->32/high->256/xhigh->512, multimodal placeholders all
   pinned in tests/test_cycle52_muse_spark.py). Cross-checked against the
   public gpt-oss Harmony template (minpeter/gpt-oss mirror): Muse Spark's
   template is its own Glimmer-derived variant (<|eot|> closers,
   `<|start|>assistant to=user<|message|>`), not gpt-oss's channel template
   -- the repo models the correct one. One prose discrepancy noted and
   dismissed: secondary coverage quotes the effort mapping as 16/64/256/512
   while the actual jinja (primary source) renders 8/32/256/512; the repo
   pins the jinja. No catalog change. Next candidates if templates emerge:
   models newly supported in transformers v5.20.0 (Muse Glimmer, Ling 3.0
   Flash, Kolibri 1, MiniCPM-V 4.7) -- need concrete template text first.

## 2026-10-16 (cycle 62: doctor portability-matrix adversarial audit)

Closed the cycle-59 gap: adversarial differential audit of
doctor._portability_check (doctor.py:159) and doctor.portability_matrix
(doctor.py:273) using SYNTHETIC tokenizers only (a fake `transformers`
module injected into sys.modules; no checkpoints, no network, no real HF
objects) -- tests/test_cycle62_doctor_portability_audit.py, 60 tests.
Established the exact consumed interfaces (primary: .decode only;
reference: .encode(text, add_special_tokens=False);
.convert_tokens_to_ids + 7 special-token attrs), 57 hand-pinned
branch tests, 220+130 seeded differential fuzz subtests vs independent
oracles with full-dict equality, and a 40-case cross-check reproducing
core.template_tokenizer_compatibility exactly. Notable pinned non-bugs:
tuple()-normalisation of list/generator encodes, empty-tuple reference
encode -> mismatch (not degradation), per-literal failure isolation,
bool ids excluded via type() is int. Mutation gate 6/6 killed on /tmp
copies (control pristine PASS). No genuine bugs -- CLEAN. Deliberately
not verified: real transformers/tokenizers against real checkpoint files;
examine_model/doctor() integration beyond these functions. README test
count updated to 3782.

## 2026-10-16 (cycle 63: search/held-out separation adversarial audit)

Adversarial audit of the search/held-out separation hard constraint
(tests/test_cycle63_heldout_separation_audit.py, 41 tests). Separation
mechanism as found: Split frozen dataclass (core.py:136-188) enforces
non-empty partitions, globally unique NFC-normalized fixture IDs, no
shared groups, no identical inputs across the split; CSV loader assigns
partition from an explicit column (fixtures.py:150-208); search stage
evaluates once per candidate (experiments.py:1008); selection gate
(len(winners)!=1 -> abstain) runs BEFORE any held-out evaluate
(experiments.py:1032-1034); held-out measured exactly once on the unique
winner (1035,1042-1056); per-call request seeds sha256(seed, case.id)
keep stages disjoint under seed-keyed caches; reporting strips held-out
observations by default. Method: SeedSpyBackend (output a pure function
of (candidate, fixture) from a hidden behavior table), planted canary
held-out fixture with distinctive answer, full-equality assertions on
exact ordered backend call sequences, 24 seeded fuzz trials vs an
independent oracle. Mutation gate 4/4 caught on /tmp copies (canonical
leak: search loop measuring split.held_out -- detected by the canary;
dropped unique-winner guard; held-out reusing search observations;
held-out measuring split.search). No genuine leaks -- CLEAN. By-design
note: `mpl debug` REPL merges partitions for interactive inspection,
outside the automated selection pipeline. Deliberately not verified:
live-model backends; statistical strength of the split itself. README
test count updated to 3823.

## 2026-10-16 (cycle 64: CLI-surface adversarial audit)

Adversarial audit of the full CLI surface (tests/test_cycle64_cli_surface_audit.py,
110 tests + 78 subtests): all 8 subcommands (catalog, inspect, doctor, run,
fixtures build, diff-reports, diff-backends, debug) plus top-level -- bare
invocation, unknown subcommand, abbreviation rejection, --help everywhere,
missing/wrong-type/out-of-range args, malformed --boundary-ids and
--boundary-map, nonexistent/unreadable/directory/FIFO paths, existing
--output, sidecar collision, the 32 MiB results.json cap, stdout-vs-stderr
routing per row, debug REPL read-only-ness (Split/CATALOG snapshot-identical
after a command battery; no experiments import; partitions merged into a
fresh tuple), and console-script equivalence (model-protocol-lab vs
python -m). Every USAGE.md-documented error code emitted verbatim; exit
discipline holds (2 argparse, 1 fixed-code input failure, 0/1/2/130 run
decisions). No genuine code bugs -- CLEAN. One genuine doc fix from the
audit: USAGE.md (and the load_results docstring) claimed diff-reports
refuses directories with results_not_regular_file -- reality: a directory
is accepted as a report directory holding results.json (diffreports.py:77;
pinned by test_report_directory_form_accepted). Fixed both wordings.
Precision notes (not bugs): `mpl` is docs shorthand, not an installed
command (only model-protocol-lab is); the 1 MB --prompt-text/--output-text
argparse bounds are unreachable via real execve on Linux (MAX_ARG_STRLEN
128 KiB) -- defense-in-depth, pinned unit-level; argparse reports a missing
command instead of "unrecognized arguments" for unknown top-level flags
(exit code still 2). Deliberately not verified: SIGINT exit-130 path;
live-model run/doctor; pip install of the console script. README test
count updated to 3933.

## 2026-10-16 (cycle 65: backend seed-determinism & cache-isolation audit)

Adversarial audit of the determinism foundation
(tests/test_cycle65_backend_determinism_audit.py, 18 tests + 39
subtests): LocalHFBackend (backends.py:728-1047), TruncatedPromptBackend
(678-726), _TokenizerBackend (diffbackends.py:830-871). Proven by test:
byte-identical (seed, input_ids, cap) outputs across independent
instances, repeats, and a fresh subprocess; greedy seed-invariance;
same-seed/different-prompts never confused; a 40-request "polluted"
instance matches a fresh instance; generate/forward_logits leave global
torch and random RNG state untouched; config.use_cache=False confirmed;
seeds 0 and 2**64-1 valid while -1, 2**64, "7", True, None fail closed
with invalid_seed; full evaluate + save_report twice -> byte-identical
results.json AND report.html (zero allowed-difference fields -- no
timestamps in the report path). Structural scan: no lru_cache/cache, no
module-level mutable containers, no per-instance cache (the only "cache"
is HF's KV-cache, disabled). Cycle-63's cited claims (backends.py:732-733,
853-859) re-verified in the current file. No genuine bugs -- CLEAN.
Deliberately not verified: cross-machine/GPU determinism; constructor-time
RNG side effects; emulate_* stop functions (pure ports, out of scope).
README test count updated to 3951.

## 2026-10-16 (cycle 66: fixture-loading adversarial audit + 2 bug fixes)

Adversarial audit of fixture loading (tests/test_cycle66_fixture_loading_audit.py,
73 tests + 1 skipped): fixtures.read_authoring_file / parse_fixture_csv
(fixtures.py:60-208) and datasets.load_dataset / parse_dataset
(datasets.py:85-138). Partition column honored exactly/case-sensitively;
NFC/NFD duplicate IDs, group overlap, message leakage rejected via Split;
CRLF/quoted fields/BOM/CRLF edge cases; all size caps enforced at exact
boundaries; FIFO/dir/dev/null/symlink/dangling/missing/null-byte inputs ->
fixed ValueErrors; UTF-16-with-BOM and latin-1 refused. TWO REAL BUGS
FOUND AND FIXED (with regression tests):
- BUG-1 (medium): parse_fixture_csv -- a single CSV field over the csv
  module's 131072-char field limit raised _csv.Error (not ValueError),
  escaping the documented contract; `fixtures build` catches only
  ValueError -> unhandled traceback. Fix: catch csv.Error around the
  parse loop, raise ValueError("CSV field exceeds the parser limit near
  row N").
- BUG-2 (low): read_authoring_file / load_dataset -- read-time OSError
  after a successful open (e.g. /proc/self/mem passes S_ISREG, read
  raises EIO) escaped as OSError -> CLI traceback. Fix: wrap the read,
  map OSError to the existing "Unreadable ..." ValueError codes.
Negative control: the 3 updated regression tests fail against the
reverted code and pass against the fixed code. Mutation gate on the
audit (4 mutants, /tmp copies) 4/4 caught. Minor doc drift noted (not
fixed): whitespace-only lines are treated as data rows ("fixture id is
blank") while USAGE.md says blank lines carry no fixture. README test
count updated to 4025.

## 2026-10-16 (cycle 67: diff-reports comparison-semantics audit + 1 bug fix)

Adversarial audit of diff-reports comparison semantics
(tests/test_cycle67_diffreports_audit.py, 44 tests + 819 subtests):
decision raw !=; winner normalization; verdict rules (first candidate-name
wins, pass iff int planned/passes + eligible + planned>0 + passes==planned);
per-fixture exact string matching with order normalization, last-wins
duplicate fixture IDs, malformed rows skipped-and-counted; config drift in
fixed order; summary dedup rules; evidence-symmetry requirement for
"identical". Method: independent oracle + 400 seeded randomized pairs x2
(800 full-dict comparisons), 13 one-field minimal pairs, 4x4
evidence-presence matrix, red-team (duplicates both directions, unicode,
case-sensitivity, deeply-nested extras). ONE REAL BUG FOUND AND FIXED
(low): diff_reports kept selected_candidate="" as-is in the winner
section while _held_verdict treats "" as absent, so two winnerless runs
compared as "winner CHANGED" with no verdict change (diffreports.py:361).
Fix: normalize "" to None in the winner extraction, matching
_held_verdict. Negative control: the pinned test + 188 fuzz subtests fail
on the reverted code, all pass fixed. The audit's oracle initially
mirrored the buggy normalization to get the fuzz green -- corrected to
the fixed behavior as part of this fix. Mutation gate 4/4 caught on /tmp
copies. Documented non-bugs: first-wins/last-wins duplicate handling,
cross-stage evidence emitting absent-diffs + set drift, flips excluding
absent legs. README test count updated to 4069.

## 2026-10-16 (cycle 68: end-to-end pipeline audit)

End-to-end audit of fixtures build -> run -> diff-reports, all via
subprocess CLI invocations with synthetic/local pieces only
(tests/test_cycle68_pipeline_e2e_audit.py, 8 tests). fixtures build:
authored 2-case CSV -> exit 0, summary counts, fixture JSON + .audit.json
sidecar, round-trip through load_dataset. run: tiny GPT-2 backend built
on the fly (no downloads) -> exit code matches the decision contract
(no_candidate -> 2), results.json byte-identical through
diffreports.load_results, evidence on every search entry, self-contained
report.html. diff-reports: self-diff (dir form vs file form) ->
identical=true, zero changes; one flipped observation -> identical=false,
fixtures_flipped=1 naming exactly the perturbed cell, all other axes
untouched. Adversarial: oversize CSV field -> exit 1 with the cycle-66
fixed error (no traceback, no fixture left behind); run with torch/
transformers stubbed out -> exit 1 local_backend_initialization_failed
(fail-closed, no traceback); report directory accepted, bare directory
rejected with fixed code. No genuine bugs -- CLEAN. Deliberately not
verified: --sensitivity-sweep / --logprob-diagnostic paths
(compute-heavy). README test count updated to 4077.

## 2026-10-16 (cycle 69: run-command diagnostics audit)

Adversarial audit of the run diagnostics (tests/test_cycle69_diagnostics_audit.py,
47 tests): budget_frontier (experiments.py:275; always-on, post-hoc, zero
model calls; recomputes per-candidate pass counts and the selection
decision at every smaller token budget), logprob_diagnostic (703;
opt-in, teacher-forced per-fixture scores, bands mean>=-2.0 derailed /
mean<-5.0 weights-don't-know), prompt_sensitivity_sweep (789; opt-in,
native + one re-run per drop depth, winner_changed = winner!=native OR
decision!=native; CLI wiring asserted structurally -- no --budget-frontier
flag exists). Method: independent hand-rolled oracles with full-payload
equality, minimal-pair tests (boundary shifted 3 tokens -> exactly +3 on
that cell), adversarial edges (budget-0-never-appears, ambiguous/
no_candidate/held_out_failed natives, identical logprobs, exact band
edges, no-op perturbation -> no false flag, UTF-8 split depths, 10
ValueError rejection cases). Mutation gate 5/5 caught on /tmp copies
(frontier off-by-one, band edge >= vs >, sweep or->and, flip == vs !=,
aggregate mean denominator). No genuine bugs -- CLEAN. Deliberately not
verified: live-model backends; subprocess mpl run; report-HTML rendering
of diagnostic sections; statistical power of the sweep. README test count
updated to 4124.

## 2026-10-16 (cycle 70: upstream drift re-check)

Re-verified all 7 tracked items against primary sources (2026-10-16 ~05:40 CEST);
nothing changed vs the 2026-10-16 ~03:00 CEST baseline. No repo changes needed.
1. SGLang PR #42466 (multi-stop earliest-occurrence): still open, not merged
   (GitHub API: state=open, merged=false, updated 2026-10-12; CI red on all
   three runs). If merged, emulate_sglang_stop tie-break becomes earliest
   occurrence in decoded text (ties keep list order).
2. transformers: latest v5.20.0 (published 2026-10-15); no post-freeze change
   to stopping_criteria.py (last touch 2026-09-18, docstring fix) -- no
   StopStringCriteria change.
3. vLLM: latest v0.31.0 (2026-10-05); no detokenization/stop-handling change.
4. SGLang: latest v0.5.21 (2026-10-02); no stop-string-logic change.
5. Mistral Large 4 (mistralai/Mistral-Large-4-1T-A52B): still "Upcoming
   release", ETA October 31, 2026 (primary HF page; a secondary source claims
   Oct 27 -- primary wins).
6. Reflection Beam: still waitlist-only, no public weights; full 501B MoE
   weights ship later in October 2026 under Apache 2.0 after red-teaming.
7. New releases in last ~30h vs 29-family catalog -- no new framing:
   Meta Muse Spark Open (2026-10-15, gated repo: chat_template.jinja returns
   401, unverifiable; repo keeps muse_spark from the cycle-52/61
   cross-check, no unverified claim made), NVIDIA Nemotron 3.5 Super VL
   120B-A12B and Bolt 74B-A9B (both <|im_start|>/<|im_end|> chatml framing,
   covered by nemotron35), Stepfun Step-5-Preview (chatml-based, covered by
   chatml), LFM2.5-24B-A2B (covered by lfm25). News sweep: no other chat-model
   releases. Next round: re-check Muse Spark Open's gated template once the
   license is accepted, and any Beam weight drop.

## 2026-10-16 (cycle 71: debug REPL deeper-paths audit)

Adversarial audit of the mpl debug REPL beyond cycle 36's command basics
(tests/test_cycle71_debug_repl_audit.py, 83 tests): independent brute-force
oracles for _overhang/first-boundary (explicit-loop implementations, 25
LCG walks + hand-computed walk), minimal-pair boundaries (16 vs 17 ids,
4096 vs 4097, 1MB vs 1MB+1 .say bytes, 10000 vs 10001 lines, 512 vs 513
display ids, 2^31-1 vs 2^31), 13-command multi-turn session with prompt
transitions, state-pollution checks (failed commands leave prior state
intact), parsing edges (uppercase, tabs, unicode, CRLF, NUL, 20k-char
input, nested .read depth-counter restoration), error recovery (exploding
tokenizers degrade, EOF mid-.say ends session -- pinned as the safe choice),
fail-closed backends (tokenizer-only loading proved with garbage weight
files; loader failures collapse to fixed codes without echoing paths),
CLI wiring. Mutation gate 6/6 killed on PID-scoped /tmp copies. No genuine
bugs -- CLEAN. README test count updated to 4207.

## 2026-10-16 (cycle 72: examine_model/doctor() integration audit)

Adversarial integration audit of doctor.py's examine_model/doctor() flow
(tests/test_cycle72_doctor_integration_audit.py, 47 tests + 35 subtests):
independent-oracle differential testing on synthetic model dirs (PID-scoped
/tmp; headers only, no weight execution) covering check ordering, early-return
branches, fail-fast vs continue semantics per stage, verdict-precedence
minimal pairs (one-corrupt-shard vs all-corrupt, missing vs malformed config,
unsupported-type vs rejected-keys precedence), context-window degradation,
boundary_ids/map validation, portability informational semantics, dataset
verdict propagation, suggestion gating, determinism/JSON-serializability,
and adversarial edges (empty dir, 0-byte/garbage files, JSON-list header,
128 MiB+1 config, unicode paths, symlinked dirs, symlink loops,
both-layouts-present, bad weight indexes, adapters). Mutation gate 6/6
killed on /tmp copies. 1 GENUINE BUG FOUND AND FIXED: when _manifest()
raised ValueError (TOCTOU asset race), examine_model recorded
asset_manifest: failed but silently dropped the completed asset_layout
stage -- fixed by recording asset_layout: ok with the standard "N assets"
detail in that branch (doctor.py, +3 lines, matching the code's own
convention); regression test fails on pre-fix source, passes post-fix. Test-harness fix
(post-audit): test_oversized_config_rejected wrote a real 128 MiB file to
the 512 MiB /tmp tmpfs and flaked once in three full-suite runs (ENOSPC
signature -- the cycle-60 failure mode); replaced with a sparse file via
truncate(), which exercises the identical _read_json oversized path
(holes read as zeros) while consuming ~0 blocks -- 5/5 green with /tmp
at 75% full.
README test count updated to 4254.

## 2026-10-16 (cycle 73: diff-backends comparison-semantics audit)

Adversarial audit of diffbackends.py's comparison layer below the emulate_*
functions (tests/test_cycle73_diffbackends_audit.py, 70 tests + 327
subtests): all four emulate_* entry points stubbed at the module boundary,
so every cell is a fully synthetic (available, fired, detail,
truncated_ids, returned_text) tuple. Independent brute-force oracle
re-derives every (class, stop_text) verdict from the docstring rules;
minimal pairs flip one input fact per verdict; adversarial edges
(fired-but-no-step, malformed truncated_ids, unavailable mixes, decode
failures, cap-boundary id lists, unicode stops); fail-closed validation
vocabulary pinned (ValueError with fixed codes, never tracebacks);
300-pair fuzz vs oracle -- identical verdict sets; CLI wiring incl.
sanitized loader/encode failure paths. Mutation gate 6/6 killed on
PID-scoped /tmp copies. No genuine bugs -- CLEAN. README test count
updated to 4324.

## 2026-10-16 (cycle 74: catalog render-correctness audit)

First adversarial audit of the 29-family prompt-format catalog's hand
renders themselves (tests/test_cycle74_catalog_render_audit.py, 1149
tests): structural invariants over all 29 families x all
candidates(mutate=True) variants (118 renders) -- render-twice
determinism, non-empty output, distinct user/assistant markers, no
None/placeholder leakage, exact one-turn mechanics rebuilt from field
values, 3-turn alternation, unicode preservation; variant inventory pin
(118 total, exact branch-variant names for the 9 generation-prompt-branch
families, base-first ordering). Primary-source cross-checks for 6 families
(fetched 2026-10-16, URLs + template lines pinned in the test docstring):
llama3 (llama-models prompt_format.md) byte-exact; gemma
(ai.google.dev/gemma/docs/formatting) byte-exact; qwen3 (Qwen3-8B
tokenizer_config.json) byte-exact; deepseek (DeepSeek-V3
tokenizer_config.json, incl. U+FF5C/U+2581) byte-exact; mistral v0.3
version-pin coherence (catalog pins the 2024-05-22 release form; current
Hub file exhibits exactly the rewrite the catalog comment describes);
hy4 (tencent/Hy4-preview chat_template.jinja + the locally preserved
byte-identical template executed under stock Jinja2 -- output minus the
documented synthesized system block is byte-identical to the catalog
render, incl. the no-think variant). Mutation gate 6/6 killed on /tmp
copies. No genuine mismatches -- CLEAN. README test count updated to 5473.

## 2026-10-16 (cycle 75: reporting determinism + section-fidelity audit)

Adversarial audit of reporting.py's report generation
(tests/test_cycle75_reporting_determinism_audit.py, 26 tests + 58
subtests): AST-based static nondeterminism scan (no timestamps/random/
uuid/id()/hash-order iteration; the two set literals are safe); 10
synthetic fixtures rendered 3x in-process plus fresh subprocesses with
PYTHONHASHSEED in {0,42,987654321,7} -- byte-identical HTML+JSON every
time; independent hand-derived oracles for selection_margin (all 5
branches incl. runner-up tie lexicographic pin), token_overhead (sorted
key order), TOC anchors, fingerprint truncation, all report sections,
evidence stripping and ASCII-only JSON in results.json, and the
allowlisted-script CSP hash; adversarial edges (empty result, hostile
<script>/unicode payloads exactly escaped, 1200-fixture bulk render,
contradictory fields pinned, no caller mutation). Mutation gate 6/6
killed on PID-scoped /tmp copies (incl. an injected time.time() caught
by the determinism check). No genuine bugs -- CLEAN. README test count
updated to 5499.
