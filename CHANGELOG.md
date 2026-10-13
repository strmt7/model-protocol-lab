# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `run` stdout now names the written artifacts: the final JSON line is
  `{"decision": ..., "selected_candidate": ..., "results": "<dir>/results.json",
  "report": "<dir>/report.html"}` so a first run points the operator at the
  report instead of leaving them guessing where it went (consistent with
  `fixtures build`, which already reports its output path).

- `doctor` suggestion now sizes `--max-new-tokens` to the checkpoint's
  declared window (`doctor._longest_prompt_tokens`): the longest fixture
  prompt rendered with every catalog candidate is encoded with the
  checkpoint tokenizer, and the suggested budget is the largest value (up
  to 64) that still fits prompt + budget within the window. When even the
  longest prompt fills the window, the suggestion keeps `--max-new-tokens 1`
  and warns to shorten the fixtures or use a larger-window checkpoint.
  Without a measurable tokenizer the suggestion keeps the default 64.
- Failed-run diagnosis: `run_experiment` summaries carry the sanitized
  `failure_code`, and the report's decision banner names it with operator
  guidance ("Failure diagnosis"). `core.evaluate` now passes through the
  built-in backend's fixed-vocabulary error codes (`context_limit_exceeded`
  and friends) via an allowlist; arbitrary backend text still collapses to
  `backend_reported_error`, so the sanitization boundary holds.
- Per-candidate response boundaries (`run --boundary-map NAME=ID,ID`,
  repeatable; `doctor --boundary-map` for pre-flight vocab checks):
  each format hypothesis is measured with its own end-of-turn marker
  instead of one sequence for all candidates, with `--boundary-ids` as
  the default. The selection rule is unchanged (exactly one candidate
  passing every search fixture under its own boundary; ties and failures
  abstain); the held-out stage, stop matrix and budget frontier use each
  candidate's own boundary, and every search entry records its
  `boundary_token_ids`. Unknown or duplicate candidate names fail fast.
- `fixtures build` (`model_protocol_lab/fixtures.py`): compile a compact
  authoring CSV (one row per fixture: `id,group,partition,expected` plus
  `user_1,assistant_1,...` message columns) into the strict fixture JSON.
  The search/held-out discipline (ID uniqueness, group non-overlap,
  identical-message leakage, NFC-normalized) is enforced at build time
  with row-numbered errors; the emitted JSON is round-tripped through the
  strict loader before it is returned, so a built file satisfies the
  schema by construction. Output is never overwritten and is written
  owner-only (0o600); `examples/fixtures.csv` builds to the equivalent of
  `examples/fixtures.json`.
- Richer report views (still a single file, no scripts or remote assets):
  a per-fixture x candidate outcome grid (✓/✗/– with the selected
  candidate starred), decisive fixtures (fixtures passed by exactly one
  candidate while all others failed -- the evidence the verdict rests
  on), per-candidate token-budget pass-curve sparklines in the frontier
  section, and a boundary-IDs column in the candidate table.
- Byte-space HF stop-string matching (`backends._hf_byte_token_mapper`,
  `LocalHFBackend.hf_stop_token_bytes`, threaded through
  `emulate_hf_stop_string`/`stop_matrix`/CLI): for byte-level and
  byte-fallback decoders the HF emulation now matches on raw token bytes
  with the stop string UTF-8-encoded, mirroring current transformers (the
  byte-space fix for upstream issue #46519); tokens the byte table cannot
  represent fall back to the isolated decode. Differentially verified
  against the real `StopStringCriteria`, including the byte-fragment shape
  where the old decode-string path stayed silent and the oracle fires.
- Token-budget robustness frontier (`experiments.budget_frontier`, always-on
  `run` diagnostic, new HTML section): recomputes the search-stage decision
  at every smaller token budget from the recorded outputs -- exact under
  greedy decoding, no extra model calls -- reporting per-candidate minimal
  passing budgets, the pass-count step curve, and the budget ranges where
  the decision or winner would have differed (`flip_ranges`,
  `budget_robust`).
- `doctor` next-command suggestion: a clean `doctor` with `--dataset` and
  `--boundary-ids` now carries a `suggestion` with the exact `run`
  invocation, `--max-context` filled from the checkpoint's declared window
  and paths shell-quoted; the format family stays an explicit placeholder.
  `examine_model` also reports `model.context_window`.
- `docs/FAQ.md`: troubleshooting and interpretation Q&A grounded in
  documented behavior, linked from README and `docs/USAGE.md`.
- Template-conformance diff (`core.template_diff`, `doctor.template_conformance`,
  surfaced in `run` reports): each hand-rendered hypothesis is compared as an
  exact token-ID sequence against the checkpoint tokenizer's own
  `apply_chat_template` rendering of the same messages, reporting the divergent
  token-ID subsequences and their positions (empty diff when byte-identical);
  "no template declared" is reported gracefully and never fails the run.
- Failure-mode taxonomy (`experiments.classify_observation`/`failure_taxonomy`,
  cited in the report's decision banner): every non-passing observation is
  classified into the closed set `no-boundary-in-N`, `boundary-text-mismatch`,
  `eos-without-boundary`, `held-out-divergence`, `backend-error`, with
  per-candidate counts; `run_experiment` accepts an optional `eos_id`
  (the CLI takes it from the checkpoint tokenizer).
- Stop-behavior isolation matrix (`experiments.stop_matrix`, emulations in
  `backends.py`, surfaced in `run` reports): the frozen winner's recorded
  outputs are re-measured under emulated HF `StopStringCriteria` overhang
  matching, vLLM detokenizer earliest-completion truncation, and EOS-appended
  stop semantics, reporting per-fixture pass/fail and which semantics flip the
  verdict. Emulations are post-hoc truncations (exact for greedy decoding).
- Cross-tokenizer boundary portability (`core.check_boundary_portability`,
  `doctor --reference-model`): the boundary marker text is re-encoded with a
  second checkpoint's tokenizer and reported portable/non-portable with both
  ID lists; non-portability is informational (`mismatch`), never a failure.
- Prompt-boundary sensitivity sweep (`experiments.prompt_sensitivity_sweep`,
  `run --sensitivity-sweep`/`--max-drop-bytes`, `backends.TruncatedPromptBackend`,
  `core.truncate_text_tail`): re-runs search with each prompt truncated
  mid-token at every depth, reporting the depths where the winner changes.
- `doctor` CLI command: pre-flight validation of checkpoint assets and fixture
  files without loading model weights (`model_protocol_lab/doctor.py`).
- `Observation.boundary_text_match`: diagnostic cross-check recording whether
  the decoded output ends with the decoded boundary text, flagging the known
  tokenization-mismatch false-negative mode. Never affects the pass criterion.
- `candidate_fingerprint` in search-stage results (sha256 over the canonical
  format JSON), surfaced in the HTML report.
- Redesigned HTML report: decision banner with per-decision explanation, run
  configuration section, pass-rate heatmap with bars and text labels, held-out
  stat cards, WCAG AA contrast (minimum 6.81:1), responsive layout, print
  stylesheet; still a single file with no external assets and a strict CSP.
- Committed synthetic sample report (`examples/sample-report/`) with a
  byte-identical regeneration golden test.
- CI workflow (`.github/workflows/ci.yml`): lint + typecheck on 3.12 without
  optional deps, full tests on 3.11/3.12/3.13 with the `hf` extra, dependency
  and static security audit (`pip-audit`, `bandit`), packaging build check;
  weekly scheduled run. Actions pinned to SHAs.
- Pre-commit config, Dependabot config (pip, github-actions, pre-commit),
  `CITATION.cff`, `CONTRIBUTING.md`, issue and PR templates.
- Property-based tests (hypothesis) for dataset loading, fixture separation,
  boundary parsing, and adversarial JSON inputs.

### Changed

- `pyproject.toml`: PEP 639 license metadata (`license-files`), dev/hf extras,
  ruff/mypy/coverage configuration (ruff `S` rules enabled).
- Report outputs now use restrictive permissions (directory `0o700`, files
  `0o600`, exclusive creation).
- `inspect` reports the fixture validation reason instead of a generic error.
- CLI surfaces sanitized error codes (`ValueError` vocabulary, `local_*`
  backend codes); `--boundary-ids` accepts ASCII digits only.
- Dataset errors identify the fixture (`search[i]`/`held_out[i]`);
  deeply-nested JSON, non-UTF-8, and char/block-device fixtures raise
  `ValueError` in the module's error vocabulary.
- README status: "unexecuted alpha" → tested state with evidence pointers; new
  "How this differs" section grounded in primary sources.

### Fixed (third hardening round, red-team, 2026-10-13)

- **Red-team round 3 — HF stop emulation:** `_hf_overhang_match` missed the
  tensor implementation's end-overlap class where the stop string ends
  *inside* the final token (e.g. token `" world"`, stop `"or"`); the stop
  matrix systematically under-reported HF stops. Now mirrors
  `_stop_string_get_matching_positions` exactly — differentially verified
  against the real `transformers.StopStringCriteria` (1500/1500 random
  trials on byte-level BPE, exhaustive 256-combination scan, zero
  divergences). Known residual: per-token `decode` strings vs HF's cleaned
  vocab strings / raw bytes can diverge on UTF-8-fragment tokens
  (documented in `docs/USAGE.md` §8.2).
- **Red-team round 3 — `template_diff` algorithmic DoS:** adversarially
  fragmented inputs made difflib super-quadratic (~45 s at n=2000, ~cubic
  scaling). The matching-blocks loop now runs under a 10 s wall-clock
  budget and degrades to `ValueError("template_diff_too_fragmented")`
  (the conformance check marks the pair `"unavailable"`); output is
  byte-identical to unbounded difflib on ordinary inputs.
- **Red-team round 3 — uncapped tokenizer loads in `doctor`:** the
  reference-model portability check and `template_conformance` loaded
  `tokenizer.json` via HF with no 128 MB pre-load cap (the round-1 fix
  covered only `LocalHFBackend`). New `_tokenizer_file()` helper enforces
  the cap at every loader call site; oversized files degrade to the fixed
  `"skipped"`/`"unavailable"` codes without parsing.
- **Red-team round 3 — hostile Jinja chat templates:** a checkpoint
  template rendering unbounded text (`{{ 'x' * 10**8 }}`) OOM-killed the
  conformance worker. Template source and rendered output are each capped
  at 1 MB chars, degrading to `"unavailable"` (`"chat_template_too_large"`);
  CPU burn inside the Jinja render itself remains the same exposure as HF's
  own `apply_chat_template` (documented residual).
- **Round-3 test hygiene:** the new `test_round3.py` broke the `[dev]`-only
  suite (7 classes imported the HF stack at test time with no skip guard);
  added the `HAS_HF` module guard so the README quick-start path is
  skip-clean again.
- 9 new regression tests (361 total, all passing, 98% line coverage).

### Fixed (third hardening round, audit, 2026-10-13)

- **Audit:** `doctor`'s config-read failure path returned a non-uniform report
  shape (missing `assets`/`special_tokens`/`added_tokens`) — now uses the same
  `_model_result` wrapper as every other early failure, with the already
  computed asset manifest attached; `_read_json` normalizes raw
  `UnicodeDecodeError`/`JSONDecodeError`/`RecursionError` from inspected asset
  files to the fixed `invalid_asset_json` code instead of leaking raw
  codec/parser text into doctor check details (mirrors
  `datasets.parse_dataset`); `run_experiment` rejects a backend missing the
  `fingerprint` attribute with `ValueError` instead of a raw `AttributeError`;
  `evaluate` compares NFC-normalized fixture IDs for uniqueness, closing the
  NFD/NFC evasion gap that round 2 fixed only in `Split`;
  `prepare_report_dir` never chmods through a symlinked path component
  (`os.chmod` follows links — the old code could tighten permissions on a
  directory outside the requested tree).
- 7 new regression tests (231 total, all passing, 97% line coverage held).

### Fixed (second hardening round, 2026-10-13)

- **Audit:** stock GPT-2 checkpoints failed with the CLI default
  `--max-context 2048` and the cause was swallowed into the generic init
  failure — now fails fast with the distinct, actionable code
  `local_context_limit_exceeds_declared_window` (documented in
  `docs/USAGE.md`); `run_experiment` validates `boundary`/`cancel`/backend
  fingerprint types up front instead of raw `TypeError`s; `evaluate` rejects
  non-`Event` `cancel`; `doctor` validates `boundary_ids` element types and
  returns a uniform report shape (always with `assets`/`special_tokens`/
  `added_tokens`) on early failures; `load_dataset` rejects non-path input
  with `ValueError`; corrected the false `candidate_fingerprint` docstring
  (the name is part of the payload).
- **Red-team round 2 (2026-10-13):** NFD/NFC variants no longer evade the
  fixture-ID uniqueness or cross-split leakage checks (`Split` compares
  NFC-normalized keys; stored data stays byte-exact); terminal escape
  sequences stripped from fixture IDs in `inspect`/`run` stderr warnings.
- **Docs:** the 3 hf-dependent tests that failed under a `[dev]`-only
  install now skip cleanly, so the README quick-start path is green without
  the heavy `hf` extra; `docs/USAGE.md` documents the GPT-2 context-window
  requirement and NFC-normalization behavior.
- 11 new regression tests (224 total, all passing, 97% line coverage held).

### Fixed

- `LocalHFBackend._directory`: `FileNotFoundError` and symlink loops now
  surface as `ValueError("invalid_local_directory")` instead of leaking OS
  error text.
- `evaluate()`: per-attempt helper closures no longer risk stale loop-variable
  capture (state passed explicitly).
- Tokenizer file size is re-checked before the HF loader parses it
  (defense-in-depth for the 128 MB JSON cap).
- **Red-team round (2026-10-12):** named-FIFO hang in `load_dataset` (now
  refuses all non-regular files via `O_NONBLOCK` + fd `fstat`); output
  directory now claimed before backend init and released on failure;
  `candidate_fingerprint` normalized to match `local_repairs` dedup;
  report parent dirs all 0o700; `doctor` tokenizer-deep failures sanitized;
  empty-`expected` fixtures warn in `inspect`/`run`.

## [0.1.0a0] — 2026-10-12

Initial source snapshot: renderer, evaluator, catalog, local HF adapter,
experiment orchestration, reports, CLI, fixture schema, docs and test source.
Unexecuted at release of the snapshot.
