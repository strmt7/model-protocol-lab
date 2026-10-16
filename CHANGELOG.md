# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (cycle 104 — mypy strict compliance, 2026-10-16)

- The full package now passes `mypy --strict` (37 errors fixed across 7
  files): type annotations added to the untyped dataset helpers
  (`_unique_object`, `_reject_constant`, `_fields`, `_text`, `_fixture`),
  `experiments._failure_code` / `_summary` (now typed on
  `EvaluationReport`), `budget_frontier` (corrected to
  `list[dict[str, Any]]` — entries are search-record dicts, not Fixtures;
  the wrong first attempt tripped `comparison-overlap`), bare `dict` /
  `list` / `tuple` generics given type arguments, `cli.main` argv typed,
  and the five `PreTrainedTokenizerFast` call sites (which mypy resolves
  to the untyped `TokenizersBackend` factory) marked with deliberate
  `type: ignore[no-untyped-call]` at the documented untyped
  third-party boundary. `strict = true` is now set in pyproject.toml so
  the CI lint job enforces it.
- `TestMypyStrictPinned` in `tests/test_ci_portability_guard.py` fails
  if `strict` is ever flipped back to false. Test count 6122 -> 6123.

### Added (cycle 103 — public API docstring completion, 2026-10-16)

- All 16 public classes/functions in the shipped package that lacked
  docstrings now have them: the seven core data-model dataclasses
  (`Message`, `Format`, `Fixture`, `Split`, `Generation`, `Observation`,
  `EvaluationReport`), `catalog.candidates`, the CLI entry points
  (`parse_boundary`, `build_parser`, `main`), `datasets.parse_dataset`
  / `load_dataset`, `experiments.run_experiment`, and
  `reporting.render_html` / `save_report`. Public docstring coverage is
  now 74/74.
- `TestPublicApiDocstrings` in `tests/test_ci_portability_guard.py`:
  AST-scans the package and fails on any public top-level def/class
  without a docstring, so the coverage cannot regress. Test count
  6121 -> 6122.

### Added (cycle 102 — true OSError-branch coverage, 2026-10-16)

- `tests/test_cycle102_oserror_branches.py` (5 tests): the
  ``except OSError`` branches in ``examine_model`` are defense-in-depth
  for I/O errors *after* a successful open (disk errors during
  ``read()``) — ``_open_regular_read`` converts open-time failures to
  ``ValueError`` first, so they were uncovered. A tracked-fd
  ``os.fdopen`` replacement injects ``OSError`` at read time for a named
  file, pinning fail-closed behavior for the `_model_assets`,
  `_manifest`, config, and tokenizer branches (fixed
  ``missing_or_unsafe_asset`` vocabulary, never a raw OSError).
- Fixed misleading docstrings on the four cycle-92 "without_root"
  tests: they exercise the ``ValueError`` conversion path, not the raw
  ``except OSError`` branch — now stated accurately with a pointer to
  the new coverage. Test count 6116 -> 6121.

### Added (cycle 101 — warning-cleanliness audit, 2026-10-16)

- `tests/test_cycle101_warning_cleanliness.py` (4 tests): pins the
  verified property that the package emits zero warnings — all module
  imports, a `doctor` smoke run, a report render, and CLI startup each
  run warn-free under `python -W error` in a subprocess. Full-suite
  audit (`python -W error -m pytest`, 6105 tests) passes with every
  warning promoted to an error: no DeprecationWarning, FutureWarning,
  ResourceWarning, or any other warning fires anywhere. Test count
  6112 -> 6116.

### Added (cycle 100 — HTML report injection audit, 2026-10-16)

- `tests/test_cycle100_html_injection_audit.py` (9 tests): hostile
  payload battery (`<script>`, event handlers, `<svg>`, `javascript:`,
  title-breakout, iframe, U+2028/29) through every `render_html` path —
  no raw payload survives, escaped forms present, no raw angle brackets
  in text content, inline script has no `</script` breakout, CSP meta
  present, hostile decision maps to neutral CSS class, output parses as
  well-formed HTML. Finding: no vulnerabilities — every interpolation is
  escaped (directly or via an outer `escape()`), the only `href` uses
  fixed-literal anchors, and there are no data-driven URL attributes.
  Test count 6103 -> 6112.

### Fixed (cycle 99 — stale development plan framing, 2026-10-16)

- `DEVELOPMENT_PLAN.md` still opened by framing the lab as a "proposed"
  tool with "no runtime validation established" — contradicting the 0.1.0
  release-ready state (and the owner's no-"early development" rule). Added
  a dated status note recording substantial implementation (6100+ tests,
  green CI, release-readiness review) while preserving the design sections
  as the authoritative architecture reference. Pinned by 3 regression
  tests in `tests/test_cycle99_plan_currency.py`. Test count 6100 -> 6103.

### Added (cycle 98 — property-based core algorithm audit, 2026-10-16)

- `tests/test_cycle98_boundary_properties.py` (7 Hypothesis property
  tests): universal properties of `first_boundary` — found index is an
  exact match, None iff absent, result is the *first* occurrence, index
  in range, empty stop raises, overlong stop returns None, embedded stop
  found at or before the known position. Thousands of generated cases;
  no counterexamples. Test count 6093 -> 6100.

### Added (cycle 97 — hostile-environment robustness audit, 2026-10-16)

- `tests/test_cycle97_hostile_environment.py` (10 tests): hostile
  *environment* audit of the write paths — the red-team sweeps covered
  hostile inputs; this covers adverse filesystem/OS conditions.
  `prepare_report_dir` / `save_report` under mocked PermissionError
  (clean OSError, no partial dirs/files, half-built pair cleanup),
  concurrent directory claims (exactly one wins via O_EXCL), existing
  file/non-empty dir rejection, and `write_fixture_file` permission and
  no-clobber behavior. All write paths confirmed robust; no code changes
  needed. Test count 6083 -> 6093.

## [0.1.0] — 2026-10-16

### Added (cycles 14–15, 52, 54 — catalog to 29 families, 2026-10-15)

- Format catalog expanded 23 -> 29 families (all pinned by byte-exact
  tests in `tests/test_cycle14_*`, `tests/test_cycle15_*`,
  `tests/test_cycle52_muse_spark.py`, `tests/test_cycle54_qwen38.py`):
  `k2horizon` (cycle 14, K2-Horizon), `hy4` (Tencent Hy4 preview),
  `minimax3` (MiniMax-M3), `nanbeige42` (Nanbeige 4.2-3B; all three cycle
  15, re-verified against primary sources this round -- official
  chat_template.jinja / tokenizer_config.json preserved byte-identical
  under `docs/evidence/`), `muse_spark` (cycle 52, Meta Muse Spark Open),
  `qwen38` (cycle 54, Qwen3.8-27B, thinking-on default opening the
  generation prompt with `<think>`). The `doctor` portability matrix now
  renders 118 mutate candidates (110 -> 114 -> 118). Evidence tiers and
  out-of-scope branches are documented in the catalog comments and
  `docs/RESEARCH_LOG.md`.

### Added (cycles 45–83 — audit record, 2026-10-15 → 2026-10-16)

- Audit cycles 45–51, 55–65, 68–75, 77–83 (see
  `docs/RESEARCH_LOG.md`): differential re-verifications of
  run_experiment, docs command examples, the sample-report pipeline,
  canary evidence-leak sweeps, logprob numerics, HF stop-emulation,
  core.evaluate, the prompt-sensitivity sweep, budget frontier, HTML
  escaping, boundary portability, harness concurrency, doctor
  portability-matrix semantics, search/held-out separation, the CLI
  surface, backend seed-determinism, LocalHFBackend lifecycle, reporting
  determinism, classification/failure taxonomy, run orchestration, the
  core.evaluate engine, and the special-token auditor. Upstream drift
  re-checks (cycles 51, 61, 70): no change. Cycles with real findings are
  under Fixed; the rest are CLEAN with mutation gates run on hermetic
  /tmp copies.
- `tests/test_ci_portability_guard.py`: standing AST guard forbidding
  exact `sys.version_info` tuple comparisons and bare torch/transformers
  `__version__` string equality anywhere in the test suite (the failure
  class behind the cycles-79/80 CI red).
- Cycle-83 CLI wiring audit (`tests/test_cycle83_cli_wiring_audit.py`,
  63 tests): the exit-code contract is pinned by subprocess -- 0/1/2/130
  for compatible/failed/cancelled/ambiguous/held_out_failed/
  no_candidate run decisions, bad subcommands, missing args, invalid
  enum values, unreadable inputs, unwritable outputs, malformed fixtures,
  nonexistent checkpoints, contradictory ranges -- plus `--help`
  byte-stability across runs for every subcommand (fixed COLUMNS, no
  PID/tmpdir/timestamp leaks) and console-script equivalence with
  `python -m model_protocol_lab`.
- Cycle-82 fixture serialization round-trip audit
  (`tests/test_cycle82_fixture_serialization_audit.py`, 80 tests):
  `fixtures.split_to_json_text` round-trip fidelity over adversarial
  Splits, `write_fixture_file` exclusive creation (0o600, partial-write
  cleanup), and the fixed dataset rejection vocabulary. CLEAN, no source
  changes.

### Fixed (cycles 36–83 — real bugs and robustness, 2026-10-15 → 2026-10-16)

- **REAL BUG, CI-red root cause (cycles 79/80):** `examine_model`
  caught only `ValueError` around config/tokenizer reads, so an
  unreadable asset (`OSError`, e.g. `PermissionError` on
  `path.open("rb")`) escaped instead of failing closed. `doctor.py` now
  catches `OSError` alongside `ValueError` in all four blocks, mapping
  to the fixed `missing_or_unsafe_asset` check code. Four
  root-independent regression tests mock `Path.open` to raise
  `PermissionError` (config/tokenizer/manifest/weight-index blocks);
  negative control: all four fail on reverted source. Sibling fixes in
  the same round: the Python-version test now checks the
  `requires-python` range (not `(3, 12)`), the torch test compares the
  base version (so CI's CUDA wheel passes), the transformers test checks
  membership in the `hf` pin range (not `"5.19.0"`), and the bandit test
  skips when no `bandit` binary is on PATH (it only exists in the
  security-audit job).
- **REAL BUG (cycle 76, scoring):** `_f32` in `scoring.py` raised
  `OverflowError` for finite float64 log-probabilities outside the
  float32 range (legitimate from fp32-range logits via max-shift
  log-sum-exp), collapsing through `score_target_logprobs` to
  `unavailable`/`forward_failed` and mislabeling a computable score.
  `_f32` now saturates finite out-of-range values to +/-inf, matching
  IEEE-754 round-to-nearest-even and torch/numpy float32 conversion
  (`+_FLT_MAX` constant); the old `test_out_of_float32_range_raises` pin
  was updated to the corrected saturation behavior. Pinned by
  `tests/test_cycle76_scoring_semantics_audit.py` (27 tests + 75
  subtests; negative control: 12 failures on reverted code).
- **REAL BUG (cycle 67, diff-reports):** two winnerless runs compared
  as "winner CHANGED" because `diff_reports` kept
  `selected_candidate=""` while `_held_verdict` treats `""` as absent.
  Winner extraction now normalizes `""` to `None`
  (`diffreports.py:361`). Pinned by `tests/test_cycle67_diffreports_audit.py`
  (44 tests + 819 subtests; negative control: the pinned test + 188 fuzz
  subtests fail on reverted code).
- **REAL BUG (cycle 66, fixtures):** a single CSV field over the csv
  module's 131072-char field limit raised `csv.Error`, escaping the
  documented `ValueError` contract and giving `fixtures build` an
  unhandled traceback; read-time `OSError` after a successful open
  (e.g. `/proc/self/mem`) escaped as `OSError` to the CLI. `csv.Error`
  is now caught around the parse loop and raised as
  `ValueError("CSV field exceeds the parser limit near row N")`; reads
  wrap the read call and map `OSError` to the existing
  `"Unreadable ..."` codes. Pinned by
  `tests/test_cycle66_fixture_loading_audit.py`.
- **REAL BUG (cycle 36, debug REPL):** truncated transcripts could never
  be saved -- truncation kept `_MAX_SAVE_BYTES` of content then
  prepended the 29-char marker, pushing the total back over the bound so
  `.save` always refused. The marker is now budgeted inside the
  truncation (`debug.py`). Pinned by
  `TestTranscript.test_truncated_transcript_save_succeeds`.
- ROBUSTNESS (cycle 39, catalog): `candidates()` type-gates non-str
  family keys before the `CATALOG` lookup, so the documented
  `ValueError("Unknown format family")` holds for unhashable and non-str
  keys instead of leaking `TypeError`. Pinned by
  `tests/test_cycle39_catalog_integrity.py`.

### Fixed (cycles 84–93 — real bugs and CI robustness, 2026-10-16)

- **CI RED ROOT CAUSE, PROVEN (cycle 93):** the Python 3.13-only test-job
  failures on cycles 87, 88 and 92 were *not* the docs-coherence test --
  the CI job log names `tests/test_cycle86_serialization_properties.py`:
  its module docstring contained a real lone-surrogate character (a
  single-backslash `'\ud800'` escape in docstring prose, meant as literal
  text). CPython 3.13's `compile()` encodes `__doc__` as strict UTF-8,
  so the module fails at import time with `UnicodeEncodeError`, while
  3.11/3.12 accept it (which masked it locally). Surrogate *constants*
  in function bodies remain fine on 3.13 -- only docstrings are
  affected. Fixed by doubling the backslash so the docstring renders the
  escape as text (zero runtime change). `tests/test_ci_portability_guard.py`
  gains `TestNoDocstringSurrogates`, an AST guard that fails any test
  module whose module/class/function docstring holds a lone surrogate.
- **TEST ROBUSTNESS (cycle 94):** `test_template_diff_long_equal_prefix_stays_fast`
  asserted a 5.0s wall-clock bound calibrated for uninstrumented runs,
  but the CI gate runs under `coverage run` (measured 3.5x overhead:
  1.47s -> 5.10s for the same input), so it failed spuriously under
  coverage. Threshold raised to 15.0s with a comment documenting the
  overhead; still catches algorithmic blowup (minutes-scale).

- **REAL BUG (cycle 86, fixtures):** `split_to_json_text` raised a bare
  `UnicodeEncodeError` for lone surrogates (which this hypothesis
  version's `st.characters()` does emit) instead of the documented
  `ValueError`. Now fails closed with `ValueError("... contains a lone
  surrogate ...")`. Pinned by
  `tests/test_cycle86_serialization_properties.py`.
- **REAL BUGS (cycle 89, signal handling):** (1) `save_report` left a
  half-written `results.json` when `KeyboardInterrupt` landed mid-write;
  now unlinks on `BaseException` per file and removes the pair if the
  second write fails. (2) SIGTERM during `run` used the default
  disposition (exit -15, claimed output dir leaked); now wired to the
  same cooperative cancel `Event` as SIGINT, draining to a valid
  "cancelled" report and exit 130. Pinned by
  `tests/test_cycle89_signal_handling_audit.py`.
- **REAL BUG (cycle 90, cancellation):** `logprob_diagnostic` ignored the
  cancel `Event`, running its full forward-pass loop even with the event
  set. Now accepts an optional `cancel` event checked per fixture,
  returning `available=False` / `detail="cancelled"`. Pinned by
  `tests/test_cycle90_cancel_responsiveness_audit.py`.
- **CI ROBUSTNESS (cycle 92):** the real-model signal tests
  (`RunSignalTests`) signalled a child process on a timing guess: the
  "100%" stderr readiness marker they polled for is never printed by
  transformers for local checkpoint loads, so the signal could land
  before the child's signal handlers were installed (SIGTERM -> -15
  instead of 130 on slow CI CPUs). `cli.run` now writes a
  `MODEL_PROTOCOL_LAB_TEST_SIGNAL_READY_FILE` marker immediately after
  installing its handlers (test-only, env-gated, zero production
  impact); the tests wait for the marker deterministically. The two
  slow real-model tests are gated behind `MPL_RUN_SLOW_SIGNAL_TESTS=1`
  (skipped in the default CI gate); the deterministic in-process signal
  tests stay in the gate. Also removed import-time `os.chdir` /
  `sys.path` mutation from `test_cycle87_docs_coherence_audit.py`
  (import-order hazard during unittest discovery).
- Cycles 84, 85, 87, 88, 91: CLEAN audits (release versioning,
  README quick-start verification, docs coherence, hostile-input
  fuzzing, hostile backend-output probing) with pinning tests, no
  source changes required.

### Added

- `mpl diff-reports OLD_RESULTS NEW_RESULTS`: deterministic comparison of
  two experiment result directories (or two `results.json` paths) for the
  iterate-and-compare operator loop. Reports the decision flip, the winner
  change, per-candidate verdict flips on search and held-out, per-fixture
  outcome diffs (changed-only by default, `--all` lists every compared
  fixture), and a loud config-drift warning section when the runs used
  different candidate families, candidate sets, per-candidate boundaries,
  `--max-new-tokens`, backend fingerprints, or fixture sets. JSON goes to
  stdout, a human-readable summary to stderr; identical runs report "no
  differences". Per-fixture comparison needs `--include-evidence` on both
  runs and degrades loudly otherwise. Exit 0 on a successful comparison
  even when differences are found. Hostile inputs (oversized, malformed,
  non-regular files, duplicate JSON fields) collapse to fixed error codes
  with no tracebacks. See `docs/USAGE.md` section 12.
- Format catalog expanded to 23 families: `ling` (inclusionAI
  Ling-3.1-flash, released 2026-10-15, MIT; evidence tier 1 -- the official
  chat_template.jinja "Bailing V3", preserved at
  docs/evidence/ling31-chat-template.jinja, derived line-by-line in the
  catalog comment). The plain-text no-tools core: an always-emitted system
  block (documented, not modeled -- the lab's Format renders user/assistant
  turns only), user turns `<role>HUMAN</role>{content}<|role_end|>`,
  assistant history turns `<role>ASSISTANT</role>\n<think></think>{content}
  <|role_end|>`, and -- thinking defaults ON (the lab models the default;
  the thinking_option dependence is documented) -- a generation prompt
  `<role>ASSISTANT</role>\n<think>`. BOS `<|startoftext|>` is tier-1
  verified from the official tokenizer_config.json bos_token field; the
  template itself emits no BOS literal and add_bos_token is false. EOS
  caveat documented, not modeled: the official eos_token field is
  `<|role_end|>`, NOT `<|endoftext|>` (that token is the pad_token and a
  special added token). The thinking-off branch (generation prompt
  prefills an empty `<think></think>` block) rides the `--mutate` path as
  `ling/thinking-off` (kolibri/thinking-off convention). Tool-call
  branches, reasoning-content branches, and the multi-step tool scan are
  out of scope for the plain-text core. `tests/test_cycle13_families.py`
  pins byte-exact renders, the thinking-off mutation, the dedup-key
  no-collision, diffs-vs-all-other-families, and the README naming/count
  (22 -> 23).
- Format catalog expanded to 22 families: `nemotron35` (NVIDIA Nemotron 3.5
  Lightning, released 2026-08-11; evidence tier 2 -- official HF model card,
  NeMo SFT docs, community osaurus renderer), `muse_glimmer` (Meta Muse
  Glimmer 30B, released 2026-08-10, Apache 2.0; evidence tier 2 -- official
  quickstart, HF transformers docs, HF blog, generation prompt
  community-refined), and `laguna` (Poolside Laguna 2.1, Jul 2026,
  OpenMDW-1.1; evidence tier 2 -- official HF cards plus the renderers
  library's token-exact LagunaXS21/S21 renderers). Nemotron 3.5's ChatML
  per-turn frames render byte-identical to chatml (only the thinking-on
  generation tail `<|im_start|>assistant\n<think>\n` differs); its BOS
  behavior is UNVERIFIED/conflicting at official level, so the catalog
  models bos="" as an absence-of-evidence placeholder and asserts no BOS
  token. Its thinking-off branch (generation prompt prefills an empty
  `<think></think>` block, no trailing newlines -- unlike kolibri's tail)
  rides the `--mutate` path as `nemotron35/thinking-off`. Muse Glimmer
  frames `<|start|>{role}<|message|>{content}<|eot|>` with a
  `<|begin_of_text|>` BOS; assistant turns use recipient channels and are
  modeled as `to=user` (the `to=self` reasoning trace is not rendered); the
  system appendage (`\nReasoning strength: high.\n# Valid recipients:
  "self", "user".`), the `<|end_of_text|>` EOS, and the end-of-message
  marker are documented, not modeled. Laguna 2.1 renders hugging tags with
  no inner newlines (`<user>{content}</user>`, assistant turns closed by
  `</assistant>`); the base entry is XS-2.1 defaults (enable_thinking FALSE,
  gen prompt `<assistant></think>`), and S-2.1's thinking-on default rides
  the `--mutate` path as `laguna/thinking-on` (gen prompt
  `<assistant><think>`, history turns render an EMPTY think block verbatim);
  EOS is decode-only and never rendered. Mistral Large 4 and Reflection
  Beam are NOT shipped (not released) -- tracked as watched in
  docs/RESEARCH_LOG.md. `tests/test_cycle11_families.py` pins byte-exact
  renders, the thinking-branch mutations, diffs-vs-closest-family, the
  nemotron35 no-BOS-assertion, and the evidence tiers; README family count
  updated 19 -> 22.
- Format catalog expanded to 19 families: `kolibri` (Aleph-Alpha
  Kolibri-1, released 2026-10-03, Apache 2.0; evidence tier 1 -- official
  tokenizer_config.json chat_template byte-verified by an independent fork)
  and `lfm25` (LiquidAI LFM2.5-24B-A2B, released 2026-10-13; evidence tier
  2 -- official model card plus HF blog rendered example). Kolibri-1's
  text-only no-system core is byte-identical to chatml (the always-emitted
  system block with its effort-dependent sentence, the default-high effort
  choice, and the thinking on/off branches are documented in the catalog
  comment; only byte-verified prefixes are pinned in tests). The Kolibri-1
  thinking-off branch (generation prompt prefills an empty `<think>` block)
  rides the `--mutate` path as `kolibri/thinking-off` rather than becoming
  a second catalog entry. LFM2.5 prepends `<|startoftext|>` and always
  thinks (generation prompt appends `<think>`); its bos_token-vs-template
  and system-block caveats are documented, not asserted.
  `tests/test_cycle10_families.py` pins byte-exact renders, the mutation,
  the diffs-vs-chatml, the evidence tiers, the README family count, and the
  deepseek41 system-framing re-verification (V4.1's fullwidth System frame
  CONFIRMED against the primary source 2026-10-15; the sibling V4 encoder
  renders system raw -- the catalog claim stands, no fix needed).
- Format catalog expanded to 17 families: `gemma4` (Gemma 4 — completely
  new turn/channel protocol, `<|turn>role` / `<turn|>`, assistant rendered
  as `model`, literal `<bos>` prepend; evidence tiers documented in the
  catalog comment: response_template start anchor and special-token
  inventory from the official tokenizer_config.json, request-side framing
  corroborated by three independent byte-level implementations).
- Automatic selection-margin diagnostic (`selection_margin` in
  `results.json`, one line in the decision banner): the closed-form number
  of single-fixture outcome flips that would change the decision --
  descriptive only, never changes the selection rule.
- Automatic protocol token-overhead card (`token_overhead` in
  `results.json`, "Token overhead" HTML section): per-candidate
  mean/max prompt tokens, cheapest candidate, and the winner's token tax
  versus cheapest; counts are retained-input/output counts, not
  equal-compute accounting.
- Teacher-forced log-likelihood scoring engine
  (`model_protocol_lab/scoring.py`): `score_target_logprobs` scores an
  exact target token-ID sequence via one forward pass with causal-shifted
  gather; `check_alignment` refuses (status `unavailable`, never a silent
  guess) when a token straddles the prompt/answer boundary; NaN/inf
  rejected, float64 accumulation, deterministic. Implements
  DEVELOPMENT_PLAN.md §9.3–§9.5 (acceptance gate: hand-computed logit
  tensors cross-checked against an independent reference). Not yet wired
  into `run` -- report integration is planned next.
- `LocalHFBackend.forward_logits` (single full-sequence forward returning
  fp32 CPU logits) and a `vocab_size` property, the missing primitives the
  scoring engine needs (`generate` only exposed last-position logits).
- Opt-in teacher-forced log-likelihood diagnostic (`run
  --logprob-diagnostic`, the DEVELOPMENT_PLAN.md §9.3–§9.5 "report wiring
  planned next"): after search, each candidate's expected answer is scored
  against the recorded prompts via `experiments.logprob_diagnostic`;
  `results.json` carries `logprob_diagnostic` and the HTML report renders a
  "Log-likelihood diagnostic" section with per-candidate total/mean
  logprob and a plain-language reading separating "greedy derailed" from
  "weights don't know" (declared bands: mean >= -2.0 / < -5.0 nats/token).
  Alignment refusals surface as `unavailable` with a fixed reason, never a
  silent guess; diagnostic only, the selection rule is untouched.
- Static template↔tokenizer portability matrix (`doctor
  --portability-matrix SECOND_CHECKPOINT`, research basis
  `docs/RESEARCH_PORTABILITY_2026-10-15.md`): for every catalog candidate
  (base formats plus bounded local repairs), each special-token inventory
  literal is checked against the second checkpoint's tokenizer --
  registered in added/special tokens, encodes as exactly one token id,
  encoded id == registered id -- via the new
  `core.template_tokenizer_compatibility` (a generalization of
  `core.check_boundary_portability`). Tokenizer only, never weights; same
  128 MB cap and fixed-vocabulary error codes as the other doctor checks.
  **Asymmetry is the design:** `incompatible` means the format cannot be
  that checkpoint's native format (answered at ~zero compute);
  `compatible` predicts NOTHING about experimental success -- the report
  section, the CLI table (stderr; stdout stays parseable JSON), and
  `docs/USAGE.md` §2 all carry the warning loudly, and no portability
  score is ever emitted. BOS-less formats skip the BOS requirement
  (`bos_skipped` per row); empty inventories and encode failures report
  `unknown`, never `compatible`; hostile second checkpoints degrade to
  `unavailable` without failing the doctor verdict. Pinned by
  `tests/test_cycle12_portability.py` (38 tests).

### Added

- Selection-rule veto pins (`tests/test_cycle8_selection.py`): a failed
  search evaluation in any candidate vetoes selection even when an
  earlier-listed candidate would be the unique winner; a held-out failure
  still names the unique search winner while the decision records where
  it failed.

### Added

- Operator docs for the two new commands: README capabilities now describe
  all 16 format families plus `mpl debug` and `mpl diff-backends`;
  `docs/USAGE.md` gained §10 (`mpl debug`: every dot-command, flags, a
  real session transcript) and §11 (`mpl diff-backends`: flags, the 5
  divergence classes, JSON reading guide, scope limits) — every flag
  verified against live `--help`.
- Hypothesis property tests for the divergence classifier
  (`tests/test_cycle7_properties.py`, derandomized): verdict ⟺
  divergences-empty, divergence well-formedness, severity totality,
  section-aggregation invariants, input-length bound, empty-input safety.

### Added

- Report "Backend divergence" section: the `run` flow now classifies the
  winner's recorded generations with the diff-backends divergence taxonomy
  and renders a verdict badge, a divergence-class × semantics matrix with
  severity badges, and `<details>` callout boxes per divergence
  (presentation: infer-check/Csmith style). Degrades to absent when no
  usable observations exist; the committed sample report is byte-identical.
- Format catalog expanded to 16 families: `mistral_large`
  (Mistral-Large-Instruct-2411 V7, re-verified against the live model card)
  and `glm53` (GLM-5.3-Flash with the confirmed whitespace deltas; the
  auto-injected system line is deliberately omitted like deepseek41's
  preamble).

### Fixed

- `divergence_section_data`'s `observations_processed` now counts the
  observations actually emulated (`len(per_fixture)`) instead of the number
  offered after truncation (`len(candidates)`) -- with skipped
  non-dict/invalid observations the old code overcounted (verified:
  total=4, processed=4, per_fixture=1). Verdicts were never affected (they
  derive from `per_fixture`); this is an informational-contract fix, and the
  docstring now states the contract exactly. Pinned by a new
  `tests/test_cycle11_bounds.py` regression test (junk mixed in -> 4
  handed in, 1 emulated, processed==1, verdict "agree").
- One-line comment at the `DivergenceClass` definition documenting why
  `HARNESS_NORMALIZATION_RELIED`'s SGLang column is vacuous by
  construction (`emulate_sglang_stop` trims at the leftmost stop-string
  occurrence, so its returned text can never contain the stop string) --
  so a future reader does not "fix" it into a false positive.
- Narrowed the exception scope around the new divergence wiring so a future
  bug there degrades only that section, not the whole diagnosis block
  (red-team suggestion).
- Red-team H1: `diff_backends` token-id cap lowered from 131072 to the
  named `MAX_OUTPUT_IDS = 4096` (prompts keep the named
  `MAX_PROMPT_IDS = 131072`, linear decode cost). The emulations do one
  `backend.decode(prefix)` per output step -- quadratic -- so the old cap
  burned tens of minutes per call on the linear reference stub; at the new
  cap a single stop completes in ~2-3 s (measured 2026-10-15). 4096 also
  matches `mpl run --max-new-tokens`' own cap, so the lab can never record
  a longer generated output. Over-cap input is rejected with the fixed
  `output_ids_too_long` code; over-cap recorded observations are skipped.
- Red-team M1: stop texts are now bounded -- 1024 chars per stop
  (`stop_text_too_long`), 16 stops per call (`too_many_stop_texts`) --
  enforced in `_checked_stop_texts`, the `diff-backends` CLI path, and the
  `run` diagnosis path (which passes `backend.decode(winner_boundary)`
  through the same validation). Dedup is now set-based (linear, not
  O(m^2)) and runs before the count check.
- Red-team L1: tokenizer-loader failures in `mpl debug` and
  `mpl diff-backends` collapse to the fixed `tokenizer_load_failed` code
  via an exact-match allowlist (`doctor._loader_failure_code`); hostile
  checkpoint bytes can no longer reach the terminal, and non-ValueError
  parse failures no longer produce a traceback. The debug REPL's error
  path strips terminal-control characters from exception text.
- Red-team cycle 13, `diff-reports` (all pinned by new
  `tests/test_cycle13_diffreports.py` regression tests):
  - **H1 (CI-blocking):** `pytest` added to the `dev` extras in
    `pyproject.toml` (`pytest>=8.0`) -- it is the suite's actual runner
    (the README test-count pin test shells out to `python -m pytest
    --collect-only`), and CI's `static`/`test` jobs install `.[dev]` /
    `.[dev,hf]`, so the dev extra is exactly what CI installs. Pin test
    verified passing in the repo venv.
  - **M1:** `render_summary` now runs the existing `_display_id`
    sanitizer over the decision strings and the source labels before
    they reach stderr -- hostile `decision` values (ANSI escapes, OSC
    sequences) can no longer inject terminal control bytes.
  - **M2:** `_config_snapshot` candidate-name dedup is now set-based
    (linear) instead of an O(n^2) list scan -- the old code hung ~1.5h
    on 800k candidate names.
  - **M3:** `_search_verdicts` now also returns per-candidate `planned`
    counts, and the diff emits `planned:<candidate>` config-drift
    entries (surfaced in the stderr drift section) when they differ --
    planned 2 vs 50 no longer compares as silently identical.
  - **L1:** evidence-presence asymmetry now sinks `identical` (new
    machine-readable `summary["evidence_asymmetric"]` flag), and the
    "no differences" stderr line names the axes actually compared
    (decision, winner, candidate verdicts, configuration, plus
    per-fixture evidence only when both sides carry it).
  - Documented (comment at `_fixture_outcomes`, no behavior change): the
    duplicate-fixture-ID last-wins vs duplicate-candidate first-wins
    inconsistency is deliberate -- reconciling it would change only
    hostile-input behavior.
- `emulate_vllm_holdback` now skips the stop check on steps whose token
  decodes to no new text, porting upstream `check_stop_strings`'
  `if not new_char_count ...: return None`
  (`vllm/v1/engine/detokenizer.py`). Without it, the find window start
  `1 - len(stop)` is 0 for single-character stops (e.g. `"\n"`), so a stop
  completed during the `min_tokens` gate was refound at the first
  zero-new-char step -- upstream never refires it. Found by a new
  differential harness (`tests/test_cycle14_holdback.py`): a line-faithful
  port of upstream's `update`/`check_stop_strings` compared against the
  emulation over 32 adversarial cases (stop at position 0, stops spanning
  token/gate boundaries, empty-decoding tokens, unicode, both
  `include_stop_str_in_output` modes, empty output) plus a 400-trial seeded
  fuzz; all 435 subtests agree with upstream after the fix.
- `emulate_hf_stop_string` now treats empty token representations as
  transparent to the match walk (upstream assigns them valid positions
  `1..len(stop)-1` and zero length contribution, so a live walk always
  passes through them) and bounds the walk to the last `len(stop)`
  representations, mirroring upstream's
  `input_ids[:, -maximum_token_len:]` truncation -- found by a new
  differential harness (`tests/test_cycle14_hfstop.py`) comparing the
  emulation against the real `transformers.StopStringCriteria` (5.19.0)
  over the docstring's match/no-match cases, stop-at-0/empty-output/
  unicode/overlapping/overlong stops, prompt-boundary overhangs, the
  byte-space path (UTF-8 fragments, byte-fallback hex tokens, OOV ids),
  multi-stop earliest-completion, plus a 525-trial seeded fuzz; the one
  pre-fix disagreement was `["s", "", "topper"]` vs `"stop"`, where
  upstream fires at step 3 and the port stayed silent. Out-of-vocabulary
  ids on the byte path now map to an empty `_OOVTokenBytes` marker (still
  `== b""`, so they can never match) that keeps killing the walk exactly
  like upstream's dummy-row clamp; previously the shared `b""` could not
  be told apart from a genuine empty token. All 3000+ differential trials
  agree with upstream after the fix.

### Added

- `mpl debug`: an interactive protocol-debugger REPL on stdlib `cmd`
  (sqlite3-style dot-commands: `.use`, `.render` with per-segment maps,
  `.tokens`, `.step` for walking boundary detection, `.audit`, `.say`
  multiline entry, `.save`, bounded `.read`, `.quit`; `--script` for
  non-interactive runs). Attaches a checkpoint tokenizer without loading
  weights; operator-visible strings are terminal-escape sanitized.
- `mpl diff-backends`: differential-testing of the emulated stop semantics
  (HF stop-string, vLLM strip/keep, EOS) over one recorded token sequence,
  classifying divergences (`KEPT_VS_STRIPPED`, `FIRING_STEP_DIFFERS`,
  `MIN_TOKENS_GATED`, `EOS_PRECEDENCE`, `HARNESS_NORMALIZATION_RELIED` --
  the last scoped to failed strips so the `agree` verdict is meaningful).
  Grounded in vendors' bug trackers (vLLM #49391/#46113/#47137, SGLang
  #28802, lm-eval #3587); see `docs/RESEARCH_LOG.md` Round 4 Cycle 2.
- Format catalog expanded to 14 families: `kimi3` (K3 XTML, structure
  spot-checked against the live `encoding_k3.py`), `deepseek41`
  (V4.1-Flash chat mode), `glm45` (GLM-4.5/4.6 think variant).

### Fixed

- `HARNESS_NORMALIZATION_RELIED` no longer fires tautologically on
  kept-disposition cells (red-team: it made `verdict="agree"` unreachable
  whenever any stop fired); now reports only failed strips, with a
  regression test for the genuine anomaly.
- `mpl debug` hardening (red-team): terminal-escape sanitization of all
  fixture-derived output, `.read` refuses non-regular files, transcript
  and `.say` buffer memory caps, `.save` path behavior documented.
- `diff-backends` input caps (1MB texts, 131072 token IDs).
- Doctor suggestion-sizing test updated for the 14-family catalog (the
  longest render now exceeds the tiny test model's 64-token window; the
  test pins the exact sizing formula on a 512-window variant).

### Added

- Format catalog expanded from 3 to 11 families, each hand-rendered from
  primary sources (see `docs/RESEARCH_LOG.md` Round 4): `qwen3`, `deepseek`
  (V3/R1 framing), `mistral` (v0.3 release revision), `phi4`, `gemma3`,
  `llama4`, `glm4` (4-9B), `kimi2`. `Format` gained optional
  `user_turn_end` / `assistant_turn_end` / `open_suffix` overrides for
  families whose per-role terminators differ (e.g. DeepSeek's unterminated
  user turn, GLM-4's bare `<|assistant|>` generation prompt). Also fixed a
  pre-existing `UnboundLocalError` that broke the `catalog` subcommand.
- Hypothesis property-based fuzz tests (`tests/test_fuzz_boundary.py`,
  derandomized): stop-overhang suffix completeness, first-firing-step
  minimality against a naive reference, UTF-8 incremental-decode
  round-trips, NFC byte-offset disagreement, byte-fallback token chunks.
- Stop-matrix hardening: the vLLM cell now exposes the competitor's text
  (`StopOutcome.returned_text`), strips the stop string by default
  (`include_stop_str_in_output=False`, the upstream default -- HF always
  keeps it, surfaced per cell as `stop_text: kept|stripped`), and models
  `min_tokens` gating upstream-exact (check skipped while
  `num_output_tokens <= min_tokens`; the find window's negative start
  counts from the end of the text, so a stop fully completed during the
  gate is not refound). FAQ documents the lab's no-implicit-EOS discipline
  vs both backends' defaults and how to match it per backend.
- Special-token literal audit (GHSA-g5f9-3xfg-p9mf): `fixtures build`
  warns on stderr and writes a `<output>.audit.json` sidecar when fixture
  text contains literals colliding with candidate formats' special tokens
  (role-boundary forgery hygiene); `doctor` reports the same as a warning.
- Report upgrades (still one self-contained file, no external assets):
  sortable tables (small inline script, inert-safe), sticky anchor TOC,
  exclusive `<details name>` accordions, SVG budget-frontier sparkline, and
  a tested no-remote-asset property.

### Fixed (observation-count bound, 2026-10-15)

- Red-team residual of the H1/M1/L1 fix cycle: `divergence_section_data`
  now bounds the *number* of recorded observations with the named constant
  `MAX_DIVERGENCE_OBSERVATIONS = 512`. Legitimate runs hand over at most
  200 observations (the experiment fixture cap is 100 search + 100
  held-out fixtures, and only the winner's observations feed the section),
  so real runs are never affected; a hostile report with millions of
  observations is now fixed work instead of open-ended emulation time.
  Over-count input degrades by omission, never by exception: the first 512
  observations are processed in input order and the section records the
  fixed-vocabulary `observations_truncated: true` plus `observations_total`
  / `observations_processed`; the HTML report names the bound explicitly.
  Pinned by `tests/test_cycle11_bounds.py` (boundary N/N+1, first-N order,
  determinism, no-usable-observations still degrades to an omitted
  section). Per-observation bounds (H1/M1) are unchanged.

### Fixed

- vLLM `min_tokens` emulation corrected to upstream-exact semantics after
  a source-level re-check of `vllm/main` `v1/engine/detokenizer.py`
  (negative `find` start counts from the end; straddling-update offset
  logic), replacing an earlier never-fire approximation.
- Report `data-sort` attributes now escape hostile strings (attribute
  injection caught by cross-agent review before release); a regression test
  pins the fix.
- `fixtures build` no longer leaves an orphan fixture file if the audit
  sidecar loses a concurrent-creation race: the just-written fixture is
  unlinked and the collision is reported distinctly from authoring errors.
- Empty output with `min_tokens=0` in the vLLM emulation is labeled
  `no_stop_within_cap`, not `gated_by_min_tokens` (nothing was gated).

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
- HTML report upgrades, still a single self-contained file with no external
  assets: sticky section table of contents with `:target` highlighting;
  click-to-sort tables via one small inline script (SHA-256 hash-allowlisted
  in the report's Content-Security-Policy -- the report reads identically
  with scripting disabled); mutually exclusive `<details>` accordions via
  `name=`; inline-SVG pass-curve sparklines for the token-budget frontier
  (same quantized data as the text sparklines, prints and scales reliably).
- Red-team (follow-up, 2026-10-15): `diff-reports` config-drift field
  reached stderr unsanitized -- the `planned:<candidate>` drift field is
  built from the raw candidate name, and `render_summary` interpolated the
  field raw, so a hostile candidate name with ANSI/OSC escapes reached
  stderr (confirmed live). Fix: the field is passed through the existing
  `_display_id` sanitizer at the render loop, which covers every drift
  field type (planned:, fixtures:search/held_out, and the fixed config
  fields). Pinned by a regression test in
  `tests/test_cycle13_diffreports.py` (hostile candidate name in a
  planned-drift entry -> no raw escape bytes in the rendered summary).
- Red-team (follow-up, 2026-10-15): the README test-count pin test never
  inspected `proc.returncode`/`proc.stderr`, so a pytest collection error
  surfaced as the misleading "could not parse pytest collection count".
  Fix: the test now asserts `proc.returncode == 0` first, and both
  failure messages include the last ~2000 characters of stderr/stdout so
  CI collection failures are debuggable from the test output alone.

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

### Added

- SGLang column for `mpl diff-backends` (implements the Cycle 1 research
  recommendation): new `sglang_stop` emulated semantic
  (`backends.emulate_sglang_stop`, a port of SGLang's
  `Req._check_str_based_finish` with `DetokenizerManager.trim_matched_stop`
  leftmost-occurrence stripping; deliberately takes no `min_tokens`
  parameter -- SGLang's `min_new_tokens` only clips the length budget and
  the finish path never consults it) plus three new divergence classes:
  `MULTI_STOP_SELECTION` (high; SGLang first-listed vs vLLM
  earliest-completion multi-stop selection, vLLM PR #49391),
  `NO_MIN_TOKENS_GATE` (medium; the SGLang column on `MIN_TOKENS_GATED`),
  and `STR_OVER_EOS_PRECEDENCE` (high; same-step stop+EOS tie where
  SGLang's shipped #28802 fix gives the stop string the win while TGI's
  documented precedence gives EOS the win). The report's "Backend
  divergence" section renders the new classes with severity badges and a
  "SGLang stop" matrix column. Deliberately no TGI cell: every TGI stop
  rule is covered by composition of the existing cells (documented in
  `USAGE.md` §11); SGLang `stop_regex` gets a USAGE note only. Pinned by
  `tests/test_cycle10_sglang.py` (27 tests).

## [0.1.0a0] — 2026-10-12

Initial source snapshot: renderer, evaluator, catalog, local HF adapter,
experiment orchestration, reports, CLI, fixture schema, docs and test source.
Unexecuted at release of the snapshot.
