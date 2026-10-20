# Validation status

## Evidence — continuous development, cycles 10–153 (executed 2026-10-14 → 2026-10-18, Python 3.12.3, Linux x86-64)

Consolidated record for the continuous autonomous development period
(2026-10-14 → 2026-10-18). All commands run from the repository root in
the project's `.venv` (torch 2.14.1+cpu, transformers 5.19.0, hypothesis
installed; the `[dev]` extra now includes `pytest>=8.0`).

- `python -m pytest -q -p no:cacheprovider` — **6782 tests collected**
  (cycle 176, 2026-10-20). Subtests numbered in the 70k+
  range in recent cycles.
- `python -m unittest discover -s tests -t .` — **6782 tests (cycle 176,
  2026-10-20; both runners agree exactly). Cycle 106 closed the pytest-only gap: 9 test files that
  unittest silently skipped (2533 pytest-style test executions) were
  converted to `unittest.TestCase`, which pytest also runs. Both runners
  now execute the full suite. Includes 29 format families (chatml, llama3, llama4, gemma, gemma3,
  gemma4, qwen3, qwen38, deepseek, deepseek41, mistral, mistral_large,
  phi4, glm4, glm45, glm53, kimi2, kimi3, kolibri, lfm25, nemotron35,
  muse_glimmer, muse_spark, laguna, ling, k2horizon, hy4, minimax3,
  nanbeige42), the `debug` REPL, `diff-backends` (HF/vLLM/SGLang),
  `diff-reports`, `doctor --portability-matrix`, the teacher-forced
  log-likelihood diagnostic, the fixture-loader contract fuzzer
  (`fixtures fuzz`), and differential-verification suites with
  mutation gates for every module.
- `ruff check` — clean; `ruff format --check` — clean (240 files).
- `mypy model_protocol_lab` — clean (45 source files).
- `bandit -r model_protocol_lab` — **no issues identified** at any
  severity/confidence (the targeted `# nosec B105` suppressions on
  report-schema fields carry justifications).
- `pip-audit` — no known vulnerabilities in auditable dependencies;
  `torch 2.14.1+cpu` keeps its local version tag (invisible to the OSV DB —
  the accepted residual, unchanged).
- Red-team: 3 findings fixed with regression tests (quadratic hang in
  vLLM holdback emulation → split input caps; unbounded stop texts →
  length/count bounds; hostile terminal text → fixed vocabulary), plus
  8 follow-up findings across two more passes (7 fixed with regression
  tests; 1 documented in a code comment with no behavior change). No
  genuine finding left open.
- GitHub Actions CI on `main` — green on the pushed tree (tests on
  3.11/3.12/3.13, lint+typecheck, security audit, build check).

## Evidence — third hardening round, red-team (executed 2026-10-13, Python 3.12.3, Linux x86-64)

Adversarial round 3 against the five new capabilities, with fixes and
re-verification. All commands run from the repository root.

- Fresh installs: `python -m venv` + `pip install -U pip setuptools`, then
  `pip install -e .` (base: package imports, `catalog` smoke OK) and
  `pip install -e ".[dev]"` (ruff, mypy, coverage, hypothesis present).
- `python -m unittest discover -s tests -t .` — **361 tests, all passing**
  (352 carried over + 9 new regression tests), across **3 consecutive green
  executions** (9.79 s / 9.50 s / 9.87 s — no variance, no flakiness).
  The `[dev]`-only suite (fresh venv, no `hf` extra) is green again:
  **OK (skipped=73)** — the round-3 test file had broken this path with
  unguarded HF imports (fixed with the `HAS_HF` skip guard).
- `ruff check` and `ruff format --check` — clean on
  `model_protocol_lab tests examples` (21 files).
- `mypy model_protocol_lab` — clean (10 source files).
- `coverage run -m unittest discover -s tests -t .` — **98% total line
  coverage** (1493 statements), above the 97% bar; new code is covered by
  the new regression tests (the two misses are the provably-dead difflib
  block-merge branch and pre-existing defensive branches).
- `bandit -r model_protocol_lab` — **0 issues** at any severity/confidence.
- `pip-audit` (2026-10-13) — **no known vulnerabilities** in the installed
  set; `torch 2.14.1+cpu` keeps its local version tag (invisible to the
  OSV DB — the accepted residual; CVE posture re-verified manually per
  `docs/SECURITY.md` F2).
- `python -m build` + `twine check` — re-verified clean in round 1;
  packaging metadata unchanged since.
- Differential verification of the fixed HF stop emulation against the
  real `transformers.StopStringCriteria` (locally trained byte-level BPE
  tokenizers, no downloads): **1500/1500 random trials agree**, plus an
  exhaustive 256-combination scan with zero divergences. The pre-fix port
  diverged on 18/300 trials (all lab-miss-real-fire).
- `template_diff` equivalence: 400 randomized inputs produce byte-identical
  output to unbounded difflib; the adversarial alternating pattern (45 s at
  n=2000 pre-fix) now degrades in ~2 s with a 2 s budget.
- CLI smoke (tiny locally-generated GPT-2 checkpoint, no downloads):
  `catalog`, `inspect`, `doctor` (plain, `--boundary-ids`,
  `--reference-model`, invalid-IDs exit 1), `run` (plain,
  `--mutate --sensitivity-sweep --max-drop-bytes 2 --include-evidence`;
  all diagnostics sections present; `--max-drop-bytes 9` rejected by
  argparse). Sample-report golden test regenerates byte-identically.
- Config files validated: CI YAML (4 jobs, actions pinned to SHAs),
  pre-commit YAML (ruff-pre-commit v0.17.0), Dependabot YAML (pip,
  github-actions, pre-commit), `CITATION.cff` (title/authors present).
- Red-team round 3 findings fixed: H3-1 (HF emulation missed mid-token
  end-overlaps — stop matrix under-reported stops), H3-2 (`template_diff`
  super-quadratic blowup — 10 s degradation budget), H3-3 (uncapped
  `tokenizer.json` loads in `doctor`'s reference/conformance paths),
  H3-4 (hostile Jinja template OOM — 1 MB source/render caps), H3-5
  (`[dev]`-only suite broken by unguarded HF imports). Each with a
  regression test; full triage in `docs/SECURITY.md`. Probed and defended
  without changes: report XSS via hostile candidate names (all four new
  sections), `--reference-model` traversal/symlink attacks, UTF-8-tail
  truncation (no lone surrogates), sweep compute bounds and cancellation,
  taxonomy classification (17 adversarial shapes), vLLM emulation vs the
  upstream detokenizer source.

## Evidence — development cycles 1–4 (executed 2026-10-13, Python 3.12.3, Linux x86-64)

Four research → plan → develop cycles followed the red-team round, each
logged with sources in `docs/RESEARCH_LOG.md` (prior competitive research:
lm-eval harness, vLLM, llama.cpp, HF transformers, lighteval, HELM, Unitxt —
10 ideas killed as already covered). Cycles stop only after consecutive
insignificant rounds; Cycles 3 and 4 were both quiet, winding the process
down.

- Cycle 1: byte-space HF stop matching (ports upstream's decoder-mode
  detection; differentially proven against real `StopStringCriteria`),
  token-budget robustness frontier, `doctor` next-command suggestion,
  `docs/FAQ.md` (12 Q&As).
- Cycle 2: per-candidate response boundaries (`run --boundary-map`),
  `fixtures build` CSV authoring with leakage-unbuildable discipline,
  report views (per-fixture outcome grid, decisive fixtures, budget
  sparklines). The llama.cpp stop cell was verified against primary
  sources and killed as duplicative (documented in `docs/USAGE.md` §8.2).
- Cycle 3 (quiet): `doctor` suggestion window-fitting, failed-run
  diagnosis with sanitized failure codes, 3 mutation-found boundary
  tests; cosmic-ray feasibility trial on `fixtures.py` (87% interim kill
  rate; full-codebase run judged impractical per-commit).
- Cycle 4 (quiet): scoped cosmic-ray trial on `core.py` (56 mutants: 42
  killed, 12 equivalent, 2 real gaps fixed with kill-verified tests);
  first-run UX fixes. No upstream drift found in any cycle.

Final verification (independent runs for this section):
- `python -m unittest discover -s tests -t .` — **708 tests, all passing**,
  consecutive green executions.
- `coverage` — **95% total line coverage** (2322 statements).
- `ruff check` and `ruff format --check` — clean (25 files);
  `mypy model_protocol_lab` — clean (11 files).
- CLI smoke: `catalog`, `inspect examples/fixtures.json`,
  `fixtures build examples/fixtures.csv --output` (output byte-equivalent
  to `examples/fixtures.json`).

## Evidence — second hardening round (executed 2026-10-13, Python 3.12.3, Linux x86-64)

A fresh-eyes second pass (independent audit of all 10 modules, a new
adversarial red-team round with novel vectors, re-verification of every
workflow and documented command). All commands run from the repository root.

- `python -m unittest discover -s tests -t .` — **224 tests, all passing**
  (213 carried over + 11 new regression tests), across **4 consecutive green
  executions** (no variance — no flakiness). Suite now also passes with only
  the `[dev]` extra installed: `OK (skipped=40)` — the 3 hf-dependent tests
  that previously *failed* without torch now skip cleanly like the other 37,
  so the README quick-start `[dev]`-only path is green.
- `ruff check` and `ruff format --check` — clean on
  `model_protocol_lab tests examples` (20 files).
- `mypy model_protocol_lab` — clean (10 source files).
- `coverage run -m unittest discover -s tests -t .` — **97% total line
  coverage** (961 statements), holding the bar; new code is covered by the
  new regression tests.
- `bandit -r model_protocol_lab` — **0 issues** at any severity/confidence.
- `pip-audit` (fresh venv, 2026-10-13) — 12 known vulnerabilities, all in the
  venv's bundled `pip 24.0` itself; **zero in any project dependency**
  (torch, transformers, safetensors, tokenizers audited clean).
- `python -m build` + `twine check` — re-verified clean in round 1; packaging
  metadata unchanged since.
- Fresh-eyes audit findings fixed: 1 medium (stock GPT-2 unusable with the
  CLI default `--max-context 2048` — now fails fast with the distinct,
  actionable code `local_context_limit_exceeds_declared_window`, documented
  in `docs/USAGE.md`) and 6 lows (false `candidate_fingerprint` docstring
  corrected; `run_experiment` now validates `boundary`/`cancel`/fingerprint
  types up front instead of raw `TypeError`s; `evaluate` rejects non-`Event`
  `cancel`; `doctor` validates `boundary_ids` element types and returns a
  uniform report shape on early failures). Each with a regression test.
- Red-team round 2 findings fixed: U1 (NFD/NFC evasion of ID-uniqueness and
  cross-split leakage checks — `Split` now compares NFC-normalized keys,
  stored data stays byte-exact) and T1 (terminal escape injection via raw
  fixture IDs in `inspect`/`run` stderr warnings — control characters
  stripped). 9 of 10 probed vectors verified defended with PoCs. Full
  triage in `docs/SECURITY.md`.
- Docs: `docs/USAGE.md` documents the GPT-2 context-window requirement and
  the NFC-normalization behavior; round-1 "not established" items below are
  unchanged.

## Evidence (executed 2026-10-12, Python 3.12.3, Linux x86-64)

The following were actually run against this working tree; commands are
reproducible from the repository root in a fresh virtual environment.

- `python -m pip install -e .` and `python -m pip install -e ".[dev,hf]"`
  (torch 2.14.1+cpu, transformers 5.19.0, safetensors 0.8.0, tokenizers —
  CPU-only wheels; no model weights downloaded at any point).
- `python -m unittest discover -s tests -t .` — **213 tests, all passing**
  across repeated runs (8+ consecutive green executions after a transient
  tmpfs-space flake in an oversized-file test was fixed with a sparse file).
  Suite composition: core invariants, dataset validation, workflows/selection,
  CLI end-to-end (subprocess), adapter integration against a locally generated
  tiny GPT-2 safetensors checkpoint (config validation, strict
  loading-info checks, greedy determinism, cancellation, failure records),
  `doctor` checks, sample-report golden test, hypothesis property tests
  (dataset loading, fixture separation, boundary parsing), and regression
  tests for every security/red-team finding.
- `ruff check` and `ruff format --check` — clean on
  `model_protocol_lab tests examples` (includes flake8-bandit `S` rules).
- `mypy model_protocol_lab` — clean, both with the `hf` extra installed and
  in a venv without it (optional imports silenced via per-module override).
- `coverage run -m unittest discover -s tests -t .` — **97% total line
  coverage**; `datasets.py`, `cli.py` (99%), `experiments.py`, `catalog.py` at
  100%/near-100%, `core.py` at 98%, `reporting.py` at 98%, `backends.py` at
  93%, `doctor.py` at 95%. Remaining misses are defensive/racy branches:
  mid-manifest file substitution, lock-race cancellation windows, non-finite
  logits guards, undecodable boundary diagnostics, the non-`local_` backend
  error branch, and the `__main__` SystemExit path — deliberately not chased
  with flaky tests.
- `python -m build` + `twine check` — sdist and wheel build; metadata valid
  (`License-Expression: MIT`, `License-File: LICENSE`).
- CLI smoke: `catalog`, `inspect examples/fixtures.json`, and `doctor` against
  synthetic and real (locally generated) checkpoint directories.
- HTML report validated programmatically: required structure, zero external
  assets (no http/https/script/link/img/iframe, strict CSP), all text pairs at
  WCAG AA contrast (minimum measured 6.81:1), hostile-input escaping, byte-identical
  re-rendering, all six decision values rendering safely.
- Config files validated: CI YAML, Dependabot YAML, pre-commit YAML and
  `CITATION.cff` parse and match their schemas; GitHub Action versions
  (`checkout@v7.0.1`, `setup-python@v7.0.0`) and pre-commit hook IDs/tag
  (`ruff-check`, `ruff-format`, `v0.17.0`) verified against the official
  release/tag APIs and the hook manifest. (`pre-commit run` itself could not
  execute in this sandbox — outbound git to github.com times out — but runs
  exactly the `ruff check`/`ruff format` commands verified clean here.)
- Adversarial red-team round: 1 high (FIFO hang), 1 medium (output-dir
  ordering), 4 low findings — all fixed with regression tests; 10 attack
  vectors verified as defended (selection-rule gaming, boundary bypasses,
  locale-independent byte-identical reports, symlink escapes, XSS, supply
  chain, ReDoS, algorithmic DoS, generation-config influence, docs
  spot-checks). Full report in `docs/SECURITY.md`.
- `bandit` — 0 issues on `model_protocol_lab/`; `pip-audit` — clean on the
  installed set (after upgrading install-time `pip`/`setuptools`).

## Design notes with primary-source grounding

- Raw token evaluation (`add_special_tokens=False`) follows the workflow the
  Hugging Face chat-templating docs prescribe for avoiding duplicated special
  tokens; the BOS-duplication pitfall it avoids is documented by both HF and
  lm-evaluation-harness.
- The token-ID boundary match is exact by design; the known false-negative
  mode (same boundary text, different tokenization — cf. HF transformers'
  `StopStringCriteria`) is surfaced per observation as `boundary_text_match`
  and documented in `docs/USAGE.md` §4. It never changes the pass criterion.
- Decode order of operations: the lab cuts in token-ID space, then decodes
  (`decode(output_ids[:cut])`, `skip_special_tokens=False`,
  `clean_up_tokenization_spaces=False`); lm-eval decodes first and truncates
  at the string level. These can differ at the cut edge for subword
  tokenizers; boundary IDs should be verified in context (`doctor
  --boundary-ids`).
- The conservative selection rule (exactly one all-pass candidate; ties and
  failures abstain; winner frozen before held-out) has no counterpart in the
  surveyed harnesses; it is this lab's own discipline, not an industry
  standard.

## Not established

- No real (non-synthetic) model run has been performed; adapter behavior on
  production checkpoints, CPU/GPU parity, and VRAM fit are unverified.
- No browser rendering check of `report.html` (structure/contrast/escaping
  validated programmatically only).
- No full-codebase mutation-testing run (scoped cosmic-ray trials on
  `fixtures.py` and `core.py` only; see `docs/RESEARCH_LOG.md` Cycles 3–4);
  no benchmark of any kind.
- The weekly CI schedule and Dependabot have not yet executed on GitHub.
- `pre-commit run --all-files` has not executed end-to-end (see above).

## Important limitations

1. The local adapter's supported architecture and dependency combinations have
   integration tests only against a synthetic tiny GPT-2 checkpoint.
   GPT-2/Llama-only loading is intentionally narrow; unsupported configurations
   should fail closed.
2. Exact string fixtures are strict behavioral probes, not general model-quality metrics. Group IDs and exact-duplicate checks do not establish semantic independence.
3. A unique all-pass search candidate can still be a false selection on an insufficient fixture set. There is no statistical confidence estimate.
4. The supplied example is only a schema demonstration. Build separate representative search and held-out families before drawing conclusions.
5. Full-prefix greedy decoding is slow and float32 weights can consume substantial memory. No general laptop-size or GPU-fit guarantee exists.
6. Cancellation is cooperative. A running kernel, model load, tokenizer operation or filesystem operation may not stop immediately.
7. Asset hashes provide identity checks, not a sandbox or a guarantee against concurrent file substitution. Use immutable trusted model assets.
8. Default JSON omits observations but still contains metadata and format strings. Opt-in evidence can reveal text through token IDs. There is no guaranteed secure erasure.
9. Report creation is not a multi-file atomic transaction. A write failure may leave a partial output directory; retry with a new directory after inspection.
10. Current counts do not measure model FLOPs, repeated prefix processing, total wall time or equal search budgets. Fair-compute comparisons are future work.
11. There is no live training instrumentation, arbitrary template inference, Jinja export, model-weight reverse engineering, distributed execution or production web service.

## Release gate

Do not tag this source as a validated release until the execution, dependency, integration, privacy, scientific-evaluation and reproducibility gates in DEVELOPMENT_PLAN.md are satisfied with retained evidence. Record failures as well as passing results. Do not weaken assertions merely to obtain a green suite.

## Fresh-install and CLI smoke test (cycle 94, 2026-10-16)

- Fresh venv (`python3 -m venv`, no extras): `pip install .` exits 0;
  `model_protocol_lab.__version__` reports `0.1.0`.
- All 8 CLI subcommands (`catalog`, `inspect`, `doctor`, `run`,
  `fixtures`, `diff-reports`, `diff-backends`, `debug`) print `--help`
  cleanly from the installed entry point.
- Functional smoke (installed package, no dev checkout):
  `catalog` lists families; `inspect examples/fixtures.json` ->
  `{"search_cases": 2, "held_out_cases": 2, "valid": true}`;
  `doctor --model` on a weights-less stub fails closed with a
  structured `failed` verdict (`require_one_standard_safetensors_layout`);
  `fixtures build examples/fixtures.csv --output` ->
  `inspect` round-trip valid; `diff-reports` on the sample report
  against itself reports "No differences".
- `run` / `debug` / `diff-backends` were not exercised here: they need
  a real tokenizer/checkpoint (`hf` extra), covered by the
  integration tests instead.
