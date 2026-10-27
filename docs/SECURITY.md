# Security and data handling

## Trust boundaries

Treat checkpoints, tokenizer files, fixtures, generated text and reports as separate trust boundaries. Local-only loading and safetensors reduce exposure; they do not make arbitrary files, native dependencies or GPU drivers safe. Run unfamiliar model assets in an appropriately isolated environment with resource limits.

Use trusted, immutable checkpoint directories. Do not grant this process unnecessary filesystem or network access. The adapter requests local-only model resolution, disables remote model code and rejects pickle weight layouts; runtime network isolation is verified by `test_no_network_access_during_doctor_and_backend_init`. Run `doctor` on any unfamiliar checkpoint before evaluation; it validates assets without loading weights.

## Security audit (2026-10-12)

A full audit was conducted against this tree and every finding was resolved or
explicitly accepted below. Methodology: `pip-audit 2.10.1` (OSV DB) on the
installed dependency set; `bandit 1.9.4` on `model_protocol_lab/`; ruff
flake8-bandit (`S`) rules in CI; manual line-by-line review of all package
modules; adversarial PoCs (XSS payloads through `render_html`, path-traversal
names through `_member`, 100 kB ReDoS probes, deeply-nested/NaN JSON through
the dataset loader). `semgrep` could not run (sandbox `/tmp` exhaustion);
bandit + ruff `S` + manual review covered the same ground.

### Findings and resolutions

| ID | Severity | Finding | Resolution |
|----|----------|---------|------------|
| F1 | Medium | No lockfile/hashes; `torch>=2.6` unbounded — fresh installs trust PyPI tip-of-range. | Upper-bound discipline documented; `pip-audit` runs as a CI gate (`security` job) on every build; torch CVE posture tracked below. `torch` capped `<3` in cycle 201 (2026-10-23); the lockfile gap remains accepted (residual 2). |
| F2 | Low | `pip-audit` cannot see `torch 2.14.1+cpu` (local version tag). | Manual CVE review 2026-10-12: postdates fixes for CVE-2025-3000 and CVE-2025-32434; CVE-2026-4538 (`torch.export.load`) is not reachable — the codebase never calls `torch.export.load`/`torch.load`; weights load only via `from_pretrained(use_safetensors=True, weights_only=True)`. Re-verify on every torch bump. |
| F3 | Low | Report files created with default umask (0644) — readable by other local users. | Fixed: `save_report` now creates the directory with `0o700` and files with `0o600`, using `O_EXCL` exclusive creation. Regression test asserts the modes. |
| F4 | Info | `_read_json` caps JSON at 128 MB but HF re-reads `tokenizer.json` uncapped. | Fixed: `stat()` size check (128 MB) on `tokenizer.json` before `PreTrainedTokenizerFast` loads it; sparse-file regression test. |
| F5 | Info | Deeply nested JSON raised bare `RecursionError`, not `ValueError`. | Fixed: caught and re-raised as `ValueError("Dataset nesting too deep")`. |
| F6 | Info | `load_dataset` on a TTY/char device could block indefinitely. | Fixed: refuse character/block devices with `ValueError`. Superseded by H1 (2026-10-12): FIFOs, sockets and directories are now refused too; only regular files are accepted. |
| F7 | Low | CI used mutable action tags and had no audit gate. | Fixed: actions pinned to SHAs; new `security` CI job runs `pip-audit` (after `pip install -U pip setuptools`) and `bandit`, both gating. |

### Red-team round (2026-10-12)

An adversarial review round found 1 high, 1 medium and 4 low issues; all
fixed with regression tests:

| ID | Severity | Finding | Resolution |
|----|----------|---------|------------|
| H1 | High | F6 was fixed incompletely: a named FIFO with no writer hangs `load_dataset` forever (no timeout), contradicting F6's own rationale. | Fixed: `load_dataset` now opens with `O_NONBLOCK` and `fstat`s the fd, refusing everything that is not a regular file (char/block devices, FIFOs, sockets, directories). The fd-based check also closes the stat→open TOCTOU. Watchdog-thread regression test proves no hang. |
| M1 | Medium | Output-path failures surfaced only after the expensive run; the completed result was discarded with it. | Fixed: the CLI claims the output directory (via new `prepare_report_dir`) before backend init; on any failure before the report is saved the empty dir is released so retries still see a fresh path. Regression tests cover init failure and path-through-file. |
| L1 | Low | Empty `expected` answers are trivially satisfiable (a backend emitting only the boundary passes) with no warning. | Fixed: `inspect` and `run` now warn on stderr naming the vacuous fixtures; documented in `docs/USAGE.md`. |
| L2 | Low | `candidate_fingerprint` was syntactic: `repeat_bos=True` with `bos=""` fingerprinted differently from `repeat_bos=False`, contradicting `local_repairs`' own dedup notion of identity. | Fixed: the fingerprint payload is normalized exactly like the dedup key. |
| L3 | Low | Intermediate report parent dirs were 0o755, overstating the "readable only by the invoking user" claim. | Fixed: `prepare_report_dir` chmods every created path component to 0o700; regression test asserts all components. |
| L4 | Low | `doctor`'s `tokenizer_deep` failure interpolated raw dependency exception text into machine-readable output — the one hole in the sanitized-error discipline. | Fixed: maps to the fixed code `tokenizer_deep_check_failed`. |

The red team also verified dynamically (socket firewall raising on any
`connect`) that `doctor`, backend init and generation perform zero network
access — this retires residual risk #4 below. A regression test
(`test_no_network_access_during_doctor_and_backend_init`) now guards it.

`pip-audit` raw results: 6 vulns in `pip 24.0` and 2 in `setuptools 78.1.0`
(install-time tools only; the package has zero runtime dependencies) —
resolved by upgrading (`pip install -U pip setuptools`); post-upgrade audit is
clean. `transformers 5.19.0`, `safetensors 0.8.0`, `tokenizers 0.23.3` audited
clean.

### Red-team round 2 (2026-10-13)

A second adversarial round probed novel vectors only (round-1 vectors were not
re-tested). 9 of 10 vectors came back defended with PoC evidence: path
traversal variants (`../`, absolute, `..\`, URL-encoded, null bytes — IDs never
reach the filesystem); symlink TOCTOU (16,731 file↔symlink swaps: 6,480 clean
loads, 14,791 sanitized `ValueError`s, zero unexpected exceptions, zero hangs);
malicious fixtures (200k-deep nesting, duplicate keys, `NaN`/`Infinity`, 50 MB
strings, invalid UTF-8 — all rejected; control chars and lone surrogates flow
safely via `ensure_ascii` JSON and HTML escaping); HTML injection round 2 (12
hostile payloads through the full pipeline: zero raw payloads in `report.html`;
CSP verified byte-for-byte; `frame-ancestors` cannot be expressed via meta tag
— noted limitation); concurrent runs (atomic `mkdir(exist_ok=False)` + `O_EXCL`
prevent corruption; loser fails fast); resource exhaustion (caps exact,
cooperative cancellation measured at 0.41 s into a ~18 s run, flat memory);
CLI edge cases (`/`, `/dev/null`, dangling symlinks, fullwidth digits, 5000-digit
ints — all clean argparse errors); tempfile hygiene (zero `tempfile` usage;
reports written directly with `O_EXCL`). Two genuine findings, both fixed:

| ID | Severity | Finding | Resolution |
|----|----------|---------|------------|
| U1 | Low–Medium | Unicode normalization evasion: NFD/NFC variants of the same visual string (e.g. `café` vs `café`) evaded the fixture-ID uniqueness and cross-split identical-message checks, allowing silent held-out contamination. | Fixed: `Split.__post_init__` compares NFC-normalized keys for IDs, groups and message contents; stored fixture data stays byte-exact so the exact-token-sequence principle is preserved. Homoglyph/confusable detection is out of scope (no normalization form unifies those) and is documented in `core.py`. Regression tests cover ID, message and group variants. |
| T1 | Low | Terminal escape injection: `inspect`/`run` interpolated raw fixture IDs into stderr warnings, so a hostile fixture file could rewrite the operator's terminal title or clear the screen. | Fixed: control characters (C0/C1) are stripped from IDs before display in both warnings (`cli._display_id`). Regression test asserts no ESC bytes on stderr. |

`doctor`'s machine-readable output was additionally reviewed: checkpoint-controlled
strings there are always emitted through `json.dumps` (`ensure_ascii=True`),
which neutralizes terminal escapes — no change needed.

### Red-team round 3 (2026-10-13)

A third adversarial round targeted the five new round-3 capabilities plus
re-verification of every documented command. Novel vectors only; round-1/2
vectors were covered by the regression suite. Five genuine findings, all
fixed with regression tests:

| ID | Severity | Finding | Resolution |
|----|----------|---------|------------|
| H3-1 | High | `_hf_overhang_match` (the `StopStringCriteria` port behind the stop matrix) missed the tensor implementation's end-overlap class where the stop string ends *inside* the final token (e.g. token `" world"`, stop `"or"`; token `"xstop"`, stop `"stop"`). The port only tried prefix overhangs (`stop[-i:] == token[:i]`), so the stop matrix systematically under-reported HF stops. Found by differential testing against the real `transformers` criterion (18/300 random trials diverged, all lab-miss-real-fire). | Fixed: the port now mirrors `_stop_string_get_matching_positions` exactly (every end-overlap from dropping `k` trailing chars of the final token, then the valid-position walk backwards). Re-verified: 1500/1500 random trials agree on a byte-level BPE tokenizer, plus an exhaustive 256-combination scan on a second vocab with zero divergences. Residual (documented): token texts come from per-token `decode` while HF matches on cleaned vocab strings / raw bytes, so UTF-8-fragment tokens can still diverge — unfixable through the `Backend` protocol. |
| H3-2 | Medium | `template_diff` is super-quadratic on adversarially fragmented inputs: an alternating equal/divergent token pattern of just n=2000 takes ~45 s (≈cubic scaling — ~6 min at n=4000), hanging `template_conformance` per (format, fixture) pair. Reachable via a hostile checkpoint template + hostile fixture. | Fixed: the difflib matching-blocks loop is replicated with a 10 s wall-clock budget checked between `find_longest_match` calls (each call is linear, so the total is bounded); on expiry it raises `ValueError("template_diff_too_fragmented")`, which the existing per-pair handler degrades to `"unavailable"`. Output is byte-identical to unbounded difflib on 400 randomized inputs; ordinary 20K-token diffs still complete in ~0.1 s. |
| H3-3 | Medium | `doctor`'s reference-model path (`_portability_check`) and `template_conformance` loaded `tokenizer.json` through `PreTrainedTokenizerFast` with **no** 128 MB pre-load cap — the round-1 fix applied only to `LocalHFBackend`. A swapped-in oversized `tokenizer.json` in a reference checkpoint is parsed into memory uncapped (PoC: 200 MB sparse file triggered a real loader call). | Fixed: new `_tokenizer_file()` helper enforces the cap before every HF loader call site (reference portability, primary deep check, conformance); oversized files degrade to the fixed `"skipped"`/`"unavailable"` codes without parsing. Sparse-file regression tests assert the loader is never invoked. |
| H3-4 | High | A hostile Jinja `chat_template` in `tokenizer_config.json` renders unbounded text: `{{ 'x' * 100000000 }}` OOM-killed the `template_conformance` worker (exit 137) during render/encode — no exception, the process dies. | Fixed: the template source is capped at 1 MB chars (`"unavailable"` / `"chat_template_too_large"`) and every rendered output at 1 MB chars (pair degrades to `"unavailable"` instead of encoding the bomb). Regression tests assert both degradations and that the loader/encoder never sees the bomb. Residual: CPU burn *inside* the Jinja render itself is the same exposure as HF's own `apply_chat_template` — run unfamiliar checkpoints with resource limits (see trust boundaries). |
| H3-5 | Medium | The round-3 test file broke the `[dev]`-only suite: 7 test classes import `tokenizers`/`transformers` at test time with no skip guard, so the README quick-start path (`pip install -e ".[dev]"`) errored instead of skipping. (Round 2 had validated that path green before `test_round3.py` existed.) | Fixed: `HAS_HF` module guard + `@unittest.skipUnless` on the seven HF-dependent classes, restoring `[dev]`-only to skip-clean. |

Probed and verified defended (PoC evidence, no change needed):

- **XSS via hostile candidate names:** 5 hostile payloads (`<script>`, `<img onerror>`, `<svg onload>`, entity/quote breaking, `{{7*7}}`) through `run_experiment` → `save_report` → `render_html`, plus hostile names injected into all four new sections (conformance, stop matrix, sweep, taxonomy banner): zero raw payloads in `report.html`; `{{7*7}}` renders as inert text (no HTML metacharacters). New regression test pins this.
- **`doctor --reference-model` traversal/symlink:** the asset name is the fixed literal `"tokenizer.json"` (`_member` rejects `..`/separators/symlinks); a symlinked `tokenizer.json` → `"skipped"`; a symlink loop → `"skipped"` (no hang); a symlinked *directory* resolves per operator choice and works. No traversal possible by construction.
- **`truncate_text_tail`:** 20 adversarial cases (multi-byte splits, emoji, flag grapheme clusters, combining marks, empty/negative/bool inputs, lone-surrogate input): never emits lone surrogates (decode either succeeds to a valid `str` or the cut degrades to `None`); lone-surrogate input raises `UnicodeEncodeError`, a `ValueError` subclass, consistent with the contract. Grapheme-cluster splits stay at codepoint granularity as documented.
- **`prompt_sensitivity_sweep`:** compute bounded (`max_drop_bytes` 1–8 enforced in CLI and function; depth count == bound); pre-cancelled sweep returns in 0.00 s (cancellation honored through every `run_experiment`); codepoint-splitting depths record the documented `"failed"` decision.
- **`failure_taxonomy`:** 17 adversarial observation shapes (backend-error priority, cancelled grouping, mismatch-before-EOS priority, corrupt `output_ids`, empty dict, falsy non-bools) all classify as documented; non-dict input rejected.
- **`check_boundary_portability`:** raising/lying `secondary_encode` callables degrade to the fixed `secondary_encode_failed` / `secondary_encode_invalid` codes, never raise.
- **vLLM emulation vs the real `vllm/v1/engine/detokenizer.py`:** the `find` window, earliest-completion selection, `bisect_left` truncation and `stop_buffer_length` semantics all match the fetched upstream source; the `new_char_count == 0` early-return is now ported too -- the old "cannot observably diverge" claim was wrong: for single-character stops (e.g. `"\n"`) the window start `1 - len(stop)` is 0 on a zero-new-char step, so a stop completed during the `min_tokens` gate was refound where upstream stays silent (found by the `tests/test_cycle14_holdback.py` differential harness, fixed 2026-10-15). Approximation documented: per-prefix batch decodes vs incremental `DecodeStream` (can differ on UTF-8 fragments), `min_tokens=0` assumed.
- **Old surfaces:** fixture-loading hostile inputs, report-dir permissions and the sanitized error vocabulary re-verified via the full suite (361 tests); new code paths emit only fixed-vocabulary details (int step counts, int ID lists) — no raw checkpoint/backend text.

#### Red-team round 4 — final pre-release sweep (2026-10-16)

Three parallel adversarial agents probed novel surfaces only (rounds 1–3
vectors were covered by the regression suite): the `model-protocol-lab debug` REPL + CLI
input handling, `doctor`/`examine_model` hostile checkpoints, and the
scoring / report-diff / backend-emulation modules. Eight genuine findings,
all fixed with regression tests in `tests/test_cycle96_redteam_sweep.py`:

| ID | Severity | Finding | Resolution |
|----|----------|---------|------------|
| R4-1 | Medium | Lone surrogates (`\ud800`) in hostile `results.json` / fixture content crashed strict stream writes: `model-protocol-lab diff-reports` died with uncaught `UnicodeEncodeError` on stderr, and the debug REPL session died (violating its "never die" invariant). The `_CONTROL_CHARS` sanitizers did not cover `\ud800–\udfff`. | All four `_CONTROL_CHARS` regexes (`debug.py`, `cli.py`, `diffreports.py`, `fixtures.py`) now strip lone surrogates; astral characters (single codepoints) are unaffected. |
| R4-2 | Medium | `BaseException` from a misbehaving backend escaped the sanitization boundary: `core.evaluate` (`except Exception` around `backend.generate`) and `scoring.score_target_logprobs` (`except Exception` around `forward_logits`) propagated, turning an hours-long run into a total loss instead of one failed observation. | Both sites now catch `BaseException` and map to the fixed codes (`generate_failed` / `forward_failed`), with explicit re-raise of `KeyboardInterrupt`/`SystemExit` so Ctrl+C keeps working. |
| R4-3 | Medium | Raw `RecursionError` escaped `examine_model` via a deeply-nested safetensors header (`doctor._check_weight_headers`): `json.loads` was wrapped in `except (UnicodeDecodeError, json.JSONDecodeError)` only, unlike `_read_json` which also catches `RecursionError`. | Added `RecursionError` to the except tuple, mirroring `_read_json`. |
| R4-4 | Medium | FIFO-swap hang: `_read_json`, `_manifest` (backends.py) and `_check_weight_headers` (doctor.py) used plain blocking `open()` after validation, so a FIFO swapped in for an asset hung the opener forever. The race window spans the whole weight-hashing phase. | New shared `_open_regular_read()` helper: `O_NONBLOCK` open + `fstat` `S_ISREG` check on the fd (the `datasets.py` pattern); `_manifest` now fstats the opened fd for its before-snapshot. All three call sites fail closed with fixed vocabulary. |
| R4-5 | Medium | Jinja chat-template memory bomb: `template_conformance` called `apply_chat_template` which materializes the full rendered string *before* the 1MB `_RENDERED_MAX_CHARS` check -- a 26-byte `{{ 'x' * 500_000_000 }}` template grew RSS by 1.9GB first (larger multipliers OOM-kill). | The render now runs under a transient `RLIMIT_AS` (current virtual size + 256MB headroom): a bomb raises catchable `MemoryError` -> `"unavailable"` with zero RSS growth; legitimate renders unaffected. Falls back to unbounded render on non-Unix platforms. |
| R4-6 | Low–Medium | `portability_matrix` raised raw `OSError` on an unreadable second checkpoint (`_read_json` wrapped in `except ValueError` only), breaking its "never raises for checkpoint problems" contract for direct API users. | Now catches `OSError` -> `"unavailable"` / `"missing_or_unsafe_asset"`. |
| R4-7 | Low | Unbounded checkpoint-controlled `model_type` interpolated into machine-readable output (`f"unsupported model_type: {model_type!r}"`): a 5MB `model_type` produced a 5MB detail string, breaking the fixed-vocabulary discipline. | Preview truncated to 200 chars. |
| R4-8 | Low | Quadratic transcript accounting in the debug REPL: `onecmd` re-summed the entire transcript on every command (~20s for a 10k-line script at the advertised caps). | Running `self._transcript_chars` counter, recomputed only on truncation. |

Probed and verified defended (PoC evidence, no change needed): `.read`
recursion/nesting (14 sub-probes), hostile renders (escape stripping at
byte level), `.save` path handling, `_parse_id_list` strictness, the 128MB
tokenizer cap on the debug path, argparse edges on all subcommands,
`_build_primary_tokenizer` hostiles, symlink/special-file games,
`examine_dataset` hostiles, scoring NaN/inf/empty/shape edges,
`diffreports`/`diffbackends` hostile inputs (40MB files, 10k nesting,
duplicate keys, FIFOs), SGLang/HF/vLLM emulation bounds, and
`experiments.py` cancellation/corruption handling.

#### Red-team round 5 — wave-2 adversarial audit (2026-10-18)

Six read-only specialists (security red-team A, adversarial red-team B
tasked with *disproving* A's findings, cross-verifier, perf, researcher,
docs-coherence) audited the cycles 134–144 feature code with a mandatory
cross-check protocol. 10 CONFIRMED findings; the builder re-verified
each against the current tree before fixing (cycle 145):

| ID | Severity | Finding | Resolution |
|----|----------|---------|------------|
| C1 | High | `margin_trajectory` still O(B·C) quadratic: the point cap bounds total points, not candidates — C candidates × 1 point each projected to ~1–2h per `diff-reports` pair on a hostile few-MB payload (measured ~4.1× per doubling by two agents). | Restructured to an O((E+B) log (E+C)) event sweep with a lazily-deleted leader heap; 0/1400 differential mismatches vs the old sweep; C=2000 in 0.008s (was ~1.4s). Perf regression test pins subquadratic scaling. |
| C2 | Low | Raw `old_decision`/`new_decision` interpolation injected terminal escapes into stderr in the transfer/robustness/perturbation drift renderers, falsifying `render_summary`'s sanitization comment. | All three wrapped in `_display_id`; sanitization tests pin escape-stripping. |
| C3 | Low | Non-string decision/winner/tier values crashed `render_summary` with `TypeError` (`_display_id` regex-sub on non-str), contradicting the degrade-gracefully contract. | `_display_id` is now total (None → "(none)", others via `str()`); non-string payloads render instead of crashing. |
| C4/C5 | Low | `diff_transfer`/`diff_contamination` `changed` flags had unpinned disjuncts (cell-drift-only, tier-escalation-only mutants survived the suite); same gap class in `diff_perturbation`. | Single-cause pinning tests for every disjunct; all five mutants re-run and killed. |
| C6 | Cosmetic | `checkpoint_drift` renderer uncapped. | Capped at 10 with "... and N more", like `cell_drift`. |

Disputed as security findings (real patterns, proven unreachable —
JSON keys are always strings; in-repo callers pass str-keyed data):
non-string dict keys in HTML sections, missing key-type check in
`diff_boundary_audit`. Template firewall note (F5): the static AST
scan misses concatenated-dunder subscripts by construction; six bypass
payloads were thrown at the stack and all died in
`ImmutableSandboxedEnvironment` + jinja2 3.1.6 — the "denies the RCE
primitives" docstring now describes the static layer's actual scope.

New security-relevant features since round 4, all covered above and by
their own test suites: the **template conformance firewall** (cycle
136: static Jinja AST scan denying dunder access, `|attr`, dangerous
globals before any render; `doctor.template_conformance` refuses
denied templates fail-closed), **chat-marker isolation conformance**
(cycle 135: hostile marker strings must not become control tokens),
and the **cycle-141 hostile-input hardening** (three `TypeError`
crashes on non-string dict keys fixed fail-closed).

## Proven safe (adversarial PoCs, observed not inferred)

- **XSS:** hostile payloads in every dynamic `render_html` field render fully
  neutralized (`html.escape`, `quote=True`), including inside `data-sort`
  attribute values; CSS classes from internal dicts; CSP `default-src 'none'`
  with `script-src` allowlisting only the SHA-256 hash of the single fixed
  inline table sorter; no external assets, no other `<script>`.
- **Path traversal:** `_member` rejects `..`/separators/symlinks (regex
  `^[A-Za-z0-9_.-]+$` + `is_symlink`/`is_file` checks); shard names constrained
  to `[A-Za-z0-9_.-]+\.safetensors`; `save_report` writes only fixed filenames.
- **ReDoS:** the two regexes are linear char classes (100 kB near-miss in ~1 ms);
  no nested quantifiers or alternation.
- **Deserialization:** `NaN`/`Infinity`/duplicate keys rejected; 2 MB cap before
  `json.loads`; no `pickle`/`yaml`/`marshal`/`eval`/`subprocess` in the package.
- **Supply chain:** the only network-adjacent calls are two `from_pretrained`
  (`local_files_only=True`, `trust_remote_code=False`, `token=False`), one
  `PreTrainedTokenizerFast(tokenizer_file=…)` (local file), and
  `importlib.metadata.version()` (local). No `hf_hub_download`,
  `snapshot_download`, sockets, or HTTP clients anywhere. This is the
  structural defense against the CVE-2026-80047 class (transformers writing
  remote repository code to the local module cache before the trust-consent
  check): the lab never loads remote repositories and never enables
  `trust_remote_code`, so the vulnerable code path is unreachable. Pinned by
  `tests/test_cycle117_local_loading_invariant_audit.py` (AST guards:
  no `trust_remote_code=True`, every `from_pretrained` passes
  `local_files_only=True`, no download helpers, no truthy `token=`).
  **Deserialization:** the model-weight `from_pretrained` additionally passes
  `use_safetensors=True` (transformers never touches the pickle unpickler)
  and `weights_only=True` (defense in depth if a fallback ever engaged),
  and no `torch.load` / `pickle.load(s)` appears in the package. This is the
  structural defense against the CVE-2025-32434 (torch <2.6.0) and
  CVE-2026-24747 (torch <=2.9.1, fixed 2.10.0; CVSS 8.8 — the
  `weights_only=True` unpickler itself was bypassable) classes. The
  installed torch 2.14.1+cpu is not affected by either, but the lab's
  defense does not depend on that: pinned by
  `tests/test_cycle119_pickle_deserialization_invariant_audit.py`
  (AST guards: no `torch.load`/`pickle.load(s)`; every model-weight
  `from_pretrained` passes `use_safetensors=True` and `weights_only=True`).
- **TOCTOU:** `_manifest` hashes every asset before and after HF loading and
  compares size/mtime/ino + sha256; swaps fail closed.
- **Cancellation:** the generate `Event` is checked before lock acquisition, in
  the lock-retry loop, and between forward steps; all public caps are hard.

### Residual risks (accepted)

1. torch's `+cpu` local tag stays invisible to `pip-audit`; its CVE posture
   needs manual re-verification on every bump.
2. No hash-pinned lockfile: fresh installs trust PyPI within the declared
   ranges; mitigated by the CI `pip-audit` gate. Partially tightened
   2026-10-23 (cycle 201): the `hf` extra now caps `torch` below its next
   major (`torch>=2.6,<3`, matching the existing `transformers<6` cap), so
   the largest dependency can no longer cross a major release silently;
   the lockfile gap itself remains accepted as recorded here.
3. ~~`_manifest` reads every weight file twice per run (before/after init) —
   hundreds of GB of I/O on very large checkpoints; documented, not a defect.~~
   **Retired 2026-10-17 (cycle 115):** the backend now hashes each weight
   file once and verifies a cheap stat fingerprint
   (`st_size`, `st_mtime_ns`, `st_ino`) after the load; the double full-hash
   is gone. Documented trade-off: a content swap preserving size,
   nanosecond mtime *and* inode inside the load window would evade the
   stat check — catching it needs the full re-hash.
4. ~~Runtime network isolation of the adapter has not been verified by test
   (code review only).~~ **Retired 2026-10-12:** verified dynamically under a
   socket firewall (zero `connect` attempts across `doctor`, backend init and
   generation); guarded by
   `test_no_network_access_during_doctor_and_backend_init`.
5. The HF stop emulation matches on per-token `decode` strings while real HF
   matches on cleaned vocab strings / raw bytes: UTF-8-fragment tokens can
   make real HF fire where the emulation does not (verified: no divergence
   on 1500 random byte-level trials or an exhaustive 256-combination scan
   outside the fragment case). Unfixable through the `Backend` protocol;
   documented in `docs/USAGE.md` §8.2.
6. A hostile Jinja chat template can burn CPU inside the render itself
   (transformers' sandbox restricts code execution, not resource use).
   Template source and rendered output are capped at 1 MB each (round-3
   fix); on Unix, in the main thread, each render additionally runs under
   a 10 s SIGALRM wall-clock budget (`_render_with_time_budget`) and
   aborts with `_RenderTimeBudgetExceeded`, degrading the pair to
   `"unavailable"` (cycle-116 fix). Residual: off the main thread or on
   non-Unix platforms the watchdog cannot be installed and the render
   runs unbounded -- the same exposure as HF's own `apply_chat_template`.
   Run unfamiliar checkpoints with resource limits. (Re-verified
   dynamically in the 2026-10-22 oversight audit: rendering a
   CPU-bomb template (a `{% for %}` loop over a 200k range, ~1.6e10
   inner steps) from a worker thread left the render still running
   after 16 s while the calling thread proceeded — confirming the
   watchdog never arms off the main thread, exactly as stated here.)
7. `template_diff` degrades (not hangs) after a 10 s budget per
   (format, fixture) pair on adversarially fragmented inputs; a fully
   hostile checkpoint can still cost ~10 s per pair before each degrades.
8. Invisible format characters (zero-width space, BOM, soft hyphen, word
   joiner, ...) survive the terminal sanitizers by design: they cannot
   reorder text or inject controls, so they are preserved for diagnostic
   fidelity. Residual: lookalike fixture IDs (`admin` vs `admin\u200b`)
   render identically in listings and diffs -- verify IDs byte-wise before
   acting on them.
9. Unicode bidi controls survive into `report.html` text: HTML-escaping
   keeps them inert (no tag breakout, no script execution; verified
   byte-level), so at most the operator's own local report could
   visually spoof ordering. The terminal path strips them. Accepted.

## Data disclosure

No network reporting or telemetry is implemented in this package. Dependency behavior has not been audited. Default report JSON excludes raw observation arrays, but names, format strings, fingerprints, counts and decisions remain. Evidence mode retains answer text and input/output token IDs. Token IDs are not anonymization.

Do not publish actual prompts, outputs, datasets, weights, local paths, credentials, account configuration or dependency diagnostic logs without reviewing them. The source archive contains synthetic examples only. `.gitignore` is a convenience, not a secret scanner or access-control boundary.

The report renderer escapes displayed strings and uses a restrictive content-security policy. Browser behavior and hostile-content rendering still require testing. Do not expand it to render model output as HTML or execute supplied JavaScript. The one inline script is fixed, author-controlled and hash-pinned in the CSP; it never touches model-derived text.

## Operational limits

Current safeguards limit fixture size, candidate count, token output and configured context. They do not guarantee bounded memory, runtime or GPU utilization. Model loading can be expensive before evaluation starts. Cancellation does not preempt a running native operation.

Do not expose the library or CLI directly as a multi-user network service. Authentication, tenant isolation, safe job scheduling, audit logging and resource enforcement are not implemented.

## Supported versions

Security fixes ship on the latest release line only; older lines are
not backported. Supported Python versions are those in
`pyproject.toml`'s `requires-python` that CI exercises (currently
3.11, 3.12, 3.13). No response-time commitment is made for reports;
the private reporting route below is the channel, and fixes land on
the latest line.

## Verifying a release

Releases are built reproducibly and published with PEP 740
attestations (see `.github/workflows/release.yml`). To verify an
artifact you downloaded:

```sh
gh attestation verify <artifact> --repo strmt7/model-protocol-lab
```

To reproduce the artifacts yourself, build twice from the tagged
tree with the tagged commit's timestamp as the epoch and compare
SHA-256 hashes with the published assets:

```sh
EPOCH=$(git log -1 --format=%ct <tag>)
python tools/release_build.py --source . --output /tmp/mpl-verify-dist --epoch "$EPOCH"
```

The base wheel installs zero dependencies — check its METADATA:
every `Requires-Dist` line is gated on an extra (`extra == "hf"`
or `extra == "dev"`), so a plain `pip install` pulls nothing.
The `[hf]` extra is declared as version ranges in
`pyproject.toml`, not a lockfile;
inventory your resolved set after installing it (for example with
`pip freeze`) rather than trusting any bill of materials the
release does not ship.

### Why no SBOM is shipped (cycle 210 adjudication)

No SPDX/CycloneDX bill of materials is attached to releases, by a
twice-considered decision (cycle 208 S3, re-adjudicated on the
procurement merits in cycle 210). A bill of materials for this wheel
would contain exactly one package — the wheel itself — because it has
zero runtime dependencies; the inventory information it would carry is
already inside the artifact as METADATA, verifiable with
`unzip -p dist/*.whl '*/METADATA'`. A bill of materials for the
`[hf]` resolved set would be stale on arrival (version ranges) or
would require the multi-GB CUDA download in the release job. And any
SBOM step would live in the tag-only release job, which never runs
until the first real release — adding a blocking third-party step
there, unexercised, is the wrong trade. This is revisited only at the
first release rehearsal, once the publish path has actually run.

### Workflow security and dependency review (cycle 215)

The workflows hold the OIDC grants (PyPI publishing, Codecov,
Pages), so their own hygiene is enforced, not just practised.
Workflow permissions are job-scoped: a top-level block grants at
most `contents: read`, and every write grant sits on the one job
that needs it, with a comment saying why. A dedicated lane lints
`.github/workflows/` with zizmor (pinned at 1.31.1, pedantic
persona, lowest severity and confidence floor, offline) on every
push and pull request that touches a workflow; run locally with
the exact CI command over the tree this cycle fixed, it reported
no findings — the fourteen findings the same command produced
before the fixes (five workflow-level write grants, five
undocumented grants, one template expansion inside a run script,
two unnamed jobs, one missing concurrency limit) were each fixed
in the workflows themselves, with no suppressions: there is no
zizmor config file and no inline ignore anywhere in the tree.

Dependency changes get a second, separate gate: a dependency
review runs on pull requests only (never on pushes, so it cannot
block the push pipeline) and fails the check when a pull request
would introduce a vulnerability of severity high or above in the
runtime or development scopes, or a dependency under a copyleft
license (the GPL, AGPL and LGPL families). The shipped package has
zero runtime dependencies, so in practice the gate watches the
dev/docs tooling and any future dependency. Two limits are the
action's own, stated here so the gate is not over-read: a
dependency whose license cannot be detected is reported but does
not fail the check, and the deny-licenses option is deprecated
upstream in the pinned v5.0.0 (functional there; a future major
bump must re-decide the license posture rather than inherit it).

## Reporting defects

For public issues, provide a minimal synthetic reproduction and dependency versions. Exclude credentials, proprietary prompts, private model files and personally identifying data. For sensitive defects, use GitHub's private vulnerability reporting for this repository — it is enabled (verified through the GitHub API, 2026-10-23): open the repository's **Security** tab and choose **Report a vulnerability**, or go to `https://github.com/strmt7/model-protocol-lab/security/advisories/new`. Reports there are visible only to the maintainer until a fix ships.
