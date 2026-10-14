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
| F1 | Medium | No lockfile/hashes; `torch>=2.6` unbounded — fresh installs trust PyPI tip-of-range. | Upper-bound discipline documented; `pip-audit` runs as a CI gate (`security` job) on every build; torch CVE posture tracked below. |
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
- **vLLM emulation vs the real `vllm/v1/engine/detokenizer.py`:** the `find` window, earliest-completion selection, `bisect_left` truncation and `stop_buffer_length` semantics all match the fetched upstream source; the `new_char_count == 0` early-return cannot observably diverge (an empty-decoding token cannot complete a non-empty stop, and a present stop would already have fired). Approximation documented: per-prefix batch decodes vs incremental `DecodeStream` (can differ on UTF-8 fragments), `min_tokens=0` assumed.
- **Old surfaces:** fixture-loading hostile inputs, report-dir permissions and the sanitized error vocabulary re-verified via the full suite (361 tests); new code paths emit only fixed-vocabulary details (int step counts, int ID lists) — no raw checkpoint/backend text.

### Proven safe (adversarial PoCs, observed not inferred)

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
  `snapshot_download`, sockets, or HTTP clients anywhere.
- **TOCTOU:** `_manifest` hashes every asset before and after HF loading and
  compares size/mtime/ino + sha256; swaps fail closed.
- **Cancellation:** the generate `Event` is checked before lock acquisition, in
  the lock-retry loop, and between forward steps; all public caps are hard.

### Residual risks (accepted)

1. torch's `+cpu` local tag stays invisible to `pip-audit`; its CVE posture
   needs manual re-verification on every bump.
2. No hash-pinned lockfile: fresh installs trust PyPI within the declared
   ranges; mitigated by the CI `pip-audit` gate.
3. `_manifest` reads every weight file twice per run (before/after init) —
   hundreds of GB of I/O on very large checkpoints; documented, not a defect.
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
6. A hostile Jinja chat template can still burn CPU inside the render itself
   (transformers' sandbox restricts code execution, not resource use) — the
   same exposure as HF's own `apply_chat_template`. Template source and
   rendered output are capped at 1 MB each (round-3 fix); run unfamiliar
   checkpoints with resource limits.
7. `template_diff` degrades (not hangs) after a 10 s budget per
   (format, fixture) pair on adversarially fragmented inputs; a fully
   hostile checkpoint can still cost ~10 s per pair before each degrades.

## Data disclosure

No network reporting or telemetry is implemented in this package. Dependency behavior has not been audited. Default report JSON excludes raw observation arrays, but names, format strings, fingerprints, counts and decisions remain. Evidence mode retains answer text and input/output token IDs. Token IDs are not anonymization.

Do not publish actual prompts, outputs, datasets, weights, local paths, credentials, account configuration or dependency diagnostic logs without reviewing them. The source archive contains synthetic examples only. `.gitignore` is a convenience, not a secret scanner or access-control boundary.

The report renderer escapes displayed strings and uses a restrictive content-security policy. Browser behavior and hostile-content rendering still require testing. Do not expand it to render model output as HTML or execute supplied JavaScript. The one inline script is fixed, author-controlled and hash-pinned in the CSP; it never touches model-derived text.

## Operational limits

Current safeguards limit fixture size, candidate count, token output and configured context. They do not guarantee bounded memory, runtime or GPU utilization. Model loading can be expensive before evaluation starts. Cancellation does not preempt a running native operation.

Do not expose the library or CLI directly as a multi-user network service. Authentication, tenant isolation, safe job scheduling, audit logging and resource enforcement are not implemented.

## Reporting defects

For public issues, provide a minimal synthetic reproduction and dependency versions. Exclude credentials, proprietary prompts, private model files and personally identifying data. For sensitive defects, use a repository's enabled private security-reporting route when available; do not assume one has been configured.
