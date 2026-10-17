# Release-readiness report — model-protocol-lab 0.1.0

**Date:** 2026-10-16

> **Addendum 2026-10-17:** historical report — the residual risks § carries
> its original 2026-10-16 numbers and wording (e.g. 6083 tests); see
> `docs/SECURITY.md`, `VALIDATION.md`, and `CHANGELOG.md` `[Unreleased]` for
> the current state (4481 tests, both runners agree; SECURITY.md residuals
> updated through cycle 132). Residual items marked **Retired** below stay
> retired; the remaining risks are still accepted.

**Scope:** every item of the release-readiness definition of done, with
evidence (commands, dates, counts). Prepared by the autonomous
development coordinator across cycles 84–96. **No git tag has been
created** — tagging is the owner's decision.

## Checklist

### 1. Version number — DONE

`pyproject.toml` and `model_protocol_lab/__init__.py` both declare
`0.1.0` (cycle 84; previously `0.1.0a0`). The public API (CLI
subcommands, `core`/`doctor`/`reporting`/`scoring` function signatures,
`results.json` schema) has been stable across 90+ cycles; 0.1.0 is the
honest initial release number, not a 1.0 claim.

### 2. CHANGELOG — DONE

`CHANGELOG.md` finalized as `[0.1.0] — 2026-10-16` (cycle 95), Keep-a-Changelog
format: Added (catalog expansion 23→29 families, audit cycles 45–83),
Fixed (every genuine finding from cycles 84–96, including the eight
final-sweep findings below), Security (red-team rounds 1–4). No
aspirational entries; every entry names the cycle and the evidence.

### 3. README claims verified — DONE

- Test count: **6083 tests** collected
  (`python -m pytest --collect-only -q -p no:cacheprovider`, 2026-10-16;
  README line 16 and VALIDATION.md updated from 6063 after the final
  sweep added 20 regression tests).
- Coverage: 95% line gate enforced by CI (`coverage report
  --fail-under=95`) on every push.
- `ruff check`, `ruff format --check`, `mypy`, `bandit` clean — all CI-gated.
- No "early development" framing; no benchmark-superiority or
  unique-invention claims. `VALIDATION.md` lists what is *not*
  established (real-model runs beyond the synthetic checkpoint, browser
  rendering checks).

### 4. Packaging — DONE

`python -m build` + `twine check` run as the CI `build` job on every
push — green on all recent commits. Entry point `model-protocol-lab`
verified from an installed wheel path in the fresh-venv test (cycle 94).

### 5. Docs coherent — DONE

`docs/USAGE.md`, `docs/FAQ.md`, `docs/SECURITY.md`,
`docs/RESEARCH_LOG.md`, `VALIDATION.md`, `CHANGELOG.md`,
`docs/RELEASE_NOTES_0.1.0.md` — cycle 87 ran a docs-coherence audit
against the current code (all 8 CLI subcommands, every documented
command example executed). Cycle 96 updated SECURITY.md (round-4
findings) and the test-count claims.

### 6. Zero real TODOs — DONE

`grep -rn "TODO\|FIXME\|XXX"` over `model_protocol_lab/`: the only hits
are a quoted upstream Jinja comment, template-syntax documentation
("placeholder" as a term of art), and a documented explicit placeholder
family — no unfinished work in shipped code.

### 7. Release notes — DONE

`docs/RELEASE_NOTES_0.1.0.md` drafted (cycle 95): what the tool is,
what's new, install/upgrade, honest limitations. Marked DRAFT pending
the owner's tag.

### 8. Fresh-venv install test — DONE (cycle 94)

Fresh venv, `pip install .` (no extras): installs clean,
`__version__ == "0.1.0"`. All 8 CLI subcommands print `--help` from the
installed entry point. Functional smoke: `catalog` lists families;
`inspect` validates the example fixtures; `doctor` fails closed with a
structured verdict on a weights-less stub; `fixtures build` →
`inspect` round-trips; `diff-reports` on the sample report vs itself
reports no differences. Recorded in `VALIDATION.md`.

### 9. CLI smoke test — DONE (cycle 94)

Covered by the fresh-venv smoke above; `run`/`debug`/`diff-backends`
need a real tokenizer (`hf` extra) and are covered by the integration
tests against the locally generated tiny GPT-2 checkpoint.

### 10. Final red-team sweep — DONE (cycle 96)

Three parallel adversarial agents probed novel surfaces with executable
PoCs (debug REPL + CLI, doctor hostile checkpoints, scoring / diff /
emulation modules). **Eight genuine findings, all fixed** with 20
regression tests (`tests/test_cycle96_redteam_sweep.py`):

| ID | Severity | Finding |
|----|----------|---------|
| R4-1 | Medium | Lone surrogates crashed strict stream writes (diff-reports stderr, debug REPL session kill) — all four `_CONTROL_CHARS` sanitizers extended |
| R4-2 | Medium | `BaseException` from a misbehaving backend escaped the sanitization boundary (`core.evaluate`, `scoring.score_target_logprobs`) — now caught, mapped to fixed codes; `KeyboardInterrupt`/`SystemExit` still propagate |
| R4-3 | Medium | Raw `RecursionError` escaped `examine_model` via deeply-nested safetensors header — now caught |
| R4-4 | Medium | FIFO-swap hang in `_read_json` / `_manifest` / `_check_weight_headers` — new `_open_regular_read()` (`O_NONBLOCK` + `fstat` `S_ISREG`) |
| R4-5 | Medium | Jinja chat-template memory bomb materializing before the render cap (+1.9GB RSS) — transient `RLIMIT_AS` around the render; bomb now raises `MemoryError` with zero RSS growth |
| R4-6 | Low–Medium | `portability_matrix` raised raw `OSError` on unreadable second checkpoint — now `unavailable` |
| R4-7 | Low | Unbounded checkpoint-controlled `model_type` in output — truncated to 200 chars |
| R4-8 | Low | Quadratic transcript accounting in debug REPL — running counter |

Prior rounds (documented in `docs/SECURITY.md`): full audit 2026-10-12
(7 findings), red-team rounds 1–3 (1 high + 1 medium + 4 low; 2
findings; 5 findings incl. 2 high), per-cycle sweeps through cycle 95.
Total: every finding ever raised has a fix and a regression test.

## Verification state at report time

- Full pytest: 6083 tests collected; green (6056 passed, 7 skipped,
  68267 subtests on the pre-sweep baseline; 20 new regression tests all
  green).
- Exact CI command `coverage run -m unittest discover -s tests -t .`:
  green at cycle-95 baseline; re-run after final-sweep fixes in
  progress at report write time — the push waits for it.
- `coverage report --fail-under=95`: 97% on the sweep tree (above the
  gate).
- `ruff check`, `ruff format --check`, `mypy`, `bandit`: clean.
- CI on main: green on all six jobs for the cycle-95 commit
  (`d7b00935`); this report and the final-sweep fixes ship as the next
  commit, whose CI must go green before the cycle closes.

## Residual risks (accepted, documented in docs/SECURITY.md)

1. `torch`'s `+cpu` local tag is invisible to `pip-audit` (manual CVE
   re-verification on every bump).
2. No hash-pinned lockfile (CI `pip-audit` gate mitigates).
3. ~~`_manifest` reads weight files twice per run (I/O cost on very large
   checkpoints).~~ **Retired 2026-10-17 (cycle 115):** the backend now
   hashes each weight file once and verifies a cheap stat fingerprint
   after the load; the double full-hash is gone (documented trade-off in
   `docs/SECURITY.md`: a content swap preserving size, nanosecond mtime
   *and* inode inside the load window would evade the stat check).
4. Hostile Jinja templates can still burn CPU *inside* the render on
   off-main-thread or non-Unix platforms (same exposure as HF's own
   `apply_chat_template`); on Unix, in the main thread, each render runs
   under a 10 s SIGALRM wall-clock budget (cycle 116) and degrades to
   "unavailable" on expiry. Template source and rendered output are
   capped at 1 MB each; memory bombs are capped.
5. `template_diff` degrades (not hangs) after a 10s budget on
   adversarially fragmented inputs.
6. Do not expose the CLI as a multi-user network service (no auth,
   tenant isolation, or audit logging — documented operational limit).

## Decision for the owner

Everything above is done and evidenced. The remaining step is yours:
create the `v0.1.0` tag (and GitHub release from
`docs/RELEASE_NOTES_0.1.0.md`) when ready. No tag has been created by
the automation.
