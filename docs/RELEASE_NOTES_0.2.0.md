# Release notes — model-protocol-lab 0.2.0

**Status: DRAFT notes.** Prepared 2026-10-31 (cycle 226) for the
0.2.0 version story; extended 2026-11-02 (cycles 227–228) and
2026-11-09 (cycles 229–241). These notes are pending the owner's
release tag; no tag has been created and nothing has been
published. The release workflow uses this file for the release
notes when its version matches the tag, and falls back to a
fixed template when no such file exists.

## What this is

Model Protocol Lab measures whether a local language model fails
because of its weights or because its conversations are formatted
incorrectly. It renders a 29-family catalog of text-only
chat-format hypotheses (plus bounded BOS, prefix and thinking-mode
mutations) as exact token sequences, sends them to a local
checkpoint, and compares task answers and response boundaries
token-for-token. A strict search/held-out fixture discipline
selects a single winning format or abstains — no leaderboards.

## What's new in 0.2.0

Everything since the 0.1.0 notes (2026-10-16), which described a
tree that was never tagged. The headline additions:

- **Probe instruments beyond the format catalog**, each with a
  frozen protocol and a diff surface: surface-noise,
  template-robustness, record-notation, canonical-form,
  invisible-chars and confusable-skeleton probes, plus experiment
  bundle packing and verification (`pack` / `verify --bundle`).
- **An MCP server** (`model-protocol-lab-mcp`) exposing the probe
  and control tools to MCP clients, with JSON-RPC error mapping
  that separates client input errors from server failures.
- **A findings-integrity pass** (cycle 225): code-scanning alerts
  fixed in code rather than dismissed, and every suppression
  annotation withdrawn so the remaining analyzer findings stand
  open and enumerated.
- **Production-readiness corrections** (cycle 226): two research
  citations that did not survive primary-source verification were
  removed from the research log with their downstream inferences;
  the remaining lagging JSON parsers were brought under the same
  strict discipline (depth cap, no NaN/Infinity, no duplicate
  keys, no lone surrogates) as their hardened siblings; and the
  CLI surface gained one documented exit-code table.
- **The deferred correctness findings closed** (cycle 227): every
  correctness finding deferred from the readiness pass was fixed
  in the producer and its test oracle together, each with a
  reproducer that failed before the fix and passes after —
  manifest validation of non-finite values, MCP error-code
  discipline for malformed versus unprocessable input, digest
  verification failing closed at five sites, and the diff
  reports' remaining partial-equality seams.
- **Release machinery and tooling integrity** (cycle 228): the
  PEP 517 build backend is pinned (`setuptools==84.2.1`) and the
  canonical sdist normalizes tar member modes, so identical
  content builds byte-identically under any umask; the crawl4ai
  and CocoIndex tool lockfiles are now complete hash-pinned
  closures like the docs and CI locks, each proven by executed
  `--require-hashes` installs; the contamination sweep
  runs under a 10-second deadline budget and reports a partial
  prefix instead of consuming unbounded time on large corpora;
  and the CLI's test signal hook can no longer truncate an
  existing file (exclusive-create open).
- **DeepSource fix-down waves** (cycles 229–230): 559 analyzer
  occurrences fixed in code across two waves, verified by local
  proxy analyzers and whole-tree rescans (DeepSource itself
  produced no analysis run for these heads). The derived
  standing-open remainder (352) is enumerated per code in
  `docs/SECURITY.md`; nothing was dismissed, ignored, or
  suppressed.
- **Three more frozen probes** (cycles 231, 233, 237):
  compatibility-form substitution (`--compatibility-form`),
  enumeration convention (`--enumeration-convention`), and
  typographic emphasis (`--typographic-emphasis`) — the last is
  the lab's first equivalence-certifying instrument: when its
  certificate event holds, an arm is labelled
  `equivalence_certified` at δ = 0.34, with the completed bound
  and the family-wise price printed beside every label. A
  producer/mirror truthiness asymmetry the bug hunt found in
  the emphasis decomposer was fixed in cycle 238.
- **Docs truth-repair** (cycle 232): the README's comparative
  claims were re-verified against primary sources (26 tools)
  and narrowed to what the sources carry, and "pre-registered"
  was replaced by "frozen in advance" across the living
  documents — the rules are fixed in code before a run, but no
  independent registry is claimed.
- **Test-infrastructure hygiene** (cycle 234): the Unicode
  census reads only the vendored, digest-pinned
  `tests/data/UnicodeData.txt` (the workspace-local fallback
  behind the red foreign-interpreter CI legs of cycles 231–233
  is gone), and the debug signal tests pin the child's signal
  dispositions instead of relying on a timing window. The
  cycle's closing CI was the first fully green one since
  cycle 230.
- **Real-checkpoint evidence** (cycles 235 and 240): the lab's
  first measurements on real checkpoints. Four checkpoints
  were acquired with revision pins and SHA-256 digests (GPT-2
  124M/355M/774M and TinyLlama-1.1B-Chat) and run through a
  frozen battery: seven rows, 324 generations, every decision
  `no_candidate` (0/12 exact-and-boundary passes per candidate;
  the held-out split was never read), with the per-row failure
  taxonomy recorded in `docs/REAL_CHECKPOINT_VALIDATION.md` —
  including the measured boundary finding from the GPT-2 core
  battery (the configured boundary token sequence occurred
  0 times in 96 generations while a string-level stop would
  have fired in 95). One row (TinyLlama × llama3) is carried
  as a named remainder after six host-reboot kills; the gated
  Llama-3.2 rows wait on the owner's Hugging Face token, the
  3B row is RAM-excluded, and Gemma is adapter-excluded, each
  named in the page rather than silently dropped.
- **One-branch docs publishing** (cycle 236): the docs site
  publishes by GitHub Actions workflow and the `gh-pages`
  branch is deleted — the repository keeps exactly one branch,
  `main`. Deployment records minted by each publish are
  deleted immediately after verification under the standing
  duty.
- **Security-policy records** (cycle 239): `docs/SECURITY.md`
  records the 2026-11-06 re-verification of the 13 open
  filelock Dependabot alerts — the two-pin remedy is proven
  against litellm 1.109.0rc1 and lands when the first stable
  litellm ≥ 1.109.0 publishes; the alerts stand open and
  undismissed — and the closure record for Dependabot PR #7,
  blocked by sympy's `mpmath<1.4` cap and torch 2.15.0's exact
  `cuda-toolkit==13.2.2` pin, not by the filelock chain.
- **Stored-payload robustness** (cycle 241): the findings of
  bug-hunt waves 4–6, one family, fixed together. The diff
  and report mirrors and the HTML renderer now type-check
  stored payload values before consuming them, so a damaged
  or crafted results file degrades to the frozen
  `available: "neither"` vocabulary instead of crashing
  `diff-reports` or the renderer with a raw traceback; two
  stored-field grammars (the confidence mirror's final-line
  grammar and the bundle manifest's SHA-256 screen) now apply
  fullmatch semantics, and the distraction plant grammar was
  converted to fullmatch after a census proved no shipped
  fixture carries a newline-terminated expected value. Every
  fix is pinned by a regression test proven to fail on the
  pre-fix code.

## Verifying a release artifact

Releases are built reproducibly and published with PEP 740
attestations (see `.github/workflows/release.yml`). To verify an
artifact you downloaded:

```sh
gh attestation verify <artifact> --repo strmt7/model-protocol-lab
```

To reproduce the artifacts yourself, build twice from the tagged
tree with the tagged commit's timestamp as the epoch and compare
SHA-256 hashes with the published assets.
