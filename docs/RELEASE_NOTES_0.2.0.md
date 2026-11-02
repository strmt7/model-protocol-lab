# Release notes — model-protocol-lab 0.2.0

**Status: DRAFT notes.** Prepared 2026-10-31 (cycle 226) for the
0.2.0 version story; extended 2026-11-02 (cycles 227–228). These
notes are pending the owner's release tag; no tag has been
created and nothing has been published. The release workflow
uses this file for the release notes when its version matches
the tag, and falls back to a fixed template when no such file
exists.

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
