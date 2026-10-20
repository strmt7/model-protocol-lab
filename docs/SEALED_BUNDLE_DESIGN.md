# Sealed result bundles (`pack` / `verify --bundle`)

**Utility (Monday morning).** An auditor asks "prove these result files are
unchanged since the run was packed." `model-protocol-lab pack` seals a report
directory into one hash-bound bundle; `model-protocol-lab verify --bundle`
re-checks the seal offline and then re-derives the experiment's decision from
the sealed `results.json` (the `verify` replay) as an independent second layer.

## What the seal proves — and what it does not

The seal is **hash-based and unsigned**. It proves exactly one thing: the
files in the bundle are bit-identical to the files that were packed, *when
the bundle digest is compared against a digest obtained through an
independent channel* (for example the hex printed by `pack` on stderr and
published where the result itself is published, or passed via
`verify --bundle --expect-digest`). A digest that only travels inside the
bundle authenticates nothing: anyone can edit the files and re-pack. The
seal also does **not** prove who produced the bundle, that the experiment
was actually run (a fabricated but internally consistent `results.json`
passes both the seal and the replay — the replay proves the recorded
decision follows from the recorded artifact, nothing more), that the
experiment was well-designed, or that the bundle is the *latest* one (there
is no freshness or rollback protection; the out-of-band record carries the
date and label). Terminology: this is a *hash seal* providing
*tamper-evidence* and *decision re-derivation*. It is not an attestation, a
signature, or tamper-proofing — those words imply a keyholder, and there is
none.

## Format `mpl-bundle/1`

A bundle is a single ZIP archive, written uncompressed (`ZIP_STORED`),
whose entries are sorted by name. Every `ZipInfo` field that affects bytes
is normalised (`date_time=(1980, 1, 1)`, `create_system=0`,
`external_attr=0o644 << 16`, no extra fields, no per-entry comments), so
rebuilding from identical inputs under the same package and interpreter
version produces byte-identical output. Verification guarantees *content*
integrity (per-file hashes plus the manifest digest below); it does not
require the container bytes to equal what the current `pack` would emit.

Entries:

- `manifest.json` — the seal manifest (below).
- One entry per sealed file, at its canonical relative path: the report
  directory's top-level files at their names, `--include` files under
  `inputs/<name>`. `pack` never emits directory entries.
- The archive comment is exactly `mpl-bundle-sha256:<64 lowercase hex>` —
  the SHA-256 of the exact stored bytes of the `manifest.json` entry.
  Verifiers hash the stored bytes once; they never re-serialise the parsed
  manifest to recompute the digest.

### Manifest

Canonical JSON (sorted keys, compact separators, `ensure_ascii`), exact
types throughout (`type(x) is int` / `is str`; booleans are not ints):

```json
{
  "context": {
    "backend_fingerprint": "<copied from results.json>",
    "catalog_sha256": "<sha256 of the canonical catalog definition>",
    "package_version": "0.1.0"
  },
  "files": [
    {"path": "report.html", "role": "report", "sha256": "<hex>", "size": 123},
    {"path": "results.json", "role": "results", "sha256": "<hex>", "size": 456}
  ],
  "format": "mpl-bundle/1"
}
```

- `files` never lists `manifest.json`; paths follow the canonical grammar:
  `/`-separated, no empty / `.` / `..` segments, no leading `/`, no drive
  prefix, no backslash, no control characters. Comparisons are exact-string.
- `role` is informational only — one of `results`, `report`, `input`,
  assigned by location. No verification decision branches on it; unknown
  roles and unknown top-level keys are accepted (forward compatibility).
  The replay target is the entry whose path is exactly `results.json`.
- `catalog_sha256` is the SHA-256 of the canonical JSON of the catalog's
  template definitions (`{family: [Format fields]}`), computed at pack
  time. It is a *recorded claim*: `verify` reports it but does not
  recompute it, exactly as for `backend_fingerprint`. The decision rule is
  bound by `package_version`: the replay semantics are those of the
  verifying package version, and the manifest records the packing one.

### Limits (enforced in this order, before content is read)

| Constant | Value | Enforced |
|---|---|---|
| Outer bundle size | 288 MiB | `fstat` on the single open fd, before reading |
| Entry count | 1024 | EOCD pre-parse, before `ZipFile` construction |
| Per-entry uncompressed size | 64 MiB | Central directory, before that entry is read |
| Total uncompressed size | 256 MiB | Central directory running total |
| Sealed `results.json` | 32 MiB (`MAX_RESULTS_BYTES`) | Shared bounded JSON reader |

The container is strict: the first local header is at offset 0 (no
prepended data), the file ends exactly at the end of the EOCD record (no
trailing data), the EOCD disk fields are zero, and ZIP64 sentinel
values in the EOCD are rejected (a format-1 bundle is far below ZIP64
sizes; needing it would be a format bump). The disk and ZIP64 checks
are EOCD-level — the pre-parse reads the end-of-central-directory
record, not a ZIP64 extended record or per-entry extra fields.
Duplicate archive entry names and duplicate manifest paths are rejected
before any entry data is read; all reads iterate `infolist()` entries and
stream in bounded chunks, asserting the hashed length equals the declared
size.

### Filesystem discipline

`pack` enumerates the report directory with `os.scandir` and opens every
file once with `O_RDONLY | O_NOFOLLOW | O_NONBLOCK`, requires `fstat` to
report a regular file, and hashes exactly the bytes it stores. Symlinks,
FIFOs, devices, and subdirectories are errors — evidence is never silently
skipped. `--output` may not live inside the report directory and may not
overwrite (created with `O_EXCL`); the default is the sibling path
`<report-dir>.bundle.zip`. `verify --bundle` opens the bundle with the
same single-fd discipline as `load_results`, so a FIFO cannot hang it.

### Verification layers and exit codes

`verify --bundle` report shape mirrors `verify_results`
(`{"verdict", "checks", ...}`). Seal checks are namespaced `seal_*`
(`seal_container`, `seal_manifest_schema`, `seal_comment_digest`,
`seal_file_set`, `seal_file_hashes`, optional `seal_expected_digest`);
replay checks keep their existing `verify_results` ids. If the seal
layer fails, the replay is not run and its checks are recorded
`unavailable` with detail `seal_failed`. A sealed `results.json` that the
plain `verify` path would call unreadable (duplicate fields, `NaN`,
oversize, wrong shape) is unreadable in a bundle too — same exit class,
same fixed vocabulary. Exit codes: 0 = seal ok and replay ok; 2 = seal or
replay failed; 1 = unreadable/invalid input. Callers must branch on the
JSON report, not the exit code alone. The stderr summary always prints
the scope line: `seal: integrity since packing only — origin not
authenticated`.

## Why no `diff-reports` section

A bundle is a container for one experiment artifact, not a measurement;
`diff-reports` compares two experiments' measurements. Bundle presence is
reported by `verify --bundle` itself. (Deviation from the generic
"every feature ships a diff section" default, recorded here deliberately:
a presence-only diff section would carry no measurement content.)

## Uniqueness (researched 2026-10-20, see `docs/RESEARCH_LOG.md`)

No mainstream evaluation harness ships result sealing. The combination —
an unsigned hash-sealed bundle binding results, inputs, catalog and
package version, verified offline with an independent decision replay as
the second layer, in a local deterministic harness — is the contribution.
Unsigned eval receipts exist elsewhere without the replay layer; provenance
metadata in other harnesses is unsealed. Academic and standards precedents
(SLSA, in-toto) are signing-based and are cited, not claimed.
