# Determinism self-audit — design contract

`model-protocol-lab verify --determinism --model <checkpoint> <report-dir>`
re-executes a recorded run's search observations against a locally loaded
checkpoint and reports whether the recorded output tokens reproduce
exactly. This document is the contract: what the audit claims, what it
refuses to claim, and the invariants the implementation is held to.
The audit lives in `model_protocol_lab/determinism.py`; `verify.py`
wires it into the `verify` command.

## Why this exists

Every lab result rests on one assumption: greedy generation on the
recorded checkpoint is deterministic, so the recorded `output_ids` are
the tokens that checkpoint produces for the recorded `input_ids`. Plain
`verify` re-derives the *decision* from the stored artifact (decision
replay); nothing re-checked the *generation* itself. The determinism
audit is the generation-replay counterpart: it turns the assumption
into evidence, one way or the other.

Monday-morning utility: a reviewer challenges a published result —
"prove the backend actually generated those tokens deterministically" —
and this command answers with a certificate or a per-cell divergence
table instead of an assertion.

## What the audit proves, and what it does not

- It proves a statement about **the re-execution**: in this audit run,
  with the checkpoint identified by the recorded backend fingerprint,
  every re-executed recorded search cell did or did not reproduce its
  recorded `output_ids` token-for-token.
- It does **not** prove the original run was deterministic. The
  environment may have changed since the original run; the audit
  observes only its own re-execution.
- A divergence is a **backend-determinism finding** — evidence about
  the backend, its numeric path, or its environment. It is not an
  instrument failure, not an artifact-integrity failure (plain `verify`
  covers artifact self-consistency), and not proof that the original
  run's *decision* was wrong: it shows the recorded tokens do not
  reproduce here, now, on this backend.
- The audit **detects** divergence; it does not diagnose the
  mechanism. The known mechanism in batched serving stacks is lack of
  batch invariance — reduction order in normalisation, matmul, and
  attention kernels varies with batch composition, so temperature-0
  outputs vary run to run (He, "Defeating Nondeterminism in LLM
  Inference", Thinking Machines Lab, 2025, DOI 10.64434/tml.20250910:
  1,000 greedy completions of one prompt produced 80 distinct outputs;
  batch-invariant kernels produced 1,000 identical ones). The lab's
  backend is sequential, batch size 1, full-prefix recompute, fp32 —
  that mechanism is largely absent in-process, so divergence under
  this backend points more at checkpoint, library-version, device, or
  context drift. The fingerprint gate (below) rules most of those out
  before any cell runs; what remains is a genuine numeric-path finding.
- Search observations only. Held-out observations are not re-executed:
  the search set is the selection evidence, and the bound keeps the
  audit's cost at or below the largest search phase the lab can record.
- Greedy only, like everything in the lab.

## The certificate

When every claimed cell reproduces, the audit emits a determinism
certificate. The word is a local label, not a term of art: the
certificate is **unsigned and self-issued**, exactly like the `pack`
seal. It states who ran it (the operator, on their own machine), what
it covers, and what it does not. It binds to the exact cells compared
via `cells_digest` — SHA-256 over the canonical JSON of
`(candidate, fixture, input_ids, output_ids)` for every attempted cell,
in recorded order — plus the backend fingerprint, the run seed, and
the token budget, so a certificate cannot be transplanted onto another
artifact or another cell set. It also states what is **not**
certified: the recorded prompts are replayed as stored; their
authenticity against the fixture files and format definitions is a
property of the original run, not of this audit.

## Invariants

- **I1 — one seed derivation.** The audit re-derives each cell's
  per-request seed through `core.request_seed_for`, the same function
  `core.evaluate` uses (extracted verbatim from the original inline
  derivation). Fixture IDs are used byte-exactly as recorded
  (`surrogatepass` encoding, no normalisation anywhere in the audit
  path): an NFC and an NFD spelling of the same visible ID are
  different IDs with different seeds, and the audit must not "help"
  by normalising. (`surfacenoise.py` carries a pre-existing private
  copy of the same formula for its own protocol; a drift-guard test
  pins the two derivations equal on edge-case IDs.)
- **I2 — fingerprint gate before any generation.** If the replay
  backend's fingerprint differs from the recorded
  `backend_fingerprint`, the audit is unavailable and **zero**
  `generate` calls happen. The fingerprint binds checkpoint assets,
  library versions, device, and context limit, so a mismatch means
  "a different experiment", never "nondeterminism".
- **I3 — recorded cells are validated before they are blamed.**
  A legitimately recorded completed cell satisfies hard invariants of
  the recording pipeline: non-empty `input_ids`, `output_ids` of
  exactly `max_new_tokens` (a completed generation reached the cap),
  integer non-boolean IDs, and — when the replay backend exposes a
  vocabulary bound — IDs inside the vocabulary. A cell violating any
  of these cannot have been recorded by this pipeline against this
  backend; it is an artifact defect and is reported as a skip, never
  as divergence and never as a backend error.
- **I4 — fail closed, in fixed vocabulary.** The audit never raises on
  a malformed or hostile artifact. Every defect lands in a documented
  detail code (below). Backend error text is never exported: replay
  errors are classified through the same allowlist `core` uses for
  recorded observations (`core._BACKEND_ERROR_CODES`); anything else
  collapses to `backend_reported_error`. A replay
  `context_limit_exceeded` is a per-cell skip
  (`replay_context_limit_exceeded`): with the fingerprint binding the
  context limit, a completed recorded cell cannot legitimately hit it,
  so it marks an artifact inconsistency, not a backend failure.
- **I5 — full denominators.** `n_cells` counts every recorded search
  observation. Skipped and errored cells are counted and listed, never
  dropped. A certificate requires every *claimed* (completed) cell to
  have been attempted and reproduced: any blocking skip (stripped
  entry, invalid/length-inconsistent/out-of-vocabulary recorded IDs,
  replay context limit, cell budget) or backend error caps the verdict
  at `inconclusive`. Cells the original run itself did not complete
  (`observation_not_completed`) are scope, not gaps: the certificate
  states their count.
- **I6 — a partial run earns no certificate.** If the artifact records
  an aborted search (decision `cancelled`/`failed`, or any search
  entry not aggregate-eligible), the recorded cells are still replayed
  and reported, but the verdict is capped at `inconclusive` with
  detail `search_incomplete`: the audit does not certify a selection
  process that never finished.
- **I7 — bounded cost.** At most `MAX_AUDIT_CELLS = 2400` cells are
  attempted — 24 candidates × 100 search fixtures, the largest search
  phase `run_experiment` accepts — so a foreign artifact (the loader
  admits up to 32 MiB) cannot turn the audit into an unbounded
  generation job. Every claimed cell is re-executed individually;
  duplicate prompts are **not** memoised, because a backend whose
  output depends on call count is precisely the nondeterminism this
  audit exists to catch.
- **I8 — plain `verify` is untouched.** Without `--determinism`,
  output is byte-identical to before. The audit section and its check
  exist only in the opt-in mode, and backend construction is skipped
  entirely when the artifact precheck fails, so a malformed artifact
  never triggers a checkpoint load.

## Verdict and check mapping

The audit verdict is one of `deterministic`, `diverged`,
`inconclusive`, or unavailable (`available: false` with a detail
code: `artifact_malformed`, `unsupported_schema_version`,
`observations_stripped`, `no_replayable_cells`,
`backend_fingerprint_mismatch`). Inside `verify`, the
`determinism_replay` check maps: `deterministic` → passed;
`diverged` → failed (the recorded outputs positively contradict the
re-execution — verify's standing doctrine for `failed`); any
`inconclusive` or unavailable outcome → unavailable, with counts in
the detail. Consequence, stated plainly: `verify --determinism` can
exit 0 while its determinism check is unavailable (for example when
the backend errors during re-execution). Exit 0 means no check
failed; the JSON `determinism` section is the audit's actual result
and must be read.

Per-cell outcomes: `reproduced`, `diverged` (with `first_divergence`,
`shared_prefix`, and both lengths), `error` (fixed detail codes:
`generate_failed`, `invalid_generation`, `backend_reported_cancelled`,
an allowlisted backend code, or `backend_reported_error`), and
`skipped` (`recorded_ids_invalid`, `recorded_length_inconsistent`,
`recorded_ids_out_of_vocab`, `observation_not_completed`,
`observations_stripped`, `replay_context_limit_exceeded`,
`audit_cell_budget_exceeded`). Cells carry entry index, candidate, and
fixture for identity, and never the token IDs themselves — those live
in the artifact; `cells_digest` binds them.

## Falsifiable claims (pre-registered, cycle 173)

- **F1.** A deterministic scripted backend whose output is a pure
  function of `(input_ids, seed)` reproduces every recorded completed
  search cell token-exactly: verdict `deterministic`, certificate
  issued, digest matching an independent recomputation.
- **F2.** A call-count-dependent backend diverges on at least one
  cell, and the reported `first_divergence` equals an independent
  oracle's computation.
- **F3.** The seeds the audit uses equal the seeds `core.evaluate`
  used for the same artifact — shown end-to-end with a seed-recording
  backend double, with expectations computed by an independent
  inline `hashlib` oracle, not by the shared helper.
- **F4.** A fingerprint mismatch produces zero `generate` calls.
- **F5.** Hostile artifacts (string IDs, negative IDs, boolean seed,
  non-list observations, missing keys, lone surrogates in fixture IDs)
  produce fixed-vocabulary output — never an exception, never a
  traceback.
- **F6.** Flipping one recorded output token makes exactly that cell
  diverge, at exactly that token index; truncating one recorded
  output makes that cell a `recorded_length_inconsistent` skip, not
  a divergence.
- **F7.** An artifact whose search aborted earns no certificate even
  when every recorded completed cell reproduces.

## Provenance

Design reviewed before implementation by a four-specialist round
(researcher, red-team A, red-team B with the explicit job of
disproving A, verifier); the uniqueness pass is logged in
`docs/RESEARCH_LOG.md` (cycle 173). Findings that changed this design:
recorded-length and vocabulary validation before any blame (A1/A3),
the replay-cost bound (A4/B2), the failed/unavailable doctrine
mapping (A6), the aborted-run certificate block (A8/B), and the
rejection of replay memoisation (B1).
