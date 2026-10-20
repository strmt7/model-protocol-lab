# Candidate aliasing audit — design contract

Status: shipped in cycle 181 (wave-7 research rank 2). This document
is the frozen contract for `model_protocol_lab/aliasing.py` and the
`aliasing` payload; the digest definition travels inside every
payload as protocol data, so this prose and the payloads can be
checked against each other.

## What it answers

Candidates are names for renderings. On a given tokenizer and
fixture set, two differently named formats can render
byte-identical prompt token sequences on every search fixture — a
BOS mutation that is a no-op for a family whose framing already
starts with that token, two families whose framing coincides on
short single-turn messages, a field the fixture set's turn shapes
never exercise. A leaderboard cannot see the collapse (tied rows
look normal); this lab can, because it renders every candidate
itself and abstains on ties — and a tie between two names for one
token sequence is a different event from a tie between two
genuinely different prompts.

Utility, in one sentence: an operator who ran a 24-candidate sweep
uses the audit to decide whether to prune the candidate set before
the next (expensive) run — and whether an `ambiguous` abstention
was two genuinely different templates tying, or one token sequence
wearing two name tags.

## Two class notions, never conflated

The design's first draft defined one equivalence (identical
`input_ids`) and spent it as three different things. The cross-check
(red-team A F1, upheld as amended by red-team B) split it:

- **Render class** — identical recorded `input_ids` on every
  fixture of the cohort. Supports exactly one claim: these names
  produced the same prompt token sequences on this fixture set.
- **Measurement class** — a render class refined by the candidate's
  own response boundary (`boundary_token_ids`). Boundaries are part
  of the measurement: each candidate is judged under its own
  terminator, and under `--boundary-map` two candidates with
  identical prompts and different boundaries are different
  experiments that happen to share a prompt.

Effective counts are reported for both notions
(`n_effective_render`, `n_effective_measurement`) next to
`n_candidates_recorded` and `n_primary_cohort`. Pruning language
and the alias-suspect flag cite measurement classes only. In the
default configuration (no `--boundary-map`) every candidate shares
one boundary and the two notions coincide; the distinction is
load-bearing exactly in the mode where it matters.

## Cohorts and evaluable cells (A1')

Classes exist only inside **cohorts** of candidates with identical
completed fixture sets: pairwise identity over differing sets is
not transitive, so union "classes" would not be classes. On genuine
runs the eligible candidates always share the full search set and
at most one ineligible interrupter exists; the cohort discipline is
what keeps foreign or interrupted records honest. The primary
cohort is chosen by a total deterministic order (most candidates,
then most fixtures, then lexicographic) so the producer and the
diff validator derive the same cohort from the same records.

A cell counts only if it is **evaluable**: `status == "completed"`,
`input_ids` a nonempty list/tuple of exact ints (bools excluded)
`>= 0`, the fixture unique within its entry, and the observation's
`candidate` field present and equal to the entry's name. The
nonempty rule is load-bearing: the abort path in `core.evaluate`
records `input_ids == ()` for render/encode failures, so the empty
tuple is a failure sentinel that collides with itself — two failed
candidates would "coincide" on it. A completed cell with empty or
malformed IDs is an artifact defect (`recorded_ids_invalid`, the
determinism audit's vocabulary), never a class member. Duplicate
fixture IDs and candidate-field mismatches fail the whole candidate
closed with named details; duplicate search entries for one name
make that name unavailable (`duplicate_candidate_entry`).

Classes are computed from recorded `input_ids` **only** — never
from `candidate_fingerprint`, the `format` dict, or `input_text`.
The fingerprint hashes `asdict(fmt)` *including the name* (its own
docstring says so deliberately), so fingerprint classes would be
singletons by construction: the audit would report "no aliasing" on
every run, forever. Named tests pin the two executed counter-pairs
(name-only difference with `repeat_bos` under an empty BOS;
`assistant_turn_end` difference on single-turn fixtures) as aliased
despite different fingerprints and different format dicts.

## The redundant count (one formula)

`n_redundant_completed_observations` = Σ over the primary cohort's
fixtures of (candidates completed on the fixture − distinct
`(input_ids, boundary_token_ids)` tuples on it). It counts each
redundant observation once, is additive over fixtures, sees partial
aliasing that a global-class formula misses, and is recomputable
from the ground records. The render analogue travels under its own
name (`n_redundant_render_observations`); the two are never summed,
and pair-matrix entries are never summed into either (pairs overlap
by construction). The word "wasted" appears nowhere in the payload:
a boundary-distinct evaluation is a distinct measurement, not
waste, and even a measurement-redundant evaluation is the replicate
the A2' cross-check consumes.

## The determinism tripwire (A2')

Within one measurement class, the recorded `output_ids` are
identical per fixture, and the outcome fields (`exact_match`,
`boundary_found`, `boundary_text_match`, `status`) are identical —
checked as independent corollaries, not consequences (the outcome
fields pass through the backend's decode). The premise is checkable
and checked: `core.request_seed_for(seed, case_id)` derives the
per-request seed from the run seed and the fixture ID only — the
candidate is not an input — and the generation cap is run-global,
so identical measurement configurations present identical
`generate()` inputs.

A violation is reported per (class, fixture) as a
**determinism-suspect cell**, kind `output` (output digests differ),
`outcome` (outcome fields differ), or `both`: recorded cells a
deterministic backend could not have produced from one measurement
configuration, attributable to backend nondeterminism (stateful or
order-dependent backends included — candidates are evaluated
sequentially in one run) or artifact corruption. The audit asserts
neither cause; `verify --determinism` replay is the discriminating
follow-up. This mirrors the house precedent in
`experiments.ws_canonicalization` (`n_same_ids_verdict_flips`:
same IDs, different verdict ⇒ determinism anomaly, never counted as
the probe's own effect), at the native-run layer.

## The alias-suspect flag (A3')

`alias_suspect_abstention` fires iff (i) the recorded decision is
`ambiguous`; (ii) the passing set — primary-cohort, eligible
candidates whose every evaluable cell passes — recomputed from the
ground records has ≥ 2 members; (iii) the passing set is contained
in a single measurement class; and (iv) that class contains no
outcome-suspect cell. Withholding reasons are produced in a fixed
priority order, and the validator recomputes the expected reason in
the same order: `candidate_unavailable`, `decision_not_ambiguous`,
`passing_set_inconsistent`, `determinism_suspect_in_class`,
`passing_spans_classes`, `boundaries_differ_within_passing_set`.
`alias_suspect_detail` is `""` exactly when the flag fires (the
scorer-lens `winner_detail` discipline); the fired case's class
members travel in `alias_suspect_class_members`.

Two negative cases are pinned as tests because users will ask about
them. A tie whose passing set *spans* classes never fires: two
different prompts both passing everything is a genuine tie between
distinct measurements — the lab working as intended, not an
artifact. And a *post-boundary-only* output divergence inside the
containing class does not suppress the flag: every
selection-relevant quantity is identical there, so the aliasing
explanation of the tie is intact; the cell is still reported as a
suspect cell. The flag is advisory — it never modifies the recorded
decision or `selected_candidate`, and it asserts the structural
containment only, not that any candidate would have survived
held-out (held-out was never run for any of them).

## The payload and its ground layer

Token IDs in a payload are prompt text by proxy (the house strips
recorded ID subsequences from the public view for exactly this
reason), so the payload carries, per candidate: the boundary tuple
(operator configuration, already public in every search entry),
eligibility, planned count, and per-fixture ground records
`{fixture, status, evaluable, input_sha256, output_sha256,
exact_match, boundary_found, boundary_text_match}` — digests and
small enums only. Digests are `None` for non-evaluable cells, never
a digest of the empty list (the abort sentinel must not re-enter
through the digest layer). The digest definition is one canonical
form, frozen as protocol data: SHA-256 of the compact JSON encoding
of the ID list, version string `sha256-json-intlist-v1`, computed
identically for input and output IDs. At top level the payload also
carries `decision`, `selected_candidate`, `backend_fingerprint`,
`seed`, and `max_new_tokens`, so it is interpretable standalone at
the diff's payload-only interface.

What the validator's recompute buys, stated without overselling:
`_valid_aliasing` re-derives cohorts, both class partitions, both
effective counts, both redundant counts, the pair matrix, every
suspect cell, the passing set, and the flag from the ground records
with independent code, and rejects duplicate candidate names,
duplicate fixtures, and payloads beyond the run caps (24 candidates,
2400 cells). That is an **internal-consistency** recompute — it
detects aggregates inconsistent with their records; it cannot
detect fabricated ground records. That limitation is the standing
one for every house ground layer (refusals, scorer-lens), and the
design says so rather than claiming verifiability. Digest handling
in the validator is version-agnostic (equality only): a coherent
foreign digest version validates internally, and a version mismatch
across a diff is `protocol_drift`.

Two size/privacy facts settle the flag-gating: the ground layer is
the largest per-cell payload the lab ships — measured 921,192
bytes (≈0.92 MB) at the 2400-cell cap in the indented public JSON —
so the audit runs behind `run --aliasing`, following the
refusals/scorer-lens precedent. And an unsalted SHA-256 of a short,
structured token sequence is a guess-confirmation oracle: anyone
holding the payload, a candidate prompt, and the tokenizer can
verify a guess offline. The docs claim exactly that and no more:
no raw token IDs or text; per-cell digests permit offline
confirmation of a guessed prompt. The same trade was already
accepted for `prompt_sha256` in the surface-noise probe.

## Back-porting, stated precisely

The profile runs over the in-memory result at run time. Back-porting
to stored artifacts works over `--include-evidence` artifacts only:
`reporting._public_view` strips `search[].observations` from the
default public `results.json`. An entry with no `observations` key
degrades its candidate to `observations_missing`; the top-level
payload is `{"available": false, "detail": "observations_stripped"}`
only when *every* search entry lacks the key (the
`determinism._plan` test) — otherwise the top level stays available
with per-candidate records (the refusal-profile precedent). A run
made without `--aliasing` carries no `aliasing` key at all, and
reporting/diff render that as "not recorded", never "no aliasing
found".

## Diff semantics

`diff_aliasing` reports measurement-partition drift over the common
candidates (`pairs_joined` / `pairs_split`), per-candidate boundary
drift, count drift over the effective counts, the redundant count,
and the suspect-cell count, flag drift, candidates
added/removed/excluded, and the evaluable-mismatch change. Three
drift gates set `comparable: false` while still listing drift:
`fixture_set_drift` (classes are fixture-set-relative),
`backend_drift`, and `protocol_drift` (digest definition).
`backend_drift` means the recorded backend fingerprints differ —
tokenizer identity *not proven equal*, because the fingerprint also
binds device, context limit, and dependency versions; it must never
be paraphrased as "a different tokenizer". The pair matrix is
pairwise data with explicit denominators and is never clustered
into "partial classes" in the HTML or the diff.

## Provenance

Wave-7 research rank 2. Red-team A returned DO NOT BUILD AS
SPECIFIED (10 findings). Red-team B's cross-check upheld the
verdict, downgraded five severities, refuted one sub-claim (the
scorer-lens precedent A cited for duplicate fixtures accepts them —
the duplicate rule is new, not a transplant), caught an internal
contradiction in A's own `n_nominal` definitions, and repaired
four of A's prescribed fixes (the stripped-artifact trigger, the
redundant-formula ambiguity, flag suppression keyed on any suspect
cell instead of outcome-suspect cells, and the inverted
detail-field precedent), plus four findings A missed (the
guess-confirmation wording, the missing run-configuration fields,
the digest-of-empty rule, duplicate search entries). The build
follows gates 1, 4, 10 as A wrote them and gates 2, 3, 5, 6, 7, 8,
9 as B amended them. All A1'–A3' claims are pinned by named tests
in `tests/test_cycle181_aliasing.py`, including a brute-force
oracle differential for A1' and a `run_experiment` construction of
the flag case through the real run flow.
