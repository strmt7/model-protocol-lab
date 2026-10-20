# Scorer-lens regrade — design contract

Status: shipped in cycle 180 (wave-7 research rank 1). This document
is the frozen contract for the ladder in `model_protocol_lab/scorerlens.py`;
the ladder itself also travels inside every payload as data, with a
SHA-256 digest, so this prose and the data can be checked against
each other.

## What it answers

The lab's native verdict is byte-exact. That strictness is the
point — but it cannot distinguish "the template produced the wrong
answer" from "the template produced the right answer in a surface
form the scorer discards" (trailing punctuation, a markdown fence,
`1,000` for `1000`). The scorer lens regrades the *recorded* search
answers under progressively more permissive, fully specified
normalisations and reports, per candidate, how many failures each
rung recovers, which fixture flips first at which rung, and whether
the search-phase winner changes. Post-hoc, zero model calls.

Utility, in one sentence: after a template change, decide whether
the template actually got worse or the byte-exact scorer is
discarding human-acceptable answers — and whether the winner
survives the mildest defensible scorer.

## The ladder

One pipeline: each rung consumes the previous rung's output, which
makes monotonicity (claim S2) a theorem of the construction rather
than a hope. R4 is a union with R3, never a replacement comparison.

- **R0** — identity. Reproduces the native per-fixture verdicts.
- **R1** — trim; collapse runs of spaces/tabs to one space;
  collapse 3+ consecutive newlines to two. Single/double newlines
  survive: line structure can be content.
- **R2** — NFC, casefold, combining-mark removal (NFD, drop
  category Mn, recompose). The mark step is the refusal detector's
  fix for Turkish `İ` (casefold leaves `i` + combining dot). The
  decomposition is canonical (NFD), deliberately not the
  detector's compatibility NFKC: fullwidth digits and other
  compatibility forms are not folded at R2 (an early build used
  NFKD for the mark strip and silently folded them; the
  monotonicity property test caught it, and the rung now does
  exactly what this contract says).
  Casefold equates German `ß` with `ss` — Unicode default folding
  as specified; a known false-recovery risk on name fixtures,
  surfaced by the per-flip attribution (`flip_via`: `nfc` when NFC
  alone achieved equality, else `casefold`).
- **R3** — markdown-fence strip (line-anchored code fences, any info
  string discarded, both sides), then iterative edge stripping of
  a small presentational set: asterisk, underscore, backtick,
  double and single quotes, period, comma, semicolon, colon,
  exclamation and question marks, and round, square and curly
  brackets. (The payload's ladder data carries the set as an
  explicit character list.)
  The set excludes `-`/`+` (signs are content) and `%` (a unit).
  A leading `.` directly before a digit is never stripped (`.5`
  is not `5`). Removing punctuation at all positions is forbidden
  by the protocol (it would equate `3.5` with `35`). An expected
  text that is itself fenced or punctuation-bearing is stripped
  too; recovered fixtures whose expected side was altered by R3
  are counted in the payload (`n_flips_expected_altered`).
- **R4** — R3, or both R3 forms fully match one ASCII numeric
  grammar and parse to finite, equal `Decimal` values:
  `^[+-]?([0-9]{1,3}(,[0-9]{3})+|[0-9]+)(\.[0-9]+)?([eE][+-]?[0-9]+)?$`
  or `^[+-]?\.[0-9]+([eE][+-]?[0-9]+)?$`. Underscores, non-ASCII
  digits, `inf`/`NaN` spellings, and decimal commas never fire R4
  (`1,5` is not `15`; the ladder has no locale input). Comparison
  is `Decimal.__eq__` on parsed values — never `normalize()`,
  which under the default 28-digit context collapses distinct
  long decimals. `-0 == 0` holds and is accepted.

## Verdicts, denominators, withholding

- A rung *verdict* is `status == completed ∧ boundary_found ∧
  rung text match`. The boundary conjunct is a recorded fact no
  rung can move; per-candidate `n_boundary_blocked` is reported
  next to every winner line. A diagnostic text-only match series
  is reported alongside the verdict series.
- At rungs ≥ 1 a pass requires both normalised forms nonempty;
  an erasure-to-empty is a no-flip. Fixtures whose expected text
  normalises to empty at R1 are excluded from the lens entirely,
  with `n_excluded_empty_expected` reported per candidate.
- The winner readout is the **search-phase** winner under each
  rung (conservative bar re-applied to rung verdict counts);
  held-out is never regraded. It is withheld with a named detail
  (`candidate_unavailable`, `candidate_not_eligible`,
  `evaluable_mismatch`) whenever the field is partial — never
  re-applied to a prefix-measured field. The surface-failure
  share (failures recovered by R4 ÷ native verdict failures,
  R4 recoveries included in the numerator) is withheld per
  candidate on the same eligibility gate as the refusal rates.

## Payload discipline

Counts, fixture IDs, rung numbers, and small enums only: no answer
text, no expected text, no normalised forms, no lengths. The
per-fixture ground records (`fixture`, `completed`,
`boundary_found`, `exact_match`, `excluded`, `first_flip_rung`,
`flip_via`, `expected_r3_altered`, `newline_differs`) plus
per-candidate `eligible` let the `diff-reports` validator
recompute every aggregate, the mismatch flag, and the winner
readout — a payload whose aggregates disagree with its records
is rejected as corruption. A coherent *foreign* ladder validates
and is reported by the diff as protocol drift.

## Pre-registered claims (sharpened by the pre-build cross-check)

- **S1.** R0 pass ⟺ (`status == "completed"` and `answer ==
  expected`) reproduces every recorded `exact_match` flag; R0 ∧
  `boundary_found` reproduces every candidate's native pass count.
  Oracle-differential over synthetic and adversarial observations.
- **S2.** The ladder is monotone: a fixture passing at rung k
  passes at every rung > k. Rung match is cumulative ("passes
  under the ladder up to this rung"), which makes monotonicity a
  theorem of the pipeline + union construction — including for
  identical all-punctuation pairs, which pass at R0 and therefore
  at every rung, while *distinct* pairs that would only meet at
  an empty form pass at none (S3). A property test over an
  adversarial corpus (NaN/inf spellings, all-punctuation strings,
  fenced blocks, decimal commas, empty strings) guards the
  implementation.
- **S3.** At rungs R1–R4, no pass is recorded when either side's
  normalised form is empty; erasure-to-empty is a no-flip. Pinned
  by named tests (`?`/`!` at R3, `***`/empty at R1–R4).
- **S4.** R4 fires iff both R3 outputs match the frozen grammar,
  parse via `Decimal`, and are finite and equal. `Answer: 42`
  fires at no rung — the ladder has no label-strip operation.
  `1,5`, `12,34`, `1,2,3`, `50%`, `1_000`, `inf`, `NaN` never
  fire R4; `42` vs `17` never match at any rung. Each is a named
  test.

## Provenance

The design was red-teamed before code existed (wave-7): red-team
A returned DO NOT BUILD AS SPECIFIED with 12 findings — the
recorded result carries no `expected` field (blocker; the profile
takes the expected map as an explicit parameter, the verbosity
precedent), the native boundary conjunct was undefined against
the ladder, the `Answer: 42` motivating example flips at no rung,
R4's `1,000` example is false under a naive `Decimal` parse, and
S1 ∧ S3 were jointly unsatisfiable on legal empty-expected
fixtures. Red-team B independently re-derived every finding,
upheld all of them (one sub-claim refuted: NFC + casefold *does*
equate the `ﬁ` ligature via casefold), and added two binding
amendments — one ASCII-only grammar for all R4 inputs (A's own
`\d` regex matched Unicode digits), and ground records rich
enough for the validator to recompute denominators and the
winner. This contract is the corrected design that survived.
