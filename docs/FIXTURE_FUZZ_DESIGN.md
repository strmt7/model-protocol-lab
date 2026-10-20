# Fixture fuzzer — design

**Status:** built in cycle 172 (`model_protocol_lab/fixturefuzz.py`);
this document is the published contract table the corpus implements. **Instrument name (honest):**
*fixture fuzzer* — a contract fuzzer for the lab's own fixture loaders.
It is not coverage-guided fuzzing and does not explore: the mutator
table is fixed, published below, and every case carries its expected
outcome from the loaders' documented contract.

**Monday-morning utility:** a maintainer who just changed the fixture
loaders (or is about to accept contributed fixture files) runs
`model-protocol-lab fixtures fuzz` to prove the loaders still reject
hostile or malformed fixture files gracefully — with the loaders' own
`ValueError` vocabulary — instead of crashing mid-run or silently
accepting a file the rest of the pipeline cannot handle.

**Positioning (web-checked 2026-10-20, see `docs/RESEARCH_LOG.md`
cycle 172):** fuzzing *models* is occupied (PromptFuzz/ps-fuzz and
garak fire jailbreak/injection attacks at endpoints), and fuzzing
*code* is occupied (Atheris/libFuzzer are coverage-guided engines a
developer wires to a target). No LLM evaluation harness checked ships
a first-class command that fuzz-tests *its own fixture loaders*
against a published accept/reject contract, across both of its input
formats (strict JSON at runtime, CSV at authoring time). That
combination is the contribution; nothing stronger is claimed.

## 1. The contract under test

The loaders (`datasets.parse_dataset` / `load_dataset`,
`fixtures.parse_fixture_csv` / `read_authoring_file`, discipline via
`core.Split`) document a fail-closed contract: any malformed input is
refused with `ValueError`; valid input is accepted exactly as written
(stored fixture data stays byte-exact; only comparisons normalize).
Outcomes:

- `accept` — the loader returns normally.
- `reject` — the loader raises `ValueError` (any message).
- `accept_warn` — CSV only: the loader returns normally *and* the
  fixture is named in the returned empty-`expected` list (the
  documented vacuous-pass warning channel).

A case whose observed outcome differs is a **finding**, classified:

- `crash` — a non-`ValueError` exception escaped the loader;
- `unexpected_accept` / `unexpected_reject` — the outcome flipped;
- `warning_mismatch` — the warning channel disagreed.

## 2. Corpus

Two base corpora, both minimal and known-valid (each carries a control
case proving the harness itself is green before any mutation is
trusted): a JSON dataset (2 search + 2 held-out fixtures, distinct
ids/groups/message sets) and the equivalent authoring CSV. Mutators
apply exactly one contract violation (or one documented non-violation)
each; combination mutations are out of scope because their expected
outcome is not independently derivable. Seed-parameterized mutators
(overlong lengths, confusable choice) vary values, never the expected
outcome class. Case classes, per target:

- **JSON:** marker literal in content/expected (accept — content has
  no blocklist at load time; the special-token audit is a separate,
  warning-only layer), homoglyph twin id/group/messages (accept —
  documented out-of-scope in `core.Split`: no normalization form
  unifies confusables), NFD/NFC twin id, group, and message set plus
  the U+2126/U+03A9 canonical-singleton twin (reject — NFC
  uniqueness/leakage), the U+FB01 ligature near-twin and
  within-partition identical messages and trailing-space/case-differing
  cross-partition groups (accept — `Split` compares NFC-normalized,
  unstripped, case-sensitive keys across partitions only), overlong
  and at-limit fields (char-counted, including a 16,384-astral-char
  field), empty and whitespace-only `expected` (accept, no warning
  channel), duplicate ids (same/cross partition), cross-partition
  group, identical cross-partition messages, missing/extra top-level
  keys, duplicate JSON keys (top-level, nested, and
  escape-equivalent), a legitimately escape-spelled key (accept),
  `NaN`/`Infinity` constants, `1e999` overflow and 5,000-digit
  integers, deep nesting (5,000 — outcome-asserted only; the guard
  that fires is depth- and interpreter-dependent), truncated,
  trailing-garbage, BOM-prefixed, empty, and whitespace-only text,
  top-level non-object, wrong field types (number/null/float in
  string fields), missing / extra fixture and message fields,
  messages not a list, empty / over-long / misalternating /
  assistant-final message lists, blank id/group with the exact code
  point named (space rejects; U+00A0 rejects — `strip()` removes it;
  U+200B-only accepts — it does not), empty partition, 101-fixture
  partition, 100-fixture partitions (accept boundary), byte-oversize
  document, exactly-2,000,000-byte document (accept boundary),
  control characters raw in the source string (reject — strict JSON)
  vs escape form (accept), zero-width content (accept), key order
  permuted (accept), lone surrogates per field (§3).
- **CSV:** bad header (case-changed), BOM-prefixed header (reject —
  deliberately visible per the module docstring), fully quoted header
  (accept), trailing empty header column (reject), message columns
  out of order / too many / none, bad partition, blank id/group with
  another cell present, an all-empty row (accept — silently skipped
  by the reader; pinned so the leniency is contractual, not
  accidental), overlong message cell / `expected` / id / group
  (reject — id/group are §3-class fixes, see below), id at the limit
  (accept), first-message empty, gap-then-content, assistant-final
  conversation, short row ending after a user message (accept —
  trailing columns are simply absent), duplicate ids,
  cross-partition group and identical messages, NFD twin id, too
  many rows, partition overflow, missing partition, field over the
  csv parser limit, bare-CR line endings (reject), extra cells in a
  row, quoted newline in a cell (accept — documented), CRLF line
  endings (accept), quoting leniencies (unterminated quote at EOF,
  quoted field with trailing junk, quote mid-field — all accept:
  the csv reader is non-strict and the stored value is pinned),
  marker literal in a cell (accept), empty `expected`
  (`accept_warn`), whitespace-only `expected` (accept, *no* warning —
  the warning channel fires on the exact empty string only),
  literal `null`/`123` field text (accept — CSV fields are strings),
  empty and whitespace-only documents (reject by different paths),
  lone-surrogate cell (reject — §3). Raw NUL in a CSV cell is
  **excluded**: it accepts on CPython 3.12, but the behaviour is not
  verified on 3.11/3.13, and a version-dependent case would falsify
  F3 — the table does not pin what it cannot guarantee.
- **Files** (written to a fresh temp directory by the runner):
  valid control, invalid UTF-8 bytes, CESU-8 surrogate bytes
  (reject as not-valid-UTF-8 — the file-level surrogate case),
  BOM bytes (reject at parse), byte-oversize file, directory as
  path, missing file, FIFO as path (reject, no hang — the readers'
  `O_NONBLOCK` + `fstat` defence), symlink to a regular file
  (accept — the readers follow links by design) — for both
  `load_dataset` and the `read_authoring_file` + parse path.

Two construction rules the swarm forced (red-team A2/B8): a case's
payload construction is part of its identity — raw-text cases are
built as raw text, structured cases via `json.dumps` (which escapes
control characters) — because the same semantic value has different
contract outcomes in the two constructions; and a reject case may
carry a **message pin**: where the loaders' vocabulary is genuinely
fixed (their own literal messages: the nonfinite-constant message,
the size-limit messages, the csv parser-limit message, and the §3
surrogate messages), the case asserts the exception is exactly
`ValueError` and the message contains the pinned text. Everywhere
else (JSON syntax errors, nesting) only the `ValueError` class is
asserted — those messages are the interpreter's and vary by version
(red-team B3).

The corpus digest is SHA-256 over the canonical case list (names,
targets, payloads); it is reported with every run so two runs can be
compared without diffing output.

## 3. Pre-registered findings (found by probing and the swarm, pre-build)

Three genuine loader defects, all of one family — *the loaders
accept or report something their own contract disowns* — each fixed
fail-closed in cycle 172 and pinned by fuzzer cases plus direct
regression tests:

1. **Lone surrogates (JSON).** Cycle 86 settled the *serializer*:
   a lone surrogate "can never be written to a UTF-8 fixture file",
   and `split_to_json_text` refuses it deliberately. The loader was
   never extended the same rule — `parse_dataset` **accepts** JSON
   escape-form surrogates (`"\ud800"` in the source text) into every
   fixture field, producing Splits the repo's own serializer refuses
   and that strict-encode helpers downstream meet with bare
   `UnicodeEncodeError`s. (Rendering and evaluation tolerate
   surrogates in degraded form — red-team B verified — so the claim
   is *unwritable and un-round-trippable*, not "unusable
   everywhere"; the loader extension is a new decision registered
   here, with its falsifier below.) Both parsers also rejected
   *raw*-surrogate source text only accidentally: the byte-cap
   `len(text.encode("utf-8"))` raises a bare `UnicodeEncodeError`
   (a `ValueError` subclass, so callers survive, but the message is
   a codec leak, not the loaders' vocabulary). Fix: `datasets._text`
   rejects any field containing a lone surrogate (covers the escape
   form in every stored field — id, group, expected, content; a
   decoded check, so valid surrogate *pairs* decoding to astral
   characters stay accepted); both parsers convert the byte-cap
   encode failure into a deliberate `ValueError` naming the
   surrogate. CSV needs no decoded-value check: it has no escape
   form, and files cannot carry raw surrogates past the readers'
   strict UTF-8 decode. Falsifier: if any in-repo consumer is shown
   to legitimately require surrogate fixture text *through the
   loader*, the cases flip to `accept` and the fix is reverted —
   none is known (the one surrogate-tolerant consumer, `core`
   seed derivation, constructs fixtures directly), and the burden
   of proof sits with acceptance, because acceptance is what
   produces unserializable Splits. Note the swarm's vacuity catch
   (red-team B1): outcome-only assertions cannot pin this fix for
   raw-surrogate text (it already raised a `ValueError` subclass),
   so the raw-surrogate cases carry message pins.
2. **CSV id/group length (red-team A1).** `parse_fixture_csv`
   length-checked message cells and `expected` but never `id` or
   `group`: a 17,000-char id was accepted, and the resulting Split
   then failed `split_to_json_text`'s round-trip — the authoring
   path accepting exactly what its own by-construction guarantee
   refuses, and a JSON↔CSV coherence break with no documented
   reason. Fix: the same `MAX_TEXT` check, same message, extended
   to `id` and `group`.
3. **csv.Error mislabel (red-team A4).** The CSV reader's error
   handler reported *every* parse error as "field exceeds the
   parser limit" — e.g. bare-CR line endings, which contain no
   large field at all. Fix: the size message is kept for the
   genuine field-limit error (existing pins depend on it); other
   parse errors report as parse errors.

## 4. Determinism

Generation is a pure function of the seed: parameter values are
derived from SHA-256 over `(seed, case name, parameter)` — the
repo's own seed-derivation precedent — never from `random.Random`
methods beyond `random()`, whose cross-version reproducibility
CPython does not guarantee (red-team B4). Same seed → same case
list → same digest. The digest is SHA-256 over case names, targets,
and payload **bytes** (file-case payloads are their bytes; no temp
path ever enters a case identity, the digest, or the report —
red-team B6). Reports contain no timestamps and **no exception
message text**: a conforming case records a fixed detail token, and
a finding records the outcome classification plus, for a crash, the
exception class name — because interpreter messages (nesting
guards, integer-digit limits, JSON syntax positions) vary by
version and configuration even when the outcome does not (red-team
B3). Exception details are sanitized (control/bidi/surrogate-
stripped, length-bounded) before they enter any human-facing
summary — hostile fixture text must not reach a terminal through
the fuzzer's own output. The runner executes cases on the calling
thread only: the JSON parser's recursion failure mode is
stack-sensitive off the main thread (red-team B5). Running the
corpus is time-bounded by construction: both loaders check the
2 MB byte cap before parsing, the largest case is a ~2 MB document,
and the `Split` discipline checks are set-based over at most
100 fixtures per partition.

## 5. Claims and falsifiers

- **F1** Every reject case raises `ValueError` and nothing else; a
  single non-`ValueError` escape falsifies the loaders' fail-closed
  claim, and the fuzzer exists to catch exactly that.
- **F2** Every accept case returns a `Split` (or split + warnings)
  whose stored data equals the loader's *decoded* field values
  exactly (red-team A17 rescoped this from "byte-exact against the
  input": JSON escapes resolve, CSV quoting is removed, CRLF is
  stripped, short rows are padded — stored data is post-decode by
  construction; NFC normalization is comparison-only and never
  rewrites it). A stored value that differs from the decoded source
  field falsifies F2.
- **F3** Same seed → identical digest and identical report, on any
  supported Python (3.11–3.13). A case whose outcome depends on the
  interpreter version falsifies F3 and must be redesigned, not
  pinned per-version.
- **F4 (honest limit)** A green fuzz run proves the published table
  holds — nothing more. It does not prove the loaders accept exactly
  the valid files, nor that unmutated classes are safe.

## 6. Non-goals

No diff-reports section and no HTML report block: fuzz output is a
maintenance artifact, not experiment output (the `pack`/`verify`
precedent). No fixture-*quality* judgment (a boring but valid fixture
passes; that is correct). No coverage-guided search (see
Positioning). No new dependencies; stdlib only.
