# Canonical-form ladder — cycle 209 design (ADJUDICATED: buildable on the amendments below; researcher, blind red-teams A and B, and verifier converged, coordinator adjudication in ~/workspace/agents_out/cycle209/build_log.md)

## Background: the gap this closes

The lab ships exactly one Unicode transform over prompt text, and it
is a compatibility transform: surface-noise's `unicode_variant`
operator (`model_protocol_lab/perturbation.py:101-123`, registered
at `:131`) replaces a seeded subset of ASCII alphanumerics with
their fullwidth forms (U+FF01–U+FF5E), which are NFKC-foldable and
visibly different characters. A full-tree census of every
`unicodedata.normalize` call in `model_protocol_lab/` (core.py:201,
:758; contamination.py:55; confidentiality.py:249; leading.py:153;
refusals.py:149,155; scorerlens.py:156,171) confirms every other
use is a comparison key, a detection signature, output
classification, or post-hoc answer regrading. Stored fixture data
stays byte-exact; only the comparisons normalize (Split docstring,
core.py:168-175). Nothing shipped re-renders prompt or fixture
message text under a different canonical form and re-measures
verdicts or re-derives a winner.

Canonical equivalence (UAX #15 §1.1) is "a fundamental equivalency
between characters or sequences of characters which represent the
same abstract character, and which when correctly displayed should
always have the same visual appearance and behavior." UAX #15 §10,
point 3: "Higher-level processes that transform or compare strings,
or that perform other higher-level functions, must respect
canonical equivalence or problems will result." The annex defines
the equivalence and warns about it; it measures nothing about
models or tokenizers. Sclar et al. (arXiv:2310.11324) measure
visible formatting sensitivity (up to 76 accuracy points on
LLaMA-2-13B), not canonical form. Rana et al.
(arXiv:2511.03237v1, "IndicSuperTokenizer") put NFKC in a
pre-tokenization pipeline and compare NFC/NFD/NFKC intrinsically by
fertility (§3.2, Table 16; metrics in §3.5) — tokens-per-word, not
task verdicts, not winner re-selection. The v1 pin matters: arXiv
now lists v2 (2026-03-22) retitled "MUTANT: A Recipe for
Multilingual Tokenizer Design"; the IndicSuperTokenizer title and
§3.2/Table 16 content are v1 facts, and any doc citing them must
say v1 or a future verifier resolving the bare ID will "refute" a
true claim.

The differentiator is the combination, never "first": canonically
equivalent, visually identical prompt text × the lab's 29
hand-rendered template families × conservative winner re-selection
with abstention, on a local checkpoint with exact token sequences.
A web pass over the named neighbours (LM Eval Harness, HELM, DSPy,
PromptPerfect, OpenAI evals, 2026-10-26) found no prompt-side
canonical-form instrument; the closest neighbour found, Promptfoo's
`normalizeUnicode`, normalizes at the assertion side (equals /
contains answer comparison, NFC by default), the same side as the
lab's scorer-lens rung — not a prompt re-render. That pass is a
search result, not proof of nonexistence, and the claim below does
not depend on it.

## Pre-registered claim (falsifiable)

Over generated fact-recall cases whose NFC and NFD renderings are
byte-different but canonically equivalent, the selected winner and
every per-case verdict are identical under both forms — conditioned
on the falsification population defined below. The claim states
exactly what would prove it wrong:

- The falsification population is ID-DIVERGENT paired cases only:
  cases where both arms completed AND the two arms' full render id
  sequences differ. Under greedy decoding, a case whose two arms
  produce identical id sequences cannot flip a verdict except
  through backend nondeterminism, so pooling id-identical cases
  into the denominator dilutes discordance toward a fake acquittal.
- Per case, the report is three-way: ids-identical /
  divergent-concordant / divergent-discordant, with the per-case
  id-length delta (NFC token count minus NFD token count) so that
  trivial byte-fallback divergence is visible as such rather than
  pooled with merge-level divergence.
- The claim is falsified per candidate when `invariance_falsified`
  is true (see Decision vocabulary). A same-ids verdict flip is
  recorded as a determinism finding (`n_same_ids_verdict_flips`),
  never as canonical-form fragility — the ws-scan precedent
  (`wscanonical.py` docstring claim 2), which conditions fragility
  on id divergence and records same-ids flips separately.
- The winner half is falsified only by `winner_changed` (both arms
  select, winners differ). Selection-vs-abstention is NEVER a
  winner change: it vacates the winner half to
  `winner_abstention_asymmetric` (exactly one side selects) or
  `winner_undefined` (neither selects).

Scope limits, printed with the numbers: the probe measures whether
the NFC/NFD distinction survives this checkpoint's full text→ids
pipeline and, if it does, whether it moves verdicts. A run that
lands 100% in the ids-identical bucket has measured the
checkpoint's tokenizer normalizer, full stop: the readout is
"instrument inapplicable to this tokenizer", never "the model is
robust to canonical form". A no-discordance result acquits the
candidate only on this generated population, never in general. A
discordant result proves the winner picked on NFC text can change
when the same text arrives decomposed; it proves nothing about
weights.

## Monday-morning utility

An engineer whose prompt text arrives from mixed sources (files,
web copy-paste, user uploads) uses the readout to decide whether
their harness must NFC-normalize all incoming text before
evaluation — or whether the template winner they picked on NFC
text silently changes when the same text arrives decomposed.

## Invariants (binding on the build)

1. Two arms only: `nfc`, `nfd`. No NFKC arm in v1 (NFKC is a
   compatibility transform per UAX #15 Table 1; §1.2 warns the K
   forms "must not be blindly applied to arbitrary text" because
   they erase distinctions; a K arm would also sit closest to the
   already-shipped fullwidth operator). No as-authored control arm:
   for an NFC-authoring generator it is byte-identical to NFC by
   construction and would burn a third of the generation budget
   re-measuring the NFC arm.
2. THE SCORER PIN. Verdicts are `core.evaluate`'s byte-exact
   `answer == case.expected` (core.py:797). A model echoing the
   prompt's canonical form flips its verdict with zero
   behavioural defect if the expected value itself carries
   accents — executed through the real pipeline in adjudication
   (NFC True, NFD False for an accented value; both True for an
   ASCII value). Therefore: decomposable characters live in
   carrier prose, topic, and question ONLY; every expected value
   is pure ASCII (placement-style `word-number` codes). The
   generator enforces this and a test pins it. The alternative
   (compare this probe's verdicts after NFC) is REJECTED: it would
   change what "verdict" means relative to every sibling probe
   and to the diff validator's digest-identity verdict.
3. Generated cases only. No user-fixture arm in v1: the shipped
   fixtures are pure ASCII (0 non-ASCII strings in
   examples/fixtures.json and fixtures.csv), and the arm is
   structurally unbuildable through the fixture path anyway —
   Split compares message inputs by NFC key (core.py:193-214), so
   an NFC fixture and its NFD twin are the same fixture by the
   lab's own anti-leakage discipline. Generated cases are
   stage-local (placement precedent) and never enter the user's
   Split, so no variant can cross the search/held-out boundary.
4. Message-content transform only, never the whole render.
   Templates are not touched: the transform applies to generated
   case text; template renders are byte-identical across arms by
   construction except for the message content they carry. (For
   ASCII messages, all 29 catalog families render
   canonical-invariantly — executed in adjudication.)
5. Single source of truth for the arms. The generator authors
   case text in NFC (precomposed literals). The NFD arm is
   `unicodedata.normalize("NFD", content)` of that one authored
   render — never independently authored, which would invite drift
   the asserts then have to catch.
6. Alphabet by mapping, not by block. A character is eligible
   iff `NFC(ch) != NFD(ch)`. Singletons (characters whose NFD is
   a single different codepoint, e.g. U+2126 OHM SIGN, U+212B
   ANGSTROM SIGN) are excluded: they are canonical-equivalent but
   identity-changing in a way that invites misreading. Hangul
   syllables are deferred to a labelled stress variant: NFD renders
   run ~3x the code points, confounding form with length.
   Compatibility-only characters (NFC == NFD, e.g. ligature U+FB00,
   vulgar fraction U+00BD) are excluded automatically by the
   divergence assert. The generator reports its effective alphabet
   size; no block count (such as "Latin-1 letters") is asserted.
7. Canonical ordering by construction. Authored text is NFC, so
   its NFD derivation is canonically ordered by definition of the
   normalization algorithm. Marks are never hand-assembled from
   base letters plus combining marks, which would make the
   authored text a third form equal to neither arm.
8. The ONLY load-bearing pre-generation assert is byte-divergence:
   `NFC(content) != NFD(content)` byte-wise, plus
   `authored == NFC(authored)`. Cases failing either are excluded
   pre-generation WITH counts (`n_excluded_non_divergent`),
   reported, never silently dropped. The equivalence assert
   `NFC(NFD(x)) == NFC(x)` is a theorem of canonical equivalence
   (holds universally, spot-checked in adjudication) and is NOT
   presented as a validation check anywhere.
9. Id divergence is measured at encode time on the FULL render id
   sequence (what the model sees), per (case, candidate) — a
   per-(case, backend) property, never a property of the text
   alone. Template text contributes nothing to it beyond the
   message content it carries.
10. Content commitment (the field-order pattern, not placement's).
    Records carry SHA-256 digests of both arms' message contents.
    The diff validator's independent mirror regenerates both forms
    from the seed and checks: each digest against the mirror's
    form, NFC digest != NFD digest, `ids_identical` implies equal
    arm token counts, and `id_length_delta` equals the token-count
    difference. A validator that checked token counts alone could
    never establish that the NFD arm's prompts were the NFD form
    of the case text. Payload-supplied normalized strings or
    equivalence flags would be tautological and are never trusted.
11. Fail-closed vocabulary:
    `{"available": False, "detail": "canonical_form_unavailable"}`,
    with stage details `cancelled`,
    `canonical_form_record_budget_exceeded`,
    `canonical_form_screen_failed` in the same pattern as
    placement. Per-cell details reuse the fixed pair
    `arm_context_exceeded` / `arm_evaluation_failed`. Rates carry
    denominators; a rate is None only when its denominator is 0.
12. Reporting discipline: counts and rates only; no payload,
    question, or answer text is rendered. Every payload-derived
    string goes through `_escape_text` (the cycle-206/207 lesson,
    not bare `escape()`).

## Decision vocabulary (total functions)

Floors transplanted from placement (placement.py:172-173):
`MIN_PAIRED_EVALUABLE = 6`, `MIN_DISCORDANT = 2`. They gate the
label only, never a reported count or rate.

Per-candidate classification over the id-divergent paired
population (`n_paired` = both cells completed AND ids diverged;
`n_nfc_only` / `n_nfd_only` = discordant directions):

- `insufficient` — `n_paired < 6`.
- `no_discordance_observed` — discordant == 0.
- `discordant_balanced` — `n_nfc_only == n_nfd_only >= 2`.
- `discordant_asymmetric` — `max(n_nfc_only, n_nfd_only) >= 2`.
- `below_threshold` — otherwise (discordant == 1 at sufficient
  population: observed, but below the floor — NOT falsified).

`invariance_falsified` is true exactly when the classification is
`discordant_balanced` or `discordant_asymmetric`. The direction is
stated here deliberately: falsified means the invariance claim
fell, not that an effect was confirmed absent.

Per-arm selection mirrors placement's `_selection_over`: over the
shared fixture set (recorded for every candidate), a candidate
"passes" an arm iff it passes every completed case on that arm;
exactly one passer selects, several passers give `ambiguous`,
none give `no_candidate`; withheld (`unavailable`) when completed
sets differ across candidates or are empty. The winner headline
compares the two arms' selections:

- `winner_changed` — both arms select, winners differ.
- `winner_stable` — both arms select, same winner.
- `winner_abstention_asymmetric` — exactly one side selects.
- `winner_undefined` — neither side selects.

## Record shape (per candidate)

```
{
  "fixture": "canonicalform-00",
  "topic": "<accented topic, NFC authored>",
  "expected_digest": sha256(ASCII value),
  "nfc_tokens": int, "nfd_tokens": int,
  "ids_identical": bool,
  "id_length_delta": int,            # nfc_tokens - nfd_tokens
  "nfc_content_digest": sha256, "nfd_content_digest": sha256,
  "cells": {"nfc": cell | None, "nfd": cell | None},
  "cell_details": {"nfc": "", "nfd": ""},
}
```

Cell shape (placement): verdict, boundary_found,
boundary_text_match, input_tokens, answer_digest, where a cell's
verdict is re-derived by the validator as
`answer_digest == expected_digest` conjoined with the boundary
flag, and a completed cell's `input_tokens` must equal the
record's arm token count. Aggregates additionally carry the
per-case `id_length_delta_by_case` list (in case order) and
`max_abs_id_delta`, so a reader can separate "diverged by 1 id"
from "diverged into byte ids" at a glance.

Generation budget: 24 cases × 2 arms × up to 24 candidates =
1,152 generations worst case (placement's cap). Records bytes
bounded by the house 12 MiB, as placement.

## Risks, pre-registered readings

- R1 (tokenizer erasure): SentencePiece normalizes with an NFKC
  variant before tokenization by default (default rule `nmt_nfkc`),
  so on SentencePiece-family checkpoints both arms may be
  token-identical on every case. That outcome is a tokenizer
  finding ("instrument inapplicable to this tokenizer"), never
  model robustness. HF `tokenizers` pipelines declare their
  normalizer explicitly per checkpoint; the probe discovers it by
  measuring ids, never asserts it in docs.
- R2 (byte-fallback divergence): for byte-level tokenizers, NFD
  text may fall back to per-byte tokens where NFC had a merged
  token. The per-case delta list is the mitigation that exists.
- R3 (scripted backends): the suite's fake backends encode at
  word/character level; on them id divergence is guaranteed and
  meaningless as tokenizer evidence. Payload language scopes
  bucket-1/2/3 to the run's backend fingerprint, and no test
  result is ever described as a tokenizer finding.

## Rejected alternatives (with reasons)

- NFKC third arm: compatibility, not canonical; closest to
  shipped territory (fullwidth operator); least new information.
  Deferred as a separately labelled compatibility arm, never
  pooled with the canonical headline.
- As-authored control arm: degenerate (== NFC for NFC-authored
  literals); a genuinely distinct mixed-form control would make
  the control's distance to each arm case-specific and break
  discordance accounting.
- NFC-compare in the verdict layer: changes what "verdict" means
  relative to every sibling probe and to the validator's digest
  identity; the ASCII-value pin achieves the same defusal by
  construction, without redefining verdicts.
- Fixture arm: vacuous on shipped data and structurally
  unbuildable through the fixture path (Split NFC keys).
- Single-discordance falsification: uncalibrated — one scorer
  artifact (or one noisy case) would close the question. The
  floors (6 paired, 2 discordant) plus the `below_threshold`
  state are the transplant that fixes it.

## Sources

- UAX #15, Unicode Normalization Forms (unicode.org/reports/tr15/,
  rev. 58): §1.1 canonical equivalence definition; Table 1 (NFD =
  Canonical Decomposition, NFC = decomposition + composition, K
  forms = compatibility); §1.2 K-form warning; §5 composition
  exclusions; §10 point 3 ("must respect canonical equivalence
  or problems will result").
- Sclar, Choi, Tsvetkov, Suhr, arXiv:2310.11324 (FormatSpread;
  "up to 76 accuracy points", LLaMA-2-13B; visible formatting
  features, not canonical form).
- Rana et al., arXiv:2511.03237v1, "IndicSuperTokenizer: An
  Optimized Tokenizer for Indic Multilingual LLMs" (§3.2 NFKC in
  pre-tokenization + Table 16 fertility by NFC/NFD/NFKC; §3.5
  intrinsic metrics). Cite as v1, never unversioned.
- SentencePiece, "Text Normalization in SentencePiece"
  (github.com/google/sentencepiece, doc/normalization.md):
  normalization before tokenization by default, default rule
  `nmt_nfkc`, partial-NFKC caveat.
- Hugging Face `tokenizers` Normalizers API
  (huggingface.co/docs/tokenizers/api/normalizers): NFC, NFD,
  NFKC, NFKD, Precompiled normalizer components exist as
  pipeline components; per-checkpoint configuration is a
  per-checkpoint fact.
- Promptfoo deterministic assertions
  (promptfoo.dev/docs/configuration/expected-outputs/deterministic/):
  `normalizeUnicode` (NFC default, NFKC option) applies to
  equals/contains answer comparison — answer-side, not a prompt
  re-render. The neighbour that normalizes, on the other side of
  the measurement from this probe.
- Tree: `model_protocol_lab/placement.py` (generator, floors,
  classification, selection mirror), `experiments.py:9220,9271`
  (cell/stage), `diffreports.py:10386,10530` (recompute/diff),
  `reporting.py:6885` (section), `cli.py:612,1893,3883` (flag,
  runner, dispatch), `core.py:797` (byte-exact verdict),
  `core.py:193-214` (Split NFC keys), `wscanonical.py`
  (id-divergence discipline), `catalog.py` (29 families).
