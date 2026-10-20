# Template surface-noise robustness — design v2 (cycle 166)

v2 addresses the cycle-166 red-team review (A: 22 findings; B: cross-check —
21 survive, 1 downgraded, 0 refuted — plus new N1). Every blocking finding is
resolved below; each resolution cites the finding ID.

## What it is

A per-candidate measurement for `model-protocol-lab run --template-robustness`: for each
candidate chat template, how well does it *preserve correct verdicts* when
surface noise hits user-supplied content? The template is the independent
variable; model weights, fixtures, backend, and protocol are fixed.

## Monday-morning use

You are choosing a chat template to ship and your users type sloppily. For each
candidate, this reports the fraction of its *correct* fixture verdicts that
survive surface-noise perturbations — so you can see which template degrades
most gracefully, before you ship it.

## The unit of analysis and the verdict function

For candidate template T (a `Format`; stage keys by candidate name exactly as
in `results.json`'s `candidates` dict — F21), fixture f, backend B:

- `render_T(f) = T.render(f.messages)` (user-role message contents may be
  perturbed; roles, order, and non-user messages are byte-identical — F10).
- `V(T, f) = verdict(B.encode(render_T(f)))`, pass/fail/abstain by exact answer
  match, same fail-closed semantics as the probe's `_evaluate_one`.
- For perturbation variant w (see "Operator set"): `f_w` = f with w applied to
  user-role message contents; `V_w(T, f) = verdict(B.encode(render_T(f_w)))`.
- A **flip** for (T, f, w): `V_w(T, f) != V(T, f)` over mutually-evaluable pairs
  (both non-abstain). Either direction counts; the rationale is the probe's:
  a flip is a verdict change under noise, regardless of direction (F16).

## Operator set (F1)

The probe's full operator set is **not** reused. Perturbing fixture content with
`typo` or `punctuation_strip` can change the task's correct answer while
`expected` stays pinned (e.g. "2.5 + 2.45" → "25 + 245") — the R6 inversion
the probe was designed to avoid. The template-robustness operator set is
restricted to **presentation-level** operators, defined as perturbing
presentation without changing the correct answer:

- `whitespace` (frozen modes), `case_flip`, `unicode_variant`.

Excluded, with reason: `typo` (can change the correct answer: "cat"→"car"),
`punctuation_strip` (can change the correct answer: "2.5"→"25"). The set is
frozen in the protocol; results are conditional on it. Pathological fixtures
whose correct answer is presentation-sensitive (e.g. whitespace-significant
code) are outside the instrument's scope — stated here, not hidden (F1).

## Headline metric (N1, F7)

Naive "mean stability" is **rejected**: P(flip | baseline pass) >> P(flip |
baseline fail) for exact-match tasks (breaking a match needs any output change;
repairing one needs noise to produce exactly the expected string), so mean
stability mechanically rewards low-accuracy templates (B's derivation; the
asymmetry vanishes for binary-choice tasks — stated as the scope limit).

The primary metric is the **verdict preservation rate**, baseline-conditioned:

- Let `P_T = {f : V(T,f) = pass}` (pass-baseline fixtures).
- For f ∈ P_T, let `W_f = {w : V_w(T,f) ≠ abstain}` (mutually-evaluable variants).
  f is *preserved* iff ∀w ∈ W_f: `V_w(T,f) = pass`. If `W_f` is empty, f is
  *unevaluable* (excluded, counted in `n_unevaluable` — never counted as stable).
- **Preservation rate** = |{f preserved}| / |{f ∈ P_T : W_f ≠ ∅}|.
  `None` (not 1.0, not 0.0) when the denominator is 0 — with the explanatory
  note, per the wave-5 abstain policy (`surfacenoise.py`, wave-5 fix #2).

Secondary, informational only:

- **Noise repair rate** = |{f : V(T,f)=fail ∧ ∃w ∈ W_f: V_w(T,f)=pass}|
  / |{f : V(T,f)=fail ∧ W_f ≠ ∅}| — how often noise accidentally fixes a wrong
  verdict. Never blended into the primary metric.
- **Per-operator preservation vectors**: the preservation rate restricted to
  variants of one operator. Reported per operator; **never** a cross-operator
  minimum or blend (F5 — the probe's R12b rejection applies here too).
- **Nearest observed flip**, per operator per (T, f): minimum content-span
  distance (below) over flipping variants of that operator; "none observed"
  when empty. Cross-operator minima are not reported.

## Distance (F3, F17, N2)

Pinned: **normalized token indel-edit rate** `d = indel/(n+m) ∈ [0,1]`
(`token_edit_distance`, which returns a raw int, divided by the combined
length — the probe's unit), computed on the **message-content span
(pre-render)** tokenized with the backend's tokenizer: `d(f, f_w)` over
user-role message contents. Because the content span is identical across
templates for the same (f, w), the distance is template-invariant by
construction — this eliminates the verbosity bias (verbose templates
mechanically scoring "closer") that rendered-input distance would introduce
(F3). The model saw the rendered form; the distance describes the
perturbation magnitude. Stated, not hidden.

## No-op filter (F4)

Variants whose perturbed user content is token-identical to the original
(token IDs of the content span, via the backend tokenizer) are **no-ops**:
listed in `no_ops`, never counted in any denominator. Token-ID identity is
used — not text identity — because the distance `d = indel/(n+m)` is computed
on token IDs: a textually-different-but-identically-tokenized variant has
d=0, can never flip, and would otherwise pad the denominator as
guaranteed-preserved.

## What the number means — and doesn't (F2, F12, F15, N1)

- With weights, fixtures, backend, protocol, and operator set fixed,
  cross-template differences in the preservation rate reflect the template's
  causal contribution **to this measured quantity** — which includes genuine
  robustness differences *and* template-induced measurement differences
  (tokenization, abstention propensity). Denominators (`n_evaluable`,
  `n_unevaluable`, `n_base_abstain`) are reported per template so the latter
  are visible; we do not claim to separate them (F2).
- The number is a **template×backend-tokenizer interaction**, not a
  backend-independent template property. Cross-backend comparison requires
  identical (fixtures, backend, protocol-fingerprint) triples (F15).
- The per-template readout **is** a comparative ranking across the run's
  candidates — valid only within identical triples, never an absolute quality
  score, never blended into winner selection (which stays purely
  accuracy-driven and conservative) (F12).
- A template can be perfectly robust and perfectly wrong; the preservation
  rate is reported *next to* accuracy, and the accuracy-driven decision does
  not see it (F12, N1).

## Falsifiability (F6)

This measurement would be proven uninformative if: (1) the controls harness
(below) fails to discriminate a known-robust from a known-fragile template
implementation; (2) the preservation rate is identical across all templates
on every fixture set tried (no signal); (3) the preservation rate does not
reproduce when the run is repeated byte-identically (nondeterminism — the
instrument's core promise). Note the number is a template×backend-tokenizer
interaction (see below): changing the tokenizer is *expected* to change it,
which is a scope boundary, not a retirement criterion.

## Frozen protocol (F9, N3)

`freeze_template_robustness_protocol(backend_fingerprint, fixtures)` /
`verify_template_robustness_protocol(protocol, backend_fingerprint, fixtures)`:
instrument name + version, `perturbation_code_sha256` (perturbation.py),
**scorer code hash** (hash of the template-robustness implementation, not just
perturbation.py — F9a), the frozen search plan (3 operators, seeds/modes fully
specified — F9d), fixture ids + hashes (bound at verify time — the wave-5
finding-1 fix), backend fingerprint. The renderer is per-candidate context:
protocol records template name + sha256 of the `Format`'s defining fields and
of the rendered base *and* perturbed inputs (N3). The `subject_span` text
states the fixture-scope condition explicitly (v1.0.1, wave-6 red-team A
finding 3: results are conditional on fixtures whose correct answer is
invariant under the operators; presentation-sensitive fixtures are out of
scope). Comparability rule (F9c): rankings are only valid across runs with
identical (fixtures, backend, protocol-fingerprint) triples.

## Degradation granularity (F8, N5)

Per-variant backend exceptions → `abstain` (fail-closed, inherited from
`_evaluate_one`); they enter denominators, never the numerator. `unavailable`
is reserved for **stage-level** failure (backend down, protocol mismatch,
frozen-plan drift) and is **prominent** in the report (N5a) — a silently
missing measurement is worse than a noisy one for the ship decision. The stage
runs **post-winner-selection** over the run's candidate set (N5b), never
raises, never fails the run (the `--verbosity` precedent).

## Controls (F14)

The probe's controls don't transfer (different unit of analysis). The stage
ships two implementation controls, with the probe's honest caveat (they
validate the machinery, not the concept):

1. **Determinism**: the same template probed twice yields byte-identical
   preservation vectors.
2. **Scripted-response discrimination**: against a backend scripted to flip on
   any content change, preservation rate = 0.0; against one that never flips,
   1.0. (Validates the flip-counting machinery end-to-end.)

## Scope decisions

- **Split**: search split — the fixtures the winner was selected on; the
  decision-relevant question is whether the *selection* is noise-fragile
  (F22).
- **Multi-turn**: user-role messages in any fixture are perturbed; assistant
  (few-shot demonstration) messages are never perturbed — they are not user
  content, and perturbing them would be the R6 inversion again (F10).
- **Zero denominators**: no user-role messages → fixture excluded with reason;
  all variants no-op → `n_unevaluable`; baseline abstain → excluded from the
  preservation denominator, counted in `n_base_abstain` (F11).
- **Candidate key**: the `Format` candidates exactly as passed to
  `run_experiment`, keyed by candidate name as in `results.json` (F21).

## Cost (F18)

Per candidate: F fixtures × (1 baseline + V variants) generations, plus
F×(1+V) encodes for verdicts and F×(1+V) encodes for the content-span
no-op filter/distance (2× encode factor). V = 8 seeds × 2 seeded operators
(case_flip, unicode_variant; whitespace modes are data-dependent, counted
exactly). Total per run ≈ C × F × (17 + n_whitespace_modes) generations.
Opt-in (`model-protocol-lab run --template-robustness`); the cost formula
is in `--help`.

## Differentiation (F13)

Web-research pass logged in `docs/RESEARCH_LOG.md` (cycle 166): PromptBench /
NoisyPromptBench measure *model* robustness (PDR); arXiv:2604.18389 studies
prompt-wording sensitivity as a paper, not a tool; no shipped tool measures
per-chat-template surface-noise robustness deterministically. Framed as "the
combination", never "first". Academic work cited, not claimed over.

## Artifacts (F20)

- `results.json`: `template_robustness` key (per-candidate vectors +
  denominators + protocol fingerprint).
- HTML report: per-candidate preservation bars with denominators, per-operator
  vectors, nearest-flip display, the "nearest observed flip is an upper bound
  over the searched subset (d_obs ≥ d*), never a guarantee" warning, and
  "Scope — read first."
- `diffreports.diff_template_robustness`: per-candidate drift with the
  comparability gate (protocol-drift → not comparable).
- `model-protocol-lab surface-noise` (per-prompt probe) is unchanged.
