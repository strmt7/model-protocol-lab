# Token-depth sensitivity sweep — cycle 206 design (ADJUDICATED: buildable on the amendments below; red-team A, red-team B, and verifier converged, coordinator adjudication in ~/workspace/agents_out/cycle206/coordinator_adjudication.md)

## Background: the gap this closes

The lab ships a prompt-boundary sensitivity sweep (`run --sensitivity-sweep`,
USAGE §8.5): the search re-runs with the prompt truncated mid-token at each
depth 1..`--max-drop-bytes` (bytes, tail-only, cap 8), reporting the depths
where the selected winner or decision changes. Wave-9 red-team B proved the
shipped coverage is byte-granularity, tail-only, depth ≤ 8 bytes, and
recorded the qualification: for templates whose turn-closing suffix spans
more than 8 bytes (multi-token assistant headers do), a token-granularity
ladder at depths beyond that suffix is measured by no shipped readout —
not by the sweep, not by `budget_frontier` (output side), not by
stop-cause. The boundary-truncation ladder candidate was dropped on that
evidence; the token-depth extension of the existing sweep is the recorded
backlog item. This design extends the existing sweep. It is not a new
probe.

## Pre-registered claim (falsifiable)

For a given run, the selected winner and decision are invariant to dropping
up to K trailing prompt tokens. The claim is falsified by a flip (winner or
decision change) at any depth ≤ K; the readout is the first flip depth in
tokens, or "no flip up to K". Scope: a per-run readout. A flip exhibits
prompt sensitivity at token depth; it proves nothing about weights, and a
no-flip result acquits the run only up to K, never in general.

## Monday-morning utility

A prompt-format researcher deciding whether a comparison's winner survives
losing its trailing prompt tokens — for example the multi-token assistant
turn-closing header — checks the token-depth flip readout before trusting
the verdict.

## Invariants (binding on the build)

1. The token mode shares the sweep's search and selection machinery
   (same native baseline, same decision rule via `run_experiment`).
   It is a parallel implementation — a new truncation wrapper, a
   parameterised/second sweep, CLI validation, and reporting — not a
   unit swap of the byte sweep, which hardcodes
   `TruncatedPromptBackend` (experiments.py:3497).
2. The depth unit is the backend's own tokenization: dropping K trailing
   token ids from the encoded prompt. No text decode round-trip sits in
   the measurement path — detokenization would confound token identity
   with decode fidelity.
3. Fail-closed with an explicit status: every token depth carries a
   per-depth `status` field. `winner_changed` is gated on
   `status == "applied"`; `flips_at` derives from the gated flag; and
   inapplicable depths are listed separately in `inapplicable_at`.
   Inapplicability is decided by a pre-run token-length check (K at or
   beyond the prompt's token count), never inferred post-hoc from a
   `failed` decision — otherwise a genuine run failure is relabelled
   inapplicable and vice versa. Status is per-depth, because one
   un-truncatable prompt aborts the whole depth (the `run_experiment`
   early-return path, experiments.py:3742–3744). An inapplicable depth
   keeps `decision: "failed"` for that depth, and the spec states that
   `status` governs all flip consumption. This corrects the byte
   sweep's known flaw, where a failed depth is counted as a flip
   (experiments.py:3509, :3523) — byte mode itself stays untouched,
   as test_cycle17 pins its behaviour.
4. The payload labels its granularity: byte and token readouts must be
   distinguishable in the recorded result, so no consumer can mistake a
   token depth for a byte depth. The token payload is a separate key,
   `sensitivity_sweep_tokens`, carrying `granularity`,
   `max_drop_tokens`, and per-depth `drop_tokens` / `status`; the
   sweep payload carries no fingerprint, so granularity must live in
   these fields. The token wrapper's fingerprint suffix is
   `+prompt-trunc-{K}t`, distinct from the byte wrapper's `b` suffix
   (backends.py:788).
5. Compute multiplies by depth count, exactly as the byte sweep's does.
   Ladder bounds must be justified against that cost.
6. Eligibility: a backend is eligible when its `generate` consumes ids
   as given. The rule covers id-rewriting wrappers, not only char-wise
   backends: `TokenHealingBackend.generate` forwards healed ids, so a
   token wrapper stacked over it would measure healed ids, not prompt
   ids. The token wrapper is defined on the base prompt encoding: it
   must never naively slice a prefill-wrapped encoding, because
   `PrefillBackend.encode` appends prefill ids that a naive `[:-K]`
   would drop first, leaving the prompt intact. Scripted backends are
   plumbing-only: they may prove the machinery runs, but never P4
   evidence (a flip on a scripted backend says nothing about real
   weights).

## Frozen acceptance probes (what the build must demonstrate)

- P1 (shape): token mode on a known fixture produces the sweep's readout
  shape with granularity labeled and a native baseline present.
- P2 (fail-closed): a depth beyond the prompt's token count records a
  failed decision for that depth with `status: "inapplicable"`, is
  excluded from `flips_at`, and appears in `inapplicable_at` — not a
  skip, and not a flip.
- P3 (no regression): byte mode behaves exactly as before; its existing
  tests pass with no behavior change, and byte report HTML is
  byte-identical.
- P4-reach (discrimination, proven existentially before build): every
  shipped suffix is longer than 8 bytes and begins with a special token
  (chatml 22 B, llama3 47 B, gemma 21 B, measured from `Format.render`
  and the catalog), so dropping all suffix tokens yields the pre-suffix
  prompt, which no byte drop ≤ 8 can yield. P4-reach existence is
  therefore proven for all three families, tokenizer-free. What the
  build must still measure, per checkpoint tokenizer (amendment A4),
  is *which* K that whole-suffix depth is per candidate.
- P4-flip (reported measurement, never a build gate): on a named
  checkpoint, a token-depth flip is reported either way. A no-flip
  result is a measurement, not a build failure, and a flip on a
  scripted backend is plumbing-only, never P4 evidence.

## Resolved design questions (adjudicated)

- Q1 Ladder: contiguous 1..max. Default 4, cap 8 are working
  hypotheses, frozen only after A4 (per-candidate suffix token counts
  measured on the real checkpoint tokenizer); A4 may raise the cap
  with justification, never silently. Contiguous beats sparse: a
  sparse ladder can skip the whole-suffix depth. Cost: native + one
  `run_experiment` per depth, so 5 runs at default 4 and 9 at cap 8
  per mode.
- Q2 CLI surface: separate combinable flags `--token-depth-sweep` +
  `--max-drop-tokens` alongside `--sensitivity-sweep` /
  `--max-drop-bytes`, sharing one native run. Sharing saves exactly
  one `run_experiment` versus two sweep calls (not two: the CLI main
  run is already a separate native). Token flag validation mirrors the
  byte flag's 1..8 range (cli.py:2967–2968) unless A4 justifies a
  wider cap. Token mode inherits the `diagnostics_unavailable`
  fallback (cli.py:4095–4096); this is documented, not hidden.
- Q3 Reporting: separate key `sensitivity_sweep_tokens` and a separate
  report section, via reporting.py's three touchpoints (compute, TOC
  tuple, body concatenation). Generalising `_sweep_section` is allowed
  only if byte HTML stays byte-identical (test_cycle33 pins it).
- Q4 Motivation: the assistant-header case is named as the motivating
  probe; the ladder itself stays generic.

## Uniqueness (combination, not "first")

Exact token-sequence selection machinery + a token-depth truncation ladder
+ a flip-depth readout on a local checkpoint. Neighbors checked in the
research pass (RESEARCH_LOG cycle-206 entry): prompt-sensitivity work
studies paraphrase and template sensitivity of scores (Hua et al.,
arXiv:2509.01790), not truncation ladders; lm-eval-harness truncation is
a length cap on few-shot examples when prompts are too long, not a
sensitivity probe; no token-depth prompt-truncation ladder over a fixed
selection rule was found in HELM or LightEval. The surviving claim is
this combination.

## As built (cycle 206)

- Pre-run length check scope: the minimum prompt token length is
  taken over search AND held-out prompts for every candidate (a
  conservative superset, stated in the sweep docstring: a depth
  whose truncated winner's held-out prompt alone was too short is
  labelled inapplicable rather than claimed as a flip, because a
  flip claim must never rest on a prompt that never ran).
- B1 is enforced, not just documented: `TokenTruncatedPromptBackend`
  refuses a prefill-wrapped inner at construction. B2 is documented
  with fingerprint transparency: a sweep over an id-rewriting backend
  measures healed ids, and the fingerprint chain shows the healing
  step; such runs are plumbing-only, never evidence.
- Shared native: when both sweeps are requested the CLI computes
  one native and passes it to both (saving exactly one
  `run_experiment` versus two sweep calls); each sweep alone
  recomputes its native, as before. Byte sweep behaviour, payload,
  and HTML are byte-identical; its failed-depth-counts-as-flip
  semantics are pinned and unchanged.
- Default 4 / cap 8 remain provisional: A4 (per-candidate suffix
  token counts on a real checkpoint tokenizer) was NOT measured in
  this cycle — no checkpoint is available in the gate environment
  — and no real-checkpoint count is claimed anywhere. The payload's
  `suffix_token_counts` (standalone-suffix encoding, documented as
  an annotation, not in-prompt tokenization) is the mechanism by
  which a future run performs A4 and may raise the cap with
  justification, never silently.

## What would prove this design wrong

- Token-depth flips turn out to be fully explained by byte-depth flips
  at corresponding boundaries on every fixture tried.
- The token unit cannot be defined without a decode round-trip on the
  lab's backends (invariant 2 fails).
- (Retired by adjudication: "P4 cannot be constructed" — the
  existential whole-suffix argument proves P4-reach for chatml,
  llama3, and gemma before any build.)
