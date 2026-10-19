# Surface-noise self-consistency probe — design

**Status:** frozen design for cycle 165. **Instrument name (honest):**
*surface-noise self-consistency probe* — never "prompt-quality meter",
never unqualified "robustness score".

**Monday-morning utility:** a developer about to ship a prompt runs the
probe to see which surface-noise family flips their task success and how
close the nearest observed flip is — before production typos find out first.

**Positioning (web-verified 2026-10-19):** ctxray (MIT) is the *static*
tier — deterministic 0–100 scoring over 30+ prompt features, lint,
rewrite, MCP server — but no perturbation, no radius, no counterfactuals,
not stdlib-only. This probe is the *dynamic* tier: perturbation-based
verdict stability over a frozen operator set. The two compose; they do not
overlap. Academic credit: CoIPO/NoisyPromptBench "decoding radius" R(a)
(arXiv:2603.03314, ICLR 2026) is the model-centric precedent for a
perturbation radius; the 2026 PCE paper is the precedent for minimal-flip
counterfactuals (LLM-based, research-only). Our combination — per-prompt,
frozen deterministic protocol, stdlib-only MCP, zero LLM calls — has no
complete competitor as of 2026-10-19 (see `docs/RESEARCH_LOG.md`).

## 1. The verdict function V (closes R1)

**Subject.** The instrument under test is a *prompt text* P: the
instruction/template span the user supplies. Fixture message content is
**never** modified (this resolves R6: perturbing task content while the
expected answer stays pinned would punish the prompt for the fixture's
brittleness).

**Rendering (frozen).** Each evaluation renders

```
Instruction: <P>

<role>: <content>
<role>: <content>
...
```

with fixture messages verbatim, then generates with the backend (greedy,
exact token sequences). Template choice is out of scope — the lab's other
experiments measure template effects; the rendering here is fixed and
documented.

**Definition.** V(P, F, B) is the pass/fail vector over fixtures
F = (f_1 … f_n) with backend B. Position i is:

- `"pass"` — generation completed and the decoded output exactly equals
  `f_i.expected`;
- `"abstain"` — generation did not complete (`cancelled`/`error`);
- `"fail"` — otherwise.

**Baseline.** V computed on the unperturbed P with the same (F, B).

**Flip predicate.** A variant P′ flips iff its pass/fail vector differs
from the baseline vector on any mutually-evaluable fixture (evaluable in
both runs). A change in *either* direction counts — a variant that fixes
a failing fixture is still instability, because the prompt's behavior
changed under noise. No τ threshold: any vector difference is a flip.
This is the strictest well-defined predicate and has no magic numbers.

**Falsifiability.** "P is stable under operator O" is falsified by
exhibiting one seed in the frozen plan whose variant flips. "No flip
observed under plan S" is falsified by exhibiting any flip within S.

## 2. Distance (closes R2)

One normalized unit *within* each operator family — never a cross-operator
scalar. The token indel-edit rate

```
d(P, P′) = indel(encode(P), encode(P′)) / (len(encode(P)) + len(encode(P′)))
```

where `indel` is the insertion/deletion edit distance (a substitution
counts as delete + insert = 2), computed exactly by Myers' O(ND) greedy
algorithm — exact and fast for near-identical sequences, which
perturbations are. d ∈ [0, 1] by construction (indel ≤ n + m). Reported
per operator; cross-operator comparison is explicitly not-comparable
(different perturbation classes). There is no collapsed "robustness
gauge" — the per-operator bars are the report. The substitution-=2
convention is stated, not hidden: what the unit measures is "fraction of
combined token mass inserted or deleted".

## 3. Nearest observed flip (closes R4)

Per operator: d_obs = min d over flipping *effective* variants in the
frozen search plan S, or `"none observed"`. Labeled everywhere as
**"nearest observed flip under frozen search plan S"** with S published
verbatim and coverage stated ("8 of 2⁶⁴ seeds searched"). Exact where
enumerable: `punctuation_strip` (exactly 1 outcome),
`whitespace` (exactly 3 modes enumerated). Seeded operators (`typo`,
`case_flip`, `unicode_variant`): heuristic, 8 seeds each.

d_obs ≥ d\* always (min over a subset ≥ min over the full space): the
reported number is an **upper bound** on the true minimal-flip distance
and errs toward **false confidence**. It is never a guarantee, never a
certificate. There is no separate "margin" metric — the nearest observed
flip *is* the margin, reported with its truncation stated (R12a).

## 4. Frozen protocol (F4)

`freeze_surface_noise_protocol()` publishes, *before* any scoring:

- instrument name + version, operator names + sha256 of
  `perturbation.py` (code hash),
- the search plan verbatim: `{operator: [seeds]}` or `"modes"`/`"single"`,
- the prompt/task split ("perturbation applies to the prompt text span
  only; fixture message content is never modified"),
- the frozen rendering, the flip predicate, the denominator definition,
- backend identity + weight-file fingerprint (R13),
- fixture id list + hashes, creation timestamp.

The probe refuses to run when the protocol's code hash differs from the
current `perturbation.py`, or the backend fingerprint differs from the
frozen one. Scores are comparable only under identical
(fixtures, backend, protocol-fingerprint) triples (R12c).

## 5. Denominators (closes R8, R12d)

Every result reports, per operator,
`n_total / n_effective / n_abstained / n_error`.
`n_effective` = variants whose perturbed prompt span changed the
**token-id sequence** (via `backend.encode`) — text-level "changed" is
insufficient (e.g. case changes can be tokenizer-invisible). No-op
variants are listed and never counted as stable. Stability per operator
= n_unchanged / n_effective.

## 6. Controls (R9, with the circularity documented)

Self-contained calibration harness: each control declares the backend
output it expects; `run_surface_noise_controls` builds dedicated
fixtures from them and scores them with a fixed calibration backend
(`_ControlCalibrationBackend`: exact `"exactly: OK"` → `"OK"`, exact
`"SAY IT LOUD: HELLO"` → `"LOUD"`, any other non-empty prompt →
`"BLUE"`) under a freshly frozen protocol. The sturdy controls are
stable *by construction* (perturbations never produce empty text, so the
non-empty rule never flips: stability exactly 1.0); the fragile controls
flip deterministically under the operators (measured: 0.15 / 0.0).
Discrimination check: mean sturdy stability > mean fragile stability.
A pass validates the *implementation* against the *specification*.
Provenance per control is designer-authored and documented as encoding
the operator set's own theory of fragility — the circularity is stated,
not hidden (Adapter Rule). One held-out control set lives outside the
test suite; CI pins its count and provenance, not its content.

## 7. Static lint tier (advisory)

Five fully-specified deterministic checks (`nonempty`,
`bounded_length`, `no_bidi_controls`, `terminal_punctuation`,
`sentence_repetition` with an exact predicate). Advisory: never gates the
sweep (R12e), never blended into a scalar (R12b). Displayed side-by-side
with the dynamic panel. (ctxray remains the serious static tier; this is
a cheap pre-sweep sanity panel, labeled as such.)

## 8. Verbosity guardrail (R3)

Prompt token length and the repetition flag are displayed alongside the
radii. Documented: noise-robustness can *anti-correlate* with concision
(redundancy survives character noise); the improvement loop must never
optimize the radius alone.

## 9. Not measured (R7 — prominent, not a footnote)

Semantic paraphrase of instructions, conflicting-instruction override /
prompt injection, hierarchy attacks, tool-call schema adherence under
noise, multi-turn drift, context-length degradation. The instrument
measures **surface-form noise only**. The MCP tool is named
`probe_surface_noise_stability` — the name is the scope.

## 10. Pseudoscience checklist (R12)

- (a) No separate margin metric; nearest-observed-flip carries its
  truncation statement.
- (b) Static and dynamic tiers side-by-side; never a blended scalar.
- (c) Cross-prompt ranking requires identical
  (fixtures, backend, protocol-fingerprint) triples, else labeled
  not-comparable.
- (d) Full denominators on every call; abstentions explicit.
- (e) The lint tier is advisory; any gating would be reported as
  censoring.

## 11. Adapter Rule (scope paragraph, shown with every result)

> This probe measures stability of task success under a frozen set of
> surface-form text perturbations. A high stability number means the
> prompt's pass/fail pattern survived *those* perturbations on *those*
> fixtures with *that* backend — nothing more. It is not a certificate
> of prompt quality, not a guarantee against real-world typos (the
> searched seeds are a sliver of the space; the reported distance is an
> upper bound), and not a measurement of semantic, adversarial, or
> injection robustness. Controls validate discrimination within the
> operator set's theory of fragility, not against real-world prompt
> failure.

## 12. Improvement-loop gating (R14)

`surface_noise_hardening_notes` surfaces the probe's own counterfactuals
(weakest operator family, nearest flips, lint findings) as *information*.
It does not run an optimization loop. Any generate→score→generate loop
built on the probe must validate acceptance on a **held-out operator
family**, keep the verbosity guardrail, and report full attempt
denominators — otherwise it optimizes the metric's blind spots, not
prompt quality.

## 13. MCP server

Stdlib-only stdio JSON-RPC 2.0 server (`tools/surface_noise_mcp_server.py`).
`--self-test` runs one embedded probe end-to-end and exits 0. Contract
wording: "local and offline" (the VulnerabilityScreener skeleton's
"never touches the network" docstring was proven false in its own code;
corrected on port).

## 14. CLI and dashboard

`mpl surface-noise probe --prompt F --fixtures F.json --boundary-ids 7
--backend selftest|local-hf [--model P] --output DIR [--max-new-tokens N]
[--seed S]` — writes `probe.json` + `dashboard.html` (refuses to
overwrite an existing path). `mpl surface-noise diff OLD.json NEW.json`
— compares two probe.json files (before/after a prompt edit).
`mpl surface-noise protocol` — prints the frozen instrument descriptor
(no scoring). Monday-morning use is printed in `--help` and the descriptor.

The dashboard is self-contained HTML (inline CSS+SVG, no external
assets, `default-src 'none'` CSP). Every interpolated string is escaped,
including SVG text nodes (R11). The prompt text is never rendered raw —
only its sha256. Minimal-flip counterfactual excerpts appear only when
the caller supplies the prompt text *and* its sha256 matches the report.
The Adapter Rule is the first thing on the page; the not-measured list
and the comparability warning are prominent, not footnotes.

## 15. Comparing two probe runs

`model_protocol_lab.diffreports.diff_surface_noise(old, new)` compares
two probe.json payloads: per-operator stability drift and nearest-flip
drift, baseline pass-count change, prompt-text change, and protocol
fingerprint change (any protocol drift flags the pair not-comparable).
`changed`/`identical` follow the module's conventions.

Intentionally NOT wired into the results.json dispatcher: the probe is a
per-prompt instrument producing standalone probe.json files, not a
per-candidate experiment. Forcing it into `mpl run` would conflate two
workflows; the improve-then-re-probe loop is `probe → edit → probe → diff`.

`tools/surface_noise_mcp_server.py`, stdlib-only (json, sys, threading —
no third-party; must run on the no-optional-deps CI job). Ported from
the VulnerabilityScreener stdio skeleton (framing, MCPError, TOOLS
registry, handle_request, real-stdio serve loop) with the contract
wording corrected ("local and offline" — the skeleton's "never touches
the network" is false in its own code). Liberal protocol acceptance:
`tools/list` / `tools/call` work with **no prior `initialize`**
(stateless, per the 2026-07-28 revision as reported in secondary
write-ups — the spec text itself could not be fetched directly at build
time and this is documented), the 2024-11-05 handshake is also accepted
but never required; `server/discover` is implemented. Bounds (FlitzZip
pattern, POSIX enforcement): 1 MiB prompt cap (stated rationale:
memory-DoS prevention; the sweep multiplies input size across variants),
≤ 50 fixtures, operators/seeds fixed by the frozen plan (no free
`n_seeds`, no free timeout), one-sweep-at-a-time lock, cooperative
cancellation between forwards with fail-closed `status: "cancelled"` on
budget expiry (R5 — never a partial score), progress content blocks
(R10).
