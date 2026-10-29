# Architecture decision records — index

Every instrument in the lab exists because a design round decided
it should, in a specific shape, after adversarial review. This page
is the one-line index of those decisions: what was decided, and
what came of it. The full reasoning — population, scorer, decision
rule, and the counterexamples that killed the first sketches —
lives in the linked document; this index states outcomes, it does
not re-argue them.

The index lists only decisions whose documents ship on `main`. A
design round that is still in adjudication, or that returned
NO-BUILD with a resubmission candidate, has no entry here until
its document ships. A regression test keeps this page and the
shipped set in step: every design document in the site navigation
must appear below, and nothing may be listed that is not in the
navigation.

## Probe and instrument designs

| Document | Decision | Outcome |
| --- | --- | --- |
| [Aliasing audit](../ALIASING_AUDIT_DESIGN.md) | Audit candidates for byte-identical renderings before comparing them; rendering identity and verdict identity are two class notions, never conflated. | Shipped as `run --aliasing`. |
| [Batch path](../BATCH_PATH_DESIGN.md) | Give the local backend a real batched generation path and test whether verdicts measured solo survive it, with a frozen verdict vocabulary. | Shipped as `run --batch-invariance`. |
| [Canonical form](../CANONICALFORM_DESIGN.md) | Measure NFC-vs-NFD sensitivity with accents confined to prose and answer values kept pure ASCII; two arms only, no NFKC. | Built in cycle 209 as `run --canonical-form`. |
| [Choice audit](../CHOICE_AUDIT_DESIGN.md) | Audit log-likelihood choice-scoring against the lab's greedy-generation winners, per template family. | Built in cycle 195 as `run --choice-audit`, with amendments after the red-team cross-check. |
| [Confidence](../CONFIDENCE_DESIGN.md) | Append one frozen suffix asking for a stated confidence, and set it against the lab's own exact-match verdict. | Shipped as `run --confidence`. |
| [Dateline](../DATELINE_DESIGN.md) | Change only the prepended "Today is …" line and count verdict flips per family. | Built in cycle 188 as `run --dateline`. |
| [Density](../DENSITY_DESIGN.md) | Ladder simultaneous-instruction density, with every verdict re-derived by the diff validator; the adjudicated v2 replaced the defective v1. | Built in cycle 204 as `run --density`. |
| [Determinism audit](../DETERMINISM_AUDIT_DESIGN.md) | Self-audit the determinism assumption every recorded result rests on, and state plainly what the audit does not prove. | Shipped as the determinism self-audit (cycle 173). |
| [Distraction](../DISTRACTION_DESIGN.md) | Pour irrelevant context around a chosen candidate's prompt and find where it starts failing — a deployment question the selection loop cannot see. | Shipped as `run --distract`. |
| [Field order](../FIELDORDER_DESIGN.md) | Move the main answer field first vs last in a JSON schema and compare the values produced for it. | Built in cycle 205 as `run --field-order`. |
| [Fixture fuzz](../FIXTURE_FUZZ_DESIGN.md) | Fuzz the lab's own fixture loaders against a published contract table. | Built in cycle 172; the document is the contract the corpus implements. |
| [History frame](../HISTORYFRAME_DESIGN.md) | Prepend one frozen six-turn synthetic history to every search fixture and reframe only that prepended segment in the frozen chatml/llama3 framings, holding the tail, bos, and open suffix at the candidate's own; verdict discordance gated on render/token-id divergence with a McNemar classification. | Built in cycle 219 as `run --history-frame`, as amended (A1–A10) after the fresh review. |
| [Instruction distance](../INSTRUCTIONDISTANCE_DESIGN.md) | Dose the rendered-token separation between one frozen format directive and the block it governs, with filler presence differenced out by a dose-matched displaced control; compliance scored under a frozen grammar, separately from correctness. | Built in cycle 218 as `run --instruction-distance`, as amended (A–J) after the fresh review. |
| [Leading claim](../LEADING_DESIGN.md) | Measure whether a candidate caves to a wrong assumption the user states in the prompt. | Shipped as `run --leading`. |
| [Multi-turn drift](../MULTITURN_DESIGN.md) | Re-run selection with the model's own earlier replies in context, not just single exchanges. | Built in cycle 189 as `run --multiturn`, amended after an initial do-not-build round. |
| [Needle depth](../NEEDLE_DESIGN.md) | Plant a fact at anchored depths in surrounding bulk and read recall against depth, per family; verdicts are digest-bound. | Built in cycle 191 as `run --needle-depth`. |
| [Negation](../NEGATION_DESIGN.md) | Claim one-directional antonym-negation discordance (direct / negated / flipped arms, an (8,5) decision rule) — never "negation invariance". | Built in cycle 212 as `run --negation`. |
| [Numeric form](../NUMFORM_DESIGN.md) | Vary numeric surface form per task kind and candidate, and watch copying, comparison and addition separately. | Built in cycle 186 as `run --numform`, as amended. |
| [Placement](../PLACEMENT_DESIGN.md) | Question before vs after the payload, as the only variable. | Built in cycle 193 as `run --placement`. |
| [Precision](../PRECISION_DESIGN.md) | Treat load precision as a second condition in the established selection-survival pattern, not as a new instrument shape. | Shipped as `run --precision`. |
| [Repetition](../REPETITION_DESIGN.md) | Repeat the request inside the turn while holding prompt length constant. | Shipped as `run --repeat`. |
| [Rotation](../ROTATION_DESIGN.md) | Cyclically rotate the same retrieved chunks and compare verdicts. | Built in cycle 203 as `run --rotate`. |
| [Scorer lens](../SCORER_LENS_DESIGN.md) | Regrade recorded outputs under relaxed scorers without re-running generation, to separate "wrong answer" from "right answer, wrong surface". | Shipped as `run --scorer-lens`. |
| [Sealed bundle](../SEALED_BUNDLE_DESIGN.md) | Seal a report directory into one hash-bound bundle with offline re-verification; the seal is unsigned and claims bit-integrity only, never attestation. | Shipped in cycle 171 as `pack` / `verify --bundle` (format `mpl-bundle/1`). |
| [Surface noise](../SURFACE_NOISE_PROBE_DESIGN.md) | Measure verdict stability over a frozen operator set, honestly named a self-consistency probe — never an unqualified robustness score. | Frozen for cycle 165; shipped as the surface-noise probe. |
| [Template probe](../TEMPLATE_PROBE_DESIGN.md) | Probe templates behaviourally: the static AST firewall cannot see content-conditional logic written in legal Jinja. | Shipped as `doctor --template-probe`. |
| [Template robustness](../TEMPLATE_ROBUSTNESS_DESIGN.md) | Report the fraction of a candidate's correct verdicts that survive surface-noise perturbations. | Shipped as the per-template robustness stage (design v2, cycle 166). |
| [Token depth](../TOKENDEPTH_DESIGN.md) | Sweep trailing-token loss and read whether a comparison's winner survives it, before trusting the comparison. | Built in cycle 206 as `run --token-depth-sweep`, on the round's converged amendments. |

## Standing decisions recorded in other shipped documents

| Record | Decision | Outcome |
| --- | --- | --- |
| [Security and data handling](../SECURITY.md) | No SBOM is shipped: a dist-scoped SBOM of a zero-runtime-dependency wheel adds no inventory beyond the artifact's own metadata (cycle 210, upheld on re-adjudication). | Recorded in the security document; the workflow-security lanes added in cycle 215 (zizmor at the pedantic floor, dependency review on pull requests, job-scoped workflow permissions) are recorded there too. |
| [Release readiness 0.1.0](../RELEASE_READINESS_0.1.0.md) | Release readiness is assessed against evidence, not asserted; building reproducibly and publishing through Trusted Publishing are set up, and cutting the tag remains the owner's decision alone. | The readiness record carries the checklist and its evidence. |
