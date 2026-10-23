# Load-precision × template (`run --precision`) — design

Cycle 198 (wave-8 rank #7). This document is the shipped
contract. It folds the wave-8 research brief
(docs/RESEARCH_LOG.md, wave-8 round), the cycle-198 researcher
pass, red-team A's nine amendments (A1–A9), red-team B's nine
amendments (B1–B9), and a verifier pass that confirmed all ten
load-bearing factual claims the two red teams made about this
tree. Where the brief and the shipped contract differ, this
document wins, and the differences are the amendments.

## What this is, honestly framed

`--precision` is a **new condition arm of the lab's established
second-condition selection-survival pattern** (transfer,
token-healing, perturbation, placement, …), not a new
instrument. What is new is the condition itself — the **load
dtype of identical weights** — which is the first
weights-representation condition in the lab (every prior
condition varied checkpoint, prompt, fixtures, decoding, or
batch composition), and the one-run packaging: paired
per-fixture verdict flips, an output-divergence readout,
teacher-forced logprob deltas, and selection re-application
under the second condition, all fail-closed diff-validated.

Monday-morning utility: a practitioner who will serve the
checkpoint in bf16 learns, before deploying, whether the
template winner the lab selected under its fp32 measurement is
still the winner when the same weights are loaded in bf16 —
and gets the per-fixture flip list if it is not.

It measures **dtype precision**, not quantization. The words
"precision" and "quantization" are not interchangeable
anywhere in the CLI help, this document, or USAGE: the bf16 arm
is a load-time cast of the same unquantized safetensors assets,
and the backend's existing `quantization_config` rejection is
untouched, which is what structurally keeps the two apart.

## Prior art and the scoped uniqueness claim

No surveyed tool or study crosses load precision with a
candidate set of hand-rendered template families and asks
whether the *selection* survives. The neighbours, named so a
reviewer does not have to discover them:

- Yuan et al., "Give Me FP32 or Give Me Death?"
  (arXiv:2506.09501): bf16 numerics demonstrably flip greedy
  outputs (up to 9% accuracy spread) when GPU count/type/batch
  size vary *under* bf16. Runtime configuration is the variable;
  no rendering, no selection. It is this probe's motivation,
  not its answer.
- Du et al., "Greedy Decoding Is Not Precision-Invariant"
  (arXiv:2609.26621): BF16-vs-FP16 output divergence is large
  (49–100% of prompts); prompt format appears only as a
  one-model robustness control for an inference intervention.
  No candidate set, no selection rule, no winner survival.
- Szalontai et al., HumanEvalFix reproducibility study
  (Software 2025, 4, 17): precision moves *aggregate* scores
  little (bf16→fp32 mean shift −2.19%/+1.01%; greedy-baseline
  SD across precision/quantization settings 1.49), and the
  paper's own conclusion is that precision is a minor factor
  for aggregate results. Cited with that conclusion attached:
  an aggregate mean cannot see per-candidate flips that cancel,
  and the study never crosses precision with template or asks
  about a selection. That gap is the probe's justification.
- Zheng et al., Qwen3 PTQ study (arXiv:2505.02214): model
  quality under post-training quantization. Out of scope by
  construction; it is what this feature will be mis-cited as
  replicating, hence the disclaimer discipline above.
- A practitioner blog's remark that computing log-softmax in
  bf16 "loses enough precision to flip close choices" is a
  folk claim about *scoring-compute* precision, with no
  experiment behind it; the lab's scoring path already upcasts
  logits to fp32 and accumulates in float64, so the sentence
  does not describe this pipeline's mechanism and is **not**
  cited as evidence anywhere in the shipped docs. In a bf16
  load, the precision loss occurs in the model's forward pass
  (weights and activations), not in a bf16 softmax.

## The measurement

Arms. The fp32 arm is the recorded native run — never
re-executed. The bf16 arm is one additional search-split
evaluation of every candidate under a second `LocalHFBackend`
instance loaded with `dtype="bfloat16"` from the same
checkpoint directory. The primary backend is **always fp32**:
`--precision` adds an arm; it never replaces the primary, never
writes `result["decision"]` or `result["selected_candidate"]`,
and never affects the exit code.

Native gate (A1). The stage runs only when the native search
is complete: the choice-audit wrapper gate is adopted verbatim
— decision in {failed, cancelled} or `len(search) !=
len(formats)` yields
`{"available": False, "detail": "native_search_incomplete"}`
with zero forwards. Per candidate, native outcomes come from
`native_search_outcomes`; a candidate whose native report is
not aggregate-eligible has `passed=None` and is excluded from
every cell, named in the payload's unavailable list.

Forwarding set (A2). The bf16 instance and stage receive
exactly: the same checkpoint path, `device=args.device`,
`max_context=args.max_context`, `max_new_tokens=args.max_new_tokens`,
`seed=args.seed`, the same `formats` tuple, `tuple(split.search)`
as the fixture set, and `boundary=args.boundary_ids` **and**
`boundary_map=boundary_map`. The bf16 instance is constructed
only in the stage phase, after `run_experiment` has returned;
the top-level `backend_fingerprint` is never overwritten. Any
omission silently compares two different measurements, which
is the one failure this lab exists not to produce. EOS needs
no forwarding: `evaluate` never passes an EOS to `generate`.

Dtype integrity (A3/A4/B4). `LocalHFBackend` gains a
keyword-only `dtype` parameter restricted to
`{"float32", "bfloat16"}`; anything else (including
`"float16"` and `"auto"`) fails through the existing
constructor wrapper as
`RuntimeError("local_backend_initialization_failed")`. Only the
`dtype=` argument value changes in the `from_pretrained` call;
every other kwarg is byte-identical, so the SECURITY.md
loading invariants stand (their AST pins scan kwarg structure,
not the dtype value — stated here so nobody overclaims what
they pin). The identity dict's dtype entry takes its literal
from the constructor parameter — exactly `"float32"` or
`"bfloat16"`, never `str(torch.dtype)` and never the loaded
config (executed fact: after a bf16 load the config object
still reports float32; deriving the identity from config would
collide the two arms' fingerprints). After load, the
constructor asserts the model's actual parameter dtype equals
the requested dtype and fails closed otherwise; the backend
exposes the verified dtype. The bf16 arm is an unconditional
load-time cast of the same assets, regardless of storage
dtype — there is no config-permission gate. The payload
records the checkpoint's declared dtype (both the `dtype` and
`torch_dtype` config keys, as found) and both arms' verified
dtypes, and scopes its wording as *load/compute* precision:
a checkpoint stored in bf16 loads today as fp32 compute over
bf16-stored weights, and the report says so when that is the
case.

Native-dtype pin (B5). The payload records `native_dtype`
from the primary backend's verified dtype, and the stage is
unavailable when it is not `"float32"` — the reuse argument is
an executable check, not a design-doc paragraph. A pin test
holds the default path's fingerprint byte-identical:
`LocalHFBackend(root)` == `LocalHFBackend(root,
dtype="float32")` != `LocalHFBackend(root, dtype="bfloat16")`.

Stage shape (A5/B2). The stage is healing-shaped, not a
`transfer_stage` reuse: per candidate it returns
`{"passed": bool | None, "per_fixture": {fixture_id: bool},
"n_passed", "n_total"}` for the bf16 arm, with `passed=None`
and empty `per_fixture` on any non-aggregate-eligible report.
Pairing is by fixture ID against `native_search_outcomes`,
never by list position.

## The readouts

Flip matrix (A5/A6). Per candidate, over fixtures completed
in **both** arms: the four paired cells (pass/pass, pass/fail,
fail/pass, fail/fail) with `n_paired` beside them, plus
`n_fp32_only` and `n_bf16_only`, plus a separate unpaired list
(candidate, fixture, arm) — a fixture an arm never reached is
unpaired, never an arm-failure. The flip list proper contains
only true paired flips, sorted by (candidate, fixture), each
with its direction. Rates are `None` at `n_paired == 0`,
never a division (a NaN would crash the report write itself:
results serialise with `allow_nan=False`).

Output divergence (B3). Per paired fixture, the stage also
records whether the bf16 generation's token IDs diverge from
the native generation's (`determinism.compare_token_ids`
semantics), the first-divergence index, and both generations'
output digests. This is the sensitive leading indicator;
verdict flips are the lagging one. Per-candidate and overall
divergence counts ride beside the flip matrix with their
denominators.

Logprob deltas (B3). Per paired fixture, the teacher-forced
logprob of the expected answer under each arm
(`scoring.score_target_logprobs`, float64 accumulation,
recorded from reported values), and their delta (bf16 − fp32).
The fp32 values are computed on the native backend during the
same run — one forward per candidate × fixture, the cost class
the lab already accepts for the opt-in logprob diagnostic.
Per-candidate mean delta and the min/max fixture deltas are
derived by the validator from the recorded per-fixture values,
never trusted from the producer.

Winner survival (A6). Defined only when the primary decision
is winner-bearing (`compatible_on_fixture_set` /
`held_out_failed` with a selected candidate, the
`_valid_primary_pair` vocabulary). The comparison re-applies
the selection rule to **both** arms over their shared paired
population (the placement precedent) and compares
`(decision, winner)` tuples — fp32 `selected W` vs bf16
`ambiguous` is a survival failure a winner-only comparison
would miss. The bf16 arm's decision uses the existing
vocabulary `{selected, ambiguous, no_candidate, unavailable}`.
Survival is withheld — reported as unavailable with its fixed
detail — on any per-candidate cross-arm completed-set
mismatch, on a non-winner-bearing primary, on bf16
unavailability, and at `n_paired == 0`.

Falsification, scoped (B3). Zero verdict flips across all
candidates falsifies a precision effect **on the search split,
at the verdict level, in a single bf16 run compared against
the recorded fp32 run, for this checkpoint and fixture set** —
and the readout's sensitivity is conditioned on the pass base
rate: on a fixture set where no candidate passes in either arm
(the random-weight tiny checkpoint is exactly that), zero
flips is guaranteed by the floor effect, not earned by
precision robustness, and the divergence/logprob readouts are
the informative ones there. The falsification readout is
`None` over zero paired cells. Two things are **not**
claimable and the docs must not claim them: a per-step
top1–top2 margin ("short margin to the runner-up token") is
not observable post-hoc — `core.Observation` records no
logits or margins and `generate()` discards each step's logits
after argmax — and no causal attribution of any single flip.
Held-out behaviour under bf16 is never measured by this probe.

Operator action (B7). The fp32 selection stands as the run's
decision. A non-surviving winner means: do not serve this
winner in bf16 on this evidence. The bf16 arm's winner is a
search-split readout, not a selection; it carries no held-out
evidence, and there is no supported primary-bf16 re-run —
USAGE says all of this in plain words.

## Payload and diff validation (A7/B6)

The precision section carries digest-bound ground records for
both arms: per candidate × fixture, `{verdict, boundary_found,
answer_digest}` per arm in the placement-cell style, with
`verdict == (answer_digest == expected_digest) and
boundary_found` re-derivable by the validator and
`expected_digest` computed from the fixture's expected answer
with the tree's surrogatepass-safe digest. **Digests only —
never answer text**: stage payloads pass verbatim into the
default results.json. Divergence records carry both output
digests, never raw token IDs beyond the first-divergence
index. A protocol object binds: version, seed,
`max_new_tokens`, per-candidate boundary token IDs, the
fixture-set SHA-256 over the search fixture IDs in split
order, candidate names, both backend fingerprints, the
declared config dtype, and both verified arm dtypes; the
validator requires `bf16_fingerprint != fp32_fingerprint` and
the payload's context fingerprint equal to the fp32 arm's.
The diff validator recomputes **everything** derived — matrix
cells and denominators, flip list (exact set and order),
divergence counts, logprob deltas and their summaries,
per-candidate verdicts, both paired-population selections,
the survival flag, the falsification readout — compared
type-strictly (`_recomputed_equal`), under records caps
before any recompute loop (legal max 24 candidates × 100
fixtures). Protocol or fingerprint drift surfaces as
`precision_backend_drift` and *not comparable*, while the
section is still listed (the placement drift pattern).
Honest ceiling, stated plainly: user-fixture expected answers
cannot be regenerated by the validator, so the ground cells
are attested at the digest binding — the same residue
`_valid_primary_pair` documents for the primary pair. Where
the same results.json carries search observations (evidence
mode), the validator additionally cross-checks the fp32
ground cells against them.

Determinism scope (B6). A `--precision` results.json still
verifies under `verify --determinism` because the primary
fingerprint is the fp32 one; the certificate covers the
native fp32 search observations only — the bf16 arm's
observations are never re-executed — and the precision section
and USAGE both say so.

## Cost (A8/B8)

`--precision` adds: a second full search evaluation, a second
full checkpoint load (asset re-hashing included), and both
models resident at once (≈1.5× the fp32 weight footprint plus
activations). Measured in this lab's sandbox, bf16 generation
on CPU ran up to ~6× slower than fp32 on the toy checkpoints —
on CPU the stage can dominate run time; on CUDA bf16 is the
serving-fast path (untestable here, per VALIDATION). The help
text says this, replacing any "reuse halves the cost" framing:
reuse is a saving relative to running both arms fresh, not
relative to the status quo. The stage threads the run's cancel
event into its `evaluate` calls inside an
`except Exception → precision_unavailable` wrapper; the
remaining SIGINT exposure of the whole stage phase (signal
handlers are restored before it) is inherited from transfer
and is named here rather than inherited silently.

## Guards and failure vocabulary

The stage uses `generate`/`evaluate` only — the paths whose
non-finite guards are verified: `generate` checks
`torch.isfinite(logits)` before argmax and returns
`invalid_next_token_logits`, which `evaluate` collapses to the
fixed `backend_reported_error`; the candidate degrades to
`None`. A forced-non-finite bf16 test pins this (the tiny
checkpoints do not exhibit bf16-only non-finites under
deliberate attack — LayerNorm/RMSNorm shield the residual
stream — so the pin is a mocked-forward requirement, not a
tiny-checkpoint claim). Stage-level failure uses the tree's
fail-closed shape `{"available": False, "detail":
"precision_unavailable"}` or the specific fixed details
above; no new verdict words are invented.

## Tests

- Backend: fingerprint pins (default == explicit fp32,
  != bf16); dtype rejection (`"float16"`, `"auto"`, non-string)
  through the constructor wrapper; post-load param-dtype
  assertion; declared-dtype recording for both config keys;
  bf16 load/generate on the tiny GPT-2 and Llama checkpoints;
  forced-non-finite bf16 → candidate `None`.
- Stage/payload: scripted factory fakes (the transfer/healing
  test pattern) exercise real flips, divergence, survival and
  non-survival, withheld paths, the native gate, and unpaired
  denominators. The tiny-checkpoint e2e asserts only
  availability and the degenerate matrix (no passes in either
  arm) — no doc cites it as sensitivity evidence.
- Invariant: a `--precision` run's top-level fingerprint,
  decision, selected candidate, and exit code equal the same
  run without the flag.
- Validator: the full hostile battery at the tree's standard —
  one broken identity at a time across every derived field,
  fingerprint-equality forgery, drift-as-incomparable, caps.
- Blast-radius pins, updated in the same change: the new pure
  module moves the test_cycle44 source-file pin (57 → 58);
  CLI audits (cycle 31/64/84), the args-Namespace contract,
  the cycle-179 dispatch e2e, README/VALIDATION counts
  (measured); the fp32-as-absolute statements at
  `backends.py`'s class docstring, `docs/USAGE.md`, and
  `README.md` are scoped to the primary/default measurement
  in the same change (cycle-172 lesson). No doctor plumbing:
  doctor never constructs a backend.
