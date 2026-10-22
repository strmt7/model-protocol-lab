# Real batch path for the local backend — design

Cycle 192 (wave-8 rank 4). An operator who will serve the
checkpoint batched learns whether the lab's solo greedy
verdicts survive the lab's own batched execution
token-for-token — the audit that since cycle 153 could only
answer `batch_unsupported` on the real backend — before
trusting a winner measured one request at a time.

This is a capability cycle, not a new probe. Cycle 153 built
the audit (`model_protocol_lab/batchinv.py`,
`experiments.batch_stage`, `run --batch-invariance`) around
an optional backend contract and pre-registered as its
claim 1 that no backend in the tree exposes a batch path.
This cycle builds the path — `LocalHFBackend.generate_batch`
— and supersedes claim 1 by construction. The audit, its
verdict vocabulary (`batch_invariant` / `batch_variant` /
`unavailable`), and its diff section do not change; the
stage changes in exactly the ways §"Stage amendments"
lists, each forced by a red-team finding.

Design history: v1 froze two premises the red teams
disproved. Red-team A returned DO NOT BUILD AS SPECIFIED
(nine amendments); red-team B tried to disprove A and
confirmed every finding (two were understated), rejected
the coordinator's chunking adjudication as specified
(unrecorded chunk composition would manufacture invariance
for the longest prompts), and corrected the coordinator's
validation-precedence candidate in one place. This v2 is
the adjudicated contract: A's amendments 1–9 with
amendment 1 implemented as B's C1-with-changes, amendment 8
as B's C3-with-changes, and amendment 2 (the bound)
resolved as an honest byte-derived refusal with a distinct
stage detail — the resolution B's closing asymmetry argues
for and this repo's `batch_unsupported` precedent embodies:
a silent weaker measurement is worse than an honest
refusal. Briefs: `~/workspace/agents_out/wave8/`
(`cycle192_redteam_a.md`, `cycle192_redteam_b.md`).

## What supersedes what, exactly

- `batchinv.py`'s pre-registered claim 1 ("No backend in the
  current tree exposes a batch path, so on the real local
  backend the stage records `batch_unsupported`") becomes
  false the moment `generate_batch` lands. The claim text is
  rewritten in the same change to record the history: true
  through cycle 191, superseded at cycle 192 by the real
  path on `LocalHFBackend`. Claims 2 and 3 stand unchanged.
- Backends without the method — including the tree's own
  wrapper backends (`TruncatedPromptBackend`,
  `TokenHealingBackend`), which transform the token stream
  in their solo `generate` — still record
  `batch_unsupported`. The wrappers deliberately do not
  forward the inner backend's batch path: their solo
  contract rewrites inputs the batch path would bypass, so
  inheriting it would compare two different streams and
  call the difference "batch variance". (Verified at source:
  no `__getattr__` exists anywhere in the package, so the
  stage's `getattr` probe on a wrapper finds nothing.)
- The backend `fingerprint` does NOT gain a batch-capability
  field. The fingerprint already binds code and policy
  (library versions, adapter and generation-policy strings)
  — deliberately: it binds everything that can change the
  **solo recorded observations**, and batch capability
  cannot change a single solo token. Changing it would make
  every pre-192 artifact unreplayable under
  `verify --determinism` (fingerprint-gated, solo-only,
  search-observations-only) for zero measurement gain. The
  `batch_invariance` payload itself records availability.

## The implementation contract

`LocalHFBackend.generate_batch(prompts, *, max_new_tokens,
seed, cancel)` implements the signature `batchinv.py`
already documents:

- Input: a tuple of prompt id-tuples. Returns a tuple of
  continuation-only id-tuples, one per prompt, in input
  order, each of length exactly `max_new_tokens`.
- Decoding mirrors the solo `generate` exactly in every
  respect that is not batching itself: fp32 full-prefix
  recompute (`use_cache=False`), greedy argmax of the
  last-position logits, no EOS or boundary stopping, the
  same model call kwargs **plus** `attention_mask` (the real
  mask, where solo passes all-ones) and `position_ids`
  (below — solo passes none because an unpadded row's
  implicit positions are already correct), under the backend
  lock, in inference mode. The seed is validated and, as in
  solo greedy, does not influence selection. All tensors —
  input, mask, position ids, appended columns — are built
  with `device=self._device` and long dtype, matching solo.
  The method uses `self._torch` exactly like `generate` /
  `forward_logits` and adds **no imports**: `backends.py`
  imports torch only under `TYPE_CHECKING` and in `__init__`,
  which is what keeps the package importable without the hf
  extra (the cycle-105 audit guards that regime), and the
  stage's `getattr` probe must find the attribute without
  importing anything.
- Batching mechanics: prompts are left-padded to the batch's
  longest prompt with pad id 0. The attention mask is 0 on
  pads and 1 on real and generated positions. Position ids
  are **recomputed from the current mask at every step**
  (`mask.cumsum(-1) - 1`, clamped at 0), including steps
  after generated columns are appended, so every row's real
  tokens sit at the same positions they occupy solo. The
  rationale is architecture-scoped: for learned-absolute-
  position models (GPT-2) omitting position ids demonstrably
  diverges on padded rows; for RoPE models (Llama) a uniform
  pad shift preserves relative positions and the omission
  happened to be harmless in testing — the ids are passed
  uniformly for both, which was exact on both. Each
  generation step runs ONE forward over the whole padded
  batch and takes the per-row argmax at the last column
  (left padding puts every row's frontier there). The chosen
  token column and a 1-column of mask are appended; repeat
  for exactly `max_new_tokens` steps.
- Unpadding is by recorded lengths only: a row's returned
  ids are exactly the `max_new_tokens` tokens its own argmax
  produced, collected per row during the loop. Nothing is
  sliced out of the padded tensor and nothing is scanned for
  a pad id — a pad id is a legitimate token id (the pad
  value used is 0, which real prompts may contain), so any
  scan-based unpadding would be a bug by construction.
  Per-step bookkeeping extracts plain Python ints per row
  and drops the step's result/logits references before the
  next step (solo's `result = logits = None` pattern): a
  retained `logits[:, -1, :]` view would keep every step's
  full `(rows, L, vocab)` tensor alive for the whole call.
- Finite check: each step's last-position logits block must
  have shape `[rows, vocab]` and be all-finite, mirroring
  the solo per-step check; any violation fails the whole
  batch (below), never a per-row salvage — a partial batch
  result would be unattributable in the audit.
- Lock and cancel mirror solo `generate`'s discipline: the
  lock is acquired by polling (`acquire(timeout=0.05)`) in a
  loop that re-checks cancel; cancel is checked before
  acquisition, after acquisition, before each forward, and
  after each forward before committing the step's tokens.
  The batch holds the lock for its whole run, as a solo
  generation holds it for its own; concurrent callers wait
  with their own polling-cancel, and the CLI run path is
  sequential anyway. Deliberate raises never sit inside the
  inference try: cancel and failure detected mid-inference
  are carried out as flags and raised after it, because the
  tree's `heal_input_ids` re-raise guard cannot distinguish
  the method's own codes from torch `RuntimeError`s raised
  by tensor ops in the same try — inside the try, every
  exception converts to the failure flag, so nothing raw
  can leak and the cancelled code is never eaten by the
  conversion to `local_batch_failed`.

## Invariants

- **I1 (shape, falsifiable).** For valid inputs the result
  is a tuple of `len(prompts)` tuples, each exactly
  `max_new_tokens` long, each id an `int` (not bool) with
  `0 <= id < vocab_size`.
- **I2 (no pad leakage — construction invariant).** Returned
  ids come only from per-row argmax bookkeeping; the pad
  value never enters a returned tuple *as padding*. This is
  not output-falsifiable (a genuinely generated pad id is
  indistinguishable), so it is verified by construction and
  review; its observable consequence — prompts containing
  id 0 behave exactly as solo — is pinned by I4's cases.
- **I3 (real batching, anti-theater, falsifiable).** Every
  generation step issues exactly one model forward whose
  batch dimension equals the number of prompts. The pin is
  the tree's white-box mechanism (cycle-81 precedent):
  `patch.object(backend._model, "forward", wrapper)`
  records the full shape sequence — `max_new_tokens`
  forwards, every call's `input_ids.shape == (rows,
  longest + step)`. A backend-level wrapper cannot observe
  forwards and proves nothing. Secretly looping solo calls
  is the one forbidden implementation.
- **I4 (equivalence is measured, not assumed).** Property
  tests pin, on the tree's tiny GPT-2 checkpoint: batched ==
  solo token-exactly at batch sizes {1, 2, 3, 5} over
  seeded prompt sets of mixed lengths (including prompts
  containing token id 0), one 32-row case (the real caller
  batches up to 100 rows; the pin's composition should not
  stop at toy sizes), and permutation invariance (the same
  prompt with different companions and at different
  positions returns identical ids). Pre-build prototypes by
  the coordinator and both red teams confirmed these
  properties on tiny GPT-2 and tiny Llama (0 divergences
  across randomized, near-tie, near-context, and 100-row
  sets). On any real checkpoint where they fail, the audit
  reports `batch_variant` with first-divergence positions:
  divergence is a finding, not a defect to tune away.
  Nothing in this cycle adds batch-invariant kernels or any
  numeric correction.
- **I5 (fixed vocabulary, falsifiable).** Every failure is
  a `ValueError` or `RuntimeError` carrying one fixed code
  from the sets below. New `RuntimeError("local_*")` codes
  are added to `cli.py`'s `_KNOWN_LOCAL_FAILURE_CODES` in
  the same change and raised in the literal
  `raise RuntimeError("local_batch_…")` form the cycle-132
  AST pin scans for (a test pins the sets equal).
- **I6 (cancel means no result, falsifiable).** If the
  cancel event is set before or during the batch, the
  method raises `RuntimeError("local_batch_cancelled")`. It
  never returns partial rows: the audit compares against
  complete solo runs, so partial rows would manufacture
  phantom divergences — the exact failure the stage's
  contract forbids. (The stage passes a fresh event, so this
  is unreachable inside the audit; it binds direct callers.)

## Failure vocabulary and precedence

Validation is `type()`-strict throughout, mirroring solo
(`type(x) is int` for scalars — `True` is not a valid seed
or cap; `type(prompts) is tuple`, per-prompt tuples checked
by the same `_ids_valid` solo uses; `isinstance(cancel,
Event)`). The precedence order is pinned (B's C3-with-
changes — solo's order, with the batch-specific slots
named):

1. prompts container type, then per-prompt ids →
   `ValueError("invalid_input_ids")`. Deliberate precedent
   choice: `generate_batch` is a *raising* method, so it
   uses the raising methods' code (`forward_logits`,
   `heal_input_ids`), not solo `generate`'s returned
   `invalid_or_empty_input`.
2. `max_new_tokens` not an int or outside 1..4096 →
   `ValueError("invalid_output_cap")`.
3. `seed` not an int or outside `[0, 2**64)` →
   `ValueError("invalid_seed")`.
4. `cancel` not a `threading.Event` →
   `ValueError("invalid_cancellation_event")`.
5. `cancel` already set → `RuntimeError(
   "local_batch_cancelled")` (mirrors solo, which reports
   cancellation before the context check).
6. Any prompt with `len(prompt) + max_new_tokens` over the
   backend's context limit →
   `ValueError("context_limit_exceeded")`.
7. `_ready()` — an uninitialized backend raises
   `RuntimeError("local_backend_not_initialized")`, as in
   every sibling method.
8. Empty prompt tuple → return `()`. The shortcut sits
   here, after all validation: an empty call returns `()`
   exactly when a valid one-prompt call would have
   proceeded, and scalar/cancel violations are never masked
   by emptiness. (`batch_stage` never calls it empty.)
9. Size bound (below) → `ValueError("batch_size_exceeded")`.
10. Lock polling, then the step loop. Runtime failures —
    the model forward raising, non-finite or mis-shaped
    logits, any other inference error — raise
    `RuntimeError("local_batch_failed") from None`; cancel
    observed at any check point raises
    `local_batch_cancelled` instead.

### The size bound (byte-derived refusal)

The v1 position-count bound is deleted. It refused
ordinary legal runs (a 100-fixture split with
paragraph-length prompts at default settings) while
permitting multi-GB single tensors, because positions do
not proxy bytes — the dominant transient, HF's
full-sequence logits, scales with the vocabulary. v2
refuses on an estimated peak, computed from the backend's
own quantities:

```
L       = longest_prompt + max_new_tokens   (final padded length)
peak    = 4 * rows * L * vocab_size          (full-sequence fp32 logits)
        + 2 * (4 * rows * heads * L^2)       (eager attention scores:
                                              matmul output and softmax
                                              output co-live, one layer)
        + 4 * rows * L^2                     (the (rows,1,L,L) fp32
                                              additive mask, held for
                                              the whole forward)
```

with `vocab_size` the backend's `_vocab_size` (the bound
`generate` validates against, not raw `config.vocab_size`)
and `heads` the loaded config's query-head count
(`num_attention_heads`; `n_head` on GPT-2 configs). If
`peak > _BATCH_FORWARD_BYTE_BUDGET` (2^33 bytes, 8 GiB),
the call raises `ValueError("batch_size_exceeded")`
before any forward runs.

The budget's role is stated plainly: it is a conservative
ceiling on one forward's *estimated* transient peak, set
just above the default caller scale (a 100-fixture split
of paragraph-length prompts on a GPT-2-vocabulary model
estimates ≈ 7.8 GB by this formula), so that an audit call
degrades to a labelled refusal instead of an OOM kill —
which, being a process kill rather than a Python
exception, would defeat the probe's never-fail-the-run
contract. It is not a memory guarantee: weights are already
resident, allocator behaviour varies, and no byte budget
consulted from Python can promise a machine will hold a
tensor. The recorded limitation, not hidden: at
large-vocab × long-context × full-split scale the audit
records `batch_size_exceeded` (surfaced distinctly — see
the stage amendments) and the operator's levers are
`--max-new-tokens` and the fixture count. Chunked execution
with recorded composition is the known future remedy if
that limitation bites in practice; it is explicitly
deferred, not silently absorbed — unrecorded chunking was
rejected in this cycle's red-team round because singleton
chunks manufacture invariance for exactly the longest
prompts while diluting the divergence rate.

## Stage amendments (`experiments.batch_stage`)

Forced by F1 (the stage fed the batch path every prompt,
including solo-ineligible ones, so one long fixture
collapsed the whole audit to `batch_failed` — the exact
per-prompt degradation the stage exists to avoid):

1. `generate_batch` is called with only the solo-evaluable
   prompts, in the stage's sorted order; the arity/contract
   check and the `zip` use the evaluable id list in the
   same edit. (Strengthening fact: the evaluable subset
   provably cannot trip the backend's caller-contract
   `ValueError`s — solo success implies the context and
   vocab checks passed, and the stage pre-validates seed
   and cap — so residual batch failure from the stage path
   is runtime failure or the size refusal, each with its
   own honest detail.)
2. When no prompt is evaluable the call is skipped: the
   stage returns an available payload with
   `n_evaluable == 0` and verdict `"unavailable"`, every
   prompt carrying its `solo_failed` entry. (Today that
   case surfaces as `batch_failed`; the change is owned
   here and pinned by a test.)
3. A `ValueError("batch_size_exceeded")` from the batch
   path maps to `{"available": False, "detail":
   "batch_size_exceeded"}` (new `batchinv` constant
   `BATCH_SIZE_EXCEEDED`), never plain `batch_failed`. The
   diff validator does not validate unavailable payloads'
   detail strings and tolerates this addition; no other
   detail vocabulary changes.
4. The stage docstring's "runs all prompts together" is
   amended: the batch path runs the solo-evaluable prompts
   together.

## Edges pinned by tests

- batch sizes {1, 2, 3, 5} vs solo, mixed prompt lengths,
  prompts containing id 0 and the pad value as real content;
  one 32-row case; permutation invariance
- forward counting/shapes (I3) via the model-forward patch,
  full `(rows, longest + step)` sequence
- cancel set before the call; cancel set mid-batch (a
  forward wrapper sets it during the first forward) — both
  raise `local_batch_cancelled`, no partial result; cancel
  set while waiting on a held lock (polling acquire)
- every `ValueError` code in the precedence list, including
  multi-violation cases that pin the order (bool seed/cap;
  bad ids + bad cap → ids win; pre-set cancel + over-context
  prompt → cancelled wins, mirroring solo)
- empty batch returns `()` on a ready backend; empty batch
  with a bad cap raises `invalid_output_cap`; empty batch
  on an unready backend raises
  `local_backend_not_initialized`
- the size bound just under and just over the line (small
  config: scaled-down budget via the module constant's
  seam — the test computes a prompt set whose estimate
  crosses 2^33 only if the formula is honest; a cheaper
  equivalent pins the estimate arithmetic against the
  formula on the tiny config and the raise at a patched
  budget)
- non-finite logits (hostile forward wrapper) →
  `local_batch_failed`; a raising forward →
  `local_batch_failed`; the flag-carrying conversion
  (§"Lock and cancel": cancellation detected inside the
  inference try surfaces as `local_batch_cancelled`, never
  eaten by the conversion to `local_batch_failed`)
- `batch_stage` end-to-end on the real backend: available
  payload, verdict `batch_invariant`; the CLI probe
  (`_run_batch_probe`) recording the available payload
  instead of `batch_unsupported`; a mixed stage case (one
  over-context prompt among good ones) degrading per-prompt
  with the audit still available (the F1 regression test);
  the all-solo-fail case returning the pinned
  available/`unavailable` payload; a size-refusing stub
  backend mapping to the `batch_size_exceeded` detail
- a deliberately batch-variant stub through the real stage
  wiring (already pinned at cycle 153; re-verified, not
  re-built)

## Non-goals

- No KV cache, no continuous batching, no padding-free
  packing, no batch-invariant kernels, no chunking in this
  cycle, no new dependency (stdlib + the existing
  torch/transformers only).
- No new module, CLI flag, diff section, or report section:
  the mypy source-file pin (55) and both diff oracles are
  untouched. The stage payload gains no fields. README/
  VALIDATION counts move only by the new test count,
  measured.
- No change to solo `generate`, `forward_logits`, or the
  diff code, except: the `batchinv.py` claim-1 text, the
  stage amendments above, and the doc/help wording that
  asserts the local backend is batch-less — `batchinv.py`
  claim 1, `docs/USAGE.md` §8.21's honesty note, and
  `README.md`'s Capabilities bullet ("the local backend
  exposes no batch path…"). The CLI flag help and probe
  docstrings are conditional ("when one exists") and stay
  true unchanged — checked, not assumed.
- Determinism scope, stated to prevent a wrong assumption:
  `verify --determinism` replays solo search observations
  only; the batch payload is covered by no determinism
  certificate, before or after this cycle.

## Monday-morning utility

An operator about to serve a checkpoint batched runs
`model-protocol-lab run --batch-invariance` and learns —
from the lab's own real batched execution, not a stub —
whether the winner it measured solo survives batching
token-for-token, before trusting that winner in production.
