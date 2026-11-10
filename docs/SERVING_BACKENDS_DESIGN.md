# Serving-stack backends: protocol spine and llama.cpp measurement

## Status and purpose

Shipped in cycle 243 as the `exec-backends` command and the
`serving_backends` report section.

A template author or release engineer uses this on Monday morning to
decide whether a surprising generation came from the checkpoint and
template under test, or from the serving stack that executed the
tokens. The instrument sends exact token IDs to a running llama.cpp
server, checks the server's identity and configuration before it
trusts a single generation, and records what the stack did in a form
that `diff-reports` can recompute later.

The combination measured here is deliberately narrow: exact token-ID
requests, a fail-closed identity probe, a local stop predictor,
GGUF-declared EOS attribution, a measured cache protocol, and
records that carry IDs and SHA-256 digests rather than answer text.
No claim is made that another tool cannot measure a serving stack;
the claim is only that this lab's selection evidence and its
serving-stack evidence now share the same discipline: denominators,
abstentions, and failure classes are recorded instead of smoothed
over.

## Scope

Cycle 243 ships:

- a protocol spine in `model_protocol_lab/servingbackends.py`, using
  only the Python standard library;
- one execution backend, llama.cpp over its native HTTP API, pinned
  to build b12182;
- standalone stop/EOS cells;
- a fidelity mode that replays recorded observations through the
  stack and compares them with the lab's existing stop predictor and
  divergence vocabulary;
- a four-position cache protocol: cold, warm repeat, shared prefix,
  and a divergent control;
- an opt-in `serving_backends` stage and report section. It is never
  part of candidate selection and is never run implicitly by `run`.

Not shipped in cycle 243:

- a vLLM adapter;
- a `serve` extra in `pyproject.toml`;
- a supported-version window for vLLM;
- a GGUF converter or any other checkpoint-supply tooling;
- any change to the Hugging Face generation backend.

The schema and vocabularies carry vLLM-shaped request fields so the
later adapter does not need a schema change. Against llama.cpp,
exercising an unsupported field records a `capability_absent` cell:
`stop_token_ids`, positive `min_tokens`,
`include_stop_str_in_output`, and `cache_salt` are data about the
stack's limits, never a reason to execute a degraded substitute.

## The protocol spine

`ExecutionBackend` is a separate protocol beside `core.Backend`. The
generation backend answers "generate from this checkpoint"; the
execution backend answers "execute these exact token IDs on that
already-running stack and prove which stack it was". The two are not
merged: a serving stack has an identity, a posture, and a cache
machine that a local generation backend does not.

A `StackRequest` is frozen at construction. It carries the full
prompt ID sequence, the derived request seed, the output cap, stop
texts, EOS policy, minimum-token and stop-token fields, the
stop-inclusion flag, a cache salt, and the cache position. Invalid
requests fail with the fixed code `stack_request_invalid`; nothing
is executed partially.

Capabilities are recorded in three places, always with the same
shape: the probe result, every execution record's stage, and the
report section. A capability is either present, absent, or not
attempted. There is no silent fallback anywhere in the design: if a
cell cannot be executed as specified, the cell exists and says why.

Verdicts are keyed per stack × weights digest × configuration
fingerprint. A verdict measured against one build, one checkpoint,
or one sampler configuration is never stated as a verdict about
another.

## The supported llama.cpp window

The llama.cpp window is a singleton: build **b12182**, commit
`fd27b2034905f8dadd8bd6d2cc44595f4929d391`. Identity binds three
things together:

1. the `build_info` string reported by `/props`, whose commit
   component must prefix-match the pinned commit (at least seven
   characters);
2. the server's resolved `model_path`, which must resolve to the
   operator-supplied GGUF file;
3. `total_slots`, which must be exactly 1.

The build counter inside `build_info` is recorded verbatim but does
not gate. A shallow clone reports `b1-fd27b20` for the same commit;
that is an acquisition artefact, not a different stack. A different
commit is `stack_version_unsupported`; an unparseable identity is
`stack_identity_unavailable`; a slot count other than 1 is
`stack_topology_mismatch`.

## Launch posture

The operator starts the server; the lab does not spawn one. The
ratified launch line is:

```sh
llama-server \
  --host 127.0.0.1 --port <port> \
  --model <checkpoint.gguf> \
  --parallel 1 \
  --api-key <operator key> \
  --no-webui \
  --slot-save-path <dedicated directory> \
  --cache-ram 0
```

Create the slot-save directory before starting the server: the
b12182 argument parser refuses a `--slot-save-path` that does not
already exist.

Why each pin is load-bearing:

- **Loopback bind and `--parallel 1`.** The protocol addresses slot
  0 and treats it as the whole machine. A second slot would make
  cache and erase observations ambiguous.
- **`--api-key`.** At b12182 the key check runs before routing: an
  unauthenticated `GET /props`, `GET /slots`, and slot erase all
  answer 401, and the probe proves this on the actual server before
  measuring (condition C-1). An unguarded server is refused, never
  measured.
- **`--slot-save-path`.** Slot erase does not exist without it at
  b12182. The directory is dedicated to the operator's server. The
  lab never invokes slot save or restore; erase performs no file
  operation; the directory stayed empty through every executed
  probe and cache sequence in the cycle evidence. Any files present
  there are operator residue, and the posture makes slot-state
  persistence to that directory possible for key-holders — the lab
  creates no such files (condition C-2).
- **`--cache-ram 0`.** With a RAM prompt cache, erasing a slot
  clears the slot layer only; the RAM cache survives. Under the
  default cache size there is one narrow pattern — this lab's own
  pinned requests — where a restore after erase happens not to
  fire, but only because the restore gate computes
  `f_keep = 0/0 = NaN` and the comparison closes by accident. An
  unpinned request restores 14 tokens after the same erase. A
  guarantee that holds by a NaN is not a guarantee: `--cache-ram 0`
  means the prompt-cache object is never constructed, and it is the
  only unconditional cold guarantee. This fragility, verified
  against the b12182 binary, is why the pin is in the launch line.
- **`--no-webui`.** This is the spelling the ratified posture and
  the executed spike used. At b12182 it is the deprecated alias of
  the canonical `--no-ui`; both are accepted. The window is a
  singleton build, so the alias costs nothing here, and the probe
  records the reported UI state either way.

Fallback, stated once: if the operator's server has no
`--slot-save-path`, erase is unavailable, the cache cells record
`capability_absent`, and the stop cells proceed. The lab does not
restart the server, does not substitute a different freshness
mechanism, and does not infer cold state from anything else.

## What /props does and does not say

The design round expected the cache machine to be fingerprintable
from `/props` fields naming the RAM cache size, idle-slot state,
context-checkpoint count, and warmup state. The b12182 `/props`
payload contains none of those fields. It reports `build_info`,
`total_slots`, `model_path`, the endpoint flags, the UI state, the
default generation settings, and the BOS/EOS strings.

The fingerprint therefore binds the probed equivalents, and says so:

- posture, proven by acting on it: `api_key_required` (the 401
  probes), `slot_erase_available` (a probe erase that also makes
  the cache protocol's cold arm genuinely post-erase),
  `endpoint_slots`, and the reported UI state;
- a measured cache probe: the cold and warm `timings.cache_n`
  values of the probe's own sentinel pair;
- the Tier-A sampler configuration the probe pinned and had echoed
  back;
- a digest of the default generation settings;
- the GGUF-declared EOS ID set;
- the checkpoint SHA-256 and the tokenizer fingerprint;
- the protocol version string, `exec-backends-protocol-1`.

The stored identity carries the model's path as a digest, not the
raw path. `config_fingerprint` is the SHA-256 of the canonical JSON
of that identity block; `diff-reports` recomputes it from the stored
block, so a record cannot be quietly re-pointed at another stack.

## The probe: five gates, fail closed

No cell executes until the probe passes, in this order:

- **A3, authentication.** Unauthenticated `/props` must answer 401
  and authenticated `/props` must answer 200.
- **G1, identity.** The window, path, and topology checks above.
- **Probe erase.** Establishes `slot_erase_available` as data and
  clears slot 0 before the cache gates.
- **G2, configuration echo.** The probe sends the sentinel prompt
  with the full Tier-A tuple pinned. The completion's
  `generation_settings` echo must match all nine fields exactly.
  The same check runs on every later execution's response; a drift
  mid-run is a `config_echo_mismatch` cell, not a footnote.
- **G3, prompt echo, text form.** The completion's `prompt` field
  must equal, as text, both the local detokenization of the sent
  IDs and the server's own `/detokenize` of the same IDs. There is
  deliberately no IDs-form prompt assertion: at b12182 the field is
  text, and treating it as IDs would invent a contract the stack
  does not offer.
- **G4, vocabulary round trip.** Detokenizing the sentinel IDs plus
  the declared end-of-turn ID, then re-tokenizing through the
  server and locally, must return the same IDs.
- **G5, cache quantity.** A cold/warm sentinel pair with
  `cache_prompt` on. The quantity is `timings.cache_n` — nothing
  else. `tokens_cached` and `tokens_evaluated` are recorded as
  reported, as context, and no gate, verdict, or mirror expectation
  may use them. Reporting counts as present only when `cache_n`
  discriminates (warm > cold); a flat or absent `cache_n` records
  `cached_token_reporting: false`, which is capability data, not a
  probe failure.

The Tier-A tuple (spike amendment R-1): sampler order
`top_k, top_p, min_p, temperature`; temperature 0.0; top_k 0;
top_p 1.0; min_p 0.0; **repeat_penalty 1.0**; dynatemp_range 0.0;
n_probs 0; and the request seed. The neutral repeat penalty at
b12182 is 1.0, established by three executed reads during the
spike. Any document or test that calls 1.1 the server default is
wrong.

The sentinel is the text `The capital of France is`. On the spike
checkpoint it tokenizes to `[464, 3139, 286, 4881, 318]`, with EOS
50256 declared in the GGUF (amendment A11 held on the real file).

## Executions and records

Each execution posts the prompt as a pure list of token IDs to
`/completion` on slot 0, non-streaming, with the Tier-A tuple, the
stop list, the EOS policy, and `cache_prompt` set exactly when the
request's cache position is not standalone. The response is
classified under a fixed finish vocabulary — `stop`, `length`,
`abort`, `error`, `repetition`, `eos`, `word`, `none`,
`unclassified` — driven by llama.cpp's `stop_type` under the same
allowlist discipline as the design's finish classes. A `stop_type`
outside `{eos, word, limit, none}` is `stack_response_unclassified`
with the token IDs kept; it is never exported raw.

Caps (design §8.5): prompt IDs reuse the differential stage's
131,072 cap; `max_new_tokens` is 1..4096; a run executes at most
512 cells; the records artifact is capped at 16 MiB. Returned
length may exceed the cap by at most one token (the
partial-multibyte overshoot llama.cpp can produce); beyond that the
cell is `stack_response_invalid`. Over-budget work degrades by
labelled omission — a budget record with the fixed detail
`serving_backends_record_budget_exceeded` — never by exception and
never by silent truncation.

A record carries: the full prompt IDs; the request configuration
with stop texts represented by their digests; the returned IDs; the
SHA-256 digests of the returned text in its raw and
special-token-free forms (the second is a vLLM-shaped field, always
null on llama.cpp); the stop winner as an index into the stop list
or an EOS ID; the finish class; the cache quantities exactly as
reported (`cache_n`, `tokens_cached`, `tokens_evaluated`; the
vLLM-shaped cache-creation field is null on llama.cpp); the mirror
block described below; and a `cell_detail` from the fixed
vocabulary. Raw stop text and raw answer text appear nowhere in a
record.

## The stop predictor and fidelity

The predictor is the lab's existing linear scan, shared with
`diff-backends`: decode each prefix, find the first step at which
each stop text completes and the first position of the EOS ID, and
let the earliest win. A stop and EOS completing on the same step is
a `precedence_tie`: the stop wins the prediction and the tie is
flagged, so a stack resolving the tie to the stop agrees, and a
stack resolving it the other way is a precedence divergence, not a
step divergence.

Per cell, the comparison yields exactly one verdict:

- `agree`;
- `diverge`, with a class from the existing vocabulary
  (`MISSING_STOP_AT_CAP`, `SPURIOUS_STOP`, `FIRING_STEP_DIFFERS`,
  `EOS_OVER_STR_PRECEDENCE`, `STR_OVER_EOS_PRECEDENCE`,
  `KEPT_VS_STRIPPED`);
- `unobservable`, when the stack's answer cannot be classified or
  its disposition cannot be recomputed from the stored digests;
- `capability_absent`;
- `predictor_detokenization_mismatch`, when the stack's own
  `/detokenize` disagrees with the predictor's detokenization of
  the same IDs. That is an instrument-side finding (adjudication
  A7): it is aggregated in its own row and is never counted as a
  vendor divergence.

EOS attribution (A11) binds the GGUF-declared IDs: a stream the
stack calls an EOS stop, ending on an ID the checkpoint never
declared, fails attribution for that cell
(`eos_attribution_mismatch`) whatever the predictor said.

Dispositions are computed from digests, never from held text: the
stack's returned-text digest is compared against the predictor's
full-text digest (`kept`) and stripped-text digest (`stripped`).
For llama.cpp the predicted disposition of a stop winner is
`stripped`.

Fidelity mode (`diffbackends.fidelity_section_data`) replays each
recorded observation twice — a strip arm and a keep arm — and
conditions every verdict on the token stream: when the stack's
stream differs from the recorded stream, the row carries
`GENERATION_STREAM_DIVERGED` with the first-divergence index as
data, beside the stop verdict, not instead of it. Against
llama.cpp the keep arm is `capability_absent` (the stack has no
stop-inclusion switch), so a llama.cpp fidelity section aggregates
to `fidelity_unavailable` whenever a keep arm was attempted; the
strip arm's verdict stands on its own row. The section aggregate
is `fidelity_diverge` if any cell diverges, else
`fidelity_unavailable` if any cell is unobservable, absent, or a
detokenization mismatch, else `fidelity_agree`.

## The cache protocol

A cache sequence runs four arms against slot 0, in order: cold,
warm repeat (the cold prompt again), shared prefix (the cold prompt
plus a suffix), and a divergent control (a prompt sharing nothing
load-bearing with the cold prompt). Identity is re-read before the
first arm and after the last (A8(i)): a change at the start voids
the sequence with `stack_identity_changed_during_run` and no
records; a change at the end keeps the records and labels the
sequence. Slot 0 is erased before the arms and again after them.

"Cold" is defined operationally (condition C-3): the first arm of a
sequence, run after a completed erase, on the probed server, under
the launch line above. The divergent control is the tripwire: cold
or divergent reuse above one quantum — one token, the
block/slot quantum at b12182 — marks the whole sequence
`cache_protocol_contaminated`, whatever the erase call returned.
There is no freshness inference and no substitute mechanism: the
spike's cold-arm-first alternative was considered and narrowed out,
because freshness the lab cannot verify is not a measurement.

## Persistence

Records persist incrementally (A8(iv)). While a run executes, the
artifact is JSON Lines: a header carrying the identity,
capabilities, and the full plan; one flushed and fsynced line per
record as its execution completes; a footer when the run finishes.
A run that dies leaves a valid prefix, and the reader reconstructs
each planned-but-absent cell as a `cell_not_executed` row (or a
budget row when the byte budget, not the process, ended the run).
On completion the command replaces the artifact, atomically, with
the finalized section document. Readers accept both forms; a JSON
Lines artifact is by definition an interrupted run.

## The report section and its mirror

`experiments.serving_backends_analysis` derives three companions
from the stage: a stop matrix (one row per record, in the
`stop_matrix` row shape), the measured cache block, and the
fidelity aggregate. Companions that were not attempted carry
`available: false` and the detail `not_attempted`; the section's
keys are always all present.

`diff-reports` mirrors the section with a fresh validator, written
for execution observations — not adapted from the generation
mirror. From the stored data alone it recomputes: every digest,
the configuration fingerprint, each cell's disposition from its
digest triple, EOS attribution, the overshoot rule, the stop
matrix rows, the cache positions and their contamination
coherence, and the fidelity counts and aggregate verdict. The
predictor itself is not re-run in the mirror (there is no tokenizer
there); per-fixture fidelity rows and the audit block are
shape-checked, as the house mirrors treat stage-attested blocks.
Every stored leaf is type-gated before any membership test, lookup,
subscript, or hash, and every grammar is a full match — the
post-cycle-241 discipline, applied to this section from its first
commit. The differ reports identity drift (stack, version,
checkpoint, fingerprint, and per-key configuration differences) as
"not comparable": two sections from different stacks are a
finding, not a diff.

The HTML report renders the section under its own name,
"Serving-stack measurement (exec-backends)". Serving-stack outputs
never wear the `diff-backends` name, and `diff-backends` outputs
never wear this one.

## The command

```sh
model-protocol-lab exec-backends \
  --base-url http://127.0.0.1:8080 \
  --api-key <operator key> \
  --gguf path/to/checkpoint.gguf \
  --model path/to/checkpoint \
  --out serving-records.json
```

`--model` supplies the tokenizer only; weights are never loaded.
The endpoint must be plain HTTP on a loopback host with no
userinfo, query, or fragment — anything else is
`stack_endpoint_refused` before any request is made. The API key
is required: the ratified posture has no unguarded form.

Exit codes, stated identically in the command's help and in
`docs/USAGE.md`: **0** when a report was produced (including a
stage that failed closed at a probe gate — the artifact is the
output, and it records the gate's fixed detail); **2** for invalid
input (endpoint refusal, unparsable arguments, an unusable results
file); **1** when no report could be produced (the tokenizer or
checkpoint could not be loaded, the artifact could not be written,
or the run was interrupted before finalization).

## Determinism, stated as a rate

The spike executed the pinned request five times at b12182 and got
five identical streams (5/5), and replayed four recorded streams
with four identical outcomes (4/4). Those are rates with their
denominators, measured on one checkpoint on one host. They are
regression references for this build, not a property claimed of
llama.cpp, of other checkpoints, or of other hosts.

## The vLLM arm, recorded but not built

The request schema, capability shape, and record fields already
carry everything the vLLM adapter needs; no vLLM code ships in
cycle 243, no `serve` extra exists, and the §8.6 vLLM window is
unset. Three facts are recorded now so the later cycle does not
have to re-derive them:

1. The distribution route is refuted for CPU-only hosts: the
   vLLM 0.33.0 PyPI wheel ships only the CUDA extension, and no
   official CPU wheel exists. A supported vLLM host is therefore a
   GPU host, or a CPU host running a from-source
   `VLLM_TARGET_DEVICE=cpu` build. Any future install instruction
   that implies `pip install` suffices on a CPU-only host is
   wrong.
2. The adapter will pin, before importing vLLM, the five-variable
   environment block: `VLLM_NO_USAGE_STATS=1`,
   `VLLM_DO_NOT_TRACK=1`, `HF_HUB_OFFLINE=1`,
   `HF_HUB_DISABLE_TELEMETRY=1`, and `VLLM_TARGET_DEVICE=cpu`
   (the fifth scoped to CPU hosts; the first four are the egress
   controls, read back at the vLLM probe's G2 equivalent and
   fingerprinted).
3. The gpt2 GGUF used as spike evidence was produced by an
   operator-side conversion that strips twelve `*.attn.bias`
   buffers the b12182 converter forwards incorrectly. That strip
   step belongs to the evidence pipeline. The lab ships no
   converter (the §9.5 non-goal stands), and no shipped code path
   depends on the stripped file.

## Falsification

This design's claims are falsified, specifically, by: a b12182
server under the launch line whose unauthenticated slot endpoints
answer anything but 401; a post-erase cold arm under
`--cache-ram 0` reporting `cache_n` above one quantum without the
sequence being marked contaminated; a record, anywhere in the
artifact, containing a raw stop text or answer text; a
`diff-reports` mirror accepting a section whose stored fingerprint
does not recompute from its stored identity block; or a capability
the stack lacks being executed in a degraded form instead of
recorded absent. None has been observed; the cycle-243 test suite
pins each one against a hostile fake, and the probe and cache
claims were executed against the real b12182 binary before the
code shipped.
