# Real-checkpoint validation — cycle 235 case studies (2026-11-05), cycle 240 remainder (2026-11-08)

This page records the lab's first executed runs against real
(non-synthetic) checkpoints: what was acquired, what the shipped
adapter did with it, and what the shipped instruments measured.
Every number below traces to an executed run record from these
cycles (commands, seeds, digests and raw summaries are retained
with the cycles' work logs). Cycle 240 (2026-11-06 →
2026-11-08) executed the remainder rows cycle 235 left: five
of its six rows completed and are recorded below; the sixth is
named in the remainder section with the evidence of why it did
not. Nothing here is a benchmark, a
model-quality ranking, or a comparison against other tools: the
fixtures are strict exact-answer probes, and the measurements
describe format/boundary behaviour of fixed weights under the
lab's own protocol.

## Environment and constraints

- Host: 4-core AMD EPYC 9D25, 7.7 GiB RAM, no GPU. All runs on
  CPU, in the project's `.venv` (torch 2.15.0+cpu,
  transformers 5.22.1), through the shipped `LocalHFBackend`:
  local files only, safetensors only, float32, greedy decoding,
  no KV cache (full-prefix recompute), exactly-N generation.
- Checkpoints were acquired out-of-band (the lab never downloads
  by design) into a directory outside this repository; weights
  are never committed. Revision pins and SHA-256 digests below
  make each row re-derivable by anyone who acquires the same
  checkpoint revision.

## Acquisition record

| Checkpoint (HF repo) | Revision | Weights (bytes) | Weights SHA-256 | Architecture |
|---|---|---|---|---|
| `openai-community/gpt2` | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | 548,105,171 | `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707` | gpt2, 12 layers, n_embd 768, window 1024 |
| `openai-community/gpt2-medium` | `6dcaa7a952f72f9298047fd5137cd6e4f05f41da` | 1,519,984,962 | `fc5a354a19255ad494f3d71549390baca1ccf61d1d822b9408971705c687c9cd` | gpt2, 24 layers, n_embd 1024, window 1024 |
| `openai-community/gpt2-large` | `32b71b12589c2f8d625668d2335a01cac3249519` | 3,247,159,078 | `5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce` | gpt2, 36 layers, n_embd 1280, window 1024 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` | 2,200,119,864 | `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933` | llama, window 2048, vocab 32000, BOS 1 / EOS 2 |

Gated rows (owner licence decision recorded 2026-11-05 — the
licences will be accepted; the rows are in scope but were not
executed this cycle):

- `meta-llama/Llama-3.2-1B-Instruct` — inside the adapter's
  reach, but gated download needs HF authentication as the
  owner and no HF token exists on this machine yet (checked:
  no `HF_TOKEN` in the environment, no hub token file). Parked
  on token delivery through the owner's credential flow.
- `meta-llama/Llama-3.2-3B-Instruct` — additionally, its float32
  weights alone (~12.4 GB) exceed this host's RAM, so the
  primary-dtype battery cannot run for this row here regardless
  of licensing.
- Gemma rows — the owner will accept Google's licence, but the
  shipped adapter accepts only `gpt2`/`llama` configurations;
  Gemma architectures are outside its reach at any licence
  state. Adapter breadth is a separate program item; no Gemma
  weights were downloaded.

GPT-2 XL (1.5B) was not acquired: ~6.0 GB of float32 weights
leaves no safe headroom on a 7.7 GiB host.

## Adapter verdicts

`doctor` returned verdict `ok` for all four acquired
checkpoints (asset layout, weight headers, tokenizer and the
configured boundary verified). Load-and-generate through the
adapter is proven by executed runs for `gpt2` (full battery
below, cycle 235), for `gpt2-medium` and `gpt2-large` (full
battery, cycle 240 below), and for TinyLlama-1.1B-Chat (a
two-fixture smoke run in cycle 235: fp32 RSS ~4.56 GB; results
and report written; backend fingerprint
`sha256:0ac30fca97222743e85ac1aaa…`; and its full chatml
battery row in cycle 240 below). Cycle 235's version of this
section stated gpt2-medium and gpt2-large as doctor-verified,
generation-unproven; the cycle-240 rows below supersede that
statement with executed evidence.

## Native-template results (doctor instruments)

- The three GPT-2 checkpoints ship no chat template:
  `template_conformance` reports `no_template_declared`, the
  firewall `no_template`, and the behavioural template probe
  `no_template`. Marker isolation: 1 special token, isolation
  rate 1.0 (48 probes).
- TinyLlama-1.1B-Chat ships a template (SHA-256
  `66291cf0045c2425a3a667cf3cbb7af2b11f09e025c02f97245323ab79119362`).
  It is a Zephyr-style template — `<|user|>`, `<|system|>`,
  `<|assistant|>` role markers, turns terminated by the EOS
  token — and matches no family in the lab's catalog, so the
  `run` families below are hand-rendered hypotheses for this
  checkpoint too; its native surface is what the doctor
  instruments measure. The template firewall reports
  `allowed`; conformance `ok`; the behavioural template probe
  rendered its 8 frozen word-swap pairs through the
  checkpoint's own template (10 renders) and reported **no
  findings** — no content-conditional template logic detected
  on the probe set. The probe's bound applies: a no-findings
  result covers the frozen swaps and bounds only
  non-probe-aware templates (see `docs/USAGE.md` section 2).
  Marker isolation on this tokenizer: 3 special tokens,
  isolation rate 0.875.

## The battery (frozen settings)

Per checkpoint: `run --family chatml` and `run --family
llama3`, each with `--mutate --stop-cause`, 64 new tokens,
seed 0, on a dedicated fixture set authored for this cycle
(12 search + 6 held-out exact-answer items; SHA-256
`043c1cd7300bd1f8f3ed8ab32af96b0838da0be97320fc7df4729b6462e464f6`;
the full set is reproduced in the appendix). Each candidate
was measured with its own format's genuine closing marker as
the boundary, tokenized by the checkpoint's tokenizer and
verified with `doctor` (GPT-2: chatml boundary
`[27, 91, 320, 62, 437, 91, 29, 198]`, llama3
`[27, 91, 68, 313, 62, 312, 91, 29]`; TinyLlama: chatml
`[529, 29989, 326, 29918, 355, 29989, 29958, 13]`, llama3
`[529, 29989, 29872, 327, 29918, 333, 29989, 29958]`).
Selection statements come from the search split only; the
held-out split is read only for a frozen winner.

### GPT-2 (124M) — both rows completed

| Family | Candidates | Generations | Exact-and-boundary passes | Decision | Wall time |
|---|---|---|---|---|---|
| chatml | 3 | 36 (3 × 12 search) | 0/12 per candidate | `no_candidate` | 1,365 s |
| llama3 | 5 | 60 (5 × 12 search) | 0/12 per candidate | `no_candidate` | 2,808 s |

Run records: chatml backend fingerprint
`sha256:b12161f729aa6ee0f2b56dd53…`, configuration digest
`sha256:7660973f5cf7f53d117a481c3…`; llama3 configuration
digest `sha256:6fed5f3fcd682a7170d07b8d3…` (same backend
fingerprint — same checkpoint and environment).

What the instruments recorded, per family:

- **Failure taxonomy:** every one of the 96 generations is
  classified `no-boundary-in-N` — the configured boundary
  token sequence occurred in **0 of 96** generations. All
  other modes (`boundary-text-mismatch`,
  `eos-without-boundary`, `backend-error`) are 0.
- **Stop-cause:** chatml 36/36 and llama3 59/60 generations
  are classified `stop_string` (one llama3 generation is
  `max_length`); the would-stop-early rate is 1.0 and 0.9833.
  In the stop-cause stage's terms, a string-level stop on the
  marker text would have terminated 95 of the 96 generations.
- **Boundary audit:** 36 observations (chatml), overall rate
  0.0556; 60 observations (llama3), overall rate 0.0.
- **Held-out:** not evaluated in either row. With no unique
  all-pass search candidate the selection rule abstains, and
  the held-out split is never read to break an abstention.

Reading, scoped to the recorded fields: on this template-less
base checkpoint, neither headline format hypothesis produced
a single completion that is simultaneously the exact expected
answer and terminated by that format's own boundary token
sequence. The recorded failing component in every case is the
token-ID boundary, while the marker's *text* evidently
appears in the generations (the string-stop classification) —
the exact tokenization mismatch `docs/USAGE.md` section 4
warns a boundary audit exists to catch. The run records are
digest-only by design, so this page makes no claim about how
often the bare answer text alone was right; the conjunction
is what was measured, and its failing component is the one
the taxonomy records.

### GPT-2 medium (355M) — both rows completed (cycle 240)

| Family | Candidates | Generations | Exact-and-boundary passes | Decision | Wall time |
|---|---|---|---|---|---|
| chatml | 3 | 36 (3 × 12 search) | 0/12 per candidate | `no_candidate` | 4,003 s |
| llama3 | 5 | 60 (5 × 12 search) | 0/12 per candidate | `no_candidate` | 8,015 s |

Run records: backend fingerprint
`sha256:259783bb79bce39cbbe35ed5…` (same checkpoint and
environment for both rows); configuration digests chatml
`sha256:caeb357cf36c483eb83a5a8f…`, llama3
`sha256:8b23753288567aae34aa7747…`.

What the instruments recorded, per family:

- **Failure taxonomy:** chatml classifies 35 of 36
  generations `no-boundary-in-N` and one
  `boundary-text-mismatch` — the decoded output ends with
  the decoded boundary text but no token-ID boundary
  occurred (the tokenization-mismatch false-negative mode);
  it is the only generation in either cycle's GPT-2 rows
  classified into that mode. llama3 classifies all 60
  generations
  `no-boundary-in-N`. All other modes are 0 in both rows.
- **Stop-cause:** chatml 22/36 `stop_string` and 14/36
  `max_length`; would-stop-early rates 0.8333, 0.1667,
  0.8333 per candidate. llama3 51/60 `stop_string` and 9/60
  `max_length`; would-stop-early rates 0.8333, 0.8333, 0.75,
  0.8333, 1.0.
- **Boundary audit:** 36 observations (chatml), overall rate
  0.0; 60 observations (llama3), overall rate 0.0333.
- **Held-out:** not evaluated in either row — the selection
  rule abstains on the search split, exactly as in cycle 235.

### GPT-2 large (774M) — both rows completed (cycle 240)

| Family | Candidates | Generations | Exact-and-boundary passes | Decision | Wall time |
|---|---|---|---|---|---|
| chatml | 3 | 36 (3 × 12 search) | 0/12 per candidate | `no_candidate` | 7,892 s |
| llama3 | 5 | 60 (5 × 12 search) | 0/12 per candidate | `no_candidate` | 21,532 s |

Run records: backend fingerprint
`sha256:25d57c88d350883a99bcfdc8…`; configuration digests
chatml `sha256:a6de25f2eff0a404ef21a31b…`, llama3
`sha256:f6ce9cde1d9a52c4c97148fc…`.

What the instruments recorded, per family:

- **Failure taxonomy:** all 96 generations are classified
  `no-boundary-in-N`; all other modes are 0.
- **Stop-cause:** chatml 33/36 `stop_string` and 3/36
  `max_length`; would-stop-early rate 0.9167 per candidate.
  llama3 55/60 `stop_string` and 5/60 `max_length`;
  would-stop-early rates 0.9167, 0.8333, 0.9167, 0.9167, 1.0.
- **Boundary audit:** 36 observations (chatml), overall rate
  0.0556; 60 observations (llama3), overall rate 0.0.
- **Held-out:** not evaluated in either row.

The llama3 row's 21,532 s (5 h 59 min) is the longest single
run the lab has executed; it completed on the fourth attempt
after three host reboots killed the earlier ones mid-flight
(no partial result from a killed attempt is counted anywhere
on this page — that rule is the same one cycle 235 applied).

### TinyLlama-1.1B-Chat — chatml row completed (cycle 240); llama3 row not completed

| Family | Candidates | Generations | Exact-and-boundary passes | Decision | Wall time |
|---|---|---|---|---|---|
| chatml | 3 | 36 (3 × 12 search) | 0/12 per candidate | `no_candidate` | 8,736 s |
| llama3 | — | not completed (see "Remainder") | — | — | — |

Run record (chatml): backend fingerprint
`sha256:0ac30fca97222743e85ac1aa…` — identical to the
cycle-235 smoke run's fingerprint, as expected for the same
checkpoint and environment; configuration digest
`sha256:2f72b87e952c2f06b5f9c9fa…`.

What the instruments recorded (chatml):

- **Failure taxonomy:** all 36 generations are classified
  `no-boundary-in-N`; all other modes are 0.
- **Stop-cause:** the profile differs sharply from every
  GPT-2 row in either cycle: only 12/36 generations are
  `stop_string` while 24/36 are `max_length` — in two-thirds
  of generations the marker text never appeared within the
  64-token budget at all. The would-stop-early rate is
  0.3333 per candidate.
- **Boundary audit:** 36 observations, overall rate 0.4722 —
  the highest recorded on any row. The taxonomy still
  classifies every generation `no-boundary-in-N`, so no
  generation satisfied the exact-boundary conjunction the
  battery measures; the two instruments' readings are both
  recorded here as measured, not reconciled into a claim
  neither supports alone.
- **Held-out:** not evaluated.

Reading across both cycles, scoped to the recorded fields:
the completed matrix is seven rows, 324 generations over
four checkpoints, and every candidate in every row scored
0/12 exact-and-boundary passes, so all seven decisions are
`no_candidate` and the held-out split was never read. The
failing component the taxonomy records is uniformly the
token-ID boundary; the variation between checkpoints is in
how the generations end — GPT-2 rows overwhelmingly produce
the marker text (a string-level stop would have terminated
most of them), while the TinyLlama chatml row mostly never
reaches the marker text inside the token budget. As in cycle
235, the records are digest-only: this page claims nothing
about how often the bare answer text alone was right.

### Cost record

Full-prefix greedy decoding at float32 on this 4-core host
cost 1,365 s (chatml) and 2,808 s (llama3) for the 124M
checkpoint's rows, 831 s for TinyLlama's two-fixture smoke
(3 generations plus stages), and 139 s for the equivalent
GPT-2 smoke. Cycle 240 adds 4,003 s and 8,015 s for the
gpt2-medium rows, 7,892 s and 21,532 s for the gpt2-large
rows, and 8,736 s for the TinyLlama chatml row — plus the
killed attempts the reboot regime consumed across the
cycle (more than 12 hours on the TinyLlama llama3 row's
six attempts alone, and further killed attempts on four
other rows), which produced no records and are counted
nowhere above. These are the shipped adapter's
real operating costs on commodity CPU and are part of the
evidence for any deployment decision.

## Remainder — what these cycles did not establish

Stated plainly, so the rows above are never read as covering
more than they do:

- **TinyLlama llama3 battery row: not completed.** Cycle
  240 attempted it six times (2026-11-07 → 2026-11-08) and
  a host reboot killed every attempt mid-flight — about
  3 h 38 min, 1 h 45 min, 1 h 37 min, 1 h 30 min, 1 h 29 min
  and 2 h 14 min into the run, respectively. At the chatml
  row's measured per-generation cost the row needs roughly
  4 hours of uninterrupted host time; the two longest
  windows the cycle received (11 h 07 min and 8 h 08 min)
  were consumed by other rows, and no attempt of this row
  ever received one. No partial result is counted. The
  block is the host's reboot regime, not a measured defect:
  before the row's first attempt, `doctor` verified the
  llama3 boundary token set against TinyLlama's tokenizer
  (verdict `ok`, "8 ids within vocabulary"), and the sibling
  chatml row completed on the same checkpoint in the same
  cycle. Cycle 235's version of this section also listed
  the TinyLlama chatml row and both gpt2-medium/gpt2-large
  row pairs as not established; those five rows are
  established by the cycle-240 records above, and this row
  alone carries forward.
- **Gated rows: not run** (licence decided by the owner; HF
  token not yet present on this machine; the 3B row is
  additionally RAM-blocked at float32, and Gemma is outside
  the adapter's reach).
- **Second-ring families** (gemma, mistral, qwen3, phi4 as
  rendered hypotheses): excluded before any battery outcome
  was read, on calibration arithmetic — the core pair alone
  is a multi-hour CPU program per checkpoint on this host.
- CPU/GPU parity and VRAM fit remain unmeasured (no GPU on
  this host).

A follow-up cycle on a stable host owns the one remaining
battery row; the settings, fixtures and boundary tables
above are frozen for it verbatim.

## Appendix — the R2 fixture set

The exact dataset of record (search split first in the file
as executed; SHA-256 above). One-turn user messages;
`expected` is the exact answer string.

```json
{
  "search": [
    {"id": "search-copy-blue", "group": "copy-colors", "messages": [{"role": "user", "content": "Reply with exactly BLUE and nothing else."}], "expected": "BLUE"},
    {"id": "search-copy-maple", "group": "copy-words", "messages": [{"role": "user", "content": "Repeat this word exactly: MAPLE"}], "expected": "MAPLE"},
    {"id": "search-number-42", "group": "numbers", "messages": [{"role": "user", "content": "Return only the numeral 42."}], "expected": "42"},
    {"id": "search-capital-france", "group": "capitals", "messages": [{"role": "user", "content": "What is the capital of France? Answer with only the city name."}], "expected": "Paris"},
    {"id": "search-sum-9", "group": "arithmetic", "messages": [{"role": "user", "content": "Compute 4 + 5. Answer with only the number."}], "expected": "9"},
    {"id": "search-copy-otter", "group": "copy-words", "messages": [{"role": "user", "content": "Repeat this word exactly: OTTER"}], "expected": "OTTER"},
    {"id": "search-weekday-friday", "group": "weekdays", "messages": [{"role": "user", "content": "What day comes immediately after Thursday? Answer with only the day name."}], "expected": "Friday"},
    {"id": "search-copy-green", "group": "copy-colors", "messages": [{"role": "user", "content": "Reply with exactly GREEN and nothing else."}], "expected": "GREEN"},
    {"id": "search-yes", "group": "yes-no", "messages": [{"role": "user", "content": "Is the sky blue on a clear day? Answer with only YES or NO."}], "expected": "YES"},
    {"id": "search-capital-japan", "group": "capitals", "messages": [{"role": "user", "content": "What is the capital of Japan? Answer with only the city name."}], "expected": "Tokyo"},
    {"id": "search-product-12", "group": "arithmetic", "messages": [{"role": "user", "content": "Compute 3 * 4. Answer with only the number."}], "expected": "12"},
    {"id": "search-copy-falcon", "group": "copy-words", "messages": [{"role": "user", "content": "Repeat this word exactly: FALCON"}], "expected": "FALCON"}
  ],
  "held_out": [
    {"id": "held-copy-amber", "group": "held-colors", "messages": [{"role": "user", "content": "Reply with exactly AMBER and nothing else."}], "expected": "AMBER"},
    {"id": "held-number-13", "group": "held-numbers", "messages": [{"role": "user", "content": "Return only the numeral 13."}], "expected": "13"},
    {"id": "held-capital-italy", "group": "held-capitals", "messages": [{"role": "user", "content": "What is the capital of Italy? Answer with only the city name."}], "expected": "Rome"},
    {"id": "held-sum-11", "group": "held-arithmetic", "messages": [{"role": "user", "content": "Compute 5 + 6. Answer with only the number."}], "expected": "11"},
    {"id": "held-copy-bison", "group": "held-words", "messages": [{"role": "user", "content": "Repeat this word exactly: BISON"}], "expected": "BISON"},
    {"id": "held-weekday-tuesday", "group": "held-weekdays", "messages": [{"role": "user", "content": "What day comes immediately after Monday? Answer with only the day name."}], "expected": "Tuesday"}
  ]
}
```
