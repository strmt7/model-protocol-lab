# Real-checkpoint validation — cycle 235 case studies (2026-11-05)

This page records the lab's first executed runs against real
(non-synthetic) checkpoints: what was acquired, what the shipped
adapter did with it, and what the shipped instruments measured.
Every number below traces to an executed run record from this
cycle (commands, seeds, digests and raw summaries are retained
with the cycle's work log). Nothing here is a benchmark, a
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
below) and for TinyLlama-1.1B-Chat (a two-fixture smoke run:
fp32 RSS ~4.56 GB; results and report written; backend
fingerprint `sha256:0ac30fca97222743e85ac1aaa…`). For
gpt2-medium and gpt2-large the battery rows that would carry
their load evidence did not complete this cycle (see
"Remainder") — their adapter status is doctor-verified,
generation-unproven, and is stated that way.

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

### Cost record

Full-prefix greedy decoding at float32 on this 4-core host
cost 1,365 s (chatml) and 2,808 s (llama3) for the 124M
checkpoint's rows, 831 s for TinyLlama's two-fixture smoke
(3 generations plus stages), and 139 s for the equivalent
GPT-2 smoke. These are the shipped adapter's real operating
costs on commodity CPU and are part of the evidence for any
deployment decision.

## Remainder — what this cycle did not establish

Stated plainly, so the rows above are never read as covering
more than they do:

- **TinyLlama battery rows (chatml, llama3): not completed.**
  The chatml row was attempted three times and killed three
  times by host reboots (~25, ~100 and ~102 minutes in; this
  host rebooted at ~18:00, ~19:41 and ~21:59 UTC on
  2026-11-05, and the row needs roughly 2.3 hours). No partial
  result is counted. The llama3 row was not attempted after
  the third kill.
- **gpt2-medium / gpt2-large battery rows: not run** (queued
  behind TinyLlama in the frozen order; the cycle closed on
  the split above).
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

A follow-up cycle on a stable host owns the remainder rows;
the settings, fixtures and boundary tables above are frozen
for it verbatim.

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
