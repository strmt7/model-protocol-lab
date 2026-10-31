# Architecture — how the lab fits together

This page is the orientation map for a reader who arrives cold: the
components, the path a measurement takes through them, and where the
authoritative record for each part lives. It states outcomes and
points; the reasoning behind each design decision is in the
[architecture decision index](adr-index.md), and the security record
is [SECURITY.md](../SECURITY.md). If this page and a linked record
ever disagree, the record wins.

## The question the system is built around

A local language model gives a bad answer. Two explanations compete:
the weights cannot do the task, or the conversation was formatted in a
way the model mishandles. Tools that always apply the checkpoint's own
chat template cannot separate those explanations. This lab can,
because it controls the exact token sequences the checkpoint is
measured on: it renders *format hypotheses* itself, and a format that
fixes the failure proves the failure was prompt-fixable. (The lab can
prove that direction only — never that a failure is a weights
limitation. See "How to interpret a result" in the README.)

## Data flow

```text
fixtures (JSON/CSV)          format hypotheses              local checkpoint
search / held-out     ──►    catalog families +      ──►    HF adapter
strict split, NFC            bounded mutations              greedy, exactly N
                                                            tokens, no EOS stop
        │                           │                             │
        └──────────────►  exact token sequences  ◄────────────────┘
                                    │
                        boundary detection (token-ID
                        subsequence, not string stops)
                                    │
                        selection rule: exactly one search
                        winner advances to held-out;
                        ties and failures abstain
                                    │
              results.json (digests, counts, verdicts) ──► report.html
                                    │                        (one file,
              second-opinion layer: diff-reports,             no external
              verify, pack, diff-backends                     assets)
```

## Components

All measurement code is in the single package `model_protocol_lab/`:

- **Catalog and rendering** — the 29 conversation-format families with
  their framing fields, plus the bounded BOS/prefix/thinking-mode
  mutations. Hypotheses are rendered by hand and encoded with
  `add_special_tokens=False`, so a checkpoint can be measured on token
  sequences its own template would never produce.
- **Fixture loading** — strict, bounded JSON/CSV loading with the
  search/held-out separation enforced structurally (IDs, groups and
  message inputs cannot leak across the split). `fixtures build` makes
  a leaking CSV unbuildable; `fixtures fuzz` is the loaders' contract
  fuzzer.
- **Evaluation** — the local Hugging Face adapter (GPT-2 and Llama
  causal-LM configurations, standalone safetensors weights only). The
  adapter never downloads models, never executes remote model code,
  and generates exactly N tokens with no implicit EOS stop, so stop
  behaviour stays a measured variable instead of a hidden default.
- **Boundary detection and selection** — response boundaries are
  matched as explicit token-ID subsequences with a text-level
  cross-check recorded as a diagnostic. The selection rule is
  deliberately conservative: a single candidate must pass every
  search fixture to reach held-out evaluation; anything else abstains.
- **Probes** — the `run --…` diagnostics (USAGE §8, sixty-one
  subsections) are post-hoc or re-measurement instruments over the
  same observations: stop-semantics robustness, cross-checkpoint
  transfer, perturbation, template conformance, and the generated-
  panel probes. Each exists because a design round decided it should,
  in a frozen shape — see the decision index.
- **Reporting** — `results.json` carries verdicts, counts and answer
  digests, not raw generations (one designed exception, documented in
  USAGE §6); `report.html` is one self-contained file. Reports are
  byte-deterministic for identical inputs.
- **Second-opinion tools** — `diff-reports` compares two result
  directories section by section; `verify` re-derives a stored result
  through independent code paths; `pack` seals a report directory
  under a SHA-256 manifest; `diff-backends` isolates emulated backend
  stop semantics. None of them trusts the primary path's arithmetic.
- **Interfaces** — the `model-protocol-lab` CLI (`catalog`, `inspect`,
  `doctor`, `run`, `fixtures`, `debug`, `diff-reports`,
  `diff-backends`, `verify`, `pack`, `surface-noise`) and a stdlib-only
  MCP server (`model-protocol-lab-mcp`) exposing the surface-noise
  probe to agents.

## Trust boundaries, briefly

Checkpoints and chat templates are untrusted input: templates are
statically scanned for code-execution primitives before any render,
weights load only as safetensors, and nothing in the measurement path
touches the network. The full record — trust boundaries, the dated
audit log with findings and explicitly accepted residuals — is
[SECURITY.md](../SECURITY.md); this page deliberately does not
summarize it into a second copy.

## Where the records live

- What the tool does and how to run it: [USAGE](../USAGE.md)
  (reference) and the [FAQ](../FAQ.md).
- Why each instrument has its shape: the design documents, indexed
  from [the decision index](adr-index.md).
- What was researched, and what was killed and why: the
  [research log](../RESEARCH_LOG.md).
- What is verified, and what is *not* established: the repository's
  `VALIDATION.md`.
- What changed: the repository's `CHANGELOG.md`.
