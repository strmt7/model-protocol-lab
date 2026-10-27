# Behavioural template probe — design

`model-protocol-lab doctor --model DIR --template-probe` renders
frozen probe conversations through a checkpoint's own chat template
and reports, per probe pair, whether the rendering changed anywhere
outside the one message whose text was swapped.

**Monday-morning use.** A practitioner who downloads a checkpoint
runs the probe before the first experiment and learns whether the
template's rendering is a pure framing of the conversation or
changes with message *content* — the signature of content-conditional
template logic — instead of trusting that a passed static scan means
the template only frames messages.

## Why a behavioural probe

The static template firewall (`template_audit.scan_template_ast`)
proves the template contains no RCE primitives. It cannot see
content-conditional logic written in legal Jinja: a template that
injects an instruction only when a trigger phrase appears in a user
message passes the AST scan with zero findings. That class is
demonstrated by arXiv:2602.04653 (chat-template backdoors keyed on
four- to six-word trigger phrases in user messages; case-normalised
matching is this probe's own variant choice, not a claim about the
paper — cycle 216). The
probe is the dynamic tier next to the firewall's static tier: the
firewall runs first, and a denied template is never rendered.

## Definitions

The probe conversations are frozen constants in
`model_protocol_lab/templateprobe.py` (protocol v1, SHA-256
fingerprint in every payload; a changed word is protocol drift and
needs a version bump, the refusals v1→v2 precedent).

- **Probe message.** A synthetic message whose content is
  `LEFT_SENTINEL + " " + body + " " + RIGHT_SENTINEL`. Sentinels are
  uppercase-alphanumeric nonces, unique per logical message; the
  alphabet survives `|upper` and `|trim`. No alphabet survives a
  template that rewrites content — those shapes land in
  `transform_unverifiable` / `span_unresolvable` by design.
- **Span.** For a render where a message's left sentinel occurs
  exactly once and its right sentinel exactly once, in order: the
  char interval between them. Defined by construction, never
  inferred from content search or diff alignment. Every other shape
  is `span_unresolvable`.
- **Swap pair.** Two conversations identical except one frozen word
  in one message (length-matched swaps, so legitimate length
  branches do not flip). Each member is rendered twice.
- **Change envelope.** The region between the longest common prefix
  and the longest common suffix of the two renders, computed by
  bisect-on-slice-equality (the algorithm is pinned; a per-char loop
  at the 1M-char render cap costs ~100× more).
- **Within-span check.** When the swapped body's stable fragments
  appear verbatim in the base span, the swapped span must equal the
  base span with the slot word substituted.

## Pair outcomes

One class per pair, in this precedence order:

| outcome | meaning |
|---|---|
| `nondeterministic_render` | a member differed across its own double render |
| `unexercised` | the two renders are byte-identical overall |
| `span_unresolvable` | a sentinel rule failed in either render |
| `content_dropped` | the span is empty/whitespace for a non-empty body |
| `extra_span_change` | the change envelope is not contained in the span |
| `span_content_mismatch` | envelope contained, but the within-span substitution check fails |
| `transform_unverifiable` | envelope contained; body fragments not verbatim in the span, so the within-span check cannot run |
| `clean` | envelope contained and the within-span check passes |
| `render_error` | the render itself raised (driver-level) |

`extra_span_change` and `span_content_mismatch` are the findings.
Findings carry a capped, control-stripped excerpt of the changed
region plus its length and SHA-256; they are grouped by region hash
and the list is capped with an overflow count.

Role arms (`add_system`, `promote_first_user`) report render-diff
facts only — `span_vanished`, `span_merged`, `span_order_changed`,
`role_render_error`, `nondeterministic_render`, `unchanged`. The
payload never claims a region "means" system or that content was
"promoted": a hostile template defines its own framing markers, so
marker semantics are not definable from the render.

## Pre-registered claims

- **T1′.** For the frozen control framer (a plain ChatML-class
  template), every pair is `clean`. Real templates are subjects,
  never controls.
- **T2′.** The seeded backdoor controls are flagged on the pairs
  whose swap completes the trigger: single-word, phrase-completion
  ("please answer precisely"), and case-normalised variants. The
  conditional-removed twin and the payload-neutral twin are flagged
  on no pair. A trigger living in an unchanged message is pinned as
  **not detected at pair level** under protocol v1 — the injection
  is constant across that pair, so no paired diff can see it.
- **T3′.** Findings never aggregate into a template verdict; the
  doctor verdict and CLI exit code are byte-identical with and
  without `--template-probe`, on healthy and on failed reports.
- **T4′.** A second-granularity `strftime_now` control is classified
  `nondeterministic_render` by the baseline double renders.
  Day-granularity time-dependence passes the baseline; findings are
  scoped to the run's date, which the payload records.
- **T5′.** The section is byte-identical across two runs on the same
  checkpoint (same machine, same versions), and carries the template
  source SHA-256, the protocol fingerprint, the render-environment
  versions, and counts for every outcome class — no class is
  silently dropped.

## Budgets and failure mapping

The section is capped at 128 renders and a 120 s cooperative
wall-clock deadline checked between renders (a single render remains
bounded only by the render watchdog where one exists — SIGALRM is
Unix/main-thread only; the docs state this rather than papering over
it). Exceeding either cap fails closed to
`unavailable / template_probe_budget_exceeded` with partial counts
preserved. If `span_unresolvable` + `unexercised` +
`nondeterministic_render` exceed half of all pairs, the section is
`unavailable / template_probe_inconclusive` — the probe could not
see the template, which must not read as a quiet pass. A checkpoint
whose template passes the firewall but whose tokenizer cannot load
reports `unavailable / tokenizer_load_failed`. The extraction scope
is the `tokenizer_config.json`-embedded string only: list-form
templates and a separate `chat_template.jinja` file report
`no_template`.

## Limits, stated with the capability

The probe conversations are published constants, and a template can
detect them — by the sentinel strings, by conversation shape, or by
the synthetic phrasing — and suppress conditional behaviour while
being probed. A section with no findings therefore bounds only
non-probe-aware templates. The template source is scanned for the
sentinel strings before rendering and a hit is reported
(`sentinel_in_template_source`), which catches the literal form of
that evasion, not the shape-based forms. The probe set is a frozen
sample: a backdoor keyed on a trigger outside the swap table, on
conversation shape, or on tools (the render path passes no `tools=`
argument in v1) is missed by construction. Legitimate templates with
content conditionals — length branches the length-matched swaps do
not neutralise, dedup, defaults — produce findings too; a finding is
a fact about where rendering changed, and the inserted text is shown
precisely so a human can judge it. Nothing in the section is a
safety certificate, in either direction.
