# Distractor dilution probe — design

`model-protocol-lab run --distract` answers a deployment question
the selection loop cannot see: once a template is chosen, how much
*irrelevant context* can be poured around the real prompt before
the chosen candidate starts failing — and does irrelevant text that
merely *looks* like it contains a different answer actively capture
the model, or does failure under load come from dilution alone?
Retrieval pipelines, long system preambles, and multi-document
prompts all add exactly this kind of load, so the answer decides
how much context budget a deployment can spend before the protocol
itself becomes the risk.

The probe is post-hoc and observational: it reuses the recorded
search observations as the native (dose 0) arm, generates treated
cells on the same backend, and never touches candidate selection.
The primary decision is reported per condition only as a flip
marker against the recorded outcome.

## Conditions and protocol

Ten conditions, frozen in code (`distraction.CONDITIONS`):

- neutral filler at 64, 256, and 1024 tokens, placed before and
  after the fixture's last user message (6 cells per fixture);
- near-miss filler at 64 and 256 tokens, before and after
  (4 cells per fixture).

Near-miss cells exist only where a *planted answer* is defined
(see below); at 1024 the near-miss arm does not exist, so capture
curves terminate at 256. That is a scope decision, not an
omission: the near-miss claim is about capture at moderate load,
and the generation budget (below) is spent on the neutral dose
curve first.

Filler is whole corpus sentences joined by a single newline,
nested by construction — the dose-*d* span's sentence sequence is
a prefix of the dose-1024 sequence — with the final sentence
truncated to hit the exact token target where whole sentences
cannot. The sizing rule (`whole-sentences-plus-one-truncated-suffix-v1`)
searches candidate spans by rendered token delta against the
native render and takes the first exact hit within a frozen
encode budget (512 encodes per cell); a cell that cannot be sized
exactly is recorded `arm_unmatchable`, never approximated. The
rendered dose is an identity the diff validator re-derives for
every compared cell: `cell.input_tokens - native.input_tokens`
must equal the nominal level.

Both corpora are frozen constants whose SHA-256 digests ride in
the protocol block: a 71-sentence neutral corpus (no digits, no
special-token literals) and four near-miss sentence templates with
a single `{planted}` slot, cycled to fill the span. Every span is
screened before generation: special-token literals for any
candidate, or the candidate's own boundary token texts appearing
in the span, reject the cell as `arm_screened` — filler must never
smuggle protocol structure into the prompt.

## The planted answer (digit-string surgery v2)

A fixture's expected answer yields a planted string only when it
is a plain ASCII number: optional ASCII sign, an integer part
either plain or comma-grouped in threes (or a bare fraction like
`.5`), an optional fraction, an optional exponent. The plant is
the same string with the last mantissa digit incremented, carry
propagating through digits only, regrouping commas on overflow
(`"9,999"` → `"10,000"`, `"007"` → `"008"`, `"-0"` → `"-1"`,
`"42.0"` → `"42.1"`, `"1,000.5"` → `"1,000.6"`). Everything else —
surrounding whitespace, underscores, non-ASCII digits, `inf`/`NaN`,
dates, number words — yields no plant. The rule is total and
deterministic, independent of any ambient decimal context; the
exact battery lives in the cycle-184 tests. Where no plant exists, or the plant equals
another fixture's expected answer, or the candidate's native
answer already *is* the plant, the near-miss arm is undefined for
that (candidate, fixture) and its cells record
`near_miss_undefined`.

Scope, stated plainly: near-miss capture is defined only for
fixtures whose expected answer is a number in the ASCII shape
above. On answer sets without such fixtures the near-miss arm is
empty by construction and the probe reduces to the neutral dose
curves.

## Claims and denominators

- **D1′ (dose curve).** Per candidate, per filler kind and
  placement, the paired pass rate at each level: records where the
  native arm passes or fails, the cell was generated, and both
  sides exist. Rates are reported per dose, never pooled across
  placements or kinds.
- **D2′ (first flip).** Over native-passing records with a
  complete curve, the distribution of the first failing level,
  plus two counters the flip histogram alone would hide: *returns*
  (passing levels above the first flip — recovery under more load
  is evidence the flip was noise, not a threshold) and *gains*
  (native-failing records that pass at some dose).
- **D3′ (capture, an upper bound).** Over native-passing records
  whose near-miss cell was generated, the share whose answer is
  exactly the planted string. Byte-exact scoring can only
  undercount capture (a captured answer with stray formatting
  scores as a plain failure), so the rate is labelled an upper
  bound on *clean* capture in the report. The headline number is
  the **excess over neutral**: records captured under near-miss
  whose same-dose neutral answer is not the plant. Dilution alone
  cannot manufacture the planted string in the neutral arm, so the
  excess separates active capture from generic load failure.
- **D4′ (per-condition selection).** The conservative selection
  rule re-run per condition over the candidates that completed
  that condition on the *same* fixture set. Where completed sets
  differ, selection is withheld for that condition
  (`selection_withheld_mismatched_sets`) rather than compared
  across different evidence. Flips are marked against the recorded
  primary decision under the same vocabulary mapping the diff
  validator uses.
- **D5′ (construction honesty).** Every cell records how its
  treated prompt relates to the native render: `id_exact` when the
  token ids decompose exactly into native prefix + inserted span +
  native suffix (the intended surgery), `text_level` when only the
  text decomposes (a tokenizer seam merged across the junction).
  Seam token ids (first/last inserted id) are recorded for
  text-level cells. A dose claim is only as strong as its
  construction, and the report keeps the two apart.

Vacuous fixtures (empty expected answer, or expected contained in
the prompt) and echo fixtures are counted and excluded from every
rate. Records whose native arm is unavailable make the whole
candidate unavailable: the probe never mixes a partial native arm
with generated cells.

## Cost

Worst case is frozen in the protocol: 10 conditions × 2,400
ground records = 24,000 generations, where ground records are
candidate × fixture pairs with a completed native arm. The CLI
help states the count and the local backend's forwards formula
(`M·L + M(M−1)/2` for M forwards of growing length L with the KV
cache off) so the price is visible before the run, not after.
Per-cell sizing probes are capped separately (512 encodes) and
never generate.

## What the probe does not do

- No grouping readout in v1: the think/non-think style grouping
  used by the repetition probe's disclosure is not repeated here;
  construction class and capture carry the honesty load.
- No capture claim at 1024 (see Conditions).
- No cross-backend claims: the protocol fingerprint binds the
  backend fingerprint, both corpus digests, the rule versions,
  the budgets, and the fixture set; a diff across differing
  protocols reports drift and refuses comparability.
- The diff validator recomputes every derived value from the
  records (curves, first-flip state machine, capture, per-condition
  selection, fingerprint) and rejects payloads whose shape
  pre-checks fail (record caps, byte bound, digest shapes) before
  recomputing. Like every recompute validator in this project, it
  proves internal consistency of the payload against the frozen
  protocol — it cannot prove the generations really happened; that
  is what the sealed bundle and the determinism audit are for.
