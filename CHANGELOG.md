# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (cycle 186 — numeric surface-form probe, 2026-10-21)

- `model-protocol-lab run --numform`: numeric surface-form probe
  (wave-7 rank 7, the last wave-7 item). The lab generates its own
  integer tasks (copy / compare / add; values stratified by digit
  band and by round, margin, and carry construction classes) and
  renders every value in four surface forms — plain, comma-grouped,
  narrow-no-break-space-grouped (U+202F), scientific — under one
  shared fixture id, with frozen digit-free zero-shot wording and
  the pass bar held constant (exact value, in the form the operands
  arrived in, boundary found). Paired per-task form contrasts with
  a frozen classification over discordant counts are the headline;
  every cell decomposes into exact / right-value-wrong-form /
  wrong-value (copy and add) or wrong-label / non-label (compare),
  so a grouping-keeping failure is never read as a miscalculation.
  The tokenizer's digit grouping is measured at run time and
  reported as a run label. Records store the generated integers in
  the clear, and `diff-reports` recomputes the section in full
  (`diff_numform`), with protocol/generator/backend drift marking
  comparisons not comparable. No per-form selection. The research
  round narrowed the claim to the hand-rendered chat-template axis
  (Reddy et al. 2026 and NUPA already vary numeric formats outside
  it), the red-team pair's amendments are in the shipped contract
  (total value-only renderers, clear canonical values,
  grammar-gated answer parsing with caps before conversion, no
  numeric worked example in the prompts), and the design is frozen
  in `docs/NUMFORM_DESIGN.md`. The cycle also fixes a latent gap
  the verifier found in the diff test oracles: both oracles'
  `identical` conjunctions omitted the leading-claim section;
  the clauses are now present in both.

### Added (cycle 185 — leading-claim caving, 2026-10-21)

- `model-protocol-lab run --leading`: leading-claim caving probe
  (wave-7 rank 6). One frozen claim sentence is appended to each
  search fixture's last user message: a wrong claim planted from
  another fixture's correct answer (donor-first construction —
  smallest-id eligible donor from a different group, digit-surgery
  fallback, per-record plant source) and the right claim as the
  control arm, each in first-person and third-person framings that
  form a minimal pair (identical certainty wording). A plant is
  eligible only when it is not equivalent to any expected answer
  in the target's group under the scorer-lens R4 rung plus a
  substring-containment screen. The headline metric is the
  per-candidate per-framing caving count over native-passing
  paired fixtures (pass right, fail wrong), with byte-exact
  adoption reported beside the failed-otherwise instability
  bucket; the classification is a frozen truth table over the
  discordant pair (the research brief's marginal-rate acquittal
  rule was disproved by counterexample pre-build and deleted),
  the McNemar exact p is diagnostic only, per-condition
  selection is withheld on mismatched or undefined completed
  sets, and the common-population readout is the only ranked
  cross-candidate comparison. Payloads carry ground records
  (ids, counts, digests) and `diff-reports` recomputes every
  derived value, including cross-record donor consistency and
  the donor-map fingerprint. The pre-build swarm corrected the
  brief's literature attributions (SycEval is Fanous et al.; the
  first-person effect is Wang et al.), the verifier replaced
  the brief's named mirror stage (hierarchy records none of the
  required per-fixture data), and red-team A's DO NOT BUILD AS
  SPECIFIED verdict (15 findings) survived B's cross-check with
  amendments; the shipped design is the amended gate set. The
  probe measures factual caving on verifiable fixtures, not
  social sycophancy; no authority arm ships in v1 because the
  literature disagrees on its size. Design:
  `docs/LEADING_DESIGN.md`.

### Added (cycle 184 — distractor dilution, 2026-10-20)

- `model-protocol-lab run --distract`: distractor-dilution probe
  (wave-7 rank 5). Irrelevant filler is spliced into each search
  fixture's last user message at exact rendered token doses —
  64/256/1024 tokens of a frozen neutral corpus and 64/256 of
  near-miss filler, before and after the real content — against
  the recorded native run as dose 0. The near-miss arm plants a
  wrong answer derived from the fixture's expected answer by a
  frozen ASCII digit-increment rule (digit-string surgery v2,
  adopted after the red-team cross-check refuted the original
  Decimal-based plant rule as non-total), and the headline
  capture number is the excess over the same-dose neutral arm,
  with the raw rate labelled an upper bound. Per-dose paired
  pass curves, a first-flip state machine with returns and gains
  counters, per-cell construction classification (id-exact vs
  text-level with seam ids), per-span filler screening,
  per-condition selection withheld over mismatched completed
  sets, and a full-recompute `diff-reports` validator with shape
  pre-checks. The wave-7 scope cut is recorded in the design
  doc: ten conditions, 24,000 generations at the run caps, no
  capture claim at 1024. The pre-build cross-check upheld
  red-team A's DO NOT BUILD verdict on the original sketch (the
  named mirror stage records none of the required per-fixture
  data) and red-team B then found A's corrected design still
  unbuildable (the near-miss filler text was never constructed);
  the shipped design is A's skeleton as amended by B.

### Added (cycle 183 — prompt repetition × template, 2026-10-20)

- `model-protocol-lab run --repeat`: repetition probe crossing
  prompt repetition with the candidate templates (wave-7 rank
  4, arXiv:2512.14982). Per search fixture, the last user
  message is repeated verbatim inside the same turn (x2)
  against a token-length-matched filler control (pad) and the
  recorded native run (x1, reused — the baseline costs no
  generations). The headline repetition effect is x2 minus pad
  at identical prompt length; pairs that cannot be matched
  exactly leave the denominator, with the pad-evaluable share
  and delta sign counts disclosed per candidate. Per-fixture
  degradation (context budget, screening, evaluation failure),
  a frozen pre-repetition detector with an excluding-flagged
  subpopulation readout, per-arm winner re-derivation with the
  selection readout withheld over mismatched completed sets,
  and a full-recompute `diff-reports` validator. The run's
  cancel event is threaded dispatch → wrapper → stage →
  evaluate (the first `_run_*` probe to thread it end-to-end),
  and a failed or cancelled native search gates the probe to
  zero generations. Doubled prompts cost real compute under
  this lab's full-prefix decoder; no latency claim is imported
  from the paper. Design: `docs/REPETITION_DESIGN.md`; the
  design survived a red-team pair cross-check that rejected
  the research sketch's construction invariant, payload shape,
  and padding control before any code was written.

### Added (cycle 182 — behavioural template probe, 2026-10-20)

- `model-protocol-lab doctor --template-probe`: behavioural probe
  of a checkpoint's chat template — the dynamic tier next to the
  static template firewall, which proves the absence of RCE
  primitives but cannot see content-conditional logic in legal
  Jinja. Frozen sentinel-bracketed probe conversations (protocol
  v1, SHA-256 fingerprint in the payload) are rendered in
  word-swap pairs through the checkpoint's own tokenizer, each
  member twice; pairs are classified by where the renders change
  (change envelope vs the swapped message's sentinel span, plus a
  within-span substitution check), and role arms report
  render-diff facts only. Findings are per-pair facts with capped,
  control-stripped excerpts — never a template verdict, and the
  doctor verdict and exit code are unchanged. Budgets (128
  renders, 120 s cooperative deadline) fail closed with partial
  counts; a probe that mostly cannot see the template reports
  `template_probe_inconclusive`, not a quiet pass. The probe
  conversations are published constants and a template can detect
  and evade them; the design doc states that limit with the
  capability. Design: `docs/TEMPLATE_PROBE_DESIGN.md`; the
  definitions survived a red-team pair cross-check that refuted
  both the research design's span notion and the first corrected
  draft's exercised-pair rule.

### Added (cycle 181 — candidate aliasing audit, 2026-10-20)

- `model-protocol-lab run --aliasing`: post-hoc audit of the
  run's candidate set over the recorded search `input_ids`, zero
  extra model calls — candidates are grouped into render classes
  (identical prompt token sequences) and measurement classes
  (identical prompts and identical response boundary) inside
  cohorts of identical completed fixture sets, reporting
  effective vs recorded candidate counts for both notions,
  redundant completed observations under one per-fixture formula
  (never "wasted": they are also the replicates the determinism
  cross-check consumes), a per-pair coincidence matrix with
  explicit denominators, determinism-suspect cells inside a
  measurement class (output-digest or outcome disagreement,
  attributable to backend nondeterminism or artifact corruption;
  the audit asserts neither), and an alias-suspect abstention
  flag when an `ambiguous` decision's passing set sits inside
  one measurement class with no outcome-suspect cell. Classes
  are computed from recorded IDs only — never the candidate
  fingerprint, which hashes the format's name. The payload
  carries per-cell SHA-256 digests under a frozen canonical
  definition (`sha256-json-intlist-v1`) and counts only — no
  token IDs, no text; digests do permit offline confirmation of
  a guessed prompt, and the docs say so. New pure module
  `aliasing.py`; `diff_aliasing` in diff-reports recomputes
  every aggregate from the ground records (an
  internal-consistency recompute) and reports partition,
  boundary, count, and flag drift with fixture-set / backend /
  digest-version comparability gates. The design was red-teamed
  pre-build (red-team A: DO NOT BUILD AS SPECIFIED, 10 findings;
  red-team B upheld the verdict, downgraded five severities,
  refuted one sub-claim, and amended six of the ten build
  gates); the corrected contract is
  `docs/ALIASING_AUDIT_DESIGN.md`.

### Added (cycle 180 — scorer-lens regrade, 2026-10-20)

- `model-protocol-lab run --scorer-lens`: post-hoc regrade of the
  recorded search answers under a frozen normalisation ladder
  (R0 exact; R1 whitespace; R2 NFC + casefold + combining-mark
  removal; R3 markdown-fence strip and edge-punctuation strip;
  R4 numeric value under a strict ASCII grammar), reporting
  per-candidate verdict counts per rung (the boundary fact is
  always conjoined and its blocked count reported), first-flip
  attribution per fixture, the surface share of each candidate's
  native failures, and the search-phase winner re-derived per
  rung (withheld, with a named reason, over any partial or
  ineligible field). The ladder travels in the payload as data
  with a SHA-256 digest; the payload carries counts and fixture
  IDs only — no answer or expected text. The design was
  red-teamed pre-build (12 findings upheld on cross-check, two
  binding amendments added); the corrected contract, including
  the no-label-strip and no-decimal-comma honesty boundaries, is
  `docs/SCORER_LENS_DESIGN.md`. New pure module `scorerlens.py`;
  `diff_scorerlens` in diff-reports recomputes every aggregate
  from the per-fixture ground records before trusting a payload.
- Also fixed: the cycle-67 diff oracle's `identical` conjunction
  was missing the refusals terms added in cycle 174 (latent —
  the implementation had them; the fuzz pairs never isolated
  the difference).

### Fixed (cycle 179 — `--perturb` dispatch nested inside `--transfer-model`, 2026-10-20)

- `model-protocol-lab run --perturb N` silently did nothing
  unless `--transfer-model` was also passed: the dispatch block
  added in cycle 144 was indented inside the
  `if args.transfer_model:` block in `cli.py`, so the flag
  parsed, the stage existed and was unit-tested, but on a
  perturb-only run the branch never fired — no perturbation
  section in the results, no error, exit 0. The block is now a
  sibling dispatch at the same level as every other probe
  branch. Found by the new parametrized dispatch tests below,
  which exist to fail on exactly this class of wiring error;
  no existing test pinned the broken nesting.

### Added (cycle 179 — parametrized CLI dispatch e2e + ablation mutation gates, 2026-10-20)

- `tests/test_cycle179_cli_dispatch.py`: all 24 `run` stage
  flags parse-pinned (dest set with the flag, falsy without);
  the 19 `_run_*_probe` helpers pinned for both halves of the
  cycle-174 dispatch pattern — with the stage layer stubbed, the
  helper attaches a real payload under its result key, and with
  the stage stubbed to raise, it attaches exactly the
  `{"available": False, "detail": "<key>_unavailable"}` marker
  without raising; the five inline flags (`--transfer-model`,
  `--perturb`, `--token-healing`, `--logprob-diagnostic`,
  `--sensitivity-sweep`) driven through the real run flow with
  stubbed stages. Plus five tempdir mutation gates
  for `ablation.py` (swap dropped, no-op arm included, flip
  direction swapped, selection bar weakened, inapplicable set
  inverted) — the last pure module whose exactness-critical
  lines had no gate.

### Fixed (cycle 178 — docs currency audit, 2026-10-20)

- README's prompt-boundary canonicality bullet inverted the
  rate: it called the fraction of pairs that *do* re-encode to
  the recorded ids "the Prompt Boundary Problem rate"; the audit
  (`boundary_audit.py`) defines the rate as the fraction that
  does *not* re-encode (`n_problem / n_usable`). The bullet now
  says so.
- `docs/SEALED_BUNDLE_DESIGN.md` overstated the bundle container
  pre-parse (wave-7 F8): "disk fields are zero, and ZIP64 is
  rejected" read as container-wide guarantees; the code enforces
  both at the EOCD record only (EOCD disk fields, ZIP64 sentinel
  values). The sentence now names that scope.
- README Capabilities omitted shipped probes and commands:
  bullets added for `--structured`, `--deliforge`, `--few-shot`,
  `--fake-turn`, `--tool-call`, `--verbosity`,
  `--template-robustness`, `--ablate-fields`, `--cache-profile`,
  `--confidentiality`, `--answer-collision`, `--refusals`,
  `--logprob-diagnostic`, `--include-evidence`, `pack`,
  `fixtures fuzz`, `inspect`, and `verify --bundle`; the
  diff-only probes (`--synth-history`, `--batch-invariance`,
  `--position`, `--stop-cause`) got their own bullets, and the
  `diff-reports` bullet now names the twelve comparisons it had
  omitted. The "What the report looks like" paragraph claimed a
  token-healing section the committed sample does not render
  (and a failure-mode diagnosis its clean-pass banner does not
  carry); it now describes the 31 sections the sample renders
  and names the eight it omits.
- USAGE §6's report inventory omitted fourteen renderer
  sections and claimed the committed sample shows "every section
  above" — false in both directions; the inventory is complete
  and the sample claim now names the absent sections. §12's
  `diff-reports` inventory omitted 22 stage keys and 44 summary
  keys; both lists are now complete, with a per-stage comparison
  bullet. §§8.33–8.36 sat physically after §14; they now follow
  §8.32 inside §8.
- FAQ stopped at the token-budget frontier (cycle ~130):
  entries added for the probe catalogue and its cost split,
  `pack` / `verify --bundle`, `verify --determinism`, and
  `fixtures fuzz`, plus the confidentiality and refusal probes.
- CHANGELOG backfilled for cycles 164, 165 (Added), 166, 167,
  and 172, which shipped without entries.
- VALIDATION's top evidence header claimed "cycles 10–153"
  while its own section cites cycles 176–177; the range now
  runs to cycle 178. Its ruff line stated "(241 files)" without
  the scope that produces the number; the line now names the CI
  scope (`model_protocol_lab tests examples`; 242 files with
  this cycle's pin test included), and a new pin test compares
  the stated count against a live CI-scope
  `ruff format --check` run.

### Fixed (cycle 177 — wave-7 red-team fixes round 3, 2026-10-20)

- `diff-reports` validators for cache profile and
  confidentiality now recompute their derived values
  (wave-7 F6b/F6c), completing the F6 cluster: cache means and
  fractions must equal their counts' ratios under the payload's
  block size, and the gap/extremes/uniformity/exclusions are
  recomputed from the cells; confidentiality rates must equal
  leaks over the paired denominator, the McNemar p-value is
  recomputed from the discordant counts, the positive/base-rate/
  evaluable flags are recomputed from the producer's
  definitions, run fractions must be consistent with the leak
  counts, and the profile's extremes, spread, and positive set
  are recomputed from the measurable cells.

### Fixed (cycle 176 — wave-7 red-team fixes round 2, 2026-10-20)

- Confidentiality leak detector measured only marker *prefixes*
  (wave-7 F4): recovering the marker's last three tokens scored
  0.0. It now scores the longest contiguous run of marker tokens
  anywhere in the output, as documented; three-of-four recoveries
  register as leaks.
- Confidentiality cell validator rejected marker offset 0
  (wave-7 F5): an offset is a position and 0 is defined
  (prefix-free formats); offsets now use a non-negative check.
- `diff-reports` validators for refusals and ablation now
  recompute the derived values their producers guarantee
  (wave-7 F6a/F6d), matching the answer-collision standard:
  refusal rates must equal their counts' ratio, the rate/note
  binding and note vocabulary are enforced, hit openers must
  come from the payload's own frozen list, duplicate candidate
  names are rejected; ablation flip counts must satisfy
  n_flips = pass_to_fail + fail_to_pass = len(flipped ids) with
  string ids, and load-bearing fields are recomputed from the
  base/bar story.

### Fixed (cycle 175 — wave-7 red-team fixes, 2026-10-20)

- Determinism audit certificate was fail-open (wave-7 F3, HIGH):
  a claimed-completed cell with invalid recorded token IDs was
  skipped without blocking the certificate, contradicting the
  design doc's I5. `recorded_ids_invalid` now blocks, and cell
  status is judged before ID validity so an aborted observation's
  legitimately empty inputs are labelled
  `observation_not_completed` (scope), never an artifact defect.
- Refusal detector v2 (wave-7 F1 HIGH + F2): bare prefix matching
  fired on natural non-refusals ("I will note..." matched
  `i will not`; "As an Air Force pilot..." matched `as an ai`).
  Matching now requires a word boundary (Unicode L*) with the
  longest boundary-satisfying opener reported; normalisation
  treats format characters as spaces and strips combining marks
  after decomposition. Same opener list (digest unchanged);
  detector id is now `frozen_refusal_openers_v2`; v1 and v2 rates
  are not comparable. Documented cost: glued missing-space
  refusals are misses.
- `pack` refused at collection (wave-7 F7): a report file named
  `manifest.json` previously sealed into a bundle pack's own
  verifier rejects; the name is now reserved
  (`pack_entry_name_reserved`).

### Added (cycle 174 — suspected-refusal profile, 2026-10-20)

- New opt-in probe `model-protocol-lab run --refusals` (wave-6
  idea 6): per-template suspected-refusal rate on benign search
  fixtures, classified post-hoc over the recorded answers with
  zero extra model calls. The detector is frozen as
  `frozen_refusal_openers_v1` (pure `refusals.py`): XSTest
  Appendix D core minus "no,"/"no." (dropped deliberately — they
  fire on correct short yes/no answers; deviation documented in
  the module), plus itemised OR-Bench Appendix M and GCG-lineage
  supplements; beginning-anchored after a frozen normalisation,
  longest-match attribution, list digest carried in every
  result's method block so a detector change surfaces in
  `diff-reports` as protocol drift. Rates are named *suspected*
  throughout (string matching is the field's known-weak baseline;
  no published accuracy transfers to this fixture set), travel
  with exact counts and a suspected × passed cross-tab, and are
  withheld for candidates whose search did not complete. Hits
  name the fixture and the opener — never the answer text, so
  the section is safe in the public results file. HTML section
  and `diff_refusals` included; USAGE §8.36; uniqueness pass and
  swarm record in `docs/RESEARCH_LOG.md`. Informational only;
  never touches selection.

### Added (cycle 173 — determinism self-audit, 2026-10-20)

- New opt-in mode `model-protocol-lab verify --determinism --model
  <checkpoint>` (wave-6 idea 8): the recorded search observations of
  a run are re-executed against the local checkpoint — same recorded
  `input_ids`, same token budget, per-request seeds re-derived
  through the new shared `core.request_seed_for` — and compared
  token-for-token. All cells reproduce: an unsigned, self-issued
  determinism certificate bound by SHA-256 to the exact cells
  compared. Any cell diverges: a per-cell divergence table, reported
  as a backend-determinism finding (the audit proves the
  re-execution, never the original run; divergence is not an
  artifact-integrity failure). A fingerprint gate runs before any
  generation (a different checkpoint, device, or context limit is a
  different experiment), recorded cells are validated against the
  recording pipeline's own invariants before any blame is assigned,
  and replay cost is capped at the largest search phase a run can
  record. Pure `determinism.py`; contract and pre-registered
  falsifiable claims in `docs/DETERMINISM_AUDIT_DESIGN.md`; USAGE
  §13.2.

### Added (cycle 172 — fixture-loader contract fuzzer, 2026-10-20)

- New command `model-protocol-lab fixtures fuzz [--seed N]`
  (wave-6 idea 7): a seeded contract fuzzer for the fixture
  loaders — a fixed, published 133-case mutator table (74 JSON
  text, 44 CSV text, 9 JSON file, 6 CSV file) checked against the
  loaders' documented accept/reject contract. Mutator parameters
  derive from SHA-256 (never `random.Random`); reports carry no
  fixture text, paths, or timestamps, and the same seed
  reproduces the report byte-for-byte across runs and
  `PYTHONHASHSEED` values. Accept cases pin stored decoded
  values; reject cases pin messages only where the loaders'
  wording is fixed vocabulary. Five tempdir mutation gates, each
  proven non-vacuous. Pure `fixturefuzz.py`; USAGE §9. The
  corpus is a fixed table, not coverage-guided fuzzing, and the
  docs say so.

### Fixed (cycle 172 — loader defects found by the fuzzer, 2026-10-20)

- Lone surrogates accepted by the JSON loader: escape-form
  `"\ud800"` entered fixture fields, producing splits the repo's
  own serializer refuses; raw-surrogate text was rejected only
  via a bare `UnicodeEncodeError` codec leak. Both loaders now
  reject lone surrogates deliberately in the loaders' own
  vocabulary; valid surrogate pairs still accept.
- CSV `id`/`group` had no length check: a 17,000-character id
  parsed, then failed the round-trip writer — the authoring path
  accepting what its own writer refuses. The `MAX_TEXT` check
  now covers every field; a max-length id round-trips.
- Every `csv.Error` was mislabeled "field exceeds the parser
  limit". The size message is kept for the genuine limit error;
  other parse errors now report as parse errors.

### Added (cycle 171 — sealed result bundle, 2026-10-20)

- New commands `model-protocol-lab pack` and
  `model-protocol-lab verify --bundle` (wave-6 idea 5): a report
  directory is sealed into one hash-bound bundle (per-file SHA-256
  manifest binding results, report, included inputs, catalog digest,
  and package version); verification re-hashes everything offline,
  fails closed on any mismatch, and then re-runs the `verify` decision
  replay on the sealed `results.json` as an independent second layer.
  The seal is unsigned and the docs say so in the first paragraph: it
  proves bit-integrity since packing only against a digest obtained
  through an independent channel (`pack` prints it for exactly that
  purpose; `--expect-digest` checks it). Pure `pack.py`; format
  contract in `docs/SEALED_BUNDLE_DESIGN.md`; `parse_results_bytes`
  factored out of `diffreports.load_results` so plain `verify`, `pack`,
  and `verify --bundle` reject exactly the same artifacts.

### Added (cycle 170 — answer-collision audit, 2026-10-20)

- New research instrument `mpl run --answer-collision` (wave-6 idea 4):
  the lab audits its OWN token-ID boundary detector. Synthetic expected
  answers embedding marker-like text (the boundary configuration's own
  marker at start / interior / end / repeated positions, every hostile
  marker string, one clean control) are emitted verbatim through the
  evaluation path by scripted backends (per-fixture caps) and classified
  by a total order — collision (the detector cut a correct answer and
  discarded content) vs ends-at-marker (the emission simply ended
  there; never conflated, and "model stopped early" is unobservable
  under scripting, so the phrase is not used) vs intact — in a reserved
  arm (natural encoding; the measurement) and a subword control arm
  (decode-identical re-encoding via `deliforge.find_encoding_pair`;
  `no_encoding_pair` recorded where no pair exists). The unit is the
  boundary configuration, not the template (boundaries are
  operator-supplied; shared-boundary cells are identical by
  construction). Scope is stated in CLI help, docs, and report copy:
  configured boundaries and this tokenizer only; the audit measures the
  detector, never a model. Pure `collision.py`,
  `experiments.answer_collision_stage` / `answer_collision_profile`,
  HTML section (`id="answer-collision"`, counts and rates only — marker
  text is never rendered), `diff_answer_collision` with
  protocol-fingerprint drift (`answer_collision_changed` /
  `answer_collision_asymmetric` feed `identical`). Pre-registered
  falsifiable claims F1–F3 logged in `docs/RESEARCH_LOG.md` (cycle 170);
  `uniform_intact` is the regression-guard pass. Sample report exercises
  the audit with genuine collisions under the demo `|` boundary.

### Added (cycle 169 — labeled-block confidentiality probe, 2026-10-20)

- New research instrument `mpl run --confidentiality` (wave-6 idea 3):
  per candidate template, whether the privilege label on a prepended
  block reduces verbatim exposure of its content under frozen extraction
  elicitations. A fixed nonce marker is rendered inside the
  privilege-labeled block (treatment; the lab has no system role, so the
  construct is labeled-block exposure, reusing hierarchy's
  `privilege_block` verbatim) and as the identical bare marker at the
  same position (control); direct / authority-claim / memory-reframe
  elicitations are mirrored exactly across arms. Detection runs on the
  full decoded output (token-subsequence, NFC + casefold), headline leak
  at 3 of 4 marker tokens contiguous; paired statistics (discordant
  counts, exact McNemar p), fail-closed `None` gaps for zero control
  rates and under-floor candidates, and `uniform_nonpositive` as the
  publishable null (Tensor Trust Appendix H reading). Reported quantity
  is a **verbatim-exposure lower bound** -- paraphrase leakage is out of
  scope and it is never a "safe to ship secrets" claim. Pure
  `confidentiality.py`, `experiments.confidentiality_stage` /
  `confidentiality_profile`, HTML section (`id="confidentiality"`),
  `diff_confidentiality` with protocol-fingerprint drift
  (`confidentiality_changed` / `confidentiality_asymmetric` feed
  `identical`), oracles in test_cycle67/test_cycle27, sample exercise
  with genuine per-template gap variation, and `docs/USAGE.md` 8.34.
  Red-team A 15 findings (3 CRITICAL: no system role, position confound,
  per-candidate markers) cross-checked by red-team B (12 confirmed; new
  M1 asymmetric-elicitation fixed by mirroring); design and uniqueness
  pass in `docs/RESEARCH_LOG.md`.

### Added (cycle 168 — prefix-cache sharability profile, 2026-10-20)

- New research instrument `mpl run --cache-profile` (wave-6 idea 2): per
  template family, the fraction of the run's recorded prompt tokens that
  exact-prefix KV-cache reuse (vLLM automatic prefix caching) could skip,
  computed offline from the recorded `input_ids` with zero extra model
  calls and block-quantized to full blocks (default 16, vLLM's block
  size; `--cache-profile-block-size` re-quantizes for other engines).
  The headline metric is the **achievable block fraction** -- exact, not
  a bound: every shared prefix costs exactly one first-occurrence miss
  in every request order (proven pre-code; brute-force simulation over
  all permutations of random prompt sets in the test suite), so the
  instrument also reports the token-level diagnostic and the
  eviction-robust "potential" upper bounds with their closed-world
  scope labeled. Pure `cacheprofile.py`, post-hoc
  `experiments.cache_profile_stage` / `cache_profile_profile`, HTML
  section (`id="cache-profile"`), `diff_cache_profile` (`cache_profile_changed`
  / `cache_profile_asymmetric` feed `identical`), oracles in
  test_cycle67/test_cycle27, sample exercise with genuine signal, and
  `docs/USAGE.md` 8.33. Pre-registered falsifiable claims and scope
  limits in `cacheprofile.py`; terminology fence ("prefix sharability",
  a cost input never a cost verdict, no "efficiency" term). Red-team A
  (13 findings; the design-defining R1: the naive "cacheable fraction"
  headline provably misranks templates against *every* realizable order,
  hence the achievable quantity; R3: pairwise LCP dropped as
  length-confounded/quadratic/redundant) and red-team B (0 refutations,
  13 survive, independent simulation agreement) reviewed the design
  pre-code. Informational only -- never touches selection.

### Fixed (cycle 168 — HTML body missing two sections)

- Two report sections were registered in the sticky table of contents
  but never concatenated into the HTML body: the template-robustness
  (`#template-robustness`) and field-ablation (`#field-ablation`)
  sections, so their TOC links pointed at missing anchors. The body now
  renders them in TOC order; a regression test asserts every TOC href
  resolves to a body id.

### Added (cycle 167 — template field ablation, 2026-10-20)

- New opt-in probe `model-protocol-lab run --ablate-fields`
  (wave-6 research idea 1): single-field causal ablation of the
  format's structural fields — one field at a time (BOS, role
  markers, separators, suffix) replaced by its neutral value,
  the conservative selection rule re-applied per arm — so the
  winner's verdict is attributed to the fields that actually
  carry it. Pure `ablation.py`; HTML section and
  `diff_ablation`; USAGE §8.32. Informational only; never
  touches selection.

### Added (cycle 166 — per-template surface-noise robustness, 2026-10-20)

- New opt-in probe `model-protocol-lab run --template-robustness`:
  for each candidate, the fraction of its *correct* search
  verdicts whose pass survives presentation-level surface noise
  (whitespace, case, Unicode variants) applied to the user
  message contents — preservation measured on correct verdicts
  only, so a template cannot look robust by failing
  consistently. `surfacenoise.probe_template_robustness` under a
  frozen protocol; HTML section and
  `diff_template_robustness`; USAGE §8.31. Informational only;
  never touches selection.

### Added (cycle 165 — surface-noise self-consistency probe, 2026-10-19)

- New command family `model-protocol-lab surface-noise`
  (`probe` / `diff` / `protocol`) and a stdlib-only MCP server
  (`tools/surface_noise_mcp_server.py`, four tools): a
  self-consistency probe for a single prompt — a frozen set of
  surface-noise operators (typo, case flip, Unicode variant,
  whitespace modes, punctuation strip) perturbs the prompt, the
  fixtures re-score it, and the report gives a per-operator
  stability vector with minimal-flip counterfactuals plus a
  self-contained HTML dashboard. The reported nearest observed
  flip is an upper bound over the searched subset, never a
  guarantee; academic precedent (CoIPO/NoisyPromptBench decoding
  radius, PCE minimal-flip counterfactuals) is cited, not
  claimed over. Pure `surfacenoise.py`; contract in
  `docs/SURFACE_NOISE_PROBE_DESIGN.md`.

### Fixed (cycle 165 — mutation-gate recursion + Codecov badge, 2026-10-19)

- Mutation-gate self-recursion eliminated in all four in-place gates
  (cycles 124/125/131/134): each gate's subprocess ran `pytest` on its whole
  test file, gate included, so the gate re-entered itself once per nesting
  level and stacked mutants — the cycle-125 gate's 22-site anchor fanned
  out to 184 processes in one traced run, and the stacked mutants poisoned
  the file left on disk, failing 136 tests in the next suite run. The
  subprocess now passes `-k "not TestMutationGate"` (`--deselect` was tried
  first but silently never matches: `__file__` is absolute under unittest
  discovery while pytest normalizes node IDs to relative paths). The kill
  verdict now comes from the audit tests alone. Genuine-kill audit of all
  16 mutants found two that had been surviving vacuously behind the
  recursion confound, both now pinned: cycle-125's span-merge `budget - 2`
  mutant (new `test_adjacent_spans_merge`) and cycle-134's per-candidate
  `"rate": 0.0` mutant (extended `test_planted_problem_shows_in_rate` to
  assert the per-candidate rate). The cycle-135/136 gates already ran
  pytest against single test targets and were unaffected.
- README Codecov badge switched to shields.io with `label=Codecov`
  (capital C in the rendered image), mirroring the owner's
  rejection-rejector repo; `test_readme_codecov_badge_exact` pins the line
  byte-exact.

### Added (cycle 164 — response verbosity profile, 2026-10-19)

- New opt-in readout `model-protocol-lab run --verbosity`
  (wave-4 research idea 5): a post-hoc verbosity profile per
  template family — recorded generation lengths per correct
  answer, computed from the finished run with zero extra model
  calls. Framed as a cost/behavior profile, not a quality
  score; generations cut off by the token cap count as lower
  bounds. Pure `verbosity.py`; HTML section and
  `diff_verbosity`; USAGE §8.30. Informational only; never
  touches selection.

### Added (cycle 163 — prompt-only tool-call adherence × template, wave-4 research idea #4, 2026-10-19)

- `mpl run --tool-call`: prompt-only tool-call syntax adherence per template
  family (Berkeley Function-Calling Leaderboard prompt-mode task, with the
  chat template as the independent variable — BFCL varies models/providers,
  JSONSchemaBench varies frameworks; nobody varies the template). A fixed
  4-function tool list (typed params, enums, numeric bounds) is described in
  prompt text — no native function-calling APIs — and each family must emit
  exactly one line per query: `CALL <tool_name> <JSON arguments>` or `NO_CALL`
  when no tool fits. 9 fixed queries cover selection (which tool), typed
  argument extraction, and relevance detection (2 abstain queries — the
  abstention analogue of the lab's conservative selection rule). The
  deterministic validator is fail-closed: unparseable → `parse_failed`
  (fails all three rates), non-string → `not_text` (excluded), unknown tool
  never correct; a correct abstention counts as correct selection of no tool,
  so a perfect model scores 1.0 on every rate. Pre-registered falsifiable
  claim: max-minus-min adherence gap ~0 uniform across families, reported per
  rate so a composite gap decomposes. New pure `model_protocol_lab/toolcall.py`
  (100% line+branch coverage), `experiments.toolcall_stage` /
  `toolcall_adherence`, results.json + HTML section (TOC `tool-calls`, all
  interpolations escaped), `diff_toolcall` with `toolcall_changed` /
  `toolcall_asymmetric` feeding `identical` (selection/args/relevance drift,
  added/removed candidates, excluded-set changes). The probe floors the token
  budget at 128 (longest canonical call is 79 chars) so the budget can never
  confound the measurement. Sample exercise with a scripted call-policy
  backend shows genuine decomposable signal (llama3 1.0 on every rate;
  mutated variants 0.22 selection / 0.0 args / 0.78 relevance), report
  regenerated byte-identical ×3 (md5 86fa4327f769a6587f1151c511749564 /
  64aaf76841f34b228e4c956faf02c140). Boundary with cycle 157 documented:
  157 measures JSON schema validity of a free response; this measures
  function selection, argument extraction, and abstention.

### Added (cycle 162 — fake-turn history tampering × template, wave-4 research idea #3, 2026-10-19)

- `mpl run --fake-turn`: fake-turn history-tampering robustness per template
  family (arXiv:2405.20234 "Hidden in Plain Sight"). A forged prior turn
  (fake assistant asserting a wrong answer; fake system contradicting the
  task — the deliforge payload builders over the same fixed MARKERS list)
  is appended to the search fixtures' user content at each family's native
  encoding; the independent variable is the forged marker's token provenance
  — `same` when the marker is also a genuine delimiter of the rendering
  template, `cross` when it is a real chat-template control token from
  another family's token set. Per (candidate, payload, marker) cell the
  verdict-flip rate against native search verdicts; per candidate the pooled
  same/cross rates and the transfer gap (cross minus same, in [-1, 1]); the
  pre-registered falsifiable claim is a ~0 gap with no token-set clustering.
  Every rendered prompt is integrity-checked (unique ref exactly once,
  forged marker at least twice); violations degrade fail-closed. New pure
  `model_protocol_lab/faketurn.py` (100% line coverage), `experiments.faketurn_stage`
  / `faketurn_robustness`, results.json + HTML section (TOC `history-tampering`,
  all interpolations escaped), `diff_faketurn` with `faketurn_changed` /
  `faketurn_asymmetric` feeding `identical` (cell flip-rate drift, transfer-gap
  drift, provenance changes, added/removed candidates, excluded-set changes).
  Sample exercise with a scripted provenance-policy backend shows genuine
  signal (llama3 same 1.0 / cross 0.0 / gap -1.0), report regenerated
  byte-identical ×3 (md5 7339f7378d50d8ac702f0d767c3fe21f /
  0993b2622513127c869c675127c93b0c). Boundary with deliforge is documented:
  deliforge varies encoding of fixed text; faketurn varies delimiter
  provenance at native encoding.

### Added (cycle 161 — mandatory dev-tool integration: cocoindex-code, caveman, crawl4ai, 2026-10-19)

- Mandatory development-tool integration, ported from strmt7/FlitzZip and made
  obligatory the same way (AGENTS.md prose + `tools/agent_context.py` startup
  fail-closed gate + contract tests; no CI enforcement, per the adversarial
  cross-check verdict in `~/workspace/agents_out/xcheck/VERDICT.md`):
  `.agents/skills/caveman/SKILL.md` (lite compression for internal AI
  communication only, never in persisted text), `.agents/skills/cocoindex-code-search/SKILL.md`
  (mandatory semantic code navigation before opening many files, exact `rg`
  stays authoritative), and the self-hosted Crawl4AI research route
  (`docs/crawl4ai.md`, `tools/crawl4ai_tool.py` with install/doctor/crawl/qualify/check-lock,
  version-pinned `tools/requirements/crawl4ai.txt`, 32-site qualification manifest).
  `tools/agent_context.py` is a byte-faithful port (bounded reads, provenance-aware
  notes, startup gate); `tools/cocoindex_agent_search.py` ports the external-mirror
  semantic search wrapper (pinned CocoIndex Code 0.2.41, CPU embeddings, telemetry
  off, stale-index refusal, query-hash receipts). FlitzZip-specific machinery
  (Windows process-ownership contracts, NLTK/repaired-wheel source builds) is
  intentionally not ported. README carries the three upstream badges. 33 new
  contract tests (all offline, no network/model downloads).

### Added (cycle 160 — few-shot × template, wave-4 research idea #2, 2026-10-19)

- `mpl run --few-shot`: few-shot demonstration gain × template family
  (arXiv:2511.09700). Two demos are sampled (seeded, v1 k=2, the
  two-orders design) from the search split — never held-out — and every
  candidate is evaluated 0-shot and with both demo orders on the
  held-out fixtures, demos prepended as real user/assistant turns so
  each family renders them through its own template. Per candidate:
  `gain` (mean over the two orders of the 0-shot → k-shot
  verdict-change rate, in [0, 1]) and `order_sensitivity` (population
  std of the two per-order rates, in [0, 0.5]); the selection rule is
  re-applied per arm (baseline / order_a / order_b) with
  winner-vs-primary flips. New pure `fewshot.py` (100% line+branch
  coverage), `results.json` `fewshot` payload, TOC-anchored HTML
  section, and `diff_fewshot` (`fewshot_changed` /
  `fewshot_asymmetric` feed `identical`). Gain is a verdict-change
  rate, never a quality judgment. 85 new tests, 4/4 tempdir-module
  mutation gates, 200-case brute-force oracle differential, both diff
  oracles extended; USAGE §8.27 documents the method and its honest
  framing.

### Added (cycle 159 — delimiter authority, wave-4 research idea #1, 2026-10-19)

- `mpl run --deliforge`: reserved-token vs subword delimiter authority
  per template family (arXiv:2609.35932). Forged chat-template markers
  (fake assistant turn / fake system instruction) are sent in two
  encodings of identical decoded text — the family's native single-id
  ("reserved") encoding vs forced subword tokens — and the
  `authority_gap` (reserved-minus-subword verdict-flip rate, in
  [-1, 1]) is measured per family against the native search verdicts.
  The subword arm is built by text-level construction (the exact
  inserted string is located exactly once in the rendered prompt; only
  forged marker occurrences are re-encoded segment-wise, so the
  template's genuine markers keep their reserved ids — verified by id
  count). New pure `deliforge.py` (100% line coverage), `results.json`
  `deliforge` payload, TOC-anchored HTML section, and `diff_deliforge`
  (`deliforge_changed` / `deliforge_asymmetric` feed `identical`).
  Mid-build, a 200-case brute-force oracle differential proved the
  first implementation's difflib id-alignment unsound (it could splice
  a genuine marker while leaving a forged one reserved); the shipped
  code uses text-level construction instead. 69 new tests, 4/4
  tempdir-module mutation gates, both diff oracles extended; USAGE
  §8.26 documents the method and its honest framing.

### Fixed (cycle 158 — wave-4 red-team fix-ups, 2026-10-19)

- B1 [moderate, honest-measurement correction]: cycle 156 documented
  that a flipped verdict on *identical* ids is a backend-determinism
  anomaly (`n_same_ids_verdict_flips`), never whitespace fragility —
  but the shipped aggregation counted same-ids flips in every fragility
  surface (`fragile`, `n_fragile_mutations`, `fragile_candidates`,
  `whitespace_robust`). The aggregation now implements the documented
  split: fragility counts only divergent-ids flips
  (`n_fragile_flips = n_verdict_flips - n_same_ids_verdict_flips`, a new
  exposed per-mutation field), `aggregate_mutation_records` rejects
  records where `n_same_ids_verdict_flips > n_verdict_flips`, and the
  docstrings/RESEARCH_LOG/USAGE text now match the shipped behavior.
  Regression tests plus tempdir-module mutation gates pin the exclusion.
- F1 [moderate, latent]: non-finite floats (`NaN`, `±Infinity`) were
  classified `adherent` for `number` fields with minimum-only/no bounds
  (RFC 8259 §6: not JSON numbers, though `json.loads` accepts them as a
  Python extension) — reachable through the real pipeline via a backend
  emitting the 3-character text `NaN`. The number branch of
  `_field_value_ok` now requires finiteness (with in-module
  `_is_finite_number` precedent); `validate_structured_specs` rejects
  non-finite answers, `build_structured_fixtures`' `_canonical_json`
  runs with `allow_nan=False`, and non-finite enum members are rejected
  at spec validation. The independent oracle shared the blind spot —
  it was fixed too, and the brute-force value pool now generates
  non-finite floats so the differential covers the hole.
- F2 [low]: `structured_stage` silently absorbed malformed schema dicts
  (every response classifying `not_text`, indistinguishable from
  backend non-text) despite its docstring promising `ValueError` on bad
  schemas. The stage now validates every schema with
  `validate_schema_def` up front; the `OUTCOMES` docstring now states
  that `not_text` marks a non-string response *or* a malformed schema
  against which nothing can be evaluated.
- F3 [low]: `layout_options` silently shortened layouts on duplicate
  options; duplicates are now rejected `ValueError` (fail-closed, per
  the module's style and the docstring's "Raises ValueError on bad
  input").
- F4 [low]: `minimum`/`maximum` were accepted (and carried) on
  non-numeric field types, producing unsatisfiable schemas that fail
  every response with a green validation. Bounds are now rejected on
  non-`integer`/`number` fields, matching the docstring.
- F5 [low]: `_valid_structured` accepted self-contradictory cells
  (`adherent_rate > parse_rate`); it now enforces the producer
  invariants documented by `candidate_adherence`
  (`adherent_rate <= parse_rate`, `schema_rate <= parse_rate`).

### Added (cycle 157 — structured-output / JSON-mode adherence, wave-3 research idea #5, 2026-10-19)

- New `mpl run --structured` experiment measuring prompt-only JSON
  adherence per template family (arXiv:2501.10868v2 JSONSchemaBench;
  askpert.dev JSON-mode notes): 9 synthetic fixtures across 3 shapes
  (extraction, classification, routing), each with a flat hand-rolled
  JSON schema (zero runtime dependencies) and a fixture-identity marker
  `(ref: structured/<id>)`. Responses are classified independently of
  the strict `exact_match` verdict (`parse_ok` / `parse_failed` /
  `not_text`, then `schema_ok` / `schema_failed` / `schema_invalid`
  against the hand-rolled validator), aggregated into per-candidate
  parse/schema/adherent rates with the invariant
  `adherent_rate <= parse_rate`. The pre-registered falsifiable claim is
  that the template change produces a measurably positive adherence gap
  (max-minus-min adherent rate) rather than ~0. Honest framing: the lab
  cannot constrain a decoder, so this is the prompt-only elicitation
  baseline, not a constrained-decoding claim. `results.json` gains the
  `structured` section (per-candidate cells, `weakest_candidates`,
  `adherence_gap`, `uniform` null-hypothesis flag); the HTML report gains
  the TOC-anchored "Structured-output adherence" section; `diff-reports`
  gains `diff_structured` (rate/gap drift, weakest added/removed,
  candidates added/removed) with `structured_changed` /
  `structured_asymmetric` feeding `identical`. 142 new tests (4/4
  tempdir mutation gates, 200-case brute-force oracle differential,
  Hypothesis properties, single-cause pins, hostile-payload tests);
  both diff oracles extended. This was the LAST unconsumed wave-3
  research idea — the wave-3 backlog is now exhausted.

### Added (cycle 156 — whitespace canonicalization, wave-3 research idea #4, 2026-10-19)

- New `mpl run --ws-scan` experiment auditing whitespace at template
  boundaries per family (aicassindra "Tokenization, in depth" production
  case study; arXiv:2405.17067 "Tokenization Matters!"): every search
  candidate's rendered prompts are re-encoded under 8 whitespace
  mutations of the render string (`trailing_space`,
  `double_trailing_space`, `trailing_newline`, `trailing_tab`,
  `leading_space`, `leading_newline`, `strip_trailing_space`,
  `collapse_double_space`), id sequences are compared exactly, and
  fixtures on divergent variants are re-run through the normal
  `core.evaluate()` pipeline via an encode-time backend wrapper
  (`WsScanBackend`, mirroring `PrefillBackend`) to measure verdict
  flips. New pure `model_protocol_lab/wscanonical.py` (100% line
  coverage: the mutation table, exact id-sequence comparison,
  per-mutation aggregation with the `whitespace_robust` verdict);
  `experiments.ws_stage` / `ws_canonicalization` (fail-closed per
  candidate x mutation cell); `results.json` `ws_canonical` section;
  HTML "Whitespace canonicalization" section with TOC anchor
  (id-divergence / verdict-flip rate matrix with `=` / `D` / `F`
  cells); `diffreports.diff_ws_canonical`
  (`ws_canonical_changed` / `ws_canonical_asymmetric` feed
  `identical`); sample-report exercise with a scripted
  whitespace-sensitive parity backend showing genuine id divergence
  and verdict flips. Honest split: divergent ids + flipped verdict is
  whitespace fragility; a flipped verdict on identical ids is recorded
  separately as a backend-determinism anomaly, never as fragility.
  122 new tests (4/4 mutation gates, 200-case brute-force oracles,
  Hypothesis properties, 8 single-cause `changed`-disjunct pins);
  both diff oracles extended.

### Added (cycle 155 — stop-cause termination audit, wave-3 research idea #3, 2026-10-19)

- New `mpl run --stop-cause` experiment classifying every search
  candidate's recorded generations into `eos_id` / `stop_string` /
  `max_length` / `stop_unknown` (arXiv:2407.20042 "When to Stop?";
  llama-cpp `"stop"` conflates EOS token and stop-string match). New
  pure `model_protocol_lab/stopcause.py` (100% line coverage:
  first-EOT positions, the classify_cause decision table with the
  documented stop-string-beats-EOS same-step tie rule, over-generation
  bleed arithmetic, cause-rate aggregation); `experiments.stop_stage`
  (pure post-hoc on stored observations -- no new generations;
  per-candidate and overall aggregates; fail-closed
  `stop_unknown`); `results.json` `stop_cause` section; HTML
  "Stop-cause termination audit" section; `diffreports.diff_stop_cause`
  (`stop_cause_changed` / `stop_cause_asymmetric` feed `identical`);
  sample-report exercise with a scripted EOT-emitting backend showing
  genuine `eos_id` vs `stop_string` variation and over-generation
  bleed. Honest framing: the lab's decoder never stops early, so every
  cause is post-hoc classification of what a stop-aware engine would
  have done. 61 new tests (4/4 mutation gates, 200-case brute-force
  oracles, Hypothesis properties); both diff oracles extended.

### Added (cycle 154 — position sensitivity, wave-3 research idea #1, 2026-10-18)

- New `mpl run --position` experiment measuring option-position bias as a
  template-family property (arXiv:2507.13949 primacy effect; corroborated
  by arXiv:2512.02665): synthetic multiple-choice fixtures whose correct
  option is permuted between the first, middle, and last positions
  (seeded distractor order), with the selection rule re-applied per
  condition. New pure `model_protocol_lab/position.py` (seeded layouts,
  `primacy_index` = P(correct|first) - P(correct|last) in [-1, 1]);
  `experiments.position_stage` / `position_sensitivity` (fail-closed
  per-cell, all-pass verdicts, winner flips vs primary, condition
  effect); `results.json` `position` section; HTML "Position
  sensitivity" section; `diffreports.diff_position`
  (`position_changed` / `position_asymmetric` feed `identical`);
  sample-report exercise with a scripted primacy/recency policy backend
  showing a genuine +1.0/-1.0 index split and winner flips. 82 new
  tests (4/4 mutation gates, 200-case brute-force oracle, Hypothesis
  properties, 14-test renderer/edge coverage class); both diff oracles
  extended.

### Added (cycle 153 — batch-invariance audit, wave-3 research idea #2, 2026-10-18)

- New `mpl run --batch-invariance` experiment auditing the lab's core
  "exact token sequences" invariant against batch-composition
  nondeterminism (Thinking Machines, "Defeating Nondeterminism in LLM
  Inference", Oct 2026): every search fixture's exact prompt token
  sequence is decoded greedily in isolation and co-batched through the
  backend's optional `generate_batch` path, and continuation ids are
  compared token-for-token. New pure `model_protocol_lab/batchinv.py`
  (first-divergence comparison, divergence rate, `batch_invariant` /
  `batch_variant` / `unavailable` verdicts); `experiments.batch_stage`
  fails closed (`batch_unsupported` when no batch path exists --
  never a synthesized batch; `batch_contract_failed` /
  `batch_failed` on contract violations; per-prompt solo failures
  degrade to error markers, never phantom divergences);
  `results.json` `batch_invariance` section; HTML "Batch-invariance
  audit" section; `diffreports.diff_batchinv`
  (`batch_invariance_changed` / `batch_invariance_asymmetric` feed
  `identical`); sample-report exercise with a scripted batch-capable
  backend demonstrating a genuine detected divergence. 57 new tests
  (4/4 mutation gates, 200-case brute-force oracles, Hypothesis
  properties, 8 fake-backend stage scenarios); both diff oracles
  extended.

### Fixed (cycle 153 — self red-team: broken report TOC anchors, 2026-10-18)

- The report's table of contents linked to `#selection-robustness`,
  `#transfer`, `#perturbation-robustness`, `#token-healing`,
  `#instruction-hierarchy`, `#synthetic-history` (cycle 150),
  `#batch-invariance` (this cycle), and `#prefill-sensitivity`, but
  those sections rendered `<section class="card">` with no `id` --
  every one of those TOC links was dead, contradicting the `_toc`
  docstring's "never points at a missing id" claim (found while
  verifying the new section's anchor). All eight sections now carry
  their anchor as `id`; verified against the regenerated sample
  report.

### Fixed (cycle 152 — wave-3 red-team A terminal-escape fixes, 2026-10-18)

- `diff-reports` stderr rendering interpolated hostile-controlled
  group labels raw in three summary renderers
  (`_render_synthhistory_summary`, `_render_hierarchy_summary`,
  `_render_prefill_summary`): a `group_pass_rates` label like
  `\x1b[2J\x1b[31mPWNED\x1b[0m` cleared/recolored the terminal and
  `\u202e` enabled bidi reordering. All three labels are now wrapped
  in `_display_id`, completing the wave-2 C2 remediation. 4 new
  regression tests.

### Fixed (cycle 151 — wave-3 red-team fixes for synthhistory, 2026-10-18)

- `synthhistory_stability` flip detection compared the selection
  vocabulary ("selected"/...) against the raw run-level primary
  decision ("compatible_on_fixture_set"), recording a phantom "flip"
  wherever history re-selected the identical winner. The two run-level
  success decisions now map onto ("selected", winner) for the flip
  comparison (other outcomes compare verbatim); the stored
  `primary_decision` stays raw. Plus: whitespace-only candidate names
  degrade to `invalid_candidates` instead of raising (the "never
  raises" contract), and zero-fixture stages degrade to invalid
  (aligning with the diff validator). 4 new regression tests; sample
  report regenerated (genuine flip only).

### Added (cycle 150 — synthetic-history selection stability, 2026-10-18)

- Synthetic-history selection stability experiment (wave-2 research
  idea #5, the last brief item): `mpl run --synth-history` prepends
  fixed benign synthetic prior turns (one exchange for `history_2`,
  three for `history_6`) to every fixture and re-applies the
  conservative selection rule per condition. New pure module
  `model_protocol_lab/synthhistory.py` (history construction with
  id/group namespacing; mechanical delimiter-explicitness labels
  derived from the lab's own `Format` hypotheses -- 22 explicit / 7
  loose across the catalog); `experiments.synthhistory_stage` /
  `synthhistory_stability` (fail-closed cells, flips vs primary,
  per-candidate pass rates, delimiter-group means reported as
  correlation); HTML section; `diff_synthhistory` with
  `synthhistory_changed` / `synthhistory_asymmetric` feeding
  `identical`; sample-report exercise with a scripted many-shot
  steering policy (genuine winner flip under `history_6`).
  Pre-registered falsifiable claims in the module docstring.

### Added (cycle 149 — Codecov wiring, owner-mandated add-on, 2026-10-18)

- Codecov coverage upload in CI, mirroring the owner's other repos:
  the test job emits `coverage xml -o coverage.xml` after the
  `coverage report --fail-under=95` gate and uploads it with the
  SHA-pinned `codecov-action` (v7.1.1) via OIDC (`use_oidc: true`,
  `fail_ci_if_error: true`; the job gains `id-token: write`). New
  `.codecov.yml` (project target auto, patch target 95% = the CI
  gate, `tests/`/`examples/`/`__main__.py` ignored, comment
  `require_changes: true`) and a codecov README badge directly after
  the CI badge. Pinned by new CI-shape tests (pinned SHA, OIDC,
  `fail_ci_if_error`, `id-token: write`, byte-exact badge URL).

### Added (cycle 149 — oracle-mixture / protocol-portfolio gap, 2026-10-18)

- Protocol portfolio analysis (wave-2 research idea #4): pure post-hoc
  over the stored search observations, computed on every `mpl run` at
  zero extra model cost. Answers "how much does the single global
  winner leave on the table versus a per-fixture oracle, and which
  small protocol portfolio covers every solvable fixture?" -- the
  routing lens of RouteLLM applied within the lab to *prompt
  protocols*, not models. New pure module
  `model_protocol_lab/portfolio.py` (input validation, `oracle_rates`,
  `per_fixture_oracle_winners` with deterministic name-sorted
  tie-breaks, `greedy_cover` with deterministic lexicographic
  tie-breaks), `experiments.portfolio_analysis` (derives per-candidate
  per-fixture pass from stored observations -- exact answer AND
  explicit boundary, never re-scored; misaligned candidates excluded),
  an HTML "Protocol portfolio" section (gap, cover list, per-fixture
  winner table), and `diffreports.diff_portfolio` (`portfolio_changed`
  / `portfolio_asymmetric` feed `identical`; compares the gap, rates,
  solvable-fixture set, portfolio membership, per-fixture oracle
  winners, and per-member covers). Two honesty points stated in the
  report itself: the portfolio is a greedy set-cover *approximation*,
  not a proven optimum, and fixtures no protocol passes are reported
  as unsolvable, never folded into the portfolio. `gap >= 0` is pinned
  by Hypothesis properties; the stage never fails the run.

### Added (cycle 148 — assistant-prefill sensitivity experiment, 2026-10-18)

- `mpl run --prefill` (wave-2 research idea #2): re-evaluates every
  search candidate with benign format-lock prefills (`Answer: `,
  `Final answer: `, `Result: `) appended as raw token ids after the
  generation prompt, then re-applies the conservative selection rule
  per prefill. The scored span is `prefill + generated text` (the
  deterministic contract that the model "said" the prefill); grading
  needs no fixture transformation since `prefill + generated ==
  prefill + expected` iff `generated == expected`. Honoring is
  detected exactly and separately: a model that did not accept the
  prefill as its own opening restarts its turn and duplicates the
  prefill text. New pure module `model_protocol_lab/prefill.py`
  (spec validation, fixture construction, the honoring predicate, the
  think-channel predicate computed from each candidate's own
  generation-prompt tail -- 10 catalog families open a reasoning
  channel, thinking on/off variants labeled independently -- and the
  `PrefillBackend` wrapper), `experiments.prefill_stage` /
  `prefill_adherence` (per-prefill verdicts, flips vs primary,
  per-candidate honoring rates, think-channel group means reported as
  correlation, not causation), an HTML "Assistant-prefill sensitivity"
  section, and `diffreports.diff_prefill` (`prefill_changed` /
  `prefill_asymmetric` feed `identical`). Fail-closed throughout: a
  prefill that cannot be encoded, does not decode back to its text, or
  contains a candidate's response boundary degrades that cell instead
  of silently mis-scoring. The stage never fails the run; roughly
  quadruples search compute.

### Added (cycle 147 — instruction-hierarchy adherence probe, 2026-10-18)

- `mpl run --hierarchy` (wave-2 research idea #1): re-evaluates every
  search candidate on 8 synthetic system-vs-user conflict fixtures and
  re-applies the conservative selection rule per privilege-block
  position. The lab never sends system messages, so the system
  instruction is delivered as a privilege-marked block folded into the
  user turn, before (primacy) or after (recency) the conflicting user
  instruction; adherence is exact string equality against the two
  expected answers. New pure module `model_protocol_lab/hierarchy.py`
  (fixture construction, `FAMILY_SYSTEM_TREATMENT` label map derived
  only from the catalog's own documentation: 14 dedicated-frame / 3
  folded / 12 unknown families), `experiments.hierarchy_stage` /
  `hierarchy_adherence` (per-position verdicts, flips vs primary,
  primacy-vs-recency position effect, documentation-derived group
  means reported as correlation, not causation), an HTML
  "Instruction-hierarchy adherence" section, and
  `diffreports.diff_hierarchy` (`hierarchy_changed` /
  `hierarchy_asymmetric` feed `identical`). The stage never fails the
  run; roughly triples search compute.

### Added (cycle 146 — token-healing selection sensitivity, 2026-10-18)

- `mpl run --token-healing` (wave-2 research idea #3): replays the whole
  experiment once more with every prompt healed before generation, then
  re-applies the conservative selection rule to the healed outcomes.
  The implementation ports HF `GenerationMixin.heal_tokens` faithfully
  into the lab's manual greedy decoder — decode the prompt, re-encode it
  (canonical re-tokenization), extend the tail token with every
  vocabulary token that starts with it (deterministic sorted scan,
  space→`Ġ` whitespace trick), trim the tail, take a single biased
  argmax (+10.0 per extension, +1.0 on the original tail). New pure
  module `model_protocol_lab/healing.py` (injected decode/re-encode/
  vocab/argmax callables), `LocalHFBackend.heal_input_ids` (single
  forward pass on the trimmed prompt), and `TokenHealingBackend`
  (fingerprint `+token-healing`, `n_healed`/`n_fallback` counters,
  validates the healed `tuple[int, ...]` contract at the boundary and
  falls back to native ids). Everything fails closed; the stage never
  fails the run.
- A `falsification` block testing the cycle-134 prompt-boundary
  audit's own prediction, one-directional by design: a verdict may
  change only for a candidate the audit flagged as seam-fragile (a
  fragile-but-stable candidate need not flip — an iff would have
  falsely violated). `results.json` carries the healed (decision,
  winner), the flip flag, per-candidate (native, healed) outcomes,
  per-fixture flips, and the falsification table; the report renders a
  "Token-healing selection sensitivity" HTML section.
- `mpl diff-reports` now compares the `token_healing` section between
  runs: `diff_token_healing` (healed-verdict drift, per-candidate
  (native, healed) drift, falsification drift). New `healing_changed` /
  `healing_asymmetric` summary flags feed `identical`; stderr summary
  lines. Both oracles updated (cycle-27 availability-level + cycle-67
  full reimplementation with fuzz + hardcoded expected summary +
  identical mutation-gate anchor); 51 tests, 5/5 `changed`-disjunct
  mutants killed.

### Fixed (cycle 145 — wave-2 adversarial-audit hardening, 2026-10-18)

- `margin_trajectory` was still O(B·C) quadratic: the wave-1 point cap
  bounds total points, not candidates, so a hostile payload with C
  candidates × 1 distinct point each meant C² work (~1.4s at C=2000,
  projecting to hours per `diff-reports` pair — measured ~4.1× per
  doubling by two independent agents). Restructured to an O((E+B) log
  (E+C)) event sweep with a lazily-deleted leader heap: 0/1400
  differential mismatches against the old sweep on seeded frontiers
  (including count revisits and decreases), C=2000 now in 0.008s.
- `diff-reports` stderr renderers interpolated raw
  `old_decision`/`new_decision` (transfer, robustness, perturbation
  drift summaries), injecting terminal escapes from a hostile
  results.json and falsifying `render_summary`'s sanitization comment —
  all three now wrapped in `_display_id`; non-string
  decision/winner/tier values no longer crash `render_summary` with
  `TypeError` (`_display_id` is now total: None → "(none)", others via
  `str()`), keeping the degrade-gracefully contract.
- `changed`-flag disjunct pinning: single-cause tests for
  cell-drift-only (`diff_transfer`), tier-escalation-only
  (`diff_contamination`), and stability-only / variant-drift-only /
  removed-only (`diff_perturbation`) — all five disjunct mutants re-run
  and killed.
- `checkpoint_drift` stderr rendering capped at 10 with "... and N
  more", like `cell_drift`; the cycle-67 differential fuzz now emits
  real payloads for the four newest diff sections; the template
  firewall docstring now describes the static layer's actual scope.
- Docs: restored the RELEASE_NOTES 0.1.0 historical snapshot against
  the actual 0.1.0 tree (measured 6063 tests / 135 files / 15 files),
  fixed the CHANGELOG cycle-135 count (34), refreshed stale lists and
  filled README/USAGE/RESEARCH_LOG/SECURITY gaps. 16 new tests in
  `tests/test_cycle145_wave2_hardening.py`.

### Added (cycle 144 — fixture-perturbation robustness experiment, 2026-10-18)

- New `run --perturb N` (1-8): re-evaluates every search candidate
  against the primary checkpoint on N deterministically perturbed
  fixture variants (search split only) and re-applies the conservative
  selection rule per variant -- answering "does the winner survive
  fixture phrasing perturbations, or did it win on phrasing luck?"
  New pure `model_protocol_lab/perturbation.py` (seeded operators:
  typo, case flip, whitespace, Unicode variant, punctuation strip --
  message content only; ids, groups, expected answers untouched),
  `experiments.perturbation_stage` (re-runs the evaluation pipeline
  per variant, fail-closed per cell, honest `changed_fixtures`
  counts), and pure `experiments.perturbation_robustness`
  (per-variant verdicts, flips, stability summary, per-operator
  breakdown with mean fixture-change rates). Wired into
  `results.json`, the HTML "Fixture-perturbation robustness" section,
  and `diff-reports` (`diff_perturbation` with
  `perturbation_changed` / `perturbation_asymmetric` feeding
  `identical`). Completes the winner-reliability trilogy with
  stop-semantics robustness (cycle 137) and cross-checkpoint transfer
  (cycle 138). 48 new tests, 2/2 mutation gates killed.

### Added (cycle 143 — contamination comparison in diff-reports, 2026-10-18)

- `diff-reports` now compares the `contamination` report between runs
  (`diff_contamination`): worst-tier drift, pairs added/removed, tier
  escalations -- answering "did fixture hygiene improve or degrade?"
  after a dataset update. New `contamination_changed` /
  `contamination_asymmetric` summary flags feed `identical`.

### Verified (cycle 142 — contamination HTML safety audit, 2026-10-18)

- Security audit of the contamination detector's HTML integration
  (cycle 132 feature): confirmed the `worst_tier` is escaped at the
  `_config_section` boundary (no injection); added a regression test
  pinning the escaping guarantee at the integration point. The transfer
  report section was likewise verified to escape all dynamic values.

### Fixed (cycle 141 — hostile-input crash hardening, 2026-10-18)

- Fixed three `TypeError` crashes on hostile/malformed input with
  non-string dict keys: `experiments.transfer_portability` (mixed-type
  checkpoint IDs crashed `sorted()`, violating its "never raises"
  contract — now fails closed with `invalid_candidates_or_transfer`),
  `diffreports.diff_transfer` and
  `diffreports.diff_selection_robustness` (mixed-type
  per-checkpoint/per-semantic/matrix keys crashed `sorted()` on a
  hostile results.json — the structural validators now require string
  keys, so the malformed side degrades to asymmetric instead of
  crashing `mpl diff-reports`).

### Added (cycle 140 — transfer comparison in diff-reports, 2026-10-18)

- `diff-reports` now compares the `transfer` experiment between runs
  (`diff_transfer`): transfer-verdict drift, per-checkpoint (decision,
  winner) drift, per-cell candidate x checkpoint verdict drift, and
  checkpoints added/removed — the fleet-portability regression signal
  after a checkpoint or fixture update. New `transfer_changed` /
  `transfer_asymmetric` summary flags feed `identical`.

### Changed (cycle 139 — test-suite speed, 2026-10-18)

- The two `pytest --collect-only` subprocess spawns (README test-count
  pin in `test_cycle4` and the every-file-collected gate in
  `test_ci_portability_guard`, ~30-60s each) now share one memoized
  collection per session via `tests/_collect_cache.py`. Zero coverage
  change; both gates keep their retry-on-transient-failure behavior
  (only successful collections are cached).

### Added (cycle 138 — cross-checkpoint transfer experiment, 2026-10-18)

- New `run --transfer-model` (repeatable, max 4; IDEAS #1): after the
  primary run, every search candidate is re-evaluated against each
  additional local checkpoint on the search split only (same boundary,
  same `--max-new-tokens`; greedy, exact) via
  `experiments.transfer_stage`, and the conservative selection rule is
  re-applied per checkpoint via pure
  `experiments.transfer_portability`. `results.json` carries `transfer`
  with checkpoint ids/paths/fingerprints, per-checkpoint
  decision/winner, the candidate x checkpoint verdict matrix
  (`transfers` / `wins_there` / `ties_there` / `fails_there` /
  `checkpoint_incompatible`), and the transfer verdict (`full` /
  `partial` / `none` / `unavailable`). The HTML report gains a
  "Cross-checkpoint transfer" section. Failed checkpoints degrade to
  incompatible (fail-closed, never silently skipped); the stage never
  fails the run.

### Added (cycle 137 — selection robustness under stop semantics, 2026-10-18)

- New `experiments.selection_robustness` (IDEAS #3): pure post-hoc
  re-application of the conservative selection rule to *every* search
  candidate under each emulated stop semantic (native, HF
  StopStringCriteria, vLLM detokenizer, EOS-appended) — answering "would
  a different protocol have won under my serving stack's stop
  semantics?" Tokenizer-only, no model calls. Reports per-semantic
  verdicts, the per-candidate pass matrix, `excluded_candidates`, and
  `flips`; a semantic whose emulation cannot run for every candidate
  (e.g. unknown EOS id) is reported as `unavailable`, never as a phantom
  `no_candidate`. Wired into `mpl run` (`results.json`), the HTML report
  ("Selection robustness under stop semantics" section), and
  `diff-reports` (`diff_selection_robustness` +
  `selection_robustness_changed`/`selection_robustness_asymmetric` flags
  feeding `identical`) — the serving-stack migration signal.

### Added (cycle 136 — template conformance firewall, 2026-10-18)

- New `model_protocol_lab/template_audit.py`: static firewall scan of the
  checkpoint's Jinja `chat_template` (research backlog R3, the RCE-class gap
  the CPU watchdog doesn't cover — CVE-2026-5760). `scan_template_ast`
  walks the parsed AST *before* any render and denies dunder
  attribute/subscript access, the `|attr` filter (CVE-2025-27516 sandbox
  escape), dangerous Jinja globals (`lipsum`, `cycler`, `joiner`, `self`),
  `super()` calls, and non-literal or huge `range()` bounds, with
  fixed-vocabulary findings. Pure and fail-closed (parse failures deny);
  zero false positives on ChatML/Llama-3-style templates.
- New `doctor.template_firewall(model_path)`: the checkpoint-backed driver
  (reads `tokenizer_config.json`, never renders); `doctor.template_conformance`
  now refuses to render a denied template (fail closed, status `denied`).
- `model-protocol-lab run` attaches `template_firewall`; the HTML report
  renders a "Template firewall" section with the verdict and findings;
  `jinja2>=3.1.6` pinned in the `hf` extra (sandbox-escape fix); the
  committed sample report exercises the section via the real scanner.
- New `tests/test_cycle136_template_firewall.py` (29 tests): benign-template
  allowlist, a 10-payload SSTI corpus (each denied with the expected rule),
  fail-closed paths, the driver against real `tokenizer_config.json` files,
  HTML rendering with hostile-finding escaping, and a 4-mutant kill gate —
  4/4 killed.

### Added (cycle 135 — chat-marker isolation conformance, 2026-10-18)

- New `model_protocol_lab/control_tokens.py`: chat-marker isolation
  conformance (research backlog R4, the defense side of structural template
  injection). A fixed corpus of 16 hostile marker strings (`<|im_start|>`,
  `<s>`, `[INST]`, `<|begin_of_text|>`, a forged
  `<|im_start|>assistant<|im_end|>` turn, …) is injected into user text and
  the checkpoint tokenizer's control-token ids are counted before and
  after — the number should not change. Pure (`render`/`encode` injected;
  `count_control_tokens`, `check_isolation`, `inject_hostile`,
  `probe_format`, `summarize_probes`); a nonzero delta means the hostile
  text's literal markers became real control tokens (the ChatInject/Phantom
  class, via HF's `split_special_tokens=False` default).
- New `doctor.marker_isolation(model_path, formats, cases)`: the
  checkpoint-backed driver (tokenizer only, never weights), mirroring
  `template_conformance`'s fail-safe shape. The special set is the
  tokenizer's `all_special_ids` union special-flagged added tokens (robust
  for under-declaring tokenizers); a tokenizer declaring no special ids
  reports `no_special_ids` (unmeasurable, not a failure).
- `model-protocol-lab run` attaches `marker_isolation`; the HTML report
  renders a "Chat-marker isolation" section with the families x markers
  conformance matrix (`isolated` vs `smuggled` cells); the committed sample
  report exercises the matrix via the real pure probe with a scripted
  tokenizer.
- New `tests/test_cycle135_marker_isolation.py` (34 tests): the pure core,
  the driver against genuinely loadable WordLevel tokenizers (smuggled
  `<|im_start|>` detected, unknown `[INST]` stays inert), HTML matrix
  rendering with hostile-name escaping, and a 4-mutant kill gate — 4/4
  killed.

### Added (cycle 134 — prompt-boundary canonicality audit, 2026-10-17)

- New `model_protocol_lab/boundary_audit.py`: the Prompt Boundary Problem
  (PBP) rate per template family (research backlog R2). For every stored
  rendered prompt x greedy completion, `encode(prompt_text +
  completion_text)` must equal `prompt_ids + completion_ids`; the module
  reports the first divergence index per observation and aggregates to
  overall, per-candidate, and per-family rates. Pure and deterministic
  (`encode`/`decode` injected); backend failures degrade a check to
  "unusable", never a crash.
- `core.Observation` now stores `input_text` (the exact rendered prompt
  string that produced `input_ids`) — decoding `input_ids` is not a
  substitute, since re-decoding is itself canonicalizing. Threaded through
  every `evaluate()` record path; defaults to "" on pre-render failures.
- `model-protocol-lab run` attaches `boundary_audit` (post-hoc, one extra
  encode per observation); the HTML report renders a "Prompt-boundary
  canonicality" section with a per-family table (a zero rate is presented
  as a certificate); `model-protocol-lab diff-reports` compares overall
  and per-family rates (`boundary_audit_changed` /
  `boundary_audit_asymmetric` summary flags feed `identical`); the
  committed sample report exercises the section.
- New `tests/test_cycle134_boundary_audit.py` (33 tests): the pure check
  (identity, planted seam-merge PBP, divergence index, unusable paths),
  rate aggregation, family rollup, end-to-end `input_text` recording, the
  HTML section, the diff-reports comparison, and a committed 4-mutant kill
  gate (4/4 killed; two mutants additionally verified killed by the real
  tests during development).

### Added (cycle 133 — docs-coherence audit fixes, 2026-10-17)

- Fixed every factually-wrong and stale claim found by the specialist
  docs-coherence audit of USAGE.md, README.md, VALIDATION.md, and the
  release documents (verified claim-by-claim against the cycle-132 tree): thinking/effort-variant lists in
  `docs/USAGE.md` §4 now name all 10 generation-prompt variants
  (`kolibri/thinking-off`, `nemotron35/thinking-off`, `ling/thinking-off`,
  `qwen38/thinking-off`, `nanbeige42/thinking-off`, `laguna/thinking-on`,
  `k2horizon/effort-medium`, `k2horizon/effort-low`, `hy4/no-think`,
  `minimax3/thinking-disabled`, `minimax3/thinking-enabled`); `safe_budget_floor`
  documented as JSON `null` (not "absent") at the sweep limit; the
  margin-trajectory chart, frontier chart, decision ribbon, sparkline
  table, run-configuration and backend-divergence sections added to the
  `report.html` inventory (§6); `held_out_floor` and
  `fixture_diffs_include_unchanged` added to the §12 stdout-JSON key list;
  README capabilities cover the held-out verification-budget floor and the
  frontier chart/ribbon/margin-trajectory visuals; VALIDATION.md header
  range now reads cycles 10–133.
- `cli.py` gained an `if __name__ == "__main__"` guard so
  `python -m model_protocol_lab.cli --help` prints usage (exit 0) instead
  of nothing; also eliminates the misleading dead entry path.
- Removed the nonexistent `mpl` command shorthand from all user-facing docs
  (USAGE.md, SECURITY.md, RELEASE_NOTES_0.1.0.md, diffbackends docstring) —
  the console script is `model-protocol-lab`; the debug REPL's literal
  `mpl:<family>` prompt is real and untouched.
- `docs/RELEASE_NOTES_0.1.0.md`: added a dated post-0.1.0 additions list
  (cycles 121–132) and marked the 2026-10-16 verification section as a
  historical snapshot; `docs/RELEASE_READINESS_0.1.0.md`: added a dated
  addendum noting superseded counts.
- Also fixed a real drift the audit predated: `RELEASE_NOTES_0.1.0.md`
  claimed 8 subcommands; the `verify` subcommand (cycle 131) makes 9.
- New `tests/test_cycle133_docs_coherence_audit.py` (3 tests): the
  `__main__` guard behavior and a no-`mpl`-shorthand scan of user docs.

### Added (cycle 132 — security red-team fixes + holdout contamination detector, 2026-10-17)

- New `model_protocol_lab/contamination.py`: pure search/held-out
  contamination detector (IDEAS #2). Every held-out fixture is compared
  against every search fixture over NFKC-normalized, format-char-stripped,
  lowercased, whitespace-collapsed user text plus expected answer, using
  character-level `SequenceMatcher` ratio (autojunk=False) and token-3-gram
  Jaccard. Tiers are advisory (warnings, never failures): `duplicate`
  (ratio >= 0.98), `near_duplicate` (>= 0.90), `paraphrase_suspect`
  (>= 0.80 or Jaccard >= 0.85). Deterministic ordering; fail-closed on
  empty/hostile inputs. Wired into `mpl inspect` (stderr warning + JSON
  `contamination` block), `mpl fixtures build` (same), `doctor --dataset`
  (`examine_dataset`), and `mpl run` (`results.json` + a "Fixture hygiene"
  line in the report's run-configuration section). New
  `tests/test_cycle132_contamination.py` (31 tests); 2/2 targeted mutants
  killed.
- Security red-team findings fixed (all reproduced against the current
  tree before fixing; `agents_out/security/REPORT.md`):
  - `render_html` no longer crashes on hostile frontier/rate payloads:
    mixed-type candidate keys are skipped before `sorted()`, non-list
    `fixture_ids` count as zero, non-list `min_budget_per_fixture` as
    empty, malformed winner-curve points skipped, and non-finite/bool/str
    pass rates degrade to the "incomplete" badge (also fixes the
    `width:nan%` cosmetic).
  - `prepare_report_dir` rejects a symlinked leaf outright
    (`FileExistsError`), even with `_allow_empty_existing` -- report files
    are never written through a link.
  - `_render_with_memory_cap` no longer swallows a watchdog alarm landing
    mid-`RLIMIT_AS`-restore: the budget exception is caught explicitly,
    the restore retried (one-shot alarm is consumed), then re-raised so
    the caller degrades the pair to "unavailable".
  - `_backend_failure_message` uses an explicit allowlist of the 7 fixed
    `local_*` codes (pinned against `backends.py` raise sites) instead of
    a prefix match.
  - `render_portability_table` control-strips every interpolated field at
    the stderr render boundary.
  - Cycle-58 escaping audit hardened: the kitchen-sink fixture now renders
    the frontier chart SVG, margin SVG, decision ribbon, and winner curve
    with hostile data; `_MarkupAudit` allowlists the measured
    g/path/line/text/circle attribute surface (incl. the fixed-literal
    `stroke-dasharray`); the fill-`url()` test now audits a real chart
    render instead of hand-written markup.
- Docs: `docs/USAGE.md` documents the contamination check (§1, §9), the
  run-report hygiene line (§6), and the debug-REPL forged-status-line
  caveat (§10); `docs/SECURITY.md` records the accepted trade-offs for
  invisible format characters in listings and bidi controls in HTML;
  `save_report`'s docstring no longer claims kill-9 atomicity (O_EXCL +
  unlink-on-Python-level-failure, precisely).
- `tests/test_cycle132_security_findings.py` (29 tests) pins every fix.

### Added (cycle 131 — independent results-integrity replay + hostile-payload robustness, 2026-10-17)

- New `mpl verify` subcommand: independently re-derives an experiment's
  recorded decision, winner, and post-hoc diagnostics from `results.json`
  through independent code paths (N-version cross-check). The selection
  rule is re-implemented as `verify.oracle_select_winner` (Hypothesis
  cross-tested against `experiments._select_winner`); the token-budget
  frontier and held-out budget floor are recomputed from the recorded
  observations and compared for exact equality; the search/held-out
  split is checked for fixture overlap. Every check reports
  passed/failed/unavailable -- cancelled runs and observation-stripped
  artifacts report "unavailable" instead of pretending. Exit 0/2/1.
  New `tests/test_cycle131_verify_audit.py` (62 tests); 4/4 targeted
  mutants killed; the new module is at 100% line coverage.
- Hostile-payload hardening for the frontier/diff surface (perf findings,
  both reproduced before fixing): a shared `MAX_FRONTIER_STEP_POINTS`
  cap (100k) in `_frontier_series`/`_valid_frontier` -- oversized
  frontiers degrade to unchartable/one-sided instead of hanging the
  comparison for hours; O(log n) step lookups (`_count_at_budget` key
  hoisting, `_step_function` bisect); budget-only sort plus
  first-occurrence dedup of duplicate `decision_steps` budgets (a foreign
  payload with mixed None/str winners crashed `diff_frontiers` with an
  uncaught TypeError). 6 regression tests.
- Docs: fixed 5 wrong/stale items from the coherence audit (retired
  residual risks #3/#4 in RELEASE_READINESS, RELEASE_NOTES counts,
  `local_repairs` reference and the incomplete thinking-variant list in
  USAGE.md §4 and the `--boundary-map` help); removed the stale `build/`
  directory.

### Added (cycle 130 — held-out budget floor, 2026-10-17)

- New `experiments.held_out_budget_floor`: post-hoc token-budget floor
  for the held-out verification stage -- the first budget from which
  the verification verdict holds unchanged (the maximum of the
  per-fixture minimal passing budgets; None when the held-out failed
  natively), plus `budget_robust`. The held-out verdict is monotone in
  the budget, so this single number is the whole story. `mpl run`
  attaches it as `held_out_budget_floor`, the HTML held-out section
  calls it out, and `mpl diff-reports` compares it between runs
  (`held_out_floor` section feeding `identical`, with summary lines).
  This cycle also extracts the shared `_observation_min_budget` helper
  used by both `budget_frontier` and the new function (no behavior
  change -- the oracle suites prove it).
- New `tests/test_cycle130_held_out_floor_audit.py` (35 tests): exact
  derivation and validation, the shared helper, end-to-end agreement on
  a real run, the HTML callout, and the diff-reports wiring; 4/4
  targeted mutants killed.

### Added (cycle 129 — selection-rule consistency audit, 2026-10-17)

- Proved and pinned: the frontier's post-hoc `decision_at` is exactly the
  pipeline's selection rule (`_select_winner` over full-pass candidates)
  within the completed-decision contract. The `budget_frontier`
  docstring now documents the consistency theorem and its three
  load-bearing invariants (the `exact_match` construction gate, the
  `aggregate_eligible` full-coverage invariant, and min-budget range).
- New `tests/test_cycle129_selection_consistency_audit.py` (10 tests):
  end-to-end agreement of the post-hoc decision at max budget with the
  recorded native decision across winner / no-candidate / ambiguous /
  held-out-failed real runs; the construction invariant; min-budget
  range; the foreign-payload status gate (fail-closed); decision-step
  coverage of the sweep maximum; and the sharp empty-expected edge case
  where the status gate is the only thing keeping a failed observation
  from counting as a pass. 2/2 targeted mutants killed.

### Added (cycle 128 — flip-attribution comparison in diff-reports, 2026-10-17)

- `mpl diff-reports` now compares `flip_attribution` per side via
  `diff_frontiers`: two runs can share identical flip spans while the
  *drivers* of the flip differ (e.g. a template tweak moves the flip
  from one fixture to another), and the diff no longer calls that
  identical. Each side is normalized (malformed entries skipped,
  crossings sorted, so crossing order never counts), a side without the
  field compares as unknown, and differences feed the `changed` flag
  and the `identical` verdict. The stderr summary prints per-span
  old/new crossing lines with terminal-control characters stripped.
- New `tests/test_cycle128_attribution_diff_audit.py` (16 tests):
  normalization, identical/same-span/different-crossings, one-sided and
  two-sided missing fields, malformed entries, order-insensitivity,
  `identical` wiring, summary wording and sanitization, determinism;
  4/4 targeted mutants killed.

### Added (cycle 127 — flip-span attribution, 2026-10-17)

- The `budget_frontier` payload now records `flip_attribution`: for each
  flip span, the (candidate, fixture) observations whose minimal passing
  budget falls inside the span, sorted by (min_budget, candidate,
  fixture). Pass counts can only change at those marks, so these are
  exactly the pass-status changes the span brackets -- the observations
  to inspect when asking why the decision flips there. A span with no
  marks inside it records an empty crossing list (the decision already
  differed at its start, e.g. a cold-start region).
- Proved and documented: the per-budget full-passer set only grows with
  the budget, so once the native decision is reached it can never be left
  again -- `flip_ranges` is always `[]` or a single prefix span `[[1, K]]`
  (2000-trial brute-force check, zero counterexamples; pinned by a
  property test). The HTML frontier section renders each span's
  crossings as an escaped attribution line under the span.
- New `tests/test_cycle127_flip_attribution_audit.py` (18 tests):
  hand-computed derivations, independent-oracle agreement on randomized
  inputs, the prefix-span theorem as a property test, HTML wording /
  hostile-name escaping / missing-or-malformed attribution tolerance /
  determinism; 4/4 targeted mutants killed (unbounded attribution,
  exclusive upper bound, dropped sort, unescaped rendering).

### Added (cycle 126 — safe budget floor readout, 2026-10-17)

- The `budget_frontier` payload now records `safe_budget_floor`: the
  first token budget from which the native selection decision never
  flips again (1 when nothing flips; derived exactly from the merged
  `flip_ranges`, which are exact by construction). The new pure
  `reporting.safe_budget_floor` re-derives it defensively (re-sorted,
  re-merged, clipped) for payloads recorded before the field existed; a
  well-typed recorded field wins.
- The HTML frontier section shows the floor as a first-class callout
  ("safe floor ≥ K tokens", or an explicit no-floor warning), and both
  the frontier chart and the margin trajectory chart draw a dashed amber
  floor marker (legend swatch + aria-label, deterministic integer
  geometry, no scripts).
- `mpl diff-reports` compares the floor per side
  (`frontier.safe_budget_floor`: old/new/changed), feeds the
  `identical` verdict, and prints a `Safe budget floor: 6 -> (none)`
  stderr line.
- New `tests/test_cycle126_safe_budget_floor_audit.py` (30 tests):
  exact derivation table, span re-validation, recorded-field
  precedence, recorded-vs-derived agreement on real
  `experiments.budget_frontier` payloads, HTML callout wording, chart
  markers + determinism, and diff-reports wiring; 4/4 targeted mutants
  killed.

### Added (cycle 125 — token-budget frontier comparison in diff-reports, 2026-10-17)

- `mpl diff-reports` now compares the runs' `budget_frontier` payloads
  via the new pure `diffreports.diff_frontiers`: sweep-range drift,
  budget-robustness flips, and the exact merged budget spans where the
  post-hoc decision, post-hoc winner, or per-budget leader (reusing
  `reporting.margin_trajectory`) differ. One-sided or missing frontiers
  are reported plainly instead of compared, mirroring the tool's
  evidence philosophy; malformed decision steps skip that axis rather
  than inventing differences.
- The diff dict gains a `frontier` section, the summary gains
  `frontier_changed` / `frontier_asymmetric` flags (both break
  `identical`, like the evidence axis), and the stderr summary renders a
  frontier block with sanitized span lines ("budgets 3–4: alpha ->
  beta").
- New `tests/test_cycle125_frontier_diff_audit.py` (16 tests):
  hand-derived leader/decision/winner spans, sweep drift with common
  prefix, robustness flips, availability variants, determinism,
  integration with `identical`, summary rendering, terminal-escape
  sanitization, and a committed 4-mutant kill gate.

### Added (cycle 124 — selection margin trajectory, 2026-10-17)

- New pure `reporting.margin_trajectory(frontier)`: per-budget leader
  and lead margin (leader pass count minus best runner-up; 0 on a tied
  lead) derived post-hoc from the recorded `pass_steps`, answering "how
  decisive was the field at each token budget?" — complementing
  `selection_margin` (a native-budget scalar) and the decision ribbon
  (which outcome, not by how much).
- The HTML report's token-budget frontier section now renders a
  "Margin trajectory" chart directly under the frontier chart: a
  deterministic inline-SVG step curve on the same x-scale, each segment
  stroked in the leading candidate's frontier-chart color (gray for tied
  leads), with an area wash, emphasized zero baseline, and per-span
  leader/margin coverage in the `aria-label`. No scripts, no external
  assets.
- New `tests/test_cycle124_margin_trajectory_audit.py` (17 tests):
  hand-derived decision-table semantics, hand-derived segment geometry,
  shared-palette ordering, escaping, determinism, section wiring, and a
  committed 4-mutant kill gate.

### Added (cycle 123 — sample report exercises the frontier, 2026-10-17)

- `examples/make_sample_report.py` now attaches the token-budget
  frontier exactly as `mpl run` does, so the committed golden sample
  report demonstrates the frontier chart and decision ribbon (5
  candidates, decision flip at budget 6 in the demo). The
  `test_json_report_schema_complete` audit's pinned top-level key list
  now includes `budget_frontier`, locking the coverage in.

### Added (cycle 122 — decision ribbon on the frontier chart, 2026-10-17)

- The token-budget frontier chart now carries a decision ribbon: a
  horizontal band under the x-axis coloring every budget span by the
  post-hoc selection outcome (selected / ambiguous / no candidate),
  drawn from the recorded `decision_steps`. Same integer-arithmetic
  determinism as the curves, distinct darker shades that never collide
  with the candidate palette, decision swatches first in the legend in
  fixed order, and per-span coverage in the SVG `aria-label`.
- New `tests/test_cycle122_decision_ribbon_audit.py` (13 tests):
  hand-derived ribbon geometry (including an ambiguous span), tiling
  proof (spans cover the full width with no gaps/overlaps), legend and
  aria-label checks, missing/malformed `decision_steps` robustness, and
  a 3-mutant mutation gate (control passes).

### Added (cycle 121 — token-budget frontier chart, 2026-10-17)

- New multi-series token-budget frontier chart in the HTML report
  (`model_protocol_lab.reporting._frontier_chart_svg`): one exact step
  curve per candidate (pass count over token budgets 1..max), with axes,
  integer tick labels, gridlines, a wrapped legend, and the native winner
  drawn last with a heavier stroke. Built from the already-recorded
  `pass_steps` data (post-hoc, no extra model calls). Deterministic
  inline SVG: integer-only geometry, fixed palette assigned in sorted
  candidate-name order, all dynamic text HTML-escaped, no scripts or
  external assets — the report's self-contained constraint holds. The
  chart renders between the winner's pass-curve table and the per-candidate
  sparkline table in the "Token-budget frontier" section.
- New `tests/test_cycle121_frontier_chart_audit.py` (20 tests): hand-derived
  SVG geometry literals, byte-determinism across processes/hash seeds and
  insertion orders, degenerate-input robustness, input non-mutation,
  hostile-name escaping, and a 5-mutant mutation gate (control passes).

### Added (cycle 120 — selection-rule extraction + property tests, 2026-10-17)

- Extracted the pure search-phase selection decision from
  `run_experiment` into `model_protocol_lab.experiments._select_winner`
  (returns the decision string plus the selected candidate or `None` on
  abstention). The rule is unchanged -- exactly one candidate passing
  every search fixture wins; ties and failures abstain; a cancelled run
  reports "cancelled" -- but it is now reachable without a backend, so
  `tests/test_cycle120_selection_rule_properties.py` exhaustively pins
  the decision table with Hypothesis: decision vocabulary, cancel
  dominance, unique-winner selection (with object identity), all-fail and
  tie abstention, order invariance, and a count-parameterized decision
  table. Mutation testing: 4/4 mutants killed (tie-selects, swapped
  abstention labels, dropped cancel branch, always-select). (7 new
  tests.)

### Added (cycle 119 — pickle-deserialization loading invariant, 2026-10-17)

- Extended the cycle-117 local-loading AST invariant to the
  pickle-deserialization attack class (CVE-2025-32434, CVE-2026-24747):
  `tests/test_cycle119_pickle_deserialization_invariant_audit.py` asserts no
  `torch.load`/`pickle.load(s)` in the package and that every model-weight
  `from_pretrained` call passes `use_safetensors=True` (transformers never
  touches the pickle unpickler) and `weights_only=True` (defense in depth).
  The installed torch 2.14.1+cpu is not affected by either CVE, but the
  lab's defense no longer depends on the installed version.
  `docs/SECURITY.md` supply-chain posture updated. (4 new tests.)

### Fixed (cycle 118 — render watchdog hardening, 2026-10-17)

- Red-team audit of the cycle-116 SIGALRM render watchdog found two
  robustness gaps in `model_protocol_lab/doctor.py::_render_with_time_budget`,
  both fixed: (1) a non-finite `budget_s` (`float("inf")`) reached
  `setitimer`, which raises `OverflowError` -- not covered by the
  `except (OSError, ValueError)` fallback -- violating the defensive
  "fall back to the unbounded render" contract; non-finite budgets now
  disable the watchdog like non-positive ones; (2) teardown did not block
  `SIGALRM` while disarming the timer and restoring the previous handler,
  so an alarm pending in that window could fire mid-restore and -- because
  `_RenderTimeBudgetExceeded` is a `ValueError` -- be swallowed by the
  cleanup guards, leaving the handler inconsistent; teardown now blocks
  `SIGALRM` via `pthread_sigmask` across the restore. Regression tests in
  `tests/test_cycle118_render_watchdog_hardening_audit.py` (4 tests).

### Security (cycle 117 — local-only loading invariant, 2026-10-17)

- Pinned the local-only model loading invariant with AST regression tests
  (`tests/test_cycle117_local_loading_invariant_audit.py`, 4 tests): no
  `trust_remote_code=True` anywhere in the package, every `from_pretrained`
  call passes `local_files_only=True`, no `snapshot_download` /
  `hf_hub_download` / `huggingface_hub` imports, no truthy `token=`. This is
  the structural defense against the CVE-2026-80047 class (transformers
  4.49.0–5.8.1 writing remote repository code to the local module cache
  before the trust-consent check): the lab never loads remote repositories
  and never enables `trust_remote_code`, so the vulnerable code path is
  unreachable. Documented in `docs/SECURITY.md`.
- Retired SECURITY.md residual risk #3: cycle 115's single-pass weight
  manifesting (hash once + stat-fingerprint verify) eliminated the double
  full-hash of weight files.

### Security (cycle 116 — CPU budget for Jinja renders, 2026-10-17)

- Closed the remaining hostile-template CPU-burn residual: a checkpoint's
  Jinja `chat_template` could burn CPU without allocating (e.g.
  `{% for i in range(10**12) %}{% set x = i %}{% endfor %}`), which the
  RLIMIT_AS memory cap cannot see. Each render in `template_conformance`
  now runs under a 10 s SIGALRM wall-clock budget
  (`_render_with_time_budget` in `doctor.py`, matching the `template_diff`
  fragmentation budget); expiry raises `_RenderTimeBudgetExceeded`
  (a `ValueError`, not `KeyboardInterrupt`) so the per-pair handler degrades
  the pair to `"unavailable"` exactly like the memory-bomb path. The
  previous SIGALRM handler/timer are saved and restored; off the main
  thread or on platforms without `setitimer` the render runs unbounded
  (documented pre-existing behavior). Verified against a real Jinja
  CPU-burn loop: aborts at the budget. New
  `tests/test_cycle116_render_cpu_budget_audit.py` (8 tests).

### Performance (cycle 115 — single-pass weight manifesting, 2026-10-17)

- `LocalHFBackend` no longer sha256-hashes every weight file twice per run.
  It hashes once (the sha256 record kept in `identity["assets"]`) and then
  re-verifies the assets are unchanged after transformers loads the model
  with the cheap `(st_size, st_mtime_ns, st_ino)` stat fingerprint captured
  while hashing (`_manifest_with_fingerprints` /
  `_verify_manifest_fingerprints` in `backends.py`). This closes residual
  risk #3 from the release-readiness report (double full read of
  multi-gigabyte checkpoints). The fail-closed contract is unchanged:
  replacement (new inode), in-place modification (new mtime/size), and
  deletion all raise `ValueError("assets_changed")` / fail backend init.
  Trade-off documented in code: a content swap preserving all three stat
  fields (nanosecond mtime forgery inside the load window) would need the
  full re-hash to detect; the stat triple catches every realistic change.
  New `tests/test_cycle115_manifest_fingerprint_audit.py` (9 tests).

### Security (cycle 114 — bidi-control sanitization, 2026-10-17)

- Closed a terminal-spoofing gap in all four operator-visible sanitizers
  (`cli._display_id`, `debug._sanitize_display`, `diffreports._display_id`,
  `fixtures._safe`): they stripped C0/C1 controls and lone surrogates but
  left Unicode bidi controls (U+200E/U+200F, U+2028/U+2029, U+202A–U+202E,
  U+2066–U+2069) intact, so a hostile fixture ID like `"safe-\u202e-evil"`
  rendered right-to-left in the operator's terminal (Trojan Source spoofing,
  CVE-2021-42574). All four now strip the bidi-control set. Other invisible
  format characters (BOM, zero-width space, soft hyphen, ...) are deliberately
  preserved: they cannot reorder displayed text, and the BOM is
  diagnostically valuable (a BOM-prefixed CSV header rejection shows the
  operator exactly why it failed — asserted by
  `test_bom_prefixed_header_rejected`; stripping it would display a
  seemingly-valid header). New
  `tests/test_cycle114_bidi_sanitization_audit.py` (14 tests) covers the bidi
  battery, parity across all four sanitizers, preservation guarantees, and an
  end-to-end REPL listing check. Test counts: pytest 4132 → 4146 collected,
  unittest 4132 → 4146; README/VALIDATION.md updated.

### Changed (cycle 113 — test_cycle66 conversion diagnostic re-attempt, 2026-10-17)

- Re-attempted the `tests/test_cycle66_fixture_loading_audit.py` pytest-to-
  `unittest.TestCase` conversion (reverted in cycle 108 after a CI-only
  3.12/3.13 failure with no diagnosable output) as a controlled diagnostic
  experiment: cycle 112's CI failure-artifact upload now captures the exact
  failing test output if CI fails again, so the root cause can finally be
  identified instead of guessed. The `_UNITTEST_EXEMPT` entry for this file
  was lifted for the experiment. Both runners now agree exactly: pytest
  4141 → 4132 collected (parametrize cases consolidate into subTests),
  unittest 4067 → 4132. README/VALIDATION.md counts updated.

### Added (cycle 112 — CI failure diagnosability, 2026-10-17)

- The CI `test` job now captures the full `unittest discover` output to a file
  and uploads it as a build artifact when the test step fails
  (`actions/upload-artifact@v4`, pinned to SHA, least-privilege `permissions`
  on the job). CI job logs are not API-accessible, which previously made
  CI-only failures (e.g. the cycle-107 test_cycle66 3.12/3.13 failure)
  undiagnosable; the artifact closes that gap permanently for all future
  failures. `pipefail` preserves unittest's exit status, so the step still
  fails exactly when tests fail.

### Added (cycle 111 — end-to-end determinism audit, 2026-10-17)

- New `tests/test_cycle111_determinism_audit.py`: runs the full `run` pipeline
  twice in separate subprocesses with different `PYTHONHASHSEED` values and
  asserts `results.json` and `report.html` are byte-identical across runs.
  Any hash-order-dependent serialization (e.g. set iteration leaking into
  output) would flip bytes and fail the test. The audit verified determinism
  holds for the base path and for `--mutate` + `--sensitivity-sweep`. Pytest
  4140 → 4141; unittest 4066 → 4067. README/VALIDATION.md counts updated.

### Added (cycle 109 — Hypothesis properties for Split anti-leakage, 2026-10-17)

- New `tests/test_cycle109_split_properties.py`: 5 Hypothesis property tests
  verifying the universal anti-leakage guarantees of `model_protocol_lab.core.Split`
  (the contamination firewall). For arbitrary NFC/NFD string pairs: byte-different
  but canonically-equal fixture IDs are always rejected as duplicates; NFC-equal
  groups across the split are always rejected; NFC-equal message content across
  the split is always rejected; valid Splits preserve byte-exact storage (NFD
  stays NFD — only comparisons normalize); and distinct IDs/groups/messages
  always construct successfully. Mutation-tested: removing NFC normalization
  from any of the three checks is caught by exactly the corresponding property
  test. Pytest 4129 → 4134; unittest 4055 → 4060. README/VALIDATION.md counts
  updated.

### Fixed (cycle 108 — redo TestCase conversions one file at a time, 2026-10-17)

- Added the TestDualRunnerContribution meta-tests to
  test_ci_portability_guard.py (5th of the cycle-107 conversions, with
  hardening): test_every_test_file_collected_by_unittest now documents a
  _UNITTEST_EXEMPT set (test_cycle66 excluded pending its conversion
  root-cause investigation); test_every_test_file_collected_by_pytest
  retries the subprocess collection up to 3 times with 300s timeout and
  reports diagnostics instead of failing opaquely. Pytest 4127 → 4129;
  unittest 4053 → 4055. README/VALIDATION.md counts updated.
- Converted test_cycle69_diagnostics_audit.py to `unittest.TestCase`
  (fourth of the cycle-107 conversions, redone individually). Pytest
  collection 4136 → 4127; unittest discovery 4015 → 4053.
  README/VALIDATION.md counts updated.
- **Isolated the cycle-107 CI failure to test_cycle66_fixture_loading_audit.py**:
  converting files #1 (test_cycle29) and #2 (test_cycle30) passed CI on all
  Pythons, but file #3 (test_cycle66) reproduced the exact 3.12/3.13 failure
  (3.11 green). Reverted test_cycle66 to the cycle-106 pytest-style version
  to restore green CI. The converted version passes locally on 3.12 under
  both runners; the failure is a CI-environment × 3.12/3.13 interaction
  still under investigation. Files #1-2 remain converted and green.
- Converted test_cycle66_fixture_loading_audit.py to `unittest.TestCase`
  (third of the cycle-107 conversions, redone individually). Pytest
  collection 4136 → 4127; unittest discovery 4015 → 4080.
  README/VALIDATION.md counts updated. [REVERTED — see above]
- Converted test_cycle30_datasets_verify.py to `unittest.TestCase`
  (second of the cycle-107 conversions, redone individually). Pytest
  collection 4854 → 4136; unittest discovery 3953 → 4015.
  README/VALIDATION.md counts updated.
- Converted test_cycle29_fixtures_verify.py to `unittest.TestCase`
  (first of the cycle-107 conversions, redone individually with CI
  verification per file after the cycle-107 CI red). `@pytest.mark.parametrize`
  became `subTest` loops (cases preserved). Pytest collection 4913 → 4854;
  unittest discovery 3873 → 3953. README/VALIDATION.md counts updated.
- Hardened test_readme_test_count_matches_discovered_suite
  (test_cycle4.py): the pytest-subprocess collection now retries up to 3
  times with a 300s timeout (was single attempt, 180s) and reports both
  counts plus collection diagnostics on failure, instead of failing
  opaquely under load.

### Fixed (CI red revert — cycle 107 conversions rolled back, 2026-10-17)

- Reverted cycle 107's pytest-to-TestCase conversions (test_cycle29,
  test_cycle30, test_cycle66, test_cycle69) and the new
  TestDualRunnerContribution meta-tests: the converted tree failed CI's
  unittest gate on Python 3.12/3.13 (3.11 passed) with an undiagnosed
  root cause that does not reproduce locally. Main is restored to the
  cycle-106 green state (4913 pytest / 3873 unittest). The conversions
  will be redone carefully with CI verification per file.

### Fixed (cycle 107 — continued pytest-to-TestCase conversions, 2026-10-17)

- Converted 4 more pytest-style files to `unittest.TestCase`
  (test_cycle29, test_cycle30, test_cycle66, test_cycle69):
  `@pytest.mark.parametrize` cases became `subTest` loops (cases
  preserved; pytest's per-param count consolidated). Removed a leftover
  `import pytest` / `pytest.approx` from test_cycle69 (now
  `assertAlmostEqual`). Both runners now agree at 4120 tests. README and
  VALIDATION.md counts updated (4913/3873 → 4120/4120).

### Fixed (cycle 106 — CI unittest gate coverage, 2026-10-17)

- Closed a major test-runner gap: CI's `coverage run -m unittest discover`
  silently skipped 2533 test executions because 9 test files were written
  in pytest style (module-level `def test_*` functions, pytest-style
  classes) that unittest discovery collects zero tests from. The 9 files
  (cycles 13, 49, 55, 62, 63, 74, 82, 86, 88) are now `unittest.TestCase`
  subclasses, which pytest also runs natively — both runners now execute
  the full suite (unittest: 3598 -> 3873 tests; pytest: 6131 -> 4913
  collected items, with parametrize cases as subtests). This included
  security-relevant suites: hostile parser properties, held-out
  separation, diff-reports security, and doctor portability. README and
  VALIDATION.md test counts updated.

### Added (cycle 104 — mypy strict compliance, 2026-10-16)

- The full package now passes `mypy --strict` (37 errors fixed across 7
  files): type annotations added to the untyped dataset helpers
  (`_unique_object`, `_reject_constant`, `_fields`, `_text`, `_fixture`),
  `experiments._failure_code` / `_summary` (now typed on
  `EvaluationReport`), `budget_frontier` (corrected to
  `list[dict[str, Any]]` — entries are search-record dicts, not Fixtures;
  the wrong first attempt tripped `comparison-overlap`), bare `dict` /
  `list` / `tuple` generics given type arguments, `cli.main` argv typed,
  and the five `PreTrainedTokenizerFast` call sites (which mypy resolves
  to the untyped `TokenizersBackend` factory) marked with deliberate
  `type: ignore[no-untyped-call]` at the documented untyped
  third-party boundary. `strict = true` is now set in pyproject.toml so
  the CI lint job enforces it.
- `TestMypyStrictPinned` in `tests/test_ci_portability_guard.py` fails
  if `strict` is ever flipped back to false. Test count 6122 -> 6123.
- Follow-up fix (same cycle): the first attempt used
  `type: ignore[no-untyped-call]` on the five `PreTrainedTokenizerFast`
  call sites, which broke CI's lint job — it runs without the `hf`
  extra, so the import becomes `Any` via `ignore_missing_imports` and
  strict's `warn_unused_ignores` flags the then-unused ignore. Replaced
  with a typed `Callable[..., Any]` binding, which is clean in both
  environments (verified with `--no-site-packages`).

### Added (cycle 103 — public API docstring completion, 2026-10-16)

- All 16 public classes/functions in the shipped package that lacked
  docstrings now have them: the seven core data-model dataclasses
  (`Message`, `Format`, `Fixture`, `Split`, `Generation`, `Observation`,
  `EvaluationReport`), `catalog.candidates`, the CLI entry points
  (`parse_boundary`, `build_parser`, `main`), `datasets.parse_dataset`
  / `load_dataset`, `experiments.run_experiment`, and
  `reporting.render_html` / `save_report`. Public docstring coverage is
  now 74/74.
- `TestPublicApiDocstrings` in `tests/test_ci_portability_guard.py`:
  AST-scans the package and fails on any public top-level def/class
  without a docstring, so the coverage cannot regress. Test count
  6121 -> 6122.

### Added (cycle 102 — true OSError-branch coverage, 2026-10-16)

- `tests/test_cycle102_oserror_branches.py` (5 tests): the
  ``except OSError`` branches in ``examine_model`` are defense-in-depth
  for I/O errors *after* a successful open (disk errors during
  ``read()``) — ``_open_regular_read`` converts open-time failures to
  ``ValueError`` first, so they were uncovered. A tracked-fd
  ``os.fdopen`` replacement injects ``OSError`` at read time for a named
  file, pinning fail-closed behavior for the `_model_assets`,
  `_manifest`, config, and tokenizer branches (fixed
  ``missing_or_unsafe_asset`` vocabulary, never a raw OSError).
- Fixed misleading docstrings on the four cycle-92 "without_root"
  tests: they exercise the ``ValueError`` conversion path, not the raw
  ``except OSError`` branch — now stated accurately with a pointer to
  the new coverage. Test count 6116 -> 6121.

### Added (cycle 101 — warning-cleanliness audit, 2026-10-16)

- `tests/test_cycle101_warning_cleanliness.py` (4 tests): pins the
  verified property that the package emits zero warnings — all module
  imports, a `doctor` smoke run, a report render, and CLI startup each
  run warn-free under `python -W error` in a subprocess. Full-suite
  audit (`python -W error -m pytest`, 6105 tests) passes with every
  warning promoted to an error: no DeprecationWarning, FutureWarning,
  ResourceWarning, or any other warning fires anywhere. Test count
  6112 -> 6116.

### Added (cycle 100 — HTML report injection audit, 2026-10-16)

- `tests/test_cycle100_html_injection_audit.py` (9 tests): hostile
  payload battery (`<script>`, event handlers, `<svg>`, `javascript:`,
  title-breakout, iframe, U+2028/29) through every `render_html` path —
  no raw payload survives, escaped forms present, no raw angle brackets
  in text content, inline script has no `</script` breakout, CSP meta
  present, hostile decision maps to neutral CSS class, output parses as
  well-formed HTML. Finding: no vulnerabilities — every interpolation is
  escaped (directly or via an outer `escape()`), the only `href` uses
  fixed-literal anchors, and there are no data-driven URL attributes.
  Test count 6103 -> 6112.

### Fixed (cycle 99 — stale development plan framing, 2026-10-16)

- `DEVELOPMENT_PLAN.md` still opened by framing the lab as a "proposed"
  tool with "no runtime validation established" — contradicting the 0.1.0
  release-ready state (and the owner's no-"early development" rule). Added
  a dated status note recording substantial implementation (6100+ tests,
  green CI, release-readiness review) while preserving the design sections
  as the authoritative architecture reference. Pinned by 3 regression
  tests in `tests/test_cycle99_plan_currency.py`. Test count 6100 -> 6103.

### Added (cycle 98 — property-based core algorithm audit, 2026-10-16)

- `tests/test_cycle98_boundary_properties.py` (7 Hypothesis property
  tests): universal properties of `first_boundary` — found index is an
  exact match, None iff absent, result is the *first* occurrence, index
  in range, empty stop raises, overlong stop returns None, embedded stop
  found at or before the known position. Thousands of generated cases;
  no counterexamples. Test count 6093 -> 6100.

### Added (cycle 97 — hostile-environment robustness audit, 2026-10-16)

- `tests/test_cycle97_hostile_environment.py` (10 tests): hostile
  *environment* audit of the write paths — the red-team sweeps covered
  hostile inputs; this covers adverse filesystem/OS conditions.
  `prepare_report_dir` / `save_report` under mocked PermissionError
  (clean OSError, no partial dirs/files, half-built pair cleanup),
  concurrent directory claims (exactly one wins via O_EXCL), existing
  file/non-empty dir rejection, and `write_fixture_file` permission and
  no-clobber behavior. All write paths confirmed robust; no code changes
  needed. Test count 6083 -> 6093.

## [0.1.0] — 2026-10-16

### Added (cycles 14–15, 52, 54 — catalog to 29 families, 2026-10-15)

- Format catalog expanded 23 -> 29 families (all pinned by byte-exact
  tests in `tests/test_cycle14_*`, `tests/test_cycle15_*`,
  `tests/test_cycle52_muse_spark.py`, `tests/test_cycle54_qwen38.py`):
  `k2horizon` (cycle 14, K2-Horizon), `hy4` (Tencent Hy4 preview),
  `minimax3` (MiniMax-M3), `nanbeige42` (Nanbeige 4.2-3B; all three cycle
  15, re-verified against primary sources this round -- official
  chat_template.jinja / tokenizer_config.json preserved byte-identical
  under `docs/evidence/`), `muse_spark` (cycle 52, Meta Muse Spark Open),
  `qwen38` (cycle 54, Qwen3.8-27B, thinking-on default opening the
  generation prompt with `<think>`). The `doctor` portability matrix now
  renders 118 mutate candidates (110 -> 114 -> 118). Evidence tiers and
  out-of-scope branches are documented in the catalog comments and
  `docs/RESEARCH_LOG.md`.

### Added (cycles 45–83 — audit record, 2026-10-15 → 2026-10-16)

- Audit cycles 45–51, 55–65, 68–75, 77–83 (see
  `docs/RESEARCH_LOG.md`): differential re-verifications of
  run_experiment, docs command examples, the sample-report pipeline,
  canary evidence-leak sweeps, logprob numerics, HF stop-emulation,
  core.evaluate, the prompt-sensitivity sweep, budget frontier, HTML
  escaping, boundary portability, harness concurrency, doctor
  portability-matrix semantics, search/held-out separation, the CLI
  surface, backend seed-determinism, LocalHFBackend lifecycle, reporting
  determinism, classification/failure taxonomy, run orchestration, the
  core.evaluate engine, and the special-token auditor. Upstream drift
  re-checks (cycles 51, 61, 70): no change. Cycles with real findings are
  under Fixed; the rest are CLEAN with mutation gates run on hermetic
  /tmp copies.
- `tests/test_ci_portability_guard.py`: standing AST guard forbidding
  exact `sys.version_info` tuple comparisons and bare torch/transformers
  `__version__` string equality anywhere in the test suite (the failure
  class behind the cycles-79/80 CI red).
- Cycle-83 CLI wiring audit (`tests/test_cycle83_cli_wiring_audit.py`,
  63 tests): the exit-code contract is pinned by subprocess -- 0/1/2/130
  for compatible/failed/cancelled/ambiguous/held_out_failed/
  no_candidate run decisions, bad subcommands, missing args, invalid
  enum values, unreadable inputs, unwritable outputs, malformed fixtures,
  nonexistent checkpoints, contradictory ranges -- plus `--help`
  byte-stability across runs for every subcommand (fixed COLUMNS, no
  PID/tmpdir/timestamp leaks) and console-script equivalence with
  `python -m model_protocol_lab`.
- Cycle-82 fixture serialization round-trip audit
  (`tests/test_cycle82_fixture_serialization_audit.py`, 80 tests):
  `fixtures.split_to_json_text` round-trip fidelity over adversarial
  Splits, `write_fixture_file` exclusive creation (0o600, partial-write
  cleanup), and the fixed dataset rejection vocabulary. CLEAN, no source
  changes.

### Fixed (cycles 36–83 — real bugs and robustness, 2026-10-15 → 2026-10-16)

- **REAL BUG, CI-red root cause (cycles 79/80):** `examine_model`
  caught only `ValueError` around config/tokenizer reads, so an
  unreadable asset (`OSError`, e.g. `PermissionError` on
  `path.open("rb")`) escaped instead of failing closed. `doctor.py` now
  catches `OSError` alongside `ValueError` in all four blocks, mapping
  to the fixed `missing_or_unsafe_asset` check code. Four
  root-independent regression tests mock `Path.open` to raise
  `PermissionError` (config/tokenizer/manifest/weight-index blocks);
  negative control: all four fail on reverted source. Sibling fixes in
  the same round: the Python-version test now checks the
  `requires-python` range (not `(3, 12)`), the torch test compares the
  base version (so CI's CUDA wheel passes), the transformers test checks
  membership in the `hf` pin range (not `"5.19.0"`), and the bandit test
  skips when no `bandit` binary is on PATH (it only exists in the
  security-audit job).
- **REAL BUG (cycle 76, scoring):** `_f32` in `scoring.py` raised
  `OverflowError` for finite float64 log-probabilities outside the
  float32 range (legitimate from fp32-range logits via max-shift
  log-sum-exp), collapsing through `score_target_logprobs` to
  `unavailable`/`forward_failed` and mislabeling a computable score.
  `_f32` now saturates finite out-of-range values to +/-inf, matching
  IEEE-754 round-to-nearest-even and torch/numpy float32 conversion
  (`+_FLT_MAX` constant); the old `test_out_of_float32_range_raises` pin
  was updated to the corrected saturation behavior. Pinned by
  `tests/test_cycle76_scoring_semantics_audit.py` (27 tests + 75
  subtests; negative control: 12 failures on reverted code).
- **REAL BUG (cycle 67, diff-reports):** two winnerless runs compared
  as "winner CHANGED" because `diff_reports` kept
  `selected_candidate=""` while `_held_verdict` treats `""` as absent.
  Winner extraction now normalizes `""` to `None`
  (`diffreports.py:361`). Pinned by `tests/test_cycle67_diffreports_audit.py`
  (44 tests + 819 subtests; negative control: the pinned test + 188 fuzz
  subtests fail on reverted code).
- **REAL BUG (cycle 66, fixtures):** a single CSV field over the csv
  module's 131072-char field limit raised `csv.Error`, escaping the
  documented `ValueError` contract and giving `fixtures build` an
  unhandled traceback; read-time `OSError` after a successful open
  (e.g. `/proc/self/mem`) escaped as `OSError` to the CLI. `csv.Error`
  is now caught around the parse loop and raised as
  `ValueError("CSV field exceeds the parser limit near row N")`; reads
  wrap the read call and map `OSError` to the existing
  `"Unreadable ..."` codes. Pinned by
  `tests/test_cycle66_fixture_loading_audit.py`.
- **REAL BUG (cycle 36, debug REPL):** truncated transcripts could never
  be saved -- truncation kept `_MAX_SAVE_BYTES` of content then
  prepended the 29-char marker, pushing the total back over the bound so
  `.save` always refused. The marker is now budgeted inside the
  truncation (`debug.py`). Pinned by
  `TestTranscript.test_truncated_transcript_save_succeeds`.
- ROBUSTNESS (cycle 39, catalog): `candidates()` type-gates non-str
  family keys before the `CATALOG` lookup, so the documented
  `ValueError("Unknown format family")` holds for unhashable and non-str
  keys instead of leaking `TypeError`. Pinned by
  `tests/test_cycle39_catalog_integrity.py`.

### Fixed (cycles 84–93 — real bugs and CI robustness, 2026-10-16)

- **CI RED ROOT CAUSE, PROVEN (cycle 93):** the Python 3.13-only test-job
  failures on cycles 87, 88 and 92 were *not* the docs-coherence test --
  the CI job log names `tests/test_cycle86_serialization_properties.py`:
  its module docstring contained a real lone-surrogate character (a
  single-backslash `'\ud800'` escape in docstring prose, meant as literal
  text). CPython 3.13's `compile()` encodes `__doc__` as strict UTF-8,
  so the module fails at import time with `UnicodeEncodeError`, while
  3.11/3.12 accept it (which masked it locally). Surrogate *constants*
  in function bodies remain fine on 3.13 -- only docstrings are
  affected. Fixed by doubling the backslash so the docstring renders the
  escape as text (zero runtime change). `tests/test_ci_portability_guard.py`
  gains `TestNoDocstringSurrogates`, an AST guard that fails any test
  module whose module/class/function docstring holds a lone surrogate.
- **TEST ROBUSTNESS (cycle 94):** `test_template_diff_long_equal_prefix_stays_fast`
  asserted a 5.0s wall-clock bound calibrated for uninstrumented runs,
  but the CI gate runs under `coverage run` (measured 3.5x overhead:
  1.47s -> 5.10s for the same input), so it failed spuriously under
  coverage. Threshold raised to 15.0s with a comment documenting the
  overhead; still catches algorithmic blowup (minutes-scale).

- **REAL BUG (cycle 86, fixtures):** `split_to_json_text` raised a bare
  `UnicodeEncodeError` for lone surrogates (which this hypothesis
  version's `st.characters()` does emit) instead of the documented
  `ValueError`. Now fails closed with `ValueError("... contains a lone
  surrogate ...")`. Pinned by
  `tests/test_cycle86_serialization_properties.py`.
- **REAL BUGS (cycle 89, signal handling):** (1) `save_report` left a
  half-written `results.json` when `KeyboardInterrupt` landed mid-write;
  now unlinks on `BaseException` per file and removes the pair if the
  second write fails. (2) SIGTERM during `run` used the default
  disposition (exit -15, claimed output dir leaked); now wired to the
  same cooperative cancel `Event` as SIGINT, draining to a valid
  "cancelled" report and exit 130. Pinned by
  `tests/test_cycle89_signal_handling_audit.py`.
- **REAL BUG (cycle 90, cancellation):** `logprob_diagnostic` ignored the
  cancel `Event`, running its full forward-pass loop even with the event
  set. Now accepts an optional `cancel` event checked per fixture,
  returning `available=False` / `detail="cancelled"`. Pinned by
  `tests/test_cycle90_cancel_responsiveness_audit.py`.
- **CI ROBUSTNESS (cycle 92):** the real-model signal tests
  (`RunSignalTests`) signalled a child process on a timing guess: the
  "100%" stderr readiness marker they polled for is never printed by
  transformers for local checkpoint loads, so the signal could land
  before the child's signal handlers were installed (SIGTERM -> -15
  instead of 130 on slow CI CPUs). `cli.run` now writes a
  `MODEL_PROTOCOL_LAB_TEST_SIGNAL_READY_FILE` marker immediately after
  installing its handlers (test-only, env-gated, zero production
  impact); the tests wait for the marker deterministically. The two
  slow real-model tests are gated behind `MPL_RUN_SLOW_SIGNAL_TESTS=1`
  (skipped in the default CI gate); the deterministic in-process signal
  tests stay in the gate. Also removed import-time `os.chdir` /
  `sys.path` mutation from `test_cycle87_docs_coherence_audit.py`
  (import-order hazard during unittest discovery).
- Cycles 84, 85, 87, 88, 91: CLEAN audits (release versioning,
  README quick-start verification, docs coherence, hostile-input
  fuzzing, hostile backend-output probing) with pinning tests, no
  source changes required.

### Added

- `mpl diff-reports OLD_RESULTS NEW_RESULTS`: deterministic comparison of
  two experiment result directories (or two `results.json` paths) for the
  iterate-and-compare operator loop. Reports the decision flip, the winner
  change, per-candidate verdict flips on search and held-out, per-fixture
  outcome diffs (changed-only by default, `--all` lists every compared
  fixture), and a loud config-drift warning section when the runs used
  different candidate families, candidate sets, per-candidate boundaries,
  `--max-new-tokens`, backend fingerprints, or fixture sets. JSON goes to
  stdout, a human-readable summary to stderr; identical runs report "no
  differences". Per-fixture comparison needs `--include-evidence` on both
  runs and degrades loudly otherwise. Exit 0 on a successful comparison
  even when differences are found. Hostile inputs (oversized, malformed,
  non-regular files, duplicate JSON fields) collapse to fixed error codes
  with no tracebacks. See `docs/USAGE.md` section 12.
- Format catalog expanded to 23 families: `ling` (inclusionAI
  Ling-3.1-flash, released 2026-10-15, MIT; evidence tier 1 -- the official
  chat_template.jinja "Bailing V3", preserved at
  docs/evidence/ling31-chat-template.jinja, derived line-by-line in the
  catalog comment). The plain-text no-tools core: an always-emitted system
  block (documented, not modeled -- the lab's Format renders user/assistant
  turns only), user turns `<role>HUMAN</role>{content}<|role_end|>`,
  assistant history turns `<role>ASSISTANT</role>\n<think></think>{content}
  <|role_end|>`, and -- thinking defaults ON (the lab models the default;
  the thinking_option dependence is documented) -- a generation prompt
  `<role>ASSISTANT</role>\n<think>`. BOS `<|startoftext|>` is tier-1
  verified from the official tokenizer_config.json bos_token field; the
  template itself emits no BOS literal and add_bos_token is false. EOS
  caveat documented, not modeled: the official eos_token field is
  `<|role_end|>`, NOT `<|endoftext|>` (that token is the pad_token and a
  special added token). The thinking-off branch (generation prompt
  prefills an empty `<think></think>` block) rides the `--mutate` path as
  `ling/thinking-off` (kolibri/thinking-off convention). Tool-call
  branches, reasoning-content branches, and the multi-step tool scan are
  out of scope for the plain-text core. `tests/test_cycle13_families.py`
  pins byte-exact renders, the thinking-off mutation, the dedup-key
  no-collision, diffs-vs-all-other-families, and the README naming/count
  (22 -> 23).
- Format catalog expanded to 22 families: `nemotron35` (NVIDIA Nemotron 3.5
  Lightning, released 2026-08-11; evidence tier 2 -- official HF model card,
  NeMo SFT docs, community osaurus renderer), `muse_glimmer` (Meta Muse
  Glimmer 30B, released 2026-08-10, Apache 2.0; evidence tier 2 -- official
  quickstart, HF transformers docs, HF blog, generation prompt
  community-refined), and `laguna` (Poolside Laguna 2.1, Jul 2026,
  OpenMDW-1.1; evidence tier 2 -- official HF cards plus the renderers
  library's token-exact LagunaXS21/S21 renderers). Nemotron 3.5's ChatML
  per-turn frames render byte-identical to chatml (only the thinking-on
  generation tail `<|im_start|>assistant\n<think>\n` differs); its BOS
  behavior is UNVERIFIED/conflicting at official level, so the catalog
  models bos="" as an absence-of-evidence placeholder and asserts no BOS
  token. Its thinking-off branch (generation prompt prefills an empty
  `<think></think>` block, no trailing newlines -- unlike kolibri's tail)
  rides the `--mutate` path as `nemotron35/thinking-off`. Muse Glimmer
  frames `<|start|>{role}<|message|>{content}<|eot|>` with a
  `<|begin_of_text|>` BOS; assistant turns use recipient channels and are
  modeled as `to=user` (the `to=self` reasoning trace is not rendered); the
  system appendage (`\nReasoning strength: high.\n# Valid recipients:
  "self", "user".`), the `<|end_of_text|>` EOS, and the end-of-message
  marker are documented, not modeled. Laguna 2.1 renders hugging tags with
  no inner newlines (`<user>{content}</user>`, assistant turns closed by
  `</assistant>`); the base entry is XS-2.1 defaults (enable_thinking FALSE,
  gen prompt `<assistant></think>`), and S-2.1's thinking-on default rides
  the `--mutate` path as `laguna/thinking-on` (gen prompt
  `<assistant><think>`, history turns render an EMPTY think block verbatim);
  EOS is decode-only and never rendered. Mistral Large 4 and Reflection
  Beam are NOT shipped (not released) -- tracked as watched in
  docs/RESEARCH_LOG.md. `tests/test_cycle11_families.py` pins byte-exact
  renders, the thinking-branch mutations, diffs-vs-closest-family, the
  nemotron35 no-BOS-assertion, and the evidence tiers; README family count
  updated 19 -> 22.
- Format catalog expanded to 19 families: `kolibri` (Aleph-Alpha
  Kolibri-1, released 2026-10-03, Apache 2.0; evidence tier 1 -- official
  tokenizer_config.json chat_template byte-verified by an independent fork)
  and `lfm25` (LiquidAI LFM2.5-24B-A2B, released 2026-10-13; evidence tier
  2 -- official model card plus HF blog rendered example). Kolibri-1's
  text-only no-system core is byte-identical to chatml (the always-emitted
  system block with its effort-dependent sentence, the default-high effort
  choice, and the thinking on/off branches are documented in the catalog
  comment; only byte-verified prefixes are pinned in tests). The Kolibri-1
  thinking-off branch (generation prompt prefills an empty `<think>` block)
  rides the `--mutate` path as `kolibri/thinking-off` rather than becoming
  a second catalog entry. LFM2.5 prepends `<|startoftext|>` and always
  thinks (generation prompt appends `<think>`); its bos_token-vs-template
  and system-block caveats are documented, not asserted.
  `tests/test_cycle10_families.py` pins byte-exact renders, the mutation,
  the diffs-vs-chatml, the evidence tiers, the README family count, and the
  deepseek41 system-framing re-verification (V4.1's fullwidth System frame
  CONFIRMED against the primary source 2026-10-15; the sibling V4 encoder
  renders system raw -- the catalog claim stands, no fix needed).
- Format catalog expanded to 17 families: `gemma4` (Gemma 4 — completely
  new turn/channel protocol, `<|turn>role` / `<turn|>`, assistant rendered
  as `model`, literal `<bos>` prepend; evidence tiers documented in the
  catalog comment: response_template start anchor and special-token
  inventory from the official tokenizer_config.json, request-side framing
  corroborated by three independent byte-level implementations).
- Automatic selection-margin diagnostic (`selection_margin` in
  `results.json`, one line in the decision banner): the closed-form number
  of single-fixture outcome flips that would change the decision --
  descriptive only, never changes the selection rule.
- Automatic protocol token-overhead card (`token_overhead` in
  `results.json`, "Token overhead" HTML section): per-candidate
  mean/max prompt tokens, cheapest candidate, and the winner's token tax
  versus cheapest; counts are retained-input/output counts, not
  equal-compute accounting.
- Teacher-forced log-likelihood scoring engine
  (`model_protocol_lab/scoring.py`): `score_target_logprobs` scores an
  exact target token-ID sequence via one forward pass with causal-shifted
  gather; `check_alignment` refuses (status `unavailable`, never a silent
  guess) when a token straddles the prompt/answer boundary; NaN/inf
  rejected, float64 accumulation, deterministic. Implements
  DEVELOPMENT_PLAN.md §9.3–§9.5 (acceptance gate: hand-computed logit
  tensors cross-checked against an independent reference). Not yet wired
  into `run` -- report integration is planned next.
- `LocalHFBackend.forward_logits` (single full-sequence forward returning
  fp32 CPU logits) and a `vocab_size` property, the missing primitives the
  scoring engine needs (`generate` only exposed last-position logits).
- Opt-in teacher-forced log-likelihood diagnostic (`run
  --logprob-diagnostic`, the DEVELOPMENT_PLAN.md §9.3–§9.5 "report wiring
  planned next"): after search, each candidate's expected answer is scored
  against the recorded prompts via `experiments.logprob_diagnostic`;
  `results.json` carries `logprob_diagnostic` and the HTML report renders a
  "Log-likelihood diagnostic" section with per-candidate total/mean
  logprob and a plain-language reading separating "greedy derailed" from
  "weights don't know" (declared bands: mean >= -2.0 / < -5.0 nats/token).
  Alignment refusals surface as `unavailable` with a fixed reason, never a
  silent guess; diagnostic only, the selection rule is untouched.
- Static template↔tokenizer portability matrix (`doctor
  --portability-matrix SECOND_CHECKPOINT`, research basis
  `docs/RESEARCH_PORTABILITY_2026-10-15.md`): for every catalog candidate
  (base formats plus bounded local repairs), each special-token inventory
  literal is checked against the second checkpoint's tokenizer --
  registered in added/special tokens, encodes as exactly one token id,
  encoded id == registered id -- via the new
  `core.template_tokenizer_compatibility` (a generalization of
  `core.check_boundary_portability`). Tokenizer only, never weights; same
  128 MB cap and fixed-vocabulary error codes as the other doctor checks.
  **Asymmetry is the design:** `incompatible` means the format cannot be
  that checkpoint's native format (answered at ~zero compute);
  `compatible` predicts NOTHING about experimental success -- the report
  section, the CLI table (stderr; stdout stays parseable JSON), and
  `docs/USAGE.md` §2 all carry the warning loudly, and no portability
  score is ever emitted. BOS-less formats skip the BOS requirement
  (`bos_skipped` per row); empty inventories and encode failures report
  `unknown`, never `compatible`; hostile second checkpoints degrade to
  `unavailable` without failing the doctor verdict. Pinned by
  `tests/test_cycle12_portability.py` (38 tests).

### Added

- Selection-rule veto pins (`tests/test_cycle8_selection.py`): a failed
  search evaluation in any candidate vetoes selection even when an
  earlier-listed candidate would be the unique winner; a held-out failure
  still names the unique search winner while the decision records where
  it failed.

### Added

- Operator docs for the two new commands: README capabilities now describe
  all 16 format families plus `mpl debug` and `mpl diff-backends`;
  `docs/USAGE.md` gained §10 (`mpl debug`: every dot-command, flags, a
  real session transcript) and §11 (`mpl diff-backends`: flags, the 5
  divergence classes, JSON reading guide, scope limits) — every flag
  verified against live `--help`.
- Hypothesis property tests for the divergence classifier
  (`tests/test_cycle7_properties.py`, derandomized): verdict ⟺
  divergences-empty, divergence well-formedness, severity totality,
  section-aggregation invariants, input-length bound, empty-input safety.

### Added

- Report "Backend divergence" section: the `run` flow now classifies the
  winner's recorded generations with the diff-backends divergence taxonomy
  and renders a verdict badge, a divergence-class × semantics matrix with
  severity badges, and `<details>` callout boxes per divergence
  (presentation: infer-check/Csmith style). Degrades to absent when no
  usable observations exist; the committed sample report is byte-identical.
- Format catalog expanded to 16 families: `mistral_large`
  (Mistral-Large-Instruct-2411 V7, re-verified against the live model card)
  and `glm53` (GLM-5.3-Flash with the confirmed whitespace deltas; the
  auto-injected system line is deliberately omitted like deepseek41's
  preamble).

### Fixed

- `divergence_section_data`'s `observations_processed` now counts the
  observations actually emulated (`len(per_fixture)`) instead of the number
  offered after truncation (`len(candidates)`) -- with skipped
  non-dict/invalid observations the old code overcounted (verified:
  total=4, processed=4, per_fixture=1). Verdicts were never affected (they
  derive from `per_fixture`); this is an informational-contract fix, and the
  docstring now states the contract exactly. Pinned by a new
  `tests/test_cycle11_bounds.py` regression test (junk mixed in -> 4
  handed in, 1 emulated, processed==1, verdict "agree").
- One-line comment at the `DivergenceClass` definition documenting why
  `HARNESS_NORMALIZATION_RELIED`'s SGLang column is vacuous by
  construction (`emulate_sglang_stop` trims at the leftmost stop-string
  occurrence, so its returned text can never contain the stop string) --
  so a future reader does not "fix" it into a false positive.
- Narrowed the exception scope around the new divergence wiring so a future
  bug there degrades only that section, not the whole diagnosis block
  (red-team suggestion).
- Red-team H1: `diff_backends` token-id cap lowered from 131072 to the
  named `MAX_OUTPUT_IDS = 4096` (prompts keep the named
  `MAX_PROMPT_IDS = 131072`, linear decode cost). The emulations do one
  `backend.decode(prefix)` per output step -- quadratic -- so the old cap
  burned tens of minutes per call on the linear reference stub; at the new
  cap a single stop completes in ~2-3 s (measured 2026-10-15). 4096 also
  matches `mpl run --max-new-tokens`' own cap, so the lab can never record
  a longer generated output. Over-cap input is rejected with the fixed
  `output_ids_too_long` code; over-cap recorded observations are skipped.
- Red-team M1: stop texts are now bounded -- 1024 chars per stop
  (`stop_text_too_long`), 16 stops per call (`too_many_stop_texts`) --
  enforced in `_checked_stop_texts`, the `diff-backends` CLI path, and the
  `run` diagnosis path (which passes `backend.decode(winner_boundary)`
  through the same validation). Dedup is now set-based (linear, not
  O(m^2)) and runs before the count check.
- Red-team L1: tokenizer-loader failures in `mpl debug` and
  `mpl diff-backends` collapse to the fixed `tokenizer_load_failed` code
  via an exact-match allowlist (`doctor._loader_failure_code`); hostile
  checkpoint bytes can no longer reach the terminal, and non-ValueError
  parse failures no longer produce a traceback. The debug REPL's error
  path strips terminal-control characters from exception text.
- Red-team cycle 13, `diff-reports` (all pinned by new
  `tests/test_cycle13_diffreports.py` regression tests):
  - **H1 (CI-blocking):** `pytest` added to the `dev` extras in
    `pyproject.toml` (`pytest>=8.0`) -- it is the suite's actual runner
    (the README test-count pin test shells out to `python -m pytest
    --collect-only`), and CI's `static`/`test` jobs install `.[dev]` /
    `.[dev,hf]`, so the dev extra is exactly what CI installs. Pin test
    verified passing in the repo venv.
  - **M1:** `render_summary` now runs the existing `_display_id`
    sanitizer over the decision strings and the source labels before
    they reach stderr -- hostile `decision` values (ANSI escapes, OSC
    sequences) can no longer inject terminal control bytes.
  - **M2:** `_config_snapshot` candidate-name dedup is now set-based
    (linear) instead of an O(n^2) list scan -- the old code hung ~1.5h
    on 800k candidate names.
  - **M3:** `_search_verdicts` now also returns per-candidate `planned`
    counts, and the diff emits `planned:<candidate>` config-drift
    entries (surfaced in the stderr drift section) when they differ --
    planned 2 vs 50 no longer compares as silently identical.
  - **L1:** evidence-presence asymmetry now sinks `identical` (new
    machine-readable `summary["evidence_asymmetric"]` flag), and the
    "no differences" stderr line names the axes actually compared
    (decision, winner, candidate verdicts, configuration, plus
    per-fixture evidence only when both sides carry it).
  - Documented (comment at `_fixture_outcomes`, no behavior change): the
    duplicate-fixture-ID last-wins vs duplicate-candidate first-wins
    inconsistency is deliberate -- reconciling it would change only
    hostile-input behavior.
- `emulate_vllm_holdback` now skips the stop check on steps whose token
  decodes to no new text, porting upstream `check_stop_strings`'
  `if not new_char_count ...: return None`
  (`vllm/v1/engine/detokenizer.py`). Without it, the find window start
  `1 - len(stop)` is 0 for single-character stops (e.g. `"\n"`), so a stop
  completed during the `min_tokens` gate was refound at the first
  zero-new-char step -- upstream never refires it. Found by a new
  differential harness (`tests/test_cycle14_holdback.py`): a line-faithful
  port of upstream's `update`/`check_stop_strings` compared against the
  emulation over 32 adversarial cases (stop at position 0, stops spanning
  token/gate boundaries, empty-decoding tokens, unicode, both
  `include_stop_str_in_output` modes, empty output) plus a 400-trial seeded
  fuzz; all 435 subtests agree with upstream after the fix.
- `emulate_hf_stop_string` now treats empty token representations as
  transparent to the match walk (upstream assigns them valid positions
  `1..len(stop)-1` and zero length contribution, so a live walk always
  passes through them) and bounds the walk to the last `len(stop)`
  representations, mirroring upstream's
  `input_ids[:, -maximum_token_len:]` truncation -- found by a new
  differential harness (`tests/test_cycle14_hfstop.py`) comparing the
  emulation against the real `transformers.StopStringCriteria` (5.19.0)
  over the docstring's match/no-match cases, stop-at-0/empty-output/
  unicode/overlapping/overlong stops, prompt-boundary overhangs, the
  byte-space path (UTF-8 fragments, byte-fallback hex tokens, OOV ids),
  multi-stop earliest-completion, plus a 525-trial seeded fuzz; the one
  pre-fix disagreement was `["s", "", "topper"]` vs `"stop"`, where
  upstream fires at step 3 and the port stayed silent. Out-of-vocabulary
  ids on the byte path now map to an empty `_OOVTokenBytes` marker (still
  `== b""`, so they can never match) that keeps killing the walk exactly
  like upstream's dummy-row clamp; previously the shared `b""` could not
  be told apart from a genuine empty token. All 3000+ differential trials
  agree with upstream after the fix.

### Added

- `mpl debug`: an interactive protocol-debugger REPL on stdlib `cmd`
  (sqlite3-style dot-commands: `.use`, `.render` with per-segment maps,
  `.tokens`, `.step` for walking boundary detection, `.audit`, `.say`
  multiline entry, `.save`, bounded `.read`, `.quit`; `--script` for
  non-interactive runs). Attaches a checkpoint tokenizer without loading
  weights; operator-visible strings are terminal-escape sanitized.
- `mpl diff-backends`: differential-testing of the emulated stop semantics
  (HF stop-string, vLLM strip/keep, EOS) over one recorded token sequence,
  classifying divergences (`KEPT_VS_STRIPPED`, `FIRING_STEP_DIFFERS`,
  `MIN_TOKENS_GATED`, `EOS_PRECEDENCE`, `HARNESS_NORMALIZATION_RELIED` --
  the last scoped to failed strips so the `agree` verdict is meaningful).
  Grounded in vendors' bug trackers (vLLM #49391/#46113/#47137, SGLang
  #28802, lm-eval #3587); see `docs/RESEARCH_LOG.md` Round 4 Cycle 2.
- Format catalog expanded to 14 families: `kimi3` (K3 XTML, structure
  spot-checked against the live `encoding_k3.py`), `deepseek41`
  (V4.1-Flash chat mode), `glm45` (GLM-4.5/4.6 think variant).

### Fixed

- `HARNESS_NORMALIZATION_RELIED` no longer fires tautologically on
  kept-disposition cells (red-team: it made `verdict="agree"` unreachable
  whenever any stop fired); now reports only failed strips, with a
  regression test for the genuine anomaly.
- `mpl debug` hardening (red-team): terminal-escape sanitization of all
  fixture-derived output, `.read` refuses non-regular files, transcript
  and `.say` buffer memory caps, `.save` path behavior documented.
- `diff-backends` input caps (1MB texts, 131072 token IDs).
- Doctor suggestion-sizing test updated for the 14-family catalog (the
  longest render now exceeds the tiny test model's 64-token window; the
  test pins the exact sizing formula on a 512-window variant).

### Added

- Format catalog expanded from 3 to 11 families, each hand-rendered from
  primary sources (see `docs/RESEARCH_LOG.md` Round 4): `qwen3`, `deepseek`
  (V3/R1 framing), `mistral` (v0.3 release revision), `phi4`, `gemma3`,
  `llama4`, `glm4` (4-9B), `kimi2`. `Format` gained optional
  `user_turn_end` / `assistant_turn_end` / `open_suffix` overrides for
  families whose per-role terminators differ (e.g. DeepSeek's unterminated
  user turn, GLM-4's bare `<|assistant|>` generation prompt). Also fixed a
  pre-existing `UnboundLocalError` that broke the `catalog` subcommand.
- Hypothesis property-based fuzz tests (`tests/test_fuzz_boundary.py`,
  derandomized): stop-overhang suffix completeness, first-firing-step
  minimality against a naive reference, UTF-8 incremental-decode
  round-trips, NFC byte-offset disagreement, byte-fallback token chunks.
- Stop-matrix hardening: the vLLM cell now exposes the competitor's text
  (`StopOutcome.returned_text`), strips the stop string by default
  (`include_stop_str_in_output=False`, the upstream default -- HF always
  keeps it, surfaced per cell as `stop_text: kept|stripped`), and models
  `min_tokens` gating upstream-exact (check skipped while
  `num_output_tokens <= min_tokens`; the find window's negative start
  counts from the end of the text, so a stop fully completed during the
  gate is not refound). FAQ documents the lab's no-implicit-EOS discipline
  vs both backends' defaults and how to match it per backend.
- Special-token literal audit (GHSA-g5f9-3xfg-p9mf): `fixtures build`
  warns on stderr and writes a `<output>.audit.json` sidecar when fixture
  text contains literals colliding with candidate formats' special tokens
  (role-boundary forgery hygiene); `doctor` reports the same as a warning.
- Report upgrades (still one self-contained file, no external assets):
  sortable tables (small inline script, inert-safe), sticky anchor TOC,
  exclusive `<details name>` accordions, SVG budget-frontier sparkline, and
  a tested no-remote-asset property.

### Fixed (observation-count bound, 2026-10-15)

- Red-team residual of the H1/M1/L1 fix cycle: `divergence_section_data`
  now bounds the *number* of recorded observations with the named constant
  `MAX_DIVERGENCE_OBSERVATIONS = 512`. Legitimate runs hand over at most
  200 observations (the experiment fixture cap is 100 search + 100
  held-out fixtures, and only the winner's observations feed the section),
  so real runs are never affected; a hostile report with millions of
  observations is now fixed work instead of open-ended emulation time.
  Over-count input degrades by omission, never by exception: the first 512
  observations are processed in input order and the section records the
  fixed-vocabulary `observations_truncated: true` plus `observations_total`
  / `observations_processed`; the HTML report names the bound explicitly.
  Pinned by `tests/test_cycle11_bounds.py` (boundary N/N+1, first-N order,
  determinism, no-usable-observations still degrades to an omitted
  section). Per-observation bounds (H1/M1) are unchanged.

### Fixed

- vLLM `min_tokens` emulation corrected to upstream-exact semantics after
  a source-level re-check of `vllm/main` `v1/engine/detokenizer.py`
  (negative `find` start counts from the end; straddling-update offset
  logic), replacing an earlier never-fire approximation.
- Report `data-sort` attributes now escape hostile strings (attribute
  injection caught by cross-agent review before release); a regression test
  pins the fix.
- `fixtures build` no longer leaves an orphan fixture file if the audit
  sidecar loses a concurrent-creation race: the just-written fixture is
  unlinked and the collision is reported distinctly from authoring errors.
- Empty output with `min_tokens=0` in the vLLM emulation is labeled
  `no_stop_within_cap`, not `gated_by_min_tokens` (nothing was gated).

- `run` stdout now names the written artifacts: the final JSON line is
  `{"decision": ..., "selected_candidate": ..., "results": "<dir>/results.json",
  "report": "<dir>/report.html"}` so a first run points the operator at the
  report instead of leaving them guessing where it went (consistent with
  `fixtures build`, which already reports its output path).

- `doctor` suggestion now sizes `--max-new-tokens` to the checkpoint's
  declared window (`doctor._longest_prompt_tokens`): the longest fixture
  prompt rendered with every catalog candidate is encoded with the
  checkpoint tokenizer, and the suggested budget is the largest value (up
  to 64) that still fits prompt + budget within the window. When even the
  longest prompt fills the window, the suggestion keeps `--max-new-tokens 1`
  and warns to shorten the fixtures or use a larger-window checkpoint.
  Without a measurable tokenizer the suggestion keeps the default 64.
- Failed-run diagnosis: `run_experiment` summaries carry the sanitized
  `failure_code`, and the report's decision banner names it with operator
  guidance ("Failure diagnosis"). `core.evaluate` now passes through the
  built-in backend's fixed-vocabulary error codes (`context_limit_exceeded`
  and friends) via an allowlist; arbitrary backend text still collapses to
  `backend_reported_error`, so the sanitization boundary holds.
- Per-candidate response boundaries (`run --boundary-map NAME=ID,ID`,
  repeatable; `doctor --boundary-map` for pre-flight vocab checks):
  each format hypothesis is measured with its own end-of-turn marker
  instead of one sequence for all candidates, with `--boundary-ids` as
  the default. The selection rule is unchanged (exactly one candidate
  passing every search fixture under its own boundary; ties and failures
  abstain); the held-out stage, stop matrix and budget frontier use each
  candidate's own boundary, and every search entry records its
  `boundary_token_ids`. Unknown or duplicate candidate names fail fast.
- `fixtures build` (`model_protocol_lab/fixtures.py`): compile a compact
  authoring CSV (one row per fixture: `id,group,partition,expected` plus
  `user_1,assistant_1,...` message columns) into the strict fixture JSON.
  The search/held-out discipline (ID uniqueness, group non-overlap,
  identical-message leakage, NFC-normalized) is enforced at build time
  with row-numbered errors; the emitted JSON is round-tripped through the
  strict loader before it is returned, so a built file satisfies the
  schema by construction. Output is never overwritten and is written
  owner-only (0o600); `examples/fixtures.csv` builds to the equivalent of
  `examples/fixtures.json`.
- Richer report views (still a single file, no scripts or remote assets):
  a per-fixture x candidate outcome grid (✓/✗/– with the selected
  candidate starred), decisive fixtures (fixtures passed by exactly one
  candidate while all others failed -- the evidence the verdict rests
  on), per-candidate token-budget pass-curve sparklines in the frontier
  section, and a boundary-IDs column in the candidate table.
- Byte-space HF stop-string matching (`backends._hf_byte_token_mapper`,
  `LocalHFBackend.hf_stop_token_bytes`, threaded through
  `emulate_hf_stop_string`/`stop_matrix`/CLI): for byte-level and
  byte-fallback decoders the HF emulation now matches on raw token bytes
  with the stop string UTF-8-encoded, mirroring current transformers (the
  byte-space fix for upstream issue #46519); tokens the byte table cannot
  represent fall back to the isolated decode. Differentially verified
  against the real `StopStringCriteria`, including the byte-fragment shape
  where the old decode-string path stayed silent and the oracle fires.
- Token-budget robustness frontier (`experiments.budget_frontier`, always-on
  `run` diagnostic, new HTML section): recomputes the search-stage decision
  at every smaller token budget from the recorded outputs -- exact under
  greedy decoding, no extra model calls -- reporting per-candidate minimal
  passing budgets, the pass-count step curve, and the budget ranges where
  the decision or winner would have differed (`flip_ranges`,
  `budget_robust`).
- `doctor` next-command suggestion: a clean `doctor` with `--dataset` and
  `--boundary-ids` now carries a `suggestion` with the exact `run`
  invocation, `--max-context` filled from the checkpoint's declared window
  and paths shell-quoted; the format family stays an explicit placeholder.
  `examine_model` also reports `model.context_window`.
- `docs/FAQ.md`: troubleshooting and interpretation Q&A grounded in
  documented behavior, linked from README and `docs/USAGE.md`.
- Template-conformance diff (`core.template_diff`, `doctor.template_conformance`,
  surfaced in `run` reports): each hand-rendered hypothesis is compared as an
  exact token-ID sequence against the checkpoint tokenizer's own
  `apply_chat_template` rendering of the same messages, reporting the divergent
  token-ID subsequences and their positions (empty diff when byte-identical);
  "no template declared" is reported gracefully and never fails the run.
- Failure-mode taxonomy (`experiments.classify_observation`/`failure_taxonomy`,
  cited in the report's decision banner): every non-passing observation is
  classified into the closed set `no-boundary-in-N`, `boundary-text-mismatch`,
  `eos-without-boundary`, `held-out-divergence`, `backend-error`, with
  per-candidate counts; `run_experiment` accepts an optional `eos_id`
  (the CLI takes it from the checkpoint tokenizer).
- Stop-behavior isolation matrix (`experiments.stop_matrix`, emulations in
  `backends.py`, surfaced in `run` reports): the frozen winner's recorded
  outputs are re-measured under emulated HF `StopStringCriteria` overhang
  matching, vLLM detokenizer earliest-completion truncation, and EOS-appended
  stop semantics, reporting per-fixture pass/fail and which semantics flip the
  verdict. Emulations are post-hoc truncations (exact for greedy decoding).
- Cross-tokenizer boundary portability (`core.check_boundary_portability`,
  `doctor --reference-model`): the boundary marker text is re-encoded with a
  second checkpoint's tokenizer and reported portable/non-portable with both
  ID lists; non-portability is informational (`mismatch`), never a failure.
- Prompt-boundary sensitivity sweep (`experiments.prompt_sensitivity_sweep`,
  `run --sensitivity-sweep`/`--max-drop-bytes`, `backends.TruncatedPromptBackend`,
  `core.truncate_text_tail`): re-runs search with each prompt truncated
  mid-token at every depth, reporting the depths where the winner changes.
- `doctor` CLI command: pre-flight validation of checkpoint assets and fixture
  files without loading model weights (`model_protocol_lab/doctor.py`).
- `Observation.boundary_text_match`: diagnostic cross-check recording whether
  the decoded output ends with the decoded boundary text, flagging the known
  tokenization-mismatch false-negative mode. Never affects the pass criterion.
- `candidate_fingerprint` in search-stage results (sha256 over the canonical
  format JSON), surfaced in the HTML report.
- Redesigned HTML report: decision banner with per-decision explanation, run
  configuration section, pass-rate heatmap with bars and text labels, held-out
  stat cards, WCAG AA contrast (minimum 6.81:1), responsive layout, print
  stylesheet; still a single file with no external assets and a strict CSP.
- Committed synthetic sample report (`examples/sample-report/`) with a
  byte-identical regeneration golden test.
- CI workflow (`.github/workflows/ci.yml`): lint + typecheck on 3.12 without
  optional deps, full tests on 3.11/3.12/3.13 with the `hf` extra, dependency
  and static security audit (`pip-audit`, `bandit`), packaging build check;
  weekly scheduled run. Actions pinned to SHAs.
- Pre-commit config, Dependabot config (pip, github-actions, pre-commit),
  `CITATION.cff`, `CONTRIBUTING.md`, issue and PR templates.
- Property-based tests (hypothesis) for dataset loading, fixture separation,
  boundary parsing, and adversarial JSON inputs.
- HTML report upgrades, still a single self-contained file with no external
  assets: sticky section table of contents with `:target` highlighting;
  click-to-sort tables via one small inline script (SHA-256 hash-allowlisted
  in the report's Content-Security-Policy -- the report reads identically
  with scripting disabled); mutually exclusive `<details>` accordions via
  `name=`; inline-SVG pass-curve sparklines for the token-budget frontier
  (same quantized data as the text sparklines, prints and scales reliably).
- Red-team (follow-up, 2026-10-15): `diff-reports` config-drift field
  reached stderr unsanitized -- the `planned:<candidate>` drift field is
  built from the raw candidate name, and `render_summary` interpolated the
  field raw, so a hostile candidate name with ANSI/OSC escapes reached
  stderr (confirmed live). Fix: the field is passed through the existing
  `_display_id` sanitizer at the render loop, which covers every drift
  field type (planned:, fixtures:search/held_out, and the fixed config
  fields). Pinned by a regression test in
  `tests/test_cycle13_diffreports.py` (hostile candidate name in a
  planned-drift entry -> no raw escape bytes in the rendered summary).
- Red-team (follow-up, 2026-10-15): the README test-count pin test never
  inspected `proc.returncode`/`proc.stderr`, so a pytest collection error
  surfaced as the misleading "could not parse pytest collection count".
  Fix: the test now asserts `proc.returncode == 0` first, and both
  failure messages include the last ~2000 characters of stderr/stdout so
  CI collection failures are debuggable from the test output alone.

### Changed

- `pyproject.toml`: PEP 639 license metadata (`license-files`), dev/hf extras,
  ruff/mypy/coverage configuration (ruff `S` rules enabled).
- Report outputs now use restrictive permissions (directory `0o700`, files
  `0o600`, exclusive creation).
- `inspect` reports the fixture validation reason instead of a generic error.
- CLI surfaces sanitized error codes (`ValueError` vocabulary, `local_*`
  backend codes); `--boundary-ids` accepts ASCII digits only.
- Dataset errors identify the fixture (`search[i]`/`held_out[i]`);
  deeply-nested JSON, non-UTF-8, and char/block-device fixtures raise
  `ValueError` in the module's error vocabulary.
- README status: "unexecuted alpha" → tested state with evidence pointers; new
  "How this differs" section grounded in primary sources.

### Fixed (third hardening round, red-team, 2026-10-13)

- **Red-team round 3 — HF stop emulation:** `_hf_overhang_match` missed the
  tensor implementation's end-overlap class where the stop string ends
  *inside* the final token (e.g. token `" world"`, stop `"or"`); the stop
  matrix systematically under-reported HF stops. Now mirrors
  `_stop_string_get_matching_positions` exactly — differentially verified
  against the real `transformers.StopStringCriteria` (1500/1500 random
  trials on byte-level BPE, exhaustive 256-combination scan, zero
  divergences). Known residual: per-token `decode` strings vs HF's cleaned
  vocab strings / raw bytes can diverge on UTF-8-fragment tokens
  (documented in `docs/USAGE.md` §8.2).
- **Red-team round 3 — `template_diff` algorithmic DoS:** adversarially
  fragmented inputs made difflib super-quadratic (~45 s at n=2000, ~cubic
  scaling). The matching-blocks loop now runs under a 10 s wall-clock
  budget and degrades to `ValueError("template_diff_too_fragmented")`
  (the conformance check marks the pair `"unavailable"`); output is
  byte-identical to unbounded difflib on ordinary inputs.
- **Red-team round 3 — uncapped tokenizer loads in `doctor`:** the
  reference-model portability check and `template_conformance` loaded
  `tokenizer.json` via HF with no 128 MB pre-load cap (the round-1 fix
  covered only `LocalHFBackend`). New `_tokenizer_file()` helper enforces
  the cap at every loader call site; oversized files degrade to the fixed
  `"skipped"`/`"unavailable"` codes without parsing.
- **Red-team round 3 — hostile Jinja chat templates:** a checkpoint
  template rendering unbounded text (`{{ 'x' * 10**8 }}`) OOM-killed the
  conformance worker. Template source and rendered output are each capped
  at 1 MB chars, degrading to `"unavailable"` (`"chat_template_too_large"`);
  CPU burn inside the Jinja render itself remains the same exposure as HF's
  own `apply_chat_template` (documented residual).
- **Round-3 test hygiene:** the new `test_round3.py` broke the `[dev]`-only
  suite (7 classes imported the HF stack at test time with no skip guard);
  added the `HAS_HF` module guard so the README quick-start path is
  skip-clean again.
- 9 new regression tests (361 total, all passing, 98% line coverage).

### Fixed (third hardening round, audit, 2026-10-13)

- **Audit:** `doctor`'s config-read failure path returned a non-uniform report
  shape (missing `assets`/`special_tokens`/`added_tokens`) — now uses the same
  `_model_result` wrapper as every other early failure, with the already
  computed asset manifest attached; `_read_json` normalizes raw
  `UnicodeDecodeError`/`JSONDecodeError`/`RecursionError` from inspected asset
  files to the fixed `invalid_asset_json` code instead of leaking raw
  codec/parser text into doctor check details (mirrors
  `datasets.parse_dataset`); `run_experiment` rejects a backend missing the
  `fingerprint` attribute with `ValueError` instead of a raw `AttributeError`;
  `evaluate` compares NFC-normalized fixture IDs for uniqueness, closing the
  NFD/NFC evasion gap that round 2 fixed only in `Split`;
  `prepare_report_dir` never chmods through a symlinked path component
  (`os.chmod` follows links — the old code could tighten permissions on a
  directory outside the requested tree).
- 7 new regression tests (231 total, all passing, 97% line coverage held).

### Fixed (second hardening round, 2026-10-13)

- **Audit:** stock GPT-2 checkpoints failed with the CLI default
  `--max-context 2048` and the cause was swallowed into the generic init
  failure — now fails fast with the distinct, actionable code
  `local_context_limit_exceeds_declared_window` (documented in
  `docs/USAGE.md`); `run_experiment` validates `boundary`/`cancel`/backend
  fingerprint types up front instead of raw `TypeError`s; `evaluate` rejects
  non-`Event` `cancel`; `doctor` validates `boundary_ids` element types and
  returns a uniform report shape (always with `assets`/`special_tokens`/
  `added_tokens`) on early failures; `load_dataset` rejects non-path input
  with `ValueError`; corrected the false `candidate_fingerprint` docstring
  (the name is part of the payload).
- **Red-team round 2 (2026-10-13):** NFD/NFC variants no longer evade the
  fixture-ID uniqueness or cross-split leakage checks (`Split` compares
  NFC-normalized keys; stored data stays byte-exact); terminal escape
  sequences stripped from fixture IDs in `inspect`/`run` stderr warnings.
- **Docs:** the 3 hf-dependent tests that failed under a `[dev]`-only
  install now skip cleanly, so the README quick-start path is green without
  the heavy `hf` extra; `docs/USAGE.md` documents the GPT-2 context-window
  requirement and NFC-normalization behavior.
- 11 new regression tests (224 total, all passing, 97% line coverage held).

### Fixed

- `LocalHFBackend._directory`: `FileNotFoundError` and symlink loops now
  surface as `ValueError("invalid_local_directory")` instead of leaking OS
  error text.
- `evaluate()`: per-attempt helper closures no longer risk stale loop-variable
  capture (state passed explicitly).
- Tokenizer file size is re-checked before the HF loader parses it
  (defense-in-depth for the 128 MB JSON cap).
- **Red-team round (2026-10-12):** named-FIFO hang in `load_dataset` (now
  refuses all non-regular files via `O_NONBLOCK` + fd `fstat`); output
  directory now claimed before backend init and released on failure;
  `candidate_fingerprint` normalized to match `local_repairs` dedup;
  report parent dirs all 0o700; `doctor` tokenizer-deep failures sanitized;
  empty-`expected` fixtures warn in `inspect`/`run`.

### Added

- SGLang column for `mpl diff-backends` (implements the Cycle 1 research
  recommendation): new `sglang_stop` emulated semantic
  (`backends.emulate_sglang_stop`, a port of SGLang's
  `Req._check_str_based_finish` with `DetokenizerManager.trim_matched_stop`
  leftmost-occurrence stripping; deliberately takes no `min_tokens`
  parameter -- SGLang's `min_new_tokens` only clips the length budget and
  the finish path never consults it) plus three new divergence classes:
  `MULTI_STOP_SELECTION` (high; SGLang first-listed vs vLLM
  earliest-completion multi-stop selection, vLLM PR #49391),
  `NO_MIN_TOKENS_GATE` (medium; the SGLang column on `MIN_TOKENS_GATED`),
  and `STR_OVER_EOS_PRECEDENCE` (high; same-step stop+EOS tie where
  SGLang's shipped #28802 fix gives the stop string the win while TGI's
  documented precedence gives EOS the win). The report's "Backend
  divergence" section renders the new classes with severity badges and a
  "SGLang stop" matrix column. Deliberately no TGI cell: every TGI stop
  rule is covered by composition of the existing cells (documented in
  `USAGE.md` §11); SGLang `stop_regex` gets a USAGE note only. Pinned by
  `tests/test_cycle10_sglang.py` (27 tests).

## [0.1.0a0] — 2026-10-12

Initial source snapshot: renderer, evaluator, catalog, local HF adapter,
experiment orchestration, reports, CLI, fixture schema, docs and test source.
Unexecuted at release of the snapshot.

### Added (cycle 110 — Hypothesis properties for truncate_text_tail, 2026-10-17)

- New `tests/test_cycle110_truncate_properties.py`: 6 Hypothesis property tests
  verifying the universal invariants of `model_protocol_lab.core.truncate_text_tail`
  (mid-token truncation for the boundary sensitivity sweep). For arbitrary Unicode
  text and cut positions: a non-None result is always a strict prefix; it encodes
  to exactly len(raw)-drop_bytes bytes; None is returned iff the cut splits a
  UTF-8 codepoint; cutting at a character boundary never returns None; out-of-range
  drop_bytes and wrong types raise ValueError. Mutation-tested: 3/3 mutants killed
  (empty-string-on-split, off-by-one, boundary acceptance). Pytest 4134 → 4140;
  unittest 4060 → 4066. README/VALIDATION.md counts updated.
- Documented the cycle-107/108 test_cycle66 CI mystery investigation in
  docs/RESEARCH_LOG.md: the TestCase conversion passes under every local
  condition tested (local venv, fresh venv, isolation, full suite, coverage)
  but fails CI on Python 3.12/3.13. Kept as `_UNITTEST_EXEMPT` pending a
  CI-environment reproduction.
