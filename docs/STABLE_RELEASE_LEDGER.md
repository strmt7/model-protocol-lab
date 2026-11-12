# Stable-release ledger: baseline record, finding status, readiness design

**Verdict at this baseline: NOT RELEASE-READY.** This ledger is the
running evidence record of the stable-release program that implements
the [2026-11-10 audit plan](audit/2026-11-10/PLAN.md) (work packages
W0–W8). It records the frozen baseline, the status of every audit and
wave-9 finding with its evidence, and the design of the readiness
entry point that W1/W8 will build. Later work packages update the
finding rows **only** with fresh evidence bound to a newer source;
a checklist entry is never itself proof.

Companion document: the [stability contract](STABILITY_CONTRACT.md)
(capability inventory, per-section contract table, boundary invariants).

## 1. Baseline record (W0)

| Item | Value |
| --- | --- |
| Baseline head | `1ffdb64005f87d7b95a9ff98d462241c9054b52f` (`main`, 2026-11-10) |
| Audit baseline | `d8a1346343e1b50b28bf567ab6b193b38fc79608` |
| Tracked files at baseline head | **587** (blob entries in the recursive git tree, counted from the GitHub API tree, not truncated) |
| Tracked files at audit baseline | **579** (same method; matches the audit's own inventory count) |
| Delta | +8 files, 0 removed, 2 changed — all documentation: added `docs/HANDOVER.md` and the 7 files of `docs/audit/2026-11-10/`; changed `AGENTS.md` (audit pointer) and `CHANGELOG.md` (audit entry) |
| Package identity check | Every blob under `model_protocol_lab/` (76 files), `tests/` (256), `tools/` (20), `examples/` (5), `fuzz/` (1) and `.github/` (17) is **byte-identical** between the audit baseline and this baseline (API tree comparison). The audit's "documentation-only delta" claim is verified, and every audit finding stated against `d8a1346` is stated against the code this ledger covers. |
| Suite at audit baseline | 9,676 tests, CI run 42835070179 green on the audited head (audit evidence) |
| Coverage floor at baseline | 95.35207221373815% (precise, from `coverage json`; set by cycle 244) |
| CI at this baseline (read 2026-11-10 ~20:30 CET) | Workflows completed green: Docs, CodeQL, Secret scan, Scorecard, Cross-OS smoke. Main CI run 42880807349: lint/typecheck, build/twine and dependency/security jobs green; the three test legs still in progress at read time — recorded as **not yet passing**, per the rule that missing evidence is non-passing. Findings guard failed on its DeepSource leg only (blocked-external, RR-05). |
| Deployments at baseline | 0 (the two records minted by the audit commits' Docs publishes were retired by the standing deactivate-then-delete duty) |
| Already fixed before the audit baseline | The 13 filelock Dependabot alerts (cycle 245, commit `d8a1346` itself): 0 open Dependabot alerts at baseline. Not an audit finding; recorded so the fix is not re-litigated. |

Top-level manifest shape at the baseline head (blob counts):
`tests/` 256, `site/` 101 (tracked generated site output — RR-16),
`docs/` 76, `model_protocol_lab/` 76, `tools/` 20, `.github/` 17,
`model_protocol_lab.egg-info/` 6, `examples/` 5, `.agents/` 3,
`.clusterfuzzlite/` 3, and 21 single files at the root (including
`pylock.toml`, `pyproject.toml`, `mkdocs.yml`). Total 587.

## 2. Finding-status ledger

Status vocabulary: **open** (defect/gap present at baseline),
**fixed-with-evidence** (a named commit + the execution that proves
it), **blocked-external** (cannot be advanced from this host; the
blocker is named), **design-debt-accepted-for-W<n>** (the audit itself
classifies the item as design debt with an owning package),
**pending-verification** (a lane's execution is still running; no
status is guessed). "Verification" cites the strongest evidence behind
the status: the audit's own execution at `d8a1346` (tree-identical to
this baseline, §1), the semantic-keys lane, the projection-census
lane, the bug-hunt wave-9 report, or the audit-verify lane's execution
ledger (`~/workspace/agents_out/audit_verify/ledger.md`, outside the
repo). Priorities and owning packages are the audit's.

### 2.1 Audit findings RR-01 – RR-18

| ID | Pri | Finding (short) | Status | Owning WP | Verification |
| --- | --- | --- | --- | --- | --- |
| RR-01 | P1 | `.gitleaks.toml` defines no rules; the configured secret scan is vacuous | **fixed-with-evidence** | W1 | audit-executed; **fixed in W1**: the config now extends the default rule set (`[extend] useDefault = true`). The miss was reproduced pre-fix with the checksum-verified pinned 8.30.1 (private-key/JWT/Slack canaries missed under the old config in both dir and history modes; fabricated `ghp_`/OpenAI shapes are not caught even by default rules and were excluded as canaries). The permanent proof is `tests/test_gitleaks_sensitivity.py`: exactly its two repo-config canary tests fail on the pre-fix config, all six pass after, and the Secret scan workflow runs the module with the pinned binary before scanning. Both flagged fixture strings were replaced at source with invariant-preserving equivalents; the working tree scans clean under default rules with NO allowlist, and the full history (read-only public clone at the pre-W1 head) scans clean under the repaired config and shows exactly the 2 adjudicated fixture findings without it. The two cycle-210 exact-regex entries were REMOVED on the owner's 2026-11-11 no-exceptions order ("solve the findings or leave them be"), executed at 3efc51db: `.gitleaks.toml` now carries no [allowlist] at all, and the cycle-210 pin asserts zero allowlist regexes. The Secret scan history job therefore reported exactly the 2 known verified-synthetic fixture findings (test_cycle79/test_cycle55 strings in pre-W1 commits) until the owner closed the question by authorizing a full git history rewrite on 2026-11-12: both fixture strings were replaced in all historical blobs with the same replacement strings used at source in W1 (`ARBITRARY-TEXT-leak-1`, `probe-value-hunter2-leakprobe`), the config's own enumeration comments were updated to match, and a post-rewrite full-history scan with the pinned gitleaks 8.30.1 reports zero findings (the Secret scan history job is green at the rewritten head). Nothing stands open on this row. All commit SHAs from the earliest affected commit onward changed as a consequence of the rewrite — SHAs cited in records written before 2026-11-12 (including the removal commit cited above) are pre-rewrite and historical |
| RR-02 | P2 | Malformed stored reports crash consumers (frontier `decision_steps` scalar; floor `budget_robust: []`; precision mixed-type sort) | **fixed-with-evidence** | W2 | audit-executed through the public CLI; **lane-confirmed verbatim** (both CLI cases incl. the valid-JSON-before-failure floor shape, and the precision sort in both input orders) **Fixed in W2 push 1 (2026-11-11)**: consumers gate container/element shapes before iteration, sorting and lookup across diffreports and reporting (frontier decision_steps/flip_ranges/pass_steps/budget_robust, held-out floor budget_robust, precision unavailable_candidates), and diff-reports builds both outputs fully before emitting either — no partial-success artifact. Regression: tests/test_w2_input_boundary.py (every defect case fails on the pre-fix tree, every control passes) |
| RR-03 | P2 | Negation section is produced but never diffed; a 16/16 → 0/16 change reads `identical: true` | **fixed-with-evidence** | W2 | audit-executed with producer pair; independently confirmed structurally at baseline (no `_valid_negation`, no dispatch — stability contract §2.2) **Fixed in W2 push 3 (2026-11-12)**: `diff_negation` dispatches the section under the recompute standard (`_valid_negation` gates exact key sets, the int protocol seed, and candidate/case counts against the frozen case generator, and compares every stored aggregate against `negation.aggregate_candidate` recomputed from the stored records; malformed records fail closed), the differ reports protocol/candidate/headline/arm/discordance drift, the summary gains negation_changed/negation_asymmetric, and render_summary names the drift. The instrument census is now a mechanical registry in diffreports (SECTION_DIFF_KEYS: 56 dispatched results sections; UNDISPATCHED_SECTIONS: 8 named opt-outs with reasons — the four cli.py fail-softened diagnostics and the four doctor/diff-backends surfaces), pinned against the bare-pair diff dict and the cycle-67 oracle by tests/test_w2_negation.py, which also drives the genuine producer pair (16/16 → 0/16 no longer compares identical). The cycle-67 and cycle-27 oracles gained independent negation mirrors. Measured reconciliation: the shipped diff surface carries 55 section keys at baseline (W0 counted 53 dispatched; the two renamed sections, budget_frontier→frontier and held_out_budget_floor→held_out_floor, account for the difference in counting basis) |
| RR-04 | P2 | `_recomputed_equal` conflates JSON bool with number (`False` validates as `0.0`, incl. nested); precision type gates inconsistent | **fixed-with-evidence** | W2 | audit-executed (primitive + admitted artifact) **Fixed in W2 push 2 (2026-11-11)**: `_recomputed_equal` is now type-strict for every scalar on both sides (the pre-fix fallthrough admitted a stored bool against an expected float, and a stored int against an expected float; cycle 227's pin — a stored 0.0 against a recomputed int 0 is a rejection — fixes the contract as exactness against this package's own producers), the cycle-67 oracle mirrors it from the same spec, the six remaining `not in (True, False, None)` equality admissions (distraction ×2, leading-cell, numform boundary_text_match; precision fp32/bf16_passed; confidence native_verdict) are exact-type `_bool_or_none` gates, and the precision record loop gates `diverged` as bool-or-None before its identity probes. Regression: tests/test_w2_exact_type.py (every defect case fails on the pre-fix tree, every control passes) |
| RR-05 | P1 | DeepSource unavailable as a signal | **blocked-external** | W1 | Owner statement 2026-11-10: the DeepSource **quota is exhausted**; the service is parked by owner order. No capture, inventory run or login check is attempted. The `deepsource-quota-watch` hook polls for a refill and is the refill detector; readiness records this as blocked-external (quota), never as green and never as a missing-credential problem (the CLI login exists; the vendor-side analysis halt of 2026-11-01 is the underlying record). |
| RR-06 | P1 | Publication path does not enforce the quality gates (no exact-SHA readiness receipt consumed by `release.yml`) | open | W1/W8 | audit inspection of the workflow graph; publication itself remains owner-gated and prohibited under the plan. **W1 fence landed (2026-11-11)**: `tools/release_policy.py` — a release-policy verifier that, given an exact commit SHA and the version a release would claim, reads the GitHub API and accepts only when the CI run, the Secret scan run and the CodeQL run for THAT SHA all completed successfully, the `dependency and static security audit` job inside the CI run succeeded, and the tree at that SHA carries the claimed version; every gap rejects with a named reason (absent / incomplete / failed / cancelled / skipped / wrong-sha / job-not-success / wrong-version). The evaluation core is pure and covered by recorded-fixture tests (`tests/test_release_policy.py`: acceptance + every rejection mode + CLI recorded-evidence runs, no network). A non-publishing `.github/workflows/release-readiness.yml` (workflow_dispatch only, no tag triggers, no publish steps, no PyPI environment, read-only token) re-runs the full gate at the supplied SHA and then runs the verifier. `release.yml` is untouched. The row stays **open**: making `release.yml` consume the receipt is W8 rehearsal scope and the publication decision itself remains the owner's |
| RR-07 | P1 | Windows install from the checked-in locks fails (unmarked Linux-only `nvidia-cufile` in the cocoindex and dev-hf closures) | open | W3 | audit-executed on Windows (resolver); **lane premise executed from this host**: PyPI serves only manylinux wheels for `nvidia-cufile` 1.17.1.22, a Windows-platform download fails twice, and the census finds **38 unmarked platform-specific pins** across the two locks. The full Windows install itself is not executable from this host and stays an open W3 evidence cell |
| RR-08 | P2 | Dev setup and typing fail at the advertised boundary (Windows mypy: 34 errors in 3 files; dev-only collection errors) | open | W3 | audit-executed on Windows |
| RR-09 | P2 | `identical` is narrower than artifact equivalence: two invalid sections compare identical with no diagnostic; held-out planned-count drift (1/1 → 50/50) is invisible | **fixed-with-evidence** | W2 | audit-executed; **lane-refined**: the collapse reproduces on `structured.adherence_gap`, `cache_profile.sharability_gap`, `confidentiality.gap` and `answer_collision.max_collision_rate`, but **not** on `boundary_audit.overall_rate` — the blast radius is section-dependent. The held-out count half reproduces exactly; **fixed in W2 push 4**: every diff now publishes `section_states` for all 56 dispatched sections (compared / one_sided / absent / unvalidated, from the Push-3 registry post-pass in `diffreports._section_states`); a section unvalidated on both sides whose payloads are present and not recomputed-equal is named in `summary.unvalidated_drift_sections`, rendered in the text summary, and breaks `identical` — identical garbage on both sides is unvalidated but not drift. The held-out stage's evidence counts are compared under the search stage's planned-count rule (`_held_counts`): differing planned/attempted for the common selected candidate produce `held_out_planned:{name}` / `held_out_attempted:{name}` config-drift entries (malformed counts compare as None, the search convention). Regression tests: `tests/test_w2_section_states.py` (the audit reproducer's 1/1 vs 50/50 pair and both-invalid differing pairs fail on the pre-fix tree); the cycle-67 and cycle-27 oracles carry independent mirrors, and cycle-13's planned-drift pin now also asserts the two held-out entries its fixture implies |
| RR-10 | P1 | Hostile template rendering has only platform-dependent fallbacks (no portable containment/deadline) | open | W4 | known limitation, documented in code; audit-verified |
| RR-11 | P2 | Release smoke can import the checkout instead of the installed wheel | open | W5 | audit inspection + isolated re-run |
| RR-12 | P2 | Sdist-shipped tests lack their dependencies; the sdist test scope is not self-contained | open | W5 | audit inspection of `MANIFEST.in` and test imports |
| RR-13 | P2 | No real positive-path evidence: 7 rows / 324 generations, every decision `no_candidate`, held-out confirmation never entered | open | W6 | standing record (`docs/REAL_CHECKPOINT_VALIDATION.md`); the negative rows are evidence, not defects to weaken |
| RR-14 | P2 | Run-level budget/recovery incomplete (per-stage caps only; no durable partial-run resume) | open | W4 | audit inspection; consistent with the R2 remainder history |
| RR-15 | P2 | Duplicated contracts/gates; the cycle-124 mutation gate counts any non-zero child exit (incl. collection failure) as a kill | open | W7 | audit-executed; **lane-confirmed deterministically** (shadow child exit 2, zero semantic tests executed, gate returns OK in ~0.5 s). New datum from the lane: the gate's *unmutated control* passed only 2 of 3 runs on the verification copy (tree byte-identical after every run, no cause asserted) — the baseline itself is not deterministic there; W7 owns both |
| RR-16 | P2 | Docs/generated metadata drift: `site/` (101 files) and `model_protocol_lab.egg-info/` tracked (107 generated files total, lane-confirmed exact); egg-info says 0.1.0 vs package 0.2.0; living SECURITY.md claims contradicted by the serving implementation (its "no HTTP clients anywhere" claim vs `servingbackends.py`'s `urllib.request`); USAGE §6 table stale in three places (R4 adjudication record) | open | W7 | audit census + tree counts in §1; lane census |
| RR-17 | P2 | Scanner-exclusion policy not mechanically enforced | open | W1/W7 | audit lexical census (706 marker *lines*); **lane occurrence census**: 1,050 marker occurrences total, **310 in package source** (noqa 231, nosec 74, `type: ignore` 5), and **`skipcq` occurrences are 0 everywhere** — the cycle-225 withdrawal is complete on this head. Same conclusion, sharper arithmetic; the residual surface is noqa/nosec/type-ignore only. **W1 wave 1 (2026-11-11)**: the mechanical inventory is now a durable artifact (occurrence census over every `.py` in the package/tests/examples/tools/fuzz trees; repo total 1,089 = noqa 656, nosec 80, `type: ignore` 335, `pragma: no cover` 18, skipcq 0; package 314 = noqa 231, nosec 74, `type: ignore` 5, pragma 4 — method and per-file rows in the W1 work area, `marker_inventory.csv`). One bounded class removed by genuine typing fixes: **all 5 package `type: ignore` markers** (deliforge.py ×2 [arg-type], experiments.py [index], mcp_server.py [attr-defined] + [arg-type]). Regression-first: stripping the markers alone produced exactly 5 mypy errors with the same codes; the fixes (a `TypeGuard` on deliforge's `_valid_ids` + `Iterable` narrowing at its fail-closed boundary, walrus narrowing in the experiments selection comprehension, and `_build_backend` declaring its real `Backend` return type) leave `mypy` and `mypy --strict` clean with the markers gone, and a 1,176-combination differential run shows zero behaviour change in the restructured deliforge validator. The row stays **open**: the remaining classes (package noqa 231 — BLE001 ×223 deliberate fail-closed sites, nosec 74, tests-tree markers) are future waves, and mechanical enforcement (a census gate) is W7 scope |
| RR-18 | P2 | Support-matrix evidence thin (Linux CI + 3-OS smoke do not prove the advertised surface) | open | W3/W8 | W0's stability contract is the first deliverable against this item; evidence completion belongs to W3/W8, so the finding stays open |

### 2.2 Subsystem findings SA / SCI-B / RTI / ORCH

| ID | Pri | Finding (short) | Status | Owning WP | Verification |
| --- | --- | --- | --- | --- | --- |
| SA-01 | P2 | Template evaluator bypasses core generation validation (`_evaluate_template_case` vs `_evaluate_one`) | **fixed-with-evidence** | W2 | audit-executed (synthetic adapter violations) **Fixed in W2 push 1 (2026-11-11)**: _evaluate_template_case applies core._valid_generation at the same position as _evaluate_one; a None generation and a short limit generation now abstain instead of crashing / scoring as a pass |
| SA-02 | P2 | Boundary-audit exclusion counts do not conserve (observations ≠ usable + unusable on decode failure) | open | W6 | audit-executed |
| SA-03 | P2 | Duplicate JSON tool arguments collapse before grading; a conflicting duplicate still grades adherent | **fixed-with-evidence** | W2/W6 | audit-executed on a shipped query **Fixed in W2 push 1 (2026-11-11)**: toolcall arguments are parsed with a duplicate-rejecting object hook; a duplicated member grades parse_failed (bad_json) in both member orders and at any nesting depth |
| SA-04 | P2 | Custom structured `number` schema overflows on a 401-digit integer (float conversion of arbitrary-precision ints) | **fixed-with-evidence** | W2 | audit-executed **Fixed in W2 push 2 (2026-11-11)**: `structured._is_finite_number` returns True for ints without conversion (an int is finite by construction) and uses `math.isfinite` for floats; the cycle-157 differential oracle carries the same fix from the same reasoning. Regression: tests/test_w2_exact_type.py |
| SA-05 | P2 | Custom schema enums conflate bool and number via Python equality (int field accepts `true` enum match and vice versa); the test oracle repeats the defect | **fixed-with-evidence** | W2 | audit-executed; JSON Schema §4.2.2 is the reference **Fixed in W2 push 2 (2026-11-11)**: enum membership in `structured._field_value_ok` goes through `_enum_member` (bool only matches bool; numbers match mathematically across int/float — an int field with enum [1] still accepts 1.0; other values match by exact type and value), and the cycle-157 oracle's `_oracle_enum_member` implements the same §4.2.2 rule independently. Regression: tests/test_w2_exact_type.py (incl. the preserved-equivalence controls) |
| SCI-B-01 | P1 | Position `capital` key is ambiguous: Lisbon and Oslo are both European capitals; only Oslo is accepted | fixed-with-evidence | W6 (fix scheduled right after W0, alongside W2) | **execution-verified by the semantic-keys lane**: a Lisbon answer scores `exact_match = False` in the default fixture set. Repair specified (distractor Lisbon → Cairo) and applied 2026-11-11 by the semantic-key repair cycle; the pins in `tests/test_semantic_key_repairs.py` failed on the pre-fix tree and pass after it. See §2.3 |
| SCI-B-02 | P2 | Cache profile calls prefix sharing exact duplication (distinct prompts sharing a prefix count as duplicates) | open | W6 | audit-executed against an independent oracle (the hit formula itself verified correct) |
| SCI-B-03 | P2 | Few-shot comparisons mix fixture cohorts (order rates computed on different populations than the headline) | open | W6 | audit-executed (two-fixture stage) |
| SCI-B-04 | P2 | Cache prefix storage is quadratic in prompt length | open | W4/W6 | audit-measured (256/512/1024 → 32,896/131,328/524,800 retained references) |
| RTI-02 | P2 | Pack producer can emit bundles its own verifier rejects (> 1,024 entries incl. manifest; no producer-side aggregate ceilings) | open | W5 | audit-executed (1,025-entry bundle packed, then rejected) |
| RTI-03 | P2 | Cheap invalid input causes expensive work before rejection (MCP probe constructs a backend before cap/seed validation; exec-backends loads/probes before validating settings) | **fixed-with-evidence** | W2/W4 | audit-executed with recording constructors (no real model/server) **Fixed in W2 push 1 (2026-11-11)**: MCP tool_probe validates max_new_tokens/seed before _build_backend runs; exec-backends validates stop texts, id lists, the request's scalar fields, the --results file and the cache ids before the tokenizer load, backend construction and probe — each failure keeps its established exit code |
| RTI-04 | P2 | Serving artifact ingestion bypasses the strict bounded readers (unbounded read, permissive parse, unvalidated hash keys) | **fixed-with-evidence** | W2/W4 | audit-executed; overlaps wave-9 A8-1/A8-2 (§2.4) — one fix surface **Fixed in W2 push 1 (2026-11-11)**: load_records_artifact reads at most the sink budget plus a fixed envelope allowance, decodes UTF-8 and parses with non-finite-constant rejection under the one fixed code, and builds dedup/matching keys only from validated string fields; the exec-backends --results ingestion is bounded by the results-size budget |
| RTI-05 | P2 | Aborted cache sequences skip end teardown (erase/identity recheck only on normal completion) | open | W4 | audit-executed with a controlled persistence callback |
| RTI-06 | P2 | Invalid MCP envelopes and argument shapes execute (`jsonrpc: "1.0"` ping and `arguments: []` accepted) | **fixed-with-evidence** | W2 | audit-executed against the real stdio subprocess **Fixed in W2 push 1 (2026-11-11)**: handle_request validates the envelope before dispatch (JSON-RPC 2.0 section 5.1: -32600 with a null id when the id itself is invalid) and tools/call arguments are no longer falsey-coerced — absent/null means no arguments, any other non-object is -32602 and never executes |
| RTI-07 | P3 | Debug save exceeds its final byte cap (components checked, encoded total is not; 4,999,999 → 5,000,764 bytes observed) | open | W4 | audit-executed |
| RTI-08 | P3 | HTTP error-body read faults escape transport classification in the llama.cpp backend | open | W4 | audit-executed with fake streams (no leak claim; CLI traceback not claimed) |
| ORCH-01 | P1 | Default export retains private evidence fields (see below) | open | W2 | **execution-verified by the projection-census lane** (sentinel values, twice, on the baseline tree): `transfer.checkpoints[].path`, serving `records[].prompt_ids`/`returned_ids`, and `answer_text` in 8 sections reach default JSON; only 4 of 62 sections have a projection rule; HTML/summary clean by renderer discipline only. Full census: stability contract §3.2 |
| ORCH-02 | P1 | Optional stages disconnect cancellation (owner event restored before optional compute; batch stage mints private events) | open | W4 | audit-executed with a synthetic backend (event disconnection proven; real-signal integration not claimed) |
| ORCH-03 | P2 | Transfer evaluation retains multiple loaded backends simultaneously | open | W4 | audit-executed loader trace (`load, load, load, evaluate`); no real allocation attempted |
| ORCH-04 | P2 | Unavailable serving cells render as measured negatives (`fired=false` for capability-absent cells reads as "no fire") | **fixed-with-evidence** | W2 | audit inspection of producer + renderer; **fixed in W2 push 4**: the producer (`experiments.serving_backends_analysis`) now records `fired: None` (unobservable) for capability-absent cells — a measured no-fire keeps `fired: false` — the diff mirror (`diffreports._serving_recompute_rows`) and the cycle-67 oracle recompute the same shape, and the HTML renderer (`reporting._serving_backends_section`) prints "unobservable" for unavailable cells while measured cells keep "fired"/"no fire". Regression tests: `tests/test_w2_section_states.py` (producer shape, renderer wording for both cell classes, and a genuine capability-absent section validating both-sides through the differ on cycle-243's fake-server machinery) |
| ORCH-D01 | P2 | Export lacks version/schema admission (`save_report` accepts `schema_version=999`) | design-debt-accepted-for-W2 | W2 | audit inspection; classified design debt in the audit itself; the lane reproduced the acceptance of `schema_version: 999` on this baseline |
| ORCH-D02 | P2 | Report pairs lack a crash completion boundary (JSON visible before HTML exists) | design-debt-accepted-for-W4/W5 | W4/W5 | **lane: defect half NOT-REPRODUCED** — injecting a failure on the second file open removes both artifacts; the Python-exception cleanup works exactly as the audit credits it. The crash half (a half-written pair visible after a hard process kill) is not executable without killing a process mid-write and stands as design debt only |

### 2.3 Semantic-key siblings of SCI-B-01 (semantic-keys lane)

The lane re-derived **every** frozen semantic answer key by execution
or primary source. The large tables are clean (all 117
compatibility-form Unicode members under UCD 15.0.0; dateline's 8
dates; numform's 48 arithmetic entries; the enumeration, confusable-
skeleton, invisible-chars and canonical-form tables). Seven root keys
are defective (11 module occurrences — four keys are duplicated in
`structured.py` and `fieldorder.py` and both copies are graded).
SCI-B-01 is D-1; the six siblings:

| Key | Defect | Status | Repair specified by the lane |
| --- | --- | --- | --- |
| `extract-person` (structured + fieldorder) | **Wrong fact**: context asserts Ada Lovelace "was 36 when she wrote the first algorithm" — she was 27 when the Notes were published in Sept 1843; 36 was her age at death | fixed-with-evidence | 36 → 27 in context and answer, both modules, plus doc/test mirrors (applied 2026-11-11) |
| `route-ticket` (structured + fieldorder) | Ambiguous: `priority=3` pinned by nothing in the context (schema allows 1–5, no urgency cue) | fixed-with-evidence | context carries the priority ("Ticket (priority 3 of 5): …") (applied 2026-11-11) |
| `route-alert` (structured + fieldorder) | Ambiguous: `channel=pager` not entailed (email/sms equally defensible); the module itself records `paired=False` yet still scores it | fixed-with-evidence | context cues the channel ("Page the on-call engineer") (applied 2026-11-11) |
| `route-order` (structured + fieldorder) | Ambiguous, arguably wrong: `warehouse=south` for Porto (northern Portugal); `express` unpinned by "by Friday" | fixed-with-evidence | answer → `north`; context pins express ("express delivery was requested") (applied 2026-11-11) |
| `route-complaint` (fieldorder only) | Ambiguous: `team=technical` with `paired=True`, but `support` is defensible (support received and confirmed the complaint) | fixed-with-evidence | context pins the escalation to the technical team (preferred option taken 2026-11-11; `paired` unchanged) |
| `docs-healing` (toolcall) | Ambiguous: expected call pins `top_k=3`, an unstated default for an optional argument; any value 1–5 (or omission) is equally defensible | fixed-with-evidence | query mirrors the sibling's phrasing ("show the top 3 results") (applied 2026-11-11) |

All seven repairs are one-line context/distractor changes preserving
each fixture's shape and difficulty. **Repaired 2026-11-11** (the
semantic-key fix cycle, the first post-baseline fix recorded in this
ledger): the repairs landed in `position.py`, `structured.py`,
`fieldorder.py` (including its checking-rule worked example) and
`toolcall.py`, in the diff validator's mirrors in `diffreports.py`,
in the reporting prose and the field-order design doc, and in the
test mirrors, with regression pins in
`tests/test_semantic_key_repairs.py` — every pin failed on the
pre-fix tree for its named defect and passes after it, under a full
green verification gate on the repair head. The `paired` flags were
not changed: the paired population is frozen with fieldorder's
pre-registered claim.

### 2.4 Wave-9 findings (bug-hunt wave 9, 2026-11-10)

Execution-verified by the wave-9 lane on `3e0b1f99` (reproducers run
at least twice; two findings spot-verified independently by the
parent), and **independently re-verified on this baseline by the
audit-verify lane (all six OPEN)**. The target file
`servingbackends.py` is byte-identical between `3e0b1f99` and this
baseline (§1 method), so the findings stand on current code. They
ride inside W2/W4 scope; S3-1/S3-2 were not covered by the external
audit. A8-1/A8-2 are the two halves of RTI-04.

| ID | Finding (short) | Status | Owning WP |
| --- | --- | --- | --- |
| W9-A8-1 | `load_records_artifact` raises raw `TypeError` on unhashable record/plan key values (list/dict candidate/fixture/cache_position) | **fixed-with-evidence** | W2/W4 (with RTI-04) **Fixed in W2 push 1 (2026-11-11)**: with RTI-04: dedup/matching keys in load_records_artifact are built only from validated string fields; an unhashable key value is accounted (record kept, cell missing), never raised |
| W9-A8-2 | Non-UTF-8 artifact bytes leak a raw `UnicodeDecodeError` instead of the frozen `serving_backends_artifact_invalid` code (258/263 corpus raises) | **fixed-with-evidence** | W2/W4 (with RTI-04) **Fixed in W2 push 1 (2026-11-11)**: with RTI-04: non-UTF-8 artifact bytes fail with the frozen serving_backends_artifact_invalid code |
| W9-S3-1 | A hostile/broken server's oversized float-typed sampler-echo values raise `OverflowError` out of `_echo_matches` and crash the whole `exec-backends` run (the stage wrapper catches only cancellation/`ValueError`) | **fixed-with-evidence** | W2/W4 **Fixed in W2 push 1 (2026-11-11)**: _echo_matches compares float-typed fields exactly (no float() conversion); an oversized echoed integer is a config_echo_mismatch, proven end to end through the fake server |
| W9-S3-2 | An out-of-vocab returned token ID raises a raw `KeyError` in `execute_cell`, same run-crashing blast radius | **fixed-with-evidence** | W2/W4 **Fixed in W2 push 1 (2026-11-11)**: _classify locally decodes the type-gated token ids; ids the tokenizer cannot decode become the stack_response_invalid observation instead of a KeyError out of execute_cell (a non-text decode result still raises token_vocab_mismatch for the probe gates) |
| W9-S4-1 | Exported `aggregate_fidelity` raises raw `TypeError` on an unhashable verdict value (no stored path reaches it today) | **fixed-with-evidence** | W2 **Fixed in W2 push 1 (2026-11-11)**: aggregate_fidelity counts only string verdicts; an unhashable verdict counts as unobservable |
| W9-S4-2 | Exported `predict_stop_outcome` raises raw `TypeError` on non-string stop texts (callers validate upstream today) | **fixed-with-evidence** | W2 **Fixed in W2 push 1 (2026-11-11)**: predict_stop_outcome validates stop texts at entry and raises the module's own invalid_stop_texts code for non-string members |

### 2.5 Status counts at baseline

The ledger carries **52 rows**: 18 RR + 5 SA + 4 SCI-B + 6 semantic
siblings + 7 RTI + 6 ORCH + 6 wave-9. Of these: **49 open**,
**1 blocked-external** (RR-05), **2 design-debt-accepted**
(ORCH-D01, ORCH-D02), **0 fixed** — no finding in this ledger has
been fixed yet; the program starts here. Every open row names its
owning work package in its table. Post-baseline status changes are
recorded in the rows themselves; the first is the semantic-key
repair of 2026-11-11 (SCI-B-01 and its six §2.3 siblings: 7 rows
now fixed-with-evidence, 42 open). The second is W2 push 1
(2026-11-11, input-boundary totality): RR-02, SA-01, SA-03,
RTI-03, RTI-04, RTI-06 and the six wave-9 rows of §2.4 — 12 rows
fixed-with-evidence (19 fixed, 30 open). The third is W2 push 2
(2026-11-11, exact-type admission): RR-04, SA-04, SA-05 —
3 rows fixed-with-evidence (22 fixed, 27 open). The fourth is W2 push 3
(2026-11-12, negation coverage): RR-03 — 1 row fixed-with-evidence
(23 fixed, 26 open). The fifth is W2 push 4 (2026-11-12, comparison
evidence states): RR-09, ORCH-04 — 2 rows fixed-with-evidence
(25 fixed, 24 open).

### 2.6 Independent re-verification (audit-verify lane — landed)

The audit-verify lane re-executed every audit reproduction and every
wave-9 reproducer against this baseline in an isolated copy (each
reproducer at least twice in fresh processes, valid controls
included; its ledger and 20 reproducers are preserved outside the
repo). Result over its 37 rows: **36 OPEN reproduced, 0 FIXED,
1 NOT-REPRODUCED** (ORCH-D02's defect half, §2.2 — the credited
cleanup works). No status in this ledger changed as a result: every
reproduced row confirms the audit, and the lane's refinements are
folded into the rows they sharpen (RR-01 canary class and history
mode; RR-07 premise and pin census; RR-09 section-dependence;
RR-15 control instability; RR-16 exact generated-file count and the
SECURITY.md line; RR-17 occurrence arithmetic and the `skipcq`
zero). One numerical note: the lane's ORCH-02 trace shows 4
generation calls after the owner event was set (3 solo + 1 batched)
where the audit reported 3 — the disconnection claim is unaffected.
The lane did not cover RR-05 (blocked-external), RR-06, RR-08,
RR-10 – RR-14 or RR-18; those rows rest on the audit's evidence and
W0's census, as their verification cells state.

## 3. Readiness entry-point design (proposed, not built)

**There is no `tools/local_ci.py` or equivalent in this repo, and no
readiness verdict has ever been issued by one.** What follows is the
W0 design; W1 builds the security-fence pieces, W8 runs the finished
entry point as the release rehearsal. Until then, readiness is
assessed exactly as this ledger does: per-lane evidence, read by a
human, with missing evidence counted as non-passing.

**Proposed form:** `python tools/local_ci.py --sha <full-sha>
[--json-out <path>]`, runnable from a clean checkout, refusing to run
against a dirty tree.

**Lanes it must mirror** (each produces its own verdict; no lane may
borrow another's evidence):

1. `static` — ruff check and ruff format over both scopes (CI scope
   and the bare repo), mypy and mypy `--strict`, bandit exit 0.
2. `suite` — full pytest, the exact CI command (`coverage run -m
   unittest discover -s tests -t .`), and precise coverage from
   `coverage json` against the floor recorded in this ledger.
3. `docs` — `mkdocs build --strict` plus the docs-coherence tests.
4. `security` — the gitleaks sensitivity self-test (canary detected
   by the configured command; W1), pip-audit over every manifest,
   and the code-scanning alert inventory read from the API.
5. `analyzers` — DeepSource inventory bound to the exact SHA
   (currently **blocked-external**: quota exhausted; the lane reports
   `blocked`, which is non-passing, never `skipped`-as-pass), CodeQL
   conclusions at the SHA.
6. `artifacts` — wheel and sdist built twice from clean inputs, hash
   equality, twine check, and the installed-artifact smoke run from
   an empty working directory with the checkout off the import path
   (W5 contract).
7. `platforms` — the supported cells of the stability contract §1.5,
   each either executed locally or read back from CI **by exact SHA**
   (a cell that cannot be evidenced is `missing`, non-passing).
8. `ci-readback` — every workflow's conclusion at the exact SHA via
   the API (not the runs list): CI, Docs, CodeQL, Secret scan,
   Scorecard, Cross-OS smoke, Findings guard.

**Verdict format** (JSON): `{sha, tool_version, lanes: [{id, status,
evidence}], overall}` where `status` ∈ `pass | fail | blocked |
missing` and each lane's `evidence` names the command or run ID and
the artifact hashes it relied on. **Fail-closed rules:** `overall` is
`ready` only if every required lane is `pass` with evidence bound to
the same SHA; `blocked`, `missing`, stale (evidence bound to another
SHA), cancelled and skipped are all non-passing and are reported with
their reason. The tool never tags, releases, uploads or dispatches
anything; its only write is the verdict file the caller names.

## 4. Program state

W0 (this census) is the only completed work package, and it changes
no product code. The baseline is **not release-ready**: 49 findings
stand open, one is blocked on an external quota, and the readiness
entry point does not exist yet. Next, per the plan's dependency
order: W1 (security controls and the non-publishing release fence),
with the semantic-key fix cycle (§2.3) and W2's first correctness
wave alongside it.
