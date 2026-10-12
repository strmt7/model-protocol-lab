# Validation status

## Evidence available for this source snapshot

- Source-level review of renderer and evaluator invariants, failure handling, cancellation, split separation, report escaping and privacy defaults.
- Test source covering deterministic fake-backend behavior, invalid inputs, boundary truncation, partial runs, search selection, dataset parsing and report serialization.
- Source inspection and archive packaging checks are distinct from application execution.

## Not established

The application and its tests have not been executed. No package installation, import check, bytecode compilation, dependency resolution, CPU/GPU model run, browser rendering test, coverage measurement, mutation-testing run or benchmark has been performed for this snapshot. The presence of a test does not mean it passes. This is a source handoff, not a validated release.

## Important limitations

1. The local adapter's supported architecture and dependency combinations require integration testing. GPT-2/Llama-only loading is intentionally narrow; unsupported configurations should fail closed.
2. Exact string fixtures are strict behavioral probes, not general model-quality metrics. Group IDs and exact-duplicate checks do not establish semantic independence.
3. A unique all-pass search candidate can still be a false selection on an insufficient fixture set. There is no statistical confidence estimate.
4. The supplied example is only a schema demonstration. Build separate representative search and held-out families before drawing conclusions.
5. Full-prefix greedy decoding is slow and float32 weights can consume substantial memory. No general laptop-size or GPU-fit guarantee exists.
6. Cancellation is cooperative. A running kernel, model load, tokenizer operation or filesystem operation may not stop immediately.
7. Asset hashes provide identity checks, not a sandbox or a guarantee against concurrent file substitution. Use immutable trusted model assets.
8. Default JSON omits observations but still contains metadata and format strings. Opt-in evidence can reveal text through token IDs. There is no guaranteed secure erasure.
9. Report creation is not a multi-file atomic transaction. A write failure may leave a partial output directory; retry with a new directory after inspection.
10. Current counts do not measure model FLOPs, repeated prefix processing, total wall time or equal search budgets. Fair-compute comparisons are future work.
11. There is no live training instrumentation, arbitrary template inference, Jinja export, model-weight reverse engineering, distributed execution or production web service.

## Release gate

Do not tag this source as a validated release until the execution, dependency, integration, privacy, scientific-evaluation and reproducibility gates in DEVELOPMENT_PLAN.md are satisfied with retained evidence. Record failures as well as passing results. Do not weaken assertions merely to obtain a green suite.
