# Tutorials

A tutorial is a lesson: it takes you through a complete first
experiment in order, and every step is a documented command. Work
through these in sequence the first time; afterwards the
[how-to guides](../how-to/index.md) and the
[reference](../reference/index.md) are the faster routes.

## 1. Install and look at the catalog

Install the package and list the format catalog — the 29 conversation
formats the lab can render — then inspect the bundled example
fixtures. This is the quick start in the project README, and
[USAGE §1](../USAGE.md) documents the fixture schema the `inspect`
output summarises: two partitions (`search` and `held_out`), strict
message alternation, and the contamination check that runs over every
dataset.

## 2. Pre-flight a checkpoint with `doctor`

Before any model run, `doctor` validates a checkpoint directory and a
fixture file without loading weights, and — when everything passes —
suggests the exact `run` command to use next. [USAGE §2](../USAGE.md)
documents the checks and the suggestion rules. If you have no
checkpoint handy, generate the synthetic sample report instead
(`examples/make_sample_report.py` in the repository): a scripted
backend stands in for a model, the output is clearly labeled, and it
shows the report shape every real run produces.

## 3. Run the experiment

[USAGE §4–§5](../USAGE.md) walk through the `run` invocation: family,
mutation, boundary IDs, token budgets. Boundary token IDs must come
from the checkpoint's actual tokenizer — `doctor` reports the
tokenizer's declared `added_tokens` to help choose them.

## 4. Read the report

[USAGE §6](../USAGE.md) documents the outputs: `results.json` (the
decision, per-candidate verdicts, counts and token IDs) and the
self-contained HTML report. The decision is a single selected format
or an abstention — ties and failures abstain rather than crown a
winner, and the [FAQ](../FAQ.md) explains the common verdicts.

## 5. Go further

- Author your own fixtures from a spreadsheet: the
  [how-to guide](../how-to/index.md) routes you to `fixtures build`.
- Probe one behaviour at a time: the probe sections of USAGE §8 each
  isolate a single surface variable (stop behaviour, position,
  numeric form, negation, and more).
- Understand what a result can and cannot prove:
  [Explanation](../explanation/index.md).
