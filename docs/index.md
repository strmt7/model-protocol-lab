# Model Protocol Lab — documentation

Model Protocol Lab answers one question with measurements: is a local
language model failing because of its weights, or because its
conversations are formatted incorrectly? It renders chat-format
hypotheses — a 29-family catalog, plus bounded BOS, prefix and
thinking-mode mutations — as exact token sequences, sends them to a
local checkpoint, and compares task answers and response boundaries
token-for-token. A strict search/held-out fixture discipline selects a
single winning format or abstains.

This documentation is organised in four parts, by what you are trying
to do:

- **[Tutorials](tutorials/index.md)** — learning-oriented. A guided
  path through a first experiment, step by step.
- **[How-to guides](how-to/index.md)** — task-oriented. Short routes
  from a concrete job (choosing boundary IDs, authoring fixtures,
  comparing two runs) to the section that documents it.
- **[Reference](reference/index.md)** — information-oriented. The
  complete usage reference, the security and data-handling record,
  and the validation evidence.
- **[Explanation](explanation/index.md)** — understanding-oriented.
  Why the lab measures what it measures, and the design record behind
  each probe.

New here? Start with the [tutorial path](tutorials/index.md). Running
a model today? Go straight to the [usage reference](USAGE.md) — and
read it before your first run: boundary token IDs must be established
from the actual checkpoint tokenizer.

The project itself lives at
[strmt7/model-protocol-lab](https://github.com/strmt7/model-protocol-lab);
the validation record is
[VALIDATION.md](https://github.com/strmt7/model-protocol-lab/blob/main/VALIDATION.md)
and the change history is the
[CHANGELOG](https://github.com/strmt7/model-protocol-lab/blob/main/CHANGELOG.md).
