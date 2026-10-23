# Evidence files — attribution

The five chat templates in this directory are third-party files,
preserved byte-identical so the catalog families derived from them can
be re-verified against the exact upstream text at any time (the
derivations and checks are recorded in `docs/RESEARCH_LOG.md`). They
are measurement evidence, not part of this project's own code, and all
rights in them remain with their respective upstream authors. Each was
taken from the upstream Hugging Face repository named below; the
license shown is the license the upstream repository declares in its
Hugging Face metadata, re-checked 2026-10-23.

| File | Upstream source | Upstream license |
|---|---|---|
| `hy4-chat-template.jinja` | `tencent/Hy4-preview` (`chat_template.jinja`, md5 `a0b7e28386487d804009340155e7be2b`) | Apache-2.0 |
| `k2horizon-chat-template.jinja` | `IFM/K2-Horizon-375B-A23B` (`chat_template.jinja`, md5 `150b0fd46f7babfc6bfd8b769ad0475c`; one template shared across the K2-Horizon fleet) | Apache-2.0 |
| `ling31-chat-template.jinja` | `inclusionAI/Ling-3.1-flash` (`chat_template.jinja`, md5 `24b514df8b812d0bb0e9c061d0bb164c`) | MIT |
| `minimax-chat-template.jinja` | `MiniMaxAI/MiniMax-M3` (`chat_template.jinja`, md5 `455403f9cba16ea73388aeb8898da752`) | Custom terms declared by MiniMax (Hugging Face license field: `other`) — see the upstream repository |
| `nanbeige-chat-template.jinja` | `Nanbeige/Nanbeige4.2-3B` (the `chat_template` string embedded in `tokenizer_config.json`, extracted verbatim; the repo ships no standalone template file; extracted-file md5 `1fb98b0fe70936f799f9aa1602af5b3c`) | Apache-2.0 |

If you redistribute this repository, keep this attribution file with
the templates. If an upstream source or license above looks wrong,
open an issue — the md5 values let anyone re-derive each file's origin
exactly.
