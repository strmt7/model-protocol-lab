# Dependency license inventory

Generated 2026-11-01 (cycle 226, readiness item B-27) from the PyPI
JSON API (`https://pypi.org/pypi/<name>/<version>/json`), one entry
per package at the exact version the development lock
(`pylock.toml`) pins. License labels are the packages' own declared
metadata (SPDX `license_expression` where published, otherwise the
`License :: OSI Approved` classifier or the `license` field).
Refresh this file whenever the lock is refreshed (see the release
runbook); a version bump without a license re-check leaves this
inventory stale.

## Development set (locked in `pylock.toml`)

| Package | Version | License |
| --- | --- | --- |
| ast-serialize | 0.13.1 | MIT |
| coverage | 7.17.0 | Apache-2.0 |
| hypothesis | 6.170.14 | MPL-2.0 |
| iniconfig | 2.3.1 | MIT |
| librt | 0.16.0 | MIT |
| mypy | 2.4.1 | MIT |
| mypy-extensions | 1.1.0 | MIT (not stated in PyPI metadata; verified from the wheel's `dist-info/licenses/LICENSE` file) |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pathspec | 1.1.1 | MPL-2.0 |
| pluggy | 1.7.0 | MIT |
| pygments | 2.21.0 | BSD-2-Clause |
| pytest | 9.2.0 | MIT |
| ruff | 0.17.3 | MIT |
| sortedcontainers | 2.4.0 | Apache-2.0 |
| typing-extensions | 4.16.0 | PSF-2.0 |

All fifteen are permissive or file-level-copyleft (MPL-2.0)
licenses; none imposes source-disclosure obligations on this
project's own code. These packages are development tools only —
none is distributed with the library, whose runtime is standard
library only.

## The `[hf]` extra (range-pinned, not locked)

The optional Hugging Face adapter stack is not in the lock (see
`pylock.toml`'s header), so the versions below are the current
PyPI releases at inventory time, shown for license posture only:

| Package | PyPI release checked | License |
| --- | --- | --- |
| torch | 2.15.0 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib (the wheel's bundled-components expression) |
| transformers | 5.22.1 | Apache 2.0 License |
| safetensors | 0.8.0 | Apache Software License |
| jinja2 | 3.1.6 | BSD License |

## Tooling licenses vendored in this directory

The agent-tooling license texts kept alongside this inventory
(Caveman, CocoIndex Code, Crawl4AI) cover the research tooling
under `tools/`, not Python dependencies; see the individual files.
