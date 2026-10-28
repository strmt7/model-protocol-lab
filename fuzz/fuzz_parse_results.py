"""Fuzz the results loader's byte contract (cycle 217, ClusterFuzzLite).

Target: ``model_protocol_lab.diffreports.parse_results_bytes`` — the
shared bounded reader for ``results.json`` bytes (used by
``load_results``, ``pack``, and ``verify --bundle``). Every rejection
is a fixed ``ValueError`` code (oversize, non-UTF-8, duplicate fields,
non-finite numbers, excessive nesting, non-object roots, wrong shape);
only a non-``ValueError`` escape is a fuzzer finding, because it would
mean an input class the loader's contract does not cover.

Run standalone (atheris installed)::

    python fuzz/fuzz_parse_results.py -runs=1000 -max_len=1048576

The in-code length guard mirrors the loader's own cap so oversized
corpus entries are out of contract rather than findings.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import atheris  # noqa: E402

with atheris.instrument_imports():  # coverage-guided, not blind runs
    from model_protocol_lab.diffreports import (  # noqa: E402
        MAX_RESULTS_BYTES,
        parse_results_bytes,
    )


def TestOneInput(data: bytes) -> None:
    """Feed one input; fixed-vocabulary rejections are expected."""
    if len(data) > MAX_RESULTS_BYTES:
        return
    try:
        parse_results_bytes(data)
    except ValueError:
        return


def main() -> None:
    """Wire the target into atheris; crashes are the findings."""
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
