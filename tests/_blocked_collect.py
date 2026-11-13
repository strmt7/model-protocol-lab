#!/usr/bin/env python3
"""Simulated-absence collection child (W3-3, productized from the
readiness lane's collect_blocked.py).

Blocks the manifest's ``blocked_in_simulation`` modules (torch,
numpy, transformers, tokenizers, resource) behind an import hook,
then runs ``pytest --collect-only`` over one layer's modules:

    python tests/_blocked_collect.py --layer base
    python tests/_blocked_collect.py --layer excluded

``base`` must collect with zero errors (the .[dev]-only / no-resource
collection guarantee); ``excluded`` collects the hf + posix modules
plus the control tests/test_workflows.py and must error on exactly
the hf modules -- the posix module collects cleanly since W3-3 (its
import of ``resource`` is conditional; its POSIX need is a runtime
skip, not a collection error) and the control collects too. That is
the negative control that proves the blocker actually blocks.

Fidelity note: a genuinely absent module makes ``import`` raise but
``importlib.util.find_spec`` return ``None`` (the suite's HAS_*
probes rely on the second shape). The meta-path blocker alone would
raise through find_spec too and over-block two dozen base modules,
so the child also wraps ``importlib.util.find_spec`` to return
``None`` for blocked roots -- both call shapes then match a real
absent-module environment.

Underscore-prefixed: neither test runner collects this file, and it
is outside the manifest's test_*.py universe. It is a subprocess
child by design -- the suite process already has the blocked modules
in sys.modules, so the simulation is only honest in a fresh
interpreter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
CONTROL = "test_workflows.py"


def main(argv: list[str] | None = None) -> int:
    """Run one blocked collection."""
    parser = argparse.ArgumentParser(description="Blocked-import collection simulation")
    parser.add_argument("--layer", choices=("base", "excluded"), required=True)
    args = parser.parse_args(argv)

    manifest = json.loads((TESTS_DIR / "test_layers.json").read_text(encoding="utf-8"))
    blocked = set(manifest["blocked_in_simulation"])
    layers = manifest["layers"]
    if args.layer == "base":
        names = list(layers["base"])
    else:
        names = list(layers["hf"]) + list(layers["posix"]) + [CONTROL]

    class Blocker:
        """Import hook raising ImportError for blocked roots."""

        def find_spec(self, name: str, path=None, target=None):  # noqa: ANN001, ANN202
            if name.split(".")[0] in blocked:
                raise ImportError(f"No module named '{name}' (simulated absence)")
            return None

    sys.meta_path.insert(0, Blocker())
    for name in list(sys.modules):
        if name.split(".")[0] in blocked:
            del sys.modules[name]

    import importlib.util

    real_find_spec = importlib.util.find_spec

    def find_spec(name: str, package: str | None = None):  # noqa: ANN202
        if name.split(".")[0] in blocked:
            return None
        return real_find_spec(name, package)

    importlib.util.find_spec = find_spec  # type: ignore[assignment]
    os.environ["MPL_BLOCKED_COLLECT_CHILD"] = "1"
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))

    import pytest

    return pytest.main(
        ["--collect-only", "-q", "-p", "no:cacheprovider", *[f"tests/{n}" for n in names]]
    )


if __name__ == "__main__":
    sys.exit(main())
