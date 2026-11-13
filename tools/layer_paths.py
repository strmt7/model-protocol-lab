#!/usr/bin/env python3
"""Print the test modules of one W3-3 layer, one per line.

Who uses this on Monday morning: a contributor (or a CI job) who
needs to run only the tests that work in the environment at hand --
the base layer on a .[dev]-only or Windows setup, the HF layer where
the adapter stack is installed -- without hand-maintaining a file
list. The single source of truth is tests/test_layers.json (pinned
against an AST import scan by tests/test_w3_test_layers.py); this
tool only reads it.

Usage:
    python tools/layer_paths.py --layer base            # tests/test_*.py paths
    python tools/layer_paths.py --layer hf --format modules
    python -m pytest -q $(python tools/layer_paths.py --layer base)
    python -m unittest $(python tools/layer_paths.py --layer base --format modules)

Layers: base, hf, posix, excluded (hf + posix), all.
Formats: paths (tests/<file>) or modules (tests.<stem>, for unittest).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "tests" / "test_layers.json"
LAYERS = ("base", "hf", "posix", "excluded", "all")


def layer_members(manifest: dict, layer: str) -> list[str]:
    """Return the sorted module filenames for one layer."""
    layers = manifest["layers"]
    if layer in layers:
        return sorted(layers[layer])
    if layer == "excluded":
        return sorted(layers["hf"] + layers["posix"])
    if layer == "all":
        return sorted(name for members in layers.values() for name in members)
    raise ValueError(f"unknown layer: {layer!r} (expected one of {', '.join(LAYERS)})")


def main(argv: list[str] | None = None) -> int:
    """Run the layer-paths printer."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layer", required=True, choices=LAYERS)
    parser.add_argument("--format", choices=("paths", "modules"), default="paths")
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name in layer_members(manifest, args.layer):
        if args.format == "paths":
            print(f"tests/{name}")
        else:
            print(f"tests.{Path(name).stem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
