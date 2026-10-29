"""Repo-root launcher for the shipped surface-noise MCP server.

The server itself lives in :mod:`model_protocol_lab.mcp_server`
(shipped in the wheel with the ``model-protocol-lab-mcp`` console
script). This file exists only so ``python
tools/surface_noise_mcp_server.py`` keeps working from a checkout,
including the test suite's subprocess spawns. It contains no server
logic of its own: the deferred import in :func:`main` is the whole
implementation.

The repository root is added to ``sys.path`` so the launcher also
works from a bare checkout where the package is not installed; in
an installed environment that insert is a harmless no-op.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    """Run the shipped surface-noise MCP server."""
    from model_protocol_lab.mcp_server import serve

    return serve()


if __name__ == "__main__":
    sys.exit(main())
