"""Cycle 105: optional-dependency audit.

Background: the base package declares ``dependencies = []`` — it is meant to
be importable and usable (catalog/inspect/fixtures/diff) with zero
third-party packages. The ``hf`` extra (torch/transformers/safetensors) is
only needed for real model backends, and every third-party import must be
deferred (function-local) or ``TYPE_CHECKING``-guarded so the base install
never hard-fails at import time.

This cycle pins two things:

1. **No module-level third-party imports.** An AST scan over the package
   asserts every top-level ``import``/``from`` is either stdlib, relative,
   or inside an ``if TYPE_CHECKING:`` block. A stray top-level
   ``import torch`` would break every base install while passing CI (CI
   installs the dev extras, so the import would resolve there).

2. **Actionable missing-extra error.** ``LocalHFBackend`` used to collapse
   a missing ``hf`` extra into the generic
   ``local_backend_initialization_failed``. It now raises the distinct
   ``local_hf_extra_not_installed`` code, and the CLI renders it with the
   install hint instead of a dead-end message.

Implementation note: the missing-extra simulation patches
``builtins.__import__`` to raise ``ImportError`` for the hf top-levels. It
deliberately does NOT delete entries from ``sys.modules`` — evicting C
extension modules (torch) and letting a later test re-import them
segfaults the interpreter.
"""

from __future__ import annotations

import ast
import builtins
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = REPO_ROOT / "model_protocol_lab"

# Top-level names the optional hf extra provides.
_HF_TOP_LEVELS = {"torch", "transformers", "safetensors"}


def _stdlib_names() -> set[str]:
    """Stdlib names."""
    # sys.stdlib_module_names exists on 3.10+; the package targets 3.11+.
    return set(sys.stdlib_module_names)


def _block_hf_imports():
    """Context manager making the hf extra unimportable (safe variant).

    Patches ``builtins.__import__`` so ``import torch`` /
    ``from transformers import ...`` raise ``ImportError``. ``sys.modules``
    is left untouched: deleting C extension modules and re-importing them
    later segfaults.
    """
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Fake import."""
        # noqa: ANN001, ANN202
        if name.split(".")[0] in _HF_TOP_LEVELS:
            raise ImportError(f"No module named {name!r} (simulated)")
        return real_import(name, globals, locals, fromlist, level)

    return mock.patch("builtins.__import__", side_effect=fake_import)


class TestNoModuleLevelThirdPartyImports(unittest.TestCase):
    """AST guard: third-party imports must be deferred or TYPE_CHECKING-only."""

    def test_package_has_no_top_level_third_party_imports(self) -> None:
        """Test package has no top level third party imports."""
        stdlib = _stdlib_names()
        violations: list[str] = []
        for path in sorted(PKG_ROOT.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        if isinstance(node, ast.Import):
                            top = alias.name.split(".")[0]
                        else:
                            top = (node.module or "").split(".")[0]
                        if not top or top in stdlib or top in ("model_protocol_lab",):
                            continue
                        # Relative imports (level > 0) are first-party.
                        if isinstance(node, ast.ImportFrom) and node.level:
                            continue
                        violations.append(f"{path.name}:{node.lineno}: {top}")
                elif isinstance(node, ast.If):
                    # if TYPE_CHECKING: blocks may import anything.
                    test = node.test
                    is_tc = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
                    )
                    if not is_tc:
                        # A non-TYPE_CHECKING module-level if containing a
                        # third-party import would still execute at import.
                        for child in ast.walk(node):
                            if not isinstance(child, (ast.Import, ast.ImportFrom)):
                                continue
                            if child is node:
                                continue
                            for alias in child.names:
                                if isinstance(child, ast.Import):
                                    mod = alias.name.split(".")[0]
                                else:
                                    mod = (child.module or "").split(".")[0]
                                if mod and mod not in stdlib and mod != "model_protocol_lab":
                                    violations.append(
                                        f"{path.name}:{child.lineno}: {mod} (in module-level if)"
                                    )
        self.assertEqual(violations, [], f"module-level third-party imports: {violations}")

    def test_hf_imports_are_present_but_deferred(self) -> None:
        """Test hf imports are present but deferred."""
        # Negative control: the scan above must actually see the hf imports
        # somewhere (deferred), otherwise it proves nothing. Only torch and
        # transformers are imported directly; safetensors is a transitive
        # dependency declared in the hf extra but never imported by name.
        found: set[str] = set()
        for path in sorted(PKG_ROOT.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    found.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    found.add(node.module.split(".")[0])
        for name in ("torch", "transformers"):
            self.assertIn(name, found, f"expected deferred import of {name} somewhere")


class TestMissingHfExtraFailsActionable(unittest.TestCase):
    """With the hf extra blocked, backend init must name the missing extra."""

    @staticmethod
    def _healthy_root() -> Path:
        """Healthy root."""
        root = Path(tempfile.mkdtemp(prefix="mpl_cycle105_"))
        (root / "config.json").write_text(json.dumps({"model_type": "gpt2", "n_positions": 1024}))
        (root / "tokenizer.json").write_text(json.dumps({"model": {}}))
        (root / "model.safetensors").write_bytes(b"\x00" * 32)
        return root

    def test_missing_extra_raises_distinct_code(self) -> None:
        """Test missing extra raises distinct code."""
        from model_protocol_lab.backends import LocalHFBackend

        with _block_hf_imports(), self.assertRaises(RuntimeError) as ctx:
            LocalHFBackend(self._healthy_root())
        self.assertEqual(str(ctx.exception), "local_hf_extra_not_installed")

    def test_blocked_import_actually_blocks(self) -> None:
        """Test blocked import actually blocks."""
        # Negative control: the simulation must really block the import.
        with _block_hf_imports(), self.assertRaises(ImportError):
            import torch  # noqa: F401

    def test_distinct_code_is_not_the_generic_one(self) -> None:
        """Test distinct code is not the generic one."""
        # Guards against a future refactor collapsing the codes again.
        from model_protocol_lab import backends

        src = Path(backends.__file__).read_text(encoding="utf-8")
        self.assertIn('"local_hf_extra_not_installed"', src)


class TestCliHint(unittest.TestCase):
    """The CLI must render the missing-extra code with an install hint."""

    def test_hint_names_the_extra(self) -> None:
        """Test hint names the extra."""
        from model_protocol_lab.cli import _backend_failure_message

        msg = _backend_failure_message("local_hf_extra_not_installed")
        self.assertIn("local_hf_extra_not_installed", msg)
        self.assertIn('pip install "model-protocol-lab[hf]"', msg)

    def test_other_local_codes_stay_terse(self) -> None:
        """Test other local codes stay terse."""
        from model_protocol_lab.cli import _backend_failure_message

        msg = _backend_failure_message("local_backend_initialization_failed")
        self.assertIn("[local_backend_initialization_failed]", msg)
        self.assertNotIn("pip install", msg)

    def test_arbitrary_text_never_leaks(self) -> None:
        """Test arbitrary text never leaks."""
        from model_protocol_lab.cli import _backend_failure_message

        evil = "oops'; rm -rf / #"
        msg = _backend_failure_message(evil)
        self.assertNotIn(evil, msg)
        self.assertNotIn("rm -rf", msg)


if __name__ == "__main__":
    unittest.main()
