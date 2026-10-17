"""Cycle 119: pickle-deserialization loading invariant audit.

Background: CVE-2025-32434 (torch <2.6.0) and CVE-2026-24747 (torch <=2.9.1,
fixed in 2.10.0, CVSS 8.8) — both remote-code-execution flaws in the
``torch.load`` deserialization path. CVE-2026-24747 is the worse of the two:
it broke the ``weights_only=True`` unpickler itself (improper validation of
pickle opcodes and storage metadata), the mechanism explicitly intended to
load model data safely. The installed torch (2.14.1+cpu) is not affected by
either, but the lab's structural defense must not depend on the installed
version: a future edit that drops ``use_safetensors=True`` from a
``from_pretrained`` call would silently re-engage the pickle path for
``.bin`` checkpoints, and CI would stay green.

This cycle pins the invariant with AST scans over the package:

1. **No ``torch.load`` / ``pickle.load(s)`` anywhere in the package.** The
   only sanctioned deserialization path is transformers' ``from_pretrained``
   with safetensors forced.
2. **Every model-weight ``from_pretrained`` call passes
   ``use_safetensors=True``** (literal ``True``). With safetensors forced,
   transformers never touches the pickle unpickler, so the CVE-2025-32434 /
   CVE-2026-24747 classes are unreachable regardless of the installed
   torch version. (``AutoConfig.from_pretrained`` loads config.json — no
   weights — and is out of scope.)
3. **Every model-weight ``from_pretrained`` call passes
   ``weights_only=True``** (literal ``True``). Defense in depth: even if
   transformers ever fell back to ``torch.load``, the restricted unpickler
   is engaged.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = REPO_ROOT / "model_protocol_lab"


def _package_trees() -> list[tuple[Path, ast.Module]]:
    """Parse every package module; return (path, tree) pairs."""
    out: list[tuple[Path, ast.Module]] = []
    for path in sorted(PKG_ROOT.glob("*.py")):
        out.append((path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))))
    return out


def _is_model_from_pretrained(node: ast.Call) -> bool:
    """True for ``X.from_pretrained(...)`` where X looks like a model class.

    ``AutoConfig.from_pretrained`` loads config.json (no weights) and is
    out of scope; model classes (``AutoModelForCausalLM``, ...) carry
    "Model" in the name.
    """
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "from_pretrained":
        return False
    value = func.value
    return isinstance(value, ast.Name) and "Model" in value.id


def _has_literal_true_kw(node: ast.Call, name: str) -> bool:
    return any(
        kw.arg == name and isinstance(kw.value, ast.Constant) and kw.value.value is True
        for kw in node.keywords
    )


class TestNoTorchLoadInPackage(unittest.TestCase):
    """torch.load / pickle.load(s) must not appear in the package."""

    def test_no_torch_load_calls(self) -> None:
        offenders: list[str] = []
        for path, tree in _package_trees():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "load"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "torch"
                ):
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(
            offenders,
            [],
            "torch.load must not appear in the package "
            "(CVE-2025-32434 / CVE-2026-24747 class):\n" + "\n".join(offenders),
        )

    def test_no_pickle_load_calls(self) -> None:
        offenders: list[str] = []
        for path, tree in _package_trees():
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "pickle":
                    offenders.append(f"{path.name}:{node.lineno} (from pickle import)")
                    continue
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "pickle" or alias.name.startswith("pickle."):
                            offenders.append(f"{path.name}:{node.lineno} (import pickle)")
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr in ("load", "loads")
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "pickle"
                ):
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(
            offenders,
            [],
            "pickle.load(s) must not appear in the package:\n" + "\n".join(offenders),
        )


class TestModelLoadingForcesSafetensors(unittest.TestCase):
    """Every model-weight from_pretrained call must pass use_safetensors=True."""

    def test_all_model_from_pretrained_force_safetensors(self) -> None:
        offenders: list[str] = []
        sites = 0
        for path, tree in _package_trees():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not _is_model_from_pretrained(node):
                    continue
                sites += 1
                if not _has_literal_true_kw(node, "use_safetensors"):
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertGreater(sites, 0, "expected at least one model from_pretrained call site")
        self.assertEqual(
            offenders,
            [],
            "model from_pretrained without use_safetensors=True "
            "(pickle path would re-engage for .bin checkpoints):\n" + "\n".join(offenders),
        )


class TestModelLoadingWeightsOnly(unittest.TestCase):
    """Every model-weight from_pretrained call must pass weights_only=True."""

    def test_all_model_from_pretrained_weights_only(self) -> None:
        offenders: list[str] = []
        sites = 0
        for path, tree in _package_trees():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not _is_model_from_pretrained(node):
                    continue
                sites += 1
                if not _has_literal_true_kw(node, "weights_only"):
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertGreater(sites, 0, "expected at least one model from_pretrained call site")
        self.assertEqual(
            offenders,
            [],
            "model from_pretrained without weights_only=True "
            "(defense-in-depth against torch.load fallback):\n" + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
