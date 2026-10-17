"""Cycle 117: local-only model loading invariant audit.

Background: CVE-2026-80047 (Hugging Face transformers 4.49.0-5.8.1, CERT/CC)
— the library wrote attacker-controlled Python files from a model
repository to the local disk (``~/.cache/huggingface/modules``) *before*
evaluating the user's trust consent prompt. ``get_cached_module_file()``
copied ``custom_generate/generate.py`` unconditionally; declining the
``trust_remote_code`` prompt left the files behind, and reused cache paths
could execute them later. The lab is architecturally immune by construction
— it never loads remote repositories, never enables ``trust_remote_code``,
and passes ``local_files_only=True`` — but nothing in the test suite pinned
that down. A future edit that "simplifies" a ``from_pretrained`` call by
dropping a keyword would silently reopen the exact attack class the CVE
describes, and CI would stay green (CI installs from the local tree, so the
call still resolves).

This cycle pins the invariant with AST scans over the package:

1. **No ``trust_remote_code=True`` anywhere.** Every keyword occurrence in
   ``model_protocol_lab/`` must be the literal ``False``.
2. **Every ``from_pretrained(...)`` call passes ``local_files_only=True``.**
   Local-only loading is what keeps transformers from touching the hub or
   the module cache at all.
3. **No remote-download helpers.** ``snapshot_download`` /
   ``hf_hub_download`` / direct ``huggingface_hub`` imports must not appear
   in the package — the only sanctioned entry points are the two
   ``from_pretrained`` calls in ``backends.py``.
4. **No truthy ``token=``.** Authentication tokens are never passed to
   transformers calls (``token=False`` is the local-only convention); a
   string or ``True`` would imply hub access.
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


class TestTrustRemoteCodeNeverEnabled(unittest.TestCase):
    """trust_remote_code=True must never appear in the package."""

    def test_no_trust_remote_code_true(self) -> None:
        offenders: list[str] = []
        for path, tree in _package_trees():
            for node in ast.walk(tree):
                if not isinstance(node, ast.keyword):
                    continue
                if node.arg != "trust_remote_code":
                    continue
                value = node.value
                is_false = isinstance(value, ast.Constant) and value.value is False
                if not is_false:
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(
            offenders,
            [],
            "trust_remote_code must only ever be False (CVE-2026-80047 class):\n"
            + "\n".join(offenders),
        )


class TestFromPretrainedIsLocalOnly(unittest.TestCase):
    """Every from_pretrained call must pass local_files_only=True."""

    def test_all_from_pretrained_calls_are_local_only(self) -> None:
        offenders: list[str] = []
        for path, tree in _package_trees():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = ""
                if isinstance(func, ast.Attribute):
                    name = func.attr
                elif isinstance(func, ast.Name):
                    name = func.id
                if name != "from_pretrained":
                    continue
                has_local_only = any(
                    kw.arg == "local_files_only"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                    for kw in node.keywords
                )
                if not has_local_only:
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertNotEqual(
            [p.name for p, _ in _package_trees() if "from_pretrained" in p.read_text()],
            [],
            "expected at least one from_pretrained call site to audit",
        )
        self.assertEqual(
            offenders,
            [],
            "from_pretrained without local_files_only=True:\n" + "\n".join(offenders),
        )


class TestNoRemoteDownloadHelpers(unittest.TestCase):
    """snapshot_download / hf_hub_download must not appear in the package."""

    def test_no_remote_download_helpers(self) -> None:
        offenders: list[str] = []
        for path, tree in _package_trees():
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and "huggingface_hub" in node.module
                ):
                    offenders.append(f"{path.name}:{node.lineno} (from-import)")
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if "huggingface_hub" in alias.name:
                            offenders.append(f"{path.name}:{node.lineno} (import)")
                if isinstance(node, ast.Call):
                    func = node.func
                    name = ""
                    if isinstance(func, ast.Attribute):
                        name = func.attr
                    elif isinstance(func, ast.Name):
                        name = func.id
                    if name in {"snapshot_download", "hf_hub_download"}:
                        offenders.append(f"{path.name}:{node.lineno} ({name})")
        self.assertEqual(
            offenders,
            [],
            "remote download helpers must not appear in the package:\n" + "\n".join(offenders),
        )


class TestNoTruthyToken(unittest.TestCase):
    """token= must never carry a truthy value (hub access implies network)."""

    def test_no_truthy_token_keyword(self) -> None:
        offenders: list[str] = []
        for path, tree in _package_trees():
            for node in ast.walk(tree):
                if not isinstance(node, ast.keyword):
                    continue
                if node.arg != "token":
                    continue
                value = node.value
                is_false = isinstance(value, ast.Constant) and value.value is False
                is_none = isinstance(value, ast.Constant) and value.value is None
                if not (is_false or is_none):
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(
            offenders,
            [],
            "token= must be False/None (never a string/True):\n" + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
