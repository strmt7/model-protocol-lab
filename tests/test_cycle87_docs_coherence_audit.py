"""Cycle 87: adversarial docs-coherence audit.

METHOD
------
This file pins, by execution rather than by eyeballing, the user-facing
documentation's claims about repository assets. The audited docs are
``README.md``, ``VALIDATION.md``, ``docs/USAGE.md``, ``docs/SECURITY.md``,
``CHANGELOG.md``, ``docs/RELEASE_NOTES_0.1.0.md`` and ``docs/FAQ.md``:

1. Every relative ``examples/``, ``docs/``, ``model_protocol_lab/`` or
   ``tests/`` path mentioned in those docs resolves to a real file or
   directory in the repo tree (URL path fragments and glob patterns such as
   ``tests/test_cycle14_*`` are recognised and excluded).
2. Every flag used by a documented CLI invocation exists in the live
   ``build_parser()`` argparse tree for that subcommand; positional
   arguments are checked too.
3. The example assets are valid: ``examples/fixtures.json`` parses via
   ``datasets.load_dataset`` (a ``Split``, not a tuple), and
   ``examples/fixtures.csv`` builds through the same pipeline the
   ``fixtures build`` subcommand uses, producing the same fixture IDs.
4. The documented ``debug`` REPL session transcript in ``docs/USAGE.md``
   is reproduced verbatim via ``--script`` (non-interactive), pinning the
   deterministic output lines.

FINDINGS (2026-10-16)
---------------------
- No broken path references: all 33 extracted relative paths resolve
  (3 initial candidates were false positives: one URL fragment and two
  glob patterns).
- No broken CLI flags: every documented flag exists in the parser
  (including ``diff-reports``'s positionals, which argparse handles).
- Example assets valid: 2 search + 2 held-out fixtures in both JSON and
  CSV builds, with identical fixture IDs.
- ``pip-audit --desc``: no known vulnerabilities; ``bandit -r``: 0 issues.
- SECURITY.md torch CVE note re-verified against public sources: the repo
  pins torch 2.14.1+cpu, postdating the CVE-2025-3000 fix (torch>=2.13.0),
  and no ``torch.export.*``/bare ``torch.load`` call exists, so the
  CVE-2026-4538 not-reachable claim holds.

No doc fixes were required; this file records the sweep so regressions in
doc/asset coherence fail loudly.

Verified by running: 2026-10-16.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parent.parent
# NOTE (cycle 92): no os.chdir / sys.path mutation at import time. An earlier
# revision did both; mutating process-global CWD/sys.path during unittest
# discovery is an import-order hazard (it changes name resolution for every
# test module imported afterwards). All paths here are anchored at REPO_ROOT
# and subprocesses get an explicit cwd=, so the mutations were unnecessary.

from model_protocol_lab import datasets, fixtures  # noqa: E402
from model_protocol_lab.cli import build_parser  # noqa: E402

AUDITED_DOCS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "VALIDATION.md",
    REPO_ROOT / "docs" / "USAGE.md",
    REPO_ROOT / "docs" / "SECURITY.md",
    REPO_ROOT / "CHANGELOG.md",
    REPO_ROOT / "docs" / "RELEASE_NOTES_0.1.0.md",
    REPO_ROOT / "docs" / "FAQ.md",
]

_PATH_RE = re.compile(r"\b(?:examples|docs|model_protocol_lab|tests)/[\w\-\./]+\b")
_FLAG_RE = re.compile(r"--[a-z][a-z0-9\-]*")

# Documented invocations (from the audited docs) paired with their subcommand.
# Ellipsis commands ("run ...") carry the flags actually shown; checkpoint
# paths are never executed here, only their flags are checked.
DOCUMENTED_INVOCATIONS = [
    ("catalog", ["python -m model_protocol_lab catalog"]),
    ("inspect", ["python -m model_protocol_lab inspect examples/fixtures.json"]),
    (
        "doctor",
        [
            "python -m model_protocol_lab doctor --model path/to/checkpoint"
            " --dataset examples/fixtures.json",
            "python -m model_protocol_lab doctor --model PRIMARY"
            " --portability-matrix SECOND_CHECKPOINT",
            "python -m model_protocol_lab doctor --model PRIMARY --reference-model REFERENCE",
        ],
    ),
    (
        "run",
        [
            "python -m model_protocol_lab run --model LOCAL_MODEL_DIRECTORY"
            " --dataset YOUR_FIXTURES.json --family chatml --mutate"
            " --boundary-ids TOKEN_ID --max-new-tokens 64 --max-context 2048"
            " --device cpu --output NEW_REPORT_DIRECTORY",
            "python -m model_protocol_lab run --family chatml --mutate",
            "python -m model_protocol_lab run --sensitivity-sweep --max-drop-bytes 3",
            "python -m model_protocol_lab run --logprob-diagnostic",
        ],
    ),
    (
        "fixtures build",
        ["python -m model_protocol_lab fixtures build examples/fixtures.csv --output OUT"],
    ),
    (
        "diff-reports",
        ["python -m model_protocol_lab diff-reports OLD_RESULTS NEW_RESULTS"],
    ),
    (
        "diff-backends",
        [
            "python -m model_protocol_lab diff-backends --model path/to/checkpoint"
            ' --prompt-text "the answer is" --output-text "42 STOP"'
            ' --stops "STOP"'
        ],
    ),
    (
        "debug",
        [
            "python -m model_protocol_lab debug --dataset examples/fixtures.json",
            "python -m model_protocol_lab debug --dataset examples/fixtures.json"
            " --model path/to/checkpoint",
            "python -m model_protocol_lab debug --dataset examples/fixtures.json"
            " --script session.txt",
        ],
    ),
]


def _subcommand_parsers() -> dict[str, argparse.ArgumentParser]:
    """Map 'fixtures build'-style paths to their argparse subparsers."""
    root = build_parser()
    out: dict[str, argparse.ArgumentParser] = {}
    stack: list[tuple[str, argparse.ArgumentParser]] = [("", root)]
    while stack:
        path, parser = stack.pop()
        for action in parser._actions:  # noqa: SLF001
            if isinstance(action, argparse._SubParsersAction):
                choices = action.choices
                assert choices is not None
                for name, sub in choices.items():
                    key = name if not path else f"{path} {name}"
                    sub_parser = cast(argparse.ArgumentParser, sub)
                    out[key] = sub_parser
                    stack.append((key, sub_parser))
    return out


def _subcommand_flags(
    sub_parser: argparse.ArgumentParser,
) -> tuple[set[str], set[str]]:
    flags: set[str] = set()
    positionals: set[str] = set()
    for action in sub_parser._actions:  # noqa: SLF001
        if action.option_strings:
            flags.update(action.option_strings)
        else:
            positionals.add(action.dest)
    return flags, positionals


class TestDocumentedPathsExist(unittest.TestCase):
    def test_every_documented_path_resolves(self) -> None:
        missing: list[str] = []
        checked = 0
        for doc in AUDITED_DOCS:
            text = doc.read_text(encoding="utf-8")
            for match in _PATH_RE.finditer(text):
                raw = match.group(0)
                path = raw.rstrip(".")
                if "*" in path:
                    continue  # glob pattern, not a concrete path claim
                if text[match.end() : match.end() + 1] == "*":
                    continue  # regex stopped at the "*"; a glob prefix
                before = text[max(0, match.start() - 40) : match.start()]
                if "://" in before or re.search(r"[a-z0-9]\.[a-z]{2,}/$", before):
                    continue  # path fragment inside a URL, not a repo path
                target = REPO_ROOT / path
                checked += 1
                if not target.exists():
                    missing.append(f"{doc.name}: {path}")
        self.assertGreater(checked, 0, "no paths extracted from docs")
        self.assertEqual(missing, [], f"broken path references: {missing}")


class TestDocumentedFlagsExist(unittest.TestCase):
    def test_documented_flags_match_live_parser(self) -> None:
        subs = _subcommand_parsers()
        problems: list[str] = []
        for sub_name, invocations in DOCUMENTED_INVOCATIONS:
            self.assertIn(sub_name, subs, f"subcommand missing from parser: {sub_name}")
            flags, positionals = _subcommand_flags(subs[sub_name])
            for cmd in invocations:
                for flag in set(_FLAG_RE.findall(cmd)):
                    if flag not in flags:
                        problems.append(f"{cmd}: unknown flag {flag}")
        self.assertEqual(problems, [], f"doc/CLI flag drift: {problems}")

    def test_diff_reports_positionals_exist(self) -> None:
        subs = _subcommand_parsers()
        _, positionals = _subcommand_flags(subs["diff-reports"])
        self.assertIn("old_results", positionals)
        self.assertIn("new_results", positionals)


class TestExampleAssetsValid(unittest.TestCase):
    def test_fixtures_json_parses(self) -> None:
        split = datasets.load_dataset(REPO_ROOT / "examples" / "fixtures.json")
        self.assertEqual(len(split.search), 2)
        self.assertEqual(len(split.held_out), 2)

    def test_fixtures_csv_builds_with_same_fixture_ids(self) -> None:
        text = fixtures.read_authoring_file(REPO_ROOT / "examples" / "fixtures.csv")
        split, empty_ids = fixtures.parse_fixture_csv(text)
        self.assertEqual(empty_ids, [])
        self.assertEqual(len(split.search), 2)
        self.assertEqual(len(split.held_out), 2)
        from_json = datasets.load_dataset(REPO_ROOT / "examples" / "fixtures.json")
        self.assertEqual(
            sorted(f.id for f in split.search),
            sorted(f.id for f in from_json.search),
        )
        self.assertEqual(
            sorted(f.id for f in split.held_out),
            sorted(f.id for f in from_json.held_out),
        )

    def test_fixtures_build_cli_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(prefix=f"mpl-cycle87-{os.getpid()}-") as tmp:
            out = Path(tmp) / "fixtures.json"
            proc = subprocess.run(  # noqa: S603
                [
                    sys.executable,
                    "-m",
                    "model_protocol_lab",
                    "fixtures",
                    "build",
                    "examples/fixtures.csv",
                    "--output",
                    str(out),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            rebuilt = datasets.load_dataset(out)
            self.assertEqual(len(rebuilt.search), 2)
            self.assertEqual(len(rebuilt.held_out), 2)


class TestDocumentedDebugTranscript(unittest.TestCase):
    def test_scripted_session_matches_usages_example(self) -> None:
        with tempfile.TemporaryDirectory(prefix=f"mpl-cycle87-{os.getpid()}-") as tmp:
            script = Path(tmp) / "session.txt"
            script.write_text(
                ".fixture 1\n.render\n.boundary 8\n.step 1,2,8,3\n.quit\n",
                encoding="utf-8",
            )
            proc = subprocess.run(  # noqa: S603
                [
                    sys.executable,
                    "-m",
                    "model_protocol_lab",
                    "debug",
                    "--dataset",
                    "examples/fixtures.json",
                    "--script",
                    str(script),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            out = proc.stdout
            # Lines copied verbatim from docs/USAGE.md's example session.
            for line in (
                "selected fixture 1: search-copy-blue",
                "render: family=chatml fixture=search-copy-blue messages=1 chars=91",
                "<|im_start|>user",
                "Reply with exactly BLUE and nothing else.<|im_end|>",
                "[0] bos              chars 0:0",
                "[1] user_prefix      chars 0:17",
                "[2] user_content     chars 17:58",
                "[3] user_turn_end    chars 58:69",
                "[4] open_suffix      chars 69:91",
                "boundary set: [8]",
            ):
                self.assertIn(line, out, f"doc transcript drift: {line!r}")


if __name__ == "__main__":
    unittest.main()
