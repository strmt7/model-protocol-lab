"""Cycle 83: adversarial audit of CLI wiring below the cycle-64 surface.

Cycle 64 audited subcommand existence, help text, dispatch basics and the
fixed error-code vocabulary (110 tests, 1 doc fix). This file attacks the
wiring *beneath* that surface in ``model_protocol_lab/cli.py``:

1. argparse edge cases: conflicting/repeated flags, ``--flag=value`` vs
   ``--flag value``, unknown args, subcommand-before-global-flag ordering,
   ``=`` inside values, empty-string args, unicode args, flag-looking args
   (``-``, ``--``, ``---``), huge arg lists, NUL bytes.
2. The exit-code contract: every failure path exits non-zero with a
   machine-parseable stderr line (or, for ``doctor``, machine-parseable JSON
   on stdout); success exits 0. Exact codes (0/1/2/130) are pinned for bad
   subcommands, missing args, invalid enum values, unreadable inputs,
   unwritable outputs, malformed fixtures, nonexistent checkpoints and
   contradictory flags.
3. ``--help`` byte-stability across runs for every subcommand (no
   timestamps/PIDs/paths leaking in); the ``--version`` mandate assumption
   is checked against reality (finding 1 below).
4. Console-script equivalence: the ``model-protocol-lab`` entry point vs
   ``python -m model_protocol_lab`` on representative invocations including
   a failure path.
5. Environment interplay: ``MPL_*`` vars (finding 2) and a polluted
   ``sys.argv[0]``.

METHOD
------
Unit-level (imported ``build_parser`` / ``main`` with explicit argv lists,
never mocked ``sys.argv``) for pure argparse edges; subprocess
(``sys.executable -m model_protocol_lab``) for every exit-code / stderr /
stdout-routing pin, because those are real-process contracts. Handlers that
would load a model are never executed: the ``run`` decision-code pins reuse
the fake-backend pattern (patched ``LocalHFBackend`` / ``run_experiment`` /
reporting helpers), and the ``run`` output-claim pins stop before any
backend init. Hermetic: no network, no model downloads, PID-scoped
``TemporaryDirectory()`` only, no hard-coded ``/tmp`` paths, no exact
Python/torch/transformers version asserts, no sleeps.

FINDINGS
--------
CLEAN: no genuine wiring bug found. Every argparse edge behaves per stdlib
contract, every failure path exits non-zero with a fixed, machine-parseable
message (never a traceback, never leaked file content), help text is
byte-stable across runs, the console script is byte-identical to
``python -m``, ``MPL_*`` environment variables do not exist and cannot steer
the CLI, and a hostile ``sys.argv[0]`` cannot crash argument parsing (the
usage prog is hard-coded to ``model-protocol-lab``).

PRECISION NOTES (mandate assumptions checked against reality -- not bugs):
1. Mandate item 3 assumes a ``--version`` flag whose output must match the
   package ``__version__``. No such flag exists: the parser defines no
   ``--version`` option, and cycle 64 explicitly pins
   ``model-protocol-lab --version`` -> exit 2 (unrecognized top-level flag)
   in ``test_undefined_top_level_flag_exits_2``. Adding the flag now would
   break that pinned contract, so it is deliberately NOT added here; the
   version lives only in ``model_protocol_lab.__version__``. Pinned below
   by ``test_no_version_flag_top_level_help``.
2. No ``MPL_*`` environment variable is read anywhere in the package
   (verified by source scan and by behavioral test): a polluted environment
   cannot change any CLI behavior.
3. ``doctor --model <nonexistent>`` exits 1 with the verdict JSON on
   *stdout* and empty stderr. This is the deliberate ``doctor`` contract
   (stdout stays machine-parseable JSON; the exit code carries the
   verdict), not a missing-error-message bug.
4. ``debug --script <directory>`` echoes the operator-supplied path in its
   fixed ``debug: unreadable script file:`` stderr line. The path is
   operator input, not leaked file content; the prefix is fixed and the
   exit code is 1.
5. ``model-protocol-lab --output x catalog`` (global-flag-looking token
   before the subcommand) exits 2 with ``argument command: invalid choice:
   'x'`` -- stdlib argparse behavior with required subparsers, exit code
   still correct.

DELIBERATELY NOT VERIFIED
-------------------------
* The exit-130 SIGINT path (needs a live run plus real signal delivery).
* Interactive ``debug`` REPL stdin behavior (covered by cycle-64 read-only
  pins; wiring only here).

Verified by RUNNING this file: ``pytest -p no:cacheprovider
tests/test_cycle83_cli_wiring_audit.py`` -- 2026-10-16.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

import model_protocol_lab
from model_protocol_lab import backends, cli, experiments
from model_protocol_lab import doctor as doctor_mod
from model_protocol_lab import reporting as reporting_mod
from model_protocol_lab.cli import build_parser, main

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_FIXTURES = REPO_ROOT / "examples" / "fixtures.json"
EXAMPLE_CSV = REPO_ROOT / "examples" / "fixtures.csv"
CALL_TIMEOUT = 120


def _top_level_subcommands() -> list[str]:
    """Every registered top-level subcommand, derived from the parser itself."""
    parser = build_parser()
    subparsers: Any = parser._subparsers  # noqa: SLF001
    for action in subparsers._group_actions:  # noqa: SLF001
        if action.dest == "command":
            return sorted(action.choices.keys())
    raise AssertionError("no subparsers found")


SUBCOMMANDS = _top_level_subcommands()


def run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Invoke the real CLI entry point in a subprocess; never mocked."""
    cmd_env = dict(os.environ)
    if env:
        cmd_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "model_protocol_lab", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=CALL_TIMEOUT,
        env=cmd_env,
        check=False,
    )


def _stderr_tail(proc: subprocess.CompletedProcess[str]) -> str:
    """Stderr tail."""
    lines = proc.stderr.strip().splitlines()
    return lines[-1] if lines else ""


def _write_fixture(directory: str) -> Path:
    """Minimal schema-valid fixture JSON; returns its path."""
    path = Path(directory) / "fixture.json"
    path.write_text(
        json.dumps(
            {
                "search": [
                    {
                        "id": "s1",
                        "group": "g1",
                        "messages": [{"role": "user", "content": "hi"}],
                        "expected": "yo",
                    }
                ],
                "held_out": [
                    {
                        "id": "h1",
                        "group": "g2",
                        "messages": [{"role": "user", "content": "hey"}],
                        "expected": "yo",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _call_main(argv: list[str]) -> tuple[object, str, str]:
    """Call main() with an explicit argv; return (code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code: object = main(argv)
    except SystemExit as exc:
        code = f"SystemExit({exc.code})"
    return code, out.getvalue(), err.getvalue()


class ArgparseEdgeUnitTests(unittest.TestCase):
    """argparse wiring edges, in-process with explicit argv lists."""

    def test_repeated_scalar_flag_last_wins(self):
        """Test repeated scalar flag last wins."""
        args = build_parser().parse_args(
            [
                "run",
                "--model",
                "m",
                "--dataset",
                "d",
                "--family",
                "chatml",
                "--boundary-ids",
                "198",
                "--output",
                "o",
                "--max-new-tokens",
                "10",
                "--max-new-tokens",
                "20",
            ]
        )
        self.assertEqual(args.max_new_tokens, 20)

    def test_flag_equals_form_matches_space_form(self):
        """Test flag equals form matches space form."""
        parser = build_parser()
        via_equals = parser.parse_args(["doctor", "--model=/x/ckpt", "--dataset=/y/f.json"])
        via_space = parser.parse_args(["doctor", "--model", "/x/ckpt", "--dataset", "/y/f.json"])
        self.assertEqual(via_equals.model, via_space.model)
        self.assertEqual(via_equals.dataset, via_space.dataset)

    def test_equals_inside_value_preserved(self):
        """Test equals inside value preserved."""
        args = build_parser().parse_args(
            [
                "diff-backends",
                "--model",
                "m",
                "--prompt-text=a=b",
                "--output-text",
                "x",
                "--stops",
                "STOP",
            ]
        )
        self.assertEqual(args.prompt_text, "a=b")

    def test_empty_string_value_accepted_verbatim(self):
        """Test empty string value accepted verbatim."""
        args = build_parser().parse_args(
            [
                "diff-backends",
                "--model",
                "m",
                "--prompt-text",
                "",
                "--output-text",
                "x",
                "--stops",
                "STOP",
            ]
        )
        self.assertEqual(args.prompt_text, "")

    def test_unicode_value_preserved(self):
        """Test unicode value preserved."""
        args = build_parser().parse_args(
            [
                "diff-backends",
                "--model",
                "m",
                "--prompt-text",
                "héllo→✓",
                "--output-text",
                "x",
                "--stops",
                "STOP",
            ]
        )
        self.assertEqual(args.prompt_text, "héllo→✓")

    def test_dashdash_separator_takes_flag_looking_positional(self):
        """Test dashdash separator takes flag looking positional."""
        args = build_parser().parse_args(["inspect", "--", "-"])
        self.assertEqual(str(args.dataset), "-")

    def test_triple_dash_rejected(self):
        """Test triple dash rejected."""
        code, _, err = _call_main(["inspect", "---"])
        self.assertEqual(code, "SystemExit(2)")
        # stdlib argparse quirk: a subparser stashes unknown optionals in
        # the namespace instead of reporting them, so the missing positional
        # fires instead of "unrecognized arguments". Exit code is still 2.
        self.assertIn("the following arguments are required: dataset", err)

    def test_bare_dashdash_means_missing_command(self):
        """Test bare dashdash means missing command."""
        code, _, err = _call_main(["--"])
        self.assertEqual(code, "SystemExit(2)")
        self.assertIn("the following arguments are required: command", err)

    def test_unknown_subcommand_flag_rejected(self):
        """Test unknown subcommand flag rejected."""
        code, _, err = _call_main(["catalog", "--frobnicate"])
        self.assertEqual(code, "SystemExit(2)")
        self.assertIn("unrecognized arguments: --frobnicate", err)

    def test_extra_positional_rejected(self):
        """Test extra positional rejected."""
        code, _, err = _call_main(["catalog", "extra"])
        self.assertEqual(code, "SystemExit(2)")
        self.assertIn("unrecognized arguments: extra", err)

    def test_main_empty_argv_exits_2(self):
        """Test main empty argv exits 2."""
        code, out, err = _call_main([])
        self.assertEqual(code, "SystemExit(2)")
        self.assertEqual(out, "")
        self.assertIn("the following arguments are required: command", err)

    def test_abbreviated_subcommand_rejected(self):
        """Test abbreviated subcommand rejected."""
        # allow_abbrev=False everywhere: no prefix matching.
        code, _, err = _call_main(["cata"])
        self.assertEqual(code, "SystemExit(2)")
        self.assertIn("invalid choice", err)

    def test_duplicate_boundary_map_name_exits_2(self):
        """Test duplicate boundary map name exits 2."""
        code, out, err = _call_main(
            ["doctor", "--model", "m", "--boundary-map", "a=1", "--boundary-map", "a=2"]
        )
        self.assertEqual(code, "SystemExit(2)")
        self.assertEqual(out, "")
        self.assertIn("Duplicate --boundary-map candidate: 'a'", err)

    def test_unknown_boundary_map_name_on_run_exits_2(self):
        """Test unknown boundary map name on run exits 2."""
        # candidates() is pure (no model); the typo fails fast, exit 2.
        code, out, err = _call_main(
            [
                "run",
                "--model",
                "m",
                "--dataset",
                "d",
                "--family",
                "chatml",
                "--boundary-ids",
                "198",
                "--output",
                "o",
                "--boundary-map",
                "nope=1",
            ]
        )
        self.assertEqual(code, "SystemExit(2)")
        self.assertEqual(out, "")
        self.assertIn("--boundary-map names unknown candidate 'nope'", err)

    def test_boundary_map_missing_equals_rejected(self):
        """Test boundary map missing equals rejected."""
        code, _, err = _call_main(["doctor", "--model", "m", "--boundary-map", "abc"])
        self.assertEqual(code, "SystemExit(2)")
        self.assertIn("Use NAME=ID,ID", err)

    def test_boundary_map_empty_name_rejected(self):
        """Test boundary map empty name rejected."""
        code, _, err = _call_main(["doctor", "--model", "m", "--boundary-map", "=1"])
        self.assertEqual(code, "SystemExit(2)")
        self.assertIn("Use NAME=ID,ID", err)

    def test_boundary_ids_and_map_for_same_candidate_are_not_contradictory(self):
        """Test boundary ids and map for same candidate are not contradictory."""
        # The map overrides the default for that candidate; both parse.
        args = build_parser().parse_args(
            [
                "run",
                "--model",
                "m",
                "--dataset",
                "d",
                "--family",
                "chatml",
                "--boundary-ids",
                "198",
                "--boundary-map",
                "chatml=199",
                "--output",
                "o",
            ]
        )
        self.assertEqual(args.boundary_ids, (198,))
        self.assertEqual(args.boundary_map, [("chatml", (199,))])

    def test_huge_positional_list_rejected_without_hang(self):
        """Test huge positional list rejected without hang."""
        code, _, err = _call_main(["catalog", *[f"x{i}" for i in range(20000)]])
        self.assertEqual(code, "SystemExit(2)")
        self.assertIn("unrecognized arguments", err)

    def test_repeated_append_flag_accumulates(self):
        """Test repeated append flag accumulates."""
        args = build_parser().parse_args(
            ["doctor", "--model", "m", "--boundary-map", "a=1", "--boundary-map", "b=2"]
        )
        self.assertEqual(args.boundary_map, [("a", (1,)), ("b", (2,))])


class ExitCodeContractTests(unittest.TestCase):
    """Every failure path: non-zero exit + machine-parseable stderr line."""

    def _assert_failure(
        self,
        proc: subprocess.CompletedProcess[str],
        code: int,
        stderr_fragment: str = "",
        stdout_must_be_empty: bool = True,
    ) -> None:
        """Assert failure."""
        self.assertEqual(proc.returncode, code)
        tail = _stderr_tail(proc)
        self.assertTrue(tail, "failure must leave a machine-parseable stderr line")
        if stderr_fragment:
            self.assertIn(stderr_fragment, proc.stderr)
        if stdout_must_be_empty:
            self.assertEqual(proc.stdout, "", "failure must not pollute stdout")

    def test_bad_subcommand_exits_2(self):
        """Test bad subcommand exits 2."""
        proc = run_cli("frobnicate")
        self._assert_failure(proc, 2, "invalid choice")
        self.assertIn("usage:", proc.stderr)

    def test_missing_required_arg_exits_2(self):
        """Test missing required arg exits 2."""
        proc = run_cli("run", "--model", "/x")
        self._assert_failure(proc, 2, "the following arguments are required")

    def test_invalid_family_exits_2(self):
        """Test invalid family exits 2."""
        proc = run_cli(
            "run",
            "--model",
            "m",
            "--dataset",
            "d",
            "--family",
            "CHATML",
            "--boundary-ids",
            "198",
            "--output",
            "o",
        )
        self._assert_failure(proc, 2, "invalid choice: 'CHATML'")

    def test_invalid_device_exits_2(self):
        """Test invalid device exits 2."""
        proc = run_cli(
            "run",
            "--model",
            "m",
            "--dataset",
            "d",
            "--family",
            "chatml",
            "--boundary-ids",
            "198",
            "--output",
            "o",
            "--device",
            "tpu",
        )
        self._assert_failure(proc, 2, "invalid choice: 'tpu'")

    def test_unicode_family_rejected_as_invalid_choice(self):
        """Test unicode family rejected as invalid choice."""
        proc = run_cli(
            "run",
            "--model",
            "m",
            "--dataset",
            "d",
            "--family",
            "chatmlé",
            "--boundary-ids",
            "198",
            "--output",
            "o",
        )
        self._assert_failure(proc, 2, "invalid choice")

    def test_malformed_fixture_json_exits_1(self):
        """Test malformed fixture json exits 1."""
        with TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            proc = run_cli("inspect", str(bad))
        self._assert_failure(proc, 1, "inspect: invalid fixture file:")

    def test_fixture_directory_exits_1(self):
        """Test fixture directory exits 1."""
        with TemporaryDirectory() as tmp:
            proc = run_cli("inspect", tmp)
        self._assert_failure(proc, 1, "inspect: invalid fixture file:")

    def test_fixture_path_through_regular_file_exits_1(self):
        """Test fixture path through regular file exits 1."""
        # Root-proof "unreadable": a path component that is a regular file.
        with TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "blocker"
            blocker.write_text("x", encoding="utf-8")
            proc = run_cli("inspect", str(blocker / "fixture.json"))
        self._assert_failure(proc, 1, "inspect: invalid fixture file:")

    @unittest.skipIf(os.geteuid() == 0, "permission bits do not block root")
    def test_unreadable_fixture_mode_000_exits_1(self):
        """Test unreadable fixture mode 000 exits 1."""
        with TemporaryDirectory() as tmp:
            fixture = _write_fixture(tmp)
            fixture.chmod(0o000)
            try:
                proc = run_cli("inspect", str(fixture))
            finally:
                fixture.chmod(0o600)
        self._assert_failure(proc, 1, "inspect: invalid fixture file:")

    def test_nonexistent_fixture_exits_1_without_path_leak(self):
        """Test nonexistent fixture exits 1 without path leak."""
        proc = run_cli("inspect", "/nonexistent-héllo→.json")
        self._assert_failure(proc, 1, "inspect: invalid fixture file: Unreadable fixture file")
        self.assertNotIn("héllo", proc.stderr, "unicode path must not leak into stderr")

    def test_empty_string_dataset_exits_1(self):
        """Test empty string dataset exits 1."""
        proc = run_cli("inspect", "")
        self._assert_failure(proc, 1, "inspect: invalid fixture file:")

    def test_triple_dash_after_separator_is_a_literal_filename(self):
        """Test triple dash after separator is a literal filename."""
        # After `--`, `---` is just a positional value: a nonexistent path
        # then fails as an unreadable fixture (exit 1), not a parse error.
        proc = run_cli("inspect", "--", "---")
        self._assert_failure(proc, 1, "inspect: invalid fixture file:")

    def test_run_output_through_regular_file_exits_1(self):
        """Test run output through regular file exits 1."""
        # Unwritable output (root-proof): claim path goes through a file.
        with TemporaryDirectory() as tmp:
            fixture = _write_fixture(tmp)
            blocker = Path(tmp) / "blocker"
            blocker.write_text("x", encoding="utf-8")
            proc = run_cli(
                "run",
                "--model",
                "/nonexistent",
                "--dataset",
                str(fixture),
                "--family",
                "chatml",
                "--boundary-ids",
                "198",
                "--output",
                str(blocker / "out"),
            )
        self._assert_failure(proc, 1, "Operation failed. Check input paths")

    def test_run_existing_output_dir_exits_2(self):
        """Test run existing output dir exits 2."""
        with TemporaryDirectory() as tmp:
            existing = Path(tmp) / "existing"
            existing.mkdir()
            proc = run_cli(
                "run",
                "--model",
                "m",
                "--dataset",
                "d",
                "--family",
                "chatml",
                "--boundary-ids",
                "198",
                "--output",
                str(existing),
            )
        self._assert_failure(proc, 2, "Output path already exists")

    def test_contradictory_range_flags_exit_2(self):
        """Test contradictory range flags exit 2."""
        cases = [
            (["--max-new-tokens", "0"], "--max-new-tokens 1-4096"),
            (["--max-new-tokens", "4097"], "--max-new-tokens 1-4096"),
            (["--max-context", "0"], "--max-context 1-32768"),
            (["--seed", "-1"], "Seed must be an unsigned 64-bit integer"),
            (["--seed", str(2**64)], "Seed must be an unsigned 64-bit integer"),
            (["--max-drop-bytes", "0"], "--max-drop-bytes must be between 1 and 8"),
            (["--max-drop-bytes", "9"], "--max-drop-bytes must be between 1 and 8"),
        ]
        with TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "o")
            for extra, fragment in cases:
                with self.subTest(extra=extra):
                    proc = run_cli(
                        "run",
                        "--model",
                        "m",
                        "--dataset",
                        "d",
                        "--family",
                        "chatml",
                        "--boundary-ids",
                        "198",
                        "--output",
                        out,
                        *extra,
                    )
                    self._assert_failure(proc, 2, fragment)

    def test_negative_looking_int_flag_value_rejected(self):
        """Test negative looking int flag value rejected."""
        # argparse binds "-5" as the flag value; the range check then fires.
        with TemporaryDirectory() as tmp:
            proc = run_cli(
                "run",
                "--model",
                "m",
                "--dataset",
                "d",
                "--family",
                "chatml",
                "--boundary-ids",
                "198",
                "--output",
                str(Path(tmp) / "o"),
                "--max-new-tokens",
                "-5",
            )
        self._assert_failure(proc, 2, "--max-new-tokens 1-4096")

    def test_nonexistent_checkpoint_doctor_exits_1_with_json_verdict(self):
        """Test nonexistent checkpoint doctor exits 1 with json verdict."""
        # Deliberate doctor contract: machine-parseable JSON on stdout, the
        # exit code carries the verdict; stderr stays empty.
        proc = run_cli("doctor", "--model", "/nonexistent-ckpt-xyz")
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(proc.stderr, "")
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["verdict"], "failed")

    def test_nonexistent_checkpoint_diff_backends_exits_1_fixed_code(self):
        """Test nonexistent checkpoint diff backends exits 1 fixed code."""
        proc = run_cli(
            "diff-backends",
            "--model",
            "/nonexistent-ckpt-xyz",
            "--prompt-text",
            "hi",
            "--output-text",
            "yo",
            "--stops",
            "STOP",
        )
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(proc.stdout, "")
        tail = _stderr_tail(proc)
        self.assertTrue(tail.startswith("diff-backends: "), tail)

    def test_diff_reports_missing_input_exits_1(self):
        """Test diff reports missing input exits 1."""
        proc = run_cli("diff-reports", "/nonexistent-a", "/nonexistent-b")
        self._assert_failure(proc, 1, "diff-reports: results_path_missing (old input)")

    def test_debug_unreadable_script_exits_1(self):
        """Test debug unreadable script exits 1."""
        with TemporaryDirectory() as tmp:
            proc = run_cli("debug", "--dataset", str(EXAMPLE_FIXTURES), "--script", tmp)
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(proc.stdout, "")
        self.assertIn("debug: unreadable script file:", proc.stderr)

    def test_fixtures_build_output_through_file_exits_1(self):
        """Test fixtures build output through file exits 1."""
        with TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "blocker"
            blocker.write_text("x", encoding="utf-8")
            proc = run_cli(
                "fixtures",
                "build",
                str(EXAMPLE_CSV),
                "--output",
                str(blocker / "out.json"),
            )
        self._assert_failure(proc, 1, "fixtures: invalid authoring source:")

    def test_global_flag_before_subcommand_exits_2(self):
        """Test global flag before subcommand exits 2."""
        # No global flags exist; the stray value is treated as the command.
        proc = run_cli("--output", "x", "catalog")
        self._assert_failure(proc, 2, "invalid choice: 'x'")

    def test_nul_byte_dataset_exits_1_in_process(self):
        """Test nul byte dataset exits 1 in process."""
        # subprocess cannot carry NUL in argv; exercise main() directly.
        code, out, err = _call_main(["inspect", "a\x00b"])
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("inspect: invalid fixture file:", err)


class _FakeBackend:
    """Stand-in for LocalHFBackend: records constructor args, no weights."""

    eos_token_id = None

    def __init__(self, model: object, device: object = None, max_context: object = None) -> None:
        self.model = model
        self.device = device
        self.max_context = max_context


class RunDecisionExitCodeTests(unittest.TestCase):
    """``run`` maps experiment decisions to exact exit codes (0/1/2/130)."""

    def _run_with_decision(self, decision: str) -> tuple[int, dict[str, object]]:
        """Run with decision."""
        with TemporaryDirectory() as tmp:
            fixture = _write_fixture(tmp)
            outdir = Path(tmp) / "report"
            result_doc: dict[str, object] = {
                "decision": decision,
                "selected_candidate": None,
                "search": [],
                "held_out": None,
            }
            patches = contextlib.ExitStack()
            patches.enter_context(mock.patch.object(backends, "LocalHFBackend", _FakeBackend))
            patches.enter_context(mock.patch.object(cli, "run_experiment", return_value=result_doc))
            patches.enter_context(
                mock.patch.object(cli, "save_report", return_value=("results.json", "report.md"))
            )
            patches.enter_context(
                mock.patch.object(doctor_mod, "template_conformance", return_value={"ok": True})
            )
            patches.enter_context(
                mock.patch.object(experiments, "budget_frontier", return_value={"ok": True})
            )

            def fake_prepare(path: object) -> Path:
                """Fake prepare."""
                target = Path(str(path))
                target.mkdir(parents=True, exist_ok=False)
                return target

            patches.enter_context(
                mock.patch.object(reporting_mod, "prepare_report_dir", fake_prepare)
            )
            out, err = io.StringIO(), io.StringIO()
            with patches, redirect_stdout(out), redirect_stderr(err):
                code = main(
                    [
                        "run",
                        "--model",
                        "ckpt-dir",
                        "--dataset",
                        str(fixture),
                        "--family",
                        "chatml",
                        "--boundary-ids",
                        "198,199",
                        "--output",
                        str(outdir),
                        "--seed",
                        "7",
                    ]
                )
            self.assertEqual(err.getvalue(), "", "no stderr expected on these paths")
            payload = json.loads(out.getvalue())
            self.assertIsInstance(code, int)
            return code, payload

    def test_compatible_exits_0(self):
        """Test compatible exits 0."""
        code, payload = self._run_with_decision("compatible_on_fixture_set")
        self.assertEqual(code, 0)
        self.assertEqual(payload["decision"], "compatible_on_fixture_set")

    def test_failed_exits_1(self):
        """Test failed exits 1."""
        code, payload = self._run_with_decision("failed")
        self.assertEqual(code, 1)
        self.assertEqual(payload["decision"], "failed")

    def test_cancelled_exits_130(self):
        """Test cancelled exits 130."""
        code, payload = self._run_with_decision("cancelled")
        self.assertEqual(code, 130)
        self.assertEqual(payload["decision"], "cancelled")

    def test_ambiguous_exits_2(self):
        """Test ambiguous exits 2."""
        code, payload = self._run_with_decision("ambiguous")
        self.assertEqual(code, 2)
        self.assertEqual(payload["decision"], "ambiguous")

    def test_held_out_failed_exits_2(self):
        """Test held out failed exits 2."""
        code, payload = self._run_with_decision("held_out_failed")
        self.assertEqual(code, 2)
        self.assertEqual(payload["decision"], "held_out_failed")

    def test_no_candidate_exits_2(self):
        """Test no candidate exits 2."""
        code, payload = self._run_with_decision("no_candidate")
        self.assertEqual(code, 2)
        self.assertEqual(payload["decision"], "no_candidate")


class HelpStabilityTests(unittest.TestCase):
    """--help output is byte-stable across runs; no run-varying data leaks."""

    @staticmethod
    def _help_argvs() -> list[list[str]]:
        """Help argvs."""
        argvs = [["--help"]]
        for sub in SUBCOMMANDS:
            argvs.append([sub, "--help"])
        argvs.append(["fixtures", "build", "--help"])
        return argvs

    def test_help_byte_stable_across_runs(self):
        """Test help byte stable across runs."""
        env = {"COLUMNS": "80"}
        for argv in self._help_argvs():
            with self.subTest(argv=argv):
                first = run_cli(*argv, env=env)
                second = run_cli(*argv, env=env)
                self.assertEqual(first.returncode, 0, argv)
                self.assertEqual(second.returncode, 0, argv)
                self.assertEqual(second.stdout, first.stdout, argv)
                self.assertEqual(second.stderr, "", argv)

    def test_help_contains_no_run_varying_data(self):
        """Test help contains no run varying data."""
        env = {"COLUMNS": "80"}
        pid = str(os.getpid())
        tmpdir = tempfile.gettempdir()
        for argv in self._help_argvs():
            with self.subTest(argv=argv):
                text = run_cli(*argv, env=env).stdout
                self.assertNotIn(pid, text, argv)
                self.assertNotIn(tmpdir, text, argv)
                self.assertIsNone(
                    re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", text),
                    f"timestamp-like text in help for {argv}",
                )

    def test_catalog_stdout_byte_stable(self):
        """Test catalog stdout byte stable."""
        first = run_cli("catalog")
        second = run_cli("catalog")
        self.assertEqual(first.returncode, 0)
        self.assertEqual(second.stdout, first.stdout)
        self.assertEqual(second.stderr, "")

    def test_no_version_flag_top_level_help(self):
        """Test no version flag top level help."""
        # Mandate item 3 assumed a --version flag; none exists and cycle 64
        # pins `--version` -> exit 2 as an unrecognized flag, so it is NOT
        # added here. The version lives only in model_protocol_lab.__version__.
        text = run_cli("--help").stdout
        self.assertNotIn("--version", text)
        self.assertTrue(model_protocol_lab.__version__)
        self.assertNotIn("--version", build_parser().format_help())


class ConsoleScriptEquivalenceTests(unittest.TestCase):
    """``model-protocol-lab`` entry point == ``python -m model_protocol_lab``."""

    script: str

    @classmethod
    def setUpClass(cls):
        """Set up the test fixture."""
        found = shutil.which("model-protocol-lab")
        if found:
            cls.script = found
            return
        candidate = Path(sys.prefix) / "bin" / "model-protocol-lab"
        if candidate.is_file():
            cls.script = str(candidate)
            return
        raise unittest.SkipTest("model-protocol-lab console script not installed")

    def _run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run script."""
        return subprocess.run(
            [self.script, *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=CALL_TIMEOUT,
            check=False,
        )

    def _assert_identical(self, argv: tuple[str, ...]) -> None:
        """Assert identical."""
        via_module = run_cli(*argv)
        via_script = self._run_script(*argv)
        self.assertEqual(via_script.returncode, via_module.returncode, argv)
        self.assertEqual(via_script.stdout, via_module.stdout, argv)
        self.assertEqual(via_script.stderr, via_module.stderr, argv)

    def test_top_level_help_identical(self):
        """Test top level help identical."""
        self._assert_identical(("--help",))

    def test_subcommand_help_identical(self):
        """Test subcommand help identical."""
        self._assert_identical(("run", "--help"))

    def test_failure_path_inspect_bad_fixture_identical(self):
        """Test failure path inspect bad fixture identical."""
        with TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{nope", encoding="utf-8")
            self._assert_identical(("inspect", str(bad)))

    def test_argparse_failure_path_identical(self):
        """Test argparse failure path identical."""
        self._assert_identical(("run", "--model", "/x"))

    def test_diff_reports_failure_path_identical(self):
        """Test diff reports failure path identical."""
        self._assert_identical(("diff-reports", "/nonexistent-a", "/nonexistent-b"))


class EnvironmentInterplayTests(unittest.TestCase):
    """MPL_* env vars do not exist; hostile argv[0] cannot crash parsing."""

    HOSTILE_ENV = {
        "MPL_DEBUG": "1",
        "MPL_MODEL": "/evil-ckpt",
        "MPL_DATASET": "/evil.json",
        "MPL_BOUNDARY_IDS": "1,2,3",
        "MPL_OUTPUT": "/evil-out",
        "MPL_FOO": "bar",
        "MPL_": "",
    }

    def test_mpl_env_vars_do_not_change_catalog(self):
        """Test mpl env vars do not change catalog."""
        plain = run_cli("catalog")
        polluted = run_cli("catalog", env=self.HOSTILE_ENV)
        self.assertEqual(polluted.returncode, 0)
        self.assertEqual(polluted.stdout, plain.stdout)
        self.assertEqual(polluted.stderr, plain.stderr)

    def test_mpl_env_vars_do_not_change_inspect(self):
        """Test mpl env vars do not change inspect."""
        plain = run_cli("inspect", str(EXAMPLE_FIXTURES))
        polluted = run_cli("inspect", str(EXAMPLE_FIXTURES), env=self.HOSTILE_ENV)
        self.assertEqual(polluted.returncode, plain.returncode)
        self.assertEqual(polluted.stdout, plain.stdout)
        self.assertEqual(polluted.stderr, plain.stderr)

    def test_package_reads_no_mpl_env_vars(self):
        """Test package reads no mpl env vars."""
        package = Path(cli.__file__).resolve().parent
        hits = [
            path.name
            for path in sorted(package.glob("*.py"))
            if "MPL_" in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(hits, [], f"MPL_ references found in: {hits}")

    def test_weird_argv0_does_not_crash(self):
        """Test weird argv0 does not crash."""
        # main() with argv=None reads sys.argv; a hostile argv[0] must not
        # affect parsing because the usage prog is hard-coded.
        old_argv = sys.argv
        sys.argv = ["\x00garbage\xff", "catalog"]
        out, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                code = main()
        finally:
            sys.argv = old_argv
        self.assertEqual(code, 0)
        json.loads(out.getvalue())

    def test_empty_argv0_help_shows_hardcoded_prog(self):
        """Test empty argv0 help shows hardcoded prog."""
        old_argv = sys.argv
        sys.argv = ["", "--help"]
        out, err = io.StringIO(), io.StringIO()
        try:
            with (
                redirect_stdout(out),
                redirect_stderr(err),
                self.assertRaises(SystemExit) as ctx,
            ):
                main()
        finally:
            sys.argv = old_argv
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("usage: model-protocol-lab", out.getvalue())

    def test_prog_hardcoded_under_python_m(self):
        """Test prog hardcoded under python m."""
        # Under `python -m`, argv[0] is the __main__.py path; usage must
        # still show the hard-coded prog name.
        proc = run_cli("frobnicate")
        self.assertTrue(
            proc.stderr.startswith("usage: model-protocol-lab"),
            proc.stderr.splitlines()[0] if proc.stderr else "",
        )


if __name__ == "__main__":
    unittest.main()
