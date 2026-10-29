"""Cycle 64: adversarial audit of the CLI surface.

HARD CONSTRAINTS UNDER TEST
---------------------------
1. Every subcommand's argument validation behaves per its code: missing
   required args / wrong types / out-of-range values -> exit 2 (argparse)
   with usage on stderr; nonexistent or unreadable path inputs -> exit 1
   with a fixed error code on stderr and no traceback; stdout stays
   machine-parseable (JSON) on the success paths.
2. Error-code discipline: every error code documented in docs/USAGE.md is
   still produced verbatim by the code (no drift), and every produced code
   follows the fixed-vocabulary rule (no raw paths, no loader text, no
   tracebacks on the terminal).
3. The `debug` REPL cannot mutate the selection pipeline: it merges the
   search + held_out partitions into a fresh tuple for inspection only
   (debug.py ``DebugShell.__init__``), keeps all session state shell-local,
   never imports the experiments module, and its only file write is the
   operator's own transcript via `.save`.
4. `python -m model_protocol_lab` and the installed `model-protocol-lab`
   console script behave identically (--help output, error paths).
5. Red-team inputs: flag-looking values, `--` separators, unicode paths,
   NUL bytes, oversized inputs, `--help` on every subcommand, unknown
   subcommands, bare invocation. There are no mutually exclusive flag
   pairs on any subcommand (checked in cli.py:build_parser).

METHOD
------
* Subprocess (`sys.executable -m model_protocol_lab`) for every error-code
  and exit-code path: the audit asserts the real process behavior
  (returncode, stdout vs stderr routing, exact stderr text).
* Unit-level (imported `main` / `build_parser` / `parse_boundary` /
  `DebugShell` / `diff_backends`) for speed where the path is pure: the
  validation table, the REPL read-only property, the fixed-code
  vocabulary. No mocked sys.argv: `main([...])` is called with an explicit
  argv list.
* The validation table was built from reading cli.py, debug.py,
  diffreports.py, diffbackends.py, datasets.py, fixtures.py, backends.py
  and doctor.py; every row below was then executed.

FINDINGS
--------
CLEAN: no genuine validation bug found. Every row of the validation table
matches the code's behavior; every USAGE.md-documented error code is still
emitted verbatim; exit codes follow the documented discipline (2 for
argparse errors, 1 for fixed-code input failures, 0/1/2/130 for run
decisions); the debug REPL is read-only w.r.t. experiment state (verified
by snapshot comparison of the Split and CATALOG before/after a full
command battery, plus a static no-experiments-import check).

PRECISION NOTES (doc-vs-reality, not code bugs):
1. `mpl` is docs shorthand, not an installed command: the distribution
   exposes exactly one console script, `model-protocol-lab`; there is no
   `mpl` entry point (importlib.metadata and PATH both checked).
2. docs/USAGE.md names 4 diff-reports fixed codes (results_too_large,
   results_not_regular_file, results_duplicate_field,
   results_nonfinite_number); the code emits 11 fixed codes. The 4
   documented all match reality; the 7 additional codes
   (results_path_missing, results_unreadable, results_not_utf8,
   results_invalid_json, results_not_an_object, results_wrong_shape,
   results_nesting_too_deep) follow the same fixed-vocabulary discipline.
   The docs are a subset, not wrong -- except for one overstatement:
   USAGE.md says "the path must be a regular file -- FIFOs, devices,
   sockets and directories are refused (results_not_regular_file)", but a
   directory input is actually treated as a report directory holding
   results.json (the subcommand help documents "report directory
   (containing results.json) or results.json path", and it works).
3. The 1 MB --prompt-text/--output-text caps are argparse errors (exit 2),
   not fixed error codes (cycle-46 already corrected the USAGE.md wording;
   re-confirmed here). Related: on Linux a 1 MB argv string can never
   reach the process (MAX_ARG_STRLEN is 128 KiB), so these two bounds are
   unreachable via real invocation and only trigger in-process -- pinned
   here via unit-level main([...]) calls.
4. stdlib argparse quirk (not a repo bug): an unrecognized top-level flag
   (e.g. `--frobnicate`) reports "the following arguments are required:
   command" instead of "unrecognized arguments". The exit code is still
   the correct 2.

DELIBERATELY NOT VERIFIED
-------------------------
* The exit-130 SIGINT path (needs a live run plus real signal delivery).
* Live-model `run` / `doctor` against real checkpoints (no weights in this
  environment); synthetic tokenizer-only checkpoints stand in for the
  loader paths.
* pip installation of the console script itself.
"""

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import unittest
from dataclasses import asdict
from importlib.metadata import distribution
from pathlib import Path
from tempfile import TemporaryDirectory

from model_protocol_lab.catalog import CATALOG
from model_protocol_lab.cli import build_parser, main, parse_boundary, parse_boundary_map
from model_protocol_lab.datasets import load_dataset
from model_protocol_lab.debug import DebugShell

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_FIXTURES = REPO_ROOT / "examples" / "fixtures.json"
EXAMPLE_CSV = REPO_ROOT / "examples" / "fixtures.csv"
CALL_TIMEOUT = 120


# Every registered top-level subcommand, derived from the parser itself so
# the list cannot drift from build_parser.
def _top_level_subcommands() -> list[str]:
    """Top level subcommands."""
    parser = build_parser()
    for action in parser._subparsers._group_actions:  # noqa: SLF001
        if action.dest == "command":
            return sorted(action.choices.keys())
    raise AssertionError("no subparsers found")


SUBCOMMANDS = _top_level_subcommands()


def run_cli(*args: str):
    """Invoke the real CLI entry point in a subprocess; never mocked."""
    return subprocess.run(
        [sys.executable, "-m", "model_protocol_lab", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=CALL_TIMEOUT,
        check=False,
    )


def make_tiny_checkpoint(directory: Path) -> Path:
    """Write a minimal tokenizer.json checkpoint (tokenizer only)."""
    from tokenizers import Tokenizer  # noqa: PLC0415
    from tokenizers.models import WordLevel  # noqa: PLC0415
    from tokenizers.pre_tokenizers import Whitespace  # noqa: PLC0415

    vocab = {
        "[UNK]": 0,
        "hello": 1,
        "world": 2,
        "answer": 3,
        "STOP": 4,
        "done": 5,
        "x": 6,
        "the": 7,
        "is": 8,
        "42": 9,
        "a": 10,
    }
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.save(str(directory / "tokenizer.json"))
    return directory


def _valid_fixture_bytes() -> bytes:
    """Valid fixture bytes."""
    return EXAMPLE_FIXTURES.read_bytes()


class TopLevelSurfaceTests(unittest.TestCase):
    """Bare invocation, unknown subcommands, abbreviation handling."""

    def test_no_args_exits_2_with_usage_on_stderr(self):
        """Test no args exits 2 with usage on stderr."""
        proc = run_cli()
        self.assertEqual(proc.returncode, 2)
        self.assertIn("usage:", proc.stderr)
        self.assertEqual(proc.stdout, "")

    def test_unknown_subcommand_exits_2(self):
        """Test unknown subcommand exits 2."""
        proc = run_cli("frobnicate")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("invalid choice", proc.stderr)
        self.assertEqual(proc.stdout, "")

    def test_abbreviated_subcommand_rejected(self):
        """Test abbreviated subcommand rejected."""
        # allow_abbrev=False on every parser: "catal" must not match "catalog".
        proc = run_cli("catal")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("invalid choice", proc.stderr)

    def test_top_level_help_lists_every_subcommand(self):
        """Test top level help lists every subcommand."""
        proc = run_cli("--help")
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stderr, "")
        for name in SUBCOMMANDS:
            self.assertIn(name, proc.stdout)

    def test_subcommand_set_matches_expectation(self):
        """Test subcommand set matches expectation."""
        self.assertEqual(
            SUBCOMMANDS,
            [
                "catalog",
                "debug",
                "diff-backends",
                "diff-reports",
                "doctor",
                "fixtures",
                "inspect",
                "pack",
                "run",
                "surface-noise",
                "verify",
            ],
        )

    def test_undefined_top_level_flag_exits_2(self):
        """Test undefined top level flag exits 2."""
        # stdlib argparse quirk with required subparsers: an unrecognized
        # top-level flag is swallowed and the missing-subcommand error fires
        # instead of "unrecognized arguments". Exit code is still 2.
        proc = run_cli("--version")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("usage:", proc.stderr)
        self.assertIn("the following arguments are required: command", proc.stderr)
        self.assertEqual(proc.stdout, "")


class HelpSurfaceTests(unittest.TestCase):
    """--help works on the top level and every (sub-)subcommand."""

    def test_help_on_every_subcommand(self):
        """Test help on every subcommand."""
        cases = [
            (("catalog", "--help"), ["usage:", "catalog"]),
            (("inspect", "--help"), ["usage:", "dataset"]),
            (("doctor", "--help"), ["usage:", "--model"]),
            (("run", "--help"), ["usage:", "--max-new-tokens", "--boundary-ids"]),
            (("fixtures", "--help"), ["usage:", "build"]),
            (("fixtures", "build", "--help"), ["usage:", "--output", "source"]),
            (("diff-reports", "--help"), ["usage:", "old_results", "--all"]),
            (("diff-backends", "--help"), ["usage:", "--min-tokens", "--stops"]),
            (("debug", "--help"), ["usage:", "--dataset", "--script"]),
        ]
        for argv, markers in cases:
            with self.subTest(argv=argv):
                proc = run_cli(*argv)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stderr, "")
                for marker in markers:
                    self.assertIn(marker, proc.stdout)

    def test_fixtures_without_subcommand_exits_2(self):
        """Test fixtures without subcommand exits 2."""
        proc = run_cli("fixtures")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("usage:", proc.stderr)


class InspectValidationTests(unittest.TestCase):
    """`inspect`: positional dataset validation table."""

    def test_missing_positional_exits_2(self):
        """Test missing positional exits 2."""
        proc = run_cli("inspect")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("usage:", proc.stderr)

    def test_nonexistent_path_exits_1_with_fixed_message_on_stderr(self):
        """Test nonexistent path exits 1 with fixed message on stderr."""
        proc = run_cli("inspect", "/nonexistent-fixture-xyz.json")
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(proc.stdout, "")
        self.assertIn("inspect: invalid fixture file:", proc.stderr)
        self.assertIn("Unreadable fixture file", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_directory_exits_1(self):
        """Test directory exits 1."""
        with TemporaryDirectory() as tmp:
            proc = run_cli("inspect", tmp)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("inspect: invalid fixture file:", proc.stderr)
        self.assertIn("Fixture path is not a regular file", proc.stderr)

    def test_invalid_json_exits_1(self):
        """Test invalid json exits 1."""
        with TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            proc = run_cli("inspect", str(bad))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("inspect: invalid fixture file:", proc.stderr)

    def test_duplicate_json_field_exits_1(self):
        """Test duplicate json field exits 1."""
        with TemporaryDirectory() as tmp:
            bad = Path(tmp) / "dup.json"
            bad.write_text('{"search": [], "search": [], "held_out": []}', encoding="utf-8")
            proc = run_cli("inspect", str(bad))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("Duplicate JSON field", proc.stderr)

    def test_non_utf8_exits_1(self):
        """Test non utf8 exits 1."""
        with TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bin.json"
            bad.write_bytes(b"\xff\xfe\x00not-utf8")
            proc = run_cli("inspect", str(bad))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("Fixture file is not valid UTF-8", proc.stderr)

    def test_oversized_fixture_exits_1(self):
        """Test oversized fixture exits 1."""
        with TemporaryDirectory() as tmp:
            big = Path(tmp) / "big.json"
            big.write_bytes(b'{"search": [], "held_out": [], "pad": "' + b"x" * 2_100_000 + b'"}')
            proc = run_cli("inspect", str(big))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("Dataset exceeds size limit", proc.stderr)

    def test_valid_fixture_reports_counts_as_json_on_stdout(self):
        """Test valid fixture reports counts as json on stdout."""
        proc = run_cli("inspect", str(EXAMPLE_FIXTURES))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(
            payload,
            {
                "search_cases": 2,
                "held_out_cases": 2,
                "valid": True,
                # The example fixtures are contamination-clean.
                "contamination": {"worst_tier": "clean", "pairs": []},
            },
        )

    def test_unicode_nonexistent_path_exits_1_without_crash(self):
        """Test unicode nonexistent path exits 1 without crash."""
        with TemporaryDirectory() as tmp:
            proc = run_cli("inspect", str(Path(tmp) / "fïxturé-ünïcode-ß.json"))
        self.assertEqual(proc.returncode, 1)
        self.assertNotIn("Traceback", proc.stderr)

    def test_dash_dash_separator_treats_dash_value_as_path(self):
        """Test dash dash separator treats dash value as path."""
        proc = run_cli("inspect", "--", "-weird-path")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("Unreadable fixture file", proc.stderr)

    def test_lone_dash_exits_1(self):
        """Test lone dash exits 1."""
        proc = run_cli("inspect", "-")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("inspect: invalid fixture file:", proc.stderr)


class ParseBoundaryUnitTests(unittest.TestCase):
    """Unit-level table for the shared boundary-ID grammars (fast)."""

    def test_parse_boundary_accepts(self):
        """Test parse boundary accepts."""
        for text, expected in [
            ("1", (1,)),
            ("1,2,3", (1, 2, 3)),
            (" 4 , 5 ", (4, 5)),
            ("0", (0,)),
            ("2147483647", (2147483647,)),
            (",".join(["7"] * 16), (7,) * 16),
        ]:
            with self.subTest(text=text):
                self.assertEqual(parse_boundary(text), expected)

    def test_parse_boundary_rejects(self):
        """Test parse boundary rejects."""
        for text in [
            "",
            " ",
            "1,,2",
            ",1",
            "1,",
            "abc",
            "1.5",
            "-1",
            "+1",
            "0x10",
            "１２３",  # fullwidth digits: must not silently coerce
            "2147483648",  # 2**31
            ",".join(["7"] * 17),  # > 16 parts
        ]:
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                parse_boundary(text)

    def test_parse_boundary_map_accepts(self):
        """Test parse boundary map accepts."""
        name, ids = parse_boundary_map("chatml=198")
        self.assertEqual((name, ids), ("chatml", (198,)))
        # Candidate names may contain "/"; split is on the FIRST "=".
        name, ids = parse_boundary_map("chatml/toggle-bos=1,2")
        self.assertEqual((name, ids), ("chatml/toggle-bos", (1, 2)))

    def test_parse_boundary_map_rejects(self):
        """Test parse boundary map rejects."""
        for text in ["chatml", "=198", "   =198", "chatml=", "chatml=abc"]:
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                parse_boundary_map(text)


class DoctorValidationTests(unittest.TestCase):
    """`doctor`: flag validation table (loader paths use a bad checkpoint)."""

    def test_missing_model_exits_2(self):
        """Test missing model exits 2."""
        proc = run_cli("doctor")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("usage:", proc.stderr)

    def test_bad_boundary_ids_exits_2(self):
        """Test bad boundary ids exits 2."""
        cases = [
            ("abc", "comma-separated ASCII token IDs"),
            ("1,,2", "comma-separated ASCII token IDs"),
            ("１２３", "comma-separated ASCII token IDs"),
            (",".join(["1"] * 17), "comma-separated ASCII token IDs"),
            ("2147483648", "Token IDs must be < 2^31"),
        ]
        for bad, fragment in cases:
            with self.subTest(bad=bad):
                proc = run_cli("doctor", "--model", "/nonexistent", "--boundary-ids", bad)
                self.assertEqual(proc.returncode, 2, bad)
                self.assertIn(fragment, proc.stderr)

    def test_boundary_map_without_equals_exits_2(self):
        """Test boundary map without equals exits 2."""
        proc = run_cli("doctor", "--model", "/nonexistent", "--boundary-map", "chatml")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("NAME=ID,ID", proc.stderr)

    def test_duplicate_boundary_map_exits_2(self):
        """Test duplicate boundary map exits 2."""
        proc = run_cli(
            "doctor",
            "--model",
            "/nonexistent",
            "--boundary-map",
            "chatml=1",
            "--boundary-map",
            "chatml=2",
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("Duplicate --boundary-map candidate", proc.stderr)

    def test_bad_checkpoint_dir_reports_json_verdict_and_exits_1(self):
        """Test bad checkpoint dir reports json verdict and exits 1."""
        with TemporaryDirectory() as tmp:
            proc = run_cli("doctor", "--model", tmp, "--dataset", str(EXAMPLE_FIXTURES))
        self.assertEqual(proc.returncode, 1)
        payload = json.loads(proc.stdout)
        self.assertIn(payload["verdict"], ("failed", "error"))
        self.assertNotIn("Traceback", proc.stderr)


class RunValidationTests(unittest.TestCase):
    """`run`: full argument-validation table (no model work is reached)."""

    @staticmethod
    def _base(tmp: str, **overrides):
        """Base."""
        argv = [
            "run",
            "--model",
            "/nonexistent-model",
            "--dataset",
            "/nonexistent-dataset.json",
            "--family",
            "chatml",
            "--boundary-ids",
            "198",
            "--output",
            str(Path(tmp) / "fresh-output"),
        ]
        for key, value in overrides.items():
            argv.extend([key, value])
        return argv

    def test_missing_each_required_flag_exits_2(self):
        """Test missing each required flag exits 2."""
        required = ["--model", "--dataset", "--family", "--boundary-ids", "--output"]
        with TemporaryDirectory() as tmp:
            full = self._base(tmp)[1:]  # drop the "run" subcommand token
            for flag in required:
                with self.subTest(flag=flag):
                    argv = []
                    skip_next = False
                    for token in full:
                        if skip_next:
                            skip_next = False
                            continue
                        if token == flag:
                            skip_next = True
                            continue
                        argv.append(token)
                    proc = run_cli("run", *argv)
                    self.assertEqual(proc.returncode, 2, flag)
                    self.assertIn("usage:", proc.stderr)

    def test_invalid_family_exits_2(self):
        """Test invalid family exits 2."""
        with TemporaryDirectory() as tmp:
            proc = run_cli(
                "run",
                "--model",
                "/x",
                "--dataset",
                "/y",
                "--family",
                "not-a-family",
                "--boundary-ids",
                "1",
                "--output",
                str(Path(tmp) / "o"),
            )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("invalid choice", proc.stderr)

    def test_out_of_range_numeric_flags_exit_2(self):
        """Test out of range numeric flags exit 2."""
        cases = [
            ("--max-new-tokens", "0"),
            ("--max-new-tokens", "4097"),
            ("--max-context", "0"),
            ("--max-context", "32769"),
            ("--seed", "-1"),
            ("--seed", str(2**64)),
            ("--max-drop-bytes", "0"),
            ("--max-drop-bytes", "9"),
        ]
        with TemporaryDirectory() as tmp:
            for flag, value in cases:
                with self.subTest(flag=flag, value=value):
                    proc = run_cli(
                        "run",
                        "--model",
                        "/x",
                        "--dataset",
                        "/y",
                        "--family",
                        "chatml",
                        "--boundary-ids",
                        "1",
                        "--output",
                        str(Path(tmp) / f"o-{flag.strip('-')}-{value}"),
                        flag,
                        value,
                    )
                    self.assertEqual(proc.returncode, 2, (flag, value))
                    self.assertIn("usage:", proc.stderr)

    def test_wrong_type_numeric_flags_exit_2(self):
        """Test wrong type numeric flags exit 2."""
        with TemporaryDirectory() as tmp:
            for flag, value in [("--max-new-tokens", "abc"), ("--seed", "1.5")]:
                with self.subTest(flag=flag, value=value):
                    proc = run_cli(
                        "run",
                        "--model",
                        "/x",
                        "--dataset",
                        "/y",
                        "--family",
                        "chatml",
                        "--boundary-ids",
                        "1",
                        "--output",
                        str(Path(tmp) / "o2"),
                        flag,
                        value,
                    )
                    self.assertEqual(proc.returncode, 2)
                    self.assertIn("invalid int value", proc.stderr)

    def test_invalid_device_exits_2(self):
        """Test invalid device exits 2."""
        with TemporaryDirectory() as tmp:
            proc = run_cli(
                "run",
                "--model",
                "/x",
                "--dataset",
                "/y",
                "--family",
                "chatml",
                "--boundary-ids",
                "1",
                "--output",
                str(Path(tmp) / "o"),
                "--device",
                "tpu",
            )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("invalid choice", proc.stderr)

    def test_bad_boundary_ids_exits_2(self):
        """Test bad boundary ids exits 2."""
        with TemporaryDirectory() as tmp:
            proc = run_cli(
                "run",
                "--model",
                "/x",
                "--dataset",
                "/y",
                "--family",
                "chatml",
                "--boundary-ids",
                "1,,2",
                "--output",
                str(Path(tmp) / "o"),
            )
        self.assertEqual(proc.returncode, 2)

    def test_boundary_map_bad_syntax_exits_2(self):
        """Test boundary map bad syntax exits 2."""
        with TemporaryDirectory() as tmp:
            proc = run_cli(
                "run",
                "--model",
                "/x",
                "--dataset",
                "/y",
                "--family",
                "chatml",
                "--boundary-ids",
                "1",
                "--output",
                str(Path(tmp) / "o"),
                "--boundary-map",
                "chatml",
            )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("NAME=ID,ID", proc.stderr)

    def test_boundary_map_duplicate_name_exits_2(self):
        """Test boundary map duplicate name exits 2."""
        with TemporaryDirectory() as tmp:
            proc = run_cli(
                "run",
                "--model",
                "/x",
                "--dataset",
                "/y",
                "--family",
                "chatml",
                "--boundary-ids",
                "1",
                "--output",
                str(Path(tmp) / "o"),
                "--boundary-map",
                "chatml=1",
                "--boundary-map",
                "chatml=2",
            )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("Duplicate --boundary-map candidate", proc.stderr)

    def test_boundary_map_unknown_name_exits_2(self):
        """Test boundary map unknown name exits 2."""
        with TemporaryDirectory() as tmp:
            proc = run_cli(
                "run",
                "--model",
                "/x",
                "--dataset",
                "/y",
                "--family",
                "chatml",
                "--boundary-ids",
                "1",
                "--output",
                str(Path(tmp) / "o"),
                "--boundary-map",
                "typo-name=1",
            )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("unknown candidate", proc.stderr)

    def test_existing_output_exits_2(self):
        """Test existing output exits 2."""
        with TemporaryDirectory() as tmp:
            existing = Path(tmp) / "taken"
            existing.mkdir()
            proc = run_cli(
                "run",
                "--model",
                "/x",
                "--dataset",
                "/y",
                "--family",
                "chatml",
                "--boundary-ids",
                "1",
                "--output",
                str(existing),
            )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("Output path already exists", proc.stderr)

    def test_nonexistent_dataset_exits_1_and_claimed_dir_is_cleaned_up(self):
        """Test nonexistent dataset exits 1 and claimed dir is cleaned up."""
        # The output dir is claimed BEFORE load_dataset; on failure the CLI
        # must release it so a retry still sees a fresh path.
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "fresh-output"
            proc = run_cli(
                "run",
                "--model",
                "/x",
                "--dataset",
                "/nonexistent-dataset.json",
                "--family",
                "chatml",
                "--boundary-ids",
                "1",
                "--output",
                str(output),
            )
            self.assertEqual(proc.returncode, 1)
            self.assertIn("Operation failed", proc.stderr)
            self.assertIn("Unreadable fixture file", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertFalse(output.exists(), "claimed output dir was not released")

    def test_flag_looking_value_rejected_as_missing_argument(self):
        """Test flag looking value rejected as missing argument."""
        # Argument injection: "--model --dataset" must not let --dataset
        # become the model path; argparse refuses the flag-looking value.
        with TemporaryDirectory() as tmp:
            proc = run_cli(
                "run",
                "--model",
                "--dataset",
                "/y",
                "--family",
                "chatml",
                "--boundary-ids",
                "1",
                "--output",
                str(Path(tmp) / "o"),
            )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("expected one argument", proc.stderr)

    def test_equals_form_defeats_flag_injection_then_name_validation_fires(self):
        """Test equals form defeats flag injection then name validation fires."""
        # "--boundary-map=-x=1" smuggles a dash-leading value past argparse,
        # but the candidate-name check still rejects it (exit 2, not a flag).
        with TemporaryDirectory() as tmp:
            proc = run_cli(
                "run",
                "--model",
                "/x",
                "--dataset",
                "/y",
                "--family",
                "chatml",
                "--boundary-ids",
                "1",
                "--output",
                str(Path(tmp) / "o"),
                "--boundary-map=-x=1",
            )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("unknown candidate", proc.stderr)

    def test_extremely_long_boundary_ids_rejected_fast(self):
        """Test extremely long boundary ids rejected fast."""
        with TemporaryDirectory() as tmp:
            proc = run_cli(
                "run",
                "--model",
                "/x",
                "--dataset",
                "/y",
                "--family",
                "chatml",
                "--boundary-ids",
                ",".join(["1"] * 10_000),
                "--output",
                str(Path(tmp) / "o"),
            )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("comma-separated ASCII token IDs", proc.stderr)


class FixturesValidationTests(unittest.TestCase):
    """`fixtures build`: authoring-input validation table."""

    def test_missing_positional_exits_2(self):
        """Test missing positional exits 2."""
        with TemporaryDirectory() as tmp:
            proc = run_cli("fixtures", "build", "--output", str(Path(tmp) / "x.json"))
        self.assertEqual(proc.returncode, 2)

    def test_missing_output_exits_2(self):
        """Test missing output exits 2."""
        proc = run_cli("fixtures", "build", str(EXAMPLE_CSV))
        self.assertEqual(proc.returncode, 2)

    def test_nonexistent_source_exits_1(self):
        """Test nonexistent source exits 1."""
        with TemporaryDirectory() as tmp:
            proc = run_cli(
                "fixtures",
                "build",
                "/nonexistent.csv",
                "--output",
                str(Path(tmp) / "out.json"),
            )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("fixtures: invalid authoring source:", proc.stderr)
        self.assertIn("Unreadable authoring file", proc.stderr)

    def test_existing_output_refused_with_exit_1(self):
        """Test existing output refused with exit 1."""
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            out.write_text("{}", encoding="utf-8")
            proc = run_cli("fixtures", "build", str(EXAMPLE_CSV), "--output", str(out))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("already exists", proc.stderr)

    def test_missing_parent_dir_exits_1(self):
        """Test missing parent dir exits 1."""
        with TemporaryDirectory() as tmp:
            proc = run_cli(
                "fixtures",
                "build",
                str(EXAMPLE_CSV),
                "--output",
                str(Path(tmp) / "nope" / "out.json"),
            )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("parent directory does not exist", proc.stderr)

    def test_sidecar_collision_writes_nothing_and_exits_1(self):
        """Test sidecar collision writes nothing and exits 1."""
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            sidecar = Path(str(out) + ".audit.json")
            sidecar.write_text("{}", encoding="utf-8")
            proc = run_cli("fixtures", "build", str(EXAMPLE_CSV), "--output", str(out))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("audit sidecar already exists", proc.stderr)
        self.assertFalse(out.exists(), "fixture file must not be left behind")

    def test_valid_csv_builds_fixture_and_sidecar(self):
        """Test valid csv builds fixture and sidecar."""
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            proc = run_cli("fixtures", "build", str(EXAMPLE_CSV), "--output", str(out))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["output"], str(out))
            self.assertIn("search_cases", payload)
            self.assertTrue(out.exists())
            self.assertTrue(Path(str(out) + ".audit.json").exists())
            built = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("search", built)
            self.assertIn("held_out", built)

    def test_malformed_csv_exits_1(self):
        """Test malformed csv exits 1."""
        with TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.csv"
            bad.write_text("not,a,valid,header\n1,2,3\n", encoding="utf-8")
            proc = run_cli(
                "fixtures",
                "build",
                str(bad),
                "--output",
                str(Path(tmp) / "out.json"),
            )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("fixtures: invalid authoring source:", proc.stderr)


class DiffReportsValidationTests(unittest.TestCase):
    """`diff-reports`: fixed error-code table (exit 1, stderr, no traceback)."""

    MINIMAL = {"decision": "x", "search": [], "selected_candidate": None, "held_out": None}

    @staticmethod
    def _write(tmp: Path, name: str, data) -> Path:
        """Write."""
        path = tmp / name
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            path.write_text(data, encoding="utf-8")
        return path

    def test_missing_positional_exits_2(self):
        """Test missing positional exits 2."""
        with TemporaryDirectory() as tmp:
            proc = run_cli("diff-reports", str(Path(tmp) / "a.json"))
        self.assertEqual(proc.returncode, 2)

    def test_missing_old_path_reports_results_path_missing(self):
        """Test missing old path reports results path missing."""
        with TemporaryDirectory() as tmp:
            new = self._write(Path(tmp), "new.json", json.dumps(self.MINIMAL))
            proc = run_cli("diff-reports", "/nonexistent-old.json", str(new))
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(proc.stdout, "")
        self.assertIn("diff-reports: results_path_missing (old input)", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_fifo_input_reports_results_not_regular_file(self):
        """Test fifo input reports results not regular file."""
        # FIFOs are refused via the fstat-on-opened-fd check. (A directory
        # input is NOT refused: it is treated as a report directory holding
        # results.json -- see test_report_directory_form_accepted. USAGE.md's
        # blanket "directories are refused" is imprecise; flagged in the
        # module docstring findings.)
        with TemporaryDirectory() as tmp:
            fifo = Path(tmp) / "input.fifo"
            os.mkfifo(fifo)
            new = self._write(Path(tmp), "new.json", json.dumps(self.MINIMAL))
            proc = run_cli("diff-reports", str(fifo), str(new))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("diff-reports: results_not_regular_file (old input)", proc.stderr)

    def test_report_directory_form_accepted(self):
        """Test report directory form accepted."""
        with TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report"
            report_dir.mkdir()
            (report_dir / "results.json").write_text(json.dumps(self.MINIMAL), encoding="utf-8")
            new = self._write(Path(tmp), "new.json", json.dumps(self.MINIMAL))
            proc = run_cli("diff-reports", str(report_dir), str(new))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(json.loads(proc.stdout)["summary"]["identical"])

    def test_oversized_input_reports_results_too_large(self):
        """Test oversized input reports results too large."""
        with TemporaryDirectory() as tmp:
            # Sparse file: load_results reads byte content (holes read as
            # zeros), so the >32 MiB results_too_large path triggers while
            # consuming ~0 tmpfs blocks. A real 34 MiB write flakes with
            # ENOSPC on the 512 MiB /tmp tmpfs (cycles 60/72 precedent).
            big = Path(tmp) / "big.json"
            with big.open("wb") as handle:
                handle.truncate(34 * 1024 * 1024)
            new = self._write(Path(tmp), "new.json", json.dumps(self.MINIMAL))
            proc = run_cli("diff-reports", str(big), str(new))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("diff-reports: results_too_large (old input)", proc.stderr)

    def test_duplicate_field_reports_results_duplicate_field(self):
        """Test duplicate field reports results duplicate field."""
        with TemporaryDirectory() as tmp:
            dup = self._write(Path(tmp), "dup.json", '{"decision": "a", "decision": "b"}')
            new = self._write(Path(tmp), "new.json", json.dumps(self.MINIMAL))
            proc = run_cli("diff-reports", str(dup), str(new))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("diff-reports: results_duplicate_field (old input)", proc.stderr)

    def test_nonfinite_number_reports_results_nonfinite_number(self):
        """Test nonfinite number reports results nonfinite number."""
        with TemporaryDirectory() as tmp:
            bad = self._write(Path(tmp), "bad.json", '{"decision": "a", "x": NaN}')
            new = self._write(Path(tmp), "new.json", json.dumps(self.MINIMAL))
            proc = run_cli("diff-reports", str(bad), str(new))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("diff-reports: results_nonfinite_number (old input)", proc.stderr)

    def test_non_utf8_reports_results_not_utf8(self):
        """Test non utf8 reports results not utf8."""
        with TemporaryDirectory() as tmp:
            bad = self._write(Path(tmp), "bad.json", b"\xff\xfe{}")
            new = self._write(Path(tmp), "new.json", json.dumps(self.MINIMAL))
            proc = run_cli("diff-reports", str(bad), str(new))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("diff-reports: results_not_utf8 (old input)", proc.stderr)

    def test_wrong_shape_reports_results_wrong_shape(self):
        """Test wrong shape reports results wrong shape."""
        with TemporaryDirectory() as tmp:
            bad = self._write(Path(tmp), "bad.json", json.dumps({"nope": 1}))
            new = self._write(Path(tmp), "new.json", json.dumps(self.MINIMAL))
            proc = run_cli("diff-reports", str(bad), str(new))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("diff-reports: results_wrong_shape (old input)", proc.stderr)

    def test_identical_reports_exit_0_json_on_stdout_summary_on_stderr(self):
        """Test identical reports exit 0 json on stdout summary on stderr."""
        with TemporaryDirectory() as tmp:
            old = self._write(Path(tmp), "old.json", json.dumps(self.MINIMAL))
            new = self._write(Path(tmp), "new.json", json.dumps(self.MINIMAL))
            proc = run_cli("diff-reports", str(old), str(new), "--all")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["summary"]["identical"])
        self.assertIn("diff-reports:", proc.stderr)


class DiffBackendsValidationTests(unittest.TestCase):
    """`diff-backends`: argparse table (exit 2) + loader/fixed-code paths."""

    @classmethod
    def setUpClass(cls):
        """Set up the test fixture."""
        try:
            import tokenizers  # noqa: F401  (availability probe)
        except ImportError:
            raise unittest.SkipTest("tokenizers package unavailable") from None
        cls._tmp = TemporaryDirectory()
        cls.checkpoint = make_tiny_checkpoint(Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        """Tear down the test fixture."""
        cls._tmp.cleanup()

    def _base(self, *extra: str):
        """Base."""
        return [
            "diff-backends",
            "--model",
            str(self.checkpoint),
            "--prompt-text",
            "the answer is",
            "--output-text",
            "42 STOP done",
            "--stops",
            "STOP",
            *extra,
        ]

    def test_missing_required_flags_exit_2(self):
        """Test missing required flags exit 2."""
        for drop in ["--model", "--prompt-text", "--output-text", "--stops"]:
            with self.subTest(drop=drop):
                argv = self._base()
                idx = argv.index(drop)
                del argv[idx : idx + 2]
                proc = run_cli(*argv)
                self.assertEqual(proc.returncode, 2, drop)

    def test_negative_min_tokens_exits_2(self):
        """Test negative min tokens exits 2."""
        proc = run_cli(*self._base("--min-tokens", "-1"))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--min-tokens must be a non-negative integer", proc.stderr)

    def test_negative_eos_id_exits_2(self):
        """Test negative eos id exits 2."""
        proc = run_cli(*self._base("--eos-id", "-1"))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--eos-id must be a non-negative integer", proc.stderr)

    def test_non_integer_eos_id_exits_2(self):
        """Test non integer eos id exits 2."""
        proc = run_cli(*self._base("--eos-id", "abc"))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("invalid int value", proc.stderr)

    def test_empty_stops_exits_2(self):
        """Test empty stops exits 2."""
        proc = run_cli(*self._base("--stops", ""))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("non-empty stop texts", proc.stderr)

    def test_blank_stop_entry_exits_2(self):
        """Test blank stop entry exits 2."""
        proc = run_cli(*self._base("--stops", "STOP,,END"))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("non-empty stop texts", proc.stderr)

    def _base_argv(self, *extra: str):
        """Base argv."""
        return [
            "diff-backends",
            "--model",
            str(self.checkpoint),
            "--prompt-text",
            "the answer is",
            "--output-text",
            "42 STOP done",
            "--stops",
            "STOP",
            *extra,
        ]

    def test_prompt_text_over_1mb_exits_2(self):
        """Test prompt text over 1mb exits 2."""
        # Linux MAX_ARG_STRLEN (128 KiB per argv string) means a 1 MB flag
        # value can never cross execve: the argparse bound is unreachable
        # via subprocess, so it is exercised unit-level with an explicit
        # argv list (no sys.argv mocking).
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer), self.assertRaises(SystemExit) as ctx:
            main(self._base_argv("--prompt-text", "a" * 1_000_001))
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("at most 1MB", buffer.getvalue())

    def test_output_text_over_1mb_exits_2(self):
        """Test output text over 1mb exits 2."""
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer), self.assertRaises(SystemExit) as ctx:
            main(self._base_argv("--output-text", "a" * 1_000_001))
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("at most 1MB", buffer.getvalue())

    def test_nonexistent_model_exits_1_with_sanitized_code(self):
        """Test nonexistent model exits 1 with sanitized code."""
        proc = run_cli(
            "diff-backends",
            "--model",
            "/nonexistent-checkpoint",
            "--prompt-text",
            "hi",
            "--output-text",
            "there",
            "--stops",
            "STOP",
        )
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(proc.stdout, "")
        # The raw loader error is sanitized to a fixed code; the missing
        # path itself must not reach the terminal.
        self.assertIn("diff-backends: tokenizer_load_failed", proc.stderr)
        self.assertNotIn("nonexistent-checkpoint", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_checkpoint_without_tokenizer_json_exits_1(self):
        """Test checkpoint without tokenizer json exits 1."""
        with TemporaryDirectory() as tmp:
            proc = run_cli(
                "diff-backends",
                "--model",
                tmp,
                "--prompt-text",
                "hi",
                "--output-text",
                "there",
                "--stops",
                "STOP",
            )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("diff-backends: tokenizer_load_failed", proc.stderr)

    def test_positive_path_exit_0_json_verdict(self):
        """Test positive path exit 0 json verdict."""
        proc = run_cli(*self._base())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertIn(payload["verdict"], ("agree", "diverge"))
        self.assertEqual(payload["stop_texts"], ["STOP"])

    def test_stop_text_too_long_exit_1_fixed_code(self):
        """Test stop text too long exit 1 fixed code."""
        proc = run_cli(*self._base("--stops", "s" * 1025))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("diff-backends: invalid input: stop_text_too_long", proc.stderr)

    def test_too_many_stop_texts_exit_1_fixed_code(self):
        """Test too many stop texts exit 1 fixed code."""
        stops = ",".join(f"stop{i}" for i in range(17))
        proc = run_cli(*self._base("--stops", stops))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("diff-backends: invalid input: too_many_stop_texts", proc.stderr)

    def test_output_ids_too_long_exit_1_fixed_code(self):
        """Test output ids too long exit 1 fixed code."""
        # 5000 whitespace-separated tokens > MAX_OUTPUT_IDS (4096).
        proc = run_cli(*self._base("--output-text", "x " * 5000))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("diff-backends: invalid input: output_ids_too_long", proc.stderr)

    def test_prompt_text_at_exactly_1mb_passes_cli_bound(self):
        """Test prompt text at exactly 1mb passes cli bound."""
        # Unit-level (see above): exactly 1 MB is within the bound, encodes
        # to a single token with the tiny checkpoint, and the run succeeds.
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            code = main(self._base_argv("--prompt-text", "a" * 1_000_000))
        self.assertEqual(code, 0, buffer.getvalue())
        self.assertNotIn("at most 1MB", buffer.getvalue())


class DebugValidationTests(unittest.TestCase):
    """`debug`: dataset/script/model validation table."""

    def test_missing_dataset_exits_2(self):
        """Test missing dataset exits 2."""
        proc = run_cli("debug")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("usage:", proc.stderr)

    def test_nonexistent_dataset_exits_1(self):
        """Test nonexistent dataset exits 1."""
        proc = run_cli("debug", "--dataset", "/nonexistent.json")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("debug: invalid dataset:", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_nonexistent_script_exits_1(self):
        """Test nonexistent script exits 1."""
        proc = run_cli(
            "debug",
            "--dataset",
            str(EXAMPLE_FIXTURES),
            "--script",
            "/nonexistent-script.txt",
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("debug: unreadable script file", proc.stderr)

    def test_directory_script_exits_1(self):
        """Test directory script exits 1."""
        with TemporaryDirectory() as tmp:
            proc = run_cli("debug", "--dataset", str(EXAMPLE_FIXTURES), "--script", tmp)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("debug: unreadable script file", proc.stderr)

    def test_oversized_script_refused(self):
        """Test oversized script refused."""
        with TemporaryDirectory() as tmp:
            big = Path(tmp) / "big.txt"
            big.write_bytes(b".families\n" * 200_000)  # > 1 MB
            proc = run_cli("debug", "--dataset", str(EXAMPLE_FIXTURES), "--script", str(big))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("larger than 1MB", proc.stderr)

    def test_scripted_quit_session_exits_0(self):
        """Test scripted quit session exits 0."""
        with TemporaryDirectory() as tmp:
            script = Path(tmp) / "s.txt"
            script.write_text(".families\n.quit\n", encoding="utf-8")
            proc = run_cli("debug", "--dataset", str(EXAMPLE_FIXTURES), "--script", str(script))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("chatml", proc.stdout)

    def test_script_error_does_not_kill_session(self):
        """Test script error does not kill session."""
        # The REPL converts command failures to "error: ..." and continues.
        with TemporaryDirectory() as tmp:
            script = Path(tmp) / "s.txt"
            script.write_text(".bogus-command\n.families\n.quit\n", encoding="utf-8")
            proc = run_cli("debug", "--dataset", str(EXAMPLE_FIXTURES), "--script", str(script))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("error:", proc.stdout)
        self.assertIn("chatml", proc.stdout)

    def test_bad_model_dir_exits_1_with_sanitized_code(self):
        """Test bad model dir exits 1 with sanitized code."""
        with TemporaryDirectory() as tmp:
            proc = run_cli("debug", "--dataset", str(EXAMPLE_FIXTURES), "--model", tmp)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("debug: tokenizer_load_failed", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_session_does_not_modify_dataset_file(self):
        """Test session does not modify dataset file."""
        with TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "d.json"
            dataset.write_bytes(_valid_fixture_bytes())
            before = dataset.read_bytes()
            script = Path(tmp) / "s.txt"
            script.write_text(
                ".use llama3\n.fixtures\n.fixture 1\n.render\n.boundary 1,2\n"
                ".stream 1,2,3\n.step\n.step\n.audit\n.save "
                + str(Path(tmp) / "t.txt")
                + "\n.quit\n",
                encoding="utf-8",
            )
            proc = run_cli("debug", "--dataset", str(dataset), "--script", str(script))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(dataset.read_bytes(), before)


class DiffBackendsCodeTableUnitTests(unittest.TestCase):
    """Unit-level: `diff_backends` raises the exact fixed-code vocabulary."""

    @staticmethod
    def _stub_backend():
        """Stub backend."""

        class Stub:
            """Stub."""

            @staticmethod
            def decode(ids):
                """Decode."""
                return "".join(f"<{i}>" for i in ids)

        return Stub()

    def test_fixed_code_vocabulary(self):
        """Test fixed code vocabulary."""
        from model_protocol_lab.diffbackends import diff_backends  # noqa: PLC0415

        backend = self._stub_backend()
        cases = [
            (
                {"prompt_ids": (1,) * 131073, "output_ids": (), "stop_texts": ["s"]},
                "prompt_ids_too_long",
            ),
            (
                {"prompt_ids": (), "output_ids": (1,) * 4097, "stop_texts": ["s"]},
                "output_ids_too_long",
            ),
            (
                {"prompt_ids": (), "output_ids": (), "stop_texts": ["s" * 1025]},
                "stop_text_too_long",
            ),
            (
                {"prompt_ids": (), "output_ids": (), "stop_texts": [f"s{i}" for i in range(17)]},
                "too_many_stop_texts",
            ),
            ({"prompt_ids": (), "output_ids": (), "stop_texts": []}, "empty_stop_texts"),
            (
                {"prompt_ids": (), "output_ids": (), "stop_texts": ["s"], "eos_id": -1},
                "invalid_eos_id",
            ),
            (
                {"prompt_ids": (), "output_ids": (), "stop_texts": ["s"], "min_tokens": -1},
                "invalid_min_tokens",
            ),
        ]
        for kwargs, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(ValueError) as ctx:
                    diff_backends(backend=backend, **kwargs)
                self.assertEqual(str(ctx.exception), code)


class ErrorCodeDisciplineTests(unittest.TestCase):
    """Every USAGE.md-documented error code is still emitted verbatim."""

    DOCUMENTED = {
        # code -> (module, needles that must all appear in its source).
        # prompt_ids_too_long / output_ids_too_long are raised via the
        # f"{name}_too_long" template in _checked_ids, so the needles pin
        # the template plus both call-site names; the exact emitted codes
        # are asserted at runtime in DiffBackendsCodeTableUnitTests.
        "results_too_large": ("diffreports.py", ['"results_too_large"']),
        "results_not_regular_file": ("diffreports.py", ['"results_not_regular_file"']),
        "results_duplicate_field": ("diffreports.py", ['"results_duplicate_field"']),
        "results_nonfinite_number": ("diffreports.py", ['"results_nonfinite_number"']),
        "output_ids_too_long": (
            "diffbackends.py",
            ["{name}_too_long", '"output_ids"'],
        ),
        "prompt_ids_too_long": (
            "diffbackends.py",
            ["{name}_too_long", '"prompt_ids"'],
        ),
        "stop_text_too_long": ("diffbackends.py", ['"stop_text_too_long"']),
        "too_many_stop_texts": ("diffbackends.py", ['"too_many_stop_texts"']),
        "local_context_limit_exceeds_declared_window": (
            "backends.py",
            ['"local_context_limit_exceeds_declared_window"'],
        ),
    }

    def test_documented_codes_present_in_source(self):
        """Test documented codes present in source."""
        package = REPO_ROOT / "model_protocol_lab"
        for code, (module, needles) in self.DOCUMENTED.items():
            with self.subTest(code=code):
                source = (package / module).read_text(encoding="utf-8")
                for needle in needles:
                    self.assertIn(needle, source, code)

    def test_no_traceback_on_any_input_failure_path(self):
        """Test no traceback on any input failure path."""
        # Spot-check the fixed-code paths already exercised above: none may
        # leak a traceback. (Full per-path assertions live in the classes
        # above; this is the discipline-level pin.)
        with TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            good = Path(tmp) / "good.json"
            good.write_text(
                json.dumps(
                    {"decision": "x", "search": [], "selected_candidate": None, "held_out": None}
                ),
                encoding="utf-8",
            )
            for argv in [
                ("inspect", str(bad)),
                ("inspect", "/nonexistent.json"),
                ("diff-reports", str(bad), str(good)),
                ("diff-reports", "/nonexistent.json", str(good)),
                ("fixtures", "build", "/nonexistent.csv", "--output", str(Path(tmp) / "o.json")),
            ]:
                with self.subTest(argv=argv[:2]):
                    proc = run_cli(*argv)
                    self.assertEqual(proc.returncode, 1, argv)
                    self.assertNotIn("Traceback", proc.stderr, argv)
                    self.assertNotIn("Traceback", proc.stdout, argv)


def _snapshot_split(split):
    """Snapshot split."""

    def snap_case(case):
        """Snap case."""
        return (
            case.id,
            case.group,
            tuple((m.role, m.content) for m in case.messages),
            case.expected,
        )

    return (
        tuple(snap_case(c) for c in split.search),
        tuple(snap_case(c) for c in split.held_out),
    )


class DebugReadOnlyTests(unittest.TestCase):
    """The debug REPL cannot mutate the selection pipeline (unit-level)."""

    def setUp(self):
        """Set up the test fixture."""
        self.split = load_dataset(EXAMPLE_FIXTURES)
        self.split_snapshot = _snapshot_split(self.split)
        self.catalog_snapshot = {name: asdict(fmt) for name, fmt in CATALOG.items()}
        self.stdout = io.StringIO()
        self.shell = DebugShell(self.split, dataset_label="audit", stdout=self.stdout)

    def _run_battery(self):
        """Run battery."""
        shell = self.shell
        for line in [
            ".use llama3",
            ".fixtures",
            ".fixture 1",
            ".render",
            ".boundary 5,6",
            ".stream 5,6,7",
            ".step",
            ".step",
            ".step",
            ".step",  # exhausted stream -> "error: ...", session continues
            ".audit",
            ".use chatml",
            ".fixture 2",
        ]:
            shell.onecmd(line)

    def test_split_content_unchanged_after_command_battery(self):
        """Test split content unchanged after command battery."""
        self._run_battery()
        self.assertEqual(_snapshot_split(self.split), self.split_snapshot)

    def test_split_partition_objects_are_shared_not_copied(self):
        """Test split partition objects are shared not copied."""
        # The REPL merges partitions into a NEW tuple for inspection; the
        # underlying Fixture objects are shared references, never mutated.
        self._run_battery()
        self.assertIs(self.shell._cases[0], self.split.search[0])  # noqa: SLF001
        self.assertIs(self.shell._cases[-1], self.split.held_out[-1])  # noqa: SLF001

    def test_catalog_unchanged_after_command_battery(self):
        """Test catalog unchanged after command battery."""
        self._run_battery()
        self.assertEqual(
            {name: asdict(fmt) for name, fmt in CATALOG.items()},
            self.catalog_snapshot,
        )

    def test_session_state_is_shell_local(self):
        """Test session state is shell local."""
        self._run_battery()
        shell = self.shell
        self.assertEqual(shell._boundary, (5, 6))  # noqa: SLF001
        self.assertEqual(shell._stream, (5, 6, 7))  # noqa: SLF001
        self.assertEqual(shell._seen, [5, 6, 7])  # noqa: SLF001
        self.assertEqual(shell._family, "chatml")  # noqa: SLF001
        # Nothing on the shell reaches back into the experiment pipeline.
        self.assertFalse(hasattr(shell, "run_experiment"))
        self.assertFalse(hasattr(shell, "selected_candidate"))

    def test_debug_module_never_imports_experiments(self):
        """Test debug module never imports experiments."""
        # "experiments" appears in prose (module docstring); what must not
        # appear is an import binding the selection pipeline into the REPL.
        source = (REPO_ROOT / "model_protocol_lab" / "debug.py").read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                self.assertNotIn("experiments", stripped, line)

    def test_say_and_save_only_touch_operator_files(self):
        """Test say and save only touch operator files."""
        with TemporaryDirectory() as tmp:
            script = Path(tmp) / "s.txt"
            script.write_text(
                ".say\nuser: hello\nassistant: hi\n.\n.render\n.save "
                + str(Path(tmp) / "transcript.txt")
                + "\n.quit\n",
                encoding="utf-8",
            )
            shell = DebugShell(self.split, dataset_label="audit", stdout=io.StringIO())
            self.assertTrue(shell.read_commands(script))
            saved = Path(tmp) / "transcript.txt"
            self.assertTrue(saved.exists())
            text = saved.read_text(encoding="utf-8")
            self.assertIn(".render", text)  # transcript echo, not experiment state
        self.assertEqual(_snapshot_split(self.split), self.split_snapshot)
        self.assertEqual(
            {name: asdict(fmt) for name, fmt in CATALOG.items()},
            self.catalog_snapshot,
        )


def _console_script_path():
    """Console script path."""
    found = shutil.which("model-protocol-lab")
    if found:
        return found
    candidate = Path(sys.prefix) / "bin" / "model-protocol-lab"
    return str(candidate) if candidate.is_file() else None


class ConsoleScriptEquivalenceTests(unittest.TestCase):
    """`model-protocol-lab` entry point == `python -m model_protocol_lab`."""

    @classmethod
    def setUpClass(cls):
        """Set up the test fixture."""
        cls.script = _console_script_path()
        if cls.script is None:
            raise unittest.SkipTest("model-protocol-lab console script not installed")

    def _run_script(self, *args: str):
        """Run script."""
        return subprocess.run(
            [self.script, *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=CALL_TIMEOUT,
            check=False,
        )

    def _assert_identical(self, argv):
        """Assert identical."""
        via_module = run_cli(*argv)
        via_script = self._run_script(*argv)
        self.assertEqual(via_script.returncode, via_module.returncode, argv)
        self.assertEqual(via_script.stdout, via_module.stdout, argv)
        self.assertEqual(via_script.stderr, via_module.stderr, argv)

    def test_help_identical(self):
        """Test help identical."""
        self._assert_identical(("--help",))

    def test_subcommand_help_identical(self):
        """Test subcommand help identical."""
        self._assert_identical(("catalog", "--help"))

    def test_success_path_identical(self):
        """Test success path identical."""
        self._assert_identical(("inspect", str(EXAMPLE_FIXTURES)))

    def test_argparse_error_path_identical(self):
        """Test argparse error path identical."""
        self._assert_identical(("frobnicate",))

    def test_fixed_code_error_path_identical(self):
        """Test fixed code error path identical."""
        self._assert_identical(("inspect", "/nonexistent-xyz.json"))

    def test_run_validation_error_path_identical(self):
        """Test run validation error path identical."""
        self._assert_identical(("run", "--model", "/x"))

    def test_unit_main_catalog_matches_subprocess(self):
        """Test unit main catalog matches subprocess."""
        # Unit-level main([...]) with an explicit argv list (no sys.argv
        # mocking) agrees with the subprocess entry point.
        buffer = io.StringIO()
        old = sys.stdout
        sys.stdout = buffer
        try:
            code = main(["catalog"])
        finally:
            sys.stdout = old
        self.assertEqual(code, 0)
        proc = run_cli("catalog")
        self.assertEqual(json.loads(buffer.getvalue()), json.loads(proc.stdout))

    def test_console_scripts_no_mpl_alias(self):
        """Test console scripts no mpl alias."""
        # `mpl` appears in docs as shorthand; it is NOT an installed entry
        # point. Pin the real surface: the CLI plus, since cycle 208
        # (M2(a): the MCP server ships in the package), the MCP server
        # script — and nothing else. Cycle-208 lesson: this test reads
        # the INSTALLED distribution, so after any [project.scripts]
        # change the venv must be reinstalled before gating, or stale
        # dist-info entry points make it pass locally while CI's fresh
        # install fails.
        entry_points = [
            ep.name
            for ep in distribution("model-protocol-lab").entry_points
            if ep.group == "console_scripts"
        ]
        self.assertEqual(entry_points, ["model-protocol-lab", "model-protocol-lab-mcp"])


class RedTeamTests(unittest.TestCase):
    """Hostile / degenerate inputs across the CLI surface."""

    def test_unicode_path_no_crash(self):
        """Test unicode path no crash."""
        with TemporaryDirectory() as tmp:
            proc = run_cli("inspect", str(Path(tmp) / "ünïcödé-☃-🚫.json"))
        self.assertEqual(proc.returncode, 1)
        self.assertNotIn("Traceback", proc.stderr)

    def test_nul_byte_in_path_no_crash(self):
        """Test nul byte in path no crash."""
        # NUL bytes cannot cross execve at all (the OS refuses the argv, and
        # Python's subprocess raises ValueError before the child starts), so
        # this red-team case is exercised unit-level with an explicit argv
        # list: the CLI must still exit 1 with a fixed message, no traceback.
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            code = main(["inspect", "a\x00b.json"])
        self.assertEqual(code, 1)
        self.assertIn("inspect: invalid fixture file:", buffer.getvalue())
        self.assertNotIn("Traceback", buffer.getvalue())

    def test_control_characters_in_fixture_id_sanitized(self):
        """Test control characters in fixture id sanitized."""
        # A hostile fixture id must not inject terminal escapes into the
        # stderr warning; cli._display_id strips control characters.
        with TemporaryDirectory() as tmp:
            hostile = {
                "search": [
                    {
                        "id": "x\x1b[31mRED\x1b[0m",
                        "group": "g",
                        "messages": [{"role": "user", "content": "hi"}],
                        "expected": "",
                    }
                ],
                "held_out": [
                    {
                        "id": "h1",
                        "group": "h",
                        "messages": [{"role": "user", "content": "bye"}],
                        "expected": "x",
                    }
                ],
            }
            path = Path(tmp) / "hostile.json"
            path.write_text(json.dumps(hostile), encoding="utf-8")
            proc = run_cli("inspect", str(path))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("\x1b", proc.stderr)
        self.assertIn("empty expected answer", proc.stderr)

    def test_extremely_long_single_stop_rejected(self):
        """Test extremely long single stop rejected."""
        # Unit-level: a 200_000-char argv string exceeds MAX_ARG_STRLEN, so
        # this goes through main() directly. The CLI --stops check only
        # rejects empties; the length bound fires inside diff_backends as
        # the fixed code stop_text_too_long (exit 1).
        with TemporaryDirectory() as tmp:
            checkpoint = make_tiny_checkpoint(Path(tmp))
            buffer = io.StringIO()
            with contextlib.redirect_stderr(buffer):
                code = main(
                    [
                        "diff-backends",
                        "--model",
                        str(checkpoint),
                        "--prompt-text",
                        "hi",
                        "--output-text",
                        "there",
                        "--stops",
                        "s" * 200_000,
                    ]
                )
        self.assertEqual(code, 1)
        self.assertIn("stop_text_too_long", buffer.getvalue())

    def test_many_duplicate_stops_deduplicated_not_rejected(self):
        """Test many duplicate stops deduplicated not rejected."""
        # 100 identical stops dedupe to 1: below MAX_STOP_TEXTS after dedup.
        with TemporaryDirectory() as tmp:
            checkpoint = make_tiny_checkpoint(Path(tmp))
            proc = run_cli(
                "diff-backends",
                "--model",
                str(checkpoint),
                "--prompt-text",
                "hi",
                "--output-text",
                "there",
                "--stops",
                ",".join(["STOP"] * 100),
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["stop_texts"], ["STOP"])

    def test_debug_script_with_crlf_and_comments(self):
        """Test debug script with crlf and comments."""
        with TemporaryDirectory() as tmp:
            script = Path(tmp) / "s.txt"
            script.write_bytes(b"# comment\r\n.families\r\n\r\n.quit\r\n")
            proc = run_cli("debug", "--dataset", str(EXAMPLE_FIXTURES), "--script", str(script))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("chatml", proc.stdout)

    def test_empty_string_flag_values_rejected(self):
        """Test empty string flag values rejected."""
        # --model "" -> _directory rejects empty; surfaces as a fixed code.
        proc = run_cli(
            "diff-backends",
            "--model",
            "",
            "--prompt-text",
            "hi",
            "--output-text",
            "there",
            "--stops",
            "STOP",
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("tokenizer_load_failed", proc.stderr)

    def test_catalog_rejects_positional(self):
        """Test catalog rejects positional."""
        proc = run_cli("catalog", "extra")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("unrecognized arguments", proc.stderr)

    def test_repeated_scalar_flag_last_wins_no_crash(self):
        """Test repeated scalar flag last wins no crash."""
        proc = run_cli("inspect", str(EXAMPLE_FIXTURES), "--help", "--help")
        self.assertEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
