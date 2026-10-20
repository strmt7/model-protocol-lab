"""Cycle 31: verification of model_protocol_lab/cli.py (CLI parsing + dispatch).

Pins the behavioral contract of the command-line interface:

* every subcommand is reachable and dispatches to the right handler with
  the parsed arguments (handlers are monkeypatched; no model, no network);
* argument parsing / validation rules (types, choices, ranges, required
  flags) and their error paths (argparse errors -> SystemExit(2), handler
  failures -> small integer return codes);
* determinism (repeated invocations produce identical results);
* hostile argv (missing args, unknown commands, abbreviations, type
  errors, unicode digit tricks, control characters, pathological lists).

Handlers that would load a model (``run``) or reach the network are never
executed: the ``run`` path is tested with a fake backend and patched
experiment/reporting helpers.  Everything else exercises the real code.

BUGS FOUND
----------
None as of this writing; see the docstring-divergence notes pinned below.

DOC / HELP-TEXT NOTES (code behavior pinned; source untouched)
--------------------------------------------------------------
1. ``run --help`` advertises ``--max-drop-bytes`` as "(1-8)" and the code
   enforces ``1 <= max_drop_bytes <= 8`` -- consistent.
2. ``fixtures build --output`` help says "an existing path is never
   overwritten" -- pinned: rebuilding onto an existing fixture path or an
   existing ``.audit.json`` sidecar exits 1 and writes nothing new.
3. ``doctor --portability-matrix`` help routes the human-readable table to
   stderr so stdout stays parseable JSON -- pinned.
4. ``parse_boundary`` help text says "Token IDs must be < 2^31" while the
   code enforces ``token > 2**31 - 1`` (i.e. the max accepted value is
   2^31 - 1, which is < 2^31) -- consistent.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import model_protocol_lab
from model_protocol_lab import backends, cli, experiments
from model_protocol_lab import doctor as doctor_mod
from model_protocol_lab import reporting as reporting_mod
from model_protocol_lab.catalog import CATALOG
from model_protocol_lab.cli import (
    _display_id,
    build_parser,
    main,
    parse_boundary,
    parse_boundary_map,
)


def run_cli(argv):
    """Run main(argv) capturing stdio.

    Returns (kind, code, stdout, stderr) where kind is "exit" (SystemExit
    raised, code is the exit status) or "return" (main returned, code is
    the integer return value).
    """
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
    except SystemExit as exc:
        return ("exit", exc.code, out.getvalue(), err.getvalue())
    return ("return", code, out.getvalue(), err.getvalue())


def write_fixture_file(directory):
    """Write a minimal valid fixture JSON file; return its Path."""
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
        )
    )
    return path


def write_csv(directory, name="authoring.csv"):
    path = Path(directory) / name
    path.write_text(
        "id,group,partition,expected,user_1\ns1,g1,search,yo,hi\nh1,g2,held_out,yo,hey\n"
    )
    return path


def run_argv(family="chatml", **overrides):
    """Base argv for the ``run`` subcommand; overrides replace flags."""
    argv = [
        "run",
        "--model",
        "ckpt-dir",
        "--dataset",
        "fixture.json",
        "--family",
        family,
        "--boundary-ids",
        "198",
        "--output",
        "fresh-output-dir",
    ]
    for flag, value in overrides.items():
        argv.extend(["--" + flag.replace("_", "-"), value])
    return argv


class TestParseBoundary(unittest.TestCase):
    def test_single(self):
        self.assertEqual(parse_boundary("198"), (198,))

    def test_multiple(self):
        self.assertEqual(parse_boundary("1,2,3"), (1, 2, 3))

    def test_whitespace_tolerated(self):
        self.assertEqual(parse_boundary(" 12 , 34 "), (12, 34))

    def test_sixteen_ids_ok(self):
        self.assertEqual(parse_boundary(",".join(["7"] * 16)), (7,) * 16)

    def test_zero_allowed(self):
        self.assertEqual(parse_boundary("0"), (0,))

    def test_max_value_boundary(self):
        self.assertEqual(parse_boundary(str(2**31 - 1)), (2**31 - 1,))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_boundary(str(2**31))

    def test_rejects(self):
        bad = [
            "",
            " ",
            ",",
            "1,",
            ",1",
            "1,,2",
            "abc",
            "1,abc",
            "12.5",
            "-5",
            "+5",
            "0x10",
            "1_0",
            ",".join(["1"] * 17),
            "１２３",  # fullwidth digits: isdecimal() would accept these
            "١٢٣",  # arabic-indic digits
            "²",  # superscript: isdecimal() accepts, not ASCII
            "−5",  # U+2212 minus sign
            "5\u00a05",  # nbsp inside a part
            "9" * 40,  # huge but still < 2^31? no -> rejected by ceiling
        ]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_boundary(value)

    def test_leading_zeros_ok(self):
        self.assertEqual(parse_boundary("007"), (7,))

    def test_returns_tuple_of_int(self):
        result = parse_boundary("3,4")
        self.assertIs(type(result), tuple)
        self.assertTrue(all(type(v) is int for v in result))


class TestParseBoundaryMap(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(parse_boundary_map("chatml=198"), ("chatml", (198,)))

    def test_name_with_slash(self):
        name, ids = parse_boundary_map("chatml/toggle-bos=1,2")
        self.assertEqual((name, ids), ("chatml/toggle-bos", (1, 2)))

    def test_split_on_first_equals(self):
        # Names never contain "=", ids never contain it either; the first
        # "=" is the separator and the rest goes to parse_boundary.
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_boundary_map("a=b=1")

    def test_name_stripped(self):
        self.assertEqual(parse_boundary_map("  chatml =198"), ("chatml", (198,)))

    def test_rejects(self):
        for value in ["", "=198", "chatml", "chatml=", "chatml=abc", "= ", "chatml=1,,2"]:
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_boundary_map(value)


class TestDisplayId(unittest.TestCase):
    def test_strips_control_chars(self):
        self.assertEqual(_display_id("s1\x1b[2Jm\x00x\x7f"), "s1[2Jmx")

    def test_plain_unchanged(self):
        self.assertEqual(_display_id("fixture-12"), "fixture-12")

    def test_unicode_kept(self):
        self.assertEqual(_display_id("fïxturé"), "fïxturé")


class TestParserContract(unittest.TestCase):
    def test_prog_name(self):
        self.assertEqual(build_parser().prog, "model-protocol-lab")

    def test_all_subcommands_registered(self):
        parser = build_parser()
        actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        self.assertEqual(len(actions), 1)
        self.assertEqual(
            set(actions[0].choices),
            {
                "catalog",
                "inspect",
                "doctor",
                "run",
                "fixtures",
                "diff-reports",
                "diff-backends",
                "debug",
                "verify",
                "surface-noise",
                "pack",
            },
        )

    def test_command_required(self):
        kind, code, _out, err = run_cli([])
        self.assertEqual((kind, code), ("exit", 2))
        self.assertIn("the following arguments are required: command", err)

    def test_abbreviations_rejected(self):
        # allow_abbrev=False: prefixes must not resolve to subcommands.
        for argv in (["cat"], ["insp"], ["ru"], ["deb"], ["diff"]):
            with self.subTest(argv=argv):
                kind, code, _out, err = run_cli(argv)
                self.assertEqual((kind, code), ("exit", 2))
                self.assertIn("invalid choice", err)

    def test_option_abbreviations_rejected(self):
        # allow_abbrev=False on the main parser and on every subparser:
        # unambiguous option prefixes must not resolve.
        argv = run_argv()
        argv[argv.index("--boundary-ids")] = "--bound"
        cases = [
            ["--hel"],  # main parser: would resolve to --help
            ["catalog", "--hel"],  # subparser: would resolve to --help
            ["doctor", "--mod", "m"],  # subparser: would resolve to --model
            argv,  # run: --bound would resolve to --boundary-ids
        ]
        for argv_case in cases:
            with self.subTest(argv=argv_case[:3]):
                kind, code, _out, _err = run_cli(argv_case)
                self.assertEqual((kind, code), ("exit", 2))

    def test_unknown_command(self):
        kind, code, _out, err = run_cli(["frobnicate"])
        self.assertEqual((kind, code), ("exit", 2))
        self.assertIn("invalid choice", err)

    def test_run_defaults(self):
        ns = build_parser().parse_args(run_argv())
        self.assertEqual(ns.max_new_tokens, 64)
        self.assertEqual(ns.max_context, 2048)
        self.assertEqual(ns.seed, 0)
        self.assertEqual(ns.device, "cpu")
        self.assertFalse(ns.mutate)
        self.assertEqual(ns.max_drop_bytes, 3)
        self.assertFalse(ns.sensitivity_sweep)
        self.assertFalse(ns.logprob_diagnostic)
        self.assertFalse(ns.include_evidence)
        self.assertIsNone(ns.boundary_map)

    def test_run_types(self):
        ns = build_parser().parse_args(run_argv())
        self.assertIsInstance(ns.model, Path)
        self.assertIsInstance(ns.dataset, Path)
        self.assertIsInstance(ns.output, Path)
        self.assertEqual(ns.boundary_ids, (198,))

    def test_doctor_defaults(self):
        ns = build_parser().parse_args(["doctor", "--model", "m"])
        self.assertIsNone(ns.dataset)
        self.assertIsNone(ns.boundary_ids)
        self.assertIsNone(ns.boundary_map)
        self.assertIsNone(ns.reference_model)
        self.assertIsNone(ns.portability_matrix)

    def test_diff_backends_types(self):
        ns = build_parser().parse_args(
            [
                "diff-backends",
                "--model",
                "m",
                "--prompt-text",
                "p",
                "--output-text",
                "o",
                "--stops",
                "STOP",
                "--eos-id",
                "5",
                "--min-tokens",
                "2",
            ]
        )
        self.assertEqual(ns.eos_id, 5)
        self.assertEqual(ns.min_tokens, 2)
        self.assertIsNone(
            build_parser()
            .parse_args(
                [
                    "diff-backends",
                    "--model",
                    "m",
                    "--prompt-text",
                    "p",
                    "--output-text",
                    "o",
                    "--stops",
                    "s",
                ]
            )
            .eos_id
        )

    def test_help_exits_zero_everywhere(self):
        argvs = [
            ["--help"],
            ["catalog", "--help"],
            ["inspect", "--help"],
            ["doctor", "--help"],
            ["run", "--help"],
            ["fixtures", "--help"],
            ["fixtures", "build", "--help"],
            ["diff-reports", "--help"],
            ["diff-backends", "--help"],
            ["debug", "--help"],
        ]
        for argv in argvs:
            with self.subTest(argv=argv):
                kind, code, out, _err = run_cli(argv)
                self.assertEqual((kind, code), ("exit", 0))
                self.assertIn("model-protocol-lab", out)

    def test_help_text_matches_enforced_ranges(self):
        # --max-drop-bytes help advertises "(1-8)"; the validator enforces
        # exactly 1..8.
        _kind, _code, out, _err = run_cli(["run", "--help"])
        self.assertIn("1-8", out)

    def test_fixtures_requires_subcommand(self):
        kind, code, _out, err = run_cli(["fixtures"])
        self.assertEqual((kind, code), ("exit", 2))
        self.assertIn("the following arguments are required", err)

    def test_parse_deterministic(self):
        argv = run_argv()
        first = build_parser().parse_args(argv)
        second = build_parser().parse_args(argv)
        self.assertEqual(first, second)


class TestCatalogDispatch(unittest.TestCase):
    def test_catalog_returns_zero_and_json(self):
        kind, code, out, _err = run_cli(["catalog"])
        self.assertEqual((kind, code), ("return", 0))
        payload = json.loads(out)
        self.assertEqual(payload, [asdict(fmt) for fmt in CATALOG.values()])
        names = [entry["name"] for entry in payload]
        self.assertEqual(len(names), len(set(names)))

    def test_catalog_deterministic(self):
        first = run_cli(["catalog"])[2]
        second = run_cli(["catalog"])[2]
        self.assertEqual(first, second)


class TestHandlerDispatch(unittest.TestCase):
    def test_doctor_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = write_fixture_file(tmp)
            seen = {}

            def fake_doctor(
                model,
                dataset,
                boundary_ids=None,
                boundary_map=None,
                reference_model_path=None,
                portability_matrix_path=None,
                template_probe=False,
            ):
                seen.update(
                    model=model,
                    dataset=dataset,
                    boundary_ids=boundary_ids,
                    boundary_map=boundary_map,
                    reference_model_path=reference_model_path,
                    portability_matrix_path=portability_matrix_path,
                    template_probe=template_probe,
                )
                return {"verdict": "ok"}

            with mock.patch.object(doctor_mod, "doctor", fake_doctor):
                kind, code, out, _err = run_cli(
                    [
                        "doctor",
                        "--model",
                        "ckpt-dir",
                        "--dataset",
                        str(fixture),
                        "--boundary-ids",
                        "1,2",
                        "--boundary-map",
                        "x=3",
                        "--reference-model",
                        "ref-dir",
                    ]
                )
            self.assertEqual((kind, code), ("return", 0))
            self.assertEqual(seen["model"], Path("ckpt-dir"))
            self.assertEqual(seen["dataset"], Path(str(fixture)))
            self.assertEqual(seen["boundary_ids"], (1, 2))
            self.assertEqual(seen["boundary_map"], {"x": (3,)})
            self.assertEqual(seen["reference_model_path"], Path("ref-dir"))
            self.assertIsNone(seen["portability_matrix_path"])
            self.assertIs(seen["template_probe"], False)
            self.assertEqual(json.loads(out)["verdict"], "ok")

    def test_doctor_verdict_not_ok_returns_one(self):
        with mock.patch.object(doctor_mod, "doctor", return_value={"verdict": "bad"}):
            kind, code, _out, _err = run_cli(["doctor", "--model", "m"])
        self.assertEqual((kind, code), ("return", 1))

    def test_doctor_portability_matrix_renders_to_stderr(self):
        section = {"rows": []}

        def fake_doctor(model, dataset, **kwargs):
            return {"verdict": "ok", "portability_matrix": section}

        with (
            mock.patch.object(doctor_mod, "doctor", fake_doctor),
            mock.patch.object(
                doctor_mod, "render_portability_table", return_value="TABLE"
            ) as rendered,
        ):
            kind, code, out, err = run_cli(
                ["doctor", "--model", "m", "--portability-matrix", "ref-dir"]
            )
        self.assertEqual((kind, code), ("return", 0))
        rendered.assert_called_once_with(section)
        self.assertIn("TABLE", err)
        # stdout stays parseable JSON
        json.loads(out)

    def test_doctor_duplicate_boundary_map_exits_two(self):
        kind, code, _out, err = run_cli(
            [
                "doctor",
                "--model",
                "m",
                "--boundary-map",
                "x=1",
                "--boundary-map",
                "x=2",
            ]
        )
        self.assertEqual((kind, code), ("exit", 2))
        self.assertIn("Duplicate --boundary-map", err)

    def test_diff_reports_dispatch(self):
        seen = {}

        def fake(args, parser):
            seen.update(args=args, parser=parser)
            return 7

        with mock.patch("model_protocol_lab.diffreports.run_diff_reports", fake):
            kind, code, _out, _err = run_cli(
                ["diff-reports", "old-results", "new-results", "--all"]
            )
        self.assertEqual((kind, code), ("return", 7))
        self.assertEqual(seen["args"].old_results, Path("old-results"))
        self.assertEqual(seen["args"].new_results, Path("new-results"))
        self.assertTrue(seen["args"].all)
        self.assertEqual(seen["parser"].prog, "model-protocol-lab")

    def test_diff_backends_dispatch(self):
        seen = {}

        def fake(args, parser):
            seen["args"] = args
            return 0

        with mock.patch("model_protocol_lab.diffbackends.run_diff_backends", fake):
            kind, code, _out, _err = run_cli(
                [
                    "diff-backends",
                    "--model",
                    "m",
                    "--prompt-text",
                    "hello",
                    "--output-text",
                    "world",
                    "--stops",
                    "STOP,</answer>",
                    "--eos-id",
                    "2",
                    "--min-tokens",
                    "1",
                ]
            )
        self.assertEqual((kind, code), ("return", 0))
        args = seen["args"]
        self.assertEqual(args.prompt_text, "hello")
        self.assertEqual(args.output_text, "world")
        self.assertEqual(args.stops, "STOP,</answer>")
        self.assertEqual(args.eos_id, 2)
        self.assertEqual(args.min_tokens, 1)

    def test_debug_dispatch(self):
        seen = {}

        def fake(args):
            seen["args"] = args
            return 3

        with tempfile.TemporaryDirectory() as tmp:
            fixture = write_fixture_file(tmp)
            with mock.patch("model_protocol_lab.debug.run_debug", fake):
                kind, code, _out, _err = run_cli(
                    ["debug", "--dataset", str(fixture), "--script", "s.read"]
                )
        self.assertEqual((kind, code), ("return", 3))
        args = seen["args"]
        self.assertEqual(args.dataset, Path(str(fixture)))
        self.assertIsNone(args.model)
        self.assertEqual(args.script, Path("s.read"))

    def test_debug_missing_dataset_exits_two(self):
        kind, code, _out, _err = run_cli(["debug"])
        self.assertEqual((kind, code), ("exit", 2))

    def test_diff_reports_missing_positional_exits_two(self):
        kind, code, _out, _err = run_cli(["diff-reports", "old-results"])
        self.assertEqual((kind, code), ("exit", 2))


class TestInspect(unittest.TestCase):
    def test_valid_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = write_fixture_file(tmp)
            kind, code, out, _err = run_cli(["inspect", str(fixture)])
        self.assertEqual((kind, code), ("return", 0))
        payload = json.loads(out)
        self.assertEqual(
            payload,
            {
                "search_cases": 1,
                "held_out_cases": 1,
                "valid": True,
                "contamination": {"worst_tier": "clean", "pairs": []},
            },
        )

    def test_missing_positional_exits_two(self):
        kind, code, _out, _err = run_cli(["inspect"])
        self.assertEqual((kind, code), ("exit", 2))

    def test_invalid_json_returns_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{not json")
            kind, code, _out, err = run_cli(["inspect", str(bad)])
        self.assertEqual((kind, code), ("return", 1))
        self.assertIn("inspect: invalid fixture file:", err)

    def test_missing_file_returns_one(self):
        kind, code, _out, err = run_cli(["inspect", "/nonexistent/f.json"])
        self.assertEqual((kind, code), ("return", 1))
        self.assertIn("inspect: invalid fixture file:", err)

    def test_directory_returns_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            kind, code, _out, err = run_cli(["inspect", tmp])
        self.assertEqual((kind, code), ("return", 1))

    def test_empty_expected_warns_and_sanitizes(self):
        # A hostile fixture id with terminal escapes must be stripped in
        # the stderr warning (no ESC byte may reach the terminal).
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "evil.json"
            fixture.write_text(
                json.dumps(
                    {
                        "search": [
                            {
                                "id": "s1\x1b[2J",
                                "group": "g1",
                                "messages": [{"role": "user", "content": "hi"}],
                                "expected": "",
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
                )
            )
            kind, code, out, err = run_cli(["inspect", str(fixture)])
        self.assertEqual((kind, code), ("return", 0))
        self.assertNotIn("\x1b", err)
        self.assertIn("empty expected answer", err)
        self.assertIn("s1[2J", err)
        self.assertEqual(json.loads(out)["valid"], True)

    def test_nul_byte_path_returns_one(self):
        kind, code, _out, err = run_cli(["inspect", "a\x00b"])
        self.assertEqual((kind, code), ("return", 1))
        self.assertIn("inspect: invalid fixture file:", err)


class TestFixturesBuild(unittest.TestCase):
    def test_build_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = write_csv(tmp)
            out = Path(tmp) / "f.json"
            kind, code, stdout, _err = run_cli(
                ["fixtures", "build", str(src), "--output", str(out)]
            )
            self.assertEqual((kind, code), ("return", 0))
            payload = json.loads(stdout)
            self.assertEqual(
                payload,
                {
                    "search_cases": 1,
                    "held_out_cases": 1,
                    "output": str(out),
                    "audit_findings": 0,
                    "audit_path": str(out) + ".audit.json",
                    "contamination": {"worst_tier": "clean", "pairs": []},
                },
            )
            self.assertTrue(out.exists())
            self.assertTrue(Path(str(out) + ".audit.json").exists())

    def test_build_missing_args_exit_two(self):
        kind, code, _out, _err = run_cli(["fixtures", "build"])
        self.assertEqual((kind, code), ("exit", 2))

    def test_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = write_csv(tmp)
            out = Path(tmp) / "f.json"
            out.write_text("sentinel")
            kind, code, _out, err = run_cli(["fixtures", "build", str(src), "--output", str(out)])
            self.assertEqual((kind, code), ("return", 1))
            self.assertIn("fixtures: invalid authoring source:", err)
            self.assertEqual(out.read_text(), "sentinel")

    def test_existing_sidecar_blocks_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = write_csv(tmp)
            out = Path(tmp) / "f.json"
            Path(str(out) + ".audit.json").write_text("sentinel")
            kind, code, _out, err = run_cli(["fixtures", "build", str(src), "--output", str(out)])
            self.assertEqual((kind, code), ("return", 1))
            self.assertIn("audit sidecar already exists", err)
            self.assertFalse(out.exists())

    def test_invalid_csv_returns_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "bad.csv"
            src.write_text("id,group,partition,expected,user_1\ns1,g1,search,yo\n")
            kind, code, _out, err = run_cli(
                ["fixtures", "build", str(src), "--output", str(Path(tmp) / "o.json")]
            )
        self.assertEqual((kind, code), ("return", 1))
        self.assertIn("fixtures: invalid authoring source:", err)

    def test_missing_source_returns_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            kind, code, _out, err = run_cli(
                [
                    "fixtures",
                    "build",
                    str(Path(tmp) / "nope.csv"),
                    "--output",
                    str(Path(tmp) / "o.json"),
                ]
            )
        self.assertEqual((kind, code), ("return", 1))
        self.assertIn("fixtures: invalid authoring source:", err)

    def test_nonexistent_parent_dir_returns_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = write_csv(tmp)
            kind, code, _out, err = run_cli(
                ["fixtures", "build", str(src), "--output", str(Path(tmp) / "nodir" / "o.json")]
            )
        self.assertEqual((kind, code), ("return", 1))
        self.assertIn("fixtures: invalid authoring source:", err)


class TestRunValidation(unittest.TestCase):
    """All validation failures must exit 2 *before* any model work."""

    def test_missing_required_flags(self):
        base = ["run", "--model", "m", "--dataset", "d", "--family", "chatml"]
        cases = [
            base,  # missing --boundary-ids and --output
            base + ["--boundary-ids", "1"],  # missing --output
            base + ["--output", "o"],  # missing --boundary-ids
            [
                "run",
                "--dataset",
                "d",
                "--family",
                "chatml",
                "--boundary-ids",
                "1",
                "--output",
                "o",
            ],  # missing --model
            [
                "run",
                "--model",
                "m",
                "--family",
                "chatml",
                "--boundary-ids",
                "1",
                "--output",
                "o",
            ],  # missing --dataset
            [
                "run",
                "--model",
                "m",
                "--dataset",
                "d",
                "--boundary-ids",
                "1",
                "--output",
                "o",
            ],  # missing --family
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                kind, code, _out, err = run_cli(argv)
                self.assertEqual((kind, code), ("exit", 2))
                self.assertIn("required", err)

    def test_invalid_choices(self):
        for argv in (run_argv(family="nope"), run_argv(device="tpu")):
            with self.subTest(argv=argv[-2:]):
                kind, code, _out, err = run_cli(argv)
                self.assertEqual((kind, code), ("exit", 2))
                self.assertIn("invalid choice", err)

    def test_numeric_ranges(self):
        cases = [
            ("--max-new-tokens", "0", "Token/context limits are outside supported ranges"),
            ("--max-new-tokens", "4097", "Token/context limits are outside supported ranges"),
            ("--max-new-tokens", "-1", "Token/context limits are outside supported ranges"),
            ("--max-context", "0", "Token/context limits are outside supported ranges"),
            ("--max-context", "32769", "Token/context limits are outside supported ranges"),
            ("--seed", "-1", "Seed must be an unsigned 64-bit integer"),
            ("--seed", str(2**64), "Seed must be an unsigned 64-bit integer"),
            ("--max-drop-bytes", "0", "--max-drop-bytes must be between 1 and 8"),
            ("--max-drop-bytes", "9", "--max-drop-bytes must be between 1 and 8"),
        ]
        for flag, value, message in cases:
            with self.subTest(flag=flag, value=value):
                kind, code, _out, err = run_cli(run_argv() + [flag, value])
                self.assertEqual((kind, code), ("exit", 2))
                # parser.error is invoked on the top-level parser here
                self.assertIn("model-protocol-lab: error", err)
                self.assertIn(message, err)

    def test_numeric_range_edges_accepted_at_parse(self):
        # Parsing itself must accept the edges; the run block then fails
        # only on the missing output/model fixtures, not on validation.
        ns = build_parser().parse_args(run_argv() + ["--max-new-tokens", "4096"])
        self.assertEqual(ns.max_new_tokens, 4096)
        ns = build_parser().parse_args(run_argv() + ["--seed", str(2**64 - 1)])
        self.assertEqual(ns.seed, 2**64 - 1)
        ns = build_parser().parse_args(run_argv() + ["--max-drop-bytes", "8"])
        self.assertEqual(ns.max_drop_bytes, 8)

    def test_non_integer_values_exit_two(self):
        for flag in ("--max-new-tokens", "--max-context", "--seed", "--max-drop-bytes"):
            with self.subTest(flag=flag):
                kind, code, _out, err = run_cli(run_argv() + [flag, "abc"])
                self.assertEqual((kind, code), ("exit", 2))
                self.assertIn("invalid int value", err)

    def test_bad_boundary_ids_exit_two(self):
        kind, code, _out, err = run_cli(run_argv()[:-2] + ["--boundary-ids", "abc"])
        self.assertEqual((kind, code), ("exit", 2))
        self.assertIn("Use 1-16 comma-separated ASCII token IDs", err)

    def test_existing_output_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            kind, code, _out, err = run_cli(run_argv()[:-2] + ["--output", tmp])
            self.assertEqual((kind, code), ("exit", 2))
            self.assertIn("Output path already exists", err)

    def test_unknown_boundary_map_name_exits_two(self):
        kind, code, _out, err = run_cli(run_argv() + ["--boundary-map", "nope=1"])
        self.assertEqual((kind, code), ("exit", 2))
        self.assertIn("unknown candidate", err)

    def test_duplicate_boundary_map_exits_two(self):
        kind, code, _out, err = run_cli(
            run_argv() + ["--boundary-map", "chatml=1", "--boundary-map", "chatml=2"]
        )
        self.assertEqual((kind, code), ("exit", 2))
        self.assertIn("Duplicate --boundary-map", err)

    def test_malformed_boundary_map_exits_two(self):
        kind, code, _out, err = run_cli(run_argv() + ["--boundary-map", "chatml"])
        self.assertEqual((kind, code), ("exit", 2))


class _FakeBackend:
    """Stand-in for LocalHFBackend: records constructor args, no weights."""

    def __init__(self, model, device=None, max_context=None):
        self.model = model
        self.device = device
        self.max_context = max_context

    eos_token_id = None


class TestRunDispatch(unittest.TestCase):
    def _run_with(self, result, argv_extra=()):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = write_fixture_file(tmp)
            outdir = Path(tmp) / "report"
            result_doc = dict(result)
            patches = contextlib.ExitStack()
            patches.enter_context(mock.patch.object(backends, "LocalHFBackend", _FakeBackend))
            patches.enter_context(mock.patch.object(cli, "run_experiment", return_value=result_doc))
            patches.enter_context(
                mock.patch.object(
                    cli,
                    "save_report",
                    return_value=(str(outdir / "results.json"), str(outdir / "report.md")),
                )
            )
            patches.enter_context(
                mock.patch.object(doctor_mod, "template_conformance", return_value={"ok": True})
            )
            patches.enter_context(
                mock.patch.object(experiments, "budget_frontier", return_value={"available": False})
            )

            def fake_prepare(path):
                Path(path).mkdir(parents=True, exist_ok=False)
                return Path(path)

            patches.enter_context(
                mock.patch.object(reporting_mod, "prepare_report_dir", fake_prepare)
            )
            with patches:
                argv = [
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
                    *argv_extra,
                ]
                outcome = run_cli(argv)
            return outcome, result_doc

    def test_run_dispatches_and_returns_zero(self):
        (kind, code, out, _err), _ = self._run_with(
            {
                "decision": "compatible_on_fixture_set",
                "selected_candidate": None,
                "search": [],
                "held_out": {},
            }
        )
        self.assertEqual((kind, code), ("return", 0))
        payload = json.loads(out)
        self.assertEqual(set(payload), {"decision", "selected_candidate", "results", "report"})
        self.assertEqual(payload["decision"], "compatible_on_fixture_set")

    def test_decision_to_exit_code(self):
        mapping = {
            "compatible_on_fixture_set": 0,
            "failed": 1,
            "held_out_failed": 2,
            "ambiguous": 2,
            "no_candidate": 2,
            "cancelled": 130,
        }
        for decision, expected in mapping.items():
            with self.subTest(decision=decision):
                (kind, code, _out, _err), _ = self._run_with(
                    {"decision": decision, "selected_candidate": None, "search": [], "held_out": {}}
                )
                self.assertEqual((kind, code), ("return", expected))

    def test_parsed_args_reach_handlers(self):
        seen = {}

        def fake_experiment(backend, formats, split, **kwargs):
            seen.update(backend=backend, formats=formats, split=split, kwargs=kwargs)
            return {
                "decision": "compatible_on_fixture_set",
                "selected_candidate": None,
                "search": [],
                "held_out": {},
            }

        with tempfile.TemporaryDirectory() as tmp:
            fixture = write_fixture_file(tmp)
            outdir = Path(tmp) / "report"
            with contextlib.ExitStack() as patches:
                patches.enter_context(mock.patch.object(backends, "LocalHFBackend", _FakeBackend))
                patches.enter_context(mock.patch.object(cli, "run_experiment", fake_experiment))
                patches.enter_context(
                    mock.patch.object(
                        cli,
                        "save_report",
                        return_value=("r", "p"),
                    )
                )
                patches.enter_context(
                    mock.patch.object(doctor_mod, "template_conformance", return_value={})
                )
                patches.enter_context(
                    mock.patch.object(
                        experiments, "budget_frontier", return_value={"available": False}
                    )
                )
                patches.enter_context(
                    mock.patch.object(
                        reporting_mod,
                        "prepare_report_dir",
                        lambda p: Path(p).mkdir(parents=True) or Path(p),
                    )
                )
                kind, code, _out, _err = run_cli(
                    [
                        "run",
                        "--model",
                        "ckpt-dir",
                        "--dataset",
                        str(fixture),
                        "--family",
                        "chatml",
                        "--mutate",
                        "--boundary-ids",
                        "198",
                        "--boundary-map",
                        "chatml=5",
                        "--max-new-tokens",
                        "32",
                        "--seed",
                        "7",
                        "--device",
                        "cpu",
                        "--output",
                        str(outdir),
                    ]
                )
        self.assertEqual((kind, code), ("return", 0))
        self.assertIsInstance(seen["backend"], _FakeBackend)
        self.assertEqual(seen["backend"].model, Path("ckpt-dir"))
        self.assertEqual(seen["kwargs"]["boundary"], (198,))
        self.assertEqual(seen["kwargs"]["boundary_map"], {"chatml": (5,)})
        self.assertEqual(seen["kwargs"]["max_new_tokens"], 32)
        self.assertEqual(seen["kwargs"]["seed"], 7)
        # --mutate adds the mutated candidates on top of the base family
        names = {fmt.name for fmt in seen["formats"]}
        self.assertIn("chatml", names)
        self.assertTrue(any("/" in name for name in names))

    def test_empty_expected_warning_on_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "empty.json"
            fixture.write_text(
                json.dumps(
                    {
                        "search": [
                            {
                                "id": "s1",
                                "group": "g1",
                                "messages": [{"role": "user", "content": "hi"}],
                                "expected": "",
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
                )
            )
            outdir = Path(tmp) / "report"
            with contextlib.ExitStack() as patches:
                patches.enter_context(mock.patch.object(backends, "LocalHFBackend", _FakeBackend))
                patches.enter_context(
                    mock.patch.object(
                        cli,
                        "run_experiment",
                        return_value={
                            "decision": "compatible_on_fixture_set",
                            "selected_candidate": None,
                            "search": [],
                            "held_out": {},
                        },
                    )
                )
                patches.enter_context(
                    mock.patch.object(cli, "save_report", return_value=("r", "p"))
                )
                patches.enter_context(
                    mock.patch.object(doctor_mod, "template_conformance", return_value={})
                )
                patches.enter_context(
                    mock.patch.object(
                        experiments, "budget_frontier", return_value={"available": False}
                    )
                )
                patches.enter_context(
                    mock.patch.object(
                        reporting_mod,
                        "prepare_report_dir",
                        lambda p: Path(p).mkdir(parents=True) or Path(p),
                    )
                )
                kind, code, _out, err = run_cli(
                    [
                        "run",
                        "--model",
                        "ckpt-dir",
                        "--dataset",
                        str(fixture),
                        "--family",
                        "chatml",
                        "--boundary-ids",
                        "198",
                        "--output",
                        str(outdir),
                    ]
                )
        self.assertEqual((kind, code), ("return", 0))
        self.assertIn("run: warning: empty expected answer", err)


class TestHostileArgv(unittest.TestCase):
    def test_unicode_digit_boundary_ids_rejected(self):
        for value in ["１２３", "١٢٣", "²"]:
            with self.subTest(value=value):
                kind, code, _out, err = run_cli(["doctor", "--model", "m", "--boundary-ids", value])
                self.assertEqual((kind, code), ("exit", 2))
                self.assertIn("ASCII token IDs", err)

    def test_seventeen_ids_rejected(self):
        kind, code, _out, _err = run_cli(
            ["doctor", "--model", "m", "--boundary-ids", ",".join(["1"] * 17)]
        )
        self.assertEqual((kind, code), ("exit", 2))

    def test_dash_dash_terminator(self):
        kind, code, _out, _err = run_cli(["--"])
        self.assertEqual((kind, code), ("exit", 2))

    def test_option_as_positional_value(self):
        kind, code, _out, _err = run_cli(["inspect", "--dataset"])
        self.assertEqual((kind, code), ("exit", 2))

    def test_repeated_boundary_map_doctor(self):
        # doctor has no candidate set: duplicates still fail, distinct
        # names accumulate.
        argv = ["doctor", "--model", "m"]
        for i in range(20):
            argv += ["--boundary-map", f"name{i}={i}"]
        with mock.patch.object(doctor_mod, "doctor", return_value={"verdict": "ok"}) as fake:
            kind, code, _out, _err = run_cli(argv)
        self.assertEqual((kind, code), ("return", 0))
        self.assertEqual(
            fake.call_args.kwargs["boundary_map"],
            {f"name{i}": (i,) for i in range(20)},
        )

    def test_very_long_token_id_rejected(self):
        kind, code, _out, _err = run_cli(["doctor", "--model", "m", "--boundary-ids", "9" * 5000])
        self.assertEqual((kind, code), ("exit", 2))

    def test_boundary_map_name_with_equals_inside_ids(self):
        kind, code, _out, _err = run_cli(["doctor", "--model", "m", "--boundary-map", "x=1=2"])
        self.assertEqual((kind, code), ("exit", 2))


class TestDeterminism(unittest.TestCase):
    def test_catalog_byte_identical(self):
        self.assertEqual(run_cli(["catalog"])[2], run_cli(["catalog"])[2])

    def test_repeated_parse_identical(self):
        argv = run_argv()
        self.assertEqual(build_parser().parse_args(argv), build_parser().parse_args(argv))


class TestMainModule(unittest.TestCase):
    @staticmethod
    def _pkg_root():
        return Path(model_protocol_lab.__file__).resolve().parent.parent

    def _run_module(self, argv):
        root = self._pkg_root()
        env = dict(os.environ, PYTHONPATH=str(root))
        return subprocess.run(
            [sys.executable, "-m", "model_protocol_lab", *argv],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_module_help(self):
        proc = self._run_module(["--help"])
        self.assertEqual(proc.returncode, 0)
        self.assertIn("model-protocol-lab", proc.stdout)

    def test_module_catalog(self):
        proc = self._run_module(["catalog"])
        self.assertEqual(proc.returncode, 0)
        payload = json.loads(proc.stdout)
        self.assertEqual(len(payload), len(CATALOG))

    def test_module_unknown_command_exits_two(self):
        proc = self._run_module(["frobnicate"])
        self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
