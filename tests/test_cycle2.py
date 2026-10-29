"""Tests for Cycle 2 additions.

A. Per-candidate response boundaries (``--boundary-map``): each format
   hypothesis is measured with its own boundary sequence, with
   ``--boundary-ids`` as the default. The conservative selection rule is
   unchanged: exactly one candidate passing every search fixture under its
   own boundary is selected; ties and failures abstain.
B. ``fixtures build``: compact CSV authoring for the strict fixture schema,
   with the search/held-out discipline enforced at build time (row-numbered
   errors) and the emitted JSON verified through ``parse_dataset`` by
   construction.
C. Richer report views: per-fixture outcome grid, decisive fixtures, and
   per-candidate token-budget sparklines -- all scriptless, zero-asset,
   escaped.
"""

import argparse
import importlib.util
import io
import json
import os
import stat
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from model_protocol_lab.cli import (
    _resolve_boundary_map,
    build_parser,
    main,
    parse_boundary_map,
)
from model_protocol_lab.core import Format, Generation, Message, Split
from model_protocol_lab.doctor import _next_command, examine_model
from model_protocol_lab.experiments import (
    _resolve_boundaries,
    budget_frontier,
    prompt_sensitivity_sweep,
    run_experiment,
)
from model_protocol_lab.fixtures import (
    parse_fixture_csv,
    read_authoring_file,
    split_to_json_text,
    write_fixture_file,
)
from model_protocol_lab.reporting import (
    _decisive_section,
    _frontier_spark_table,
    _outcome_grid_section,
    _sparkline,
    render_html,
)
from tests.test_core import FakeBackend, case, ids
from tests.test_doctor import write_model_dir

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("transformers", "tokenizers", "torch")
)

FMT_A = Format("cand-a", "", "<u>", "<a>", "<e>")
FMT_B = Format("cand-b", "", "<u>", "<a>", "<e>")


def _split():
    return Split((case("s1"), case("s2")), (case("h1"),))


def _gen(text, cap=8):
    # A "limit" generation must carry exactly max_new_tokens tokens; pad
    # after the boundary so the pass analysis is unaffected.
    return Generation(ids(text) + ids("x") * (cap - len(text)), "limit")


def _run_map(backend, boundary_map, formats=(FMT_A, FMT_B)):
    return run_experiment(
        backend,
        formats,
        _split(),
        boundary=ids("|"),
        boundary_map=boundary_map,
        max_new_tokens=8,
        seed=0,
    )


# ---------------------------------------------------------------------------
# A. Per-candidate boundaries


class ResolveBoundariesTests(unittest.TestCase):
    def test_none_map_applies_default_to_all(self):
        resolved = _resolve_boundaries((FMT_A, FMT_B), ids("|"), None)
        self.assertEqual(resolved, {"cand-a": ids("|"), "cand-b": ids("|")})

    def test_override_applies_per_candidate(self):
        resolved = _resolve_boundaries((FMT_A, FMT_B), ids("|"), {"cand-b": ids("#")})
        self.assertEqual(resolved["cand-a"], ids("|"))
        self.assertEqual(resolved["cand-b"], ids("#"))

    def test_unknown_candidate_rejected(self):
        with self.assertRaises(ValueError):
            _resolve_boundaries((FMT_A,), ids("|"), {"nope": ids("#")})

    def test_non_dict_map_rejected(self):
        with self.assertRaises(ValueError):
            _resolve_boundaries((FMT_A,), ids("|"), [("cand-a", ids("#"))])  # type: ignore[arg-type]

    def test_bad_boundary_values_rejected(self):
        for bad in ((), (1, -2), "12", (True,)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _resolve_boundaries((FMT_A,), ids("|"), {"cand-a": bad})


class RunExperimentBoundaryMapTests(unittest.TestCase):
    def test_each_candidate_measured_with_own_boundary(self):
        # cand-a emits "ok|" (passes with "|"), cand-b emits "ok#" (passes
        # with "#" under the map, fails without it).
        backend = FakeBackend(
            generate_script=[_gen("ok|"), _gen("ok|"), _gen("ok#"), _gen("ok#"), _gen("ok|")]
        )
        result = _run_map(backend, {"cand-b": ids("#")})
        self.assertEqual(result["decision"], "ambiguous")
        by_name = {entry["format"]["name"]: entry for entry in result["search"]}
        self.assertEqual(by_name["cand-a"]["boundary_token_ids"], list(ids("|")))
        self.assertEqual(by_name["cand-b"]["boundary_token_ids"], list(ids("#")))

    def test_without_map_second_candidate_fails(self):
        backend = FakeBackend(
            generate_script=[_gen("ok|"), _gen("ok|"), _gen("ok#"), _gen("ok#"), _gen("ok|")]
        )
        result = _run_map(backend, None)
        # cand-b's "ok#" has no "|" boundary: cand-a is the unique winner.
        self.assertEqual(result["decision"], "compatible_on_fixture_set")
        self.assertEqual(result["selected_candidate"], "cand-a")
        self.assertEqual(result["selected_boundary_token_ids"], list(ids("|")))

    def test_held_out_uses_winner_boundary(self):
        backend = FakeBackend(
            generate_script=[_gen("ok#"), _gen("ok#"), _gen("ok~"), _gen("ok~"), _gen("ok#")]
        )
        result = _run_map(backend, {"cand-a": ids("#")})
        self.assertEqual(result["selected_candidate"], "cand-a")
        self.assertEqual(result["selected_boundary_token_ids"], list(ids("#")))
        self.assertEqual(result["decision"], "compatible_on_fixture_set")

    def test_entries_record_boundaries_without_map(self):
        backend = FakeBackend(generate_script=[_gen("ok|")] * 5)
        result = _run_map(backend, None)
        for entry in result["search"]:
            self.assertEqual(entry["boundary_token_ids"], list(ids("|")))


class BudgetFrontierBoundaryTests(unittest.TestCase):
    @staticmethod
    def _entry(name, output_text, boundary_ids):
        obs = {
            "fixture": "s1",
            "status": "completed",
            "exact_match": True,
            "boundary_found": True,
            "output_ids": list(ids(output_text)),
        }
        return {
            "format": {"name": name},
            "boundary_token_ids": list(boundary_ids),
            "observations": [obs],
        }

    def test_entry_boundary_overrides_argument(self):
        search = [self._entry("a", "ok|", ids("|"))]
        frontier = budget_frontier(
            search,
            boundary=(999,),
            max_new_tokens=8,
            native_decision="no_candidate",
            native_winner=None,
        )
        # With the entry's own "|" boundary the minimal budget is 3; the
        # (999,) argument boundary would have found no boundary at all.
        self.assertEqual(frontier["candidates"]["a"]["min_budget_per_fixture"], [3])

    def test_missing_entry_boundary_falls_back_to_argument(self):
        obs = {
            "fixture": "s1",
            "status": "completed",
            "exact_match": True,
            "boundary_found": True,
            "output_ids": list(ids("ok|")),
        }
        search = [{"format": {"name": "a"}, "observations": [obs]}]
        frontier = budget_frontier(
            search,
            boundary=ids("|"),
            max_new_tokens=8,
            native_decision="no_candidate",
            native_winner=None,
        )
        self.assertEqual(frontier["candidates"]["a"]["min_budget_per_fixture"], [3])

    def test_malformed_entry_boundary_rejected(self):
        search = [self._entry("a", "ok|", ())]
        with self.assertRaises(ValueError):
            budget_frontier(
                search,
                boundary=ids("|"),
                max_new_tokens=8,
                native_decision="no_candidate",
                native_winner=None,
            )

    def test_tuple_entry_boundary_accepted(self):
        search = [self._entry("a", "ok|", ids("|"))]
        search[0]["boundary_token_ids"] = tuple(ids("|"))
        frontier = budget_frontier(
            search,
            boundary=(999,),
            max_new_tokens=8,
            native_decision="no_candidate",
            native_winner=None,
        )
        self.assertEqual(frontier["candidates"]["a"]["min_budget_per_fixture"], [3])

    def test_observations_without_usable_output_ids(self):
        search = [
            {
                "format": {"name": "a"},
                "observations": [
                    {
                        "fixture": "s1",
                        "status": "completed",
                        "exact_match": True,
                        "boundary_found": True,
                        # no output_ids at all
                    },
                    {
                        "fixture": "s2",
                        "status": "completed",
                        "exact_match": True,
                        "boundary_found": True,
                        "output_ids": [1, "two", 3],  # mistyped token
                    },
                    {
                        "fixture": "s3",
                        "status": "completed",
                        "exact_match": True,
                        "boundary_found": True,
                        "output_ids": list(ids("ok~")),  # boundary absent
                    },
                ],
            }
        ]
        frontier = budget_frontier(
            search,
            boundary=ids("|"),
            max_new_tokens=8,
            native_decision="no_candidate",
            native_winner=None,
        )
        self.assertEqual(frontier["candidates"]["a"]["min_budget_per_fixture"], [None, None, None])


class SensitivitySweepBoundaryMapTests(unittest.TestCase):
    def test_map_threads_through_sweep(self):
        backend = FakeBackend(generate_script=[_gen("ok|")] * 40)
        formats = (replace(FMT_A, name="a"),)
        result = prompt_sensitivity_sweep(
            backend,
            formats,
            Split((case("s1"),), (case("h1"),)),
            boundary=ids("|"),
            boundary_map={"a": ids("|")},
            max_drop_bytes=2,
            max_new_tokens=8,
            seed=0,
        )
        self.assertEqual(result["native_decision"], "compatible_on_fixture_set")
        self.assertEqual(len(result["depths"]), 2)


class ParseBoundaryMapTests(unittest.TestCase):
    def test_valid_forms(self):
        self.assertEqual(parse_boundary_map("chatml=1,2"), ("chatml", (1, 2)))
        self.assertEqual(parse_boundary_map("chatml/toggle-bos=5"), ("chatml/toggle-bos", (5,)))

    def test_invalid_forms_rejected(self):
        for bad in ("chatml", "=1,2", "chatml=", "chatml=1,,2", "chatml=x"):
            with self.subTest(bad=bad), self.assertRaises(argparse.ArgumentTypeError):
                parse_boundary_map(bad)


class ResolveBoundaryMapTests(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_none_returns_none(self):
        self.assertIsNone(_resolve_boundary_map(None, {"a"}, self.parser))

    def test_valid_names_resolve(self):
        resolved = _resolve_boundary_map([("a", (1,)), ("b", (2, 3))], {"a", "b"}, self.parser)
        self.assertEqual(resolved, {"a": (1,), "b": (2, 3)})

    def test_duplicate_name_exits_two(self):
        with self.assertRaises(SystemExit) as ctx:
            _resolve_boundary_map([("a", (1,)), ("a", (2,))], {"a"}, self.parser)
        self.assertEqual(ctx.exception.code, 2)

    def test_unknown_name_exits_two(self):
        with self.assertRaises(SystemExit) as ctx:
            _resolve_boundary_map([("zzz", (1,))], {"a"}, self.parser)
        self.assertEqual(ctx.exception.code, 2)

    def test_unknown_names_allowed_without_candidate_set(self):
        # doctor does not know the candidate set: duplicates are still
        # rejected, unknown names pass through to per-name vocab checks.
        resolved = _resolve_boundary_map([("zzz", (1,))], None, self.parser)
        self.assertEqual(resolved, {"zzz": (1,)})
        with self.assertRaises(SystemExit):
            _resolve_boundary_map([("zzz", (1,)), ("zzz", (2,))], None, self.parser)


class RunBoundaryMapCliTests(unittest.TestCase):
    @staticmethod
    def _run_args(tmp, extra):
        return [
            "run",
            "--model",
            str(Path(tmp) / "missing"),
            "--dataset",
            "examples/fixtures.json",
            "--family",
            "chatml",
            "--boundary-ids",
            "62",
            *extra,
            "--output",
            str(Path(tmp) / "out"),
        ]

    def test_duplicate_map_flag_exits_two(self):
        with (
            TemporaryDirectory() as tmp,
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as ctx,
        ):
            main(
                self._run_args(
                    tmp,
                    [
                        "--boundary-map",
                        "chatml=62",
                        "--boundary-map",
                        "chatml=63",
                    ],
                )
            )
        self.assertEqual(ctx.exception.code, 2)

    def test_unknown_map_name_exits_two(self):
        with (
            TemporaryDirectory() as tmp,
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as ctx,
        ):
            main(self._run_args(tmp, ["--boundary-map", "not-a-candidate=62"]))
        self.assertEqual(ctx.exception.code, 2)

    def test_unknown_map_name_lists_valid_names(self):
        with (
            TemporaryDirectory() as tmp,
            redirect_stderr(io.StringIO()) as err,
            self.assertRaises(SystemExit),
        ):
            main(self._run_args(tmp, ["--mutate", "--boundary-map", "not-a-candidate=62"]))
        self.assertIn("chatml/prefix-newline", err.getvalue())


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class RunBoundaryMapEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_adapter_integration import build_tiny_gpt2

        cls._tmp = TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.model_dir = Path(cls._tmp.name) / "tiny"
        cls.model_dir.mkdir()
        build_tiny_gpt2(cls.model_dir)
        cls.dataset = Path(__file__).resolve().parent.parent / "examples" / "fixtures.json"

    def test_run_with_boundary_map_records_per_candidate_boundaries(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        output = Path(tmp.name) / "report"
        with redirect_stdout(io.StringIO()):
            code = main(
                [
                    "run",
                    "--model",
                    str(self.model_dir),
                    "--dataset",
                    str(self.dataset),
                    "--family",
                    "chatml",
                    "--mutate",
                    "--boundary-ids",
                    "62",
                    "--boundary-map",
                    "chatml/prefix-newline=63",
                    "--max-new-tokens",
                    "8",
                    "--max-context",
                    "64",
                    "--device",
                    "cpu",
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(code, 2)  # random weights: the lab abstains honestly
        results = json.loads((output / "results.json").read_text(encoding="utf-8"))
        by_name = {entry["format"]["name"]: entry for entry in results["search"]}
        self.assertEqual(by_name["chatml/prefix-newline"]["boundary_token_ids"], [63])
        self.assertEqual(by_name["chatml"]["boundary_token_ids"], [62])
        html = (output / "report.html").read_text(encoding="utf-8")
        self.assertIn("Boundary token IDs (chatml/prefix-newline)", html)


class DoctorBoundaryMapTests(unittest.TestCase):
    @staticmethod
    def _model(root):
        root.mkdir(parents=True, exist_ok=True)
        return write_model_dir(root)

    def test_map_checks_degrade_to_skipped_without_transformers(self):
        # Without the hf extra the vocabulary is unknown: per-name checks
        # degrade to "skipped", never to a raw failure. Simulate the missing
        # extra by blocking the transformers import the deep check uses.
        import sys
        from unittest.mock import patch

        with TemporaryDirectory() as tmp:
            root = self._model(Path(tmp) / "model")
            with patch.dict(sys.modules, {"transformers": None}):
                report = examine_model(root, boundary_map={"a": (0,), "b": (1,)})
        checks = report["model"]["checks"]
        self.assertEqual(checks["tokenizer_deep"]["status"], "skipped")
        self.assertEqual(checks["boundary_ids:a"]["status"], "skipped")
        self.assertEqual(checks["boundary_ids:b"]["status"], "skipped")

    def test_invalid_map_shape_fails_closed(self):
        with TemporaryDirectory() as tmp:
            root = self._model(Path(tmp) / "model")
            report = examine_model(root, boundary_map=[("a", (0,))])  # type: ignore[arg-type]
        self.assertEqual(report["model"]["checks"]["boundary_map"]["status"], "failed")

    @unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
    def test_map_validated_against_vocabulary(self):
        with TemporaryDirectory() as tmp:
            root = self._model(Path(tmp) / "model")
            report = examine_model(root, boundary_map={"a": (0,), "b": (10**9,)})
        checks = report["model"]["checks"]
        self.assertEqual(checks["boundary_ids:a"]["status"], "ok")
        self.assertEqual(checks["boundary_ids:b"]["status"], "failed")
        self.assertEqual(report["verdict"], "failed")

    def test_suggestion_note_mentions_boundary_map(self):
        report = {
            "verdict": "ok",
            "model": {"context_window": 64},
        }
        suggestion = _next_command("m", "d.json", (62,), report)
        self.assertIn("--boundary-map", suggestion["note"])


# ---------------------------------------------------------------------------
# B. fixtures build


VALID_CSV = """\
id,group,partition,expected,user_1,assistant_1,user_2
s1,g1,search,ok,question one,,
s2,g2,search,ok2,question two,answer two,follow up?
h1,g3,held_out,ok,held question,,
"""


class ParseFixtureCsvTests(unittest.TestCase):
    def test_valid_csv_builds_split(self):
        split, empty = parse_fixture_csv(VALID_CSV)
        self.assertEqual([c.id for c in split.search], ["s1", "s2"])
        self.assertEqual([c.id for c in split.held_out], ["h1"])
        self.assertEqual(split.search[1].messages[1], Message("assistant", "answer two"))
        self.assertEqual(split.search[0].expected, "ok")
        self.assertEqual(empty, [])

    def test_quoted_newlines_preserved(self):
        text = (
            "id,group,partition,expected,user_1\n"
            's1,g1,search,ok,"line one\nline two"\n'
            "h1,g2,held_out,ok,q\n"
        )
        split, _ = parse_fixture_csv(text)
        self.assertEqual(split.search[0].messages[0].content, "line one\nline two")

    def test_blank_lines_and_short_rows_tolerated(self):
        text = (
            "\n"
            "id,group,partition,expected,user_1,assistant_1\n"
            "\n"
            "s1,g1,search,ok,q\n"
            "h1,g2,held_out,ok,q2\n"
            "\n"
        )
        split, _ = parse_fixture_csv(text)
        self.assertEqual(len(split.search), 1)

    def test_empty_expected_reported_not_rejected(self):
        text = "id,group,partition,expected,user_1\ns1,g1,search,,q\nh1,g2,held_out,ok,q2\n"
        split, empty = parse_fixture_csv(text)
        self.assertEqual(empty, ["s1"])
        self.assertEqual(split.search[0].expected, "")

    def test_row_numbers_in_errors(self):
        cases = [
            (
                "id,group,partition,expected,user_1\ns1,g1,bogus,ok,q\nh1,g2,held_out,ok,q2\n",
                "row 2",
            ),
            (
                "id,group,partition,expected,user_1\n, g1,search,ok,q\nh1,g2,held_out,ok,q2\n",
                "row 2",
            ),
            (
                "id,group,partition,expected,user_1,assistant_1\n"
                "s1,g1,search,ok,q,a\nh1,g2,held_out,ok,q2,a2\n",
                "row 2",
            ),
            (
                "id,group,partition,expected,user_1,assistant_1,user_2\n"
                "s1,g1,search,ok,q,,q2\nh1,g2,held_out,ok,q3\n",
                "row 2",
            ),
        ]
        for text, marker in cases:
            with self.subTest(text=text), self.assertRaises(ValueError) as ctx:
                parse_fixture_csv(text)
            self.assertIn(marker, str(ctx.exception))

    def test_discipline_violations_refuse_build(self):
        # Group overlap across partitions.
        text = "id,group,partition,expected,user_1\ns1,g1,search,ok,q1\nh1,g1,held_out,ok,q2\n"
        with self.assertRaises(ValueError) as ctx:
            parse_fixture_csv(text)
        self.assertIn("split discipline", str(ctx.exception))
        # Duplicate IDs.
        text = "id,group,partition,expected,user_1\ns1,g1,search,ok,q1\ns1,g2,held_out,ok,q2\n"
        with self.assertRaises(ValueError):
            parse_fixture_csv(text)
        # Identical message sequences across partitions.
        text = "id,group,partition,expected,user_1\ns1,g1,search,ok,same\nh1,g2,held_out,ok,same\n"
        with self.assertRaises(ValueError):
            parse_fixture_csv(text)

    def test_bad_headers_rejected(self):
        for header in (
            "identifier,group,partition,expected,user_1",
            "id,group,partition,expected,assistant_1",
            "id,group,partition,expected,user_1,user_2",
            "id,group,partition,expected",
        ):
            text = header + "\ns1,g1,search,ok,q\nh1,g2,held_out,ok,q2\n"
            with self.subTest(header=header), self.assertRaises(ValueError):
                parse_fixture_csv(text)

    def test_missing_partition_rejected(self):
        text = "id,group,partition,expected,user_1\nh1,g2,held_out,ok,q2\n"
        with self.assertRaises(ValueError) as ctx:
            parse_fixture_csv(text)
        self.assertIn("no 'search' fixtures", str(ctx.exception))

    def test_empty_source_rejected(self):
        with self.assertRaises(ValueError):
            parse_fixture_csv("")
        with self.assertRaises(ValueError):
            parse_fixture_csv("id,group,partition,expected,user_1\n")

    def test_oversize_and_non_text_sources_rejected(self):
        from model_protocol_lab.datasets import MAX_BYTES

        with self.assertRaises(ValueError):
            parse_fixture_csv(123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            parse_fixture_csv("x" * (MAX_BYTES + 1))

    def test_too_many_fixtures_rejected(self):
        header = "id,group,partition,expected,user_1\n"
        rows = "".join(f"s{i},g{i},search,ok,q{i}\n" for i in range(101)) + "".join(
            f"h{i},gg{i},held_out,ok,q{i}\n" for i in range(101)
        )
        with self.assertRaises(ValueError) as ctx:
            parse_fixture_csv(header + rows)
        self.assertIn("more than 200", str(ctx.exception))

    def test_too_many_cells_rejected(self):
        text = "id,group,partition,expected,user_1\ns1,g1,search,ok,q,EXTRA\nh1,g2,held_out,ok,q2\n"
        with self.assertRaises(ValueError) as ctx:
            parse_fixture_csv(text)
        self.assertIn("row 2", str(ctx.exception))

    def test_blank_group_rejected(self):
        text = "id,group,partition,expected,user_1\ns1,,search,ok,q\nh1,g2,held_out,ok,q2\n"
        with self.assertRaises(ValueError) as ctx:
            parse_fixture_csv(text)
        self.assertIn("group is blank", str(ctx.exception))

    def test_oversize_cell_rejected(self):
        from model_protocol_lab.datasets import MAX_TEXT

        text = (
            "id,group,partition,expected,user_1\n"
            f"s1,g1,search,ok,{'y' * (MAX_TEXT + 1)}\n"
            "h1,g2,held_out,ok,q2\n"
        )
        with self.assertRaises(ValueError) as ctx:
            parse_fixture_csv(text)
        self.assertIn("exceeds size limit", str(ctx.exception))

    def test_missing_user_message_rejected(self):
        text = (
            "id,group,partition,expected,user_1,assistant_1\n"
            "s1,g1,search,ok,,\n"
            "h1,g2,held_out,ok,q2,\n"
        )
        with self.assertRaises(ValueError) as ctx:
            parse_fixture_csv(text)
        self.assertIn("user_1 message", str(ctx.exception))

    def test_too_many_per_partition_rejected(self):
        header = "id,group,partition,expected,user_1\n"
        rows = "".join(f"s{i},g{i},search,ok,q{i}\n" for i in range(101))
        rows += "h1,gg1,held_out,ok,q\n"
        with self.assertRaises(ValueError) as ctx:
            parse_fixture_csv(header + rows)
        self.assertIn("'search' fixtures", str(ctx.exception))

    def test_control_chars_stripped_from_error_ids(self):
        text = "id,group,partition,expected,user_1\n\x1b[31mx,g1,bogus,ok,q\nh1,g2,held_out,ok,q2\n"
        with self.assertRaises(ValueError) as ctx:
            parse_fixture_csv(text)
        self.assertNotIn("\x1b", str(ctx.exception))


class SplitToJsonTests(unittest.TestCase):
    def test_emitted_json_passes_strict_loader(self):
        split, _ = parse_fixture_csv(VALID_CSV)
        text = split_to_json_text(split)
        from model_protocol_lab.datasets import parse_dataset

        reparsed = parse_dataset(text)
        self.assertEqual([c.id for c in reparsed.search], [c.id for c in split.search])
        self.assertEqual(text.splitlines()[0], "{")

    def test_rejects_non_split(self):
        with self.assertRaises(ValueError):
            split_to_json_text("nope")  # type: ignore[arg-type]


class ReadAuthoringFileTests(unittest.TestCase):
    def test_reads_regular_file(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.csv"
            path.write_text(VALID_CSV, encoding="utf-8")
            self.assertEqual(read_authoring_file(path), VALID_CSV)

    def test_refuses_directory(self):
        with TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            read_authoring_file(Path(tmp))

    def test_refuses_non_utf8(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.csv"
            path.write_bytes(b"id,\xff\n")
            with self.assertRaises(ValueError):
                read_authoring_file(path)

    def test_refuses_oversize(self):
        from model_protocol_lab.datasets import MAX_BYTES

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.csv"
            path.write_bytes(b"x" * (MAX_BYTES + 1))
            with self.assertRaises(ValueError):
                read_authoring_file(path)

    def test_refuses_non_path(self):
        with self.assertRaises(ValueError):
            read_authoring_file(123)  # type: ignore[arg-type]


class WriteFixtureFileTests(unittest.TestCase):
    def test_writes_with_owner_only_permissions(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "out.json"
            write_fixture_file("{}", target)
            self.assertEqual(target.read_text(encoding="utf-8"), "{}")
            mode = stat.S_IMODE(os.stat(target).st_mode)
            self.assertEqual(mode, 0o600)

    def test_refuses_existing_path(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "out.json"
            target.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                write_fixture_file("{}", target)

    def test_refuses_missing_parent(self):
        with TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            write_fixture_file("{}", Path(tmp) / "nope" / "out.json")

    def test_rejects_bad_argument_types(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "out.json"
            with self.assertRaises(ValueError):
                write_fixture_file(123, target)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                write_fixture_file("{}", 123)  # type: ignore[arg-type]

    def test_unwritable_path_reports_cleanly(self):
        # ENAMETOOLONG from the OS surfaces as the fixed "not writable"
        # vocabulary, never raw OS text.
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / ("x" * 300)
            with self.assertRaises(ValueError) as ctx:
                write_fixture_file("{}", target)
            self.assertIn("not writable", str(ctx.exception))

    def test_partial_write_leaves_no_file(self):
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import os as _os
        from unittest.mock import patch

        with (
            TemporaryDirectory() as tmp,
            patch.object(_os, "fdopen", side_effect=OSError("boom")),
            self.assertRaises(OSError),
        ):
            target = Path(tmp) / "out.json"
            write_fixture_file("{}", target)
        self.assertFalse(target.exists())


class FixturesBuildCliTests(unittest.TestCase):
    def _run_build(self, csv_text, extra=()):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        source = Path(tmp.name) / "cases.csv"
        source.write_text(csv_text, encoding="utf-8")
        output = Path(tmp.name) / "fixtures.json"
        with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
            code = main(["fixtures", "build", str(source), "--output", str(output), *extra])
        return code, output, out.getvalue(), err.getvalue()

    def test_build_writes_valid_fixture_json(self):
        code, output, stdout, _ = self._run_build(VALID_CSV)
        self.assertEqual(code, 0)
        summary = json.loads(stdout)
        self.assertEqual((summary["search_cases"], summary["held_out_cases"]), (2, 1))
        from model_protocol_lab.datasets import load_dataset

        split = load_dataset(output)
        self.assertEqual([c.id for c in split.search], ["s1", "s2"])

    def test_build_warns_on_empty_expected(self):
        text = "id,group,partition,expected,user_1\ns1,g1,search,,q\nh1,g2,held_out,ok,q2\n"
        code, _, _, stderr = self._run_build(text)
        self.assertEqual(code, 0)
        self.assertIn("pass vacuously", stderr)

    def test_build_rejects_bad_csv(self):
        code, output, _, stderr = self._run_build(
            "id,group,partition,expected,user_1\ns1,g1,bogus,ok,q\nh1,g2,held_out,ok,q2\n"
        )
        self.assertEqual(code, 1)
        self.assertIn("row 2", stderr)
        self.assertFalse(output.exists())

    def test_build_rejects_existing_output(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        source = Path(tmp.name) / "cases.csv"
        source.write_text(VALID_CSV, encoding="utf-8")
        output = Path(tmp.name) / "fixtures.json"
        output.write_text("{}", encoding="utf-8")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = main(["fixtures", "build", str(source), "--output", str(output)])
        self.assertEqual(code, 1)
        self.assertEqual(output.read_text(encoding="utf-8"), "{}")

    def test_missing_source_exits_one(self):
        with (
            TemporaryDirectory() as tmp,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            code = main(
                [
                    "fixtures",
                    "build",
                    str(Path(tmp) / "missing.csv"),
                    "--output",
                    str(Path(tmp) / "out.json"),
                ]
            )
        self.assertEqual(code, 1)


# ---------------------------------------------------------------------------
# C. Richer report views


def _result_with_search():
    def obs(fid, passed):
        return {
            "fixture": fid,
            "status": "completed",
            "exact_match": passed,
            "boundary_found": passed,
        }

    return {
        "decision": "compatible_on_fixture_set",
        "selected_candidate": "cand-a",
        "boundary_token_ids": [124],
        "search": [
            {
                "format": {"name": "cand-a"},
                "boundary_token_ids": [124],
                "summary": {"status": "completed"},
                "observations": [obs("s1", True), obs("s2", True)],
            },
            {
                "format": {"name": "cand-b"},
                "boundary_token_ids": [124],
                "summary": {"status": "completed"},
                "observations": [obs("s1", True), obs("s2", False)],
            },
        ],
    }


class OutcomeGridTests(unittest.TestCase):
    def test_glyphs_and_winner_star(self):
        html = _outcome_grid_section(_result_with_search())
        self.assertIn("Per-fixture outcomes", html)
        self.assertIn("\u2713", html)  # pass
        self.assertIn("\u2717", html)  # fail
        self.assertIn("cand-a \u2605", html)  # selected candidate starred
        self.assertIn("s1", html)

    def test_incomplete_observations_render_dash(self):
        result = _result_with_search()
        result["search"][1]["observations"][0]["status"] = "error"
        html = _outcome_grid_section(result)
        self.assertIn("\u2013", html)

    def test_hostile_fixture_ids_escaped(self):
        result = _result_with_search()
        result["search"][0]["observations"][0]["fixture"] = '<script>alert("x")</script>'
        html = _outcome_grid_section(result)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_empty_search_renders_nothing(self):
        self.assertEqual(_outcome_grid_section({"search": []}), "")


class DecisiveFixtureTests(unittest.TestCase):
    def test_winner_decisive_fixture_listed(self):
        html = _decisive_section(_result_with_search())
        self.assertIn("Decisive fixtures", html)
        self.assertIn("cand-a", html)
        self.assertIn("s2", html)  # only cand-a passed s2
        self.assertNotIn("s1", html.split("</ul>")[0].split("<ul>")[-1])

    def test_no_solo_pass_message(self):
        result = _result_with_search()
        for entry in result["search"]:
            for obs_item in entry["observations"]:
                obs_item["exact_match"] = True
                obs_item["boundary_found"] = True
        html = _decisive_section(result)
        self.assertIn("No fixture was passed by exactly one candidate", html)

    def test_incomplete_observation_disqualifies(self):
        result = _result_with_search()
        result["search"][1]["observations"][1]["status"] = "error"
        html = _decisive_section(result)
        self.assertIn("No fixture was passed by exactly one candidate", html)


class SparklineTests(unittest.TestCase):
    def test_scaling(self):
        self.assertEqual(_sparkline([0, 2, 4], 4), "\u2581\u2585\u2588")
        self.assertEqual(_sparkline([4, 4], 4), "\u2588\u2588")

    def test_degenerate_inputs(self):
        self.assertEqual(_sparkline([], 4), "?")
        self.assertEqual(_sparkline([1], 0), "?")

    def test_frontier_table_renders_all_candidates(self):
        frontier = {
            "max_new_tokens": 8,
            "candidates": {
                "a": {
                    "fixture_ids": ["s1", "s2"],
                    "min_budget_per_fixture": [3, 5],
                    "full_pass_budget": 5,
                },
                "b": {
                    "fixture_ids": ["s1", "s2"],
                    "min_budget_per_fixture": [None, None],
                    "full_pass_budget": None,
                },
            },
        }
        html = _frontier_spark_table(frontier)
        self.assertIn("Pass curves by candidate", html)
        self.assertIn("a", html)
        self.assertIn("b", html)
        self.assertIn('aria-label="pass count', html)
        self.assertIn("\u2014", html)  # no full-pass budget for b

    def test_frontier_table_handles_bad_input(self):
        self.assertEqual(_frontier_spark_table({}), "")
        self.assertEqual(_frontier_spark_table({"candidates": {}}), "")
        self.assertEqual(
            _frontier_spark_table({"candidates": {"a": {}}, "max_new_tokens": "8"}), ""
        )
        self.assertEqual(
            _frontier_spark_table({"candidates": {1: {}}, "max_new_tokens": 8}),
            "",
        )

    def test_frontier_table_samples_large_budgets(self):
        frontier = {
            "max_new_tokens": 4096,
            "candidates": {
                "a": {
                    "fixture_ids": ["s1"],
                    "min_budget_per_fixture": [4000],
                    "full_pass_budget": 4000,
                },
            },
        }
        html = _frontier_spark_table(frontier)
        self.assertIn("over budgets 1 to 4096", html)

    def test_outcome_grid_tolerates_malformed_runs(self):
        result = {
            "search": [
                "not-a-dict",
                {"format": {"name": ""}, "observations": []},
                {"format": {}, "observations": [{"fixture": 1}]},
            ]
        }
        self.assertEqual(_outcome_grid_section(result), "")
        self.assertEqual(_decisive_section(result), "")


class ReportViewsIntegrationTests(unittest.TestCase):
    def test_render_html_includes_new_sections(self):
        result = _result_with_search()
        result["budget_frontier"] = {
            "max_new_tokens": 8,
            "budget_robust": True,
            "native_winner": "cand-a",
            "candidates": {
                "cand-a": {
                    "fixture_ids": ["s1", "s2"],
                    "min_budget_per_fixture": [3, 5],
                    "pass_steps": [[1, 0], [3, 1], [5, 2]],
                    "full_pass_budget": 5,
                },
                "cand-b": {
                    "fixture_ids": ["s1", "s2"],
                    "min_budget_per_fixture": [3, None],
                    "pass_steps": [[1, 0], [3, 1]],
                    "full_pass_budget": None,
                },
            },
        }
        html = render_html(result)
        self.assertIn("Per-fixture outcomes", html)
        self.assertIn("Decisive fixtures", html)
        self.assertIn("Pass curves by candidate", html)
        self.assertIn("Boundary IDs", html)  # search-table column
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)

    def test_config_section_lists_per_candidate_boundaries(self):
        result = _result_with_search()
        result["search"][1]["boundary_token_ids"] = [125]
        html = render_html(result)
        self.assertIn("Boundary token IDs (cand-a)", html)
        self.assertIn("Boundary token IDs (cand-b)", html)

    def test_config_section_single_line_when_uniform(self):
        html = render_html(_result_with_search())
        self.assertIn("Boundary token IDs", html)
        self.assertNotIn("Boundary token IDs (cand-a)", html)


if __name__ == "__main__":
    unittest.main()
