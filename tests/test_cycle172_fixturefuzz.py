"""Cycle 172 -- ``fixtures fuzz``: the fixture-loader contract fuzzer.

Pins, in order: the shipped corpus conforms end-to-end (controls
first); generation and reporting are byte-deterministic per seed;
case constructors fail closed; the outcome classifier distinguishes
crash / unexpected accept / unexpected reject / warning mismatch /
message mismatch / stored mismatch with fake loaders (the message-pin
tests are what keep the surrogate cases non-vacuous -- a
UnicodeEncodeError is a ValueError subclass, so outcome-only checks
cannot pin the cycle-172 loader fix); the three loader fixes
themselves (lone surrogates, CSV id/group length, csv.Error mapping)
with direct regression tests; the CLI surface; and tempdir mutation
gates over the classifier, the pin check, the report verdict, and the
corpus digest.
"""

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import model_protocol_lab.fixturefuzz as ff
from model_protocol_lab.datasets import parse_dataset
from model_protocol_lab.fixtures import parse_fixture_csv, split_to_json_text

_BASE_TEXT = ff._json_text(ff._json_base())
_BASE_SPLIT = parse_dataset(_BASE_TEXT)


def _case(**kw):
    kw.setdefault("name", "unit_case")
    kw.setdefault("target", "json")
    kw.setdefault("payload", _BASE_TEXT)
    kw.setdefault("expected", "accept")
    kw.setdefault("clause", "unit clause")
    return ff.FuzzCase(**kw)


def _eval(case, **loaders):
    with tempfile.TemporaryDirectory(prefix="c172_eval_") as tmp:
        return ff.evaluate_case(case, Path(tmp), **loaders)


class CorpusConformanceTests(unittest.TestCase):
    def test_full_corpus_conforms_seed_zero(self):
        report = ff.run_fuzz(0)
        self.assertEqual(report.total, 133)
        failures = [r for r in report.results if not r.conform]
        self.assertEqual(failures, [], failures[:3])
        self.assertTrue(report.ok)

    def test_full_corpus_conforms_other_seeds(self):
        for seed in (1, 7, 12345):
            with self.subTest(seed=seed):
                self.assertTrue(ff.run_fuzz(seed).ok)

    def test_controls_present_and_conforming(self):
        report = ff.run_fuzz(0)
        by_name = {r.name: r for r in report.results}
        for name in ("json_control", "csv_control", "jfile_control", "cfile_control"):
            self.assertIn(name, by_name)
            self.assertTrue(by_name[name].conform, name)

    def test_corpus_shape_and_table_invariants(self):
        cases = ff.generate_cases(0)
        self.assertEqual(len(cases), 133)
        self.assertEqual(len({c.name for c in cases}), 133)
        by_target = {}
        for case in cases:
            by_target[case.target] = by_target.get(case.target, 0) + 1
            self.assertTrue(case.clause, case.name)
            if case.message_pin is not None:
                self.assertEqual(case.expected, "reject", case.name)
            if case.expected == "accept_warn":
                self.assertEqual(case.target, "csv", case.name)
                self.assertIsNotNone(case.warn_id, case.name)
        self.assertEqual(by_target, {"json": 74, "csv": 44, "json_file": 9, "csv_file": 6})


class DeterminismTests(unittest.TestCase):
    def test_same_seed_identical_report(self):
        first = json.dumps(ff.run_fuzz(3).to_dict(), sort_keys=True)
        second = json.dumps(ff.run_fuzz(3).to_dict(), sort_keys=True)
        self.assertEqual(first, second)

    def test_digest_changes_with_seed(self):
        self.assertNotEqual(
            ff.corpus_digest(ff.generate_cases(0)),
            ff.corpus_digest(ff.generate_cases(1)),
        )

    def test_derive_stable_bounded_and_case_sensitive(self):
        self.assertEqual(ff._derive(5, "case", "p", 97), ff._derive(5, "case", "p", 97))
        self.assertTrue(0 <= ff._derive(5, "case", "p", 97) < 97)
        self.assertNotEqual(ff._derive(5, "case", "p", 10**6), ff._derive(5, "other", "p", 10**6))

    def test_digest_covers_payload_and_pin(self):
        one = ff.FuzzCase("n", "json", "{}", "reject", "c")
        two = ff.FuzzCase("n", "json", "{ }", "reject", "c")
        three = ff.FuzzCase("n", "json", "{}", "reject", "c", message_pin="x")
        self.assertNotEqual(ff.corpus_digest((one,)), ff.corpus_digest((two,)))
        self.assertNotEqual(ff.corpus_digest((one,)), ff.corpus_digest((three,)))

    def test_report_has_no_paths_or_timestamps(self):
        blob = json.dumps(ff.run_fuzz(0).to_dict())
        self.assertNotIn(tempfile.gettempdir(), blob)
        self.assertNotIn("mpl-fixture-fuzz", blob)


class ConstructorValidationTests(unittest.TestCase):
    def test_rejects_bad_target(self):
        with self.assertRaises(ValueError):
            ff.FuzzCase("n", "yaml", "x", "accept", "c")

    def test_rejects_bad_outcome(self):
        with self.assertRaises(ValueError):
            ff.FuzzCase("n", "json", "x", "maybe", "c")

    def test_rejects_bytes_for_text_target(self):
        with self.assertRaises(ValueError):
            ff.FuzzCase("n", "json", b"x", "accept", "c")

    def test_rejects_str_for_file_target(self):
        with self.assertRaises(ValueError):
            ff.FuzzCase("n", "json_file", "x", "accept", "c")

    def test_rejects_bad_file_kind(self):
        with self.assertRaises(ValueError):
            ff.FuzzCase("n", "json_file", b"x", "accept", "c", file_kind="socket")

    def test_rejects_warn_on_file_target(self):
        with self.assertRaises(ValueError):
            ff.FuzzCase("n", "csv_file", b"x", "accept_warn", "c", warn_id="s1")

    def test_rejects_warn_without_csv_target(self):
        with self.assertRaises(ValueError):
            ff.FuzzCase("n", "json", "x", "accept_warn", "c", warn_id="s1")

    def test_rejects_warn_without_warn_id(self):
        with self.assertRaises(ValueError):
            ff.FuzzCase("n", "csv", "x", "accept_warn", "c")

    def test_rejects_pin_on_accept(self):
        with self.assertRaises(ValueError):
            ff.FuzzCase("n", "json", "x", "accept", "c", message_pin="x")

    def test_rejects_malformed_stored(self):
        with self.assertRaises(ValueError):
            ff.FuzzCase("n", "json", "x", "accept", "c", stored=("middle", 0, "id", "v"))
        with self.assertRaises(ValueError):
            ff.FuzzCase("n", "json", "x", "accept", "c", stored=("search", 0, "middle", "v"))


class ClassificationTests(unittest.TestCase):
    def test_crash_on_non_valueerror(self):
        def boom(_text):
            raise TypeError("internal bug")

        result = _eval(_case(expected="reject"), parse_json=boom)
        self.assertEqual(result.kind, "crash")
        self.assertFalse(result.conform)
        self.assertIn("TypeError", result.detail)

    def test_crash_on_accept_case(self):
        def boom(_text):
            raise RuntimeError("internal bug")

        result = _eval(_case(), parse_json=boom)
        self.assertEqual(result.kind, "crash")

    def test_unexpected_accept(self):
        result = _eval(_case(expected="reject"), parse_json=lambda _t: _BASE_SPLIT)
        self.assertEqual(result.kind, "unexpected_accept")

    def test_unexpected_reject(self):
        def nope(_text):
            raise ValueError("no")

        result = _eval(_case(), parse_json=nope)
        self.assertEqual(result.kind, "unexpected_reject")

    def test_warning_mismatch_missing_warning(self):
        case = ff.FuzzCase("w", "csv", "x", "accept_warn", "c", warn_id="s1")
        result = _eval(case, parse_csv=lambda _t: (_BASE_SPLIT, []))
        self.assertEqual(result.kind, "warning_mismatch")

    def test_warning_mismatch_wrong_fixture_named(self):
        case = ff.FuzzCase("w", "csv", "x", "accept_warn", "c", warn_id="s1")
        result = _eval(case, parse_csv=lambda _t: (_BASE_SPLIT, ["other"]))
        self.assertEqual(result.kind, "warning_mismatch")

    def test_warning_mismatch_unexpected_warning(self):
        case = ff.FuzzCase("w", "csv", "x", "accept", "c")
        result = _eval(case, parse_csv=lambda _t: (_BASE_SPLIT, ["s1"]))
        self.assertEqual(result.kind, "warning_mismatch")

    def test_warn_conforms_when_named(self):
        case = ff.FuzzCase("w", "csv", "x", "accept_warn", "c", warn_id="s1")
        result = _eval(case, parse_csv=lambda _t: (_BASE_SPLIT, ["s1"]))
        self.assertTrue(result.conform)
        self.assertEqual(result.observed, "accept_warn")

    def test_message_mismatch_codec_subclass(self):
        # The B1 vacuity guard: a UnicodeEncodeError IS a ValueError,
        # so an outcome-only check would pass it; the pin must not.
        def codec(_text):
            raise UnicodeEncodeError("utf-8", "x", 0, 1, "surrogates not allowed")

        case = _case(expected="reject", message_pin="lone surrogate")
        result = _eval(case, parse_json=codec)
        self.assertEqual(result.kind, "message_mismatch")
        self.assertFalse(result.conform)

    def test_message_mismatch_wrong_message(self):
        def wrong(_text):
            raise ValueError("some other problem")

        case = _case(expected="reject", message_pin="lone surrogate")
        result = _eval(case, parse_json=wrong)
        self.assertEqual(result.kind, "message_mismatch")

    def test_message_pin_conforms_on_exact_valueerror(self):
        def pinned(_text):
            raise ValueError("Text field contains a lone surrogate")

        case = _case(expected="reject", message_pin="lone surrogate")
        result = _eval(case, parse_json=pinned)
        self.assertTrue(result.conform)
        self.assertEqual(result.detail, "rejected with the pinned message")

    def test_unpinned_reject_conforms_any_wording(self):
        def anywording(_text):
            raise ValueError("wording that belongs to the interpreter")

        result = _eval(_case(expected="reject"), parse_json=anywording)
        self.assertTrue(result.conform)

    def test_stored_mismatch(self):
        case = _case(stored=("search", 0, "id", "different"))
        result = _eval(case, parse_json=lambda _t: _BASE_SPLIT)
        self.assertEqual(result.kind, "stored_mismatch")

    def test_stored_index_out_of_range_is_mismatch(self):
        case = _case(stored=("search", 99, "id", "s1"))
        result = _eval(case, parse_json=lambda _t: _BASE_SPLIT)
        self.assertEqual(result.kind, "stored_mismatch")

    def test_counts_mismatch(self):
        case = _case(counts=(3, 3))
        result = _eval(case, parse_json=lambda _t: _BASE_SPLIT)
        self.assertEqual(result.kind, "stored_mismatch")

    def test_stored_fields_all_kinds(self):
        split = _BASE_SPLIT
        self.assertEqual(ff._stored_value(split, ("search", 0, "id", "")), "s1")
        self.assertEqual(ff._stored_value(split, ("search", 0, "group", "")), "gs")
        self.assertEqual(ff._stored_value(split, ("search", 0, "expected", "")), "a")
        self.assertEqual(ff._stored_value(split, ("search", 0, "content", "")), "q one")
        self.assertEqual(ff._stored_value(split, ("search", 0, "content_last", "")), "q one")
        self.assertIsNone(ff._stored_value(split, ("search", 7, "id", "")))

    def test_file_target_reject_via_fake_loader(self):
        def nope(_path):
            raise ValueError("unreadable")

        case = ff.FuzzCase("f", "json_file", b"{}", "reject", "c")
        result = _eval(case, load_json_file=nope)
        self.assertTrue(result.conform)
        self.assertEqual(result.observed, "reject")

    def test_file_target_crash_via_oserror(self):
        def boom(_path):
            raise OSError("disk went away")

        case = ff.FuzzCase("f", "csv_file", b"x", "accept", "c")
        result = _eval(case, read_csv_file=boom)
        self.assertEqual(result.kind, "crash")

    def test_type_invariant_guards_surface_as_crash(self):
        # The constructors make wrong-typed payloads unrepresentable;
        # these guards are the last line if a case object is ever
        # corrupted in memory (proven here by bypassing the frozen
        # constructor), and they must fail closed as crash findings.
        json_case = _case()
        object.__setattr__(json_case, "payload", b"bytes-not-str")
        result = _eval(json_case)
        self.assertEqual(result.kind, "crash")
        self.assertIn("TypeError", result.detail)
        csv_case = ff.FuzzCase("c", "csv", "x", "accept", "c")
        object.__setattr__(csv_case, "payload", b"bytes-not-str")
        result = _eval(csv_case)
        self.assertEqual(result.kind, "crash")
        file_case = ff.FuzzCase("f", "json_file", b"{}", "accept", "c")
        object.__setattr__(file_case, "payload", "str-not-bytes")
        result = _eval(file_case)
        self.assertEqual(result.kind, "crash")
        self.assertIn("TypeError", result.detail)


class LoaderFixRegressionTests(unittest.TestCase):
    def test_escape_surrogate_rejected_in_every_json_field(self):
        base = _BASE_TEXT
        variants = {
            "content": base.replace('"q one"', '"a\\ud800"'),
            "id": base.replace('"s1"', '"a\\ud800"', 1),
            "group": base.replace('"gs"', '"a\\ud800"', 1),
            "expected": base.replace('"expected": "a"', '"expected": "a\\ud800"'),
        }
        for field, text in variants.items():
            with self.subTest(field=field):
                with self.assertRaises(ValueError) as info:
                    parse_dataset(text)
                self.assertIs(type(info.exception), ValueError)
                self.assertIn("lone surrogate", str(info.exception))

    def test_surrogate_pair_escape_still_accepted(self):
        text = _BASE_TEXT.replace('"q one"', '"\\ud83d\\ude00"')
        split = parse_dataset(text)
        self.assertEqual(split.search[0].messages[0].content, "\U0001f600")

    def test_raw_surrogate_text_is_deliberate_valueerror(self):
        doc = ff._json_base()
        doc["search"][0]["messages"][0]["content"] = "q\ud800one"
        with self.assertRaises(ValueError) as info:
            parse_dataset(ff._json_text(doc))
        self.assertIs(type(info.exception), ValueError)
        self.assertIn("Dataset text contains a lone surrogate", str(info.exception))

    def test_non_str_still_size_message(self):
        class Sub(str):
            pass

        with self.assertRaisesRegex(ValueError, "Dataset exceeds size limit"):
            parse_dataset(Sub(_BASE_TEXT))

    def test_csv_overlong_id_and_group_rejected(self):
        header = ff._CSV_HEADER
        for row in (
            "x" * 17_000 + ",gs,search,ans one,q one,a one,q follow",
            "s1," + "g" * 17_000 + ",search,ans one,q one,a one,q follow",
        ):
            text = header + "\n" + row + "\nh1,gh,held_out,ans three,q three,a three,q held\n"
            with self.assertRaisesRegex(ValueError, "text field exceeds size limit"):
                parse_fixture_csv(text)

    def test_csv_id_at_limit_accepted_and_round_trips(self):
        fid = "y" * ff._MAX_TEXT
        text = (
            ff._CSV_HEADER
            + "\n"
            + f"{fid},gs,search,ans one,q one,a one,q follow\n"
            + "h1,gh,held_out,ans three,q three,a three,q held\n"
        )
        split, _warnings = parse_fixture_csv(text)
        self.assertEqual(split.search[0].id, fid)
        parse_dataset(split_to_json_text(split))  # the by-construction promise

    def test_csv_bare_cr_is_parse_error_not_size_lie(self):
        text = ff._csv_text(eol="\r")
        with self.assertRaisesRegex(ValueError, "CSV parse error"):
            parse_fixture_csv(text)

    def test_csv_genuine_field_limit_keeps_its_message(self):
        rows = list(ff._CSV_ROWS)
        rows[0] = f"s1,gs,search,ans one,{'z' * 131_073},a one,q follow"
        with self.assertRaisesRegex(ValueError, "parser limit"):
            parse_fixture_csv(ff._csv_text(tuple(rows)))

    def test_csv_raw_surrogate_is_deliberate_valueerror(self):
        rows = list(ff._CSV_ROWS)
        rows[0] = "s1,gs,search,ans one,q\ud800one,a one,q follow"
        with self.assertRaises(ValueError) as info:
            parse_fixture_csv(ff._csv_text(tuple(rows)))
        self.assertIs(type(info.exception), ValueError)
        self.assertIn("CSV source contains a lone surrogate", str(info.exception))


def _run_main(argv):
    from model_protocol_lab.cli import main

    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


class CliTests(unittest.TestCase):
    def test_fuzz_command_exit_zero_and_json_report(self):
        code, out, err = _run_main(["fixtures", "fuzz"])
        self.assertEqual(code, 0, err)
        report = json.loads(out)
        self.assertEqual(report["schema"], ff.SCHEMA_ID)
        self.assertEqual(report["seed"], 0)
        self.assertEqual(report["cases"], 133)
        self.assertTrue(report["ok"])
        self.assertIn("133/133 cases conform", err)

    def test_fuzz_command_seed_flag(self):
        code0, out0, _ = _run_main(["fixtures", "fuzz", "--seed", "0"])
        code5, out5, _ = _run_main(["fixtures", "fuzz", "--seed", "5"])
        self.assertEqual((code0, code5), (0, 0))
        self.assertNotEqual(json.loads(out0)["digest"], json.loads(out5)["digest"])

    def test_fuzz_help_and_fixtures_help(self):
        from model_protocol_lab.cli import main

        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as info:
            main(["fixtures", "--help"])
        self.assertEqual(info.exception.code, 0)
        self.assertIn("fuzz", out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as info:
            main(["fixtures", "fuzz", "--help"])
        self.assertEqual(info.exception.code, 0)
        self.assertIn("Corpus seed", out.getvalue())

    def test_run_fixtures_fuzz_exit_one_on_finding(self):
        original = ff.generate_cases
        bad = ff.FuzzCase("bad", "json", "not json at all", "accept", "rigged")
        ff.generate_cases = lambda _seed=0: (bad,)  # noqa: E731
        try:
            import argparse

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                code = ff.run_fixtures_fuzz(argparse.Namespace(seed=0), argparse.ArgumentParser())
        finally:
            ff.generate_cases = original
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(stdout.getvalue())["ok"])

    def test_monday_morning_use_names_the_contract(self):
        self.assertIn("published table", ff.MONDAY_MORNING_USE)
        self.assertIn("accept/reject contract", ff.MONDAY_MORNING_USE)


# Tempdir mutation gates: each mutant is exec'd under the package
# namespace (relative imports) from a /tmp copy; the scenario must
# pass on the real module and fail on the mutant.
# ---------------------------------------------------------------------------


class FixtureFuzzMutationGateTests(unittest.TestCase):
    def _run_gate(self, name, old_text, new_text, scenario):
        source = Path(ff.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c172_mut_", dir="/tmp") as tmpdir:
            mod_path = os.path.join(tmpdir, f"fixturefuzz_mut_{name}.py")
            with open(mod_path, "w", encoding="utf-8") as handle:
                handle.write(mutated)
            dotted = f"model_protocol_lab.fixturefuzz_mut_{name}"
            spec = importlib.util.spec_from_file_location(dotted, mod_path)
            assert spec is not None and spec.loader is not None
            mutant = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mutant)
            scenario(ff)  # passes on the real module: non-vacuous
            with self.assertRaises(AssertionError, msg=f"mutant {name} was NOT caught"):
                scenario(mutant)

    def test_mutant_report_always_ok_caught(self):
        def scenario(mod):
            report = mod.FuzzReport(
                seed=0,
                digest="x",
                results=(mod.CaseResult("n", "json", "accept", "crash", False, "crash", "d"),),
            )
            assert not report.ok

        self._run_gate(
            "always_ok",
            "return self.conforming == self.total",
            "return True",
            scenario,
        )

    def test_mutant_message_pin_ignored_caught(self):
        def scenario(mod):
            def wrong(_text):
                raise ValueError("unrelated wording")

            case = mod.FuzzCase("n", "json", "x", "reject", "c", message_pin="lone surrogate")
            with tempfile.TemporaryDirectory(prefix="c172_g2_") as tmp:
                result = mod.evaluate_case(case, Path(tmp), parse_json=wrong)
            assert result.kind == "message_mismatch", result
            assert not result.conform

        self._run_gate(
            "pin_ignored",
            "case.message_pin not in str(exc)",
            "False",
            scenario,
        )

    def test_mutant_crash_blurred_to_reject_caught(self):
        def scenario(mod):
            def boom(_text):
                raise TypeError("internal bug")

            case = mod.FuzzCase("n", "json", "x", "reject", "c")
            with tempfile.TemporaryDirectory(prefix="c172_g3_") as tmp:
                result = mod.evaluate_case(case, Path(tmp), parse_json=boom)
            assert result.kind == "crash", result

        self._run_gate(
            "crash_blur",
            'return "crash", None, [], exc',
            'return "reject", None, [], exc',
            scenario,
        )

    def test_mutant_digest_ignores_payloads_caught(self):
        def scenario(mod):
            one = mod.FuzzCase("n", "json", "{}", "reject", "c")
            two = mod.FuzzCase("n", "json", "{ }", "reject", "c")
            assert mod.corpus_digest((one,)) != mod.corpus_digest((two,))

        self._run_gate(
            "digest_blind",
            "hasher.update(raw)",
            'hasher.update(b"")',
            scenario,
        )

    def test_mutant_stored_check_dropped_caught(self):
        def scenario(mod):
            case = mod.FuzzCase(
                "n",
                "json",
                "x",
                "accept",
                "c",
                stored=("search", 0, "id", "different"),
            )
            with tempfile.TemporaryDirectory(prefix="c172_g5_") as tmp:
                result = mod.evaluate_case(case, Path(tmp), parse_json=lambda _t: _BASE_SPLIT)
            assert result.kind == "stored_mismatch", result

        self._run_gate(
            "stored_dropped",
            "and _stored_value(split, case.stored) != case.stored[3]",
            "and False",
            scenario,
        )


if __name__ == "__main__":
    unittest.main()
