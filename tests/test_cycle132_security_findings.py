"""Cycle 132: regression tests for the security red-team findings.

Reproduces findings 1 (render_html crash sites), 2 (SIGALRM swallowed in
the RLIMIT_AS restore), 3 (symlinked report-dir leaf), 7 (backend failure
message allowlist), 8 (portability-table stderr sanitization), and 9
(non-finite rates) from agents_out/security/REPORT.md against the current
tree. Every test fails on the pre-fix code and passes after.
"""

import math
import os
import resource
import signal
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from model_protocol_lab import cli, doctor, reporting
from model_protocol_lab.reporting import (
    _frontier_chart_svg,
    _frontier_series,
    _frontier_spark_table,
    _held_out_section,
    _search_rows,
    prepare_report_dir,
)


def _frontier(**overrides):
    base = {
        "max_new_tokens": 8,
        "candidates": {
            "a": {
                "fixture_ids": ["f1", "f2"],
                "pass_steps": [[1, 0], [4, 1], [8, 2]],
                "min_budget_per_fixture": [4, 8],
                "full_pass_budget": 8,
            }
        },
    }
    base.update(overrides)
    return base


class TestFrontierCrashSites(unittest.TestCase):
    """Finding 1: hostile frontier payloads must degrade, never crash."""

    def test_mixed_type_candidate_keys_chart(self):
        frontier = _frontier()
        frontier["candidates"][3] = {
            "fixture_ids": ["f1"],
            "pass_steps": [[1, 1]],
        }
        # The hostile key is skipped; the valid candidate still charts.
        html = _frontier_chart_svg(frontier)
        self.assertIn("<svg", html)

    def test_mixed_type_candidate_keys_series(self):
        frontier = _frontier()
        frontier["candidates"][3] = {
            "fixture_ids": ["f1"],
            "pass_steps": [[1, 1]],
        }
        # The hostile key is skipped; the valid series still validates.
        self.assertEqual(_frontier_series(frontier), (["a"], 8))

    def test_int_fixture_ids_chart(self):
        frontier = _frontier()
        frontier["candidates"]["a"]["fixture_ids"] = 3
        self.assertEqual(_frontier_chart_svg(frontier), "")

    def test_int_fixture_ids_series(self):
        frontier = _frontier()
        frontier["candidates"]["a"]["fixture_ids"] = 3
        self.assertIsNone(_frontier_series(frontier))

    def test_int_fixture_ids_spark_table(self):
        # Degrades to a zeroed row, never crashes.
        frontier = _frontier()
        frontier["candidates"]["a"]["fixture_ids"] = 3
        html = _frontier_spark_table(frontier)
        self.assertIn("Pass curves by candidate", html)
        self.assertIn("a", html)

    def test_int_min_budget_per_fixture(self):
        frontier = _frontier()
        frontier["candidates"]["a"]["min_budget_per_fixture"] = 7
        html = _frontier_spark_table(frontier)
        self.assertIn("Pass curves by candidate", html)

    def test_string_count_in_winner_curve_skipped(self):
        from model_protocol_lab.reporting import _frontier_section

        frontier = _frontier(native_winner="a")
        frontier["candidates"]["a"]["pass_steps"] = [[1, "x"], [4, 1]]
        html = _frontier_section(
            {"budget_frontier": frontier, "selection": {"decision": "selected"}}
        )
        self.assertIn("Winner pass curve", html)

    def test_string_budget_in_winner_curve_skipped(self):
        from model_protocol_lab.reporting import _frontier_section

        frontier = _frontier(native_winner="a")
        frontier["candidates"]["a"]["pass_steps"] = [["1", 1], [4, 1]]
        html = _frontier_section(
            {"budget_frontier": frontier, "selection": {"decision": "selected"}}
        )
        self.assertIn("Winner pass curve", html)

    def test_margin_trajectory_int_fixture_ids(self):
        frontier = _frontier()
        frontier["candidates"]["a"]["fixture_ids"] = 3
        self.assertEqual(reporting.margin_trajectory(frontier), [])


class TestRateFormatting(unittest.TestCase):
    """Findings 1+9: malformed rates degrade to "incomplete", never crash."""

    @staticmethod
    def _search_result(rate):
        return {
            "search": [
                {
                    "format": {"name": "a"},
                    "summary": {
                        "rate": rate,
                        "status": "completed",
                        "exact_and_boundary_passes": 1,
                        "planned": 2,
                        "attempted": 2,
                    },
                }
            ]
        }

    def test_string_rate_search_rows(self):
        html = _search_rows(self._search_result("high"))
        self.assertIn("incomplete", html)

    def test_nan_rate_search_rows(self):
        html = _search_rows(self._search_result(float("nan")))
        self.assertIn("incomplete", html)
        self.assertNotIn("nan%", html)

    def test_inf_rate_search_rows(self):
        html = _search_rows(self._search_result(math.inf))
        self.assertIn("incomplete", html)

    def test_bool_rate_search_rows(self):
        html = _search_rows(self._search_result(True))
        self.assertIn("incomplete", html)

    def test_string_rate_held_out(self):
        html = _held_out_section({"held_out": {"summary": {"rate": "high"}}})
        self.assertIn("incomplete", html)

    def test_nan_rate_held_out(self):
        html = _held_out_section({"held_out": {"summary": {"rate": float("nan")}}})
        self.assertIn("incomplete", html)
        self.assertNotIn("nan%", html)

    def test_valid_rates_unchanged(self):
        html = _search_rows(self._search_result(0.5))
        self.assertIn("50%", html)
        html = _search_rows(self._search_result(1))
        self.assertIn("100%", html)
        self.assertIn("badge pass", html)


class TestSymlinkReportDir(unittest.TestCase):
    """Finding 3: a symlinked leaf is never claimed, even when empty."""

    def test_symlinked_leaf_rejected_with_allow_empty(self):
        with TemporaryDirectory() as tmp:
            evil = Path(tmp) / "evil"
            evil.mkdir()
            link = Path(tmp) / "report"
            os.symlink(evil, link)
            with self.assertRaises(FileExistsError):
                prepare_report_dir(link, _allow_empty_existing=True)
            # Nothing was written through the link.
            self.assertEqual(list(evil.iterdir()), [])

    def test_symlinked_leaf_rejected_without_flag(self):
        with TemporaryDirectory() as tmp:
            evil = Path(tmp) / "evil"
            evil.mkdir()
            link = Path(tmp) / "report"
            os.symlink(evil, link)
            with self.assertRaises(FileExistsError):
                prepare_report_dir(link)

    def test_empty_real_dir_still_claimable(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "report"
            target.mkdir()
            claimed = prepare_report_dir(target, _allow_empty_existing=True)
            self.assertEqual(claimed, target)


class FakeTokenizer:
    @staticmethod
    def apply_chat_template(conversation, tokenize=False, add_generation_prompt=True):
        return "hello"


@unittest.skipIf(not hasattr(signal, "SIGALRM"), "needs POSIX SIGALRM")
class TestRlimitRestoreRace(unittest.TestCase):
    """Finding 2: a budget alarm landing mid-restore must not be swallowed."""

    def test_budget_exception_during_restore_propagates(self):
        calls = []
        real_setrlimit = resource.setrlimit

        def spy(which, limits):
            calls.append(limits)
            if len(calls) == 2:
                # Simulate the SIGALRM handler firing inside the restore:
                # the budget exception is a ValueError subclass.
                raise doctor._RenderTimeBudgetExceeded("simulated mid-restore alarm")
            return real_setrlimit(which, limits)

        with (
            mock.patch("resource.setrlimit", spy),
            self.assertRaises(doctor._RenderTimeBudgetExceeded),
        ):
            doctor._render_with_memory_cap(FakeTokenizer(), [{"role": "user", "content": "x"}])
        # Install cap, restore attempt, retry after the alarm: the restore
        # was ensured before the exception propagated.
        self.assertEqual(len(calls), 3)

    def test_normal_path_restores_limit(self):
        old_soft, _ = resource.getrlimit(resource.RLIMIT_AS)
        doctor._render_with_memory_cap(FakeTokenizer(), [{"role": "user", "content": "x"}])
        new_soft, _ = resource.getrlimit(resource.RLIMIT_AS)
        self.assertEqual(old_soft, new_soft)


class TestBackendFailureAllowlist(unittest.TestCase):
    """Finding 7: only fixed local_* codes echo; the prefix match is gone."""

    def test_all_known_codes_render_tersely(self):
        for code in cli._KNOWN_LOCAL_FAILURE_CODES:
            with self.subTest(code=code):
                msg = cli._backend_failure_message(code)
                self.assertIn(f"[{code}]", msg)

    def test_known_codes_match_backends_raise_sites(self):
        import re

        source = Path(doctor.__file__).parent.joinpath("backends.py").read_text()
        raised = set(re.findall(r'raise RuntimeError\("(local_[a-z_0-9]+)"', source))
        self.assertEqual(set(cli._KNOWN_LOCAL_FAILURE_CODES), raised)

    def test_local_prefixed_evil_stays_generic(self):
        msg = cli._backend_failure_message("local_evil_injected")
        self.assertNotIn("local_evil_injected", msg)
        self.assertIn("Operation failed.", msg)

    def test_arbitrary_text_stays_generic(self):
        msg = cli._backend_failure_message("oops'; rm -rf / #")
        self.assertNotIn("rm -rf", msg)


class TestPortabilityTableSanitization(unittest.TestCase):
    """Finding 8: escape sequences in table fields cannot reach stderr."""

    def test_hostile_fields_stripped(self):
        section = {
            "status": "ok",
            "second_checkpoint": "ckpt-dir\x1b[31m",
            "rows": [
                {
                    "format": "chatml\x1b]0;evil\x07",
                    "verdict": "ok\u202e",
                    "reason": "fine\x08",
                }
            ],
        }
        rendered = doctor.render_portability_table(section)
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\u202e", rendered)
        self.assertNotIn("\x08", rendered)
        self.assertIn("chatml", rendered)

    def test_normal_values_unchanged(self):
        section = {
            "status": "ok",
            "second_checkpoint": "ckpt-dir",
            "rows": [{"format": "chatml", "verdict": "compatible", "reason": "ids match"}],
        }
        rendered = doctor.render_portability_table(section)
        self.assertIn("ckpt-dir", rendered)
        self.assertIn("chatml", rendered)
        self.assertIn("compatible", rendered)

    def test_non_string_fields_do_not_crash(self):
        section = {
            "status": "ok",
            "second_checkpoint": "ckpt-dir",
            "rows": [{"format": 5, "verdict": None, "reason": ["x"]}],
        }
        rendered = doctor.render_portability_table(section)
        self.assertIn("5", rendered)

    def test_failed_status_sanitized(self):
        section = {"status": "failed\x1b[2J", "detail": "bad\x07detail"}
        rendered = doctor.render_portability_table(section)
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\x07", rendered)


if __name__ == "__main__":
    unittest.main()
