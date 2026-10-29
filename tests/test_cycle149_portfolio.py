"""Cycle 149: oracle-mixture / protocol-portfolio gap (research wave-2 idea 4).

Tests for the pure ``model_protocol_lab/portfolio.py`` module, the
``experiments.portfolio_analysis`` driver, the ``diff_portfolio``
comparison, the stderr summary lines, the HTML section (escaping), four
tempdir-module mutation gates, and a brute-force oracle differential
for the greedy cover.
"""

import importlib
import os
import random
import tempfile
import unittest
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from model_protocol_lab import diffreports
from model_protocol_lab.core import Fixture, Message
from model_protocol_lab.experiments import portfolio_analysis
from model_protocol_lab.portfolio import (
    greedy_cover,
    oracle_rates,
    per_fixture_oracle_winners,
    validate_fixture_ids,
    validate_passing,
)
from model_protocol_lab.reporting import _portfolio_section


def _case(fixture_id, expected="A"):
    """Case."""
    return Fixture(
        id=fixture_id,
        messages=(Message(role="user", content="q"),),
        expected=expected,
        group="g",
    )


def _obs(candidate, fixture_id, exact, boundary):
    """Obs."""
    return {
        "candidate": candidate,
        "fixture": fixture_id,
        "input_ids": (1,),
        "output_ids": (2,),
        "answer": "x",
        "exact_match": exact,
        "boundary_found": boundary,
        "boundary_text_match": None,
        "raw_reason": None,
        "error": None,
    }


def _result(entries, decision="ambiguous", winner=None):
    """Result."""
    return {"decision": decision, "selected_candidate": winner, "search": entries}


def _entry(name, obs_list):
    """Entry."""
    return {"format": {"name": name}, "observations": obs_list}


FIXTURES = ("f1", "f2", "f3")


class ValidatePassingTests(unittest.TestCase):
    """Validate passing tests."""

    def test_round_trip_normalizes_sorted(self):
        """Test round trip normalizes sorted."""
        out = validate_passing({"b": ("f2", "f1", "f1"), "a": ()}, FIXTURES)
        self.assertEqual(out, {"b": ("f1", "f2"), "a": ()})

    def test_rejects_non_dict(self):
        """Test rejects non dict."""
        with self.assertRaises(ValueError):
            validate_passing([("a", ())], FIXTURES)

    def test_rejects_empty_map(self):
        """Test rejects empty map."""
        with self.assertRaises(ValueError):
            validate_passing({}, FIXTURES)

    def test_rejects_blank_candidate_name(self):
        """Test rejects blank candidate name."""
        with self.assertRaises(ValueError):
            validate_passing({"": ("f1",)}, FIXTURES)

    def test_rejects_non_tuple_passed(self):
        """Test rejects non tuple passed."""
        with self.assertRaises(ValueError):
            validate_passing({"a": ["f1"]}, FIXTURES)

    def test_rejects_unknown_fixture(self):
        """Test rejects unknown fixture."""
        with self.assertRaises(ValueError):
            validate_passing({"a": ("f9",)}, FIXTURES)

    def test_rejects_empty_fixture_ids(self):
        """Test rejects empty fixture ids."""
        with self.assertRaises(ValueError):
            validate_passing({"a": ()}, ())

    def test_rejects_duplicate_fixture_ids(self):
        """Test rejects duplicate fixture ids."""
        with self.assertRaises(ValueError):
            validate_passing({"a": ()}, ("f1", "f1"))

    def test_rejects_non_string_fixture_id(self):
        """Test rejects non string fixture id."""
        with self.assertRaises(ValueError):
            validate_passing({"a": (7,)}, FIXTURES)

    def test_rejects_non_string_fixture_id_in_ids(self):
        """Test rejects non string fixture id in ids."""
        with self.assertRaises(ValueError):
            validate_fixture_ids(("f1", 7))


class OracleRatesTests(unittest.TestCase):
    """Oracle rates tests."""

    def test_rates_and_gap(self):
        """Test rates and gap."""
        passing = validate_passing({"alpha": ("f1", "f2"), "beta": ("f2", "f3")}, FIXTURES)
        rates = oracle_rates(passing, FIXTURES, "alpha")
        self.assertEqual(rates["n_fixtures"], 3)
        self.assertEqual(rates["n_solvable"], 3)
        self.assertEqual(rates["solvable_fixture_ids"], ["f1", "f2", "f3"])
        self.assertEqual(rates["unsolvable_fixture_ids"], [])
        self.assertAlmostEqual(rates["oracle_pass_rate"], 1.0)
        self.assertAlmostEqual(rates["winner_pass_rate"], 2 / 3)
        self.assertAlmostEqual(rates["gap"], 1 / 3)
        self.assertGreaterEqual(rates["gap"], 0.0)

    def test_gap_zero_when_winner_covers_all_solvable(self):
        """Test gap zero when winner covers all solvable."""
        passing = validate_passing({"alpha": ("f1", "f2", "f3")}, FIXTURES)
        rates = oracle_rates(passing, FIXTURES, "alpha")
        self.assertAlmostEqual(rates["gap"], 0.0)

    def test_no_winner_gap_is_oracle_rate(self):
        """Test no winner gap is oracle rate."""
        passing = validate_passing({"alpha": ("f1",)}, FIXTURES)
        rates = oracle_rates(passing, FIXTURES, None)
        self.assertIsNone(rates["winner_pass_rate"])
        self.assertAlmostEqual(rates["gap"], 1 / 3)

    def test_unsolvable_fixtures_excluded_from_oracle(self):
        """Test unsolvable fixtures excluded from oracle."""
        passing = validate_passing({"alpha": ("f1",)}, FIXTURES)
        rates = oracle_rates(passing, FIXTURES, "alpha")
        self.assertEqual(rates["n_solvable"], 1)
        self.assertEqual(rates["unsolvable_fixture_ids"], ["f2", "f3"])

    def test_rejects_unknown_winner(self):
        """Test rejects unknown winner."""
        passing = validate_passing({"alpha": ("f1",)}, FIXTURES)
        with self.assertRaises(ValueError):
            oracle_rates(passing, FIXTURES, "zzz")

    def test_rejects_blank_winner(self):
        """Test rejects blank winner."""
        passing = validate_passing({"alpha": ("f1",)}, FIXTURES)
        with self.assertRaises(ValueError):
            oracle_rates(passing, FIXTURES, "")


class OracleWinnersTests(unittest.TestCase):
    """Oracle winners tests."""

    def test_per_fixture_winners_sorted_and_solvable_only(self):
        """Test per fixture winners sorted and solvable only."""
        passing = validate_passing({"beta": ("f2",), "alpha": ("f1", "f2")}, FIXTURES)
        winners = per_fixture_oracle_winners(passing, FIXTURES)
        # Deterministic tie-break: sorted by family name.
        self.assertEqual(winners, {"f1": ["alpha"], "f2": ["alpha", "beta"]})

    def test_tie_break_independent_of_input_order(self):
        """Test tie break independent of input order."""
        forward = validate_passing({"alpha": ("f1",), "beta": ("f1",)}, ("f1",))
        backward = validate_passing({"beta": ("f1",), "alpha": ("f1",)}, ("f1",))
        self.assertEqual(
            per_fixture_oracle_winners(forward, ("f1",)),
            per_fixture_oracle_winners(backward, ("f1",)),
        )


class GreedyCoverTests(unittest.TestCase):
    """Greedy cover tests."""

    def test_single_cover(self):
        """Test single cover."""
        passing = validate_passing({"alpha": ("f1", "f2")}, ("f1", "f2"))
        self.assertEqual(greedy_cover(passing, ("f1", "f2")), ["alpha"])

    def test_greedy_picks_max_coverage_first(self):
        """Test greedy picks max coverage first."""
        passing = validate_passing(
            {"alpha": ("f1", "f2", "f3"), "beta": ("f1",)}, ("f1", "f2", "f3")
        )
        self.assertEqual(greedy_cover(passing, ("f1", "f2", "f3")), ["alpha"])

    def test_tie_broken_by_lexicographically_smallest_name(self):
        """Test tie broken by lexicographically smallest name."""
        passing = validate_passing({"beta": ("f1",), "alpha": ("f1",)}, ("f1",))
        self.assertEqual(greedy_cover(passing, ("f1",)), ["alpha"])

    def test_cover_order_deterministic_under_input_shuffle(self):
        """Test cover order deterministic under input shuffle."""
        passing = validate_passing(
            {"gamma": ("f3",), "alpha": ("f1", "f2"), "beta": ("f2", "f3")}, FIXTURES
        )
        first = greedy_cover(passing, FIXTURES)
        rng = random.Random(149)  # noqa: S311 -- deterministic test seed
        names = list(passing)
        for _ in range(20):
            rng.shuffle(names)
            shuffled = validate_passing({n: passing[n] for n in names}, FIXTURES)
            self.assertEqual(greedy_cover(shuffled, FIXTURES), first)

    def test_unsolvable_fixtures_are_not_cover_targets(self):
        """Test unsolvable fixtures are not cover targets."""
        passing = validate_passing({"alpha": ("f1",)}, FIXTURES)
        self.assertEqual(greedy_cover(passing, FIXTURES), ["alpha"])

    def test_empty_cover_when_nothing_solvable(self):
        """Test empty cover when nothing solvable."""
        passing = validate_passing({"alpha": ()}, ("f1",))
        self.assertEqual(greedy_cover(passing, ("f1",)), [])

    def test_honest_approximation_documented(self):
        """Test honest approximation documented."""
        # The greedy cover is reported as an approximation everywhere it
        # surfaces; pin the disclaimer so it cannot silently vanish.
        doc = __import__("model_protocol_lab.portfolio", fromlist=["x"]).__doc__
        self.assertIn("approximation", doc)


class PortfolioProperties(unittest.TestCase):
    """Portfolio properties."""

    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=4),
            st.lists(st.sampled_from(["f1", "f2", "f3"]), unique=True).map(tuple),
            min_size=1,
            max_size=4,
        )
    )
    @settings(max_examples=120)
    def test_gap_nonnegative_for_every_winner(self, raw):
        """Test gap nonnegative for every winner."""
        passing = validate_passing(raw, FIXTURES)
        for winner in passing:
            rates = oracle_rates(passing, FIXTURES, winner)
            self.assertGreaterEqual(rates["gap"], 0.0)

    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=4),
            st.lists(st.sampled_from(["f1", "f2", "f3"]), unique=True).map(tuple),
            min_size=1,
            max_size=4,
        )
    )
    @settings(max_examples=120)
    def test_portfolio_covers_every_solvable_fixture(self, raw):
        """Test portfolio covers every solvable fixture."""
        passing = validate_passing(raw, FIXTURES)
        portfolio = greedy_cover(passing, FIXTURES)
        covered = {fid for name in portfolio for fid in passing[name]}
        solvable = {fid for fid in FIXTURES if any(fid in p for p in passing.values())}
        self.assertEqual(covered, solvable)
        for name in portfolio:
            self.assertIn(name, passing)

    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=4),
            st.lists(st.sampled_from(["f1", "f2", "f3"]), unique=True).map(tuple),
            min_size=1,
            max_size=4,
        )
    )
    @settings(max_examples=60)
    def test_oracle_winners_match_passing_sets(self, raw):
        """Test oracle winners match passing sets."""
        passing = validate_passing(raw, FIXTURES)
        winners = per_fixture_oracle_winners(passing, FIXTURES)
        for fid in FIXTURES:
            expected = sorted(n for n, p in passing.items() if fid in p)
            self.assertEqual(winners.get(fid, []), expected)


class GreedyCoverBruteForceTests(unittest.TestCase):
    """Differential: greedy_cover vs an independent brute-force reimplementation."""

    @staticmethod
    def _brute_force(passing, fixture_ids):
        """Brute force."""
        # Independent reimplementation: same greedy rule, written from the
        # docstring (max new coverage, name tie-break), different code.
        targets = sorted(fid for fid in fixture_ids if any(fid in p for p in passing.values()))
        uncovered = set(targets)
        portfolio = []
        while uncovered:
            scored = sorted(
                (
                    (len(set(passing[name]) & uncovered), name)
                    for name in passing
                    if set(passing[name]) & uncovered
                ),
                key=lambda pair: (-pair[0], pair[1]),
            )
            if not scored:
                break
            _, name = scored[0]
            portfolio.append(name)
            uncovered -= set(passing[name])
        return portfolio

    def test_matches_brute_force_on_seeded_cases(self):
        """Test matches brute force on seeded cases."""
        rng = random.Random(20261018)  # noqa: S311 -- deterministic test seed
        names = ["alpha", "beta", "gamma", "delta", "epsilon"]
        fixtures = tuple(f"f{i}" for i in range(8))
        for _ in range(200):
            raw = {name: tuple(f for f in fixtures if rng.random() < 0.4) for name in names}
            passing = validate_passing(raw, fixtures)
            self.assertEqual(
                greedy_cover(passing, fixtures),
                self._brute_force(passing, fixtures),
            )


class PortfolioAnalysisTests(unittest.TestCase):
    """Portfolio analysis tests."""

    def test_gap_scenario_from_observations(self):
        """Test gap scenario from observations."""
        cases = (_case("f1"), _case("f2"))
        result = _result(
            [
                _entry(
                    "alpha",
                    [_obs("alpha", "f1", True, True), _obs("alpha", "f2", False, True)],
                ),
                _entry(
                    "beta",
                    [_obs("beta", "f1", False, True), _obs("beta", "f2", True, True)],
                ),
            ],
            decision="ambiguous",
        )
        analysis = portfolio_analysis(result, cases)
        self.assertTrue(analysis["available"])
        self.assertIsNone(analysis["winner"])
        self.assertAlmostEqual(analysis["oracle"]["gap"], 1.0)
        self.assertEqual(analysis["portfolio"]["members"], ["alpha", "beta"])

    def test_winner_rate_from_observations(self):
        """Test winner rate from observations."""
        cases = (_case("f1"), _case("f2"))
        result = _result(
            [
                _entry(
                    "alpha",
                    [_obs("alpha", "f1", True, True), _obs("alpha", "f2", True, True)],
                ),
                _entry(
                    "beta",
                    [_obs("beta", "f1", True, False), _obs("beta", "f2", False, False)],
                ),
            ],
            decision="selected",
            winner="alpha",
        )
        analysis = portfolio_analysis(result, cases)
        self.assertEqual(analysis["winner"], "alpha")
        self.assertAlmostEqual(analysis["oracle"]["winner_pass_rate"], 1.0)
        self.assertAlmostEqual(analysis["oracle"]["gap"], 0.0)
        self.assertEqual(analysis["portfolio"]["members"], ["alpha"])

    def test_boundary_miss_is_not_a_pass(self):
        """Test boundary miss is not a pass."""
        # exact_match without boundary_found must not count as passing.
        cases = (_case("f1"),)
        result = _result(
            [_entry("alpha", [_obs("alpha", "f1", True, False)])], decision="no_candidate"
        )
        analysis = portfolio_analysis(result, cases)
        self.assertEqual(analysis["oracle"]["n_solvable"], 0)
        self.assertEqual(analysis["portfolio"]["members"], [])

    def test_misaligned_candidate_excluded_not_fatal(self):
        """Test misaligned candidate excluded not fatal."""
        cases = (_case("f1"), _case("f2"))
        result = _result(
            [
                _entry("alpha", [_obs("alpha", "f1", True, True)]),  # one obs short
                _entry(
                    "beta",
                    [_obs("beta", "f1", True, True), _obs("beta", "f2", True, True)],
                ),
            ],
            decision="selected",
            winner="beta",
        )
        analysis = portfolio_analysis(result, cases)
        self.assertTrue(analysis["available"])
        self.assertEqual(analysis["excluded_candidates"], ["alpha"])
        self.assertEqual(analysis["n_candidates"], 1)

    def test_all_misaligned_is_unavailable(self):
        """Test all misaligned is unavailable."""
        cases = (_case("f1"),)
        result = _result([_entry("alpha", [])], decision="no_candidate")
        analysis = portfolio_analysis(result, cases)
        self.assertFalse(analysis["available"])

    def test_hostile_inputs_never_raise(self):
        """Test hostile inputs never raise."""
        cases = (_case("f1"),)
        hostile = [
            None,
            42,
            "junk",
            {},
            {"search": "junk"},
            {"search": []},
            {"search": [{"format": "junk"}]},
            {"search": [{"format": {"name": 7}, "observations": []}]},
            {"search": [{"format": {"name": "a"}, "observations": "junk"}]},
            {"search": [{"format": {"name": "a"}, "observations": [None]}]},
        ]
        for bad in hostile:
            with self.subTest(bad=bad):
                analysis = portfolio_analysis(bad, cases)
                self.assertFalse(analysis["available"])
        # A candidate passing nothing is legitimate input, not hostile:
        # the analysis stays available with an empty oracle.
        zero_pass = {
            "search": [{"format": {"name": "a"}, "observations": [{"exact_match": "yes"}]}]
        }
        analysis = portfolio_analysis(zero_pass, cases)
        self.assertTrue(analysis["available"])
        self.assertEqual(analysis["oracle"]["n_solvable"], 0)
        # Bad cases tuple is also unavailable, never a raise.
        good = _result([_entry("a", [_obs("a", "f1", True, True)])])
        self.assertFalse(portfolio_analysis(good, ())["available"])
        self.assertFalse(portfolio_analysis(good, "junk")["available"])

    def test_non_dict_observation_excluded(self):
        """Test non dict observation excluded."""
        cases = (_case("f1"),)
        result = _result([_entry("alpha", ["junk"])])
        analysis = portfolio_analysis(result, cases)
        self.assertFalse(analysis["available"])


class PortfolioMutationGateTests(unittest.TestCase):
    """Tempdir-module mutation gates (the real tree is never rewritten)."""

    def _run_gate(self, module_name, name, old_text, new_text, scenario):
        """Run gate."""
        mod = __import__(f"model_protocol_lab.{module_name}", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c149_mut_", dir="/tmp") as tmpdir:
            mod_path = os.path.join(tmpdir, f"{module_name}_mut_{name}.py")
            with open(mod_path, "w", encoding="utf-8") as handle:
                handle.write(mutated)
            dotted = f"model_protocol_lab.{module_name}_mut_{name}"
            spec = importlib.util.spec_from_file_location(dotted, mod_path)
            assert spec is not None and spec.loader is not None
            mutant = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mutant)
            scenario(mod)  # passes on the real module: non-vacuous
            with self.assertRaises(AssertionError, msg=f"mutant {name} was NOT caught"):
                scenario(mutant)

    @staticmethod
    def _gap_scenario(mod):
        """Gap scenario."""
        passing = mod.validate_passing({"alpha": ("f1", "f2"), "beta": ("f2", "f3")}, FIXTURES)
        rates = mod.oracle_rates(passing, FIXTURES, "alpha")
        # Oracle covers f3 which alpha misses: gap is strictly positive.
        assert rates["gap"] > 0, f"expected positive gap, got {rates['gap']}"

    def test_mutant_inverted_gap_sign_caught(self):
        """Test mutant inverted gap sign caught."""
        self._run_gate(
            "portfolio",
            "inverted_gap",
            "        gap = oracle_rate - winner_rate",
            "        gap = winner_rate - oracle_rate  # MUTANT",
            self._gap_scenario,
        )

    @staticmethod
    def _tie_scenario(mod):
        """Tie scenario."""
        # alpha and beta cover the same fixtures: the lexicographically
        # smallest name must win the tie.
        passing = mod.validate_passing({"beta": ("f1",), "alpha": ("f1",)}, ("f1",))
        cover = mod.greedy_cover(passing, ("f1",))
        assert cover == ["alpha"], f"expected ['alpha'], got {cover}"

    def test_mutant_broken_tie_break_caught(self):
        """Test mutant broken tie break caught."""
        self._run_gate(
            "portfolio",
            "broken_tiebreak",
            """        best = max(
            sorted(passing),
            key=partial(_coverage_size, passing, uncovered),
        )""",
            """        best = max(
            sorted(passing, reverse=True),  # MUTANT
            key=partial(_coverage_size, passing, uncovered),
        )""",
            self._tie_scenario,
        )

    @staticmethod
    def _order_scenario(mod):
        """Order scenario."""
        # Insertion order is beta-first; the oracle must still sort.
        passing = mod.validate_passing({"beta": ("f1",), "alpha": ("f1",)}, ("f1",))
        winners = mod.per_fixture_oracle_winners(passing, ("f1",))
        assert winners == {"f1": ["alpha", "beta"]}, f"expected sorted, got {winners}"

    def test_mutant_unsorted_oracle_winners_caught(self):
        """Test mutant unsorted oracle winners caught."""
        self._run_gate(
            "portfolio",
            "unsorted_winners",
            "        names = sorted(name for name, passed in passing.items() if fixture_id in passed)",  # noqa: E501
            "        names = list(name for name, passed in passing.items() if fixture_id in passed)  # MUTANT",  # noqa: E501
            self._order_scenario,
        )

    @staticmethod
    def _pass_predicate_scenario(mod):
        """Pass predicate scenario."""
        # A fixture with exact_match but no boundary_found is NOT passed:
        # the oracle must see zero solvable fixtures.
        cases = (_case("f1"),)
        result = _result(
            [_entry("alpha", [_obs("alpha", "f1", True, False)])], decision="no_candidate"
        )
        analysis = mod.portfolio_analysis(result, cases)
        assert analysis["oracle"]["n_solvable"] == 0, analysis["oracle"]

    def test_mutant_loosened_pass_predicate_caught(self):
        """Test mutant loosened pass predicate caught."""
        self._run_gate(
            "experiments",
            "loosened_pass",
            '                if obs.get("exact_match") is True and obs.get("boundary_found") is True:',  # noqa: E501
            '                if obs.get("exact_match") is True or obs.get("boundary_found") is True:  # MUTANT',  # noqa: E501
            self._pass_predicate_scenario,
        )


def _portfolio_payload(**overrides):
    """Portfolio payload."""
    payload = {
        "available": True,
        "n_fixtures": 3,
        "n_candidates": 2,
        "decision": "ambiguous",
        "winner": None,
        "excluded_candidates": [],
        "oracle": {
            "n_solvable": 3,
            "solvable_fixture_ids": ["f1", "f2", "f3"],
            "unsolvable_fixture_ids": [],
            "oracle_pass_rate": 1.0,
            "winner_pass_rate": None,
            "gap": 1.0,
            "per_fixture_winners": {"f1": ["alpha"], "f2": ["alpha", "beta"], "f3": ["beta"]},
        },
        "portfolio": {
            "size": 2,
            "members": ["alpha", "beta"],
            "per_member_covers": {"alpha": ["f1", "f2"], "beta": ["f2", "f3"]},
            "covered_fixture_ids": ["f1", "f2", "f3"],
            "uncovered_fixture_ids": [],
            "greedy_note": "greedy set-cover approximation; not guaranteed optimal",
        },
    }
    payload.update(overrides)
    return payload


class DiffPortfolioTests(unittest.TestCase):
    """Diff portfolio tests."""

    def test_self_diff_is_unchanged(self):
        """Test self diff is unchanged."""
        payload = _portfolio_payload()
        diff = diffreports.diff_portfolio(payload, payload)
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_availability_matrix(self):
        """Test availability matrix."""
        payload = _portfolio_payload()
        for old, new, expected in [
            (payload, payload, "both"),
            (payload, None, "old"),
            (None, payload, "new"),
            (None, None, "neither"),
            ({"available": False}, {"available": False}, "neither"),
        ]:
            with self.subTest(expected=expected):
                diff = diffreports.diff_portfolio(old, new)
                self.assertEqual(diff["available"], expected)
                self.assertFalse(diff["changed"])

    def test_hostile_keys_degrade_to_asymmetric(self):
        """Test hostile keys degrade to asymmetric."""
        payload = _portfolio_payload()
        hostile = _portfolio_payload()
        hostile["oracle"]["per_fixture_winners"] = {7: ["alpha"]}
        diff = diffreports.diff_portfolio(payload, hostile)
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])
        # Non-string member keys are likewise rejected, never raised on.
        hostile2 = _portfolio_payload()
        hostile2["portfolio"]["per_member_covers"] = {8.5: ["f1"]}
        diff2 = diffreports.diff_portfolio(payload, hostile2)
        self.assertEqual(diff2["available"], "old")

    def _changed_only(self, mutate_new):
        """Changed only."""
        old = _portfolio_payload()
        new = _portfolio_payload()
        mutate_new(new)
        diff = diffreports.diff_portfolio(old, new)
        self.assertTrue(diff["changed"], "expected the single-cause change to flip changed")

    def test_rates_only_change_pins_disjunct(self):
        """Test rates only change pins disjunct."""

        def mutate(new):
            """Mutate."""
            new["oracle"]["gap"] = 0.5

        self._changed_only(mutate)

    def test_solvable_only_change_pins_disjunct(self):
        """Test solvable only change pins disjunct."""

        def mutate(new):
            """Mutate."""
            new["oracle"]["solvable_fixture_ids"] = ["f1", "f2"]

        self._changed_only(mutate)

    def test_members_only_change_pins_disjunct(self):
        """Test members only change pins disjunct."""

        def mutate(new):
            """Mutate."""
            new["portfolio"]["members"] = ["alpha", "beta", "gamma"]

        self._changed_only(mutate)

    def test_fixture_winner_drift_only_pins_disjunct(self):
        """Test fixture winner drift only pins disjunct."""

        def mutate(new):
            """Mutate."""
            new["oracle"]["per_fixture_winners"]["f1"] = ["beta"]

        self._changed_only(mutate)

    def test_cover_drift_only_pins_disjunct(self):
        """Test cover drift only pins disjunct."""

        def mutate(new):
            """Mutate."""
            new["portfolio"]["per_member_covers"]["alpha"] = ["f1"]

        self._changed_only(mutate)

    def test_diff_reports_summary_flags_feed_identical(self):
        """Test diff reports summary flags feed identical."""
        old = {"decision": "selected", "selected_candidate": "alpha", "search": []}
        new = {"decision": "selected", "selected_candidate": "alpha", "search": []}
        diff = diffreports.diff_reports(old, new)
        self.assertFalse(diff["summary"]["portfolio_changed"])
        self.assertFalse(diff["summary"]["portfolio_asymmetric"])
        self.assertTrue(diff["summary"]["identical"])

        new["portfolio"] = _portfolio_payload()
        diff = diffreports.diff_reports(old, new)
        self.assertFalse(diff["summary"]["portfolio_changed"])
        self.assertTrue(diff["summary"]["portfolio_asymmetric"])
        self.assertFalse(diff["summary"]["identical"])

        old["portfolio"] = _portfolio_payload()
        diff = diffreports.diff_reports(old, new)
        self.assertFalse(diff["summary"]["portfolio_changed"])
        self.assertFalse(diff["summary"]["portfolio_asymmetric"])
        self.assertTrue(diff["summary"]["identical"])

        new["portfolio"]["oracle"]["gap"] = 0.25
        diff = diffreports.diff_reports(old, new)
        self.assertTrue(diff["summary"]["portfolio_changed"])
        self.assertFalse(diff["summary"]["portfolio_asymmetric"])
        self.assertFalse(diff["summary"]["identical"])

    def test_stderr_summary_names_portfolio_axis_when_identical(self):
        """Test stderr summary names portfolio axis when identical."""
        report = {
            "decision": "selected",
            "selected_candidate": "alpha",
            "search": [],
            "portfolio": _portfolio_payload(),
        }
        diff = diffreports.diff_reports(report, dict(report))
        self.assertTrue(diff["summary"]["identical"])
        summary = diffreports.render_summary(diff)
        self.assertIn("protocol-portfolio gap", summary)


class PortfolioHtmlTests(unittest.TestCase):
    """Portfolio html tests."""

    def test_renders_nothing_when_unavailable(self):
        """Test renders nothing when unavailable."""
        self.assertEqual(_portfolio_section({}), "")
        self.assertEqual(_portfolio_section({"portfolio": {"available": False}}), "")

    def test_escapes_hostile_values(self):
        """Test escapes hostile values."""
        payload = _portfolio_payload()
        payload["winner"] = "<script>alert(1)</script>"
        payload["portfolio"]["members"] = ["<img src=x onerror=alert(1)>"]
        payload["oracle"]["per_fixture_winners"] = {"f1": ["<b>bold</b>"]}
        html = _portfolio_section({"portfolio": payload})
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img src=x", html)
        self.assertNotIn("<b>bold</b>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_greedy_disclaimer_rendered(self):
        """Test greedy disclaimer rendered."""
        html = _portfolio_section({"portfolio": _portfolio_payload()})
        self.assertIn("not guaranteed optimal", html)
        self.assertIn('id="protocol-portfolio"', html)

    def test_gap_line_present(self):
        """Test gap line present."""
        html = _portfolio_section({"portfolio": _portfolio_payload()})
        self.assertIn("Oracle pass rate", html)
        self.assertIn("gap", html)


if __name__ == "__main__":
    unittest.main()
