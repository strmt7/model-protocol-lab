"""Cycle 22: differential verification of reporting diagnostics + evidence-exclusion red team.

Part A -- independent oracles for ``selection_margin``, ``token_overhead`` and
``attach_diagnostics`` (model_protocol_lab/reporting.py), written from the
functions' docstrings only: their own arithmetic, no reuse of the
implementation's helpers. The oracles are fuzzed against the implementation
with seeded random result dicts (1-5 candidates, varied pass counts, ties,
empty observations, missing keys, hostile shapes), plus targeted edge cases.
Sensitivity controls mutate the implementation in a copy and prove the tests
catch each mutation.

Part B -- adversarial red-team on the evidence-exclusion policy
(``_public_view`` and the JSON/HTML serialization paths in ``save_report`` /
``render_html``). The security property: with ``include_evidence=False`` (the
default), no raw prompts, generations, or token-id sequences that could
reconstruct user content may appear in the public JSON or HTML; with
``include_evidence=True`` they may. Every field the report builders touch is
probed as a smuggling vector.

Brief-vs-docstring discrepancies found while grounding the oracles:

1. ``selection_margin``'s docstring says a failed/cancelled/uncomputable run
   returns None "so the key stays absent from results.json". In fact
   ``attach_diagnostics`` unconditionally sets
   ``enriched["selection_margin"] = None``, and neither ``_public_view`` nor
   ``save_report`` strips null keys, so results.json actually carries
   ``"selection_margin": null``. The None return is correct; the "key stays
   absent" claim is not. (Pinned by
   ``SelectionMarginOracleTests.test_failed_returns_none_key_present_in_json``.)

2. ``token_overhead``'s docstring says the winner's tax is "null when there is
   no selected winner or no usable counts". The implementation additionally
   returns null when the cheapest mean is exactly 0 (division guard) --
   consistent with "no usable counts" but unstated. (Pinned by
   ``TokenOverheadOracleTests.test_tax_null_when_cheapest_mean_zero``.)

3. The docstrings are silent on ``selected_candidate`` naming a candidate that
   is not among the search runs, and on truthy non-dict ``format`` values
   (the implementation raises ``AttributeError`` on the latter). The oracles
   follow the implementation's observable behavior on both; the fuzz
   generator covers them and asserts exception parity.

Leaks found and fixed (Part B):

- R1 (REAL): ``backend_divergences`` embedded raw generated text (the
  kept-vs-stripped ``detail`` interpolates ``{returned_text!r}``) and the
  decoded expected answer (``stop_texts`` / per-finding ``stop_text``) in the
  public JSON and the HTML, even with ``include_evidence=False``.
- R2 (REAL): ``template_conformance`` divergent segments recorded the
  token-ID subsequences of the rendered prompt (``hand_ids`` /
  ``template_ids``) -- decoding them reconstructs the prompt -- in the public
  JSON and the HTML, even with ``include_evidence=False``.

Fix (minimal, in reporting.py): ``_public_view`` now also scrubs those
fields when ``include_evidence=False`` (new ``_strip_raw_diagnostics``);
``render_html`` no longer embeds the token-ID lists, stop texts, or free-text
divergence details in the HTML (the HTML never carried raw evidence per
docs/USAGE.md, so the redaction is unconditional there). The
``include_evidence=True`` JSON path is byte-identical to before (pinned by
``EvidenceExclusionTests.test_evidence_path_is_unchanged``).

By-design (documented, not fixed):

- ``boundary_token_ids`` / ``selected_boundary_token_ids`` stay public: they
  are operator-specified run configuration (the required ``--boundary-ids``
  argument), not recorded raw evidence; ``diff-reports`` consumes them from
  public results.json for boundary-drift detection, and the HTML config
  shows them deliberately. Residual: a multi-token boundary decodes to the
  operator's own marker text -- keep boundaries inert markers.
- Candidate names, fixture IDs, format markers, failure codes and taxonomy
  mode names are operator/fixture identifiers rendered HTML-escaped
  (injection-safe); they are not evidence records and the report cannot name
  its subjects without them.
- Fingerprints (``candidate_fingerprint``, ``backend_fingerprint``) are
  one-way sha256 digests: not reversible to content.
- Observation-level fields (``input_ids``, ``output_ids``, ``answer``,
  ``error``) were already stripped from the public JSON and never rendered
  in HTML; the logprob interpretation text and all other diagnostic text
  are fixed-vocabulary by construction.
"""

import copy
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path

from model_protocol_lab import reporting
from model_protocol_lab.reporting import (
    _WITHHELD_DIAGNOSTIC_DETAIL,
    _public_view,
    attach_diagnostics,
    render_html,
    save_report,
    selection_margin,
    token_overhead,
)

# ---------------------------------------------------------------------------
# Part A: independent oracles, written from the docstrings only.
# ---------------------------------------------------------------------------


def _oracle_search_outcomes(result):
    """Per-candidate, per-fixture outcomes, mirroring the documented data flow.

    Candidate names come from each search run's format mapping; an outcome is
    True only for a completed observation whose exact_match and
    boundary_found are both booleans and both true, False for completed but
    not passing, None otherwise. First-seen order is kept for names and
    fixture IDs; duplicate names/fixtures keep the first record.
    """
    names = []
    order = []
    per_candidate = {}
    for run in result.get("search") or []:
        if type(run) is not dict:
            continue
        # Same observable behavior as the implementation, including the
        # AttributeError on truthy non-dict format values (see module
        # docstring, discrepancy 3).
        fmt = run.get("format") or {}
        name = fmt.get("name")
        if type(name) is not str or not name or name in per_candidate:
            continue
        names.append(name)
        outcomes = {}
        for obs in run.get("observations") or []:
            if type(obs) is not dict:
                continue
            fid = obs.get("fixture")
            if type(fid) is not str or not fid or fid in outcomes:
                continue
            if (
                obs.get("status") == "completed"
                and type(obs.get("exact_match")) is bool
                and type(obs.get("boundary_found")) is bool
            ):
                outcomes[fid] = bool(obs.get("exact_match") and obs.get("boundary_found"))
            else:
                outcomes[fid] = None
            if fid not in order:
                order.append(fid)
        per_candidate[name] = outcomes
    return names, order, per_candidate


def oracle_selection_margin(result):
    """Independent oracle for selection_margin, from its docstring.

    Margin in fixture flips: unique winner -> smallest runner-up failing
    count (direction "to_tie"); no_candidate -> smallest failing count over
    all candidates ("to_winner"); ambiguous -> (#all-pass - 1) ("to_unique").
    failed/cancelled/uncomputable -> None. Runner-up ties break to the
    lexicographically smallest name; fixture lists are sorted.
    """
    if type(result) is not dict:
        raise ValueError("result must be a dict")
    if str(result.get("decision", "")) in ("failed", "cancelled"):
        return None
    names, order, per_candidate = _oracle_search_outcomes(result)
    if not names or not order:
        return None
    failing = {
        name: sorted(fid for fid in order if per_candidate[name].get(fid) is not True)
        for name in names
    }
    selected = result.get("selected_candidate")
    if type(selected) is str and selected:
        runners = [name for name in names if name != selected]
        if not runners:
            return {
                "value": None,
                "direction": "to_tie",
                "runner_up": None,
                "decisive_fixtures": [],
            }
        best = min(runners, key=lambda name: (len(failing[name]), name))
        return {
            "value": len(failing[best]),
            "direction": "to_tie",
            "runner_up": best,
            "decisive_fixtures": failing[best],
        }
    decision = str(result.get("decision", ""))
    if decision == "ambiguous":
        tied = [name for name in names if not failing[name]]
        if len(tied) < 2:
            return None
        shared = sorted(
            fid for fid in order if all(per_candidate[name].get(fid) is True for name in tied)
        )
        return {
            "value": len(tied) - 1,
            "direction": "to_unique",
            "runner_up": None,
            "decisive_fixtures": shared,
        }
    if decision == "no_candidate":
        best = min(names, key=lambda name: (len(failing[name]), name))
        return {
            "value": len(failing[best]),
            "direction": "to_winner",
            "runner_up": best,
            "decisive_fixtures": failing[best],
        }
    return None


def oracle_token_overhead(result):
    """Independent oracle for token_overhead, from its docstring.

    Per-candidate mean/max prompt tokens and mean output tokens over
    completed observations with list/tuple id sequences; null stats when a
    candidate has none. Cheapest = smallest mean prompt count, ties to the
    lexicographically smallest name. Winner's tax vs cheapest in percent,
    0.0 when the winner is the cheapest, null without a usable winner or
    usable counts (including a zero cheapest mean).
    """
    if type(result) is not dict:
        raise ValueError("result must be a dict")
    per_candidate = {}
    for run in result.get("search") or []:
        if type(run) is not dict:
            continue
        fmt = run.get("format")
        name = fmt.get("name") if type(fmt) is dict else None
        if type(name) is not str or not name or name in per_candidate:
            continue
        prompt_counts = []
        output_counts = []
        for obs in run.get("observations") or []:
            if type(obs) is not dict or obs.get("status") != "completed":
                continue
            input_ids = obs.get("input_ids")
            output_ids = obs.get("output_ids")
            if not isinstance(input_ids, (list, tuple)) or not isinstance(
                output_ids, (list, tuple)
            ):
                continue
            prompt_counts.append(len(input_ids))
            output_counts.append(len(output_ids))
        if prompt_counts:
            per_candidate[name] = {
                "mean_prompt_tokens": sum(prompt_counts) / len(prompt_counts),
                "max_prompt_tokens": max(prompt_counts),
                "mean_output_tokens": sum(output_counts) / len(output_counts),
            }
        else:
            per_candidate[name] = {
                "mean_prompt_tokens": None,
                "max_prompt_tokens": None,
                "mean_output_tokens": None,
            }
    cheapest = None
    cheapest_mean = None
    for name in sorted(per_candidate):
        mean = per_candidate[name]["mean_prompt_tokens"]
        if type(mean) is not float:
            continue
        if cheapest is None or (mean, name) < (cheapest_mean, cheapest):
            cheapest, cheapest_mean = name, mean
    winner_mean = None
    winner = result.get("selected_candidate")
    if type(winner) is str and winner in per_candidate:
        candidate_mean = per_candidate[winner]["mean_prompt_tokens"]
        if type(candidate_mean) is float:
            winner_mean = candidate_mean
    tax = None
    if cheapest is not None and winner_mean is not None and cheapest_mean > 0:
        tax = (winner_mean - cheapest_mean) / cheapest_mean * 100.0
    return {
        "per_candidate": {name: per_candidate[name] for name in sorted(per_candidate)},
        "cheapest_candidate": cheapest,
        "winner_tax_vs_cheapest_pct": tax,
    }


def oracle_attach_diagnostics(result):
    """Independent oracle for attach_diagnostics: a copy, never a mutation."""
    if type(result) is not dict:
        raise ValueError("result must be a dict")
    enriched = dict(result)
    enriched["selection_margin"] = oracle_selection_margin(result)
    enriched["token_overhead"] = oracle_token_overhead(result)
    return enriched


def _call(fn, result):
    """Call fn, capturing the outcome or the exception identity for parity."""
    try:
        return ("ok", fn(result))
    except Exception as exc:  # noqa: BLE001 -- parity check needs the exception identity
        return ("raised", type(exc).__name__, str(exc))


# ---------------------------------------------------------------------------
# Minimal result-dict builders (ScriptedBackend-style: dicts only, no model).
# ---------------------------------------------------------------------------


def _obs(
    fixture,
    *,
    passed=None,
    status="completed",
    input_len=0,
    output_len=0,
    drop_keys=(),
):
    """One observation dict. ``passed`` sets exact_match/boundary_found."""
    obs = {"fixture": fixture, "status": status}
    if passed is not None:
        obs["exact_match"] = passed
        obs["boundary_found"] = passed
    obs["input_ids"] = list(range(input_len))
    obs["output_ids"] = list(range(1000, 1000 + output_len))
    for key in drop_keys:
        obs.pop(key, None)
    return obs


def _run(name, observations, *, summary=None):
    run = {"format": {"name": name}, "observations": observations}
    if summary is not None:
        run["summary"] = summary
    return run


def _result(runs, *, decision="compatible_on_fixture_set", selected=None, extra=None):
    result = {"decision": decision, "search": runs}
    if selected is not None:
        result["selected_candidate"] = selected
    if extra:
        result.update(extra)
    return result


def _rand_result(rng):
    """Seeded hostile result dict: 1-5 candidates, ties, gaps, bad shapes."""
    names = rng.sample(["cand-a", "cand-b", "cand-c", "cand-d", "cand-e"], rng.randint(1, 5))
    fixtures = [f"f{i}" for i in range(rng.randint(0, 4))]
    result = {}
    if rng.random() < 0.9:
        result["decision"] = rng.choice(
            [
                "compatible_on_fixture_set",
                "ambiguous",
                "no_candidate",
                "failed",
                "cancelled",
                "bogus-decision",
            ]
        )
    if rng.random() < 0.7:
        result["selected_candidate"] = rng.choice(names + ["", "ghost", None, 42])
    if rng.random() < 0.92:
        search = []
        for name in names:
            if rng.random() < 0.04:
                search.append("not-a-dict-run")
                continue
            run = {}
            shape = rng.random()
            if shape < 0.9:
                run["format"] = {"name": name}
            elif shape < 0.94:
                run["format"] = {}  # missing name
            elif shape < 0.97:
                run["format"] = "truthy-non-dict-format"  # impl raises here
            # else: missing format entirely
            observations = []
            if rng.random() < 0.9:
                for fid in fixtures:
                    if rng.random() < 0.12:
                        continue  # missing observation
                    if rng.random() < 0.05:
                        observations.append({"fixture": fid})  # bare observation
                        continue
                    obs = {"fixture": fid}
                    outcome = rng.random()
                    if outcome < 0.55:
                        obs["status"] = "completed"
                        obs["exact_match"] = rng.random() < 0.5
                        obs["boundary_found"] = rng.random() < 0.5
                        if rng.random() < 0.06:
                            obs["exact_match"] = "yes"  # non-bool flag
                    elif outcome < 0.75:
                        obs["status"] = rng.choice(["failed", "cancelled"])
                    # else: missing status -> incomplete
                    pick = rng.random()
                    if pick < 0.8:
                        obs["input_ids"] = [rng.randint(0, 999) for _ in range(rng.randint(0, 6))]
                        obs["output_ids"] = [rng.randint(0, 999) for _ in range(rng.randint(0, 6))]
                    elif pick < 0.86:
                        obs["input_ids"] = []  # empty list still counts: 0 tokens
                        obs["output_ids"] = []
                    elif pick < 0.92:
                        obs["input_ids"] = "not-a-list"  # skipped by impl
                        obs["output_ids"] = [1, 2]
                    # else: missing id keys
                    observations.append(obs)
                if rng.random() < 0.05:
                    observations.append("not-a-dict-obs")
            elif rng.random() < 0.5:
                observations = {"not": "a-list"}  # hostile shape, still iterable
            run["observations"] = observations
            search.append(run)
        if rng.random() < 0.06:
            # duplicate candidate name: first record wins
            search.append(_run(names[0], [_obs("f9", passed=True)]))
        result["search"] = search
    elif rng.random() < 0.5:
        result["search"] = None
    # else: missing search key
    return result


# ---------------------------------------------------------------------------
# Part A: differential tests.
# ---------------------------------------------------------------------------


class SelectionMarginOracleTests(unittest.TestCase):
    def test_rejects_non_dict(self):
        for bad in ([], "x", None, 42):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                selection_margin(bad)

    def test_failed_and_cancelled_return_none(self):
        result = _result(
            [_run("a", [_obs("f1", passed=True)])],
            decision="failed",
            selected="a",
        )
        self.assertIsNone(selection_margin(result))
        result["decision"] = "cancelled"
        self.assertIsNone(selection_margin(result))

    def test_failed_returns_none_key_present_in_json(self):
        # Docstring discrepancy 1: the docstring claims the key "stays absent
        # from results.json", but attach_diagnostics always sets it, so the
        # serialized report carries "selection_margin": null.
        enriched = attach_diagnostics(
            _result([_run("a", [_obs("f1", passed=True)])], decision="failed")
        )
        self.assertIsNone(enriched["selection_margin"])
        text = json.dumps(_public_view(enriched, include_evidence=False))
        self.assertIn('"selection_margin": null', text)

    def test_unique_winner_runner_up_tie_breaks_lexicographically(self):
        runs = [
            _run("cand-a", [_obs("f1", passed=True), _obs("f2", passed=True)]),
            _run("cand-b", [_obs("f1", passed=False), _obs("f2", passed=True)]),
            _run("cand-c", [_obs("f1", passed=True), _obs("f2", passed=False)]),
        ]
        margin = selection_margin(_result(runs, selected="cand-a"))
        self.assertEqual(
            margin,
            {
                "value": 1,
                "direction": "to_tie",
                "runner_up": "cand-b",
                "decisive_fixtures": ["f1"],
            },
        )

    def test_single_candidate_has_null_value(self):
        margin = selection_margin(
            _result([_run("solo", [_obs("f1", passed=True)])], selected="solo")
        )
        self.assertEqual(
            margin,
            {
                "value": None,
                "direction": "to_tie",
                "runner_up": None,
                "decisive_fixtures": [],
            },
        )

    def test_ambiguous_counts_all_pass_minus_one(self):
        runs = [
            _run("a", [_obs("f1", passed=True), _obs("f2", passed=True)]),
            _run("b", [_obs("f1", passed=True), _obs("f2", passed=True)]),
            _run("c", [_obs("f1", passed=True), _obs("f2", passed=True)]),
            _run("d", [_obs("f1", passed=False), _obs("f2", passed=True)]),
        ]
        margin = selection_margin(_result(runs, decision="ambiguous"))
        self.assertEqual(
            margin,
            {
                "value": 2,
                "direction": "to_unique",
                "runner_up": None,
                "decisive_fixtures": ["f1", "f2"],
            },
        )

    def test_ambiguous_with_fewer_than_two_tied_returns_none(self):
        runs = [
            _run("a", [_obs("f1", passed=True)]),
            _run("b", [_obs("f1", passed=False)]),
        ]
        self.assertIsNone(selection_margin(_result(runs, decision="ambiguous")))

    def test_no_candidate_picks_closest(self):
        runs = [
            _run("a", [_obs("f1", passed=False), _obs("f2", passed=False)]),
            _run("b", [_obs("f1", passed=False), _obs("f2", passed=True)]),
        ]
        margin = selection_margin(_result(runs, decision="no_candidate"))
        self.assertEqual(
            margin,
            {
                "value": 1,
                "direction": "to_winner",
                "runner_up": "b",
                "decisive_fixtures": ["f1"],
            },
        )

    def test_empty_search_returns_none(self):
        self.assertIsNone(selection_margin(_result([], selected="a")))
        self.assertIsNone(selection_margin({"decision": "no_candidate"}))

    def test_incomplete_observations_count_as_failing(self):
        runs = [
            _run("a", [_obs("f1", passed=True)]),
            _run("b", [_obs("f1", status="failed")]),  # incomplete -> failing
        ]
        margin = selection_margin(_result(runs, selected="a"))
        self.assertEqual(margin["value"], 1)
        self.assertEqual(margin["decisive_fixtures"], ["f1"])

    def test_duplicate_candidate_name_keeps_first(self):
        runs = [
            _run("a", [_obs("f1", passed=True)]),
            _run("a", [_obs("f1", passed=False)]),
            _run("b", [_obs("f1", passed=False)]),
        ]
        margin = selection_margin(_result(runs, selected="a"))
        self.assertEqual(margin["runner_up"], "b")

    def test_selected_unknown_name_treats_all_as_runners(self):
        runs = [
            _run("a", [_obs("f1", passed=False)]),
            _run("b", [_obs("f1", passed=True)]),
        ]
        margin = selection_margin(_result(runs, selected="ghost"))
        self.assertEqual(margin["runner_up"], "b")
        self.assertEqual(margin["value"], 0)

    def test_fuzz_against_oracle(self):
        for seed in range(250):
            rng = random.Random(seed)  # noqa: S311 -- deterministic test seed
            result = _rand_result(rng)
            with self.subTest(seed=seed):
                self.assertEqual(
                    _call(selection_margin, result),
                    _call(oracle_selection_margin, result),
                )


class TokenOverheadOracleTests(unittest.TestCase):
    def test_rejects_non_dict(self):
        for bad in ([], "x", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                token_overhead(bad)

    def test_means_over_completed_only(self):
        runs = [
            _run(
                "a",
                [
                    _obs("f1", passed=True, input_len=10, output_len=4),
                    _obs("f2", passed=True, input_len=20, output_len=6),
                    _obs("f3", status="failed", input_len=999, output_len=999),
                ],
            ),
        ]
        overhead = token_overhead(_result(runs))
        self.assertEqual(
            overhead["per_candidate"]["a"],
            {
                "mean_prompt_tokens": 15.0,
                "max_prompt_tokens": 20,
                "mean_output_tokens": 5.0,
            },
        )

    def test_empty_id_lists_count_as_zero_tokens(self):
        runs = [_run("a", [_obs("f1", passed=True, input_len=0, output_len=0)])]
        stats = token_overhead(_result(runs))["per_candidate"]["a"]
        self.assertEqual(stats["mean_prompt_tokens"], 0.0)
        self.assertEqual(stats["max_prompt_tokens"], 0)

    def test_non_list_ids_are_skipped(self):
        obs = _obs("f1", passed=True, input_len=5, output_len=5)
        obs["input_ids"] = "not-a-list"
        runs = [_run("a", [obs])]
        stats = token_overhead(_result(runs))["per_candidate"]["a"]
        self.assertIsNone(stats["mean_prompt_tokens"])

    def test_candidate_without_completed_gets_null_stats(self):
        runs = [_run("a", [_obs("f1", status="failed", input_len=5)])]
        stats = token_overhead(_result(runs))["per_candidate"]["a"]
        self.assertEqual(
            stats,
            {
                "mean_prompt_tokens": None,
                "max_prompt_tokens": None,
                "mean_output_tokens": None,
            },
        )

    def test_cheapest_tie_breaks_lexicographically(self):
        runs = [
            _run("b", [_obs("f1", passed=True, input_len=10)]),
            _run("a", [_obs("f1", passed=True, input_len=10)]),
        ]
        overhead = token_overhead(_result(runs))
        self.assertEqual(overhead["cheapest_candidate"], "a")
        self.assertEqual(list(overhead["per_candidate"]), ["a", "b"])  # output sorted by name

    def test_winner_tax_zero_when_winner_is_cheapest(self):
        runs = [
            _run("a", [_obs("f1", passed=True, input_len=10)]),
            _run("b", [_obs("f1", passed=True, input_len=30)]),
        ]
        overhead = token_overhead(_result(runs, selected="a"))
        self.assertEqual(overhead["winner_tax_vs_cheapest_pct"], 0.0)

    def test_winner_tax_positive_when_winner_costlier(self):
        runs = [
            _run("a", [_obs("f1", passed=True, input_len=10)]),
            _run("b", [_obs("f1", passed=True, input_len=30)]),
        ]
        overhead = token_overhead(_result(runs, selected="b"))
        self.assertAlmostEqual(overhead["winner_tax_vs_cheapest_pct"], 200.0)

    def test_tax_null_without_winner_or_counts(self):
        runs = [_run("a", [_obs("f1", passed=True, input_len=10)])]
        self.assertIsNone(token_overhead(_result(runs))["winner_tax_vs_cheapest_pct"])
        self.assertIsNone(
            token_overhead(_result(runs, selected="ghost"))["winner_tax_vs_cheapest_pct"]
        )
        # Winner selected but with no usable counts.
        runs = [
            _run("a", [_obs("f1", passed=True, input_len=10)]),
            _run("b", [_obs("f1", status="failed")]),
        ]
        self.assertIsNone(token_overhead(_result(runs, selected="b"))["winner_tax_vs_cheapest_pct"])

    def test_tax_null_when_cheapest_mean_zero(self):
        # Docstring discrepancy 2: unstated division guard.
        runs = [_run("a", [_obs("f1", passed=True, input_len=0)])]
        overhead = token_overhead(_result(runs, selected="a"))
        self.assertEqual(overhead["cheapest_candidate"], "a")
        self.assertIsNone(overhead["winner_tax_vs_cheapest_pct"])

    def test_fuzz_against_oracle(self):
        for seed in range(250, 500):
            rng = random.Random(seed)  # noqa: S311 -- deterministic test seed
            result = _rand_result(rng)
            with self.subTest(seed=seed):
                self.assertEqual(
                    _call(token_overhead, result),
                    _call(oracle_token_overhead, result),
                )


class AttachDiagnosticsTests(unittest.TestCase):
    def test_rejects_non_dict(self):
        with self.assertRaises(ValueError):
            attach_diagnostics([])

    def test_returns_copy_with_both_diagnostics(self):
        result = _result(
            [_run("a", [_obs("f1", passed=True, input_len=8, output_len=2)])],
            selected="a",
        )
        enriched = attach_diagnostics(result)
        self.assertIsNot(enriched, result)
        self.assertEqual(enriched["selection_margin"], oracle_selection_margin(result))
        self.assertEqual(enriched["token_overhead"], oracle_token_overhead(result))
        self.assertEqual(set(enriched), set(result) | {"selection_margin", "token_overhead"})

    def test_input_is_not_mutated(self):
        result = _result([_run("a", [_obs("f1", passed=True, input_len=8)])], selected="a")
        before = copy.deepcopy(result)
        attach_diagnostics(result)
        self.assertEqual(result, before)

    def test_fuzz_matches_oracle_pair(self):
        for seed in range(500, 650):
            rng = random.Random(seed)  # noqa: S311 -- deterministic test seed
            result = _rand_result(rng)
            with self.subTest(seed=seed):
                got = _call(attach_diagnostics, result)
                want = _call(oracle_attach_diagnostics, result)
                self.assertEqual(got, want)
                if got[0] == "ok":
                    # The oracle pair also pins non-mutation on every fuzz case.
                    self.assertNotIn("selection_margin", result)
                    self.assertNotIn("token_overhead", result)


# ---------------------------------------------------------------------------
# Sensitivity controls: prove the tests catch implementation mutations.
# ---------------------------------------------------------------------------


def _load_mutant(name, replacements):
    """Load a mutated copy of reporting.py from source text."""
    source = Path(reporting.__file__).read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in source:
            raise AssertionError(f"mutant {name}: anchor not found: {old[:70]!r}")
        source = source.replace(old, new, 1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"mut_reporting_{name}.py"
        path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(f"mut_reporting_{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def _prefix_public_view(result, include_evidence):
    """The pre-fix _public_view: stripped observations only (R1/R2 present)."""
    clean = json.loads(json.dumps(result, ensure_ascii=True, allow_nan=False))
    if not include_evidence:
        for run in clean.get("search", []):
            run.pop("observations", None)
        if clean.get("held_out") is not None:
            clean["held_out"].pop("observations", None)
    clean["raw_evidence_included"] = include_evidence
    return clean


class SensitivityTests(unittest.TestCase):
    def test_margin_direction_mutation_is_caught(self):
        mutant = _load_mutant(
            "margin_direction",
            [
                (
                    '"value": len(failing[best]),\n            "direction": "to_tie",',
                    '"value": len(failing[best]),\n            "direction": "to_draw",',
                )
            ],
        )
        case = _result(
            [
                _run("a", [_obs("f1", passed=True)]),
                _run("b", [_obs("f1", passed=False)]),
            ],
            selected="a",
        )
        self.assertNotEqual(mutant.selection_margin(case), oracle_selection_margin(case))

    def test_margin_tiebreak_mutation_is_caught(self):
        mutant = _load_mutant(
            "margin_tiebreak",
            [
                (
                    "best = min(runners, key=lambda name: (len(failing[name]), name))",
                    'best = min(runners, key=lambda name: (len(failing[name]), ""))',
                )
            ],
        )
        # Search order puts cand-b first; both runners fail one fixture, so
        # the oracle picks the lexicographically smallest ("cand-a") while the
        # mutant picks the first in search order ("cand-b").
        case = _result(
            [
                _run("cand-b", [_obs("f1", passed=False)]),
                _run("cand-a", [_obs("f1", passed=False)]),
                _run("winner", [_obs("f1", passed=True)]),
            ],
            selected="winner",
        )
        self.assertEqual(oracle_selection_margin(case)["runner_up"], "cand-a")
        self.assertNotEqual(mutant.selection_margin(case), oracle_selection_margin(case))

    def test_overhead_tax_sign_mutation_is_caught(self):
        mutant = _load_mutant(
            "tax_sign",
            [
                (
                    "tax = (winner_mean - cheapest_mean) / cheapest_mean * 100.0",
                    "tax = (cheapest_mean - winner_mean) / cheapest_mean * 100.0",
                )
            ],
        )
        case = _result(
            [
                _run("a", [_obs("f1", passed=True, input_len=10)]),
                _run("b", [_obs("f1", passed=True, input_len=30)]),
            ],
            selected="b",
        )
        self.assertAlmostEqual(oracle_token_overhead(case)["winner_tax_vs_cheapest_pct"], 200.0)
        self.assertNotEqual(mutant.token_overhead(case), oracle_token_overhead(case))

    def test_attach_diagnostics_input_mutation_is_caught(self):
        mutant = _load_mutant(
            "attach_mutates",
            [("enriched = dict(result)", "enriched = result")],
        )
        case = _result([_run("a", [_obs("f1", passed=True, input_len=8)])], selected="a")
        before = copy.deepcopy(case)
        mutant.attach_diagnostics(case)
        # The mutant wrote the diagnostics into the caller's dict.
        self.assertNotEqual(case, before)

    def test_prefix_public_view_still_leaks(self):
        # The Part B regression tests must fail against the pre-fix
        # _public_view: proves they are sensitive to the R1/R2 leaks.
        view = _prefix_public_view(_hostile_result(), include_evidence=False)
        text = json.dumps(view, ensure_ascii=True)
        self.assertIn(_HOSTILE_GENERATION, text)
        self.assertIn(str(_HOSTILE_IDS), text)
        finding = view["backend_divergences"]["per_fixture"][0]["divergences"][0]
        self.assertIn("stop_text", finding)
        segment = view["template_conformance"]["diffs"][0]["diff"][0]
        self.assertIn("hand_ids", segment)


# ---------------------------------------------------------------------------
# Part B: adversarial red-team on the evidence-exclusion policy.
# ---------------------------------------------------------------------------

_HOSTILE_PROMPT = "PROMPT-SMUGGLER-7f3a the user's secret prompt text"
_HOSTILE_GENERATION = "GENERATION-SMUGGLER-9c1e the model's secret output text"
_HOSTILE_IDS = [101, 202, 303, 404]


def _hostile_result():
    """Result dict with hostile content planted in every builder-touched field."""
    return {
        "decision": "compatible_on_fixture_set",
        "selection_rule": "test-rule",
        "selected_candidate": "cand-a",
        "search": [
            {
                "format": {"name": "cand-a", "bos": "<|HOSTILE-MARKER|>"},
                "candidate_fingerprint": "sha256:" + "ab" * 32,
                "boundary_token_ids": [7],
                "summary": {
                    "status": "completed",
                    "planned": 1,
                    "attempted": 1,
                    "aggregate_eligible": True,
                    "exact_and_boundary_passes": 1,
                    "rate": 1.0,
                },
                "observations": [
                    {
                        "candidate": "cand-a",
                        "fixture": "f1",
                        "input_ids": list(_HOSTILE_IDS),
                        "output_ids": [1, 2, 3],
                        "answer": _HOSTILE_GENERATION,
                        "exact_match": True,
                        "boundary_found": True,
                        "status": "completed",
                        "error": _HOSTILE_PROMPT,
                    }
                ],
            }
        ],
        "held_out": {
            "summary": {
                "status": "completed",
                "planned": 1,
                "attempted": 1,
                "aggregate_eligible": True,
                "exact_and_boundary_passes": 1,
                "rate": 1.0,
            },
            "observations": [],
        },
        "template_conformance": {
            "status": "ok",
            "diffs": [
                {
                    "status": "ok",
                    "format": "cand-a",
                    "fixture": "f1",
                    "hand_tokens": 4,
                    "template_tokens": 4,
                    "diff": [
                        {
                            "hand_span": [0, 4],
                            "hand_ids": list(_HOSTILE_IDS),
                            "template_span": [0, 4],
                            "template_ids": list(_HOSTILE_IDS),
                        }
                    ],
                }
            ],
        },
        "backend_divergences": {
            "verdict": "diverge",
            "pairs_verdict": ("HF vs vLLM vs SGLang stop semantics: 0/1 agree; 1 divergence"),
            "per_fixture": [
                {
                    "fixture": "f1",
                    "verdict": "diverge",
                    "divergences": [
                        {
                            "class": "KEPT_VS_STRIPPED",
                            "stop_text": _HOSTILE_GENERATION,
                            "detail": (
                                "vllm_holdback_strip returned "
                                f"{_HOSTILE_GENERATION!r} while hf_stop_string "
                                f"kept {_HOSTILE_GENERATION!r}"
                            ),
                            "severity": "high",
                        }
                    ],
                }
            ],
            "matrix": {
                "KEPT_VS_STRIPPED": {
                    "hf_stop_string": 1,
                    "vllm_holdback_strip": 1,
                    "vllm_holdback_keep": 0,
                    "sglang_stop": 0,
                    "eos_appended": 0,
                }
            },
            "severities": {"KEPT_VS_STRIPPED": "high"},
            "stop_texts": [_HOSTILE_GENERATION],
            "eos_id": None,
            "min_tokens": 0,
            "observations_truncated": False,
            "observations_total": 1,
            "observations_processed": 1,
        },
        "failure_taxonomy": {
            "per_candidate": {"cand-a": {"stop_string_cut": 2}},
            "held_out": {},
        },
    }


def _public_json_text(result, include_evidence):
    return json.dumps(_public_view(result, include_evidence), ensure_ascii=True)


class EvidenceExclusionTests(unittest.TestCase):
    def test_public_json_strips_raw_diagnostics_without_evidence(self):
        text = _public_json_text(_hostile_result(), include_evidence=False)
        self.assertNotIn(_HOSTILE_PROMPT, text)
        self.assertNotIn(_HOSTILE_GENERATION, text)
        self.assertNotIn(str(_HOSTILE_IDS), text)
        # Structural pins for the R1/R2 fixes.
        view = json.loads(text)
        segment = view["template_conformance"]["diffs"][0]["diff"][0]
        self.assertNotIn("hand_ids", segment)
        self.assertNotIn("template_ids", segment)
        self.assertEqual(segment["hand_span"], [0, 4])  # spans stay
        divergences = view["backend_divergences"]
        self.assertNotIn("stop_texts", divergences)
        finding = divergences["per_fixture"][0]["divergences"][0]
        self.assertNotIn("stop_text", finding)
        self.assertEqual(finding["detail"], _WITHHELD_DIAGNOSTIC_DETAIL)
        self.assertEqual(finding["class"], "KEPT_VS_STRIPPED")  # signal stays
        self.assertFalse(view["raw_evidence_included"])

    def test_public_json_keeps_everything_with_evidence(self):
        text = _public_json_text(_hostile_result(), include_evidence=True)
        self.assertIn(_HOSTILE_PROMPT, text)  # observation error field
        self.assertIn(_HOSTILE_GENERATION, text)
        self.assertIn(str(_HOSTILE_IDS), text)
        view = json.loads(text)
        segment = view["template_conformance"]["diffs"][0]["diff"][0]
        self.assertEqual(segment["hand_ids"], _HOSTILE_IDS)
        finding = view["backend_divergences"]["per_fixture"][0]["divergences"][0]
        self.assertEqual(finding["stop_text"], _HOSTILE_GENERATION)
        self.assertIn(_HOSTILE_GENERATION, finding["detail"])
        self.assertTrue(view["raw_evidence_included"])

    def test_evidence_path_is_unchanged(self):
        # The include_evidence=True JSON must be byte-identical to the
        # pre-fix behavior: the fix only touches the excluded path.
        result = _hostile_result()
        self.assertEqual(
            _public_json_text(result, include_evidence=True),
            json.dumps(_prefix_public_view(result, True), ensure_ascii=True),
        )

    def test_html_never_embeds_raw_diagnostics(self):
        html = render_html(_hostile_result())
        self.assertNotIn(_HOSTILE_PROMPT, html)
        self.assertNotIn(_HOSTILE_GENERATION, html)
        self.assertNotIn(str(_HOSTILE_IDS), html)
        # The diagnostic signal stays; the raw carriers are withheld.
        self.assertIn("KEPT_VS_STRIPPED", html)
        self.assertIn("divergent segment", html)
        self.assertIn("withheld from this HTML view", html)
        # Observation-level raw records were already never rendered: the
        # only "observations" mention is the held-out exclusion note.
        self.assertIn("Held-out observations are excluded from this HTML view", html)

    def test_save_report_end_to_end(self):
        for include_evidence in (False, True):
            with (
                self.subTest(include_evidence=include_evidence),
                tempfile.TemporaryDirectory() as tmp,
            ):
                json_path, html_path = save_report(
                    _hostile_result(),
                    Path(tmp) / "report",
                    include_evidence=include_evidence,
                )
                json_text = json_path.read_text(encoding="utf-8")
                html_text = html_path.read_text(encoding="utf-8")
                if include_evidence:
                    self.assertIn(_HOSTILE_GENERATION, json_text)
                else:
                    self.assertNotIn(_HOSTILE_PROMPT, json_text)
                    self.assertNotIn(_HOSTILE_GENERATION, json_text)
                    self.assertNotIn(str(_HOSTILE_IDS), json_text)
                # The HTML never carries raw evidence, either way.
                self.assertNotIn(_HOSTILE_PROMPT, html_text)
                self.assertNotIn(_HOSTILE_GENERATION, html_text)
                self.assertNotIn(str(_HOSTILE_IDS), html_text)

    def test_smuggled_candidate_name_is_escaped_not_leaked(self):
        # By-design: candidate names are report identifiers, not evidence;
        # hostile markup in them must be neutralized, not smuggled.
        result = _hostile_result()
        result["search"][0]["format"]["name"] = "cand-<script>alert(1)</script>"
        html = render_html(result)
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        text = _public_json_text(result, include_evidence=False)
        self.assertEqual(
            json.loads(text)["search"][0]["format"]["name"],
            "cand-<script>alert(1)</script>",
        )

    def test_smuggled_fixture_id_is_escaped_not_leaked(self):
        result = _hostile_result()
        result["search"][0]["observations"][0]["fixture"] = "f1<img src=x onerror=y>"
        html = render_html(result)
        self.assertNotIn("<img src=x onerror=y>", html)
        self.assertIn("&lt;img src=x onerror=y&gt;", html)

    def test_hostile_error_code_is_escaped_and_guidance_is_fixed(self):
        # By-design: failure codes render escaped; unknown codes get the
        # fixed generic guidance, never raw backend text.
        result = _hostile_result()
        result["decision"] = "failed"
        result["search"][0]["summary"]["failure_code"] = "x<script>y</script>"
        html = render_html(result)
        self.assertNotIn("<script>y</script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("Failure diagnosis:", html)

    def test_hostile_taxonomy_mode_is_escaped(self):
        result = _hostile_result()
        result["decision"] = "held_out_failed"
        result["failure_taxonomy"] = {"per_candidate": {"cand-a": {"<b>evil</b>": 3}}}
        html = render_html(result)
        self.assertNotIn("<b>evil</b>", html)
        self.assertIn("&lt;b&gt;evil&lt;/b&gt;", html)

    def test_format_markers_are_config_not_evidence(self):
        # By-design: format markers travel in the public JSON (operator
        # config) but are never rendered into the HTML.
        text = _public_json_text(_hostile_result(), include_evidence=False)
        self.assertIn("<|HOSTILE-MARKER|>", text)
        html = render_html(_hostile_result())
        self.assertNotIn("<|HOSTILE-MARKER|>", html)

    def test_fingerprints_are_one_way_hashes(self):
        # By-design: sha256 digests identify candidates/backends without
        # carrying reconstructable content.
        text = _public_json_text(_hostile_result(), include_evidence=False)
        self.assertIn("sha256:" + "ab" * 32, text)
        html = render_html(_hostile_result())
        self.assertIn("sha256:" + "ab" * 32, html)

    def test_boundary_token_ids_are_public_by_design(self):
        # By-design: operator-specified run configuration (the required
        # --boundary-ids argument), consumed by diff-reports' boundary-drift
        # detection from public results.json and shown in the HTML config.
        text = _public_json_text(_hostile_result(), include_evidence=False)
        self.assertEqual(json.loads(text)["search"][0]["boundary_token_ids"], [7])
        html = render_html(_hostile_result())
        self.assertIn("Boundary token IDs", html)

    def test_hostile_observation_fields_stripped_without_evidence(self):
        result = _hostile_result()
        view = _public_view(result, include_evidence=False)
        self.assertNotIn("observations", view["search"][0])
        text = _public_json_text(result, include_evidence=True)
        self.assertIn(_HOSTILE_PROMPT, text)  # error field kept with evidence


if __name__ == "__main__":
    unittest.main()
