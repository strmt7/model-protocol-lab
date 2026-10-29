"""Cycle 75: adversarial audit of report determinism + section fidelity.

Target: ``model_protocol_lab/reporting.py`` -- the HTML/JSON report builders.
"Deterministic reports" is a hard constraint of this project; earlier cycles
audited HTML escaping (58) and pinned backend byte-identical outputs (65),
but no audit has verified the *reporting layer's* determinism and section
fidelity end to end. This file does that, adversarially.

Method
------
1. **Determinism.** Several synthetic result dicts covering every report
   section are rendered 3x in-process via ``save_report`` (both evidence
   flags) and in 2+ fresh subprocesses with varied ``PYTHONHASHSEED``;
   HTML and JSON bytes must be identical every time. Subprocess fixtures
   travel as JSON files so the fixture builders exist exactly once.
2. **Section fidelity (independent oracle).** Expected section contents are
   hand-derived literals -- selection margins, token-overhead aggregates,
   TOC anchor lists, fingerprint truncations, banner classes, matrix cells,
   sparkline SVG geometry -- computed by hand from the input dicts and
   written as literals. No expectation is built by calling the source
   functions under test.
3. **Adversarial edges.** Empty result dicts, missing optional sections,
   unicode/HTML-special payloads, a 1200-fixture bulk result, contradictory
   fields, and a caller-mutation check (rendering must not mutate the input
   dict).
4. **Mutation gate.** 6 mutants planted in /tmp *copies* of reporting.py
   (PID-scoped dirs; the repo is never mutated): dropped sort, injected
   timestamp, flipped margin arithmetic, fingerprint threshold, TOC filter,
   min->max tie-break. Every mutant is caught by at least one check; an
   unmutated control copy passes every check (proving the harness faithful).

Bugs found: none. The reporting layer is deterministic and every section
renders exactly the hand-derived expected values (see checks below).
"""

import copy
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from model_protocol_lab import reporting
from model_protocol_lab.diffbackends import MAX_DIVERGENCE_OBSERVATIONS

# ---------------------------------------------------------------------------
# Constants (hand-verified literals, not derived from source functions).
# ---------------------------------------------------------------------------

# Allowlisted script hash embedded in the report's Content-Security-Policy.
_CSP_SCRIPT_SHA256 = "WdNqqXTmjFU7IIw2w/LHX2rjE5+VTUTTCo08g0bykhI="

# Marker the evidence-excluded JSON view substitutes for raw diagnostics.
_WITHHELD = "[withheld without --include-evidence]"

# PID-scoped scratch root: concurrent pytest processes never share it.
_SCRATCH = Path(tempfile.gettempdir()) / f"cycle75_reporting_{os.getpid()}"

_REPO_ROOT = Path(reporting.__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Synthetic fixtures. Every expectation below is a hand-derived literal.
# ---------------------------------------------------------------------------


def _obs(fixture, exact, boundary, status="completed", in_len=10, out_len=3):
    """Obs."""
    obs = {
        "fixture": fixture,
        "status": status,
        "exact_match": exact,
        "boundary_found": boundary,
        "input_ids": [1] * in_len,
        "output_ids": [2] * out_len,
    }
    return obs


def _search_run(name, observations, rate, passes, planned, boundary_ids, fingerprint):
    """Search run."""
    return {
        "format": {"name": name},
        "candidate_fingerprint": fingerprint,
        "boundary_token_ids": boundary_ids,
        "summary": {
            "rate": rate,
            "status": "complete",
            "attempted": planned,
            "planned": planned,
            "exact_and_boundary_passes": passes,
            "encoded_input_tokens": 100,
            "retained_output_tokens": 24,
        },
        "observations": observations,
    }


def _result_full():
    """Every section populated. Search order is deliberately non-alphabetical
    (gamma, alpha, aardvark, beta) so ordering bugs surface.
    """
    search = [
        _search_run(
            "gamma",
            [
                _obs("f1", True, True, in_len=8, out_len=2),
                _obs("f2", False, True, in_len=8, out_len=2),
                _obs("f3", False, False, status="error", in_len=8, out_len=2),
                _obs("f4", False, False, in_len=8, out_len=2),
            ],
            0.25,
            1,
            4,
            [101, 102],
            "g" * 64,
        ),
        _search_run(
            "alpha",
            [
                _obs("f1", True, True, in_len=10, out_len=3),
                _obs("f2", True, True, in_len=20, out_len=5),
                _obs("f3", True, True, in_len=30, out_len=7),
                _obs("f4", True, True, in_len=40, out_len=9),
            ],
            1.0,
            4,
            4,
            [101, 102],
            "a" * 64,
        ),
        _search_run(
            "aardvark",
            [
                _obs("f1", True, True, in_len=100, out_len=10),
                _obs("f2", True, True, in_len=100, out_len=10),
                _obs("f3", False, True, in_len=100, out_len=10),
                _obs("f4", False, True, in_len=100, out_len=10),
            ],
            0.5,
            2,
            4,
            [201],
            "v" * 64,
        ),
        _search_run(
            "beta",
            [
                _obs("f1", True, True, in_len=12, out_len=4),
                _obs("f2", True, True, in_len=22, out_len=6),
                _obs("f3", False, True, in_len=32, out_len=8),
                _obs("f4", True, False, in_len=42, out_len=10),
            ],
            0.5,
            2,
            4,
            [101, 102],
            "b" * 64,
        ),
    ]
    return {
        "decision": "compatible_on_fixture_set",
        "selection_rule": "sole search winner",
        "measurement": "exact answer + boundary",
        "seed": 7,
        "max_new_tokens": 64,
        "statistical_confidence": "none claimed",
        "compute_comparison": "retained counts only",
        "backend_fingerprint": "0123456789ABCDEFGHIJKLMNOP",
        "selected_candidate": "alpha",
        "search": search,
        "held_out": {
            "summary": {
                "status": "complete",
                "rate": 1.0,
                "exact_and_boundary_passes": 2,
                "planned": 2,
                "aggregate_eligible": True,
            },
            "observations": [{"fixture": "h1", "status": "completed"}],
        },
        "template_conformance": {
            "status": "ok",
            "diffs": [
                {
                    "format": "chatml",
                    "fixture": "f1",
                    "hand_tokens": 10,
                    "template_tokens": 10,
                    "status": "ok",
                    "diff": [],
                },
                {
                    "format": "chatml",
                    "fixture": "f2",
                    "hand_tokens": 12,
                    "template_tokens": 14,
                    "status": "ok",
                    "diff": [
                        {
                            "hand_span": [0, 3],
                            "template_span": [0, 4],
                            "hand_ids": [1, 2, 3],
                            "template_ids": [1, 2, 3, 4],
                        }
                    ],
                },
                {"format": "llama3", "fixture": "f1", "status": "unavailable"},
            ],
        },
        "stop_matrix": {
            "semantics": [
                {"id": "native", "label": "Native", "description": "lab stop"},
                {
                    "id": "hf_stop_string",
                    "label": "HF stop string",
                    "description": "stop-string truncation",
                },
                {"id": "vllm_holdback_strip", "label": "vLLM strip", "description": "strip"},
            ],
            "search": [
                {
                    "fixture": "f1",
                    "native_pass": True,
                    "cells": {
                        "hf_stop_string": {"available": True, "pass": True, "fired": False},
                        "vllm_holdback_strip": {"available": True, "pass": False, "fired": True},
                    },
                    "flips": ["vllm_holdback_strip"],
                },
                {
                    "fixture": "f2",
                    "native_pass": False,
                    "cells": {"hf_stop_string": {"available": False, "detail": "no recording"}},
                    "flips": [],
                },
            ],
        },
        "backend_divergences": {
            "verdict": "diverge",
            "pairs_verdict": "1 of 2 pairs diverge",
            "matrix": {
                "KEPT_VS_STRIPPED": {
                    "hf_stop_string": 2,
                    "vllm_holdback_strip": 0,
                    "vllm_holdback_keep": 1,
                    "sglang_stop": 0,
                    "eos_appended": 0,
                },
                "UNKNOWN_CLASS_X": {
                    "hf_stop_string": 5,
                    "vllm_holdback_strip": 0,
                    "vllm_holdback_keep": 0,
                    "sglang_stop": 0,
                    "eos_appended": 0,
                },
            },
            "severities": {"KEPT_VS_STRIPPED": "high", "UNKNOWN_CLASS_X": "low"},
            "per_fixture": [
                {
                    "fixture": "f1",
                    "divergences": [
                        {
                            "class": "KEPT_VS_STRIPPED",
                            "severity": "high",
                            "stop_text": "SECRET",
                            "detail": "raw detail here",
                        },
                        {"class": "UNKNOWN_CLASS_X", "severity": "low", "detail": "d2"},
                    ],
                }
            ],
            "stop_texts": ["SECRET"],
            "observations_truncated": True,
            "observations_total": 600,
        },
        "budget_frontier": {
            "max_new_tokens": 64,
            "budget_robust": False,
            "native_winner": "alpha",
            "decision_steps": [
                [1, "no_candidate", None],
                [8, "no_candidate", None],
                [32, "compatible_on_fixture_set", "alpha"],
            ],
            "flip_ranges": [[1, 7], [8, 31]],
            "candidates": {
                "alpha": {
                    "fixture_ids": ["f1", "f2", "f3", "f4"],
                    "min_budget_per_fixture": [8, 16, 32, 64],
                    "full_pass_budget": 64,
                    "pass_steps": [[8, 1], [16, 2], [32, 3], [64, 4]],
                },
                "beta": {
                    "fixture_ids": ["f1", "f2"],
                    "min_budget_per_fixture": [4, 4],
                    "full_pass_budget": 4,
                    "pass_steps": [[4, 2]],
                },
            },
        },
        "sensitivity_sweep": {
            "native_winner": "alpha",
            "native_decision": "compatible_on_fixture_set",
            "flips_at": [3],
            "depths": [
                {
                    "drop_bytes": 1,
                    "decision": "compatible_on_fixture_set",
                    "selected_candidate": "alpha",
                    "winner_changed": False,
                },
                {
                    "drop_bytes": 3,
                    "decision": "no_candidate",
                    "selected_candidate": None,
                    "winner_changed": True,
                },
            ],
        },
        "logprob_diagnostic": {
            "available": True,
            "thresholds": {
                "derailed_mean_logprob_at_least": -0.5,
                "unknown_mean_logprob_below": -3.0,
            },
            "candidates": [
                {
                    "name": "alpha",
                    "aggregate": {
                        "native_passes": 4,
                        "total_logprob": -12.345,
                        "mean_logprob": -1.234,
                        "scored_token_count": 10,
                    },
                    "interpretation": {"kind": "derailed?", "text": "weights know it"},
                    "fixture_scores": [
                        {
                            "fixture": "f1",
                            "native_pass": True,
                            "status": "scored",
                            "reason": "",
                            "mean_logprob": -0.5,
                        },
                        {
                            "fixture": "f2",
                            "native_pass": False,
                            "status": "scored",
                            "reason": "x",
                            "mean_logprob": -2.0,
                        },
                        {
                            "fixture": "f3",
                            "native_pass": None,
                            "status": "unscored",
                            "mean_logprob": -1,
                        },
                    ],
                }
            ],
        },
    }


def _result_solo():
    """Single candidate: margin n/a branch, winner-is-cheapest tax branch."""
    return {
        "decision": "compatible_on_fixture_set",
        "selected_candidate": "solo",
        "search": [
            _search_run(
                "solo",
                [
                    _obs("g1", True, True, in_len=50, out_len=5),
                    _obs("g2", True, True, in_len=50, out_len=5),
                ],
                1.0,
                2,
                2,
                [7],
                "s" * 40,
            )
        ],
    }


def _result_no_candidate():
    """Result no candidate."""
    return {
        "decision": "no_candidate",
        "search": [
            _search_run(
                "m1", [_obs("f1", True, True), _obs("f2", False, True)], 0.5, 1, 2, [1], "m" * 16
            ),
            _search_run(
                "m2", [_obs("f1", False, True), _obs("f2", False, True)], 0.0, 0, 2, [1], "n" * 16
            ),
        ],
        "failure_taxonomy": {
            "per_candidate": {
                "m1": {"wrong_answer": 2, "stopped_early": 0},
                "m2": {"wrong_answer": 1, "timeout": 1},
            },
            "held_out": {"wrong_answer": 3, "timeout": 0},
        },
    }


def _result_ambiguous():
    """Result ambiguous."""
    return {
        "decision": "ambiguous",
        "search": [
            _search_run(
                "p", [_obs("f1", True, True), _obs("f2", True, True)], 1.0, 2, 2, [1], "p" * 16
            ),
            _search_run(
                "q", [_obs("f1", True, True), _obs("f2", True, True)], 1.0, 2, 2, [1], "q" * 16
            ),
            _search_run(
                "r", [_obs("f1", False, True), _obs("f2", True, True)], 0.5, 1, 2, [1], "r" * 16
            ),
        ],
    }


def _result_failed():
    """Result failed."""
    return {
        "decision": "failed",
        "seed": 1,
        "search": [
            {
                "format": {"name": "x"},
                "summary": {
                    "status": "error",
                    "failure_code": "generate_failed",
                    "attempted": 1,
                    "planned": 2,
                },
                "observations": [],
            }
        ],
        "failure_taxonomy": {},
    }


def _result_frontier_stable():
    """Tiny frontier (max_new_tokens=4 <= 48): exact SVG geometry is hand-derived."""
    return {
        "decision": "ambiguous",
        "search": [
            _search_run(
                "c", [_obs("f1", True, True), _obs("f2", True, True)], 1.0, 2, 2, [1], "c" * 16
            ),
        ],
        "budget_frontier": {
            "max_new_tokens": 4,
            "budget_robust": True,
            "native_winner": "c",
            "candidates": {
                "c": {
                    "fixture_ids": ["f1", "f2"],
                    "min_budget_per_fixture": [1, 3],
                    "full_pass_budget": 3,
                    "pass_steps": [[1, 1], [3, 2]],
                }
            },
        },
    }


def _result_edge_sections():
    """no_template_declared conformance, unavailable logprob, unknown decision."""
    return {
        "decision": "something_unexpected",
        "search": [],
        "template_conformance": {"status": "no_template_declared"},
        "logprob_diagnostic": {"available": False, "detail": ""},
    }


def _result_hostile():
    """Result hostile."""
    name = 'X <script>alert("x")</script> & "q" \'s\' caf\u00e9 \u2603'
    return {
        "decision": "ambiguous",
        "search": [
            _search_run(
                name,
                [_obs('f<1>&"2"', True, True), _obs("plain", False, True)],
                0.5,
                1,
                2,
                [1],
                "h" * 32,
            )
        ],
    }


def _result_huge(n=1200):
    """Result huge."""
    runs = []
    for cand in ("h1", "h2"):
        obs = [_obs(f"bulk-{i:05d}", i % 2 == 0, True, in_len=3, out_len=1) for i in range(n)]
        runs.append(_search_run(cand, obs, 0.5, n // 2, n, [9], cand * 32))
    return {"decision": "ambiguous", "selected_candidate": None, "search": runs}


def _result_contradictory():
    """selected_candidate names no recorded candidate: pins observed behavior."""
    return {
        "decision": "compatible_on_fixture_set",
        "selected_candidate": "ghost",
        "search": [
            _search_run(
                "c1", [_obs("f1", True, True), _obs("f2", False, True)], 0.5, 1, 2, [1], "1" * 16
            ),
            _search_run(
                "c2", [_obs("f1", False, True), _obs("f2", False, True)], 0.0, 0, 2, [1], "2" * 16
            ),
        ],
    }


_FIXTURES = {
    "full": _result_full,
    "solo": _result_solo,
    "no_candidate": _result_no_candidate,
    "ambiguous": _result_ambiguous,
    "failed": _result_failed,
    "frontier_stable": _result_frontier_stable,
    "edge_sections": _result_edge_sections,
    "hostile": _result_hostile,
    "contradictory": _result_contradictory,
    "empty": dict,
}

# ---------------------------------------------------------------------------
# Render helpers.
# ---------------------------------------------------------------------------


def _save_triple(mod, result, tmpdir, tag):
    """Render one (html, json-no-evidence, json-evidence) triple via save_report."""
    outs = []
    for ev in (False, True):
        target = tmpdir / f"{tag}_{int(ev)}"
        json_path, html_path = mod.save_report(copy.deepcopy(result), target, include_evidence=ev)
        outs.append((html_path.read_bytes(), json_path.read_bytes()))
    return outs[0][0], outs[0][1], outs[1][1]


def _read_json(raw):
    """Read json."""
    return json.loads(raw.decode("utf-8"))


def _toc_anchors(html):
    """Anchor ids of the sticky TOC, in order."""
    nav = re.search(r'<nav class="toc".*?</nav>', html, re.DOTALL)
    assert nav is not None, "TOC nav missing"
    return re.findall(r'<a href="#([^"]+)">', nav.group(0))


# ---------------------------------------------------------------------------
# Hand-derived expectation literals for the full fixture.
# ---------------------------------------------------------------------------

# Outcomes (hand-derived from _result_full observations):
#   alpha: f1..f4 pass            -> failing []
#   beta:  f1,f2 pass; f3,f4 fail -> failing [f3, f4]
#   gamma: f1 pass; f2,f4 fail; f3 incomplete -> failing [f2, f3, f4]
#   aardvark: f1,f2 pass; f3,f4 fail -> failing [f3, f4]
# Selected "alpha"; runners-up tie beta/aardvark at 2 -> lexicographic min wins.
_EXPECT_MARGIN_FULL = {
    "value": 2,
    "direction": "to_tie",
    "runner_up": "aardvark",
    "decisive_fixtures": ["f3", "f4"],
}

# Token overhead (hand-derived means over *completed* observations only;
# gamma's errored f3 observation is excluded: 3 completed obs of 8 tokens):
#   aardvark: prompt (100,100,100,100) -> 100.0 / 100 ; output -> 10.0
#   alpha:    prompt (10,20,30,40)     -> 25.0  / 40  ; output (3,5,7,9) -> 6.0
#   beta:     prompt (12,22,32,42)     -> 27.0  / 42  ; output (4,6,8,10) -> 7.0
#   gamma:    prompt (8,8,8)           -> 8.0   / 8   ; output (2,2,2) -> 2.0
# Cheapest: gamma (8.0). Winner alpha: (25.0-8.0)/8.0*100 = 212.5.
_EXPECT_OVERHEAD_FULL = {
    "per_candidate": {
        "aardvark": {
            "mean_prompt_tokens": 100.0,
            "max_prompt_tokens": 100,
            "mean_output_tokens": 10.0,
        },
        "alpha": {"mean_prompt_tokens": 25.0, "max_prompt_tokens": 40, "mean_output_tokens": 6.0},
        "beta": {"mean_prompt_tokens": 27.0, "max_prompt_tokens": 42, "mean_output_tokens": 7.0},
        "gamma": {"mean_prompt_tokens": 8.0, "max_prompt_tokens": 8, "mean_output_tokens": 2.0},
    },
    "cheapest_candidate": "gamma",
    "winner_tax_vs_cheapest_pct": 212.5,
}
_EXPECT_OVERHEAD_KEY_ORDER = ["aardvark", "alpha", "beta", "gamma"]

_EXPECT_TOC_FULL = [
    "run-configuration",
    "search-candidates",
    "per-fixture-outcomes",
    "decisive-fixtures",
    "token-overhead",
    "template-conformance",
    "stop-behavior-matrix",
    "backend-divergence",
    "token-budget-frontier",
    "sensitivity-sweep",
    "logprob-diagnostic",
    "held-out-stage",
]

# backend_fingerprint "0123456789ABCDEFGHIJKLMNOP" -> first 19 chars + ellipsis.
_EXPECT_FP_SHORT = "0123456789ABCDEFGHI\u2026"
_EXPECT_FP_FULL = "0123456789ABCDEFGHIJKLMNOP"

# ---------------------------------------------------------------------------
# Fidelity checks. Each takes the module under test so the mutant gate can
# run the identical checks against /tmp copies. Expectations are literals
# hand-derived from the fixtures above -- never computed by the module.
# ---------------------------------------------------------------------------


def check_margin_oracle(mod, tc):
    """selection_margin literals + banner lines for every decision branch."""
    cases = [
        (
            "full",
            _EXPECT_MARGIN_FULL,
            "Selection margin: 2 fixtures (runner-up: aardvark)",
            "Flipping 2 of aardvark&#x27;s failing fixtures to pass would tie the selection.",
            "Flipping 2 of aardvark&#x27;s failing fixtures (f3, f4) to pass "
            "would tie the selection. The margin is descriptive; it never changes the decision.",
        ),
        (
            "solo",
            {"value": None, "direction": "to_tie", "runner_up": None, "decisive_fixtures": []},
            "Selection margin: n/a (single candidate)",
            "With a single candidate there is no runner-up to tie against.",
            "There is no runner-up, so the margin is undefined. "
            "The margin is descriptive; it never changes the decision.",
        ),
        (
            "no_candidate",
            {"value": 1, "direction": "to_winner", "runner_up": "m1", "decisive_fixtures": ["f2"]},
            "Selection margin: 1 fixture (closest: m1)",
            "Flipping 1 of m1&#x27;s failing fixtures to pass would select m1.",
            "Flipping 1 of m1&#x27;s failing fixtures (f2) to pass would select m1. "
            "The margin is descriptive; it never changes the decision.",
        ),
        (
            "ambiguous",
            {
                "value": 1,
                "direction": "to_unique",
                "runner_up": None,
                "decisive_fixtures": ["f1", "f2"],
            },
            "Selection margin: 1 candidate",
            "Demoting 1 candidate of the 2 tied candidates would leave a unique winner.",
            "Flipping one of the shared fixtures (f1, f2) to fail for 1 candidate "
            "of the 2 tied candidates would leave a unique winner. "
            "The margin is descriptive; it never changes the decision.",
        ),
    ]
    for name, expected, label, tip, explainer in cases:
        result = _FIXTURES[name]()
        with tc.subTest(fixture=name):
            tc.assertEqual(mod.selection_margin(result), expected)
            html = mod.render_html(result)
            tc.assertIn(label, html)
            tc.assertIn('title="' + tip + '"', html)
            tc.assertIn(explainer, html)
            payload = _read_json(
                mod.save_report(copy.deepcopy(result), _fresh(f"m{name}"), include_evidence=False)[
                    0
                ].read_bytes()
            )
            tc.assertEqual(payload["selection_margin"], expected)
    # failed / cancelled -> null margin, no banner line.
    failed = _result_failed()
    with tc.subTest(fixture="failed"):
        tc.assertIsNone(mod.selection_margin(failed))
        html = mod.render_html(failed)
        tc.assertNotIn("Selection margin:", html)
        payload = _read_json(
            mod.save_report(copy.deepcopy(failed), _fresh("mfailed"), include_evidence=False)[
                0
            ].read_bytes()
        )
        tc.assertIsNone(payload["selection_margin"])


def check_overhead_oracle(mod, tc):
    """Check overhead oracle."""
    result = _result_full()
    with tc.subTest("json"):
        payload = _read_json(
            mod.save_report(copy.deepcopy(result), _fresh("o"), include_evidence=False)[
                0
            ].read_bytes()
        )
        overhead = payload["token_overhead"]
        tc.assertEqual(overhead, _EXPECT_OVERHEAD_FULL)
        # Key order is part of the contract: sorted, not insertion order.
        tc.assertEqual(list(overhead["per_candidate"].keys()), _EXPECT_OVERHEAD_KEY_ORDER)
    with tc.subTest("html"):
        html = mod.render_html(result)
        # repr() forms of the hand-derived floats, winner starred.
        tc.assertIn('<td data-sort="25.0">25.0</td>', html)
        tc.assertIn('<td data-sort="100.0">100.0</td>', html)
        tc.assertIn(">alpha \u2605</th>", html)
        tc.assertIn(
            "Winner token tax vs cheapest: alpha averages 25.0 prompt tokens vs 8.0 "
            "for gamma \u2014 a token tax of 212.5%.",
            html,
        )
    with tc.subTest("winner_is_cheapest"):
        solo = _result_solo()
        payload = _read_json(
            mod.save_report(copy.deepcopy(solo), _fresh("osolo"), include_evidence=False)[
                0
            ].read_bytes()
        )
        overhead = payload["token_overhead"]
        tc.assertEqual(overhead["cheapest_candidate"], "solo")
        tc.assertEqual(overhead["winner_tax_vs_cheapest_pct"], 0.0)
        tc.assertEqual(
            overhead["per_candidate"]["solo"],
            {"mean_prompt_tokens": 50.0, "max_prompt_tokens": 50, "mean_output_tokens": 5.0},
        )
        html = mod.render_html(solo)
        tc.assertIn(
            "The winner (solo) is also the cheapest candidate "
            "(50.0 mean prompt tokens): 0.0% token tax.",
            html,
        )
    with tc.subTest("no_winner"):
        nc = _result_no_candidate()
        payload = _read_json(
            mod.save_report(copy.deepcopy(nc), _fresh("onc"), include_evidence=False)[
                0
            ].read_bytes()
        )
        tc.assertIsNone(payload["token_overhead"]["winner_tax_vs_cheapest_pct"])
        html = mod.render_html(nc)
        tc.assertIn("No candidate was selected, so no winner token tax is reported.", html)
    with tc.subTest("failed"):
        failed = _result_failed()
        payload = _read_json(
            mod.save_report(copy.deepcopy(failed), _fresh("ofail"), include_evidence=False)[
                0
            ].read_bytes()
        )
        overhead = payload["token_overhead"]
        tc.assertEqual(
            overhead["per_candidate"]["x"],
            {"mean_prompt_tokens": None, "max_prompt_tokens": None, "mean_output_tokens": None},
        )
        tc.assertIsNone(overhead["cheapest_candidate"])
        tc.assertIsNone(overhead["winner_tax_vs_cheapest_pct"])


def check_toc(mod, tc):
    """Check toc."""
    html = mod.render_html(_result_full())
    tc.assertEqual(_toc_anchors(html), _EXPECT_TOC_FULL)
    # Minimal result: only the always-rendered sections get TOC entries.
    minimal = mod.render_html(_result_failed())
    tc.assertEqual(
        _toc_anchors(minimal),
        ["run-configuration", "search-candidates", "token-overhead", "held-out-stage"],
    )
    edge = mod.render_html(_result_edge_sections())
    tc.assertEqual(
        _toc_anchors(edge),
        [
            "run-configuration",
            "search-candidates",
            "token-overhead",
            "template-conformance",
            "logprob-diagnostic",
            "held-out-stage",
        ],
    )
    # Every TOC anchor resolves to a rendered section id.
    for anchor in _toc_anchors(html):
        tc.assertIn(f'<section id="{anchor}"', html)


def check_fingerprint(mod, tc):
    """Check fingerprint."""
    html = mod.render_html(_result_full())
    # Config section: truncated fingerprint + full fingerprint in <details>.
    tc.assertIn("<dd>" + _EXPECT_FP_SHORT + "</dd>", html)
    tc.assertIn('<p class="mono">' + _EXPECT_FP_FULL + "</p>", html)
    # Search rows: 64-char candidate fingerprint truncated to 19 + ellipsis.
    tc.assertIn("aaaaaaaaaaaaaaaaaaa\u2026", html)
    tc.assertIn("a" * 64, html)


def check_determinism(mod, tc, tmpdir):
    """3x in-process renders via save_report: byte-identical HTML and JSON."""
    for name, builder in _FIXTURES.items():
        if name == "huge":
            continue  # covered separately with 2x (slower)
        with tc.subTest(fixture=name):
            result = builder()
            triples = [_save_triple(mod, result, tmpdir, f"d_{name}_{i}") for i in range(3)]
            for triple in triples[1:]:
                tc.assertEqual(triple, triples[0])


def check_conformance_section(mod, tc):
    """Check conformance section."""
    html = mod.render_html(_result_full())
    tc.assertIn('<span class="badge pass">identical</span>', html)
    tc.assertIn('<span class="badge fail">1 divergent segment(s)</span>', html)
    # Spans only -- token-ID subsequences are withheld from this HTML view.
    tc.assertIn("hand [0, 3] \u2194 template [0, 4]", html)
    tc.assertNotIn("[1, 2, 3]", html)
    tc.assertIn('<span class="badge neutral">unavailable</span>', html)
    # Numeric data-sort keys via _sort_key; missing values sort as "None".
    tc.assertIn('data-sort="12"', html)
    tc.assertIn('data-sort="None">?</td>', html)
    # no_template_declared branch.
    edge = mod.render_html(_result_edge_sections())
    tc.assertIn("No chat template is declared in the checkpoint", edge)
    # Missing key -> no section, no TOC entry.
    minimal = mod.render_html(_result_failed())
    tc.assertNotIn('id="template-conformance"', minimal)


def check_stop_matrix_section(mod, tc):
    """Check stop matrix section."""
    html = mod.render_html(_result_full())
    # Header excludes the native semantic; legend covers all three.
    tc.assertIn('<th scope="col">HF stop string</th><th scope="col">vLLM strip</th>', html)
    tc.assertIn("<dt>Native</dt><dd>lab stop</dd>", html)
    # f1: native pass, hf pass, vllm fail with stop-fired marker, flip note.
    tc.assertIn('<span class="badge pass">pass</span>', html)
    tc.assertIn(
        '<span class="badge fail">fail</span><span class="mono"> \u00b7 stop fired</span>', html
    )
    tc.assertIn('<span class="mono">vllm_holdback_strip</span>', html)
    # f2: unavailable cell with detail, empty flips render as em dash.
    tc.assertIn('<span class="badge neutral">n/a</span> (no recording)', html)
    tc.assertIn('<span class="mono">\u2014</span>', html)
    tc.assertIn("<h3>Search stage</h3>", html)
    tc.assertNotIn("<h3>Held Out stage</h3>", html)
    # Missing key -> no section.
    tc.assertNotIn('id="stop-behavior-matrix"', mod.render_html(_result_failed()))


def check_divergence_section(mod, tc):
    """Check divergence section."""
    html = mod.render_html(_result_full())
    tc.assertIn('<span class="badge fail">diverge</span> 1 of 2 pairs diverge.', html)
    # Matrix row: severity badge + fixed class order + per-semantic counts.
    tc.assertIn('<span class="badge fail">high</span>', html)
    tc.assertIn('<span class="mono">KEPT_VS_STRIPPED</span>', html)
    tc.assertIn(
        '<td data-sort="2">2</td><td data-sort="0">0</td><td data-sort="1">1</td>'
        '<td data-sort="0">0</td><td data-sort="0">0</td>',
        html,
    )
    # Unknown class: dropped from the matrix but its callout still renders.
    tc.assertEqual(html.count("UNKNOWN_CLASS_X"), 1)
    tc.assertIn(
        '<span class="badge neutral">low</span> <span class="mono">UNKNOWN_CLASS_X</span>'
        ' on <span class="mono">f1</span>',
        html,
    )
    # Raw stop text and free-text detail are withheld from this HTML view.
    tc.assertNotIn("SECRET", html)
    tc.assertNotIn("raw detail here", html)
    tc.assertIn("[stop text and diagnostic detail withheld from this HTML view]", html)
    tc.assertIn(
        "Observation bound reached: the run recorded 600 observations; "
        "only the first 512 were differential-tested.",
        html,
    )


def check_frontier_section(mod, tc):
    """Check frontier section."""
    html = mod.render_html(_result_full())
    tc.assertIn('<span class="badge fail">budget-sensitive</span>', html)
    tc.assertIn("<li>budgets 1&ndash;7: no_candidate</li>", html)
    tc.assertIn("<li>budgets 8&ndash;31: no_candidate</li>", html)
    tc.assertIn("<h3>Winner pass curve: alpha</h3>", html)
    tc.assertIn("<td>&ge; 8 tokens</td>", html)
    tc.assertIn('<span class="rate-text">1/4</span>', html)
    # 64 budgets > 48 -> sampled to 48; geometry is integer arithmetic.
    tc.assertIn('width="144"', html)
    tc.assertIn('viewBox="0 0 48 8"', html)
    tc.assertIn('aria-label="pass count 0 to 4 of 4 over budgets 1 to 64"', html)
    tc.assertIn('aria-label="pass count 0 to 2 of 2 over budgets 1 to 64"', html)

    # Stable branch + exact SVG geometry for the tiny (<=48) frontier.
    stable = mod.render_html(_result_frontier_stable())
    tc.assertIn('<span class="badge pass">stable</span>', stable)
    tc.assertIn("identical at every token budget 1&ndash;4.", stable)
    # counts at budgets 1..4: [1,1,2,2]; levels round(c/2*8): [4,4,8,8].
    tc.assertIn(
        '<svg class="spark-svg" width="12" height="24" viewBox="0 0 4 8" role="img" '
        'aria-label="pass count 1 to 2 of 2 over budgets 1 to 4">'
        '<rect x="0" y="4" width="0.9" height="4"/>'
        '<rect x="1" y="4" width="0.9" height="4"/>'
        '<rect x="2" y="0" width="0.9" height="8"/>'
        '<rect x="3" y="0" width="0.9" height="8"/>'
        "</svg>",
        stable,
    )
    tc.assertIn("sampled left to right from budget 1 to 4.", stable)


def check_sweep_section(mod, tc):
    """Check sweep section."""
    html = mod.render_html(_result_full())
    tc.assertIn(
        "Native winner: alpha (compatible_on_fixture_set). "
        "Depths where the winner or decision changed: [3].",
        html,
    )
    tc.assertIn(
        '<th scope="row" data-sort="1">1</th>'
        '<td data-sort="compatible_on_fixture_set">compatible_on_fixture_set</td>'
        '<td data-sort="alpha">alpha</td>'
        '<td data-sort="stable"><span class="badge pass">stable</span></td>',
        html,
    )
    # selected_candidate None renders via str() as "None".
    tc.assertIn(
        '<th scope="row" data-sort="3">3</th>'
        '<td data-sort="no_candidate">no_candidate</td>'
        '<td data-sort="None">None</td>'
        '<td data-sort="changed"><span class="badge fail">changed</span></td>',
        html,
    )


def check_logprob_section(mod, tc):
    """Check logprob section."""
    html = mod.render_html(_result_full())
    tc.assertIn(
        "Declared bands: mean \u2265 -0.5 nats/token reads as greedy derailment, "
        "mean < -3.0 as missing knowledge; between the bands the diagnostic "
        "withholds judgment.",
        html,
    )
    tc.assertIn('<th scope="row" data-sort="alpha">alpha</th><td data-sort="4">4/3</td>', html)
    tc.assertIn('<td data-sort="-12.345">-12.35</td>', html)
    tc.assertIn('<td data-sort="-1.234">-1.23</td>', html)
    tc.assertIn('<td data-sort="10">10</td>', html)
    tc.assertIn("<td>derailed?</td>", html)
    tc.assertIn("<p><strong>alpha:</strong> weights know it</p>", html)
    # Per-fixture glyphs, reason codes, and the int-mean em dash.
    tc.assertIn('<span class="glyph-ok" title="pass">\u2713</span>', html)
    tc.assertIn("<td>scored (x)</td>", html)
    tc.assertIn('<span class="glyph-na" title="unknown">\u2013</span>', html)
    tc.assertIn('<td data-sort="">\u2014</td>', html)
    tc.assertIn('<td data-sort="-0.5">-0.50</td>', html)
    # Unavailable branch.
    edge = mod.render_html(_result_edge_sections())
    tc.assertIn("Log-likelihood diagnostic unavailable: no reason recorded.", edge)


def check_heldout_section(mod, tc):
    """Check heldout section."""
    html = mod.render_html(_result_full())
    tc.assertIn('<div class="k">Status</div><div class="v">complete</div>', html)
    tc.assertIn('<div class="k">Pass rate</div><div class="v">100%</div>', html)
    tc.assertIn('<div class="k">Exact + boundary passes</div><div class="v">2/2</div>', html)
    tc.assertIn('<div class="k">Aggregate eligible</div><div class="v">Yes</div>', html)
    tc.assertIn(
        "Not evaluated &mdash; no candidate was promoted from search.",
        mod.render_html(_result_failed()),
    )


def check_decision_banner(mod, tc):
    """Check decision banner."""
    cases = [
        ("full", "compatible_on_fixture_set", "pass"),
        ("no_candidate", "no_candidate", "neutral"),
        ("ambiguous", "ambiguous", "neutral"),
        ("failed", "failed", "fail"),
        ("edge_sections", "something_unexpected", "neutral"),
    ]
    for name, decision, dclass in cases:
        html = mod.render_html(_FIXTURES[name]())
        with tc.subTest(fixture=name):
            tc.assertIn(f'<section class="decision {dclass}"', html)
            tc.assertIn(f'<div class="decision-value">{decision}</div>', html)
    edge = mod.render_html(_result_edge_sections())
    tc.assertIn("Unrecognized decision value.", edge)
    # Failure diagnosis names the sanitized code with operator guidance.
    failed = mod.render_html(_result_failed())
    tc.assertIn(
        "Failure diagnosis: generate_failed: Generation failed inside the backend.",
        failed,
    )
    # Taxonomy notes.
    nc = mod.render_html(_result_no_candidate())
    tc.assertIn("Failure-mode taxonomy: timeout \u00d71, wrong_answer \u00d76.", nc)
    tc.assertIn(
        "Failure-mode taxonomy: the run did not complete, so there are no "
        "observations to classify.",
        failed,
    )
    # Compatible decisions suppress the taxonomy note even when data exists.
    full = mod.render_html(_result_full())
    tc.assertNotIn("Failure-mode taxonomy", full)


def check_config_section(mod, tc):
    """Check config section."""
    html = mod.render_html(_result_full())
    tc.assertIn("<dt>Selection rule</dt><dd>sole search winner</dd>", html)
    tc.assertIn("<dt>Seed</dt><dd>7</dd>", html)
    tc.assertIn("<dt>Candidates</dt><dd>4</dd>", html)
    # Mixed boundary ids -> per-candidate lines in search order.
    tc.assertIn("<dt>Boundary token IDs (gamma)</dt><dd>101, 102</dd>", html)
    tc.assertIn("<dt>Boundary token IDs (aardvark)</dt><dd>201</dd>", html)
    solo = mod.render_html(_result_solo())
    tc.assertIn("<dt>Boundary token IDs</dt><dd>7</dd>", solo)


def check_outcome_grid_and_decisive(mod, tc):
    """Check outcome grid and decisive."""
    html = mod.render_html(_result_full())
    # Winner star on alpha; columns follow search order gamma, alpha, aardvark, beta.
    tc.assertIn('<th scope="col">alpha \u2605</th>', html)
    # f3: gamma incomplete -> en dash cell; f4: only alpha passes.
    tc.assertIn(
        '<td data-sort="0"><span class="glyph-na" title="incomplete">\u2013</span></td>', html
    )
    tc.assertIn('<td data-sort="2"><span class="glyph-ok" title="pass">\u2713</span></td>', html)
    # Decisive: f4 is alpha's solo pass; f3 is disqualified (gamma incomplete).
    tc.assertIn("<li><strong>alpha</strong>: f4</li>", html)
    tc.assertNotIn("<li><strong>beta</strong>", html)
    # Contradictory input: no starred column (the caption still documents the marker).
    contra = mod.render_html(_result_contradictory())
    tc.assertNotIn(" \u2605</th>", contra)
    tc.assertIn("Selection margin: 1 fixture (runner-up: c1)", contra)
    tc.assertIn("The winner's token tax is unavailable", contra)


def check_json_public_view(mod, tc):
    """Check json public view."""
    result = _result_full()
    noev = _read_json(
        mod.save_report(copy.deepcopy(result), _fresh("pv0"), include_evidence=False)[
            0
        ].read_bytes()
    )
    ev = _read_json(
        mod.save_report(copy.deepcopy(result), _fresh("pv1"), include_evidence=True)[0].read_bytes()
    )
    tc.assertFalse(noev["raw_evidence_included"])
    tc.assertTrue(ev["raw_evidence_included"])
    # results.json is ASCII-only even with unicode input.
    hostile = _read_json(
        mod.save_report(copy.deepcopy(_result_hostile()), _fresh("pv2"), include_evidence=False)[
            0
        ].read_bytes()
    )
    tc.assertTrue(json.dumps(hostile, ensure_ascii=True).isascii())
    for run in noev["search"]:
        tc.assertNotIn("observations", run)
    tc.assertNotIn("observations", noev["held_out"])
    tc.assertEqual(len(ev["search"][1]["observations"]), 4)
    # Conformance token-ID subsequences stripped only without evidence.
    seg_noev = noev["template_conformance"]["diffs"][1]["diff"][0]
    tc.assertNotIn("hand_ids", seg_noev)
    tc.assertNotIn("template_ids", seg_noev)
    tc.assertEqual(seg_noev["hand_span"], [0, 3])
    seg_ev = ev["template_conformance"]["diffs"][1]["diff"][0]
    tc.assertEqual(seg_ev["hand_ids"], [1, 2, 3])
    # Divergence detail withheld / restored.
    div_noev = noev["backend_divergences"]["per_fixture"][0]["divergences"][0]
    tc.assertNotIn("stop_text", div_noev)
    tc.assertEqual(div_noev["detail"], _WITHHELD)
    tc.assertNotIn("stop_texts", noev["backend_divergences"])
    div_ev = ev["backend_divergences"]["per_fixture"][0]["divergences"][0]
    tc.assertEqual(div_ev["stop_text"], "SECRET")
    tc.assertEqual(div_ev["detail"], "raw detail here")
    tc.assertEqual(ev["backend_divergences"]["stop_texts"], ["SECRET"])
    # Diagnostics attached as top-level keys in both views.
    for payload in (noev, ev):
        tc.assertEqual(payload["selection_margin"], _EXPECT_MARGIN_FULL)
        tc.assertEqual(payload["token_overhead"], _EXPECT_OVERHEAD_FULL)


def check_self_contained(mod, tc):
    """Check self contained."""
    html = mod.render_html(_result_full())
    tc.assertIn(
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "script-src 'sha256-" + _CSP_SCRIPT_SHA256 + "';",
        html,
    )
    tc.assertIn("<style>", html)
    tc.assertTrue(html.startswith("<!doctype html>"))
    tc.assertTrue(html.rstrip().endswith("</html>"))


def check_hostile_escaping(mod, tc):
    """Check hostile escaping."""
    html = mod.render_html(_result_hostile())
    tc.assertIn(
        "X &lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &quot;q&quot; "
        "&#x27;s&#x27; caf\u00e9 \u2603",
        html,
    )
    tc.assertNotIn("<script>alert", html)
    tc.assertNotIn("f<1>", html)
    tc.assertIn("f&lt;1&gt;&amp;&quot;2&quot;", html)
    # Layout intact.
    tc.assertTrue(html.rstrip().endswith("</html>"))
    tc.assertIn("<table", html)


def check_no_caller_mutation(mod, tc):
    """Check no caller mutation."""
    for name, builder in _FIXTURES.items():
        if name == "huge":
            continue
        with tc.subTest(fixture=name):
            result = builder()
            snapshot = copy.deepcopy(result)
            mod.render_html(result)
            mod.save_report(result, _fresh(f"nm{name}"), include_evidence=False)
            mod.save_report(result, _fresh(f"nm{name}e"), include_evidence=True)
            tc.assertEqual(result, snapshot)


_FRESH_COUNTER = 0


def _fresh(tag):
    """A fresh (not yet existing) directory path under the PID-scoped scratch
    root. save_report refuses existing leaves (O_EXCL), so the path must not
    exist until save_report creates it.
    """
    # skipcq: PYL-W0603 - module-level mutable test state is the fixture mechanism
    global _FRESH_COUNTER
    _SCRATCH.mkdir(parents=True, exist_ok=True)
    _FRESH_COUNTER += 1
    return _SCRATCH / f"c75_{tag}_{os.getpid()}_{_FRESH_COUNTER}"


# ---------------------------------------------------------------------------
# Test classes: the checks above run against the real module here.
# ---------------------------------------------------------------------------


class TestInProcessDeterminism(unittest.TestCase):
    """Test in process determinism."""

    @classmethod
    def setUpClass(cls):
        """Set up the test fixture."""
        _SCRATCH.mkdir(parents=True, exist_ok=True)
        cls.tmpdir = Path(tempfile.mkdtemp(prefix="c75_det_", dir=str(_SCRATCH)))

    @classmethod
    def tearDownClass(cls):
        """Tear down the test fixture."""
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_triple_renders_byte_identical(self):
        """Test triple renders byte identical."""
        check_determinism(reporting, self, self.tmpdir)

    def test_huge_result_byte_identical(self):
        """Test huge result byte identical."""
        result = _result_huge()
        triples = [_save_triple(reporting, result, self.tmpdir, f"huge_{i}") for i in range(2)]
        self.assertEqual(triples[1], triples[0])
        html, _noev, _ev = triples[0]
        # Sanity: the bulk report is substantive but bounded.
        self.assertGreater(len(html), 100_000)
        self.assertLess(len(html), 60_000_000)


_DRIVER = (
    "import json, sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from model_protocol_lab import reporting\n"
    "result = json.load(open(sys.argv[2], encoding='utf-8'))\n"
    "for i, ev in enumerate((False, True)):\n"
    "    jp, hp = reporting.save_report(result, sys.argv[3] + '/sub%d' % i,\n"
    "                                   include_evidence=ev)\n"
    "    print(jp, hp)\n"
)


def _run_driver_subprocess(repo_root, fixture_path, outdir, hash_seed):
    """Run driver subprocess."""
    driver = outdir / "driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    proc = subprocess.run(
        [sys.executable, str(driver), str(repo_root), str(fixture_path), str(outdir)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_SCRATCH),
        check=True,
        timeout=300,
    )
    return proc


class TestSubprocessDeterminism(unittest.TestCase):
    """Fresh interpreters with varied PYTHONHASHSEED must emit identical bytes."""

    def test_cross_process_byte_identical(self):
        """Test cross process byte identical."""
        workdir = _fresh("sub")
        workdir.mkdir()
        fixture_path = workdir / "fixture.json"
        fixture_path.write_text(json.dumps(_result_full()), encoding="utf-8")
        expected = _save_triple(reporting, _result_full(), workdir, "inproc")
        digests_expected = [hashlib.sha256(blob).hexdigest() for blob in expected]
        for seed in ("0", "42", "987654321"):
            with self.subTest(hash_seed=seed):
                outdir = workdir / f"seed_{seed}"
                outdir.mkdir()
                _run_driver_subprocess(_REPO_ROOT, fixture_path, outdir, seed)
                blobs = (
                    (outdir / "sub0" / "report.html").read_bytes(),
                    (outdir / "sub0" / "results.json").read_bytes(),
                    (outdir / "sub1" / "results.json").read_bytes(),
                )
                self.assertEqual([hashlib.sha256(b).hexdigest() for b in blobs], digests_expected)

    def test_cross_process_hostile_and_empty(self):
        """Test cross process hostile and empty."""
        for name in ("hostile", "empty"):
            with self.subTest(fixture=name):
                workdir = _fresh(f"sub{name}")
                workdir.mkdir()
                fixture_path = workdir / "fixture.json"
                fixture_path.write_text(json.dumps(_FIXTURES[name]()), encoding="utf-8")
                expected = _save_triple(reporting, _FIXTURES[name](), workdir, "inproc")
                digests_expected = [hashlib.sha256(b).hexdigest() for b in expected]
                outdir = workdir / "seed_7"
                outdir.mkdir()
                _run_driver_subprocess(_REPO_ROOT, fixture_path, outdir, "7")
                blobs = (
                    (outdir / "sub0" / "report.html").read_bytes(),
                    (outdir / "sub0" / "results.json").read_bytes(),
                    (outdir / "sub1" / "results.json").read_bytes(),
                )
                self.assertEqual([hashlib.sha256(b).hexdigest() for b in blobs], digests_expected)


class TestSectionFidelity(unittest.TestCase):
    """Test section fidelity."""

    def test_margin_oracle(self):
        """Test margin oracle."""
        check_margin_oracle(reporting, self)

    def test_overhead_oracle(self):
        """Test overhead oracle."""
        check_overhead_oracle(reporting, self)

    def test_toc(self):
        """Test toc."""
        check_toc(reporting, self)

    def test_fingerprint(self):
        """Test fingerprint."""
        check_fingerprint(reporting, self)

    def test_conformance_section(self):
        """Test conformance section."""
        check_conformance_section(reporting, self)

    def test_stop_matrix_section(self):
        """Test stop matrix section."""
        check_stop_matrix_section(reporting, self)

    def test_divergence_section(self):
        """Test divergence section."""
        check_divergence_section(reporting, self)

    def test_frontier_section(self):
        """Test frontier section."""
        check_frontier_section(reporting, self)

    def test_sweep_section(self):
        """Test sweep section."""
        check_sweep_section(reporting, self)

    def test_logprob_section(self):
        """Test logprob section."""
        check_logprob_section(reporting, self)

    def test_heldout_section(self):
        """Test heldout section."""
        check_heldout_section(reporting, self)

    def test_decision_banner(self):
        """Test decision banner."""
        check_decision_banner(reporting, self)

    def test_config_section(self):
        """Test config section."""
        check_config_section(reporting, self)

    def test_outcome_grid_and_decisive(self):
        """Test outcome grid and decisive."""
        check_outcome_grid_and_decisive(reporting, self)

    def test_json_public_view(self):
        """Test json public view."""
        check_json_public_view(reporting, self)

    def test_self_contained(self):
        """Test self contained."""
        check_self_contained(reporting, self)


class TestAdversarialEdges(unittest.TestCase):
    """Test adversarial edges."""

    def test_empty_result_renders(self):
        """Test empty result renders."""
        html = reporting.render_html({})
        self.assertIn('<section class="decision neutral"', html)
        self.assertIn('<div class="decision-value"></div>', html)
        self.assertIn("Unrecognized decision value.", html)
        self.assertEqual(
            _toc_anchors(html),
            ["run-configuration", "search-candidates", "token-overhead", "held-out-stage"],
        )
        payload = _read_json(
            reporting.save_report({}, _fresh("empty"), include_evidence=False)[0].read_bytes()
        )
        self.assertIsNone(payload["selection_margin"])
        self.assertEqual(payload["token_overhead"]["per_candidate"], {})
        self.assertIsNone(payload["token_overhead"]["cheapest_candidate"])

    def test_hostile_escaping(self):
        """Test hostile escaping."""
        check_hostile_escaping(reporting, self)

    def test_no_caller_mutation(self):
        """Test no caller mutation."""
        check_no_caller_mutation(reporting, self)

    def test_contradictory_fields_pinned(self):
        """Test contradictory fields pinned."""
        # Observed behavior pin (real pipeline never emits this): the margin
        # is computed over the recorded runners even when selected_candidate
        # names nobody; the winner tax is reported unavailable.
        result = _result_contradictory()
        self.assertEqual(
            reporting.selection_margin(result),
            {"value": 1, "direction": "to_tie", "runner_up": "c1", "decisive_fixtures": ["f2"]},
        )
        payload = _read_json(
            reporting.save_report(copy.deepcopy(result), _fresh("contra"), include_evidence=False)[
                0
            ].read_bytes()
        )
        self.assertIsNone(payload["token_overhead"]["winner_tax_vs_cheapest_pct"])
        self.assertEqual(payload["token_overhead"]["cheapest_candidate"], "c1")


# ---------------------------------------------------------------------------
# Mutation gate: 6 mutants in /tmp copies (repo never mutated) + control.
# ---------------------------------------------------------------------------

_MUTANT_ROOT = _SCRATCH / "mutants"

# (name, [(old, new), ...], [check names that must catch it], prepend)
_MUTANTS = [
    (
        "m1_drop_sort",
        [
            (
                '"per_candidate": {name: per_candidate[name] for name in sorted(per_candidate)},',
                '"per_candidate": {name: per_candidate[name] for name in per_candidate},',
            )
        ],
        ["overhead"],
        "",
    ),
    (
        "m2_timestamp",
        [
            (
                '"<script>" + _SCRIPT + "</script>"',
                '"<script>" + _SCRIPT + "</script><!--" + str(time.time()) + "-->"',
            )
        ],
        ["determinism"],
        "import time\n",
    ),
    (
        "m3_margin_plus_one",
        [
            (
                '"value": len(failing[best]),\n            "direction": "to_tie",',
                '"value": len(failing[best]) + 1,\n            "direction": "to_tie",',
            )
        ],
        ["margin"],
        "",
    ),
    (
        "m4_fingerprint_threshold",
        [
            ("short = text[:19]", "short = text[:20]"),
            ("> 19 else short", "> 20 else short"),
        ],
        ["fingerprint"],
        "",
    ),
    (
        "m5_toc_includes_empty",
        [("        if html\n", "        if True  # mutant\n")],
        ["toc"],
        "",
    ),
    (
        "m6_tiebreak_max",
        [
            (
                "best = min(runners, key=lambda name: (len(failing[name]), name))",
                "best = max(runners, key=lambda name: (len(failing[name]), name))",
            )
        ],
        ["margin"],
        "",
    ),
]

_CHECKS = {
    "margin": check_margin_oracle,
    "overhead": check_overhead_oracle,
    "toc": check_toc,
    "fingerprint": check_fingerprint,
    "determinism": None,  # special-cased: needs a tmpdir
    "conformance": check_conformance_section,
    "divergence": check_divergence_section,
    "frontier": check_frontier_section,
    "sweep": check_sweep_section,
    "logprob": check_logprob_section,
    "banner": check_decision_banner,
    "json": check_json_public_view,
}


def _check_determinism_gate(mod, tc):
    """Check determinism gate."""
    tmpdir = Path(tempfile.mkdtemp(prefix="c75_mutdet_", dir=str(_SCRATCH)))
    try:
        check_determinism(mod, tc, tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class TestMutantGate(unittest.TestCase):
    """Each mutant must be caught by >=1 check; the control passes all."""

    @classmethod
    def setUpClass(cls):
        """Set up the test fixture."""
        if _MUTANT_ROOT.exists():
            shutil.rmtree(_MUTANT_ROOT)
        src = Path(reporting.__file__).read_text(encoding="utf-8")
        jobs = [("control", [], list(_CHECKS), "")]
        jobs.extend((pkg, repl, checks, pre) for pkg, repl, checks, pre in _MUTANTS)
        for pkg, replacements, _checks, prepend in jobs:
            pkgdir = _MUTANT_ROOT / pkg
            pkgdir.mkdir(parents=True)
            (pkgdir / "__init__.py").write_text("", encoding="utf-8")
            (pkgdir / "diffbackends.py").write_text(
                f"MAX_DIVERGENCE_OBSERVATIONS = {MAX_DIVERGENCE_OBSERVATIONS}\n",
                encoding="utf-8",
            )
            text = src
            for old, new in replacements:
                assert old in text, f"mutant anchor missing in {pkg}: {old!r}"
                assert text.count(old) == 1, f"mutant anchor ambiguous in {pkg}: {old!r}"
                text = text.replace(old, new, 1)
            (pkgdir / "reporting.py").write_text(prepend + text, encoding="utf-8")
        sys.path.insert(0, str(_MUTANT_ROOT))

    @classmethod
    def tearDownClass(cls):
        """Tear down the test fixture."""
        if str(_MUTANT_ROOT) in sys.path:
            sys.path.remove(str(_MUTANT_ROOT))
        shutil.rmtree(_MUTANT_ROOT, ignore_errors=True)

    def _run_checks(self, mod, names):
        """Run checks."""
        # NOTE: checks run against a bare TestCase probe, never against
        # ``self`` inside a subTest: on this interpreter a subTest nested
        # inside another subTest records its failure but does NOT propagate
        # the exception, which would let escaping mutants slip through.
        # A bare TestCase's subTest is a pass-through, so AssertionErrors
        # propagate normally here.
        probe = unittest.TestCase()
        for name in names:
            with self.subTest(check=name):
                if name == "determinism":
                    _check_determinism_gate(mod, probe)
                else:
                    _CHECKS[name](mod, probe)

    def test_control_copy_passes_all_checks(self):
        """Test control copy passes all checks."""
        mod = importlib.import_module("control.reporting")
        self._run_checks(mod, list(_CHECKS))

    def test_mutants_are_caught(self):
        """Test mutants are caught."""
        probe = unittest.TestCase()
        for pkg, _, check_names, _ in _MUTANTS:
            mod = importlib.import_module(f"{pkg}.reporting")
            caught = []
            for check_name in check_names:
                with self.subTest(mutant=pkg, check=check_name):
                    try:
                        if check_name == "determinism":
                            _check_determinism_gate(mod, probe)
                        else:
                            _CHECKS[check_name](mod, probe)
                    except AssertionError:
                        caught.append(check_name)
                    else:
                        self.fail(f"mutant {pkg} NOT caught by {check_name}")
            self.assertTrue(caught, f"mutant {pkg} escaped all checks")


def tearDownModule():
    """TearDownModule."""
    shutil.rmtree(_SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
