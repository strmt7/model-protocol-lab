"""Cycle 33: verification of model_protocol_lab/reporting.py's report internals.

Scope (deliberately NOT re-testing the cycle-22 evidence-leak fix itself,
which is already pinned by test_cycle22_reporting_verify.py): report
structure, section rendering, byte-determinism of the HTML/JSON outputs,
the self-contained-HTML property (no external assets), fixed-vocabulary
markers, summary computations rendered into the report, the JSON payload
shape produced by ``save_report``/``_public_view``/``attach_diagnostics``,
and hostile inputs (empty results, missing fields, adversarial strings --
HTML escaping is asserted exactly, not just "no exception").

Method: targeted exact-assertion tests plus a seeded fuzz
(``random.Random`` with string seeds; ``# noqa: S311`` like the other cycle
files) over hostile result dicts asserting determinism, escaping, and JSON
shape. Sensitivity is proven by mutation testing on COPIES of reporting.py
in /tmp/cycle33_mutants_<pid> (the repo is never mutated): 20 mutants, each
caught by at least one check, plus an unmutated control copy that must pass
every check.

BUGS FOUND
----------
None. Every pinned behavior holds on the real implementation.

DOCSTRING / CODE DIVERGENCES (code behavior pinned; source untouched)
---------------------------------------------------------------------
D1. (CYCLE 33 NOTE, FIXED IN CYCLE 33.) The code comment above the
    class x semantic matrix said "fixed row order (severity descending)".
    The rows actually follow the hardcoded ``class_order`` tuple in
    reporting.py (which happens to be severity-descending for the known
    classes, but the order is NOT derived from the section's ``severities``
    data), and divergence classes absent from that tuple are silently
    dropped from the matrix (their per-fixture callouts still render).
    The comment was clarified in cycle 33. Pinned by
    ``test_unknown_divergence_class_dropped_from_matrix``.
D2. ``_logprob_num_cell``'s docstring: "2-decimal logprob or an em dash".
    Only exact-``float`` values render numerically -- an ``int`` mean such
    as ``-1`` renders as the em dash, not "-1.00". Pinned by
    ``test_logprob_num_cell_int_renders_em_dash``.
D3. (CYCLE 33 NOTE, DOCUMENTED IN CYCLE 33.) ``_held_out_section`` renders
    "Not evaluated -- no candidate was promoted from search." for ANY
    non-dict ``held_out`` value, conflating "absent" with "present but
    malformed". The function had no docstring; one was added in cycle 33
    stating this conflation. Pinned by
    ``test_held_out_malformed_renders_not_evaluated``.
D4. ``_selection_margin``'s docstring says a failed/cancelled/uncomputable
    run returns None "so the key stays absent from results.json" -- as
    already documented in test_cycle22_reporting_verify.py, the key IS
    present with a null value. Not re-pinned here beyond the JSON key-order
    assertion.
D5. ``_sweep_section`` renders ``selected_candidate`` via ``str(...)``
    (so None becomes "None"); the "Vs native" column is driven by the
    ``winner_changed`` flag, not by comparing names. Pinned by
    ``test_sweep_section``.

MUTANT INVENTORY (each caught; control copy passes all checks)
--------------------------------------------------------------
M01 decision-class swap (compatible_on_fixture_set -> fail) -- banner check
M02 config section drops escape() on values -- hostile-escaping check
M03 id(result) comment appended to HTML -- determinism check
M04 _short_fingerprint threshold 19 -> 20 -- helper check
M05 _SCRIPT_SHA256 replaced with wrong constant -- CSP-consistency check
M06 _failure_diagnosis gate changed failed -> cancelled -- diagnosis check
M07 _sort_key "?" no longer maps to "" -- helper check
M08 severity badge high -> neutral -- divergence check
M09 divergence section embeds raw stop_texts -- withholding check
M10 _spark_svg level formula *8 -> *7 -- spark-geometry check
M11 _plural singular/plural boundary moved -- margin-line check
M12 token-overhead table reads max instead of mean -- overhead check
M13 footer disclaimer paragraph deleted -- structure check
M14 held-out cards Status/Pass-rate swapped -- held-out check
M15 search-row percent .0% -> .1% -- search-rows check
M16 outcome-grid pass glyph swapped -- grid check
M17 _public_view evidence flag inverted -- JSON-shape check
M18 _toc no longer skips empty sections -- toc check
M19 save_report ensure_ascii True -> False -- JSON-ascii check
M20 margin label drops the runner-up name -- margin-line check
"""

import base64
import copy
import hashlib
import importlib
import json
import os
import random
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from model_protocol_lab import reporting
from model_protocol_lab.diffbackends import MAX_DIVERGENCE_OBSERVATIONS

# ---------------------------------------------------------------------------
# Builders: small, explicit result dicts (no model calls, no network).
# ---------------------------------------------------------------------------


def _obs(fixture, exact=True, boundary=True, status="completed", **extra):
    """Obs."""
    obs = {
        "fixture": fixture,
        "status": status,
        "exact_match": exact,
        "boundary_found": boundary,
    }
    obs.update(extra)
    return obs


def _run(name, observations, fingerprint="f" * 64, boundary_ids=(1, 2), **summary_kw):
    """Run."""
    summary = {"rate": None, "status": "completed", "attempted": 0, "planned": 0}
    summary.update(summary_kw)
    return {
        "format": {"name": name},
        "candidate_fingerprint": fingerprint,
        "boundary_token_ids": list(boundary_ids),
        "summary": summary,
        "observations": observations,
    }


def _minimal_result():
    """Minimal result."""
    return {"decision": "ambiguous", "seed": 3, "search": []}


def _full_result():
    """Two candidates, mixed outcomes, held-out present: exercises every section."""
    return {
        "decision": "compatible_on_fixture_set",
        "selected_candidate": "chatml",
        "selection_rule": "conservative single-candidate",
        "measurement": "exact answer + explicit boundary",
        "seed": 7,
        "max_new_tokens": 64,
        "backend_fingerprint": "a" * 64,
        "statistical_confidence": "none claimed",
        "compute_comparison": "not equal-compute",
        "search": [
            _run(
                "chatml",
                [
                    _obs("f1", input_ids=[1, 2, 3], output_ids=[4, 5]),
                    _obs("f2", input_ids=[1, 2], output_ids=[4]),
                ],
                fingerprint="f" * 64,
                rate=1.0,
                exact_and_boundary_passes=2,
                planned=2,
                attempted=2,
                encoded_input_tokens=11,
                retained_output_tokens=5,
            ),
            _run(
                "llama3",
                [
                    _obs("f1", exact=False, input_ids=[9] * 5, output_ids=[7]),
                    _obs("f2", status="error", exact=False, boundary=False),
                ],
                fingerprint="e" * 64,
                rate=0.5,
                exact_and_boundary_passes=1,
                planned=2,
                attempted=2,
                encoded_input_tokens=9,
                retained_output_tokens=3,
            ),
        ],
        "held_out": {
            "summary": {
                "status": "completed",
                "rate": 1.0,
                "exact_and_boundary_passes": 2,
                "planned": 2,
                "aggregate_eligible": True,
            }
        },
    }


_HOSTILE_STRINGS = [
    "<script>alert(1)</script>",
    "</script><script>alert(2)</script>",
    '<img src=x onerror="alert(3)">',
    '"><svg onload=alert(4)>',
    "a&b<c>d\"e'f",
    "\u202efake",
    "<style>@import 'http://evil/x.css';</style>",
    "http://example.com/x",
    "https://example.com/y",
]

# Only payloads containing HTML-special characters must be absent raw from
# the HTML: html.escape() legitimately leaves e.g. U+202E untouched.
_NEEDS_ESCAPE = [s for s in _HOSTILE_STRINGS if any(c in s for c in '<>&"')]


# ---------------------------------------------------------------------------
# Shared checks. Each takes a reporting-like module so the same checks run
# against the real implementation and against /tmp mutant copies (PID-scoped).
# ---------------------------------------------------------------------------


def _check_determinism(rep):
    """Check determinism."""
    result = _full_result()
    before = copy.deepcopy(result)
    first = rep.render_html(result)
    second = rep.render_html(copy.deepcopy(result))
    assert first == second, "render_html is not byte-deterministic"
    assert result == before, "render_html mutated its input"
    with tempfile.TemporaryDirectory() as tmp:
        p1, h1 = rep.save_report(copy.deepcopy(result), Path(tmp) / "a")
        p2, h2 = rep.save_report(copy.deepcopy(result), Path(tmp) / "b")
        assert p1.read_bytes() == p2.read_bytes(), "results.json not byte-deterministic"
        assert h1.read_bytes() == h2.read_bytes(), "report.html not byte-deterministic"


def _check_self_contained(rep):
    """Check self contained."""
    html = rep.render_html(_full_result())
    assert "http://" not in html and "https://" not in html
    assert "url(" not in html and "@import" not in html
    assert "<link" not in html and "src=" not in html
    assert html.count("<script") == 1 and html.count("</script>") == 1
    assert "</script>" not in rep._SCRIPT, "script source must not contain </script>"
    assert "<script>" + rep._SCRIPT + "</script>" in html
    # The CSP allowlist hash must be the sha256 of the embedded script bytes.
    digest = base64.b64encode(hashlib.sha256(rep._SCRIPT.encode("utf-8")).digest()).decode()
    assert f"script-src 'sha256-{digest}'" in html
    assert f"script-src 'sha256-{rep._SCRIPT_SHA256}'" in html
    assert "default-src 'none'" in html
    assert "<style>" in html and "</style>" in html


def _check_decision_banner(rep):
    """Check decision banner."""
    expected = {
        "compatible_on_fixture_set": ("pass", "The sole search winner also passed"),
        "ambiguous": ("neutral", "The lab abstains"),
        "no_candidate": ("neutral", "The lab abstains; nothing"),
        "held_out_failed": ("fail", "did not pass the held-out cases"),
        "cancelled": ("neutral", "cancelled before completion"),
        "failed": ("fail", "failed before completion"),
    }
    for decision, (cls, help_frag) in expected.items():
        html = rep.render_html({"decision": decision, "search": []})
        assert f'<section class="decision {cls}" aria-label="Run decision">' in html
        assert f'<div class="decision-value">{decision}</div>' in html
        assert help_frag in html
    html = rep.render_html({"decision": "mystery", "search": []})
    assert '<section class="decision neutral"' in html
    assert "Unrecognized decision value." in html
    html = rep.render_html({"decision": None, "search": []})
    assert '<div class="decision-value">None</div>' in html  # str(None), still escaped


def _check_toc(rep):
    """Check toc."""
    full = _full_result()
    html = rep.render_html(full)
    sections = re.findall(r'<section id="([^"]+)"', html)
    links = re.findall(r'<a href="#([^"]+)">([^<]+)</a>', html)
    assert [anchor for anchor, _ in links] == sections
    assert "run-configuration" in sections and "held-out-stage" in sections
    # Sections whose data is absent render nothing and get no TOC entry.
    assert "template-conformance" not in sections
    assert "stop-behavior-matrix" not in sections
    assert rep._toc([("a", "A", ""), ("b", "B", "")]) == ""
    assert 'href="#b"' in rep._toc([("a", "A", ""), ("b", "B", "<x>")])


def _check_report_structure(rep):
    """Check report structure."""
    html = rep.render_html(_full_result())
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert '<html lang="en">' in html
    assert "<title>Model Protocol Lab report</title>" in html
    assert '<meta charset="utf-8">' in html
    assert "<h1>Model Protocol Lab</h1>" in html
    # Footer disclaimers are fixed text.
    assert "This report provides no confidence interval" in html
    assert "Raw prompts, answers and token traces are excluded from this HTML view." in html
    # Every rendered section has a matching h2.
    for section_id, title in re.findall(r'<section id="([^"]+)"><h2>([^<]+)</h2>', html):
        assert section_id and title


def _check_config_section(rep):
    """Check config section."""
    result = _full_result()
    html = rep._config_section(result)
    assert "<dt>Selection rule</dt><dd>conservative single-candidate</dd>" in html
    assert "<dt>Seed</dt><dd>7</dd>" in html
    assert "<dt>Candidates</dt><dd>2</dd>" in html
    # Same boundary for every candidate: one shared line.
    assert "<dt>Boundary token IDs</dt><dd>1, 2</dd>" in html
    assert "Boundary token IDs (chatml)" not in html
    assert "Full backend fingerprint" in html
    assert "a" * 64 in html  # full fingerprint in <details>, not just the short form
    # Differing boundaries: one line per candidate.
    alt = copy.deepcopy(result)
    alt["search"][1]["boundary_token_ids"] = [3, 4]
    html = rep._config_section(alt)
    assert "<dt>Boundary token IDs (chatml)</dt><dd>1, 2</dd>" in html
    assert "<dt>Boundary token IDs (llama3)</dt><dd>3, 4</dd>" in html
    # Hostile values are escaped.
    evil = {"decision": "x", "seed": "<b>1</b>", "search": []}
    html = rep._config_section(evil)
    assert "<b>1</b>" not in html and "&lt;b&gt;1&lt;/b&gt;" in html


def _check_search_rows(rep):
    """Check search rows."""
    result = _full_result()
    html = rep.render_html(result)
    # chatml: 100% pass.
    assert '<span class="badge pass">100%</span>' in html
    assert 'aria-label="100% pass rate"' in html
    assert '<span class="rate-text">2/2</span>' in html
    # llama3: 50% -> fail badge.
    assert '<span class="badge fail">50%</span>' in html
    assert '<span class="rate-text">1/2</span>' in html
    # Fingerprint: long -> first 19 chars + ellipsis in the <summary>;
    # the full value sits in the <details> body.
    assert '<summary class="mono">' + "f" * 19 + "\u2026</summary>" in html
    assert "f" * 64 in html
    # Incomplete (rate None) -> neutral badge, no bar.
    rows = rep._search_rows(
        {"search": [_run("n", [], rate=None, status="error", attempted=0, planned=2)]}
    )
    assert '<span class="badge neutral">incomplete</span>' in rows
    assert '<span class="bar"' not in rows
    assert "<td>0/2</td>" not in rows  # attempts cell keeps the data-sort + text form
    assert ">0/2</td>" in rows
    # Missing summary fields degrade to "?".
    rows = rep._search_rows({"search": [{"format": {"name": "q"}}]})
    assert "?/?" in rows
    assert '<span class="badge neutral">incomplete</span>' in rows
    # Empty search -> explicit row.
    assert "No candidates were evaluated in this run." in rep._search_rows({"search": []})


def _check_held_out(rep):
    """Check held out."""
    result = _full_result()
    html = rep._held_out_section(result)
    cards = [
        ("Status", "completed"),
        ("Pass rate", "100%"),
        ("Exact + boundary passes", "2/2"),
        ("Aggregate eligible", "Yes"),
    ]
    positions = [html.index(f'<div class="k">{k}</div><div class="v">{v}</div>') for k, v in cards]
    assert positions == sorted(positions), "held-out cards out of order"
    assert "Held-out observations are excluded from this HTML view" in html
    assert "Not evaluated &mdash; no candidate was promoted" in rep._held_out_section({})
    assert "incomplete" in rep._held_out_section(
        {"held_out": {"summary": {"status": "error", "rate": None}}}
    )
    assert ">No</div>" in rep._held_out_section(
        {"held_out": {"summary": {"aggregate_eligible": False}}}
    )


def _check_outcome_grid(rep):
    """Check outcome grid."""
    result = _full_result()
    html = rep._outcome_grid_section(result)
    # chatml f1 pass, f2 pass; llama3 f1 fail, f2 incomplete.
    assert html.count('<td data-sort="2"><span class="glyph-ok"') == 2
    assert html.count('<td data-sort="1"><span class="glyph-bad"') == 1
    assert html.count('<td data-sort="0"><span class="glyph-na"') == 1
    assert html.count("\u2713</span>") == 2  # pass glyphs, not the caption legend
    assert html.count("\u2717</span>") == 1
    assert html.count("\u2013</span>") == 1
    assert "chatml \u2605" in html  # winner star
    assert "llama3 \u2605" not in html
    assert "\u2713 exact answer and boundary" in html  # caption legend
    # No winner named -> no star on any header (the caption legend still names ★).
    html = rep._outcome_grid_section({"search": result["search"]})
    assert " \u2605</th>" not in html
    # Empty inputs -> no section.
    assert rep._outcome_grid_section({"search": []}) == ""
    assert rep._outcome_grid_section({}) == ""


def _check_decisive(rep):
    """Check decisive."""
    search = [
        _run("a", [_obs("f1"), _obs("f2", exact=False)]),
        _run("b", [_obs("f1", exact=False), _obs("f2", exact=False)]),
    ]
    html = rep._decisive_section({"search": search})
    assert "<li><strong>a</strong>: f1</li>" in html
    assert "f2</li>" not in html
    # An incomplete observation disqualifies the fixture.
    search[1]["observations"][0] = _obs("f1", status="error", exact=False, boundary=False)
    html = rep._decisive_section({"search": search})
    assert "No fixture was passed by exactly one candidate" in html
    assert rep._decisive_section({"search": []}) == ""


def _check_margin_line(rep):
    """Check margin line."""
    search = [
        _run("a", [_obs("f1"), _obs("f2")]),
        _run("b", [_obs("f1", exact=False), _obs("f2", exact=False)]),
    ]
    line = rep._selection_margin_line(
        {"decision": "compatible_on_fixture_set", "selected_candidate": "a", "search": search}
    )
    assert "Selection margin: 2 fixtures (runner-up: b)" in line
    assert "Flipping 2 of b&#x27;s failing fixtures (f1, f2) to pass would tie" in line
    assert "The margin is descriptive; it never changes the decision." in line
    line = rep._selection_margin_line({"decision": "no_candidate", "search": search})
    assert "Selection margin: 0 fixtures (closest: a)" in line
    amb = {"decision": "ambiguous", "search": search}
    # make both pass everything for the ambiguous path
    amb["search"][1]["observations"] = [_obs("f1"), _obs("f2")]
    line = rep._selection_margin_line(amb)
    assert "Selection margin: 1 candidate" in line
    assert "Demoting 1 candidate of the 2 tied candidates" in line
    # Single candidate -> null margin wording.
    line = rep._selection_margin_line(
        {"decision": "x", "selected_candidate": "solo", "search": [_run("solo", [_obs("f1")])]}
    )
    assert "Selection margin: n/a (single candidate)" in line
    assert rep._selection_margin_line({"decision": "failed", "search": search}) == ""
    # Hostile fixture ids are escaped inside the explainer.
    evil = {"decision": "no_candidate", "search": [_run("a<b>", [_obs("<i>", exact=False)])]}
    line = rep._selection_margin_line(evil)
    assert "<i>" not in line and "&lt;i&gt;" in line
    assert "a&lt;b&gt;" in line


def _check_token_overhead(rep):
    """Check token overhead."""
    search = [
        _run("a", [_obs("f1", input_ids=[1] * 10, output_ids=[2] * 3)]),
        _run("b", [_obs("f1", input_ids=[1] * 5, output_ids=[2] * 3)]),
    ]
    result = {"decision": "compatible_on_fixture_set", "selected_candidate": "a", "search": search}
    html = rep._token_overhead_section(result)
    assert ">10.0</td>" in html and ">5.0</td>" in html  # repr() of the float means
    assert "a token tax of 100.0%" in html
    assert "Winner token tax vs cheapest: a averages 10.0 prompt tokens vs 5.0 for b" in html
    html = rep._token_overhead_section(dict(result, selected_candidate="b"))
    assert "0.0% token tax" in html
    assert "is also the cheapest candidate" in html
    html = rep._token_overhead_section({"decision": "no_candidate", "search": search})
    assert "No candidate was selected, so no winner token tax is reported." in html
    html = rep._token_overhead_section(
        {"decision": "x", "selected_candidate": "a", "search": [_run("a", [])]}
    )
    assert "The winner's token tax is unavailable" in html
    assert (
        "Token overhead unavailable: no search candidates were recorded."
        in rep._token_overhead_section({"search": []})
    )
    assert "not equal-compute accounting." in html


def _check_sweep(rep):
    """Check sweep."""
    sweep = {
        "native_winner": "chatml",
        "native_decision": "compatible_on_fixture_set",
        "flips_at": [3],
        "depths": [
            {
                "drop_bytes": 1,
                "winner_changed": False,
                "decision": "compatible_on_fixture_set",
                "selected_candidate": "chatml",
            },
            {
                "drop_bytes": 3,
                "winner_changed": True,
                "decision": "ambiguous",
                "selected_candidate": None,
            },
        ],
    }
    html = rep._sweep_section({"sensitivity_sweep": sweep, "search": []})
    assert "Native winner: chatml (compatible_on_fixture_set)." in html
    assert "Depths where the winner or decision changed: [3]." in html
    assert '<span class="badge pass">stable</span>' in html
    assert '<span class="badge fail">changed</span>' in html
    assert ">None</td>" in html  # selected_candidate None renders as "None"
    assert "No depths were evaluated." in rep._sweep_section(
        {"sensitivity_sweep": {"depths": []}, "search": []}
    )
    assert rep._sweep_section({"search": []}) == ""


def _check_conformance(rep):
    """Check conformance."""
    conf = {
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
                "format": "llama3",
                "fixture": "f2",
                "hand_tokens": 12,
                "template_tokens": 13,
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
            {"format": "x", "fixture": "f3", "status": "error", "diff": []},
        ],
    }
    html = rep._conformance_section({"template_conformance": conf})
    assert '<span class="badge pass">identical</span>' in html
    assert "1 divergent segment(s)" in html
    assert "hand [0, 3] \u2194 template [0, 4]" in html
    assert "[1, 2, 3]" not in html  # token-ID subsequences withheld
    assert '<span class="badge neutral">unavailable</span>' in html
    assert "the token-ID subsequences themselves are withheld from this HTML view" in html
    html = rep._conformance_section({"template_conformance": {"status": "no_template_declared"}})
    assert "No chat template is declared" in html
    html = rep._conformance_section(
        {"template_conformance": {"status": "broken", "detail": "<d>&"}}
    )
    assert "Template-conformance check unavailable: &lt;d&gt;&amp;." in html
    assert rep._conformance_section({}) == ""
    assert "No format/fixture pairs were checked." in rep._conformance_section(
        {"template_conformance": {"status": "ok", "diffs": []}}
    )


def _check_stop_matrix(rep):
    """Check stop matrix."""
    matrix = {
        "semantics": [
            {"id": "native", "label": "Native", "description": "lab semantics"},
            {"id": "hf_stop_string", "label": "HF <s>", "description": "desc & more"},
        ],
        "search": [
            {
                "fixture": "f1",
                "native_pass": True,
                "cells": {"hf_stop_string": {"available": True, "pass": False, "fired": True}},
                "flips": ["hf"],
            },
            {
                "fixture": "f2",
                "native_pass": False,
                "cells": {"hf_stop_string": {"available": False, "detail": "no <data>"}},
                "flips": [],
            },
        ],
        "held_out": None,
    }
    html = rep._stop_matrix_section({"stop_matrix": matrix})
    assert "HF &lt;s&gt;" in html and "desc &amp; more" in html  # legend escaped
    assert '<span class="badge pass">pass</span>' in html
    assert '<span class="badge fail">fail</span>' in html
    assert " \u00b7 stop fired" in html
    assert '<span class="badge neutral">n/a</span> (no &lt;data&gt;)' in html
    assert "<h3>Search stage</h3>" in html
    assert "Held out stage" not in html
    assert rep._stop_matrix_section({"search": []}) == ""
    assert rep._stop_matrix_section({"stop_matrix": {"semantics": []}}) == ""
    # _matrix_cell_html variants.
    assert rep._matrix_cell_html(None) == '<span class="badge neutral">n/a</span> (unavailable)'
    assert rep._matrix_cell_html("x") == '<span class="badge neutral">n/a</span> (unavailable)'
    assert '<span class="badge pass">pass</span>' in rep._matrix_cell_html(
        {"available": True, "pass": True}
    )
    cell = rep._matrix_cell_html({"available": 1, "pass": 0, "fired": 0})
    assert '<span class="badge fail">fail</span>' in cell and "stop fired" not in cell


def _divergence_section_data(**kw):
    """Divergence section data."""
    section = {
        "verdict": "diverge",
        "pairs_verdict": "3 of 12 pairs diverge",
        "stop_texts": ["the expected answer"],
        "matrix": {
            "EOS_PRECEDENCE": {"hf_stop_string": 1},
            "KEPT_VS_STRIPPED": {"hf_stop_string": 2, "vllm_holdback_strip": 0},
            "NOT_A_REAL_CLASS": {"hf_stop_string": 9},
        },
        "severities": {"KEPT_VS_STRIPPED": "high", "EOS_PRECEDENCE": "mystery"},
        "per_fixture": [
            {
                "fixture": "<f1>",
                "divergences": [
                    {
                        "class": "KEPT_VS_STRIPPED",
                        "severity": "high",
                        "stop_text": "secret answer",
                        "detail": "raw generated text",
                    }
                ],
            }
        ],
    }
    section.update(kw)
    return section


def _check_divergence(rep):
    """Check divergence."""
    res = {
        "decision": "compatible_on_fixture_set",
        "backend_divergences": _divergence_section_data(),
    }
    html = rep._backend_divergence_section(res)
    assert '<span class="badge fail">diverge</span>' in html
    assert "3 of 12 pairs diverge." in html
    # Raw stop text / detail never rendered; fixed marker instead.
    assert "the expected answer" not in html
    assert "secret answer" not in html and "raw generated text" not in html
    assert "[stop text and diagnostic detail withheld from this HTML view]" in html
    assert "Stop text(s):" in html and "[withheld from this HTML view]" in html
    # Severity badges: high -> fail red; unknown -> neutral.
    assert '<span class="badge fail">high</span>' in html
    assert '<span class="badge neutral">mystery</span>' in html
    # Fixed class row order (KEPT before EOS) regardless of dict insertion order;
    # unknown classes are dropped from the matrix rows.
    assert html.index("KEPT_VS_STRIPPED") < html.index("EOS_PRECEDENCE")
    assert "NOT_A_REAL_CLASS" not in html
    # Column order is fixed.
    heads = re.findall(r'<th scope="col">([^<]+)</th>', html)
    assert heads == [
        "Divergence class",
        "HF stop string",
        "vLLM strip",
        "vLLM keep",
        "SGLang stop",
        "EOS-appended",
    ]
    # Fixture names escaped in callouts.
    assert "&lt;f1&gt;" in html
    # agree verdict -> pass badge.
    html = rep._backend_divergence_section(
        {"backend_divergences": _divergence_section_data(verdict="agree")}
    )
    assert '<span class="badge pass">agree</span>' in html
    # Observation-bound note uses the real constant.
    html = rep._backend_divergence_section(
        {
            "backend_divergences": _divergence_section_data(
                observations_truncated=True, observations_total=120
            )
        }
    )
    assert (
        "the run recorded 120 observations; only the first "
        f"{MAX_DIVERGENCE_OBSERVATIONS} were differential-tested."
    ) in html
    # Non-True truncation flag -> no note (exact-True check).
    html = rep._backend_divergence_section(
        {"backend_divergences": _divergence_section_data(observations_truncated=1)}
    )
    assert "Observation bound reached" not in html
    # Missing / malformed -> "".
    assert rep._backend_divergence_section({}) == ""
    assert rep._backend_divergence_section({"backend_divergences": {"verdict": "x"}}) == ""


def _check_logprob(rep):
    """Check logprob."""
    diag = {
        "available": True,
        "thresholds": {"derailed_mean_logprob_at_least": -0.5, "unknown_mean_logprob_below": -3.0},
        "candidates": [
            {
                "name": "<c>",
                "aggregate": {
                    "native_passes": 2,
                    "total_logprob": -1.234,
                    "mean_logprob": -0.617,
                    "scored_token_count": 2,
                },
                "interpretation": {"kind": "derailed", "text": "looks <b>derailed</b>"},
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
                        "status": "unavailable",
                        "reason": "no_alignment",
                        "mean_logprob": None,
                    },
                ],
            }
        ],
    }
    html = rep._logprob_section({"logprob_diagnostic": diag, "search": []})
    assert "mean \u2265 -0.5 nats/token" in html  # declared bands note
    assert "-0.62" in html and "-1.23" in html  # 2-decimal cells
    assert "&lt;b&gt;derailed&lt;/b&gt;" in html
    assert "&lt;c&gt;" in html
    assert "unavailable (no_alignment)" in html
    assert "\u2713" in html and "\u2717" in html
    assert "Per-fixture scores: &lt;c&gt;" in html
    html = rep._logprob_section(
        {"logprob_diagnostic": {"available": False, "detail": "<d>"}, "search": []}
    )
    assert "Log-likelihood diagnostic unavailable: &lt;d&gt;." in html
    html = rep._logprob_section(
        {"logprob_diagnostic": {"available": True, "candidates": []}, "search": []}
    )
    assert "Log-likelihood diagnostic unavailable: no reason recorded." in html
    assert rep._logprob_section({"search": []}) == ""


def _check_frontier(rep):
    """Check frontier."""
    frontier = {
        "max_new_tokens": 4,
        "budget_robust": False,
        "decision_steps": [[1, "no_candidate", None], [3, "compatible_on_fixture_set", "chatml"]],
        "flip_ranges": [[1, 2]],
        "native_winner": "chatml",
        "candidates": {
            "chatml": {
                "fixture_ids": ["f1", "f2"],
                "min_budget_per_fixture": [1, 3],
                "full_pass_budget": 3,
                "pass_steps": [[1, 1], [3, 2]],
            }
        },
    }
    html = rep._frontier_section({"budget_frontier": frontier, "search": []})
    assert '<span class="badge fail">budget-sensitive</span>' in html
    assert "budgets 1&ndash;2: no_candidate" in html
    assert "Winner pass curve: chatml" in html
    assert "Pass curves by candidate" in html
    assert 'aria-label="pass count 1 to 2 of 2 over budgets 1 to 4"' in html
    # Exact spark geometry: counts [1,1,2,2] of 2 -> levels 4,4,8,8.
    assert '<rect x="3" y="0" width="0.9" height="8"/>' in html
    html = rep._frontier_section(
        {"budget_frontier": dict(frontier, budget_robust=True), "search": []}
    )
    assert '<span class="badge pass">stable</span>' in html
    assert "1&ndash;4" in html
    assert rep._frontier_section({"search": []}) == ""
    assert rep._frontier_section({"budget_frontier": {"candidates": []}, "search": []}) == ""
    # An empty (but dict) candidates map still renders the section shell.
    assert '<section id="token-budget-frontier">' in rep._frontier_section(
        {"budget_frontier": {"candidates": {}}, "search": []}
    )


def _check_spark_helpers(rep):
    """Check spark helpers."""
    assert rep._sparkline([0, 5, 10], 10) == "\u2581\u2585\u2588"
    assert rep._sparkline([1, 2, 3], 3) == "\u2583\u2586\u2588"
    assert rep._sparkline([], 10) == "?"
    assert rep._sparkline([1], 0) == "?"
    svg = rep._spark_svg([0, 2, 4], 4, 'a"b<c>')
    # levels: 0 -> y=8 h=0; 4/8 -> y=4 h=4; 8/8 -> y=0 h=8.
    assert '<rect x="0" y="8" width="0.9" height="0"/>' in svg
    assert '<rect x="1" y="4" width="0.9" height="4"/>' in svg
    assert '<rect x="2" y="0" width="0.9" height="8"/>' in svg
    assert 'width="9"' in svg and 'viewBox="0 0 3 8"' in svg
    assert 'aria-label="a&quot;b&lt;c&gt;"' in svg
    assert rep._spark_svg([], 4, "x") == "?"


def _check_taxonomy(rep):
    """Check taxonomy."""
    taxonomy = {
        "per_candidate": {
            "a": {"exact_mismatch": 2, "boundary_miss": 1},
            "b": {"exact_mismatch": 0},
        },
        "held_out": {"exact_mismatch": 1},
    }
    note = rep._taxonomy_note({"decision": "no_candidate", "failure_taxonomy": taxonomy})
    assert note == "Failure-mode taxonomy: boundary_miss \u00d71, exact_mismatch \u00d73."
    assert (
        rep._taxonomy_note({"decision": "compatible_on_fixture_set", "failure_taxonomy": taxonomy})
        == ""
    )
    assert rep._taxonomy_note(
        {"decision": "failed", "failure_taxonomy": {"per_candidate": {}}}
    ) == (
        "Failure-mode taxonomy: the run did not complete, so there are no observations to classify."
    )
    assert rep._taxonomy_note({"decision": "no_candidate", "failure_taxonomy": {"x": 1}}) == (
        "Failure-mode taxonomy: no failing observations recorded."
    )
    assert rep._taxonomy_note({"decision": "x"}) == ""
    # Non-int / non-positive counts are ignored.
    note = rep._taxonomy_note(
        {"decision": "x", "failure_taxonomy": {"per_candidate": {"a": {"m": "2", "n": -1}}}}
    )
    assert note == "Failure-mode taxonomy: no failing observations recorded."


def _check_failure_diagnosis(rep):
    """Check failure diagnosis."""
    result = {
        "decision": "failed",
        "search": [
            {"summary": {"failure_code": "generate_failed"}},
            {"summary": {"failure_code": "generate_failed"}},  # duplicate: named once
            {"summary": {"failure_code": "mystery_code"}},
            {"summary": {}},
        ],
        "held_out": {"summary": {"failure_code": "render_failed"}},
    }
    line = rep._failure_diagnosis(result)
    assert line.startswith("Failure diagnosis: ")
    assert line.index("generate_failed") < line.index("mystery_code") < line.index("render_failed")
    assert line.count("generate_failed:") == 1
    assert "Generation failed inside the backend." in line
    assert "The backend reported this failure; re-run doctor" in line  # unknown code fallback
    assert "A fixture prompt could not be rendered." in line
    # Only for decision "failed".
    assert rep._failure_diagnosis({"decision": "ambiguous", "search": result["search"]}) == ""
    assert rep._failure_diagnosis({"decision": "failed", "search": []}) == ""
    assert rep._failure_diagnosis({"decision": "failed"}) == ""
    # Every documented code has guidance and renders.
    for code in rep._FAILURE_GUIDANCE:
        line = rep._failure_diagnosis(
            {"decision": "failed", "search": [{"summary": {"failure_code": code}}]}
        )
        assert line.startswith(f"Failure diagnosis: {code}: ")
    assert set(rep._FAILURE_GUIDANCE) == {
        "context_limit_exceeded",
        "render_failed",
        "encode_failed",
        "invalid_input_ids",
        "generate_failed",
        "backend_reported_error",
        "invalid_or_empty_input",
        "invalid_output_cap",
        "invalid_seed",
        "invalid_cancellation_event",
    }


def _check_misc_helpers(rep):
    """Check misc helpers."""
    assert rep._short_fingerprint("x" * 19) == "x" * 19
    assert rep._short_fingerprint("x" * 20) == "x" * 19 + "\u2026"
    assert rep._short_fingerprint("abc") == "abc"
    assert rep._short_fingerprint("") == ""
    assert rep._short_fingerprint(123) == ""
    assert rep._short_fingerprint(None) == ""
    assert rep._sort_key(3) == "3"
    assert rep._sort_key(2.5) == "2.5"
    assert rep._sort_key("?") == ""
    assert rep._sort_key("ab") == "ab"
    assert rep._plural(1, "fixture") == "1 fixture"
    assert rep._plural(0, "fixture") == "0 fixtures"
    assert rep._plural(2, "candidate") == "2 candidates"
    # Logprob cells: exact float -> 2 decimals; anything else -> em dash.
    assert "-0.62" in rep._logprob_num_cell(-0.617)
    assert rep._logprob_num_cell(None) == '<td data-sort="">\u2014</td>'
    assert rep._logprob_num_cell(-1) == '<td data-sort="">\u2014</td>'  # int -> em dash (D2)
    assert rep._logprob_num_cell("x") == '<td data-sort="">\u2014</td>'


def _check_json_shape(rep):
    """Check json shape."""
    result = _minimal_result()
    with tempfile.TemporaryDirectory() as tmp:
        json_path, _html_path = rep.save_report(copy.deepcopy(result), Path(tmp) / "r")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        raw = json_path.read_bytes()
    # Exact top-level key order: input keys, then diagnostics, then the flag.
    assert list(payload.keys()) == [
        "decision",
        "seed",
        "search",
        "selection_margin",
        "token_overhead",
        "raw_evidence_included",
    ]
    assert payload["raw_evidence_included"] is False
    assert payload["selection_margin"] == rep.selection_margin(result)
    assert payload["token_overhead"] == rep.token_overhead(result)
    # ensure_ascii=True: the file is pure ASCII even with non-ASCII input.
    assert raw.isascii(), "results.json is not ASCII-only"
    with tempfile.TemporaryDirectory() as tmp:
        json_path, _ = rep.save_report(
            {"decision": "caf\u00e9 \u2603", "seed": 1, "search": []}, Path(tmp) / "r"
        )
        assert json_path.read_bytes().isascii(), "results.json is not ASCII-only"
        assert "\\u2603" in json_path.read_text(encoding="utf-8")
    # include_evidence=True keeps observations and sets the flag.
    with tempfile.TemporaryDirectory() as tmp:
        json_path, _ = rep.save_report(
            copy.deepcopy(_full_result()), Path(tmp) / "r", include_evidence=True
        )
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["raw_evidence_included"] is True
    assert "observations" in json.dumps(payload["search"])
    # NaN is rejected (allow_nan=False).
    with tempfile.TemporaryDirectory() as tmp, unittest.TestCase().assertRaises(ValueError):
        rep.save_report({"decision": "x", "seed": float("nan")}, Path(tmp) / "r")
    # attach_diagnostics does not mutate the caller's dict (top level).
    result = _minimal_result()
    keys_before = set(result)
    rep.attach_diagnostics(result)
    assert set(result) == keys_before
    # _public_view returns an independent copy.
    public = rep._public_view(result, include_evidence=False)
    public["search"].append({"x": 1})
    assert result["search"] == []


def _check_save_report_io(rep):
    """Check save report io."""
    result = _minimal_result()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        json_path, html_path = rep.save_report(result, tmp / "r1")
        assert json_path == tmp / "r1" / "results.json"
        assert html_path == tmp / "r1" / "report.html"
        for path in (json_path, html_path):
            assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        assert stat.S_IMODE(os.stat(tmp / "r1").st_mode) == 0o700
        # Refuses to overwrite: O_EXCL.
        with unittest.TestCase().assertRaises(FileExistsError):
            rep.save_report(result, tmp / "r1")
        # HTML file content equals render_html of the enriched result.
        assert html_path.read_text(encoding="utf-8") == rep.render_html(
            rep.attach_diagnostics(result)
        )
    # prepare_report_dir: fresh nested parents all get 0o700.
    with tempfile.TemporaryDirectory() as tmp:
        target = rep.prepare_report_dir(Path(tmp) / "a" / "b" / "c")
        for component in (Path(tmp) / "a", Path(tmp) / "a" / "b", target):
            assert stat.S_IMODE(os.stat(component).st_mode) == 0o700
        with unittest.TestCase().assertRaises(FileExistsError):
            rep.prepare_report_dir(target)  # already exists and is empty: still refused
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "empty"
        target.mkdir()
        with unittest.TestCase().assertRaises(FileExistsError):
            rep.prepare_report_dir(target)
        claimed = rep.prepare_report_dir(target, _allow_empty_existing=True)
        assert claimed == target


def _check_hostile_escaping(rep):
    """Check hostile escaping."""
    evil_name = _HOSTILE_STRINGS[0]
    result = {
        "decision": _HOSTILE_STRINGS[1],
        "selected_candidate": evil_name,
        "seed": _HOSTILE_STRINGS[4],
        "search": [
            _run(
                evil_name,
                [_obs(_HOSTILE_STRINGS[2], input_ids=[1], output_ids=[2])],
                fingerprint=_HOSTILE_STRINGS[3],
                rate=1.0,
                exact_and_boundary_passes=1,
                planned=1,
                attempted=1,
                status=_HOSTILE_STRINGS[5],
            )
        ],
        "held_out": {"summary": {"status": _HOSTILE_STRINGS[6]}},
    }
    html = rep.render_html(result)
    # The only script element is the real one; no payload smuggles a tag.
    assert html.count("<script") == 1 and html.count("</script>") == 1
    assert "<script>" + rep._SCRIPT + "</script>" in html
    for payload in _NEEDS_ESCAPE:
        assert payload not in html, f"raw hostile payload in HTML: {payload!r}"
    # Spot-checks that escaping (not stripping) happened.
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;img src=x onerror=" in html
    # Attribute injection via data-sort is neutralized (quote=True).
    assert 'data-sort="a&quot;b' in rep._search_rows(
        {"search": [_run('a"b', [], rate=1.0, attempted=1, planned=1)]}
    )
    # JSON stays ASCII and parseable with hostile unicode.
    with tempfile.TemporaryDirectory() as tmp:
        json_path, _ = rep.save_report(result, Path(tmp) / "r")
        raw = json_path.read_bytes()
        assert raw.isascii(), "results.json is not ASCII-only"
        json.loads(raw)


def _check_empty_and_missing(rep):
    """Check empty and missing."""
    html = rep.render_html({})
    assert '<div class="decision-value"></div>' in html
    assert "No candidates were evaluated in this run." in html
    assert "Not evaluated &mdash; no candidate was promoted" in html
    assert "Token overhead unavailable: no search candidates were recorded." in html
    # The TOC links exactly the rendered sections (held-out always renders).
    assert 'class="toc"' in html
    # Missing/odd shapes degrade gracefully.
    html = rep.render_html({"decision": "x", "search": "garbage", "held_out": [1, 2]})
    assert "No candidates were evaluated in this run." in html
    assert "Not evaluated &mdash; no candidate was promoted" in html  # D3
    # Missing/odd shapes degrade gracefully. A truthy non-dict "format"
    # raises AttributeError (documented in test_cycle22_reporting_verify.py,
    # discrepancy 3); falsy/missing shapes are skipped.
    html = rep.render_html({"decision": "x", "search": [None, 42]})
    assert "No candidates were evaluated in this run." in html
    with unittest.TestCase().assertRaises(AttributeError):
        rep.render_html({"decision": "x", "search": [{"format": "nope"}]})
    assert rep.render_html({"decision": "x", "search": [{"format": {"name": ""}}]})
    # Duplicate candidate names / fixture ids keep the first record.
    dup = {
        "decision": "compatible_on_fixture_set",
        "selected_candidate": "a",
        "search": [
            _run("a", [_obs("f1")]),
            _run("a", [_obs("f1", exact=False)]),  # duplicate name: ignored
        ],
    }
    html = rep._outcome_grid_section(dup)
    assert html.count('<td data-sort="2"><span class="glyph-ok"') == 1


_CHECKS = {
    "determinism": _check_determinism,
    "self_contained": _check_self_contained,
    "decision_banner": _check_decision_banner,
    "toc": _check_toc,
    "report_structure": _check_report_structure,
    "config_section": _check_config_section,
    "search_rows": _check_search_rows,
    "held_out": _check_held_out,
    "outcome_grid": _check_outcome_grid,
    "decisive": _check_decisive,
    "margin_line": _check_margin_line,
    "token_overhead": _check_token_overhead,
    "sweep": _check_sweep,
    "conformance": _check_conformance,
    "stop_matrix": _check_stop_matrix,
    "divergence": _check_divergence,
    "logprob": _check_logprob,
    "frontier": _check_frontier,
    "spark_helpers": _check_spark_helpers,
    "taxonomy": _check_taxonomy,
    "failure_diagnosis": _check_failure_diagnosis,
    "misc_helpers": _check_misc_helpers,
    "json_shape": _check_json_shape,
    "save_report_io": _check_save_report_io,
    "hostile_escaping": _check_hostile_escaping,
    "empty_and_missing": _check_empty_and_missing,
}


class TestReportInternals(unittest.TestCase):
    """Run every shared check against the real implementation."""

    def test_all_checks(self):
        """Test all checks."""
        for name, check in _CHECKS.items():
            with self.subTest(check=name):
                check(reporting)


class TestDecisionVocabulary(unittest.TestCase):
    """Test decision vocabulary."""

    def test_all_six_decisions_have_banner_class(self):
        """Test all six decisions have banner class."""
        expected = {
            "compatible_on_fixture_set": "pass",
            "ambiguous": "neutral",
            "no_candidate": "neutral",
            "held_out_failed": "fail",
            "cancelled": "neutral",
            "failed": "fail",
        }
        for decision, cls in expected.items():
            with self.subTest(decision=decision):
                html = reporting.render_html({"decision": decision, "search": []})
                self.assertIn(f'<section class="decision {cls}"', html)
                # The decision value itself is escaped in the banner.
                self.assertIn(f'<div class="decision-value">{decision}</div>', html)


class TestSummaryComputations(unittest.TestCase):
    """Summary numbers rendered into the report match independent recomputation."""

    def test_search_row_percent_matches_summary(self):
        """Test search row percent matches summary."""
        rng = random.Random("cycle33-summary")  # noqa: S311 -- deterministic test seed
        for trial in range(25):
            with self.subTest(trial=trial):
                passes = rng.randint(0, 9)
                planned = rng.randint(1, 9)
                passes = min(passes, planned)
                rate = passes / planned
                rows = reporting._search_rows(
                    {
                        "search": [
                            _run(
                                "c",
                                [],
                                rate=rate,
                                exact_and_boundary_passes=passes,
                                planned=planned,
                                attempted=planned,
                            )
                        ]
                    }
                )
                self.assertIn(f">{passes}/{planned}</span>", rows)
                if rate == 1:
                    self.assertIn('<span class="badge pass">100%</span>', rows)
                else:
                    self.assertIn('<span class="badge fail">', rows)

    def test_held_out_pass_rate_card(self):
        """Test held out pass rate card."""
        html = reporting._held_out_section(
            {
                "held_out": {
                    "summary": {
                        "status": "completed",
                        "rate": 0.75,
                        "exact_and_boundary_passes": 3,
                        "planned": 4,
                        "aggregate_eligible": True,
                    }
                }
            }
        )
        self.assertIn('<div class="k">Pass rate</div><div class="v">75%</div>', html)
        self.assertIn('<div class="k">Exact + boundary passes</div><div class="v">3/4</div>', html)

    def test_margin_value_matches_failing_count(self):
        """Test margin value matches failing count."""
        rng = random.Random("cycle33-margin")  # noqa: S311 -- deterministic test seed
        for trial in range(25):
            with self.subTest(trial=trial):
                fixtures = [f"f{i}" for i in range(rng.randint(1, 4))]
                obs_a = [_obs(f, exact=rng.random() < 0.7) for f in fixtures]
                obs_b = [_obs(f, exact=rng.random() < 0.7) for f in fixtures]
                result = {
                    "decision": "compatible_on_fixture_set",
                    "selected_candidate": "a",
                    "search": [_run("a", obs_a), _run("b", obs_b)],
                }
                margin = reporting.selection_margin(result)
                failing_b = sum(1 for o in obs_b if not (o["exact_match"] and o["boundary_found"]))
                self.assertEqual(margin["value"], failing_b)
                self.assertEqual(margin["direction"], "to_tie")
                self.assertEqual(margin["runner_up"], "b")
                line = reporting._selection_margin_line(result)
                self.assertIn(f"Selection margin: {failing_b} fixture", line)

    def test_token_overhead_means_match_recomputation(self):
        """Test token overhead means match recomputation."""
        counts = [4, 8, 12]
        result = {
            "decision": "x",
            "search": [
                _run(
                    "a",
                    [
                        _obs(f"f{i}", input_ids=list(range(c)), output_ids=[0] * 2)
                        for i, c in enumerate(counts)
                    ],
                )
            ],
        }
        overhead = reporting.token_overhead(result)
        self.assertEqual(overhead["per_candidate"]["a"]["mean_prompt_tokens"], 8.0)
        self.assertEqual(overhead["per_candidate"]["a"]["max_prompt_tokens"], 12)
        self.assertEqual(overhead["per_candidate"]["a"]["mean_output_tokens"], 2.0)
        self.assertEqual(overhead["cheapest_candidate"], "a")


# ---------------------------------------------------------------------------
# Fuzz: seeded hostile result dicts -> determinism + escaping + JSON shape.
# ---------------------------------------------------------------------------


def _fuzz_string(rng, hostile_pool):
    """Fuzz string."""
    choices = ["alpha", "beta", "f1", "chatml", ""] + hostile_pool
    return rng.choice(choices)


class TestFuzzReportInternals(unittest.TestCase):
    """Test fuzz report internals."""

    @staticmethod
    # skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
    def _random_result(rng, hostile_pool):
        """Random result."""
        decisions = [
            "compatible_on_fixture_set",
            "ambiguous",
            "no_candidate",
            "held_out_failed",
            "cancelled",
            "failed",
            "weird",
            None,
        ]
        used = []

        def hs():
            """Hs."""
            s = _fuzz_string(rng, hostile_pool)
            if s in _NEEDS_ESCAPE:
                used.append(s)
            return s

        search = []
        for ci in range(rng.randint(0, 3)):
            name = hs() or f"cand{ci}"
            observations = []
            for fi in range(rng.randint(0, 3)):
                status = rng.choice(["completed", "completed", "error"])
                observations.append(
                    {
                        "fixture": hs() or f"f{fi}",
                        "status": status,
                        "exact_match": rng.random() < 0.6,
                        "boundary_found": rng.random() < 0.6,
                        "input_ids": [rng.randint(0, 999)] * rng.randint(0, 6),
                        "output_ids": [rng.randint(0, 999)] * rng.randint(0, 4),
                    }
                )
            planned = max(1, len(observations))
            passes = sum(1 for o in observations if o["exact_match"] and o["boundary_found"])
            search.append(
                {
                    "format": {"name": name},
                    "candidate_fingerprint": hs() or "fp",
                    "boundary_token_ids": [rng.randint(0, 999) for _ in range(rng.randint(0, 3))],
                    "summary": {
                        "rate": passes / planned if observations else None,
                        "exact_and_boundary_passes": passes,
                        "planned": planned,
                        "attempted": len(observations),
                        "status": hs() or "completed",
                    },
                    "observations": observations,
                }
            )
        result = {
            "decision": rng.choice(decisions),
            "selected_candidate": hs() or None,
            "seed": rng.randint(0, 2**31),
            "search": search,
        }
        if rng.random() < 0.5:
            result["held_out"] = {
                "summary": {
                    "status": hs() or "completed",
                    "rate": rng.choice([None, 0.0, 0.5, 1.0]),
                    "exact_and_boundary_passes": rng.randint(0, 3),
                    "planned": 3,
                    "aggregate_eligible": rng.random() < 0.5,
                }
            }
        if rng.random() < 0.4:
            result["failure_taxonomy"] = {
                "per_candidate": {hs() or "c": {"m1": rng.randint(0, 3)} for _ in range(2)}
            }
        if rng.random() < 0.3:
            result["backend_divergences"] = _divergence_section_data(
                stop_texts=[hs() or "stop"],
                per_fixture=[
                    {
                        "fixture": hs() or "f",
                        "divergences": [
                            {
                                "class": rng.choice(["KEPT_VS_STRIPPED", "BOGUS"]),
                                "severity": rng.choice(["high", "low", "weird"]),
                                "stop_text": hs() or "s",
                                "detail": hs() or "d",
                            }
                        ],
                    }
                ],
            )
        if rng.random() < 0.3:
            result["budget_frontier"] = {
                "max_new_tokens": rng.randint(1, 60),
                "budget_robust": rng.random() < 0.5,
                "decision_steps": [[1, "no_candidate", None]],
                "flip_ranges": [[1, 2]],
                "native_winner": hs() or "c",
                "candidates": {
                    hs() or "c": {
                        "fixture_ids": ["f1", "f2"],
                        "min_budget_per_fixture": [1, 3],
                        "full_pass_budget": 3,
                        "pass_steps": [[1, 1], [3, 2]],
                    }
                },
            }
        return result, used

    def test_fuzz_determinism_and_escaping(self):
        """Test fuzz determinism and escaping."""
        for trial in range(150):
            with self.subTest(trial=trial):
                rng = random.Random(f"cycle33-fuzz-{trial}")  # noqa: S311 -- deterministic seed
                result, used = self._random_result(rng, _HOSTILE_STRINGS)
                first = reporting.render_html(copy.deepcopy(result))
                second = reporting.render_html(copy.deepcopy(result))
                self.assertEqual(first, second)
                # Only the genuine script element may exist.
                self.assertEqual(first.count("<script"), 1)
                self.assertEqual(first.count("</script>"), 1)
                for payload in used:
                    self.assertNotIn(payload, first)

    def test_fuzz_json_shape(self):
        """Test fuzz json shape."""
        for trial in range(60):
            with self.subTest(trial=trial):
                rng = random.Random(f"cycle33-fuzzjson-{trial}")  # noqa: S311 -- deterministic seed
                result, _ = self._random_result(rng, _HOSTILE_STRINGS)
                include_evidence = trial % 2 == 0
                with tempfile.TemporaryDirectory() as tmp:
                    p1, _ = reporting.save_report(
                        copy.deepcopy(result), Path(tmp) / "a", include_evidence=include_evidence
                    )
                    p2, _ = reporting.save_report(
                        copy.deepcopy(result), Path(tmp) / "b", include_evidence=include_evidence
                    )
                    raw1, raw2 = p1.read_bytes(), p2.read_bytes()
                self.assertEqual(raw1, raw2)
                self.assertTrue(raw1.isascii(), "results.json is not ASCII-only")
                payload = json.loads(raw1)
                self.assertIs(payload["raw_evidence_included"], include_evidence)
                self.assertIn("selection_margin", payload)
                self.assertIn("token_overhead", payload)


# ---------------------------------------------------------------------------
# Mutation-style sensitivity controls: copies in /tmp, never the repo.
# ---------------------------------------------------------------------------

# PID-scoped so two pytest processes running this file concurrently stage
# into disjoint directories (a fixed name raced -> ModuleNotFoundError).
_MUTANT_ROOT = Path(tempfile.gettempdir()) / f"cycle33_mutants_{os.getpid()}"

_MUTANTS = [
    (
        "m01_decision_class",
        [
            (
                '    "compatible_on_fixture_set": "pass",\n',
                '    "compatible_on_fixture_set": "fail",\n',
            )
        ],
        ["decision_banner"],
    ),
    (
        "m02_config_escape",
        [
            (
                '<dt>" + _escape_text(key) + "</dt><dd>" + _escape_text(value) + "</dd>',
                '<dt>" + _escape_text(key) + "</dt><dd>" + value + "</dd>',
            )
        ],
        ["hostile_escaping", "config_section"],
    ),
    (
        "m03_determinism",
        [
            (
                '"<script>" + _SCRIPT + "</script>"',
                '"<script>" + _SCRIPT + "</script><!--" + str(id(result)) + "-->"',
            )
        ],
        ["determinism"],
    ),
    (
        "m04_fingerprint",
        [("short = text[:19]", "short = text[:20]"), ("> 19 else short", "> 20 else short")],
        ["misc_helpers", "search_rows"],
    ),
    (
        "m05_csp_hash",
        [("_SCRIPT_SHA256 = base64.b64encode(", '_SCRIPT_SHA256 = "QUJD"\n# ')],
        ["self_contained"],
    ),
    (
        "m06_diagnosis_gate",
        [
            (
                'if str(result.get("decision", "")) != "failed":',
                'if str(result.get("decision", "")) != "cancelled":',
            )
        ],
        ["failure_diagnosis"],
    ),
    (
        "m07_sort_key",
        [('return "" if text == "?" else text', "return text")],
        ["misc_helpers"],
    ),
    (
        "m08_severity_badge",
        [
            (
                'severity_badge = {"high": "fail",',
                'severity_badge = {"high": "neutral",',
            )
        ],
        ["divergence"],
    ),
    (
        "m09_stop_text_embed",
        [
            (
                "[withheld from this HTML view]</span>.</p>'",
                "' + escape(\", \".join(stop_texts)) + '</span>.</p>'",
            )
        ],
        ["divergence"],
    ),
    (
        "m10_spark_svg",
        [("round(count / total * 8))", "round(count / total * 7))")],
        ["spark_helpers", "frontier"],
    ),
    (
        "m11_plural",
        [('("" if count == 1 else "s")', '("" if count == 0 else "s")')],
        ["misc_helpers", "margin_line"],
    ),
    (
        "m12_overhead_stat",
        [
            (
                'for key in ("mean_prompt_tokens", "max_prompt_tokens",',
                'for key in ("max_prompt_tokens", "max_prompt_tokens",',
            )
        ],
        ["token_overhead"],
    ),
    (
        "m13_footer",
        [("<p>Raw prompts, answers and token traces are excluded", "<p>")],
        ["report_structure"],
    ),
    (
        "m14_held_out_cards",
        [
            (
                '("Status", str(summary.get("status", "?"))),\n'
                '            ("Pass rate", rate_text),',
                '("Pass rate", rate_text),\n'
                '            ("Status", str(summary.get("status", "?"))),',
            )
        ],
        ["held_out"],
    ),
    (
        "m15_search_percent",
        [('percent = f"{rate:.0%}"', 'percent = f"{rate:.1%}"')],
        ["search_rows"],
    ),
    (
        "m16_grid_glyph",
        [
            (
                '<td data-sort="2"><span class="glyph-ok" title="pass">\\u2713</span></td>',
                '<td data-sort="2"><span class="glyph-ok" title="pass">\\u2717</span></td>',
            )
        ],
        ["outcome_grid"],
    ),
    (
        "m17_evidence_flag",
        [("    if not include_evidence:", "    if include_evidence:")],
        ["json_shape"],
    ),
    (
        "m18_toc_filter",
        [
            (
                "        for anchor, title, html in entries\n        if html\n",
                "        for anchor, title, html in entries\n",
            )
        ],
        ["toc"],
    ),
    (
        "m19_json_ascii",
        [
            (
                "ensure_ascii=True, allow_nan=False, indent=2",
                "ensure_ascii=False, allow_nan=False, indent=2",
            )
        ],
        ["json_shape"],
    ),
    (
        "m20_margin_label",
        [
            (
                'label = f"Selection margin: {unit} (runner-up: {runner})"',
                'label = f"Selection margin: {unit}"',
            )
        ],
        ["margin_line"],
    ),
]


class TestMutantSensitivity(unittest.TestCase):
    """Each mutant must be caught by at least one shared check; the control passes all."""

    @classmethod
    def setUpClass(cls):
        """Set up the test fixture."""
        # PID-scoped (module constant): two pytest processes staging mutants
        # concurrently must not share a directory (a fixed name raced: one
        # process's rmtree deleted the other's in-use packages ->
        # ModuleNotFoundError).
        cls.root = _MUTANT_ROOT
        if cls.root.exists():
            shutil.rmtree(cls.root)
        src = Path(reporting.__file__).read_text(encoding="utf-8")
        # Sanity: every anchor below must match the real source exactly,
        # otherwise a mutant would silently test nothing.
        jobs = [("control", [], list(_CHECKS))]
        jobs.extend(_MUTANTS)
        for pkg, replacements, _ in jobs:
            pkgdir = cls.root / pkg
            pkgdir.mkdir(parents=True)
            (pkgdir / "__init__.py").write_text("", encoding="utf-8")
            (pkgdir / "diffbackends.py").write_text(
                f"MAX_DIVERGENCE_OBSERVATIONS = {MAX_DIVERGENCE_OBSERVATIONS}\n", encoding="utf-8"
            )
            text = src
            for old, new in replacements:
                assert old in text, f"mutant anchor missing in {pkg}: {old!r}"
                assert text.count(old) == 1, f"mutant anchor ambiguous in {pkg}: {old!r}"
                text = text.replace(old, new, 1)
            (pkgdir / "reporting.py").write_text(text, encoding="utf-8")
        sys.path.insert(0, str(cls.root))

    @classmethod
    def tearDownClass(cls):
        """Tear down the test fixture."""
        if str(cls.root) in sys.path:
            sys.path.remove(str(cls.root))
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_control_copy_passes_all_checks(self):
        """Test control copy passes all checks."""
        mod = importlib.import_module("control.reporting")
        for name, check in _CHECKS.items():
            with self.subTest(check=name):
                check(mod)

    def test_mutants_are_caught(self):
        """Test mutants are caught."""
        for pkg, _, check_names in _MUTANTS:
            mod = importlib.import_module(f"{pkg}.reporting")
            for check_name in check_names:
                with self.subTest(mutant=pkg, check=check_name), self.assertRaises(AssertionError):
                    _CHECKS[check_name](mod)


class TestCrossProcessDeterminism(unittest.TestCase):
    """PYTHONHASHSEED must not affect the rendered report bytes."""

    def test_render_stable_across_hash_seeds(self):
        """Test render stable across hash seeds."""
        script = (
            "import copy, hashlib, json, sys; sys.path.insert(0, '.');"
            "from model_protocol_lab import reporting;"
            "result = {'decision': 'ambiguous', 'seed': 3,"
            " 'search': [{'format': {'name': 'a'}, 'candidate_fingerprint': 'f'*64,"
            " 'boundary_token_ids': [1, 2],"
            " 'summary': {'rate': 1.0, 'attempted': 2, 'planned': 2},"
            " 'observations': [{'fixture': 'f1', 'status': 'completed',"
            " 'exact_match': True, 'boundary_found': True}]}]};"
            "print(hashlib.sha256(reporting.render_html(result).encode()).hexdigest())"
        )
        digests = set()
        env = dict(os.environ)
        for seed in ("0", "999"):
            env["PYTHONHASHSEED"] = seed
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env=env,
                cwd=str(Path(reporting.__file__).resolve().parent.parent),
                check=True,
            )
            digests.add(proc.stdout.strip())
        self.assertEqual(len(digests), 1)


if __name__ == "__main__":
    unittest.main()
