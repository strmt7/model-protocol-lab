"""Cycle 58 adversarial audit: HTML escaping in the report renderers.

Renderers audited
-----------------
1. ``model_protocol_lab.reporting.render_html`` -- the self-contained run-report
   HTML page (sections: decision banner, config, search candidates, outcome
   grid, decisive fixtures, token overhead, template conformance, stop matrix,
   backend divergence, budget frontier, sensitivity sweep, logprob diagnostic,
   held-out stage).
2. ``model_protocol_lab.reporting.save_report`` -- writes ``report.html``
   (via ``render_html``) and ``results.json`` to disk.
3. ``examples/make_sample_report.py`` -- the sample-report generator; it calls
   ``save_report``, so it shares renderer (1). Its output is audited too.
4. ``model_protocol_lab.diffreports.render_summary`` -- terminal-only drift
   summary; verified below to have NO HTML entry point.

Method
------
Fixture text (fixture ids, candidate names, decision strings, failure codes,
conformance diffs, stop-matrix labels, divergence severities/classes,
sweep values, logprob interpretations, frontier candidate names, config
values, fingerprints, selection-margin runner-ups) is UNTRUSTED input, so
each payload below is injected into every renderer-visible field. The audit
then asserts, per payload, that the raw payload string does not appear in
the HTML source while its ``html.escape`` form does, and runs a parse-based
audit (``html.parser.HTMLParser``) over the generated markup asserting:

* exactly one ``<script>`` tag (the renderer's own table-sorter),
* no ``on*`` event-handler attributes on any element,
* no ``javascript:`` (or other dangerous scheme) attribute values,
* no ``img``/``iframe``/``object``/``embed``/``textarea``/``base`` tags,
* the renderer's own ``<svg>`` sparklines only carry allow-listed attributes.

FINDINGS
--------
No REAL escaping holes found: every payload rendered only in escaped form in
all sections, and the parse-based audit found no executable markup. A
Content-Security-Policy meta tag (``script-src 'none'``) is additionally
present as defense in depth.

Mutation gate (run in a /tmp copy of the repo, never in the repo itself):
* mutant A (``escape`` replaced by identity) -- CAUGHT: payload-escaping
  tests fail.
* mutant B (escape only ``<``, not ``>``) -- CAUGHT: raw ``>`` payloads and
  the parser audit fail.
* mutant C (double ``html.escape``) -- CAUGHT by the no-corruption test
  (``&amp;lt;`` appears).
"""

import html
import importlib.util
import json
import unittest
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory

from model_protocol_lab import diffreports
from model_protocol_lab.reporting import render_html, save_report

REPO_ROOT = Path(__file__).resolve().parent.parent

# (label, raw payload). Payloads cover: script tags, event handlers,
# javascript: URLs, textarea/title breakouts, HTML comments, CSS
# expression(), dotless-i / fullwidth unicode tag tricks, attribute
# injection via quotes, and mixed-case tags.
PAYLOADS = [
    ("script_tag", "<script>alert(1)</script>"),
    ("img_handler", "<img src=x onerror=alert(2)>"),
    ("js_url", "javascript:alert(3)"),
    ("textarea_breakout", "</textarea><script>alert(4)</script>"),
    ("html_comment", "<!-- <script>alert(5)</script> -->"),
    ("css_expression", '<div style="x:expression(alert(6))">'),
    ("dotless_i", "<scrıpt>alert(7)</scrıpt>"),  # U+0131 dotless i
    ("fullwidth", "＜script＞alert(8)＜/script＞"),  # fullwidth, inert but checked
    ("attr_injection", '" onmouseover="alert(9)'),
    ("quote_break", "'><svg onload=alert(10)>"),
    ("title_breakout", "</title><script>alert(11)</script>"),
    ("svg_payload", "<svg onload=alert(12)>"),
    ("line_sep", "a <script>alert(13)</script>b"),  # U+2028 line separator
    ("mixed_case", "<ScRiPt>alert(14)</ScRiPt>"),
]

DANGEROUS_TAGS = {"img", "iframe", "object", "embed", "textarea", "base", "link", "form"}

# Only these attributes can carry a navigable/executable URL. Generic
# attributes such as data-sort hold inert text (the table sorter reads them
# via getAttribute and only compares/reorders rows), so a "javascript:"
# *string* there is not an executable vector.
URL_ATTRS = {
    "href",
    "src",
    "action",
    "formaction",
    "xlink:href",
    "cite",
    "data",
    "poster",
    "srcset",
    "background",
}

SVG_ATTR_ALLOW = {"class", "width", "height", "viewbox", "role", "aria-label"}
RECT_ATTR_ALLOW = {"x", "y", "width", "height", "fill"}
# Attribute surface measured from the chart builders' actual output
# (frontier chart, margin trajectory, sparklines): every dynamic value in
# these attributes is an integer from validated counts or a fixed palette
# literal -- hostile strings can only ever appear as escaped <text>
# content, never in attributes.
G_ATTR_ALLOW = {"transform"}
PATH_ATTR_ALLOW = {"d", "fill", "fill-opacity", "stroke", "stroke-width"}
LINE_ATTR_ALLOW = {"x1", "y1", "x2", "y2", "stroke", "stroke-width", "stroke-dasharray"}
TEXT_ATTR_ALLOW = {"x", "y", "fill", "font-size", "text-anchor", "dominant-baseline"}
CIRCLE_ATTR_ALLOW = {"cx", "cy", "r", "fill"}
# Elements whose fill could reference an external paint server.
_FILL_URL_CHECK_TAGS = {"rect", "path", "circle"}


class _MarkupAudit(HTMLParser):
    """Collect tags/attributes and flag anything executable."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.tags = []
        self.problems = []

    def _check(self, tag, attrs):
        self.tags.append((tag, attrs))
        if tag in DANGEROUS_TAGS:
            self.problems.append(f"dangerous tag <{tag}>")
        for name, value in attrs:
            lname = name.lower()
            if lname.startswith("on"):
                self.problems.append(f"event handler attribute {name!r} on <{tag}>")
            if (
                lname in URL_ATTRS
                and value
                and value.strip().lower().startswith(("javascript:", "vbscript:", "data:text/html"))
            ):
                self.problems.append(f"dangerous URL scheme in {name!r} on <{tag}>")
        if tag == "svg":
            for name, _ in attrs:
                if name.lower() not in SVG_ATTR_ALLOW:
                    self.problems.append(f"unexpected <svg> attribute {name!r}")
        if tag == "rect":
            for name, _ in attrs:
                if name.lower() not in RECT_ATTR_ALLOW:
                    self.problems.append(f"unexpected <rect> attribute {name!r}")
        for chart_tag, allowed in (
            ("g", G_ATTR_ALLOW),
            ("path", PATH_ATTR_ALLOW),
            ("line", LINE_ATTR_ALLOW),
            ("text", TEXT_ATTR_ALLOW),
            ("circle", CIRCLE_ATTR_ALLOW),
        ):
            if tag == chart_tag:
                for name, _ in attrs:
                    if name.lower() not in allowed:
                        self.problems.append(f"unexpected <{tag}> attribute {name!r}")
        if tag in _FILL_URL_CHECK_TAGS:
            for name, value in attrs:
                # fill is a non-executable presentation attribute, but a
                # url() payload could reference an external resource.
                if name.lower() == "fill" and value and value.strip().lower().startswith("url("):
                    self.problems.append(f"external paint reference in <{tag}> fill")
        if tag == "meta":
            attr = {n.lower(): (v or "") for n, v in attrs}
            if "http-equiv" in attr and attr["http-equiv"].lower() != "content-security-policy":
                self.problems.append(f"unexpected meta http-equiv {attr['http-equiv']!r}")

    def handle_starttag(self, tag, attrs):
        self._check(tag, attrs)

    def handle_startendtag(self, tag, attrs):
        self._check(tag, attrs)


def _audit_markup(testcase, rendered, label):
    """Parse-based audit: no executable markup may be present."""
    audit = _MarkupAudit()
    audit.feed(rendered)
    audit.close()
    testcase.assertEqual(audit.problems, [], f"{label}: executable markup found: {audit.problems}")
    script_tags = [t for t, _ in audit.tags if t == "script"]
    testcase.assertEqual(
        len(script_tags), 1, f"{label}: expected exactly the renderer's own <script>"
    )
    # Note: the literal text "javascript:..." may legitimately appear as
    # *escaped text content* (payload rendered inert); the executable vector
    # -- a javascript: URL in an attribute -- is covered by the parser's
    # attribute-value check above.


def _search_run(name, fixture, extra_summary=None):
    summary = {
        "rate": 1.0,
        "status": "completed",
        "exact_and_boundary_passes": 1,
        "planned": 1,
        "attempted": 1,
        "encoded_input_tokens": 10,
        "retained_output_tokens": 4,
    }
    if extra_summary:
        summary.update(extra_summary)
    return {
        "format": {"name": name},
        "summary": summary,
        "boundary_token_ids": [124],
        "candidate_fingerprint": f"fp-{name}",
        "observations": [
            {
                "fixture": fixture,
                "status": "completed",
                "exact_match": True,
                "boundary_found": True,
            }
        ],
    }


def build_result(payload):
    """Result dict with ``payload`` injected into every renderer-visible field."""
    other = "zz-other-candidate"
    return {
        "decision": "failed",  # exercises _failure_diagnosis via failure_code
        "selected_candidate": payload,
        "selection_rule": payload,
        "measurement": payload,
        "seed": 7,
        "max_new_tokens": 8,
        "statistical_confidence": payload,
        "compute_comparison": payload,
        "backend_fingerprint": payload,
        "search": [
            _search_run(payload, payload, {"failure_code": payload}),
            _search_run(other, "zz-fixture-b"),
        ],
        "template_conformance": {
            "status": "mismatch",
            "detail": payload,
            "diffs": [
                {
                    "status": "mismatch",
                    "format": payload,
                    "fixture": payload,
                    "hand_tokens": [1],
                    "template_tokens": [2],
                    "diff": [{"hand_span": payload, "template_span": payload}],
                }
            ],
        },
        "stop_matrix": {
            "semantics": [
                {"id": "native", "label": "native", "description": "baseline"},
                {"id": "emu", "label": payload, "description": payload},
            ],
            "search": [
                {
                    "fixture": payload,
                    "native_pass": True,
                    "cells": {"emu": {"available": True, "pass": True, "fired": False}},
                    "flips": [payload],
                }
            ],
        },
        "backend_divergences": {
            "pairs_verdict": payload,
            "per_fixture": [],
            "matrix": {},
            "severities": [payload],
        },
        "sensitivity_sweep": {
            "depths": [
                {
                    "drop_bytes": 1,
                    "winner_changed": False,
                    "decision": payload,
                    "selected_candidate": payload,
                }
            ],
            "native_winner": payload,
            "native_decision": payload,
            "flips_at": payload,
        },
        "logprob_diagnostic": {
            "available": True,
            "thresholds": {
                "derailed_mean_logprob_at_least": -0.5,
                "unknown_mean_logprob_below": -3.0,
            },
            "candidates": [
                {
                    "name": payload,
                    "aggregate": {
                        "native_passes": 1,
                        "scored_token_count": 4,
                        "total_logprob": -1.0,
                        "mean_logprob": -0.25,
                    },
                    "interpretation": {"kind": payload, "text": payload},
                    "fixture_scores": [
                        {
                            "fixture": payload,
                            "native_pass": True,
                            "status": "scored",
                            "reason": "",
                            "mean_logprob": -0.25,
                        }
                    ],
                }
            ],
        },
        "budget_frontier": {
            "max_new_tokens": 8,
            # Hostile pass_steps/decision_steps/native_winner: the frontier
            # chart SVG, margin SVG, decision ribbon, and winner curve must
            # all render with hostile data (finding 4: the audit previously
            # never executed these paths with hostile input).
            "native_winner": payload,
            "decision_steps": [[1, payload, payload], [5, "selected", payload]],
            "candidates": {
                payload: {
                    "fixture_ids": ["f1"],
                    "pass_steps": [[1, 0], [3, 1], [8, 1]],
                    "min_budget_per_fixture": [3],
                    "full_pass_budget": 3,
                }
            },
        },
        "held_out": {
            "summary": {
                "status": payload,
                "rate": 1.0,
                "exact_and_boundary_passes": 1,
                "planned": 1,
                "failure_code": payload,
                "aggregate_eligible": True,
            }
        },
    }


def build_kitchen_sink():
    """One result with a DIFFERENT payload in each field (cross-field check)."""
    names = [label for label, _ in PAYLOADS]
    by_label = dict(PAYLOADS)

    def p(i):
        return by_label[names[i % len(names)]]

    result = {
        "decision": "compatible_on_fixture_set",  # exercises the margin line
        "selected_candidate": p(0),
        "selection_rule": p(1),
        "measurement": p(2),
        "statistical_confidence": p(3),
        "compute_comparison": p(4),
        "backend_fingerprint": p(5),
        "search": [
            _search_run(p(6), p(7)),
            _search_run(p(8), p(9)),
        ],
        "template_conformance": {
            "status": "mismatch",
            "detail": p(10),
            "diffs": [
                {
                    "status": "mismatch",
                    "format": p(11),
                    "fixture": p(12),
                    "hand_tokens": [1],
                    "template_tokens": [2],
                    "diff": [{"hand_span": p(13), "template_span": p(0)}],
                }
            ],
        },
        "stop_matrix": {
            "semantics": [
                {"id": "native", "label": "native", "description": "baseline"},
                {"id": "emu", "label": p(1), "description": p(2)},
            ],
            "search": [
                {
                    "fixture": p(3),
                    "native_pass": False,
                    "cells": {"emu": {"available": True, "pass": False, "fired": True}},
                    "flips": [p(4)],
                }
            ],
        },
        "backend_divergences": {
            "pairs_verdict": p(5),
            "per_fixture": [],
            "matrix": {},
            "severities": [p(6)],
        },
        "sensitivity_sweep": {
            "depths": [
                {
                    "drop_bytes": 2,
                    "winner_changed": True,
                    "decision": p(7),
                    "selected_candidate": p(8),
                }
            ],
            "native_winner": p(9),
            "native_decision": p(10),
            "flips_at": p(11),
        },
        "logprob_diagnostic": {
            "available": True,
            "thresholds": {
                "derailed_mean_logprob_at_least": -0.5,
                "unknown_mean_logprob_below": -3.0,
            },
            "candidates": [
                {
                    "name": p(12),
                    "aggregate": {
                        "native_passes": 1,
                        "scored_token_count": 4,
                        "total_logprob": -1.0,
                        "mean_logprob": -0.25,
                    },
                    "interpretation": {"kind": p(13), "text": p(0)},
                    "fixture_scores": [
                        {
                            "fixture": p(1),
                            "native_pass": False,
                            "status": "scored",
                            "reason": "",
                            "mean_logprob": -0.25,
                        }
                    ],
                }
            ],
        },
        "budget_frontier": {
            "max_new_tokens": 8,
            # Hostile chart inputs (see build_result): the frontier SVG, the
            # margin SVG, the decision ribbon, and the winner curve render
            # with hostile candidate names, decisions, and winners.
            "native_winner": p(2),
            "decision_steps": [[1, p(7), p(2)], [5, "selected", p(2)]],
            "candidates": {
                p(2): {
                    "fixture_ids": ["f1"],
                    "pass_steps": [[1, 0], [3, 1], [8, 1]],
                    "min_budget_per_fixture": [3],
                    "full_pass_budget": 3,
                }
            },
        },
        "boundary_audit": {
            # Hostile family names: the canonicality table must escape them.
            "available": True,
            "n_observations": 4,
            "n_usable": 4,
            "n_unusable": 0,
            "n_problem": 1,
            "overall_rate": 0.25,
            "per_candidate": {},
            "per_family": {
                p(14): {"n_usable": 2, "n_problem": 1, "rate": 0.5},
                p(15): {"n_usable": 2, "n_problem": 0, "rate": 0.0},
            },
        },
        "marker_isolation": {
            # Hostile family and marker names: the isolation matrix must
            # escape them.
            "status": "ok",
            "special_token_count": 2,
            "n_probes": 2,
            "n_isolated": 1,
            "overall_isolation_rate": 0.5,
            "family_scores": {
                p(16): {"n_probes": 2, "n_isolated": 1, "isolation_rate": 0.5},
            },
            "probes": [
                {
                    "format": p(16),
                    "fixture": "f1",
                    "marker": p(17),
                    "status": "ok",
                    "baseline": 2,
                    "probed": 3,
                    "delta": 1,
                    "isolated": False,
                },
                {
                    "format": p(16),
                    "fixture": "f1",
                    "marker": "[INST]",
                    "status": "ok",
                    "baseline": 2,
                    "probed": 2,
                    "delta": 0,
                    "isolated": True,
                },
            ],
        },
        "template_firewall": {
            # Hostile rule/detail strings: the firewall section must escape
            # them. A denied verdict exercises the findings table.
            "status": "denied",
            "detail": "template_firewall_denied",
            "findings": [
                {"rule": p(18), "detail": p(19)},
                {"rule": "dunder_getattr", "detail": "dunder attribute access"},
            ],
        },
        "held_out": {
            "summary": {
                "status": p(3),
                "rate": 0.5,
                "exact_and_boundary_passes": 1,
                "planned": 2,
                "aggregate_eligible": False,
            }
        },
    }
    return result, set(by_label.values())


class HtmlEscapingAuditTests(unittest.TestCase):
    def test_every_payload_escaped_in_every_field(self):
        for label, payload in PAYLOADS:
            with self.subTest(payload=label):
                rendered = render_html(build_result(payload))
                escaped = html.escape(payload)
                if escaped != payload:
                    self.assertNotIn(payload, rendered, f"raw payload {label!r} present unescaped")
                self.assertIn(escaped, rendered, f"escaped payload {label!r} missing")

    def test_kitchen_sink_payloads_all_escaped(self):
        result, payload_set = build_kitchen_sink()
        rendered = render_html(result)
        for payload in payload_set:
            escaped = html.escape(payload)
            if escaped != payload:
                self.assertNotIn(payload, rendered, f"raw payload present: {payload[:40]!r}")
            self.assertIn(escaped, rendered)

    def test_parse_based_audit_kitchen_sink(self):
        result, _ = build_kitchen_sink()
        _audit_markup(self, render_html(result), "render_html")

    def test_raw_dangerous_substrings_absent(self):
        result, _ = build_kitchen_sink()
        rendered = render_html(result)
        # Exactly one <script> open/close pair: the renderer's own sorter.
        self.assertEqual(rendered.count("<script"), 1)
        self.assertEqual(rendered.count("</script>"), 1)
        # Raw tag breakouts must never survive; "onerror="/ "onload=" as *text*
        # are inert once escaped (covered per-payload above), so only the
        # parse-based audit judges attribute context.
        for token in ("</textarea>", "<img", "</title><script", "<svg onload"):
            self.assertNotIn(token, rendered, f"dangerous token present: {token!r}")

    def test_save_report_html_escaped_and_json_roundtrips(self):
        result, payload_set = build_kitchen_sink()
        with TemporaryDirectory() as tmp:
            json_path, html_path = save_report(result, Path(tmp) / "report")
            rendered = Path(html_path).read_text(encoding="utf-8")
            _audit_markup(self, rendered, "save_report report.html")
            for payload in payload_set:
                escaped = html.escape(payload)
                if escaped != payload:
                    self.assertNotIn(payload, rendered)
                self.assertIn(escaped, rendered)
            # JSON output: separate escaping domain -- values must round-trip
            # verbatim through the parsed structure (proving JSON escaping
            # is intact and independent of the HTML escaping).
            parsed = json.loads(Path(json_path).read_text(encoding="utf-8"))

            def _strings(node):
                if isinstance(node, str):
                    yield node
                elif isinstance(node, dict):
                    for value in node.values():
                        yield from _strings(value)
                elif isinstance(node, list):
                    for item in node:
                        yield from _strings(item)

            values = set(_strings(parsed))
            for payload in payload_set:
                self.assertIn(payload, values, f"payload lost in JSON: {payload[:40]!r}")

    def test_legitimate_content_not_corrupted(self):
        legit = {
            "unicode": "café naïve — Δ",
            "special_tokens": "<|begin_of_text|><|start_header_id|>user",
            "code": "if a < b and c > d: print('x & y')",
            "quotes": "say \"hi\" & 'bye'",
        }
        result = build_result("zz")
        # Inject into fields known to render (decision="failed" renders no
        # selected_candidate line, so use search/config fields instead).
        result["backend_fingerprint"] = legit["unicode"]
        result["search"][0]["format"]["name"] = legit["code"]
        result["search"][0]["observations"][0]["fixture"] = legit["quotes"]
        result["search"][1]["format"]["name"] = legit["special_tokens"]
        rendered = render_html(result)
        for value in legit.values():
            self.assertIn(html.escape(value), rendered)
        # No double-escaping of already-escaped entities.
        self.assertNotIn("&amp;lt;", rendered)
        self.assertNotIn("&amp;gt;", rendered)
        self.assertNotIn("&amp;amp;", rendered)

    def test_csp_meta_present(self):
        rendered = render_html(build_result("zz"))
        self.assertIn('http-equiv="Content-Security-Policy"', rendered)

    def test_sample_report_html_passes_audit(self):
        script = REPO_ROOT / "examples" / "make_sample_report.py"
        spec = importlib.util.spec_from_file_location("make_sample_report", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as tmp:
            module.build_sample_report(Path(tmp) / "sample")
            rendered = (Path(tmp) / "sample" / "report.html").read_text(encoding="utf-8")
            _audit_markup(self, rendered, "sample report.html")
            self.assertEqual(rendered.count("<script"), 1)

    def test_real_chart_fills_are_fixed_literals(self):
        # The old version of this test fed hand-written markup to the
        # audit's own parser -- it tested the harness, not the code. Now a
        # real frontier chart renders with a hostile candidate name and the
        # audit runs over the actual builder output: every fill must be a
        # fixed literal (no url() paint references), and the chart elements
        # must carry only allowlisted attributes.
        from model_protocol_lab.reporting import _frontier_chart_svg

        hostile = "<script>alert(1)</script>"
        frontier = {
            "max_new_tokens": 8,
            "native_winner": hostile,
            "decision_steps": [[1, hostile, hostile], [5, "selected", hostile]],
            "candidates": {
                hostile: {
                    "fixture_ids": ["f1"],
                    "pass_steps": [[1, 0], [3, 1], [8, 1]],
                    "min_budget_per_fixture": [3],
                    "full_pass_budget": 3,
                }
            },
        }
        rendered = _frontier_chart_svg(frontier)
        self.assertIn("<svg", rendered)
        self.assertNotIn(hostile, rendered)
        audit = _MarkupAudit()
        audit.feed(rendered)
        audit.close()
        self.assertEqual(audit.problems, [], audit.problems)
        # And the url() check itself is not vacuous: a hostile fill is
        # still flagged when one is present.
        audit2 = _MarkupAudit()
        audit2.feed('<svg><circle cx="1" cy="1" r="2" fill="url(http://evil/x)"/></svg>')
        audit2.close()
        self.assertTrue(
            any("external paint reference" in p for p in audit2.problems),
            audit2.problems,
        )

    def test_diffreports_has_no_html_entry_point(self):
        self.assertFalse(hasattr(diffreports, "render_html"))
        source = Path(diffreports.__file__).read_text(encoding="utf-8")
        self.assertNotIn("<!doctype", source.lower())
        self.assertNotIn("<html", source.lower())
        # render_summary is terminal text with a hostile source label: the
        # existing ANSI-escape sanitization must hold (separate from HTML).
        hostile = "\x1b[31m<script>alert(1)</script>\x1b[0m"
        diff = {
            "old": {"source": hostile},
            "new": {"source": "run-b.json"},
            "config_drift": [],
            "decision": {"old": "failed", "new": hostile},
            "winner": {"old": None, "new": None},
            "frontier": {"available": "neither", "changed": False},
            "held_out_floor": {"available": "neither", "changed": False},
            "boundary_audit": {"available": "neither", "changed": False},
            "candidate_verdicts": [],
            "evidence": {"old": False, "new": False},
            "summary": {
                "identical": False,
                "decision_changed": True,
                "winner_changed": False,
                "frontier_changed": False,
                "frontier_asymmetric": False,
                "candidates_changed_status": 0,
                "fixtures_flipped": 0,
                "config_drift_warnings": 0,
            },
            "fixture_diffs": [],
        }
        summary = diffreports.render_summary(diff)
        self.assertIsInstance(summary, str)
        self.assertNotIn("\x1b", summary)


if __name__ == "__main__":
    unittest.main()
