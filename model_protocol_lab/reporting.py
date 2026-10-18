"""Local JSON and escaped HTML reports with no external assets."""

import base64
import contextlib
import hashlib
import json
import math
import os
from bisect import bisect_right
from html import escape
from pathlib import Path
from typing import Any

# Table sorting: the only script in the report, and the zero-JS -> tiny-JS
# change. The report is fully readable with scripting disabled (plain
# tables, <details> accordions, SVG sparklines); with scripting enabled,
# headers of <table class="sortable"> sort their rows on click using each
# cell's data-sort key (numeric when both keys parse as numbers). Vanilla
# JS, no frameworks, no network, no new DOM nodes -- it only reorders rows
# and sets aria-sort (▲▼ indicators are pure CSS ::after). It is allowlisted
# in the Content-Security-Policy by its SHA-256 hash (computed below), so
# the policy stays byte-deterministic and the script cannot be swapped
# without invalidating the hash. The source must never contain the literal
# "</script>" sequence, which would break out of the script element.
_SCRIPT = """/* Sortable tables (progressive enhancement): inert without JavaScript --
   the report is unchanged and fully readable; with it, <table class="sortable">
   headers sort rows on click by each cell's data-sort key (numeric when both
   keys parse as numbers). No frameworks, no network, no new DOM nodes. */
(function () {
  if (!document.querySelectorAll) return;
  document.documentElement.classList.add("has-js");
  function key(cell) {
    if (!cell) return "";
    var k = cell.getAttribute("data-sort");
    return k === null ? cell.textContent.trim() : k;
  }
  function cmp(x, y) {
    var a = parseFloat(x), b = parseFloat(y);
    if (x !== "" && y !== "" && !isNaN(a) && !isNaN(b)) return a - b;
    return x < y ? -1 : x > y ? 1 : 0;
  }
  document.querySelectorAll("table.sortable").forEach(function (table) {
    var body = table.tBodies[0];
    if (!body) return;
    table.querySelectorAll("thead th").forEach(function (th, index) {
      th.addEventListener("click", function () {
        var asc = th.getAttribute("aria-sort") !== "ascending";
        var rows = Array.prototype.slice.call(body.rows).sort(function (r, s) {
          return cmp(key(r.cells[index]), key(s.cells[index]));
        });
        if (!asc) rows.reverse();
        rows.forEach(function (r) { body.appendChild(r); });
        table.querySelectorAll("thead th").forEach(function (h) {
          h.removeAttribute("aria-sort");
        });
        th.setAttribute("aria-sort", asc ? "ascending" : "descending");
      });
    });
  });
})();
"""

_SCRIPT_SHA256 = base64.b64encode(hashlib.sha256(_SCRIPT.encode("utf-8")).digest()).decode("ascii")

_CSP = (
    "default-src 'none'; script-src 'sha256-"
    + _SCRIPT_SHA256
    + "'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"
)

_CSS = (
    ":root{color-scheme:dark}"
    "*{box-sizing:border-box}"
    "body{font:16px/1.6 system-ui,-apple-system,'Segoe UI',sans-serif;"
    "max-width:1100px;margin:0 auto;padding:2rem 1.25rem;"
    "background:#0f172a;color:#f1f5f9}"
    "header h1{font-size:1.9rem;margin:0 0 .25rem}"
    ".subtitle{color:#a9bcd2;margin:0 0 2rem;max-width:75ch}"
    ".decision{border-radius:.75rem;padding:1.25rem 1.5rem;margin:0 0 2rem;"
    "border:1px solid #334155;border-left-width:.4rem}"
    ".decision.pass{border-left-color:#22c55e;background:#052e16}"
    ".decision.fail{border-left-color:#ef4444;background:#450a0a}"
    ".decision.neutral{border-left-color:#a9bcd2;background:#1e293b}"
    ".decision-label{font-size:.8rem;text-transform:uppercase;"
    "letter-spacing:.08em;color:#a9bcd2}"
    ".decision-value{font-size:1.6rem;font-weight:700;margin:.15rem 0 .5rem;"
    "overflow-wrap:anywhere}"
    ".decision-help{margin:0;color:#cbd5e1;max-width:70ch}"
    "section{margin:0 0 2.5rem}"
    "h2{font-size:1.25rem;margin:0 0 1rem;padding-bottom:.5rem;"
    "border-bottom:1px solid #334155}"
    "dl.config{display:grid;grid-template-columns:minmax(11rem,15rem) 1fr;"
    "gap:.4rem 1rem;margin:0}"
    "dl.config dt{color:#a9bcd2}"
    "dl.config dd{margin:0;font-variant-numeric:tabular-nums;"
    "overflow-wrap:anywhere}"
    ".table-wrap{overflow-x:auto;border-radius:.75rem;border:1px solid #334155}"
    "table{border-collapse:collapse;width:100%;background:#1e293b}"
    "caption{text-align:left;padding:.75rem 1rem;color:#a9bcd2;font-size:.9rem;"
    "background:#0f172a}"
    "th,td{padding:.8rem 1rem;text-align:left;border-top:1px solid #334155;"
    "vertical-align:middle}"
    "thead th{border-top:0;background:#0f172a;color:#a9bcd2;font-size:.8rem;"
    "text-transform:uppercase;letter-spacing:.06em;white-space:nowrap}"
    "tbody th{font-weight:600;overflow-wrap:anywhere}"
    ".badge{display:inline-block;padding:.2rem .7rem;border-radius:999px;"
    "font-size:.85rem;font-weight:600;color:#f8fafc;white-space:nowrap}"
    ".badge.pass{background:#166534}.badge.fail{background:#991b1b}"
    ".badge.neutral{background:#475569}"
    ".attribution{display:block;margin-top:.25rem;font-size:.85rem;color:#a9bcd2}"
    ".attribution code{font-family:ui-monospace,monospace;font-size:.8rem;"
    "color:#e2e8f0}"
    ".bar{display:inline-block;width:8rem;height:.6rem;border-radius:999px;"
    "background:#334155;vertical-align:middle;margin-right:.6rem;overflow:hidden}"
    ".bar-fill{display:block;height:100%;border-radius:999px}"
    ".bar-fill.pass{background:#22c55e}.bar-fill.fail{background:#ef4444}"
    ".bar-fill.neutral{background:#a9bcd2}"
    ".glyph-ok{color:#22c55e;font-weight:700}"
    ".glyph-bad{color:#ef4444;font-weight:700}"
    ".glyph-na{color:#a9bcd2}"
    ".spark{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
    "letter-spacing:.15em;white-space:nowrap}"
    ".rate-text{font-variant-numeric:tabular-nums;white-space:nowrap}"
    ".mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.85em;"
    "overflow-wrap:anywhere}"
    ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(10rem,1fr));"
    "gap:1rem;margin:0 0 1rem}"
    ".card{background:#1e293b;border:1px solid #334155;border-radius:.75rem;"
    "padding:1rem 1.25rem}"
    ".card .k{font-size:.8rem;color:#a9bcd2;text-transform:uppercase;"
    "letter-spacing:.06em}"
    ".card .v{font-size:1.4rem;font-weight:700;font-variant-numeric:tabular-nums;"
    "overflow-wrap:anywhere}"
    "details{margin-top:.4rem}"
    "summary{cursor:pointer;color:#a9bcd2}"
    "footer{color:#a9bcd2;font-size:.9rem;border-top:1px solid #334155;"
    "padding-top:1.5rem}"
    "footer p{max-width:75ch}"
    "@media (max-width:640px){body{padding:1.25rem .75rem}"
    "dl.config{grid-template-columns:1fr}}"
    "html.has-js table.sortable thead th{cursor:pointer}"
    "html.has-js table.sortable thead th:hover{text-decoration:underline}"
    "table.sortable thead th:focus-visible{outline:2px solid #38bdf8;outline-offset:-2px}"
    'table.sortable thead th[aria-sort=ascending]::after{content:" \\25B2"}'
    'table.sortable thead th[aria-sort=descending]::after{content:" \\25BC"}'
    "nav.toc{position:sticky;top:0;z-index:1;background:#0f172a;"
    "border:1px solid #334155;border-radius:.75rem;padding:.75rem 1.25rem;margin:0 0 2rem}"
    "nav.toc ul{list-style:none;display:flex;flex-wrap:wrap;gap:.2rem 1.25rem;"
    "margin:0;padding:0}"
    "nav.toc a{color:#a9bcd2;font-size:.9rem}"
    "section{scroll-margin-top:6rem}"
    "section:target{outline:2px solid #38bdf8;outline-offset:.4rem;border-radius:.25rem}"
    ".spark-svg{display:block;max-width:100%;height:auto}"
    ".spark-svg rect{fill:#38bdf8}"
    ".frontier-svg{display:block;max-width:100%;height:auto}"
    ".chart-wrap{overflow-x:auto;border:1px solid #334155;border-radius:.75rem;"
    "padding:1rem;background:#0f172a;margin:0 0 1rem}"
    ".chart-note{color:#a9bcd2;font-size:.9rem;max-width:75ch}"
    "@media print{:root{color-scheme:light}"
    "body{background:#fff;color:#111;max-width:none}"
    "nav.toc{position:static}"
    ".decision{border:2px solid #111;background:#fff!important}"
    ".decision-help,.subtitle,footer,dl.config dt{color:#333}"
    "table{background:#fff}th,td{border-color:#999}thead th{background:#eee;color:#111}}"
)

_DECISION_CLASS = {
    "compatible_on_fixture_set": "pass",
    "ambiguous": "neutral",
    "no_candidate": "neutral",
    "held_out_failed": "fail",
    "cancelled": "neutral",
    "failed": "fail",
}

_DECISION_HELP = {
    "compatible_on_fixture_set": (
        "The sole search winner also passed the supplied held-out cases under "
        "this exact run configuration. This does not recover the model's "
        "training template or establish general compatibility."
    ),
    "ambiguous": (
        "More than one candidate passed every search fixture. The lab abstains "
        "instead of guessing; the held-out set was not used to break the tie."
    ),
    "no_candidate": (
        "No candidate passed every search fixture. The lab abstains; nothing "
        "was promoted to held-out evaluation."
    ),
    "held_out_failed": (
        "The search winner did not pass the held-out cases, so compatibility is not claimed."
    ),
    "cancelled": (
        "The run was cancelled before completion. Partial evidence is retained "
        "but cannot support a decision."
    ),
    "failed": (
        "The run failed before completion. Partial evidence is retained but "
        "cannot support a decision."
    ),
}


_FAILURE_GUIDANCE = {
    # Sanitized backend failure codes (fixed vocabulary) mapped to operator
    # guidance. Surfaced in the decision banner when a run fails, so the
    # cause is visible without --include-evidence.
    "context_limit_exceeded": (
        "Prompt tokens plus --max-new-tokens exceeded the checkpoint's "
        "declared context window. Retry with a smaller --max-new-tokens "
        "(the doctor suggestion sizes it to fit the fixtures)."
    ),
    "render_failed": ("A fixture prompt could not be rendered. Re-run doctor on the dataset."),
    "encode_failed": (
        "A rendered prompt could not be encoded with the checkpoint "
        "tokenizer. Re-run doctor against this checkpoint."
    ),
    "invalid_input_ids": (
        "The backend rejected the encoded prompt. Re-run doctor "
        "--boundary-ids against this checkpoint."
    ),
    "generate_failed": (
        "Generation failed inside the backend. Check --device and that the "
        "checkpoint assets are intact (re-run doctor)."
    ),
    "backend_reported_error": (
        "The backend reported an error during generation. Check --device and "
        "the checkpoint assets (re-run doctor)."
    ),
    "invalid_or_empty_input": (
        "The backend rejected the encoded prompt as empty or out of "
        "vocabulary. Re-run doctor against this checkpoint."
    ),
    "invalid_output_cap": (
        "The requested --max-new-tokens is outside the backend's accepted "
        "range. Use a value between 1 and 4096."
    ),
    "invalid_seed": "The requested --seed is outside the accepted range.",
    "invalid_cancellation_event": "Internal error: the cancellation event was invalid.",
}


def _failure_diagnosis(result: dict[str, Any]) -> str:
    """Actionable failure-cause line for the decision banner.

    Only for decision "failed": names the sanitized backend failure code(s)
    behind the failure with operator guidance. The codes are a fixed
    vocabulary, so this is safe in the default evidence-excluded report.
    """
    if str(result.get("decision", "")) != "failed":
        return ""
    codes: list[str] = []
    entries = result.get("search")
    if type(entries) is list:
        for entry in entries:
            summary = entry.get("summary") if type(entry) is dict else None
            code = summary.get("failure_code") if type(summary) is dict else None
            if type(code) is str and code and code not in codes:
                codes.append(code)
    held = result.get("held_out")
    if type(held) is dict:
        summary = held.get("summary")
        code = summary.get("failure_code") if type(summary) is dict else None
        if type(code) is str and code and code not in codes:
            codes.append(code)
    if not codes:
        return ""
    parts = []
    for code in codes:
        guidance = _FAILURE_GUIDANCE.get(code)
        if guidance is None:
            guidance = (
                "The backend reported this failure; re-run doctor and check the checkpoint assets."
            )
        parts.append(f"{code}: {guidance}")
    return "Failure diagnosis: " + " ".join(parts)


def _public_view(result: dict[str, Any], include_evidence: bool) -> dict[str, Any]:
    clean: dict[str, Any] = json.loads(json.dumps(result, ensure_ascii=True, allow_nan=False))
    if not include_evidence:
        for run in clean.get("search", []):
            run.pop("observations", None)
        if clean.get("held_out") is not None:
            clean["held_out"].pop("observations", None)
        _strip_raw_diagnostics(clean)
    clean["raw_evidence_included"] = include_evidence
    return clean


#: Replacement for diagnostic free text that can carry raw prompts or
#: generations in an evidence-excluded public view. The divergence class
#: (fixed vocabulary) stays on the record; only the free text is withheld.
_WITHHELD_DIAGNOSTIC_DETAIL = "[withheld without --include-evidence]"


def _strip_raw_diagnostics(clean: dict[str, Any]) -> None:
    """Remove raw-evidence carriers from an evidence-excluded public view.

    ``observations`` (stripped by the caller) are not the only place raw
    content hides in a result dict:

    - ``template_conformance`` records, per divergent segment, the token-ID
      subsequences of the rendered prompt (``hand_ids``/``template_ids``) --
      decoding them reconstructs the prompt text;
    - ``backend_divergences`` embeds the decoded stop text (the expected
      answer) in ``stop_texts``/``stop_text``, and raw generated text in the
      free-text ``detail`` of kept-vs-stripped findings.

    None of these may reach the public JSON without ``--include-evidence``.
    Spans, counts, classes and severities stay: they carry no raw content.
    The evidence-included path is untouched.
    """
    conformance = clean.get("template_conformance")
    if type(conformance) is dict:
        diffs = conformance.get("diffs")
        if type(diffs) is list:
            for entry in diffs:
                if type(entry) is not dict:
                    continue
                segments = entry.get("diff")
                if type(segments) is list:
                    for segment in segments:
                        if type(segment) is not dict:
                            continue
                        segment.pop("hand_ids", None)
                        segment.pop("template_ids", None)
    divergences = clean.get("backend_divergences")
    if type(divergences) is dict:
        divergences.pop("stop_texts", None)
        per_fixture = divergences.get("per_fixture")
        if type(per_fixture) is list:
            for entry in per_fixture:
                if type(entry) is not dict:
                    continue
                findings = entry.get("divergences")
                if type(findings) is list:
                    for finding in findings:
                        if type(finding) is not dict:
                            continue
                        finding.pop("stop_text", None)
                        if "detail" in finding:
                            finding["detail"] = _WITHHELD_DIAGNOSTIC_DETAIL


def _short_fingerprint(value: object) -> str:
    text = value if type(value) is str else ""
    short = text[:19]
    return short + "\u2026" if len(text) > 19 else short


def _sort_key(value: Any) -> str:
    """Lexicographic data-sort key for a sortable table cell.

    Ints and floats are emitted plain so the inline sorter compares them
    numerically; anything else sorts as text. The "?" placeholder used for
    missing values sorts as empty.
    """
    if type(value) is int or type(value) is float:
        return str(value)
    text = str(value)
    return "" if text == "?" else text


def _toc(entries: list[tuple[str, str, str]]) -> str:
    """Sticky table of contents linking to the sections actually rendered.

    Each entry is (anchor, title, section HTML); entries whose section
    rendered empty are skipped, so the TOC never points at a missing id.
    Anchors and titles are fixed literals, never model-derived text.
    """
    items = "".join(
        '<li><a href="#' + anchor + '">' + title + "</a></li>"
        for anchor, title, html in entries
        if html
    )
    if not items:
        return ""
    return '<nav class="toc" aria-label="Report sections"><ul>' + items + "</ul></nav>"


def _taxonomy_note(result: dict[str, Any]) -> str:
    """One-line failure-taxonomy citation for the decision banner."""
    taxonomy = result.get("failure_taxonomy")
    if type(taxonomy) is not dict:
        return ""
    if str(result.get("decision", "")) == "compatible_on_fixture_set":
        return ""
    totals: dict[str, int] = {}
    per_candidate = taxonomy.get("per_candidate")
    if type(per_candidate) is dict:
        for counts in per_candidate.values():
            if type(counts) is not dict:
                continue
            for mode, count in counts.items():
                if type(count) is int and count > 0:
                    totals[str(mode)] = totals.get(str(mode), 0) + count
    held_out = taxonomy.get("held_out")
    if type(held_out) is dict:
        for mode, count in held_out.items():
            if type(count) is int and count > 0:
                totals[str(mode)] = totals.get(str(mode), 0) + count
    if not totals:
        if str(result.get("decision", "")) in ("cancelled", "failed"):
            return (
                "Failure-mode taxonomy: the run did not complete, so there "
                "are no observations to classify."
            )
        return "Failure-mode taxonomy: no failing observations recorded."
    parts = ", ".join(f"{mode} \u00d7{totals[mode]}" for mode in sorted(totals))
    return f"Failure-mode taxonomy: {parts}."


def _conformance_section(result: dict[str, Any]) -> str:
    conformance = result.get("template_conformance")
    if type(conformance) is not dict:
        return ""
    status = conformance.get("status")
    if status == "no_template_declared":
        body = (
            "<p>No chat template is declared in the checkpoint's "
            "tokenizer_config.json, so hand-rendered hypotheses could not be "
            "checked against the template's own rendering.</p>"
        )
    elif status != "ok":
        body = (
            "<p>Template-conformance check unavailable: "
            + escape(str(conformance.get("detail", "")))
            + ".</p>"
        )
    else:
        rows = []
        for entry in conformance.get("diffs") or []:
            if type(entry) is not dict:
                continue
            if entry.get("status") != "ok":
                cell = '<span class="badge neutral">unavailable</span>'
                diff_key = "unavailable"
            else:
                segments = entry.get("diff") or []
                if not segments:
                    cell = '<span class="badge pass">identical</span>'
                    diff_key = "identical"
                else:
                    detail_items = []
                    for segment in segments:
                        if type(segment) is not dict:
                            continue
                        # Spans only: the token-ID subsequences would
                        # reconstruct the rendered prompt, so they are never
                        # embedded in this HTML view (see docs/USAGE.md).
                        detail_items.append(
                            "hand "
                            + escape(str(segment.get("hand_span")))
                            + " \u2194 template "
                            + escape(str(segment.get("template_span")))
                        )
                    cell = (
                        '<span class="badge fail">'
                        + escape(f"{len(segments)} divergent segment(s)")
                        + "</span>"
                        + '<details name="conformance-details">'
                        + '<summary class="mono">segments</summary>'
                        '<p class="mono">' + "<br>".join(detail_items) + "</p></details>"
                    )
                    diff_key = "divergent"
            rows.append(
                '<tr><th scope="row" data-sort="'
                + escape(str(entry.get("format", "?")))
                + '">'
                + escape(str(entry.get("format", "?")))
                + '</th><td data-sort="'
                + escape(str(entry.get("fixture", "?")))
                + '">'
                + escape(str(entry.get("fixture", "?")))
                + '</td><td data-sort="'
                + escape(_sort_key(entry.get("hand_tokens")))
                + '">'
                + escape(str(entry.get("hand_tokens", "?")))
                + '</td><td data-sort="'
                + escape(_sort_key(entry.get("template_tokens")))
                + '">'
                + escape(str(entry.get("template_tokens", "?")))
                + '</td><td data-sort="'
                + diff_key
                + '">'
                + cell
                + "</td></tr>"
            )
        if not rows:
            body = "<p>No format/fixture pairs were checked.</p>"
        else:
            body = (
                '<div class="table-wrap"><table class="sortable">'
                "<caption>Hand-rendered hypothesis vs the checkpoint tokenizer's "
                "own chat-template rendering, compared as exact token-ID "
                "sequences.</caption>"
                '<thead><tr><th scope="col">Candidate</th><th scope="col">Fixture</th>'
                '<th scope="col">Hand tokens</th><th scope="col">Template tokens</th>'
                '<th scope="col">Diff</th></tr></thead><tbody>'
                + "".join(rows)
                + "</tbody></table></div>"
            )
    return (
        '<section id="template-conformance"><h2>Template conformance</h2>'
        + body
        + "<p>Each hand-rendered "
        'hypothesis is encoded with <span class="mono">add_special_tokens=False</span> '
        "and compared token-ID by token-ID against the checkpoint's own "
        '<span class="mono">apply_chat_template</span> output for the same '
        "messages. A divergent segment lists the positions that differ -- the "
        "token-ID subsequences themselves are withheld from this HTML view; "
        "an identical rendering reports an empty "
        "diff.</p></section>"
    )


def _matrix_cell_html(cell: dict[str, Any] | None) -> str:
    if type(cell) is not dict or not cell.get("available"):
        detail = cell.get("detail", "unavailable") if type(cell) is dict else "unavailable"
        return '<span class="badge neutral">n/a</span> ' + escape(f"({detail})")
    color = "pass" if cell.get("pass") else "fail"
    label = "pass" if cell.get("pass") else "fail"
    fired = " \u00b7 stop fired" if cell.get("fired") else ""
    return (
        '<span class="badge ' + color + '">' + label + "</span>"
        '<span class="mono">' + escape(fired) + "</span>"
    )


def _stop_matrix_section(result: dict[str, Any]) -> str:
    matrix = result.get("stop_matrix")
    if type(matrix) is not dict:
        return ""
    semantics = matrix.get("semantics")
    if type(semantics) is not list or not semantics:
        return ""
    semantic_ids = [s.get("id") for s in semantics if type(s) is dict and type(s.get("id")) is str]
    other_semantics = [s for s in semantics if type(s) is dict and s.get("id") != "native"]
    header = "".join(
        '<th scope="col">' + escape(str(s.get("label", s.get("id", "?")))) + "</th>"
        for s in other_semantics
    )

    def stage_table(stage: str, rows: object) -> str:
        body_rows = []
        if type(rows) is list:
            for row in rows:
                if type(row) is not dict:
                    continue
                native = row.get("native_pass")
                native_cell = (
                    '<span class="badge pass">pass</span>'
                    if native
                    else '<span class="badge fail">fail</span>'
                )
                cells = row.get("cells") or {}
                flips = row.get("flips") or []
                flip_note = (
                    '<span class="mono">' + escape(", ".join(str(f) for f in flips)) + "</span>"
                    if type(flips) is list and flips
                    else '<span class="mono">\u2014</span>'
                )
                body_rows.append(
                    '<tr><th scope="row">'
                    + escape(str(row.get("fixture", "?")))
                    + "</th><td>"
                    + native_cell
                    + "</td>"
                    + "".join(
                        "<td>"
                        + _matrix_cell_html(cells.get(sid) if type(cells) is dict else None)
                        + "</td>"
                        for sid in semantic_ids
                        if sid != "native"
                    )
                    + "<td>"
                    + flip_note
                    + "</td></tr>"
                )
        if not body_rows:
            return "<p>No rows for this stage.</p>"
        return (
            '<div class="table-wrap"><table>'
            + f"<caption>Stop-behavior isolation matrix, {escape(stage)} stage.</caption>"
            '<thead><tr><th scope="col">Fixture</th><th scope="col">Native (lab)</th>'
            + header
            + '<th scope="col">Flips vs native</th></tr></thead><tbody>'
            + "".join(body_rows)
            + "</tbody></table></div>"
        )

    legend = "".join(
        "<dt>"
        + escape(str(s.get("label", s.get("id", "?"))))
        + "</dt><dd>"
        + escape(str(s.get("description", "")))
        + "</dd>"
        for s in semantics
        if type(s) is dict
    )
    sections = ""
    for stage in ("search", "held_out"):
        rows = matrix.get(stage)
        if rows is None:
            continue
        sections += f"<h3>{escape(stage.replace('_', ' ').title())} stage</h3>" + stage_table(
            stage, rows
        )
    return (
        '<section id="stop-behavior-matrix"><h2>Stop-behavior isolation matrix</h2>'
        "<p>The frozen winner's recorded outputs, re-measured under emulated "
        "competitor stop semantics. Each cell recomputes the lab's pass rule "
        "(exact answer and token-ID boundary) on the output as truncated by "
        "that semantic; a flip means the semantic would have changed the "
        "verdict. Emulations are post-hoc truncations of the greedy "
        "N-token outputs.</p>" + sections + '<dl class="config">' + legend + "</dl></section>"
    )


def _backend_divergence_section(result: dict[str, Any]) -> str:
    """Backend stop-semantics divergence section.

    Renders ``result["backend_divergences"]`` (built by
    ``diffbackends.divergence_section_data`` from the winner's recorded
    observations): the one-line verdict, a divergence-class x semantic
    count matrix with severity badges, and one ``<details>`` callout per
    divergence. When the section data records ``observations_truncated``,
    a fixed-vocabulary note names how many observations were recorded and
    how many were actually tested. Returns "" when the key is missing, so runs without the
    diagnosis render exactly as before (the sample-report golden test).
    Fixture names are escaped; severities, classes and counts are the
    module's fixed vocabulary. Stop texts and free-text detail lines are
    withheld from this HTML view entirely (they can carry raw prompts or
    generations) -- only the fixed-vocabulary markers render. Severity
    badges reuse the existing badge
    palette (high: fail-red; medium/low: neutral slate, with the severity
    word in the badge text) -- no new CSS, so byte-identical rendering
    for reports without the section is preserved.
    """
    section = result.get("backend_divergences")
    if type(section) is not dict:
        return ""
    pairs_verdict = section.get("pairs_verdict")
    if type(pairs_verdict) is not str or not pairs_verdict:
        return ""
    per_fixture = section.get("per_fixture")
    matrix = section.get("matrix")
    severities = section.get("severities")
    if type(per_fixture) is not list or type(matrix) is not dict:
        return ""
    if type(severities) is not dict:
        severities = {}
    verdict = section.get("verdict")
    verdict_badge = (
        '<span class="badge fail">diverge</span>'
        if verdict == "diverge"
        else '<span class="badge pass">agree</span>'
    )

    # Divergence-class x semantic count matrix: fixed row order (severity
    # descending over the known classes; classes outside class_order get no
    # matrix row, though their <details> callouts below still render),
    # fixed column order, sortable like the other tables.
    class_order = (
        "KEPT_VS_STRIPPED",
        "EOS_PRECEDENCE",
        "STR_OVER_EOS_PRECEDENCE",
        "MULTI_STOP_SELECTION",
        "FIRING_STEP_DIFFERS",
        "MIN_TOKENS_GATED",
        "NO_MIN_TOKENS_GATE",
        "HARNESS_NORMALIZATION_RELIED",
    )
    semantic_order = (
        ("hf_stop_string", "HF stop string"),
        ("vllm_holdback_strip", "vLLM strip"),
        ("vllm_holdback_keep", "vLLM keep"),
        ("sglang_stop", "SGLang stop"),
        ("eos_appended", "EOS-appended"),
    )
    severity_badge = {"high": "fail", "medium": "neutral", "low": "neutral"}
    matrix_rows = []
    for kind in class_order:
        counts = matrix.get(kind)
        if type(counts) is not dict:
            continue
        severity = severities.get(kind)
        sev_badge = (
            '<span class="badge '
            + severity_badge.get(str(severity), "neutral")
            + '">'
            + escape(str(severity) if type(severity) is str else "?")
            + "</span>"
        )
        matrix_rows.append(
            '<tr><th scope="row" data-sort="'
            + escape(kind)
            + '">'
            + sev_badge
            + " "
            + '<span class="mono">'
            + escape(kind)
            + "</span></th>"
            + "".join(
                '<td data-sort="'
                + escape(_sort_key(counts.get(sid, 0)))
                + '">'
                + escape(str(counts.get(sid, 0)))
                + "</td>"
                for sid, _label in semantic_order
            )
            + "</tr>"
        )
    matrix_table = (
        '<div class="table-wrap"><table class="sortable">'
        "<caption>Divergence-class \u00d7 emulated-semantic counts: each cell counts "
        "the recorded observations whose stop-semantics differential test fired "
        "that divergence class for that semantic.</caption>"
        '<thead><tr><th scope="col">Divergence class</th>'
        + "".join('<th scope="col">' + label + "</th>" for _sid, label in semantic_order)
        + "</tr></thead><tbody>"
        + "".join(matrix_rows)
        + "</tbody></table></div>"
        if matrix_rows
        else "<p>No divergences fired: every recorded observation agreed across "
        "the emulated stop semantics.</p>"
    )

    # One <details> callout per divergence: fixture, class with severity.
    # The stop text and the recorded detail are withheld from this HTML view
    # (the stop text is the decoded expected answer; the detail can carry raw
    # generated text) -- the class, severity and matrix counts keep the
    # diagnostic signal, and --include-evidence keeps the full record in JSON.
    callouts = []
    for entry in per_fixture:
        if type(entry) is not dict:
            continue
        fixture = entry.get("fixture")
        findings = entry.get("divergences")
        if type(findings) is not list:
            continue
        for finding in findings:
            if type(finding) is not dict:
                continue
            finding_class = finding.get("class")
            finding_severity = finding.get("severity")
            # The stop text (the decoded expected answer) and the free-text
            # detail (which can carry raw generated text) are never embedded
            # in this HTML view; the class, severity and fixture identify the
            # finding, and --include-evidence keeps the full record in JSON.
            callouts.append(
                "<details><summary>"
                + '<span class="badge '
                + severity_badge.get(str(finding_severity), "neutral")
                + '">'
                + escape(str(finding_severity) if type(finding_severity) is str else "?")
                + "</span> "
                + '<span class="mono">'
                + escape(str(finding_class))
                + "</span>"
                + " on "
                + '<span class="mono">'
                + escape(str(fixture))
                + "</span>"
                + "</summary>"
                + '<p class="mono">[stop text and diagnostic detail withheld '
                "from this HTML view]</p>" + "</details>"
            )

    stop_texts = section.get("stop_texts")
    params = ""
    if type(stop_texts) is list and stop_texts:
        # The stop text is the decoded expected answer: never rendered here.
        params = '<p>Stop text(s): <span class="mono">[withheld from this HTML view]</span>.</p>'
    # Observation-count bound note (fixed vocabulary): the diagnosis
    # processed only the first MAX_DIVERGENCE_OBSERVATIONS observations
    # when the run recorded more than that. Numbers are ints by
    # construction; still escaped via str() like the rest of this section.
    truncation_note = ""
    if section.get("observations_truncated") is True:
        from .diffbackends import MAX_DIVERGENCE_OBSERVATIONS

        total_obs = section.get("observations_total")
        truncation_note = (
            "<p>Observation bound reached: the run recorded "
            + escape(str(total_obs))
            + " observations; only the first "
            + escape(str(MAX_DIVERGENCE_OBSERVATIONS))
            + " were differential-tested.</p>"
        )
    return (
        '<section id="backend-divergence"><h2>Backend divergence</h2>'
        "<p>Differential test of the emulated backend stop semantics "
        "(HF stop-string, vLLM detokenizer strip/keep, SGLang stop, "
        "EOS-appended) over "
        "the frozen winner's recorded token sequences -- the same "
        "post-hoc truncation measurement the stop-behavior matrix uses. "
        + verdict_badge
        + " "
        + escape(pairs_verdict)
        + ".</p>"
        + params
        + truncation_note
        + matrix_table
        + ("<h3>Divergences</h3>" + "".join(callouts) if callouts else "")
        + "</section>"
    )


def _sweep_section(result: dict[str, Any]) -> str:
    sweep = result.get("sensitivity_sweep")
    if type(sweep) is not dict:
        return ""
    rows = []
    for depth in sweep.get("depths") or []:
        if type(depth) is not dict:
            continue
        changed = depth.get("winner_changed")
        drop = depth.get("drop_bytes")
        rows.append(
            '<tr><th scope="row" data-sort="'
            + escape(str(drop) if type(drop) is int else "")
            + '">'
            + escape(str(depth.get("drop_bytes", "?")))
            + '</th><td data-sort="'
            + escape(str(depth.get("decision", "?")))
            + '">'
            + escape(str(depth.get("decision", "?")))
            + '</td><td data-sort="'
            + escape(str(depth.get("selected_candidate", "?")))
            + '">'
            + escape(str(depth.get("selected_candidate", "?")))
            + '</td><td data-sort="'
            + ("changed" if changed else "stable")
            + '">'
            + (
                '<span class="badge fail">changed</span>'
                if changed
                else '<span class="badge pass">stable</span>'
            )
            + "</td></tr>"
        )
    table = (
        '<div class="table-wrap"><table class="sortable">'
        "<caption>Search re-run with the prompt truncated mid-token.</caption>"
        '<thead><tr><th scope="col">Dropped bytes</th><th scope="col">Decision</th>'
        '<th scope="col">Selected</th><th scope="col">Vs native</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
        if rows
        else "<p>No depths were evaluated.</p>"
    )
    return (
        '<section id="sensitivity-sweep"><h2>Prompt-boundary sensitivity sweep</h2>'
        "<p>Each depth drops that many trailing bytes from the rendered prompt "
        "and re-encodes it, truncating the prompt in the middle of its final "
        "token, then re-runs the full search. Native winner: "
        + escape(str(sweep.get("native_winner", "?")))
        + " ("
        + escape(str(sweep.get("native_decision", "?")))
        + "). Depths where the winner or decision changed: "
        + escape(str(sweep.get("flips_at", "?")))
        + ".</p>"
        + table
        + "</section>"
    )


def _logprob_num_cell(value: Any) -> str:
    """One sortable numeric table cell: 2-decimal logprob or an em dash."""
    if type(value) is float:
        return (
            '<td data-sort="' + escape(_sort_key(value)) + '">' + escape(f"{value:.2f}") + "</td>"
        )
    return '<td data-sort="">' + escape("\u2014") + "</td>"


def _logprob_fixture_details(candidate: dict[str, Any]) -> str:
    """Per-fixture score table for one candidate, inside a <details>."""
    name = candidate.get("name")
    scores = candidate.get("fixture_scores")
    if type(scores) is not list or not scores:
        return ""
    rows = []
    for score in scores:
        if type(score) is not dict:
            continue
        native = score.get("native_pass")
        glyph = (
            '<span class="glyph-ok" title="pass">\u2713</span>'
            if native is True
            else '<span class="glyph-bad" title="fail">\u2717</span>'
            if native is False
            else '<span class="glyph-na" title="unknown">\u2013</span>'
        )
        status = score.get("status")
        reason = score.get("reason")
        status_text = str(status) if type(status) is str else "?"
        if type(reason) is str and reason:
            # Reason codes are the engine's fixed vocabulary; escaped anyway.
            status_text += " (" + reason + ")"
        rows.append(
            "<tr><td>"
            + escape(str(score.get("fixture")))
            + "</td><td>"
            + glyph
            + "</td><td>"
            + escape(status_text)
            + "</td>"
            + _logprob_num_cell(score.get("mean_logprob"))
            + "</tr>"
        )
    return (
        "<details><summary>Per-fixture scores: "
        + escape(str(name))
        + "</summary>"
        + '<div class="table-wrap"><table>'
        "<caption>Teacher-forced score of the expected answer per fixture; "
        "mean is logprob per scored token in nats.</caption>"
        "<thead><tr><th>Fixture</th><th>Native pass</th>"
        "<th>Score status</th><th>Mean / token</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div></details>"
    )


def _logprob_section(result: dict[str, Any]) -> str:
    """Teacher-forced log-likelihood diagnostic section (§9.3-§9.5).

    Renders ``result["logprob_diagnostic"]`` (built by
    ``experiments.logprob_diagnostic`` when ``run --logprob-diagnostic`` is
    set): per-candidate total/mean logprob over the expected answers, one
    plain-language reading per candidate separating "weights don't know"
    from "greedy derailed", and per-fixture scores in <details>. Returns ""
    when the key is missing, so runs without the flag render exactly as
    before (the sample-report golden test). Every model/fixture-derived
    string is escaped; numbers, statuses and reason codes are fixed-shape.
    Unavailable scores render as an explicit "unavailable" note -- never a
    guess.
    """
    diag = result.get("logprob_diagnostic")
    if type(diag) is not dict:
        return ""
    thresholds = diag.get("thresholds")
    band_note = ""
    if type(thresholds) is dict:
        hi = thresholds.get("derailed_mean_logprob_at_least")
        lo = thresholds.get("unknown_mean_logprob_below")
        if type(hi) in (int, float) and type(lo) in (int, float):
            band_note = (
                f" Declared bands: mean \u2265 {hi} nats/token reads as "
                f"greedy derailment, mean < {lo} as missing knowledge; "
                "between the bands the diagnostic withholds judgment."
            )
    intro = (
        "<p>After search, each candidate's <em>expected answer</em> was "
        "teacher-forced through the checkpoint -- one forward pass per "
        "candidate \u00d7 fixture over the recorded prompt plus the encoded "
        "expected answer; prompt positions are masked and answer tokens are "
        "gathered with the causal shift. Diagnostic only: it never changes "
        "the selection rule. Totals are summed log-probabilities in nats; "
        "the mean is per scored token." + band_note + "</p>"
    )
    candidates = diag.get("candidates")
    rows: list[str] = []
    readings: list[str] = []
    details: list[str] = []
    if type(candidates) is list:
        for candidate in candidates:
            if type(candidate) is not dict:
                continue
            name = candidate.get("name")
            aggregate = candidate.get("aggregate")
            if type(aggregate) is not dict:
                aggregate = {}
            interpretation = candidate.get("interpretation")
            if type(interpretation) is not dict:
                interpretation = {}
            scores = candidate.get("fixture_scores")
            scored_total = len(scores) if type(scores) is list else 0
            passes = aggregate.get("native_passes")
            pass_text = f"{passes}/{scored_total}" if type(passes) is int else "?"
            tokens = aggregate.get("scored_token_count")
            rows.append(
                '<tr><th scope="row" data-sort="'
                + escape(str(name))
                + '">'
                + escape(str(name))
                + '</th><td data-sort="'
                + escape(_sort_key(passes))
                + '">'
                + escape(pass_text)
                + "</td>"
                + _logprob_num_cell(aggregate.get("total_logprob"))
                + _logprob_num_cell(aggregate.get("mean_logprob"))
                + '<td data-sort="'
                + escape(_sort_key(tokens))
                + '">'
                + escape(str(tokens) if type(tokens) is int else "?")
                + "</td><td>"
                + escape(str(interpretation.get("kind", "?")))
                + "</td></tr>"
            )
            reading = interpretation.get("text")
            readings.append(
                "<p><strong>"
                + escape(str(name))
                + ":</strong> "
                + escape(str(reading) if type(reading) is str else "?")
                + "</p>"
            )
            details.append(_logprob_fixture_details(candidate))
    if diag.get("available") is True and rows:
        body = (
            '<div class="table-wrap"><table class="sortable">'
            "<caption>Per-candidate teacher-forced log-likelihood of the "
            "expected answers: summed total in nats, mean per scored token, "
            "and the plain-language reading.</caption>"
            '<thead><tr><th scope="col">Candidate</th>'
            '<th scope="col">Search pass</th><th scope="col">Total logprob</th>'
            '<th scope="col">Mean / token</th><th scope="col">Scored tokens</th>'
            '<th scope="col">Reading</th>'
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
            + "".join(readings)
            + "".join(details)
        )
    else:
        detail = diag.get("detail")
        detail_text = detail if type(detail) is str and detail else "no reason recorded"
        body = (
            "<p>Log-likelihood diagnostic unavailable: "
            + escape(detail_text)
            + ". The alignment check or the forward pass refused every "
            "score; nothing was guessed.</p>"
        )
    return (
        '<section id="logprob-diagnostic"><h2>Log-likelihood diagnostic</h2>'
        + intro
        + body
        + "</section>"
    )


def _search_outcomes(
    result: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, dict[str, bool | None]]]:
    """Per-candidate, per-fixture search outcomes from the recorded report.

    Returns (candidate names, fixture IDs in first-seen order, outcomes)
    where outcomes[name][fixture_id] is True (exact answer and boundary),
    False (completed but not passing), or None (incomplete or missing
    observation). Pure post-hoc reading of recorded observations: no model
    calls, no re-evaluation.
    """
    names: list[str] = []
    order: list[str] = []
    per_candidate: dict[str, dict[str, bool | None]] = {}
    for run in result.get("search") or []:
        if type(run) is not dict:
            continue
        name = (run.get("format") or {}).get("name")
        if type(name) is not str or not name or name in per_candidate:
            continue
        names.append(name)
        outcomes: dict[str, bool | None] = {}
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


def _outcome_grid_section(result: dict[str, Any]) -> str:
    """Per-fixture x candidate outcome grid: which fixtures separate whom."""
    names, order, per_candidate = _search_outcomes(result)
    if not names or not order:
        return ""
    winner = result.get("selected_candidate")
    header = "".join(
        '<th scope="col">' + escape(name) + (" \u2605" if name == winner else "") + "</th>"
        for name in names
    )
    rows = []
    for fid in order:
        cells = []
        for name in names:
            outcome = per_candidate[name].get(fid)
            if outcome is True:
                cells.append(
                    '<td data-sort="2"><span class="glyph-ok" title="pass">\u2713</span></td>'
                )
            elif outcome is False:
                cells.append(
                    '<td data-sort="1"><span class="glyph-bad" title="fail">\u2717</span></td>'
                )
            else:
                cells.append(
                    '<td data-sort="0"><span class="glyph-na" title="incomplete">\u2013</span></td>'
                )
        rows.append(
            '<tr><th scope="row" data-sort="'
            + escape(fid)
            + '">'
            + escape(fid)
            + "</th>"
            + "".join(cells)
            + "</tr>"
        )
    return (
        '<section id="per-fixture-outcomes"><h2>Per-fixture outcomes</h2>'
        '<div class="table-wrap"><table class="sortable">'
        "<caption>Search-stage outcome per candidate and fixture: "
        "\u2713 exact answer and boundary, \u2717 completed without passing, "
        "\u2013 incomplete or missing observation. \u2605 marks the selected "
        "candidate.</caption>"
        '<thead><tr><th scope="col">Fixture</th>'
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></section>"
    )


def _decisive_section(result: dict[str, Any]) -> str:
    """Fixtures passed by exactly one candidate while all others failed.

    These are the fixtures that decided the selection: the winner's list is
    the evidence the verdict actually rests on. An empty list with a unique
    winner means no single fixture separates it from the whole field (the
    win came from a combination). Incomplete observations disqualify a
    fixture -- a candidate that errored cannot be counted as failing it.
    """
    names, order, per_candidate = _search_outcomes(result)
    if not names or not order:
        return ""
    solo: dict[str, list[str]] = {name: [] for name in names}
    for fid in order:
        passed = [name for name in names if per_candidate[name].get(fid) is True]
        if len(passed) != 1:
            continue
        if any(per_candidate[name].get(fid) is None for name in names):
            continue
        solo[passed[0]].append(fid)
    items = "".join(
        "<li><strong>" + escape(name) + "</strong>: " + escape(", ".join(fids)) + "</li>"
        for name, fids in solo.items()
        if fids
    )
    if not items:
        body = (
            "<p>No fixture was passed by exactly one candidate with every "
            "other candidate failing it.</p>"
        )
    else:
        body = (
            "<p>Fixtures passed by exactly one candidate while every other "
            "candidate failed them -- the fixtures that separated the field:</p><ul>"
            + items
            + "</ul>"
        )
    return '<section id="decisive-fixtures"><h2>Decisive fixtures</h2>' + body + "</section>"


def selection_margin(result: dict[str, Any]) -> dict[str, Any] | None:
    """Fragility of the selection decision, measured in fixture flips.

    Pure function of the recorded search outcomes: no model calls, and it
    never alters the decision -- it only measures how close the recorded
    outcomes were to a different one.

    - unique winner (``selected_candidate`` set): the minimum, over
      runners-up, of their failing search-fixture counts; flipping that
      many of the closest runner-up's failing fixtures to pass would tie
      the selection. ``direction`` is ``"to_tie"``.
    - ``no_candidate``: the minimum, over all candidates, of their failing
      counts; flipping that many of the closest candidate's failing
      fixtures to pass would select it. ``direction`` is ``"to_winner"``.
    - ``ambiguous``: ``(# all-pass candidates - 1)``; demoting that many of
      the tied candidates -- one flipped fixture each -- would leave a
      unique winner. ``direction`` is ``"to_unique"``; ``runner_up`` is
      null (no single candidate carries the margin) and
      ``decisive_fixtures`` are the fixtures every tied candidate passed.
    - ``failed`` / ``cancelled`` (or anything uncomputable): None, so the
      key stays absent from ``results.json``.

    A "failing" fixture is one the candidate did not pass: completed
    without passing, or incomplete/missing. Runner-up ties break to the
    lexicographically smallest candidate name; fixture lists are sorted.
    """
    if type(result) is not dict:
        raise ValueError("result must be a dict")
    if str(result.get("decision", "")) in ("failed", "cancelled"):
        return None
    names, order, per_candidate = _search_outcomes(result)
    if not names or not order:
        return None
    failing: dict[str, list[str]] = {
        name: sorted(fid for fid in order if per_candidate[name].get(fid) is not True)
        for name in names
    }
    selected = result.get("selected_candidate")
    if type(selected) is str and selected:
        runners = [name for name in names if name != selected]
        if not runners:
            # A lone candidate has no runner-up to tie against: the margin
            # is undefined, reported as a null value rather than invented.
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


MAX_FRONTIER_STEP_POINTS = 100_000
"""Cap on validated frontier step points (hostile-payload guard).

Native ``budget_frontier`` payloads carry at most a few change points per
candidate (pass counts change only where a fixture's minimal budget
falls), so legitimate data never approaches this bound. A foreign
payload can stuff millions of ``pass_steps`` points inside the 32 MiB
``results.json`` cap and hang the chart, trajectory, and span
derivations -- which are superlinear in the point count -- for hours.
Beyond the bound the frontier is treated as unchartable: the same
graceful degradation as malformed input. The cap also bounds the SVG
chart's per-point circles. Public so ``diffreports`` can share the bound
(lazy import there, mirroring the existing reporting import).
"""


def _valid_rate(rate: Any) -> float | None:
    """A pass rate as a finite float, or None when it is not one.

    Hostile result payloads can carry a string, a bool, or NaN/Infinity
    (the JSON parser accepts the latter two by default); formatting any of
    those with ``:.0%`` raises or renders ``"nan%"``. ``bool`` is excluded
    explicitly: ``True == 1`` but a boolean is never a genuine rate.
    Malformed rates degrade to the "incomplete" badge, never a crash.
    """
    if type(rate) is bool or type(rate) not in (int, float):
        return None
    value = float(rate)
    return value if math.isfinite(value) else None


def _fixture_total(data: dict[str, Any]) -> int:
    """Number of fixture ids recorded for a frontier candidate.

    Malformed ``fixture_ids`` (anything but a list) count as zero so a
    hostile payload can never crash the report through ``len()``.
    """
    ids = data.get("fixture_ids")
    return len(ids) if type(ids) is list else 0


def _sorted_str_keys(candidates: dict[str, Any]) -> list[str]:
    """Sorted string keys of a candidate mapping; non-string keys skipped.

    ``sorted()`` on mixed-type keys raises ``TypeError`` -- a hostile
    payload must degrade to skipped entries, never a crash.
    """
    return sorted(key for key in candidates if type(key) is str)


def _frontier_series(frontier: dict[str, Any]) -> tuple[list[str], int] | None:
    """Validated frontier series: (sorted candidate names, max_new_tokens).

    Shared validation for the frontier chart and the margin trajectory:
    candidates must be a nonempty dict, ``max_new_tokens`` a positive int,
    and each candidate must carry at least one well-typed ``pass_steps``
    point inside the budget range with a nonzero fixture total. The total
    validated point count is capped at :data:`MAX_FRONTIER_STEP_POINTS`
    (hostile-payload guard -- see the constant). Malformed entries are
    skipped; returns None when nothing chartable remains.
    """
    if type(frontier) is not dict:
        return None
    candidates = frontier.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    max_tokens = frontier.get("max_new_tokens")
    if type(max_tokens) is not int or max_tokens <= 0:
        return None
    names: list[str] = []
    total_points = 0
    for name in _sorted_str_keys(candidates):
        data = candidates[name]
        if type(data) is not dict:
            continue
        if _fixture_total(data) <= 0:
            continue
        points = 0
        for point in data.get("pass_steps") or []:
            if (
                type(point) is list
                and len(point) == 2
                and type(point[0]) is int
                and type(point[1]) is int
                and 1 <= point[0] <= max_tokens
            ):
                points += 1
        if not points:
            continue
        total_points += points
        if total_points > MAX_FRONTIER_STEP_POINTS:
            return None
        names.append(name)
    if not names:
        return None
    return (names, max_tokens)


def _frontier_candidate_colors(frontier: dict[str, Any]) -> dict[str, str]:
    """Deterministic per-candidate palette for frontier-derived charts.

    Assigned in sorted candidate-name order (cycled when there are more
    candidates than colors), so the same frontier always renders the same
    colors regardless of dict insertion order or hash seed. The frontier
    chart and the margin trajectory chart share this assignment, so a
    candidate keeps its color across both charts.
    """
    validated = _frontier_series(frontier)
    if validated is None:
        return {}
    names, _ = validated
    return {
        name: _FRONTIER_CHART_COLORS[i % len(_FRONTIER_CHART_COLORS)]
        for i, name in enumerate(names)
    }


_MARGIN_TIE_COLOR = "#94a3b8"
"""Segment color for tied-lead spans of the margin trajectory.

Slate, deliberately outside the candidate palette and the decision
ribbon colors: a gray span reads as "no unique leader" at a glance.
"""


def margin_trajectory(frontier: dict[str, Any]) -> list[tuple[int, str | None, int]]:
    """Per-budget selection margin over the token-budget sweep.

    Pure post-hoc derivation from a recorded ``budget_frontier``: at every
    token budget where any candidate's pass count changes, the pass counts
    of all candidates (from their recorded ``pass_steps``), the unique
    leader (the candidate with the strictly highest pass count), and the
    margin -- the leader's count minus the highest count among the
    remaining candidates. When the lead is tied, or a lone candidate has
    no competition, the margin is 0; the leader is None for a tied lead.

    Returns change points ``[(budget, leader, margin), ...]`` sorted by
    budget with consecutive duplicates collapsed, so the margin is
    constant between adjacent change points and on [last, max_new_tokens].
    The margin answers "how decisive was the field at each budget?",
    complementing :func:`selection_margin` (a single native-budget scalar)
    and the decision ribbon (which outcome, not by how much). Malformed
    input yields [] -- never raises, never guesses.
    """
    validated = _frontier_series(frontier)
    if validated is None:
        return []
    names, max_tokens = validated
    candidates = frontier["candidates"]
    steps_of: dict[str, list[tuple[int, int]]] = {}
    keys_of: dict[str, list[int]] = {}
    for name in names:
        data = candidates[name]
        by_budget: dict[int, int] = {}
        for point in data.get("pass_steps") or []:
            if (
                type(point) is list
                and len(point) == 2
                and type(point[0]) is int
                and type(point[1]) is int
                and 1 <= point[0] <= max_tokens
                and 0 <= point[1] <= _fixture_total(data)
            ):
                by_budget[point[0]] = point[1]
        steps_of[name] = sorted(by_budget.items())
        # Hoisted out of the per-budget loop: rebuilding the key list on
        # every _count_at_budget call made this derivation quadratic in the
        # point count (perf finding 1A).
        keys_of[name] = [b for b, _ in steps_of[name]]
    change_budgets = sorted({b for ordered in steps_of.values() for b, _ in ordered})
    points: list[tuple[int, str | None, int]] = []
    for budget in change_budgets:
        counts = {name: _count_at_budget(steps_of[name], keys_of[name], budget) for name in names}
        top = max(counts.values())
        leaders = [name for name in names if counts[name] == top]
        if len(leaders) == 1:
            leader: str | None = leaders[0]
            rest = [c for name, c in counts.items() if name != leader]
            margin = top - max(rest) if rest else 0
        else:
            leader = None
            margin = 0
        if points and points[-1][1] == leader and points[-1][2] == margin:
            continue
        points.append((budget, leader, margin))
    return points


def _count_at_budget(steps: list[tuple[int, int]], keys: list[int], budget: int) -> int:
    """Pass count at ``budget`` from sorted (budget, count) step points.

    ``keys`` is the precomputed ``[b for b, _ in steps]`` list, hoisted by
    the caller: building it per call made the margin trajectory quadratic
    in the point count on foreign payloads (perf finding 1A).
    """
    index = bisect_right(keys, budget) - 1
    return steps[index][1] if index >= 0 else 0


_SAFE_FLOOR_ABSENT: Any = object()
"""Sentinel distinguishing a missing ``safe_budget_floor`` field from None."""


def safe_budget_floor(frontier: Any) -> int | None:
    """First token budget from which the native decision never flips.

    Derived exactly from the recorded ``flip_ranges``: the merged spans
    over 1..max_new_tokens where the post-hoc selection decision differs
    from native. A recorded ``safe_budget_floor`` field wins when present
    and well-typed (an int >= 1, or None meaning "no floor in sweep");
    otherwise the floor is derived here, so payloads recorded before the
    field existed still get an exact answer.

    Returns 1 when nothing flips, the budget just past the last flip
    span when that span ends before the sweep maximum, and None when a
    flip span reaches the maximum (the decision still differs from
    native at the top of the sweep) or the payload is unusable.
    Defensive: spans are re-validated, clipped to the sweep, re-sorted
    and re-merged, so hand-built payloads cannot produce a wrong floor.
    Never raises.
    """
    if type(frontier) is not dict:
        return None
    recorded = frontier.get("safe_budget_floor", _SAFE_FLOOR_ABSENT)
    if recorded is not _SAFE_FLOOR_ABSENT and (
        recorded is None or (type(recorded) is int and recorded >= 1)
    ):
        return recorded
    max_tokens = frontier.get("max_new_tokens")
    if type(max_tokens) is not int or max_tokens <= 0:
        return None
    spans: list[list[int]] = []
    for span in frontier.get("flip_ranges") or []:
        if type(span) is not list or len(span) != 2:
            continue
        lo, hi = span
        if type(lo) is not int or type(hi) is not int:
            continue
        lo = max(1, lo)
        hi = min(max_tokens, hi)
        if lo > hi:
            continue
        spans.append([lo, hi])
    spans.sort()
    merged: list[list[int]] = []
    for lo, hi in spans:
        if merged and merged[-1][1] >= lo - 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    if not merged:
        return 1
    last_hi = merged[-1][1]
    return last_hi + 1 if last_hi < max_tokens else None


def token_overhead(result: dict[str, Any]) -> dict[str, Any]:
    """Per-candidate token counts from the recorded observations.

    Prompt tokens per candidate per fixture are the recorded encoded-input
    lengths: ``evaluate`` stores the backend's exact encoding of the
    rendered prompt, and the backend contract requires encoding the
    complete rendered prompt with no implicit BOS/EOS insertion -- so
    ``len(input_ids)`` is exactly
    ``len(tokenizer.encode(rendered_prompt, add_special_tokens=False))``
    with the run's own checkpoint tokenizer. Zero new model calls. Output
    counts are the recorded retained-output lengths. Means are over
    completed observations only; candidates with none get null stats.
    ``cheapest_candidate`` is the smallest mean prompt count
    (lexicographically-smallest name breaks ties);
    ``winner_tax_vs_cheapest_pct`` is the selected winner's mean prompt
    count relative to the cheapest, in percent -- 0.0 when the winner is
    the cheapest -- or null when there is no selected winner or no usable
    counts.
    """
    if type(result) is not dict:
        raise ValueError("result must be a dict")
    per_candidate: dict[str, dict[str, float | int | None]] = {}
    for run in result.get("search") or []:
        if type(run) is not dict:
            continue
        fmt = run.get("format")
        name = fmt.get("name") if type(fmt) is dict else None
        if type(name) is not str or not name or name in per_candidate:
            continue
        prompt_counts: list[int] = []
        output_counts: list[int] = []
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
    cheapest: str | None = None
    cheapest_mean = 0.0
    for name in sorted(per_candidate):
        mean = per_candidate[name]["mean_prompt_tokens"]
        if type(mean) is not float:
            continue
        if cheapest is None or mean < cheapest_mean or (mean == cheapest_mean and name < cheapest):
            cheapest, cheapest_mean = name, mean
    tax: float | None = None
    winner_mean: float | None = None
    winner = result.get("selected_candidate")
    if type(winner) is str and winner in per_candidate:
        candidate_mean = per_candidate[winner]["mean_prompt_tokens"]
        if type(candidate_mean) is float:
            winner_mean = candidate_mean
    if cheapest is not None and winner_mean is not None and cheapest_mean > 0:
        tax = (winner_mean - cheapest_mean) / cheapest_mean * 100.0
    return {
        "per_candidate": {name: per_candidate[name] for name in sorted(per_candidate)},
        "cheapest_candidate": cheapest,
        "winner_tax_vs_cheapest_pct": tax,
    }


def attach_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    """Copy of ``result`` with the automatic diagnostics attached.

    Adds ``selection_margin`` and ``token_overhead`` (both pure functions
    of the recorded search outcomes; see their docstrings) as top-level
    keys so ``results.json`` carries them. The input dict is not mutated.
    """
    if type(result) is not dict:
        raise ValueError("result must be a dict")
    enriched = dict(result)
    enriched["selection_margin"] = selection_margin(result)
    enriched["token_overhead"] = token_overhead(result)
    return enriched


def _plural(count: int, singular: str) -> str:
    return f"{count} {singular}" + ("" if count == 1 else "s")


def _selection_margin_line(result: dict[str, Any]) -> str:
    """One-line decision-fragility note for the decision banner.

    Renders the ``selection_margin`` object as a hoverable line plus a
    visible explainer sentence. Returns "" when the margin is absent, so
    failed/cancelled runs render exactly as before. Never alters the
    decision itself.
    """
    margin = selection_margin(result)
    if margin is None:
        return ""
    value = margin["value"]
    direction = margin["direction"]
    runner = margin["runner_up"]
    fixtures = margin["decisive_fixtures"]
    ids_text = " (" + ", ".join(fixtures) + ")" if fixtures else ""
    descriptive = "The margin is descriptive; it never changes the decision."
    if direction == "to_tie" and value is None:
        label = "Selection margin: n/a (single candidate)"
        tip = "With a single candidate there is no runner-up to tie against."
        explainer = "There is no runner-up, so the margin is undefined. " + descriptive
    elif direction == "to_tie":
        unit = _plural(int(value), "fixture")
        label = f"Selection margin: {unit} (runner-up: {runner})"
        tip = f"Flipping {value} of {runner}'s failing fixtures to pass would tie the selection."
        explainer = (
            f"Flipping {value} of {runner}'s failing fixtures{ids_text} to pass "
            f"would tie the selection. {descriptive}"
        )
    elif direction == "to_winner":
        unit = _plural(int(value), "fixture")
        label = f"Selection margin: {unit} (closest: {runner})"
        tip = f"Flipping {value} of {runner}'s failing fixtures to pass would select {runner}."
        explainer = (
            f"Flipping {value} of {runner}'s failing fixtures{ids_text} to pass "
            f"would select {runner}. {descriptive}"
        )
    else:  # "to_unique"
        unit = _plural(int(value), "candidate")
        tied = int(value) + 1
        label = f"Selection margin: {unit}"
        tip = f"Demoting {unit} of the {tied} tied candidates would leave a unique winner."
        if fixtures:
            explainer = (
                f"Flipping one of the shared fixtures{ids_text} to fail for {unit} "
                f"of the {tied} tied candidates would leave a unique winner. {descriptive}"
            )
        else:
            explainer = (
                f"Demoting {unit} of the {tied} tied candidates -- one flipped "
                f"fixture each -- would leave a unique winner. {descriptive}"
            )
    return (
        '<span title="' + escape(tip) + '">' + escape(label) + "</span> \u2014 " + escape(explainer)
    )


def _token_overhead_section(result: dict[str, Any]) -> str:
    """Efficiency card: per-candidate token counts and the winner's tax.

    Renders ``token_overhead`` as a sortable table plus the winner's token
    tax versus the cheapest candidate. Mirrors the README's disclaimer
    verbatim: counts are reported, not equal-compute claims.
    """
    overhead = token_overhead(result)
    per_candidate = overhead["per_candidate"]
    if not per_candidate:
        return (
            '<section id="token-overhead"><h2>Token overhead</h2>'
            "<p>Token overhead unavailable: no search candidates were recorded.</p>"
            "</section>"
        )
    winner = result.get("selected_candidate")
    cheapest = overhead["cheapest_candidate"]
    rows = []
    for name in per_candidate:
        stats = per_candidate[name]
        marker = " \u2605" if name == winner else ""
        cells = []
        for key in ("mean_prompt_tokens", "max_prompt_tokens", "mean_output_tokens"):
            stat = stats[key]
            if stat is None:
                cells.append('<td data-sort="">\u2014</td>')
            else:
                text = repr(stat)
                cells.append('<td data-sort="' + escape(text) + '">' + escape(text) + "</td>")
        rows.append(
            '<tr><th scope="row" data-sort="'
            + escape(name)
            + '">'
            + escape(name)
            + marker
            + "</th>"
            + "".join(cells)
            + "</tr>"
        )
    tax = overhead["winner_tax_vs_cheapest_pct"]
    winner_stats = per_candidate.get(winner) if type(winner) is str else None
    cheapest_stats = per_candidate.get(cheapest) if type(cheapest) is str else None
    winner_mean = winner_stats["mean_prompt_tokens"] if type(winner_stats) is dict else None
    cheapest_mean = cheapest_stats["mean_prompt_tokens"] if type(cheapest_stats) is dict else None
    if (
        tax is not None
        and type(winner_mean) is float
        and type(cheapest_mean) is float
        and type(winner) is str
        and type(cheapest) is str
    ):
        if winner == cheapest:
            tax_line = (
                "<p>The winner ("
                + escape(winner)
                + ") is also the cheapest candidate ("
                + escape(repr(winner_mean))
                + " mean prompt tokens): 0.0% token tax.</p>"
            )
        else:
            tax_line = (
                "<p>Winner token tax vs cheapest: "
                + escape(winner)
                + " averages "
                + escape(repr(winner_mean))
                + " prompt tokens vs "
                + escape(repr(cheapest_mean))
                + " for "
                + escape(cheapest)
                + " \u2014 a token tax of "
                + escape(repr(tax))
                + "%.</p>"
            )
    elif type(winner) is str and winner:
        tax_line = (
            "<p>The winner's token tax is unavailable: no usable token counts were recorded.</p>"
        )
    else:
        tax_line = "<p>No candidate was selected, so no winner token tax is reported.</p>"
    return (
        '<section id="token-overhead"><h2>Token overhead</h2>'
        "<p>Prompt tokens are the encoded length of each candidate's rendered prompt "
        '(<span class="mono">add_special_tokens=False</span>); output tokens are the '
        "retained output length. Counts are means over completed search observations.</p>"
        '<div class="table-wrap"><table class="sortable">'
        "<caption>Prompt and output token counts per candidate.</caption>"
        '<thead><tr><th scope="col">Candidate</th><th scope="col">Mean prompt tokens</th>'
        '<th scope="col">Max prompt tokens</th><th scope="col">Mean output tokens</th>'
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        + tax_line
        + "<p>Token counts in reports are retained-input/output counts, not equal-compute "
        "accounting.</p>"
        "</section>"
    )


def _search_rows(result: dict[str, Any]) -> str:
    rows = []
    for run in result.get("search") or []:
        if type(run) is not dict:
            continue
        summary = run.get("summary") or {}
        format_info = run.get("format") or {}
        name = format_info.get("name", "?")
        boundary_ids = run.get("boundary_token_ids")
        boundary_text = (
            ", ".join(str(token) for token in boundary_ids) if type(boundary_ids) is list else "?"
        )
        rate = _valid_rate(summary.get("rate"))
        if rate is None:
            cell = '<span class="badge neutral">incomplete</span>'
            bar = ""
        else:
            color = "pass" if rate == 1 else "fail"
            percent = f"{rate:.0%}"
            passes = summary.get("exact_and_boundary_passes", "?")
            planned = summary.get("planned", "?")
            bar = (
                '<span class="bar" role="img" aria-label="'
                + escape(f"{percent} pass rate")
                + '"><span class="bar-fill '
                + color
                + '" style="width:'
                + escape(percent)
                + '"></span></span>'
            )
            cell = (
                '<span class="badge '
                + color
                + '">'
                + escape(percent)
                + "</span> "
                + '<span class="rate-text">'
                + escape(f"{passes}/{planned}")
                + "</span>"
            )
        fingerprint = _short_fingerprint(run.get("candidate_fingerprint"))
        rate_key = f"{rate:.6f}" if rate is not None else ""
        attempted = summary.get("attempted")
        attempted_key = str(attempted) if type(attempted) is int else ""
        rows.append(
            '<tr><th scope="row" data-sort="'
            + escape(str(name))
            + '">'
            + escape(str(name))
            + '</th><td data-sort="'
            + escape(rate_key)
            + '">'
            + bar
            + cell
            + '</td><td data-sort="'
            + escape(str(summary.get("status", "?")))
            + '">'
            + escape(str(summary.get("status", "?")))
            + '</td><td data-sort="'
            + escape(attempted_key)
            + '">'
            + escape(f"{summary.get('attempted', '?')}/{summary.get('planned', '?')}")
            + '</td><td class="mono" data-sort="'
            + escape(boundary_text)
            + '">'
            + escape(boundary_text)
            + '</td><td data-sort="'
            + escape(str(run.get("candidate_fingerprint", "")))
            + '">'
            + '<details name="candidate-details"><summary class="mono">'
            + escape(fingerprint)
            + "</summary>"
            + '<p class="mono">'
            + escape(str(run.get("candidate_fingerprint", "")))
            + "<br>encoded input tokens: "
            + escape(str(summary.get("encoded_input_tokens", "?")))
            + "<br>retained output tokens: "
            + escape(str(summary.get("retained_output_tokens", "?")))
            + "</p></details>"
            + "</td></tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="6">No candidates were evaluated in this run.</td></tr>')
    return "".join(rows)


def _contamination_line(contamination: Any) -> str:
    """One-line fixture-hygiene summary for the run-configuration section.

    Advisory: a flagged worst tier warns that the held-out split may be
    contaminated by search near-duplicates; it never changes any verdict.
    Malformed payloads render "not recorded", never crash.
    """
    if type(contamination) is not dict:
        return "not recorded"
    worst = contamination.get("worst_tier")
    pairs = contamination.get("pairs")
    count = len(pairs) if type(pairs) is list else 0
    if worst == "clean":
        return "clean: no search/held-out near-duplicate pairs"
    if type(worst) is str and count:
        return (
            f"{worst}: {count} near-duplicate pair(s) flagged -- "
            "held-out split may be contaminated (advisory)"
        )
    return "not recorded"


def _config_section(result: dict[str, Any]) -> str:
    fingerprint = _short_fingerprint(result.get("backend_fingerprint"))
    # Per-candidate boundaries (Cycle 2): when every candidate was measured
    # with the same sequence the config shows one line, as before; when they
    # differ, each candidate's own boundary is listed explicitly.
    per_candidate: list[tuple[str, str]] = []
    for run in result.get("search") or []:
        if type(run) is not dict:
            continue
        name = (run.get("format") or {}).get("name", "?")
        ids = run.get("boundary_token_ids")
        per_candidate.append(
            (str(name), ", ".join(str(token) for token in ids) if type(ids) is list else "?")
        )
    distinct = {ids for _, ids in per_candidate}
    if len(distinct) == 1:
        boundary_items: tuple[tuple[str, str], ...] = (
            ("Boundary token IDs", next(iter(distinct))),
        )
    else:
        boundary_items = tuple((f"Boundary token IDs ({name})", ids) for name, ids in per_candidate)
    items = (
        ("Selection rule", str(result.get("selection_rule", ""))),
        ("Measurement", str(result.get("measurement", ""))),
        ("Seed", str(result.get("seed", ""))),
        ("Max new tokens", str(result.get("max_new_tokens", ""))),
        *boundary_items,
        ("Candidates", str(len(result.get("search") or []))),
        ("Fixture hygiene", _contamination_line(result.get("contamination"))),
        ("Backend fingerprint", fingerprint),
        ("Statistical confidence", str(result.get("statistical_confidence", ""))),
        ("Compute comparison", str(result.get("compute_comparison", ""))),
    )
    terms = "".join(
        "<dt>" + escape(key) + "</dt><dd>" + escape(value) + "</dd>" for key, value in items
    )
    return (
        '<section id="run-configuration"><h2>Run configuration</h2>'
        '<dl class="config">'
        + terms
        + "</dl>"
        + '<details name="config-details">'
        + '<summary>Full backend fingerprint</summary><p class="mono">'
        + escape(str(result.get("backend_fingerprint", "")))
        + "</p></details></section>"
    )


def _held_out_section(result: dict[str, Any]) -> str:
    """Held-out stage cards, or the not-evaluated notice.

    A non-dict ``held_out`` (absent or malformed) renders the same
    "Not evaluated" notice; the section does not distinguish the two.
    """
    held = result.get("held_out")
    if type(held) is not dict:
        return (
            '<section id="held-out-stage"><h2>Held-out stage</h2>'
            "<p>Not evaluated &mdash; no candidate was promoted from search.</p>"
            "</section>"
        )
    summary = held.get("summary") or {}
    valid_rate = _valid_rate(summary.get("rate"))
    rate_text = "incomplete" if valid_rate is None else f"{valid_rate:.0%}"
    cards = "".join(
        '<div class="card"><div class="k">'
        + escape(key)
        + '</div><div class="v">'
        + escape(value)
        + "</div></div>"
        for key, value in (
            ("Status", str(summary.get("status", "?"))),
            ("Pass rate", rate_text),
            (
                "Exact + boundary passes",
                f"{summary.get('exact_and_boundary_passes', '?')}/{summary.get('planned', '?')}",
            ),
            ("Aggregate eligible", "Yes" if summary.get("aggregate_eligible") else "No"),
        )
    )
    return (
        '<section id="held-out-stage"><h2>Held-out stage</h2>'
        '<div class="cards">'
        + cards
        + "</div>"
        + _held_out_floor_callout(result.get("held_out_budget_floor"))
        + "<p>Held-out observations are excluded from this HTML view; "
        "see results.json with --include-evidence for the raw records.</p>"
        "</section>"
    )


def _held_out_floor_callout(floor: Any) -> str:
    """Verification budget-floor callout for the held-out section.

    Renders nothing for missing, unavailable-marker, or malformed
    payloads; never raises.
    """
    if type(floor) is not dict or floor.get("available", True) is False:
        return ""
    native = floor.get("native_verdict")
    safe = floor.get("safe_budget_floor")
    max_tokens = floor.get("max_new_tokens", "?")
    if native == "compatible_on_fixture_set" and floor.get("budget_robust") is True:
        return (
            '<p><span class="badge pass">verification robust</span> '
            "The held-out verification passes at every token budget 1&ndash;"
            + escape(str(max_tokens))
            + ".</p>"
        )
    if native == "compatible_on_fixture_set" and type(safe) is int:
        return (
            '<p><span class="badge pass">verification floor</span> '
            "Held-out verification floor: <strong>&ge; "
            + escape(str(safe))
            + " tokens</strong> &mdash; the verification passes unchanged "
            "from budget " + escape(str(safe)) + " through " + escape(str(max_tokens)) + ".</p>"
        )
    if native == "held_out_failed":
        return (
            '<p><span class="badge fail">verification failed</span> '
            "The held-out verification failed natively; no budget floor "
            "applies &mdash; the failing fixtures never pass within the sweep.</p>"
        )
    return ""


def _sparkline(counts: list[int], total: int) -> str:
    """Text step-curve: block elements scaled 0..total, no new attack surface.

    All inputs are ints derived from recorded observations; the output is a
    fixed-alphabet string rendered like the existing pass-rate bars (with an
    aria label for screen readers).
    """
    levels = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    if not counts or total <= 0:
        return "?"
    return "".join(levels[min(7, round(count / total * 7))] for count in counts)


def _spark_svg(counts: list[int], total: int, label: str) -> str:
    """Inline-SVG step curve: one rect per sampled budget, 8 integer levels.

    The same quantized pass-count data as the text sparkline (counts
    0..``total`` mapped to 8 levels), drawn as SVG rects so the curve
    prints and scales without relying on block-element glyph coverage in
    the viewing font. All geometry is integer arithmetic over recorded
    observation counts, so the output is byte-deterministic; the accessible
    name states the exact data rather than describing the picture.
    """
    if not counts or total <= 0:
        return "?"
    bars: list[str] = []
    for index, count in enumerate(counts):
        level = min(8, round(count / total * 8))
        bars.append(
            '<rect x="'
            + str(index)
            + '" y="'
            + str(8 - level)
            + '" width="0.9" height="'
            + str(level)
            + '"/>'
        )
    width = len(counts)
    return (
        '<svg class="spark-svg" width="'
        + str(3 * width)
        + '" height="24" viewBox="0 0 '
        + str(width)
        + ' 8" role="img" aria-label="'
        + escape(label)
        + '">'
        + "".join(bars)
        + "</svg>"
    )


_FRONTIER_CHART_COLORS = (
    "#38bdf8",
    "#22c55e",
    "#f59e0b",
    "#ef4444",
    "#a78bfa",
    "#f472b6",
    "#2dd4bf",
    "#facc15",
)
"""Deterministic per-candidate palette for the frontier chart.

Assigned in sorted candidate-name order (cycled when there are more
candidates than colors), so the same frontier always renders the same
colors regardless of dict insertion order or hash seed.
"""

_FRONTIER_CHART_W = 560
_FRONTIER_CHART_H = 240

_DECISION_RIBBON_COLORS = {
    "selected": "#16a34a",
    "ambiguous": "#d97706",
    "no_candidate": "#64748b",
}
"""Ribbon colors for the selection-decision band under the frontier chart.

Darker shades than the candidate palette so the two legends never share
a color; unknown decision values fall back to the neutral gray.
"""

_DECISION_RIBBON_LABELS = {
    "selected": "selected",
    "ambiguous": "ambiguous",
    "no_candidate": "no candidate",
}
_DECISION_RIBBON_ORDER = ("selected", "ambiguous", "no_candidate")


def _frontier_chart_svg(frontier: dict[str, Any]) -> str:
    """Multi-series token-budget frontier chart as deterministic inline SVG.

    One exact step curve per candidate: pass count (0..n fixtures) over
    token budget (1..max_new_tokens), drawn from each candidate's recorded
    ``pass_steps``. Axes with integer tick labels, gridlines, a wrapped
    legend, and the native winner drawn last with a heavier stroke. Below
    the x-axis, a decision ribbon colors every budget span by the
    post-hoc selection outcome (selected / ambiguous / no candidate)
    from the recorded ``decision_steps``.

    All geometry is integer arithmetic over recorded observation counts,
    the palette is assigned in sorted name order, and all dynamic text is
    HTML-escaped, so the output is byte-deterministic across processes and
    hash seeds. The SVG carries no scripts or external references, so the
    report's self-contained/no-external-asset constraint holds.
    Returns "" when there is nothing chartable.
    """
    candidates = frontier.get("candidates")
    if type(candidates) is not dict or not candidates:
        return ""
    max_tokens = frontier.get("max_new_tokens")
    if type(max_tokens) is not int or max_tokens <= 0:
        return ""
    # Validate each series: strictly-typed step points inside the budget
    # range; malformed entries are skipped, never crash the report.
    series: list[tuple[str, int, list[tuple[int, int]], int | None]] = []
    for name in _sorted_str_keys(candidates):
        data = candidates[name]
        if type(data) is not dict:
            continue
        total = _fixture_total(data)
        if total <= 0:
            continue
        by_budget: dict[int, int] = {}
        for point in data.get("pass_steps") or []:
            if (
                type(point) is list
                and len(point) == 2
                and type(point[0]) is int
                and type(point[1]) is int
                and 1 <= point[0] <= max_tokens
                and 0 <= point[1] <= total
            ):
                by_budget[point[0]] = point[1]
        steps = sorted(by_budget.items())
        if not steps:
            continue
        full = data.get("full_pass_budget")
        series.append((name, total, steps, full if type(full) is int else None))
    if not series:
        return ""

    # Decision spans: (lo, hi, decision) with 1 <= lo <= hi <= max_tokens,
    # derived from the recorded decision_steps; adjacent spans with the
    # same decision merge. Malformed entries are skipped.
    raw_decisions = frontier.get("decision_steps") or []
    dec_points: list[tuple[int, str]] = []
    for point in raw_decisions:
        if (
            type(point) is list
            and len(point) == 3
            and type(point[0]) is int
            and type(point[1]) is str
        ):
            dec_points.append((point[0], point[1]))
    dec_points.sort()
    dec_spans: list[tuple[int, int, str]] = []
    for i, (budget, decision) in enumerate(dec_points):
        lo = max(1, min(budget, max_tokens))
        hi = dec_points[i + 1][0] - 1 if i + 1 < len(dec_points) else max_tokens
        hi = max(lo, min(hi, max_tokens))
        if dec_spans and dec_spans[-1][2] == decision and dec_spans[-1][1] == lo - 1:
            dec_spans[-1] = (dec_spans[-1][0], hi, decision)
        else:
            dec_spans.append((lo, hi, decision))

    width, height = _FRONTIER_CHART_W, _FRONTIER_CHART_H
    denom_x = max(max_tokens - 1, 1)

    def x_of(budget: int) -> int:
        return (budget - 1) * width // denom_x

    # One shared Y axis over the largest fixture total.
    y_total = max(total for _, total, _, _ in series)

    def y_of(count: int) -> int:
        return (y_total - count) * height // y_total

    # X ticks: up to 5 integer budgets spanning 1..max_tokens.
    tick_k = min(5, max_tokens)
    x_ticks = (
        sorted({1 + i * (max_tokens - 1) // (tick_k - 1) for i in range(tick_k)})
        if tick_k > 1
        else [1]
    )
    # Y ticks: every integer when few fixtures, quartiles otherwise.
    y_ticks = (
        list(range(y_total + 1)) if y_total <= 6 else sorted({i * y_total // 4 for i in range(5)})
    )

    elems: list[str] = []
    for tick in x_ticks:
        x = x_of(tick)
        elems.append(
            '<line x1="'
            + str(x)
            + '" y1="0" x2="'
            + str(x)
            + '" y2="'
            + str(height)
            + '" stroke="#1e293b" stroke-width="1"/>'
        )
        elems.append(
            '<text x="'
            + str(x)
            + '" y="'
            + str(height + 16)
            + '" text-anchor="middle" font-size="10" fill="#a9bcd2">'
            + str(tick)
            + "</text>"
        )
    for tick in y_ticks:
        y = y_of(tick)
        elems.append(
            '<line x1="0" y1="'
            + str(y)
            + '" x2="'
            + str(width)
            + '" y2="'
            + str(y)
            + '" stroke="#1e293b" stroke-width="1"/>'
        )
        elems.append(
            '<text x="-6" y="'
            + str(y + 3)
            + '" text-anchor="end" font-size="10" fill="#a9bcd2">'
            + str(tick)
            + "</text>"
        )
    # Axes.
    elems.append(
        '<line x1="0" y1="0" x2="0" y2="' + str(height) + '" stroke="#475569" stroke-width="1"/>'
    )
    elems.append(
        '<line x1="0" y1="'
        + str(height)
        + '" x2="'
        + str(width)
        + '" y2="'
        + str(height)
        + '" stroke="#475569" stroke-width="1"/>'
    )
    elems.append(
        '<text x="'
        + str(width // 2)
        + '" y="'
        + str(height + 28)
        + '" text-anchor="middle" font-size="10" fill="#a9bcd2">'
        "token budget</text>"
    )
    # Decision ribbon: one rect per decision span, directly below the
    # x-axis. Spans tile [0, width] exactly (x_of is monotone), so the
    # band is fully covered with no gaps or overlaps.
    ribbon_top = height + 38
    ribbon_h = 12
    for lo, hi, decision in dec_spans:
        rx = x_of(lo)
        rx1 = x_of(hi + 1) if hi < max_tokens else width
        elems.append(
            '<rect x="'
            + str(rx)
            + '" y="'
            + str(ribbon_top)
            + '" width="'
            + str(rx1 - rx)
            + '" height="'
            + str(ribbon_h)
            + '" fill="'
            + _DECISION_RIBBON_COLORS.get(decision, "#64748b")
            + '"/>'
        )
    elems.append(
        '<text x="-6" y="'
        + str(ribbon_top + ribbon_h // 2)
        + '" text-anchor="end" dominant-baseline="central" font-size="10" '
        'fill="#a9bcd2">decision</text>'
    )

    # The native winner draws last (on top) with a heavier stroke.
    # Colors identify candidates: assigned in sorted name order via
    # _frontier_candidate_colors, so a candidate keeps its color
    # regardless of which candidate won -- and matches the margin
    # trajectory chart below.
    winner = frontier.get("native_winner")
    color_of = _frontier_candidate_colors(frontier)
    ordered = sorted(series, key=lambda item: (item[0] == winner, item[0]))
    label_bits: list[str] = []
    drawn: list[tuple[str, str]] = []
    for name, _total, steps, full in ordered:
        color = color_of[name]
        first_budget, first_count = steps[0]
        path = "M " + str(x_of(first_budget)) + " " + str(y_of(first_count))
        for budget, count in steps[1:]:
            path += " H " + str(x_of(budget)) + " V " + str(y_of(count))
        path += " H " + str(x_of(max_tokens))
        stroke_w = "3" if name == winner else "2"
        elems.append(
            '<path d="'
            + path
            + '" fill="none" stroke="'
            + color
            + '" stroke-width="'
            + stroke_w
            + '"/>'
        )
        for budget, count in steps:
            elems.append(
                '<circle cx="'
                + str(x_of(budget))
                + '" cy="'
                + str(y_of(count))
                + '" r="2.5" fill="'
                + color
                + '"/>'
            )
        drawn.append((name, color))
        label_bits.append(
            name
            + ": "
            + str(steps[0][1])
            + " passing at budget 1"
            + (", full pass at budget " + str(full) if full is not None else ", never fully passes")
        )

    # Safe-floor marker: a dashed vertical line at the first budget from
    # which the native decision never flips, spanning the plot and the
    # decision ribbon. Drawn on top of the curves. Skipped when the floor
    # is 1 (it would sit on the y-axis) or absent.
    floor = safe_budget_floor(frontier)
    floor_marker: int | None = None
    if floor is not None and 1 < floor <= max_tokens:
        floor_marker = floor
        fx = x_of(floor)
        ribbon_bottom = ribbon_top + ribbon_h
        elems.append(
            '<line x1="'
            + str(fx)
            + '" y1="0" x2="'
            + str(fx)
            + '" y2="'
            + str(ribbon_bottom)
            + '" stroke="#fbbf24" stroke-width="2" stroke-dasharray="5 4"/>'
        )
        label_bits.append("safe budget floor at budget " + str(floor))

    # Legend: decision swatches first (they key the ribbon above), then
    # candidates. Wrapped rows below the plot; item widths use a fixed
    # per-character estimate so layout is byte-deterministic.
    legend_top = height + 58
    lx, ly = 0, legend_top
    row_h = 18
    legend_elems: list[str] = []
    legend_items: list[tuple[str, str]] = [
        (_DECISION_RIBBON_LABELS[d], _DECISION_RIBBON_COLORS[d])
        for d in _DECISION_RIBBON_ORDER
        if any(s[2] == d for s in dec_spans)
    ]
    legend_items.extend(drawn)
    for name, color in legend_items:
        item_w = 14 + 7 * len(name) + 12
        if lx > 0 and lx + item_w > width:
            lx, ly = 0, ly + row_h
        legend_elems.append(
            '<rect x="'
            + str(lx)
            + '" y="'
            + str(ly)
            + '" width="10" height="10" fill="'
            + color
            + '"/>'
        )
        legend_elems.append(
            '<text x="'
            + str(lx + 14)
            + '" y="'
            + str(ly + 5)
            + '" dominant-baseline="central" font-size="11" fill="#f8fafc">'
            + escape(name)
            + "</text>"
        )
        lx += item_w
    if floor_marker is not None:
        # The floor marker gets a dashed-line swatch so the vertical line
        # in the plot is keyed, reusing the same deterministic wrapping.
        floor_name = "safe floor \u2265 " + str(floor_marker)
        item_w = 14 + 7 * len(floor_name) + 12
        if lx > 0 and lx + item_w > width:
            lx, ly = 0, ly + row_h
        legend_elems.append(
            '<line x1="'
            + str(lx)
            + '" y1="'
            + str(ly + 5)
            + '" x2="'
            + str(lx + 10)
            + '" y2="'
            + str(ly + 5)
            + '" stroke="#fbbf24" stroke-width="2" stroke-dasharray="4 3"/>'
        )
        legend_elems.append(
            '<text x="'
            + str(lx + 14)
            + '" y="'
            + str(ly + 5)
            + '" dominant-baseline="central" font-size="11" fill="#f8fafc">'
            + escape(floor_name)
            + "</text>"
        )
        lx += item_w
    legend_h = ly - legend_top + row_h

    margin_left, margin_top, margin_right = 40, 8, 8
    outer_w = margin_left + width + margin_right
    outer_h = margin_top + height + 58 + legend_h
    decision_bits = [
        "decision "
        + _DECISION_RIBBON_LABELS.get(decision, decision)
        + " at budgets "
        + str(lo)
        + "\u2013"
        + str(hi)
        for lo, hi, decision in dec_spans
    ]
    aria = (
        "Token-budget frontier chart: "
        + str(len(ordered))
        + " candidates, "
        + str(y_total)
        + " fixtures, budgets 1 to "
        + str(max_tokens)
        + ". "
        + "; ".join(label_bits)
        + ("; " + "; ".join(decision_bits) if decision_bits else "")
    )
    return (
        '<svg class="frontier-svg" width="'
        + str(outer_w)
        + '" height="'
        + str(outer_h)
        + '" viewBox="0 0 '
        + str(outer_w)
        + " "
        + str(outer_h)
        + '" role="img" aria-label="'
        + escape(aria)
        + '">'
        + '<g transform="translate('
        + str(margin_left)
        + ","
        + str(margin_top)
        + ')">'
        + "".join(elems)
        + "".join(legend_elems)
        + "</g></svg>"
    )


def _margin_chart_svg_block(frontier: dict[str, Any]) -> str:
    """The margin trajectory chart wrapped with heading and note."""
    chart_svg = _margin_chart_svg(frontier)
    if not chart_svg:
        return ""
    return (
        "<h3>Margin trajectory: how decisive was the lead</h3>"
        '<div class="chart-wrap">' + chart_svg + "</div>"
        '<p class="chart-note">At each token budget, the margin is the '
        "leading candidate\u2019s pass count minus the best remaining "
        "candidate\u2019s; a tied lead has margin 0. Each step segment is "
        "colored by the leading candidate, using the same palette as the "
        "frontier chart above (gray: tied lead). Derived post-hoc from the "
        "recorded pass steps; no extra model calls. The dashed amber line "
        "marks the safe budget floor, as in the frontier chart.</p>"
    )


def _margin_chart_svg(frontier: dict[str, Any]) -> str:
    """Selection-margin trajectory as deterministic inline SVG.

    One exact step curve: at each token budget, the leading candidate's
    pass count minus the best remaining candidate's (0 on a tied lead),
    from :func:`margin_trajectory`. Each step segment is stroked in the
    leading candidate's frontier-chart color (gray for tied leads), so the
    chart reads as "who was ahead, and by how much, at every budget".
    Same x-scale as the frontier chart above; integer geometry, integer
    tick labels, HTML-escaped text, no scripts or external references.
    Returns "" when there is nothing chartable.
    """
    traj = margin_trajectory(frontier)
    if not traj:
        return ""
    validated = _frontier_series(frontier)
    if validated is None:  # margin_trajectory already validated; defensive
        return ""
    _, max_tokens = validated
    color_of = _frontier_candidate_colors(frontier)

    width, height = _FRONTIER_CHART_W, _FRONTIER_CHART_H
    denom_x = max(max_tokens - 1, 1)

    def x_of(budget: int) -> int:
        return (budget - 1) * width // denom_x

    y_max = max(margin for _, _, margin in traj)
    denom_y = max(y_max, 1)

    def y_of(margin: int) -> int:
        return (y_max - margin) * height // denom_y

    elems: list[str] = []
    # X ticks mirror the frontier chart: up to 5 integer budgets.
    tick_k = min(5, max_tokens)
    x_ticks = (
        sorted({1 + i * (max_tokens - 1) // (tick_k - 1) for i in range(tick_k)})
        if tick_k > 1
        else [1]
    )
    for tick in x_ticks:
        x = x_of(tick)
        elems.append(
            '<line x1="'
            + str(x)
            + '" y1="0" x2="'
            + str(x)
            + '" y2="'
            + str(height)
            + '" stroke="#1e293b" stroke-width="1"/>'
        )
        elems.append(
            '<text x="'
            + str(x)
            + '" y="'
            + str(height + 16)
            + '" text-anchor="middle" font-size="10" fill="#a9bcd2">'
            + str(tick)
            + "</text>"
        )
    y_ticks = list(range(y_max + 1)) if y_max <= 6 else sorted({i * y_max // 4 for i in range(5)})
    for tick in y_ticks:
        y = y_of(tick)
        stroke = "#475569" if tick == 0 else "#1e293b"
        elems.append(
            '<line x1="0" y1="'
            + str(y)
            + '" x2="'
            + str(width)
            + '" y2="'
            + str(y)
            + '" stroke="'
            + stroke
            + '" stroke-width="1"/>'
        )
        elems.append(
            '<text x="-6" y="'
            + str(y + 3)
            + '" text-anchor="end" font-size="10" fill="#a9bcd2">'
            + str(tick)
            + "</text>"
        )
    # Axes and labels.
    elems.append(
        '<line x1="0" y1="0" x2="0" y2="' + str(height) + '" stroke="#475569" stroke-width="1"/>'
    )
    elems.append(
        '<line x1="0" y1="'
        + str(height)
        + '" x2="'
        + str(width)
        + '" y2="'
        + str(height)
        + '" stroke="#475569" stroke-width="1"/>'
    )
    elems.append(
        '<text x="'
        + str(width // 2)
        + '" y="'
        + str(height + 28)
        + '" text-anchor="middle" font-size="10" fill="#a9bcd2">'
        "token budget</text>"
    )
    elems.append(
        '<text x="-6" y="-8" text-anchor="end" font-size="10" fill="#a9bcd2">'
        "lead margin (passes)</text>"
    )

    # Area under the curve, drawn first (behind the segments).
    area = "M " + str(x_of(traj[0][0])) + " " + str(y_of(traj[0][2]))
    for budget, _, margin in traj[1:]:
        area += " H " + str(x_of(budget)) + " V " + str(y_of(margin))
    area += " H " + str(x_of(max_tokens)) + " V " + str(height)
    area += " H " + str(x_of(traj[0][0])) + " Z"
    elems.append('<path d="' + area + '" fill="#38bdf8" fill-opacity="0.10"/>')

    # Step segments, each stroked in its leader's color (gray on ties).
    seg_bits: list[str] = []
    extended = traj + [(max_tokens, traj[-1][1], traj[-1][2])]
    for (b0, leader0, m0), (b1, _leader1, m1) in zip(extended, extended[1:], strict=False):
        color = color_of.get(leader0, _MARGIN_TIE_COLOR) if leader0 else _MARGIN_TIE_COLOR
        elems.append(
            '<path d="M '
            + str(x_of(b0))
            + " "
            + str(y_of(m0))
            + " H "
            + str(x_of(b1))
            + " V "
            + str(y_of(m1))
            + '" fill="none" stroke="'
            + color
            + '" stroke-width="2.5"/>'
        )
        seg_bits.append(
            "budgets "
            + str(b0)
            + "\u2013"
            + str(b1)
            + ": leader "
            + (leader0 if leader0 else "tied")
            + ", margin "
            + str(m0)
        )
    # Safe-floor marker, same x-scale and style as the frontier chart:
    # the first budget from which the native decision never flips.
    floor = safe_budget_floor(frontier)
    if floor is not None and 1 < floor <= max_tokens:
        fx = x_of(floor)
        elems.append(
            '<line x1="'
            + str(fx)
            + '" y1="0" x2="'
            + str(fx)
            + '" y2="'
            + str(height)
            + '" stroke="#fbbf24" stroke-width="2" stroke-dasharray="5 4"/>'
        )
        seg_bits.append("safe budget floor at budget " + str(floor))
    aria = (
        "Selection margin trajectory chart: "
        + str(len(color_of))
        + " candidates, budgets 1 to "
        + str(max_tokens)
        + ". "
        + "; ".join(seg_bits)
    )
    margin_left, margin_top, margin_right = 40, 16, 8
    outer_w = margin_left + width + margin_right
    outer_h = margin_top + height + 34
    return (
        '<svg class="margin-svg" width="'
        + str(outer_w)
        + '" height="'
        + str(outer_h)
        + '" viewBox="0 0 '
        + str(outer_w)
        + " "
        + str(outer_h)
        + '" role="img" aria-label="'
        + escape(aria)
        + '">'
        + '<g transform="translate('
        + str(margin_left)
        + ","
        + str(margin_top)
        + ')">'
        + "".join(elems)
        + "</g></svg>"
    )


def _frontier_chart_svg_block(frontier: dict[str, Any]) -> str:
    """The frontier chart wrapped with its heading and explanatory note."""
    chart_svg = _frontier_chart_svg(frontier)
    if not chart_svg:
        return ""
    return (
        "<h3>Frontier chart: all candidates</h3>"
        '<div class="chart-wrap">' + chart_svg + "</div>"
        '<p class="chart-note">Exact step curves: each candidate\u2019s pass '
        "count at every token budget, derived post-hoc from the recorded "
        "minimal passing budgets (no extra model calls). The native "
        "winner, when one exists, is drawn last with a heavier stroke. "
        "The decision ribbon below the axis colors every budget span by "
        "the post-hoc selection outcome. A dashed amber line marks the "
        "safe budget floor (first budget from which the native decision "
        "never flips) when one exists inside the sweep.</p>"
    )


def _frontier_spark_table(frontier: dict[str, Any]) -> str:
    """One sparkline per candidate: pass count over token budgets 1..max."""
    candidates = frontier.get("candidates")
    if type(candidates) is not dict or not candidates:
        return ""
    max_tokens = frontier.get("max_new_tokens")
    if type(max_tokens) is not int or max_tokens <= 0:
        return ""
    # Sample at most 48 budgets; the curve is a step function derived from
    # the recorded per-fixture minimal passing budgets (exact, post-hoc).
    if max_tokens <= 48:
        budgets = list(range(1, max_tokens + 1))
    else:
        budgets = sorted({1 + index * (max_tokens - 1) // 47 for index in range(48)})
    rows = []
    for name, data in candidates.items():
        if type(name) is not str or type(data) is not dict:
            continue
        total = _fixture_total(data)
        mins_raw = data.get("min_budget_per_fixture")
        mins = mins_raw if type(mins_raw) is list else []
        counts = [sum(1 for mb in mins if type(mb) is int and mb <= budget) for budget in budgets]
        label = f"pass count {counts[0]} to {counts[-1]} of {total} over budgets 1 to {max_tokens}"
        spark = _spark_svg(counts, total, label)
        full = data.get("full_pass_budget")
        rows.append(
            '<tr><th scope="row" data-sort="'
            + escape(name)
            + '">'
            + escape(name)
            + '</th><td data-sort="'
            + str(counts[-1])
            + '">'
            + spark
            + '</td><td data-sort="'
            + (str(full) if type(full) is int else "")
            + '">'
            + escape(str(full) if full is not None else "\u2014")
            + "</td></tr>"
        )
    if not rows:
        return ""
    return (
        "<h3>Pass curves by candidate</h3>"
        '<div class="table-wrap"><table class="sortable">'
        "<caption>Search fixtures passing at or below each token budget, "
        "sampled left to right from budget 1 to " + escape(str(max_tokens)) + ".</caption>"
        '<thead><tr><th scope="col">Candidate</th>'
        '<th scope="col">Pass curve</th><th scope="col">Full-pass budget</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _frontier_section(result: dict[str, Any]) -> str:
    """Token-budget robustness frontier: post-hoc and exact.

    Shows the winner's pass curve over token budgets and the budget ranges
    where the selection decision or winner would have differed from native.
    Renders nothing when the frontier was unavailable.
    """
    frontier: Any = result.get("budget_frontier")
    if type(frontier) is not dict or type(frontier.get("candidates")) is not dict:
        return ""
    max_tokens = frontier.get("max_new_tokens", "?")
    parts = [
        '<section id="token-budget-frontier"><h2>Token-budget frontier</h2>',
        "<p>Post-hoc recomputation of the search-stage decision at every "
        "smaller token budget, from the recorded outputs alone (exact under "
        "greedy decoding; no extra model calls). Passing at budget <i>b</i> "
        "means the observation passed natively with its boundary completing "
        "within <i>b</i> tokens.</p>",
    ]
    if frontier.get("budget_robust"):
        parts.append(
            '<p><span class="badge pass">stable</span> The selection decision is '
            "identical at every token budget 1&ndash;" + escape(str(max_tokens)) + ".</p>"
        )
    else:
        steps = frontier.get("decision_steps") or []
        # Flip-span attribution, keyed defensively by span: foreign or older
        # payloads may carry no attribution, or one that does not line up
        # with the rendered spans -- rendering never raises.
        attribution: dict[tuple[Any, Any], list[Any]] = {}
        raw_attribution = frontier.get("flip_attribution")
        if type(raw_attribution) is list:
            for entry in raw_attribution:
                if (
                    type(entry) is dict
                    and type(entry.get("span")) is list
                    and len(entry["span"]) == 2
                    and type(entry.get("crossings")) is list
                ):
                    attribution[(entry["span"][0], entry["span"][1])] = entry["crossings"]
        items = []
        for span in frontier.get("flip_ranges") or []:
            if type(span) is not list or len(span) != 2:
                continue
            lo = span[0]
            flip_decision, flip_winner = "?", None
            for point in steps:
                if type(point) is list and len(point) == 3 and point[0] <= lo:
                    flip_decision, flip_winner = point[1], point[2]
            label = (
                "budgets "
                + escape(str(lo))
                + "&ndash;"
                + escape(str(span[1]))
                + ": "
                + escape(str(flip_decision))
            )
            if flip_winner:
                label += " (winner: " + escape(str(flip_winner)) + ")"
            crossings = attribution.get((lo, span[1]))
            if crossings is None:
                items.append("<li>" + label + "</li>")
                continue
            detail_bits = []
            for crossing in crossings:
                if type(crossing) is not dict:
                    continue
                detail_bits.append(
                    "<code>"
                    + escape(str(crossing.get("candidate")))
                    + "</code>/<code>"
                    + escape(str(crossing.get("fixture")))
                    + "</code> passes at budget "
                    + escape(str(crossing.get("min_budget")))
                )
            if detail_bits:
                noun = "pass-status change" if len(detail_bits) == 1 else "pass-status changes"
                detail = (
                    '<span class="attribution">'
                    + noun
                    + " inside this span: "
                    + "; ".join(detail_bits)
                    + "</span>"
                )
            else:
                detail = (
                    '<span class="attribution">no pass-status change inside this '
                    "span &mdash; the decision already differs from native at "
                    "its start</span>"
                )
            items.append("<li>" + label + detail + "</li>")
        parts.append(
            '<p><span class="badge fail">budget-sensitive</span> The selection '
            "decision would have differed at these token budgets:</p><ul>"
            + "".join(items)
            + "</ul>"
        )
    floor = safe_budget_floor(frontier)
    if floor is None:
        parts.append(
            '<p><span class="badge fail">no safe floor</span> No safe budget '
            "floor within the sweep: the decision still differs from native "
            "at the maximum budget " + escape(str(max_tokens)) + ".</p>"
        )
    elif floor <= 1:
        parts.append(
            '<p><span class="badge pass">safe floor</span> Safe budget floor: '
            "<strong>1 token</strong> &mdash; the native decision holds "
            "unchanged at every budget 1&ndash;" + escape(str(max_tokens)) + ".</p>"
        )
    else:
        parts.append(
            '<p><span class="badge pass">safe floor</span> Safe budget floor: '
            "<strong>&ge; "
            + escape(str(floor))
            + " tokens</strong> &mdash; the native decision holds unchanged "
            "from budget " + escape(str(floor)) + " through " + escape(str(max_tokens)) + ".</p>"
        )
    winner: Any = frontier.get("native_winner")
    candidates: Any = frontier.get("candidates")
    data = candidates.get(winner) if type(winner) is str else None
    if type(data) is dict:
        total = _fixture_total(data)
        rows = []
        for point in data.get("pass_steps") or []:
            if type(point) is not list or len(point) != 2:
                continue
            budget, count = point
            if type(budget) is not int or type(count) is not int:
                continue
            percent = f"{(count / total):.0%}" if total else "?"
            bar = (
                '<span class="bar" role="img" aria-label="'
                + escape(f"{percent} pass rate at budget {budget}")
                + '"><span class="bar-fill '
                + ("pass" if count == total else "fail")
                + '" style="width:'
                + escape(percent)
                + '"></span></span>'
            )
            rows.append(
                "<tr><td>&ge; "
                + escape(str(budget))
                + " tokens</td><td>"
                + bar
                + '<span class="rate-text">'
                + escape(f"{count}/{total}")
                + "</span></td></tr>"
            )
        parts.append(
            "<h3>Winner pass curve: " + escape(str(winner)) + "</h3>"
            '<div class="table-wrap"><table><caption>Search fixtures passing '
            "at or below each token budget.</caption><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )
    parts.append(_frontier_chart_svg_block(frontier))
    parts.append(_margin_chart_svg_block(frontier))
    parts.append(_frontier_spark_table(frontier))
    parts.append("</section>")
    return "".join(parts)


def _boundary_audit_section(result: dict[str, Any]) -> str:
    """Prompt-boundary canonicality: the Prompt Boundary Problem rate.

    For every stored rendered prompt x greedy completion, re-encoding the
    concatenated text must reproduce prompt_ids + completion_ids. The table
    reports the fraction of observations where it does not, per template
    family. A zero rate is a certificate of boundary-canonicality for the
    run's templates; a nonzero rate names the families whose prompts the
    tokenizer re-segments. Renders nothing when the audit is unavailable.
    """
    audit: Any = result.get("boundary_audit")
    if type(audit) is not dict or not audit.get("available"):
        return ""
    n_usable = audit.get("n_usable", 0)
    n_problem = audit.get("n_problem", 0)
    overall = audit.get("overall_rate")
    rows = []
    per_family = audit.get("per_family")
    if type(per_family) is dict:
        for family in sorted(per_family):
            entry = per_family[family]
            if type(entry) is not dict:
                continue
            rate = entry.get("rate")
            rate_text = "n/a" if type(rate) not in (int, float) else f"{rate:.1%}"
            badge = (
                '<span class="badge pass">canonical</span>'
                if rate == 0
                else '<span class="badge fail">non-canonical</span>'
                if rate is not None
                else '<span class="badge">n/a</span>'
            )
            rows.append(
                '<tr><th scope="row">' + escape(str(family)) + "</th>"
                '<td data-sort="' + escape(rate_text) + '">' + badge + "</td>"
                "<td>" + escape(rate_text) + "</td>"
                "<td>" + escape(str(entry.get("n_problem", "?"))) + "</td>"
                "<td>" + escape(str(entry.get("n_usable", "?"))) + "</td></tr>"
            )
    overall_text = "n/a" if type(overall) not in (int, float) else f"{overall:.1%}"
    parts = [
        '<section id="boundary-canonicality"><h2>Prompt-boundary canonicality</h2>',
        "<p>For every stored rendered prompt &times; greedy completion, the "
        "lab re-encodes the concatenated text and requires the joint encoding "
        "to equal the recorded <i>prompt_ids + completion_ids</i>. A "
        "mismatch is the Prompt Boundary Problem: the tokenizer re-segments "
        "the text at the prompt/completion seam, so the ids the model saw "
        "are not what the stored text would reproduce. "
        + escape(str(n_problem))
        + " of "
        + escape(str(n_usable))
        + " usable observations diverge (overall rate "
        + escape(overall_text)
        + ").</p>",
    ]
    if overall == 0:
        parts.append(
            '<p><span class="badge pass">certificate</span> Every usable '
            "observation re-encodes exactly: the run's templates are "
            "boundary-canonical on this tokenizer.</p>"
        )
    if rows:
        parts.append(
            '<div class="table-wrap"><table class="sortable">'
            "<caption>Prompt Boundary Problem rate by template family.</caption>"
            '<thead><tr><th scope="col">Family</th><th scope="col">Status</th>'
            '<th scope="col">PBP rate</th><th scope="col">Divergent</th>'
            '<th scope="col">Usable</th></tr></thead><tbody>'
            + "".join(rows)
            + "</tbody></table></div>"
        )
    parts.append("</section>")
    return "".join(parts)


def _marker_isolation_section(result: dict[str, Any]) -> str:
    """Chat-marker isolation: the families x markers conformance matrix.

    For each template family and each hostile marker string, the lab
    injected the marker into user text and counted the checkpoint
    tokenizer's control-token ids before and after. A cell is "isolated"
    when the count did not change (the marker stayed inert text) and
    "smuggled" when the count grew (the marker became real control
    tokens). Renders nothing when the check is unavailable.
    """
    check: Any = result.get("marker_isolation")
    if type(check) is not dict or check.get("status") != "ok":
        return ""
    scores: Any = check.get("family_scores")
    if type(scores) is not dict or not scores:
        return ""
    probes: Any = check.get("probes")
    # family -> marker -> worst (least isolated) probe for the matrix cell.
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    if type(probes) is list:
        for probe in probes:
            if type(probe) is not dict or probe.get("status") != "ok":
                continue
            family = str(probe.get("format", "")).split("/", 1)[0] or "unknown"
            marker = probe.get("marker")
            if type(marker) is not str:
                continue
            prev = cells.setdefault(family, {}).get(marker)
            if prev is None or (
                type(probe.get("delta")) is int
                and type(prev.get("delta")) is int
                and probe["delta"] > prev["delta"]
            ):
                cells[family][marker] = probe
    from .control_tokens import HOSTILE_MARKER_STRINGS

    header = "".join('<th scope="col">' + escape(m) + "</th>" for m in HOSTILE_MARKER_STRINGS)
    rows = []
    for family in sorted(cells):
        fam_cells = cells[family]
        tds = []
        for marker in HOSTILE_MARKER_STRINGS:
            probe = fam_cells.get(marker)
            if probe is None:
                tds.append('<td><span class="badge">n/a</span></td>')
            elif probe.get("isolated"):
                tds.append('<td><span class="badge pass">isolated</span></td>')
            else:
                delta = probe.get("delta", "?")
                tds.append(
                    '<td><span class="badge fail">smuggled</span> +' + escape(str(delta)) + "</td>"
                )
        score = scores.get(family, {})
        rate = score.get("isolation_rate")
        rate_text = "n/a" if type(rate) not in (int, float) else f"{rate:.0%}"
        rows.append(
            '<tr><th scope="row">'
            + escape(family)
            + "</th><td>"
            + escape(rate_text)
            + "</td>"
            + "".join(tds)
            + "</tr>"
        )
    overall = check.get("overall_isolation_rate")
    overall_text = "n/a" if type(overall) not in (int, float) else f"{overall:.1%}"
    parts = [
        '<section id="marker-isolation"><h2>Chat-marker isolation</h2>',
        "<p>Each hostile marker string was injected into user text and the "
        "checkpoint tokenizer's control-token ids were counted before and "
        "after. <em>Isolated</em> means the count did not change (the marker "
        "stayed inert text); <em>smuggled</em> means the marker became real "
        "control tokens — the structural-injection class behind ChatInject "
        "and Phantom. Overall isolation rate " + escape(overall_text) + ".</p>",
        '<div class="table-wrap"><table class="sortable">'
        "<caption>Marker isolation by template family.</caption>"
        '<thead><tr><th scope="col">Family</th><th scope="col">Isolated</th>'
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>",
        "</section>",
    ]
    return "".join(parts)


def _selection_robustness_section(result: dict[str, Any]) -> str:
    """Selection robustness: does the winner survive other stop semantics?

    The stop matrix re-measures only the winner under emulated stop
    semantics; this section re-applies the conservative selection rule to
    *every* search candidate under each semantic (tokenizer-only, no model
    calls) and reports whether the decision or the winner changes. A flip
    means the protocol that wins in the lab would not win on that serving
    stack -- the deployment-relevant question the winner-only matrix cannot
    answer. Renders nothing when the analysis did not run.
    """
    analysis: Any = result.get("selection_robustness")
    if type(analysis) is not dict or not analysis.get("available"):
        return ""
    labels = {
        "native": "Native (lab)",
        "hf_stop_string": "HF StopStringCriteria",
        "vllm_holdback": "vLLM detokenizer",
        "eos_appended": "EOS-appended",
    }
    per_semantic = analysis.get("per_semantic")
    per_candidate = analysis.get("per_candidate")
    flips = analysis.get("flips")
    if type(per_semantic) is not dict or type(per_candidate) is not dict:
        return ""
    semantics = [s for s in labels if s in per_semantic]
    # Verdict line: per-semantic decision + winner.
    verdict_items = []
    for semantic in semantics:
        verdict = per_semantic[semantic]
        if type(verdict) is not dict:
            continue
        decision = str(verdict.get("decision", "?"))
        winner = verdict.get("winner")
        badge_cls = (
            "pass"
            if decision == "selected"
            else ("neutral" if decision in ("ambiguous", "unavailable") else "fail")
        )
        verdict_items.append(
            "<li><strong>"
            + escape(labels[semantic])
            + ':</strong> <span class="badge '
            + badge_cls
            + '">'
            + escape(decision)
            + "</span>"
            + (
                ' winner <span class="mono">' + escape(str(winner)) + "</span>"
                if winner is not None
                else ""
            )
            + "</li>"
        )
    # Candidate x semantic pass matrix.
    names = sorted(str(n) for n in per_candidate)
    header = "".join('<th scope="col">' + escape(labels[s]) + "</th>" for s in semantics)
    body_rows = []
    for name in names:
        row = per_candidate[name]
        cells = []
        for semantic in semantics:
            passed = row.get(semantic) if type(row) is dict else None
            if passed is None:
                cells.append('<td><span class="badge neutral">n/a</span></td>')
            else:
                cells.append(
                    '<td><span class="badge '
                    + ("pass" if passed else "fail")
                    + '">'
                    + ("pass" if passed else "fail")
                    + "</span></td>"
                )
        body_rows.append(
            '<tr><td><span class="mono">' + escape(name) + "</span></td>" + "".join(cells) + "</tr>"
        )
    flip_html = ""
    if type(flips) is list and flips:
        items = []
        for flip in flips:
            if type(flip) is not dict:
                continue
            items.append(
                "<li><strong>"
                + escape(labels.get(str(flip.get("semantic", "?")), str(flip.get("semantic", "?"))))
                + ":</strong> decision "
                + escape(str(flip.get("decision_from", "?")))
                + " &rarr; "
                + escape(str(flip.get("decision_to", "?")))
                + ", winner "
                + escape(str(flip.get("winner_from")))
                + " &rarr; "
                + escape(str(flip.get("winner_to")))
                + "</li>"
            )
        flip_html = (
            "<details><summary><strong>Winner flips under other stop "
            "semantics ("
            + str(len(items))
            + ")</strong></summary><ul>"
            + "".join(items)
            + "</ul></details>"
        )
    else:
        flip_html = (
            "<p>The selection is <strong>robust</strong>: every evaluable "
            "emulated stop semantic selects the same decision and winner "
            "as native.</p>"
        )
    unavailable = analysis.get("unavailable_semantics")
    if type(unavailable) is list and unavailable:
        flip_html += (
            "<p>Not evaluable (missing emulation input, e.g. unknown EOS "
            "id): "
            + escape(", ".join(labels.get(str(s), str(s)) for s in unavailable if type(s) is str))
            + ".</p>"
        )
    excluded = analysis.get("excluded_candidates")
    if type(excluded) is list and excluded:
        flip_html += (
            "<p>Excluded from the analysis (unaligned observations): "
            + escape(", ".join(str(s) for s in excluded))
            + ".</p>"
        )
    return (
        '<section class="card"><h2>Selection robustness under stop semantics</h2>'
        "<p>Each serving stack stops generation differently. This re-applies "
        "the lab's conservative selection rule to every search candidate "
        "under each emulated stop semantic (exact answer + boundary on the "
        "truncated output); no model calls, derived from the recorded "
        "observations.</p>"
        "<ul>" + "".join(verdict_items) + "</ul>"
        '<div class="table-wrap"><table><thead><tr><th scope="col">Candidate</th>'
        + header
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table></div>"
        + flip_html
        + "</section>"
    )


def _transfer_section(result: dict[str, Any]) -> str:
    """Cross-checkpoint transfer: does the winner win elsewhere too?

    Re-applies the conservative selection rule to every search candidate
    under each transfer checkpoint (search split only) and reports
    whether the primary winner transfers, ties, or fails there. A
    checkpoint where a candidate could not be evaluated (e.g. boundary
    IDs outside its vocabulary) is recorded as incompatible, never
    silently skipped. Renders nothing when the transfer experiment did
    not run.
    """
    transfer: Any = result.get("transfer")
    if type(transfer) is not dict or not transfer.get("available"):
        return ""
    checkpoints = transfer.get("checkpoints")
    per_checkpoint = transfer.get("per_checkpoint")
    matrix = transfer.get("matrix")
    if (
        type(checkpoints) is not list
        or type(per_checkpoint) is not dict
        or type(matrix) is not dict
    ):
        return ""
    verdict = str(transfer.get("transfer_verdict", "?"))
    primary = transfer.get("primary_winner")
    verdict_labels = {
        "full": "transfers to every checkpoint",
        "partial": "transfers to some checkpoints",
        "none": "transfers to no checkpoint",
        "unavailable": "not evaluable",
    }
    cell_labels = {
        "transfers": ("pass", "transfers"),
        "wins_there": ("pass", "wins"),
        "ties_there": ("neutral", "ties"),
        "fails_there": ("fail", "fails"),
        "checkpoint_incompatible": ("neutral", "n/a"),
    }
    # Checkpoint columns: id + fingerprint.
    ckpt_ids: list[str] = []
    header_cells = []
    for ckpt in checkpoints:
        if type(ckpt) is not dict:
            continue
        cid = str(ckpt.get("id", "?"))
        ckpt_ids.append(cid)
        fingerprint = ckpt.get("fingerprint")
        header_cells.append(
            '<th scope="col"><span class="mono">'
            + escape(cid)
            + "</span>"
            + (
                '<br><span class="mono">' + escape(str(fingerprint)[:16]) + "…</span>"
                if type(fingerprint) is str and fingerprint
                else ""
            )
            + "</th>"
        )
    names = sorted(str(n) for n in matrix)
    body_rows = []
    for name in names:
        row = matrix[name]
        cells = []
        for cid in ckpt_ids:
            v = row.get(cid) if type(row) is dict else None
            badge, label = cell_labels.get(str(v), ("neutral", "?"))
            cells.append('<td><span class="badge ' + badge + '">' + label + "</span></td>")
        marker = " ★" if name == primary else ""
        body_rows.append(
            '<tr><td><span class="mono">'
            + escape(name)
            + escape(marker)
            + "</span></td>"
            + "".join(cells)
            + "</tr>"
        )
    return (
        '<section class="card"><h2>Cross-checkpoint transfer</h2>'
        "<p>Every search candidate re-evaluated against each transfer "
        "checkpoint (search split only); the selection rule re-applied per "
        "checkpoint. ★ marks the primary winner.</p>"
        "<p>Transfer verdict: <strong>"
        + escape(verdict_labels.get(verdict, verdict))
        + "</strong></p>"
        '<div class="table-wrap"><table><thead><tr><th scope="col">Candidate</th>'
        + "".join(header_cells)
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table></div>"
        "</section>"
    )


def _perturbation_section(result: dict[str, Any]) -> str:
    """Fixture-perturbation robustness: does the winner survive phrasing noise?

    Re-applies the conservative selection rule to every search candidate
    on each deterministically perturbed fixture variant (search split
    only) and reports whether the primary (decision, winner) survives.
    A variant where no candidate could be evaluated is shown as n/a,
    never as a verdict. Renders nothing when the perturbation
    experiment did not run.
    """
    perturbation: Any = result.get("perturbation")
    if type(perturbation) is not dict or not perturbation.get("available"):
        return ""
    per_variant = perturbation.get("per_variant")
    stability = perturbation.get("stability")
    per_operator = perturbation.get("per_operator")
    if (
        type(per_variant) is not dict
        or type(stability) is not dict
        or type(per_operator) is not dict
    ):
        return ""
    primary_winner = perturbation.get("primary_winner")
    primary_decision = perturbation.get("primary_decision")
    n_eval = stability.get("n_evaluable")
    retained = stability.get("winner_retained")
    decision_retained = stability.get("decision_retained")
    if type(n_eval) is int and n_eval:
        if type(retained) is int:
            stability_line = f"{retained}/{n_eval} evaluable variants retain the primary winner"
        else:
            stability_line = (
                f"{decision_retained}/{n_eval} evaluable variants retain the primary decision"
            )
    else:
        stability_line = "no variant was evaluable"
    flips = perturbation.get("flips")
    n_flips = len(flips) if type(flips) is list else 0

    op_rows = []
    for op in sorted(str(o) for o in per_operator):
        agg = per_operator[op]
        if type(agg) is not dict:
            continue
        wr = agg.get("winner_retained")
        ev = agg.get("evaluable")
        rate = agg.get("mean_changed_rate")
        if type(wr) is int and type(ev) is int and ev:
            cell = f"{wr}/{ev}"
        elif type(ev) is int and ev:
            cell = f"{ev} evaluable"
        else:
            cell = "n/a"
        rate_cell = f"{rate:.0%}" if type(rate) is float else "n/a"
        op_rows.append(
            '<tr><td><span class="mono">'
            + escape(op)
            + "</span></td><td>"
            + escape(cell)
            + "</td><td>"
            + escape(rate_cell)
            + "</td></tr>"
        )
    variant_rows = []
    for vid in sorted(str(v) for v in per_variant):
        verdict = per_variant[vid]
        if type(verdict) is not dict:
            continue
        decision = verdict.get("decision")
        winner = verdict.get("winner")
        operator = verdict.get("operator")
        if decision == "unavailable":
            badge, note = "neutral", "n/a"
        elif (decision, winner) == (primary_decision, primary_winner):
            badge, note = "pass", "matches"
        else:
            badge, note = "fail", "differs"
        variant_rows.append(
            '<tr><td><span class="mono">'
            + escape(str(vid))
            + '</span></td><td><span class="mono">'
            + escape(str(operator))
            + '</span></td><td><span class="mono">'
            + escape(str(decision))
            + '</span></td><td><span class="mono">'
            + escape(str(winner) if winner is not None else "(none)")
            + '</span></td><td><span class="badge '
            + badge
            + '">'
            + note
            + "</span></td></tr>"
        )
    return (
        '<section class="card"><h2>Fixture-perturbation robustness</h2>'
        "<p>Every search candidate re-evaluated on each deterministically "
        "perturbed fixture variant (typos, case flips, whitespace, Unicode "
        "variants, punctuation stripping; search split only); the selection "
        "rule re-applied per variant. A variant that flips the "
        "(decision, winner) is phrasing luck, not merit.</p>"
        "<p>Stability: <strong>"
        + escape(stability_line)
        + "</strong>"
        + (f" — {n_flips} variant(s) flip the primary verdict." if n_flips else ".")
        + "</p>"
        + (
            '<div class="table-wrap"><table><thead><tr><th scope="col">Operator</th>'
            '<th scope="col">Winner retained</th>'
            '<th scope="col">Fixtures changed</th>'
            "</tr></thead><tbody>" + "".join(op_rows) + "</tbody></table></div>"
            if op_rows
            else ""
        )
        + '<div class="table-wrap"><table><thead><tr><th scope="col">Variant</th>'
        '<th scope="col">Operator</th><th scope="col">Decision</th>'
        '<th scope="col">Winner</th><th scope="col">vs primary</th>'
        "</tr></thead><tbody>" + "".join(variant_rows) + "</tbody></table></div>"
        "</section>"
    )


def _template_firewall_section(result: dict[str, Any]) -> str:
    """Template firewall: the static AST verdict on the checkpoint's template.

    Reports whether the checkpoint's ``chat_template`` passed the static
    RCE-primitive scan before any render, and lists denied findings. An
    "allowed" verdict is not a safety proof -- it means none of the known
    RCE primitives (dunder access, ``|attr``, dangerous Jinja globals,
    unbounded ``range()``) were found; the sandboxed render, CPU watchdog,
    and memory cap remain the backstops. Renders nothing when the check did
    not run.
    """
    check: Any = result.get("template_firewall")
    if type(check) is not dict:
        return ""
    status = check.get("status")
    if status == "allowed":
        badge = '<span class="badge pass">allowed</span>'
        note = (
            "No RCE primitives found in the checkpoint's chat template "
            "(no dunder access, no |attr filter, no dangerous Jinja globals, "
            "no unbounded range()). The template was still rendered inside "
            "the sandboxed environment with the CPU watchdog and memory cap."
        )
    elif status == "denied":
        badge = '<span class="badge fail">denied</span>'
        note = (
            "The checkpoint's chat template was DENIED by the static firewall "
            "and was never rendered: template conformance did not execute. "
            "Inspect the findings below before trusting this checkpoint."
        )
    elif status == "no_template":
        badge = '<span class="badge">n/a</span>'
        note = "The checkpoint declares no chat template; nothing was scanned."
    else:
        return ""
    findings = check.get("findings")
    rows = []
    if type(findings) is list:
        for finding in findings:
            if type(finding) is not dict:
                continue
            rows.append(
                "<tr><td>"
                + escape(str(finding.get("rule", "?")))
                + "</td><td>"
                + escape(str(finding.get("detail", "?")))
                + "</td></tr>"
            )
    parts = [
        '<section id="template-firewall"><h2>Template firewall</h2>',
        "<p>" + badge + " " + escape(note) + "</p>",
    ]
    if rows:
        parts.append(
            '<div class="table-wrap"><table>'
            "<caption>Firewall findings.</caption>"
            '<thead><tr><th scope="col">Rule</th>'
            '<th scope="col">Detail</th></tr></thead><tbody>'
            + "".join(rows)
            + "</tbody></table></div>"
        )
    parts.append("</section>")
    return "".join(parts)


def render_html(result: dict[str, Any]) -> str:
    """Render an experiment result dict as a self-contained HTML report string.

    All dynamic content is HTML-escaped (injection-safe); the output embeds
    its own styling and needs no external assets. Expects the enriched result
    shape produced by attach_diagnostics.
    """

    decision = str(result.get("decision", ""))
    dclass = _DECISION_CLASS.get(decision, "neutral")
    dhelp = _DECISION_HELP.get(decision, "Unrecognized decision value.")
    taxonomy_note = _taxonomy_note(result)
    diagnosis = _failure_diagnosis(result)
    margin_line = _selection_margin_line(result)
    banner_help = (
        escape(dhelp)
        + ('</p><p class="decision-help">' + escape(taxonomy_note) if taxonomy_note else "")
        + ('</p><p class="decision-help">' + escape(diagnosis) if diagnosis else "")
        + ('</p><p class="decision-help">' + margin_line if margin_line else "")
    )
    config_html = _config_section(result)
    search_html = (
        '<section id="search-candidates"><h2>Search-stage candidates</h2>'
        '<div class="table-wrap">'
        '<table class="sortable">'
        "<caption>Candidate summary: exact answer plus explicit boundary pass "
        "rate across search fixtures.</caption>"
        '<thead><tr><th scope="col">Candidate</th>'
        '<th scope="col">Pass rate</th><th scope="col">Status</th>'
        '<th scope="col">Attempts</th><th scope="col">Boundary IDs</th>'
        '<th scope="col">Details</th>'
        "</tr></thead><tbody>" + _search_rows(result) + "</tbody></table></div></section>"
    )
    outcome_html = _outcome_grid_section(result)
    decisive_html = _decisive_section(result)
    overhead_html = _token_overhead_section(result)
    conformance_html = _conformance_section(result)
    stop_matrix_html = _stop_matrix_section(result)
    selection_robustness_html = _selection_robustness_section(result)
    divergence_html = _backend_divergence_section(result)
    frontier_html = _frontier_section(result)
    boundary_audit_html = _boundary_audit_section(result)
    marker_isolation_html = _marker_isolation_section(result)
    template_firewall_html = _template_firewall_section(result)
    transfer_html = _transfer_section(result)
    perturbation_html = _perturbation_section(result)
    sweep_html = _sweep_section(result)
    logprob_html = _logprob_section(result)
    held_out_html = _held_out_section(result)
    toc = _toc(
        [
            ("run-configuration", "Run configuration", config_html),
            ("search-candidates", "Search-stage candidates", search_html),
            ("per-fixture-outcomes", "Per-fixture outcomes", outcome_html),
            ("decisive-fixtures", "Decisive fixtures", decisive_html),
            ("token-overhead", "Token overhead", overhead_html),
            ("template-conformance", "Template conformance", conformance_html),
            ("stop-behavior-matrix", "Stop-behavior isolation matrix", stop_matrix_html),
            (
                "selection-robustness",
                "Selection robustness under stop semantics",
                selection_robustness_html,
            ),
            ("backend-divergence", "Backend divergence", divergence_html),
            ("token-budget-frontier", "Token-budget frontier", frontier_html),
            ("boundary-canonicality", "Prompt-boundary canonicality", boundary_audit_html),
            ("marker-isolation", "Chat-marker isolation", marker_isolation_html),
            ("template-firewall", "Template firewall", template_firewall_html),
            ("transfer", "Cross-checkpoint transfer", transfer_html),
            (
                "perturbation-robustness",
                "Fixture-perturbation robustness",
                perturbation_html,
            ),
            ("sensitivity-sweep", "Prompt-boundary sensitivity sweep", sweep_html),
            ("logprob-diagnostic", "Log-likelihood diagnostic", logprob_html),
            ("held-out-stage", "Held-out stage", held_out_html),
        ]
    )
    return (
        "<!doctype html>"
        '<html lang="en">'
        "<head>"
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="' + _CSP + '">'
        "<title>Model Protocol Lab report</title>"
        "<style>" + _CSS + "</style>"
        "</head>"
        "<body><main>"
        "<header><h1>Model Protocol Lab</h1>"
        '<p class="subtitle">Exact-answer and explicit-boundary measurements '
        "on the supplied fixtures.</p></header>"
        + toc
        + '<section class="decision '
        + dclass
        + '" aria-label="Run decision">'
        '<div class="decision-label">Decision</div>'
        '<div class="decision-value">' + escape(decision) + "</div>"
        '<p class="decision-help">'
        + banner_help
        + "</p></section>"
        + config_html
        + search_html
        + outcome_html
        + decisive_html
        + overhead_html
        + conformance_html
        + stop_matrix_html
        + selection_robustness_html
        + divergence_html
        + frontier_html
        + boundary_audit_html
        + marker_isolation_html
        + template_firewall_html
        + transfer_html
        + perturbation_html
        + sweep_html
        + logprob_html
        + held_out_html
        + "<footer>"
        "<p>This report provides no confidence interval, universal "
        "compatibility guarantee, or equal-compute claim. Token counts are "
        "retained input/output counts, not equal-compute accounting.</p>"
        "<p>Raw prompts, answers and token traces are excluded from this HTML "
        "view.</p>"
        "</footer></main>"
        "<script>" + _SCRIPT + "</script>"
        "</body></html>"
    )


def prepare_report_dir(directory: str | Path, *, _allow_empty_existing: bool = False) -> Path:
    """Create a fresh report directory with owner-only permissions.

    Creates ``directory`` (and any missing parents) with mode 0o700 so the
    report is readable only by the invoking user regardless of the process
    umask. Raises FileExistsError if the leaf already exists, unless
    ``_allow_empty_existing`` is set and the leaf is an empty directory
    (used by ``cli.run``, which claims the output dir before the expensive
    experiment). A symlinked leaf always raises FileExistsError -- even with
    ``_allow_empty_existing`` -- so report files are never written through a
    link into an attacker-chosen directory. Symlinked path components are
    never chmodded through: the
    permission tightening applies only to real directories created (or
    claimed) for this report. Callers that create the directory ahead of expensive work
    should remove it again if the work fails before ``save_report``
    completes, to preserve the fresh-directory guarantee for retries.
    """
    target = Path(directory)
    # Collect the chain we are about to create so every component — not just
    # the leaf — gets owner-only permissions (mkdir's mode= applies to the
    # leaf only; parents would otherwise inherit 0o777 & ~umask).
    missing: list[Path] = []
    probe = target
    while not probe.exists():
        missing.append(probe)
        probe = probe.parent
    try:
        target.mkdir(parents=True, exist_ok=False, mode=0o700)
    except FileExistsError:
        # A symlinked leaf is never claimed: is_dir() follows links, so a
        # symlink to an empty directory would otherwise pass the freshness
        # check below and save_report would write through it into an
        # attacker-chosen directory. Fail closed regardless of
        # _allow_empty_existing.
        if target.is_symlink():
            raise
        if not (_allow_empty_existing and target.is_dir() and not any(target.iterdir())):
            raise
    for component in missing:
        # Never chmod through a symlink: os.chmod follows links, which would
        # change permissions on a directory outside the requested tree (and
        # the is_symlink re-check after mkdir also covers a swap race).
        if not component.is_symlink():
            os.chmod(component, 0o700)
    if not target.is_symlink():
        os.chmod(target, 0o700)
    return target


def save_report(
    result: dict[str, Any],
    directory: str | Path,
    *,
    include_evidence: bool = False,
    _allow_empty_existing: bool = False,
) -> tuple[Path, Path]:
    """Write the report as results.json + report.html into a fresh directory.

    The directory is created with owner-only permissions and both files are
    written with O_EXCL (no overwrite) and 0o600, since reports may contain
    sensitive evidence; a partially written file is unlinked on any
    Python-level failure. The JSON holds the public view (raw
    observations stripped unless include_evidence is set); the HTML is the
    full enriched rendering. Returns the (json_path, html_path) tuple.
    Raises ValueError if the directory is not fresh and empty, and
    FileExistsError if the leaf is (or resolves through) a symlink -- report
    files are never written through a symlinked leaf.
    """

    # Automatic diagnostics: pure functions of the recorded search
    # outcomes, attached to a copy so the caller's dict is untouched. The
    # JSON keeps them (they are safe aggregates) while _public_view strips
    # the raw observations they were computed from.
    enriched = attach_diagnostics(result)
    public = _public_view(enriched, include_evidence)
    json_text = json.dumps(public, ensure_ascii=True, allow_nan=False, indent=2)
    html_text = render_html(enriched)
    # Restrictive permissions: reports may contain sensitive evidence
    # (--include-evidence). The directory and files are readable only by the
    # invoking user, regardless of the process umask.
    target = prepare_report_dir(directory, _allow_empty_existing=_allow_empty_existing)
    json_path, html_path = target / "results.json", target / "report.html"
    written: list[Path] = []
    try:
        for path, text in ((json_path, json_text), (html_path, html_text)):
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                handle = os.fdopen(fd, "w", encoding="utf-8")
            except BaseException:
                os.close(fd)
                with contextlib.suppress(OSError):
                    os.unlink(path)
                raise
            try:
                with handle:
                    handle.write(text)
            except BaseException:
                # Never leave a half-written report behind: an interrupted
                # (KeyboardInterrupt) or failed write removes the partial
                # file, mirroring fixtures.write_fixture_file's discipline.
                with contextlib.suppress(OSError):
                    os.unlink(path)
                raise
            written.append(path)
    except BaseException:
        # The second file failed after the first was fully written: remove
        # it too, so a failed save never leaves a half-built report pair.
        for done in written:
            with contextlib.suppress(OSError):
                os.unlink(done)
        raise
    return json_path, html_path
