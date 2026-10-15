"""Local JSON and escaped HTML reports with no external assets."""

import base64
import hashlib
import json
import os
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
    clean["raw_evidence_included"] = include_evidence
    return clean


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
                        detail_items.append(
                            "hand "
                            + escape(str(segment.get("hand_span")))
                            + " "
                            + escape(str(segment.get("hand_ids")))
                            + " \u2194 template "
                            + escape(str(segment.get("template_span")))
                            + " "
                            + escape(str(segment.get("template_ids")))
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
        "messages. A divergent segment lists the token-ID subsequences that "
        "differ and their positions; an identical rendering reports an empty "
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
    Every model/fixture-derived string (fixture names, stop texts,
    detail lines) is escaped; severities, classes and counts are the
    module's fixed vocabulary. Severity badges reuse the existing badge
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
    # descending), fixed column order, sortable like the other tables.
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

    # One <details> callout per divergence: fixture, class with severity,
    # the stop text, and the recorded detail (which names the semantics
    # and firing steps that disagreed, plus the kept-vs-stripped text
    # contrast).
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
            finding_stop_text = finding.get("stop_text")
            finding_detail = finding.get("detail")
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
                + "</summary><p>Stop text: "
                + '<span class="mono">'
                + escape("\u2014" if finding_stop_text is None else str(finding_stop_text))
                + "</span></p><p>"
                + escape(str(finding_detail))
                + "</p></details>"
            )

    stop_texts = section.get("stop_texts")
    params = ""
    if type(stop_texts) is list and stop_texts:
        params = (
            "<p>Stop text(s): "
            + ", ".join(
                '<span class="mono">' + escape(str(stop)) + "</span>" for stop in stop_texts
            )
            + ".</p>"
        )
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
        rate = summary.get("rate")
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
    held = result.get("held_out")
    if type(held) is not dict:
        return (
            '<section id="held-out-stage"><h2>Held-out stage</h2>'
            "<p>Not evaluated &mdash; no candidate was promoted from search.</p>"
            "</section>"
        )
    summary = held.get("summary") or {}
    rate = summary.get("rate")
    rate_text = "incomplete" if rate is None else f"{rate:.0%}"
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
        '<div class="cards">' + cards + "</div>"
        "<p>Held-out observations are excluded from this HTML view; "
        "see results.json with --include-evidence for the raw records.</p>"
        "</section>"
    )


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
        fixture_ids = data.get("fixture_ids") or []
        total = len(fixture_ids)
        mins = data.get("min_budget_per_fixture") or []
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
            items.append("<li>" + label + "</li>")
        parts.append(
            '<p><span class="badge fail">budget-sensitive</span> The selection '
            "decision would have differed at these token budgets:</p><ul>"
            + "".join(items)
            + "</ul>"
        )
    winner: Any = frontier.get("native_winner")
    candidates: Any = frontier.get("candidates")
    data = candidates.get(winner) if type(winner) is str else None
    if type(data) is dict:
        total = len(data.get("fixture_ids") or [])
        rows = []
        for point in data.get("pass_steps") or []:
            if type(point) is not list or len(point) != 2:
                continue
            budget, count = point
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
    parts.append(_frontier_spark_table(frontier))
    parts.append("</section>")
    return "".join(parts)


def render_html(result: dict[str, Any]) -> str:
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
    divergence_html = _backend_divergence_section(result)
    frontier_html = _frontier_section(result)
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
            ("backend-divergence", "Backend divergence", divergence_html),
            ("token-budget-frontier", "Token-budget frontier", frontier_html),
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
        + divergence_html
        + frontier_html
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


def prepare_report_dir(directory: str | Path, *, _allow_empty_existing=False) -> Path:
    """Create a fresh report directory with owner-only permissions.

    Creates ``directory`` (and any missing parents) with mode 0o700 so the
    report is readable only by the invoking user regardless of the process
    umask. Raises FileExistsError if the leaf already exists, unless
    ``_allow_empty_existing`` is set and the leaf is an empty directory
    (used by ``cli.run``, which claims the output dir before the expensive
    experiment). Symlinked path components are never chmodded through: the
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
    result: dict,
    directory: str | Path,
    *,
    include_evidence=False,
    _allow_empty_existing=False,
) -> tuple[Path, Path]:
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
    for path, text in ((json_path, json_text), (html_path, html_text)):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            handle = os.fdopen(fd, "w", encoding="utf-8")
        except BaseException:
            os.close(fd)
            raise
        with handle:
            handle.write(text)
    return json_path, html_path
