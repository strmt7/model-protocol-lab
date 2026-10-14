"""Local JSON and escaped HTML reports with no external assets."""

import json
import os
from html import escape
from pathlib import Path
from typing import Any

_CSP = "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"

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
    "@media print{:root{color-scheme:light}"
    "body{background:#fff;color:#111;max-width:none}"
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
            else:
                segments = entry.get("diff") or []
                if not segments:
                    cell = '<span class="badge pass">identical</span>'
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
                        + '<details><summary class="mono">segments</summary><p class="mono">'
                        + "<br>".join(detail_items)
                        + "</p></details>"
                    )
            rows.append(
                '<tr><th scope="row">'
                + escape(str(entry.get("format", "?")))
                + "</th><td>"
                + escape(str(entry.get("fixture", "?")))
                + "</td><td>"
                + escape(str(entry.get("hand_tokens", "?")))
                + "</td><td>"
                + escape(str(entry.get("template_tokens", "?")))
                + "</td><td>"
                + cell
                + "</td></tr>"
            )
        if not rows:
            body = "<p>No format/fixture pairs were checked.</p>"
        else:
            body = (
                '<div class="table-wrap"><table>'
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
        "<section><h2>Template conformance</h2>" + body + "<p>Each hand-rendered "
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
        "<section><h2>Stop-behavior isolation matrix</h2>"
        "<p>The frozen winner's recorded outputs, re-measured under emulated "
        "competitor stop semantics. Each cell recomputes the lab's pass rule "
        "(exact answer and token-ID boundary) on the output as truncated by "
        "that semantic; a flip means the semantic would have changed the "
        "verdict. Emulations are post-hoc truncations of the greedy "
        "N-token outputs.</p>" + sections + '<dl class="config">' + legend + "</dl></section>"
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
        rows.append(
            '<tr><th scope="row">'
            + escape(str(depth.get("drop_bytes", "?")))
            + "</th><td>"
            + escape(str(depth.get("decision", "?")))
            + "</td><td>"
            + escape(str(depth.get("selected_candidate", "?")))
            + "</td><td>"
            + (
                '<span class="badge fail">changed</span>'
                if changed
                else '<span class="badge pass">stable</span>'
            )
            + "</td></tr>"
        )
    table = (
        '<div class="table-wrap"><table>'
        "<caption>Search re-run with the prompt truncated mid-token.</caption>"
        '<thead><tr><th scope="col">Dropped bytes</th><th scope="col">Decision</th>'
        '<th scope="col">Selected</th><th scope="col">Vs native</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
        if rows
        else "<p>No depths were evaluated.</p>"
    )
    return (
        "<section><h2>Prompt-boundary sensitivity sweep</h2>"
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
                cells.append('<td><span class="glyph-ok" title="pass">\u2713</span></td>')
            elif outcome is False:
                cells.append('<td><span class="glyph-bad" title="fail">\u2717</span></td>')
            else:
                cells.append('<td><span class="glyph-na" title="incomplete">\u2013</span></td>')
        rows.append('<tr><th scope="row">' + escape(fid) + "</th>" + "".join(cells) + "</tr>")
    return (
        "<section><h2>Per-fixture outcomes</h2>"
        '<div class="table-wrap"><table>'
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
    return "<section><h2>Decisive fixtures</h2>" + body + "</section>"


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
        rows.append(
            '<tr><th scope="row">'
            + escape(str(name))
            + "</th><td>"
            + bar
            + cell
            + "</td><td>"
            + escape(str(summary.get("status", "?")))
            + "</td><td>"
            + escape(f"{summary.get('attempted', '?')}/{summary.get('planned', '?')}")
            + '</td><td class="mono">'
            + escape(boundary_text)
            + "</td><td>"
            + '<details><summary class="mono">'
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
        "<section><h2>Run configuration</h2>"
        '<dl class="config">'
        + terms
        + "</dl>"
        + '<details><summary>Full backend fingerprint</summary><p class="mono">'
        + escape(str(result.get("backend_fingerprint", "")))
        + "</p></details></section>"
    )


def _held_out_section(result: dict[str, Any]) -> str:
    held = result.get("held_out")
    if type(held) is not dict:
        return (
            "<section><h2>Held-out stage</h2>"
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
        "<section><h2>Held-out stage</h2>"
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
        spark = _sparkline(counts, total)
        full = data.get("full_pass_budget")
        rows.append(
            '<tr><th scope="row">'
            + escape(name)
            + "</th><td>"
            + '<span class="spark" role="img" aria-label="'
            + escape(
                f"pass count {counts[0]} to {counts[-1]} of {total} over budgets 1 to {max_tokens}"
            )
            + '">'
            + spark
            + "</span></td><td>"
            + escape(str(full) if full is not None else "\u2014")
            + "</td></tr>"
        )
    if not rows:
        return ""
    return (
        "<h3>Pass curves by candidate</h3>"
        '<div class="table-wrap"><table>'
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
        "<section><h2>Token-budget frontier</h2>",
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
    banner_help = (
        escape(dhelp)
        + ('</p><p class="decision-help">' + escape(taxonomy_note) if taxonomy_note else "")
        + ('</p><p class="decision-help">' + escape(diagnosis) if diagnosis else "")
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
        '<section class="decision ' + dclass + '" aria-label="Run decision">'
        '<div class="decision-label">Decision</div>'
        '<div class="decision-value">' + escape(decision) + "</div>"
        '<p class="decision-help">'
        + banner_help
        + "</p></section>"
        + _config_section(result)
        + '<section><h2>Search-stage candidates</h2><div class="table-wrap">'
        "<table>"
        "<caption>Candidate summary: exact answer plus explicit boundary pass "
        "rate across search fixtures.</caption>"
        '<thead><tr><th scope="col">Candidate</th>'
        '<th scope="col">Pass rate</th><th scope="col">Status</th>'
        '<th scope="col">Attempts</th><th scope="col">Boundary IDs</th>'
        '<th scope="col">Details</th>'
        "</tr></thead><tbody>"
        + _search_rows(result)
        + "</tbody></table></div></section>"
        + _outcome_grid_section(result)
        + _decisive_section(result)
        + _conformance_section(result)
        + _stop_matrix_section(result)
        + _frontier_section(result)
        + _sweep_section(result)
        + _held_out_section(result)
        + "<footer>"
        "<p>This report provides no confidence interval, universal "
        "compatibility guarantee, or equal-compute claim. Token counts are "
        "retained input/output counts, not equal-compute accounting.</p>"
        "<p>Raw prompts, answers and token traces are excluded from this HTML "
        "view.</p>"
        "</footer></main>"
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
    public = _public_view(result, include_evidence)
    json_text = json.dumps(public, ensure_ascii=True, allow_nan=False, indent=2)
    html_text = render_html(result)
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
