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


def _search_rows(result: dict[str, Any]) -> str:
    rows = []
    for run in result.get("search") or []:
        if type(run) is not dict:
            continue
        summary = run.get("summary") or {}
        format_info = run.get("format") or {}
        name = format_info.get("name", "?")
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
        rows.append('<tr><td colspan="5">No candidates were evaluated in this run.</td></tr>')
    return "".join(rows)


def _config_section(result: dict[str, Any]) -> str:
    boundary = ", ".join(str(t) for t in result.get("boundary_token_ids", []))
    fingerprint = _short_fingerprint(result.get("backend_fingerprint"))
    items = (
        ("Selection rule", str(result.get("selection_rule", ""))),
        ("Measurement", str(result.get("measurement", ""))),
        ("Seed", str(result.get("seed", ""))),
        ("Max new tokens", str(result.get("max_new_tokens", ""))),
        ("Boundary token IDs", boundary),
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


def render_html(result: dict[str, Any]) -> str:
    decision = str(result.get("decision", ""))
    dclass = _DECISION_CLASS.get(decision, "neutral")
    dhelp = _DECISION_HELP.get(decision, "Unrecognized decision value.")
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
        + escape(dhelp)
        + "</p></section>"
        + _config_section(result)
        + '<section><h2>Search-stage candidates</h2><div class="table-wrap">'
        "<table>"
        "<caption>Candidate summary: exact answer plus explicit boundary pass "
        "rate across search fixtures.</caption>"
        '<thead><tr><th scope="col">Candidate</th>'
        '<th scope="col">Pass rate</th><th scope="col">Status</th>'
        '<th scope="col">Attempts</th><th scope="col">Details</th>'
        "</tr></thead><tbody>"
        + _search_rows(result)
        + "</tbody></table></div></section>"
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
    experiment). Callers that create the directory ahead of expensive work
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
        os.chmod(component, 0o700)
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
