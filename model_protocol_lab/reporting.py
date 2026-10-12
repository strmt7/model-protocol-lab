"""Local JSON and escaped HTML reports with no external assets."""
from html import escape
import json
from pathlib import Path


def _public_view(result: dict, include_evidence: bool) -> dict:
    clean = json.loads(json.dumps(result, ensure_ascii=True, allow_nan=False))
    if not include_evidence:
        for run in clean.get("search", []):
            run.pop("observations", None)
        if clean.get("held_out") is not None:
            clean["held_out"].pop("observations", None)
    clean["raw_evidence_included"] = include_evidence
    return clean


def render_html(result: dict) -> str:
    rows = []
    for run in result["search"]:
        summary = run["summary"]
        rate = summary["rate"]
        cell = "incomplete" if rate is None else f"{rate:.0%}"
        color = "neutral" if rate is None else ("pass" if rate == 1 else "fail")
        rows.append("<tr><th>" + escape(run["format"]["name"]) + "</th>"
                    + '<td class="' + color + '">' + escape(cell) + "</td><td>"
                    + escape(summary["status"]) + "</td><td>"
                    + escape(str(summary["attempted"])) + "/" + escape(str(summary["planned"])) + "</td></tr>")
    held = result.get("held_out")
    held_text = "Not evaluated" if held is None else json.dumps(held["summary"], indent=2)
    return """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Model Protocol Lab report</title><style>
body{font:16px system-ui;max-width:1050px;margin:3rem auto;padding:0 1rem;background:#111827;color:#f8fafc}
table{border-collapse:collapse;width:100%}th,td{padding:1rem;border:1px solid #64748b;text-align:left}
.pass{background:#14532d}.fail{background:#7f1d1d}.neutral{background:#374151}pre{white-space:pre-wrap}
</style><h1>Model Protocol Lab</h1><p>Exact-answer and explicit-boundary measurements on the supplied fixtures.</p>
<p>Decision: <strong>""" + escape(result["decision"]) + """</strong></p>
<table><caption>Search-stage candidate heatmap</caption><thead><tr><th>Candidate</th><th>Pass rate</th><th>Status</th><th>Attempts</th></tr></thead><tbody>""" + "".join(rows) + """</tbody></table>
<h2>Held-out stage</h2><pre>""" + escape(held_text) + """</pre>
<p>This report provides no confidence interval, universal compatibility guarantee, or equal-compute claim.
Raw prompts, answers and token traces are excluded from this HTML view.</p></html>"""


def save_report(result: dict, directory: str | Path, *, include_evidence=False) -> tuple[Path, Path]:
    target = Path(directory)
    public = _public_view(result, include_evidence)
    json_text = json.dumps(public, ensure_ascii=True, allow_nan=False, indent=2)
    html_text = render_html(result)
    target.mkdir(parents=True, exist_ok=False)
    json_path, html_path = target / "results.json", target / "report.html"
    with json_path.open("x", encoding="utf-8") as handle:
        handle.write(json_text)
    with html_path.open("x", encoding="utf-8") as handle:
        handle.write(html_text)
    return json_path, html_path
