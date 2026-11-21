"""reporting probes family: split from reporting.py (W7-2).

Facades keep the public surface: import from model_protocol_lab.reporting, not from this module.
"""

from html import escape
from typing import Any

from .reporting_core import (
    _escape_text,
    _float_convertible,
)


def _hierarchy_section(result: dict[str, Any]) -> str:
    """Instruction-hierarchy adherence: which instruction wins conflicts?

    Every search candidate re-evaluated on synthetic system-vs-user
    conflict fixtures (the system instruction is delivered as a
    privilege-marked block folded into the user turn, before and after
    the user instruction); the selection rule re-applied per position.
    Reports per-candidate system/user/neither adherence rates, the
    winner-vs-clean flips, the primacy-vs-recency position effect, and
    the documentation-derived dedicated-frame vs folded family
    comparison. Renders nothing when the hierarchy probe did not run.
    """
    hierarchy: Any = result.get("hierarchy")
    if type(hierarchy) is not dict or not hierarchy.get("available"):
        return ""
    per_condition = hierarchy.get("per_condition")
    if type(per_condition) is not dict:
        return ""
    primary_winner = hierarchy.get("primary_winner")
    primary_decision = hierarchy.get("primary_decision")
    n_fixtures = hierarchy.get("n_fixtures")

    def _name(value: Any) -> str:
        """Return the name."""
        return "(none)" if value is None else str(value)

    def _rate(value: Any) -> str:
        """Return the rate."""
        if type(value) in (int, float):
            try:
                return f"{value:.0%}"
            except OverflowError:
                return "n/a"
        return "n/a"

    verdict_lines = []
    for key in sorted(per_condition, key=str):
        position = str(key)
        condition = per_condition[key]
        if type(condition) is not dict:
            continue
        decision = condition.get("decision")
        winner = condition.get("winner")
        line = (
            escape(str(position))
            + ": <strong>"
            + escape(_name(decision))
            + " / "
            + escape(_name(winner))
            + "</strong> vs primary "
            + escape(_name(primary_decision))
            + " / "
            + escape(_name(primary_winner))
        )
        if (decision, winner) != (primary_decision, primary_winner):
            line += ' <span class="badge fail">differs</span>'
        else:
            line += ' <span class="badge pass">same</span>'
        verdict_lines.append("<p>" + line + "</p>")

    cand_rows = []
    for key in sorted(per_condition, key=str):
        position = str(key)
        condition = per_condition[key]
        if type(condition) is not dict:
            continue
        candidates = condition.get("per_candidate")
        if type(candidates) is not dict:
            continue
        for ckey in sorted(candidates, key=str):
            cand = str(ckey)
            entry = candidates[ckey]
            if type(entry) is not dict:
                continue
            passed = entry.get("pass")
            badge = (
                '<span class="badge pass">pass</span>'
                if passed is True
                else '<span class="badge fail">fail</span>'
                if passed is False
                else '<span class="badge">n/a</span>'
            )
            cand_rows.append(
                '<tr><td><span class="mono">'
                + escape(cand)
                + '</span></td><td><span class="mono">'
                + escape(str(position))
                + "</span></td><td>"
                + escape(_rate(entry.get("system_rate")))
                + "</td><td>"
                + escape(_rate(entry.get("user_rate")))
                + "</td><td>"
                + escape(_rate(entry.get("neither_rate")))
                + "</td><td>"
                + badge
                + "</td></tr>"
            )

    flips = hierarchy.get("flips")
    flip_lines = []
    if type(flips) is list:
        for entry in flips:
            if type(entry) is not dict:
                continue
            flip_lines.append(
                '<li><span class="mono">'
                + escape(str(entry.get("position")))
                + "</span>: "
                + escape(_name(entry.get("decision_from")))
                + " / "
                + escape(_name(entry.get("winner_from")))
                + " &rarr; "
                + escape(_name(entry.get("decision_to")))
                + " / "
                + escape(_name(entry.get("winner_to")))
                + "</li>"
            )

    position_effect = hierarchy.get("position_effect")
    effect_line = ""
    if type(position_effect) is dict:
        delta = position_effect.get("mean_system_rate_delta")
        delta_text = _rate(delta) if delta is not None else "n/a"
        changed = position_effect.get("winner_changed") is True
        effect_line = (
            "<p>Primacy vs recency: mean system-adherence delta "
            "(privilege-first minus privilege-last) <strong>"
            + escape(delta_text)
            + "</strong>; the per-position winner "
            + ("<strong>differs</strong>." if changed else "is the same.")
            + "</p>"
        )

    group_adherence = hierarchy.get("group_adherence")
    group_rows = []

    def _as_rate(value: Any) -> float | None:
        # Strict: bools are excluded (type() check, not isinstance()).
        """Return the as rate."""
        if type(value) in (int, float):
            try:
                return float(value)
            except OverflowError:
                return None
        return None

    ded_rate: float | None = None
    fold_rate: float | None = None
    if type(group_adherence) is dict:
        for label in ("dedicated", "folded", "unknown"):
            group = group_adherence.get(label)
            if type(group) is not dict:
                continue
            rate = _as_rate(group.get("mean_system_rate"))
            if label == "dedicated":
                ded_rate = rate
            elif label == "folded":
                fold_rate = rate
            group_rows.append(
                "<tr><td>"
                + escape(str(label))
                + "</td><td>"
                + escape(str(group.get("n_candidates")))
                + "</td><td>"
                + escape(str(group.get("n_families")))
                + "</td><td>"
                + escape(_rate(group.get("mean_system_rate")))
                + "</td></tr>"
            )
    group_note = ""
    if ded_rate is not None and fold_rate is not None:
        relation = (
            "higher than"
            if ded_rate > fold_rate
            else "lower than"
            if ded_rate < fold_rate
            else "equal to"
        )
        group_note = (
            "<p>Dedicated-frame families average "
            + escape(_rate(ded_rate))
            + " system adherence vs "
            + escape(_rate(fold_rate))
            + " for folded families ("
            + relation
            + " -- a correlation readout over "
            "catalog-documented labels, not a causal claim).</p>"
        )

    n_text = f" ({n_fixtures} conflict fixtures per position)" if type(n_fixtures) is int else ""
    return (
        '<section class="card" id="instruction-hierarchy"><h2>Instruction-hierarchy adherence</h2>'
        "<p>Every search candidate re-evaluated on synthetic system-vs-user "
        "conflict fixtures: the system instruction is delivered as a "
        "privilege-marked block folded into the user turn, either before "
        "(primacy) or after (recency) the conflicting user instruction; the "
        "selection rule is re-applied per position. System adherence is the "
        "fraction of fixtures answered with the system instruction's exact "
        "reply; user adherence the fraction answered with the user's."
        + escape(n_text)
        + "</p>"
        + "".join(verdict_lines)
        + effect_line
        + (
            "<ul>" + "".join(flip_lines) + "</ul>"
            if flip_lines
            else "<p>No winner flips against the primary run.</p>"
        )
        + (
            '<div class="table-wrap"><table><caption>Per-candidate adherence '
            "by position.</caption>"
            '<thead><tr><th scope="col">Candidate</th>'
            '<th scope="col">Position</th><th scope="col">System</th>'
            '<th scope="col">User</th><th scope="col">Neither</th>'
            '<th scope="col">Pass</th>'
            "</tr></thead><tbody>" + "".join(cand_rows) + "</tbody></table></div>"
            if cand_rows
            else ""
        )
        + (
            '<div class="table-wrap"><table><caption>System adherence by '
            "catalog-documented system treatment (correlation only).</caption>"
            '<thead><tr><th scope="col">Treatment</th>'
            '<th scope="col">Candidates</th><th scope="col">Families</th>'
            '<th scope="col">Mean system rate</th>'
            "</tr></thead><tbody>" + "".join(group_rows) + "</tbody></table></div>" + group_note
            if group_rows
            else ""
        )
        + "</section>"
    )


def _structured_section(result: dict[str, Any]) -> str:
    """Structured-output adherence: prompt-only JSON parse/schema rates.

    Renders the ``structured`` payload (see experiments.structured_adherence)
    as a per-candidate parse-valid / schema-valid / adherent rate table plus
    the adherence gap between the strongest and weakest templates. Every
    interpolation is escaped at this boundary.
    """
    audit: Any = result.get("structured")
    if type(audit) is not dict or not audit.get("available"):
        return ""
    per_candidate = audit.get("per_candidate")
    if type(per_candidate) is not dict:
        return ""

    def _rate(value: Any) -> str:
        """Return the rate."""
        if type(value) in (int, float):
            try:
                return f"{value:.0%}"
            except OverflowError:
                return "n/a"
        return "n/a"

    rows = []
    for key in sorted(per_candidate, key=str):
        cand = str(key)
        entry = per_candidate[key]
        if type(entry) is not dict:
            continue
        rows.append(
            '<tr><td><span class="mono">'
            + escape(cand)
            + "</span></td><td>"
            + escape(_rate(entry.get("parse_rate")))
            + "</td><td>"
            + escape(_rate(entry.get("schema_rate")))
            + "</td><td>"
            + escape(_rate(entry.get("adherent_rate")))
            + "</td><td>"
            + escape(str(entry.get("n_evaluable")))
            + "</td></tr>"
        )
    weakest = audit.get("weakest_candidates")
    weakest_line = ""
    if type(weakest) is list and weakest:
        weakest_line = (
            "<p>Weakest structured-output candidates: "
            + ", ".join(
                '<span class="mono">' + escape(str(name)) + "</span>"
                for name in weakest
                if type(name) is str
            )
            + ".</p>"
        )
    gap = audit.get("adherence_gap")
    gap_line = ""
    if type(gap) in (int, float) and _float_convertible(gap):
        gap_line = (
            "<p>Adherence gap (strongest minus weakest template): "
            + escape(f"{gap:.0%}")
            + ". A positive gap falsifies uniform adherence across templates.</p>"
        )
    uniform = audit.get("uniform")
    uniform_line = ""
    if uniform is True:
        uniform_line = "<p>Adherence was uniform across all evaluable candidates.</p>"
    return (
        '<section class="card" id="structured-adherence">'
        "<h2>Structured-output adherence</h2>"
        "<p>Prompt-only JSON adherence per template: each candidate's decoded "
        "answers on the synthetic JSON fixtures are parsed and validated "
        "against the fixture schemas with a deterministic local validator. "
        "No constrained decoder is used -- this is what the template alone "
        "elicits. <strong>Parse</strong> = fraction of responses that parsed "
        "as JSON; <strong>Schema</strong> = fraction satisfying the schema; "
        "<strong>Adherent</strong> = fraction both parsing and schema-valid.</p>"
        + weakest_line
        + gap_line
        + uniform_line
        + (
            '<div class="table-wrap"><table><caption>Adherence rates per candidate.</caption>'
            '<thead><tr><th scope="col">Candidate</th>'
            '<th scope="col">Parse</th><th scope="col">Schema</th>'
            '<th scope="col">Adherent</th><th scope="col">Evaluable</th>'
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            if rows
            else ""
        )
        + "</section>"
    )


def _faketurn_section(result: dict[str, Any]) -> str:
    """Fake-turn history tampering: forged prior turns, same vs cross delimiters.

    Renders the ``faketurn`` payload (see experiments.faketurn_robustness)
    as a per-candidate table of verdict-flip rates: forged prior turns
    (fake assistant asserting a wrong answer, fake system contradicting
    the task) injected into user content at the family's native
    encoding, with the forged delimiters wearing the family's own
    token set (same) vs another family's token set (cross). The
    pre-registered falsifiable claim is a transfer gap of ~0 uniform
    across families; a systematically positive gap means foreign
    delimiters carry more tampering authority -- arXiv:2405.20234's
    token-set-dependence claim, measured per family with the model
    fixed. "Authority" is measured strictly as verdict-flip rate on
    fixed fixtures, not open-ended harm. Every interpolation is
    escaped at this boundary.
    """
    audit: Any = result.get("faketurn")
    if type(audit) is not dict or not audit.get("available"):
        return ""
    per_candidate = audit.get("per_candidate")
    if type(per_candidate) is not dict:
        return ""

    def _rate(value: Any) -> str:
        """Return the rate."""
        if type(value) in (int, float):
            try:
                return f"{value:.2f}"
            except OverflowError:
                return "n/a"
        return "n/a"

    def _gap(value: Any) -> str:
        """Return the gap."""
        if type(value) in (int, float):
            try:
                return f"{value:+.2f}"
            except OverflowError:
                return "n/a"
        return "n/a"

    rows = []
    for key in sorted(per_candidate, key=str):
        cand = str(key)
        entry = per_candidate[key]
        if type(entry) is not dict:
            continue
        rows.append(
            '<tr><td><span class="mono">'
            + escape(cand)
            + "</span></td><td>"
            + escape(str(entry.get("n_evaluable_cells")))
            + " / "
            + escape(str(entry.get("n_cells")))
            + "</td><td>"
            + escape(_rate(entry.get("mean_flip_rate")))
            + "</td><td>"
            + escape(_rate(entry.get("same_flip_rate")))
            + "</td><td>"
            + escape(_rate(entry.get("cross_flip_rate")))
            + "</td><td>"
            + escape(_gap(entry.get("transfer_gap")))
            + "</td></tr>"
        )
    excluded = audit.get("excluded_candidates")
    excluded_line = ""
    if type(excluded) is list and excluded:
        excluded_line = (
            "<p>Candidates without an evaluable cell (provenance "
            "undeterminable, payload not intact, or no native verdicts): "
            + ", ".join(
                '<span class="mono">' + escape(str(name)) + "</span>"
                for name in excluded
                if type(name) is str
            )
            + ".</p>"
        )
    top_gap = audit.get("max_transfer_gap")
    top_line = ""
    if type(top_gap) in (int, float) and _float_convertible(top_gap):
        top_line = (
            "<p>Largest cross-minus-same flip-rate gap over all evaluable "
            "candidates: <strong>"
            + escape(f"{top_gap:+.2f}")
            + "</strong>. A positive gap means forged turns wearing "
            "another family's delimiters flipped task verdicts more "
            "often than forged turns wearing the family's own "
            "delimiters.</p>"
        )
    top_rate = audit.get("max_flip_rate")
    rate_line = ""
    if type(top_rate) in (int, float) and _float_convertible(top_rate):
        rate_line = (
            "<p>Largest per-cell verdict-flip rate over all evaluable "
            "candidates: <strong>" + escape(f"{top_rate:.2f}") + "</strong>.</p>"
        )
    return (
        '<section class="card" id="history-tampering">'
        "<h2>Fake-turn history tampering</h2>"
        "<p>A forged prior turn -- a fake assistant message asserting a "
        "wrong answer, or a fake system instruction contradicting the "
        "task -- was injected into user content and evaluated at each "
        "family's native encoding. The forged delimiters either belong "
        "to the family's own token set (<em>same</em>: the marker is "
        "also a genuine delimiter of the rendering template) or to "
        "another family's token set (<em>cross</em>). <strong>Transfer "
        "gap</strong> = cross-family flip rate minus same-family flip "
        "rate, in [-1, 1].</p>"
        + top_line
        + rate_line
        + (
            '<div class="table-wrap"><table><caption>Fake-turn flip rates per candidate.</caption>'
            '<thead><tr><th scope="col">Candidate</th>'
            '<th scope="col">Evaluable cells</th><th scope="col">Mean flip rate</th>'
            '<th scope="col">Same-family rate</th><th scope="col">Cross-family rate</th>'
            '<th scope="col">Transfer gap</th>'
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            if rows
            else ""
        )
        + excluded_line
        + "</section>"
    )


def _toolcall_section(result: dict[str, Any]) -> str:
    """Prompt-only tool-call syntax adherence per template family.

    Renders the ``toolcall`` payload (see experiments.toolcall_adherence)
    as a per-candidate table of selection / argument / relevance rates:
    a fixed 4-function tool list described in prompt text (no native
    function-calling APIs), 7 tool-relevant queries plus 2
    no-relevant-function queries answered with exactly ``NO_CALL``.
    The pre-registered falsifiable claim is a max-minus-min adherence
    gap of ~0 uniform across families; a measurably positive gap means
    the template is a first-class variable for tool-calling pipelines
    (the BFCL prompt-mode task, with the chat template as the
    independent variable). Rates are behavior measurements on fixed
    fixtures, never quality judgments. Every interpolation is escaped
    at this boundary.
    """
    audit: Any = result.get("toolcall")
    if type(audit) is not dict or not audit.get("available"):
        return ""
    per_candidate = audit.get("per_candidate")
    if type(per_candidate) is not dict:
        return ""

    def _rate(value: Any) -> str:
        """Return the rate."""
        if type(value) in (int, float):
            try:
                return f"{value:.2f}"
            except OverflowError:
                return "n/a"
        return "n/a"

    def _gap(value: Any) -> str:
        """Return the gap."""
        if type(value) in (int, float):
            try:
                return f"{value:+.2f}"
            except OverflowError:
                return "n/a"
        return "n/a"

    rows = []
    for key in sorted(per_candidate, key=str):
        cand = str(key)
        entry = per_candidate[key]
        if type(entry) is not dict:
            continue
        rows.append(
            '<tr><td><span class="mono">'
            + escape(cand)
            + "</span></td><td>"
            + escape(str(entry.get("n_evaluable")))
            + " / "
            + escape(str(entry.get("n")))
            + "</td><td>"
            + escape(_rate(entry.get("selection_rate")))
            + "</td><td>"
            + escape(_rate(entry.get("args_rate")))
            + "</td><td>"
            + escape(_rate(entry.get("relevance_rate")))
            + "</td><td>"
            + escape(_rate(entry.get("adherent_rate")))
            + "</td></tr>"
        )
    excluded = audit.get("excluded_candidates")
    excluded_line = ""
    if type(excluded) is list and excluded:
        excluded_line = (
            "<p>Candidates with no evaluable tool-call response: "
            + ", ".join(
                '<span class="mono">' + escape(str(name)) + "</span>"
                for name in excluded
                if type(name) is str
            )
            + ".</p>"
        )
    gap_lines = ""
    for rate_key, label in (
        ("selection_rate_gap", "selection"),
        ("args_rate_gap", "argument extraction"),
        ("relevance_rate_gap", "relevance detection"),
        ("adherent_rate_gap", "overall adherence"),
    ):
        gap = audit.get(rate_key)
        if type(gap) in (int, float):
            gap_lines += (
                f"<p>Max-minus-min {label} gap over evaluable candidates: "
                "<strong>" + escape(_gap(gap)) + "</strong>.</p>"
            )
    weakest = audit.get("weakest_candidates")
    weakest_line = ""
    if type(weakest) is list and weakest:
        weakest_line = (
            "<p>Weakest candidates (lowest overall adherence): "
            + ", ".join(
                '<span class="mono">' + escape(str(name)) + "</span>"
                for name in weakest
                if type(name) is str
            )
            + ".</p>"
        )
    return (
        '<section class="card" id="tool-calls">'
        "<h2>Prompt-only tool-call adherence</h2>"
        "<p>A fixed 4-function tool list was described in prompt text -- "
        "no native function-calling APIs -- and each family rendered the "
        "same 9 queries through its own template. The model must emit "
        'exactly one line: <span class="mono">CALL &lt;tool_name&gt; '
        '&lt;JSON arguments&gt;</span>, or <span class="mono">NO_CALL</span> '
        "when no tool fits. <strong>Selection</strong> = correct tool "
        "chosen; <strong>args</strong> = arguments match the expected "
        "typed values exactly; <strong>relevance</strong> = abstention "
        "decision correct. Unparseable lines count as failures on all "
        "three rates.</p>"
        + gap_lines
        + weakest_line
        + (
            '<div class="table-wrap"><table>'
            "<caption>Tool-call adherence rates per candidate.</caption>"
            '<thead><tr><th scope="col">Candidate</th>'
            '<th scope="col">Evaluable</th><th scope="col">Selection</th>'
            '<th scope="col">Args</th><th scope="col">Relevance</th>'
            '<th scope="col">Adherent</th>'
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            if rows
            else ""
        )
        + excluded_line
        + "</section>"
    )


def _aliasing_section(result: dict[str, Any]) -> str:
    """Candidate aliasing audit: render and measurement classes.

    Renders the ``aliasing`` payload (see
    experiments.aliasing_profile) as the effective-vs-recorded
    candidate counts, the render and measurement equivalence
    classes of the primary cohort, the per-pair coincidence
    matrix, and the alias-suspect abstention flag.
    Determinism-suspect cells lead the section when present: they
    are the audit's tripwire, and a fired flag must never be read
    over them. Classes are relative to this run's fixture set and
    backend tokenizer; the payload's per-cell digests are never
    rendered here. Every interpolation is escaped at this
    boundary.
    """
    audit: Any = result.get("aliasing")
    if type(audit) is not dict or not audit.get("available"):
        return ""
    candidates = audit.get("candidates")
    if type(candidates) is not list:
        return ""

    def _num(key: str) -> str:
        """Return the num."""
        value = audit.get(key)
        return escape(str(value)) if type(value) is int else "n/a"

    def _names(names: Any) -> str:
        """Return the names."""
        if type(names) is list:
            return ", ".join('<span class="mono">' + escape(str(n)) + "</span>" for n in names)
        return "n/a"

    suspect_items = []
    suspects = audit.get("determinism_suspect_cells")
    if type(suspects) is list:
        for cell in suspects:
            if type(cell) is not dict:
                continue
            fixture = cell.get("fixture")
            kind = cell.get("kind")
            suspect_items.append(
                "<li>"
                + _names(cell.get("candidates"))
                + " · fixture "
                + '<span class="mono">'
                + (escape(fixture) if type(fixture) is str else "unknown")
                + "</span> · "
                + (escape(kind) if type(kind) is str else "unknown")
                + " divergence</li>"
            )
    suspect_block = ""
    if suspect_items:
        suspect_block = (
            "<p><strong>Determinism-suspect cells:</strong> recorded "
            "cells inside one measurement class that a deterministic "
            "backend could not have produced from one measurement "
            "configuration -- backend nondeterminism or artifact "
            'corruption; <span class="mono">verify '
            "--determinism</span> replay discriminates. The audit "
            "asserts neither cause.</p><ul>" + "".join(suspect_items) + "</ul>"
        )
    class_rows = []
    measurement = audit.get("measurement_classes")
    if type(measurement) is list:
        for idx, members in enumerate(measurement):
            if type(members) is not list:
                continue
            class_rows.append(
                "<tr><td>measurement class "
                + escape(str(idx + 1))
                + "</td><td>"
                + _names(members)
                + "</td></tr>"
            )
    render = audit.get("render_classes")
    if type(render) is list and render != measurement:
        for idx, members in enumerate(render):
            if type(members) is not list:
                continue
            class_rows.append(
                "<tr><td>render class "
                + escape(str(idx + 1))
                + "</td><td>"
                + _names(members)
                + "</td></tr>"
            )
    pair_items = []
    pairs = audit.get("pairs")
    if type(pairs) is list:
        for pair in pairs:
            if type(pair) is not dict or pair.get("n_shared", 0) == 0:
                continue
            pair_items.append(
                "<li>"
                + _names(pair.get("candidates"))
                + " · identical prompts on "
                + escape(str(pair.get("n_shared")))
                + " of "
                + escape(str(pair.get("n_common_completed")))
                + " commonly completed fixtures</li>"
            )
    pairs_block = ""
    if pair_items:
        pairs_block = (
            "<p>Pairwise coincidence (per pair, over the fixtures "
            "both candidates completed; pairwise data, never "
            "clustered into classes):</p><ul>" + "".join(pair_items) + "</ul>"
        )
    if audit.get("alias_suspect_abstention") is True:
        flag_block = (
            "<p><strong>Alias-suspect abstention:</strong> the "
            'recorded decision is <span class="mono">ambiguous</span> '
            "and every passing candidate lies in one measurement "
            "class: "
            + _names(audit.get("alias_suspect_class_members"))
            + ". The tie is between names for one measurement, not "
            "between distinct prompts. This flag is advisory -- the "
            "recorded decision stands, and it does not assert any "
            "candidate would have survived held-out, which was never "
            "run for any of them.</p>"
        )
    else:
        detail = audit.get("alias_suspect_detail")
        flag_block = (
            "<p>Alias-suspect abstention flag: not fired"
            + (
                " (" + escape(detail.replace("_", " ")) + ")"
                if type(detail) is str and detail
                else ""
            )
            + ".</p>"
        )
    mismatch_line = ""
    if audit.get("evaluable_mismatch") is True:
        mismatch_line = (
            "<p><strong>Caution:</strong> candidates were not all "
            "measured on the same completed fixture set; classes "
            "are computed inside cohorts of identical completed "
            "sets, and the counts below describe the primary "
            "cohort.</p>"
        )
    unavailable_names = []
    excluded_names = []
    for entry in candidates:
        if type(entry) is not dict:
            continue
        name = entry.get("candidate")
        label = escape(str(name)) if type(name) is str and name else "unknown"
        if not entry.get("available"):
            unavailable_names.append(label)
        elif entry.get("in_primary_cohort") is not True:
            excluded_names.append(label)
    status_line = ""
    if unavailable_names:
        status_line += (
            "<p>Candidates with no usable recorded observations: "
            + ", ".join('<span class="mono">' + n + "</span>" for n in unavailable_names)
            + ".</p>"
        )
    if excluded_names:
        status_line += (
            "<p>Candidates outside the primary cohort (different "
            "completed fixture set): "
            + ", ".join('<span class="mono">' + n + "</span>" for n in excluded_names)
            + ".</p>"
        )
    return (
        '<section class="card" id="aliasing">'
        "<h2>Candidate aliasing audit</h2>"
        "<p>Did the run actually test distinct templates? "
        "Candidates are grouped over the recorded search prompts: "
        "render classes share one prompt token sequence on every "
        "fixture of the cohort; measurement classes also share "
        "the response boundary, which is part of the measurement. "
        "Counts are relative to this run's fixture set and "
        "backend tokenizer -- a collapse here is a "
        "tokenizer-by-fixture interaction, not a claim that the "
        "catalog families are duplicates. Redundant observations "
        "are counted once per fixture; they are also the "
        "replicates the determinism cross-check consumes.</p>"
        + suspect_block
        + mismatch_line
        + "<table><thead><tr><th>Count</th><th>Value</th></tr></thead><tbody>"
        + "<tr><td>Candidates recorded</td><td>"
        + _num("n_candidates_recorded")
        + "</td></tr>"
        + "<tr><td>Primary cohort</td><td>"
        + _num("n_primary_cohort")
        + "</td></tr>"
        + "<tr><td>Effective render candidates</td><td>"
        + _num("n_effective_render")
        + "</td></tr>"
        + "<tr><td>Effective measurement candidates</td><td>"
        + _num("n_effective_measurement")
        + "</td></tr>"
        + "<tr><td>Redundant completed observations</td><td>"
        + _num("n_redundant_completed_observations")
        + "</td></tr>"
        + "</tbody></table>"
        + (
            "<table><thead><tr><th>Class</th><th>Members</th></tr></thead><tbody>"
            + "".join(class_rows)
            + "</tbody></table>"
            if class_rows
            else ""
        )
        + pairs_block
        + flag_block
        + status_line
        + "</section>"
    )


def _repetition_section(result: dict[str, Any]) -> str:
    """Prompt-repetition probe: x2 vs recorded native vs padding control.

    Renders the ``repetition`` payload (see
    experiments.repetition_analysis): per candidate, the repetition
    effect (x2 pass rate minus pad pass rate over paired fixtures)
    with its paired denominator and pad-evaluable share beside it —
    never a bare effect — plus the length effect and the
    joiner-inclusive raw gain, and the per-arm selection readout.
    A withheld arm comparison leads the section: a flip computed
    over mismatched completed sets is not published, and the reader
    must see that before any number. Every interpolation is escaped
    at this boundary.
    """
    probe: Any = result.get("repetition")
    if type(probe) is not dict or not probe.get("available"):
        return ""
    candidates = probe.get("candidates")
    if type(candidates) is not dict:
        return ""

    def _rate(value: Any) -> str:
        """Return the rate."""
        if type(value) in (int, float) and not isinstance(value, bool):
            try:
                return escape(f"{float(value):+.3f}")
            except OverflowError:
                return "n/a"
        return "n/a"

    def _count(value: Any) -> str:
        """Return the count."""
        return escape(str(value)) if type(value) is int else "n/a"

    arms = probe.get("arms")
    withheld_arms = []
    arm_lines = []
    if type(arms) is dict:
        for arm in ("x2", "pad"):
            entry = arms.get(arm)
            if type(entry) is not dict:
                continue
            if entry.get("withheld") is True:
                withheld_arms.append(arm)
                continue
            winner = entry.get("winner")
            flip = entry.get("flip")
            arm_lines.append(
                "<li>"
                + escape(arm)
                + " arm: "
                + escape(str(entry.get("decision")))
                + (
                    " · winner " + '<span class="mono">' + escape(winner) + "</span>"
                    if type(winner) is str and winner
                    else ""
                )
                + (
                    " · <strong>selection flip</strong> against the recorded run"
                    if flip is True
                    else " · no selection flip"
                    if flip is False
                    else ""
                )
                + "</li>"
            )
    withheld_block = ""
    if withheld_arms:
        withheld_block = (
            "<p><strong>Selection comparison withheld for "
            + ", ".join(escape(a) for a in withheld_arms)
            + ":</strong> candidates did not all complete the same "
            "fixture set under that arm, so no winner or flip is "
            "reported for it — a flip the denominators cannot explain "
            "would be a phantom.</p>"
        )
    rows = []
    for name in sorted(candidates):
        entry = candidates[name]
        if type(entry) is not dict:
            continue
        label = '<span class="mono">' + escape(name) + "</span>"
        if entry.get("status") != "available":
            rows.append(
                "<tr><td>"
                + label
                + '</td><td colspan="4">unavailable ('
                + escape(str(entry.get("detail", "")).replace("_", " "))
                + ")</td></tr>"
            )
            continue
        contrasts = entry.get("contrasts")
        headline = contrasts.get("x2_vs_pad") if type(contrasts) is dict else None
        length = contrasts.get("pad_vs_x1") if type(contrasts) is dict else None
        raw = contrasts.get("x2_vs_x1") if type(contrasts) is dict else None
        rows.append(
            "<tr><td>"
            + label
            + "</td><td>"
            + (_rate(headline.get("repetition_effect")) if type(headline) is dict else "n/a")
            + " (n="
            + (_count(headline.get("n_paired")) if type(headline) is dict else "n/a")
            + ", pad-evaluable "
            + _rate(entry.get("pad_evaluable_share"))
            + ")</td><td>"
            + (_rate(length.get("length_effect")) if type(length) is dict else "n/a")
            + "</td><td>"
            + (_rate(raw.get("raw_gain")) if type(raw) is dict else "n/a")
            + "</td><td>"
            + _count(entry.get("n_pre_repeated"))
            + " pre-repeated, "
            + _count(entry.get("n_vacuous"))
            + " vacuous</td></tr>"
        )
    groups = probe.get("think_groups")
    group_line = ""
    if type(groups) is dict:
        parts = []
        for key, label in (("think", "think-channel"), ("no_think", "no think channel")):
            group = groups.get(key)
            if type(group) is dict and group.get("mean_repetition_effect") is not None:
                parts.append(
                    label
                    + ": "
                    + _rate(group.get("mean_repetition_effect"))
                    + " over "
                    + _count(group.get("n_candidates"))
                    + " candidates ("
                    + _count(group.get("n_distinct_renders"))
                    + " distinct renders)"
                )
        if parts:
            group_line = "<p>Repetition effect by group — " + "; ".join(parts) + ".</p>"
    return (
        '<section class="card" id="repetition">'
        "<h2>Prompt repetition</h2>"
        "<p>Does repeating the request inside the turn change the "
        "verdict? Each fixture's last user message was repeated "
        "verbatim (x2) and, as a control, followed by token-length-"
        "matched neutral filler (pad); the x1 arm is the recorded "
        "run itself. The headline contrast is x2 minus pad: at equal "
        "total prompt length, a difference is not explained by "
        "length alone — content position and repetition-specific "
        "attention are not separated by this design. The raw gain "
        "(x2 minus x1) additionally includes the effect of the "
        "joiner itself. Doubled prompts cost real compute here "
        "(full-prefix recompute, no KV cache).</p>"
        + withheld_block
        + ("<ul>" + "".join(arm_lines) + "</ul>" if arm_lines else "")
        + "<table><thead><tr><th>Candidate</th><th>Repetition effect "
        "(x2 − pad)</th><th>Length effect (pad − x1)</th><th>Raw gain "
        "(x2 − x1)</th><th>Fixtures</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + group_line
        + "</section>"
    )


def _distraction_section(result: dict[str, Any]) -> str:
    """Distractor-dilution probe: verdict curves over filler doses.

    Renders the ``distraction`` payload (see
    experiments.distraction_analysis): per candidate, the verdict
    counts per (kind, placement) curve over paired denominators,
    first-flip counts, and near-miss capture — labelled an upper
    bound wherever it appears, beside the neutral-control excess.
    Withheld condition selections lead the section: a flip
    computed over mismatched completed sets is not published.
    No curve is pooled and nothing is called decay: a rise at
    any dose is reported as-is. Every interpolation is escaped
    at this boundary.
    """
    probe: Any = result.get("distraction")
    if type(probe) is not dict or not probe.get("available"):
        return ""
    candidates = probe.get("candidates")
    if type(candidates) is not dict:
        return ""

    def _rate(value: Any) -> str:
        """Return the rate."""
        if type(value) in (int, float) and not isinstance(value, bool):
            try:
                return escape(f"{float(value):.3f}")
            except OverflowError:
                return "n/a"
        return "n/a"

    def _count(value: Any) -> str:
        """Return the count."""
        return escape(str(value)) if type(value) is int else "n/a"

    conditions = probe.get("conditions")
    withheld_keys: list[str] = []
    condition_lines: list[str] = []
    if type(conditions) is dict:
        for key in sorted(conditions, key=str):
            entry = conditions[key]
            if type(entry) is not dict:
                continue
            if entry.get("withheld") is True:
                withheld_keys.append(str(key))
                continue
            winner = entry.get("winner")
            flip = entry.get("flip")
            condition_lines.append(
                "<li>"
                + escape(str(key))
                + ": "
                + escape(str(entry.get("decision")))
                + (
                    " · winner " + '<span class="mono">' + escape(winner) + "</span>"
                    if type(winner) is str and winner
                    else ""
                )
                + (
                    " · <strong>selection flip</strong> against the recorded run"
                    if flip is True
                    else " · no selection flip"
                    if flip is False
                    else ""
                )
                + "</li>"
            )
    withheld_block = ""
    if withheld_keys:
        withheld_block = (
            "<p><strong>Selection withheld</strong> for "
            + escape(", ".join(withheld_keys))
            + ": the candidates did not complete the same fixture "
            "set under those conditions, so no winner or flip is "
            "published for them.</p>"
        )
    rows: list[str] = []
    for name in sorted(candidates, key=str):
        entry = candidates[name]
        if type(entry) is not dict:
            continue
        if entry.get("status") != "available":
            rows.append(
                '<tr><td><span class="mono">'
                + escape(str(name))
                + '</span></td><td colspan="5">unavailable</td></tr>'
            )
            continue
        curves = entry.get("curves")
        first_flip = entry.get("first_flip")
        capture = entry.get("capture")
        curve_bits: list[str] = []
        if type(curves) is dict:
            for curve_key in (
                "neutral_before",
                "neutral_after",
                "near-miss_before",
                "near-miss_after",
            ):
                points = curves.get(curve_key)
                if type(points) is not dict:
                    continue
                cells = []
                for level in sorted(
                    points,
                    key=lambda v: int(v) if str(v).isascii() and str(v).isdigit() else 0,
                ):
                    point = points[level]
                    if type(point) is not dict:
                        continue
                    cells.append(
                        escape(str(level))
                        + " tok: "
                        + _count(point.get("n_pass"))
                        + "/"
                        + _count(point.get("n_paired"))
                        + " ("
                        + _rate(point.get("rate"))
                        + ")"
                    )
                flip_entry = first_flip.get(curve_key) if type(first_flip) is dict else None
                flip_bit = ""
                if type(flip_entry) is dict:
                    flip_bit = (
                        " · first flips "
                        + escape(str(flip_entry.get("flip_counts")))
                        + " over "
                        + _count(flip_entry.get("n_population"))
                        + " native-passing; returns "
                        + _count(flip_entry.get("n_returns"))
                        + "; gains "
                        + _count(flip_entry.get("n_gains"))
                        + "/"
                        + _count(flip_entry.get("n_gain_population"))
                    )
                curve_bits.append(
                    "<div>" + escape(curve_key) + " — " + "; ".join(cells) + flip_bit + "</div>"
                )
        capture_bits: list[str] = []
        if type(capture) is dict:
            for cap_key in sorted(capture, key=str):
                cap = capture[cap_key]
                if type(cap) is not dict:
                    continue
                capture_bits.append(
                    escape(str(cap_key))
                    + ": "
                    + _count(cap.get("n_captured"))
                    + "/"
                    + _count(cap.get("n_population"))
                    + " ("
                    + _rate(cap.get("capture_rate"))
                    + "), excess over neutral "
                    + _rate(cap.get("capture_excess_over_neutral"))
                )
        rows.append(
            '<tr><td><span class="mono">'
            + escape(str(name))
            + "</span></td><td>"
            + "".join(curve_bits)
            + "</td><td>"
            + ("; ".join(capture_bits) if capture_bits else "n/a")
            + "</td><td>"
            + _count(entry.get("n_nearmiss_defined"))
            + " of "
            + _count(entry.get("n_fixtures"))
            + " (share "
            + _rate(entry.get("nearmiss_defined_share"))
            + ")</td><td>"
            + _count(entry.get("n_echo"))
            + "</td><td>"
            + _count(entry.get("n_vacuous"))
            + "</td></tr>"
        )
    return (
        '<section class="card" id="distraction">'
        "<h2>Distractor dilution</h2>"
        "<p>Does irrelevant bulk around the question change the "
        "verdict? Each fixture's last user message gained a filler "
        "span sized in exact rendered tokens — neutral prose at "
        "64/256/1024 tokens, or a frozen near-miss template "
        "carrying a planted wrong answer at 64/256 — placed "
        "before or after the content. Capture counts treated "
        "answers equal to the plant among fixtures the candidate "
        "passed natively; it is an <strong>upper bound</strong> "
        "on caused capture (a coincidental equal answer is "
        "indistinguishable at this tier), reported beside the "
        "excess over the same-dose neutral arm. Near-miss covers "
        "numeric-expected fixtures with plain ASCII formatting "
        "only, and its curves terminate at 256 tokens: at 1024 "
        "the plant would repeat dozens of times and the cell "
        "would measure repetition pressure, not dilution. Counts "
        "are per curve over paired denominators; no monotonicity "
        "is assumed.</p>"
        + withheld_block
        + ("<ul>" + "".join(condition_lines) + "</ul>" if condition_lines else "")
        + "<table><thead><tr><th>Candidate</th><th>Verdict curves "
        "(pass/paired by dose)</th><th>Capture (upper bound)</th>"
        "<th>Near-miss defined</th><th>Echo fixtures</th>"
        "<th>Vacuous</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>" + "</section>"
    )


def _multiturn_section(result: dict[str, Any]) -> str:
    """Multi-turn drift probe: does the winner survive its own history.

    Renders the ``multiturn`` payload (see
    experiments.multiturn_analysis): per candidate, the
    classification and, per depth, the paired pass rates for
    the three arms, the delivery discordant counts (drift and
    gains), and the self-history discordant pair with its net
    and McNemar diagnostic. The contrast is named as the
    bundle it is -- authorship, content, and history length
    move together -- and the generated tasks are named as an
    instrument, not a benchmark. Counts and rates only; no
    answer text is rendered. Every interpolation is escaped
    at this boundary.
    """
    probe: Any = result.get("multiturn")
    if type(probe) is not dict or not probe.get("available"):
        return ""
    candidates = probe.get("candidates")
    if type(candidates) is not dict:
        return ""

    def _rate(value: Any) -> str:
        """Return the rate."""
        if type(value) in (int, float) and not isinstance(value, bool):
            try:
                return escape(f"{float(value):.3f}")
            except OverflowError:
                return "n/a"
        return "n/a"

    def _count(value: Any) -> str:
        """Return the count."""
        return escape(str(value)) if type(value) is int else "n/a"

    rows: list[str] = []
    for name in sorted(candidates, key=str):
        entry = candidates[name]
        if type(entry) is not dict:
            continue
        if entry.get("status") != "available":
            rows.append(
                '<tr><td><span class="mono">'
                + escape(str(name))
                + '</span></td><td colspan="2">unavailable</td></tr>'
            )
            continue
        bits: list[str] = []
        classification = entry.get("classification")
        bits.append(
            "<div>Classification: <strong>" + escape(str(classification)) + "</strong>.</div>"
        )
        depths = entry.get("depths")
        if type(depths) is dict:
            for depth in sorted(depths, key=str):
                block = depths[depth]
                if type(block) is not dict:
                    continue
                rates = block.get("pass_rates")
                delivery = block.get("delivery")
                self_history = block.get("self_history")
                if type(rates) is not dict or type(delivery) is not dict:
                    continue
                line = (
                    "Depth "
                    + escape(str(depth))
                    + ": paired "
                    + _count(block.get("n_paired"))
                    + " of "
                    + _count(block.get("n_defined"))
                    + "; pass full "
                    + _rate(rates.get("full"))
                    + ", sharded "
                    + _rate(rates.get("sharded"))
                    + ", scripted "
                    + _rate(rates.get("scripted"))
                    + "; drift "
                    + _count(delivery.get("drift"))
                    + ", gains "
                    + _count(delivery.get("gains"))
                )
                if type(self_history) is dict:
                    line += (
                        "; self-history pair (sharded-only "
                        + _count(self_history.get("sh_only"))
                        + ", scripted-only "
                        + _count(self_history.get("sc_only"))
                        + ", net "
                        + _count(self_history.get("net_excess"))
                        + ", McNemar p "
                        + _rate(self_history.get("mcnemar_p"))
                        + ")"
                    )
                bits.append("<div>" + line + ".</div>")
        rows.append(
            '<tr><td><span class="mono">'
            + escape(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )
    return (
        '<section class="card" id="multiturn">'
        "<h2>Multi-turn drift</h2>"
        "<p>Generated shard tasks (ledger sums and shelf "
        "argmax, depths 2 and 3, 48 fixtures from the run seed) "
        "delivered three ways: every shard in one user turn, "
        "one shard per turn with the model's own boundary-cut "
        "replies as the history, and one shard per turn with a "
        "frozen <em>Noted.</em> history. Only the final turn is "
        "scored. The sharded-vs-scripted contrast is a bundle "
        "-- authorship, content, and history length move "
        "together, because content and length are the mechanism "
        "of self-history -- so it is reported as a discordant "
        "pair, never a net-only difference. Intermediate "
        "replies are screened for special-token literals before "
        "they become history. The generated tasks are a probe "
        "instrument, not a benchmark: they measure whether "
        "verdicts survive sharded delivery, not performance on "
        "real conversations.</p>"
        "<table><thead><tr><th>Candidate</th><th>Readout</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></section>"
    )


def _needle_section(result: dict[str, Any]) -> str:
    """Needle-depth probe: at which depth the fact gets lost.

    Renders the ``needle`` payload (see
    experiments.needle_analysis): per candidate, the dose-0
    control rate, the recall curve at five depths for both
    filler doses over the shared headline population, the
    mid-depth deficit, the needle-absent false-recall floor,
    and the duplicate-needle readout with its independence
    baseline. The scope limits are printed with the numbers,
    not under them: the filler is topically dissimilar prose,
    so the curves are a lower bound on position effects under
    realistic retrieval bulk, and each cell is one greedy
    generation. Counts and rates only; no needle values or
    answer text is rendered. Every interpolation is escaped
    at this boundary.
    """
    probe: Any = result.get("needle")
    if type(probe) is not dict or not probe.get("available"):
        return ""
    candidates = probe.get("candidates")
    readouts = probe.get("readouts")
    if type(candidates) is not dict or type(readouts) is not dict:
        return ""

    def _rate(value: Any) -> str:
        """Return the rate."""
        if type(value) in (int, float) and not isinstance(value, bool):
            try:
                return escape(f"{float(value):.3f}")
            except OverflowError:
                return "n/a"
        return "n/a"

    def _count(value: Any) -> str:
        """Return the count."""
        return escape(str(value)) if type(value) is int else "n/a"

    rows: list[str] = []
    for name in sorted(candidates, key=str):
        entry = candidates[name]
        if type(entry) is not dict:
            continue
        if entry.get("status") != "available":
            rows.append(
                '<tr><td><span class="mono">'
                + escape(str(name))
                + '</span></td><td colspan="2">unavailable</td></tr>'
            )
            continue
        bits: list[str] = []
        control = entry.get("dose0_control")
        if type(control) is dict:
            bits.append(
                "<div>Dose-0 control: "
                + _count(control.get("n_pass"))
                + " of "
                + _count(control.get("n_recorded"))
                + " passed (rate "
                + _rate(control.get("rate"))
                + ").</div>"
            )
        floors = entry.get("false_recall")
        duplicates = entry.get("duplicate")
        for dose in ("1024", "4096"):
            block = readouts.get(dose)
            if type(block) is not dict:
                continue
            curves = block.get("curves")
            curve = curves.get(name) if type(curves) is dict else None
            if type(curve) is not dict:
                continue
            points = ", ".join(
                escape(str(depth)) + "% " + _rate(point.get("rate"))
                for depth, point in sorted(curve.items(), key=lambda kv: int(kv[0]))
                if type(point) is dict
            )
            line = (
                "Dose "
                + escape(dose)
                + " (headline n "
                + _count(block.get("n_population"))
                + "): "
                + points
            )
            deficits = block.get("deficit")
            deficit = deficits.get(name) if type(deficits) is dict else None
            if deficit is None:
                line += "; deficit withheld (population below floor)"
            else:
                line += "; mid-depth deficit " + _rate(deficit)
            if type(floors) is dict and type(floors.get(dose)) is dict:
                line += "; false-recall floor " + _rate(floors[dose].get("rate"))
            if type(duplicates) is dict and type(duplicates.get(dose)) is dict:
                dup = duplicates[dose]
                line += (
                    "; duplicate pass "
                    + _rate(dup.get("pass_rate"))
                    + " vs independence "
                    + _rate(dup.get("independence_expected"))
                    + " (excess "
                    + _rate(dup.get("excess_over_independence"))
                    + "), rescued "
                    + _count(dup.get("n_rescued"))
                    + " of "
                    + _count(dup.get("n_mid_fail"))
                    + " mid-depth failures"
                )
            bits.append("<div>" + line + ".</div>")
        rows.append(
            '<tr><td><span class="mono">'
            + escape(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )
    return (
        '<section class="card" id="needle">'
        "<h2>Needle depth</h2>"
        "<p>Twenty generated cases plant one fact sentence "
        "with a unique synthetic code inside neutral filler at "
        "five depths of the bulk; the model is asked for the "
        "code. Dose is the rendered-token growth over the same "
        "case with no filler, and each depth offset is verified "
        "on the candidate's own rendered tokens, so the curves "
        "compare templates, not tokenizers. Curves, deficit, "
        "and first failure share one headline population per "
        "dose: the cases every candidate passed with no filler "
        "and completed at all five depths. The deficit is the "
        "edge mean minus the middle rate. The filler is "
        "topically dissimilar prose, so these curves are a "
        "lower bound on position effects under realistic "
        "retrieval bulk, and each cell is one greedy "
        "generation; at the 4096 dose the filler stream wraps "
        "the corpus about three times. A flat curve at both "
        "doses, at the candidate's own false-recall floor, is "
        "what 'no position effect for this template' looks "
        "like.</p>"
        "<table><thead><tr><th>Candidate</th><th>Readout</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></section>"
    )


def _placement_section(result: dict[str, Any]) -> str:
    """Instruction placement probe: question before or after the payload.

    Renders the ``placement`` payload (see
    experiments.placement_analysis): per candidate, the
    question-first vs payload-first discordant counts split by
    direction over the paired cases, the classification, the
    sandwich control level on the same cases, the closed-book
    screening yield, and the per-arm selection winners. The
    scope limits are printed with the numbers, not under them:
    the sandwich bounds what recency and exposure together
    produce without identifying which, the control is a
    deliberately conservative bar, and each cell is one greedy
    generation on generated fixtures. Counts and rates only;
    no payload, question, or answer text is rendered. Every
    interpolation is escaped at this boundary.
    """
    probe: Any = result.get("placement")
    if type(probe) is not dict or not probe.get("available"):
        return ""
    candidates = probe.get("candidates")
    conditions = probe.get("conditions")
    if type(candidates) is not dict or type(conditions) is not dict:
        return ""

    def _rate(value: Any) -> str:
        """Return the rate."""
        if type(value) in (int, float) and not isinstance(value, bool):
            try:
                return escape(f"{float(value):.3f}")
            except OverflowError:
                return "n/a"
        return "n/a"

    def _count(value: Any) -> str:
        """Return the count."""
        return escape(str(value)) if type(value) is int else "n/a"

    rows: list[str] = []
    for name in sorted(candidates, key=str):
        entry = candidates[name]
        if type(entry) is not dict:
            continue
        if entry.get("status") != "available":
            rows.append(
                '<tr><td><span class="mono">'
                + escape(str(name))
                + '</span></td><td colspan="2">unavailable</td></tr>'
            )
            continue
        bits: list[str] = []
        headline = entry.get("headline")
        if type(headline) is dict:
            bits.append(
                "<div>Paired cases "
                + _count(headline.get("n_paired"))
                + ": question-first only "
                + _count(headline.get("n_qf_only"))
                + ", payload-first only "
                + _count(headline.get("n_pf_only"))
                + ", both "
                + _count(headline.get("n_both"))
                + ", neither "
                + _count(headline.get("n_neither"))
                + " ("
                + escape(str(headline.get("classification", "")))
                + ").</div>"
            )
        control = entry.get("control_level")
        bits.append(
            "<div>Sandwich control level "
            + (_count(control) if type(control) is int else "n/a")
            + " over "
            + _count(entry.get("n_control"))
            + " shared cases; directional placement effect falsified: "
            + ("yes" if entry.get("placement_effect_falsified") is True else "no")
            + ".</div>"
        )
        arms = entry.get("arms")
        if type(arms) is dict:
            rates = ", ".join(
                escape(arm.replace("_", " ")) + " " + _rate(arms[arm].get("rate"))
                if type(arms.get(arm)) is dict
                else ""
                for arm in ("question_first", "payload_first", "sandwich")
            )
            bits.append("<div>Arm pass rates: " + rates + ".</div>")
        closed = entry.get("closed_book")
        if type(closed) is dict:
            bits.append(
                "<div>Closed-book screen: "
                + _count(entry.get("n_screened"))
                + " cases excluded (closed-book pass rate "
                + _rate(closed.get("pass_rate"))
                + "), "
                + _count(entry.get("n_closed_book_incomplete"))
                + " cases with no closed-book verdict.</div>"
            )
        rows.append(
            '<tr><td><span class="mono">'
            + escape(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )
    winners: list[str] = []
    for arm in ("question_first", "payload_first", "sandwich"):
        cond = conditions.get(arm)
        if type(cond) is not dict:
            continue
        if cond.get("withheld"):
            winners.append(escape(arm.replace("_", " ")) + ": selection withheld")
        else:
            winner = cond.get("winner")
            winners.append(
                escape(arm.replace("_", " "))
                + ": "
                + (escape(str(winner)) if winner else "no unique winner")
                + (" (flip vs payload-first)" if cond.get("flip") is True else "")
            )
    return (
        '<section class="card" id="placement">'
        "<h2>Question/payload placement</h2>"
        "<p>Twenty-four generated cases each bury one "
        "answer-bearing fact in the middle sentence of a "
        "three-sentence payload; the same question is asked "
        "before the payload, after it, twice around it (the "
        "sandwich control), and alone (the closed-book screen: "
        "a pass without the payload excludes the case for that "
        "candidate). The headline is the discordant count "
        "between the two single-question layouts, split by "
        "direction; the falsification bar is the sandwich "
        "control's discordance on the same cases, a deliberately "
        "conservative maximum that bounds what recency and "
        "exposure together produce without identifying which. "
        "Each cell is one greedy generation.</p>"
        "<table><thead><tr><th>Candidate</th><th>Readout</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + ("<p>Per-arm selection: " + "; ".join(winners) + ".</p>" if winners else "")
        + "</section>"
    )


def _instruction_distance_section(result: dict[str, Any]) -> str:
    """Instruction-distance probe: one frozen format directive
    separated from the block it governs by verified token doses.

    Renders the ``instruction_distance`` payload (see
    experiments.instruction_distance_analysis): per candidate,
    per-arm compliance and byte-exact rates with the
    decomposition and attrition (context-exceeded,
    evaluation-failed, unmatchable, screened) beside each, the
    per-dose D/S/T decision block with the S value-kept /
    value-lost split and the reverse-discordant count, the
    id-divergence report per arm, and the operating
    characteristics paragraph beside every dose decision. The
    claim discipline is amendment F of the design doc: the
    label is an excess-of-between-losses label, power is
    stated for a large drop only, the 0.25 power row is
    labelled below resolution, and no sentence describes a
    curve, a threshold, or decay. Counts and rates only; no
    payload, question, or answer text is rendered. Every
    payload-derived value below goes through _escape_text,
    never bare escape().
    """
    probe: Any = result.get("instruction_distance")
    if type(probe) is not dict or not probe.get("available"):
        return ""
    candidates = probe.get("candidates")
    if type(candidates) is not dict:
        return ""

    def _rate(value: Any) -> str:
        """Return the rate."""
        if type(value) in (int, float) and not isinstance(value, bool):
            try:
                return _escape_text(f"{float(value):.3f}")
            except OverflowError:
                return "n/a"
        return "n/a"

    def _count(value: Any) -> str:
        """Return the count."""
        return _escape_text(str(value)) if type(value) is int else "n/a"

    def _arm_line(name: str, arm: dict[str, Any]) -> str:
        """Return the arm line."""
        decomp = arm.get("decomposition")
        decomp_bits = ""
        if type(decomp) is dict:
            decomp_bits = (
                " Decomposition — compliant-correct "
                + _count(decomp.get("compliant_correct"))
                + ", compliant-wrong "
                + _count(decomp.get("compliant_wrong"))
                + ", noncompliant-correct-value "
                + _count(decomp.get("noncompliant_correct_value"))
                + ", noncompliant-wrong "
                + _count(decomp.get("noncompliant_wrong"))
                + ", boundary-only "
                + _count(decomp.get("boundary_only"))
                + "."
            )
        return (
            "<div>"
            + _escape_text(name)
            + " compliance rate "
            + _rate(arm.get("compliance_rate"))
            + " ("
            + _count(arm.get("n_compliant"))
            + " of "
            + _count(arm.get("n_completed"))
            + " completed), byte-exact pass rate "
            + _rate(arm.get("rate"))
            + " ("
            + _count(arm.get("n_pass"))
            + " of "
            + _count(arm.get("n_completed"))
            + "), "
            + _count(arm.get("n_context_exceeded"))
            + " context-exceeded, "
            + _count(arm.get("n_evaluation_failed"))
            + " evaluation-failed, "
            + _count(arm.get("n_unmatchable"))
            + " unmatchable, "
            + _count(arm.get("n_screened"))
            + " screened."
            + decomp_bits
            + "</div>"
        )

    def _oc_paragraph(direct_passes: Any) -> str:
        """Return the oc paragraph."""
        return (
            "<div>At D = "
            + _count(direct_passes)
            + ", this panel (24 cases) is a per-dose large-drop "
            "falsifier: the label fires only when separated-arm "
            "noncompliance on the adjacent compliant-correct "
            "cases reaches S &gt;= 6 and exceeds the dose-matched "
            "displaced control's losses by S - T &gt;= 5, at "
            "D &gt;= 10. Measured operating characteristics of "
            "the (10, 6, 5) rule (exact enumeration, four "
            "independent derivations): a random labeller fires "
            "0.13% of the time; a competent candidate with no "
            "distance effect fires 0.58%, and at most 0.77% "
            "under every paired null executed. Power is 90.9% at "
            "a 50% conditional drop rate — the drop size this "
            "instrument is built to catch — and 99.9% at 0.75, "
            "but only 24.6% at a 25% drop rate, which is below "
            "this panel's resolution: at that size a "
            "below_threshold is the expected outcome three times "
            "in four, not evidence of no effect. The pure-length "
            "confound (filler presence alone, nothing between "
            "the directive and the block) fires 7.88% under the "
            "independent-cells convention; that convention is "
            "the worst case only when the separated and "
            "displaced losses are non-negatively correlated — "
            "under negatively paired nulls the pure-length rate "
            "rises to at most 13.3% (at correlation -0.8, "
            "executed), while a perfectly case-correlated length "
            "effect gives S = T and can never fire. Family-wise "
            "over this panel's 48 decisions (2 doses x 24 "
            "candidates): at the no-effect rate the exact "
            "expectation is 0.28 false labels with probability "
            "0.24 of at least one; under candidate-level frailty "
            "(Beta baselines, kappa 50/20/10) the expectation is "
            "0.40-0.59 with probability 0.33-0.44; at the "
            "printed pure-length bound the expectation is about "
            "3.78 labels with probability about 0.98 — read any "
            "single fire against the panel it sits in. Verdict "
            "availability: the D &gt;= 10 floor is reached with "
            "probability 0.511 when the adjacent "
            "compliant-correct rate is 0.40 and 0.153 at 0.30, "
            "so insufficient is the common verdict for weak "
            "candidates by construction.</div>"
        )

    rows: list[str] = []
    for name in sorted(candidates, key=str):
        entry = candidates[name]
        if type(entry) is not dict:
            continue
        if entry.get("status") != "available":
            rows.append(
                '<tr><td><span class="mono">'
                + _escape_text(str(name))
                + '</span></td><td colspan="2">unavailable</td></tr>'
            )
            continue
        bits: list[str] = []
        arms = entry.get("arms")
        if type(arms) is dict:
            for arm_name in (
                "adj",
                "sep_256",
                "sep_1024",
                "disp_256",
                "disp_1024",
                "adj_after",
            ):
                arm = arms.get(arm_name)
                if type(arm) is dict:
                    bits.append(_arm_line(arm_name, arm))
        doses = entry.get("doses")
        if type(doses) is dict:
            for dose_key in ("256", "1024"):
                dose_block = doses.get(dose_key)
                if type(dose_block) is not dict:
                    continue
                bits.append(
                    "<div>Dose "
                    + _escape_text(dose_key)
                    + ": population "
                    + _count(dose_block.get("n_population"))
                    + " (completed in adjacent, separated, and "
                    "displaced), D "
                    + _count(dose_block.get("direct_passes"))
                    + ", S "
                    + _count(dose_block.get("separated_losses"))
                    + " (value kept "
                    + _count(dose_block.get("s_value_kept"))
                    + ", value lost "
                    + _count(dose_block.get("s_value_lost"))
                    + "), T "
                    + _count(dose_block.get("displaced_losses"))
                    + ", S - T "
                    + _count(dose_block.get("margin"))
                    + ", reverse discordants "
                    + _count(dose_block.get("n_reverse"))
                    + " (adjacent noncompliant, separated "
                    "compliant; printed beside the decision, never "
                    "in the rule), decision "
                    + _escape_text(str(dose_block.get("decision", "")))
                    + ".</div>"
                )
                bits.append(_oc_paragraph(dose_block.get("direct_passes")))
        ids = entry.get("ids")
        if type(ids) is dict:
            for arm_name in ("sep_256", "sep_1024", "disp_256", "disp_1024", "adj_after"):
                block = ids.get(arm_name)
                if type(block) is not dict:
                    continue
                deltas = block.get("id_length_delta_by_case")
                delta_text = ", ".join(_count(d) for d in deltas) if type(deltas) is list else "n/a"
                bits.append(
                    "<div>Ids ("
                    + _escape_text(arm_name)
                    + " vs adjacent): identical in "
                    + _count(block.get("n_ids_identical"))
                    + " cases (a rendering anomaly — the arms "
                    "differ in text by construction), same-ids "
                    "compliance flips "
                    + _count(block.get("n_same_ids_compliance_flips"))
                    + " (a determinism finding). Id-length deltas "
                    "("
                    + _escape_text(arm_name)
                    + " minus adjacent, case order): "
                    + delta_text
                    + ". Max abs id delta "
                    + _count(block.get("max_abs_id_delta"))
                    + ".</div>"
                )
        rows.append(
            '<tr><td><span class="mono">'
            + _escape_text(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )
    return (
        '<section class="card" id="instruction-distance">'
        "<h2>Instruction distance (directive-to-question separation)</h2>"
        "<p>Twenty-four generated cases, each a fixed "
        "fact+question block governed by one frozen format "
        "directive ('Reply with only the number: no units, no "
        "sentence, nothing else.' — the only directive in any "
        "prompt), rendered in six arms per candidate: adjacent "
        "(directive, block), separated at 256 and 1024 "
        "(directive, filler, block), displaced at the same "
        "doses (directive, block, filler), and a descriptive "
        "order control (block, directive). Distance here means "
        "rendered tokens between the frozen directive and the "
        "question block, verified per candidate on its own "
        "rendered tokens — never edit distance. A dose is the "
        "exact rendered-token delta of the arm over the case's "
        "adjacent render, sized from the neutral corpus; a cell "
        "whose delta cannot be sized exactly is unmatchable, "
        "never approximated. The headline scorer is compliance "
        "under a frozen grammar — the answer span is ASCII "
        "digits only over the whole span — decomposed against "
        "correctness (the value kept or lost) and printed per "
        "arm beside the byte-exact verdict, which is "
        "descriptive only. The decision conditions on the "
        "adjacent arm alone: D counts cases whose adjacent cell "
        "is compliant-correct among cases completed in the "
        "adjacent, separated, and displaced arms of the dose. "
        "The label is an excess-of-between-losses label: it "
        "certifies separated-arm losses beyond the dose-matched "
        "displaced control's, and nothing more — a pure "
        "constraint-to-generation effect yields S = T and reads "
        "below_threshold even at total collapse, and a "
        "between-effect exactly matched by the control's own "
        "trailing-filler losses does the same. Two residual "
        "channels are named, not argued away: in the separated "
        "arms the filler also sits between the directive and "
        "the fact (fact-to-question adjacency is held, and the "
        "value-kept / value-lost split of S is printed so a "
        "fact-mediated pattern is visible as such), and on the "
        "constraint-first arms, filler immediately before the "
        "block is coextensive with between-ness, so a pure "
        "filler-proximity effect — the directive never "
        "consulted — would fire this label; the separating arm "
        "is not among the six. Duplication disclosure: the "
        "displaced arm is byte-identical to a shipped "
        "distraction after-cell on the candidate fixture, and "
        "the separated arm is a constant prefix over a shipped "
        "before-cell; the residue this probe adds is the "
        "mid-message insertion point between a directive and "
        "the block it governs, the differencing of the "
        "displaced control inside the decision rule, and the "
        "compliance scorer. Closed-book floor, by construction "
        "rather than by an arm: the value is a seed-derived "
        "3-digit number stated only in the fact sentence, so a "
        "candidate that never saw the fact can be "
        "compliant-correct only by guessing (900 equiprobable "
        "values under the generator; model priors over round "
        "numbers are a stated caveat, not a hidden one). The "
        "claim is scoped to this one frozen directive string "
        "and to neutral prose filler; a second phrasing or a "
        "topically related filler is a different treatment. "
        "Each cell is one greedy generation. Records carry "
        "digests only.</p>"
        "<table><thead><tr><th>Candidate</th><th>Readout</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></section>"
    )


def _history_frame_section(result: dict[str, Any]) -> str:
    """History-frame probe: a frozen synthetic history segment
    reframed in two foreign chat-template framings.

    Renders the ``history_frame`` payload (see
    experiments.history_frame_analysis). The first sentence is
    the naming audit's disclosure: only the prepended frozen
    synthetic segment is reframed; the fixtures' own messages
    are never reframed. Per candidate: per-arm pass rates with
    attrition, the two headline contrasts (paired counts,
    decomposition, classification with its exact at-n
    random-labeller fire rate, id-divergence readout, ungated
    sensitivity), the descriptive verbatim readout with its
    inapplicable accounting, and the encoding stratum beside
    every classification. After the table: per-arm selection
    conditions, winner headlines per contrast, the flips
    against the primary run (labelled as pooling history
    presence with frame), and the operating-characteristics
    paragraph with the computed family-wise expectation, the
    jitter grid, and the named residual channels. Counts and
    rates only; no prompt, history, or answer text is
    rendered. Every payload-derived value below goes through
    _escape_text, never bare escape().
    """
    probe: Any = result.get("history_frame")
    if type(probe) is not dict or not probe.get("available"):
        return ""
    candidates = probe.get("candidates")
    if type(candidates) is not dict:
        return ""

    def _rate(value: Any) -> str:
        """Return the rate."""
        if type(value) in (int, float) and not isinstance(value, bool):
            try:
                return _escape_text(f"{float(value):.3f}")
            except OverflowError:
                return "n/a"
        return "n/a"

    def _pct(value: Any) -> str:
        """Return the pct."""
        if type(value) in (int, float) and not isinstance(value, bool):
            try:
                return _escape_text(f"{100.0 * float(value):.2f}")
            except OverflowError:
                return "n/a"
        return "n/a"

    def _count(value: Any) -> str:
        """Return the count."""
        return _escape_text(str(value)) if type(value) is int else "n/a"

    def _arm_line(name: str, arm: dict[str, Any]) -> str:
        """Return the arm line."""
        return (
            "<div>"
            + _escape_text(name)
            + " pass rate "
            + _rate(arm.get("rate"))
            + " ("
            + _count(arm.get("n_pass"))
            + " of "
            + _count(arm.get("n_completed"))
            + " completed), "
            + _count(arm.get("n_context_exceeded"))
            + " context-exceeded, "
            + _count(arm.get("n_evaluation_failed"))
            + " evaluation-failed.</div>"
        )

    stratum = probe.get("stratum")
    stratum_bits: dict[str, str] = {}
    if type(stratum) is dict:
        for arm_name in ("foreign_chatml", "foreign_llama3"):
            block = stratum.get(arm_name)
            if type(block) is dict:
                label = block.get("label")
                stratum_text = "Encoding stratum " + _escape_text(str(label)) + "."
                literals = block.get("literals")
                if label == "mixed" and type(literals) is dict:
                    stratum_text += (
                        " Per-literal token-real: "
                        + ", ".join(
                            _escape_text(str(lit)) + " " + ("yes" if real else "no")
                            for lit, real in sorted(literals.items(), key=lambda kv: str(kv[0]))
                        )
                        + "."
                    )
                stratum_bits[arm_name] = stratum_text

    rows: list[str] = []
    for name in sorted(candidates, key=str):
        entry = candidates[name]
        if type(entry) is not dict:
            continue
        if entry.get("status") != "available":
            rows.append(
                '<tr><td><span class="mono">'
                + _escape_text(str(name))
                + '</span></td><td colspan="2">unavailable</td></tr>'
            )
            continue
        bits: list[str] = []
        arms = entry.get("arms")
        if type(arms) is dict:
            for arm_name in ("own", "foreign_chatml", "foreign_llama3", "verbatim_chatml"):
                arm = arms.get(arm_name)
                if type(arm) is dict:
                    bits.append(_arm_line(arm_name, arm))
        contrasts = entry.get("contrasts")
        if type(contrasts) is dict:
            for arm_name in ("foreign_chatml", "foreign_llama3"):
                block = contrasts.get(arm_name)
                if type(block) is not dict:
                    continue
                if block.get("applicable") is not True:
                    bits.append(
                        "<div>"
                        + _escape_text(arm_name)
                        + ": inapplicable for this candidate — the foreign "
                        "history segment is render-identical to the own "
                        "segment on every fixture ("
                        + _count(block.get("n_render_identical"))
                        + " render-identical), so no classification is "
                        "emitted; an inapplicable contrast is never a "
                        "robustness reading.</div>"
                    )
                    continue
                decomp = block.get("decomposition")
                decomp_bits = ""
                if type(decomp) is dict:
                    decomp_bits = (
                        " Decomposition of the discordant pairs — "
                        "boundary-only "
                        + _count(decomp.get("boundary_only"))
                        + ", exact-only "
                        + _count(decomp.get("exact_only"))
                        + ", mixed "
                        + _count(decomp.get("mixed"))
                        + "."
                    )
                deltas = block.get("id_length_delta")
                delta_text = ", ".join(_count(d) for d in deltas) if type(deltas) is list else "n/a"
                bits.append(
                    "<div>"
                    + _escape_text(arm_name)
                    + " vs own: n "
                    + _count(block.get("n_paired"))
                    + " paired ("
                    + _count(block.get("n_render_identical"))
                    + " render-identical excluded, "
                    + _count(block.get("n_ids_identical"))
                    + " ids-identical excluded, "
                    + _count(block.get("n_same_ids_flips"))
                    + " same-ids verdict flips recorded as determinism "
                    "findings), own-only passes "
                    + _count(block.get("own_only"))
                    + ", foreign-only passes "
                    + _count(block.get("foreign_only"))
                    + ", discordant "
                    + _count(block.get("discordant"))
                    + ", paired rates own "
                    + _rate(block.get("paired_rate_own"))
                    + " / foreign "
                    + _rate(block.get("paired_rate_foreign"))
                    + ", McNemar p "
                    + _rate(block.get("mcnemar_p"))
                    + ", classification "
                    + _escape_text(str(block.get("classification", "")))
                    + ". Exact random-labeller fire rate at this n: "
                    + _pct(block.get("random_fire_rate"))
                    + " percent. Ungated sensitivity (all paired "
                    "completed fixtures, before divergence exclusions): n "
                    + _count(block.get("ungated_n"))
                    + ", own-only "
                    + _count(block.get("ungated_own_only"))
                    + ", foreign-only "
                    + _count(block.get("ungated_foreign_only"))
                    + ". Id-length deltas (own minus foreign, fixture "
                    "order): "
                    + delta_text
                    + ". Max abs id delta "
                    + _count(block.get("max_abs_id_delta"))
                    + "."
                    + decomp_bits
                    + " "
                    + stratum_bits.get(arm_name, "")
                    + "</div>"
                )
        verbatim = entry.get("verbatim")
        if type(verbatim) is dict:
            if verbatim.get("applicable") is True:
                bits.append(
                    "<div>verbatim_chatml (descriptive only): applicable "
                    "— paired "
                    + _count(verbatim.get("n_paired"))
                    + " fixtures completed in both chatml arms, "
                    "foreign-only passes "
                    + _count(verbatim.get("twin_only"))
                    + ", verbatim-only passes "
                    + _count(verbatim.get("verbatim_only"))
                    + "; that discordance is the bos-substitution share "
                    "of the foreign_chatml contrast, never a decision "
                    "input.</div>"
                )
            else:
                bits.append(
                    "<div>verbatim_chatml (descriptive only): "
                    "inapplicable for this candidate — the candidate's "
                    "bos equals chatml's, so the arm is byte-identical "
                    "to foreign_chatml; its discordance is never pooled "
                    "and never read as a bos share of zero.</div>"
                )
        rows.append(
            '<tr><td><span class="mono">'
            + _escape_text(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )

    tail: list[str] = []
    conditions = probe.get("conditions")
    selection = probe.get("selection")
    if type(conditions) is dict:
        cond_bits: list[str] = []
        for arm_name in ("own", "foreign_chatml", "foreign_llama3"):
            block = conditions.get(arm_name)
            if type(block) is not dict:
                continue
            if block.get("withheld") is True:
                cond_bits.append(
                    _escape_text(arm_name)
                    + " selection withheld (mismatched or empty completed sets)"
                )
            else:
                winner = block.get("winner")
                cond_bits.append(
                    _escape_text(arm_name)
                    + " selection "
                    + _escape_text(str(block.get("decision", "")))
                    + ((" — winner " + _escape_text(str(winner))) if winner else "")
                )
        if cond_bits:
            tail.append(
                "<p>Per-arm selection on the common completed set: "
                + "; ".join(cond_bits)
                + ".</p>"
            )
    if type(selection) is dict:
        sel_bits: list[str] = []
        for arm_name in ("foreign_chatml", "foreign_llama3"):
            block = selection.get(arm_name)
            if type(block) is dict:
                sel_bits.append(
                    _escape_text(arm_name)
                    + " winner headline "
                    + _escape_text(str(block.get("headline", "")))
                )
        if sel_bits:
            tail.append("<p>" + "; ".join(sel_bits) + ".</p>")
    flips = probe.get("flips")
    if type(flips) is list:
        if flips:
            flip_bits = ", ".join(
                _escape_text(str(f.get("arm", "")))
                + " ("
                + _escape_text(str(f.get("decision_from", "")))
                + " -> "
                + _escape_text(str(f.get("decision_to", "")))
                + ")"
                for f in flips
                if type(f) is dict
            )
            tail.append(
                "<p>Flips against the primary run: "
                + flip_bits
                + ". These flips pool history presence with frame and "
                "are printed for context only — never the headline.</p>"
            )
        else:
            tail.append("<p>No per-arm selection flips against the primary run.</p>")
    familywise = probe.get("familywise")
    fw_bits = ""
    if type(familywise) is dict:
        fw_bits = (
            " Family-wise over this run's "
            + _count(familywise.get("decisions"))
            + " classifications: the computed expectation is "
            + _rate(familywise.get("expected_false_labels"))
            + " false labels with probability "
            + _rate(familywise.get("p_at_least_one"))
            + " of at least one, summed from the exact at-n "
            "random-labeller rates printed beside each classification "
            "(at n = 100 over 48 decisions that expectation is 1.76 "
            "with probability 0.83) — read any single fire against "
            "the panel it sits in."
        )
    tail.append(
        "<p>Operating characteristics of the frozen rule (n at "
        "least 10 paired, discordant at least 6, two-sided exact "
        "McNemar p below 0.05): asymmetric fire is at most 3.6 "
        "percent on the random-labeller null and at most 3.1 "
        "percent on competent nulls at every tabulated panel size; "
        "the exact rate at each classification's own n prints "
        "beside it. Under symmetric two-arm jitter the fire rate "
        "is 0.04 percent at epsilon 0.01, 0.48 percent at 0.02, "
        "and 2.13 percent at 0.05 (D-invariant under the symmetric "
        "model, the model matching this stage, where both arms "
        "are freshly generated in one run); one-sided jitter "
        "variants are arm-asymmetric perturbations — inside the "
        "alternative, not the null. Power at a 25-point drop is "
        "0.99 at n = 100 but a coin flip (0.52) at n = 30, so on "
        "small fixture sets below_threshold is weak evidence, "
        "never invariance, and insufficient is the honest verdict "
        "when the panel cannot speak."
        + fw_bits
        + " Residual channels, named so the label is never read "
        "as a mechanism certificate: token length and marker "
        "encoding are consequences of the framing strings and "
        "move with the treatment by construction, and arm-level "
        "directional channels — batch composition and numerics "
        "correlated with the foreign prompts' length — lie "
        "inside the string-scoped claim. The label certifies "
        "string-level discordance for these named framings and "
        "this frozen history, on this checkpoint, and nothing "
        "about histories, migrations, or templates in "
        "general.</p>"
    )
    return (
        '<section class="card" id="history-frame">'
        "<h2>History frame (foreign framing of the prepended history)</h2>"
        "<p>This probe prepends the frozen six-turn synthetic "
        "history to every search fixture and reframes only that "
        "prepended segment — the fixture's own messages are never "
        "reframed. Per candidate, the prepended segment is "
        "rendered under four arms: own (the candidate's own "
        "framing), foreign_chatml and foreign_llama3 (the frozen "
        "chatml / llama3 framings of the history segment only), "
        "and a descriptive verbatim_chatml arm differing from "
        "foreign_chatml only in the leading bos string. The tail, "
        "the prompt-initial bos, and the open suffix are always "
        "the candidate's own. The own arm is byte-identical to "
        "the synth-history history_6 render of the same messages; "
        "verdict-identity with the synth-history stage is not "
        "claimed, because that stage runs namespaced ids and "
        "hence different request seeds — this stage's baseline "
        "is its own arm. The scorer is byte-exact conjoined with "
        "boundary presence; no stripping or normalisation is "
        "added. The frozen history texts and framing literals "
        "are screened for special-token literals before any "
        "generation; user fixture contents are deliberately not "
        "screened, because they are byte-identical across arms "
        "and cannot manufacture a frame contrast. The verbatim "
        "arm has a single donor by design: there is no "
        "verbatim_llama3, and that asymmetry is scope, not an "
        "omission. Each cell is one greedy generation. Records "
        "carry digests only.</p>"
        "<table><thead><tr><th>Candidate</th><th>Readout</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>" + "".join(tail) + "</section>"
    )


def _prune_sweep_section(result: dict[str, Any]) -> str:
    """Prune-sweep probe: seeded interior word loss from a
    frozen generated passage, classified per candidate.

    Renders the ``prune_sweep`` payload (see
    experiments.prune_sweep_analysis). The first sentence is
    the naming audit's disclosure: the probe prunes passage
    words. Per candidate, in the design's printed order:
    per-arm pass rates with attrition; the digest
    decomposition split per arm with the boundary-only count;
    C, A, B, and the decision, with A split by first-flip
    dose; the descriptive shape readout; the tail72
    displaced-control contrast with the word-count-loss
    reading when the two arms are within noise; the
    id-length report with the exclusion counts; and the
    dropped-set class shares. After the table: the
    operating-characteristics paragraph (exact enumerations,
    amendment A), the bundle fencing (amendment F), and the
    printed limitations. Counts and rates only; no passage,
    prompt, or answer text is rendered. Every payload-derived
    value below goes through _escape_text, never bare
    escape().
    """
    probe: Any = result.get("prune_sweep")
    if type(probe) is not dict or not probe.get("available"):
        return ""
    candidates = probe.get("candidates")
    if type(candidates) is not dict:
        return ""

    def _rate(value: Any) -> str:
        """Return the rate."""
        if type(value) in (int, float) and not isinstance(value, bool):
            try:
                return _escape_text(f"{float(value):.3f}")
            except OverflowError:
                return "n/a"
        return "n/a"

    def _count(value: Any) -> str:
        """Return the count."""
        return _escape_text(str(value)) if type(value) is int else "n/a"

    arm_names = ("dose0", "dose18", "dose36", "dose72", "tail72", "closed_book")
    rows: list[str] = []
    for name in sorted(candidates, key=str):
        entry = candidates[name]
        if type(entry) is not dict:
            continue
        if entry.get("status") != "available":
            rows.append(
                '<tr><td><span class="mono">'
                + _escape_text(str(name))
                + '</span></td><td colspan="2">unavailable</td></tr>'
            )
            continue
        bits: list[str] = []
        arms = entry.get("arms")
        if type(arms) is dict:
            for arm_name in arm_names:
                arm = arms.get(arm_name)
                if type(arm) is not dict:
                    continue
                bits.append(
                    "<div>"
                    + _escape_text(arm_name)
                    + " pass rate "
                    + _rate(arm.get("rate"))
                    + " ("
                    + _count(arm.get("n_pass"))
                    + " of "
                    + _count(arm.get("n_completed"))
                    + " completed), "
                    + _count(arm.get("n_context_exceeded"))
                    + " context-exceeded, "
                    + _count(arm.get("n_evaluation_failed"))
                    + " evaluation-failed.</div>"
                )
            for arm_name in arm_names:
                arm = arms.get(arm_name)
                if type(arm) is not dict:
                    continue
                decomp = arm.get("decomposition")
                if type(decomp) is not dict:
                    continue
                bits.append(
                    "<div>"
                    + _escape_text(arm_name)
                    + " decomposition of completed cells — exact_pass "
                    + _count(decomp.get("exact_pass"))
                    + ", wrong_value "
                    + _count(decomp.get("wrong_value"))
                    + ", non_value "
                    + _count(decomp.get("non_value"))
                    + ", boundary-only "
                    + _count(decomp.get("boundary_only"))
                    + " (the answer digest still equals the expected "
                    "one; the boundary event alone failed).</div>"
                )
        headline = entry.get("headline")
        if type(headline) is dict:
            by_flip = headline.get("a_by_first_flip")
            flip_bits = ""
            if type(by_flip) is dict:
                flip_bits = (
                    " A by first-flip dose — dose18 "
                    + _count(by_flip.get("dose18"))
                    + ", dose36 "
                    + _count(by_flip.get("dose36"))
                    + ", dose72 "
                    + _count(by_flip.get("dose72"))
                    + "."
                )
            bits.append(
                "<div>Headline over "
                + _count(headline.get("n_population"))
                + " cases (completed in all four dose arms, not "
                "closed-book screened, not ids-identical between "
                "dose0 and dose72): C (dose0 passes) "
                + _count(headline.get("direct_passes"))
                + ", A (dose72 wrong_value or non_value failures "
                "among C) "
                + _count(headline.get("drop_cases"))
                + ", boundary attrition beside A (dose72 "
                "boundary-only failures among C, never counted "
                "in A) "
                + _count(headline.get("boundary_attrition"))
                + ", B (reverse discordants) "
                + _count(headline.get("reverse"))
                + ", decision "
                + _escape_text(str(headline.get("decision", "")))
                + "."
                + flip_bits
                + "</div>"
            )
        shape = entry.get("shape")
        if type(shape) is dict:
            hist = shape.get("first_flip_histogram")
            hist_bits = ""
            if type(hist) is dict:
                hist_bits = (
                    "first-flip histogram over C — dose18 "
                    + _count(hist.get("dose18"))
                    + ", dose36 "
                    + _count(hist.get("dose36"))
                    + ", dose72 "
                    + _count(hist.get("dose72"))
                    + ", never "
                    + _count(hist.get("never"))
                    + "; "
                )
            rates = shape.get("dose_pass_rates")
            rate_bits = ""
            if type(rates) is dict:
                rate_bits = (
                    "per-dose pass rates over the headline "
                    "population — dose0 "
                    + _rate(rates.get("dose0"))
                    + ", dose18 "
                    + _rate(rates.get("dose18"))
                    + ", dose36 "
                    + _rate(rates.get("dose36"))
                    + ", dose72 "
                    + _rate(rates.get("dose72"))
                    + "."
                )
            bits.append(
                "<div>Shape readout (descriptive only — no shape "
                "is claimed or gated): "
                + hist_bits
                + "returns R (fail at dose18 or dose36, pass at "
                "dose72) " + _count(shape.get("returns")) + "; " + rate_bits + "</div>"
            )
        tail = entry.get("tail_contrast")
        if type(tail) is dict:
            rival_bits = ""
            if tail.get("rival_reading") is True:
                rival_bits = (
                    " The two arms' discordance is within noise "
                    "of each other, so the honest reading of this "
                    "panel is word-count loss at 50 percent of "
                    "the eligible words, not interior placement."
                )
            bits.append(
                "<div>Displaced control, tail72 vs dose72 over "
                + _count(tail.get("n_compared"))
                + " compared cases: dose72 fails "
                + _count(tail.get("dose72_fails"))
                + ", tail72 fails "
                + _count(tail.get("tail72_fails"))
                + ", dose72-only failures "
                + _count(tail.get("dose72_only"))
                + ", tail72-only failures "
                + _count(tail.get("tail72_only"))
                + "."
                + rival_bits
                + "</div>"
            )
        ids = entry.get("ids")
        if type(ids) is dict:
            id_bits: list[str] = []
            for arm_name in ("dose18", "dose36", "dose72", "tail72"):
                block = ids.get(arm_name)
                if type(block) is not dict:
                    continue
                deltas = block.get("id_length_delta_by_case")
                delta_text = ", ".join(_count(d) for d in deltas) if type(deltas) is list else "n/a"
                id_bits.append(
                    _escape_text(arm_name)
                    + " id-length deltas (vs dose0, case order): "
                    + delta_text
                    + "; max abs id delta "
                    + _count(block.get("max_abs_id_delta"))
                    + ", ids-identical "
                    + _count(block.get("n_ids_identical"))
                    + ", same-ids verdict flips "
                    + _count(block.get("n_same_ids_flips"))
                )
            if id_bits:
                exclusion_bits = ""
                if type(headline) is dict:
                    exclusion_bits = (
                        " Exclusions from the headline population: "
                        + _count(headline.get("n_ids_identical"))
                        + " ids-identical (dose0 vs dose72), "
                        + _count(headline.get("n_screened"))
                        + " closed-book screened, "
                        + _count(headline.get("n_closed_book_incomplete"))
                        + " closed-book incomplete (kept in), "
                        + _count(headline.get("n_headline_incomplete"))
                        + " incomplete in a headline arm."
                    )
                bits.append("<div>" + "; ".join(id_bits) + "." + exclusion_bits + "</div>")
        shares = entry.get("dropped_shares")
        if type(shares) is dict:
            share_bits: list[str] = []
            for arm_name in ("dose18", "dose36", "dose72", "tail72"):
                block = shares.get(arm_name)
                if type(block) is dict:
                    share_bits.append(
                        _escape_text(arm_name)
                        + " numeral "
                        + _rate(block.get("numeral"))
                        + " / negation "
                        + _rate(block.get("negation"))
                    )
            if share_bits:
                bits.append(
                    "<div>Dropped-set shares by class (expected 0 "
                    "under the screens): " + "; ".join(share_bits) + ".</div>"
                )
        rows.append(
            '<tr><td><span class="mono">'
            + _escape_text(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )

    tail_paragraphs = [
        "<p>Operating characteristics of the frozen rule (C at "
        "least 12, A at least 5, two-sided exact McNemar p below "
        "0.05), by exact enumeration — no Monte Carlo figures. "
        "Stated conditional basis (dose0 pass Bernoulli p0; "
        "dose72 failure given a dose0 pass with probability q; "
        "dose72 pass given a dose0 failure with probability r; "
        "arms conditionally independent): the random labeller "
        "(p0 0.5, q and r 0.5) fires 0.0121 per decision, the "
        "competent no-effect panel (p0 0.9, q and r 0.1) fires "
        "0.0143, and jitter (p0 0.97, q and r 0.02) fires "
        "0.000005; family-wise over a 24-candidate panel that "
        "basis expects 0.34 false labels with probability 0.29 "
        "of at least one. Exchangeable basis (per-case joint: "
        "pass/pass = m - v, pass/fail = fail/pass = v, "
        "fail/fail = 1 - m - v, v the per-direction "
        "discordance): the competent panel at m 0.9 fires "
        "0.0030 at v 0.09, 0.0004 at v 0.05, and 0.000004 at "
        "v 0.02, and the random labeller agrees with the "
        "stated row at 0.0121; family-wise expectation 0.07 "
        "with probability 0.07 of at least one. Power (stated "
        "basis, r 0.1): a 25 percent top-dose drop is detected "
        "0.3969 of the time, a 50 percent drop 0.9601, a 75 "
        "percent drop 0.9996 — graded and threshold-at-72 "
        "shapes are one computation, because the rule reads "
        "only dose0 and dose72. The rule is therefore powered "
        "for large drops only: a below_threshold is weak "
        "evidence, never invariance, and an insufficient is "
        "withheld, never an acquittal. The enumerations are "
        "computed on the population the rule conditions on — "
        "cases completed in all four headline arms and not "
        "screened or ids-identical — so they take completion "
        "as given; per-arm attrition prints beside every rate "
        "above, and closed-book screening only shrinks the "
        "population (stated basis: the competent rate falls "
        "from 0.0143 at 24 cases to 0.0017 at 16; the random "
        "rate from 0.0121 to 0.0072 at 17 and 0.0019 at "
        "14).</p>",
        "<p>What the label attributes: the drop is attributed "
        "to the named word-loss treatment as a bundle — "
        "information loss, prompt shortening, fact-position "
        "shift, and sentence breakage move together under the "
        "scattered operator. At dose72 the fact sentence "
        "shifts earlier by a mean of 36.0 words (hypergeometric "
        "sd 3.01) against exactly 0 under tail72, and an "
        "expected 15.95 of the 16 context sentences are "
        "partially broken at dose72 (0.02 removed whole, 0.02 "
        "intact) against 0 broken under tail72, which instead "
        "relocates the fact to passage-final position. The "
        "tail72 contrast therefore separates scatter-vs-block "
        "jointly with fact position, never either alone; read "
        "any prune_drop_observed with its tail72 row. Two "
        "shape gates were executed and rejected, so no shape "
        "is gated: a returns cap collapses graded 50 percent "
        "power from 0.9601 to 0.0941 under the printed "
        "interpolation (per-dose failure probability linear "
        "in dose), and an intermediate-support requirement "
        "collapses threshold-at-72 power to exactly 0.</p>",
        "<p>Printed limitations: the answer-bearing sentence "
        "is never pruned, so this instrument decides nothing "
        "about trims that can delete the answer itself — real "
        "summarisers can, and that failure mode is out of "
        "scope by construction. Doses are word counts: a "
        "checkpoint whose tokenizer splits the passage's words "
        "unevenly realises a different token dose per family "
        "at the same word dose, and the id-length deltas above "
        "are the printed evidence of that. There is no winner "
        "readout and no selection over this generated panel. "
        "The claim is scoped to this frozen panel, this "
        "checkpoint, and the named seeded interior word-loss "
        "treatment — nothing about truncation, compression, "
        "or summarisation in general.</p>",
    ]
    return (
        '<section class="card" id="prune-sweep">'
        "<h2>Prune sweep (seeded interior word loss from a frozen passage)</h2>"
        "<p>This probe prunes passage words: each generated "
        "case is one user message — a frozen instruction, a "
        "17-sentence passage whose 16 context sentences carry "
        "144 eligible words around a six-word answer-bearing "
        "fact sentence that is never pruned, and a frozen "
        "question — and the treated arms delete the first 18, "
        "36, or 72 words of the case's frozen sha256 drop "
        "order over the eligible positions. The tail72 arm "
        "deletes the same count from the passage tail instead, "
        "and the closed_book arm deletes the passage entirely "
        "as a screen. The scorer is byte-exact conjoined with "
        "boundary presence; every arm is a fixture with the "
        "same id, so request seeds are identical across arms. "
        "Each cell is one greedy generation. Records carry "
        "digests only.</p>"
        "<table><thead><tr><th>Candidate</th><th>Readout</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + "".join(tail_paragraphs)
        + "</section>"
    )


def _prefill_section(result: dict[str, Any]) -> str:
    """Assistant-prefill sensitivity: does the winner survive prefill?

    Every search candidate re-evaluated with benign format-lock prefills
    appended after the generation prompt; the selection rule re-applied
    per prefill. Reports per-candidate honoring rates (did the model
    accept the prefill as its own opening, or restart its turn and
    duplicate it?), the winner-vs-native flips, and the think-channel
    group comparison. Renders nothing when the prefill experiment did
    not run.
    """
    prefill: Any = result.get("prefill")
    if type(prefill) is not dict or not prefill.get("available"):
        return ""
    per_condition = prefill.get("per_condition")
    if type(per_condition) is not dict:
        return ""
    primary_winner = prefill.get("primary_winner")
    primary_decision = prefill.get("primary_decision")
    n_fixtures = prefill.get("n_fixtures")

    def _name(value: Any) -> str:
        """Return the name."""
        return "(none)" if value is None else str(value)

    def _rate(value: Any) -> str:
        """Return the rate."""
        if type(value) in (int, float):
            try:
                return f"{value:.0%}"
            except OverflowError:
                return "n/a"
        return "n/a"

    verdict_lines = []
    for key in sorted(per_condition, key=str):
        condition_id = str(key)
        condition = per_condition[key]
        if type(condition) is not dict:
            continue
        decision = condition.get("decision")
        winner = condition.get("winner")
        line = (
            escape(condition_id)
            + ": <strong>"
            + escape(_name(decision))
            + " / "
            + escape(_name(winner))
            + "</strong> vs primary "
            + escape(_name(primary_decision))
            + " / "
            + escape(_name(primary_winner))
        )
        if (decision, winner) != (primary_decision, primary_winner):
            line += ' <span class="badge fail">differs</span>'
        else:
            line += ' <span class="badge pass">same</span>'
        verdict_lines.append("<p>" + line + "</p>")

    cand_rows = []
    for key in sorted(per_condition, key=str):
        condition_id = str(key)
        condition = per_condition[key]
        if type(condition) is not dict:
            continue
        candidates = condition.get("per_candidate")
        if type(candidates) is not dict:
            continue
        for ckey in sorted(candidates, key=str):
            cand = str(ckey)
            entry = candidates[ckey]
            if type(entry) is not dict:
                continue
            passed = entry.get("pass")
            badge = (
                '<span class="badge pass">pass</span>'
                if passed is True
                else '<span class="badge fail">fail</span>'
                if passed is False
                else '<span class="badge">n/a</span>'
            )
            think = entry.get("think_channel")
            think_text = "think-channel" if think is True else "plain" if think is False else "n/a"
            cand_rows.append(
                '<tr><td><span class="mono">'
                + escape(cand)
                + '</span></td><td><span class="mono">'
                + escape(str(condition_id))
                + "</span></td><td>"
                + escape(_rate(entry.get("honor_rate")))
                + "</td><td>"
                + escape(_rate(entry.get("restart_rate")))
                + "</td><td>"
                + escape(think_text)
                + "</td><td>"
                + badge
                + "</td></tr>"
            )

    flips = prefill.get("flips")
    flip_lines = []
    if type(flips) is list:
        for entry in flips:
            if type(entry) is not dict:
                continue
            flip_lines.append(
                '<li><span class="mono">'
                + escape(str(entry.get("prefill")))
                + "</span>: "
                + escape(_name(entry.get("decision_from")))
                + " / "
                + escape(_name(entry.get("winner_from")))
                + " &rarr; "
                + escape(_name(entry.get("decision_to")))
                + " / "
                + escape(_name(entry.get("winner_to")))
                + "</li>"
            )

    think_effect = prefill.get("think_effect")
    think_rows = []
    think_note = ""

    def _as_rate(value: Any) -> float | None:
        # Strict: bools are excluded (type() check, not isinstance()).
        """Return the as rate."""
        if type(value) in (int, float):
            try:
                return float(value)
            except OverflowError:
                return None
        return None

    if type(think_effect) is dict:
        think_rate: float | None = None
        plain_rate: float | None = None
        for label in ("think_channel", "plain"):
            group = think_effect.get(label)
            if type(group) is not dict:
                continue
            rate = _as_rate(group.get("mean_honor_rate"))
            if label == "think_channel":
                think_rate = rate
            else:
                plain_rate = rate
            think_rows.append(
                "<tr><td>"
                + escape(str(label))
                + "</td><td>"
                + escape(str(group.get("n_candidates")))
                + "</td><td>"
                + escape(_rate(rate))
                + "</td></tr>"
            )
        if think_rate is not None and plain_rate is not None:
            relation = (
                "lower than"
                if think_rate < plain_rate
                else "higher than"
                if think_rate > plain_rate
                else "equal to"
            )
            think_note = (
                "<p>Think-channel candidates average "
                + escape(_rate(think_rate))
                + " honoring vs "
                + escape(_rate(plain_rate))
                + " for plain candidates ("
                + relation
                + " -- a correlation readout over the "
                "generation-prompt structure, not a causal claim).</p>"
            )

    n_text = f" ({n_fixtures} fixtures per prefill)" if type(n_fixtures) is int else ""
    return (
        '<section class="card" id="prefill-sensitivity"><h2>Assistant-prefill sensitivity</h2>'
        "<p>Every search candidate re-evaluated with benign format-lock "
        "prefills appended after the generation prompt (the model "
        "generates the continuation; the scored span is prefill + "
        "generated text); the selection rule is re-applied per prefill. "
        "A model <em>honors</em> the prefill when it accepts it as its "
        "own opening and continues; it <em>restarts</em> when it ignores "
        "the prefill and re-emits its preferred opener, duplicating the "
        "prefill text."
        + escape(n_text)
        + "</p>"
        + "".join(verdict_lines)
        + (
            "<ul>" + "".join(flip_lines) + "</ul>"
            if flip_lines
            else "<p>No winner flips against the primary run.</p>"
        )
        + (
            '<div class="table-wrap"><table><caption>Per-candidate honoring '
            "by prefill.</caption>"
            '<thead><tr><th scope="col">Candidate</th>'
            '<th scope="col">Prefill</th><th scope="col">Honored</th>'
            '<th scope="col">Restarted</th><th scope="col">Prompt</th>'
            '<th scope="col">Pass</th>'
            "</tr></thead><tbody>" + "".join(cand_rows) + "</tbody></table></div>"
            if cand_rows
            else ""
        )
        + (
            '<div class="table-wrap"><table><caption>Honoring by '
            "generation-prompt structure (correlation only).</caption>"
            '<thead><tr><th scope="col">Group</th>'
            '<th scope="col">Candidates</th><th scope="col">Mean honor rate</th>'
            "</tr></thead><tbody>" + "".join(think_rows) + "</tbody></table></div>" + think_note
            if think_rows
            else ""
        )
        + "</section>"
    )
