"""reporting forms family: split from reporting.py (W7-2).

Facades keep the public surface: import from model_protocol_lab.reporting, not from this module.
"""

from html import escape
from typing import Any

from .reporting_core import (
    _escape_text,
    _float_convertible,
)


def _position_section(result: dict[str, Any]) -> str:
    """Position sensitivity: does option placement change the winner?

    Every search candidate re-evaluated on synthetic multiple-choice
    fixtures whose correct option is permuted between the first, middle,
    and last positions (seeded distractor order); the selection rule
    re-applied per condition. Reports per-condition winner verdicts,
    the per-candidate primacy index (P(correct | first) minus P(correct
    | last)), the winner flips against the primary run, and the
    condition effect. Renders nothing when the position probe did not
    run.
    """
    position: Any = result.get("position")
    if type(position) is not dict or not position.get("available"):
        return ""
    per_condition = position.get("per_condition")
    if type(per_condition) is not dict:
        return ""
    primary_winner = position.get("primary_winner")
    primary_decision = position.get("primary_decision")
    n_fixtures = position.get("n_fixtures")

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

    def _index(value: Any) -> str:
        """Return the index."""
        if type(value) in (int, float):
            try:
                return f"{value:+.2f}"
            except OverflowError:
                return "n/a"
        return "n/a"

    verdict_lines = []
    for key in sorted(per_condition, key=str):
        condition = str(key)
        verdict = per_condition[key]
        if type(verdict) is not dict:
            continue
        decision = verdict.get("decision")
        winner = verdict.get("winner")
        line = (
            escape(condition)
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
        condition = str(key)
        verdict = per_condition[key]
        if type(verdict) is not dict:
            continue
        candidates = verdict.get("per_candidate")
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
                + escape(condition)
                + "</span></td><td>"
                + escape(_rate(entry.get("correct_rate")))
                + "</td><td>"
                + badge
                + "</td></tr>"
            )

    index_rows = []
    primacy = position.get("primacy_index")
    if type(primacy) is dict:
        for ckey in sorted(primacy, key=str):
            cand = str(ckey)
            index_rows.append(
                '<tr><td><span class="mono">'
                + escape(cand)
                + "</span></td><td>"
                + escape(_index(primacy[ckey]))
                + "</td></tr>"
            )

    effect = position.get("condition_effect")
    effect_line = ""
    if type(effect) is dict:
        mean = effect.get("mean_index")
        mean_text = _index(mean) if type(mean) in (int, float) else "n/a"
        n_pos = effect.get("n_positive")
        n_neg = effect.get("n_negative")
        n_cand = effect.get("n_candidates")
        effect_line = (
            "<p>Condition effect: mean primacy index <strong>"
            + escape(mean_text)
            + "</strong> over "
            + escape(str(n_cand))
            + " evaluable candidates ("
            + escape(str(n_pos))
            + " positive, "
            + escape(str(n_neg))
            + " negative). Positive means more correct answers when the "
            "correct option is listed first; negative the reverse.</p>"
        )

    flips = position.get("flips")
    flip_lines = []
    if type(flips) is list:
        for entry in flips:
            if type(entry) is not dict:
                continue
            flip_lines.append(
                '<li><span class="mono">'
                + escape(str(entry.get("condition")))
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

    n_text = f" ({n_fixtures} fixtures per condition)" if type(n_fixtures) is int else ""
    return (
        '<section class="card" id="position-sensitivity"><h2>Position sensitivity</h2>'
        "<p>Every search candidate re-evaluated on synthetic multiple-choice "
        "fixtures whose correct option is permuted between the first, middle, "
        "and last positions (seeded distractor order); the selection rule is "
        "re-applied per condition. The primacy index is P(correct | first) "
        "minus P(correct | last), in [-1, 1]; a nonzero index means option "
        "placement moves the task verdict."
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
            '<div class="table-wrap"><table><caption>Per-candidate correct '
            "rates by condition.</caption>"
            '<thead><tr><th scope="col">Candidate</th>'
            '<th scope="col">Condition</th><th scope="col">Correct</th>'
            '<th scope="col">Pass</th>'
            "</tr></thead><tbody>" + "".join(cand_rows) + "</tbody></table></div>"
            if cand_rows
            else ""
        )
        + (
            '<div class="table-wrap"><table><caption>Primacy index per '
            "candidate.</caption>"
            '<thead><tr><th scope="col">Candidate</th>'
            '<th scope="col">Primacy index</th>'
            "</tr></thead><tbody>" + "".join(index_rows) + "</tbody></table></div>"
            if index_rows
            else ""
        )
        + "</section>"
    )


def _leading_section(result: dict[str, Any]) -> str:
    """Leading-claim caving probe: claim agreement per template.

    Renders the ``leading`` payload (see
    experiments.leading_analysis): per candidate and framing,
    the caving counts over the native-passing paired population
    (adopted vs failed-otherwise), the discordance
    classification, and both claim arms' paired rates.
    Withheld condition selections lead the section: a flip
    computed over mismatched completed sets is not published.
    Adoption is a behavioural equality at this tier -- the
    recorded answer is the planted string -- and the section
    says so. Every interpolation is escaped at this boundary;
    no claim, planted, or answer text is ever rendered.
    """
    probe: Any = result.get("leading")
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
        framings = entry.get("framings")
        framing_bits: list[str] = []
        if type(framings) is dict:
            for framing in ("first_person", "third_person"):
                block = framings.get(framing)
                if type(block) is not dict:
                    continue
                caving = block.get("caving")
                if type(caving) is not dict:
                    continue
                wrong_arm = block.get("wrong_arm")
                right_arm = block.get("right_arm")
                framing_bits.append(
                    "<div>"
                    + escape(framing)
                    + " — caved "
                    + _count(caving.get("n_caved"))
                    + "/"
                    + _count(caving.get("n_paired"))
                    + " ("
                    + _rate(caving.get("caving_rate"))
                    + "), adopted "
                    + _count(caving.get("n_adopted"))
                    + " (boundary-found "
                    + _count(caving.get("n_adopted_boundary_found"))
                    + "), failed otherwise "
                    + _count(caving.get("n_failed_otherwise"))
                    + ", right-only "
                    + _count(caving.get("n_right_only"))
                    + " · "
                    + escape(str(caving.get("classification")))
                    + " · wrong-arm pass "
                    + (
                        _count(wrong_arm.get("n_pass")) + "/" + _count(wrong_arm.get("n_paired"))
                        if type(wrong_arm) is dict
                        else "n/a"
                    )
                    + ", right-arm pass "
                    + (
                        _count(right_arm.get("n_pass")) + "/" + _count(right_arm.get("n_paired"))
                        if type(right_arm) is dict
                        else "n/a"
                    )
                    + "</div>"
                )
        contrast = entry.get("framing_contrast")
        contrast_bit = ""
        if type(contrast) is dict:
            contrast_bit = "caving-rate delta (first − third person): " + _rate(
                contrast.get("caving_rate_delta")
            )
        rows.append(
            '<tr><td><span class="mono">'
            + escape(str(name))
            + "</span></td><td>"
            + "".join(framing_bits)
            + "</td><td>"
            + (contrast_bit or "n/a")
            + "</td><td>"
            + _count(entry.get("n_wrong_defined"))
            + " of "
            + _count(entry.get("n_fixtures"))
            + "</td><td>"
            + _count(entry.get("n_echo"))
            + "</td><td>"
            + _count(entry.get("n_vacuous"))
            + "</td></tr>"
        )
    common = probe.get("common_population")
    common_bits: list[str] = []
    if type(common) is dict:
        for framing in ("first_person", "third_person"):
            block = common.get(framing)
            if type(block) is not dict:
                continue
            per_candidate = block.get("per_candidate")
            counts = []
            if type(per_candidate) is dict:
                for cand_name in sorted(per_candidate, key=str):
                    cand = per_candidate[cand_name]
                    if type(cand) is dict:
                        counts.append(
                            escape(str(cand_name))
                            + " caved "
                            + _count(cand.get("n_caved"))
                            + " ("
                            + _rate(cand.get("caving_rate"))
                            + ")"
                        )
            common_bits.append(
                "<div>"
                + escape(framing)
                + " — common population "
                + _count(block.get("n_common"))
                + (" (" + escape(str(block.get("detail"))) + ")" if block.get("detail") else "")
                + (": " + "; ".join(counts) if counts else "")
                + "</div>"
            )
    return (
        '<section class="card" id="leading">'
        "<h2>Leading-claim caving</h2>"
        "<p>Does a confident wrong claim in the user's own turn "
        "move the answer? Each fixture's last user message gained "
        "one frozen claim sentence — a wrong claim (another "
        "fixture's answer from a different group, or an "
        "off-by-one plant of the fixture's own) or a control "
        "asserting the true answer — in first-person and "
        "third-person framings, against the recorded native run. "
        "A fixture <strong>caves</strong> when the candidate "
        "passes it natively and under the right claim but fails "
        "it under the wrong one; the failure splits into "
        "<em>adopted</em> (the recorded answer is byte-exactly "
        "the planted string — a behavioural equality that cannot "
        "distinguish believing the claim from copying the "
        "asserted string) and <em>failed otherwise</em> "
        "(instability, never counted as caving). The "
        "classification compares discordant pairs only, with the "
        "McNemar p reported as a diagnostic, never thresholded; "
        "balanced discordance is reported as claim-following, "
        "not caving. This measures factual caving on short "
        "exact-match answers — not social sycophancy. Counts are "
        "per framing over paired denominators; framings are "
        "never pooled.</p>"
        + withheld_block
        + ("<ul>" + "".join(condition_lines) + "</ul>" if condition_lines else "")
        + "<table><thead><tr><th>Candidate</th><th>Caving by framing "
        "(caved/paired, classification, arm passes)</th>"
        "<th>Framing contrast</th><th>Wrong claim defined</th>"
        "<th>Echo fixtures</th><th>Vacuous</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + (
            "<h3>Common population</h3><p>The only ranked "
            "cross-candidate comparison: fixtures in every "
            "available candidate's caving population.</p>" + "".join(common_bits)
            if common_bits
            else ""
        )
        + "</section>"
    )


def _numform_section(result: dict[str, Any]) -> str:
    """Numeric surface-form probe: written form per template.

    Renders the ``numform`` payload (see
    experiments.numform_analysis): per candidate, task, and
    form, exact pass counts with the value/form decomposition,
    and the paired form contrasts over the values completed in
    both forms. The scope is stated in the quantity's
    definition, not a footnote: generated integer tasks,
    isolated operands, one frozen zero-shot wording, this
    checkpoint. The tokenizer grouping is a run-level label
    (one backend per run), never a within-run correlation, and
    an ``unknown`` classification suppresses every grouping
    sentence beyond the label itself. Every interpolation is
    escaped at this boundary.
    """
    probe: Any = result.get("numform")
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

    grouping = probe.get("grouping")
    grouping_line = ""
    if type(grouping) is dict:
        classification = grouping.get("classification")
        if classification == "unknown":
            grouping_line = (
                "<p>Tokenizer digit grouping: <strong>unknown</strong> "
                "-- the probe battery's counts match no frozen class, "
                "so no grouping reading is offered; the per-form "
                "counts below do not depend on the label.</p>"
            )
        elif type(classification) is str and classification:
            grouping_line = (
                "<p>Tokenizer digit grouping (measured on the "
                "attached tokenizer, one label for the whole run): "
                f"<strong>{escape(classification)}</strong>. The "
                "label describes this run only; a grouping claim "
                "across tokenizers needs a cross-run diff of "
                "different checkpoints, which confounds weights "
                "with tokenizer.</p>"
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
                + '</span></td><td colspan="2">unavailable</td></tr>'
            )
            continue
        aggregates = entry.get("aggregates")
        contrasts = entry.get("contrasts")
        task_bits: list[str] = []
        if type(aggregates) is dict:
            for task in ("copy", "compare", "add"):
                forms = aggregates.get(task)
                if type(forms) is not dict:
                    continue
                form_bits: list[str] = []
                for form in ("plain", "comma", "space", "scientific"):
                    agg = forms.get(form)
                    if type(agg) is not dict:
                        continue
                    decomposition = agg.get("decomposition")
                    decomp_bit = ""
                    if type(decomposition) is dict and decomposition:
                        if task == "compare":
                            decomp_bit = (
                                " (wrong label "
                                + _count(decomposition.get("wrong_label"))
                                + ", non-label "
                                + _count(decomposition.get("non_label"))
                                + ")"
                            )
                        else:
                            decomp_bit = (
                                " (right value, wrong form "
                                + _count(decomposition.get("value_pass_form_fail"))
                                + ", wrong value "
                                + _count(decomposition.get("value_fail"))
                                + ")"
                            )
                    form_bits.append(
                        escape(form)
                        + " "
                        + _count(agg.get("n_exact"))
                        + "/"
                        + _count(agg.get("n_completed"))
                        + " ("
                        + _rate(agg.get("exact_rate"))
                        + ")"
                        + decomp_bit
                    )
                contrast_bits: list[str] = []
                if type(contrasts) is dict:
                    task_contrasts = contrasts.get(task)
                    if type(task_contrasts) is dict:
                        for pair in sorted(task_contrasts, key=str):
                            block = task_contrasts[pair]
                            if type(block) is not dict:
                                continue
                            contrast_bits.append(
                                escape(pair.replace("__", " vs "))
                                + ": "
                                + escape(str(block.get("classification")))
                                + " (paired "
                                + _count(block.get("n_paired"))
                                + ", discordant "
                                + _count(block.get("n_pass_a_only"))
                                + "/"
                                + _count(block.get("n_pass_b_only"))
                                + ")"
                            )
                task_bits.append(
                    "<div><strong>"
                    + escape(task)
                    + "</strong> — "
                    + "; ".join(form_bits)
                    + ("<br>contrasts: " + "; ".join(contrast_bits) if contrast_bits else "")
                    + "</div>"
                )
        rows.append(
            '<tr><td><span class="mono">'
            + escape(str(name))
            + "</span></td><td>"
            + "".join(task_bits)
            + "</td></tr>"
        )
    return (
        '<section class="card" id="numform">'
        "<h2>Numeric surface form</h2>"
        "<p>The same generated integers, written four ways -- "
        "plain, comma-grouped (3-3-3), space-grouped (U+202F), "
        "scientific -- across copy, compare, and add tasks, "
        "with the expected answer in the operands' form (for "
        "compare the answer is the word <em>first</em> or "
        "<em>second</em>, the one arm where the scorer cannot "
        "be a form confound). A pass is the right value, in "
        "the requested form, with the response boundary found; "
        "the decomposition separates <em>right value, wrong "
        "form</em> from <em>wrong value</em>, and only the "
        "wrong-value class licenses an arithmetic reading. "
        "Rates are per task and never pooled; the headline is "
        "the paired contrast over the values completed in both "
        "forms, and a contrast of <em>no_difference_observed</em> "
        "across the board is a publishable result: "
        "normalisation does not matter here. Generated integer "
        "tasks with isolated operands, one frozen zero-shot "
        "wording, this checkpoint -- a probe, not a numeracy "
        "benchmark, and no selection is derived from the "
        "synthetic set.</p>"
        + grouping_line
        + "<table><thead><tr><th>Candidate</th><th>Exact passes "
        "by task and form, with paired contrasts</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + "</section>"
    )


def _dateline_section(result: dict[str, Any]) -> str:
    """Date-line sensitivity probe: the injected date per template.

    Renders the ``dateline`` payload (see
    experiments.dateline_analysis): per candidate, the
    date-attributable loss per frozen date value against the
    length-matched no-date control, the anchor's weekday-lie
    split, and the pure-date discordant pairs (same format, two
    date values) that carry the headline. The construct is named
    in the prose: a date line immediately before the final user
    message, which is the head of the conversation only for
    single-turn fixtures. Every interpolation is escaped at this
    boundary.
    """
    probe: Any = result.get("dateline")
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

    n_search = probe.get("n_search_fixtures")
    n_multi = probe.get("n_multi_turn_fixtures")
    multi_line = ""
    if type(n_search) is int and type(n_multi) is int and n_search:
        multi_line = (
            "<p>Multi-turn share of the search set: "
            + _count(n_multi)
            + "/"
            + _count(n_search)
            + " fixtures have more than one user turn; for those, "
            "the block sits immediately before the final user "
            "message, not at the head of the conversation.</p>"
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
                + '</span></td><td colspan="2">unavailable</td></tr>'
            )
            continue
        bits: list[str] = []
        if entry.get("date_content_falsified"):
            bits.append(
                "<div><strong>Date content falsified on this "
                "fixture set</strong>: every pure-date pair's "
                "discordant count is zero and every date value's "
                "loss count is zero.</div>"
            )
        loss = entry.get("date_loss")
        if type(loss) is dict:
            loss_bits: list[str] = []
            for value in sorted(loss, key=str):
                block = loss[value]
                if type(block) is not dict:
                    continue
                loss_bits.append(
                    escape(value)
                    + " "
                    + _count(block.get("n_loss"))
                    + "/"
                    + _count(block.get("n_evaluable"))
                    + " ("
                    + _rate(block.get("loss_rate"))
                    + ", counterpart "
                    + _count(block.get("n_counterpart"))
                    + ")"
                )
                lie_split = block.get("lie_split")
                if type(lie_split) is dict:
                    loss_bits.append(
                        "anchor lie split: any-date-text "
                        + _count(lie_split.get("n_loss_any_text"))
                        + ", content-specific "
                        + _count(lie_split.get("n_loss_content_specific"))
                        + " of "
                        + _count(lie_split.get("n_evaluable"))
                    )
            if loss_bits:
                bits.append(
                    "<div><strong>Date-attributable loss</strong> "
                    "(pass the no-date control, fail the date value)"
                    " — " + "; ".join(loss_bits) + "</div>"
                )
        pairs = entry.get("pairs_by_format")
        if type(pairs) is dict:
            pair_bits: list[str] = []
            for fmt in sorted(pairs, key=str):
                table = pairs[fmt]
                if type(table) is not dict:
                    continue
                for pair in sorted(table, key=str):
                    block = table[pair]
                    if type(block) is not dict:
                        continue
                    discordant = 0
                    if type(block.get("n_a_only")) is int and type(block.get("n_b_only")) is int:
                        discordant = block["n_a_only"] + block["n_b_only"]
                    if discordant:
                        pair_bits.append(
                            escape(fmt)
                            + " "
                            + escape(pair.replace("|", " vs "))
                            + ": "
                            + escape(str(block.get("classification")))
                            + " (paired "
                            + _count(block.get("n_paired"))
                            + ", discordant "
                            + _count(block.get("n_a_only"))
                            + "/"
                            + _count(block.get("n_b_only"))
                            + ")"
                        )
            if pair_bits:
                bits.append(
                    "<div><strong>Pure-date discordant pairs</strong>"
                    " — " + "; ".join(pair_bits) + "</div>"
                )
        control = entry.get("control_evaluable")
        if type(control) is dict:
            control_bits = []
            for fmt in sorted(control, key=str):
                block = control[fmt]
                if type(block) is dict:
                    control_bits.append(
                        escape(fmt)
                        + " "
                        + _count(block.get("n_evaluable"))
                        + "/"
                        + _count(block.get("n_live"))
                    )
            if control_bits:
                bits.append(
                    "<div>Control length-match evaluable: "
                    + "; ".join(control_bits)
                    + ". Screened fixtures: "
                    + _count(entry.get("n_screened"))
                    + "; message-date-bearing: "
                    + _count(entry.get("n_message_date_bearing"))
                    + ".</div>"
                )
        rows.append(
            '<tr><td><span class="mono">'
            + escape(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )
    return (
        '<section class="card" id="dateline">'
        "<h2>Date-line sensitivity</h2>"
        "<p>A frozen <em>[Context] Today's date is ... "
        "[/Context]</em> block prepended to each search fixture's "
        "last user message -- a date line immediately before the "
        "final user message -- across eight frozen date values "
        "(anchor 2026-10-21, &plusmn;1 day, &plusmn;1 year, leap "
        "day 2028-02-29, a year-boundary pair) in three formats "
        "(ISO, long, weekday+ISO), against a no-date control whose "
        "neutral text is sized to the anchor arm's rendered length "
        "and one weekday-lie arm (the anchor with its weekday name "
        "shifted a day). Dates are protocol constants, never the "
        "wall clock, so the run is reproducible on any day. The "
        "headline is the per-pair discordant table within a format "
        "-- not an any-pair flip rate, which would saturate under "
        "generic fragility -- and the date-attributable loss "
        "subset, printed beside its asymmetric counterpart. "
        "Fixtures whose expected answer is itself date-derived "
        "are screened out and counted, since the date line would "
        "hand the model the string to copy.</p>"
        + multi_line
        + "<table><thead><tr><th>Candidate</th><th>Date loss, lie "
        "split, and discordant pairs</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + "</section>"
    )


def _rotation_section(result: dict[str, Any]) -> str:
    """Context rotation probe: does cyclically rotating the block span move the verdict?

    Renders the ``rotation`` payload (see
    experiments.rotation_analysis): per candidate, the
    headline invariance block over token-exact rotations, the
    pass-by-absolute-slot curve, the mirror control block, the
    reversal discordance, the fact-absent false-recall floor,
    the screening accounting, and the per-arm selection
    winners. The scope limits are printed with the numbers,
    not under them: the construction classification is a token
    inventory claim (segmentation at the seams is not
    verified), the slot curve is a cross-check against the
    needle probe's depth curve rather than a second
    measurement of it, the mirror control compares the same
    any-flip statistic on matched slot multisets, and each
    cell is one greedy generation on generated fixtures.
    Counts and rates only; no block, question, or answer text
    is rendered. Every interpolation is escaped at this
    boundary.
    """
    probe: Any = result.get("rotation")
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
                return _escape_text(f"{float(value):.3f}")
            except OverflowError:
                return "n/a"
        return "n/a"

    def _count(value: Any) -> str:
        """Return the count."""
        return _escape_text(str(value)) if type(value) is int else "n/a"

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
        headline = entry.get("headline")
        if type(headline) is dict:
            bits.append(
                "<div>Headline cases "
                + _count(headline.get("n_headline"))
                + " (evaluable share "
                + _rate(headline.get("evaluable_share"))
                + "): invariant "
                + _count(headline.get("n_invariant"))
                + " (all-pass "
                + _count(headline.get("n_all_pass"))
                + ", all-fail "
                + _count(headline.get("n_all_fail"))
                + "), flips "
                + _count(headline.get("n_flip_any"))
                + ", invariance rate "
                + _rate(headline.get("invariance_rate"))
                + " ("
                + _escape_text(str(entry.get("classification", "")))
                + "); invariance falsified: "
                + ("yes" if entry.get("invariance_falsified") is True else "no")
                + ". Token-exact share of completed rotation cells "
                + _rate(headline.get("id_exact_share"))
                + ".</div>"
            )
            bits.append(
                "<div>Flips from an r=0 pass: "
                + _count(headline.get("n_flip_from_pass"))
                + " of "
                + _count(headline.get("n_r0_pass"))
                + " ("
                + _rate(headline.get("flip_rate_from_pass"))
                + ").</div>"
            )
        curve = entry.get("slot_curve")
        if type(curve) is dict and type(curve.get("slots")) is dict:
            slots = curve["slots"]
            bits.append(
                "<div>Passes by the asked fact's absolute slot: "
                + ", ".join("slot " + str(s) + " " + _count(slots.get(str(s))) for s in range(5))
                + " of "
                + _count(curve.get("n_headline"))
                + " headline cases each.</div>"
            )
        mirror = entry.get("mirror")
        if type(mirror) is dict:
            bits.append(
                "<div>Mirror control over "
                + _count(mirror.get("n_paired"))
                + " paired cases: rotation flips "
                + _count(mirror.get("n_flips_paired"))
                + ", mirror flips "
                + _count(mirror.get("n_flips_mirror"))
                + " (the same any-flip statistic over the "
                "non-cyclic control set, on the same cases).</div>"
            )
        reversal = entry.get("reversal")
        if type(reversal) is dict:
            bits.append(
                "<div>Reversal discordance: "
                + _count(reversal.get("n_discordant"))
                + " of "
                + _count(reversal.get("n_paired"))
                + " ("
                + _rate(reversal.get("discordance_rate"))
                + ").</div>"
            )
        absent = entry.get("absent")
        if type(absent) is dict:
            bits.append(
                "<div>Fact-absent false recall: "
                + _count(absent.get("n_false_pass"))
                + " of "
                + _count(absent.get("n_evaluable"))
                + " ("
                + _rate(absent.get("rate"))
                + ").</div>"
            )
        closed = entry.get("closed_book")
        if type(closed) is dict:
            bits.append(
                "<div>Screens: "
                + _count(entry.get("n_screened"))
                + " cases excluded by the closed-book screen "
                "(closed-book pass rate "
                + _rate(closed.get("pass_rate"))
                + "), "
                + _count(entry.get("n_boundary_screened"))
                + " cases boundary-screened, "
                + _count(entry.get("n_closed_book_incomplete"))
                + " cases with no closed-book verdict.</div>"
            )
        rows.append(
            '<tr><td><span class="mono">'
            + _escape_text(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )
    winners: list[str] = []
    for arm in ("rot_0", "rot_1", "rot_2", "rot_3", "rot_4", "mir_4"):
        cond = conditions.get(arm)
        if type(cond) is not dict:
            continue
        label = "reversal" if arm == "mir_4" else arm.replace("_", " ")
        if cond.get("withheld"):
            winners.append(_escape_text(label) + ": selection withheld")
        else:
            winner = cond.get("winner")
            winners.append(
                _escape_text(label)
                + ": "
                + (_escape_text(str(winner)) if winner else "no unique winner")
                + (" (flip vs rot 0)" if cond.get("flip") is True else "")
            )
    return (
        '<section class="card" id="rotation">'
        "<h2>Context rotation</h2>"
        "<p>Twenty generated cases each assemble five "
        "fact-bearing blocks (one asked fact, four fact-shaped "
        "distractors) into a span with the question pinned "
        "after it. The span is cyclically rotated through all "
        "five orders -- same content, same total text length, "
        "same cyclic relative order -- and four mirror arms "
        "rotate the reversed order the same way, forming a "
        "generic-reordering control set that places the asked "
        "fact at the same slots. A span arm counts as "
        "construction-exact only when its rendered token "
        "inventory matches the unrotated arm's exactly; that is "
        "an inventory claim, and segmentation at the block "
        "seams is not verified. The headline is verdict "
        "invariance over the token-exact rotations; the "
        "pass-by-slot curve is a cross-check against the needle "
        "probe's depth curve, not a second measurement of it. "
        "A fact-absent arm prices false recall, and a "
        "closed-book screen (question alone; a pass excludes "
        "the case for that candidate) keeps ungrounded answers "
        "out of the populations. Each cell is one greedy "
        "generation.</p>"
        "<table><thead><tr><th>Candidate</th><th>Readout</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + ("<p>Per-arm selection: " + "; ".join(winners) + ".</p>" if winners else "")
        + "</section>"
    )


def _density_section(result: dict[str, Any]) -> str:
    """Instruction-density ladder: which instructions does a growing stack drop?

    Renders the ``density`` payload (see
    experiments.density_analysis): per candidate, satisfaction
    by stack depth with the solo composition control beside
    each point, task-value presence, position bands at full
    density, per-kind solo-vs-full rates, joint satisfaction,
    the length-matched sham readout, the screen accounting,
    and the pre-registered endpoint decision. The scope limits
    are printed with the numbers, not under them: the
    constraint sets are jointly satisfiable by construction
    (a perfect follower passes every arm), satisfaction is
    derived from the answers' word-hash commitments and never
    stored, the decision is scoped to the solo-vs-full
    endpoint contrast (intermediate depths are descriptive),
    positional-kind demands repeat their target at full
    density, and each record is one greedy generation on
    generated cases. Counts and rates only; no case,
    constraint, or answer text is rendered. Every
    interpolation is escaped at this boundary.
    """
    probe: Any = result.get("density")
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

    def _pvalue(value: Any) -> str:
        """Return the pvalue."""
        if type(value) in (int, float) and not isinstance(value, bool):
            try:
                return escape(f"{float(value):.3g}")
            except OverflowError:
                return "n/a"
        return "n/a"

    def _mean(value: Any) -> str:
        """Return the mean."""
        if type(value) in (int, float) and not isinstance(value, bool):
            try:
                return escape(f"{float(value):.1f}")
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
        bits.append(
            "<div>Records: "
            + _count(entry.get("n_generated"))
            + " generated, "
            + _count(entry.get("n_boundary_screened"))
            + " boundary-screened, "
            + _count(entry.get("n_detail_screened"))
            + " detail-screened, "
            + _count(entry.get("n_no_boundary"))
            + " without a boundary, "
            + _count(entry.get("n_zero_word"))
            + " live with no words, "
            + _count(entry.get("n_live"))
            + " live.</div>"
        )
        decision = entry.get("decision")
        if type(decision) is dict:
            verdict = str(decision.get("verdict", ""))
            if verdict == "insufficient":
                decision_text = "withheld (insufficient: fewer than six discordant cases)"
            elif decision.get("falsified") is True:
                decision_text = "density falsified"
            else:
                decision_text = "not falsified"
            bits.append(
                "<div>Endpoint decision (solo vs full density): "
                + escape(decision_text)
                + " -- discordant pairs f "
                + _count(decision.get("f"))
                + " (solo passes, full fails), p "
                + _count(decision.get("p"))
                + " (reverse), over "
                + _count(decision.get("discordant_cases"))
                + " discordant cases; exact sign test p "
                + _pvalue(decision.get("sign_p"))
                + ".</div>"
            )
        by_density = entry.get("by_density")
        if type(by_density) is list:
            points: list[str] = []
            for row in by_density:
                if type(row) is not dict:
                    continue
                points.append(
                    "K="
                    + _count(row.get("density"))
                    + ": "
                    + _rate(row.get("rate"))
                    + " ("
                    + _count(row.get("satisfied"))
                    + "/"
                    + _count(row.get("checked"))
                    + "; solo control "
                    + _rate(row.get("solo_rate"))
                    + ", "
                    + _count(row.get("solo_satisfied"))
                    + "/"
                    + _count(row.get("solo_checked"))
                    + ")"
                )
            if points:
                bits.append(
                    "<div>Satisfaction by stack depth (the raw "
                    "curve is not composition-controlled; the "
                    "solo control beside each point is): " + "; ".join(points) + ".</div>"
                )
        joint = entry.get("joint_k16")
        if type(joint) is dict:
            bits.append(
                "<div>Joint satisfaction at full density (all "
                "16 constraints at once): "
                + _count(joint.get("satisfied"))
                + " of "
                + _count(joint.get("live"))
                + " ("
                + _rate(joint.get("rate"))
                + ").</div>"
            )
        bands = entry.get("position_bands")
        if type(bands) is list:
            band_bits: list[str] = []
            for band in bands:
                if type(band) is not dict:
                    continue
                band_bits.append(
                    escape(str(band.get("band", "")))
                    + " "
                    + _rate(band.get("rate"))
                    + " ("
                    + _count(band.get("satisfied"))
                    + "/"
                    + _count(band.get("checked"))
                    + ")"
                )
            if band_bits:
                bits.append(
                    "<div>Position bands at full density "
                    "(positions 1-5 early, 6-11 mid, 12-16 late): "
                    + ", ".join(band_bits)
                    + ".</div>"
                )
        by_kind = entry.get("by_kind")
        if type(by_kind) is list:
            kind_bits: list[str] = []
            for row in by_kind:
                if type(row) is not dict:
                    continue
                kind_bits.append(
                    escape(str(row.get("kind", "")))
                    + " solo "
                    + _rate(row.get("solo_rate"))
                    + " ("
                    + _count(row.get("solo_satisfied"))
                    + "/"
                    + _count(row.get("solo_checked"))
                    + "), full "
                    + _rate(row.get("k16_rate"))
                    + " ("
                    + _count(row.get("k16_satisfied"))
                    + "/"
                    + _count(row.get("k16_checked"))
                    + ")"
                )
            if kind_bits:
                bits.append(
                    "<div>Per kind (positional kinds repeat "
                    "their target at full density): " + "; ".join(kind_bits) + ".</div>"
                )
        presence = entry.get("task_presence")
        if type(presence) is list:
            by_arm = {str(row.get("arm")): row for row in presence if type(row) is dict}
            presence_bits: list[str] = []
            for arm in ("solo_1", "sham", "k16"):
                row = by_arm.get(arm)
                if row is None:
                    continue
                presence_bits.append(
                    escape(arm)
                    + " "
                    + _rate(row.get("rate"))
                    + " ("
                    + _count(row.get("present"))
                    + "/"
                    + _count(row.get("live"))
                    + ")"
                )
            if presence_bits:
                bits.append(
                    "<div>Task value present in the answer: " + ", ".join(presence_bits) + ".</div>"
                )
        sham = entry.get("sham")
        if type(sham) is dict:
            bits.append(
                "<div>Sham (first constraint plus length-matched "
                "neutral text), over "
                + _count(sham.get("cases"))
                + " cases live in all three arms: position-1 "
                "satisfaction solo "
                + _rate(sham.get("solo_1_rate"))
                + ", sham "
                + _rate(sham.get("sham_rate"))
                + ", full "
                + _rate(sham.get("k16_rate"))
                + "; mean block words sham "
                + _mean(sham.get("sham_block_words_mean"))
                + " vs full "
                + _mean(sham.get("k16_block_words_mean"))
                + ".</div>"
            )
        rows.append(
            '<tr><td><span class="mono">'
            + escape(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )
    return (
        '<section class="card" id="density">'
        "<h2>Instruction density</h2>"
        "<p>Twenty generated cases each carry sixteen "
        "simultaneously satisfiable constraints -- include a "
        "word, start with a word, end with a word, stay within "
        "a word limit, four of each, with the start and end "
        "targets shared across the case -- over one small "
        "fact-recall task. Each candidate answers every "
        "constraint alone, then the first 2, 4, 8, and all 16 "
        "stacked, plus a sham arm whose text is length-matched "
        "to the full stack but carries only the first "
        "constraint. Answers are committed as ordered per-word "
        "hash lists and every verdict is derived from the "
        "commitment. The claim under test is scoped to the "
        "endpoint contrast: per-constraint satisfaction at full "
        "density equals its solo satisfaction. The decision "
        "aggregates solo-vs-full discordance per case and "
        "applies an exact sign test; below six discordant "
        "cases the verdict is withheld as insufficient. The "
        "level is per candidate -- across 24 candidates at "
        "alpha 0.05, at least one false falsification is "
        "expected with probability about 0.71. Each record is "
        "one greedy generation.</p>"
        "<table><thead><tr><th>Candidate</th><th>Readout</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>" + "</section>"
    )


def _field_order_section(result: dict[str, Any]) -> str:
    """Schema field order: does the answer field's position change its value?

    Renders the ``field_order`` payload (see
    experiments.fieldorder_analysis): per candidate, adherence
    by arm in the shipped parse/schema decomposition, the
    answer field's accuracy by arm, whole-document accuracy,
    the pairing attrition with the ungated sensitivity counts,
    and the pre-registered endpoint decision with its printed
    operating characteristics. The scope limits are printed
    with the numbers: flat schemas of 1-4 fields, the endpoint
    contrast only, canonical coinciding with an endpoint arm
    on 6 of the 10 paired specs (its numbers there are
    replicates), two shipped specs excluded from pairing
    because their frozen answer labels were not entailed by
    their contexts as originally shipped (the contexts have
    since been repaired so the labels are entailed; the
    pairing population is frozen and unchanged), adherence
    gating selecting against discordance, and duplicate-key
    documents scored by their last occurrence outside the
    pairing. Counts and rates
    only; no spec, schema, or answer text is rendered. Every
    interpolation is escaped at this boundary.
    """
    probe: Any = result.get("field_order")
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

    def _pvalue(value: Any) -> str:
        """Return the pvalue."""
        if type(value) in (int, float) and not isinstance(value, bool):
            try:
                return _escape_text(f"{float(value):.3g}")
            except OverflowError:
                return "n/a"
        return "n/a"

    def _count(value: Any) -> str:
        """Return the count."""
        return _escape_text(str(value)) if type(value) is int else "n/a"

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
        bits.append(
            "<div>Records: "
            + _count(entry.get("n_generated"))
            + " generated ("
            + _count(entry.get("n_distinct_prompts"))
            + " distinct prompts; coincident orders are generated "
            "twice under a uniform code path), "
            + _count(entry.get("n_boundary_screened"))
            + " boundary-screened, "
            + _count(entry.get("n_detail_screened"))
            + " detail-screened, "
            + _count(entry.get("n_no_boundary"))
            + " without a boundary, "
            + _count(entry.get("n_live"))
            + " live, "
            + _count(entry.get("n_duplicate_keys"))
            + " live with duplicate keys.</div>"
        )
        decision = entry.get("decision")
        pairing = entry.get("pairing")
        if type(decision) is dict:
            verdict = str(decision.get("verdict", ""))
            if verdict == "insufficient":
                decision_text = "withheld (insufficient: fewer than six discordant paired specs)"
            elif verdict == "falsified":
                decision_text = (
                    "field-order invariance falsified (direction: "
                    + _escape_text(str(decision.get("direction", "")))
                    + ")"
                )
            else:
                decision_text = "not falsified"
            bits.append(
                "<div>Endpoint decision (answer field first vs last, paired specs): "
                + _escape_text(decision_text)
                + " -- first-only "
                + _count(decision.get("n_first_only"))
                + ", last-only "
                + _count(decision.get("n_last_only"))
                + ", over "
                + _count(decision.get("n_paired"))
                + " paired specs; exact sign test p "
                + _pvalue(decision.get("sign_p"))
                + ".</div>"
            )
        if type(pairing) is dict:
            ungated = pairing.get("ungated")
            if type(ungated) is not dict:
                ungated = {}
            bits.append(
                "<div>Pairing: "
                + _count(pairing.get("n_flagged"))
                + " specs carry the paired flag; "
                + _count(pairing.get("n_order_same"))
                + " lost to identical endpoint orders, "
                + _count(pairing.get("n_duplicate_excluded"))
                + " to duplicate keys, "
                + _count(pairing.get("n_not_adherent"))
                + " to non-adherence in an endpoint arm, leaving "
                + _count(pairing.get("n_paired"))
                + " paired. Answer-field accuracy on the paired "
                "population: first "
                + _rate(pairing.get("first_rate"))
                + ", last "
                + _rate(pairing.get("last_rate"))
                + ", delta "
                + _rate(pairing.get("delta"))
                + ". Ungated sensitivity (non-adherent records "
                "scored by their values): "
                + _count(ungated.get("n_first_only"))
                + " first-only, "
                + _count(ungated.get("n_last_only"))
                + " last-only over "
                + _count(ungated.get("n_specs"))
                + " specs.</div>"
            )
        by_arm = entry.get("by_arm")
        answer_by_arm = entry.get("answer_by_arm")
        whole_doc_by_arm = entry.get("whole_doc_by_arm")
        if type(by_arm) is dict and type(answer_by_arm) is dict:
            points: list[str] = []
            for arm in ("canonical", "answer_first", "answer_last", "reversed"):
                row = by_arm.get(arm)
                ans = answer_by_arm.get(arm)
                if type(row) is not dict or type(ans) is not dict:
                    continue
                whole = whole_doc_by_arm.get(arm) if type(whole_doc_by_arm) is dict else None
                points.append(
                    _escape_text(arm)
                    + ": adherent "
                    + _rate(row.get("adherent_rate"))
                    + " (parse "
                    + _rate(row.get("parse_rate"))
                    + ", schema "
                    + _rate(row.get("schema_rate"))
                    + "), answer-field accuracy "
                    + _rate(ans.get("rate"))
                    + ", whole-document "
                    + _rate(whole.get("rate") if type(whole) is dict else None)
                    + " over "
                    + _count(row.get("n"))
                    + " live"
                )
            if points:
                bits.append(
                    "<div>By arm (canonical is a replicate of an "
                    "endpoint arm on specs whose answer field sits "
                    "at a canonical endpoint): " + "; ".join(points) + ".</div>"
                )
        extraction = entry.get("extraction_decision")
        split = entry.get("field_count_split")
        if type(extraction) is dict and type(split) is dict:
            two = split.get("two")
            if type(two) is not dict:
                two = {}
            three = split.get("three_plus")
            if type(three) is not dict:
                three = {}
            bits.append(
                "<div>Sensitivities: extraction-only decision "
                + _escape_text(str(extraction.get("verdict", "")))
                + " over "
                + _count(extraction.get("n_paired"))
                + " paired extraction specs; paired counts by "
                "field count -- 2-field: "
                + _count(two.get("n_first_only"))
                + " first-only / "
                + _count(two.get("n_last_only"))
                + " last-only over "
                + _count(two.get("n_paired"))
                + "; 3+-field: "
                + _count(three.get("n_first_only"))
                + " first-only / "
                + _count(three.get("n_last_only"))
                + " last-only over "
                + _count(three.get("n_paired"))
                + ". On 2-field specs the answer field's "
                "promotion is the companion field's demotion.</div>"
            )
        rows.append(
            '<tr><td><span class="mono">'
            + _escape_text(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )
    return (
        '<section class="card" id="field_order">'
        "<h2>Schema field order</h2>"
        "<p>Fifteen frozen flat-JSON specs -- the shipped "
        "structured-output set plus six 3-4-field specs -- are "
        "rendered in four field orders: canonical (alphabetical, "
        "the shipped presentation), the designated answer field "
        "first, the answer field last, and fully reversed. "
        "Answers are committed as per-field canonical values and "
        "every verdict -- per-field correctness, schema "
        "adherence, the pairing -- is derived from the "
        "commitment, never stored. The claim under test is "
        "scoped to the endpoint contrast: the answer field's "
        "value accuracy is invariant between the first and last "
        "arms, decided per candidate by an exact sign test over "
        "specs adherent in both endpoint arms (gating that "
        "itself selects against discordance; ungated counts are "
        "printed beside it). Below six discordant paired specs "
        "the verdict is withheld as insufficient: at 6-8 "
        "discordant specs only a unanimous split falsifies, and "
        "a true 30-point accuracy gap falsifies only about 15% "
        "of the time, so the instrument detects only very large "
        "effects. Two shipped specs (route-alert, route-order) "
        "are excluded from pairing because their frozen answer "
        "labels were not entailed by their contexts as "
        "originally shipped; the contexts have since been "
        "repaired so the labels are entailed, but the pairing "
        "population is frozen and unchanged. The level is "
        "per candidate: across 24 candidates at alpha 0.05, "
        "about 1.2 false falsifications are expected at the "
        "nominal rate (the probability of at least one is about "
        "0.71). Minimum discordant specs: 6. Field values per "
        "record are capped at 4096 characters in total; the "
        "payload is capped at 8388608 bytes over at most 1440 "
        "records. Each record is one greedy generation.</p>"
        "<table><thead><tr><th>Candidate</th><th>Readout</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>" + "</section>"
    )


def _choiceaudit_section(result: dict[str, Any]) -> str:
    """Choice-scoring audit: does likelihood scoring pick what generation picks?

    Renders the ``choice_audit`` payload (see
    experiments.choiceaudit_analysis): per candidate, choice
    accuracy under three rankings of the same teacher-forced
    cells, the generation x choice discordance table with the
    generation-fail row split by boundary, and the method-level
    agreement over the common evaluable population. The scope
    limits are printed with the numbers, not under them: the
    donors are unmatched answers to other fixtures (choice
    accuracy is an upper bound on item-matched accuracy), the
    per-token mean is not lm-eval's acc_norm (which is
    per-byte), the generation side is greedy only, and the
    audit is a second opinion that never selects. Counts and
    rates only; no fixture, prompt, or answer text is rendered.
    Every interpolation is escaped at this boundary.
    """
    probe: Any = result.get("choice_audit")
    if type(probe) is not dict or not probe.get("available"):
        return ""
    candidates = probe.get("candidates")
    method = probe.get("method_comparison")
    discordance = probe.get("discordance")
    if type(candidates) is not dict or type(method) is not dict:
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
        accuracy = entry.get("accuracy")
        if type(accuracy) is dict:
            parts = []
            for ranking, label in (
                ("total", "total"),
                ("byte", "per-byte"),
                ("mean", "per-token mean"),
            ):
                block = accuracy.get(ranking)
                if type(block) is dict:
                    parts.append(
                        label
                        + " "
                        + _rate(block.get("accuracy"))
                        + " ("
                        + _count(block.get("correct"))
                        + "/"
                        + _count(block.get("n"))
                        + ", "
                        + _count(block.get("tied"))
                        + " tied)"
                    )
            bits.append("<div>Choice accuracy: " + "; ".join(parts) + ".</div>")
        disc = entry.get("discordance")
        if type(disc) is dict:
            bits.append(
                "<div>Discordance over "
                + _count(disc.get("n_paired"))
                + " paired fixtures: generation-pass/choice-incorrect "
                + _count(disc.get("gen_pass_choice_incorrect"))
                + ", generation-fail/choice-correct "
                + _count(disc.get("gen_fail_choice_correct"))
                + " (discordant "
                + _count(disc.get("n_discordant"))
                + "); generation failures split by boundary found "
                + _count(disc.get("gen_fail_boundary_found"))
                + " / not found "
                + _count(disc.get("gen_fail_boundary_not_found"))
                + "; "
                + _count(disc.get("n_choice_tied"))
                + " choice ties reported beside the table.</div>"
            )
        bits.append(
            "<div>Evaluable "
            + _count(entry.get("n_evaluable"))
            + " of "
            + _count(entry.get("n_constructible"))
            + " constructible fixtures; "
            + _count(entry.get("n_choice_unavailable"))
            + " without a choice verdict, "
            + _count(entry.get("n_generation_missing"))
            + " without a generation verdict.</div>"
        )
        rows.append(
            '<tr><td><span class="mono">'
            + escape(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )
    agreement = str(method.get("agreement", ""))
    if agreement == "agree":
        method_line = (
            "Over the common evaluable population ("
            + _count(method.get("n_common"))
            + " fixtures), the choice method's top candidate agrees with "
            "the primary winner (" + escape(str(method.get("primary_winner"))) + ")."
        )
    elif agreement == "disagree":
        method_line = (
            "Over the common evaluable population ("
            + _count(method.get("n_common"))
            + " fixtures), the choice method's top candidate ("
            + escape(str(method.get("choice_top_candidate")))
            + ") disagrees with the primary winner ("
            + escape(str(method.get("primary_winner")))
            + ")."
        )
    elif agreement == "choice_tied":
        method_line = (
            "Over the common evaluable population ("
            + _count(method.get("n_common"))
            + " fixtures), the choice method's top is tied between "
            + escape(", ".join(str(n) for n in method.get("choice_top_tied", [])))
            + "; ties are reported, never broken."
        )
    elif agreement == "native_abstain":
        method_line = (
            "The primary run abstained ("
            + escape(str(method.get("primary_decision")))
            + "), so there is no primary winner for the choice method "
            "to agree with; the comparison is reported as an abstention."
        )
    else:
        method_line = (
            "The method-level agreement is withheld: the common "
            "evaluable population ("
            + _count(method.get("n_common"))
            + ") is below the floor of "
            + _count(method.get("floor"))
            + ". The counts above still stand; only the agreement "
            "claim is gated."
        )
    falsified_line = ""
    if type(discordance) is dict:
        falsified = discordance.get("falsified")
        if falsified is True:
            falsified_line = (
                "<p>Zero discordance on "
                + _count(discordance.get("n_cells"))
                + " completed, fully-evaluable cells under the total "
                "ranking: 'the scoring method matters for this "
                "checkpoint/fixture set' is falsified for greedy "
                "generation on this fixture set.</p>"
            )
        elif falsified is False:
            falsified_line = (
                "<p>Discordance on "
                + _count(discordance.get("n_discordant"))
                + " of "
                + _count(discordance.get("n_cells"))
                + " completed, fully-evaluable cells: the scoring "
                "method changes answers on this checkpoint/fixture "
                "set.</p>"
            )
    return (
        '<section class="card" id="choice-audit">'
        "<h2>Choice-scoring audit (second opinion)</h2>"
        "<p>Each search fixture's answer is scored against three "
        "donor answers drawn from other fixtures' expected "
        "answers, screened so no donor is the same answer twice; "
        "every choice is teacher-forced under each candidate over "
        "the recorded search prompt, with no extra generations. "
        "The headline ranking is total log-likelihood (lm-eval's "
        "<em>acc</em> semantics); the per-byte ranking is its "
        "<em>acc_norm</em>. The per-token mean is a third quantity "
        "the harness does not report and is never labelled "
        "acc_norm here. Donors are unmatched -- answers to "
        "unrelated fixtures, screened for equivalence only -- so "
        "choice accuracy is an upper bound on item-matched "
        "choice accuracy. A choice cell marked unavailable for a "
        "boundary-merge reason failed tokenisation alignment at "
        "the prompt/answer seam; that is a different fact from a "
        "generation boundary not being found, and the two are "
        "never pooled. This audit is a second opinion: it never "
        "changes the selection.</p>"
        '<table class="data"><thead><tr><th scope="col">Candidate</th>'
        '<th scope="col">Choice scoring vs generation</th></tr></thead>'
        "<tbody>" + "".join(rows) + "</tbody></table>"
        "<p>" + method_line + "</p>" + falsified_line + "</section>"
    )


def _precision_section(result: dict[str, Any]) -> str:
    """Load precision: does the fp32 winner survive a bf16 load?

    Renders the ``precision`` payload (see experiments.precision_stage):
    per candidate, the verdict-flip matrix over fixtures completed in
    both arms (the fp32 arm is the recorded native run; the bf16 arm is
    the same weights loaded in bfloat16), the output-divergence rate,
    and the teacher-forced expected-answer logprob deltas; then the
    winner-survival readout over the shared paired population and the
    scoped falsification line. The scope limits are printed with the
    numbers: search split only, verdict level, one bf16 run against
    the recorded fp32 run, dtype precision -- not quantization -- and
    the determinism certificate covers the native fp32 observations
    only. Counts, rates, and fixture IDs only; no answer text is
    rendered. Every interpolation is escaped at this boundary.
    """
    probe: Any = result.get("precision")
    if type(probe) is not dict or not probe.get("available"):
        return ""
    candidates = probe.get("candidates")
    totals = probe.get("totals")
    selection = probe.get("selection")
    if type(candidates) is not dict or type(totals) is not dict or type(selection) is not dict:
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

    def _delta(value: Any) -> str:
        """Return the delta."""
        if type(value) in (int, float) and not isinstance(value, bool):
            try:
                return escape(f"{float(value):+.3f}")
            except OverflowError:
                return "n/a"
        return "n/a"

    def _passed(value: Any) -> str:
        """Return the passed."""
        if value is True:
            return "passes"
        if value is False:
            return "fails"
        return "unavailable"

    rows: list[str] = []
    for name in sorted(candidates, key=str):
        entry = candidates[name]
        if type(entry) is not dict:
            continue
        cells = entry.get("cells")
        logprob = entry.get("logprob")
        if type(cells) is not dict or type(logprob) is not dict:
            continue
        bits = [
            "<div>Paired fixtures "
            + _count(entry.get("n_paired"))
            + ": pass/pass "
            + _count(cells.get("pass_pass"))
            + ", pass/fail "
            + _count(cells.get("pass_fail"))
            + ", fail/pass "
            + _count(cells.get("fail_pass"))
            + ", fail/fail "
            + _count(cells.get("fail_fail"))
            + "; completed in one arm only: fp32 "
            + _count(entry.get("n_fp32_only"))
            + ", bf16 "
            + _count(entry.get("n_bf16_only"))
            + ".</div>",
            "<div>All-pass over the full search set: fp32 "
            + _passed(entry.get("fp32_passed"))
            + ", bf16 "
            + _passed(entry.get("bf16_passed"))
            + ". Generations diverged on "
            + _count(entry.get("n_diverged"))
            + " paired fixtures (rate "
            + _rate(entry.get("divergence_rate"))
            + ").</div>",
            "<div>Expected-answer logprob delta (bf16 - fp32) over "
            + _count(logprob.get("n"))
            + " paired fixtures: mean "
            + _delta(logprob.get("mean_delta"))
            + ", min "
            + _delta(logprob.get("min_delta"))
            + ", max "
            + _delta(logprob.get("max_delta"))
            + ".</div>",
        ]
        rows.append(
            '<tr><td><span class="mono">'
            + escape(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )
    flips = probe.get("flips")
    flip_line = ""
    if type(flips) is list and flips:
        items = []
        for flip in flips:
            if type(flip) is not dict:
                continue
            direction = (
                "pass to fail" if flip.get("direction") == "pass_to_fail" else "fail to pass"
            )
            items.append(
                escape(str(flip.get("candidate")))
                + " / "
                + escape(str(flip.get("fixture")))
                + " ("
                + direction
                + ")"
            )
        if items:
            flip_line = "<p>Verdict flips: " + "; ".join(items) + ".</p>"
    unavailable = probe.get("unavailable_candidates")
    unavailable_line = ""
    if type(unavailable) is list and unavailable:
        unavailable_line = (
            "<p>Candidates with no completed fixture in either arm: "
            + escape(", ".join(str(n) for n in unavailable))
            + ".</p>"
        )
    survived = selection.get("survived")
    if survived is True:
        selection_line = (
            "<p>Over the shared paired population, re-applying the "
            "selection rule to both arms gives the same outcome: the "
            "fp32 winner survives the bf16 load on the search split.</p>"
        )
    elif survived is False:
        selection_line = (
            "<p>Over the shared paired population, the bf16 selection "
            "differs from the fp32 selection: the fp32 winner does not "
            "survive the bf16 load on this evidence. The fp32 selection "
            "stands as the run's decision; do not serve this winner in "
            "bf16 on this evidence.</p>"
        )
    else:
        selection_line = (
            "<p>Winner survival is withheld ("
            + escape(str(selection.get("withheld")))
            + "): the counts above still stand; only the survival "
            "claim is gated.</p>"
        )
    falsified = probe.get("precision_effect_falsified")
    if falsified is True:
        falsified_line = (
            "<p>Zero verdict flips over "
            + _count(totals.get("n_paired"))
            + " paired fixtures: a precision effect is falsified on "
            "the search split, at the verdict level, for this "
            "checkpoint and fixture set in a single bf16 run against "
            "the recorded fp32 run. Where no candidate passes in "
            "either arm, zero flips is the floor effect, not evidence "
            "of robustness -- the divergence and logprob numbers above "
            "are the informative readouts there.</p>"
        )
    elif falsified is False:
        falsified_line = (
            "<p>Verdict flips over "
            + _count(totals.get("n_paired"))
            + " paired fixtures: load precision changes verdicts on "
            "the search split for this checkpoint and fixture set.</p>"
        )
    else:
        falsified_line = (
            "<p>No paired fixtures completed in both arms, so the "
            "falsification readout is unavailable.</p>"
        )
    return (
        '<section class="card" id="precision">'
        "<h2>Load precision (fp32 vs bf16)</h2>"
        "<p>The same checkpoint was loaded a second time in "
        "bfloat16 and every search candidate was re-evaluated on "
        "the search split; the fp32 arm is the recorded native run. "
        "This is dtype precision, not quantization: the bf16 arm is "
        "a load-time cast of the same unquantized weights. Held-out "
        "behaviour under bf16 is not measured, and the determinism "
        "certificate covers the native fp32 search observations "
        "only.</p>"
        '<table class="data"><thead><tr><th scope="col">Candidate</th>'
        '<th scope="col">fp32 vs bf16</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
        + flip_line
        + unavailable_line
        + selection_line
        + falsified_line
        + "</section>"
    )


def _confidence_section(result: dict[str, Any]) -> str:
    """Verbalized confidence: does the stated number track correctness?

    Renders the ``confidence`` payload (see experiments.confidence_stage):
    per candidate, the elicitation compliance rate under the frozen line
    grammar, mean stated confidence on correct vs incorrect compliant
    answers with their descriptive gap, the frozen four-bin table, the
    verdict flips the suffix itself caused against the recorded native
    run, and -- over the generated pair -- whether stated confidence
    drops when the answer is not knowable from the prompt (knowable
    planted-fact cases vs their donor-key controls). The framing limits
    are printed with the numbers: a stated confidence is model output
    text, not a probability, and no readout here is a calibration
    score; the probe covers the search split only (held-out fixtures
    are never treated), and the determinism certificate covers the
    recorded native search observations only -- these elicited cells
    are outside its scope. Counts, rates, and digests only; no answer
    text is rendered. Every interpolation is escaped at this boundary.
    """
    probe: Any = result.get("confidence")
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

    def _num(value: Any) -> str:
        """Return the num."""
        if type(value) in (int, float) and not isinstance(value, bool):
            try:
                return escape(f"{float(value):.1f}")
            except OverflowError:
                return "n/a"
        return "n/a"

    def _count(value: Any) -> str:
        """Return the count."""
        return escape(str(value)) if type(value) is int else "n/a"

    def _block_lines(block: Any, label: str) -> list[str]:
        """Return the block lines."""
        if type(block) is not dict:
            return []
        joint = block.get("joint")
        if type(joint) is not dict:
            return []
        lines = [
            "<div>"
            + escape(label)
            + ": completed "
            + _count(block.get("n_completed"))
            + " of "
            + _count(block.get("n_population"))
            + ", compliant "
            + _count(block.get("n_compliant"))
            + " (rate "
            + _rate(block.get("compliance_rate"))
            + "); compliant answers split correct "
            + _count(joint.get("compliant_correct"))
            + " / incorrect "
            + _count(joint.get("compliant_incorrect"))
            + ", non-compliant "
            + _count(joint.get("noncompliant"))
            + ".</div>"
        ]
        gap = block.get("gap")
        if isinstance(gap, (int, float)) and not isinstance(gap, bool) and _float_convertible(gap):
            lines.append(
                "<div>Mean stated confidence: correct "
                + _num(block.get("mean_correct"))
                + ", incorrect "
                + _num(block.get("mean_incorrect"))
                + " -- descriptive gap "
                + escape(f"{float(gap):+.1f}")
                + " (agreement between the stated number and the "
                "verdict on compliant cells; not a calibration "
                "score).</div>"
            )
        else:
            lines.append(
                "<div>Separation gap withheld ("
                + escape(str(block.get("gap_detail")))
                + "); the compliant counts above still stand.</div>"
            )
        bins = block.get("bins")
        if type(bins) is list and bins:
            parts = []
            for entry in bins:
                if type(entry) is not dict:
                    continue
                parts.append(
                    escape(str(entry.get("lo")))
                    + "-"
                    + escape(str(entry.get("hi")))
                    + ": n "
                    + _count(entry.get("n"))
                    + ", accuracy "
                    + _rate(entry.get("accuracy"))
                )
            if parts:
                lines.append(
                    "<div>Stated-confidence bins (compliant cells): " + "; ".join(parts) + ".</div>"
                )
        return lines

    rows: list[str] = []
    for name in sorted(candidates, key=str):
        entry = candidates[name]
        if type(entry) is not dict:
            continue
        operator = entry.get("operator")
        generated = entry.get("generated")
        if type(operator) is not dict or type(generated) is not dict:
            continue
        bits = _block_lines(operator, "Search fixtures (elicited)")
        flips = operator.get("flips")
        if type(flips) is dict:
            bits.append(
                "<div>Elicitation cost vs the recorded native run: "
                + _count(flips.get("n_compared"))
                + " compared, pass to fail "
                + _count(flips.get("pass_to_fail"))
                + ", fail to pass "
                + _count(flips.get("fail_to_pass"))
                + ".</div>"
            )
        bits += _block_lines(generated.get("knowable"), "Generated knowable cases")
        bits += _block_lines(generated.get("control"), "Generated donor-key control")
        knowability = generated.get("knowability")
        if type(knowability) is dict:
            falsified = knowability.get("falsified")
            if falsified is True:
                verdict = (
                    "mean stated confidence on the control arm is at "
                    "or above the knowable arm over paired compliant "
                    "cases: knowability sensitivity is falsified on "
                    "this generated instrument, for this checkpoint "
                    "and seed."
                )
            elif falsified is False:
                verdict = (
                    "mean stated confidence drops from the knowable "
                    "arm to the donor-key control over paired "
                    "compliant cases."
                )
            else:
                verdict = (
                    "knowability readout withheld ("
                    + escape(str(knowability.get("detail")))
                    + "); the per-arm counts above still stand."
                )
            bits.append(
                "<div>Knowability: paired compliant cases "
                + _count(knowability.get("n_paired_compliant"))
                + ", knowable mean "
                + _num(knowability.get("knowable_mean"))
                + ", control mean "
                + _num(knowability.get("control_mean"))
                + " -- "
                + verdict
                + "</div>"
            )
        rows.append(
            '<tr><td><span class="mono">'
            + _escape_text(name)
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )
    unavailable = probe.get("unavailable_candidates")
    unavailable_line = ""
    if type(unavailable) is list and unavailable:
        unavailable_line = (
            "<p>Candidates with no completed elicited cell: "
            + _escape_text(", ".join(str(n) for n in unavailable))
            + ".</p>"
        )
    return (
        '<section class="card" id="confidence">'
        "<h2>Verbalized confidence</h2>"
        "<p>Every search fixture was re-run with one frozen suffix "
        "asking the model to end its reply with a stated confidence "
        "line, parsed under a frozen grammar and scored against the "
        "same exact-match verdicts as the native run. A stated "
        "confidence is model output text, not a probability; the "
        "gap and bins below describe agreement between the number "
        "and the verdict, and are not calibration scores. The "
        "generated cases are instrument cases (planted synthetic "
        "facts), not operator fixtures. Search split only; the "
        "determinism certificate covers the recorded native search "
        "observations only, so these elicited cells are outside its "
        "scope.</p>"
        '<table class="data"><thead><tr><th scope="col">Candidate</th>'
        '<th scope="col">Stated confidence vs verdicts</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
        + unavailable_line
        + "</section>"
    )
