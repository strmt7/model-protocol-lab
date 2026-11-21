"""reporting text family: split from reporting.py (W7-2).

Facades keep the public surface: import from model_protocol_lab.reporting, not from this module.
"""

from html import escape
from typing import Any

from .reporting_core import (
    _escape_text,
    _float_convertible,
)

_WSCAN_MUTATIONS = (
    "trailing_space",
    "double_trailing_space",
    "trailing_newline",
    "trailing_tab",
    "leading_space",
    "leading_newline",
    "strip_trailing_space",
    "collapse_double_space",
)


def _wscanonical_section(result: dict[str, Any]) -> str:
    """Whitespace canonicalization: are prompts whitespace-fragile?

    Every search candidate's rendered prompts are re-encoded under
    whitespace mutations of the render string (trailing/leading space,
    newline, tab, stripping, collapsing); id sequences are compared
    exactly and fixtures on divergent variants are re-run to measure
    verdict flips. Reports per-candidate per-mutation id divergence and
    verdict-flip rates plus the whitespace-robust verdict. A flipped
    verdict on identical ids is a backend-determinism anomaly, not
    whitespace fragility, and is reported as such. Renders nothing when
    the whitespace scan did not run.
    """
    audit: Any = result.get("ws_canonical")
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

    def _bool(value: Any) -> str:
        """Return the bool."""
        if value is True:
            return "yes"
        if value is False:
            return "no"
        return "n/a"

    rows = []
    for key in sorted(per_candidate, key=str):
        cand = str(key)
        entry = per_candidate[key]
        if type(entry) is not dict:
            continue
        cells = []
        mutations = entry.get("per_mutation")
        if type(mutations) is not dict:
            mutations = {}
        for mutation in _WSCAN_MUTATIONS:
            cell = mutations.get(mutation)
            if type(cell) is not dict or not cell.get("evaluable"):
                cells.append("<td>n/a</td>")
                continue
            div = _rate(cell.get("ids_divergence_rate"))
            flip = _rate(cell.get("verdict_flip_rate"))
            marker = "F" if cell.get("fragile") else ("D" if cell.get("n_ids_divergent") else "=")
            cells.append(
                '<td><span class="mono">'
                + escape(marker)
                + "</span> "
                + escape(div)
                + " / "
                + escape(flip)
                + "</td>"
            )
        aggregate = entry.get("aggregate")
        if type(aggregate) is not dict:
            aggregate = {}
        rows.append(
            '<tr><td><span class="mono">'
            + escape(cand)
            + "</span></td>"
            + "".join(cells)
            + "<td>"
            + escape(_rate(aggregate.get("max_ids_divergence_rate")))
            + "</td>"
            + "<td>"
            + escape(_rate(aggregate.get("max_verdict_flip_rate")))
            + "</td>"
            + "<td>"
            + escape(_bool(entry.get("whitespace_robust")))
            + "</td></tr>"
        )
    fragile = audit.get("fragile_candidates")
    fragile_line = ""
    if type(fragile) is list and fragile:
        fragile_line = (
            "<p>Whitespace-fragile candidates: "
            + ", ".join(
                '<span class="mono">' + escape(str(name)) + "</span>"
                for name in fragile
                if type(name) is str
            )
            + ".</p>"
        )
    return (
        '<section class="card" id="whitespace-canonicalization">'
        "<h2>Whitespace canonicalization</h2>"
        "<p>Each rendered prompt re-encoded under whitespace mutations "
        "of the render string; id sequences compared exactly; fixtures "
        "on divergent variants re-run to measure verdict flips. Cell "
        "legend: <strong>=</strong> identical ids and verdicts, "
        "<strong>D</strong> divergent ids without a verdict flip, "
        "<strong>F</strong> at least one verdict flip (shown as "
        "id-divergence rate / verdict-flip rate).</p>"
        + fragile_line
        + (
            '<div class="table-wrap"><table><caption>Id-divergence / '
            "verdict-flip rates per candidate and mutation.</caption>"
            '<thead><tr><th scope="col">Candidate</th>'
            + "".join(
                '<th scope="col">' + escape(mutation.replace("_", " ")) + "</th>"
                for mutation in _WSCAN_MUTATIONS
            )
            + '<th scope="col">Max id-div.</th>'
            '<th scope="col">Max flip</th><th scope="col">Robust</th>'
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            if rows
            else ""
        )
        + "</section>"
    )


def _canonical_form_section(result: dict[str, Any]) -> str:
    """Canonical-form ladder probe: NFC vs NFD prompt text.

    Renders the ``canonical_form`` payload (see
    experiments.canonicalform_analysis): per candidate, the
    id-divergent paired counts split by direction, the
    classification, the ids-identical share (the
    instrument-inapplicable share, never robustness), the
    same-ids verdict flips (a determinism finding, never
    fragility), and the winner headline comparing the two arms'
    selections. The scope limits are printed with the numbers,
    not under them: the headline conditions on cases whose full
    render id sequences actually differ, and each cell is one
    greedy generation on generated fixtures. Counts and rates
    only; no payload, question, or answer text is rendered.
    Every payload-derived value below goes through
    _escape_text, never bare escape().
    """
    probe: Any = result.get("canonical_form")
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
                "<div>Id-divergent paired cases "
                + _count(headline.get("n_paired"))
                + ": NFC only "
                + _count(headline.get("n_nfc_only"))
                + ", NFD only "
                + _count(headline.get("n_nfd_only"))
                + ", both "
                + _count(headline.get("n_both"))
                + ", neither "
                + _count(headline.get("n_neither"))
                + " ("
                + _escape_text(str(headline.get("classification", "")))
                + ").</div>"
            )
        bits.append(
            "<div>Invariance falsified: "
            + ("yes" if entry.get("invariance_falsified") is True else "no")
            + ".</div>"
        )
        ids = entry.get("ids")
        if type(ids) is dict:
            bits.append(
                "<div>Ids identical "
                + _count(ids.get("n_ids_identical"))
                + " cases (instrument-inapplicable share), divergent "
                + _count(ids.get("n_ids_divergent"))
                + ", same-ids verdict flips "
                + _count(ids.get("n_same_ids_verdict_flips"))
                + " (determinism finding), max abs id-length delta "
                + _count(ids.get("max_abs_id_delta"))
                + ".</div>"
            )
        arms = entry.get("arms")
        if type(arms) is dict:
            rates = ", ".join(
                _escape_text(str(arm)) + ": " + _rate((arms.get(arm) or {}).get("rate"))
                for arm in ("nfc", "nfd")
            )
            bits.append("<div>Pass rates — " + rates + ".</div>")
        rows.append(
            '<tr><td><span class="mono">'
            + _escape_text(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )
    selection = probe.get("selection")
    winner_line = ""
    if type(selection) is dict:
        winner = selection.get("winner")
        if type(winner) is dict:
            winner_line = (
                "<p>Winner headline: "
                + _escape_text(str(winner.get("headline", "")))
                + " (winner changed: "
                + (
                    "yes"
                    if winner.get("winner_changed") is True
                    else "no"
                    if winner.get("winner_changed") is False
                    else "undefined"
                )
                + ").</p>"
            )
    return (
        '<section class="card" id="canonical-form">'
        "<h2>Canonical-form sensitivity (NFC vs NFD)</h2>"
        "<p>Twenty-four generated cases each bury one "
        "answer-bearing fact in the middle sentence of a "
        "three-sentence payload; the same message text is "
        "rendered in NFC (composed) and NFD (decomposed) form "
        "per candidate. Accents live in carrier prose, topic, "
        "and question only; every expected value is pure ASCII, "
        "so the byte-exact scorer cannot manufacture "
        "discordance. The headline conditions on cases whose "
        "full render id sequences actually differ; cases whose "
        "ids are identical are reported as the "
        "instrument-inapplicable share, never as robustness. "
        "Each cell is one greedy generation.</p>"
        "<table><thead><tr><th>Candidate</th><th>Readout</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>" + winner_line + "</section>"
    )


def _record_notation_section(result: dict[str, Any]) -> str:
    """Record-notation probe: the same generated table in two
    frozen notations, described per candidate.

    Renders the ``record_notation`` payload (see
    experiments.record_notation_analysis). The first sentence
    states the treatment exactly: identical records, two
    notations. Per candidate, in the design's printed order:
    the audit counts (closed-book screening) first; per-arm
    pass rates with attrition and the completed-cell
    decomposition; the echo and parsed-value rates beside the
    failure counts they are shares of; the headline paired
    discordance with its direction, decomposition, paired
    per-arm rates, and resolution; the id report; and the
    sensitivities (unscreened discordance; R1 and fence
    regrade discordance). After the table: the estimand
    sentence, the operating-characteristics paragraph (the
    executed reason no classification ships, every power
    figure with its marginals), and the printed limitations.
    The instrument is descriptive -- there is no decision to
    render. Counts and rates only; no table, prompt, or answer
    text is rendered. Every payload-derived value below goes
    through _escape_text, never bare escape().
    """
    probe: Any = result.get("record_notation")
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

    def _points(value: Any) -> str:
        """Return the points."""
        if type(value) in (int, float) and not isinstance(value, bool):
            try:
                return _escape_text(f"{float(value):.1f}")
            except OverflowError:
                return "n/a"
        return "n/a"

    arm_names = ("prose", "canonical_json", "closed_book")
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
        audit = entry.get("audit")
        if type(audit) is dict:
            bits.append(
                "<div>Closed-book screen: "
                + _count(audit.get("n_screened"))
                + " cases screened out of this candidate's live "
                "populations (passed with no table at all), "
                + _count(audit.get("n_closed_book_incomplete"))
                + " screen cells incomplete (never treated as "
                "screened or as passed).</div>"
            )
        arms = entry.get("arms")
        if type(arms) is dict:
            for arm_name in arm_names:
                arm = arms.get(arm_name)
                if type(arm) is not dict:
                    continue
                decomp = arm.get("decomposition")
                decomp_bit = ""
                if type(decomp) is dict:
                    decomp_bit = (
                        " Decomposition of completed cells -- pass "
                        + _count(decomp.get("pass"))
                        + ", boundary-only "
                        + _count(decomp.get("boundary_only"))
                        + " (the answer digest still equals the "
                        "expected one; the boundary event alone "
                        "failed -- a boundary effect, never a "
                        "notation effect), other failures "
                        + _count(decomp.get("other_failure"))
                        + "."
                    )
                diag_bit = ""
                if arm_name != "closed_book":
                    diag_bit = (
                        " Of "
                        + _count(arm.get("n_failures"))
                        + " completed verdict failures, the echo "
                        "rate (the answer contains the expected "
                        "value as a substring) is "
                        + _rate(arm.get("echo_rate"))
                        + " and the parsed-value rate (the answer "
                        "parses as a JSON object whose code value "
                        "equals the expected value) is "
                        + _rate(arm.get("parsed_rate"))
                        + " -- two independent diagnostics, "
                        "neither a subset of the other: JSON "
                        "string escapes let a parsed hit occur "
                        "with no literal substring hit."
                    )
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
                    + " evaluation-failed."
                    + decomp_bit
                    + diag_bit
                    + "</div>"
                )
        headline = entry.get("headline")
        if type(headline) is dict:
            decomp = headline.get("decomposition")
            decomp_bit = ""
            if type(decomp) is dict:
                decomp_bit = (
                    " Decomposition of the discordant pairs -- "
                    "exact-only "
                    + _count(decomp.get("exact_only"))
                    + ", boundary-only "
                    + _count(decomp.get("boundary_only"))
                    + ", mixed "
                    + _count(decomp.get("mixed"))
                    + " (reported, never attributed)."
                )
            bits.append(
                "<div>Headline over "
                + _count(headline.get("n_paired"))
                + " paired cases (live -- not closed-book "
                "screened -- with both headline cells completed, "
                "minus same-ids anomalies): discordant "
                + _count(headline.get("discordant"))
                + ", of which prose-only passes "
                + _count(headline.get("n_prose_only"))
                + " and canonical_json-only passes "
                + _count(headline.get("n_canonical_json_only"))
                + " (this instrument has no baseline arm; "
                "neither arm is a control). Paired pass rates -- "
                "prose "
                + _rate(headline.get("prose_rate"))
                + ", canonical_json "
                + _rate(headline.get("canonical_json_rate"))
                + ". Resolution: one paired case is "
                + _points(headline.get("resolution_points"))
                + " points of discordance rate at this paired "
                "count; the count is the readout, and no finer "
                "claim is printed." + decomp_bit + "</div>"
            )
        ids = entry.get("ids")
        if type(ids) is dict:
            deltas = ids.get("id_length_delta")
            delta_bit = ""
            if type(deltas) is list:
                delta_bit = (
                    " Per-case id-length deltas (canonical JSON "
                    "minus prose, in case order): "
                    + _escape_text(", ".join(str(d) for d in deltas))
                    + "."
                )
            bits.append(
                "<div>Token-ID report (reported, not gated): "
                + _count(ids.get("n_ids_identical"))
                + " ids-identical cases (excluded from the "
                "headline as rendering anomalies), maximum "
                "absolute id-length delta "
                + _count(ids.get("max_abs_id_delta"))
                + ", "
                + _count(ids.get("n_same_ids_verdict_flips"))
                + " same-ids verdict flips (a determinism "
                "finding, never notation fragility)." + delta_bit + "</div>"
            )
        sensitivities = entry.get("sensitivities")
        if type(sensitivities) is dict:
            unscreened = sensitivities.get("unscreened")
            unscreened_bit = ""
            if type(unscreened) is dict:
                unscreened_bit = (
                    " Over the unscreened paired set ("
                    + _count(unscreened.get("n_paired"))
                    + " cases): discordant "
                    + _count(unscreened.get("discordant"))
                    + ", prose-only "
                    + _count(unscreened.get("n_prose_only"))
                    + ", canonical_json-only "
                    + _count(unscreened.get("n_canonical_json_only"))
                    + "."
                )
            bits.append(
                "<div>Sensitivities:" + unscreened_bit + " Regrade discordance over the headline "
                "population, boundary conjunct unchanged -- R1 "
                "(whitespace trim/collapse) "
                + _count(sensitivities.get("regrade_r1_discordant"))
                + ", fence strip "
                + _count(sensitivities.get("regrade_fence_discordant"))
                + ".</div>"
            )
        rows.append(
            '<tr><td><span class="mono">'
            + _escape_text(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )
    oc: dict[str, Any] = {}
    for entry in candidates.values():
        if type(entry) is dict and type(entry.get("operating_characteristics")) is dict:
            oc = entry["operating_characteristics"]
            break

    def _oc(key: str) -> str:
        """Return the oc."""
        return _rate(oc.get(key))

    def _sci(key: str) -> str:
        """Return the sci."""
        value = oc.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                return _escape_text(f"{float(value):.2e}")
            except OverflowError:
                return "n/a"
        return "n/a"

    tail_paragraphs = [
        "<p>The estimand, stated once: discordance between "
        "these two frozen renderings, over live cases both arms "
        "completed, for this candidate's template family on "
        "this checkpoint. The claim is descriptive pairwise "
        "discordance between two named frozen renderings, and "
        "nothing else -- never a causal claim, never a claim "
        "about weights, never a claim about notations not "
        "frozen here. The treatment is the two renderings as "
        "wholes: token length, delimiters and punctuation, "
        "lexical framing, field order as a reading sequence in "
        "prose, and output priming move together and are named "
        "as the bundle, never attributed individually.</p>",
        "<p>Why no classification ships (exact multinomial "
        "enumerations over per-case outcomes at 24 paired "
        "cases, executed, not assumed): the canonical-form "
        "transplant rule fires "
        + _oc("transplant_fire_competent_independent")
        + " on a competent independent null (family-wise "
        "expectation "
        + _rate(oc.get("transplant_familywise_expected"))
        + " false labels per 24-candidate panel), "
        + _oc("transplant_fire_competent_paired_phi05")
        + " under a paired null at correlation 0.5, and "
        + _oc("transplant_fire_delta_005")
        + " under fair-direction noise at delta 0.05; Rule A "
        "fires "
        + _oc("rule_a_fire_competent_independent")
        + " / "
        + _oc("rule_a_fire_competent_paired_phi05")
        + " / "
        + _oc("rule_a_fire_delta_005")
        + " on the same three nulls. The family-wise-controlled "
        "rules are near-inert at the gaps that matter: McNemar "
        "power is "
        + _oc("mcnemar_power_gap25_marginals_85_60")
        + " at a 25-point gap with marginals .85/.60 and "
        + _oc("mcnemar_power_gap20_marginals_90_70")
        + " at a 20-point gap with marginals .90/.70 -- gap "
        "alone does not determine power; at the competent base "
        ".90 a 25-point gap gives McNemar power "
        + _oc("mcnemar_power_gap25_base_90")
        + ". The Bonferroni sign rule's power is "
        + _oc("sign_bonferroni_power_gap25_marginals_85_60")
        + " at the 25-point gap. The larger panel that would "
        "license a directional claim (96 cases: Bonferroni "
        "false-fire "
        + _oc("panel96_bonferroni_false_fire")
        + ", power "
        + _oc("panel96_bonferroni_power_gap25_marginals_85_60")
        + " at marginals .85/.60 and "
        + _oc("panel96_bonferroni_power_gap25_base_90")
        + " at the .90 base) costs 6,912 generations for a "
        "claim this readout does not consume. There is no "
        "winner readout either: a candidate passes an arm only "
        "by passing every completed case, so the pass "
        "probability over 24 lookups is "
        + _oc("winner_pass_probability_95")
        + " at a per-case rate of .95 and "
        + _oc("winner_pass_probability_80")
        + " at .80 -- the winner headline would be undefined "
        "or abstention in the large majority of runs; the "
        "per-arm accuracies above are the selection-adjacent "
        "readout.</p>",
        "<p>The closed-book screen is retained as fail-closed "
        "hygiene against generator or code-space drift: a "
        "closed-book pass requires guessing one value of "
        "216,000 (probability "
        + _sci("screen_guess_probability")
        + " per case), so expected exclusions are "
        + _sci("screen_expected_exclusions_per_candidate")
        + " per candidate; its cost (576 of 1,728 generations "
        "at the legal maximum) is stated here as in the CLI "
        "help. The question every case asks embeds the key "
        "value: the answer's word equals the key's word in "
        "about 4.17 percent of cases by construction (analytic "
        "8999/215999) -- arm-symmetric, because the question is "
        "byte-identical across arms, so it cannot manufacture "
        "discordance; no redraw rule is applied, because a "
        "redraw would change the population. Printed "
        "limitations: no typed values, no CSV/XML/YAML arms, "
        "no length matching and no component-level attribution; "
        "no generalisation beyond the two frozen renderings, "
        "the generated code-table population, and a candidate's "
        "own template family. Records carry digests, except "
        "headline cells' answer text, which is digest-bound "
        "and never rendered here.</p>",
    ]
    return (
        '<section class="card" id="record-notation">'
        "<h2>Record notation (prose sentences vs canonical JSON)</h2>"
        "<p>This probe renders the same generated data table "
        "in two frozen notations: each of the 24 generated cases "
        "is a code table (4 to 6 records of 3 to 4 fields; every "
        "value a single token from a frozen 24-word list over a "
        "216,000 code space, all values in a table distinct) "
        "placed inside the user message as prose sentences in "
        "one arm and as canonical JSON in the other, the record "
        "sequence, the question, and the output contract held "
        "byte-identical across arms, plus a closed-book screen "
        "arm that sees the question alone. The prose rendering "
        "is injective over this population -- it has an "
        "exhibited inverse -- so a reader given either rendering "
        "recovers the identical table. The scorer is byte-exact "
        "conjoined with boundary presence; every arm is a "
        "fixture with the same id, so request seeds are "
        "identical across arms. Each cell is one greedy "
        "generation. The readout is descriptive: paired "
        "discordance counts, their direction, and their "
        "decomposition -- no classification, no winner, scoped "
        "for this checkpoint and this candidate's template family.</p>"
        "<table><thead><tr><th>Candidate</th><th>Readout</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + "".join(tail_paragraphs)
        + "</section>"
    )


def _invisible_chars_section(result: dict[str, Any]) -> str:
    """Invisible-character insertion probe: does one frozen
    invisible code point in a fact sentence move answers to
    wrong values?

    Renders the ``invisible_chars`` payload (see
    experiments.invisible_chars_analysis). The first sentence
    states the treatment exactly: one code point at two frozen
    sites in the fact sentence, the question and carriers
    byte-identical across arms. Per candidate, in the design's
    printed order: per-arm pass rates with attrition and the
    six-class completed-cell decomposition; the per-arm
    token-ID report (the regime gate); and per treated arm the
    guarded headline over the id-divergent paired population
    -- label, D, J, reverse discordants, incomplete count, the
    paired control rate beside the label, the pair
    decomposition with the failing cells' classes, and the
    non-answer count. After the table: the estimand sentence,
    the operating-characteristics paragraph (exact figures,
    one convention, the family-wise price printed beside the
    licence), the guard paragraph (what incompletion can and
    cannot do, with the executed worlds), and the printed
    limitations. Counts and rates only; no prompt or answer
    text is rendered. Every payload-derived value below goes
    through _escape_text, never bare escape().
    """
    probe: Any = result.get("invisible_chars")
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

    arm_names = (
        "control",
        "zwsp",
        "vs16",
        "tag",
        "softhyphen",
        "wj",
        "visible_hyphen",
    )
    class_names = (
        "pass",
        "boundary_only",
        "wrong_value",
        "echo_invisible",
        "form_only",
        "non_answer",
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
            for arm_name in arm_names:
                arm = arms.get(arm_name)
                if type(arm) is not dict:
                    continue
                decomp = arm.get("decomposition")
                decomp_bit = ""
                if type(decomp) is dict:
                    decomp_bit = (
                        " Decomposition of completed cells -- "
                        + ", ".join(
                            class_name + " " + _count(decomp.get(class_name))
                            for class_name in class_names
                        )
                        + " (in the frozen order: a boundary-only "
                        "cell is a boundary effect, never an "
                        "insertion effect; wrong-value counts "
                        "only values inside the frozen grammar; "
                        "echo and form classes are the model "
                        "reproducing the treated surface with the "
                        "value intact)."
                    )
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
                    + " evaluation-failed."
                    + decomp_bit
                    + "</div>"
                )
        ids = entry.get("ids")
        headline = entry.get("headline")
        if type(ids) is dict and type(headline) is dict:
            for arm_name in arm_names[1:]:
                arm_ids = ids.get(arm_name)
                arm_head = headline.get(arm_name)
                if type(arm_ids) is not dict or type(arm_head) is not dict:
                    continue
                deltas = arm_ids.get("id_length_delta")
                delta_bit = ""
                if type(deltas) is list:
                    delta_bit = (
                        " Per-case id-length deltas (arm minus "
                        "control, in case order): "
                        + _escape_text(", ".join(str(d) for d in deltas))
                        + "."
                    )
                pair = arm_head.get("pair_decomposition")
                pair_bit = ""
                if type(pair) is dict:
                    pair_bit = (
                        " Pair decomposition of the discordant "
                        "pairs -- exact-only "
                        + _count(pair.get("exact_only"))
                        + ", boundary-only "
                        + _count(pair.get("boundary_only"))
                        + ", mixed "
                        + _count(pair.get("mixed"))
                        + " (reported, never attributed)."
                    )
                failing = arm_head.get("failing_classes")
                failing_bit = ""
                if type(failing) is dict:
                    failing_bit = (
                        " The discordant pairs' failing cells "
                        "decompose as "
                        + ", ".join(
                            class_name + " " + _count(failing.get(class_name))
                            for class_name in class_names
                        )
                        + "."
                    )
                bits.append(
                    "<div>"
                    + _escape_text(arm_name)
                    + " headline over "
                    + _count(arm_head.get("n_population"))
                    + " paired cases (completed in both control "
                    "and this arm and id-divergent for this arm; "
                    + _count(arm_ids.get("n_ids_identical"))
                    + " cases were ids-identical for this arm "
                    "-- instrument-inapplicable for those, never "
                    "passes -- and "
                    + _count(arm_head.get("n_incomplete"))
                    + " id-divergent cases were incomplete): "
                    "label "
                    + _escape_text(str(arm_head.get("label")))
                    + ", D (control passes) "
                    + _count(arm_head.get("control_passes"))
                    + " at control rate "
                    + _rate(arm_head.get("control_rate"))
                    + ", J (wrong-value discordants) "
                    + _count(arm_head.get("wrong_values"))
                    + ", reverse discordants "
                    + _count(arm_head.get("n_reverse"))
                    + " (counted, never pooled into J), "
                    "arm pass rate "
                    + _rate(arm_head.get("arm_rate"))
                    + ", non-answer arm cells in the population "
                    + _count(arm_head.get("n_non_answer"))
                    + ". Same-ids verdict flips for this arm: "
                    + _count(arm_ids.get("n_same_ids_verdict_flips"))
                    + " (a determinism finding, never insertion "
                    "fragility); maximum absolute id-length delta "
                    + _count(arm_ids.get("max_abs_id_delta"))
                    + "."
                    + delta_bit
                    + pair_bit
                    + failing_bit
                    + "</div>"
                )
        rows.append(
            '<tr><td><span class="mono">'
            + _escape_text(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )
    oc: dict[str, Any] = {}
    for entry in candidates.values():
        if type(entry) is dict and type(entry.get("operating_characteristics")) is dict:
            oc = entry["operating_characteristics"]
            break

    def _oc(key: str) -> str:
        """Return the oc."""
        return _rate(oc.get(key))

    tail_paragraphs = [
        "<p>The estimand, stated once: value-changing "
        "discordance between the verbatim control and one "
        "treated rendering, over the id-divergent paired cases "
        "both arms completed, for this candidate's template "
        "family on this checkpoint. The claim is per candidate "
        "and per arm; nothing is pooled, and a fire proves only "
        "that this arm moved this checkpoint's answers to wrong "
        "values inside the frozen grammar on this generated "
        "population -- never a claim about weights, attacks, or "
        "code points outside the frozen list. An invisible arm "
        "is read as invisibility-specific only when the visible "
        "hyphen arm does not also fire for the same candidate; "
        "if both fire, the reading is that insertion at these "
        "sites disrupts this candidate, and the invisible arm "
        "adds no invisibility-specific evidence.</p>",
        "<p>Operating characteristics (exact enumeration under "
        "per-case independence, q the arm's marginal in-grammar "
        "wrong-value rate; the paired column couples the two "
        "outcomes within each case at correlation 0.5 and is "
        "computed exactly): the competent panel fires "
        + _oc("fire_competent_independent")
        + " independent and "
        + _oc("fire_competent_paired")
        + " paired; the weak panel (control rate 0.80) fires "
        + _oc("fire_weak_independent")
        + " and the random labeller "
        + _oc("fire_random_independent")
        + " -- the label is comparative evidence at a stated "
        "base rate, which is why each candidate's control rate "
        "prints beside its label above. Power at the 25-point "
        "target effect is "
        + _oc("power_drop25_base90")
        + " at base .90 and "
        + _oc("power_drop25_base85")
        + " at base .85 (paired "
        + _oc("power_drop25_paired")
        + "); a 10-point drop fires only "
        + _oc("power_drop10_base90")
        + " of the time, so an acquittal is an observed null at "
        "this panel's resolution, never proof of invariance. "
        "Family-wise, over the 120 labelled invisible "
        "comparisons the competent expectation is "
        + _rate(oc.get("familywise_competent_expected_120"))
        + " false labels (probability of at least one "
        + _oc("familywise_competent_any_120")
        + "); over all 144 printed labels it is "
        + _rate(oc.get("familywise_competent_expected_144"))
        + "; paired, the 120-comparison expectation is "
        + _rate(oc.get("familywise_paired_expected_120"))
        + " (probability of at least one "
        + _escape_text(str(oc.get("familywise_paired_any_120")))
        + "). The rejected two-directional transplant rule "
        "fires "
        + _oc("transplant_fire_competent_independent")
        + " on the same competent null, which is why the "
        "floors are this instrument's own.</p>",
        "<p>The completion guard, and what incompletion can "
        "and cannot do: fires issue on observed counts over "
        "completed cases, so removing a generation can never "
        "create a wrong-value discordant; acquittal labels "
        "issue only when stable under the worst-case completion "
        "of the pair's incomplete id-divergent cases, so "
        "removal can never manufacture an acquittal either -- "
        "in the executed attack worlds (10,000 trials each), "
        "sign-concentrated incompletion drives the unguarded "
        "rule's below-threshold rate to "
        + _oc("world_b_unguarded_below_threshold")
        + " on a true drop while the guarded rate stays "
        + _oc("world_b_guarded_below_threshold")
        + ", the displaced mass withheld as insufficient. The "
        "price of the guard is withholding: at the amended cap "
        "(incomplete id-divergent cases above 6 of 24 withhold "
        "every label) the manufacture world fires "
        + _oc("world_a_guarded_fire")
        + " and the flat-incompletion drop world "
        + _oc("world_c_guarded_fire")
        + " against its unguarded "
        + _oc("world_c_unguarded_fire")
        + ". Dose escalation in response to a null is outside "
        "this instrument: one code point per site is the frozen "
        "dose, and a null here is a legitimate pre-registered "
        "outcome. Printed limitations: the soft-hyphen arm is "
        "the weakest member of the frozen list (a renderer may "
        "show it at a line-break opportunity; the frozen sites "
        "are never at one); value changes outside the frozen "
        "grammar print as non-answer and are outside the "
        "headline claim; sanitizer cost on clean text is not "
        "measured by this arm set. Records carry digests, "
        "except cells' answer text, which is digest-bound and "
        "never rendered here.</p>",
    ]
    audit_arms: Any = {}
    protocol = probe.get("protocol")
    if type(protocol) is dict:
        population = protocol.get("population_audit")
        if type(population) is dict and type(population.get("arms")) is dict:
            audit_arms = population["arms"]
    legend_parts = []
    for arm in arm_names[1:]:
        entry = audit_arms.get(arm)
        if type(entry) is dict:
            legend_parts.append(f"{arm} ({entry.get('code_point')} {entry.get('name')})")
        else:
            legend_parts.append(arm)
    legend = "<p>The frozen arms: " + _escape_text(", ".join(legend_parts)) + ".</p>"
    return (
        '<section class="card" id="invisible-chars">'
        "<h2>Invisible characters (frozen invisible-code-point insertion)</h2>"
        "<p>This probe inserts one frozen code point at two "
        "frozen sites in a generated fact sentence: each of the "
        "24 generated cases is a fact-recall item (a frozen "
        "topic, a word-number inspection code, two neutral "
        "carrier sentences, a question), rendered verbatim in "
        "the control arm and with a single invisible code point "
        "inserted after the second letter of the topic and of "
        "the code's word in five invisible arms (zero-width "
        "space, variation selector-16, language tag, soft "
        "hyphen, word joiner -- a frozen list of "
        "Default-Ignorable code points in Unicode 15.0.0), plus "
        "a matched visible control inserting a hyphen at the "
        "same sites. The question and carriers are "
        "byte-identical across arms; the scorer is byte-exact "
        "conjoined with boundary presence; every arm is a "
        "fixture with the same id, so request seeds are "
        "identical across arms. Each cell is one greedy "
        "generation. The readout is a guarded classification "
        "per candidate and per arm over the id-divergent "
        "paired population, with the operating characteristics "
        "and the family-wise price printed beside it.</p>"
        + legend
        + "<table><thead><tr><th>Candidate</th><th>Readout</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + "".join(tail_paragraphs)
        + "</section>"
    )


def _confusable_skeleton_section(result: dict[str, Any]) -> str:
    """Confusable-skeleton substitution probe: does replacing
    Latin letters with their UTS #39 skeleton-equivalent
    Cyrillic/Greek letters move answers to wrong values?

    Renders the ``confusable_skeleton`` payload (see
    experiments.confusable_skeleton_analysis). The first
    sentences state the treatment exactly: frozen
    substitutions inside the fact sentence only (the topic and
    the value's word part), the question and carriers
    byte-identical across arms, the population frozen from
    confusables.txt at Unicode 15.0.0. Per candidate, in the
    design's printed order: per-arm pass rates with attrition
    and the seven-class completed-cell decomposition; the
    whole-word dose; the per-arm token-ID report (the regime
    gate); and per treated arm the guarded headline over the
    id-divergent paired population -- label, D, J, reverse
    discordants, incomplete count, the paired control rate
    beside the label, the pair decomposition with the failing
    cells' classes, and the non-answer count. After the table:
    the estimand sentence (with both isolation readings,
    including the whole-word sentence of amendment A5 printed
    verbatim), the operating-characteristics paragraph (exact
    figures, one convention, the family-wise price over the 72
    labelled comparisons printed beside the licence), the
    guard paragraph (what incompletion can and cannot do,
    with the executed worlds and the pinned-domain caveat of
    amendment A8), and the printed limitations. Counts and
    rates only; no prompt or answer text is rendered. Every
    payload-derived value below goes through _escape_text,
    never bare escape().
    """
    probe: Any = result.get("confusable_skeleton")
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

    def _exact(value: Any) -> str:
        """Return the exact."""
        return _escape_text(str(value)) if type(value) in (int, float) else "n/a"

    arm_names = (
        "control",
        "single_site",
        "whole_word",
        "rarity_control",
    )
    class_names = (
        "pass",
        "boundary_only",
        "wrong_value",
        "echo_confusable",
        "echo_control",
        "form_only",
        "non_answer",
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
            for arm_name in arm_names:
                arm = arms.get(arm_name)
                if type(arm) is not dict:
                    continue
                decomp = arm.get("decomposition")
                decomp_bit = ""
                if type(decomp) is dict:
                    decomp_bit = (
                        " Decomposition of completed cells -- "
                        + ", ".join(
                            class_name + " " + _count(decomp.get(class_name))
                            for class_name in class_names
                        )
                        + " (in the frozen order: a boundary-only "
                        "cell is a boundary effect, never a "
                        "substitution effect; wrong-value counts "
                        "only values inside the frozen grammar; "
                        "echo and form classes are the model "
                        "reproducing the treated surface with the "
                        "value intact)."
                    )
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
                    + " evaluation-failed."
                    + decomp_bit
                    + "</div>"
                )
        dose = entry.get("dose")
        if type(dose) is dict and type(dose.get("whole_word")) is dict:
            whole = dose["whole_word"]
            bits.append(
                "<div>Whole-word dose on this panel: "
                + _count(whole.get("eligible_letters"))
                + " of "
                + _count(whole.get("total_letters"))
                + " letters in treated words are eligible "
                "(fraction "
                + _rate(whole.get("fraction"))
                + "); the remaining letters are zero-eligible "
                "in the frozen population and no arm can "
                "substitute them.</div>"
            )
        ids = entry.get("ids")
        headline = entry.get("headline")
        if type(ids) is dict and type(headline) is dict:
            for arm_name in arm_names[1:]:
                arm_ids = ids.get(arm_name)
                arm_head = headline.get(arm_name)
                if type(arm_ids) is not dict or type(arm_head) is not dict:
                    continue
                deltas = arm_ids.get("id_length_delta")
                delta_bit = ""
                if type(deltas) is list:
                    delta_bit = (
                        " Per-case id-length deltas (arm minus "
                        "control, in case order): "
                        + _escape_text(", ".join(str(d) for d in deltas))
                        + "."
                    )
                pair = arm_head.get("pair_decomposition")
                pair_bit = ""
                if type(pair) is dict:
                    pair_bit = (
                        " Pair decomposition of the discordant "
                        "pairs -- exact-only "
                        + _count(pair.get("exact_only"))
                        + ", boundary-only "
                        + _count(pair.get("boundary_only"))
                        + ", mixed "
                        + _count(pair.get("mixed"))
                        + " (reported, never attributed)."
                    )
                failing = arm_head.get("failing_classes")
                failing_bit = ""
                if type(failing) is dict:
                    failing_bit = (
                        " The discordant pairs' failing cells "
                        "decompose as "
                        + ", ".join(
                            class_name + " " + _count(failing.get(class_name))
                            for class_name in class_names
                        )
                        + "."
                    )
                bits.append(
                    "<div>"
                    + _escape_text(arm_name)
                    + " headline over "
                    + _count(arm_head.get("n_population"))
                    + " paired cases (completed in both control "
                    "and this arm and id-divergent for this arm; "
                    + _count(arm_ids.get("n_ids_identical"))
                    + " cases were ids-identical for this arm "
                    "-- instrument-inapplicable for those, never "
                    "passes -- and "
                    + _count(arm_head.get("n_incomplete"))
                    + " id-divergent cases were incomplete): "
                    "label "
                    + _escape_text(str(arm_head.get("label")))
                    + ", D (control passes) "
                    + _count(arm_head.get("control_passes"))
                    + " at control rate "
                    + _rate(arm_head.get("control_rate"))
                    + ", J (wrong-value discordants) "
                    + _count(arm_head.get("wrong_values"))
                    + ", reverse discordants "
                    + _count(arm_head.get("n_reverse"))
                    + " (counted, never pooled into J), "
                    "arm pass rate "
                    + _rate(arm_head.get("arm_rate"))
                    + ", non-answer arm cells in the population "
                    + _count(arm_head.get("n_non_answer"))
                    + ". Same-ids verdict flips for this arm: "
                    + _count(arm_ids.get("n_same_ids_verdict_flips"))
                    + " (a determinism finding, never substitution "
                    "fragility); maximum absolute id-length delta "
                    + _count(arm_ids.get("max_abs_id_delta"))
                    + "."
                    + delta_bit
                    + pair_bit
                    + failing_bit
                    + "</div>"
                )
        rows.append(
            '<tr><td><span class="mono">'
            + _escape_text(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )
    oc: dict[str, Any] = {}
    for entry in candidates.values():
        if type(entry) is dict and type(entry.get("operating_characteristics")) is dict:
            oc = entry["operating_characteristics"]
            break

    def _oc(key: str) -> str:
        """Return the oc."""
        return _rate(oc.get(key))

    tail_paragraphs = [
        "<p>The estimand, stated once: value-changing "
        "discordance between the verbatim control and one "
        "treated rendering, over the id-divergent paired cases "
        "both arms completed, for this candidate's template "
        "family on this checkpoint. The claim is per candidate "
        "and per arm; nothing is pooled, and a fire proves only "
        "that this arm moved this checkpoint's answers to wrong "
        "values inside the frozen grammar on this generated "
        "population -- never a claim about weights, attacks, or "
        "letters outside the frozen population. The claim is "
        "skeleton-class substitution, never that a model is "
        "fooled like a human: the model never sees glyphs. A "
        "single_site fire is read as skeleton-class-specific "
        "only if rarity_control does not also fire for the same "
        "candidate; if both fire, the reading is that non-Latin "
        "substitution at these sites disrupts this candidate, "
        "not that skeleton equality is the mechanism. For "
        "whole_word the isolation reading does not extend: a "
        "whole-word solo fire reads verbatim -- "
        "skeleton-class substitution at whole-word dose "
        "disrupts this candidate; rarity at that dose is "
        "unmeasured by this design.</p>",
        "<p>Operating characteristics (exact enumeration under "
        "per-case independence, q the arm's marginal in-grammar "
        "wrong-value rate; the paired column couples the two "
        "outcomes within each case at correlation 0.5 and is "
        "computed exactly -- the design-phase Monte Carlo "
        "understated it and is not printed): the competent "
        "panel fires "
        + _oc("fire_competent_independent")
        + " independent and "
        + _exact(oc.get("fire_competent_paired"))
        + " paired; the weak panel (control rate 0.80) fires "
        + _oc("fire_weak_independent")
        + " and the random labeller "
        + _oc("fire_random_independent")
        + " -- the label is comparative evidence at a stated "
        "base rate, which is why each candidate's control rate "
        "prints beside its label above. Power at the 25-point "
        "target effect is "
        + _oc("power_drop25_base90")
        + " at base .90 and "
        + _oc("power_drop25_base85")
        + " at base .85 (paired "
        + _exact(oc.get("power_drop25_paired"))
        + "); a 10-point drop fires only "
        + _oc("power_drop10_base90")
        + " of the time, so an acquittal is an observed null at "
        "this panel's resolution, never proof of invariance. "
        "Family-wise, over the 72 labelled comparisons (3 "
        "treated arms x 24 candidates) the competent "
        "expectation is "
        + _rate(oc.get("familywise_competent_expected_72"))
        + " false labels (probability of at least one "
        + _oc("familywise_competent_any_72")
        + "); paired, the expectation is "
        + _exact(oc.get("familywise_paired_expected_72"))
        + " (probability of at least one "
        + _exact(oc.get("familywise_paired_any_72"))
        + "); the random labeller's expectation is "
        + _rate(oc.get("familywise_random_expected_72"))
        + ". The rejected two-directional transplant rule "
        "fires "
        + _oc("transplant_fire_competent_independent")
        + " on the same competent null, which is why the "
        "floors are this instrument's own.</p>",
        "<p>The completion guard, and what incompletion can "
        "and cannot do: fires issue on observed counts over "
        "completed cases, so removing a generation can never "
        "create a wrong-value discordant; acquittal labels "
        "issue only when stable under the worst-case completion "
        "of the pair's incomplete id-divergent cases (the "
        "completion ranges over divergent cases only -- an "
        "encode-time property, decidable whether or not the "
        "generation completed -- and counts each as a control "
        "pass with an arm wrong value, a deliberate blanket "
        "over-count), so removal can never manufacture an "
        "acquittal either -- in the executed attack worlds "
        "(10,000 trials each, ranging at design time over all "
        "incomplete cases, the conservative direction; the "
        "shipped battery re-executes them under the pinned "
        "domain), sign-concentrated incompletion drives the "
        "unguarded rule's below-threshold rate to "
        + _oc("world_b_unguarded_below_threshold")
        + " on a true drop while the guarded rate stays "
        + _oc("world_b_guarded_below_threshold")
        + ", the displaced mass withheld as insufficient. The "
        "price of the guard is withholding: at the frozen cap "
        "(incomplete id-divergent cases above 6 of 24 withhold "
        "every label) the manufacture world fires "
        + _oc("world_a_guarded_fire")
        + " and the flat-incompletion drop world "
        + _oc("world_c_guarded_fire")
        + " against its unguarded "
        + _oc("world_c_unguarded_fire")
        + ". Printed limitations: the population is "
        "single-version by construction (skeletons are not "
        "version-stable; a Unicode bump is a population change, "
        "never a silent upgrade); sequence-level skeleton "
        "equivalence (the rn/m class) is outside the "
        "single-code-point dose axis of v1; value changes "
        "outside the frozen grammar print as non-answer and "
        "are outside the headline claim; screening cost on "
        "clean text is not measured by this arm set. Records "
        "carry digests, except cells' answer text, which is "
        "digest-bound and never rendered here.</p>",
    ]
    audit_letters: Any = {}
    protocol = probe.get("protocol")
    if type(protocol) is dict:
        population = protocol.get("population_audit")
        if type(population) is dict and type(population.get("letters")) is dict:
            audit_letters = population["letters"]
    legend_parts = []
    for letter in sorted(audit_letters, key=str):
        entry = audit_letters.get(letter)
        if type(entry) is dict and entry.get("replacement"):
            legend_parts.append(
                f"{letter} -> {entry.get('replacement')} "
                f"{entry.get('replacement_name')} "
                f"(rarity control {entry.get('control')} "
                f"{entry.get('control_name')})"
            )
    zero_eligible = [
        letter
        for letter in sorted(audit_letters, key=str)
        if type(audit_letters.get(letter)) is dict and not audit_letters[letter].get("replacement")
    ]
    legend = (
        "<p>The frozen population (confusables data, Unicode "
        "15.0.0): "
        + _escape_text("; ".join(legend_parts))
        + ". Zero-eligible letters (no Cyrillic/Greek "
        "small-letter skeleton equivalent): " + _escape_text(", ".join(zero_eligible)) + ".</p>"
    )
    return (
        '<section class="card" id="confusable-skeleton">'
        "<h2>Confusable skeleton (frozen UTS #39 skeleton-class substitution)</h2>"
        "<p>This probe replaces Latin letters in a generated "
        "fact sentence with their UTS #39 skeleton-equivalent "
        "Cyrillic/Greek letters: each of the 24 generated cases "
        "is a fact-recall item (a frozen topic, a word-number "
        "inspection code, two neutral carrier sentences, a "
        "question), rendered verbatim in the control arm and "
        "with substitutions inside the fact sentence only in "
        "three treated arms -- single-site (the first eligible "
        "letter of the topic and of the code's word), "
        "whole-word (every eligible letter), and a rarity "
        "control (the single-site sites filled with a "
        "same-script non-confusable letter: U+0434 for the "
        "Cyrillic seat, U+03B6 for the Greek seat). The "
        "question and carriers are byte-identical across arms; "
        "the scorer is byte-exact conjoined with boundary "
        "presence; every arm is a fixture with the same id, so "
        "request seeds are identical across arms. Each cell is "
        "one greedy generation. The readout is a guarded "
        "classification per candidate and per arm over the "
        "id-divergent paired population, with the operating "
        "characteristics and the family-wise price printed "
        "beside it.</p>" + legend + "<table><thead><tr><th>Candidate</th><th>Readout</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + "".join(tail_paragraphs)
        + "</section>"
    )


def _compatibility_form_section(result: dict[str, Any]) -> str:
    """Compatibility-form substitution probe: does replacing
    one marker field's plain-text reading with its frozen
    compatibility character move answers to wrong values?

    Renders the ``compatibility_form`` payload (see
    experiments.compatibility_form_analysis). The first
    sentences state the treatment exactly: a frozen marker
    sentence inside every generated case hosts one field per
    compatibility class, and each treated arm replaces
    exactly its own field's reading with the frozen
    character from the pinned UnicodeData 15.0.0 population
    (117 characters in six classes). Per candidate, in the
    design's printed order: per-arm pass rates with attrition
    and the six-class completed-cell decomposition; the
    per-arm dose deltas; the per-arm token-ID report (the
    regime gate); and per treated arm the guarded headline
    over the id-divergent paired population -- label with
    the divergent-domain size M printed beside it (design
    amendment A1), D, J, reverse discordants, incomplete
    count, the paired control rate, the pair decomposition
    with the failing cells' classes, and the non-answer
    count. After the table: the estimand sentence, the
    operating-characteristics paragraph (exact figures, one
    convention, the M-conditioned rows, the family-wise
    price over the 144 labelled comparisons printed beside
    the licence), the guard paragraph (what incompletion
    can and cannot do, with the executed worlds and the
    pinned-domain rule), and the printed limitations.
    Counts and rates only; no prompt or answer text is
    rendered. Every payload-derived value below goes
    through _escape_text, never bare escape().
    """
    probe: Any = result.get("compatibility_form")
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

    def _exact(value: Any) -> str:
        """Return the exact."""
        return _escape_text(str(value)) if type(value) in (int, float) else "n/a"

    arm_names = (
        "control",
        "ligature",
        "enclosed",
        "fraction",
        "superscript",
        "roman",
        "katakana",
    )
    class_names = (
        "pass",
        "boundary_only",
        "wrong_value",
        "echo_compat",
        "form_only",
        "non_answer",
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
            for arm_name in arm_names:
                arm = arms.get(arm_name)
                if type(arm) is not dict:
                    continue
                decomp = arm.get("decomposition")
                decomp_bit = ""
                if type(decomp) is dict:
                    decomp_bit = (
                        " Decomposition of completed cells -- "
                        + ", ".join(
                            class_name + " " + _count(decomp.get(class_name))
                            for class_name in class_names
                        )
                        + " (in the frozen order: a boundary-only "
                        "cell is a boundary effect, never a form "
                        "effect; wrong-value counts only values "
                        "inside the frozen grammar; echo-compat "
                        "and form-only are the model rendering the "
                        "expected value in compatibility forms, "
                        "with the value intact, and never count "
                        "toward J)."
                    )
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
                    + " evaluation-failed."
                    + decomp_bit
                    + "</div>"
                )
        dose = entry.get("dose")
        if type(dose) is dict:
            for arm_name in arm_names[1:]:
                arm_dose = dose.get(arm_name)
                if type(arm_dose) is not dict:
                    continue
                bits.append(
                    "<div>" + _escape_text(arm_name) + " dose on this panel: exactly 1 frozen code "
                    "point per case; codepoint delta versus "
                    "control ranges "
                    + _count(arm_dose.get("codepoint_delta_min"))
                    + " to "
                    + _count(arm_dose.get("codepoint_delta_max"))
                    + ", UTF-8 byte delta "
                    + _count(arm_dose.get("byte_delta_min"))
                    + " to "
                    + _count(arm_dose.get("byte_delta_max"))
                    + ".</div>"
                )
        ids = entry.get("ids")
        headline = entry.get("headline")
        if type(ids) is dict and type(headline) is dict:
            for arm_name in arm_names[1:]:
                arm_ids = ids.get(arm_name)
                arm_head = headline.get(arm_name)
                if type(arm_ids) is not dict or type(arm_head) is not dict:
                    continue
                deltas = arm_ids.get("id_length_delta")
                delta_bit = ""
                if type(deltas) is list:
                    delta_bit = (
                        " Per-case id-length deltas (arm minus "
                        "control, in case order): "
                        + _escape_text(", ".join(str(d) for d in deltas))
                        + "."
                    )
                pair = arm_head.get("pair_decomposition")
                pair_bit = ""
                if type(pair) is dict:
                    pair_bit = (
                        " Pair decomposition of the discordant "
                        "pairs -- exact-only "
                        + _count(pair.get("exact_only"))
                        + ", boundary-only "
                        + _count(pair.get("boundary_only"))
                        + ", mixed "
                        + _count(pair.get("mixed"))
                        + " (reported, never attributed)."
                    )
                failing = arm_head.get("failing_classes")
                failing_bit = ""
                if type(failing) is dict:
                    failing_bit = (
                        " The discordant pairs' failing cells "
                        "decompose as "
                        + ", ".join(
                            class_name + " " + _count(failing.get(class_name))
                            for class_name in class_names
                        )
                        + "."
                    )
                bits.append(
                    "<div>"
                    + _escape_text(arm_name)
                    + " headline over "
                    + _count(arm_head.get("n_population"))
                    + " paired cases (completed in both control "
                    "and this arm and id-divergent for this arm; "
                    "the divergent-domain size M is "
                    + _count(arm_head.get("n_ids_divergent"))
                    + ", "
                    + _count(arm_ids.get("n_ids_identical"))
                    + " cases were ids-identical for this arm "
                    "-- instrument-inapplicable for those, never "
                    "passes -- and "
                    + _count(arm_head.get("n_incomplete"))
                    + " id-divergent cases were incomplete): "
                    "label "
                    + _escape_text(str(arm_head.get("label")))
                    + " at M "
                    + _count(arm_head.get("n_ids_divergent"))
                    + ", D (control passes) "
                    + _count(arm_head.get("control_passes"))
                    + " at control rate "
                    + _rate(arm_head.get("control_rate"))
                    + ", J (wrong-value discordants) "
                    + _count(arm_head.get("wrong_values"))
                    + ", reverse discordants "
                    + _count(arm_head.get("n_reverse"))
                    + " (counted, never pooled into J), "
                    "arm pass rate "
                    + _rate(arm_head.get("arm_rate"))
                    + ", non-answer arm cells in the population "
                    + _count(arm_head.get("n_non_answer"))
                    + ". Same-ids verdict flips for this arm: "
                    + _count(arm_ids.get("n_same_ids_verdict_flips"))
                    + " (a determinism finding, never form "
                    "fragility); maximum absolute id-length delta "
                    + _count(arm_ids.get("max_abs_id_delta"))
                    + "."
                    + delta_bit
                    + pair_bit
                    + failing_bit
                    + "</div>"
                )
        rows.append(
            '<tr><td><span class="mono">'
            + _escape_text(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )
    oc: dict[str, Any] = {}
    for entry in candidates.values():
        if type(entry) is dict and type(entry.get("operating_characteristics")) is dict:
            oc = entry["operating_characteristics"]
            break

    def _oc(key: str) -> str:
        """Return the oc."""
        return _rate(oc.get(key))

    tail_paragraphs = [
        "<p>The estimand, stated once: value-changing "
        "discordance between the reading-form control and one "
        "compatibility-form rendering, over the id-divergent "
        "paired cases both arms completed, for this candidate's "
        "template family on this checkpoint. The claim is per "
        "candidate and per arm; nothing is pooled, and a fire "
        "proves only that this class's compatibility forms, at "
        "one field of the carrier marker sentence, moved this "
        "checkpoint's answers to wrong values inside the frozen "
        "grammar on this generated population -- never a claim "
        "about weights, attacks, or characters outside the "
        "frozen population. For the fraction arm under an "
        "NFKC-class normalizer, the form that reached the "
        "model is the pipeline's rewritten fraction-slash form "
        "(the treated character folds to digit, U+2044, digit, "
        "which is neither the treated form nor the ASCII "
        "reading), and a fire is attributable to that rewritten "
        "form against the ASCII reading. A null does not "
        "license claims about compatibility forms in "
        "answer-bearing text: the treated field is carrier "
        "text adjacent to the fact, never the answer-bearing "
        "word.</p>",
        "<p>Operating characteristics (exact enumeration under "
        "per-case independence, q the arm's marginal in-grammar "
        "wrong-value rate; the paired column couples the two "
        "outcomes within each case at correlation 0.5 and is "
        "computed exactly): the competent panel fires "
        + _oc("fire_competent_independent")
        + " independent and "
        + _exact(oc.get("fire_competent_paired"))
        + " paired; the weak panel (control rate 0.80) fires "
        + _oc("fire_weak_independent")
        + " and the random labeller "
        + _oc("fire_random_independent")
        + " -- the label is comparative evidence at a stated "
        "base rate, which is why each candidate's control rate "
        "prints beside its label above. Power at the 25-point "
        "target effect is "
        + _oc("power_drop25_base90")
        + " at base .90 and "
        + _oc("power_drop25_base85")
        + " at base .85 (paired "
        + _exact(oc.get("power_drop25_paired"))
        + "); a 10-point drop at the full domain (M = 24) "
        "fires only "
        + _oc("power_drop10_base90")
        + " of the time, so an acquittal is an observed null "
        "at this panel's resolution, never proof of "
        "invariance. The operating characteristics condition "
        "on the divergent-domain size M printed beside every "
        "label: at M = 20 the 25-point power is "
        + _oc("m20_drop25")
        + " and the 10-point power "
        + _oc("m20_drop10")
        + "; at M = 12 they fall to "
        + _oc("m12_drop25")
        + " and "
        + _oc("m12_drop10")
        + ", while the competent-null fire rate only falls as "
        "M shrinks, so the licence is never overstated at "
        "small M -- an acquittal there is simply weaker "
        "evidence. Family-wise, over the 144 labelled "
        "comparisons (6 treated arms x 24 candidates) the "
        "competent expectation is "
        + _rate(oc.get("familywise_competent_expected_144"))
        + " false labels (probability of at least one "
        + _oc("familywise_competent_any_144")
        + "); paired, the expectation is "
        + _exact(oc.get("familywise_paired_expected_144"))
        + " (probability of at least one "
        + _exact(oc.get("familywise_paired_any_144"))
        + "); the random labeller's expectation is "
        + _rate(oc.get("familywise_random_expected_144"))
        + ". The rejected two-directional transplant rule "
        "fires "
        + _oc("transplant_fire_competent_independent")
        + " on the same competent null, which is why the "
        "floors are this instrument's own.</p>",
        "<p>The completion guard, and what incompletion can "
        "and cannot do: fires issue on observed counts over "
        "completed cases, so removing a generation can never "
        "create a wrong-value discordant; acquittal labels "
        "issue only when stable under the worst-case "
        "completion of the pair's incomplete id-divergent "
        "cases (the completion ranges over divergent cases "
        "only -- an encode-time property, decidable whether "
        "or not the generation completed -- and counts each "
        "as a control pass with an arm wrong value, a "
        "deliberate blanket over-count), so removal can never "
        "manufacture an acquittal either -- in the executed "
        "attack worlds with the ids dimension (10,000 trials "
        "each, every case id-divergent with probability 0.8, "
        "a stated simulation parameter), sign-concentrated "
        "incompletion drives the unguarded rule's "
        "below-threshold rate to "
        + _oc("world_b_unguarded_below_threshold")
        + " on a true drop while the guarded rate stays "
        + _oc("world_b_guarded_below_threshold")
        + ", the displaced mass withheld as insufficient; on "
        "the zero-effect manufacture world the guarded rule "
        "withholds "
        + _oc("world_a_guarded_insufficient")
        + " of trials as insufficient and fires "
        + _oc("world_a_guarded_fire")
        + " against the unguarded "
        + _oc("world_a_unguarded_fire")
        + ". The cap bounds incomplete cases within the "
        "divergent domain at 6; it guarantees at least 18 "
        "paired cases only when the domain is the full 24, "
        "and at smaller M labels issue at paired counts down "
        "to M - 6, priced by the M-conditioned rows above. "
        "Printed limitations: the population is "
        "single-version by construction (a Unicode bump is a "
        "population change, never a silent upgrade); the "
        "fraction reading is a plain-text transcription "
        "(solidus for U+2044), pre-registered, so the "
        "fraction pair is reading-equivalent under the frozen "
        "table rather than NFKC-equal; echo-compat is "
        "reachable only for the enclosed and superscript arms "
        "(their readings are digits, and digits occur in "
        "every expected value) -- for the other four arms the "
        "compatibility-rendering split is carried entirely by "
        "form-only, and echo rates are never compared across "
        "arms; value changes outside the frozen grammar print "
        "as non-answer and are outside the headline claim. "
        "Records carry digests, except cells' answer text, "
        "which is digest-bound and never rendered here.</p>",
    ]
    audit_classes: Any = {}
    protocol = probe.get("protocol")
    if type(protocol) is dict:
        population = protocol.get("population_audit")
        if type(population) is dict and type(population.get("classes")) is dict:
            audit_classes = population["classes"]
    legend_parts = []
    for class_name in arm_names[1:]:
        entry = audit_classes.get(class_name)
        if type(entry) is dict and type(entry.get("members")) is list:
            legend_parts.append(
                class_name
                + " ("
                + _count(entry.get("count"))
                + "): "
                + ", ".join(str(member) for member in entry["members"])
            )
    legend = (
        "<p>The frozen population (UnicodeData.txt, Unicode "
        "15.0.0; every member satisfies NFC == NFD == itself "
        "and folds under NFKC; the fullwidth class is excluded "
        "by rule): " + _escape_text("; ".join(legend_parts)) + ".</p>"
    )
    return (
        '<section class="card" id="compatibility-form">'
        "<h2>Compatibility form (frozen UAX #15 compatibility-class substitution)</h2>"
        "<p>This probe replaces one field of a frozen marker "
        "sentence with its compatibility-form character: each "
        "of the 24 generated cases is a fact-recall item (a "
        "frozen topic, a word-number inspection code, two "
        "neutral carrier sentences, a marker sentence hosting "
        "one field per compatibility class, a question), "
        "rendered in the control arm with every marker field "
        "in its plain-text reading and in six treated arms "
        "with exactly one field -- the arm's own class -- "
        "carrying the frozen compatibility character instead "
        "(a ligature, an enclosed digit, a vulgar fraction, a "
        "superscript digit, a Roman numeral, a halfwidth "
        "katakana letter). The fact, the question, and the "
        "carriers are byte-identical across arms; the scorer "
        "is byte-exact conjoined with boundary presence; every "
        "arm is a fixture with the same id, so request seeds "
        "are identical across arms. Each cell is one greedy "
        "generation. The readout is a guarded classification "
        "per candidate and per arm over the id-divergent "
        "paired population, with the operating "
        "characteristics and the family-wise price printed "
        "beside it.</p>" + legend + "<table><thead><tr><th>Candidate</th><th>Readout</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + "".join(tail_paragraphs)
        + "</section>"
    )


def _enumeration_convention_section(result: dict[str, Any]) -> str:
    """Enumeration-convention probe: does the marker system
    a list is enumerated under move answers to wrong values?

    Renders the ``enumeration_convention`` payload (see
    experiments.enumeration_convention_analysis). The first
    sentences state the treatment exactly: a frozen
    nine-item list inside every generated case is enumerated
    under one of six marker systems, and the count is never
    stated anywhere in the content. Per candidate, in the
    design's printed order: per-arm pass rates with
    attrition and the six-class completed-cell
    decomposition; the per-arm token-ID report (the regime
    gate); per treated arm the guarded headline over the
    id-divergent paired population -- label with the
    divergent-domain size M printed beside it, D, J,
    reverse discordants, incomplete count, the paired
    control rate, the pair decomposition with the failing
    cells' classes, and the non-answer count; and the
    by-position join (amendment A5): the mean id-length
    delta by target position beside the by-position D and
    J, so a length explanation is checkable against the
    label. After the table: the estimand sentence, the
    operating-characteristics paragraph (exact figures,
    one convention, the sensitivity rows, the family-wise
    price over the 120 labelled comparisons printed beside
    the licence), the guard paragraph (what incompletion
    can and cannot do, with the executed worlds and the
    pinned-domain rule), the length-pattern reading, and
    the printed limitations. Counts and rates only; no
    prompt or answer text is rendered. Every
    payload-derived value below goes through _escape_text,
    never bare escape().
    """
    probe: Any = result.get("enumeration_convention")
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

    def _exact(value: Any) -> str:
        """Return the exact."""
        return _escape_text(str(value)) if type(value) in (int, float) else "n/a"

    arm_names = (
        "control_arabic",
        "alpha_upper",
        "alpha_lower",
        "roman",
        "wrapper",
        "bullet",
    )
    class_names = (
        "pass",
        "boundary_only",
        "wrong_value",
        "echo",
        "form_only",
        "non_answer",
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
            for arm_name in arm_names:
                arm = arms.get(arm_name)
                if type(arm) is not dict:
                    continue
                decomp = arm.get("decomposition")
                decomp_bit = ""
                if type(decomp) is dict:
                    decomp_bit = (
                        " Decomposition of completed cells -- "
                        + ", ".join(
                            class_name + " " + _count(decomp.get(class_name))
                            for class_name in class_names
                        )
                        + " (in the frozen order: a boundary-only "
                        "cell is a boundary effect, never a form "
                        "effect; wrong-value counts only values "
                        "inside the frozen grammar, bare or wearing "
                        "exactly one marker-shaped prefix or "
                        "suffix; echo and form-only are the model "
                        "rendering the expected value with the "
                        "list's marker form attached, the value "
                        "intact, and never count toward J)."
                    )
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
                    + " evaluation-failed."
                    + decomp_bit
                    + "</div>"
                )
        ids = entry.get("ids")
        headline = entry.get("headline")
        positions = entry.get("positions")
        if type(ids) is dict and type(headline) is dict:
            for arm_name in arm_names[1:]:
                arm_ids = ids.get(arm_name)
                arm_head = headline.get(arm_name)
                if type(arm_ids) is not dict or type(arm_head) is not dict:
                    continue
                deltas = arm_ids.get("id_length_delta")
                delta_bit = ""
                if type(deltas) is list:
                    delta_bit = (
                        " Per-case id-length deltas (arm minus "
                        "control, in case order): "
                        + _escape_text(", ".join(str(d) for d in deltas))
                        + "."
                    )
                pair = arm_head.get("pair_decomposition")
                pair_bit = ""
                if type(pair) is dict:
                    pair_bit = (
                        " Pair decomposition of the discordant "
                        "pairs -- exact-only "
                        + _count(pair.get("exact_only"))
                        + ", boundary-only "
                        + _count(pair.get("boundary_only"))
                        + ", mixed "
                        + _count(pair.get("mixed"))
                        + " (reported, never attributed)."
                    )
                failing = arm_head.get("failing_classes")
                failing_bit = ""
                if type(failing) is dict:
                    failing_bit = (
                        " The discordant pairs' failing cells "
                        "decompose as "
                        + ", ".join(
                            class_name + " " + _count(failing.get(class_name))
                            for class_name in class_names
                        )
                        + "."
                    )
                position_bit = ""
                if type(positions) is dict and type(positions.get(arm_name)) is list:
                    position_rows = positions[arm_name]
                    position_bit = (
                        " By target position (paired population; "
                        "position, cases, control passes, "
                        "wrong-value discordants, mean id-length "
                        "delta): "
                        + _escape_text(
                            "; ".join(
                                "position "
                                + str(row.get("position"))
                                + ": "
                                + str(row.get("n_population"))
                                + " cases, "
                                + str(row.get("control_passes"))
                                + " passes, "
                                + str(row.get("wrong_values"))
                                + " wrong values, mean delta "
                                + (
                                    f"{float(row['mean_id_length_delta']):.3f}"
                                    if type(row.get("mean_id_length_delta")) in (int, float)
                                    and _float_convertible(row["mean_id_length_delta"])
                                    else "n/a"
                                )
                                for row in position_rows
                                if type(row) is dict
                            )
                        )
                        + "."
                    )
                bits.append(
                    "<div>"
                    + _escape_text(arm_name)
                    + " headline over "
                    + _count(arm_head.get("n_population"))
                    + " paired cases (completed in both control "
                    "and this arm and id-divergent for this arm; "
                    "the divergent-domain size M is "
                    + _count(arm_head.get("n_ids_divergent"))
                    + ", "
                    + _count(arm_ids.get("n_ids_identical"))
                    + " cases were ids-identical for this arm "
                    "-- instrument-inapplicable for those, never "
                    "passes -- and "
                    + _count(arm_head.get("n_incomplete"))
                    + " id-divergent cases were incomplete): "
                    "label "
                    + _escape_text(str(arm_head.get("label")))
                    + " at M "
                    + _count(arm_head.get("n_ids_divergent"))
                    + ", D (control passes) "
                    + _count(arm_head.get("control_passes"))
                    + " at control rate "
                    + _rate(arm_head.get("control_rate"))
                    + ", J (wrong-value discordants) "
                    + _count(arm_head.get("wrong_values"))
                    + ", reverse discordants "
                    + _count(arm_head.get("n_reverse"))
                    + " (counted, never pooled into J), "
                    "arm pass rate "
                    + _rate(arm_head.get("arm_rate"))
                    + ", non-answer arm cells in the population "
                    + _count(arm_head.get("n_non_answer"))
                    + ". Same-ids verdict flips for this arm: "
                    + _count(arm_ids.get("n_same_ids_verdict_flips"))
                    + " (a determinism finding, never convention "
                    "fragility); maximum absolute id-length delta "
                    + _count(arm_ids.get("max_abs_id_delta"))
                    + "."
                    + delta_bit
                    + pair_bit
                    + failing_bit
                    + position_bit
                    + "</div>"
                )
        rows.append(
            '<tr><td><span class="mono">'
            + _escape_text(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )
    oc: dict[str, Any] = {}
    for entry in candidates.values():
        if type(entry) is dict and type(entry.get("operating_characteristics")) is dict:
            oc = entry["operating_characteristics"]
            break

    def _oc(key: str) -> str:
        """Return the oc."""
        return _rate(oc.get(key))

    tail_paragraphs = [
        "<p>The estimand, stated once: value-changing "
        "discordance between the arabic control and one "
        "enumeration convention, over the id-divergent paired "
        "cases both arms completed, for this candidate's "
        "template family on this checkpoint. The claim is per "
        "candidate and per arm; nothing is pooled, and a fire "
        "proves only that this marker system, as the carrier "
        "list's enumeration, moved this checkpoint's answers "
        "to wrong values inside the frozen grammar on this "
        "generated population -- never a claim about weights, "
        "attacks, or marker systems outside the frozen six. "
        "A null does not license claims about enumeration in "
        "answer-bearing text: the markers enumerate carrier "
        "items adjacent to the fact, and the answer itself is "
        "never enumerated.</p>",
        "<p>Operating characteristics (exact enumeration under "
        "per-case independence, q the arm's marginal in-grammar "
        "wrong-value rate; the paired column couples the two "
        "outcomes within each case at correlation 0.5 and is "
        "computed exactly): the competent panel fires "
        + _oc("fire_competent_independent")
        + " independent and "
        + _exact(oc.get("fire_competent_paired"))
        + " paired; the weak panel (control rate 0.80) fires "
        + _oc("fire_weak_independent")
        + " and the random labeller "
        + _oc("fire_random_independent")
        + " -- the label is comparative evidence at a stated "
        "base rate, which is why each candidate's control rate "
        "prints beside its label above. Power at the 25-point "
        "target effect is "
        + _oc("power_drop25_base90")
        + " at base .90 and "
        + _oc("power_drop25_base85")
        + " at base .85 (paired "
        + _exact(oc.get("power_drop25_paired"))
        + "); a 10-point drop fires only "
        + _oc("power_drop10_base90")
        + " of the time and a 5-point drop "
        + _oc("power_drop5_base90")
        + ", so an acquittal is an observed null at this "
        "panel's resolution, never proof of invariance and "
        "never a few-point null. The operating characteristics "
        "condition on the divergent-domain size M printed "
        "beside every label: at M = 18 the 25-point power is "
        + _oc("sensitivity18_drop25")
        + " and the competent fire "
        + _oc("sensitivity18_competent")
        + "; at M = 12 they fall to "
        + _oc("sensitivity12_drop25")
        + " and "
        + _oc("sensitivity12_competent")
        + ", while the competent-null fire rate only falls as "
        "M shrinks, so the licence is never overstated at "
        "small M -- an acquittal there is simply weaker "
        "evidence. Family-wise, over the 120 labelled "
        "comparisons (5 treated arms x 24 candidates) the "
        "competent expectation is "
        + _rate(oc.get("familywise_competent_expected_120"))
        + " false labels (probability of at least one "
        + _oc("familywise_competent_any_120")
        + "); paired, the expectation is "
        + _exact(oc.get("familywise_paired_expected_120"))
        + " (probability of at least one "
        + _exact(oc.get("familywise_paired_any_120"))
        + "); the random labeller's expectation is "
        + _rate(oc.get("familywise_random_expected_120"))
        + ". The rejected two-directional transplant rule "
        "fires "
        + _oc("transplant_fire_competent_independent")
        + " on the same competent null, which is why the "
        "floors are this instrument's own. The corrected "
        "required-N computation (an erratum to the executed "
        "search, recorded in the design document) puts the "
        "smallest panel licensing 80 percent power at a "
        "10-point drop at N = 117 cases with floors (0, 18) "
        "-- roughly five times this panel's cost -- which is "
        "why the shipped panel is sized and priced for the "
        "25-point target only.</p>",
        "<p>The completion guard, and what incompletion can "
        "and cannot do: fires issue on observed counts over "
        "completed cases, so removing a generation can never "
        "create a wrong-value discordant; acquittal labels "
        "issue only when stable under the worst-case "
        "completion of the pair's incomplete id-divergent "
        "cases (the completion ranges over divergent cases "
        "only -- an encode-time property, decidable whether "
        "or not the generation completed -- and counts each "
        "as a control pass with an arm wrong value, a "
        "deliberate blanket over-count), so removal can never "
        "manufacture an acquittal either -- in the executed "
        "attack worlds with the ids dimension (10,000 trials "
        "each, every case id-divergent with probability 0.8, "
        "a stated simulation parameter), sign-concentrated "
        "incompletion drives the unguarded rule's "
        "below-threshold rate to "
        + _oc("world_b_unguarded_below_threshold")
        + " on a true drop while the guarded rate stays "
        + _oc("world_b_guarded_below_threshold")
        + ", the displaced mass withheld as insufficient; on "
        "the zero-effect manufacture world the guarded rule "
        "withholds "
        + _oc("world_a1_guarded_insufficient")
        + " of trials as insufficient and fires "
        + _oc("world_a1_guarded_fire")
        + " against the unguarded "
        + _oc("world_a1_unguarded_fire")
        + "; under flat incompletion the guarded rule still "
        "fires "
        + _oc("world_c_guarded_fire")
        + " on the true drop. The cap bounds incomplete cases "
        "within the divergent domain at 6; it guarantees at "
        "least 18 paired cases only when the domain is the "
        "full 24, and at smaller M labels issue at paired "
        "counts down to M - 6, priced by the M-conditioned "
        "rows above. Length-pattern reading: the Roman "
        "markers' character lengths by position are 2, 3, 4, "
        "3, 2, 3, 4, 5, 3 -- position 8 carries the longest "
        "marker and position 9 a short one -- so a "
        "Roman-arm effect concentrated at the long-marker "
        "positions with a null at position 9 reads as "
        "length-correlated, while one spread evenly or "
        "peaking at position 9 does not; the by-position join "
        "above prints the mean id-length delta beside the "
        "by-position D and J for exactly this check, for "
        "every arm. Printed limitations: the count is never "
        "stated, so the probe measures enumeration-dependent "
        "retrieval, not counting ability; the list length is "
        "frozen at nine items and the panel at 24 cases; "
        "value changes outside the frozen grammar print as "
        "non-answer and are outside the headline claim; "
        "marker interpretation in the decomposition is "
        "ASCII-pinned (a non-ASCII digit head is a "
        "non-answer, never a parsed marker). Records carry "
        "digests, except cells' answer text, which is "
        "digest-bound and never rendered here.</p>",
    ]
    return (
        '<section class="card" id="enumeration-convention">'
        "<h2>Enumeration convention (marker-system substitution on a frozen list)</h2>"
        "<p>This probe changes only the marker system a "
        "frozen list is enumerated under: each of the 24 "
        "generated cases is a fact-recall item (a nine-item "
        "list of inspection codes under a frozen intro line, "
        "two neutral carrier sentences, a question), rendered "
        "in the arabic control arm with markers 1. to 9. and "
        "in five treated arms under upper alpha (A. to I.), "
        "lower alpha (a. to i.), Roman (I. to IX.), "
        "parenthesised wrapper ((1) to (9)), and bullet "
        "(index-free). The count is never stated anywhere in "
        "the content, so the answer can only be found by "
        "enumerating the list; the items, their order, the "
        "question, and the carriers are byte-identical "
        "across arms; the scorer is byte-exact conjoined "
        "with boundary presence; every arm is a fixture with "
        "the same id, so request seeds are identical across "
        "arms. Each cell is one greedy generation. The "
        "readout is a guarded classification per candidate "
        "and per arm over the id-divergent paired "
        "population, with the operating characteristics and "
        "the family-wise price printed beside it.</p>"
        "<table><thead><tr><th>Candidate</th><th>Readout</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + "".join(tail_paragraphs)
        + "</section>"
    )


def _typographic_emphasis_section(result: dict[str, Any]) -> str:
    """Typographic-emphasis probe: does wrapping one
    topic span of a fact carrier in bold, italic,
    monospace, or capitals move answers to wrong
    values -- and can a candidate earn the certificate
    that it does not?

    Renders the ``typographic_emphasis`` payload (see
    experiments.typographic_emphasis_analysis). Per
    candidate, in the design's printed order: per-arm
    pass rates with attrition and the completed-cell
    decomposition (the five main arms); the per-arm
    dose deltas; the positive-control block (the
    bold-detection panel's counts and gate verdict);
    the per-arm token-ID report (the regime gate); and
    per treated arm the combined guarded headline over
    the id-divergent paired population -- the label
    with the divergent-domain size M, D, J, H, R, S,
    the incomplete count, the rates, the non-answer
    count, the pair decomposition with the failing
    cells' classes, and the certification companions
    (delta, alpha, the positive-control gate's status,
    and the gates that blocked a certificate). After
    the table: the estimand sentence, the
    operating-characteristics paragraph (the
    certification licence with every near-0.0122 figure
    under its full name, the fire grid, both
    family-wise prices, the positive control's deaf
    range), the guard paragraph, the section-8 readings,
    and the printed limitations. Counts and rates only;
    no prompt or answer text is rendered. Every
    payload-derived value below goes through
    _escape_text, never bare escape().
    """
    probe: Any = result.get("typographic_emphasis")
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

    def _exact(value: Any) -> str:
        """Return the exact."""
        return _escape_text(str(value)) if type(value) in (int, float) else "n/a"

    main_arms = ("control", "bold", "italic", "monospace", "caps")
    record_arms = main_arms + ("pc_plain", "pc_info", "pc_b")
    class_names = (
        "pass",
        "boundary_only",
        "wrong_value",
        "echo_bold",
        "echo_italic",
        "echo_monospace",
        "echo_caps",
        "form_only",
        "non_answer",
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
            for arm_name in record_arms:
                arm = arms.get(arm_name)
                if type(arm) is not dict:
                    continue
                decomp = arm.get("decomposition")
                decomp_bit = ""
                if type(decomp) is dict:
                    decomp_bit = (
                        " Decomposition of completed cells -- "
                        + ", ".join(
                            class_name + " " + _count(decomp.get(class_name))
                            for class_name in class_names
                        )
                        + " (in the frozen order: a boundary-only "
                        "cell is a boundary effect, never a form "
                        "effect; wrong-value counts only values "
                        "inside the frozen grammar; echo and "
                        "form-only are the model rendering the "
                        "expected value wearing the arm's own "
                        "treatment, with the value intact, and "
                        "never count toward J)."
                    )
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
                    + " evaluation-failed."
                    + decomp_bit
                    + "</div>"
                )
        dose = entry.get("dose")
        if type(dose) is dict:
            for arm_name in main_arms[1:]:
                arm_dose = dose.get(arm_name)
                if type(arm_dose) is not dict:
                    continue
                bits.append(
                    "<div>" + _escape_text(arm_name) + " dose on this panel: the fact "
                    "sentence's topic span only; codepoint delta versus "
                    "control ranges "
                    + _count(arm_dose.get("codepoint_delta_min"))
                    + " to "
                    + _count(arm_dose.get("codepoint_delta_max"))
                    + ", UTF-8 byte delta "
                    + _count(arm_dose.get("byte_delta_min"))
                    + " to "
                    + _count(arm_dose.get("byte_delta_max"))
                    + " (bold adds four asterisks, italic and "
                    "monospace two marks each, caps none -- "
                    "bold and italic differ only in asterisk "
                    "dose, the same character twice or four "
                    "times).</div>"
                )
        pc = entry.get("positive_control")
        if type(pc) is dict:
            bits.append(
                "<div>Positive control (which of the two facts' "
                "topics is written in bold; the plain arm has no "
                "correct answer by construction): "
                + _count(pc.get("n_pairs"))
                + " completed pairs, "
                + _count(pc.get("n_incomplete"))
                + " incomplete, info-arm passes "
                + _count(pc.get("info_passes"))
                + ", plain-fail/info-pass pairs "
                + _count(pc.get("plain_fail_info_pass"))
                + ", plain-pass/info-fail pairs "
                + _count(pc.get("plain_pass_info_fail"))
                + ", margin "
                + _count(pc.get("margin"))
                + "; gate "
                + ("passed" if pc.get("passed") is True else "not passed")
                + " (floors: at least 10 info passes, margin at "
                "least 8, and the margin stable under the "
                "worst-case completion of the incomplete "
                "pairs). No certificate issues for this "
                "candidate without this gate.</div>"
            )
        ids = entry.get("ids")
        headline = entry.get("headline")
        if type(ids) is dict and type(headline) is dict:
            for arm_name in main_arms[1:]:
                arm_ids = ids.get(arm_name)
                arm_head = headline.get(arm_name)
                if type(arm_ids) is not dict or type(arm_head) is not dict:
                    continue
                deltas = arm_ids.get("id_length_delta")
                delta_bit = ""
                if type(deltas) is list:
                    delta_bit = (
                        " Per-case id-length deltas (arm minus "
                        "control, in case order): "
                        + _escape_text(", ".join(str(d) for d in deltas))
                        + "."
                    )
                pair = arm_head.get("pair_decomposition")
                pair_bit = ""
                if type(pair) is dict:
                    pair_bit = (
                        " Pair decomposition of the discordant "
                        "pairs -- exact-only "
                        + _count(pair.get("exact_only"))
                        + ", boundary-only "
                        + _count(pair.get("boundary_only"))
                        + ", mixed "
                        + _count(pair.get("mixed"))
                        + " (reported, never attributed)."
                    )
                failing = arm_head.get("failing_classes")
                failing_bit = ""
                if type(failing) is dict:
                    failing_bit = (
                        " The discordant pairs' failing cells "
                        "decompose as "
                        + ", ".join(
                            class_name + " " + _count(failing.get(class_name))
                            for class_name in class_names
                        )
                        + "."
                    )
                blocked = arm_head.get("certification_blocked_by")
                blocked_bit = ""
                if type(blocked) is list:
                    blocked_bit = (
                        " Certification gates blocking a "
                        "certificate for this arm: "
                        + (_escape_text(", ".join(str(b) for b in blocked)) if blocked else "none")
                        + "."
                    )
                bits.append(
                    "<div>"
                    + _escape_text(arm_name)
                    + " headline over "
                    + _count(arm_head.get("n_population"))
                    + " paired cases (completed in both control "
                    "and this arm and id-divergent for this arm; "
                    "the divergent-domain size M is "
                    + _count(arm_head.get("n_ids_divergent"))
                    + ", "
                    + _count(arm_ids.get("n_ids_identical"))
                    + " cases were ids-identical for this arm "
                    "-- instrument-inapplicable for those, never "
                    "passes -- and "
                    + _count(arm_head.get("n_incomplete"))
                    + " id-divergent cases were incomplete): "
                    "label "
                    + _escape_text(str(arm_head.get("label")))
                    + " at M "
                    + _count(arm_head.get("n_ids_divergent"))
                    + ", D (control passes) "
                    + _count(arm_head.get("control_passes"))
                    + " at control rate "
                    + _rate(arm_head.get("control_rate"))
                    + ", J (wrong-value discordants) "
                    + _count(arm_head.get("wrong_values"))
                    + ", H (control-pass arm-failures, every "
                    "class) "
                    + _count(arm_head.get("harm"))
                    + ", reverse discordants "
                    + _count(arm_head.get("n_reverse"))
                    + " (counted, never pooled into J), net harm "
                    "S "
                    + _count(arm_head.get("net_harm"))
                    + ", arm pass rate "
                    + _rate(arm_head.get("arm_rate"))
                    + ", non-answer arm cells in the population "
                    + _count(arm_head.get("n_non_answer"))
                    + ". The label's licence: delta "
                    + _exact(arm_head.get("delta"))
                    + " at alpha "
                    + _exact(arm_head.get("alpha"))
                    + ", positive-control gate "
                    + ("passed" if arm_head.get("pc_gate_passed") is True else "not passed")
                    + ". Same-ids verdict flips for this arm: "
                    + _count(arm_ids.get("n_same_ids_verdict_flips"))
                    + " (a determinism finding, never form "
                    "fragility); maximum absolute id-length delta "
                    + _count(arm_ids.get("max_abs_id_delta"))
                    + "."
                    + delta_bit
                    + pair_bit
                    + failing_bit
                    + blocked_bit
                    + "</div>"
                )
        rows.append(
            '<tr><td><span class="mono">'
            + _escape_text(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )
    oc: dict[str, Any] = {}
    for entry in candidates.values():
        if type(entry) is dict and type(entry.get("operating_characteristics")) is dict:
            oc = entry["operating_characteristics"]
            break

    def _oc(key: str) -> str:
        """Return the oc."""
        return _rate(oc.get(key))

    tail_paragraphs = [
        "<p>The estimand, stated once: value-changing "
        "discordance, and net harm over every failure "
        "class, between the verbatim control and one "
        "typographic treatment of the fact sentence's "
        "topic span, over the id-divergent paired cases "
        "both arms completed, for this candidate's "
        "template family on this checkpoint. The claim "
        "is per candidate and per arm; nothing is pooled "
        "and there is no winner readout. A fire proves "
        "only that this treatment, at this one carrier "
        "span, moved this checkpoint's answers to wrong "
        "values inside the frozen grammar on this "
        "generated population. A certificate asserts "
        "exactly one thing: the treatment's true marginal "
        "drop is below delta 0.34, under the licence "
        "printed below; it is silent below 0.34 -- it "
        "does not exclude, for example, the published "
        "PASTA-study average effects of 0.1791 and 0.2514 "
        "-- and it says nothing about highlight or "
        "colour treatments, marks used as selectors, "
        "churn among cases both arms fail, ids-identical "
        "cases, other spans, other doses, or treatments "
        "of instruction segments.</p>",
        "<p>Operating characteristics. The certification "
        "licence (per comparison, alpha 0.02, stated over "
        "the heterogeneous class -- per-pair joint laws "
        "arbitrary, the null a pooled marginal drop of at "
        "least delta): the completed extremal bound at "
        "delta 0.34 is "
        + _exact(oc.get("certification_bound_completed_per_comparison"))
        + " per comparison. Three neighbouring figures "
        "are printed under their full names so none can "
        "be mistaken for the licence: the restricted-vertex "
        "heterogeneous bound at delta 0.34 is "
        + _exact(oc.get("certification_bound_restricted_vertex_delta_034"))
        + "; the homogeneous uniform bound at delta 0.34 "
        "(context only, never the licence) is "
        + _exact(oc.get("certification_bound_uniform_homogeneous_delta_034"))
        + "; and a real-table counterexample world "
        "certifies at "
        + _exact(oc.get("rtb_counterexample_certification_probability"))
        + " at delta, which is why the licence is stated "
        "over the completed class. Partial domains: the "
        "completed bound at M = 23 is "
        + _exact(oc.get("certification_bound_completed_m23"))
        + " (the unique partial domain the level "
        "supports) and at M = 22 is "
        + _exact(oc.get("certification_bound_completed_m22"))
        + ", above alpha -- the shipped gate nevertheless "
        "requires the full M = 24, because a guarded "
        "certificate at M implies the full-data event at "
        "size 24 under worst-case completion, by "
        "componentwise monotonicity of (D, H, S). "
        "Family-wise over the 96 labelled comparisons (4 "
        "treated arms x 24 candidates), under the "
        "independence convention, the certification "
        "expectation is "
        + _exact(oc.get("familywise_certification_expected_96"))
        + " false certificates (probability of at least "
        "one "
        + _exact(oc.get("familywise_certification_any_96"))
        + ") and the competent fire expectation is "
        + _exact(oc.get("familywise_fire_expected_96"))
        + " (probability of at least one "
        + _exact(oc.get("familywise_fire_any_96"))
        + "). The fire rule's own grid (exact enumeration "
        "under per-case independence; the paired column "
        "couples the outcomes within each case at "
        "correlation 0.5): the competent panel fires "
        + _exact(oc.get("fire_competent_independent"))
        + " independent and "
        + _exact(oc.get("fire_competent_paired"))
        + " paired; the weak panel (control rate 0.80) "
        "fires "
        + _oc("fire_weak_independent")
        + " and the random labeller "
        + _exact(oc.get("fire_random_independent"))
        + "; power is "
        + _oc("power_drop10_base90")
        + " at a 10-point drop, "
        + _oc("power_drop25_base90")
        + " at the 25-point target (paired "
        + _exact(oc.get("power_drop25_paired"))
        + "), and "
        + _oc("power_drop50_base90")
        + " at 50 points. Under the anchored coupling "
        "(independence at control rate 0.90) a true-null "
        "candidate certifies with probability "
        + _exact(oc.get("certify_anchored_delta_000_pc_090"))
        + " -- the null yield: the predicted-null "
        "instrument certifies the null it predicts, at "
        "the stated level, rather than merely failing to "
        "reject it. The positive control's deaf-guesser "
        "probability of passing its gate is "
        + _exact(oc.get("pc_gate_deaf_x_025_independence"))
        + " at the independence point of the "
        "discordant-cell parameter x (x is the "
        "probability of each discordant cell, plain-fail/"
        "info-pass and plain-pass/info-fail alike, with "
        "both-pass and both-fail carrying 0.5 - x each); "
        "along the dependence range it rises to "
        + _exact(oc.get("pc_gate_deaf_x_030"))
        + " at x = 0.30, "
        + _exact(oc.get("pc_gate_deaf_x_035"))
        + " at x = 0.35, and "
        + _exact(oc.get("pc_gate_deaf_x_050"))
        + " at x = 0.50, so the gate's licence is read "
        "as independence-conditioned.</p>",
        "<p>The completion guard, and what incompletion "
        "can and cannot do: fires issue on observed "
        "counts over completed cases, so removing a "
        "generation can never create a wrong-value "
        "discordant; acquittals and certificates issue "
        "only when stable under the worst-case completion "
        "of the pair's incomplete id-divergent cases (the "
        "completion ranges over divergent cases only -- "
        "an encode-time property, decidable whether or "
        "not the generation completed); in the executed "
        "attack worlds with the ids dimension (10,000 "
        "trials each at the designer's seed), the "
        "zero-effect manufacture world certifies guarded "
        "at "
        + _oc("world_a_guarded_certification")
        + " and fires guarded at "
        + _oc("world_a_guarded_fire")
        + ", the true-delta world's certification falls "
        "from "
        + _oc("world_b_delta_unguarded_certification")
        + " unguarded to "
        + _oc("world_b_delta_guarded_certification")
        + " guarded, and the true-drop world fires "
        "guarded at "
        + _oc("world_c_guarded_fire")
        + ". The cap bounds incomplete cases within the "
        "divergent domain at 6. One disclosed inversion: "
        "a certificate can issue at J = 0 with k > 0 "
        "where the sibling no-discordance acquittal "
        "would withhold (245 such states in the "
        "exhaustive audit) -- the certificate is the "
        "stronger claim and carries the completion guard "
        "as its own stability condition, so the inversion "
        "is a property of the two claims, not a hole. "
        "Readings, frozen in advance: a bold fire beside "
        "an italic null (or the reverse) is an "
        "asterisk-dose response, the two arms differing "
        "only in dose of the same character; a "
        "monospace-only fire reads as a code-span effect "
        "(the emphasis reading is withheld: code spans "
        "carry their own semantics); a caps fire with "
        "the mark arms quiet is a case-channel effect; "
        "all four firing is surface fragility, not four "
        "findings; the PC-B arm's rate beside bold's is "
        "a dose and site-count signal for a future "
        "design, never a contradiction of the bold "
        "label. Printed limitations: one treated "
        "occurrence per case by construction, so "
        "multi-site house styles are out of scope; quote "
        "marks belong to the punctuation axis and "
        "highlight and colour have no plain-text "
        "representation, so both sit outside the claim; "
        "value changes outside the frozen grammar print "
        "as non-answer and are outside the headline "
        "claim. Records carry digests, except cells' "
        "answer text, which is digest-bound and never "
        "rendered here.</p>",
    ]
    return (
        '<section class="card" id="typographic-emphasis">'
        "<h2>Typographic emphasis (span treatment on a frozen fact carrier)</h2>"
        "<p>This probe changes only the typographic "
        "treatment of one span: each of the 24 generated "
        "cases is a fact-recall item (a frozen topic, a "
        "word-number inspection code, two neutral "
        "carrier sentences, a fact sentence, a question), "
        "rendered in five main arms that differ only in "
        "the fact sentence's topic span -- verbatim "
        "control, bold (**span**), italic (*span*), "
        "monospace (`span`), or the span uppercased -- "
        "with the question's topic occurrence verbatim in "
        "every main arm, so exactly one occurrence is "
        "treated. A positive-control panel asks which of "
        "two facts' topics is written in bold (the main "
        "case's value under its own topic, plain or bold, "
        "beside a distractor fact), and a PC-B arm wraps "
        "the topic in ***...*** at both occurrences, "
        "reported descriptively. The scorer is byte-exact "
        "conjoined with boundary presence; every arm is a "
        "fixture with the same id, so request seeds are "
        "identical across arms. Each cell is one greedy "
        "generation. The readout is a combined guarded "
        "classification per candidate and per arm over "
        "the id-divergent paired population -- fire, "
        "sibling acquittal, or a level-controlled "
        "certificate that the treatment's marginal drop "
        "is below delta 0.34 -- with the operating "
        "characteristics and the family-wise prices "
        "printed beside it.</p>"
        "<table><thead><tr><th>Candidate</th><th>Readout</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + "".join(tail_paragraphs)
        + "</section>"
    )
