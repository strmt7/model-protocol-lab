"""reporting runlevel family: split from reporting.py (W7-2).

Facades keep the public surface: import from model_protocol_lab.reporting, not from this module.
"""

import re
from html import escape
from typing import Any

from .reporting_core import (
    _FRONTIER_CHART_H,
    _FRONTIER_CHART_W,
    _escape_text,
    _fixture_total,
    _float_convertible,
    _margin_chart_svg_block,
    _sorted_str_keys,
    _spark_svg,
    _valid_rate,
)

MAX_FRONTIER_STEP_POINTS = 100_000


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
        steps = data.get("pass_steps")
        for point in steps if type(steps) is list else ():
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


_SAFE_FLOOR_ABSENT: Any = object()


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
    # Shape before consumption (RR-02): a non-list flip_ranges is
    # treated as absent -- the "never raises" contract above covers
    # truthy scalars too, which the `or []` idiom iterated or raised on.
    flip_ranges = frontier.get("flip_ranges")
    for span in flip_ranges if type(flip_ranges) is list else ():
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
    if contamination.get("budget_exhausted") is True:
        return (
            "partial: sweep stopped at its time budget -- "
            "unevaluated held-out fixtures are not cleared (advisory)"
        )
    if worst == "clean":
        return "clean: no search/held-out near-duplicate pairs"
    if type(worst) is str and count:
        return (
            f"{worst}: {count} near-duplicate pair(s) flagged -- "
            "held-out split may be contaminated (advisory)"
        )
    return "not recorded"


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
        + _escape_text(key)
        + '</div><div class="v">'
        + _escape_text(value)
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


_DECISION_RIBBON_COLORS = {
    "selected": "#16a34a",
    "ambiguous": "#d97706",
    "no_candidate": "#64748b",
}


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
        steps = data.get("pass_steps")
        for point in steps if type(steps) is list else ():
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
    raw_steps = frontier.get("decision_steps")
    raw_decisions = raw_steps if type(raw_steps) is list else []
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
        """Map a budget to its x pixel coordinate."""
        return (budget - 1) * width // denom_x

    # One shared Y axis over the largest fixture total.
    y_total = max(total for _, total, _, _ in series)

    def y_of(count: int) -> int:
        """Map a value to its y pixel coordinate."""
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
        raw_steps = frontier.get("decision_steps")
        steps = raw_steps if type(raw_steps) is list else []
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
                    and type(entry["span"][0]) is int
                    and type(entry["span"][1]) is int
                    and type(entry.get("crossings")) is list
                ):
                    attribution[(entry["span"][0], entry["span"][1])] = entry["crossings"]
        items = []
        flip_ranges = frontier.get("flip_ranges")
        for span in flip_ranges if type(flip_ranges) is list else ():
            if (
                type(span) is not list
                or len(span) != 2
                or type(span[0]) is not int
                or type(span[1]) is not int
            ):
                continue
            lo = span[0]
            flip_decision, flip_winner = "?", None
            for point in steps:
                if (
                    type(point) is list
                    and len(point) == 3
                    and type(point[0]) is int
                    and point[0] <= lo
                ):
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
            if not _float_convertible(count):
                # The percent below divides count by the fixture
                # total; a count beyond float range is a malformed
                # point and is skipped like any other (wave-8 F5).
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
            rate_text = (
                "n/a"
                if type(rate) not in (int, float) or not _float_convertible(rate)
                else f"{rate:.1%}"
            )
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
    overall_text = (
        "n/a"
        if type(overall) not in (int, float) or not _float_convertible(overall)
        else f"{overall:.1%}"
    )
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
        '<section class="card" id="selection-robustness">'
        "<h2>Selection robustness under stop semantics</h2>"
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
            + _escape_text(cid)
            + "</span>"
            + (
                '<br><span class="mono">' + _escape_text(str(fingerprint)[:16]) + "…</span>"
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
        '<section class="card" id="transfer"><h2>Cross-checkpoint transfer</h2>'
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
        '<section class="card" id="perturbation-robustness">'
        "<h2>Fixture-perturbation robustness</h2>"
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


def _token_healing_section(result: dict[str, Any]) -> str:
    """Token-healing selection sensitivity: does healing flip the winner?

    Every search candidate re-evaluated with HF-style token healing
    applied to each prompt (search split only); the selection rule
    re-applied to the healed verdicts. The falsification table tests the
    lab's own boundary-canonicality audit as a predictor: only
    seam-fragile candidates should change verdict under healing.
    Renders nothing when the token-healing experiment did not run.
    """
    healing: Any = result.get("token_healing")
    if type(healing) is not dict or not healing.get("available"):
        return ""
    per_candidate = healing.get("per_candidate")
    falsification = healing.get("falsification")
    if type(per_candidate) is not dict or type(falsification) is not dict:
        return ""
    primary_winner = healing.get("primary_winner")
    primary_decision = healing.get("primary_decision")
    healed_winner = healing.get("healed_winner")
    healed_decision = healing.get("healed_decision")
    flipped = healing.get("flipped") is True

    def _name(value: Any) -> str:
        """Return the name."""
        return "(none)" if value is None else str(value)

    verdict_line = (
        "Healed verdict: <strong>"
        + escape(_name(healed_decision))
        + " / "
        + escape(_name(healed_winner))
        + "</strong> vs primary "
        + escape(_name(primary_decision))
        + " / "
        + escape(_name(primary_winner))
    )
    if flipped:
        verdict_line += ' <span class="badge fail">flipped</span>'
    else:
        verdict_line += ' <span class="badge pass">stable</span>'

    cand_rows = []
    for cand in sorted(str(c) for c in per_candidate):
        entry = per_candidate[cand]
        if type(entry) is not dict:
            continue
        native = entry.get("native")
        healed_v = entry.get("healed")
        changed = entry.get("verdict_changed")
        cell = (
            '<span class="badge fail">changed</span>'
            if changed is True
            else '<span class="badge pass">same</span>'
            if changed is False
            else '<span class="badge">n/a</span>'
        )
        cand_rows.append(
            '<tr><td><span class="mono">'
            + escape(cand)
            + "</span></td><td>"
            + escape(str(native))
            + "</td><td>"
            + escape(str(healed_v))
            + "</td><td>"
            + cell
            + "</td></tr>"
        )

    fals_rows = []
    fals_per_candidate = falsification.get("per_candidate")
    if type(fals_per_candidate) is dict:
        for cand in sorted(str(c) for c in fals_per_candidate):
            entry = fals_per_candidate[cand]
            if type(entry) is not dict:
                continue
            fragile = entry.get("seam_fragile")
            changed = entry.get("verdict_changed")
            holds = entry.get("prediction_holds")
            badge = (
                '<span class="badge pass">holds</span>'
                if holds is True
                else '<span class="badge fail">violated</span>'
                if holds is False
                else '<span class="badge">n/a</span>'
            )
            fals_rows.append(
                '<tr><td><span class="mono">'
                + escape(cand)
                + "</span></td><td>"
                + escape(str(fragile))
                + "</td><td>"
                + escape(str(changed))
                + "</td><td>"
                + badge
                + "</td></tr>"
            )
    n_comparable = falsification.get("n_comparable")
    n_holds = falsification.get("n_holds")
    if type(n_comparable) is int and type(n_holds) is int and n_comparable:
        fals_line = (
            f"Boundary-audit prediction holds for {n_holds}/{n_comparable} comparable candidates."
        )
    else:
        fals_line = "Boundary-audit prediction not comparable on this run."

    n_healed = healing.get("n_healed")
    n_fallback = healing.get("n_fallback")
    heal_line = ""
    if type(n_healed) is int and type(n_fallback) is int:
        heal_line = (
            "<p>"
            + escape(str(n_healed))
            + " prompts healed, "
            + escape(str(n_fallback))
            + " fell back to native ids.</p>"
        )

    flips = healing.get("fixture_flips")
    n_flips = len(flips) if type(flips) is list else 0
    return (
        '<section class="card" id="token-healing"><h2>Token-healing selection sensitivity</h2>'
        "<p>Every search candidate re-evaluated with HF-style token healing "
        "applied to each prompt (the tail token is re-chosen among "
        "vocabulary extensions of its decoded text; search split only); the "
        "selection rule re-applied to the healed verdicts. A flipped "
        "(decision, winner) means the winner depended on greedy "
        "tokenization of the prompt tail.</p>"
        "<p>"
        + verdict_line
        + "</p>"
        + heal_line
        + "<p>"
        + escape(fals_line)
        + (f" {n_flips} per-fixture verdict flip(s)." if n_flips else "")
        + "</p>"
        + (
            '<div class="table-wrap"><table><thead><tr><th scope="col">Candidate</th>'
            '<th scope="col">Native pass</th><th scope="col">Healed pass</th>'
            '<th scope="col">Verdict</th>'
            "</tr></thead><tbody>" + "".join(cand_rows) + "</tbody></table></div>"
            if cand_rows
            else ""
        )
        + (
            '<div class="table-wrap"><table><thead><tr><th scope="col">Candidate</th>'
            '<th scope="col">Seam-fragile (boundary audit)</th>'
            '<th scope="col">Verdict changed</th><th scope="col">Prediction</th>'
            "</tr></thead><tbody>" + "".join(fals_rows) + "</tbody></table></div>"
            if fals_rows
            else ""
        )
        + "</section>"
    )


_STOPCAUSE_CAUSES = ("eos_id", "stop_string", "max_length", "stop_unknown")


def _stopcause_section(result: dict[str, Any]) -> str:
    """Stop-cause termination audit: how would each generation have stopped?

    Every search candidate's recorded generations are classified
    post-hoc into eos_id (the family's end-of-turn token id was
    emitted), stop_string (the HF stop-string emulation on the
    candidate's boundary text fired), max_length (the cap was reached
    with no mechanism firing), or stop_unknown (fail-closed: neither
    the EOT id nor a stop text was available). Reports per-candidate
    termination-cause rates, the would-stop-early rate, and the
    over-generation bleed (tokens the lab's non-stopping decoder
    emitted past the first EOT token, which no real engine would
    emit). Renders nothing when the stop-cause probe did not run.
    """
    audit: Any = result.get("stop_cause")
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

    def _num(value: Any) -> str:
        """Return the num."""
        return str(value) if type(value) is int else "n/a"

    rows = []
    for key in sorted(per_candidate, key=str):
        cand = str(key)
        entry = per_candidate[key]
        if type(entry) is not dict:
            continue
        rates = entry.get("cause_rates")
        if type(rates) is not dict:
            rates = {}
        cells = "".join(
            "<td>" + escape(_rate(rates.get(cause))) + "</td>" for cause in _STOPCAUSE_CAUSES
        )
        rows.append(
            '<tr><td><span class="mono">'
            + escape(cand)
            + "</span></td>"
            + cells
            + "<td>"
            + escape(_rate(entry.get("would_stop_early_rate")))
            + "</td><td>"
            + escape(_rate(entry.get("over_generation_rate")))
            + "</td><td>"
            + escape(_num(entry.get("n_evaluable")))
            + "</td></tr>"
        )
    overall = audit.get("overall")
    overall_line = ""
    if type(overall) is dict:
        overall_rates = overall.get("cause_rates")
        if type(overall_rates) is not dict:
            overall_rates = {}
        overall_line = (
            "<p>Overall termination-cause distribution: "
            + ", ".join(
                "<strong>" + escape(cause) + "</strong> " + escape(_rate(overall_rates.get(cause)))
                for cause in _STOPCAUSE_CAUSES
            )
            + "; would-stop-early rate <strong>"
            + escape(_rate(overall.get("would_stop_early_rate")))
            + "</strong>.</p>"
        )
    n_text = ""
    max_tokens = audit.get("max_new_tokens")
    if type(max_tokens) is int:
        n_text = f" (cap {max_tokens} tokens)"
    return (
        '<section class="card" id="stop-cause"><h2>Stop-cause termination audit</h2>'
        "<p>Post-hoc classification of what a stop-aware engine would have "
        "done with each recorded generation: <strong>eos_id</strong> (the "
        "family's end-of-turn token was emitted), <strong>stop_string</strong> "
        "(the boundary stop text fired under HF stop-string semantics), "
        "<strong>max_length</strong> (the cap was reached with no mechanism "
        "firing), or <strong>stop_unknown</strong> (neither mechanism was "
        "evaluable). Over-generation is the share of generations with tokens "
        "past the first end-of-turn token -- output the lab's non-stopping "
        "decoder emits that no real engine would."
        + escape(n_text)
        + "</p>"
        + overall_line
        + (
            '<div class="table-wrap"><table><caption>Termination-cause rates '
            "per candidate.</caption>"
            '<thead><tr><th scope="col">Candidate</th>'
            '<th scope="col">eos_id</th><th scope="col">stop_string</th>'
            '<th scope="col">max_length</th><th scope="col">stop_unknown</th>'
            '<th scope="col">Would stop early</th>'
            '<th scope="col">Over-generated</th><th scope="col">Evaluable</th>'
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            if rows
            else ""
        )
        + "</section>"
    )


def _deliforge_section(result: dict[str, Any]) -> str:
    """Delimiter authority: forged markers, reserved vs subword encoding.

    Renders the ``deliforge`` payload (see experiments.deliforge_authority)
    as a per-candidate table of authority gaps: the reserved-minus-subword
    task-verdict flip rate of forged chat-template markers, held at
    identical decoded text. The pre-registered falsifiable claim is a gap
    of ~0 uniform across families; a positive gap replicates
    arXiv:2609.35932's reserved-token authority finding inside the lab.
    "Authority" is measured strictly as verdict-flip rate on fixed
    fixtures, not open-ended harm. Every interpolation is escaped at
    this boundary.
    """
    audit: Any = result.get("deliforge")
    if type(audit) is not dict or not audit.get("available"):
        return ""
    per_candidate = audit.get("per_candidate")
    if type(per_candidate) is not dict:
        return ""

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
            + escape(_gap(entry.get("mean_authority_gap")))
            + "</td><td>"
            + escape(_gap(entry.get("max_authority_gap")))
            + "</td><td>"
            + escape(_gap(entry.get("min_authority_gap")))
            + "</td></tr>"
        )
    excluded = audit.get("excluded_candidates")
    excluded_line = ""
    if type(excluded) is list and excluded:
        excluded_line = (
            "<p>Candidates without an evaluable cell (no reserved encoding "
            "found, localization failed, or no native verdicts): "
            + ", ".join(
                '<span class="mono">' + escape(str(name)) + "</span>"
                for name in excluded
                if type(name) is str
            )
            + ".</p>"
        )
    top = audit.get("max_authority_gap")
    top_line = ""
    if type(top) in (int, float) and _float_convertible(top):
        top_line = (
            "<p>Largest reserved-minus-subword flip-rate gap over all "
            "evaluable candidates: <strong>"
            + escape(f"{top:+.2f}")
            + "</strong>. A positive gap means the forged marker flipped "
            "task verdicts more often as the family's reserved control "
            "token than as subword tokens of the identical text -- the "
            "paper's authority claim, replicated per family.</p>"
        )
    marker_note = ""
    marker_indexes = audit.get("marker_indexes")
    if type(marker_indexes) is list and marker_indexes:
        marker_note = (
            "<p>Markers probed: "
            + escape(", ".join(str(i) for i in marker_indexes))
            + " (index into the lab's fixed marker list). Each cell covers "
            "one forged-turn payload and one marker; only the forged "
            "occurrences were re-encoded -- the template's own genuine "
            "markers were never touched.</p>"
        )
    return (
        '<section class="card" id="delimiter-authority">'
        "<h2>Delimiter authority: reserved vs subword encoding</h2>"
        "<p>A forged chat-template marker (e.g. &lt;|im_start|&gt;) was "
        "injected into user content and sent to every search candidate in "
        "two encodings of <em>identical decoded text</em>: the family's "
        "native single reserved control token, and forced subword tokens. "
        "The verdict-flip rate of each arm was measured against the native "
        "search verdicts. <strong>Authority gap</strong> = reserved-arm "
        "flip rate minus subword-arm flip rate, in [-1, 1].</p>"
        + top_line
        + (
            '<div class="table-wrap"><table><caption>Authority gaps per candidate.</caption>'
            '<thead><tr><th scope="col">Candidate</th>'
            '<th scope="col">Evaluable cells</th><th scope="col">Mean gap</th>'
            '<th scope="col">Max gap</th><th scope="col">Min gap</th>'
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            if rows
            else ""
        )
        + excluded_line
        + marker_note
        + "</section>"
    )


def _ablation_section(result: dict[str, Any]) -> str:
    """Template field ablation (cycle 167).

    Renders the ``ablation`` payload as a per-candidate, per-field table:
    for each structural field, the number of verdict flips when the field
    is replaced by its neutral value, the flip directions, and whether the
    arm still clears the conservative selection bar. Load-bearing fields
    (base passes all, arm does not) are highlighted. Informational only:
    the section states it never changes winner selection, that attribution
    is causal only within the measured (template, fixtures, backend)
    triple, and that only main effects are measured. Every interpolation
    is escaped at this boundary.
    """

    _control_chars = re.compile(
        r"[\x00-\x1f\x7f-\x9f\ud800-\udfff\u200e\u200f\u2028\u2029\u202a-\u202e\u2066-\u2069]"
    )

    audit: Any = result.get("ablation")
    if type(audit) is not dict or not audit.get("available"):
        return ""
    candidates = audit.get("candidates")
    if type(candidates) is not dict or not candidates:
        return ""

    def _display_id(raw: Any) -> str:
        """Return the display id."""
        return escape(_control_chars.sub("", str(raw)))

    blocks: list[str] = []
    for name in sorted(candidates):
        rep = candidates[name]
        if type(rep) is not dict:
            continue
        if rep.get("status") != "ok":
            blocks.append(
                "<h3>"
                + _display_id(name)
                + "</h3><p>unavailable: "
                + _display_id(rep.get("reason"))
                + "</p>"
            )
            continue
        arms = rep.get("arms")
        rows: list[str] = []
        load_bearing = rep.get("load_bearing_fields")
        load_set = set(load_bearing) if type(load_bearing) is list else set()
        if type(arms) is dict:
            for field in sorted(arms):
                arm = arms[field]
                if type(arm) is not dict or arm.get("status") != "ok":
                    rows.append(
                        "<tr><td>" + escape(str(field)) + "</td>"
                        '<td colspan="4">unavailable</td></tr>'
                    )
                    continue
                marker = " &#9733;" if field in load_set else ""
                rows.append(
                    "<tr><td>"
                    + escape(str(field))
                    + marker
                    + "</td><td>"
                    + escape(str(arm.get("n_flips")))
                    + "</td><td>"
                    + escape(str(arm.get("pass_to_fail")))
                    + " / "
                    + escape(str(arm.get("fail_to_pass")))
                    + "</td><td>"
                    + ("yes" if arm.get("selection_bar_held") else "no")
                    + "</td><td>"
                    + escape(", ".join(arm.get("flipped_fixture_ids") or ()))
                    + "</td></tr>"
                )
        table = (
            "<table><thead><tr><th>Field</th><th>Verdict flips</th>"
            "<th>pass&rarr;fail / fail&rarr;pass</th>"
            "<th>Selection bar held</th><th>Flipped fixtures</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table>"
        )
        note = ""
        if not rep.get("any_attribution"):
            note = (
                "<p><em>No single field explains this candidate's verdicts: "
                "ablation arms show ~zero marginal attribution. This negative "
                "result is the finding.</em></p>"
            )
        blocks.append(
            "<h3>" + _display_id(name) + "</h3>"
            "<p>Base accuracy: " + escape(str(rep.get("base_accuracy"))) + "; "
            "load-bearing fields: "
            + escape(", ".join(load_set) if load_set else "none")
            + ".</p>"
            + note
            + table
        )
    return (
        '<section class="card" id="field-ablation">'
        "<h2>Template field ablation</h2>"
        "<p>Which structural field drives the verdict? For the top-2 "
        "candidates, one <em>Format</em> field at a time is replaced by its "
        "neutral value and the candidate re-evaluated; &#9733; marks a "
        "<em>load-bearing</em> field (the base passes every search fixture "
        "but the arm does not). Attribution is causal only within the "
        "measured (template, fixtures, backend) triple; only main effects "
        "are measured, never pairwise interactions. Informational only -- "
        "it never changes winner selection.</p>" + "".join(blocks) + "</section>"
    )


def _template_robustness_section(result: dict[str, Any]) -> str:
    """Per-template surface-noise robustness (cycle 166).

    Renders the ``template_robustness`` payload as a per-candidate table:
    the verdict preservation rate (fraction of pass-baseline fixtures with
    no pass->fail flip under presentation-level surface noise), per-operator
    preservation vectors, denominators, and the scope note. Informational
    only: the section states it never changes winner selection, that the
    number is a template x backend-tokenizer interaction, and that nearest
    observed flips are upper bounds. Every interpolation is escaped at this
    boundary.
    """

    _control_chars = re.compile(
        r"[\x00-\x1f\x7f-\x9f\ud800-\udfff\u200e\u200f\u2028\u2029\u202a-\u202e\u2066-\u2069]"
    )

    audit: Any = result.get("template_robustness")
    if type(audit) is not dict or not audit.get("available"):
        return ""
    candidates = audit.get("candidates")
    if type(candidates) is not dict or not candidates:
        return ""

    def _rate(value: Any) -> str:
        """Return the rate."""
        if type(value) in (int, float):
            try:
                return f"{value:.2f}"
            except OverflowError:
                return "n/a"
        return "n/a"

    def _display_id(raw: Any) -> str:
        # Strip control characters (terminal-escape injection) then HTML-escape.
        """Return the display id."""
        return escape(_control_chars.sub("", str(raw)))

    rows: list[str] = []
    for name in sorted(candidates):
        rep = candidates[name]
        if type(rep) is not dict:
            continue
        if rep.get("status") != "ok":
            rows.append(
                "<tr><td>"
                + _display_id(name)
                + "</td><td>n/a</td><td>n/a</td><td>"
                + _display_id(rep.get("status"))
                + "</td></tr>"
            )
            continue
        ops = rep.get("operators")
        op_cells: list[str] = []
        if type(ops) is dict:
            for op in ("case_flip", "whitespace", "unicode_variant"):
                op_rep = ops.get(op)
                if type(op_rep) is dict:
                    op_cells.append(escape(str(op)) + ": " + _rate(op_rep.get("preservation_rate")))
        rows.append(
            "<tr><td>"
            + _display_id(name)
            + "</td><td>"
            + _rate(rep.get("preservation_rate"))
            + "</td><td>"
            + escape(str(rep.get("n_preserved")))
            + "/"
            + escape(str(rep.get("n_preservation_denominator")))
            + "</td><td>"
            + "; ".join(op_cells)
            + "</td></tr>"
        )
    table = (
        "<table><thead><tr><th>Template</th><th>Preservation rate</th>"
        "<th>Preserved / denominator</th><th>Per-operator</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )
    return (
        '<section class="card" id="template-robustness">'
        "<h2>Template surface-noise robustness</h2>"
        "<p>Verdict preservation rate per candidate template: the fraction "
        "of its <em>correct</em> fixture verdicts with no pass-to-fail flip "
        "under presentation-level surface noise (whitespace, case, unicode "
        "variants) applied to user message contents. Informational only -- "
        "it never changes winner selection. The number is a template x "
        "backend-tokenizer interaction, valid only within identical "
        "(fixtures, backend, protocol) triples; nearest observed flips are "
        "upper bounds, never guarantees.</p>" + table + "</section>"
    )


def _verbosity_section(result: dict[str, Any]) -> str:
    """Response verbosity profile per template family.

    Renders the ``verbosity`` payload (see
    experiments.verbosity_profile) as a per-candidate table of excess
    token distributions: for every brevity-ideal fixture (short stable
    expected answer), excess = generated tokens minus the exact
    tokenization of the expected answer, clamped at zero. Median, IQR,
    mean and max excess travel together so a gap cannot hide behind a
    long-tailed family looking typical at the median; the
    budget-saturation share marks candidates whose measured lengths
    are lower bounds (generations hit the token budget -- the
    stop-cause confound, reported not hidden). This is a cost/behavior
    profile: shorter is not better, and the section says so. Every
    interpolation is escaped at this boundary.
    """
    audit: Any = result.get("verbosity")
    if type(audit) is not dict or not audit.get("available"):
        return ""
    per_candidate = audit.get("per_candidate")
    if type(per_candidate) is not dict:
        return ""

    def _num(value: Any) -> str:
        """Return the num."""
        if type(value) in (int, float):
            try:
                return f"{value:.1f}"
            except OverflowError:
                return "n/a"
        return "n/a"

    def _share(value: Any) -> str:
        """Return the share."""
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
            + escape(str(entry.get("n_evaluable")))
            + " / "
            + escape(str(entry.get("n")))
            + "</td><td>"
            + escape(_num(entry.get("median_excess")))
            + "</td><td>"
            + escape(_num(entry.get("p25_excess")))
            + "–"
            + escape(_num(entry.get("p75_excess")))
            + "</td><td>"
            + escape(_num(entry.get("iqr_excess")))
            + "</td><td>"
            + escape(_num(entry.get("mean_excess")))
            + "</td><td>"
            + escape(_num(entry.get("max_excess")))
            + "</td><td>"
            + escape(_share(entry.get("capped_share")))
            + "</td></tr>"
        )
    gap = audit.get("median_excess_gap")
    gap_line = ""
    if type(gap) in (int, float) and _float_convertible(gap):
        gap_line = (
            "<p>Max-minus-min median excess over evaluable candidates: "
            "<strong>" + escape(f"{gap:+.1f}") + " tokens</strong>.</p>"
        )
    extremes_line = ""
    most = audit.get("most_verbose")
    least = audit.get("least_verbose")
    if type(most) is list and most and type(least) is list and least:
        extremes_line = (
            "<p>Most verbose (highest median excess): "
            + ", ".join(
                '<span class="mono">' + escape(str(name)) + "</span>"
                for name in most
                if type(name) is str
            )
            + "; least verbose: "
            + ", ".join(
                '<span class="mono">' + escape(str(name)) + "</span>"
                for name in least
                if type(name) is str
            )
            + ".</p>"
        )
    capped = audit.get("capped_candidates")
    capped_line = ""
    if type(capped) is list and capped:
        capped_line = (
            "<p>Candidates with budget-saturated generations (measured "
            "lengths are <em>lower bounds</em> on true verbosity): "
            + ", ".join(
                '<span class="mono">' + escape(str(name)) + "</span>"
                for name in capped
                if type(name) is str
            )
            + ". See the stop-cause audit for the exact max-length "
            "termination classification.</p>"
        )
    excluded = audit.get("excluded_candidates")
    excluded_line = ""
    if type(excluded) is list and excluded:
        excluded_line = (
            "<p>Candidates with no evaluable verbosity observation: "
            + ", ".join(
                '<span class="mono">' + escape(str(name)) + "</span>"
                for name in excluded
                if type(name) is str
            )
            + ".</p>"
        )
    n_brevity = audit.get("n_brevity_fixtures")
    return (
        '<section class="card" id="verbosity">'
        "<h2>Response verbosity profile</h2>"
        "<p>For each fixture with a short stable expected answer "
        "(brevity-ideal: at most 32 characters, single line), excess = "
        "generated tokens minus the exact tokenization of the expected "
        "answer, clamped at zero. The question is whether the "
        "<em>template</em> -- not the model or the prompt -- decides how "
        "much the model generates, with direct inference-cost "
        "consequences. This is a behavior/cost profile, <strong>not a "
        "quality judgment</strong>: shorter is not better.</p>"
        "<p>Brevity-ideal fixtures measured: "
        + escape(str(n_brevity))
        + ".</p>"
        + gap_line
        + extremes_line
        + capped_line
        + (
            '<div class="table-wrap"><table>'
            "<caption>Excess generated tokens per candidate (tokens).</caption>"
            '<thead><tr><th scope="col">Candidate</th>'
            '<th scope="col">Evaluable</th><th scope="col">Median</th>'
            '<th scope="col">P25–P75</th><th scope="col">IQR</th>'
            '<th scope="col">Mean</th><th scope="col">Max</th>'
            '<th scope="col">Capped</th>'
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            if rows
            else ""
        )
        + excluded_line
        + "</section>"
    )


def _refusals_section(result: dict[str, Any]) -> str:
    """Suspected-refusal profile per candidate template.

    Renders the ``refusals`` payload (see experiments.refusal_profile)
    as a per-candidate table of suspected-refusal rates on benign
    search fixtures: recorded answers classified against the frozen
    refusal-opener list (see :mod:`refusals`). Rates travel with
    their exact counts (k of n) -- a rate on a small denominator is
    never a bare headline. The rates are named *suspected*: the
    detector is beginning-anchored string matching, the field's
    known-weak baseline, and a hedged answer that fails its fixture
    is indistinguishable from a true refusal at this tier. Hits name
    the fixture and the opener that fired; the answer text itself is
    never rendered here (it lives in the evidence-bearing results,
    keyed by fixture). Every interpolation is escaped at this
    boundary.
    """
    audit: Any = result.get("refusals")
    if type(audit) is not dict or not audit.get("available"):
        return ""
    candidates = audit.get("candidates")
    if type(candidates) is not list:
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
    hit_items = []
    unavailable_names = []
    for entry in candidates:
        if type(entry) is not dict:
            continue
        name = entry.get("candidate")
        label = escape(str(name)) if type(name) is str and name else "unknown"
        if not entry.get("available"):
            unavailable_names.append(label)
            continue
        rate_text = _rate(entry.get("suspected_rate"))
        note = entry.get("rate_note")
        if type(note) is str and note:
            rate_text += " (" + escape(note.replace("_", " ")) + ")"
        rows.append(
            '<tr><td><span class="mono">'
            + label
            + "</span></td><td>"
            + escape(str(entry.get("n_completed")))
            + "</td><td>"
            + escape(str(entry.get("n_boundary_found")))
            + "</td><td>"
            + escape(str(entry.get("n_suspected")))
            + " — "
            + rate_text
            + "</td><td>"
            + escape(str(entry.get("n_suspected_passed")))
            + "</td><td>"
            + escape(str(entry.get("n_suspected_failed")))
            + "</td></tr>"
        )
        hits = entry.get("hits")
        if type(hits) is list:
            for hit in hits:
                if type(hit) is not dict:
                    continue
                fixture = hit.get("fixture")
                opener = hit.get("opener")
                hit_items.append(
                    "<li>"
                    + '<span class="mono">'
                    + label
                    + "</span> · fixture "
                    + '<span class="mono">'
                    + (escape(fixture) if type(fixture) is str else "unknown")
                    + "</span> · opener "
                    + '<span class="mono">'
                    + (escape(opener) if type(opener) is str else "unknown")
                    + "</span> · "
                    + ("passed" if hit.get("passed") is True else "failed")
                    + "</li>"
                )
    mismatch_line = ""
    if audit.get("evaluable_mismatch") is True:
        mismatch_line = (
            "<p><strong>Caution:</strong> candidates were not all "
            "measured on the same completed fixture set (an interrupted "
            "candidate is measured on a prefix); compare rates across "
            "candidates with care.</p>"
        )
    hits_block = ""
    if hit_items:
        hits_block = (
            "<p>Suspected-refusal hits (the recorded answer for each "
            "fixture is in the evidence-bearing results file):</p><ul>"
            + "".join(hit_items)
            + "</ul>"
        )
    unavailable_line = ""
    if unavailable_names:
        unavailable_line = (
            "<p>Candidates with no usable recorded observations: "
            + ", ".join('<span class="mono">' + name + "</span>" for name in unavailable_names)
            + ".</p>"
        )
    return (
        '<section class="card" id="refusals">'
        "<h2>Suspected-refusal profile</h2>"
        "<p>Does the <em>template</em> -- not the model -- change how "
        "often the checkpoint refuses benign questions? Each "
        "candidate's recorded search answers are classified against a "
        "frozen, published refusal-opener list (beginning-anchored, "
        "after normalisation). A hit is a <strong>suspected</strong> "
        "refusal: string matching both over-counts (empathy openers) "
        "and under-counts (refusals that open differently), and no "
        "published accuracy figure transfers to this fixture set. "
        "Suspected-and-passed hits are detector artifacts on "
        "echo-style fixtures whose expected answer itself opens with "
        "an opener; suspected-and-failed hits are the "
        "decision-relevant cell. Informational only -- this profile "
        "never affects selection.</p>"
        + mismatch_line
        + (
            '<div class="table-wrap"><table>'
            "<caption>Suspected refusals per candidate (completed "
            "search observations).</caption>"
            '<thead><tr><th scope="col">Candidate</th>'
            '<th scope="col">Completed</th>'
            '<th scope="col">Boundary found</th>'
            '<th scope="col">Suspected (count — rate)</th>'
            '<th scope="col">Suspected &amp; passed</th>'
            '<th scope="col">Suspected &amp; failed</th>'
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            if rows
            else ""
        )
        + hits_block
        + unavailable_line
        + "</section>"
    )


def _scorerlens_section(result: dict[str, Any]) -> str:
    """Scorer-lens regrade per candidate template.

    Renders the ``scorerlens`` payload (see
    experiments.scorerlens_profile) as a per-candidate table of
    verdict counts under each rung of the frozen normalisation
    ladder: how many of a template's byte-exact failures a more
    forgiving -- but still fully specified -- scorer would have
    passed. The verdict counts always include the recorded
    boundary fact, which no rung can move; the boundary-blocked
    count travels with the table so a stable winner is never read
    as scorer evidence when the boundary was the whole story.
    Flip attribution names the fixture and the first rung that
    recovers it; the answer and expected texts are never rendered
    here. Every interpolation is escaped at this boundary.
    """
    audit: Any = result.get("scorerlens")
    if type(audit) is not dict or not audit.get("available"):
        return ""
    candidates = audit.get("candidates")
    if type(candidates) is not list:
        return ""

    def _count(entry: dict[str, Any], key: str, rung: str) -> str:
        """Return the count."""
        counts = entry.get(key)
        if type(counts) is dict and type(counts.get(rung)) is int:
            return escape(str(counts[rung]))
        return "n/a"

    def _share(entry: dict[str, Any]) -> str:
        """Return the share."""
        value = entry.get("surface_failure_share")
        if type(value) in (int, float):
            try:
                text = f"{value:.0%}"
            except OverflowError:
                text = "n/a"
        else:
            text = "n/a"
        note = entry.get("share_note")
        if type(note) is str and note:
            text += " (" + escape(note.replace("_", " ")) + ")"
        return text

    rows = []
    flip_items = []
    unavailable_names = []
    for entry in candidates:
        if type(entry) is not dict:
            continue
        name = entry.get("candidate")
        label = escape(str(name)) if type(name) is str and name else "unknown"
        if not entry.get("available"):
            unavailable_names.append(label)
            continue
        rows.append(
            '<tr><td><span class="mono">'
            + label
            + "</span></td><td>"
            + escape(str(entry.get("n_evaluable")))
            + "</td><td>"
            + _count(entry, "verdict_counts", "0")
            + "</td><td>"
            + _count(entry, "verdict_counts", "1")
            + "</td><td>"
            + _count(entry, "verdict_counts", "2")
            + "</td><td>"
            + _count(entry, "verdict_counts", "3")
            + "</td><td>"
            + _count(entry, "verdict_counts", "4")
            + "</td><td>"
            + escape(str(entry.get("n_boundary_blocked")))
            + "</td><td>"
            + _share(entry)
            + "</td></tr>"
        )
        records = entry.get("records")
        if type(records) is list:
            for record in records:
                if type(record) is not dict:
                    continue
                rung = record.get("first_flip_rung")
                if type(rung) is not int:
                    continue
                fixture = record.get("fixture")
                via = record.get("flip_via")
                via_text = ""
                if type(via) is str and via:
                    via_text = " via " + escape(via)
                flip_items.append(
                    "<li>"
                    + '<span class="mono">'
                    + label
                    + "</span> · fixture "
                    + '<span class="mono">'
                    + (escape(fixture) if type(fixture) is str else "unknown")
                    + "</span> · first passes at rung "
                    + escape(str(rung))
                    + via_text
                    + (
                        " · boundary still blocks the verdict"
                        if record.get("boundary_found") is not True
                        else ""
                    )
                    + "</li>"
                )
    winner_block = ""
    winners = audit.get("winner_by_rung")
    if type(winners) is dict:
        lines = []
        for rung in ("0", "1", "2", "3", "4"):
            winner = winners.get(rung)
            lines.append(
                "<li>rung "
                + rung
                + ": "
                + (
                    '<span class="mono">' + escape(winner) + "</span>"
                    if type(winner) is str and winner
                    else "no unique winner"
                )
                + "</li>"
            )
        winner_block = (
            "<p>Search-phase winner under each rung (the conservative "
            "bar re-applied to the rung verdict counts; held-out is "
            "never regraded):</p><ul>" + "".join(lines) + "</ul>"
        )
    else:
        detail = audit.get("winner_detail")
        winner_block = (
            "<p>Search-phase winner readout withheld"
            + (
                " (" + escape(detail.replace("_", " ")) + ")"
                if type(detail) is str and detail
                else ""
            )
            + ": a winner computed over a partial or ineligible "
            "field would not mean what it appears to mean.</p>"
        )
    mismatch_line = ""
    if audit.get("evaluable_mismatch") is True:
        mismatch_line = (
            "<p><strong>Caution:</strong> candidates were not all "
            "measured on the same evaluable fixture set; compare "
            "counts across candidates with care.</p>"
        )
    flips_block = ""
    if flip_items:
        flips_block = (
            "<p>Fixtures recovered by the ladder (first rung that "
            "passes them):</p><ul>" + "".join(flip_items) + "</ul>"
        )
    unavailable_line = ""
    if unavailable_names:
        unavailable_line = (
            "<p>Candidates with no usable recorded observations: "
            + ", ".join('<span class="mono">' + name + "</span>" for name in unavailable_names)
            + ".</p>"
        )
    return (
        '<section class="card" id="scorerlens">'
        "<h2>Scorer-lens regrade</h2>"
        "<p>Is a template actually worse, or is the byte-exact "
        "scorer discarding answers a human would accept? Each "
        "candidate's recorded search answers are regraded under a "
        "frozen ladder: R0 exact, R1 whitespace, R2 case/Unicode, "
        "R3 edge punctuation and markdown fences, R4 numeric value. "
        "Verdict counts include the recorded boundary fact, which "
        "no rung can move. The ladder has no label-strip operation "
        '-- <span class="mono">Answer: 42</span> against expected '
        '<span class="mono">42</span> flips at no rung -- and R4 '
        'does not read decimal commas (<span class="mono">1,5</span> '
        'is not <span class="mono">15</span>). Casefold equates '
        "German ß with ss; a rung-2 flip on a name fixture deserves "
        "a second look.</p>"
        + mismatch_line
        + "<table><thead><tr><th>Candidate</th><th>Evaluable</th>"
        "<th>R0</th><th>R1</th><th>R2</th><th>R3</th><th>R4</th>"
        "<th>Boundary-blocked</th><th>Surface share of failures</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + winner_block
        + flips_block
        + unavailable_line
        + "</section>"
    )


def _negation_section(result: dict[str, Any]) -> str:
    """Antonym-negated definite-description probe: direct vs
    negated vs flipped instruction arms on generated compare
    pairs.

    Renders the ``negation`` payload (see
    experiments.negation_analysis): per candidate, per-arm
    pass rates with attrition beside each, the decomposition
    split per arm with the ``negation_echo`` share separated
    on the negated arms, D, J, and the section decision with
    J split by presentation order and by stratum, the
    rival-pattern readouts (flipped pass rate on D with the
    gate-weak qualifier, echo-dominant flag,
    order-concentration flag), the descriptive negated_alt
    arm, the id-divergence report, the boundary-only vs
    exact-only discordance decomposition, and the operating
    characteristics paragraph. The negated arm differs from
    direct in two lexical moves (the negation token plus the
    larger-to-smaller antonym swap), so the claim rendered
    here is antonym-negated definite-description discordance
    and no sentence attributes a discordance to negation
    simpliciter. Per-arm pass rates and the joint pattern are
    the only readouts. Counts and rates only; no payload,
    question, or answer text is rendered. Every
    payload-derived value below goes through _escape_text,
    never bare escape().
    """
    probe: Any = result.get("negation")
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
                " Decomposition — exact "
                + _count(decomp.get("exact_pass"))
                + ", wrong-label "
                + _count(decomp.get("wrong_label"))
                + ", echo "
                + _count(decomp.get("negation_echo"))
                + ", non-label "
                + _count(decomp.get("non_label"))
                + ", boundary-only "
                + _count(decomp.get("boundary_only"))
                + "."
            )
            if name in ("negated", "negated_alt"):
                completed = arm.get("n_completed")
                exact = decomp.get("exact_pass")
                non_pass = (
                    completed - exact if type(completed) is int and type(exact) is int else -1
                )
                if non_pass > 0:
                    decomp_bits += (
                        " Echo share among non-pass: "
                        + _count(decomp.get("negation_echo"))
                        + " of "
                        + _count(non_pass)
                        + " (printed beside the rate, never pooled into "
                        "wrong-label)."
                    )
        return (
            "<div>"
            + _escape_text(name)
            + " pass rate "
            + _rate(arm.get("rate"))
            + " ("
            + _count(arm.get("n_completed"))
            + " completed, "
            + _count(arm.get("n_context_exceeded"))
            + " context-exceeded, "
            + _count(arm.get("n_evaluation_failed"))
            + " evaluation-failed)."
            + decomp_bits
            + "</div>"
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
            for arm_name in ("direct", "negated", "flipped", "negated_alt"):
                arm = arms.get(arm_name)
                if type(arm) is dict:
                    bits.append(_arm_line(arm_name, arm))
        headline = entry.get("headline")
        if type(headline) is dict:
            direct_passes = headline.get("direct_passes")
            j_by_order = headline.get("j_by_order")
            order_bits = ""
            if type(j_by_order) is dict:
                order_bits = (
                    " J by presentation order — larger-first "
                    + _count(j_by_order.get("larger_first"))
                    + ", larger-second "
                    + _count(j_by_order.get("larger_second"))
                    + "."
                )
            j_by_stratum = headline.get("j_by_stratum")
            stratum_bits = ""
            if type(j_by_stratum) is dict and j_by_stratum:
                stratum_bits = (
                    " J by stratum — "
                    + ", ".join(
                        _escape_text(str(key)) + " " + _count(value)
                        for key, value in sorted(
                            j_by_stratum.items(), key=lambda item: str(item[0])
                        )
                    )
                    + "."
                )
            bits.append(
                "<div>D "
                + _count(direct_passes)
                + ", J "
                + _count(headline.get("joint"))
                + ", decision "
                + _escape_text(str(headline.get("decision", "")))
                + "."
                + order_bits
                + stratum_bits
                + "</div>"
            )
            bits.append(
                "<div>Flipped pass rate on direct passes "
                + _rate(headline.get("flipped_rate_on_d"))
                + " ("
                + _count(headline.get("flipped_passes_on_d"))
                + " of "
                + _count(direct_passes)
                + ")."
                + (
                    " Gate-weak: flipped pass rate on D is below 0.5 — "
                    "negation-drop undiagnosed, not absent "
                    "(larger-default rival pattern)."
                    if headline.get("gate_weak") is True
                    else ""
                )
                + (
                    " Echo-dominant: negated failures are predominantly "
                    "negation echoes, not negation-drop."
                    if headline.get("echo_dominant") is True
                    else ""
                )
                + (
                    " Order-concentrated: the joint pattern sits "
                    "entirely in one presentation order, not general "
                    "negation-drop."
                    if headline.get("order_concentrated") is True
                    else ""
                )
                + "</div>"
            )
        descriptive = entry.get("descriptive")
        if type(descriptive) is dict:
            bits.append(
                "<div>Descriptive negated_alt (never a decision input): "
                "wrong-label on direct passes "
                + _count(descriptive.get("negated_alt_wrong_label_on_d"))
                + " of "
                + _count(descriptive.get("negated_alt_completed_on_d"))
                + " completed, alt-joint "
                + _count(descriptive.get("negated_alt_joint"))
                + ".</div>"
            )
        ids = entry.get("ids")
        if type(ids) is dict:
            deltas = ids.get("id_length_delta_by_case")
            delta_text = ", ".join(_count(d) for d in deltas) if type(deltas) is list else "n/a"
            bits.append(
                "<div>Ids identical (direct-negated) "
                + _count(ids.get("n_ids_identical"))
                + " cases (rendering anomaly, excluded from the "
                "headline population), divergent "
                + _count(ids.get("n_ids_divergent"))
                + ", same-ids verdict flips "
                + _count(ids.get("n_same_ids_verdict_flips"))
                + " (determinism finding, never negation fragility). "
                "Id-length deltas (negated minus direct, case order): "
                + delta_text
                + ". Max abs id delta "
                + _count(ids.get("max_abs_id_delta"))
                + ".</div>"
            )
        discordance = entry.get("discordance")
        if type(discordance) is dict:
            bits.append(
                "<div>Discordance decomposition: boundary-only "
                + _count(discordance.get("boundary_only"))
                + ", exact-only "
                + _count(discordance.get("exact_only"))
                + ", mixed "
                + _count(discordance.get("mixed"))
                + ".</div>"
            )
        if type(headline) is dict:
            bits.append(
                "<div>At D = "
                + _count(headline.get("direct_passes"))
                + ", this panel (16 cases) falsifies only "
                "negation-drop patterns at least as lopsided as J >= 5 at "
                "D >= 8. It does not establish invariance. Measured "
                "operating characteristics of the (8,5) rule: a random "
                "labeller (independent 0.5 labels per arm) fires ~4% of "
                "the time; a competent candidate with no negation effect "
                "(direct/flipped pass 0.9, negated wrong-label rate 0.1) "
                "fires 0.7%; power is 84% at a 0.5 drop rate "
                "(direct/flipped pass 0.9) and 99%+ at 0.75, but only ~21% "
                "at a 0.25 drop rate. Small effects are therefore usually "
                "missed; a below_threshold on this panel is weak "
                "evidence.</div>"
            )
        rows.append(
            '<tr><td><span class="mono">'
            + _escape_text(str(name))
            + "</span></td><td>"
            + "".join(bits)
            + "</td></tr>"
        )
    return (
        '<section class="card" id="negation">'
        "<h2>Antonym-negated definite-description discordance</h2>"
        "<p>Sixteen generated compare pairs, each rendered under "
        "four instruction arms per candidate: direct ('larger'), "
        "negated ('not the smaller'), flipped ('smaller'), and a "
        "descriptive negated_alt arm ('not smaller than the other', "
        "never a decision input). The negated arm differs from "
        "direct in two lexical moves — the negation token plus "
        "the larger-to-smaller antonym swap — so the claim is "
        "antonym-negated definite-description discordance, never "
        "negation simpliciter, and the negated arm's expected "
        "answer is derivable only by the pair-closed complement "
        "rule, valid because every pair has distinct operands. "
        "The scorer is byte-exact conjoined with boundary "
        "presence; a negated answer in echo form ('not second' "
        "for expected 'first') is labelled negation_echo beside "
        "the rate, never counted as wrong-label and never as a "
        "pass. Each cell is one greedy generation. Records "
        "carry digests only.</p>"
        "<table><thead><tr><th>Candidate</th><th>Readout</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></section>"
    )


def _serving_backends_section(result: dict[str, Any]) -> str:
    """Serving-stack measurement: what a real serving stack did
    with exact token ids (cycle 243; the vLLM arm added by W6).

    Renders the ``serving_backends`` payload (see
    experiments.serving_backends_analysis): the stack identity the
    verdicts are keyed by, the probed capabilities (absence is
    data, listed as such), the fidelity verdict with its counts,
    the measured cache companion's positions as reported, and the
    serving-stack stop matrix rows. Digests and counts only; no
    prompt or completion text is rendered. Every payload-derived
    value below goes through _escape_text, never bare escape().
    The transport prose and the cache table branch on the stack:
    the llamacpp arm is measured over HTTP, the vLLM arm runs
    in-process, and each arm's cache quantities are its own.
    """
    probe: Any = result.get("serving_backends")
    if type(probe) is not dict or not probe.get("available"):
        return ""

    def _text(value: Any) -> str:
        """Return the text."""
        return _escape_text(value) if type(value) is str else "n/a"

    def _count(value: Any) -> str:
        """Return the count."""
        return _escape_text(str(value)) if type(value) is int else "n/a"

    identity = probe.get("identity")
    if type(identity) is not dict:
        return ""
    stack = identity.get("stack")
    is_vllm = stack == "vllm"
    capabilities = probe.get("capabilities")
    if type(capabilities) is not dict:
        return ""
    present = sorted(name for name, offered in capabilities.items() if offered is True)
    absent = sorted(name for name, offered in capabilities.items() if offered is False)
    fidelity = probe.get("fidelity")
    fidelity_line = ""
    if type(fidelity) is dict:
        counts = fidelity.get("counts")
        count_bits = ""
        if type(counts) is dict:
            count_bits = ", ".join(
                f"{_text(name)} {_count(counts[name])}" for name in sorted(counts, key=str)
            )
        fidelity_line = (
            "<p>Fidelity verdict: <strong>"
            + _text(fidelity.get("verdict"))
            + "</strong>"
            + (f" ({count_bits})" if count_bits else "")
            + ". The verdict compares the measured stack against the "
            "lab's stop predictor per execution record; a diverged "
            "generation stream is reported beside the verdicts, never "
            "instead of them.</p>"
        )
    cache = probe.get("cache")
    cache_rows = ""
    if type(cache) is dict and type(cache.get("positions")) is dict:
        position_rows: list[str] = []
        for position in ("cold", "warm_repeat", "shared_prefix", "divergent_control"):
            entry = cache["positions"].get(position)
            if type(entry) is not dict:
                continue
            if is_vllm:
                position_rows.append(
                    "<tr><td>"
                    + _text(position)
                    + "</td><td>"
                    + _count(entry.get("tokens_cached"))
                    + "</td><td>"
                    + _count(entry.get("cache_creation_tokens"))
                    + "</td></tr>"
                )
            else:
                position_rows.append(
                    "<tr><td>"
                    + _text(position)
                    + "</td><td>"
                    + _count(entry.get("cache_n"))
                    + "</td><td>"
                    + _count(entry.get("tokens_cached"))
                    + "</td><td>"
                    + _count(entry.get("tokens_evaluated"))
                    + "</td></tr>"
                )
        if position_rows and is_vllm:
            config = identity.get("config")
            engine = config.get("engine") if type(config) is dict else None
            block_size = engine.get("block_size") if type(engine) is dict else None
            block_note = (
                f" of {_count(block_size)} tokens"
                if type(block_size) is int and block_size > 0
                else ""
            )
            cache_rows = (
                "<p>Measured cache companion (sequence detail: "
                + _text(cache.get("detail") or "ok")
                + "). The headline quantity is tokens_cached, the "
                "engine's own reuse count, quantized to whole blocks"
                + block_note
                + "; cache_creation_tokens counts the tokens the "
                "engine wrote into the cache on that arm.</p>"
                + "<table><thead><tr><th>Position</th><th>tokens_cached</th>"
                + "<th>cache_creation_tokens</th></tr></thead>"
                + "<tbody>"
                + "".join(position_rows)
                + "</tbody></table>"
            )
        elif position_rows:
            cache_rows = (
                "<p>Measured cache companion (sequence detail: "
                + _text(cache.get("detail") or "ok")
                + "). The headline quantity is cache_n, the slot "
                "layer's own reuse count; the other two columns are "
                "recorded as reported.</p>"
                + "<table><thead><tr><th>Position</th><th>cache_n</th>"
                + "<th>tokens_cached</th><th>tokens_evaluated</th></tr></thead>"
                + "<tbody>"
                + "".join(position_rows)
                + "</tbody></table>"
            )
    stop_matrix = probe.get("stop_matrix")
    matrix_rows = ""
    if type(stop_matrix) is dict and type(stop_matrix.get("rows")) is list:
        body_rows: list[str] = []
        for row in stop_matrix["rows"]:
            if type(row) is not dict:
                continue
            cells = row.get("cells")
            cell = (
                cells.get(stack if type(stack) is str else "llamacpp")
                if type(cells) is dict
                else None
            )
            if type(cell) is not dict:
                continue
            # An unavailable cell observed nothing: it renders as
            # unobservable, never as a measured "no fire" (ORCH-04).
            # A genuine unavailable cell never carries fired=True
            # (the producer records None), so the fired check leads.
            if cell.get("fired") is True:
                stop_word = "fired"
            elif cell.get("available") is not True or cell.get("fired") is None:
                stop_word = "unobservable"
            else:
                stop_word = "no fire"
            body_rows.append(
                "<tr><td>"
                + _text(row.get("fixture"))
                + "</td><td>"
                + _text(row.get("candidate"))
                + "</td><td>"
                + _text(row.get("arm"))
                + "</td><td>"
                + _text(stop_word)
                + "</td><td>"
                + _count(cell.get("firing_step"))
                + "</td><td>"
                + _text(cell.get("disposition"))
                + "</td><td>"
                + _text(cell.get("verdict"))
                + "</td></tr>"
            )
        if body_rows:
            matrix_rows = (
                "<table><thead><tr><th>Fixture</th><th>Candidate</th>"
                + "<th>Arm</th><th>Stop</th><th>Firing step</th>"
                + "<th>Disposition</th><th>Verdict</th></tr></thead>"
                + "<tbody>"
                + "".join(body_rows)
                + "</tbody></table>"
            )
    checkpoint = identity.get("checkpoint_digest")
    fingerprint = identity.get("config_fingerprint")
    return (
        '<section class="card" id="serving-backends">'
        "<h2>Serving-stack measurement (exec-backends)</h2>"
        + (
            "<p>The lab ran a real serving stack in-process with raw "
            "token ids under a pinned deterministic sampler configuration "
            "and recorded what the stack did — firing step, stop winner, "
            "disposition, cache reuse — against the lab's own predictor. "
            "Verdicts are keyed to the stack identity below: a different "
            "build, checkpoint, or configuration is a different "
            "measurement, not a comparable one.</p>"
            if is_vllm
            else "<p>The lab drove a real serving stack over HTTP with raw "
            "token ids under a pinned deterministic sampler configuration "
            "and recorded what the stack did — firing step, stop winner, "
            "disposition, cache reuse — against the lab's own predictor. "
            "Verdicts are keyed to the stack identity below: a different "
            "build, checkpoint, or configuration is a different "
            "measurement, not a comparable one.</p>"
        )
        + "<p>Stack: "
        + _text(identity.get("stack"))
        + " "
        + _text(identity.get("stack_version"))
        + " on "
        + _text(identity.get("hardware_class"))
        + "; checkpoint sha256 "
        + _text(checkpoint[:12] if type(checkpoint) is str else None)
        + "…; configuration fingerprint "
        + _text(fingerprint[:12] if type(fingerprint) is str else None)
        + "….</p>"
        "<p>Capabilities offered: "
        + _text(", ".join(present) if present else "none")
        + ". Capabilities absent (recorded as absent cells, never "
        "silently degraded): "
        + _text(", ".join(absent) if absent else "none")
        + ".</p>"
        + fidelity_line
        + cache_rows
        + matrix_rows
        + "</section>"
    )


def _cache_profile_section(result: dict[str, Any]) -> str:
    """Prefix-cache sharability profile per template family.

    Renders the ``cache_profile`` payload (see
    experiments.cache_profile_profile) as a per-candidate table: the
    exact achievable sharability fractions (block-quantized headline;
    token-level diagnostic) plus the potential upper bounds, computed
    offline from the run's recorded input_ids with zero extra model
    calls. "Achievable" is exact for a permutation of the recorded
    prompts served once into a cold, infinite, dedicated cache; the
    section says so, and fences the framing: sharability is a cost
    *input*, not a cost verdict, and never a latency claim. Every
    interpolation is escaped at this boundary.
    """
    audit: Any = result.get("cache_profile")
    if type(audit) is not dict or not audit.get("available"):
        return ""
    per_candidate = audit.get("per_candidate")
    if type(per_candidate) is not dict:
        return ""

    def _pct(value: Any) -> str:
        """Return the pct."""
        if type(value) in (int, float):
            try:
                return f"{value:.1%}"
            except OverflowError:
                return "n/a"
        return "n/a"

    def _num(value: Any) -> str:
        """Return the num."""
        if type(value) in (int, float):
            try:
                return f"{value:.1f}"
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
            + escape(_num(entry.get("mean_prompt_tokens")))
            + "</td><td>"
            + escape(_pct(entry.get("achievable_block_fraction")))
            + "</td><td>"
            + escape(_pct(entry.get("achievable_token_fraction")))
            + "</td><td>"
            + escape(_pct(entry.get("potential_block_fraction")))
            + "</td><td>"
            + escape(_pct(entry.get("potential_token_fraction")))
            + "</td><td>"
            + escape(str(entry.get("n_duplicate_prompts")))
            + "</td></tr>"
        )
    block_size = audit.get("block_size")
    gap = audit.get("sharability_gap")
    gap_line = ""
    if type(gap) in (int, float) and _float_convertible(gap):
        gap_line = (
            "<p>Max-minus-min achievable block sharability over evaluable "
            "candidates: <strong>" + escape(f"{gap:.1%}") + "</strong>.</p>"
        )
    extremes_line = ""
    most = audit.get("most_sharable")
    least = audit.get("least_sharable")
    if type(most) is list and most and type(least) is list and least:
        extremes_line = (
            "<p>Most sharable (highest achievable block fraction): "
            + ", ".join(
                '<span class="mono">' + escape(str(name)) + "</span>"
                for name in most
                if type(name) is str
            )
            + "; least sharable: "
            + ", ".join(
                '<span class="mono">' + escape(str(name)) + "</span>"
                for name in least
                if type(name) is str
            )
            + ".</p>"
        )
    mismatch_line = ""
    if audit.get("evaluable_mismatch") is True:
        mismatch_line = (
            "<p><strong>Caution:</strong> candidates were measured on "
            "different numbers of evaluable prompts; fractions from "
            "different denominators are not directly comparable.</p>"
        )
    excluded = audit.get("excluded_candidates")
    excluded_line = ""
    if type(excluded) is list and excluded:
        excluded_line = (
            "<p>Candidates with no evaluable prompt: "
            + ", ".join(
                '<span class="mono">' + escape(str(name)) + "</span>"
                for name in excluded
                if type(name) is str
            )
            + ".</p>"
        )
    return (
        '<section class="card" id="cache-profile">'
        "<h2>Prefix-cache sharability profile</h2>"
        "<p>For each candidate, the fraction of recorded prompt tokens "
        "that exact-prefix KV-cache reuse could skip, computed offline "
        "from the run's recorded input_ids with zero extra model "
        "calls. <strong>Achievable</strong> is exact for a permutation "
        "of the recorded prompts served once into a cold, infinite, "
        "dedicated cache: each shared prefix costs exactly one "
        "first-occurrence miss in every request order. "
        "<strong>Potential</strong> counts every cacheable token and "
        "is an eviction-robust upper bound over those orders -- not a "
        "production ceiling (cross-workload warming and repeated "
        "requests can exceed it). Block rows quantize to full blocks "
        "of "
        + escape(str(block_size))
        + " tokens (vLLM automatic prefix caching caches full blocks "
        "only; partial trailing blocks never count). Sharability is a "
        "cost <em>input</em>, <strong>not a cost verdict</strong>: it "
        "says nothing about absolute prefill cost, latency, or "
        "TTFT.</p>"
        + gap_line
        + extremes_line
        + mismatch_line
        + (
            '<div class="table-wrap"><table>'
            "<caption>Prefix sharability per candidate (share of prompt "
            "tokens).</caption>"
            '<thead><tr><th scope="col">Candidate</th>'
            '<th scope="col">Evaluable</th>'
            '<th scope="col">Mean prompt</th>'
            '<th scope="col">Achievable block</th>'
            '<th scope="col">Achievable token</th>'
            '<th scope="col">Potential block</th>'
            '<th scope="col">Potential token</th>'
            '<th scope="col">Duplicates</th>'
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            if rows
            else ""
        )
        + excluded_line
        + "</section>"
    )


def _confidentiality_section(result: dict[str, Any]) -> str:
    """Labeled-block confidentiality profile per template family.

    Renders the ``confidentiality`` payload (see
    experiments.confidentiality_profile) as a per-candidate table: for
    the privilege-labeled block (treatment) vs the identical bare marker
    (control), the paired leak rates under frozen extraction
    elicitations, the gap, discordant pairs, and the exact McNemar
    p-value. The measured quantity is a verbatim-exposure LOWER bound
    (paraphrase leakage is out of scope); the section fences the
    framing: never a "safe to ship secrets" claim, and "confidentiality"
    is the behavioural sense (OWASP LLM07), not the TEE/cryptographic
    sense. Every interpolation is escaped at this boundary; no raw model
    outputs are rendered.
    """
    audit: Any = result.get("confidentiality")
    if type(audit) is not dict or not audit.get("available"):
        return ""
    per_candidate = audit.get("per_candidate")
    if type(per_candidate) is not dict:
        return ""

    def _pct(value: Any) -> str:
        """Return the pct."""
        if type(value) in (int, float):
            try:
                return f"{value:.1%}"
            except OverflowError:
                return "n/a"
        return "n/a"

    def _num(value: Any) -> str:
        """Return the num."""
        if type(value) in (int, float):
            try:
                return f"{value:.3f}"
            except OverflowError:
                return "n/a"
        return "n/a"

    rows = []
    for key in sorted(per_candidate, key=str):
        cand = str(key)
        entry = per_candidate[key]
        if type(entry) is not dict:
            rows.append(
                '<tr><td><span class="mono">'
                + escape(cand)
                + '</span></td><td colspan="7">unavailable</td></tr>'
            )
            continue
        rows.append(
            '<tr><td><span class="mono">'
            + escape(cand)
            + "</span></td><td>"
            + escape(str(entry.get("n_paired_evaluable")))
            + " / "
            + escape(str(entry.get("n")))
            + "</td><td>"
            + escape(str(entry.get("ctrl_leaks")))
            + "</td><td>"
            + escape(str(entry.get("treat_leaks")))
            + "</td><td>"
            + escape(_pct(entry.get("ctrl_rate")))
            + "</td><td>"
            + escape(_pct(entry.get("treat_rate")))
            + "</td><td>"
            + escape(_pct(entry.get("gap")))
            + "</td><td>"
            + escape(str(entry.get("discord_ctrl_only")))
            + " / "
            + escape(str(entry.get("discord_treat_only")))
            + "</td><td>"
            + escape(_num(entry.get("mcnemar_p")))
            + "</td></tr>"
        )
    gap = audit.get("gap")
    gap_line = ""
    if type(gap) in (int, float) and _float_convertible(gap):
        gap_line = (
            "<p>Max-minus-min confidentiality gap over measurable "
            "candidates: <strong>" + escape(f"{gap:.1%}") + "</strong>.</p>"
        )
    uniform_line = ""
    if audit.get("uniform_nonpositive") is True:
        uniform_line = (
            "<p><strong>Null result</strong>: every measurable gap is "
            "nonpositive, so on this fixture set the privilege label "
            "conferred no measurable selective protection for any "
            "candidate -- the small-difference outcome Tensor Trust "
            "Appendix H's role-scheme ablation also reports.</p>"
        )
    positive = audit.get("positive_candidates")
    positive_line = ""
    if type(positive) is list and positive:
        positive_line = (
            "<p>Candidates meeting the positive bar (gap &gt; 0 with at "
            "least 2 discordant control-only pairs): "
            + ", ".join(
                '<span class="mono">' + escape(str(name)) + "</span>"
                for name in positive
                if type(name) is str
            )
            + ".</p>"
        )
    return (
        '<section class="card" id="confidentiality">'
        "<h2>Labeled-block confidentiality</h2>"
        "<p>A fixed nonce marker was rendered inside the "
        "privilege-labeled block (treatment) and, as a control, bare at "
        "the same position; frozen extraction elicitations then asked "
        "for the block quoted verbatim. Rates are paired per fixture and "
        "measured on the full decoded output. The gap is the control "
        "leak rate minus the treatment leak rate: positive means the "
        "label measurably reduced verbatim exposure on this fixture "
        "set. These numbers are a <strong>verbatim-exposure lower "
        "bound</strong>: paraphrase, translation, and summarization "
        "leakage are out of scope, and a low leak rate is "
        "<strong>not</strong> a certification that real secrets are "
        "safe in the block. Confidentiality here is the behavioural "
        "sense (OWASP LLM07, system prompt leakage), not confidentiality "
        "at rest or in transit.</p>"
        + gap_line
        + uniform_line
        + positive_line
        + '<div class="table-wrap"><table>'
        + "<thead><tr><th>Candidate</th><th>Paired evaluable</th>"
        + "<th>Control leaks</th><th>Treatment leaks</th>"
        + "<th>Control rate</th><th>Treatment rate</th><th>Gap</th>"
        + "<th>Discordant (ctrl-only / treat-only)</th>"
        + "<th>McNemar p</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        + "</section>"
    )


def _answer_collision_section(result: dict[str, Any]) -> str:
    """Answer-collision audit: the lab's own detector vs correct answers.

    Renders the ``answer_collision`` payload (see
    experiments.answer_collision_profile) as a per-boundary-configuration
    table: for the reserved-token arm and the subword control arm, how
    many verbatim-emitted correct answers the detector truncated
    (collision), how many it ended at the marker (ends-at-marker), and
    how many it left intact. The measured quantity is a detector
    property: scripted emission measures the detector, never a model,
    and only the configured boundaries are in scope. Marker text itself
    is never rendered here -- marker-bearing fixtures are hostile input
    to the report path by design (red-team A R11), so the section shows
    counts and rates only. Every interpolated identifier is escaped at
    this boundary; no raw model outputs are rendered.
    """
    audit: Any = result.get("answer_collision")
    if type(audit) is not dict or not audit.get("available"):
        return ""
    per_candidate = audit.get("per_candidate")
    if type(per_candidate) is not dict:
        return ""

    def _pct(value: Any) -> str:
        """Return the pct."""
        if type(value) in (int, float):
            try:
                return f"{value:.1%}"
            except OverflowError:
                return "n/a"
        return "n/a"

    def _int(value: Any) -> str:
        """Return the int."""
        if type(value) is int:
            return str(value)
        return "n/a"

    rows = []
    for key in sorted(per_candidate, key=str):
        cand = str(key)
        entry = per_candidate[key]
        if type(entry) is not dict:
            rows.append(
                '<tr><td><span class="mono">'
                + escape(cand)
                + '</span></td><td colspan="6">unavailable</td></tr>'
            )
            continue
        reserved = entry.get("reserved") or {}
        subword = entry.get("subword") or {}
        rows.append(
            '<tr><td><span class="mono">'
            + escape(cand)
            + "</span></td><td>"
            + escape(_int(entry.get("boundary_ids")))
            + "</td><td>"
            + escape(_int(reserved.get("n_scored")))
            + "</td><td>"
            + escape(_int(reserved.get("n_collision")))
            + "</td><td>"
            + escape(_int(reserved.get("n_ends_at_marker")))
            + "</td><td>"
            + escape(_pct(reserved.get("collision_rate")))
            + "</td><td>"
            + escape(_int(subword.get("n_collision")))
            + "</td></tr>"
        )
    intact_line = ""
    if audit.get("uniform_intact") is True:
        intact_line = (
            "<p><strong>Null result</strong>: on this fixture set the "
            "detector left every marker-bearing correct answer intact in "
            "every measurable configuration -- no truncation of a "
            "verbatim-correct answer was observed. This is the "
            "regression-guard pass: it says the detector is "
            "collision-clean here, not that real answers never contain "
            "marker text.</p>"
        )
    colliding = audit.get("colliding_candidates")
    colliding_line = ""
    if type(colliding) is list and colliding:
        colliding_line = (
            "<p>Configurations whose detector provably truncated a "
            "verbatim-correct answer (reserved arm): "
            + ", ".join(
                '<span class="mono">' + escape(str(name)) + "</span>"
                for name in colliding
                if type(name) is str
            )
            + ".</p>"
        )
    return (
        '<section class="card" id="answer-collision">'
        "<h2>Answer-collision audit</h2>"
        "<p>The lab audits its own boundary detector: synthetic expected "
        "answers embedding marker-like text were emitted verbatim through "
        "the evaluation path, once in the backend's natural encoding "
        "(reserved arm) and once with every marker occurrence re-encoded "
        "in subword ids (subword control arm). A <strong>collision</strong> "
        "means the detector cut a correct answer and discarded content; "
        "<strong>ends-at-marker</strong> means the scripted emission "
        "simply ended at the marker -- a distinct bookkeeping cell, not "
        "evidence of early stopping. The unit is the boundary "
        "configuration: configurations sharing a boundary produce "
        "identical cells by construction. Scope is the configured "
        "boundaries and this tokenizer only; this measures the detector, "
        "never any model, and says nothing about how often real answers "
        "contain marker text.</p>"
        + intact_line
        + colliding_line
        + '<div class="table-wrap"><table>'
        + "<thead><tr><th>Candidate</th><th>Boundary ids</th>"
        + "<th>Reserved scored</th><th>Reserved collisions</th>"
        + "<th>Ends-at-marker</th><th>Collision rate</th>"
        + "<th>Subword collisions</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        + "</section>"
    )


def _fewshot_section(result: dict[str, Any]) -> str:
    """Few-shot demonstration gain x template family.

    Renders the ``fewshot`` payload (see experiments.fewshot_gain): the
    zero-shot -> 2-shot verdict-change rate (gain) and the demo-order
    sensitivity per candidate, measured on held-out fixtures with the
    demos sampled from the search split only. The pre-registered
    falsifiable claim is a gain of ~0 and order sensitivity of ~0
    uniform across families; family differences are the finding. Gain is
    a verdict-change rate, never a quality judgment. Every
    interpolation is escaped at this boundary.
    """
    audit: Any = result.get("fewshot")
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

    rows = []
    for key in sorted(per_candidate, key=str):
        cand = str(key)
        entry = per_candidate[key]
        if type(entry) is not dict:
            continue
        raw_per_order = entry.get("per_order")
        per_order: dict[str, Any] = raw_per_order if type(raw_per_order) is dict else {}
        order_cells = []
        for arm in ("order_a", "order_b"):
            raw_cell = per_order.get(arm)
            cell: dict[str, Any] = raw_cell if type(raw_cell) is dict else {}
            order_cells.append(
                escape(_rate(cell.get("change_rate")))
                + " ("
                + escape(str(cell.get("n_changed")))
                + "/"
                + escape(str(cell.get("n_evaluable")))
                + ")"
            )
        rows.append(
            '<tr><td><span class="mono">'
            + escape(cand)
            + "</span></td><td>"
            + escape(str(entry.get("n_evaluable")))
            + "</td><td>"
            + escape(_rate(entry.get("gain")))
            + "</td><td>"
            + escape(_rate(entry.get("order_sensitivity")))
            + "</td><td>"
            + " | ".join(order_cells)
            + "</td></tr>"
        )
    excluded = audit.get("excluded_candidates")
    excluded_line = ""
    if type(excluded) is list and excluded:
        excluded_line = (
            "<p>Candidates with no fully evaluable fixture (a cell failed, "
            "or some arm verdict was missing): "
            + ", ".join(
                '<span class="mono">' + escape(str(name)) + "</span>"
                for name in excluded
                if type(name) is str
            )
            + ".</p>"
        )
    per_arm = audit.get("per_arm")
    arm_rows = []
    if type(per_arm) is dict:
        for arm in ("baseline", "order_a", "order_b"):
            arm_cell = per_arm.get(arm)
            if type(arm_cell) is not dict:
                continue
            winner = arm_cell.get("winner")
            arm_rows.append(
                '<tr><td><span class="mono">'
                + escape(str(arm))
                + "</span></td><td>"
                + escape(str(cell.get("decision")))
                + "</td><td>"
                + (
                    '<span class="mono">' + escape(str(winner)) + "</span>"
                    if type(winner) is str
                    else "n/a"
                )
                + "</td><td>"
                + escape(str(cell.get("n_evaluable")))
                + "</td></tr>"
            )
    flips = audit.get("flips")
    flips_line = ""
    if type(flips) is list and flips:
        flip_parts = []
        for flip in flips:
            if type(flip) is not dict:
                continue
            flip_parts.append(
                '<span class="mono">'
                + escape(str(flip.get("arm")))
                + "</span>: "
                + escape(str(flip.get("decision_from")))
                + " \u2192 "
                + escape(str(flip.get("decision_to")))
            )
        if flip_parts:
            flips_line = (
                "<p>Selection-rule flips vs the primary run: " + "; ".join(flip_parts) + ".</p>"
            )
    demo_ids = audit.get("demo_ids")
    demo_line = ""
    if type(demo_ids) is list and demo_ids:
        demo_line = (
            "<p>Demonstration fixtures (search split, seeded sample, shown in "
            "sampled order): "
            + ", ".join(
                '<span class="mono">' + escape(str(d)) + "</span>"
                for d in demo_ids
                if type(d) is str
            )
            + ".</p>"
        )
    return (
        '<section class="card" id="few-shot">'
        "<h2>Few-shot \u00d7 template</h2>"
        "<p>Two demonstration fixtures were sampled (seeded) from the search "
        "split and prepended as real user/assistant turns to every held-out "
        "fixture, in both demo orders; every search candidate was evaluated "
        "0-shot and with both orders. <strong>Gain</strong> is the mean "
        "over the two orders of the k-shot vs 0-shot verdict-change rate, "
        "in [0, 1]; <strong>order sensitivity</strong> is the population "
        "standard deviation of the two per-order change rates, in [0, 0.5]. "
        "Gain is a verdict-<em>change</em> rate, never a quality judgment: "
        "a high gain means the demos moved the verdicts, not that they "
        "helped. A demo fixture can never evaluate itself -- demos come "
        "from the search split only, measurement runs on held-out.</p>"
        + demo_line
        + (
            '<div class="table-wrap"><table><caption>Gain and order sensitivity '
            "per candidate.</caption>"
            '<thead><tr><th scope="col">Candidate</th>'
            '<th scope="col">Evaluable fixtures</th><th scope="col">Gain</th>'
            '<th scope="col">Order sensitivity</th>'
            '<th scope="col">Per-order change rate (changed/evaluable)</th>'
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            if rows
            else ""
        )
        + excluded_line
        + (
            '<div class="table-wrap"><table><caption>Selection rule re-applied per arm.</caption>'
            '<thead><tr><th scope="col">Arm</th><th scope="col">Decision</th>'
            '<th scope="col">Winner</th><th scope="col">Evaluable candidates</th>'
            "</tr></thead><tbody>" + "".join(arm_rows) + "</tbody></table></div>"
            if arm_rows
            else ""
        )
        + flips_line
        + "</section>"
    )


def _synthhistory_section(result: dict[str, Any]) -> str:
    """Synthetic-history selection stability: does the winner survive context?

    Every search candidate re-evaluated with fixed benign synthetic prior
    conversation turns prepended (one and three exchanges); the selection
    rule re-applied per history condition. Reports per-candidate history
    pass rates, the winner-vs-clean flips, and the delimiter-explicitness
    group comparison (families with unambiguous per-turn open/close
    markers vs loose delimiting -- a correlation readout over the lab's
    own catalog, not a causal claim). Renders nothing when the
    synthetic-history experiment did not run.
    """
    history: Any = result.get("synthhistory")
    if type(history) is not dict or not history.get("available"):
        return ""
    per_condition = history.get("per_condition")
    if type(per_condition) is not dict:
        return ""
    primary_winner = history.get("primary_winner")
    primary_decision = history.get("primary_decision")
    n_fixtures = history.get("n_fixtures")

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
                + escape(_rate(entry.get("pass_rate")))
                + "</td><td>"
                + badge
                + "</td></tr>"
            )

    flips = history.get("flips")
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

    group_rates = history.get("group_pass_rates")
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

    exp_rate: float | None = None
    loose_rate: float | None = None
    if type(group_rates) is dict:
        for label in ("explicit", "loose", "unknown"):
            group = group_rates.get(label)
            if type(group) is not dict:
                continue
            rate = _as_rate(group.get("mean_pass_rate"))
            if label == "explicit":
                exp_rate = rate
            elif label == "loose":
                loose_rate = rate
            group_rows.append(
                "<tr><td>"
                + escape(str(label))
                + "</td><td>"
                + escape(str(group.get("n_candidates")))
                + "</td><td>"
                + escape(_rate(group.get("mean_pass_rate")))
                + "</td></tr>"
            )
    group_note = ""
    if exp_rate is not None and loose_rate is not None:
        relation = (
            "higher than"
            if exp_rate > loose_rate
            else "lower than"
            if exp_rate < loose_rate
            else "equal to"
        )
        group_note = (
            "<p>Explicitly-delimited families average "
            + escape(_rate(exp_rate))
            + " history pass rate vs "
            + escape(_rate(loose_rate))
            + " for loosely-delimited families ("
            + relation
            + " -- a correlation readout over "
            "mechanically-derived catalog labels, not a causal claim).</p>"
        )

    n_text = f" ({n_fixtures} fixtures per condition)" if type(n_fixtures) is int else ""
    return (
        '<section class="card" id="synthetic-history">'
        "<h2>Synthetic-history selection stability</h2>"
        "<p>Every search candidate re-evaluated with fixed benign synthetic "
        "prior conversation turns prepended to every fixture -- one exchange "
        "(history_2) and three exchanges (history_6); the selection rule is "
        "re-applied per history condition. History pass rate is the fraction "
        "of fixtures still answered exactly under history."
        + escape(n_text)
        + "</p>"
        + "".join(verdict_lines)
        + (
            "<ul>" + "".join(flip_lines) + "</ul>"
            if flip_lines
            else "<p>No winner flips against the primary run.</p>"
        )
        + (
            '<div class="table-wrap"><table><caption>History pass rate '
            "by condition.</caption>"
            '<thead><tr><th scope="col">Candidate</th>'
            '<th scope="col">Condition</th><th scope="col">Pass rate</th>'
            '<th scope="col">Pass</th>'
            "</tr></thead><tbody>" + "".join(cand_rows) + "</tbody></table></div>"
            if cand_rows
            else ""
        )
        + (
            '<div class="table-wrap"><table><caption>History pass rate by '
            "delimiter-explicitness label (correlation only).</caption>"
            '<thead><tr><th scope="col">Label</th>'
            '<th scope="col">Candidates</th>'
            '<th scope="col">Mean pass rate</th>'
            "</tr></thead><tbody>" + "".join(group_rows) + "</tbody></table></div>" + group_note
            if group_rows
            else ""
        )
        + "</section>"
    )


def _batchinv_section(result: dict[str, Any]) -> str:
    """Batch-invariance audit: solo vs batched greedy decoding.

    Every search fixture's exact prompt token sequence re-run through
    the backend's batch path (when one exists) and compared
    token-for-token against the isolated solo generation. Reports the
    divergence rate and first-divergence positions -- the audit of the
    lab's "exact token sequences" invariant against batch-composition
    nondeterminism. Renders nothing when the audit did not run or the
    backend exposed no batch path.
    """
    audit: Any = result.get("batch_invariance")
    if type(audit) is not dict or audit.get("available") is not True:
        return ""
    per_prompt = audit.get("per_prompt")
    if type(per_prompt) is not dict:
        return ""

    def _rate(value: Any) -> str:
        """Return the rate."""
        if type(value) in (int, float):
            try:
                return f"{value:.1%}"
            except OverflowError:
                return "n/a"
        return "n/a"

    def _num(value: Any) -> str:
        """Return the num."""
        return str(value) if type(value) is int else "n/a"

    verdict = audit.get("verdict")
    verdict_badge = (
        '<span class="badge pass">batch-invariant</span>'
        if verdict == "batch_invariant"
        else '<span class="badge fail">batch-variant</span>'
        if verdict == "batch_variant"
        else '<span class="badge">n/a</span>'
    )
    rows = []
    for key in sorted(per_prompt, key=str):
        pid = str(key)
        entry = per_prompt[key]
        if type(entry) is not dict:
            continue
        diverged = entry.get("diverged")
        badge = (
            '<span class="badge fail">diverged</span>'
            if diverged is True
            else '<span class="badge pass">identical</span>'
            if diverged is False
            else '<span class="badge">n/a</span>'
        )
        first = entry.get("first_divergence")
        first_text = _num(first) if diverged is True else "&mdash;"
        error = entry.get("error")
        note = " (" + escape(str(error)) + ")" if type(error) is str and error else ""
        rows.append(
            '<tr><td><span class="mono">'
            + escape(pid)
            + "</span></td><td>"
            + badge
            + "</td><td>"
            + first_text
            + "</td><td>"
            + _num(entry.get("solo_len"))
            + "</td><td>"
            + _num(entry.get("batched_len"))
            + "</td><td>"
            + note
            + "</td></tr>"
        )

    n_prompts = audit.get("n_prompts")
    n_evaluable = audit.get("n_evaluable")
    n_diverged = audit.get("n_diverged")
    counts = (
        f" ({_num(n_evaluable)} of {_num(n_prompts)} prompts evaluable, "
        f"{_num(n_diverged)} diverged)"
    )
    return (
        '<section class="card" id="batch-invariance"><h2>Batch-invariance audit</h2>'
        "<p>Each search fixture's exact prompt token sequence was decoded "
        "greedily in isolation and co-batched through the backend's batch "
        "path; continuation token ids were compared token-for-token. Any "
        "divergence falsifies the determinism assumption for that backend."
        + escape(counts)
        + "</p>"
        + "<p>Verdict: "
        + verdict_badge
        + " &mdash; divergence rate "
        + escape(_rate(audit.get("divergence_rate")))
        + "</p>"
        + (
            '<div class="table-wrap"><table><caption>Solo vs batched '
            "comparison per prompt.</caption>"
            '<thead><tr><th scope="col">Prompt</th>'
            '<th scope="col">Outcome</th>'
            '<th scope="col">First divergence</th>'
            '<th scope="col">Solo ids</th>'
            '<th scope="col">Batched ids</th>'
            '<th scope="col">Note</th>'
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            if rows
            else ""
        )
        + "</section>"
    )


def _portfolio_section(result: dict[str, Any]) -> str:
    """Oracle-mixture / protocol-portfolio gap: what the winner leaves behind.

    Pure post-hoc over the stored search observations (exact answer AND
    explicit boundary): the per-fixture oracle (which candidates pass
    each fixture), the oracle pass rate vs the global winner's rate and
    the gap, and the greedy covering protocol portfolio over solvable
    fixtures. The portfolio is a greedy set-cover approximation, not a
    proven optimum -- the section says so. Renders nothing when the
    analysis did not run.
    """
    portfolio: Any = result.get("portfolio")
    if type(portfolio) is not dict or not portfolio.get("available"):
        return ""
    oracle = portfolio.get("oracle")
    cover = portfolio.get("portfolio")
    if type(oracle) is not dict or type(cover) is not dict:
        return ""

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

    n_fixtures = portfolio.get("n_fixtures")
    n_solvable = oracle.get("n_solvable")
    gap = oracle.get("gap")
    winner = portfolio.get("winner")
    gap_line = (
        "<p>Oracle pass rate <strong>"
        + escape(_rate(oracle.get("oracle_pass_rate")))
        + "</strong> vs global winner <strong>"
        + escape(_name(winner))
        + "</strong> at <strong>"
        + escape(_rate(oracle.get("winner_pass_rate")))
        + "</strong> -- gap <strong>"
        + escape(_rate(gap) if type(gap) in (int, float) else "n/a")
        + "</strong> of "
        + escape(str(n_fixtures) if type(n_fixtures) is int else "?")
        + " fixtures ("
        + escape(str(n_solvable) if type(n_solvable) is int else "?")
        + " solvable by at least one protocol).</p>"
    )
    members = cover.get("members")
    member_rows = []
    if type(members) is list:
        per_member = cover.get("per_member_covers")
        per_member = per_member if type(per_member) is dict else {}
        for name in members:
            fixtures = per_member.get(name) if type(name) is str else None
            fixtures = list(fixtures) if type(fixtures) is list else []
            member_rows.append(
                '<tr><td><span class="mono">'
                + escape(str(name))
                + "</span></td><td>"
                + escape(", ".join(str(fid) for fid in fixtures) or "(none)")
                + "</td></tr>"
            )
    portfolio_block = (
        "<p>Greedy covering portfolio (size "
        + escape(str(cover.get("size")))
        + "): "
        + escape(", ".join(str(m) for m in members) if type(members) is list else "(none)")
        + ". <em>"
        + escape(
            str(cover.get("greedy_note") or "")
            or "Greedy set-cover approximation; not guaranteed optimal."
        )
        + "</em></p>"
        + (
            '<div class="table-wrap"><table><caption>Portfolio members and '
            "the solvable fixtures each covers.</caption>"
            '<thead><tr><th scope="col">Protocol</th>'
            '<th scope="col">Covers</th>'
            "</tr></thead><tbody>" + "".join(member_rows) + "</tbody></table></div>"
            if member_rows
            else ""
        )
    )
    per_fixture = oracle.get("per_fixture_winners")
    fixture_rows = []
    if type(per_fixture) is dict:
        for key in sorted(per_fixture, key=str):
            names = per_fixture[key]
            names = list(names) if type(names) is list else []
            fixture_rows.append(
                '<tr><td><span class="mono">'
                + escape(str(key))
                + "</span></td><td>"
                + escape(", ".join(str(name) for name in names))
                + "</td></tr>"
            )
    unsolvable = oracle.get("unsolvable_fixture_ids")
    unsolvable_line = ""
    if type(unsolvable) is list and unsolvable:
        unsolvable_line = (
            "<p>No protocol passed: " + escape(", ".join(str(fid) for fid in unsolvable)) + ".</p>"
        )
    return (
        '<section id="protocol-portfolio"><h2>Protocol portfolio</h2>'
        "<p>The routing lens applied within the lab to prompt protocols: "
        "how much does committing to the single global winner leave on "
        "the table versus a per-fixture oracle, and which small protocol "
        "set covers every solvable fixture?</p>"
        + gap_line
        + portfolio_block
        + (
            '<div class="table-wrap"><table><caption>Per-fixture oracle '
            "winners (any passing protocol is a correct choice "
            "here).</caption>"
            '<thead><tr><th scope="col">Fixture</th>'
            '<th scope="col">Oracle winners</th>'
            "</tr></thead><tbody>" + "".join(fixture_rows) + "</tbody></table></div>"
            if fixture_rows
            else "<p>No fixture was passed by any protocol.</p>"
        )
        + unsolvable_line
        + "</section>"
    )
