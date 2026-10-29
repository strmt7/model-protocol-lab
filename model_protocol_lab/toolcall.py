"""Prompt-only tool-call syntax adherence per template (research wave-4 idea 4).

The question: prompt-only tool-call adherence -- correct function
selection, correct argument extraction, and correct abstention when no
function is relevant (relevance detection) -- is uniform across the 29
template families for a fixed tool list and query set? The Berkeley
Function-Calling Leaderboard (BFCL, Databricks, "Beyond the
Leaderboard: Unpacking Function Calling Evaluation") scores function
calling in native-FC vs prompt mode, but prompt mode uses one universal
system prompt per provider -- the chat template is never the
independent variable. JSONSchemaBench varies frameworks. Nobody varies
the *chat template* while holding model, tools, queries, and decoder
fixed. The lab already has the exact machinery (template families,
exact token sequences, fixture split, conservative selection) plus the
deterministic local validation discipline from cycle 157's
structured-output probe to run the missing experiment.

Boundary with cycle 157 (structured output): 157 measures JSON *schema*
validity of a free response. This measures *function selection* (which
tool), *argument extraction* (parameter values from the query), and
*relevance detection* (abstain when no tool fits -- a direct analogue of
the lab's conservative selection rule). Different task, different
failure modes, different falsifiable claim; the deterministic validator
is new, the fail-closed classification discipline is reused.

Design: a fixed tool list (4 functions with typed params) and a fixed
query set (7 tool-relevant queries, 2 no-relevant-function queries).
The prompt describes the tools in pinned prose and demands exactly one
line: ``CALL <tool_name> <JSON arguments>`` or ``NO_CALL`` when no tool
fits. Each fixture's ``expected`` is the canonical rendering of the
correct call (or ``NO_CALL``), so the pipeline's ``exact_match`` still
grades strictly; adherence is measured independently on the decoded
text by :func:`classify_tool_call`, with a deterministic hand-rolled
call parser -- the lab has zero runtime dependencies, so there is no
BFCL harness to lean on and none is needed for the pinned syntax.

No native function-calling APIs in v1: that keeps the backend contract
identical to every other experiment (greedy text generation) and keeps
the claim honest -- what the template alone elicits from prompt text.

Pre-registered claims (falsifiable):

1. Adherence is uniform across candidates: the max-minus-min
   ``adherence_gap`` over evaluable candidates is either ~0 (uniform,
   the null hypothesis -- reported, never asserted) or measurably
   positive; the same gap is reported separately for selection,
   arguments, and relevance, so a composite gap can be decomposed.
2. Selection, argument extraction, and relevance detection decompose: a
   response can select the right tool yet extract the wrong arguments,
   and abstention is scored as its own rate -- the three are reported
   separately rather than conflated into one "accuracy".
3. Classification is total and fail-closed: non-string responses become
   ``not_text`` (excluded from rates, never silently dropped), an
   unparseable line becomes ``parse_failed`` (counts as a failure on all
   three rates, never silently dropped), and an unknown tool name can
   never count as correct selection.

Everything here is pure: tool/query validation, the call syntax,
classification, and rate aggregation need no model.
"""

from __future__ import annotations

import json
import math
from typing import Any, cast

from .core import Fixture, Message

#: Fixed tool list, in canonical order. Flat typed params only; the
#: prompt describes these in pinned prose, so the set is part of the
#: measurement contract, not a backend capability.
TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "params": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["C", "F"]},
        },
        "required": ("city", "unit"),
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount between currencies.",
        "params": {
            "amount": {"type": "number"},
            "from_currency": {"type": "string"},
            "to_currency": {"type": "string"},
        },
        "required": ("amount", "from_currency", "to_currency"),
    },
    {
        "name": "search_docs",
        "description": "Full-text search over the documentation.",
        "params": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ("query",),
    },
    {
        "name": "set_reminder",
        "description": "Set a reminder after a number of minutes.",
        "params": {
            "task": {"type": "string"},
            "minutes": {"type": "integer", "minimum": 1, "maximum": 1440},
        },
        "required": ("task", "minutes"),
    },
)

#: Fixed query set. ``call`` is the expected call as
#: ``{"name": ..., "arguments": {...}}``, or ``None`` when no tool is
#: relevant and the correct response is ``NO_CALL`` (relevance
#: detection -- the abstention analogue of the lab's conservative
#: selection rule).
TOOLCALL_QUERIES: tuple[dict[str, Any], ...] = (
    {
        "id": "weather-paris",
        "question": "What is the weather in Paris in Celsius?",
        "call": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "C"}},
    },
    {
        "id": "weather-berlin",
        "question": "Should I bring a jacket in Berlin? Give the temperature in Fahrenheit.",
        "call": {"name": "get_weather", "arguments": {"city": "Berlin", "unit": "F"}},
    },
    {
        "id": "currency-usd-eur",
        "question": "Convert 100 US dollars to euros.",
        "call": {
            "name": "convert_currency",
            "arguments": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"},
        },
    },
    {
        "id": "currency-jpy-usd",
        "question": "How much is 250 yen in dollars?",
        "call": {
            "name": "convert_currency",
            "arguments": {"amount": 250, "from_currency": "JPY", "to_currency": "USD"},
        },
    },
    {
        "id": "docs-healing",
        "question": "Find documents about token healing.",
        "call": {"name": "search_docs", "arguments": {"query": "token healing", "top_k": 3}},
    },
    {
        "id": "docs-refund",
        "question": "Look up the refund policy; show only the top 1 result.",
        "call": {"name": "search_docs", "arguments": {"query": "refund policy", "top_k": 1}},
    },
    {
        "id": "reminder-plants",
        "question": "Remind me to water the plants in 30 minutes.",
        "call": {"name": "set_reminder", "arguments": {"task": "water the plants", "minutes": 30}},
    },
    {
        "id": "abstain-capital",
        "question": "What is the capital of France?",
        "call": None,
    },
    {
        "id": "abstain-summary",
        "question": (
            "Summarize this paragraph in one sentence: the lab renders exact token sequences."
        ),
        "call": None,
    },
)

#: The pinned abstention marker. A query with no relevant tool must be
#: answered with exactly this line and nothing else.
NO_CALL = "NO_CALL"

#: Response outcome vocabulary. ``not_text`` marks a non-string
#: response (fail-closed: excluded from rates); ``parse_failed`` marks
#: an unparseable line (fail-closed: counts as a failure on all three
#: rates, never silently dropped); ``call_failed`` marks a parsed call
#: that selects the wrong tool, extracts wrong arguments, or abstains
#: wrongly.
OUTCOMES = ("adherent", "parse_failed", "call_failed", "not_text")

_PARAM_TYPES = ("string", "integer", "number", "boolean")


def _tool_map() -> dict[str, dict[str, Any]]:
    """Map tool name -> tool definition (names are unique by validation)."""
    return {tool["name"]: tool for tool in TOOLS}


def validate_tool_def(tool: object) -> dict[str, Any]:
    """Validate one tool definition; returns the normalized definition.

    Raises ``ValueError`` on bad input (fail-fast: a malformed tool can
    never silently grade calls against the wrong contract).
    """
    if type(tool) is not dict:
        raise ValueError("tool definition must be a dict")
    name = tool.get("name")
    if type(name) is not str or not name or not name.replace("_", "").isalnum():
        raise ValueError(f"tool name must be a nonempty identifier, got {name!r}")
    params = tool.get("params")
    if type(params) is not dict or not params:
        raise ValueError(f"tool {name!r} needs a nonempty params dict")
    for param_name, param in params.items():
        if type(param_name) is not str or not param_name:
            raise ValueError(f"tool {name!r} has a non-string param name")
        if type(param) is not dict:
            raise ValueError(f"tool {name!r} param {param_name!r} must be a dict")
        ptype = param.get("type")
        if ptype not in _PARAM_TYPES:
            raise ValueError(f"tool {name!r} param {param_name!r} has unknown type {ptype!r}")
        enum = param.get("enum")
        if enum is not None and (
            type(enum) is not list or not enum or not all(type(v) is str for v in enum)
        ):
            raise ValueError(f"tool {name!r} param {param_name!r} has a bad enum")
        for bound in ("minimum", "maximum"):
            value = param.get(bound)
            if value is not None and (type(value) is bool or not isinstance(value, (int, float))):
                raise ValueError(f"tool {name!r} param {param_name!r} has a non-numeric {bound}")
    required = tool.get("required")
    if type(required) is not tuple or not required or not all(p in params for p in required):
        raise ValueError(f"tool {name!r} needs a nonempty required tuple over params")
    if type(tool.get("description")) is not str:
        raise ValueError(f"tool {name!r} needs a string description")
    return {
        "name": name,
        "description": tool["description"],
        "params": dict(params),
        "required": tuple(required),
    }


def validate_toolcall_queries(queries: object) -> tuple[dict[str, Any], ...]:
    """Validate the query set; returns normalized ``(id, question, call)`` dicts.

    Each ``call`` is ``None`` (abstain) or ``{"name", "arguments"}``
    validated against the tool's contract: the name must be a known
    tool, every required param present and type-valid, no unknown
    params, enum membership and numeric bounds honored. Raises
    ``ValueError`` on bad input.
    """
    if type(queries) is not tuple or not queries:
        raise ValueError("queries must be a nonempty tuple")
    tools = _tool_map()
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for query in queries:
        if type(query) is not dict:
            raise ValueError("each query must be a dict")
        qid = query.get("id")
        if type(qid) is not str or not qid or qid in seen:
            raise ValueError(f"query id must be a unique nonempty string, got {qid!r}")
        seen.add(qid)
        question = query.get("question")
        if type(question) is not str or not question:
            raise ValueError(f"query {qid!r} needs a nonempty question")
        call = query.get("call")
        if call is not None:
            call = _validate_expected_call(call, tools, where=f"query {qid!r}")
        normalized.append({"id": qid, "question": question, "call": call})
    return tuple(normalized)


def _validate_expected_call(
    call: object, tools: dict[str, dict[str, Any]], *, where: str
) -> dict[str, Any]:
    """Validate an expected call against the tool contracts."""
    if type(call) is not dict:
        raise ValueError(f"{where}: call must be a dict")
    name = call.get("name")
    tool = tools.get(name) if type(name) is str else None
    if tool is None:
        raise ValueError(f"{where}: unknown tool {name!r}")
    arguments = call.get("arguments")
    if type(arguments) is not dict:
        raise ValueError(f"{where}: arguments must be a dict")
    params = tool["params"]
    unknown = set(arguments) - set(params)
    if unknown:
        raise ValueError(f"{where}: unknown params {sorted(unknown)}")
    missing = [p for p in tool["required"] if p not in arguments]
    if missing:
        raise ValueError(f"{where}: missing required params {missing}")
    for param_name, value in arguments.items():
        _check_arg_value(value, params[param_name], where=f"{where} param {param_name!r}")
    return {"name": name, "arguments": dict(arguments)}


def _check_arg_value(value: object, param: dict[str, Any], *, where: str) -> None:
    """Validate one argument value against its param contract. Raises ``ValueError``."""
    ptype = param["type"]
    if ptype == "string":
        ok = type(value) is str
    elif ptype == "integer":
        ok = type(value) is int
    elif ptype == "number":
        ok = (
            (type(value) is int or type(value) is float)
            # skipcq: PYL-R0124 - self-comparison is the total NaN test on untrusted JSON values
            and value == value
            and value not in (float("inf"), float("-inf"))
        )
    else:  # boolean
        ok = type(value) is bool
    if not ok:
        raise ValueError(f"{where}: {value!r} is not a valid {ptype}")
    enum = param.get("enum")
    if enum is not None and value not in enum:
        raise ValueError(f"{where}: {value!r} not in enum {enum}")
    minimum = param.get("minimum")
    if minimum is not None and value < minimum:
        raise ValueError(f"{where}: {value!r} below minimum {minimum}")
    maximum = param.get("maximum")
    if maximum is not None and value > maximum:
        raise ValueError(f"{where}: {value!r} above maximum {maximum}")


def _describe_tool(tool: dict[str, Any]) -> str:
    """One-line pinned prose description of a tool for the prompt."""
    parts = []
    for param_name, param in tool["params"].items():
        spec = param["type"]
        if param.get("enum") is not None:
            spec += " " + str(param["enum"]).replace("'", '"')
        for bound in ("minimum", "maximum"):
            if param.get(bound) is not None:
                spec += f" {bound}={param[bound]}"
        mark = "*" if param_name in tool["required"] else ""
        parts.append(f"{param_name}{mark}: {spec}")
    return f"- {tool['name']}({', '.join(parts)}): {tool['description']}"


def render_prompt(query: dict[str, Any]) -> str:
    """Render the fixture prompt for one validated query.

    The prompt carries the pinned tool list, the question, and the
    one-line output contract, and ends with a fixture-identity marker
    ``(ref: toolcall/<id>)`` so scripted backends can key their policy
    per fixture deterministically. The marker is inert for real models.
    """
    tools_block = "\n".join(_describe_tool(validate_tool_def(tool)) for tool in TOOLS)
    return (
        "You have these tools (* marks required params):\n"
        f"{tools_block}\n"
        f"Question: {query['question']}\n"
        "Emit exactly one line: CALL <tool_name> <JSON arguments>, "
        "or NO_CALL if no tool fits. Nothing else.\n"
        f"(ref: toolcall/{query['id']})"
    )


def canonical_call(call: dict[str, Any] | None) -> str:
    """Canonical rendering of an expected call (the fixture ``expected``).

    Keys are sorted and separators fixed, so ``exact_match`` grades the
    primary verdict strictly; adherence is measured independently by
    :func:`classify_tool_call`.
    """
    if call is None:
        return NO_CALL
    args = json.dumps(call["arguments"], sort_keys=True, separators=(", ", ": "))
    return f"CALL {call['name']} {args}"


def build_toolcall_fixtures(
    queries: tuple[dict[str, Any], ...],
) -> tuple[Fixture, ...]:
    """Build lab ``Fixture`` objects from validated queries.

    Each fixture's ``expected`` is :func:`canonical_call` of the
    expected call, so the pipeline's ``exact_match`` grades strictly (a
    model must emit the canonical form byte-for-byte to pass); the
    adherence analysis measures selection/args/relevance independently
    of that strict verdict.
    """
    return tuple(
        Fixture(
            id=f"toolcall/{query['id']}",
            messages=(Message(role="user", content=render_prompt(query)),),
            expected=canonical_call(query["call"]),
            group="toolcall",
        )
        for query in queries
    )


def _parse_emitted(text: str) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Parse an emitted line into ``(tool_name, arguments, error)``.

    Returns ``(None, None, None)`` for the ``NO_CALL`` abstention.
    ``error`` is a short machine tag on parse failure, else ``None``.
    Never raises on string input.
    """
    stripped = text.strip()
    if stripped == NO_CALL:
        return None, None, None
    head, sep, tail = stripped.partition(" ")
    if sep == "" or head != "CALL":
        return None, None, "bad_head"
    name, sep, args_text = tail.partition(" ")
    if sep == "" or not name.replace("_", "").isalnum():
        return None, None, "bad_shape"
    try:
        arguments = json.loads(
            args_text,
            parse_constant=_reject_constant,
            parse_float=_reject_nonfinite_float,
        )
    except ValueError:
        return None, None, "bad_json"
    except RecursionError:
        # Deeply nested input: the C scanner recurses per nesting level and
        # raises outside JSONDecodeError/ValueError. Fail-closed, never escape
        # (a single degenerate answer must not kill the whole probe).
        return None, None, "bad_json"
    if type(arguments) is not dict:
        return None, None, "args_not_object"
    return name, arguments, None


def _reject_nonfinite_float(text: str) -> float:
    """Reject float literals that overflow to infinity (e.g. ``1e999``).

    ``parse_constant`` only fires for the ``NaN``/``Infinity`` literals;
    ``float()`` conversion of a huge exponent reaches ``inf`` without ever
    calling it, so the finiteness guarantee needs this hook too.
    """
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite float literal {text!r}")
    return value


def _reject_constant(value: str) -> object:
    """Reject NaN/Infinity/-Infinity JSON constants (fail-closed)."""
    raise ValueError(f"non-finite constant {value!r}")


def classify_tool_call(text: object, expected: dict[str, Any] | None) -> dict[str, Any]:
    """Classify one decoded response against the expected call.

    Returns ``{"outcome", "selection_ok", "args_ok", "relevance_ok"}``.
    ``selection_ok``: the parsed tool name matches the expected tool
    (emitting any call when ``expected`` is ``None`` is wrong).
    ``args_ok``: the parsed arguments match the expected arguments
    exactly under the tool's typed contract (only meaningful when
    selection is correct, else ``False``). ``relevance_ok``: the
    abstention decision is correct -- ``NO_CALL`` exactly when
    ``expected`` is ``None``. A correct abstention counts as correct
    selection (of no tool) with vacuously correct arguments, so a
    perfect model scores 1.0 on every rate. Unparseable lines are
    ``parse_failed`` and count as failures on all three rates
    (fail-closed, never silently dropped); non-string responses are
    ``not_text`` (excluded from rates). An unknown tool name can never
    count as correct selection.
    """
    if type(text) is not str:
        return {
            "outcome": "not_text",
            "selection_ok": False,
            "args_ok": False,
            "relevance_ok": False,
            "not_text": True,
        }
    name, arguments, error = _parse_emitted(text)
    if error is not None:
        return {
            "outcome": "parse_failed",
            "selection_ok": False,
            "args_ok": False,
            "relevance_ok": False,
        }
    should_abstain = expected is None
    abstained = name is None
    relevance_ok = abstained == should_abstain
    if abstained or should_abstain:
        # Abstention cells: a correct abstention IS correct selection
        # (of no tool) with vacuously correct arguments, so a perfect
        # model scores 1.0 on every rate -- the rates never carry a
        # structural ceiling from the abstain queries. A wrong
        # abstention (NO_CALL where a tool fits, or any call where
        # none fits) fails all three rates.
        ok = relevance_ok
        return {
            "outcome": "adherent" if ok else "call_failed",
            "selection_ok": ok,
            "args_ok": ok,
            "relevance_ok": relevance_ok,
        }
    # From here a call was emitted (name/arguments are not None) and a
    # call was expected (expected is not None): every abstention shape
    # returned above. cast (not assert -- S101) documents the narrowing.
    expected_call = cast(dict[str, Any], expected)
    call_args = cast(dict[str, Any], arguments)
    selection_ok = name == expected_call["name"]
    args_ok = selection_ok and _args_match(call_args, expected_call["arguments"])
    ok = selection_ok and args_ok and relevance_ok
    return {
        "outcome": "adherent" if ok else "call_failed",
        "selection_ok": selection_ok,
        "args_ok": args_ok,
        "relevance_ok": relevance_ok,
    }


def _args_match(arguments: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Strict typed argument equality: same key set, type-valid values.

    Extra or missing args fail; values must match exactly (numbers
    compare by value, so ``100`` and ``100.0`` are equal -- the JSON
    number contract, not the Python type contract).
    """
    if set(arguments) != set(expected):
        return False
    for key, want in expected.items():
        got = arguments[key]
        if type(want) is bool or type(got) is bool:
            if type(want) is not type(got) or want != got:
                return False
        elif isinstance(want, (int, float)) and isinstance(got, (int, float)):
            if got != want:
                return False
        elif type(want) is not type(got) or want != got:
            return False
    return True


def candidate_adherence(classifications: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-response classifications into candidate rates.

    ``not_text`` responses are excluded from the rates (fail-closed:
    never counted as failures, never silently dropped -- they shrink
    the denominator). Returns counts plus ``selection_rate``,
    ``args_rate``, ``relevance_rate`` and ``adherent_rate`` (``None``
    when nothing is evaluable).
    """
    n = len(classifications)
    evaluable = [c for c in classifications if not c.get("not_text")]
    n_evaluable = len(evaluable)
    counts = dict.fromkeys(OUTCOMES, 0)
    for outcome in OUTCOMES:
        counts[outcome] = sum(1 for c in evaluable if c["outcome"] == outcome)
    # not_text is the one outcome that only occurs off the evaluable
    # set: count it over all classifications so the exclusion is
    # visible, not silently dropped.
    counts["not_text"] = n - n_evaluable
    rates: dict[str, float | None] = {}
    for key, rate_key in (
        ("selection_ok", "selection_rate"),
        ("args_ok", "args_rate"),
        ("relevance_ok", "relevance_rate"),
    ):
        rates[rate_key] = sum(1 for c in evaluable if c[key]) / n_evaluable if n_evaluable else None
    rates["adherent_rate"] = counts["adherent"] / n_evaluable if n_evaluable else None
    return {"n": n, "n_evaluable": n_evaluable, **counts, **rates}


def query_calls(
    queries: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, Any] | None]:
    """Map fixture id -> expected call (or ``None`` for abstain) for validated queries."""
    return {f"toolcall/{query['id']}": query["call"] for query in queries}
