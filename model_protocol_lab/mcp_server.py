"""Stdio MCP server for the surface-noise self-consistency probe.

Gives coding agents first-class access to the probe: measure a prompt's
stability under surface-form noise, read the frozen protocol, run the
built-in controls, and surface hardening counterfactuals — all
deterministic, local, and offline, with zero LLM calls beyond the
backend under test.

Standard library only (json, sys, threading). Speaks MCP over
stdio using newline-delimited JSON-RPC 2.0. Protocol acceptance is
liberal on purpose: ``tools/list`` / ``tools/call`` work with no prior
``initialize`` (stateless servers must not require it), the 2024-11-05
``initialize`` handshake is also accepted but never required, and
``server/discover`` is implemented.

Caveat, stated plainly: the 2026-07-28 MCP spec revision (stateless
rewrite) could not be verified against the published spec text at build
time — only secondary write-ups were available. The server is therefore
designed to interoperate with both framings rather than to certify
against one. Re-check before claiming conformance.

Sweep bounds (FlitzZip pattern, POSIX enforcement): every sweep
dimension is capped in the tool schema, one sweep runs at a time
(threading lock), cancellation is cooperative between forwards with a
fail-closed ``status: "cancelled"`` (never a partial score). The 600 s
wall-clock budget is enforced with ``SIGALRM``/``ITIMER_REAL`` and is
therefore POSIX-only: on platforms without ``SIGALRM`` — and on calls
served from a non-main thread, where signal calls are unavailable —
the sweep still runs, but the budget degrades to best-effort (not
enforced) rather than failing the call. A host timer armed before the
sweep is saved and re-armed afterwards, so embedding the probe never
eats the caller's alarm.

Shipped in the wheel with the ``model-protocol-lab-mcp`` console
script. Run it as:

    model-protocol-lab-mcp
    python -m model_protocol_lab.mcp_server

A thin launcher remains at ``tools/surface_noise_mcp_server.py``
for checkouts, so ``python tools/surface_noise_mcp_server.py``
keeps working from a repository root.
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import sys
import threading
import time
from typing import Any, Literal, NoReturn, cast

from model_protocol_lab import __version__, datasets, platform_caps
from model_protocol_lab import surfacenoise as sn
from model_protocol_lab.core import Backend, Fixture, Message

PROTOCOL_VERSIONS = ["2024-11-05", "2026-07-28"]
SERVER_INFO = {"name": "mpl-surface-noise-probe", "version": "1.0.0"}

# Sweep bounds. Rationale documented per bound; none is user-raisable
# (a free n_seeds / timeout parameter would be a self-DoS primitive).
MAX_REQUEST_CHARS = 1024 * 1024  # 1 MiB prompt cap: memory-DoS prevention;
# the sweep multiplies input size across variants, so the cap is on the subject.
SWEEP_TIMEOUT_SECONDS = 600.0  # wall-clock budget per probe call; large
# fixture sets should be split across calls rather than given more time.
MAX_CONTENT_BYTES = 64 * 1024  # cap on any single text content block.

_sweep_lock = threading.Lock()


class MCPError(Exception):
    """A JSON-RPC error carrying its numeric code to the client."""

    def __init__(self, code: int, message: str):
        """Initialise the MCP error with its code and message."""
        self.code = code
        self.message = message
        super().__init__(message)


def _reject_surrogates(value: str, field: str) -> None:
    """Refuse lone surrogates as a client error (-32602).

    A lone surrogate cannot be encoded as UTF-8, so letting one reach
    the probe turns a client input defect into a server-side
    UnicodeEncodeError, which the generic handler maps to -32603
    (internal error) — the misclassification cycle 202 fixed for the
    backend spec and cycle 226 (B-13) closes for the remaining text
    fields.
    """
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in value):
        raise MCPError(-32602, f"{field} contains a lone surrogate")


# ---------------------------------------------------------------------------
# Backend construction (fail-closed; no network anywhere in this file)
# ---------------------------------------------------------------------------


def _build_backend(spec: dict[str, Any]) -> Backend:
    """Build backend."""
    if type(spec) is not dict:
        raise MCPError(-32602, "params.backend must be an object")
    kind = spec.get("kind")
    if kind == "selftest":
        marker = spec.get("marker", "MAGICWORD")
        if type(marker) is not str or not marker:
            raise MCPError(-32602, "backend.marker must be a nonempty string")
        _reject_surrogates(marker, "backend.marker")
        # Defined once in the measured module (model_protocol_lab.surfacenoise).
        return sn.SelfTestBackend(marker)
    if kind == "local_hf":
        checkpoint = spec.get("checkpoint")
        if type(checkpoint) is not str or not checkpoint:
            raise MCPError(-32602, "backend.checkpoint (path) is required")
        if not os.path.isdir(checkpoint):
            raise MCPError(-32602, f"checkpoint is not a directory: {checkpoint}")
        device = spec.get("device", "cpu")
        if device not in ("cpu", "cuda"):
            raise MCPError(-32602, 'backend.device must be "cpu" or "cuda"')
        from model_protocol_lab.backends import LocalHFBackend

        try:
            return LocalHFBackend(checkpoint, device=device)
        except Exception as exc:
            # Construction failure on client-supplied params (missing
            # weights, unreadable config) is an invalid-params outcome,
            # not a server fault: -32602 with a fixed message (the raw
            # exception text may carry host paths). This deliberately
            # reclassifies cycle 196's -32603 specimen; the moved pin
            # and its reasoning live in
            # tests/test_cycle196_oversight_fixes.py (cycle 227, C-04).
            raise MCPError(
                -32602, "local_hf backend could not be constructed from params.backend"
            ) from exc
    raise MCPError(-32602, 'backend.kind must be "selftest" or "local_hf"')


def _parse_fixtures(raw: object) -> tuple[Fixture, ...]:
    """Parse fixtures."""
    if type(raw) is not list or not raw:
        raise MCPError(-32602, "params.fixtures must be a nonempty list")
    if len(raw) > sn.MAX_FIXTURES:
        raise MCPError(-32602, f"at most {sn.MAX_FIXTURES} fixtures per call")
    fixtures: list[Fixture] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        if type(item) is not dict:
            raise MCPError(-32602, f"fixtures[{i}] must be an object")
        fid = item.get("id")
        messages = item.get("messages")
        expected = item.get("expected")
        if type(fid) is not str or not fid.strip():
            raise MCPError(-32602, f"fixtures[{i}].id must be a nonblank string")
        _reject_surrogates(fid, f"fixtures[{i}].id")
        if fid in seen:
            raise MCPError(-32602, f"duplicate fixture id: {fid!r}")
        seen.add(fid)
        if type(messages) is not list or not messages:
            raise MCPError(-32602, f"fixtures[{i}].messages must be a nonempty list")
        if len(messages) > datasets.MAX_MESSAGES:
            raise MCPError(
                -32602,
                f"fixtures[{i}].messages exceeds {datasets.MAX_MESSAGES} messages per fixture",
            )
        parsed_messages = []
        for j, msg in enumerate(messages):
            if type(msg) is not dict:
                raise MCPError(-32602, f"fixtures[{i}].messages[{j}] must be an object")
            role, content = msg.get("role"), msg.get("content")
            if type(role) is not str or type(content) is not str:
                raise MCPError(-32602, f"fixtures[{i}].messages[{j}] needs str role/content")
            if role not in ("user", "assistant"):
                # core.Message accepts only user/assistant roles; a
                # third role string previously flowed through
                # unvalidated into the probe.
                raise MCPError(
                    -32602,
                    f"fixtures[{i}].messages[{j}].role must be 'user' or 'assistant'",
                )
            # The membership check above guarantees the role; the cast
            # tells the checker what the check established.
            if len(content) > sn.MAX_FIXTURE_MESSAGE_CHARS:
                raise MCPError(
                    -32602,
                    f"fixtures[{i}].messages[{j}] content exceeds "
                    f"{sn.MAX_FIXTURE_MESSAGE_CHARS} characters",
                )
            _reject_surrogates(content, f"fixtures[{i}].messages[{j}].content")
            parsed_messages.append(
                Message(role=cast(Literal["user", "assistant"], role), content=content)
            )
        if type(expected) is not str:
            raise MCPError(-32602, f"fixtures[{i}].expected must be a string")
        _reject_surrogates(expected, f"fixtures[{i}].expected")
        group = item.get("group", "probe")
        if type(group) is not str or not group.strip():
            raise MCPError(-32602, f"fixtures[{i}].group must be a nonblank string")
        _reject_surrogates(group, f"fixtures[{i}].group")
        fixtures.append(
            Fixture(
                id=fid,
                messages=tuple(parsed_messages),
                expected=expected,
                group=group,
            )
        )
    return tuple(fixtures)


def _parse_boundary(raw: object) -> tuple[int, ...]:
    # Parity with the CLI's --boundary-ids discipline (oversight
    # security S-11): 1-16 IDs, each in [0, 2^31). Before the parity
    # fix, arrays of 200,000 IDs and IDs like 2**100 passed here
    # while the CLI refused them.
    """Parse boundary."""
    if (
        type(raw) is not list
        or not raw
        or len(raw) > 16
        or any(type(t) is not int or t < 0 or t >= 2**31 for t in raw)
    ):
        raise MCPError(-32602, "params.boundary must be 1-16 token ids, each in [0, 2^31)")
    return tuple(raw)


def _parse_prompt(raw: object) -> str:
    """Parse prompt."""
    if type(raw) is not str or not raw.strip():
        raise MCPError(-32602, "params.prompt must be a nonempty string")
    if len(raw) > MAX_REQUEST_CHARS:
        raise MCPError(-32602, f"prompt exceeds {MAX_REQUEST_CHARS} characters")
    _reject_surrogates(raw, "params.prompt")
    return raw


def _parse_optional_int(raw: object, name: str, lo: int, hi: int, default: int) -> int:
    """Parse optional int."""
    if raw is None:
        return default
    if type(raw) is not int or isinstance(raw, bool) or not lo <= raw <= hi:
        raise MCPError(-32602, f"params.{name} must be an integer in [{lo}, {hi}]")
    return raw


def _text_block(text: str) -> dict[str, Any]:
    """Return the text block."""
    if len(text) > MAX_CONTENT_BYTES:
        text = text[:MAX_CONTENT_BYTES] + "\n…[truncated]"
    return {"type": "text", "text": text}


def _result(content_text: str, structured: dict[str, Any]) -> dict[str, Any]:
    """Return the result."""
    return {"content": [_text_block(content_text)], "structuredContent": structured}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def tool_probe(params: dict[str, Any]) -> dict[str, Any]:
    """Run the frozen surface-noise protocol on one prompt (one sweep at a time)."""
    prompt = _parse_prompt(params.get("prompt"))
    fixtures = _parse_fixtures(params.get("fixtures"))
    boundary = _parse_boundary(params.get("boundary"))
    backend_spec = params.get("backend")
    if backend_spec is None:
        backend_spec = {"kind": "selftest"}
    # Cheap scalar params validate before any backend construction
    # (RTI-03): construction can load a checkpoint, so an out-of-range
    # cap or seed must be rejected before it runs -- pre-fix the
    # backend was built first and the -32602 arrived only afterwards.
    max_new_tokens = _parse_optional_int(
        params.get("max_new_tokens"), "max_new_tokens", 1, 4096, 64
    )
    seed = _parse_optional_int(params.get("seed"), "seed", 0, 2**64 - 1, 0)
    # _build_backend owns the type check and raises MCPError(-32602)
    # for a non-object spec; a bare ValueError here was mapped to
    # -32603 (internal error) by the generic handler, reporting a
    # client-caused invalid-params error as a server fault (cycle 202).
    backend = _build_backend(backend_spec)

    fingerprint = backend.fingerprint
    protocol = sn.freeze_surface_noise_protocol(backend_fingerprint=fingerprint, fixtures=fixtures)
    cancel = threading.Event()
    # Client cancellation (ORCH-02): when this call was dispatched
    # through handle_request, its registered event is merged in -- a
    # daemon watcher copies it into the sweep's own event, so the
    # SIGALRM budget and the client share one cancellation surface.
    client_cancel = current_call_event()
    if client_cancel is not None and client_cancel is not cancel:

        def _watch_client() -> None:
            """Watch client."""
            client_cancel.wait()
            cancel.set()

        threading.Thread(target=_watch_client, daemon=True).start()

    # The budget deadline is platform_caps.alarm_deadline -- the single
    # typed implementation doctor's render watchdog also uses.
    # SIGALRM/ITIMER_REAL do not exist on Windows, and signal calls are
    # main-thread-only: where the deadline cannot arm, the context
    # manager yields unarmed and the sweep runs unbounded (the
    # best-effort degradation documented above) rather than letting a
    # signal error escape as -32603 from a worker thread (BH2B-10).
    # Expiry only sets the cancel event; the sweep polls it between
    # forwards. A pre-existing host timer is saved before arming and
    # re-armed at teardown by the context manager (BH2B-09) -- the
    # host's alarm is never silently eaten.
    with _sweep_lock, platform_caps.alarm_deadline(SWEEP_TIMEOUT_SECONDS, cancel.set):
        report = sn.probe_surface_noise(
            prompt,
            fixtures,
            backend,
            protocol,
            boundary=boundary,
            max_new_tokens=max_new_tokens,
            seed=seed,
            cancel=cancel,
        )

    status = report.get("status")
    if status == "cancelled" and client_cancel is not None and client_cancel.is_set():
        summary = (
            "probe cancelled by client: "
            f"{report.get('completed_variants')}/{report.get('total_variants')} "
            "variants completed. No score rendered (fail-closed)."
        )
    elif status == "cancelled":
        summary = (
            f"probe cancelled after {SWEEP_TIMEOUT_SECONDS:.0f}s budget: "
            f"{report.get('completed_variants')}/{report.get('total_variants')} "
            f"variants completed. No score rendered (fail-closed)."
        )
    elif status == "unavailable":
        summary = f"probe unavailable: {report.get('reason')}. No score rendered."
    else:
        den = report["denominators"]
        summary = (
            f"surface-noise probe ok: {den['n_effective']}/{den['n_total']} effective "
            f"variants, baseline {report['baseline']['n_pass']}/"
            f"{report['n_fixtures']} pass. "
            f"Weakest operator: {sn.hardening_notes(report)['weakest_operator']}. "
            f"{sn.ADAPTER_RULE}"
        )
    return _result(summary, report)


def tool_hardening_notes(params: dict[str, Any]) -> dict[str, Any]:
    """Surface a probe report's counterfactuals (informational, not a loop)."""
    report = params.get("report")
    if type(report) is not dict:
        raise MCPError(-32602, "params.report must be a probe result object")
    try:
        notes = sn.hardening_notes(report)
    except (ValueError, TypeError, AttributeError) as exc:
        # A type-confused report field escapes hardening_notes as a
        # TypeError/AttributeError, not a ValueError: it is still a
        # client-supplied invalid-params problem, never a server
        # fault, so it maps to -32602 like the ValueError case (F4).
        raise MCPError(-32602, f"invalid probe report: {exc}") from exc
    lines = [
        f"weakest operator: {notes['weakest_operator']} (stability {notes['weakest_stability']})",
        *notes["counterfactuals"],
        f"lint findings: {notes['lint_findings'] or 'none'}",
        notes["loop_guidance"],
    ]
    return _result("\n".join(lines), notes)


def tool_protocol_info(_params: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen instrument descriptor (no fixtures, no scoring)."""
    descriptor = sn.describe_instrument()
    summary = (
        f"{descriptor['instrument']}/{descriptor['protocol_version']}: "
        f"{len(descriptor['search_plan'])} operators. "
        f"{descriptor['adapter_rule']}"
    )
    return _result(summary, descriptor)


def tool_controls(params: dict[str, Any]) -> dict[str, Any]:
    """Run the built-in control harness; check sturdy > fragile."""
    boundary = _parse_boundary(params.get("boundary"))
    include_heldout = params.get("include_heldout", False)
    if type(include_heldout) is not bool:
        raise MCPError(-32602, "params.include_heldout must be a bool")
    result = sn.run_surface_noise_controls(
        boundary=boundary,
        include_heldout=include_heldout,
    )
    summary = (
        f"controls discriminates={result['discriminates']}: "
        + "; ".join(
            f"{c['name']}({c['expectation']}): "
            f"{c['mean_stability'] if c['mean_stability'] is not None else c['status']}"
            for c in result["controls"]
        )
        + f". {result['circularity_note']}"
    )
    return _result(summary, result)


TOOLS: dict[str, dict[str, Any]] = {
    "probe_surface_noise_stability": {
        "description": (
            "Measure a prompt's task-success stability under a frozen set of "
            "surface-form text perturbations (typos, case flips, whitespace, "
            "Unicode variants, punctuation stripping). Reports per-operator "
            "stability and nearest observed flips. Surface noise only -- not "
            "a prompt-quality certificate (see the Adapter Rule in the result)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The prompt text under test"},
                "fixtures": {
                    "type": "array",
                    "description": (
                        "1-50 fixtures: {id, messages: [{role, content}], expected, group?}"
                    ),
                    "items": {"type": "object"},
                },
                "backend": {
                    "type": "object",
                    "description": (
                        '{"kind": "selftest"} or {"kind": "local_hf", "checkpoint": path}'
                    ),
                },
                "boundary": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Answer-boundary token ids (cuts the decoded answer)",
                },
                "max_new_tokens": {"type": "integer", "default": 64},
                "seed": {"type": "integer", "default": 0},
            },
            "required": ["prompt", "fixtures", "boundary"],
        },
        "handler": tool_probe,
    },
    "surface_noise_hardening_notes": {
        "description": (
            "Surface a probe report's minimal-flip counterfactuals and weakest "
            "operator family as informational hardening notes. Does not run an "
            "optimization loop."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "report": {
                    "type": "object",
                    "description": "The probe result object from probe_surface_noise_stability",
                }
            },
            "required": ["report"],
        },
        "handler": tool_hardening_notes,
    },
    "surface_noise_protocol_info": {
        "description": (
            "Return the frozen instrument descriptor: operator set, search "
            "plan, distance definition, verdict function, radius framing, "
            "and comparability rule. Inspect the instrument before using it. "
            "No fixtures, no scoring."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
        "handler": tool_protocol_info,
    },
    "surface_noise_controls": {
        "description": (
            "Run the built-in sturdy/fragile control prompts and check that "
            "sturdy controls out-score fragile ones. Validates the "
            "implementation against the specification, not against "
            "real-world prompt failure."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "boundary": {"type": "array", "items": {"type": "integer"}},
                "include_heldout": {"type": "boolean", "default": False},
            },
            "required": ["boundary"],
        },
        "handler": tool_controls,
    },
}


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 over stdio (newline-delimited)
# ---------------------------------------------------------------------------


def _valid_request_id(value: Any) -> bool:
    """A JSON-RPC 2.0 id is a string, a number, or null.

    Exact-type checks: a bool is none of those here (``type(True)``
    is ``bool``, not ``int``), and fractional/exotic objects cannot
    be echoed back to the client.
    """
    return value is None or type(value) is str or type(value) in (int, float)


# Client cancellation (ORCH-02): a per-request registry mapping the
# id of each in-flight tools/call to its cancellation event. The
# serve loop dispatches tool calls on worker threads, so a
# ``notifications/cancelled`` frame read while a call is running can
# set that call's event; the surface-noise tool merges the
# registered event with its own SIGALRM budget event. At stdio EOF
# every outstanding event is set -- a disconnected client cancels
# its in-flight work by definition. A client must not reuse an id
# while its call is in flight; if one does, both dispatches are
# tracked under that id (BH12-4) and a cancellation naming the id
# sets every event registered under it -- the client cancelled
# that id, and both calls claim to be it.
_IN_FLIGHT: dict[Any, set[threading.Event]] = {}
_IN_FLIGHT_LOCK = threading.Lock()
_CURRENT_CALL = threading.local()


def current_call_event() -> threading.Event | None:
    """Return the cancellation event of the tool call running on
    this thread, or None outside a tools/call dispatch."""
    return getattr(_CURRENT_CALL, "event", None)


def _register_call(request_id: Any) -> threading.Event:
    """Register call."""
    event = threading.Event()
    with _IN_FLIGHT_LOCK:
        _IN_FLIGHT.setdefault(request_id, set()).add(event)
    return event


def _complete_call(request_id: Any, event: threading.Event) -> None:
    """Complete call."""
    with _IN_FLIGHT_LOCK:
        events = _IN_FLIGHT.get(request_id)
        if events is not None:
            events.discard(event)
            if not events:
                del _IN_FLIGHT[request_id]


def _cancel_call(request_id: Any) -> None:
    """Cancel call."""
    with _IN_FLIGHT_LOCK:
        events = list(_IN_FLIGHT.get(request_id, ()))
    for event in events:
        event.set()


def _cancel_all_calls() -> None:
    """Cancel all calls."""
    with _IN_FLIGHT_LOCK:
        events = [event for group in _IN_FLIGHT.values() for event in group]
    for event in events:
        event.set()


def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """Dispatch one JSON-RPC request; notifications are never executed."""
    # JSON-RPC notification discipline (oversight security S-13): a
    # frame with NO "id" member is a notification — it is never
    # answered and, for this server, never executed (a misdirected
    # notification must not start an expensive probe run). An
    # explicit "id": null is a request: it dispatches normally and
    # its errors are reported like any other request's.
    # The single exception is notifications/cancelled (ORCH-02): it
    # executes nothing -- it only sets the registered cancellation
    # event of the in-flight call it names, which is the MCP
    # protocol's own cancellation contract.
    if "id" not in request:
        if request.get("method") == "notifications/cancelled":
            params = request.get("params")
            if (
                type(params) is dict
                and "requestId" in params
                and _valid_request_id(params["requestId"])
            ):
                _cancel_call(params["requestId"])
        return None
    method = request.get("method")
    request_id = request.get("id")
    # Envelope validation (RTI-06; JSON-RPC 2.0 section 5.1): a frame
    # that is not a valid Request object -- wrong or missing jsonrpc
    # version, an id that is not a string/number/null, a non-string
    # method -- is an Invalid Request (-32600) and never dispatches.
    # When the id itself is invalid it cannot be echoed, so the error
    # is reported with a null id, as the spec requires. (Notifications
    # returned above, before this check: they are never answered, so
    # their envelope is never judged here.)
    if (
        request.get("jsonrpc") != "2.0"
        or not _valid_request_id(request_id)
        or type(method) is not str
    ):
        return {
            "jsonrpc": "2.0",
            "id": request_id if _valid_request_id(request_id) else None,
            "error": {"code": -32600, "message": "invalid request"},
        }

    try:
        # Stateless: no initialize required. The 2024-11-05 handshake is
        # accepted for older clients but never required.
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSIONS[0],
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                },
            }
        if method == "notifications/initialized":
            return None
        if method == "server/discover":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "serverInfo": SERVER_INFO,
                    "protocolVersions": PROTOCOL_VERSIONS,
                    "capabilities": {"tools": {}},
                },
            }
        if method == "ping":
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": name,
                            "description": spec["description"],
                            "inputSchema": spec["inputSchema"],
                        }
                        for name, spec in TOOLS.items()
                    ]
                },
            }
        if method == "tools/call":
            params = request.get("params")
            if params is None:
                params = {}
            if type(params) is not dict:
                raise MCPError(-32602, "params must be an object")
            name = params.get("name")
            if type(name) is not str or name not in TOOLS:
                raise MCPError(-32602, f"unknown tool: {name}")
            # No falsey coercion (RTI-06): an absent or explicit-null
            # arguments member means "no arguments"; any other
            # non-object value ([], false, 0, "") is invalid params,
            # never silently an empty call that executes the tool.
            arguments = params.get("arguments")
            if arguments is None:
                arguments = {}
            if type(arguments) is not dict:
                raise MCPError(-32602, "arguments must be an object")
            call_event = _register_call(request_id)
            _CURRENT_CALL.event = call_event
            try:
                result = TOOLS[name]["handler"](arguments)
            finally:
                _CURRENT_CALL.event = None
                _complete_call(request_id, call_event)
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        raise MCPError(-32601, f"method not found: {method}")
    except MCPError as exc:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": exc.code, "message": exc.message},
        }
    except Exception:  # noqa: BLE001 -- the server never crashes on a tool failure
        # Fixed text (oversight security S-12): the generic path used
        # to echo the exception's type and message, which leaks
        # library internals (and, for UnicodeEncodeError, hostile
        # input bytes) to the client. The code stays -32603.
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32603, "message": "internal error"},
        }


def self_test() -> int:
    """Run one embedded probe end-to-end (offline) and exit 0 on success."""

    backend = sn.SelfTestBackend()
    fixtures = (
        Fixture(
            id="self-test-1",
            messages=(Message(role="user", content="anything"),),
            expected="BLUE",
            group="self-test",
        ),
    )
    protocol = sn.freeze_surface_noise_protocol(
        backend_fingerprint=backend.fingerprint, fixtures=fixtures
    )
    report = sn.probe_surface_noise("MAGICWORD please", fixtures, backend, protocol, boundary=(7,))
    if report.get("status") != "ok":
        sys.stderr.write(f"self-test probe failed: {report.get('status')}\n")
        return 1
    # Exercise each tool's input-validation path against its declared schema.
    tools_list = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    if tools_list is None or "result" not in tools_list:
        sys.stderr.write("self-test: tools/list failed\n")
        return 1
    discover = handle_request(
        {"jsonrpc": "2.0", "id": 2, "method": "server/discover", "params": {}}
    )
    if discover is None or "result" not in discover:
        sys.stderr.write("self-test: server/discover failed\n")
        return 1
    sys.stderr.write("surface-noise-mcp-server self-test: OK\n")
    return 0


# Framing bound (oversight security S-5): one request line may be at
# most 16 MiB — far above any legitimate frame (a 1 MiB prompt plus
# 50 capped fixtures) — so an unterminated line can no longer grow
# the input buffer without bound.
MAX_LINE_CHARS = 16 * 1024 * 1024


def _parse_error_response() -> dict[str, Any]:
    """Parse error response."""
    return {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32700, "message": "parse error"},
    }


def _unique_frame_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Duplicate keys in a frame are a parse error (oversight
    security S-13): every JSON loader in the repo rejects duplicate
    fields; the MCP frame parser now applies the same discipline
    instead of Python's last-wins default.
    """
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate key in frame")
        obj[key] = value
    return obj


def _reject_frame_constant(value: str) -> NoReturn:
    """Non-finite constants are a parse error, as in every other JSON
    loader in the repo: before cycle 202 the frame parser had no
    parse_constant hook, so ``{"id": NaN}`` was accepted and the NaN
    was echoed back in the response — invalid JSON on the wire.
    """
    raise ValueError(f"non-finite constant {value!r}")


def _reject_frame_float(text: str) -> float:
    """Overflow float literals (``1e999``) are a parse error too:
    float() conversion reaches inf without ever calling
    parse_constant, so the constant hook alone leaves the hole open
    (the same pair of hooks toolcall.py carries).
    """
    value = float(text)
    if math.isnan(value) or value in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite float literal {text!r}")
    return value


_RESPONSE_LOCK = threading.Lock()


def _write_response(response: dict[str, Any]) -> bool:
    """Write one response frame; False when the client has gone away.

    A routine client disconnect surfaces as a pipe error on the
    flush — BrokenPipeError on POSIX, the WinError 109/232 or
    EINVAL shapes on Windows (see :func:`_client_gone`; pre-fix
    the server died with a traceback, exit 120). The
    session is over at that point: report False so serve() ends it
    cleanly with exit 0, and point the stdout fd at devnull so the
    interpreter's shutdown flush cannot re-raise on the dead pipe
    (F3). Writes are serialized on a module lock: tool calls run on
    worker threads (ORCH-02), so two responses can never interleave
    on the wire.
    """
    with _RESPONSE_LOCK:
        return _write_response_locked(response)


def _client_gone(exc: OSError) -> bool:
    """Whether a stdout write error means the client went away.

    POSIX reports a dead pipe as BrokenPipeError; Windows
    reports the same event as ERROR_BROKEN_PIPE (109) or
    ERROR_NO_DATA (232), or as EINVAL from the write itself —
    none of which are BrokenPipeError, so a BrokenPipeError-only
    handler lets the server die with exit 120 on Windows
    (RR-18 follow-up). Anything else is a real write failure
    and must propagate.
    """
    if isinstance(exc, BrokenPipeError):
        return True
    if getattr(exc, "winerror", None) in (109, 232):
        return True
    return os.name == "nt" and exc.errno == errno.EINVAL


def _write_response_locked(response: dict[str, Any]) -> bool:
    """Write response locked."""
    try:
        sys.stdout.write(json.dumps(response, default=str) + "\n")
        sys.stdout.flush()
    except OSError as exc:
        if not _client_gone(exc):
            raise
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(devnull, sys.stdout.fileno())
            finally:
                os.close(devnull)
        except OSError:
            pass
        return False
    return True


#: Bounds for the stdio-EOF shutdown (ORCH-02): in-flight calls
#: share ONE grace window to finish, then every outstanding event
#: is set and the workers share ONE drain window before the
#: server exits (BH12-3).
_EOF_GRACE_SECONDS = 2.0
_EOF_DRAIN_SECONDS = 10.0


def serve(argv: list[str] | None = None) -> int:
    """Serve newline-delimited JSON-RPC on stdio until EOF, or self-test.

    Arguments are parsed, never silently ignored (cycle 226, B-18):
    ``--help`` prints usage and exits 0, ``--version`` prints the
    server version and exits 0, and an unknown argument exits 2 —
    previously any argument other than ``--self-test`` was dropped
    and the server blocked on stdin as if serving.
    """
    parser = argparse.ArgumentParser(
        prog="model-protocol-lab-mcp",
        description=(
            "Stdio MCP server for the surface-noise self-consistency probe. "
            "Speaks newline-delimited JSON-RPC 2.0 on stdin/stdout until EOF."
        ),
    )
    parser.add_argument(
        "--self-test", action="store_true", help="run the built-in self-test and exit"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"model-protocol-lab-mcp {__version__}",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    stdin = sys.stdin.buffer
    workers: set[threading.Thread] = set()
    # Responses are written strictly in frame order (the server's
    # established wire contract): tool calls compute concurrently on
    # worker threads, but each response waits for its turn, and an
    # inline response waits for every earlier worker to flush.
    write_cond = threading.Condition()
    write_turn = [0]
    next_seq = [0]

    def _dispatch_call(frame: dict[str, Any], seq: int) -> None:
        """Dispatch call."""
        response = None
        try:
            response = handle_request(frame)
        finally:
            with write_cond:
                while write_turn[0] != seq:
                    write_cond.wait(timeout=5)
                try:
                    if response is not None:
                        _write_response(response)
                finally:
                    write_turn[0] += 1
                    write_cond.notify_all()
                    workers.discard(threading.current_thread())

    while True:
        raw = stdin.readline(MAX_LINE_CHARS + 1)
        if not raw:
            break
        if len(raw) > MAX_LINE_CHARS:
            # Overlong line: fail this frame closed and discard
            # through its newline so framing resynchronises on the
            # next line instead of the tail being parsed as a frame
            # of its own.
            while raw and not raw.endswith(b"\n"):
                raw = stdin.readline(MAX_LINE_CHARS + 1)
            if not _write_response(_parse_error_response()):
                break
            if not raw:
                break
            continue
        try:
            line = raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            if not _write_response(_parse_error_response()):
                break
            continue
        if not line:
            continue
        try:
            # The parse boundary normalises every malformed-frame
            # class into -32700 (oversight security S-3): syntax,
            # duplicate keys (ValueError from the hook), the
            # interpreter's integer-length limit (ValueError), and
            # pathological nesting depth (RecursionError). Only
            # JSONDecodeError was caught before, and deeply-nested
            # JSON killed the process.
            request = json.loads(
                line,
                object_pairs_hook=_unique_frame_object,
                parse_constant=_reject_frame_constant,
                parse_float=_reject_frame_float,
            )
        except (RecursionError, ValueError):
            if not _write_response(_parse_error_response()):
                break
            continue
        if type(request) is not dict:
            if not _write_response(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "invalid request"},
                }
            ):
                break
            continue
        if request.get("method") == "tools/call" and "id" in request:
            # Tool calls run on a worker thread (ORCH-02): the read
            # loop keeps reading, so a notifications/cancelled frame
            # or an EOF can reach a call while it is still running.
            seq = next_seq[0]
            next_seq[0] += 1
            worker = threading.Thread(target=_dispatch_call, args=(request, seq), daemon=True)
            workers.add(worker)
            worker.start()
            continue
        response = handle_request(request)
        if response is not None:
            with write_cond:
                while write_turn[0] != next_seq[0]:
                    write_cond.wait(timeout=5)
            if not _write_response(response):
                break
    # EOF (or a dead client): in-flight calls first get a bounded
    # grace to finish -- a client that half-closes after queueing
    # its requests still expects their responses. Anything still
    # running after the grace is abandoned work: every outstanding
    # event is set and the workers get a bounded drain before the
    # server exits. Both bounds are single shared deadlines across
    # all workers (BH12-3), never per-worker timeouts, so shutdown
    # latency does not grow with the number of in-flight calls.
    grace_deadline = time.monotonic() + _EOF_GRACE_SECONDS
    for worker in list(workers):
        worker.join(timeout=max(0.0, grace_deadline - time.monotonic()))
    _cancel_all_calls()
    drain_deadline = time.monotonic() + _EOF_DRAIN_SECONDS
    for worker in list(workers):
        worker.join(timeout=max(0.0, drain_deadline - time.monotonic()))
    return 0


if __name__ == "__main__":
    sys.exit(serve())
