"""Stdio MCP server for the surface-noise self-consistency probe.

Gives coding agents first-class access to the probe: measure a prompt's
stability under surface-form noise, read the frozen protocol, run the
built-in controls, and surface hardening counterfactuals — all
deterministic, local, and offline, with zero LLM calls beyond the
backend under test.

Standard library only (json, sys, threading, signal). Speaks MCP over
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
import json
import math
import os
import signal
import sys
import threading
from typing import Any, Literal, NoReturn, cast

from model_protocol_lab import __version__, datasets
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
    # _build_backend owns the type check and raises MCPError(-32602)
    # for a non-object spec; a bare ValueError here was mapped to
    # -32603 (internal error) by the generic handler, reporting a
    # client-caused invalid-params error as a server fault (cycle 202).
    backend = _build_backend(backend_spec)
    max_new_tokens = _parse_optional_int(
        params.get("max_new_tokens"), "max_new_tokens", 1, 4096, 64
    )
    seed = _parse_optional_int(params.get("seed"), "seed", 0, 2**64 - 1, 0)

    fingerprint = backend.fingerprint
    protocol = sn.freeze_surface_noise_protocol(backend_fingerprint=fingerprint, fixtures=fixtures)
    cancel = threading.Event()

    # SIGALRM/ITIMER_REAL do not exist on Windows, and signal calls are
    # main-thread-only: guard the POSIX-only enforcement and degrade to
    # an unbounded sweep elsewhere (documented above) rather than
    # letting signal.signal's ValueError escape as -32603 from a worker
    # thread (BH2B-10). A pre-existing host timer is saved before arming
    # and re-armed at teardown, like the handler (BH2B-09; the cycle-116
    # doctor convention) -- the host's alarm is never silently eaten.
    _has_alarm = (
        getattr(signal, "SIGALRM", None) is not None
        and hasattr(signal, "setitimer")
        and hasattr(signal, "getitimer")
        and threading.current_thread() is threading.main_thread()
    )

    def _on_alarm(_signum: int, _frame: object) -> None:
        """On alarm."""
        cancel.set()

    with _sweep_lock:
        previous = signal.signal(signal.SIGALRM, _on_alarm) if _has_alarm else None
        old_timer = signal.getitimer(signal.ITIMER_REAL) if _has_alarm else (0.0, 0.0)
        try:
            if _has_alarm:
                signal.setitimer(signal.ITIMER_REAL, SWEEP_TIMEOUT_SECONDS)
            try:
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
            finally:
                if _has_alarm:
                    signal.setitimer(signal.ITIMER_REAL, 0)
        finally:
            if _has_alarm:
                signal.signal(signal.SIGALRM, previous)
                if old_timer[0] > 0:
                    signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])

    status = report.get("status")
    if status == "cancelled":
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


def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """Dispatch one JSON-RPC request; notifications are never executed."""
    # JSON-RPC notification discipline (oversight security S-13): a
    # frame with NO "id" member is a notification — it is never
    # answered and, for this server, never executed (a misdirected
    # notification must not start an expensive probe run). An
    # explicit "id": null is a request: it dispatches normally and
    # its errors are reported like any other request's.
    if "id" not in request:
        return None
    method = request.get("method")
    request_id = request.get("id")

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
            arguments = params.get("arguments") or {}
            if type(arguments) is not dict:
                raise MCPError(-32602, "arguments must be an object")
            result = TOOLS[name]["handler"](arguments)
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


def _write_response(response: dict[str, Any]) -> bool:
    """Write one response frame; False when the client has gone away.

    A routine client disconnect surfaces as BrokenPipeError on the
    flush (pre-fix the server died with a traceback, exit 120). The
    session is over at that point: report False so serve() ends it
    cleanly with exit 0, and point the stdout fd at devnull so the
    interpreter's shutdown flush cannot re-raise on the dead pipe
    (F3).
    """
    try:
        sys.stdout.write(json.dumps(response, default=str) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
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
        response = handle_request(request)
        if response is not None and not _write_response(response):
            break
    return 0


if __name__ == "__main__":
    sys.exit(serve())
