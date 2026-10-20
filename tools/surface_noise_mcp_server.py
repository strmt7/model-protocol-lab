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
therefore POSIX-only: on platforms without ``SIGALRM`` the sweep still
runs, but the budget degrades to best-effort (not enforced) rather than
failing the call.

Run from the repository root:
    python3 tools/surface_noise_mcp_server.py
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model_protocol_lab import surfacenoise as sn  # noqa: E402
from model_protocol_lab.core import Fixture, Message  # noqa: E402

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
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


# ---------------------------------------------------------------------------
# Backend construction (fail-closed; no network anywhere in this file)
# ---------------------------------------------------------------------------


def _build_backend(spec: dict) -> object:
    if type(spec) is not dict:
        raise MCPError(-32602, "params.backend must be an object")
    kind = spec.get("kind")
    if kind == "selftest":
        marker = spec.get("marker", "MAGICWORD")
        if type(marker) is not str or not marker:
            raise MCPError(-32602, "backend.marker must be a nonempty string")
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

        return LocalHFBackend(checkpoint, device=device)
    raise MCPError(-32602, 'backend.kind must be "selftest" or "local_hf"')


def _parse_fixtures(raw: object) -> tuple[Fixture, ...]:
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
        if type(fid) is not str or not fid:
            raise MCPError(-32602, f"fixtures[{i}].id must be a nonempty string")
        if fid in seen:
            raise MCPError(-32602, f"duplicate fixture id: {fid!r}")
        seen.add(fid)
        if type(messages) is not list or not messages:
            raise MCPError(-32602, f"fixtures[{i}].messages must be a nonempty list")
        parsed_messages = []
        for j, msg in enumerate(messages):
            if type(msg) is not dict:
                raise MCPError(-32602, f"fixtures[{i}].messages[{j}] must be an object")
            role, content = msg.get("role"), msg.get("content")
            if type(role) is not str or type(content) is not str:
                raise MCPError(-32602, f"fixtures[{i}].messages[{j}] needs str role/content")
            parsed_messages.append(Message(role=role, content=content))
        if type(expected) is not str:
            raise MCPError(-32602, f"fixtures[{i}].expected must be a string")
        group = item.get("group", "probe")
        if type(group) is not str or not group.strip():
            raise MCPError(-32602, f"fixtures[{i}].group must be a nonblank string")
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
    if type(raw) is not list or not raw or any(type(t) is not int or t < 0 for t in raw):
        raise MCPError(-32602, "params.boundary must be a nonempty list of token ids")
    return tuple(raw)


def _parse_prompt(raw: object) -> str:
    if type(raw) is not str or not raw.strip():
        raise MCPError(-32602, "params.prompt must be a nonempty string")
    if len(raw) > MAX_REQUEST_CHARS:
        raise MCPError(-32602, f"prompt exceeds {MAX_REQUEST_CHARS} characters")
    return raw


def _parse_optional_int(raw: object, name: str, lo: int, hi: int, default: int) -> int:
    if raw is None:
        return default
    if type(raw) is not int or isinstance(raw, bool) or not lo <= raw <= hi:
        raise MCPError(-32602, f"params.{name} must be an integer in [{lo}, {hi}]")
    return raw


def _text_block(text: str) -> dict:
    if len(text) > MAX_CONTENT_BYTES:
        text = text[:MAX_CONTENT_BYTES] + "\n…[truncated]"
    return {"type": "text", "text": text}


def _result(content_text: str, structured: dict) -> dict:
    return {"content": [_text_block(content_text)], "structuredContent": structured}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def tool_probe(params: dict) -> dict:
    """Run the frozen surface-noise protocol on one prompt (one sweep at a time)."""
    prompt = _parse_prompt(params.get("prompt"))
    fixtures = _parse_fixtures(params.get("fixtures"))
    boundary = _parse_boundary(params.get("boundary"))
    backend_spec = params.get("backend")
    if backend_spec is None:
        backend_spec = {"kind": "selftest"}
    if type(backend_spec) is not dict:
        raise ValueError("backend must be a dict or omitted")
    backend = _build_backend(backend_spec)
    max_new_tokens = _parse_optional_int(
        params.get("max_new_tokens"), "max_new_tokens", 1, 4096, 64
    )
    seed = _parse_optional_int(params.get("seed"), "seed", 0, 2**64 - 1, 0)

    fingerprint = backend.fingerprint  # type: ignore[attr-defined]
    protocol = sn.freeze_surface_noise_protocol(backend_fingerprint=fingerprint, fixtures=fixtures)
    cancel = threading.Event()

    # SIGALRM/ITIMER_REAL do not exist on Windows: guard the POSIX-only
    # enforcement and degrade to best-effort there (documented above).
    _has_alarm = getattr(signal, "SIGALRM", None) is not None and hasattr(signal, "setitimer")

    def _on_alarm(_signum: int, _frame: object) -> None:
        cancel.set()

    with _sweep_lock:
        previous = signal.signal(signal.SIGALRM, _on_alarm) if _has_alarm else None
        try:
            if _has_alarm:
                signal.setitimer(signal.ITIMER_REAL, SWEEP_TIMEOUT_SECONDS)
            try:
                report = sn.probe_surface_noise(
                    prompt,
                    fixtures,
                    backend,  # type: ignore[arg-type]
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


def tool_hardening_notes(params: dict) -> dict:
    """Surface a probe report's counterfactuals (informational, not a loop)."""
    report = params.get("report")
    if type(report) is not dict:
        raise MCPError(-32602, "params.report must be a probe result object")
    try:
        notes = sn.hardening_notes(report)
    except ValueError as exc:
        raise MCPError(-32602, f"invalid probe report: {exc}") from exc
    lines = [
        f"weakest operator: {notes['weakest_operator']} (stability {notes['weakest_stability']})",
        *notes["counterfactuals"],
        f"lint findings: {notes['lint_findings'] or 'none'}",
        notes["loop_guidance"],
    ]
    return _result("\n".join(lines), notes)


def tool_protocol_info(params: dict) -> dict:
    """Return the frozen instrument descriptor (no fixtures, no scoring)."""
    descriptor = sn.describe_instrument()
    summary = (
        f"{descriptor['instrument']}/{descriptor['protocol_version']}: "
        f"{len(descriptor['search_plan'])} operators. "
        f"{descriptor['adapter_rule']}"
    )
    return _result(summary, descriptor)


def tool_controls(params: dict) -> dict:
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


TOOLS = {
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


def handle_request(request: dict) -> dict | None:
    method = request.get("method")
    request_id = request.get("id")
    is_notification = request_id is None

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
            params = request.get("params") or {}
            name = params.get("name")
            if name not in TOOLS:
                raise MCPError(-32602, f"unknown tool: {name}")
            result = TOOLS[name]["handler"](params.get("arguments") or {})
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        raise MCPError(-32601, f"method not found: {method}")
    except MCPError as exc:
        if is_notification:
            return None
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": exc.code, "message": exc.message},
        }
    except Exception as exc:  # noqa: BLE001 -- the server never crashes on a tool failure
        if is_notification:
            return None
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"},
        }


def self_test() -> int:
    """Run one embedded probe end-to-end (offline) and exit 0 on success."""
    from model_protocol_lab.core import Fixture, Message

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
    if "result" not in tools_list:
        sys.stderr.write("self-test: tools/list failed\n")
        return 1
    discover = handle_request(
        {"jsonrpc": "2.0", "id": 2, "method": "server/discover", "params": {}}
    )
    if "result" not in discover:
        sys.stderr.write("self-test: server/discover failed\n")
        return 1
    sys.stderr.write("surface-noise-mcp-server self-test: OK\n")
    return 0


def serve() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            response: dict | None = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "parse error"},
            }
        else:
            if type(request) is not dict:
                response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "invalid request"},
                }
            else:
                response = handle_request(request)
        if response is not None:
            sys.stdout.write(json.dumps(response, default=str) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(serve())
