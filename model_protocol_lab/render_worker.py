"""Owned worker process for checkpoint-controlled chat-template renders.

A checkpoint's Jinja ``chat_template`` is attacker-influenced code: a
legal template can burn CPU in a loop or materialize gigabytes of
output. Rendering it in the CLI's own process (the pre-W4 design)
contained those failure modes with a SIGALRM watchdog and a transient
address-space cap — two guards that degrade *silently* exactly where
containment matters most: the watchdog cannot arm off the main thread
or on Windows, and without ``resource``/``/proc`` the cap cannot
install, so the render ran unbounded in the host process (RR-10).

This module is the replacement boundary. :class:`RenderSession` (the
parent side, used by :mod:`model_protocol_lab.doctor`) spawns one
worker per doctor-section invocation — ``python -m
model_protocol_lab.render_worker`` on the parent's own interpreter, so
the transformers/jinja2 versions are the parent's by construction and
are cross-checked at the init handshake. The worker constructs the
checkpoint tokenizer, then serves render/encode requests over a
length-prefixed JSON frame protocol on its stdio. Containment is
structural, not best-effort:

* the parent enforces a per-request deadline on every platform and
  from every thread — on expiry the worker is killed, the pair
  degrades to ``"unavailable"``, and a fresh worker is respawned for
  the next pair;
* the worker renders under the same transient address-space cap as
  before (now process-local, inside the worker, via
  :mod:`model_protocol_lab.platform_caps`), so a memory bomb raises
  there or kills at most the worker — never the parent;
* the rendered-length cap is enforced in the worker *before* the text
  crosses the wire, and both directions enforce a hard frame ceiling,
  so a hostile or broken worker cannot make the parent read
  unboundedly;
* three consecutive worker deaths trip a circuit breaker: the section
  fails closed with the fixed detail ``render_worker_failed``. There
  is no in-process fallback — a fallback would be the RR-10 failure
  surface under a new name (worker-or-refuse, audit decision D-1).

The worker never sends exception text: render/encode failures cross
the wire as fixed-vocabulary codes, the same sanitization doctrine as
``doctor._loader_failure_code``. Its stderr is /dev/null for the same
reason — diagnostics from checkpoint-influenced code must not reach
the operator's terminal.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import queue
import struct
import subprocess  # nosec B404 -- spawning the owned worker is this module's purpose; see the Popen call site
import sys
import threading
from collections.abc import Callable
from importlib import metadata
from typing import IO, Any

from . import platform_caps

#: Rendered-text cap, enforced in the worker before the text crosses
#: the wire. Mirrors ``doctor._RENDERED_MAX_CHARS`` (the parent keeps
#: its own post-render check as well); a test pins the two equal.
MAX_RENDERED_CHARS = 1_000_000

#: Hard ceiling on one protocol frame, either direction. The largest
#: legitimate frame is an encode reply for a cap-sized render (at most
#: MAX_RENDERED_CHARS token ids) or a JSON-escaped cap-sized text —
#: both far below this; the ceiling exists so a hostile or broken peer
#: cannot make either side allocate unboundedly on a length prefix.
_MAX_FRAME_BYTES = 33_554_432

#: Address-space headroom for one render inside the worker — the same
#: 256 MiB the in-process cap used (doctor.py, pre-W4).
_HEADROOM_BYTES = 256 * 1024 * 1024

#: Consecutive worker deaths that trip the circuit breaker.
_MAX_CONSECUTIVE_DEATHS = 3

#: Liveness bound for the init handshake (one tokenizer load). Not a
#: measurement value and never recorded; a wedged load is a broken
#: environment, not a slow one.
_INIT_DEADLINE_S = 60.0

#: Default per-request deadline: the pre-W4 render CPU budget.
_DEFAULT_BUDGET_S = 10.0

#: How the parent spawns the worker. A module-level constant (read at
#: session construction) so tests can substitute a scripted fake worker
#: through the real protocol.
WORKER_ARGV: tuple[str, ...] = (
    sys.executable,
    "-m",
    "model_protocol_lab.render_worker",
)


class RenderWorkerError(Exception):
    """Base class for render-worker failures."""


class RenderFailed(RenderWorkerError):
    """The worker answered with an error/oversize status.

    Pair-level: the worker is alive and healthy; this one render (or
    encode) failed or exceeded the rendered cap. Degrades one pair to
    ``"unavailable"``, exactly like a render exception did pre-W4.
    """


class WorkerDied(RenderWorkerError):
    """The worker died or broke the protocol mid-request.

    Transport-level: timeout kill, EOF, a malformed or oversize frame.
    The current pair degrades; a fresh worker is spawned for the next
    request. Consecutive deaths trip the circuit breaker (see
    :class:`WorkerUnavailable`).
    """


class WorkerUnavailable(RenderWorkerError):
    """The section cannot be served at all; ``code`` is the detail.

    Raised when the worker cannot be spawned, when the init handshake
    reports a fixed failure (``transformers_not_installed`` /
    ``tokenizer_load_failed`` — the pre-W4 section details, carried
    verbatim), or when the circuit breaker trips
    (``render_worker_failed``, the one detail code W4 adds).
    """

    def __init__(self, code: str) -> None:
        """Init."""
        super().__init__(code)
        self.code = code


def _encode_frame(payload: dict[str, Any]) -> bytes:
    """Encode frame."""
    body = json.dumps(payload).encode("utf-8")
    return struct.pack(">I", len(body)) + body


def _send_frame(stream: IO[bytes], payload: dict[str, Any]) -> None:
    """Send frame."""
    stream.write(_encode_frame(payload))
    stream.flush()


def _read_exact(stream: IO[bytes], count: int) -> bytes | None:
    """Read exact."""
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(stream: IO[bytes]) -> dict[str, Any] | None:
    """Read one frame; ``None`` on EOF or an over-ceiling announcement.

    A malformed JSON body raises — the worker treats that as a dead
    channel and exits; the parent's reader thread classifies it as a
    closed worker.
    """
    header = _read_exact(stream, 4)
    if header is None:
        return None
    (size,) = struct.unpack(">I", header)
    if size > _MAX_FRAME_BYTES:
        return None
    body = _read_exact(stream, size)
    if body is None:
        return None
    parsed = json.loads(body)
    return parsed if type(parsed) is dict else None


class _FrameReader(threading.Thread):
    """Pump worker frames into a queue; any transport failure closes it.

    Runs as a daemon thread so the parent can wait on a frame with a
    deadline (the containment primitive the SIGALRM watchdog used to
    approximate, now portable to every thread and platform). A frame
    announcing more than the ceiling is a protocol violation: the pump
    stops without reading the body, so the parent never allocates on
    a hostile length prefix.
    """

    def __init__(self, stream: IO[bytes], inbox: queue.Queue[tuple[str, Any]]) -> None:
        """Init."""
        super().__init__(daemon=True, name="mpl-render-worker-reader")
        self._stream = stream
        self._inbox = inbox

    def run(self) -> None:
        """Run."""
        try:
            while True:
                header = _read_exact(self._stream, 4)
                if header is None:
                    self._inbox.put(("closed", None))
                    return
                (size,) = struct.unpack(">I", header)
                if size > _MAX_FRAME_BYTES:
                    self._inbox.put(("closed", None))
                    return
                body = _read_exact(self._stream, size)
                if body is None:
                    self._inbox.put(("closed", None))
                    return
                self._inbox.put(("frame", json.loads(body)))
        except Exception:  # noqa: BLE001 -- a broken pipe/frame is a closed worker
            self._inbox.put(("closed", None))


class RenderSession:
    """Parent-side client for one per-invocation render worker.

    One session serves one doctor-section call and is used from one
    thread. Requests are single-flight; each carries the session's
    per-request deadline. The session owns the worker lifecycle:
    spawn + init handshake (:meth:`start`), respawn after a death,
    the consecutive-death circuit breaker, and reap-on-close.
    """

    def __init__(
        self,
        tokenizer_file: str,
        template: str,
        bos_token: str | None,
        *,
        budget_s: float = _DEFAULT_BUDGET_S,
        argv: tuple[str, ...] | None = None,
    ) -> None:
        """Init."""
        if not math.isfinite(budget_s) or budget_s <= 0:
            # The pre-W4 watchdog treated a non-positive/non-finite
            # budget as "disabled" — a documented escape hatch the
            # worker design deliberately removes (the deadline is
            # structural now). Refuse the value instead of letting it
            # reach a queue wait, where inf would block forever and
            # nan would behave arbitrarily.
            raise ValueError("render budget must be positive and finite")
        self._tokenizer_file = tokenizer_file
        self._template = template
        self._bos_token = bos_token
        self._budget_s = budget_s
        self._argv = argv if argv is not None else WORKER_ARGV
        self._proc: subprocess.Popen[bytes] | None = None
        self._inbox: queue.Queue[tuple[str, Any]] | None = None
        self._deaths = 0
        self._failed_code: str | None = None
        self._closed = False

    def __enter__(self) -> RenderSession:
        """Enter."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit."""
        self.close()

    def start(self) -> None:
        """Spawn the worker and run the init handshake.

        Raises :class:`WorkerUnavailable` with the section detail the
        caller should report: the pre-W4 codes for a missing
        transformers or an unloadable tokenizer, or
        ``render_worker_failed`` when the worker itself cannot run.
        """
        self._ensure_worker()

    def render(self, conversation: list[dict[str, str]]) -> str:
        """Render one conversation in the worker."""
        reply = self._request({"op": "render", "conversation": conversation})
        if reply.get("status") == "ok":
            text = reply.get("text")
            if type(text) is str:
                return text
        raise RenderFailed(str(reply.get("code", "render_failed")))

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode text with the checkpoint tokenizer, in the worker."""
        reply = self._request({"op": "encode", "text": text})
        if reply.get("status") == "ok":
            ids = reply.get("ids")
            if type(ids) is list and all(type(i) is int for i in ids):
                return tuple(ids)
        raise RenderFailed(str(reply.get("code", "encode_failed")))

    def close(self) -> None:
        """Kill and reap the worker if one is alive; idempotent."""
        self._closed = True
        self._kill()

    def _ensure_worker(self) -> None:
        """Ensure worker."""
        if self._closed:
            raise RenderWorkerError("render session is closed")
        if self._failed_code is not None:
            raise WorkerUnavailable(self._failed_code)
        if self._proc is None:
            try:
                self._spawn_and_init()
            except WorkerUnavailable as exc:
                # Latch: a worker that cannot initialize will not start
                # succeeding inside the same section call.
                self._failed_code = exc.code
                raise

    def _spawn_and_init(self) -> None:
        """Spawn and init."""
        try:
            # Fixed argv (this interpreter, this package's worker
            # module), never a shell, never checkpoint-derived input.
            proc: subprocess.Popen[bytes] = subprocess.Popen(  # noqa: S603  # nosec B603
                list(self._argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=os.name == "posix",
            )
        except OSError:
            raise WorkerUnavailable("render_worker_failed") from None
        if proc.stdout is None or proc.stdin is None:
            self._proc = proc
            self._kill()
            raise WorkerUnavailable("render_worker_failed")
        inbox: queue.Queue[tuple[str, Any]] = queue.Queue()
        _FrameReader(proc.stdout, inbox).start()
        self._proc = proc
        self._inbox = inbox
        try:
            _send_frame(
                proc.stdin,
                {
                    "op": "init",
                    "tokenizer_file": self._tokenizer_file,
                    "template": self._template,
                    "bos_token": self._bos_token,
                },
            )
            kind, value = inbox.get(timeout=_INIT_DEADLINE_S)
        except (OSError, ValueError, queue.Empty):
            self._kill()
            raise WorkerUnavailable("render_worker_failed") from None
        if kind != "frame" or type(value) is not dict:
            self._kill()
            raise WorkerUnavailable("render_worker_failed")
        status = value.get("status")
        if status == "init_error":
            code = value.get("code")
            self._kill()
            raise WorkerUnavailable(
                code
                if code in ("transformers_not_installed", "tokenizer_load_failed")
                else "tokenizer_load_failed"
            )
        if status != "ok":
            self._kill()
            raise WorkerUnavailable("render_worker_failed")
        # The worker must be running this parent's own environment:
        # same interpreter by construction, verified here for every
        # version the parent can read — a worker reporting a different
        # transformers/jinja2 version would make every render a
        # cross-version measurement. A version the parent cannot read
        # (broken metadata) is not checked: the doctor sections stamp
        # such versions "unavailable" and proceed, by design.
        versions = value.get("versions")
        expected: dict[str, str] = {}
        for package in ("transformers", "jinja2"):
            try:
                expected[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                continue
        if type(versions) is not dict or any(
            versions.get(name) != want for name, want in expected.items()
        ):
            self._kill()
            raise WorkerUnavailable("render_worker_failed")

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Request."""
        self._ensure_worker()
        proc = self._proc
        inbox = self._inbox
        if proc is None or inbox is None or proc.stdin is None:
            raise RenderWorkerError("render worker is not running")
        if proc.poll() is not None:
            # The worker died between requests. Never write into its
            # stdin: a host process that reset SIGPIPE to SIG_DFL (the
            # CLI does, for its own stdout pipeline convention) would
            # be killed by the write to the closed pipe instead of
            # seeing the BrokenPipeError caught below.
            raise self._death() from None
        try:
            inbox.get_nowait()
        except queue.Empty:
            pass
        else:
            # The reader already reported the channel closed, or the
            # worker sent an unsolicited frame — single-flight means
            # the inbox is empty at the start of every request, and a
            # worker that broke that is dead for protocol purposes.
            # Either way, do not write into the pipe (same SIGPIPE
            # reasoning as above).
            self._kill()
            raise self._death()
        frame = _encode_frame(payload)
        if len(frame) > _MAX_FRAME_BYTES:
            raise RenderFailed("request_frame_too_large")
        try:
            proc.stdin.write(frame)
            proc.stdin.flush()
        except (OSError, ValueError):
            raise self._death() from None
        try:
            kind, value = inbox.get(timeout=self._budget_s)
        except queue.Empty:
            self._kill()
            raise self._death() from None
        if kind != "frame" or type(value) is not dict:
            self._kill()
            raise self._death()
        # Any well-formed reply — including an error status — proves
        # the worker is alive; only transport failures count as deaths.
        self._deaths = 0
        return value

    def _death(self) -> RenderWorkerError:
        """Death."""
        self._proc = None
        self._inbox = None
        self._deaths += 1
        if self._deaths >= _MAX_CONSECUTIVE_DEATHS:
            self._failed_code = "render_worker_failed"
            return WorkerUnavailable("render_worker_failed")
        return WorkerDied("render worker died or broke the protocol")

    def _kill(self) -> None:
        """Kill."""
        proc, self._proc = self._proc, None
        self._inbox = None
        if proc is not None:
            with contextlib.suppress(Exception):
                proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=10)


def _render_once(tokenizer: Any, conversation: list[dict[str, str]]) -> str:
    """Render one conversation under the worker-local address-space cap.

    The cap formula is the pre-W4 one (current VmSize + 256 MiB,
    :func:`platform_caps.address_space_cap`); what changed is whose
    process it constrains. Where the cap cannot install (no
    ``resource``/``/proc``), the render runs unbounded *inside the
    worker only* — the parent's deadline and the frame discipline
    still bound the consequence, and a memory bomb can kill at most
    this worker.
    """
    with platform_caps.address_space_cap(_HEADROOM_BYTES):
        rendered = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        if type(rendered) is not str:
            raise ValueError("chat_template_did_not_render_text")
        return rendered


def _init_worker(frame: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Init worker."""
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError:
        return None, {"status": "init_error", "code": "transformers_not_installed"}
    try:
        # Untyped transformers boundary: bind through a typed Callable
        # so strict mode stays clean with or without transformers
        # installed (the same pattern doctor.py used pre-W4).
        factory: Callable[..., Any] = PreTrainedTokenizerFast
        tokenizer: Any = factory(
            tokenizer_file=str(frame["tokenizer_file"]),
            clean_up_tokenization_spaces=False,
        )
        tokenizer.chat_template = str(frame["template"])
        bos_token = frame.get("bos_token")
        if type(bos_token) is str and bos_token:
            # The template may not reference bos_token; a setter
            # failure must not fail the init.
            with contextlib.suppress(Exception):
                tokenizer.bos_token = bos_token
        versions: dict[str, str] = {"python": platform.python_version()}
        for package in ("transformers", "jinja2"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = "unavailable"
        return tokenizer, {
            "status": "ok",
            "versions": versions,
            "tokenizer_vocab_size": len(tokenizer),
        }
    except Exception:  # noqa: BLE001 -- any load failure maps to the fixed code
        return None, {"status": "init_error", "code": "tokenizer_load_failed"}


def _render_reply(tokenizer: Any, frame: dict[str, Any]) -> dict[str, Any]:
    """Render reply."""
    conversation = frame.get("conversation")
    if type(conversation) is not list:
        return {"status": "error", "code": "render_failed"}
    try:
        text = _render_once(tokenizer, conversation)
    except Exception:  # noqa: BLE001 -- fixed vocabulary only, never exception text
        return {"status": "error", "code": "render_failed"}
    if len(text) > MAX_RENDERED_CHARS:
        # The cap is enforced here so the payload never crosses the wire.
        return {"status": "oversize"}
    return {"status": "ok", "text": text}


def _encode_reply(tokenizer: Any, frame: dict[str, Any]) -> dict[str, Any]:
    """Encode reply."""
    text = frame.get("text")
    if type(text) is not str:
        return {"status": "error", "code": "encode_failed"}
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception:  # noqa: BLE001 -- fixed vocabulary only
        return {"status": "error", "code": "encode_failed"}
    return {"status": "ok", "ids": [int(i) for i in ids]}


def main() -> int:
    """Serve framed requests on stdio until the parent closes the pipe.

    The template and tokenizer never cross the wire after init; each
    render request carries only a conversation. Malformed inbound
    frames end the worker — the parent classifies the exit as a death
    and degrades accordingly.
    """
    tokenizer: Any = None
    ready = False
    while True:
        try:
            frame = _read_frame(sys.stdin.buffer)
        except Exception:  # noqa: BLE001 -- a malformed frame is a dead channel
            return 0
        if frame is None:
            return 0
        op = frame.get("op")
        try:
            if op == "init":
                tokenizer, reply = _init_worker(frame)
                ready = tokenizer is not None
                _send_frame(sys.stdout.buffer, reply)
            elif not ready:
                _send_frame(sys.stdout.buffer, {"status": "error", "code": "not_initialized"})
            elif op == "render":
                _send_frame(sys.stdout.buffer, _render_reply(tokenizer, frame))
            elif op == "encode":
                _send_frame(sys.stdout.buffer, _encode_reply(tokenizer, frame))
            else:
                _send_frame(sys.stdout.buffer, {"status": "error", "code": "unknown_op"})
        except (OSError, ValueError):
            # The parent is gone or the pipe broke; nothing else to do.
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
