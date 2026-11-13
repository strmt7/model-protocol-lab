"""Typed platform capabilities for the POSIX containment primitives.

Three call sites used to reach for POSIX-only attributes directly --
``doctor``'s render address-space cap and ``SIGALRM`` watchdog,
``mcp_server``'s sweep deadline, and ``fixturefuzz``'s ``os.mkfifo`` --
which made ``mypy --platform win32`` fail with 34 ``attr-defined``
errors and left the deadline logic duplicated with divergent teardown.
This module is the single home for those primitives, with the exact
semantics the call sites had:

* :func:`alarm_deadline` -- one-shot ``ITIMER_REAL`` deadline. Yields
  ``True`` iff the timer was armed; on any platform or thread where
  the alarm cannot be installed it yields ``False`` and touches no
  signal state, so the caller's body runs unbounded (the documented
  best-effort degradation). Teardown blocks ``SIGALRM`` while the
  timer is disarmed and the previous handler restored, and re-arms a
  pre-existing host timer with its full original delay.
* :func:`address_space_cap` -- transient ``RLIMIT_AS`` of the current
  virtual size plus a caller-chosen headroom, installed only when the
  ambient soft limit is infinite. Yields ``True`` iff installed.
* :func:`make_fifo` / :data:`FIFO_SUPPORTED` -- ``os.mkfifo`` with an
  honest unsupported answer instead of an ``AttributeError``.

Every function that names a POSIX-only attribute checks
``sys.platform`` first, in its own body: that is what lets mypy prove
the attribute exists on each platform it type-checks, with no
per-line type-ignores and no per-module override. Calls go through the
module objects (``signal.setitimer``, ``resource.setrlimit``) so the
test suite's global patch points keep intercepting them.
"""

from __future__ import annotations

import contextlib
import math
import os
import signal
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

FIFO_SUPPORTED: bool = sys.platform != "win32" and hasattr(os, "mkfifo")


def make_fifo(path: Path) -> bool:
    """Create a FIFO at ``path``; return ``False`` iff the platform cannot.

    ``False`` means no file was created. Errors from ``mkfifo`` itself
    (permissions, an existing file) propagate, as they did when call
    sites invoked ``os.mkfifo`` directly.
    """
    if sys.platform == "win32":
        return False
    if not hasattr(os, "mkfifo"):
        return False
    os.mkfifo(path)
    return True


def alarm_available() -> bool:
    """Whether a ``SIGALRM`` deadline can be armed from this thread.

    The full conjunction the MCP sweep always required: the platform
    is not Windows, ``SIGALRM``/``setitimer``/``getitimer`` all exist,
    and this is the main thread (signals are main-thread-only). Names
    no POSIX-only attribute, so it is safe to call anywhere.
    """
    return (
        sys.platform != "win32"
        and getattr(signal, "SIGALRM", None) is not None
        and hasattr(signal, "setitimer")
        and hasattr(signal, "getitimer")
        and threading.current_thread() is threading.main_thread()
    )


@contextmanager
def alarm_deadline(seconds: float, on_expire: Callable[[], None]) -> Iterator[bool]:
    """Arm a one-shot ``ITIMER_REAL`` deadline for the ``with`` body.

    Yields ``True`` iff armed. Yields ``False`` -- with no signal
    state read or written -- when ``seconds`` is non-positive or
    non-finite, or when :func:`alarm_available` is false; the body
    then runs unbounded, which is the callers' documented best-effort
    degradation. A failure installing the handler or arming the timer
    also yields ``False`` after restoring the previous handler.
    Saving the previous handler/timer is deliberately unguarded, as it
    was at both original call sites: if it raises, no state has been
    changed yet and the exception propagates.

    ``on_expire`` runs as the ``SIGALRM`` handler: doctor passes a
    raiser (budget exception), the MCP sweep passes an event setter
    (cooperative cancellation). A pre-existing host timer is re-armed
    afterwards with its full original delay, not its remaining time.
    """
    if sys.platform == "win32":
        yield False
        return
    if seconds <= 0 or not math.isfinite(seconds) or not alarm_available():
        yield False
        return

    def _handler(_signum: int, _frame: Any) -> None:
        on_expire()

    old_handler = signal.getsignal(signal.SIGALRM)
    old_delay, old_interval = signal.getitimer(signal.ITIMER_REAL)
    try:
        signal.signal(signal.SIGALRM, _handler)
    except (OSError, ValueError, RuntimeError):
        yield False
        return
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
    except (OSError, ValueError):
        with contextlib.suppress(OSError, ValueError, RuntimeError):
            signal.signal(signal.SIGALRM, old_handler)
        yield False
        return
    try:
        yield True
    finally:
        # Block SIGALRM across teardown: disarming the timer does not
        # retract an already-pending alarm, which would otherwise fire
        # mid-restore into a half-restored handler state. A pending
        # alarm is delivered after the mask is lifted, to the restored
        # previous handler.
        saved_mask: set[int | signal.Signals] | None = None
        try:
            saved_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
        except (AttributeError, OSError, ValueError, RuntimeError):
            saved_mask = None
        try:
            with contextlib.suppress(OSError, ValueError):
                signal.setitimer(signal.ITIMER_REAL, 0)
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                signal.signal(signal.SIGALRM, old_handler)
            if old_delay > 0:
                with contextlib.suppress(OSError, ValueError):
                    signal.setitimer(signal.ITIMER_REAL, old_delay, old_interval)
        finally:
            if saved_mask is not None:
                with contextlib.suppress(OSError, ValueError, RuntimeError):
                    signal.pthread_sigmask(signal.SIG_SETMASK, saved_mask)


@contextmanager
def address_space_cap(
    headroom_bytes: int,
    *,
    reraise_on_restore: tuple[type[BaseException], ...] = (),
) -> Iterator[bool]:
    """Install a transient ``RLIMIT_AS`` of VmSize + ``headroom_bytes``.

    Yields ``True`` iff the cap was installed. Every not-installed
    path yields ``False`` with no limit state touched: the platform
    has no ``resource`` module, ``/proc/self/status`` has no readable
    VmSize, the current limit cannot be read, the ambient soft limit
    is already finite (respected, never loosened), or installing the
    cap fails. The hard limit is carried through unchanged.

    On exit the previous soft/hard pair is restored. A restore
    failure of type ``OSError``/``ValueError`` is swallowed -- except
    a type named in ``reraise_on_restore``: that triggers one retry
    under suppression and then re-raises, because doctor's budget
    exception (a ``ValueError`` subclass) must reach its caller even
    when the watchdog alarm lands mid-restore, instead of being
    swallowed by the cleanup guard and leaving the cap lowered
    process-wide with nothing signalling it.
    """
    if sys.platform == "win32":
        yield False
        return
    try:
        import resource
    except ImportError:
        yield False
        return
    vmsize: int | None = None
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmSize:"):
                    vmsize = int(line.split()[1]) * 1024
                    break
    except OSError:
        vmsize = None
    if vmsize is None:
        yield False
        return
    try:
        old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
    except OSError:
        yield False
        return
    if old_soft != resource.RLIM_INFINITY:
        # A tighter ambient limit already governs; respect it, don't touch.
        yield False
        return
    new_soft = vmsize + headroom_bytes
    try:
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, old_hard))
    except (OSError, ValueError):
        yield False
        return
    try:
        yield True
    finally:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (old_soft, old_hard))
        except reraise_on_restore:
            # The watchdog alarm landed mid-restore. The one-shot alarm
            # is consumed, so retry the restore under suppression, then
            # re-raise: the budget expiry must reach the caller.
            with contextlib.suppress(OSError, ValueError):
                resource.setrlimit(resource.RLIMIT_AS, (old_soft, old_hard))
            raise
        except (OSError, ValueError):
            pass
