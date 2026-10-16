"""Bounded JSON fixture loading without executable configuration."""

import json
import os
import stat
from pathlib import Path
from typing import Any, NoReturn

from .core import Fixture, Message, Split

MAX_BYTES = 2_000_000
MAX_CASES_PER_SPLIT = 100
MAX_MESSAGES = 21
MAX_TEXT = 16_384


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _reject_constant(_value: str) -> NoReturn:
    raise ValueError("Nonfinite JSON")


def _fields(value: Any, expected: tuple[str, ...]) -> None:
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("Unexpected or missing JSON fields")


def _text(value: Any, *, nonblank: bool = False) -> str:
    if type(value) is not str or len(value) > MAX_TEXT:
        raise ValueError("Invalid or oversized text field")
    if nonblank and not value.strip():
        raise ValueError("Blank identifier")
    return value


def _fixture(value: Any, label: str) -> Fixture:
    _fields(value, ("id", "group", "messages", "expected"))
    messages = value["messages"]
    if type(messages) is not list or not 1 <= len(messages) <= MAX_MESSAGES:
        raise ValueError(f"{label}: conversation must have 1-{MAX_MESSAGES} messages")
    parsed = []
    for index, item in enumerate(messages):
        _fields(item, ("role", "content"))
        role = item["role"]
        if role != ("user" if index % 2 == 0 else "assistant"):
            raise ValueError(f"{label}: conversation must alternate user/assistant")
        parsed.append(Message(role, _text(item["content"])))
    if parsed[-1].role != "user":
        raise ValueError(f"{label}: conversation must end with user")
    return Fixture(
        _text(value["id"], nonblank=True),
        tuple(parsed),
        _text(value["expected"]),
        _text(value["group"], nonblank=True),
    )


def parse_dataset(text: str) -> Split:
    """Parse dataset JSON text into a validated Split.

    Enforces the byte cap, rejects duplicate object keys and non-JSON
    constants, requires the "search"/"held_out" partitions with bounded
    fixture counts, and lets Split enforce the anti-leakage guarantees.
    Raises ValueError with the module's fixed error vocabulary on any
    malformed input.
    """

    if type(text) is not str or len(text.encode("utf-8")) > MAX_BYTES:
        raise ValueError("Dataset exceeds size limit")
    try:
        raw = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except RecursionError:
        # Deeply nested input within the byte cap exhausts the parser stack;
        # report it in the module's error vocabulary, not as a bare RuntimeError.
        raise ValueError("Dataset nesting too deep") from None
    _fields(raw, ("search", "held_out"))
    partitions = []
    for name in ("search", "held_out"):
        values = raw[name]
        if type(values) is not list or not 1 <= len(values) <= MAX_CASES_PER_SPLIT:
            raise ValueError("Invalid fixture count")
        partitions.append(
            tuple(_fixture(value, f"{name}[{index}]") for index, value in enumerate(values))
        )
    return Split(*partitions)


def load_dataset(path: str | Path) -> Split:
    """Load a dataset file from disk into a validated Split.

    Opens the path defensively: only regular files are accepted (never FIFOs,
    devices, sockets, or directories), reads are size-capped, and the file is
    validated through parse_dataset. Raises ValueError on any problem.
    """

    if not isinstance(path, (str, Path)):
        raise ValueError("Invalid fixture path")
    target = Path(path)
    # O_NONBLOCK so that opening a named FIFO with no writer cannot block
    # forever; the fstat below then refuses anything that is not a regular
    # file (character/block devices, FIFOs, sockets, directories). Checking
    # the opened fd (not the path) also closes the stat->open TOCTOU window.
    try:
        fd = os.open(target, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        raise ValueError("Unreadable fixture file") from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            # Refuse TTYs, block devices, FIFOs, sockets, directories:
            # opening them can block forever or produce unbounded input.
            raise ValueError("Fixture path is not a regular file")
        with os.fdopen(fd, "rb") as handle:
            fd = -1  # ownership moved to the file object
            try:
                content = handle.read(MAX_BYTES + 1)
            except OSError:
                # The open/fstat guards cannot catch every unreadable file:
                # e.g. /proc files pass S_ISREG yet fail on read (EIO).
                # Normalize so callers see the fixed error, not a traceback.
                raise ValueError("Unreadable fixture file") from None
    finally:
        if fd != -1:
            os.close(fd)
    if len(content) > MAX_BYTES:
        raise ValueError("Dataset exceeds size limit")
    try:
        return parse_dataset(content.decode("utf-8"))
    except UnicodeDecodeError:
        raise ValueError("Fixture file is not valid UTF-8") from None
