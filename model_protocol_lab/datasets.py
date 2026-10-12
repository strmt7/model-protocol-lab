"""Bounded JSON fixture loading without executable configuration."""
import json
from pathlib import Path
from .core import Fixture, Message, Split

MAX_BYTES = 2_000_000
MAX_CASES_PER_SPLIT = 100
MAX_MESSAGES = 21
MAX_TEXT = 16_384


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("Nonfinite JSON")


def _fields(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("Unexpected or missing JSON fields")


def _text(value, *, nonblank=False):
    if type(value) is not str or len(value) > MAX_TEXT:
        raise ValueError("Invalid or oversized text field")
    if nonblank and not value.strip():
        raise ValueError("Blank identifier")
    return value


def _fixture(value):
    _fields(value, ("id", "group", "messages", "expected"))
    messages = value["messages"]
    if type(messages) is not list or not 1 <= len(messages) <= MAX_MESSAGES:
        raise ValueError("Invalid conversation length")
    parsed = []
    for index, item in enumerate(messages):
        _fields(item, ("role", "content"))
        role = item["role"]
        if role != ("user" if index % 2 == 0 else "assistant"):
            raise ValueError("Conversation must alternate user/assistant")
        parsed.append(Message(role, _text(item["content"])))
    if parsed[-1].role != "user":
        raise ValueError("Conversation must end with user")
    return Fixture(_text(value["id"], nonblank=True), tuple(parsed),
                   _text(value["expected"]), _text(value["group"], nonblank=True))


def parse_dataset(text: str) -> Split:
    if type(text) is not str or len(text.encode("utf-8")) > MAX_BYTES:
        raise ValueError("Dataset exceeds size limit")
    raw = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    _fields(raw, ("search", "held_out"))
    partitions = []
    for name in ("search", "held_out"):
        values = raw[name]
        if type(values) is not list or not 1 <= len(values) <= MAX_CASES_PER_SPLIT:
            raise ValueError("Invalid fixture count")
        partitions.append(tuple(_fixture(value) for value in values))
    return Split(*partitions)


def load_dataset(path: str | Path) -> Split:
    with Path(path).open("rb") as handle:
        content = handle.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise ValueError("Dataset exceeds size limit")
    return parse_dataset(content.decode("utf-8"))
