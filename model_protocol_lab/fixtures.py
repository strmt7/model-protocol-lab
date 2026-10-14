"""Author evaluation fixtures from a compact CSV source.

The lab's runtime schema is strict JSON with a search/held-out discipline
(ID uniqueness, group non-overlap across partitions, no identical message
sequences across partitions, NFC-normalized comparisons). Hand-writing that
JSON is verbose and the discipline is easy to violate; this module offers a
compact authoring surface -- one CSV row per fixture -- that enforces the
discipline at *build* time, before any file is written:

- the ``partition`` column assigns every fixture explicitly, so leakage is
  caught while authoring (with row numbers) instead of at ``inspect`` time;
- the emitted JSON is round-tripped through the strict
  :func:`datasets.parse_dataset` loader before it is returned, so a built
  file satisfies the runtime schema by construction -- the builder refuses
  to produce a violating file.

CSV layout (header row required)::

    id,group,partition,expected,user_1,assistant_1,user_2,assistant_2,...

``partition`` is exactly ``search`` or ``held_out``. Message columns must
form ``user_1, assistant_1, user_2, ...`` pairs in order, starting with
``user_1``; the conversation ends at the first empty message cell, and an
empty cell followed by a non-empty later cell is an error (so empty-string
message content is not expressible here -- use JSON for that). Cells are
used verbatim: no whitespace stripping, no formula evaluation. Quoted cells
may contain newlines. Limits mirror the runtime schema (2 MB source, 100
fixtures per partition, 21 messages, 16,384 characters per cell).
"""

import contextlib
import csv
import io
import json
import os
import re
import stat
from pathlib import Path
from typing import Literal

from .core import Fixture, Message, Split
from .datasets import MAX_BYTES, MAX_CASES_PER_SPLIT, MAX_MESSAGES, MAX_TEXT, parse_dataset

_FIXED_COLUMNS = ("id", "group", "partition", "expected")
_PARTITIONS = ("search", "held_out")

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _safe(value: str) -> str:
    """Strip control characters so hostile authoring input cannot inject
    terminal escape sequences into error messages."""
    return _CONTROL_CHARS.sub("", value)


def read_authoring_file(path: str | Path) -> str:
    """Read a bounded UTF-8 text file, refusing non-regular files.

    Mirrors the defensive open in :func:`datasets.load_dataset` (O_NONBLOCK
    plus an fstat on the opened fd) so a named FIFO with no writer cannot
    hang the builder and character devices or directories are refused.
    """
    if not isinstance(path, (str, Path)):
        raise ValueError("Invalid authoring path")
    target = Path(path)
    try:
        fd = os.open(target, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        raise ValueError("Unreadable authoring file") from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Authoring path is not a regular file")
        with os.fdopen(fd, "rb") as handle:
            fd = -1  # ownership moved to the file object
            content = handle.read(MAX_BYTES + 1)
    finally:
        if fd != -1:
            os.close(fd)
    if len(content) > MAX_BYTES:
        raise ValueError("Authoring file exceeds size limit")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("Authoring file is not valid UTF-8") from None


def _expected_message_headers(count: int) -> tuple[str, ...]:
    headers: list[str] = []
    index = 1
    while len(headers) < count:
        headers.append(f"user_{index}")
        if len(headers) < count:
            headers.append(f"assistant_{index}")
            index += 1
    return tuple(headers)


def parse_fixture_csv(text: str) -> tuple[Split, list[str]]:
    """Parse authoring CSV into a validated :class:`Split`.

    Returns the split plus the IDs of fixtures with an empty ``expected``
    answer (which pass vacuously, exactly as in the JSON schema -- the
    caller warns, mirroring ``inspect``). Raises ValueError with a row
    number on any authoring error. The returned split has already passed
    the same separation checks ``inspect`` applies (ID uniqueness, group
    non-overlap, identical-message leakage, NFC-normalized).
    """
    if type(text) is not str:
        raise ValueError("CSV source must be text")
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise ValueError("CSV source exceeds size limit")
    reader = csv.reader(io.StringIO(text))
    header: list[str] | None = None
    rows: list[tuple[int, list[str]]] = []
    for record in reader:
        if header is None:
            if not record or all(not cell for cell in record):
                continue  # leading blank lines are not a header
            header = record
            continue
        if not record or all(not cell for cell in record):
            continue  # blank lines between rows carry no fixture
        rows.append((reader.line_num, record))
    if header is None:
        raise ValueError("CSV source has no header row")
    if tuple(header[:4]) != _FIXED_COLUMNS:
        raise ValueError(
            "CSV header must start with 'id,group,partition,expected', "
            f"got {','.join(_safe(cell) for cell in header[:4])!r}"
        )
    message_headers = tuple(header[4:])
    if not 1 <= len(message_headers) <= MAX_MESSAGES:
        raise ValueError(
            f"CSV needs 1-{MAX_MESSAGES} message columns after 'expected', "
            f"got {len(message_headers)}"
        )
    if message_headers != _expected_message_headers(len(message_headers)):
        raise ValueError(
            "Message columns must be 'user_1,assistant_1,user_2,...' in order, "
            f"got {','.join(_safe(cell) for cell in message_headers)!r}"
        )
    if len(rows) > 2 * MAX_CASES_PER_SPLIT:
        raise ValueError(f"CSV holds more than {2 * MAX_CASES_PER_SPLIT} fixtures")
    search: list[Fixture] = []
    held_out: list[Fixture] = []
    empty_expected: list[str] = []
    for line_num, record in rows:
        label = f"row {line_num}"
        cells = list(record) + [""] * (len(header) - len(record))
        if len(cells) > len(header):
            raise ValueError(f"{label}: {len(record)} cells, header has {len(header)}")
        fid, group, partition, expected = cells[0], cells[1], cells[2], cells[3]
        if not fid.strip():
            raise ValueError(f"{label}: fixture id is blank")
        if not group.strip():
            raise ValueError(f"{label} (id {_safe(fid)!r}): fixture group is blank")
        if partition not in _PARTITIONS:
            raise ValueError(
                f"{label} (id {_safe(fid)!r}): partition must be 'search' or "
                f"'held_out', got {_safe(partition)!r}"
            )
        texts = cells[4:]
        if any(len(cell) > MAX_TEXT for cell in texts) or len(expected) > MAX_TEXT:
            raise ValueError(f"{label} (id {_safe(fid)!r}): text field exceeds size limit")
        if not texts[0]:
            raise ValueError(f"{label} (id {_safe(fid)!r}): fixture needs a user_1 message")
        messages: list[Message] = []
        ended = False
        for position, cell in enumerate(texts):
            if not cell:
                ended = True
                continue
            if ended:
                raise ValueError(
                    f"{label} (id {_safe(fid)!r}): empty message followed by content "
                    f"at {message_headers[position]!r}"
                )
            role: Literal["user", "assistant"] = "user" if position % 2 == 0 else "assistant"
            messages.append(Message(role, cell))
        if messages[-1].role != "user":
            raise ValueError(
                f"{label} (id {_safe(fid)!r}): conversation must end with a user message"
            )
        fixture = Fixture(id=fid, messages=tuple(messages), expected=expected, group=group)
        if not expected:
            empty_expected.append(fid)
        (search if partition == "search" else held_out).append(fixture)
    for name, cases in (("search", search), ("held_out", held_out)):
        if not cases:
            raise ValueError(f"CSV has no '{name}' fixtures")
        if len(cases) > MAX_CASES_PER_SPLIT:
            raise ValueError(f"CSV has more than {MAX_CASES_PER_SPLIT} '{name}' fixtures")
    # The Split constructor enforces the search/held-out discipline (global
    # ID uniqueness, group non-overlap, identical-message leakage, all
    # NFC-normalized); a violation raises here, before any file is written.
    try:
        split = Split(tuple(search), tuple(held_out))
    except ValueError as exc:
        raise ValueError(f"CSV fixtures violate the split discipline: {exc}") from None
    return split, empty_expected


def split_to_json_text(split: Split) -> str:
    """Serialize a split to the strict fixture JSON, verified by construction.

    The text is round-tripped through :func:`datasets.parse_dataset` before
    it is returned, so only a file the runtime loader accepts can come out
    of the builder.
    """
    if type(split) is not Split:
        raise ValueError("split_to_json_text needs a Split")

    def fixture_json(case: Fixture) -> dict:
        return {
            "id": case.id,
            "group": case.group,
            "messages": [
                {"role": message.role, "content": message.content} for message in case.messages
            ],
            "expected": case.expected,
        }

    payload = {
        "search": [fixture_json(case) for case in split.search],
        "held_out": [fixture_json(case) for case in split.held_out],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    # By-construction guarantee: the runtime loader must accept what we emit.
    parse_dataset(text)
    return text


def write_fixture_file(text: str, path: str | Path) -> Path:
    """Write built fixture JSON to a new file with owner-only permissions.

    Refuses to overwrite an existing path (exclusive creation); the parent
    directory must already exist. Reports are written 0o600 for the same
    reason fixture prompts can be sensitive.
    """
    if type(text) is not str:
        raise ValueError("Fixture text must be a string")
    if not isinstance(path, (str, Path)):
        raise ValueError("Invalid fixture output path")
    target = Path(path)
    if target.parent != Path(".") and not target.parent.is_dir():
        raise ValueError("Fixture output parent directory does not exist")
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ValueError("Fixture output already exists; use a fresh path") from None
    except OSError:
        raise ValueError("Fixture output is not writable") from None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
    except BaseException:
        # A partial write must not leave a half-built fixture file behind.
        with contextlib.suppress(OSError):
            os.unlink(target)
        raise
    return target
