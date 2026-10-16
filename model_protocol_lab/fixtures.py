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

``fixtures build`` also runs a special-token literal audit over the built
split (see :func:`audit_special_token_literals`) and records the findings
in a ``<output>.audit.json`` sidecar next to the fixture file. The findings
cannot live inside the fixture JSON itself: the strict runtime schema
(:func:`datasets.parse_dataset`) rejects any top-level key beyond
``search``/``held_out``, so an in-file ``_audit`` key would make the built
file unloadable. Findings are warnings only and never block the build.
"""

import contextlib
import csv
import io
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .core import Fixture, Format, Message, Split
from .datasets import MAX_BYTES, MAX_CASES_PER_SPLIT, MAX_MESSAGES, MAX_TEXT, parse_dataset

_FIXED_COLUMNS = ("id", "group", "partition", "expected")
_PARTITIONS = ("search", "held_out")

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f\ud800-\udfff]")
# Lone surrogates are stripped too: they crash strict stream writes with
# UnicodeEncodeError and are never valid text on their own.


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
            try:
                content = handle.read(MAX_BYTES + 1)
            except OSError:
                # The open/fstat guards cannot catch every unreadable file:
                # e.g. /proc files pass S_ISREG yet fail on read (EIO).
                # Normalize so callers see the fixed error, not a traceback.
                raise ValueError("Unreadable authoring file") from None
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
    caller warns, mirroring ``inspect``). Raises ValueError naming the
    offending row for per-row authoring errors; header, size, column-count,
    partition-size, and split-discipline errors name the problem without a
    row number. The returned split has already passed
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
    try:
        for record in reader:
            if header is None:
                if not record or all(not cell for cell in record):
                    continue  # leading blank lines are not a header
                header = record
                continue
            if not record or all(not cell for cell in record):
                continue  # blank lines between rows carry no fixture
            rows.append((reader.line_num, record))
    except csv.Error:
        # A single field bigger than the csv module's field limit (131072
        # chars) aborts the parser before the per-cell MAX_TEXT check below
        # can run. Normalize to ValueError so callers (which catch only
        # ValueError) report a fixed error instead of a traceback.
        raise ValueError(f"CSV field exceeds the parser limit near row {reader.line_num}") from None
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

    def fixture_json(case: Fixture) -> dict[str, Any]:
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
    # A lone surrogate is legal in a Python str but cannot be encoded to the
    # UTF-8 bytes a fixture file requires. Refuse deliberately with the fixed
    # ValueError vocabulary instead of letting the encode (or the later file
    # write) die with a bare UnicodeEncodeError.
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "split_to_json_text: fixture text contains a lone surrogate "
            "(not valid Unicode -- cannot be written to a UTF-8 fixture file)"
        ) from exc
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


# ---------------------------------------------------------------------------
# Special-token literal audit (fixture hygiene).
#
# Advisory GHSA-g5f9-3xfg-p9mf (Decepticon v1.1.17): role-boundary forgery via
# chat-template special-token literals embedded in untrusted input that is
# later composed into an LLM context. For the lab this is a fixture-hygiene
# property: a fixture whose message text contains a literal such as
# ``<|im_start|>`` would poison the experiment, because the measured
# "boundary" would come from the fixture text rather than the model.
#
# The audit scans fixture message contents for exact, case-sensitive
# substring matches against the special-token inventory of the candidate
# formats under test. The inventory is *derived from the formats
# themselves* (their bos/user_prefix/assistant_prefix/turn_end framing
# fields, plus the optional per-role turn terminators and open suffix when
# set), never from a hardcoded token list, so the audit stays in sync with
# whatever catalog the operator audits against. The extractor is
# deliberately conservative: it recognizes only four fixed token shapes --
# ``<|...|>``, ``<...>``, ``[...]`` (letter- or slash-leading, e.g.
# ``[INST]``, ``[/INST]``, ``[s]``), and the full-width DeepSeek-style
# ``＜｜...｜＞`` -- matched as literal substrings, case-sensitive, with no
# regex heuristics beyond those four shapes. Anything not token-shaped in
# the format fields is not audited (plain words like ``user`` or trailing
# newlines are ordinary text, not special tokens).


@dataclass(frozen=True)
class AuditFinding:
    """One special-token literal found in audited text.

    ``fixture_id`` identifies the fixture ("" for raw-string input);
    ``field`` locates the text: ``"<role>_<n>"`` per message for fixtures
    (``user_1``, ``assistant_1``, ``user_2``, ...), ``"text"`` for a single
    raw string, ``"text_<n>"`` for each item of a raw-string tuple.
    ``literal`` is the exact matched substring, ``format_name`` the
    candidate format whose inventory contains it, and ``offset`` the
    character offset of the match within the field text.
    """

    fixture_id: str
    field: str
    literal: str
    format_name: str
    offset: int

    def __post_init__(self) -> None:
        if (
            type(self.fixture_id) is not str
            or type(self.field) is not str
            or type(self.literal) is not str
            or type(self.format_name) is not str
        ):
            raise ValueError("AuditFinding text fields must be strings")
        if not self.field.strip() or not self.literal or not self.format_name.strip():
            raise ValueError("AuditFinding field, literal and format_name must be nonblank")
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError("AuditFinding offset must be a nonnegative int")


# Fixed token shapes, most specific first so ``<|...|>`` is extracted as one
# token rather than also matching the looser ``<...>`` shape (the inventory
# dedupes by string anyway; the order only keeps first-seen order stable).
_TOKEN_SHAPES = (
    re.compile(r"＜｜[^＜＞\n]+｜＞"),  # full-width DeepSeek-style ＜｜...｜＞
    re.compile(r"<\|[^<>\n]+\|>"),  # <|im_start|> style
    re.compile(r"\[[A-Za-z/][^\[\]\n]*\]"),  # [INST], [/INST], [s]
    re.compile(r"<[^<>\n]+>"),  # <bos>, <s>, <start_of_turn>
)


def _extract_special_tokens(fmt: Format) -> tuple[str, ...]:
    """Token-shaped substrings of a format's framing fields, deduplicated.

    Scans bos, user_prefix, assistant_prefix, turn_end, then the optional
    per-role turn terminators and open suffix (None fields are skipped --
    the base turn_end/assistant_prefix they fall back to was already
    scanned, so only non-None overrides can add new text). First-seen
    order is the inventory order the audit reports in.
    """
    if type(fmt) is not Format:
        raise ValueError("_extract_special_tokens needs a Format")
    seen: list[str] = []
    fields = (
        fmt.bos,
        fmt.user_prefix,
        fmt.assistant_prefix,
        fmt.turn_end,
        fmt.user_turn_end,
        fmt.assistant_turn_end,
        fmt.open_suffix,
    )
    for text in fields:
        if text is None:
            continue
        for shape in _TOKEN_SHAPES:
            for match in shape.finditer(text):
                token = match.group(0)
                if token not in seen:
                    seen.append(token)
    return tuple(seen)


def _audit_targets(
    fixture_or_texts: Fixture | Split | tuple[Fixture | str, ...] | str,
) -> tuple[tuple[str, str, str], ...]:
    """Normalize audit input to (fixture_id, field, text) triples."""
    targets: list[tuple[str, str, str]] = []

    def add_fixture(case: Fixture) -> None:
        for index, message in enumerate(case.messages):
            targets.append((case.id, f"{message.role}_{index // 2 + 1}", message.content))

    if type(fixture_or_texts) is Fixture:
        add_fixture(fixture_or_texts)
    elif type(fixture_or_texts) is Split:
        for case in (*fixture_or_texts.search, *fixture_or_texts.held_out):
            add_fixture(case)
    elif type(fixture_or_texts) is str:
        targets.append(("", "text", fixture_or_texts))
    elif type(fixture_or_texts) is tuple:
        for position, item in enumerate(fixture_or_texts):
            if type(item) is Fixture:
                add_fixture(item)
            elif type(item) is str:
                targets.append(("", f"text_{position + 1}", item))
            else:
                raise ValueError("audit input tuple must hold Fixture or str items")
    else:
        raise ValueError("audit input must be a Fixture, Split, str, or tuple of Fixture/str")
    return tuple(targets)


def audit_special_token_literals(
    fixture_or_texts: Fixture | Split | tuple[Fixture | str, ...] | str,
    formats: tuple[Format, ...],
) -> tuple[AuditFinding, ...]:
    """Scan fixture message text for special-token literals.

    Advisory GHSA-g5f9-3xfg-p9mf: a special-token literal (e.g.
    ``<|im_start|>``, ``＜｜User｜＞``, ``[INST]``) embedded in untrusted
    fixture text can forge a role boundary when the fixture is composed
    into a prompt, poisoning the experiment. This audit flags exact,
    case-sensitive substring collisions between message contents and the
    special-token inventory derived from each given format's framing
    fields (bos, user_prefix, assistant_prefix, turn_end, plus non-None
    per-role turn terminators and open suffix -- see
    :func:`_extract_special_tokens`).

    ``fixture_or_texts`` is a Fixture, a Split (both partitions scanned),
    a raw string, or a tuple mixing Fixtures and raw strings. ``formats``
    is a nonempty tuple of Format. Returns findings in deterministic
    order: fixtures and messages in input order, then formats in the given
    order, then each format's inventory order, then ascending offsets
    (overlapping occurrences of one literal each get a finding).
    """
    if type(formats) is not tuple or not formats or any(type(fmt) is not Format for fmt in formats):
        raise ValueError("audit needs a nonempty tuple of Format")
    inventories = tuple((fmt.name, _extract_special_tokens(fmt)) for fmt in formats)
    findings: list[AuditFinding] = []
    for fixture_id, field, text in _audit_targets(fixture_or_texts):
        for format_name, tokens in inventories:
            for token in tokens:
                start = 0
                while True:
                    offset = text.find(token, start)
                    if offset < 0:
                        break
                    findings.append(
                        AuditFinding(
                            fixture_id=fixture_id,
                            field=field,
                            literal=token,
                            format_name=format_name,
                            offset=offset,
                        )
                    )
                    start = offset + 1
    return tuple(findings)


_AUDIT_SCHEMA = "model-protocol-lab.audit/v1"


def audit_findings_json(findings: tuple[AuditFinding, ...], *, source: str) -> str:
    """Serialize audit findings to the sidecar document for a built fixture.

    The strict fixture JSON schema (:func:`datasets.parse_dataset`) rejects
    any top-level key beyond ``search``/``held_out``, so findings cannot be
    embedded in the fixture file without breaking the loader; ``fixtures
    build`` writes this document to ``<output>.audit.json`` instead.
    ``source`` names the fixture file the findings belong to.
    """
    if type(findings) is not tuple or any(type(item) is not AuditFinding for item in findings):
        raise ValueError("audit_findings_json needs a tuple of AuditFinding")
    if type(source) is not str:
        raise ValueError("audit sidecar source must be a string")
    payload = {
        "schema": _AUDIT_SCHEMA,
        "source": source,
        "findings": [
            {
                "fixture_id": finding.fixture_id,
                "field": finding.field,
                "literal": finding.literal,
                "format_name": finding.format_name,
                "offset": finding.offset,
            }
            for finding in findings
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
