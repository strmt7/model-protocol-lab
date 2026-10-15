"""Interactive protocol debugger REPL (``mpl debug``).

An offline, stdlib-only (:mod:`cmd`-based) shell for poking at format
hypotheses the way the experiments do: render a catalog hypothesis over a
fixture's messages, inspect the per-segment char spans (and exact token IDs
when a tokenizer is attached), step through response-boundary detection one
token at a time, and audit fixtures for special-token literals -- all in one
terminal session, scriptable via ``.read`` files.

Conventions are stolen from the sqlite3 CLI: dot-command meta namespace,
unambiguous abbreviations, multiline entry with a continuation prompt,
``.help PATTERN`` search, ``#`` comments, ``.read FILE`` scripted sessions,
and a startup banner showing the active context.

The optional ``--model`` backend is tokenizer-only: it reuses doctor.py's
:func:`~model_protocol_lab.doctor._build_primary_tokenizer` path, which
builds a ``PreTrainedTokenizerFast`` from the checkpoint's ``tokenizer.json``
(with the same 128 MB pre-load size cap). Model weights are never loaded,
no model code is executed, and nothing touches the network.
"""

from __future__ import annotations

import argparse
import cmd
import io
import re
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Literal

from . import __version__
from .catalog import CATALOG
from .core import Fixture, Format, Message, Split, first_boundary
from .datasets import load_dataset
from .fixtures import audit_special_token_literals

# Bounds: a scripted session is operator input, but .read must stay a cheap,
# deterministic operation -- never a vector for unbounded memory growth.
_MAX_SCRIPT_BYTES = 1_000_000
_MAX_SCRIPT_LINES = 10_000
_MAX_READ_DEPTH = 5
# Token-stream bounds mirror cli.parse_boundary's 16-ID cap for boundaries;
# streams are longer but still capped so a pasted blob cannot blow up memory.
_MAX_BOUNDARY_IDS = 16
_MAX_STREAM_TOKENS = 4096
_MAX_SAVE_BYTES = 5_000_000
# Display cap for printed ID lists; counts are always exact.
_TOKEN_DISPLAY_CAP = 512

# Optional "role:" prefix on .say lines, e.g. "user: hello".
_ROLE_RE = re.compile(r"^([A-Za-z]+)\s*:\s*(.*)$", re.DOTALL)

# Control characters are stripped from every operator-visible string derived
# from fixture data, user input, or audit findings before printing, so a
# hostile fixture file cannot inject terminal escape sequences (notably ESC,
# which introduces ANSI sequences) into the operator's terminal. Tab, LF and
# CR are kept so multi-line renders stay readable. Mirrors cli._display_id's
# rationale (which strips all controls from short IDs); kept local so
# debug.py stays importable without cli.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _sanitize_display(value: str) -> str:
    """Strip terminal-control characters from an operator-visible string."""
    return _CONTROL_CHARS.sub("", value)


def _parse_id_list(value: str, *, what: str, max_len: int) -> tuple[int, ...]:
    """Strict comma-separated ASCII token-ID list (cli.parse_boundary style)."""
    parts = [part.strip() for part in value.split(",")]
    if not 1 <= len(parts) <= max_len or any(
        not part or any(not ("0" <= ch <= "9") for ch in part) for part in parts
    ):
        raise ValueError(f"{what}: expected 1-{max_len} comma-separated ASCII token IDs")
    ids = tuple(int(part) for part in parts)
    if any(token_id > 2**31 - 1 for token_id in ids):
        raise ValueError(f"{what}: token IDs must be < 2^31")
    return ids


def _overhang(seen: tuple[int, ...], boundary: tuple[int, ...]) -> int:
    """Longest k such that the seen stream ends with ``boundary[:k]``.

    This is the running partial-match state the stepper shows: how much of
    the boundary suffix is currently "in flight" without a full match yet.
    """
    for k in range(min(len(seen), len(boundary)), -1, -1):
        if seen[len(seen) - k :] == boundary[:k]:
            return k
    return 0


def _render_segments(fmt: Format, messages: tuple[Message, ...]) -> tuple[tuple[str, str], ...]:
    """Name every char span of ``Format.render``; asserts exact reconstruction.

    Mirrors ``Format.render``'s assembly order (bos, per-message
    prefix/content/turn-end, optional repeated bos, open-reply suffix) so the
    map the REPL prints is the render the experiments measure -- not a
    second implementation that could drift. The reconstruction assert turns
    any future drift into a loud failure instead of a wrong map.
    """
    rendered = fmt.render(messages)
    segments: list[tuple[str, str]] = [("bos", fmt.bos)]
    for index, message in enumerate(messages):
        if index and fmt.repeat_bos:
            segments.append(("bos", fmt.bos))
        if message.role == "user":
            prefix, per_role_end = fmt.user_prefix, fmt.user_turn_end
        else:
            prefix, per_role_end = fmt.assistant_prefix, fmt.assistant_turn_end
        turn_end = fmt.turn_end if per_role_end is None else per_role_end
        segments.append((f"{message.role}_prefix", prefix))
        segments.append((f"{message.role}_content", message.content))
        segments.append((f"{message.role}_turn_end", turn_end))
    if fmt.repeat_bos:
        segments.append(("bos", fmt.bos))
    suffix = fmt.open_suffix if fmt.open_suffix is not None else fmt.assistant_prefix
    segments.append(("open_suffix", suffix))
    if "".join(text for _, text in segments) != rendered:
        raise AssertionError("segment map does not reconstruct the render")
    return tuple(segments)


def _parse_say_line(line: str) -> tuple[Literal["user", "assistant"] | None, str]:
    """Split an optional ``role:`` prefix off a .say line.

    Returns ``(role, content)`` with role None when the line carries no
    recognized prefix; unrecognized ``word:`` prefixes are kept as content
    rather than rejected, so prose like ``"note: ..."`` is never mangled.
    """
    match = _ROLE_RE.match(line.strip())
    if match and match.group(1).lower() in ("user", "assistant"):
        role: Literal["user", "assistant"] = (
            "user" if match.group(1).lower() == "user" else "assistant"
        )
        return role, match.group(2)
    return None, line


def attach_tokenizer(model_path: str | Path) -> Any:
    """Load a checkpoint's tokenizer for exact token IDs; never model weights.

    Reuses doctor.py's tokenizer-only load path
    (``PreTrainedTokenizerFast`` from ``tokenizer.json`` with the 128 MB
    pre-load size cap). Raises ImportError when transformers is unavailable,
    ValueError for a bad checkpoint directory.
    """
    from .backends import _directory
    from .doctor import _build_primary_tokenizer

    return _build_primary_tokenizer(_directory(model_path))


class DebugShell(cmd.Cmd):
    """sqlite3-flavored REPL over one dataset and one format hypothesis.

    Drive programmatically by feeding ``cmdqueue`` and calling ``cmdloop()``
    (or single-step with ``onecmd``); every command's output goes to
    ``self.stdout`` and is also appended to the in-memory transcript used by
    ``.save``. All output is deterministic: no timestamps, no randomness.
    """

    def __init__(
        self,
        split: Split,
        *,
        dataset_label: str = "<dataset>",
        family: str = "chatml",
        tokenizer: Any = None,
        stdin: Any = None,
        stdout: Any = None,
    ) -> None:
        if type(split) is not Split:
            raise ValueError("DebugShell needs a Split")
        if family not in CATALOG:
            raise ValueError(f"unknown family {family!r}")
        super().__init__(stdin=stdin, stdout=stdout)
        # Never block on real stdin: scripted/queued sessions must terminate
        # on EOF instead of waiting at an interactive prompt.
        self.use_rawinput = False
        self._cases: tuple[Fixture, ...] = (*split.search, *split.held_out)
        self._family = family
        self._case_index = 0
        self._tokenizer = tokenizer
        self._boundary: tuple[int, ...] | None = None
        self._stream: tuple[int, ...] | None = None
        self._seen: list[int] = []
        self._say_buffer: list[str] | None = None
        self._buffered: tuple[Message, ...] | None = None
        self._transcript: list[str] = []
        self._transcript_truncated = False
        self._read_depth = 0
        entries: tuple[tuple[str, str, Callable[[str], bool]], ...] = (
            (".help", "Show help for dot-commands", self._cmd_help),
            (".use", "Switch the format hypothesis family", self._cmd_use),
            (".families", "List catalog families", self._cmd_families),
            (".fixtures", "List loaded fixture ids", self._cmd_fixtures),
            (".fixture", "Select a fixture by number or id", self._cmd_fixture),
            (".render", "Render the hypothesis over messages + segment map", self._cmd_render),
            (".tokens", "Exact token IDs of the full render", self._cmd_tokens),
            (".boundary", "Set the boundary token-ID sequence", self._cmd_boundary),
            (".stream", "Set the recorded token stream to walk", self._cmd_stream),
            (".step", "Step boundary detection one token at a time", self._cmd_step),
            (".audit", "Audit the fixture for special-token literals", self._cmd_audit),
            (".say", "Enter messages multiline, end with a lone '.'", self._cmd_say),
            (".save", "Append the session transcript to a file", self._cmd_save),
            (".read", "Execute a scripted command file", self._cmd_read),
            (".quit", "Exit the debugger", self._cmd_quit),
            (".q", "Exit the debugger", self._cmd_quit),
        )
        self._commands: dict[str, Callable[[str], bool]] = {name: func for name, _, func in entries}
        self._summaries: dict[str, str] = {name: summary for name, summary, _ in entries}
        tok = "attached" if tokenizer is not None else "none (rerun with --model PATH)"
        self.intro = (
            f"model-protocol-lab debug {__version__}\n"
            f"dataset: {dataset_label} ({len(self._cases)} fixtures) | "
            f"family: {self._family} | tokenizer: {tok}\n"
            "Enter .help for usage hints."
        )
        self._refresh_prompt()

    @property
    def _fmt(self) -> Format:
        return CATALOG[self._family]

    @property
    def _case(self) -> Fixture:
        return self._cases[self._case_index]

    def _refresh_prompt(self) -> None:
        if self._say_buffer is not None:
            self.prompt = "...> "
        else:
            self.prompt = f"mpl:{self._family} fixture-{self._case_index + 1} > "

    # -- core loop -----------------------------------------------------

    def onecmd(self, line: str) -> bool:
        """Run one line; capture its output into the transcript; never die."""
        if not line.strip():
            return False
        echo_prompt = self.prompt
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            try:
                stop = self._dispatch(line)
            except Exception as exc:  # noqa: BLE001 -- REPL boundary: every failure becomes "error: ..."
                # Exception text can carry hostile bytes (control
                # characters from fixture/dataset errors); strip them so
                # the terminal never sees raw escapes.
                print(f"error: {_sanitize_display(str(exc))}")
                stop = False
        text = buffer.getvalue()
        self._transcript.append(f"{echo_prompt}{line}\n")
        if text:
            self._transcript.append(text)
        # Bound transcript memory: keep the most recent output; .save notes
        # the truncation so a saved transcript never silently loses its head.
        total = sum(len(chunk) for chunk in self._transcript)
        if total > _MAX_SAVE_BYTES:
            kept: list[str] = []
            remaining = _MAX_SAVE_BYTES
            for chunk in reversed(self._transcript):
                if remaining <= 0:
                    break
                kept.append(chunk[-remaining:] if len(chunk) > remaining else chunk)
                remaining -= len(kept[-1])
            kept.reverse()
            self._transcript = ["[... transcript truncated ...]\n", *kept]
            self._transcript_truncated = True
        self.stdout.write(text)
        return stop

    def _dispatch(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if stripped == "EOF":
            return True
        if self._say_buffer is not None:
            # Multiline message entry: everything is content until the lone
            # "." terminator (documented; dot-commands are unavailable here).
            # Bounded like .read so an accidental paste cannot grow memory
            # without limit.
            if stripped == ".":
                self._finish_say()
            else:
                if len(self._say_buffer) >= _MAX_SCRIPT_LINES:
                    raise ValueError(
                        f"message buffer full (max {_MAX_SCRIPT_LINES} lines); end with a lone '.'"
                    )
                self._say_buffer.append(line)
                if sum(len(entry) for entry in self._say_buffer) > _MAX_SCRIPT_BYTES:
                    self._say_buffer = None
                    self._refresh_prompt()
                    raise ValueError(
                        "message buffer full (max 1MB); buffer discarded, "
                        "end with a lone '.' to start over"
                    )
            return False
        if stripped.startswith("#"):
            return False
        if not stripped.startswith("."):
            word = stripped.split(None, 1)[0]
            raise ValueError(f"unknown command {word!r}; dot-commands start with '.' (try .help)")
        parts = stripped.split(None, 1)
        name, arg = parts[0], (parts[1] if len(parts) > 1 else "")
        return self._commands[self._resolve_command(name)](arg.strip())

    def _resolve_command(self, name: str) -> str:
        """Exact match, else the unique unambiguous abbreviation (sqlite3)."""
        if name in self._commands:
            return name
        matches = [key for key in self._commands if key.startswith(name)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise ValueError(f"ambiguous command {name!r}; matches: {', '.join(sorted(matches))}")
        raise ValueError(f"unknown dot-command {name!r} (try .help)")

    # -- shared helpers -------------------------------------------------

    def _active_messages(self) -> tuple[tuple[Message, ...], str]:
        if self._buffered is not None:
            return self._buffered, "buffered"
        return self._case.messages, f"fixture={_sanitize_display(self._case.id)}"

    def _format_ids(self, ids: tuple[int, ...]) -> str:
        if len(ids) > _TOKEN_DISPLAY_CAP:
            head = ", ".join(str(token_id) for token_id in ids[:_TOKEN_DISPLAY_CAP])
            return f"[{head}, ... ({len(ids)} total)]"
        return f"[{', '.join(str(token_id) for token_id in ids)}]"

    def _encode(self, text: str) -> tuple[int, ...]:
        return tuple(self._tokenizer.encode(text, add_special_tokens=False))

    def _print_step(
        self,
        position: int,
        total: int,
        token_id: int,
        seen: tuple[int, ...],
        boundary: tuple[int, ...],
    ) -> None:
        at = first_boundary(seen, boundary)
        overhang = _overhang(seen, boundary)
        parts = [f"token {position}/{total} id={token_id}"]
        if self._tokenizer is not None:
            try:
                text = self._tokenizer.decode([token_id])
            except Exception:  # noqa: BLE001 -- decode failure degrades to no text, never kills the walk
                text = ""
            if text:
                parts.append(f"text={text!r}")
        parts.append(f"overhang={overhang}/{len(boundary)}")
        parts.append(f"boundary_at={at if at is not None else 'none'}")
        print(" ".join(parts))

    def _finish_say(self) -> None:
        buffer = self._say_buffer or []
        self._say_buffer = None
        self._refresh_prompt()
        if not buffer:
            print("message buffer discarded (empty)")
            return
        messages: list[Message] = []
        expected: Literal["user", "assistant"] = "user"
        for raw in buffer:
            role, content = _parse_say_line(raw)
            messages.append(Message(role=role if role is not None else expected, content=content))
            if role is None:
                expected = "assistant" if expected == "user" else "user"
        self._buffered = tuple(messages)
        print(f"buffered {len(messages)} message(s); .render uses them until the next .say")

    def read_commands(self, path: str | Path) -> bool:
        """Execute a scripted command file; bounded. True when .quit is seen."""
        target = str(path).strip()
        if not target:
            raise ValueError("usage: .read <path>")
        if self._read_depth >= _MAX_READ_DEPTH:
            raise ValueError(f"nested .read too deep (max {_MAX_READ_DEPTH})")
        script = Path(target).expanduser()
        try:
            # stat() size is unreliable for non-regular files (0 for /proc
            # entries, blocking for FIFOs): only regular files are scripts.
            if not script.is_file():
                raise ValueError(f"unreadable script file: {script}")
            size = script.stat().st_size
        except OSError:
            raise ValueError(f"unreadable script file: {script}") from None
        if size > _MAX_SCRIPT_BYTES:
            raise ValueError(f"refusing script larger than 1MB: {script}")
        try:
            text = script.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"cannot read script {script}: {exc}") from None
        lines = text.splitlines()
        if len(lines) > _MAX_SCRIPT_LINES:
            raise ValueError(f"refusing script longer than {_MAX_SCRIPT_LINES} lines: {script}")
        self._read_depth += 1
        try:
            for raw in lines:
                stripped = raw.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if self.onecmd(raw):
                    return True
        finally:
            self._read_depth -= 1
        return False

    # -- dot-commands ----------------------------------------------------

    def _cmd_help(self, arg: str) -> bool:
        """Usage: .help [COMMAND|PATTERN]

        With no argument, list every dot-command. With an argument, show
        full help for one command (unambiguous abbreviations allowed, e.g.
        ".help ren"); when nothing matches exactly, list the commands whose
        name or summary contains PATTERN, sqlite3-style.
        """
        target = arg.strip()
        if not target:
            width = max(len(name) for name in self._commands)
            for name in sorted(self._commands):
                print(f"{name:<{width}}  {self._summaries[name]}")
            return False
        key = target if target.startswith(".") else f".{target}"
        try:
            resolved = self._resolve_command(key)
        except ValueError:
            pattern = target.lower().lstrip(".")
            hits = [
                name
                for name in sorted(self._commands)
                if pattern in name or pattern in self._summaries[name].lower()
            ]
            if not hits:
                raise
            print("matching commands:")
            for name in hits:
                print(f"  {name}: {self._summaries[name]}")
            return False
        print(f"{resolved}: {self._summaries[resolved]}")
        print((self._commands[resolved].__doc__ or "").strip())
        return False

    def _cmd_use(self, arg: str) -> bool:
        """Usage: .use <family>

        Switch the format hypothesis to another catalog family (validated
        against CATALOG; see .families). The prompt and subsequent .render
        / .tokens / .audit calls follow the new hypothesis.
        """
        family = arg.strip()
        if not family:
            raise ValueError("usage: .use <family> (see .families)")
        if family not in CATALOG:
            raise ValueError(f"unknown family {family!r}; valid families: {', '.join(CATALOG)}")
        self._family = family
        self._refresh_prompt()
        print(f"using family {family}")
        return False

    def _cmd_families(self, arg: str) -> bool:
        """Usage: .families

        List every format-hypothesis family in the catalog.
        """
        if arg:
            raise ValueError("usage: .families (takes no arguments)")
        for name in CATALOG:
            print(name)
        return False

    def _cmd_fixtures(self, arg: str) -> bool:
        """Usage: .fixtures

        List the loaded fixtures (search split then held-out split) with
        1-based numbers for .fixture; the current one is marked with '*'.
        """
        if arg:
            raise ValueError("usage: .fixtures (takes no arguments)")
        for index, case in enumerate(self._cases):
            marker = " *" if index == self._case_index else ""
            print(f"{index + 1}: {_sanitize_display(case.id)}{marker}")
        return False

    def _cmd_fixture(self, arg: str) -> bool:
        """Usage: .fixture <n|id>

        Select the active fixture by its 1-based .fixtures number or by id.
        A numeric argument is tried as a number first, then as a literal id.
        """
        target = arg.strip()
        if not target:
            raise ValueError("usage: .fixture <n|id> (see .fixtures)")
        index: int | None = None
        if target.isascii() and target.isdigit():
            number = int(target)
            if 1 <= number <= len(self._cases):
                index = number - 1
        if index is None:
            for position, case in enumerate(self._cases):
                if case.id == target:
                    index = position
                    break
        if index is None:
            raise ValueError(f"unknown fixture {target!r} (see .fixtures)")
        self._case_index = index
        self._refresh_prompt()
        print(f"selected fixture {index + 1}: {_sanitize_display(self._case.id)}")
        return False

    def _cmd_render(self, arg: str) -> bool:
        """Usage: .render

        Print the active hypothesis rendered over the active messages (the
        selected fixture's messages, or the .say buffer when one exists),
        followed by a per-segment map: segment kind, char span, and -- when
        a tokenizer is attached -- the exact token IDs of each segment.
        """
        if arg:
            raise ValueError("usage: .render (takes no arguments)")
        messages, label = self._active_messages()
        segments = _render_segments(self._fmt, messages)
        rendered = "".join(text for _, text in segments)
        print(
            f"render: family={self._family} {label} messages={len(messages)} chars={len(rendered)}"
        )
        print(_sanitize_display(rendered))
        print("segments:")
        offset = 0
        for position, (kind, text) in enumerate(segments):
            start, end = offset, offset + len(text)
            offset = end
            entry = f"  [{position}] {kind:<16} chars {start}:{end}"
            if self._tokenizer is not None:
                entry += f" tokens={self._format_ids(self._encode(text))}"
            print(entry)
        return False

    def _cmd_tokens(self, arg: str) -> bool:
        """Usage: .tokens

        Print the exact token IDs of the full render (needs --model).
        """
        if arg:
            raise ValueError("usage: .tokens (takes no arguments)")
        if self._tokenizer is None:
            print("no tokenizer attached; rerun with --model PATH for exact token IDs.")
            return False
        messages, label = self._active_messages()
        rendered = self._fmt.render(messages)
        ids = self._encode(rendered)
        print(f"tokens: family={self._family} {label} count={len(ids)}")
        print(f"ids={self._format_ids(ids)}")
        return False

    def _cmd_boundary(self, arg: str) -> bool:
        """Usage: .boundary ID,ID,...

        Set the boundary token-ID sequence the stepper looks for
        (1-16 ASCII IDs, same grammar as the CLI --boundary flag).
        """
        if not arg:
            raise ValueError("usage: .boundary ID,ID,...")
        self._boundary = _parse_id_list(arg, what="boundary", max_len=_MAX_BOUNDARY_IDS)
        print(f"boundary set: {list(self._boundary)}")
        return False

    def _cmd_stream(self, arg: str) -> bool:
        """Usage: .stream ID,ID,...

        Record a token stream for .step to walk (up to 4096 ASCII IDs);
        resets the walk to the start of the new stream.
        """
        if not arg:
            raise ValueError("usage: .stream ID,ID,...")
        self._stream = _parse_id_list(arg, what="stream", max_len=_MAX_STREAM_TOKENS)
        self._seen = []
        print(f"stream set: {len(self._stream)} tokens; walk it with .step")
        return False

    def _cmd_step(self, arg: str) -> bool:
        """Usage: .step [ID,ID,...]

        Walk boundary detection one token at a time. With no argument, take
        the next token from the recorded .stream; with IDs, walk that typed
        stream instead. Each step prints the running overhang (how much of
        the boundary is currently matched as a stream suffix) and the first
        full-boundary position via core.first_boundary.
        """
        if self._boundary is None:
            raise ValueError("no boundary set; use .boundary ID,ID,... first")
        boundary = self._boundary
        if arg:
            typed = _parse_id_list(arg, what="step", max_len=_MAX_STREAM_TOKENS)
            seen: list[int] = []
            for token_id in typed:
                seen.append(token_id)
                self._print_step(len(seen), len(typed), token_id, tuple(seen), boundary)
            return False
        if self._stream is None:
            raise ValueError("no recorded stream; use .stream ID,ID,... or .step ID,ID,...")
        if len(self._seen) >= len(self._stream):
            raise ValueError(
                f"stream exhausted ({len(self._stream)} tokens); use .stream to load a new one"
            )
        token_id = self._stream[len(self._seen)]
        self._seen.append(token_id)
        self._print_step(len(self._seen), len(self._stream), token_id, tuple(self._seen), boundary)
        return False

    def _cmd_audit(self, arg: str) -> bool:
        """Usage: .audit

        Run fixtures.audit_special_token_literals on the selected fixture
        against the active hypothesis: flags special-token literals (e.g.
        <|im_start|>) embedded in fixture text that could forge a role
        boundary and poison the experiment.
        """
        if arg:
            raise ValueError("usage: .audit (takes no arguments)")
        case = self._case
        findings = audit_special_token_literals(case, (self._fmt,))
        sanitized_id = _sanitize_display(case.id)
        print(
            f"audit: {len(findings)} finding(s) on fixture {sanitized_id} "
            f"with family {self._family}"
        )
        for finding in findings:
            print(
                f"  {_sanitize_display(finding.fixture_id)} {_sanitize_display(finding.field)} "
                f"literal={_sanitize_display(finding.literal)!r} "
                f"format={_sanitize_display(finding.format_name)} offset={finding.offset}"
            )
        return False

    def _cmd_say(self, arg: str) -> bool:
        """Usage: .say

        Start multiline message entry (sqlite3-style continuation prompt
        "...>"). Each line is one message: an optional "user:"/"assistant:"
        prefix sets the role explicitly, otherwise roles alternate starting
        with user. A lone "." on its own line ends entry; .render then uses
        the buffered messages instead of the fixture's.
        """
        if arg:
            raise ValueError("usage: .say (takes no arguments)")
        if self._say_buffer is not None:
            raise ValueError("already entering messages; end with a lone '.'")
        self._say_buffer = []
        self._refresh_prompt()
        print("enter messages (one per line); end with a lone '.'")
        return False

    def _cmd_save(self, arg: str) -> bool:
        """Usage: .save <path>

        Append the session transcript (command echoes plus their output) to
        a file. The parent directory must exist; an existing file must be a
        regular file and stay under 5MB. The path is used as given --
        absolute and parent-relative paths are intentional (the operator's
        own REPL command, e.g. saving under /tmp); a shared --script file
        can therefore direct output anywhere the operator could write.
        """
        target = arg.strip()
        if not target:
            raise ValueError("usage: .save <path>")
        if "\x00" in target:
            raise ValueError("invalid path")
        path = Path(target).expanduser()
        if not path.parent.is_dir():
            raise ValueError(f"not a directory: {path.parent}")
        if path.exists():
            if not path.is_file():
                raise ValueError(f"not a regular file: {path}")
            if path.stat().st_size > _MAX_SAVE_BYTES:
                raise ValueError(f"refusing to append to a file over 5MB: {path}")
        text = "".join(self._transcript)
        if len(text) > _MAX_SAVE_BYTES:
            raise ValueError("transcript too large to save (over 5MB)")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text)
        print(f"saved {len(text)} chars of transcript to {_sanitize_display(str(path))}")
        return False

    def _cmd_read(self, arg: str) -> bool:
        """Usage: .read <path>

        Execute a scripted command file line by line (bounded: refused over
        1MB or 10k lines; '#' comments and blank lines skipped; nesting
        capped at 5). A .quit inside the script stops the session.
        """
        return self.read_commands(arg)

    def _cmd_quit(self, arg: str) -> bool:
        """Usage: .quit | .q

        Exit the debugger.
        """
        if arg:
            raise ValueError("usage: .quit (takes no arguments)")
        return True


def run_script(shell: DebugShell, path: str | Path) -> int:
    """Run a .read script non-interactively; deterministic output."""
    try:
        shell.read_commands(path)
    except ValueError as exc:
        print(f"debug: {exc}", file=sys.stderr)
        return 1
    return 0


def run_debug(args: argparse.Namespace) -> int:
    """Entry point for ``mpl debug``; never loads model weights."""
    try:
        split = load_dataset(args.dataset)  # nosec B615 -- own bounded JSON reader
    except ValueError as exc:
        print(f"debug: invalid dataset: {exc}", file=sys.stderr)
        return 1
    tokenizer: Any = None
    if args.model is not None:
        try:
            tokenizer = attach_tokenizer(args.model)
        except ImportError:
            print("debug: --model requires the 'transformers' package", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001 -- sanitization boundary, never leak loader text
            # The parse can echo hostile checkpoint bytes (and raise
            # non-ValueError); only exact-match fixed codes reach the
            # terminal, everything else collapses to the fixed code.
            from .doctor import _loader_failure_code

            print(f"debug: {_loader_failure_code(exc)}", file=sys.stderr)
            return 1
    shell = DebugShell(split, dataset_label=str(args.dataset), tokenizer=tokenizer)
    if args.script is not None:
        return run_script(shell, args.script)
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print()
    return 0
