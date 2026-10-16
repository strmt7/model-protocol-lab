"""Cycle 71: adversarial audit of the `mpl debug` REPL deeper paths.

Complements test_cycle36_debug_repl_verify.py (command basics) and
test_cycle23_debug_helpers_verify.py (helper unit tests). This file audits
what those left shallow:

* multi-turn sessions: state carried across commands (family, fixture,
  boundary, stream walk, .say buffer), and that a *failed* command leaves
  prior state untouched;
* command-parsing edges: case, tabs, unicode, dotted suffixes, very long
  input, numeric-looking fixture ids, empty/whitespace arguments;
* error recovery: bad input must print ``error: ...`` and never corrupt
  session state or kill the loop (exploding synthetic tokenizers, .say
  buffer overflow at both byte and line caps, EOF/Ctrl-D mid-.say);
* transcript/history boundaries: the 5 MB trim invariant (marker first,
  newest output fully preserved, total under cap), the 512-id display cap
  at exactly 512/513, echo ordering;
* fail-closed backends: synthetic tokenizers only; the --model path never
  reads weight files, collapses loader failures to fixed codes, and raises
  ImportError before touching the filesystem when transformers is absent;
* cli.py wiring of the ``debug`` subcommand.

Method: independent oracles hand-rolled in this file (a brute-force
overhang / first-boundary implementation written with explicit loops,
exact expected strings, CATALOG renders as the oracle for .render), plus
minimal-pair boundary tests (16 vs 17 ids, 4096 vs 4097, 1MB vs 1MB+1,
10000 vs 10001 lines, 512 vs 513 display ids, 2^31-1 vs 2^31).

Local-only: no model downloads, no network, no remote execution. The only
real dependency exercised is a hand-built ``tokenizer.json`` (via the
``tokenizers`` library) paired with *garbage* weight files, which proves
the loader is tokenizer-only.
"""

import argparse
import contextlib
import io
import os
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from unittest import mock

import model_protocol_lab.debug as debug_module
from model_protocol_lab import __version__
from model_protocol_lab.catalog import CATALOG
from model_protocol_lab.cli import build_parser
from model_protocol_lab.core import Fixture, Message, Split
from model_protocol_lab.debug import DebugShell, attach_tokenizer, run_debug, run_script
from model_protocol_lab.fixtures import split_to_json_text

_HAS_TRANSFORMERS = find_spec("transformers") is not None


class OrdTokenizer:
    """Deterministic synthetic tokenizer: one id per char (its ord()).

    ``decode`` uses ``chr()``, so ids above 0x10FFFF raise -- the same
    fail-closed decode path a hostile id would exercise.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> tuple[int, ...]:
        return tuple(ord(char) for char in text)

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class ExplodingTokenizer:
    """Every operation raises: the REPL must degrade, never die."""

    def encode(self, text: str, add_special_tokens: bool = False) -> tuple[int, ...]:
        raise RuntimeError("encode exploded")

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        raise RuntimeError("decode exploded")


class CountTokenizer:
    """encode() ignores text and returns exactly ``count`` ids."""

    def __init__(self, count: int) -> None:
        self._count = count

    def encode(self, text: str, add_special_tokens: bool = False) -> tuple[int, ...]:
        return tuple(range(self._count))

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return ""


def _make_split() -> Split:
    search = (
        Fixture(
            id="s1",
            group="g1",
            messages=(Message("user", "hello"),),
            expected="hi",
        ),
        Fixture(
            id="s2",
            group="g2",
            messages=(
                Message("user", "a"),
                Message("assistant", "b"),
                Message("user", "c"),
            ),
            expected="d",
        ),
    )
    held_out = (Fixture(id="h1", group="g3", messages=(Message("user", "q"),), expected="a"),)
    return Split(search=search, held_out=held_out)


def _make_shell(
    split: Split | None = None, tokenizer: Any = "ord", **kwargs: Any
) -> tuple[DebugShell, io.StringIO]:
    """Fresh shell writing to a buffer; never touches real stdin/stdout."""
    buf = io.StringIO()
    if tokenizer == "ord":
        tok: Any = OrdTokenizer()
    elif tokenizer == "none":
        tok = None
    elif tokenizer == "boom":
        tok = ExplodingTokenizer()
    else:
        tok = tokenizer
    shell = DebugShell(
        split if split is not None else _make_split(),
        dataset_label="test",
        tokenizer=tok,
        stdin=io.StringIO(""),
        stdout=buf,
        **kwargs,
    )
    return shell, buf


def _one(shell: DebugShell, buf: io.StringIO, command: str) -> tuple[bool, str]:
    """Run a single command; return (stop, output)."""
    buf.seek(0)
    buf.truncate(0)
    stop = shell.onecmd(command)
    return stop, buf.getvalue()


def _run(shell: DebugShell, buf: io.StringIO, *commands: str) -> str:
    """Feed queued commands through cmdloop(); return everything printed."""
    buf.seek(0)
    buf.truncate(0)
    shell.cmdqueue.extend(commands)
    shell.cmdloop()
    return buf.getvalue()


def _transcript(shell: DebugShell) -> str:
    return "".join(shell._transcript)


# ---------------------------------------------------------------------------
# Independent oracles: brute-force definitions written with explicit loops,
# deliberately structured unlike the source under test.
# ---------------------------------------------------------------------------


def _brute_overhang(seen: tuple[int, ...], boundary: tuple[int, ...]) -> int:
    """Longest k with seen[-k:] == boundary[:k], by direct comparison."""
    best = 0
    for k in range(min(len(seen), len(boundary)) + 1):
        ok = True
        for i in range(k):
            if seen[len(seen) - k + i] != boundary[i]:
                ok = False
                break
        if ok:
            best = k
    return best


def _brute_first_boundary(seen: tuple[int, ...], boundary: tuple[int, ...]) -> int | None:
    """First index i with seen[i:i+len(boundary)] == boundary, by scanning."""
    for i in range(len(seen) - len(boundary) + 1):
        if all(seen[i + j] == boundary[j] for j in range(len(boundary))):
            return i
    return None


def _expected_step_line(
    position: int,
    total: int,
    token_id: int,
    seen: tuple[int, ...],
    boundary: tuple[int, ...],
) -> str:
    """Hand-format a .step line from the brute-force oracles (no text part)."""
    overhang = _brute_overhang(seen, boundary)
    at = _brute_first_boundary(seen, boundary)
    return (
        f"token {position}/{total} id={token_id} "
        f"overhang={overhang}/{len(boundary)} "
        f"boundary_at={at if at is not None else 'none'}"
    )


class _Deterministic:
    """Tiny LCG: deterministic fuzz without the ``random`` module (S311)."""

    def __init__(self, seed: int) -> None:
        self._state = seed

    def below(self, n: int) -> int:
        self._state = (self._state * 6364136223846793005 + 1442695040888963407) % 2**64
        return self._state % n


class TestIdListGrammar(unittest.TestCase):
    """Unit oracle for _parse_id_list: the exact accept/reject grammar."""

    def _parse(self, value: str, what: str = "x", max_len: int = 16) -> tuple[int, ...]:
        return debug_module._parse_id_list(value, what=what, max_len=max_len)

    def test_valid_cases(self) -> None:
        self.assertEqual(self._parse("5"), (5,))
        self.assertEqual(self._parse("1,2,3"), (1, 2, 3))
        self.assertEqual(self._parse("007"), (7,))
        self.assertEqual(self._parse("0"), (0,))
        self.assertEqual(self._parse("2147483647"), (2147483647,))
        self.assertEqual(self._parse(" 1 , 2 "), (1, 2))
        self.assertEqual(self._parse("1,\t2"), (1, 2))

    def test_rejects_malformed(self) -> None:
        for bad in (
            "",
            "   ",
            ",",
            ",1",
            "1,",
            "1,,2",
            "a",
            "1b",
            "-1",
            "+1",
            "0x1A",
            "1.5",
            "1;2",
            "1 2",
            "\u00b9",  # superscript one: isdigit() but not ASCII
            "\uff11\uff12",  # full-width digits: not ASCII 0-9
        ):
            with self.assertRaises(ValueError, msg=bad):
                self._parse(bad)

    def test_rejects_over_int31(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._parse("2147483648")
        self.assertIn("token IDs must be < 2^31", str(ctx.exception))

    def test_error_names_what_and_max_len(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._parse("nope", what="boundary", max_len=16)
        self.assertEqual(
            str(ctx.exception),
            "boundary: expected 1-16 comma-separated ASCII token IDs",
        )
        with self.assertRaises(ValueError) as ctx2:
            self._parse("nope", what="stream", max_len=4096)
        self.assertEqual(
            str(ctx2.exception),
            "stream: expected 1-4096 comma-separated ASCII token IDs",
        )

    def test_max_len_enforced(self) -> None:
        self.assertEqual(self._parse(",".join(["1"] * 16), max_len=16), (1,) * 16)
        with self.assertRaises(ValueError):
            self._parse(",".join(["1"] * 17), max_len=16)


class TestIdListCommandBounds(unittest.TestCase):
    """Minimal pairs at the command level: 16 vs 17, 4096 vs 4097."""

    def test_boundary_16_ok_17_rejected(self) -> None:
        shell, buf = _make_shell()
        stop, out = _one(shell, buf, ".boundary " + ",".join(str(i) for i in range(16)))
        self.assertFalse(stop)
        self.assertIn("boundary set:", out)
        stop, out = _one(shell, buf, ".boundary " + ",".join(str(i) for i in range(17)))
        self.assertFalse(stop)
        self.assertIn("error: boundary: expected 1-16 comma-separated ASCII token IDs", out)

    def test_stream_4096_ok_4097_rejected(self) -> None:
        shell, buf = _make_shell()
        stop, out = _one(shell, buf, ".stream " + ",".join(str(i) for i in range(4096)))
        self.assertFalse(stop)
        self.assertIn("stream set: 4096 tokens", out)
        stop, out = _one(shell, buf, ".stream " + ",".join(str(i) for i in range(4097)))
        self.assertFalse(stop)
        self.assertIn("error: stream: expected 1-4096 comma-separated ASCII token IDs", out)

    def test_step_typed_4096_ok(self) -> None:
        shell, buf = _make_shell(tokenizer="none")
        _one(shell, buf, ".boundary 1")
        _, out = _one(shell, buf, ".step " + ",".join(["2"] * 4096))
        self.assertIn("token 4096/4096 id=2", out)


class TestStepWalkOracle(unittest.TestCase):
    """Differential: every .step line must match the brute-force oracles."""

    def test_hand_computed_walk(self) -> None:
        # boundary=(5,6,7), typed stream=(5,6,5,6,7): overhangs and the
        # boundary position below are computed by hand, not from the code.
        shell, buf = _make_shell(tokenizer="none")
        _one(shell, buf, ".boundary 5,6,7")
        _, out = _one(shell, buf, ".step 5,6,5,6,7")
        self.assertEqual(
            out.splitlines(),
            [
                "token 1/5 id=5 overhang=1/3 boundary_at=none",
                "token 2/5 id=6 overhang=2/3 boundary_at=none",
                "token 3/5 id=5 overhang=1/3 boundary_at=none",
                "token 4/5 id=6 overhang=2/3 boundary_at=none",
                "token 5/5 id=7 overhang=3/3 boundary_at=2",
            ],
        )

    def test_random_walks_match_brute_force(self) -> None:
        rng = _Deterministic(20261016)
        for _ in range(25):
            boundary = tuple(rng.below(5) for _ in range(rng.below(4) + 1))
            stream = tuple(rng.below(5) for _ in range(rng.below(12) + 1))
            shell, buf = _make_shell(tokenizer="none")
            _one(shell, buf, ".boundary " + ",".join(map(str, boundary)))
            _, out = _one(shell, buf, ".step " + ",".join(map(str, stream)))
            seen: list[int] = []
            expected = []
            for pos, token_id in enumerate(stream, start=1):
                seen.append(token_id)
                expected.append(
                    _expected_step_line(pos, len(stream), token_id, tuple(seen), boundary)
                )
            self.assertEqual(out.splitlines(), expected)

    def test_recorded_walk_matches_brute_force(self) -> None:
        shell, buf = _make_shell(tokenizer="none")
        boundary = (3, 1, 3)
        stream = (3, 1, 2, 3, 1, 3)
        _one(shell, buf, ".boundary " + ",".join(map(str, boundary)))
        _one(shell, buf, ".stream " + ",".join(map(str, stream)))
        seen: list[int] = []
        for pos, token_id in enumerate(stream, start=1):
            _, out = _one(shell, buf, ".step")
            seen.append(token_id)
            self.assertEqual(
                out.strip(),
                _expected_step_line(pos, len(stream), token_id, tuple(seen), boundary),
            )

    def test_single_token_boundary(self) -> None:
        shell, buf = _make_shell(tokenizer="none")
        _one(shell, buf, ".boundary 9")
        _, out = _one(shell, buf, ".step 4,9")
        self.assertEqual(
            out.splitlines(),
            [
                "token 1/2 id=4 overhang=0/1 boundary_at=none",
                "token 2/2 id=9 overhang=1/1 boundary_at=1",
            ],
        )

    def test_typed_stream_does_not_touch_recorded_walk(self) -> None:
        shell, buf = _make_shell(tokenizer="none")
        _one(shell, buf, ".boundary 1")
        _one(shell, buf, ".stream 7,8")
        _one(shell, buf, ".step")  # recorded: token 1/2
        _one(shell, buf, ".step 1,1,1")  # typed walk: independent
        _, out = _one(shell, buf, ".step")  # recorded resumes at 2/2
        self.assertIn("token 2/2 id=8", out)

    def test_positions_are_1_based(self) -> None:
        shell, buf = _make_shell(tokenizer="none")
        _one(shell, buf, ".boundary 1")
        _one(shell, buf, ".stream 2,3")
        _, first = _one(shell, buf, ".step")
        self.assertTrue(first.startswith("token 1/2 "))


class TestMultiTurnSessionState(unittest.TestCase):
    """State carried across commands in one session; failures keep state."""

    def test_long_scripted_session_via_cmdloop(self) -> None:
        shell, buf = _make_shell(tokenizer="none")
        out = _run(
            shell,
            buf,
            ".use llama3",
            ".fixture 2",
            ".say",
            "user: hi",
            ".",
            ".render",
            ".boundary 1,2",
            ".stream 1,2",
            ".step",
            ".step",
            ".audit",
            ".quit",
        )
        markers = [
            "using family llama3",
            "selected fixture 2: s2",
            "enter messages (one per line); end with a lone '.'",
            "buffered 1 message(s); .render uses them until the next .say",
            "render: family=llama3 buffered messages=1",
            "boundary set: [1, 2]",
            "stream set: 2 tokens; walk it with .step",
            "token 1/2 id=1 overhang=1/2 boundary_at=none",
            "token 2/2 id=2 overhang=2/2 boundary_at=0",
            "audit: ",
            "with family llama3",
        ]
        index = -1
        for marker in markers:
            found = out.find(marker, index + 1)
            self.assertGreater(found, index, f"marker out of order or missing: {marker!r}")
            index = found
        # Prompt followed the session: family switched, fixture switched,
        # continuation prompt active only mid-.say.
        self.assertEqual(shell.prompt, "mpl:llama3 fixture-2 > ")
        transcript = _transcript(shell)
        self.assertIn("mpl:llama3 fixture-1 > .fixture 2\n", transcript)
        self.assertIn("...> user: hi\n", transcript)
        self.assertIn("...> .\n", transcript)

    def test_same_session_twice_is_byte_identical(self) -> None:
        commands = (
            ".families",
            ".use llama3",
            ".fixtures",
            ".fixture 2",
            ".say",
            "user: hi",
            "assistant: hello",
            ".",
            ".render",
            ".boundary 5,6,7",
            ".step 5,6,5,6,7",
            ".audit",
            ".quit",
        )
        shell1, buf1 = _make_shell()
        shell2, buf2 = _make_shell()
        out1 = _run(shell1, buf1, *commands)
        out2 = _run(shell2, buf2, *commands)
        self.assertEqual(out1, out2)

    def test_say_buffer_survives_fixture_switch(self) -> None:
        shell, buf = _make_shell()
        _one(shell, buf, ".say")
        _one(shell, buf, "user: hi")
        _one(shell, buf, ".")
        _one(shell, buf, ".fixture 2")
        _, out = _one(shell, buf, ".render")
        self.assertIn("buffered messages=1", out)

    def test_say_buffer_survives_use_switch(self) -> None:
        shell, buf = _make_shell()
        _one(shell, buf, ".say")
        _one(shell, buf, "user: hi")
        _one(shell, buf, ".")
        _one(shell, buf, ".use llama3")
        _, out = _one(shell, buf, ".render")
        self.assertTrue(out.startswith("render: family=llama3 buffered messages=1"))

    def test_failed_use_keeps_family(self) -> None:
        shell, buf = _make_shell()
        _one(shell, buf, ".use llama3")
        _, out = _one(shell, buf, ".use nosuchfamily")
        self.assertIn("error: unknown family 'nosuchfamily'", out)
        self.assertEqual(shell.prompt, "mpl:llama3 fixture-1 > ")
        _, out = _one(shell, buf, ".render")
        self.assertTrue(out.startswith("render: family=llama3 "))

    def test_failed_fixture_keeps_index(self) -> None:
        shell, buf = _make_shell()
        _one(shell, buf, ".fixture 2")
        _, out = _one(shell, buf, ".fixture 99")
        self.assertIn("error: unknown fixture '99'", out)
        self.assertEqual(shell.prompt, "mpl:chatml fixture-2 > ")
        _, out = _one(shell, buf, ".fixtures")
        self.assertIn("2: s2 *", out)

    def test_failed_boundary_keeps_old_boundary(self) -> None:
        shell, buf = _make_shell(tokenizer="none")
        _one(shell, buf, ".boundary 7")
        _, out = _one(shell, buf, ".boundary notanid")
        self.assertIn("error: boundary: expected 1-16 comma-separated ASCII token IDs", out)
        _, out = _one(shell, buf, ".step 7")
        self.assertIn("overhang=1/1", out)
        self.assertIn("boundary_at=0", out)

    def test_failed_stream_keeps_stream_and_position(self) -> None:
        shell, buf = _make_shell(tokenizer="none")
        _one(shell, buf, ".boundary 9")
        _one(shell, buf, ".stream 1,2,3")
        _one(shell, buf, ".step")
        _, out = _one(shell, buf, ".stream oops")
        self.assertIn("error: stream: expected 1-4096 comma-separated ASCII token IDs", out)
        _, out = _one(shell, buf, ".step")
        self.assertIn("token 2/3 id=2", out)

    def test_new_stream_resets_walk(self) -> None:
        shell, buf = _make_shell(tokenizer="none")
        _one(shell, buf, ".boundary 1")
        _one(shell, buf, ".stream 1,2")
        _one(shell, buf, ".step")
        _one(shell, buf, ".stream 9,8")
        _, out = _one(shell, buf, ".step")
        self.assertIn("token 1/2 id=9", out)

    def test_boundary_change_does_not_reset_walk(self) -> None:
        shell, buf = _make_shell(tokenizer="none")
        _one(shell, buf, ".boundary 1")
        _one(shell, buf, ".stream 5,6")
        _one(shell, buf, ".step")
        _one(shell, buf, ".boundary 5")
        _, out = _one(shell, buf, ".step")
        self.assertIn("token 2/2 id=6", out)

    def test_nested_say_line_is_content_buffer_kept(self) -> None:
        # Via dispatch, a ".say" line during entry is message content (the
        # "already entering" guard in _cmd_say is direct-call-only); the
        # buffer must keep both lines, unclobbered.
        shell, buf = _make_shell()
        _one(shell, buf, ".say")
        _one(shell, buf, "hello")
        stop, out = _one(shell, buf, ".say")
        self.assertFalse(stop)
        self.assertNotIn("error:", out)
        _one(shell, buf, "bye")
        _one(shell, buf, ".")
        self.assertIsNone(shell._say_buffer)  # finalized
        _, out = _one(shell, buf, ".render")
        self.assertIn("messages=3", out)
        expected_body = CATALOG["chatml"].render(
            (Message("user", "hello"), Message("assistant", ".say"), Message("user", "bye"))
        )
        self.assertIn(expected_body, out)

    def test_direct_double_say_call_rejected_buffer_untouched(self) -> None:
        shell, _ = _make_shell()
        with contextlib.redirect_stdout(io.StringIO()):
            shell._cmd_say("")
        with self.assertRaises(ValueError) as ctx:
            shell._cmd_say("")
        self.assertIn("already entering messages", str(ctx.exception))
        self.assertEqual(shell._say_buffer, [])

    def test_say_roles_alternate_starting_with_user(self) -> None:
        shell, buf = _make_shell()
        _one(shell, buf, ".say")
        _one(shell, buf, "first")
        _one(shell, buf, "second")
        _one(shell, buf, "third")
        _one(shell, buf, ".")
        _, out = _one(shell, buf, ".render")
        expected_body = CATALOG["chatml"].render(
            (
                Message("user", "first"),
                Message("assistant", "second"),
                Message("user", "third"),
            )
        )
        self.assertIn(expected_body, out)

    def test_quit_line_inside_say_is_content(self) -> None:
        # Dot-commands are unavailable during .say: even ".quit" is content.
        shell, buf = _make_shell()
        stop, _ = _one(shell, buf, ".say")
        self.assertFalse(stop)
        stop, _ = _one(shell, buf, ".quit")
        self.assertFalse(stop)
        stop, out = _one(shell, buf, ".")
        self.assertFalse(stop)
        self.assertIn("buffered 1 message(s)", out)
        _, out = _one(shell, buf, ".render")
        self.assertIn(".quit", out)
        stop, _ = _one(shell, buf, ".quit")
        self.assertTrue(stop)

    def test_dot_read_line_inside_say_is_content(self) -> None:
        shell, buf = _make_shell()
        _one(shell, buf, ".say")
        _one(shell, buf, ".read /tmp/does_not_exist_xyz")
        _, out = _one(shell, buf, ".")
        self.assertIn("buffered 1 message(s)", out)
        _, out = _one(shell, buf, ".render")
        self.assertIn(".read /tmp/does_not_exist_xyz", out)

    def test_comment_line_inside_say_is_content_blank_line_dropped(self) -> None:
        shell, buf = _make_shell()
        _one(shell, buf, ".say")
        _one(shell, buf, "")  # blank lines never reach the buffer (onecmd drops them)
        _one(shell, buf, "# not a comment in here")
        _one(shell, buf, "hello")
        _one(shell, buf, "bye")
        _one(shell, buf, ".")
        _, out = _one(shell, buf, ".render")
        self.assertIn("messages=3", out)
        expected_body = CATALOG["chatml"].render(
            (
                Message("user", "# not a comment in here"),
                Message("assistant", "hello"),
                Message("user", "bye"),
            )
        )
        self.assertIn(expected_body, out)

    def test_eof_line_inside_say_ends_session(self) -> None:
        # Pinned behavior: a literal "EOF" line always ends the session, even
        # mid-.say. This is the safe choice -- cmd maps a real stdin EOF
        # (Ctrl-D) to the "EOF" line, and swallowing it as buffer content
        # would spin forever on a dead stdin.
        shell, buf = _make_shell()
        _one(shell, buf, ".say")
        _one(shell, buf, "user: partial")
        stop, _ = _one(shell, buf, "EOF")
        self.assertTrue(stop)

    def test_eof_line_in_script_ends_session(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "s.txt"
            script.write_text(".fixtures\nEOF\n.fixtures\n", encoding="utf-8")
            self.assertTrue(shell.read_commands(script))
        self.assertEqual(buf.getvalue().count("s1"), 1)  # second .fixtures never ran


class TestCommandParsingEdges(unittest.TestCase):
    def test_uppercase_dot_command_rejected(self) -> None:
        shell, buf = _make_shell()
        _, out = _one(shell, buf, ".HELP")
        self.assertIn("error: unknown dot-command '.HELP' (try .help)", out)

    def test_uppercase_bare_word_rejected(self) -> None:
        shell, buf = _make_shell()
        _, out = _one(shell, buf, "QUIT")
        self.assertIn("error: unknown command 'QUIT'; dot-commands start with '.'", out)

    def test_tab_between_command_and_arg(self) -> None:
        shell, buf = _make_shell()
        _, out = _one(shell, buf, ".fixture\t2")
        self.assertIn("selected fixture 2: s2", out)
        self.assertEqual(shell.prompt, "mpl:chatml fixture-2 > ")

    def test_dotted_suffix_is_unknown(self) -> None:
        shell, buf = _make_shell()
        _, out = _one(shell, buf, ".help.extra")
        self.assertIn("error: unknown dot-command '.help.extra'", out)

    def test_double_dot_is_unknown(self) -> None:
        shell, buf = _make_shell()
        _, out = _one(shell, buf, "..")
        self.assertIn("error: unknown dot-command '..'", out)

    def test_unicode_dot_command_error_keeps_unicode(self) -> None:
        shell, buf = _make_shell()
        _, out = _one(shell, buf, ".\u00fc\u00f1\u00eec\u00f8de")
        self.assertIn("error: unknown dot-command '.\u00fc\u00f1\u00eec\u00f8de'", out)

    def test_very_long_unknown_command_does_not_crash(self) -> None:
        shell, buf = _make_shell()
        _, out = _one(shell, buf, "." + "x" * 20000)
        self.assertIn("error: unknown dot-command", out)
        stop, _ = _one(shell, buf, ".quit")
        self.assertTrue(stop)

    def test_help_pattern_is_case_insensitive(self) -> None:
        shell, buf = _make_shell()
        _, out = _one(shell, buf, ".help TOKEN")
        self.assertIn("matching commands:", out)
        self.assertIn(".tokens", out)

    def test_help_dotted_abbreviation(self) -> None:
        shell, buf = _make_shell()
        _, out = _one(shell, buf, ".help .ren")
        self.assertIn(".render:", out)
        self.assertIn("Usage: .render", out)

    def test_help_with_blank_arg_lists_all(self) -> None:
        shell, buf = _make_shell()
        _, out = _one(shell, buf, ".help ")
        self.assertIn(".audit", out)
        self.assertIn(".quit", out)

    def test_help_lists_exactly_16_commands(self) -> None:
        shell, buf = _make_shell()
        _, out = _one(shell, buf, ".help")
        names = [line.split()[0] for line in out.splitlines() if line.strip()]
        self.assertEqual(
            names,
            [
                ".audit",
                ".boundary",
                ".families",
                ".fixture",
                ".fixtures",
                ".help",
                ".q",
                ".quit",
                ".read",
                ".render",
                ".save",
                ".say",
                ".step",
                ".stream",
                ".tokens",
                ".use",
            ],
        )

    def test_lone_dot_is_ambiguous(self) -> None:
        shell, buf = _make_shell()
        _, out = _one(shell, buf, ".")
        self.assertIn("error: ambiguous command '.'; matches:", out)

    def test_fixture_zero_negative_and_overflow_rejected(self) -> None:
        shell, buf = _make_shell()
        for bad in ("0", "-1", "4", "1 2"):
            _, out = _one(shell, buf, f".fixture {bad}")
            self.assertIn("error: unknown fixture", out, bad)
        self.assertEqual(shell.prompt, "mpl:chatml fixture-1 > ")

    def test_fixture_superscript_digit_rejected(self) -> None:
        # "\u00b9".isdigit() is True but isascii() is False: must not parse
        # as a number, and must not crash the session.
        shell, buf = _make_shell()
        _, out = _one(shell, buf, ".fixture \u00b9")
        self.assertIn("error: unknown fixture", out)

    def test_fixture_nul_byte_id_rejected_cleanly(self) -> None:
        shell, buf = _make_shell()
        _, out = _one(shell, buf, ".fixture \x00")
        self.assertIn("error: unknown fixture", out)
        stop, _ = _one(shell, buf, ".quit")
        self.assertTrue(stop)

    def test_use_blank_arg_is_usage_error(self) -> None:
        shell, buf = _make_shell()
        _, out = _one(shell, buf, ".use ")
        self.assertIn("error: usage: .use <family> (see .families)", out)

    def test_q_with_trailing_space_still_quits(self) -> None:
        shell, buf = _make_shell()
        stop, _ = _one(shell, buf, ".q ")
        self.assertTrue(stop)


class TestErrorRecovery(unittest.TestCase):
    """Bad input prints ``error: ...``; session state stays valid."""

    def test_render_with_exploding_tokenizer_recovers(self) -> None:
        shell, buf = _make_shell(tokenizer="boom")
        _, out = _one(shell, buf, ".render")
        self.assertIn("error: encode exploded", out)
        # Header and body printed before the segment-token failure; the
        # session is alive and the prompt is intact.
        self.assertIn("render: family=chatml", out)
        self.assertEqual(shell.prompt, "mpl:chatml fixture-1 > ")
        stop, _ = _one(shell, buf, ".quit")
        self.assertTrue(stop)

    def test_tokens_with_exploding_tokenizer_recovers(self) -> None:
        shell, buf = _make_shell(tokenizer="boom")
        _, out = _one(shell, buf, ".tokens")
        self.assertIn("error: encode exploded", out)
        _, out = _one(shell, buf, ".families")
        self.assertNotIn("error:", out)

    def test_step_decode_of_huge_id_degrades(self) -> None:
        # chr(2**31 - 1) raises in the synthetic tokenizer: _print_step must
        # degrade to no text= part, still reporting overhang/boundary.
        shell, buf = _make_shell(tokenizer="ord")
        _one(shell, buf, ".boundary 2147483647")
        _, out = _one(shell, buf, ".step 2147483647")
        self.assertEqual(out.strip(), "token 1/1 id=2147483647 overhang=1/1 boundary_at=0")
        self.assertNotIn("text=", out)

    def test_step_without_tokenizer_has_no_text_part(self) -> None:
        shell, buf = _make_shell(tokenizer="none")
        _one(shell, buf, ".boundary 1,2")
        _, out = _one(shell, buf, ".step 1,2")
        self.assertNotIn("text=", out)
        self.assertIn("token 2/2 id=2 overhang=2/2 boundary_at=0", out)

    def test_say_byte_overflow_discards_buffer_and_recovers(self) -> None:
        shell, buf = _make_shell()
        _one(shell, buf, ".say")
        _, out = _one(shell, buf, "x" * 1_000_001)
        self.assertIn(
            "error: message buffer full (max 1MB); buffer discarded, "
            "end with a lone '.' to start over",
            out,
        )
        # Buffer discarded, prompt restored, .render falls back to fixture.
        self.assertIsNone(shell._say_buffer)
        self.assertEqual(shell.prompt, "mpl:chatml fixture-1 > ")
        _, out = _one(shell, buf, ".render")
        self.assertIn("fixture=s1", out)
        self.assertNotIn("buffered", out.splitlines()[0])
        # A fresh .say works afterwards.
        _one(shell, buf, ".say")
        _one(shell, buf, "user: ok")
        _, out = _one(shell, buf, ".")
        self.assertIn("buffered 1 message(s)", out)

    def test_say_exactly_1mb_accepted(self) -> None:
        # Minimal pair: exactly 1_000_000 bytes is accepted (> is the trigger).
        shell, buf = _make_shell()
        _one(shell, buf, ".say")
        _, out = _one(shell, buf, "x" * 1_000_000)
        self.assertNotIn("error:", out)
        _, out = _one(shell, buf, ".")
        self.assertIn("buffered 1 message(s)", out)

    def test_say_line_cap_error_keeps_buffer_then_finalizes(self) -> None:
        shell, buf = _make_shell()
        _one(shell, buf, ".say")
        for i in range(10_000):
            stop, out = _one(shell, buf, f"line{i}")
            self.assertFalse(stop)
            self.assertNotIn("error:", out)
        _, out = _one(shell, buf, "one-too-many")
        self.assertIn("error: message buffer full (max 10000 lines); end with a lone '.'", out)
        # The 10000 buffered lines survived the failed append.
        self.assertEqual(len(shell._say_buffer or []), 10_000)
        _, out = _one(shell, buf, ".")
        self.assertIn("buffered 10000 message(s)", out)

    def test_error_inside_read_script_then_session_continues(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "s.txt"
            script.write_text(".boundary nope\n.fixtures\n", encoding="utf-8")
            stop, _ = _one(shell, buf, f".read {script}")
            self.assertFalse(stop)
        _, out = _one(shell, buf, ".step 1")
        self.assertIn("error: no boundary set", out)
        _, out = _one(shell, buf, ".quit")
        self.assertFalse(out)  # .quit prints nothing
        # Session still fully functional after the scripted error.
        shell2, buf2 = _make_shell()
        _, out = _one(shell2, buf2, ".families")
        self.assertIn("chatml", out)

    def test_read_missing_then_valid_script(self) -> None:
        shell, buf = _make_shell()
        _, out = _one(shell, buf, ".read /tmp/definitely_missing_xyz_123")
        self.assertIn("error: unreadable script file", out)
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "ok.txt"
            script.write_text(".families\n", encoding="utf-8")
            _, out = _one(shell, buf, f".read {script}")
        self.assertIn("chatml", out)


class TestTranscriptBoundaries(unittest.TestCase):
    """The transcript trim: marker first, newest output kept, total capped."""

    def _overflow(self, shell: DebugShell, buf: io.StringIO, cap: int, renders: int = 15) -> None:
        with mock.patch.object(debug_module, "_MAX_SAVE_BYTES", cap):
            for _ in range(renders):
                _one(shell, buf, ".render")

    def test_truncation_invariants_with_small_cap(self) -> None:
        shell, buf = _make_shell()
        self._overflow(shell, buf, 2000)
        chunks = shell._transcript
        self.assertTrue(shell._transcript_truncated)
        self.assertEqual(chunks[0], "[... transcript truncated ...]\n")
        total = sum(len(chunk) for chunk in chunks)
        self.assertLessEqual(total, 2000)
        # The newest command's output survived the trim.
        with mock.patch.object(debug_module, "_MAX_SAVE_BYTES", 2000):
            _one(shell, buf, ".use llama3")
        self.assertTrue(_transcript(shell).endswith("using family llama3\n"))

    def test_no_truncation_below_cap(self) -> None:
        shell, buf = _make_shell()
        _one(shell, buf, ".fixtures")
        self.assertFalse(shell._transcript_truncated)
        self.assertNotIn("truncated", _transcript(shell))

    def test_truncated_transcript_saves_marker_first_and_fits(self) -> None:
        shell, buf = _make_shell()
        self._overflow(shell, buf, 2000)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "t.txt"
            with mock.patch.object(debug_module, "_MAX_SAVE_BYTES", 2000):
                _, out = _one(shell, buf, f".save {target}")
            content = target.read_text(encoding="utf-8")
            self.assertTrue(content.startswith("[... transcript truncated ...]\n"))
            self.assertLessEqual(len(content), 2000)
            self.assertIn("saved ", out)

    def test_format_ids_exactly_512_not_capped(self) -> None:
        shell, buf = _make_shell(tokenizer=CountTokenizer(512))
        _, out = _one(shell, buf, ".tokens")
        ids_line = out.splitlines()[1]
        self.assertEqual(ids_line, "ids=[" + ", ".join(str(i) for i in range(512)) + "]")
        self.assertNotIn("...", ids_line)

    def test_format_ids_513_capped_with_exact_total(self) -> None:
        shell, buf = _make_shell(tokenizer=CountTokenizer(513))
        _, out = _one(shell, buf, ".tokens")
        ids_line = out.splitlines()[1]
        self.assertEqual(
            ids_line,
            "ids=[" + ", ".join(str(i) for i in range(512)) + ", ... (513 total)]",
        )

    def test_transcript_echo_order_matches_command_order(self) -> None:
        shell, buf = _make_shell()
        _one(shell, buf, ".families")
        _one(shell, buf, ".fixtures")
        transcript = _transcript(shell)
        families_at = transcript.find("mpl:chatml fixture-1 > .families\n")
        fixtures_at = transcript.find("mpl:chatml fixture-1 > .fixtures\n")
        self.assertGreater(families_at, -1)
        self.assertGreater(fixtures_at, families_at)


class TestReadScriptEdges(unittest.TestCase):
    def _write(self, tmp: str, name: str, text: str) -> Path:
        path = Path(tmp) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_read_script_with_say_block(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            script = self._write(
                tmp, "s.txt", ".say\nuser: hello\nassistant: hi\nuser: again\n.\n.render\n"
            )
            stop, _ = _one(shell, buf, f".read {script}")
            self.assertFalse(stop)
        self.assertIn("buffered", buf.getvalue())
        self.assertIn("messages=3", buf.getvalue())

    def test_read_script_crlf_line_endings(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            script = self._write(tmp, "s.txt", ".say\r\nuser: hi\r\n.\r\n")
            stop, out = _one(shell, buf, f".read {script}")
            self.assertFalse(stop)
            self.assertIn("buffered 1 message(s)", out)

    def test_read_empty_script(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            script = self._write(tmp, "s.txt", "")
            self.assertFalse(shell.read_commands(script))

    def test_read_tilde_expands(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home.mkdir()
            (home / "cmds.txt").write_text(".families\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"HOME": str(home)}):
                _, out = _one(shell, buf, ".read ~/cmds.txt")
        self.assertIn("chatml", out)

    def test_read_depth_restored_after_nested_read(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            inner = self._write(tmp, "inner.txt", ".families\n")
            outer = self._write(tmp, "outer.txt", f".read {inner}\n")
            _, out = _one(shell, buf, f".read {outer}")
            self.assertIn("chatml", out)
        self.assertEqual(shell._read_depth, 0)
        # And again: the counter did not leak.
        with tempfile.TemporaryDirectory() as tmp:
            script = self._write(tmp, "s.txt", ".fixtures\n")
            _, out = _one(shell, buf, f".read {script}")
            self.assertIn("s1", out)

    def test_read_depth_restored_after_failed_nested_read(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            # Self-recursive: terminates at the depth cap with an error.
            script = self._write(tmp, "loop.txt", ".read {p}\n".format(p=Path(tmp) / "loop.txt"))
            _, out = _one(shell, buf, f".read {script}")
            self.assertIn("error: nested .read too deep", out)
        self.assertEqual(shell._read_depth, 0)

    def test_script_ending_mid_say_abandons_buffer_without_crash(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            script = self._write(tmp, "s.txt", ".say\nuser: hi\n")
            self.assertEqual(run_script(shell, script), 0)
        # Unterminated entry leaves the partial buffer; nothing exploded.
        self.assertEqual(shell._say_buffer, ["user: hi"])


def _write_wordlevel_tokenizer_json(path: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel(vocab={"hello": 0, "world": 1, "[UNK]": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(path))


def _write_dataset(tmp: str) -> Path:
    path = Path(tmp) / "ds.json"
    path.write_text(split_to_json_text(_make_split()), encoding="utf-8")
    return path


def _namespace(**kwargs: Any) -> argparse.Namespace:
    base = {"dataset": None, "model": None, "script": None}
    base.update(kwargs)
    return argparse.Namespace(**base)


@unittest.skipIf(not _HAS_TRANSFORMERS, "transformers not installed")
class TestFailClosedBackends(unittest.TestCase):
    """--model is tokenizer-only: synthetic backends, garbage weights."""

    def test_attach_tokenizer_missing_dir_raises_valueerror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            attach_tokenizer(str(Path(tmp) / "nope"))

    def test_attach_tokenizer_file_path_raises_valueerror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "file.bin"
            target.write_bytes(b"junk")
            with self.assertRaises(ValueError):
                attach_tokenizer(target)

    def test_attach_tokenizer_never_reads_weight_files(self) -> None:
        # The checkpoint dir carries garbage where weights would live. If
        # the loader touched anything but tokenizer.json, it would fail.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_wordlevel_tokenizer_json(root / "tokenizer.json")
            (root / "model.safetensors").write_bytes(b"this is not a safetensors file")
            (root / "config.json").write_text("not json at all", encoding="utf-8")
            tokenizer = attach_tokenizer(root)
            self.assertEqual(
                list(tokenizer.encode("hello world", add_special_tokens=False)), [0, 1]
            )

    def test_attach_tokenizer_import_error_before_filesystem(self) -> None:
        # With transformers unavailable, the ImportError surfaces before any
        # checkpoint parsing -- even for a perfectly valid directory.
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(sys.modules, {"transformers": None}),
            self.assertRaises(ImportError),
        ):
            attach_tokenizer(tmp)

    def test_run_debug_oversized_tokenizer_json_collapses_to_fixed_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _write_dataset(tmp)
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            with open(ckpt / "tokenizer.json", "wb") as handle:
                handle.truncate(128 * 1024 * 1024 + 1)  # sparse: no real I/O
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                result = run_debug(_namespace(dataset=dataset, model=ckpt))
            self.assertEqual(result, 1)
            self.assertIn("debug: oversized_tokenizer_file", err.getvalue())

    def test_run_debug_bad_model_collapses_and_never_echoes_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _write_dataset(tmp)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                result = run_debug(_namespace(dataset=dataset, model="/nonexistent_ckpt_xyz"))
            self.assertEqual(result, 1)
            self.assertIn("debug: tokenizer_load_failed", err.getvalue())
            self.assertNotIn("nonexistent_ckpt_xyz", err.getvalue())

    def test_shell_with_real_attached_tokenizer_renders_exact_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_wordlevel_tokenizer_json(root / "tokenizer.json")
            tokenizer = attach_tokenizer(root)
            shell, buf = _make_shell(tokenizer=tokenizer)
            _one(shell, buf, ".say")
            _one(shell, buf, "user: hello world")
            _one(shell, buf, ".")
            _, out = _one(shell, buf, ".tokens")
        rendered = CATALOG["chatml"].render((Message("user", "hello world"),))
        expected_ids = list(tokenizer.encode(rendered, add_special_tokens=False))
        self.assertIn(f"count={len(expected_ids)}", out)
        self.assertIn(f"ids={expected_ids}", out)


class TestCliWiring(unittest.TestCase):
    """cli.py wires `mpl debug` to run_debug with dataset/model/script."""

    def test_debug_parser_flags(self) -> None:
        args = build_parser().parse_args(["debug", "--dataset", "ds.json"])
        self.assertEqual(args.command, "debug")
        self.assertEqual(args.dataset, Path("ds.json"))
        self.assertIsNone(args.model)
        self.assertIsNone(args.script)

    def test_debug_parser_model_and_script_flags(self) -> None:
        args = build_parser().parse_args(
            ["debug", "--dataset", "ds.json", "--model", "ckpt", "--script", "s.txt"]
        )
        self.assertEqual(args.model, Path("ckpt"))
        self.assertEqual(args.script, Path("s.txt"))

    def test_debug_dataset_is_required(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["debug"])

    def test_main_dispatches_debug_to_run_debug(self) -> None:
        from model_protocol_lab import cli as cli_module

        with tempfile.TemporaryDirectory() as tmp:
            dataset = _write_dataset(tmp)
            with mock.patch("model_protocol_lab.debug.run_debug", return_value=0) as mocked:
                result = cli_module.main(["debug", "--dataset", str(dataset)])
        self.assertEqual(result, 0)
        mocked.assert_called_once()
        namespace = mocked.call_args[0][0]
        self.assertEqual(namespace.command, "debug")
        self.assertEqual(namespace.dataset, dataset)

    def test_version_banner_matches_package(self) -> None:
        shell, _ = _make_shell(tokenizer="none")
        self.assertTrue((shell.intro or "").startswith(f"model-protocol-lab debug {__version__}\n"))


if __name__ == "__main__":
    unittest.main()
