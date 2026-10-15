"""Cycle 36: behavioral verification of the ``mpl debug`` REPL command loop.

Drives :class:`model_protocol_lab.debug.DebugShell` programmatically (queued
commands through ``cmdloop()`` and single ``onecmd()`` calls) with a
deterministic char-level fake tokenizer. No model weights, no network, no
randomness. The pure helpers (``_sanitize_display``, ``_parse_id_list``,
``_overhang``, ``_render_segments``, ``_parse_say_line``) were verified in
cycle 23; this file pins the REPL itself: command parsing/dispatch,
abbreviation resolution, every dot-command's behavior with stubbed state,
error paths, output formatting, exit handling, transcript behavior, and
hostile inputs.

Security posture (verified, not a finding): the REPL intentionally shows
operator/fixture content by design (docs/USAGE.md section 10 says so, and
warns against pasting ``.save`` transcripts into public issues). Every
surface that echoes fixture or operator text (``.render``, ``.fixtures``,
``.tokens``, ``.audit``, transcript echoes, ``.save``) is inside that
documented contract; control characters are stripped before printing, raw
exception text is sanitized at the REPL boundary, and the session writes
nothing to disk except through the explicit ``.save`` command (verified by
``TestNoHistoryNoNetwork``).

Doc/code discrepancies found (source behavior NOT changed; documented here):

1. docs/USAGE.md section 10's command table says ``.families`` lists "the
   16 catalog families", but the catalog now holds 28 families (it grew in
   later cycles; the section-10 text was never updated). The REPL itself
   correctly lists all 28.
2. ``.help`` with an *ambiguous* abbreviation (e.g. ``.help .fi``) falls
   through to the PATTERN search and lists the matches instead of reporting
   the ambiguity. sqlite3-flavored and harmless, but the docstring's
   "unambiguous abbreviations allowed" phrasing does not mention the
   fallback; pinned as-is.
3. ``.say`` role alternation: the alternation counter advances only for
   lines *without* a recognized ``user:``/``assistant:`` prefix, so
   ``user: x`` followed by a plain ``y`` yields two ``user`` messages. The
   USAGE.md wording ("otherwise roles alternate starting with user") matches
   this reading, but operators may expect the counter to advance; pinned
   as-is to prevent silent drift.
4. Only the exact uppercase line ``EOF`` exits the loop; lowercase ``eof``
   is an unknown command. USAGE.md does not document ``EOF`` at all;
   behavior pinned.
5. ``.save``'s "saved N chars" counts command echoes plus output, exactly
   as the USAGE.md row describes ("Append the session transcript (command
   echoes plus output) to a file"). The file holds the transcript as it
   stood when ``.save`` ran -- the ``.save`` line's own echo and output land
   in the transcript only afterwards.

Bugs found (item 1 FIXED in cycle 36; reported to the parent):

1. **FIXED in cycle 36: truncated transcripts could never be saved.**
   ``onecmd`` used to truncate the transcript to ``_MAX_SAVE_BYTES`` of
   content and then *prepend* the 29-char ``[... transcript truncated
   ...]`` marker, so the total was ``_MAX_SAVE_BYTES + 29``. ``_cmd_save``
   then refused with "transcript too large to save (over 5MB)". The fix
   budgets the marker inside the truncation (``remaining =
   _MAX_SAVE_BYTES - len(marker)``), so truncated transcripts still fit
   and ``.save`` succeeds. Pinned by
   ``TestTranscript.test_truncated_transcript_save_succeeds``.
"""

import argparse
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest import mock

import model_protocol_lab.debug as debug_module
from model_protocol_lab import __version__
from model_protocol_lab.catalog import CATALOG
from model_protocol_lab.core import Fixture, Message, Split
from model_protocol_lab.debug import DebugShell, attach_tokenizer, run_debug, run_script
from model_protocol_lab.fixtures import split_to_json_text


class FakeTokenizer:
    """Deterministic char-level tokenizer shim (no weights, no network)."""

    def encode(self, text, add_special_tokens=False):
        assert isinstance(text, str)
        return tuple(ord(char) for char in text)

    def decode(self, token_ids):
        assert isinstance(token_ids, (list, tuple))
        return "".join(chr(token_id) for token_id in token_ids)


class ExplodingDecodeTokenizer(FakeTokenizer):
    """decode() always fails: the stepper must degrade, never die."""

    def decode(self, token_ids):
        raise RuntimeError("decode exploded")


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


def _make_shell(split=None, tokenizer="fake", **kwargs) -> tuple[DebugShell, io.StringIO]:
    """Fresh shell writing to a buffer; never touches real stdin/stdout."""
    buf = io.StringIO()
    shell = DebugShell(
        split if split is not None else _make_split(),
        dataset_label="test",
        tokenizer=FakeTokenizer() if tokenizer == "fake" else tokenizer,
        stdin=io.StringIO(""),
        stdout=buf,
        **kwargs,
    )
    return shell, buf


def _run(shell: DebugShell, buf: io.StringIO, *commands: str) -> str:
    """Feed queued commands through cmdloop(); return everything printed."""
    buf.seek(0)
    buf.truncate(0)
    shell.cmdqueue.extend(commands)
    shell.cmdloop()
    return buf.getvalue()


def _tail_lines(output: str, skip: int = 3) -> list[str]:
    """Drop the 3-line startup banner; return the session's own lines."""
    return output.splitlines()[skip:]


class TestShellConstruction(unittest.TestCase):
    def test_banner_names_version_dataset_family_tokenizer(self) -> None:
        shell, buf = _make_shell(tokenizer=None)
        _run(shell, buf, ".quit")
        banner = buf.getvalue().splitlines()[:3]
        self.assertEqual(
            banner,
            [
                f"model-protocol-lab debug {__version__}",
                "dataset: test (3 fixtures) | family: chatml | "
                "tokenizer: none (rerun with --model PATH)",
                "Enter .help for usage hints.",
            ],
        )

    def test_banner_tokenizer_attached_state(self) -> None:
        attached, _ = _make_shell()  # default fake tokenizer
        self.assertIn("tokenizer: attached", attached.intro)
        none_shell, _ = _make_shell(tokenizer=None)
        self.assertIn("tokenizer: none (rerun with --model PATH)", none_shell.intro)

    def test_initial_prompt_shows_family_and_fixture_number(self) -> None:
        shell, _ = _make_shell()
        self.assertEqual(shell.prompt, "mpl:chatml fixture-1 > ")

    def test_initial_family_override(self) -> None:
        shell, _ = _make_shell(family="llama3")
        self.assertEqual(shell.prompt, "mpl:llama3 fixture-1 > ")
        self.assertIn("family: llama3", shell.intro)

    def test_rejects_non_split(self) -> None:
        with self.assertRaises(ValueError):
            DebugShell("not-a-split")  # type: ignore[arg-type]

    def test_rejects_unknown_family(self) -> None:
        with self.assertRaises(ValueError):
            _make_shell(family="nosuchfamily")

    def test_never_blocks_on_real_stdin(self) -> None:
        shell, _ = _make_shell()
        self.assertFalse(shell.use_rawinput)


class TestCommandParsing(unittest.TestCase):
    def test_bare_word_is_unknown_command(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, "hello world", ".quit")
        self.assertIn(
            "error: unknown command 'hello'; dot-commands start with '.' (try .help)",
            out,
        )

    def test_unknown_dot_command(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".nosuch", ".quit")
        self.assertIn("error: unknown dot-command '.nosuch' (try .help)", out)

    def test_ambiguous_abbreviation_lists_matches(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".fi", ".quit")
        self.assertIn("error: ambiguous command '.fi'; matches: .fixture, .fixtures", out)

    def test_lone_dot_is_ambiguous_over_everything(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".", ".quit")
        self.assertIn("error: ambiguous command '.'; matches:", out)
        for name in (".help", ".quit", ".render", ".step"):
            self.assertIn(name, out)

    def test_unique_abbreviation_resolves(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".h", ".quit")
        self.assertIn(".audit", out)  # full .help listing, not an error
        self.assertNotIn("error:", out)

    def test_exact_match_beats_prefix(self) -> None:
        # ".q" is an exact command even though ".quit" also starts with ".q".
        shell, buf = _make_shell()
        self.assertTrue(shell.onecmd(".q"))

    def test_qu_abbreviation_quits(self) -> None:
        shell, buf = _make_shell()
        self.assertTrue(shell.onecmd(".qu"))

    def test_double_dot_is_unknown(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, "..help", ".quit")
        self.assertIn("error: unknown dot-command '..help'", out)

    def test_hash_comment_is_silent(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, "# just a comment", "   # indented", ".quit")
        self.assertNotIn("error", out)
        self.assertNotIn("just a comment", out)

    def test_blank_lines_do_nothing_and_leave_no_transcript(self) -> None:
        shell, buf = _make_shell()
        before = list(shell._transcript)
        self.assertFalse(shell.onecmd(""))
        self.assertFalse(shell.onecmd("   "))
        self.assertEqual(shell._transcript, before)
        self.assertEqual(buf.getvalue(), "")

    def test_leading_whitespace_before_dot_command(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, "   .families", ".quit")
        self.assertIn("chatml", out)
        self.assertNotIn("error", out)

    def test_eof_line_quits(self) -> None:
        shell, _ = _make_shell()
        self.assertTrue(shell.onecmd("EOF"))

    def test_lowercase_eof_is_unknown_command(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, "eof", ".quit")
        self.assertIn("error: unknown command 'eof'", out)

    def test_dot_commands_unavailable_inside_say(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".say", ".render", ".", ".quit")
        # ".render" was swallowed as message content, not executed.
        self.assertIn("buffered 1 message(s)", out)
        self.assertNotIn("render: family=", out)

    def test_command_failure_never_kills_session(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".nosuch", ".use nosuch", ".fixtures", ".quit")
        self.assertIn("error:", out)
        self.assertIn("1: s1", out)


class TestHelp(unittest.TestCase):
    def test_help_lists_all_commands_sorted(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".help", ".quit")
        lines = [line for line in _tail_lines(out) if line.startswith(".")]
        names = [line.split()[0] for line in lines]
        self.assertEqual(names, sorted(shell._commands))
        self.assertEqual(len(names), 16)
        for name in shell._commands:
            self.assertIn(name, out)

    def test_help_exact_command_shows_docstring(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".help .render", ".quit")
        self.assertIn(".render: Render the hypothesis over messages + segment map", out)
        self.assertIn("Usage: .render", out)

    def test_help_without_dot_prefix(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".help step", ".quit")
        self.assertIn(".step: Step boundary detection one token at a time", out)
        self.assertIn("Usage: .step", out)

    def test_help_abbreviation(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".help ren", ".quit")
        self.assertIn(".render: Render the hypothesis over messages + segment map", out)

    def test_help_pattern_search_matches_names(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".help oundar", ".quit")
        self.assertIn("matching commands:", out)
        self.assertIn(".boundary", out)
        self.assertIn(".step", out)

    def test_help_pattern_search_matches_summaries(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".help transcript", ".quit")
        self.assertIn("matching commands:", out)
        self.assertIn(".save", out)

    def test_help_no_match_is_error(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".help nosuchxyz", ".quit")
        self.assertIn("error: unknown dot-command '.nosuchxyz' (try .help)", out)

    def test_help_ambiguous_abbreviation_falls_back_to_pattern_search(self) -> None:
        # Documented nuance (docstring item 2): no ambiguity error here.
        shell, buf = _make_shell()
        out = _run(shell, buf, ".help .fi", ".quit")
        self.assertIn("matching commands:", out)
        self.assertIn(".fixture", out)
        self.assertIn(".fixtures", out)
        self.assertNotIn("error:", out)


class TestUseAndFamilies(unittest.TestCase):
    def test_families_lists_every_catalog_family(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".families", ".quit")
        for name in CATALOG:
            self.assertIn(name, out)

    def test_families_count_matches_catalog(self) -> None:
        # USAGE.md section 10 still says "16 catalog families"; the catalog
        # has grown to 28 (see module docstring discrepancy 1).
        self.assertEqual(len(CATALOG), 28)
        shell, buf = _make_shell()
        out = _run(shell, buf, ".families", ".quit")
        self.assertEqual(len(_tail_lines(out)), 28)

    def test_families_rejects_arguments(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".families extra", ".quit")
        self.assertIn("error: usage: .families (takes no arguments)", out)

    def test_use_switches_family_prompt_and_render(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".use llama3", ".render", ".quit")
        self.assertIn("using family llama3", out)
        self.assertEqual(shell.prompt, "mpl:llama3 fixture-1 > ")
        self.assertIn("render: family=llama3", out)

    def test_use_rejects_unknown_family(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".use nosuchfamily", ".quit")
        self.assertIn("error: unknown family 'nosuchfamily'", out)
        self.assertIn("valid families:", out)
        self.assertIn("chatml", out)
        self.assertEqual(shell._family, "chatml")

    def test_use_requires_argument(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".use", ".quit")
        self.assertIn("error: usage: .use <family> (see .families)", out)


class TestFixtures(unittest.TestCase):
    def test_fixtures_lists_ids_with_current_marked(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".fixtures", ".quit")
        self.assertIn("1: s1 *", out)
        self.assertIn("2: s2", out)
        self.assertIn("3: h1", out)

    def test_fixture_select_by_number_moves_marker(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".fixture 2", ".fixtures", ".quit")
        self.assertIn("selected fixture 2: s2", out)
        self.assertIn("2: s2 *", out)
        self.assertNotIn("1: s1 *", out)
        self.assertEqual(shell.prompt, "mpl:chatml fixture-2 > ")

    def test_fixture_select_by_id(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".fixture h1", ".quit")
        self.assertIn("selected fixture 3: h1", out)

    def test_fixture_numeric_argument_tried_as_number_first(self) -> None:
        # Fixture id "1" exists as the *second* fixture; ".fixture 1" must
        # still select fixture number 1, per the documented precedence.
        split = Split(
            search=(Fixture(id="a", group="g1", messages=(Message("user", "x"),), expected="y"),),
            held_out=(Fixture(id="1", group="g2", messages=(Message("user", "z"),), expected="y"),),
        )
        shell, buf = _make_shell(split=split)
        out = _run(shell, buf, ".fixture 1", ".quit")
        self.assertIn("selected fixture 1: a", out)

    def test_fixture_falls_back_to_numeric_id(self) -> None:
        split = Split(
            search=(Fixture(id="a", group="g1", messages=(Message("user", "x"),), expected="y"),),
            held_out=(
                Fixture(id="42", group="g2", messages=(Message("user", "z"),), expected="y"),
            ),
        )
        shell, buf = _make_shell(split=split)
        out = _run(shell, buf, ".fixture 42", ".quit")
        self.assertIn("selected fixture 2: 42", out)

    def test_fixture_rejects_unknown(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".fixture 99", ".fixture nope", ".quit")
        self.assertIn("error: unknown fixture '99' (see .fixtures)", out)
        self.assertIn("error: unknown fixture 'nope' (see .fixtures)", out)

    def test_fixture_requires_argument(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".fixture", ".quit")
        self.assertIn("error: usage: .fixture <n|id> (see .fixtures)", out)

    def test_fixtures_rejects_arguments(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".fixtures extra", ".quit")
        self.assertIn("error: usage: .fixtures (takes no arguments)", out)


class TestRender(unittest.TestCase):
    def test_render_header_and_verbatim_body(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".render", ".quit")
        fmt = CATALOG["chatml"]
        rendered = fmt.render((Message("user", "hello"),))
        self.assertIn(f"render: family=chatml fixture=s1 messages=1 chars={len(rendered)}", out)
        self.assertIn(rendered, out)

    def test_render_segment_spans_cover_render(self) -> None:
        import re

        shell, buf = _make_shell()
        out = _run(shell, buf, ".render", ".quit")
        fmt = CATALOG["chatml"]
        rendered = fmt.render((Message("user", "hello"),))
        spans = re.findall(r"\[(\d+)\] (\w+) +chars (\d+):(\d+)", out)
        self.assertEqual(
            [kind for _, kind, _, _ in spans],
            ["bos", "user_prefix", "user_content", "user_turn_end", "open_suffix"],
        )
        offset = 0
        for _, _, start, end in spans:
            self.assertEqual(int(start), offset)
            offset = int(end)
        self.assertEqual(offset, len(rendered))

    def test_render_per_segment_token_ids(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".render", ".quit")
        content_line = next(line for line in out.splitlines() if "user_content" in line)
        self.assertIn("tokens=[104, 101, 108, 108, 111]", content_line)

    def test_render_without_tokenizer_omits_ids(self) -> None:
        import re

        shell, buf = _make_shell(tokenizer=None)
        out = _run(shell, buf, ".render", ".quit")
        segment_lines = [line for line in out.splitlines() if re.match(r"  \[\d+\]", line)]
        self.assertTrue(segment_lines)
        for line in segment_lines:
            self.assertNotIn("tokens=", line)

    def test_render_uses_say_buffer_with_buffered_label(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".say", "hi there", ".", ".render", ".quit")
        self.assertIn("render: family=chatml buffered messages=1", out)

    def test_render_rejects_arguments(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".render extra", ".quit")
        self.assertIn("error: usage: .render (takes no arguments)", out)

    def test_render_hostile_content_is_sanitized(self) -> None:
        split = Split(
            search=(
                Fixture(
                    id="s1",
                    group="g1",
                    messages=(Message("user", "a\x1b[31mb\x07c"),),
                    expected="x",
                ),
            ),
            held_out=(
                Fixture(id="h1", group="g2", messages=(Message("user", "q"),), expected="y"),
            ),
        )
        shell, buf = _make_shell(split=split)
        out = _run(shell, buf, ".render", ".quit")
        self.assertNotIn("\x1b", out)
        self.assertNotIn("\x07", out)
        self.assertIn("a[31mbc", out)


class TestTokens(unittest.TestCase):
    def test_tokens_exact_ids(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".tokens", ".quit")
        fmt = CATALOG["chatml"]
        expected = tuple(ord(char) for char in fmt.render((Message("user", "hello"),)))
        self.assertIn(f"tokens: family=chatml fixture=s1 count={len(expected)}", out)
        self.assertIn(f"ids=[{', '.join(str(i) for i in expected)}]", out)

    def test_tokens_without_tokenizer_prints_hint(self) -> None:
        shell, buf = _make_shell(tokenizer=None)
        shell.onecmd(".tokens")
        self.assertIn(
            "no tokenizer attached; rerun with --model PATH for exact token IDs.",
            buf.getvalue(),
        )

    def test_tokens_rejects_arguments(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".tokens extra", ".quit")
        self.assertIn("error: usage: .tokens (takes no arguments)", out)

    def test_tokens_display_cap_keeps_exact_count(self) -> None:
        split = Split(
            search=(
                Fixture(
                    id="s1",
                    group="g1",
                    messages=(Message("user", "x" * 600),),
                    expected="y",
                ),
            ),
            held_out=(
                Fixture(id="h1", group="g2", messages=(Message("user", "q"),), expected="y"),
            ),
        )
        shell, buf = _make_shell(split=split)
        out = _run(shell, buf, ".tokens", ".quit")
        fmt = CATALOG["chatml"]
        total = len(fmt.render((Message("user", "x" * 600),)))
        self.assertGreater(total, 512)
        self.assertIn(f"count={total}", out)
        self.assertIn(f"... ({total} total)]", out)


class TestBoundaryStreamStep(unittest.TestCase):
    def _walked(self, *commands: str) -> str:
        shell, buf = _make_shell(tokenizer=None)
        return _run(shell, buf, *commands)

    def test_boundary_grammar_matrix(self) -> None:
        cases = {
            "": "usage: .boundary ID,ID,...",
            "1,2,abc": "expected 1-16 comma-separated ASCII token IDs",
            "1,,2": "expected 1-16 comma-separated ASCII token IDs",
            ",": "expected 1-16 comma-separated ASCII token IDs",
            "-5": "expected 1-16 comma-separated ASCII token IDs",
            "12٣": "expected 1-16 comma-separated ASCII token IDs",
            "2147483648": "token IDs must be < 2^31",
            ",".join(["1"] * 17): "expected 1-16 comma-separated ASCII token IDs",
        }
        for arg, message in cases.items():
            with self.subTest(arg=arg):
                shell, buf = _make_shell()
                out = _run(shell, buf, f".boundary {arg}".rstrip(), ".quit")
                prefix = "" if message.startswith("usage:") else "boundary: "
                self.assertIn(f"error: {prefix}{message}", out)

    def test_boundary_accepts_edges(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".boundary 0", ".quit")
        self.assertIn("boundary set: [0]", out)
        shell, buf = _make_shell()
        out = _run(shell, buf, ".boundary 007", ".quit")
        self.assertIn("boundary set: [7]", out)
        shell, buf = _make_shell()
        out = _run(shell, buf, ".boundary 2147483647", ".quit")
        self.assertIn("boundary set: [2147483647]", out)
        shell, buf = _make_shell()
        out = _run(shell, buf, ".boundary " + ",".join(str(i) for i in range(16)), ".quit")
        self.assertIn("boundary set: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]", out)
        shell, buf = _make_shell()
        out = _run(shell, buf, ".boundary 1, 2", ".quit")
        self.assertIn("boundary set: [1, 2]", out)

    def test_stream_grammar_and_bounds(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".stream", ".quit")
        self.assertIn("error: usage: .stream ID,ID,...", out)
        shell, buf = _make_shell()
        out = _run(shell, buf, ".stream 1,zz", ".quit")
        self.assertIn("error: stream: expected 1-4096 comma-separated ASCII token IDs", out)
        shell, buf = _make_shell()
        out = _run(shell, buf, ".stream " + ",".join(["9"] * 4096), ".quit")
        self.assertIn("stream set: 4096 tokens; walk it with .step", out)
        shell, buf = _make_shell()
        out = _run(shell, buf, ".stream " + ",".join(["9"] * 4097), ".quit")
        self.assertIn("error: stream: expected 1-4096 comma-separated ASCII token IDs", out)

    def test_step_requires_boundary_first(self) -> None:
        out = self._walked(".step", ".quit")
        self.assertIn("error: no boundary set; use .boundary ID,ID,... first", out)

    def test_step_requires_stream_or_typed_ids(self) -> None:
        out = self._walked(".boundary 7", ".step", ".quit")
        self.assertIn("error: no recorded stream; use .stream ID,ID,... or .step ID,ID,...", out)

    def test_step_exhaustion(self) -> None:
        out = self._walked(".boundary 7", ".stream 1,2", ".step", ".step", ".step", ".quit")
        self.assertIn("token 1/2 id=1 overhang=0/1 boundary_at=none", out)
        self.assertIn("token 2/2 id=2 overhang=0/1 boundary_at=none", out)
        self.assertIn("error: stream exhausted (2 tokens); use .stream to load a new one", out)

    def test_step_walks_recorded_stream_with_overhang(self) -> None:
        out = self._walked(
            ".boundary 7,8",
            ".stream 1,2,7,8,9",
            ".step",
            ".step",
            ".step",
            ".step",
            ".quit",
        )
        self.assertIn("token 1/5 id=1 overhang=0/2 boundary_at=none", out)
        self.assertIn("token 2/5 id=2 overhang=0/2 boundary_at=none", out)
        self.assertIn("token 3/5 id=7 overhang=1/2 boundary_at=none", out)
        self.assertIn("token 4/5 id=8 overhang=2/2 boundary_at=2", out)

    def test_step_typed_stream_leaves_recorded_walk_alone(self) -> None:
        out = self._walked(".boundary 7", ".stream 1,2,3", ".step", ".step 5,6", ".step", ".quit")
        lines = [line for line in out.splitlines() if line.startswith("token")]
        self.assertEqual(
            lines,
            [
                "token 1/3 id=1 overhang=0/1 boundary_at=none",
                "token 1/2 id=5 overhang=0/1 boundary_at=none",
                "token 2/2 id=6 overhang=0/1 boundary_at=none",
                "token 2/3 id=2 overhang=0/1 boundary_at=none",
            ],
        )

    def test_stream_resets_walk(self) -> None:
        out = self._walked(".boundary 7", ".stream 1,2,3", ".step", ".stream 9,8", ".step", ".quit")
        lines = [line for line in out.splitlines() if line.startswith("token")]
        self.assertEqual(
            lines,
            [
                "token 1/3 id=1 overhang=0/1 boundary_at=none",
                "token 1/2 id=9 overhang=0/1 boundary_at=none",
            ],
        )

    def test_step_typed_bad_ids(self) -> None:
        out = self._walked(".boundary 7", ".step 1,nope", ".quit")
        self.assertIn("error: step: expected 1-4096 comma-separated ASCII token IDs", out)

    def test_step_shows_decoded_text_when_tokenizer_attached(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".boundary 104", ".step 104", ".quit")
        self.assertIn("token 1/1 id=104 text='h' overhang=1/1 boundary_at=0", out)

    def test_step_decode_failure_degrades_to_no_text(self) -> None:
        shell, buf = _make_shell(tokenizer=ExplodingDecodeTokenizer())
        out = _run(shell, buf, ".boundary 1", ".step 2", ".quit")
        self.assertIn("token 1/1 id=2 overhang=0/1 boundary_at=none", out)
        self.assertNotIn("text=", out)


class TestSay(unittest.TestCase):
    def test_say_enters_continuation_prompt(self) -> None:
        shell, buf = _make_shell()
        shell.onecmd(".say")
        self.assertEqual(shell.prompt, "...> ")
        self.assertIn("enter messages (one per line); end with a lone '.'", buf.getvalue())

    def test_say_rejects_arguments(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".say extra", ".quit")
        self.assertIn("error: usage: .say (takes no arguments)", out)

    def test_say_rejects_nested_entry(self) -> None:
        # Via normal dispatch ".say" inside entry is message content (pinned
        # in test_dot_commands_unavailable_inside_say); the "already
        # entering" guard is defensive and reachable only by direct call.
        shell, _ = _make_shell()
        shell.onecmd(".say")
        with self.assertRaises(ValueError) as ctx:
            shell._cmd_say("")
        self.assertIn("already entering messages", str(ctx.exception))

    def test_say_empty_buffer_discards(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".say", ".", ".render", ".quit")
        self.assertIn("message buffer discarded (empty)", out)
        self.assertIn("fixture=s1", out)  # falls back to the fixture
        self.assertEqual(shell.prompt, "mpl:chatml fixture-1 > ")

    def test_say_terminator_allows_surrounding_whitespace(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".say", "hello", "  .  ", ".quit")
        self.assertIn("buffered 1 message(s)", out)

    def test_say_role_alternation_starts_with_user(self) -> None:
        shell, buf = _make_shell()
        _run(shell, buf, ".say", "a", "b", "c", ".", ".quit")
        self.assertEqual(
            [(m.role, m.content) for m in shell._buffered],
            [("user", "a"), ("assistant", "b"), ("user", "c")],
        )

    def test_say_explicit_prefixes(self) -> None:
        shell, buf = _make_shell()
        _run(
            shell,
            buf,
            ".say",
            "user: hi",
            "assistant: hello",
            "User: caps",
            "ASSISTANT: loud",
            "user : spaced",
            ".",
            ".quit",
        )
        self.assertEqual(
            [(m.role, m.content) for m in shell._buffered],
            [
                ("user", "hi"),
                ("assistant", "hello"),
                ("user", "caps"),
                ("assistant", "loud"),
                ("user", "spaced"),
            ],
        )

    def test_say_explicit_prefix_does_not_advance_alternation(self) -> None:
        # Documented nuance (docstring item 3): the counter only moves on
        # lines without a recognized prefix.
        shell, buf = _make_shell()
        _run(shell, buf, ".say", "user: x", "y", ".", ".quit")
        self.assertEqual(
            [(m.role, m.content) for m in shell._buffered],
            [("user", "x"), ("user", "y")],
        )

    def test_say_unrecognized_prefix_kept_as_content(self) -> None:
        shell, buf = _make_shell()
        _run(shell, buf, ".say", "note: keep me", ".", ".quit")
        self.assertEqual(
            [(m.role, m.content) for m in shell._buffered],
            [("user", "note: keep me")],
        )

    def test_say_multiline_content_after_prefix(self) -> None:
        shell, buf = _make_shell()
        # cmdqueue lines are single lines; the DOTALL role regex also handles
        # an embedded newline inside one pasted line.
        _run(shell, buf, ".say", "user: line1\nline2", ".", ".quit")
        self.assertEqual(
            [(m.role, m.content) for m in shell._buffered],
            [("user", "line1\nline2")],
        )

    def test_say_buffer_used_until_next_say(self) -> None:
        shell, buf = _make_shell()
        out = _run(
            shell,
            buf,
            ".say",
            "first",
            ".",
            ".render",
            ".say",
            "second",
            ".",
            ".render",
            ".quit",
        )
        self.assertEqual(out.count("render: family=chatml buffered"), 2)
        self.assertIn("first", out)
        self.assertIn("second", out)

    def test_say_buffer_line_count_bound(self) -> None:
        old = debug_module._MAX_SCRIPT_LINES
        debug_module._MAX_SCRIPT_LINES = 2
        try:
            shell, buf = _make_shell()
            shell.onecmd(".say")
            shell.onecmd("a")
            shell.onecmd("b")
            buf.seek(0)
            buf.truncate(0)
            shell.onecmd("c")
            self.assertIn(
                "error: message buffer full (max 2 lines); end with a lone '.'",
                buf.getvalue(),
            )
            # Entry still active; buffered lines so far are kept.
            self.assertIsNotNone(shell._say_buffer)
            shell.onecmd(".")
            self.assertEqual(len(shell._buffered), 2)
        finally:
            debug_module._MAX_SCRIPT_LINES = old

    def test_say_buffer_byte_bound_discards_and_resets(self) -> None:
        old = debug_module._MAX_SCRIPT_BYTES
        debug_module._MAX_SCRIPT_BYTES = 100
        try:
            shell, buf = _make_shell()
            shell.onecmd(".say")
            shell.onecmd("x" * 60)
            buf.seek(0)
            buf.truncate(0)
            shell.onecmd("y" * 60)
            self.assertIn(
                "error: message buffer full (max 1MB); buffer discarded, "
                "end with a lone '.' to start over",
                buf.getvalue(),
            )
            self.assertIsNone(shell._say_buffer)
            self.assertEqual(shell.prompt, "mpl:chatml fixture-1 > ")
        finally:
            debug_module._MAX_SCRIPT_BYTES = old


class TestAudit(unittest.TestCase):
    def _split_with_literal(self) -> Split:
        return Split(
            search=(
                Fixture(
                    id="s1",
                    group="g1",
                    messages=(Message("user", "hello <|im_start|>"),),
                    expected="x",
                ),
            ),
            held_out=(
                Fixture(id="h1", group="g2", messages=(Message("user", "q"),), expected="y"),
            ),
        )

    def test_audit_reports_findings(self) -> None:
        shell, buf = _make_shell(split=self._split_with_literal())
        out = _run(shell, buf, ".audit", ".quit")
        self.assertIn("audit: 1 finding(s) on fixture s1 with family chatml", out)
        self.assertIn("s1 user_1 literal='<|im_start|>' format=chatml offset=6", out)

    def test_audit_clean_fixture(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".audit", ".quit")
        self.assertIn("audit: 0 finding(s) on fixture s1 with family chatml", out)

    def test_audit_rejects_arguments(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".audit extra", ".quit")
        self.assertIn("error: usage: .audit (takes no arguments)", out)

    def test_audit_follows_active_family(self) -> None:
        shell, buf = _make_shell(split=self._split_with_literal())
        out = _run(shell, buf, ".use llama3", ".audit", ".quit")
        self.assertIn("with family llama3", out)


class TestSave(unittest.TestCase):
    def test_save_writes_transcript_verbatim_with_echoes(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "t.txt")
            # .save persists the transcript exactly as it stood when invoked:
            # command echoes plus their output, excluding the .save line
            # itself (its echo/output land in the transcript afterwards).
            shell.onecmd(".fixtures")
            before = "".join(shell._transcript)
            shell.onecmd(f".save {target}")
            self.assertEqual(Path(target).read_text(encoding="utf-8"), before)
            out = buf.getvalue()
            self.assertIn(f"saved {len(before)} chars of transcript to {target}", out)
            self.assertIn("mpl:chatml fixture-1 > .fixtures\n", before)
            self.assertIn("1: s1 *\n", before)

    def test_save_appends(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "t.txt")
            shell.onecmd(".fixtures")
            before_first = "".join(shell._transcript)
            shell.onecmd(f".save {target}")
            first = Path(target).read_text(encoding="utf-8")
            self.assertEqual(first, before_first)
            shell.onecmd(".families")
            before_second = "".join(shell._transcript)
            shell.onecmd(f".save {target}")
            second = Path(target).read_text(encoding="utf-8")
            self.assertEqual(second, first + before_second)
            self.assertTrue(second.startswith(first))
            self.assertGreater(len(second), len(first))

    def test_save_empty_transcript_writes_zero_chars(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "t.txt")
            shell.onecmd(f".save {target}")
            self.assertIn("saved 0 chars of transcript", buf.getvalue())
            self.assertEqual(Path(target).read_text(encoding="utf-8"), "")

    def test_save_requires_argument(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".save", ".quit")
        self.assertIn("error: usage: .save <path>", out)

    def test_save_rejects_nul_byte(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".save /tmp/a\x00b", ".quit")
        self.assertIn("error: invalid path", out)

    def test_save_requires_existing_parent_directory(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            out = _run(shell, buf, f".save {tmp}/nodir/x.txt", ".quit")
            self.assertIn(f"error: not a directory: {tmp}/nodir", out)

    def test_save_rejects_non_regular_file(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            out = _run(shell, buf, f".save {tmp}", ".quit")
            self.assertIn(f"error: not a regular file: {tmp}", out)

    def test_save_refuses_oversized_existing_file(self) -> None:
        old = debug_module._MAX_SAVE_BYTES
        debug_module._MAX_SAVE_BYTES = 100
        try:
            shell, buf = _make_shell()
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "big.txt"
                target.write_bytes(b"x" * 101)
                out = _run(shell, buf, f".save {target}", ".quit")
                self.assertIn(f"error: refusing to append to a file over 5MB: {target}", out)
        finally:
            debug_module._MAX_SAVE_BYTES = old

    def test_save_refuses_oversized_transcript(self) -> None:
        shell, buf = _make_shell()
        _run(shell, buf, ".families", ".quit")
        old = debug_module._MAX_SAVE_BYTES
        debug_module._MAX_SAVE_BYTES = 10
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = _run(shell, buf, f".save {tmp}/t.txt", ".quit")
                self.assertIn("error: transcript too large to save (over 5MB)", out)
        finally:
            debug_module._MAX_SAVE_BYTES = old

    def test_save_relative_path_resolves_from_cwd(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                _run(shell, buf, ".fixtures", ".save rel.txt", ".quit")
                self.assertTrue((Path(tmp) / "rel.txt").is_file())
            finally:
                os.chdir(old_cwd)

    def test_save_message_sanitizes_control_chars_in_path(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "a\x07b.txt")
            out = _run(shell, buf, f".save {target}", ".quit")
            self.assertNotIn("\x07", out)
            self.assertTrue(Path(target).is_file())


class TestRead(unittest.TestCase):
    def test_read_missing_file(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            out = _run(shell, buf, f".read {tmp}/nope.txt", ".quit")
            self.assertIn(f"error: unreadable script file: {tmp}/nope.txt", out)

    def test_read_skips_comments_and_blanks(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "s.txt"
            script.write_text("# comment\n\n   \n#another\n.fixtures\n", encoding="utf-8")
            out = _run(shell, buf, f".read {script}", ".quit")
            self.assertIn("1: s1 *", out)
            self.assertNotIn("error", out)
            self.assertEqual(shell._read_depth, 0)

    def test_read_quit_inside_script_stops_session(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "s.txt"
            script.write_text(".fixtures\n.quit\n.families\n", encoding="utf-8")
            out = _run(shell, buf, f".read {script}", ".quit")
            self.assertIn("1: s1", out)
            self.assertNotIn("muse_glimmer", out)

    def test_read_error_inside_script_continues(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "s.txt"
            script.write_text(".nosuchcmd\n.fixtures\n", encoding="utf-8")
            out = _run(shell, buf, f".read {script}", ".quit")
            self.assertIn("error: unknown dot-command '.nosuchcmd'", out)
            self.assertIn("1: s1", out)

    def test_read_eof_line_inside_script_stops(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "s.txt"
            script.write_text(".fixtures\nEOF\n.families\n", encoding="utf-8")
            out = _run(shell, buf, f".read {script}")
            self.assertIn("1: s1", out)
            self.assertNotIn("muse_glimmer", out)

    def test_read_requires_argument(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".read", ".quit")
        self.assertIn("error: usage: .read <path>", out)

    def test_read_rejects_directory(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            out = _run(shell, buf, f".read {tmp}", ".quit")
            self.assertIn(f"error: unreadable script file: {tmp}", out)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX FIFO support required")
    def test_read_rejects_fifo(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            fifo = str(Path(tmp) / "fifo")
            os.mkfifo(fifo)
            out = _run(shell, buf, f".read {fifo}", ".quit")
            self.assertIn(f"error: unreadable script file: {fifo}", out)

    def test_read_rejects_non_utf8(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "bad.txt"
            script.write_bytes(b".fixtures\n\xff\xfe\n")
            out = _run(shell, buf, f".read {script}", ".quit")
            self.assertIn(f"error: cannot read script {script}:", out)

    def test_read_rejects_over_1mb(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "big.txt"
            script.write_bytes(b"x" * 1_000_001)
            out = _run(shell, buf, f".read {script}", ".quit")
            self.assertIn(f"error: refusing script larger than 1MB: {script}", out)

    def test_read_rejects_over_10000_lines(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "many.txt"
            script.write_text(".fixtures\n" * 10001, encoding="utf-8")
            out = _run(shell, buf, f".read {script}", ".quit")
            self.assertIn(f"error: refusing script longer than 10000 lines: {script}", out)

    def test_read_accepts_exactly_10000_lines(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "ok.txt"
            script.write_text("# c\n" * 10000, encoding="utf-8")
            out = _run(shell, buf, f".read {script}", ".quit")
            self.assertNotIn("error", out)

    def _write_chain(self, tmp: str, depth: int) -> str:
        """s{depth} -> ... -> s0; s0 runs .families."""
        prev = Path(tmp) / "s0.txt"
        prev.write_text(".families\n", encoding="utf-8")
        for level in range(1, depth + 1):
            cur = Path(tmp) / f"s{level}.txt"
            cur.write_text(f".read {prev}\n", encoding="utf-8")
            prev = cur
        return str(prev)

    def test_read_nesting_at_limit_succeeds(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            top = self._write_chain(tmp, 4)  # 5 nested .read calls, depths 1..5
            out = _run(shell, buf, f".read {top}", ".quit")
            self.assertNotIn("nested .read too deep", out)
            self.assertIn("muse_glimmer", out)
            self.assertEqual(shell._read_depth, 0)

    def test_read_nesting_past_limit_fails_closed(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            top = self._write_chain(tmp, 5)  # 6 nested .read calls
            out = _run(shell, buf, f".read {top}", ".quit")
            self.assertIn("error: nested .read too deep (max 5)", out)
            # The innermost script's payload never ran.
            self.assertNotIn("muse_glimmer", out)
            self.assertEqual(shell._read_depth, 0)

    def test_read_self_recursive_script_terminates(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "self.txt"
            script.write_text(f".read {script}\n", encoding="utf-8")
            out = _run(shell, buf, f".read {script}", ".quit")
            self.assertIn("error: nested .read too deep (max 5)", out)
            self.assertEqual(shell._read_depth, 0)


class TestTranscript(unittest.TestCase):
    def test_transcript_echo_format(self) -> None:
        shell, buf = _make_shell()
        _run(shell, buf, ".fixtures", ".quit")
        self.assertEqual(shell._transcript[0], "mpl:chatml fixture-1 > .fixtures\n")
        self.assertEqual(shell._transcript[1], "1: s1 *\n2: s2\n3: h1\n")

    def test_transcript_records_errors_and_comments(self) -> None:
        shell, buf = _make_shell()
        _run(shell, buf, "# note", ".nosuch", ".quit")
        joined = "".join(shell._transcript)
        self.assertIn("mpl:chatml fixture-1 > # note\n", joined)
        self.assertIn("error: unknown dot-command '.nosuch'", joined)

    def test_transcript_truncation_marks_head(self) -> None:
        old = debug_module._MAX_SAVE_BYTES
        debug_module._MAX_SAVE_BYTES = 200
        try:
            shell, buf = _make_shell()
            _run(shell, buf, ".families", ".fixtures", ".quit")
            self.assertTrue(shell._transcript_truncated)
            self.assertEqual(shell._transcript[0], "[... transcript truncated ...]\n")
            total = sum(len(chunk) for chunk in shell._transcript)
            # Cycle 36 fix: the truncation marker is budgeted, so the total
            # (marker included) stays within _MAX_SAVE_BYTES.
            self.assertLessEqual(total, 200)
        finally:
            debug_module._MAX_SAVE_BYTES = old

    def test_truncated_transcript_save_succeeds(self) -> None:
        # Cycle 36 fix (was: BUG -- see module docstring "Bugs found",
        # item 1): the truncation marker is budgeted inside _MAX_SAVE_BYTES,
        # so a truncated transcript still fits and .save succeeds, keeping
        # the truncation note the onecmd comment promises.
        old = debug_module._MAX_SAVE_BYTES
        debug_module._MAX_SAVE_BYTES = 200
        try:
            shell, buf = _make_shell()
            with tempfile.TemporaryDirectory() as tmp:
                target = str(Path(tmp) / "t.txt")
                shell.onecmd(".families")
                before = "".join(shell._transcript)
                self.assertTrue(before.startswith("[... transcript truncated ...]\n"))
                self.assertLessEqual(len(before), 200)
                buf.seek(0)
                buf.truncate(0)
                shell.onecmd(f".save {target}")
                self.assertIn("saved", buf.getvalue())
                saved = Path(target).read_text(encoding="utf-8")
                self.assertEqual(saved, before)
                self.assertTrue(saved.startswith("[... transcript truncated ...]\n"))
        finally:
            debug_module._MAX_SAVE_BYTES = old


class TestQuitExit(unittest.TestCase):
    def test_quit_and_q_return_true(self) -> None:
        for command in (".quit", ".q"):
            with self.subTest(command=command):
                shell, _ = _make_shell()
                self.assertTrue(shell.onecmd(command))

    def test_quit_rejects_arguments(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".quit extra", ".quit")
        self.assertIn("error: usage: .quit (takes no arguments)", out)

    def test_q_rejects_arguments(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".q extra", ".quit")
        self.assertIn("error: usage: .quit (takes no arguments)", out)

    def test_cmdloop_stops_on_quit(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".quit", ".families")
        self.assertNotIn("muse_glimmer", out)


class TestRunScript(unittest.TestCase):
    def test_run_script_success(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "s.txt"
            script.write_text(".fixtures\n", encoding="utf-8")
            self.assertEqual(run_script(shell, script), 0)
            self.assertIn("1: s1", buf.getvalue())

    def test_run_script_missing_file_returns_1_with_stderr(self) -> None:
        shell, _ = _make_shell()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            result = run_script(shell, "/nonexistent/script.txt")
        self.assertEqual(result, 1)
        self.assertIn("debug: unreadable script file:", err.getvalue())

    def test_run_script_quit_inside_returns_0(self) -> None:
        shell, buf = _make_shell()
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "s.txt"
            script.write_text(".fixtures\n.quit\n", encoding="utf-8")
            self.assertEqual(run_script(shell, script), 0)


class TestRunDebug(unittest.TestCase):
    def _namespace(self, **kwargs):
        kwargs.setdefault("dataset", None)
        kwargs.setdefault("model", None)
        kwargs.setdefault("script", None)
        return argparse.Namespace(**kwargs)

    def _write_dataset(self, tmp: str) -> str:
        path = str(Path(tmp) / "ds.json")
        Path(path).write_text(split_to_json_text(_make_split()), encoding="utf-8")
        return path

    def test_run_debug_invalid_dataset(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            result = run_debug(self._namespace(dataset="/nonexistent.json"))
        self.assertEqual(result, 1)
        self.assertIn("debug: invalid dataset:", err.getvalue())

    def test_run_debug_script_mode_no_banner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._write_dataset(tmp)
            script = Path(tmp) / "s.txt"
            script.write_text(".fixtures\n", encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                result = run_debug(self._namespace(dataset=dataset, script=str(script)))
            self.assertEqual(result, 0)
            self.assertNotIn("model-protocol-lab debug", out.getvalue())
            self.assertIn("1: s1", out.getvalue())

    def test_run_debug_script_failure_returns_1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._write_dataset(tmp)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                result = run_debug(self._namespace(dataset=dataset, script=f"{tmp}/nope.txt"))
            self.assertEqual(result, 1)
            self.assertIn("debug:", err.getvalue())

    def test_run_debug_bad_model_collapses_to_fixed_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._write_dataset(tmp)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                result = run_debug(self._namespace(dataset=dataset, model="/nonexistent_ckpt"))
            self.assertEqual(result, 1)
            # Raw loader text never reaches the terminal; only the fixed code.
            self.assertIn("debug: tokenizer_load_failed", err.getvalue())
            self.assertNotIn("nonexistent_ckpt", err.getvalue())

    def test_run_debug_keyboard_interrupt_exits_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._write_dataset(tmp)
            with (
                mock.patch.object(DebugShell, "cmdloop", side_effect=KeyboardInterrupt),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = run_debug(self._namespace(dataset=dataset))
            self.assertEqual(result, 0)

    def test_cli_debug_script_end_to_end(self) -> None:
        # Full `python -m model_protocol_lab debug --script` path: banner
        # absent, deterministic output, and nothing but the operator's own
        # files left behind (no history/state files -- there is no history).
        repo = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            home = str(Path(tmp) / "home")
            os.makedirs(home)
            dataset = str(Path(tmp) / "ds.json")
            Path(dataset).write_text(split_to_json_text(_make_split()), encoding="utf-8")
            script = str(Path(tmp) / "s.txt")
            Path(script).write_text(".fixtures\n.quit\n", encoding="utf-8")
            env = dict(os.environ, HOME=home)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "model_protocol_lab",
                    "debug",
                    "--dataset",
                    dataset,
                    "--script",
                    script,
                ],
                cwd=str(repo),
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertNotIn("model-protocol-lab debug", proc.stdout)
            self.assertIn("1: s1", proc.stdout)
            leftovers = [p for p in Path(home).rglob("*") if p.is_file() or p.is_symlink()]
            self.assertEqual(leftovers, [])

    def test_debug_module_does_not_pull_in_readline(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import model_protocol_lab.debug; print('readline' in sys.modules)",
            ],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.stdout.strip(), "False")


@unittest.skipUnless(
    find_spec("transformers") is not None and find_spec("tokenizers") is not None,
    "needs the optional hf stack (transformers + tokenizers)",
)
class TestAttachTokenizerEndToEnd(unittest.TestCase):
    """attach_tokenizer on a locally-built checkpoint: tokenizer-only, offline."""

    def _build_checkpoint(self, tmp: str) -> str:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit

        vocab = {
            "<|im_start|>": 0,
            "<|im_end|>": 1,
            "hello": 2,
            "user": 3,
            "\n": 4,
            "[UNK]": 5,
        }
        tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = WhitespaceSplit()
        root = str(Path(tmp) / "ckpt")
        os.makedirs(root)
        tok.save(str(Path(root) / "tokenizer.json"))
        return root

    def test_attach_tokenizer_returns_working_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tok = attach_tokenizer(self._build_checkpoint(tmp))
            self.assertEqual(tok.encode("hello", add_special_tokens=False), [2])

    def test_tokens_uses_attached_tokenizer_exact_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tok = attach_tokenizer(self._build_checkpoint(tmp))
            shell, buf = _make_shell(tokenizer=tok)
            out = _run(shell, buf, ".tokens", ".quit")
            fmt = CATALOG["chatml"]
            rendered = fmt.render((Message("user", "hello"),))
            expected = tok.encode(rendered, add_special_tokens=False)
            self.assertIn(f"count={len(expected)}", out)
            self.assertIn(f"ids=[{', '.join(str(i) for i in expected)}]", out)

    def test_render_segment_ids_match_attached_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tok = attach_tokenizer(self._build_checkpoint(tmp))
            shell, buf = _make_shell(tokenizer=tok)
            out = _run(shell, buf, ".render", ".quit")
            content_line = next(line for line in out.splitlines() if "user_content" in line)
            expected = tok.encode("hello", add_special_tokens=False)
            self.assertIn(f"tokens=[{', '.join(str(i) for i in expected)}]", content_line)


class TestAttachTokenizerBadInput(unittest.TestCase):
    def test_attach_tokenizer_rejects_bad_directory(self) -> None:
        with self.assertRaises(ValueError):
            attach_tokenizer("/nonexistent_checkpoint_dir")


class TestHostileInputs(unittest.TestCase):
    def test_fixtures_sanitizes_hostile_id(self) -> None:
        split = Split(
            search=(
                Fixture(
                    id="s1\x1b[31mred\x07",
                    group="g1",
                    messages=(Message("user", "x"),),
                    expected="y",
                ),
            ),
            held_out=(
                Fixture(id="h1", group="g2", messages=(Message("user", "q"),), expected="y"),
            ),
        )
        shell, buf = _make_shell(split=split)
        out = _run(shell, buf, ".fixtures", ".quit")
        self.assertNotIn("\x1b", out)
        self.assertNotIn("\x07", out)
        self.assertIn("1: s1[31mred *", out)

    def test_fixture_error_uses_repr_so_controls_cannot_escape(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".fixture \x1b[31mnope", ".quit")
        self.assertNotIn("\x1b", out)
        self.assertIn("error: unknown fixture", out)

    def test_use_error_uses_repr_so_controls_cannot_escape(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".use \x07bad", ".quit")
        self.assertNotIn("\x07", out)
        self.assertIn("error: unknown family", out)

    def test_hostile_exception_text_sanitized_at_repl_boundary(self) -> None:
        shell, buf = _make_shell()
        original = shell._commands[".families"]

        def hostile(arg: str) -> bool:
            raise ValueError("boom\x1b[31mred\x07injected")

        shell._commands[".families"] = hostile
        try:
            out = _run(shell, buf, ".families", ".quit")
        finally:
            shell._commands[".families"] = original
        self.assertNotIn("\x1b", out)
        self.assertNotIn("\x07", out)
        self.assertIn("error: boom[31mredinjected", out)

    def test_say_content_with_controls_sanitized_in_render(self) -> None:
        shell, buf = _make_shell()
        out = _run(shell, buf, ".say", "a\x1b[2Jb\x00c", ".", ".render", ".quit")
        self.assertNotIn("\x1b", out)
        self.assertNotIn("\x00", out)

    def test_audit_hostile_literal_sanitized(self) -> None:
        split = Split(
            search=(
                Fixture(
                    id="s1",
                    group="g1",
                    messages=(Message("user", "ok"),),
                    expected="x",
                ),
            ),
            held_out=(
                Fixture(id="h1", group="g2", messages=(Message("user", "q"),), expected="y"),
            ),
        )
        shell, buf = _make_shell(split=split)
        out = _run(shell, buf, ".audit", ".quit")
        self.assertNotIn("\x1b", out)
        # No raw control character anywhere in a hostile session's output.
        for line in out.splitlines():
            self.assertNotRegex(line, r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class TestDeterminism(unittest.TestCase):
    def test_same_commands_same_output(self) -> None:
        commands = (
            ".families",
            ".use gemma",
            ".fixtures",
            ".fixture 2",
            ".render",
            ".boundary 7,8",
            ".stream 1,7,8",
            ".step",
            ".step",
            ".step",
            ".say",
            "user: hi",
            "bye",
            ".",
            ".render",
            ".audit",
            ".quit",
        )
        shell_a, buf_a = _make_shell()
        shell_b, buf_b = _make_shell()
        self.assertEqual(_run(shell_a, buf_a, *commands), _run(shell_b, buf_b, *commands))

    def test_help_output_stable(self) -> None:
        shell_a, buf_a = _make_shell()
        shell_b, buf_b = _make_shell()
        self.assertEqual(
            _run(shell_a, buf_a, ".help", ".quit"),
            _run(shell_b, buf_b, ".help", ".quit"),
        )


if __name__ == "__main__":
    unittest.main()
