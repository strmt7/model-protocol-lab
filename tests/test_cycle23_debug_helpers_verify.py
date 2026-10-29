"""Differential verification of the pure helpers in ``mpl debug`` (debug.py).

Covers ``_render_segments``, ``_overhang``, ``_parse_id_list``,
``_parse_say_line`` and ``_sanitize_display``, plus the pure boundary-stepping
computation inside ``_cmd_step`` (``_overhang`` + ``core.first_boundary`` on
successive stream prefixes). Every assertion is checked against an
INDEPENDENT oracle written from the functions' documented behavior, not from
the implementations. The interactive ``cmd.Cmd`` loop itself is not
exercised (covered elsewhere by test_cycle6_debug.py).

Brief-vs-docstring discrepancies found while grounding the oracles:

1. The task brief claims ``_render_segments`` returns "(segment_name, text)
   pairs PLUS the char-span map". It does not: it returns only
   ``(name, text)`` pairs (signature ``tuple[tuple[str, str], ...]``). Char
   spans are derived by the caller (``_cmd_render``) by accumulating segment
   lengths. The tests below verify the tiling property on the derived spans.
2. ``_overhang``'s docstring calls the running match "how much of the
   boundary suffix is currently in flight" -- the formal definition in the
   same docstring is the precise one: the longest ``k`` such that the seen
   stream ends with ``boundary[:k]`` (a *prefix* of the boundary). The oracle
   below implements the formal definition.

Deterministic: seeded PRNGs only; no network, no model downloads.
"""

from __future__ import annotations

import random
import unittest
from dataclasses import replace

from model_protocol_lab.catalog import CATALOG
from model_protocol_lab.core import Format, Message, first_boundary
from model_protocol_lab.debug import (
    _overhang,
    _parse_id_list,
    _parse_say_line,
    _render_segments,
    _sanitize_display,
)

_FAMILIES = sorted(CATALOG)

# ---------------------------------------------------------------------------
# Independent oracles, written from the documented behavior only.
# ---------------------------------------------------------------------------


def _oracle_segments(fmt: Format, messages: tuple[Message, ...]) -> list[tuple[str, str]]:
    """Documented assembly order: bos, per-message prefix/content/turn-end,
    optional repeated bos, open-reply suffix; per-role turn terminators fall
    back to turn_end, open_suffix falls back to assistant_prefix.
    """
    pieces: list[tuple[str, str]] = [("bos", fmt.bos)]
    for index, message in enumerate(messages):
        if index and fmt.repeat_bos:
            pieces.append(("bos", fmt.bos))
        if message.role == "user":
            prefix, per_role_end = fmt.user_prefix, fmt.user_turn_end
        else:
            prefix, per_role_end = fmt.assistant_prefix, fmt.assistant_turn_end
        turn_end = fmt.turn_end if per_role_end is None else per_role_end
        pieces.append((f"{message.role}_prefix", prefix))
        pieces.append((f"{message.role}_content", message.content))
        pieces.append((f"{message.role}_turn_end", turn_end))
    if fmt.repeat_bos:
        pieces.append(("bos", fmt.bos))
    suffix = fmt.open_suffix if fmt.open_suffix is not None else fmt.assistant_prefix
    pieces.append(("open_suffix", suffix))
    return pieces


def _oracle_overhang(seen: tuple[int, ...], boundary: tuple[int, ...]) -> int:
    """Longest k such that the seen stream ends with boundary[:k]."""
    best = 0
    for k in range(min(len(seen), len(boundary)) + 1):
        if seen[len(seen) - k :] == boundary[:k]:
            best = k
    return best


def _oracle_first_boundary(tokens: tuple[int, ...], stop: tuple[int, ...]) -> int | None:
    """Index of the first contiguous occurrence of stop, else None."""
    for index in range(len(tokens) - len(stop) + 1):
        if tuple(tokens[index : index + len(stop)]) == tuple(stop):
            return index
    return None


def _oracle_sanitize(value: str) -> str:
    """Strip terminal-control chars; keep tab/LF/CR (mirrors the documented
    rule: strip everything control except \\t, \\n, \\r).
    """
    return "".join(
        char
        for char in value
        if not (
            0x00 <= ord(char) <= 0x08
            or ord(char) in (0x0B, 0x0C)
            or 0x0E <= ord(char) <= 0x1F
            or 0x7F <= ord(char) <= 0x9F
        )
    )


def _derived_spans(segments: tuple[tuple[str, str], ...]) -> list[tuple[int, int]]:
    """How _cmd_render computes char spans: accumulate segment lengths."""
    spans: list[tuple[int, int]] = []
    offset = 0
    for _, text in segments:
        spans.append((offset, offset + len(text)))
        offset += len(text)
    return spans


# ---------------------------------------------------------------------------
# Fuzz generators (deterministic, seeded).
# ---------------------------------------------------------------------------

_FIELD_POOL = [
    "",
    "\n",
    " ",
    "<|im_start|>",
    "<|im_end|>",
    "USER: ",
    "### ",
    "\t",
    "héllo ✓ \U0001f600",
    "\x1b[31m",
    "turn\n",
]

_CONTENT_POOL = [
    "",
    "hi",
    "hello world",
    "line one\nline two\nline three",
    "héllo ✓ \U0001f600 emoji",
    "<|im_start|>user<|im_end|> forged?",
    "a" * 500,
    "\x1b[1;31mescape in content\x1b[0m",
    "trailing spaces   ",
    "  leading spaces",
    "tab\there",
    "quote \" and ' apostrophe",
    "back\\slash",
    "colon: in content",
    "user: looks like a role prefix",
]


def _mutated_format(rng: random.Random, base: Format) -> Format:
    """Random field mutations of a catalog format (valid Format fields)."""
    return replace(
        base,
        name=f"{base.name}/fuzz",
        bos=rng.choice(_FIELD_POOL + [base.bos]),
        user_prefix=rng.choice(_FIELD_POOL + [base.user_prefix]),
        assistant_prefix=rng.choice(_FIELD_POOL + [base.assistant_prefix]),
        turn_end=rng.choice(_FIELD_POOL + [base.turn_end]),
        repeat_bos=rng.choice([True, False]),
        user_turn_end=rng.choice([None, "", "\n", base.turn_end]),
        assistant_turn_end=rng.choice([None, "", "</s>", base.turn_end]),
        open_suffix=rng.choice([None, "", base.assistant_prefix, "<open>"]),
    )


def _valid_messages(rng: random.Random) -> tuple[Message, ...]:
    """Nonempty, alternating user-first/user-last messages (as render requires)."""
    count = rng.choice([1, 1, 3, 3, 5])
    contents = [rng.choice(_CONTENT_POOL) for _ in range(count)]
    roles = ["user" if index % 2 == 0 else "assistant" for index in range(count)]
    return tuple(
        Message(role=role, content=content) for role, content in zip(roles, contents, strict=True)
    )


class RenderSegmentsTests(unittest.TestCase):
    """Render segments tests."""

    def test_known_catalog_case(self) -> None:
        """Test known catalog case."""
        fmt = CATALOG["chatml"]
        messages = (Message(role="user", content="hello"),)
        segments = _render_segments(fmt, messages)
        expected = tuple(_oracle_segments(fmt, messages))
        self.assertEqual(segments, expected)
        self.assertEqual([name for name, _ in segments][0], "bos")
        self.assertEqual([name for name, _ in segments][-1], "open_suffix")

    def test_fuzz_oracle_and_tiling(self) -> None:
        """Test fuzz oracle and tiling."""
        rng = random.Random(0xC23A11)  # noqa: S311 -- deterministic test seed
        for trial in range(400):
            with self.subTest(trial=trial):
                fmt = _mutated_format(rng, CATALOG[rng.choice(_FAMILIES)])
                messages = _valid_messages(rng)
                segments = _render_segments(fmt, messages)
                # Independent oracle: exact (name, text) pairs.
                self.assertEqual(segments, tuple(_oracle_segments(fmt, messages)))
                rendered = fmt.render(messages)
                # Reconstruction equals the real render.
                self.assertEqual("".join(text for _, text in segments), rendered)
                # Derived spans tile [0, len(render)) exactly: ordered, no gaps, no overlaps.
                spans = _derived_spans(segments)
                self.assertEqual(spans[0][0], 0)
                for (prev_start, prev_end), (start, end) in zip(spans, spans[1:], strict=False):
                    self.assertEqual(start, prev_end, "gap or overlap between spans")
                    self.assertGreaterEqual(end, start)
                    self.assertGreaterEqual(prev_start, 0)
                self.assertEqual(spans[-1][1], len(rendered))
                # Each span's text equals the piece at that offset.
                for (name, text), (start, end) in zip(segments, spans, strict=True):
                    self.assertEqual(rendered[start:end], text, f"span text mismatch: {name}")

    def test_fuzz_all_27_families(self) -> None:
        """Test fuzz all 27 families."""
        rng = random.Random(0xFA4117)  # noqa: S311 -- deterministic test seed
        for family in _FAMILIES:
            with self.subTest(family=family):
                fmt = CATALOG[family]
                messages = _valid_messages(rng)
                segments = _render_segments(fmt, messages)
                self.assertEqual(segments, tuple(_oracle_segments(fmt, messages)))
                self.assertEqual("".join(text for _, text in segments), fmt.render(messages))

    def test_segment_names_follow_documented_order(self) -> None:
        """Test segment names follow documented order."""
        fmt = replace(CATALOG["chatml"], name="x", repeat_bos=True)
        messages = (
            Message(role="user", content="u1"),
            Message(role="assistant", content="a1"),
            Message(role="user", content="u2"),
        )
        names = [name for name, _ in _render_segments(fmt, messages)]
        self.assertEqual(
            names,
            [
                "bos",
                "user_prefix",
                "user_content",
                "user_turn_end",
                "bos",
                "assistant_prefix",
                "assistant_content",
                "assistant_turn_end",
                "bos",
                "user_prefix",
                "user_content",
                "user_turn_end",
                "bos",
                "open_suffix",
            ],
        )

    def test_empty_content_message(self) -> None:
        """Test empty content message."""
        fmt = CATALOG["llama3"]
        messages = (Message(role="user", content=""),)
        segments = _render_segments(fmt, messages)
        content = [text for name, text in segments if name == "user_content"]
        self.assertEqual(content, [""])
        spans = _derived_spans(segments)
        self.assertEqual(spans[-1][1], len(fmt.render(messages)))

    def test_invalid_messages_raise(self) -> None:
        """Test invalid messages raise."""
        fmt = CATALOG["chatml"]
        with self.assertRaises(ValueError):
            _render_segments(fmt, ())
        with self.assertRaises(ValueError):  # not alternating
            _render_segments(
                fmt,
                (Message(role="user", content="a"), Message(role="user", content="b")),
            )
        with self.assertRaises(ValueError):  # final role must be user
            _render_segments(
                fmt,
                (Message(role="user", content="a"), Message(role="assistant", content="b")),
            )


class OverhangTests(unittest.TestCase):
    """Overhang tests."""

    def test_fuzz_against_brute_force(self) -> None:
        """Test fuzz against brute force."""
        rng = random.Random(0x0E2A96)  # noqa: S311 -- deterministic test seed
        for trial in range(2000):
            with self.subTest(trial=trial):
                seen = tuple(rng.randint(0, 5) for _ in range(rng.randint(0, 8)))
                boundary = tuple(rng.randint(0, 5) for _ in range(rng.randint(0, 6)))
                self.assertEqual(_overhang(seen, boundary), _oracle_overhang(seen, boundary))

    def test_fuzz_large_ids(self) -> None:
        """Test fuzz large ids."""
        rng = random.Random(0xB16B00B5)  # noqa: S311 -- deterministic test seed
        for trial in range(300):
            with self.subTest(trial=trial):
                seen = tuple(rng.randint(0, 2**31 - 1) for _ in range(rng.randint(0, 6)))
                boundary = tuple(rng.randint(0, 2**31 - 1) for _ in range(rng.randint(1, 4)))
                self.assertEqual(_overhang(seen, boundary), _oracle_overhang(seen, boundary))

    def test_known_cases(self) -> None:
        """Test known cases."""
        cases = [
            ((1, 2, 3), (2, 3, 4), 2),
            ((1, 2, 3), (1, 2, 3), 3),  # full match is still an overhang
            ((9, 9), (1, 2), 0),
            ((), (1, 2), 0),
            ((1, 2), (), 0),
            ((1, 2, 1, 2), (1, 2, 1, 2, 3), 4),
            ((7,), (7, 7), 1),
            ((5, 6), (6,), 1),
            ((5, 6), (5,), 0),
        ]
        for seen, boundary, expected in cases:
            with self.subTest(seen=seen, boundary=boundary):
                self.assertEqual(_overhang(tuple(seen), tuple(boundary)), expected)

    def test_result_is_longest_matching_prefix(self) -> None:
        """Test result is longest matching prefix."""
        # Property: the returned k is the LARGEST k with seen ending in boundary[:k].
        rng = random.Random(0xDECAF)  # noqa: S311 -- deterministic test seed
        for trial in range(500):
            with self.subTest(trial=trial):
                seen = tuple(rng.randint(0, 3) for _ in range(rng.randint(0, 6)))
                boundary = tuple(rng.randint(0, 3) for _ in range(rng.randint(0, 5)))
                k = _overhang(seen, boundary)
                self.assertEqual(seen[len(seen) - k :], boundary[:k])
                for bigger in range(k + 1, min(len(seen), len(boundary)) + 1):
                    self.assertNotEqual(
                        seen[len(seen) - bigger :],
                        boundary[:bigger],
                        "a longer match existed",
                    )


class ParseIdListTests(unittest.TestCase):
    """Parse id list tests."""

    def test_valid(self) -> None:
        """Test valid."""
        self.assertEqual(_parse_id_list("0", what="x", max_len=16), (0,))
        self.assertEqual(_parse_id_list("1,2,3", what="x", max_len=16), (1, 2, 3))
        self.assertEqual(_parse_id_list("  7 ,  8\t", what="x", max_len=16), (7, 8))
        self.assertEqual(_parse_id_list(str(2**31 - 1), what="x", max_len=16), (2**31 - 1,))
        self.assertEqual(_parse_id_list(",".join(["1"] * 16), what="x", max_len=16), (1,) * 16)

    def test_rejects(self) -> None:
        """Test rejects."""
        bad = [
            "",
            "   ",
            ",",
            "1,,2",
            ",1",
            "1,",
            "1,2,",
            "abc",
            "1a",
            "-1",
            "+1",
            "1.5",
            "0x10",
            "1 2",
            "1;2",
            "1_000",
            "१२३",  # non-ASCII decimal digits must be rejected
            str(2**31),  # over the 2^31-1 cap
            "999999999999999999999",
            ",".join(["1"] * 17),  # over max_len
        ]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                _parse_id_list(value, what="stream", max_len=16)

    def test_error_names_what_and_max_len(self) -> None:
        """Test error names what and max len."""
        with self.assertRaises(ValueError) as ctx:
            _parse_id_list("nope", what="boundary", max_len=16)
        self.assertIn("boundary", str(ctx.exception))
        self.assertIn("16", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            _parse_id_list(str(2**31), what="zz", max_len=4)
        self.assertIn("zz", str(ctx.exception))

    def test_fuzz_round_trip(self) -> None:
        """Test fuzz round trip."""
        rng = random.Random(0x1D1157)  # noqa: S311 -- deterministic test seed
        for trial in range(500):
            with self.subTest(trial=trial):
                ids = tuple(rng.randint(0, 2**31 - 1) for _ in range(rng.randint(1, 8)))
                text = ",".join(str(i) for i in ids)
                self.assertEqual(_parse_id_list(text, what="x", max_len=16), ids)
                # Random whitespace padding around parts must still parse.
                padded = ",".join(f" {i} " for i in ids)
                self.assertEqual(_parse_id_list(padded, what="x", max_len=16), ids)

    def test_fuzz_rejects_corrupted(self) -> None:
        """Test fuzz rejects corrupted."""
        rng = random.Random(0xBAD1D)  # noqa: S311 -- deterministic test seed
        alphabet = list("0123456789,") + [" ", "a", "-", "+", ".", ";", "\t"]
        for trial in range(800):
            with self.subTest(trial=trial):
                value = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
                parts = [part.strip() for part in value.split(",")]
                ok = (
                    1 <= len(parts) <= 5
                    and all(part and all("0" <= ch <= "9" for ch in part) for part in parts)
                    and all(int(part) <= 2**31 - 1 for part in parts)
                )
                if ok:
                    self.assertEqual(
                        _parse_id_list(value, what="x", max_len=5),
                        tuple(int(part) for part in parts),
                    )
                else:
                    with self.assertRaises(ValueError):
                        _parse_id_list(value, what="x", max_len=5)


class ParseSayLineTests(unittest.TestCase):
    """Parse say line tests."""

    def test_role_prefixes(self) -> None:
        """Test role prefixes."""
        cases = [
            ("user: hello", ("user", "hello")),
            ("User: hello", ("user", "hello")),
            ("USER: hello", ("user", "hello")),
            ("assistant: hi", ("assistant", "hi")),
            ("Assistant: hi", ("assistant", "hi")),
            ("ASSISTANT:hi", ("assistant", "hi")),
            ("user:hi", ("user", "hi")),
            ("user : spaced", ("user", "spaced")),
            ("user:\ttabbed", ("user", "tabbed")),
            ("user:", ("user", "")),
            ("user:   ", ("user", "")),
            ("user: a: b", ("user", "a: b")),
            ("user: line1\nline2", ("user", "line1\nline2")),
            ("  user: padded  ", ("user", "padded")),
        ]
        for line, expected in cases:
            with self.subTest(line=line):
                self.assertEqual(_parse_say_line(line), expected)

    def test_unrecognized_prefix_kept_as_content(self) -> None:
        """Test unrecognized prefix kept as content."""
        # Docstring contract: unrecognized "word:" prefixes are kept as
        # content, never mangled; the ORIGINAL line is returned verbatim.
        cases = [
            "note: do not mangle",
            "data: 123",
            "user1: digit in role",
            "user_name: underscore",
            ": empty role",
            "user name: space in role",
            "u: too short",
            "users: plural",
            "user",  # no colon at all
            "hello world",
            "",
            "   ",
            "  note: padded stays verbatim  ",
        ]
        for line in cases:
            with self.subTest(line=line):
                role, content = _parse_say_line(line)
                self.assertIsNone(role)
                self.assertEqual(content, line)

    def test_fuzz_prefix_grammar(self) -> None:
        """Test fuzz prefix grammar."""
        rng = random.Random(0x5A4)  # noqa: S311 -- deterministic test seed
        role_words = ["user", "assistant", "User", "ASSISTANT", "note", "data", "u1", ""]
        for trial in range(600):
            with self.subTest(trial=trial):
                word = rng.choice(role_words)
                body = rng.choice(["x", "", "a: b", "  spaced  ", "multi\nline"])
                line = f"{word}: {body}" if word else body
                role, content = _parse_say_line(line)
                if word and word.lower() in ("user", "assistant"):
                    self.assertEqual(role, word.lower())
                    # Leading whitespace after the colon and outer whitespace
                    # are consumed by the regex on the stripped line.
                    self.assertEqual(content, body.strip())
                else:
                    self.assertIsNone(role)
                    self.assertEqual(content, line)


class SanitizeDisplayTests(unittest.TestCase):
    """Sanitize display tests."""

    def test_control_chars_stripped(self) -> None:
        """Test control chars stripped."""
        stripped = "".join(
            chr(c)
            for c in list(range(0x00, 0x09))
            + [0x0B, 0x0C]
            + list(range(0x0E, 0x20))
            + list(range(0x7F, 0xA0))
        )
        self.assertEqual(_sanitize_display(f"a{stripped}b"), "ab")
        self.assertIn("\x1b", stripped)  # ESC is covered
        self.assertIn("\x9b", stripped)  # C1 CSI introducer is covered

    def test_tab_lf_cr_kept(self) -> None:
        """Test tab lf cr kept."""
        self.assertEqual(_sanitize_display("a\tb\nc\rd"), "a\tb\nc\rd")

    def test_printable_and_unicode_kept(self) -> None:
        """Test printable and unicode kept."""
        text = "héllo ✓ \U0001f600 <|im_start|> \u00a0"
        self.assertEqual(_sanitize_display(text), text)

    def test_fuzz_against_char_oracle(self) -> None:
        """Test fuzz against char oracle."""
        rng = random.Random(0x5A7172)  # noqa: S311 -- deterministic test seed
        alphabet = [chr(c) for c in range(0x20, 0x7F)] + [
            "\t",
            "\n",
            "\r",
            "\x00",
            "\x07",
            "\x1b",
            "\x7f",
            "\x85",
            "\x9b",
            "é",
            "\U0001f600",
            "",
        ]
        for trial in range(600):
            with self.subTest(trial=trial):
                value = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
                self.assertEqual(_sanitize_display(value), _oracle_sanitize(value))
        # Exhaustive single-char check: every char in 0x00..0x2FF agrees.
        for code in range(0x300):
            with self.subTest(code=code):
                char = chr(code)
                self.assertEqual(_sanitize_display(char), _oracle_sanitize(char))


class BoundaryStepComputationTests(unittest.TestCase):
    """The pure per-step computation in _cmd_step: overhang and first_boundary
    over successive prefixes of the recorded stream.
    """

    def test_fuzz_step_walk(self) -> None:
        """Test fuzz step walk."""
        rng = random.Random(0x57E9)  # noqa: S311 -- deterministic test seed
        for trial in range(400):
            with self.subTest(trial=trial):
                stream = tuple(rng.randint(0, 4) for _ in range(rng.randint(1, 12)))
                boundary = tuple(rng.randint(0, 4) for _ in range(rng.randint(1, 5)))
                for end in range(1, len(stream) + 1):
                    seen = stream[:end]
                    self.assertEqual(_overhang(seen, boundary), _oracle_overhang(seen, boundary))
                    self.assertEqual(
                        first_boundary(seen, boundary), _oracle_first_boundary(seen, boundary)
                    )

    def test_first_boundary_requires_nonempty(self) -> None:
        """Test first boundary requires nonempty."""
        with self.assertRaises(ValueError):
            first_boundary((1, 2), ())

    def test_overhang_zero_after_full_match(self) -> None:
        """Test overhang zero after full match."""
        # After a full match, a following non-matching token resets overhang
        # to the brute-force value (no stale state).
        seen, boundary = (10, 20, 30), (10, 20, 30)
        self.assertEqual(first_boundary(seen, boundary), 0)
        self.assertEqual(_overhang(seen, boundary), 3)
        seen2 = seen + (99,)
        self.assertEqual(first_boundary(seen2, boundary), 0)
        self.assertEqual(_overhang(seen2, boundary), _oracle_overhang(seen2, boundary))


class SensitivityTests(unittest.TestCase):
    """Prove the oracles catch plausible implementation mutations.

    Each mutated copy below is a small, realistic bug (off-by-one, dropped
    case, over/under-strict check). The test asserts the oracle disagrees
    with the mutation on a concrete input -- i.e. the main test classes
    above would fail against the mutated implementation.
    """

    def test_overhang_drops_zero_case(self) -> None:
        """Test overhang drops zero case."""

        def mutated(seen: tuple[int, ...], boundary: tuple[int, ...]) -> int | None:
            """Mutated."""
            for k in range(min(len(seen), len(boundary)), 0, -1):  # BUG: skips k=0
                if seen[len(seen) - k :] == boundary[:k]:
                    return k
            return None  # type: ignore[return-value]

        seen, boundary = (9,), (1, 2)
        self.assertEqual(_overhang(seen, boundary), 0)
        self.assertNotEqual(mutated(seen, boundary), _oracle_overhang(seen, boundary))

    def test_overhang_off_by_one_slice(self) -> None:
        """Test overhang off by one slice."""

        def mutated(seen: tuple[int, ...], boundary: tuple[int, ...]) -> int:
            """Mutated."""
            for k in range(min(len(seen), len(boundary)), -1, -1):
                if seen[len(seen) - k :] == boundary[: k + 1]:  # BUG: k+1
                    return k
            return 0

        seen, boundary = (1, 2, 3), (1, 2, 3, 4)
        self.assertEqual(_overhang(seen, boundary), 3)
        self.assertNotEqual(mutated(seen, boundary), _oracle_overhang(seen, boundary))

    def test_render_segments_dropped_open_suffix(self) -> None:
        """Test render segments dropped open suffix."""

        def mutated(fmt: Format, messages: tuple[Message, ...]) -> tuple[tuple[str, str], ...]:
            """Mutated."""
            pieces = list(_oracle_segments(fmt, messages))
            return tuple(pieces[:-1])  # BUG: open_suffix dropped

        fmt = CATALOG["chatml"]
        messages = (Message(role="user", content="hi"),)
        self.assertNotEqual(
            "".join(text for _, text in mutated(fmt, messages)), fmt.render(messages)
        )
        self.assertNotEqual(mutated(fmt, messages), tuple(_oracle_segments(fmt, messages)))

    def test_render_segments_swapped_order(self) -> None:
        """Test render segments swapped order."""

        def mutated(fmt: Format, messages: tuple[Message, ...]) -> tuple[tuple[str, str], ...]:
            """Mutated."""
            pieces = list(_oracle_segments(fmt, messages))
            pieces[1], pieces[2] = pieces[2], pieces[1]  # BUG: content before prefix
            return tuple(pieces)

        fmt = CATALOG["llama3"]
        messages = (Message(role="user", content="hi"),)
        self.assertNotEqual(mutated(fmt, messages), tuple(_oracle_segments(fmt, messages)))

    def test_parse_id_list_allows_over_cap(self) -> None:
        """Test parse id list allows over cap."""

        def mutated(value: str, *, what: str, max_len: int) -> tuple[int, ...]:
            """Mutated."""
            parts = [part.strip() for part in value.split(",")]  # BUG: no 2^31 cap
            return tuple(int(part) for part in parts)

        value = str(2**31)
        with self.assertRaises(ValueError):
            _parse_id_list(value, what="boundary", max_len=16)
        # The mutation silently accepts the over-cap ID: the reject-case
        # tests above catch exactly this.
        self.assertEqual(mutated(value, what="boundary", max_len=16), (2**31,))

    def test_parse_id_list_allows_extra_part(self) -> None:
        """Test parse id list allows extra part."""

        def mutated(value: str, *, what: str, max_len: int) -> tuple[int, ...]:
            """Mutated."""
            return _parse_id_list(value, what=what, max_len=max_len + 1)  # BUG: off-by-one cap

        value = ",".join(["1"] * 6)
        with self.assertRaises(ValueError):
            _parse_id_list(value, what="x", max_len=5)
        # The mutation accepts the 6th ID: the max_len reject tests catch it.
        self.assertEqual(mutated(value, what="x", max_len=5), (1,) * 6)

    def test_parse_say_line_returns_stripped_on_no_match(self) -> None:
        """Test parse say line returns stripped on no match."""

        def mutated(line: str) -> tuple[None, str]:
            """Mutated."""
            return None, line.strip()  # BUG: mangles verbatim content

        line = "  note: padded stays verbatim  "
        self.assertEqual(_parse_say_line(line), (None, line))
        self.assertNotEqual(mutated(line), (None, line))

    def test_parse_say_line_rejects_unrecognized_prefix(self) -> None:
        """Test parse say line rejects unrecognized prefix."""

        def mutated(line: str) -> tuple[str | None, str]:
            """Mutated."""
            if ":" in line:
                raise ValueError("unknown role")  # BUG: rejects prose
            return None, line

        with self.assertRaises(ValueError):
            mutated("note: do not mangle")
        self.assertEqual(_parse_say_line("note: do not mangle"), (None, "note: do not mangle"))

    def test_sanitize_strips_tab_too(self) -> None:
        """Test sanitize strips tab too."""

        def mutated(value: str) -> str:
            """Mutated."""
            return "".join(ch for ch in _sanitize_display(value) if ch != "\t")  # BUG

        self.assertEqual(_sanitize_display("a\tb"), "a\tb")
        self.assertNotEqual(mutated("a\tb"), _oracle_sanitize("a\tb"))

    def test_sanitize_keeps_esc(self) -> None:
        """Test sanitize keeps esc."""

        def mutated(value: str) -> str:
            """Mutated."""
            return value  # BUG: no sanitization at all

        self.assertEqual(_sanitize_display("\x1b[31m"), "[31m")  # ESC gone, text kept
        self.assertNotEqual(mutated("\x1b[31m"), _oracle_sanitize("\x1b[31m"))


if __name__ == "__main__":
    unittest.main()
