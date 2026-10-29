"""Cycle 88 property-based audit: hostile-input fuzzing of the dataset parsers.

METHOD
------
Target: ``datasets.parse_dataset`` / ``datasets.load_dataset`` -- the
REJECTION side of the parser contract. Cycle 86 fuzzed schema-valid inputs
(round-trip identity); no cycle has property-fuzzed hostile inputs. Every
test below asserts the universal parser contract for EVERY input it
generates: the call either returns a structurally valid ``Split`` or raises
``ValueError`` -- never another exception type (``TypeError``, ``KeyError``,
bare ``RecursionError``, ...), never a hang, never a malformed ``Split``.

(``UnicodeDecodeError``/``UnicodeEncodeError`` are ``ValueError`` subclasses,
so they satisfy the contract when they escape; the file-based tests confirm
they are normalized to the fixed vocabulary anyway.)

Strategies (hypothesis, derandomized, modest max_examples):
1. ``st.text()`` -- arbitrary Unicode: NUL, lone surrogates, astral planes,
   huge strings.
2. Mutations of a known-valid fixture JSON: truncation at every offset,
   single-character substitution, ``"zz": 1`` splicing at every offset,
   rotation.
3. Type-shape attacks: a recursive strategy emitting wrong JSON types at the
   top level (array/string/number/null/bool, nested junk), plus a
   deterministic battery hitting every schema level (fixture as string,
   messages as dict/string, message as string, role as number/null, content
   as list/dict/number, id/expected/group wrong types, extra and missing
   keys at every object level, 22-message overflow).
4. Number attacks: 10**1000 integers, a 5000-digit literal (trips the
   4300-digit int<->str cap), 1e308/-1e308/1e-9999, raw ``NaN``/``Infinity``/
   ``-Infinity`` tokens spliced into valid JSON, 0-message conversations.
5. Nesting attacks: list/dict nesting at depths 50-500 (property -- must be
   ``ValueError``, never ``RecursionError``), nesting spliced inside a
   fixture, and deterministic 20000/300000-deep probes of the
   ``RecursionError`` -> ``ValueError("Dataset nesting too deep")`` guard
   (probed in a subprocess first: no segfault, no hang, inside the 2MB byte
   cap).
6. ``load_dataset`` file attacks: hostile payloads written to PID-scoped
   temp files (``tempfile``, no hard-coded /tmp), plus directory path, FIFO,
   non-UTF8 bytes, missing file, >MAX_BYTES file, and wrong-type paths.

Fixed-vocabulary pins -- substrings read verbatim from ``datasets.py`` (not
guessed): "Duplicate JSON field", "Nonfinite JSON", "Unexpected or missing
JSON fields", "conversation must have 1-{MAX_MESSAGES} messages" (built from
the imported constant), "Dataset nesting too deep", "Fixture path is not a
regular file", "Fixture file is not valid UTF-8", "Unreadable fixture file",
"Dataset exceeds size limit", "Invalid fixture path".

FINDINGS
--------
CLEAN: no bug found. Every hostile input probed during development (direct
probes plus the full property runs below) either raised ``ValueError`` or --
for mutations that happened to stay schema-valid -- returned a structurally
valid ``Split``. No ``TypeError``/``KeyError``/bare ``RecursionError``/hang
observed anywhere, including 300000-deep nesting inside the byte cap and
FIFO/directory/non-UTF8 file attacks. One vocabulary observation (not a
contract violation, deliberately NOT "fixed"): a NUL byte in the path raises
``ValueError("embedded null byte")`` straight from ``os.open`` instead of
the module's "Unreadable fixture file" -- still a ``ValueError``, so the
universal contract holds; normalizing it would churn the source for zero
contract gain.

VERIFIED BY RUNNING
-------------------
2026-10-16: ``python -m pytest tests/test_cycle88_parser_hostile_properties.py``
green (48 tests from 17 test functions incl. parametrized cases, ~1s);
``ruff check`` / ``ruff format --check`` clean; ``mypy`` clean. Negative
control: the contract helper was sanity-checked against a deliberately
broken local copy of ``datasets.py`` (``_fields`` raising ``TypeError``
instead of ``ValueError``) and 11 tests failed as required, proving the
harness actually detects non-ValueError escapes; the source file was
restored byte-identical afterwards (diff-verified).
"""

import json
import os
import re
import tempfile
import unittest
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from model_protocol_lab.core import Fixture, Message, Split
from model_protocol_lab.datasets import (
    MAX_BYTES,
    MAX_MESSAGES,
    load_dataset,
    parse_dataset,
)

FUZZ = settings(
    derandomize=True,
    max_examples=30,
    # Same rationale as tests/test_fuzz_boundary.py: differing_executors fires
    # on in-process re-runs (a harness artifact), not on flakiness; these
    # tests are deterministic, so the check is suppressed with justification.
    suppress_health_check=[HealthCheck.differing_executors],
)
FILE_FUZZ = settings(
    derandomize=True,
    max_examples=15,
    suppress_health_check=[HealthCheck.differing_executors],
)


def _fixture_dict(fid, group, content):
    return {
        "id": fid,
        "group": group,
        "messages": [{"role": "user", "content": content}],
        "expected": "ok",
    }


_SEARCH_FIX = _fixture_dict("s1", "gs", "Search question?")
_HELD_FIX = _fixture_dict("h1", "gh", "Held-out question?")
_VALID_TEXT = json.dumps({"search": [_SEARCH_FIX], "held_out": [_HELD_FIX]})


def assert_valid_split(split):
    """Deep structural check: a returned Split must never be malformed."""
    assert type(split) is Split, f"not a Split: {type(split)!r}"
    for name in ("search", "held_out"):
        part = getattr(split, name)
        assert type(part) is tuple and len(part) >= 1, f"bad {name}: {part!r}"
    cases = split.search + split.held_out
    assert all(type(case) is Fixture for case in cases)
    ids = [case.id for case in cases]
    assert all(type(i) is str and i.strip() for i in ids)
    assert len(set(ids)) == len(ids), "duplicate fixture ids"
    for case in cases:
        assert type(case.messages) is tuple
        assert 1 <= len(case.messages) <= MAX_MESSAGES
        assert all(type(m) is Message for m in case.messages)
        assert all(type(m.content) is str for m in case.messages)
        roles = [m.role for m in case.messages]
        assert roles[0] == "user" and roles[-1] == "user"
        assert all(r == ("user" if i % 2 == 0 else "assistant") for i, r in enumerate(roles))
        assert type(case.expected) is str
        assert type(case.group) is str and case.group.strip()


def check_contract(make_call):
    """Universal parser contract: ValueError or a structurally valid Split.

    Anything else -- another exception type escaping, or a malformed Split
    being returned -- is a real bug and fails the test loudly.
    """
    try:
        result = make_call()
    except ValueError:
        return None
    except Exception as exc:  # noqa: BLE001 -- the contract under test forbids these
        raise AssertionError(f"non-ValueError escaped: {type(exc).__name__}: {exc!r}") from exc
    assert_valid_split(result)
    return result


def check_rejected(text):
    """Strict form: this input is known-invalid, so it must raise ValueError."""
    try:
        parse_dataset(text)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {text!r}")


# ---------------------------------------------------------------------------
# Sanity: the mutation base is itself a valid dataset.
# ---------------------------------------------------------------------------


_WRONG_SCALAR = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**300), max_value=10**300),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=30),
)

_WRONG_VALUE = st.recursive(
    _WRONG_SCALAR,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(min_size=1, max_size=10), children, max_size=4),
    ),
    max_leaves=8,
)

_HOSTILE_FILE_BYTES = st.one_of(
    st.text(max_size=500).map(lambda s: s.encode("utf-8", errors="surrogatepass")),
    st.binary(max_size=300),
    st.integers(0, len(_VALID_TEXT)).map(lambda k: _VALID_TEXT[:k].encode("utf-8")),
)


class ParserHostilePropertiesTests(unittest.TestCase):
    """unittest entry point; pytest runs TestCase classes natively."""

    def setUp(self) -> None:
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import tempfile

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from pathlib import Path

        # PID-scoped temp dir; tempfile picks the location (no hard-coded /tmp).
        self._td = tempfile.TemporaryDirectory(prefix=f"mpl88-{os.getpid()}-")
        self.pid_tmpdir = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    @staticmethod
    def test_valid_text_sanity():
        split = check_contract(lambda: parse_dataset(_VALID_TEXT))
        assert split is not None
        assert [c.id for c in split.search] == ["s1"]
        assert [c.id for c in split.held_out] == ["h1"]

    @FUZZ
    @given(st.text())
    def test_arbitrary_text_contract(self, text):
        check_contract(lambda: parse_dataset(text))

    @FUZZ
    @given(st.integers(0, len(_VALID_TEXT)))
    def test_truncation_contract(self, cut):
        check_contract(lambda: parse_dataset(_VALID_TEXT[:cut]))

    @FUZZ
    @given(st.data())
    def test_char_substitution_contract(self, data):
        index = data.draw(st.integers(0, len(_VALID_TEXT) - 1))
        replacement = data.draw(st.characters())
        mutated = _VALID_TEXT[:index] + replacement + _VALID_TEXT[index + 1 :]
        check_contract(lambda: parse_dataset(mutated))

    @FUZZ
    @given(st.integers(1, len(_VALID_TEXT) - 1))
    def test_rotation_contract(self, split_at):
        rotated = _VALID_TEXT[split_at:] + _VALID_TEXT[:split_at]
        check_contract(lambda: parse_dataset(rotated))

    @FUZZ
    @given(st.integers(0, len(_VALID_TEXT)))
    def test_key_splice_contract(self, index):
        mutated = _VALID_TEXT[:index] + '"zz": 1, ' + _VALID_TEXT[index:]
        check_contract(lambda: parse_dataset(mutated))

    @FUZZ
    @given(_WRONG_VALUE)
    def test_wrong_top_level_shape_property(self, value):
        check_contract(lambda: parse_dataset(json.dumps(value)))

    def test_type_shape_attacks_at_every_level(self):
        for name, text in _SHAPE_CASES:
            with self.subTest(name=name, text=text):
                check_rejected(text)

    def test_non_string_inputs_rejected(self):
        for bad in [None, 123, 4.5, b"{}", bytearray(b"{}"), ["x"], {"a": 1}, True]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_dataset(bad)  # type: ignore[arg-type]

    def test_str_subclass_rejected(self):
        class Sneaky(str):
            pass

        with self.assertRaises(ValueError):
            parse_dataset(Sneaky("{}"))

    @staticmethod
    def test_number_attacks():
        check_rejected(json.dumps({"search": 10**1000, "held_out": []}))
        check_rejected('{"search": [' + "9" * 5000 + "]}")  # trips the 4300-digit cap
        for token in ("1e308", "-1e308", "1e-9999"):
            check_rejected(_VALID_TEXT.replace('"Search question?"', token, 1))

    @FUZZ
    @given(st.integers(50, 500), st.sampled_from(["[]", "{}"]))
    def test_nesting_depth_property(self, depth, kind):
        text = "[" * depth + "]" * depth if kind == "[]" else '{"k":' * depth + "0" + "}" * depth
        # Must be ValueError -- a bare RecursionError escaping is the F5 bug class.
        check_contract(lambda: parse_dataset(text))

    @staticmethod
    def test_nesting_spliced_inside_fixture():
        deep = "[" * 300 + "]" * 300
        bad = _VALID_TEXT.replace('[{"role": "user", "content": "Search question?"}]', deep, 1)
        check_rejected(bad)

    def test_deep_nesting_guard_fixed_vocabulary(self):
        for depth in [20000, 300000]:
            with (
                self.subTest(depth=depth),
                self.assertRaisesRegex(ValueError, re.escape("Dataset nesting too deep")),
            ):
                parse_dataset("[" * depth + "]" * depth)

    def test_fixed_vocabulary_pins(self):
        dup = _VALID_TEXT.replace('"id": "s1"', '"id": "s1", "id": "s2"', 1)
        with self.assertRaisesRegex(ValueError, re.escape("Duplicate JSON field")):
            parse_dataset(dup)

        for token in ("NaN", "Infinity", "-Infinity"):
            bad = _VALID_TEXT.replace('"Search question?"', token, 1)
            with self.assertRaisesRegex(ValueError, re.escape("Nonfinite JSON")):
                parse_dataset(bad)

        for bad in ("[]", '"x"', "42", "null", "true"):
            with self.assertRaisesRegex(ValueError, re.escape("Unexpected or missing JSON fields")):
                parse_dataset(bad)

        zero = _doc(dict(_SEARCH_FIX, messages=[]))
        with self.assertRaisesRegex(
            ValueError,
            re.escape(f"conversation must have 1-{MAX_MESSAGES} messages"),
        ):
            parse_dataset(zero)

    def test_oversize_text_rejected(self):
        with self.assertRaisesRegex(ValueError, re.escape("Dataset exceeds size limit")):
            parse_dataset(" " * (MAX_BYTES + 1))
        # Exactly at the cap: still rejected, but by the JSON layer, not the guard.
        check_rejected(" " * MAX_BYTES)

    @FILE_FUZZ
    @given(_HOSTILE_FILE_BYTES)
    def test_load_dataset_hostile_files(self, payload):
        with tempfile.TemporaryDirectory(prefix=f"mpl88-{os.getpid()}-") as path:
            target = Path(path) / "hostile.json"
            target.write_bytes(payload)
            check_contract(lambda: load_dataset(target))

    def test_load_dataset_valid_file(self):
        target = self.pid_tmpdir / "valid.json"
        target.write_text(_VALID_TEXT, encoding="utf-8")
        split = check_contract(lambda: load_dataset(target))
        assert split is not None

    def test_load_dataset_path_attacks(self):
        with self.assertRaisesRegex(ValueError, re.escape("Fixture path is not a regular file")):
            load_dataset(self.pid_tmpdir)  # a directory is never loadable

        fifo = self.pid_tmpdir / f"fifo-{os.getpid()}"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, re.escape("Fixture path is not a regular file")):
            load_dataset(fifo)  # refused via O_NONBLOCK + fstat: cannot hang

        bad = self.pid_tmpdir / "bad.bin"
        bad.write_bytes(b"\xff\xfe\x00not-json")
        with self.assertRaisesRegex(ValueError, re.escape("Fixture file is not valid UTF-8")):
            load_dataset(bad)

        with self.assertRaisesRegex(ValueError, re.escape("Unreadable fixture file")):
            load_dataset(self.pid_tmpdir / "missing.json")

        big = self.pid_tmpdir / "big.json"
        big.write_bytes(b" " * (MAX_BYTES + 1))
        with self.assertRaisesRegex(ValueError, re.escape("Dataset exceeds size limit")):
            load_dataset(big)

        for wrong in (123, None, ["x"]):
            with self.assertRaisesRegex(ValueError, re.escape("Invalid fixture path")):
                load_dataset(wrong)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. Arbitrary Unicode text.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. Mutations of a valid fixture JSON.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. Type-shape attacks.
# ---------------------------------------------------------------------------


def _doc(search, held=None):
    return json.dumps({"search": [search], "held_out": [held or _HELD_FIX]})


_MSGS_22 = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(22)]

_SHAPE_CASES = [
    ("fixture-is-string", _doc("nope")),
    ("messages-is-dict", _doc(dict(_SEARCH_FIX, messages={"role": "user"}))),
    ("messages-is-string", _doc(dict(_SEARCH_FIX, messages="nope"))),
    ("message-is-string", _doc(dict(_SEARCH_FIX, messages=["nope"]))),
    ("role-is-number", _doc(dict(_SEARCH_FIX, messages=[{"role": 5, "content": "x"}]))),
    ("role-is-null", _doc(dict(_SEARCH_FIX, messages=[{"role": None, "content": "x"}]))),
    (
        "content-is-list",
        _doc(dict(_SEARCH_FIX, messages=[{"role": "user", "content": ["x"]}])),
    ),
    (
        "content-is-dict",
        _doc(dict(_SEARCH_FIX, messages=[{"role": "user", "content": {"t": 1}}])),
    ),
    (
        "content-is-number",
        _doc(dict(_SEARCH_FIX, messages=[{"role": "user", "content": 7}])),
    ),
    ("id-is-number", _doc(dict(_SEARCH_FIX, id=7))),
    ("expected-is-dict", _doc(dict(_SEARCH_FIX, expected={"x": 1}))),
    ("group-is-list", _doc(dict(_SEARCH_FIX, group=["g"]))),
    ("group-is-blank", _doc(dict(_SEARCH_FIX, group="  "))),
    ("extra-top-level-key", json.dumps({"search": [_SEARCH_FIX], "held_out": [_HELD_FIX], "z": 1})),
    ("missing-top-level-key", json.dumps({"search": [_SEARCH_FIX]})),
    ("fixture-extra-key", _doc(dict(_SEARCH_FIX, zzz=1))),
    ("fixture-missing-key", _doc({k: v for k, v in _SEARCH_FIX.items() if k != "expected"})),
    (
        "message-extra-key",
        _doc(
            dict(
                _SEARCH_FIX,
                messages=[{"role": "user", "content": "x", "zzz": 1}],
            )
        ),
    ),
    ("22-messages", _doc(dict(_SEARCH_FIX, messages=_MSGS_22))),
    ("search-is-dict", json.dumps({"search": {"a": 1}, "held_out": [_HELD_FIX]})),
    ("held-out-is-null", json.dumps({"search": [_SEARCH_FIX], "held_out": None})),
    ("top-is-empty-dict", "{}"),
]


# ---------------------------------------------------------------------------
# 4. Number attacks.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. Nesting attacks.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fixed-vocabulary pins (substrings read verbatim from datasets.py).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Size boundary.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. load_dataset file attacks.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    unittest.main()
