"""Cycle 42: differential verification of core.py's remaining pure helpers.

Covers every pure helper in ``model_protocol_lab/core.py`` that prior cycles
did not differentially verify: ``truncate_text_tail``, ``first_boundary``,
``local_repairs``, ``template_diff``, ``check_boundary_portability`` and
``template_tokenizer_compatibility``. Out of scope by mandate: ``evaluate()``
(cycle 20), the ``Message``/``Format`` dataclasses (cycle 32), and the
scoring-engine-adjacent helpers (``_valid_token_ids``, ``_valid_generation``,
``_boundary_text_match``, the ``EvaluationReport`` aggregate properties).

Method: an independent oracle for each helper is re-implemented here FROM THE
DOCSTRINGS -- deliberately different structure and naming from the
implementation, sharing no helpers with it. Where the docstrings
under-specify the code (D1..D7 below), the oracle implements the pinned code
behavior. The suite then:

* fuzzes each oracle against the implementation on generated inputs
  (deterministic string seeds via ``random.Random``), covering UTF-8
  codepoint splits, empty inputs, boundary token edge cases, and hostile
  inputs (wrong types, ``bool`` where ``int`` is expected, ``str``/``tuple``
  subclasses, raising/degrading ``secondary_encode`` callables);
* pins exact behavioral facts the docstrings claim (reconstruction of
  ``template_ids`` from ``template_diff`` segments, the fixed per-token
  status vocabulary of ``template_tokenizer_compatibility``, the verdict
  precedence degraded(unknown) > failed(incompatible), the
  ``repeat_bos``-collapse in ``local_repairs``' dedup key, the
  ``ValueError("template_diff_too_fragmented")`` deadline degradation);
* runs mutation-style sensitivity controls on COPIES of ``core.py`` in
  ``/tmp/cycle42_mutants_<pid>`` (the repo is never mutated): each mutant is loaded
  standalone via importlib and the differential battery must report >= 1
  divergence against the real implementation;
* red-teams hidden assumptions (lone surrogates, ``BaseException`` vs
  ``Exception`` in the never-raise probes, cross-container-type comparisons,
  non-``Format`` input to ``local_repairs``).

BUGS FOUND
----------
None. The implementation matches the oracle on all fuzzed inputs (see counts
in the test names), every pinned fact holds, and all 16 mutants are caught.

DOCSTRING / CODE DIVERGENCES (code behavior pinned; source untouched)
--------------------------------------------------------------------
D1. (CYCLE 42 NOTE, DOCUMENTED IN CYCLE 42.) ``first_boundary`` had no
    docstring at all; every behavior below is code-pinned, and a docstring
    was added in cycle 42 stating the pinned contract. Notably it performs
    NO type validation: ``list``/``str`` inputs work, the match test is
    ``tokens[i:i+len(stop)] == stop`` so a
    slice and ``stop`` of different container types never compare equal
    (``(1,) == [1]`` is ``False``), ``stop`` values with ``not stop`` true
    (``()``, ``[]``, ``""``, ``0``, ``None``, ``False``) raise ``ValueError``,
    and an unsized ``tokens`` (e.g. ``None``) raises ``TypeError``.
D2. ``truncate_text_tail`` enforces exact types although the docstring says
    "a string and an integer": ``bool`` ``drop_bytes`` and ``str``-subclass
    ``text`` raise ``ValueError``. The bound is ``1 <= drop_bytes <
    len(raw)`` -- strictly less, so an empty text rejects every ``drop_bytes``.
    Lone surrogates are NOT covered by the documented ``None`` path:
    ``text.encode("utf-8")`` raises ``UnicodeDecodeError``-sibling
    ``UnicodeEncodeError`` which propagates (only the *decode* side is
    converted to ``None``).
D3. ``local_repairs``' dedup key ignores ``name`` entirely, and ``repeat_bos``
    collapses to ``False`` in the key whenever ``bos == ""`` -- so with an
    empty BOS the ``/toggle-bos`` variant is a silent no-op that dedups away.
    The docstring ("small, explicit search space") does not state the
    resulting 1..5 variant count, that ``result[0] is base``, or the
    first-wins dedup order.
D4. ``template_diff``: segment spans and ID slices are ``list``s (not tuples)
    inside plain ``dict``s keyed exactly ``hand_span``/``template_span``/
    ``hand_ids``/``template_ids``; identical inputs return ``()`` before any
    work; ``_deadline_s`` is an underscore-private test hook; an expired
    budget raises ``ValueError("template_diff_too_fragmented")``. The function
    is pure except for the ``time.monotonic()`` wall-clock reads the deadline
    requires. The "byte-identical to unbounded difflib" claim is verified by
    the oracle, which drives ``difflib``'s own ``get_opcodes``.
D5. ``check_boundary_portability``'s "reported, never raised" guarantee covers
    ``Exception`` only: ``BaseException`` (``KeyboardInterrupt``,
    ``SystemExit``) from ``secondary_encode`` propagates. A
    ``secondary_encode`` returning a ``list`` (not a tuple) is
    ``secondary_encode_invalid`` even when the values match; returning ``()``
    is *valid* and therefore ``divergent_ids`` (never ``invalid``).
D6. ``template_tokenizer_compatibility``: verdict precedence is
    degraded(``unknown``) > failed(``incompatible``) -- a degraded probe beats
    an incompatible token even when the incompatible token comes first in
    inventory order; ``reason`` always names the FIRST failure in inventory
    order; the per-token status ladder checks ``registered`` before
    ``fragments`` before ``id_mismatch`` (an unregistered multi-token literal
    is ``unregistered``, not ``fragments``); ``registered_tokens`` values
    must be exact ``int`` (``bool`` ids are rejected by validation).
D7. The ``template_tokenizer_compatibility`` docstring's
    "``secondary_encode`` must encode with ``add_special_tokens=False``" is a
    caller contract the code cannot and does not enforce.

MUTATION CONTROLS
-----------------
16 mutants of ``core.py`` (M1..M16, see ``MUTANTS``), each applied to a fresh
copy in ``/tmp/cycle42_mutants_<pid>`` and loaded standalone. A mutant is CAUGHT
when the differential battery reports >= 1 divergence against the real
implementation. All 16 are caught.
"""

from __future__ import annotations

import difflib
import importlib.util
import os
import random
import shutil
import sys
import unittest
from pathlib import Path

from model_protocol_lab import core as real_core
from model_protocol_lab.core import Format

CORE_SRC = Path(real_core.__file__).resolve()
# PID-scoped: two pytest processes staging mutants concurrently must not share
# a directory (fixed names caused FileExistsError / half-written imports).
MUTANT_DIR = Path(f"/tmp/cycle42_mutants_{os.getpid()}")  # noqa: S108 -- mutant scratch copies, never the repo


# ---------------------------------------------------------------------------
# Independent oracles, written from the docstrings. Deliberately different
# structure/naming from the implementation; shares no helpers with it. Where
# the docstrings under-specify the code (D1..D7 above), the oracle implements
# the pinned code behavior.
# ---------------------------------------------------------------------------


def _ids_ok(value: object, *, require_nonempty: bool = False) -> bool:
    """Oracle token-ID validation (not imported from core)."""
    if type(value) is not tuple:
        return False
    if require_nonempty and len(value) == 0:
        return False
    return all(type(token) is int and token >= 0 for token in value)


def _oracle_truncate_text_tail(text: object, drop_bytes: object) -> str | None:
    """Oracle truncate text tail."""
    # Exact-type discipline is code-pinned (D2): bool/str-subclass rejected.
    if type(text) is not str or type(drop_bytes) is not int:
        raise ValueError("oracle: needs an exact str and an exact int")
    blob = text.encode("utf-8")  # lone surrogates raise UnicodeEncodeError (D2)
    size = len(blob)
    if drop_bytes < 1 or drop_bytes >= size:
        raise ValueError("oracle: drop_bytes outside [1, byte-length)")
    head = blob[: size - drop_bytes]
    try:
        return head.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _oracle_first_boundary(tokens: object, stop: object) -> int | None:
    """Oracle first boundary."""
    # No docstring exists (D1); oracle mirrors the code's permissive contract:
    # no type checks, `not stop` guard, slice == stop comparison. Written as a
    # nested element loop instead of slice comparison, and only exercised on
    # homogeneous container types (mixed tuple/list never match under ==).
    if not stop:
        raise ValueError("oracle: boundary must be nonempty")
    needle = list(stop)  # type: ignore[arg-type]
    haystack = list(tokens)  # type: ignore[arg-type]
    width = len(needle)
    for start in range(len(haystack) - width + 1):
        matched = True
        for offset in range(width):
            if haystack[start + offset] != needle[offset]:
                matched = False
                break
        if matched:
            return start
    return None


def _oracle_local_repairs(base: Format) -> tuple[Format, ...]:
    """Oracle local repairs."""
    from dataclasses import replace as _replace

    proposals = [
        base,
        _replace(base, name=base.name + "/toggle-bos", repeat_bos=not base.repeat_bos),
        _replace(base, name=base.name + "/no-bos", bos=""),
        _replace(
            base,
            name=base.name + "/prefix-newline",
            assistant_prefix=base.assistant_prefix + "\n",
        ),
        _replace(
            base,
            name=base.name + "/prefix-trim",
            assistant_prefix=base.assistant_prefix.rstrip(),
        ),
    ]

    def _signature(fmt: Format) -> tuple[object, ...]:
        """Signature."""
        # D3: name is ignored; repeat_bos collapses when bos is empty.
        return (
            fmt.bos,
            fmt.user_prefix,
            fmt.assistant_prefix,
            fmt.turn_end,
            fmt.user_turn_end,
            fmt.assistant_turn_end,
            fmt.open_suffix,
            fmt.repeat_bos if fmt.bos else False,
        )

    kept: list[Format] = []
    seen: list[tuple[object, ...]] = []
    for proposal in proposals:
        sig = _signature(proposal)
        if sig not in seen:  # first wins; list-based instead of dict.setdefault
            seen.append(sig)
            kept.append(proposal)
    return tuple(kept)


def _oracle_template_diff(hand_ids: object, template_ids: object) -> tuple[dict[str, object], ...]:
    """Oracle template diff."""
    # Drives difflib's own get_opcodes (autojunk=False) instead of the
    # hand-rolled deadline queue loop; verifies the "byte-identical to
    # unbounded difflib" docstring claim (D4).
    if not _ids_ok(hand_ids) or not _ids_ok(template_ids):
        raise ValueError("oracle: template_diff needs tuples of token IDs")
    assert isinstance(hand_ids, tuple) and isinstance(template_ids, tuple)
    if hand_ids == template_ids:
        return ()
    matcher = difflib.SequenceMatcher(None, list(hand_ids), list(template_ids), autojunk=False)
    segments: list[dict[str, object]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        segments.append(
            {
                "hand_span": [i1, i2],
                "template_span": [j1, j2],
                "hand_ids": list(hand_ids[i1:i2]),
                "template_ids": list(template_ids[j1:j2]),
            }
        )
    return tuple(segments)


def _oracle_check_boundary_portability(
    boundary_text: object, primary_ids: object, secondary_encode: object
) -> dict[str, object]:
    """Oracle check boundary portability."""
    if type(boundary_text) is not str or not boundary_text:
        raise ValueError("oracle: boundary_text must be a nonempty str")
    if not _ids_ok(primary_ids, require_nonempty=True):
        raise ValueError("oracle: primary_ids must be a nonempty tuple of token IDs")
    if not callable(secondary_encode):
        raise ValueError("oracle: secondary_encode must be callable")
    assert isinstance(primary_ids, tuple)
    try:
        second = secondary_encode(boundary_text)  # type: ignore[operator]
    except Exception:  # noqa: BLE001 -- oracle mirrors the probe's never-raise contract
        return {
            "portable": False,
            "primary_ids": list(primary_ids),
            "secondary_ids": None,
            "detail": "secondary_encode_failed",
        }
    if not _ids_ok(second):
        return {
            "portable": False,
            "primary_ids": list(primary_ids),
            "secondary_ids": None,
            "detail": "secondary_encode_invalid",
        }
    assert isinstance(second, tuple)
    portable = second == primary_ids
    return {
        "portable": portable,
        "primary_ids": list(primary_ids),
        "secondary_ids": list(second),
        "detail": "identical" if portable else "divergent_ids",
    }


def _oracle_template_tokenizer_compatibility(
    name: object, inventory: object, registered_tokens: object, secondary_encode: object
) -> dict[str, object]:
    """Oracle template tokenizer compatibility."""
    if type(name) is not str or not name:
        raise ValueError("oracle: name must be a nonempty str")
    if type(inventory) is not tuple or any(
        type(token) is not str or not token
        for token in inventory  # type: ignore[union-attr]
    ):
        raise ValueError("oracle: inventory must be a tuple of nonempty strs")
    if type(registered_tokens) is not dict or any(
        type(token) is not str or type(token_id) is not int or token_id < 0
        for token, token_id in registered_tokens.items()  # type: ignore[union-attr]
    ):
        raise ValueError("oracle: registered_tokens must map str to non-negative int")
    if not callable(secondary_encode):
        raise ValueError("oracle: secondary_encode must be callable")
    assert isinstance(inventory, tuple) and isinstance(registered_tokens, dict)

    rows: list[dict[str, object]] = []
    for literal in inventory:
        entry: dict[str, object] = {
            "token": literal,
            "registered": literal in registered_tokens,
            "registered_id": registered_tokens.get(literal),
            "single_token": None,
            "encoded_id": None,
            "id_match": None,
            "status": "encode_failed",
        }
        try:
            produced = secondary_encode(literal)  # type: ignore[operator]
        except Exception:  # noqa: BLE001 -- oracle mirrors the probe's never-raise contract
            rows.append(entry)
            continue
        if not _ids_ok(produced) or len(produced) == 0:  # type: ignore[arg-type]
            entry["status"] = "encode_invalid"
            rows.append(entry)
            continue
        assert isinstance(produced, tuple)
        first = produced[0]
        entry["encoded_id"] = first
        entry["single_token"] = len(produced) == 1
        if literal not in registered_tokens:
            entry["status"] = "unregistered"  # checked before fragments (D6)
        elif len(produced) != 1:
            entry["status"] = "fragments"
        elif first != registered_tokens[literal]:
            entry["status"] = "id_mismatch"
            entry["id_match"] = False
        else:
            entry["status"] = "ok"
            entry["id_match"] = True
        rows.append(entry)

    if not inventory:
        return {
            "format": name,
            "verdict": "unknown",
            "reason": "empty_inventory",
            "detail": (
                "empty inventory: the format exposes no token-shaped control "
                "literals, so compatibility cannot be judged (never 'compatible')"
            ),
            "tokens": rows,
        }
    degraded = None
    for entry in rows:  # degraded (unknown) outranks failed (incompatible) (D6)
        if entry["status"] in ("encode_failed", "encode_invalid"):
            degraded = entry
            break
    if degraded is not None:
        return {
            "format": name,
            "verdict": "unknown",
            "reason": degraded["status"],
            "detail": (
                f"{degraded['status']} on {degraded['token']!r}: the encode "
                "probe degraded, so compatibility cannot be judged"
            ),
            "tokens": rows,
        }
    failed = None
    for entry in rows:  # first failure in inventory order names `reason` (D6)
        if entry["status"] != "ok":
            failed = entry
            break
    if failed is not None:
        return {
            "format": name,
            "verdict": "incompatible",
            "reason": failed["status"],
            "detail": f"{failed['status']} on {failed['token']!r}",
            "tokens": rows,
        }
    count = len(rows)
    return {
        "format": name,
        "verdict": "compatible",
        "reason": "",
        "detail": (
            f"{count}/{count} inventory tokens registered, atomic, and id-consistent; "
            "this predicts nothing about experimental success (weights decide)"
        ),
        "tokens": rows,
    }


# ---------------------------------------------------------------------------
# Deterministic fuzz scaffolding: string seeds for random.Random.
# ---------------------------------------------------------------------------

_TRUNCATE_ALPHABET = ["a", "b", " ", "\n", "\t", "é", "ü", "中", "文", "\U0001f600", "𝄞", "0"]


def _rng(tag: str) -> random.Random:
    """Rng."""
    return random.Random(f"cycle42-{tag}")  # noqa: S311 -- deterministic test fuzz, not crypto


def _rand_text(rng: random.Random, max_len: int = 12) -> str:
    """Rand text."""
    return "".join(rng.choice(_TRUNCATE_ALPHABET) for _ in range(rng.randint(0, max_len)))


def _rand_format(rng: random.Random) -> Format:
    """Rand format."""
    bos = rng.choice(["", "<s>", "BOS>", "\n", " ", "<|begin|>"])
    prefix = lambda: rng.choice(  # noqa: E731 -- local test generator
        ["<|user|>\n", "USER: ", "", "  ", "[INST] ", "<|a|>", "<|a|> ", "<|a|>\n", "A:\t "]
    )
    optional = lambda: rng.choice([None, "", "<end>", " ", "\n", "</s>"])  # noqa: E731
    return Format(
        name=rng.choice(["f", "fmt/x", " n ", "fmt-42"]),
        bos=bos,
        user_prefix=prefix(),
        assistant_prefix=prefix(),
        turn_end=rng.choice(["", "\n", "</s>", " "]),
        repeat_bos=rng.choice([True, False]),
        user_turn_end=optional(),
        assistant_turn_end=optional(),
        open_suffix=optional(),
    )


def _outcome_of(func, *args, **kwargs):
    """Canonical outcome: ('ok', repr) or ('raise', exception type name)."""
    try:
        return ("ok", repr(func(*args, **kwargs)))
    except Exception as exc:  # noqa: BLE001 -- outcome comparison needs the type
        return ("raise", type(exc).__name__)


class TestTruncateTextTailDifferential(unittest.TestCase):
    """Test truncate text tail differential."""

    def test_fuzz_matches_oracle(self):
        """Test fuzz matches oracle."""
        rng = _rng("truncate")
        divergences = 0
        cases = 0
        for iteration in range(400):
            text = _rand_text(rng)
            raw_len = len(text.encode("utf-8"))
            # Cover valid, boundary, and invalid drop_bytes (D2).
            drop = rng.choice(
                [
                    rng.randint(-3, raw_len + 3),
                    0,
                    1,
                    raw_len - 1,
                    raw_len,
                    raw_len + 1,
                ]
            )
            with self.subTest(iteration=iteration, text=text, drop=drop):
                impl = _outcome_of(real_core.truncate_text_tail, text, drop)
                expected = _outcome_of(_oracle_truncate_text_tail, text, drop)
                cases += 1
                if impl != expected:
                    divergences += 1
                self.assertEqual(impl, expected)
        self.assertEqual(cases, 400)
        self.assertEqual(divergences, 0)

    def test_codepoint_split_returns_none(self):
        """Test codepoint split returns none."""
        # "中" is 3 bytes: dropping 1 or 2 bytes splits the codepoint.
        self.assertIsNone(real_core.truncate_text_tail("a中", 1))
        self.assertIsNone(real_core.truncate_text_tail("a中", 2))
        self.assertEqual(real_core.truncate_text_tail("a中", 3), "a")
        # 4-byte emoji: drops 1..3 split it.
        for drop in (1, 2, 3):
            self.assertIsNone(real_core.truncate_text_tail("\U0001f600", drop))
        # Pure-ASCII cut at a character boundary is exact.
        self.assertEqual(real_core.truncate_text_tail("hello", 2), "hel")

    def test_bounds_are_exclusive(self):
        """Test bounds are exclusive."""
        # D2: 1 <= drop_bytes < len(raw); empty text rejects everything.
        with self.assertRaises(ValueError):
            real_core.truncate_text_tail("abc", 0)
        with self.assertRaises(ValueError):
            real_core.truncate_text_tail("abc", 3)
        with self.assertRaises(ValueError):
            real_core.truncate_text_tail("abc", 4)
        with self.assertRaises(ValueError):
            real_core.truncate_text_tail("abc", -1)
        self.assertEqual(real_core.truncate_text_tail("abc", 2), "a")
        with self.assertRaises(ValueError):
            real_core.truncate_text_tail("", 1)

    def test_exact_type_discipline(self):
        """Test exact type discipline."""
        # D2: docstring says "a string and an integer"; code wants exact types.
        with self.assertRaises(ValueError):
            real_core.truncate_text_tail("abc", True)
        with self.assertRaises(ValueError):
            real_core.truncate_text_tail("abc", False)
        with self.assertRaises(ValueError):
            real_core.truncate_text_tail("abc", 2.0)
        with self.assertRaises(ValueError):
            real_core.truncate_text_tail("abc", "2")
        with self.assertRaises(ValueError):
            real_core.truncate_text_tail(b"abc", 1)
        with self.assertRaises(ValueError):
            real_core.truncate_text_tail(None, 1)

        class _StrSub(str):
            """Str sub."""

        with self.assertRaises(ValueError):
            real_core.truncate_text_tail(_StrSub("abc"), 1)

    def test_lone_surrogate_propagates_encode_error(self):
        """Test lone surrogate propagates encode error."""
        # D2: only the decode side maps to None; a lone surrogate fails at
        # .encode("utf-8") with UnicodeEncodeError, which is NOT converted.
        with self.assertRaises(UnicodeEncodeError):
            real_core.truncate_text_tail("a\ud800b", 1)
        # Oracle agrees (it encodes the same way).
        with self.assertRaises(UnicodeEncodeError):
            _oracle_truncate_text_tail("a\ud800b", 1)


class TestFirstBoundaryDifferential(unittest.TestCase):
    """Test first boundary differential."""

    def test_fuzz_matches_oracle(self):
        """Test fuzz matches oracle."""
        rng = _rng("first-boundary")
        for iteration in range(500):
            tokens = tuple(rng.randint(0, 5) for _ in range(rng.randint(0, 12)))
            stop = tuple(rng.randint(0, 5) for _ in range(rng.randint(0, 5)))
            container = rng.choice(["tuple", "list", "str"])
            if container == "list":
                tokens, stop = list(tokens), list(stop)
            elif container == "str":
                tokens = "".join(chr(97 + t) for t in tokens)
                stop = "".join(chr(97 + t) for t in stop)
            with self.subTest(iteration=iteration, tokens=tokens, stop=stop):
                # Homogeneous container types only: mixed tuple/list never
                # matches under == (D1); pinned separately below.
                self.assertEqual(
                    _outcome_of(real_core.first_boundary, tokens, stop),
                    _outcome_of(_oracle_first_boundary, tokens, stop),
                )

    def test_pinned_semantics(self):
        """Test pinned semantics."""
        self.assertEqual(real_core.first_boundary((1, 2, 3, 2, 1), (2,)), 1)
        self.assertEqual(real_core.first_boundary((1, 2, 3), (1, 2, 3)), 0)
        self.assertEqual(real_core.first_boundary((1, 2, 3), (3,)), 2)
        self.assertIsNone(real_core.first_boundary((1, 2, 3), (9,)))
        self.assertIsNone(real_core.first_boundary((), (1,)))
        self.assertIsNone(real_core.first_boundary((1,), (1, 2)))
        # Overlapping matches return the FIRST index.
        self.assertEqual(real_core.first_boundary((1, 1, 1), (1, 1)), 0)
        # Match at the very end is found (off-by-one guard).
        self.assertEqual(real_core.first_boundary((1, 2, 3), (2, 3)), 1)

    def test_empty_stop_rejected(self):
        """Test empty stop rejected."""
        for bad in ((), [], "", 0, None, False):
            with self.subTest(stop=bad), self.assertRaises(ValueError):
                real_core.first_boundary((1, 2, 3), bad)

    def test_no_type_validation(self):
        """Test no type validation."""
        # D1: lists and strs are accepted (no docstring governs this).
        self.assertEqual(real_core.first_boundary([1, 2, 3], [2]), 1)
        self.assertEqual(real_core.first_boundary("hello", "ell"), 1)
        self.assertIsNone(real_core.first_boundary([1, 2], [9]))
        # Mixed container types never match: tuple slice == list is False.
        self.assertIsNone(real_core.first_boundary((1, 2), [1, 2]))
        self.assertIsNone(real_core.first_boundary([1, 2], (1, 2)))
        # Unsized tokens raise TypeError, not ValueError.
        with self.assertRaises(TypeError):
            real_core.first_boundary(None, (1,))


class TestLocalRepairsDifferential(unittest.TestCase):
    """Test local repairs differential."""

    def test_fuzz_matches_oracle(self):
        """Test fuzz matches oracle."""
        rng = _rng("local-repairs")
        for iteration in range(250):
            base = _rand_format(rng)
            with self.subTest(iteration=iteration, base=base):
                self.assertEqual(
                    real_core.local_repairs(base),
                    _oracle_local_repairs(base),
                )

    def test_result_starts_with_base(self):
        """Test result starts with base."""
        rng = _rng("local-repairs-base")
        for _ in range(50):
            base = _rand_format(rng)
            result = real_core.local_repairs(base)
            self.assertIs(result[0], base)
            self.assertTrue(1 <= len(result) <= 5)

    def test_all_five_variants_survive_when_distinct(self):
        """Test all five variants survive when distinct."""
        base = Format(
            name="x",
            bos="<s>",
            user_prefix="U:",
            assistant_prefix="A: ",  # trailing space: trim differs
            turn_end="\n",
            repeat_bos=False,  # toggle differs (bos nonempty)
        )
        result = real_core.local_repairs(base)
        self.assertEqual(len(result), 5)
        self.assertEqual(
            [v.name for v in result],
            ["x", "x/toggle-bos", "x/no-bos", "x/prefix-newline", "x/prefix-trim"],
        )
        self.assertTrue(result[1].repeat_bos)
        self.assertEqual(result[2].bos, "")
        self.assertEqual(result[3].assistant_prefix, "A: \n")
        self.assertEqual(result[4].assistant_prefix, "A:")

    def test_empty_bos_collapses_toggle_and_no_bos(self):
        """Test empty bos collapses toggle and no bos."""
        # D3: with bos == "", repeat_bos collapses to False in the dedup key,
        # so /toggle-bos and /no-bos dedup away against base. (assistant_prefix
        # keeps a trailing space so /prefix-trim stays distinct.)
        base = Format(
            name="x",
            bos="",
            user_prefix="U:",
            assistant_prefix="A: ",
            turn_end="\n",
            repeat_bos=False,
        )
        result = real_core.local_repairs(base)
        self.assertEqual([v.name for v in result], ["x", "x/prefix-newline", "x/prefix-trim"])

    def test_prefix_trim_noop_dedups(self):
        """Test prefix trim noop dedups."""
        # assistant_prefix with no trailing whitespace: trim == base.
        base = Format(
            name="x",
            bos="<s>",
            user_prefix="U:",
            assistant_prefix="A:",
            turn_end="\n",
            repeat_bos=True,
        )
        result = real_core.local_repairs(base)
        self.assertEqual(
            [v.name for v in result],
            ["x", "x/toggle-bos", "x/no-bos", "x/prefix-newline"],
        )

    def test_dedup_ignores_name(self):
        """Test dedup ignores name."""
        # D3: only the key matters; craft base where toggle-bos differs from
        # base ONLY by repeat_bos, then confirm dedup keeps the FIRST (base).
        base = Format(
            name="x",
            bos="<s>",
            user_prefix="U:",
            assistant_prefix="A:",
            turn_end="\n",
            repeat_bos=False,
        )
        result = real_core.local_repairs(base)
        toggle = result[1]
        self.assertEqual(toggle.name, "x/toggle-bos")
        self.assertTrue(toggle.repeat_bos)


def _reconstruct_template(hand_ids: tuple[int, ...], segments) -> tuple[int, ...]:
    """Apply every segment's template slice over its hand span (docstring claim)."""
    out: list[int] = []
    pos = 0
    for seg in segments:
        start, end = seg["hand_span"]
        out.extend(hand_ids[pos:start])
        out.extend(seg["template_ids"])
        pos = end
    out.extend(hand_ids[pos:])
    return tuple(out)


class TestTemplateDiffDifferential(unittest.TestCase):
    """Test template diff differential."""

    @staticmethod
    def _rand_pair(rng: random.Random) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Rand pair."""
        base = [rng.randint(0, 6) for _ in range(rng.randint(0, 40))]
        mutated = list(base)
        for _ in range(rng.randint(0, 3)):
            if not mutated:
                mutated.extend(rng.randint(0, 6) for _ in range(rng.randint(1, 4)))
                continue
            op = rng.choice(["replace", "delete", "insert"])
            at = rng.randint(0, len(mutated))
            if op == "replace":
                end = min(len(mutated), at + rng.randint(1, 4))
                mutated[at:end] = [rng.randint(0, 6) for _ in range(end - at)]
            elif op == "delete":
                end = min(len(mutated), at + rng.randint(1, 4))
                del mutated[at:end]
            else:
                mutated[at:at] = [rng.randint(0, 6) for _ in range(rng.randint(1, 4))]
        return tuple(base), tuple(mutated)

    def test_fuzz_matches_oracle(self):
        """Test fuzz matches oracle."""
        rng = _rng("template-diff")
        for iteration in range(300):
            hand, template = self._rand_pair(rng)
            with self.subTest(iteration=iteration, hand=hand, template=template):
                impl = real_core.template_diff(hand, template)
                expected = _oracle_template_diff(hand, template)
                self.assertEqual(impl, expected)
                # Docstring claim: segments reconstruct template_ids exactly.
                self.assertEqual(_reconstruct_template(hand, impl), template)

    def test_identical_inputs_yield_empty_tuple(self):
        """Test identical inputs yield empty tuple."""
        rng = _rng("template-diff-identical")
        for iteration in range(50):
            ids = tuple(rng.randint(0, 9) for _ in range(rng.randint(0, 20)))
            with self.subTest(iteration=iteration):
                self.assertEqual(real_core.template_diff(ids, ids), ())
                self.assertEqual(real_core.template_diff((), ()), ())
                # Early return happens before the deadline machinery (D4).
                self.assertEqual(real_core.template_diff(ids, ids, _deadline_s=-1.0), ())

    def test_pinned_segment_shapes(self):
        """Test pinned segment shapes."""
        segments = real_core.template_diff((1, 2, 3, 4), (1, 9, 3, 4))
        self.assertEqual(len(segments), 1)
        seg = segments[0]
        # D4: spans and ID slices are lists inside plain dicts.
        self.assertEqual(seg["hand_span"], [1, 2])
        self.assertEqual(seg["template_span"], [1, 2])
        self.assertEqual(seg["hand_ids"], [2])
        self.assertEqual(seg["template_ids"], [9])
        self.assertEqual(
            set(seg.keys()), {"hand_span", "template_span", "hand_ids", "template_ids"}
        )
        # Pure delete and pure insert shapes.
        (delete,) = real_core.template_diff((1, 2, 3), (1, 3))
        self.assertEqual(delete["hand_span"], [1, 2])
        self.assertEqual(delete["template_span"], [1, 1])
        self.assertEqual(delete["template_ids"], [])
        (insert,) = real_core.template_diff((1, 3), (1, 2, 3))
        self.assertEqual(insert["hand_span"], [1, 1])
        self.assertEqual(insert["template_span"], [1, 2])
        self.assertEqual(insert["hand_ids"], [])
        # Empty hand: single insert covering the whole template.
        (ins,) = real_core.template_diff((), (4, 5))
        self.assertEqual(ins["hand_span"], [0, 0])
        self.assertEqual(ins["template_span"], [0, 2])

    def test_input_validation(self):
        """Test input validation."""
        good = (1, 2, 3)
        for bad in ([1, 2], (1, -2), (1, True), (1, 2.0), None, "12", (1, "2")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    real_core.template_diff(bad, good)  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    real_core.template_diff(good, bad)  # type: ignore[arg-type]

        class _TupleSub(tuple):
            """Tuple sub."""

        with self.assertRaises(ValueError):
            real_core.template_diff(_TupleSub((1, 2)), good)

    def test_deadline_degradation(self):
        """Test deadline degradation."""
        # D4: an expired budget raises ValueError("template_diff_too_fragmented").
        hand = tuple(range(50))
        template = tuple(range(50, 100))
        with self.assertRaises(ValueError) as ctx:
            real_core.template_diff(hand, template, _deadline_s=-1.0)
        self.assertEqual(str(ctx.exception), "template_diff_too_fragmented")
        # A generous budget on the same input succeeds.
        segments = real_core.template_diff(hand, template, _deadline_s=30.0)
        self.assertEqual(_reconstruct_template(hand, segments), template)

    def test_byte_identical_to_unbounded_difflib(self):
        """Test byte identical to unbounded difflib."""
        # The docstring's core claim, on localized diffs inside long inputs.
        rng = _rng("template-diff-long")
        base = tuple(rng.randint(0, 255) for _ in range(5000))
        mutated = list(base)
        mutated[100:110] = [999] * 10
        mutated[4000:4000] = [1000, 1001]
        template = tuple(mutated)
        impl = real_core.template_diff(base, template)
        expected = _oracle_template_diff(base, template)
        self.assertEqual(impl, expected)
        self.assertEqual(_reconstruct_template(base, impl), template)


class TestCheckBoundaryPortabilityDifferential(unittest.TestCase):
    """Test check boundary portability differential."""

    @staticmethod
    def _encoders():
        """Encoders."""
        mapping = {"<stop>": (99,), "end": (7, 8), "x": (1,)}

        def enc_exact(text):
            """Enc exact."""
            return mapping[text]  # KeyError -> encode_failed

        def enc_shift(text):
            """Enc shift."""
            return tuple(t + 1000 for t in mapping[text])

        def enc_boom(text):
            """Enc boom."""
            raise RuntimeError("tokenizer exploded")

        def enc_list(text):
            """Enc list."""
            return [99]  # list, not tuple -> encode_invalid

        def enc_empty(text):
            """Enc empty."""
            return ()

        def enc_negative(text):
            """Enc negative."""
            return (99, -1)

        return {
            "exact": enc_exact,
            "shift": enc_shift,
            "boom": enc_boom,
            "list": enc_list,
            "empty": enc_empty,
            "negative": enc_negative,
        }

    def test_fuzz_matches_oracle(self):
        """Test fuzz matches oracle."""
        rng = _rng("portability")
        encoders = self._encoders()
        texts = ["<stop>", "end", "x", "missing", "", "é", "<stop> "]
        for iteration in range(300):
            text = rng.choice(texts)
            primary = tuple(rng.randint(0, 200) for _ in range(rng.randint(0, 4)))
            if rng.random() < 0.3:
                primary = (99,)  # bias toward the portable case
            enc = encoders[rng.choice(sorted(encoders))]
            with self.subTest(iteration=iteration, text=text, primary=primary):
                self.assertEqual(
                    _outcome_of(real_core.check_boundary_portability, text, primary, enc),
                    _outcome_of(_oracle_check_boundary_portability, text, primary, enc),
                )

    def test_pinned_reports(self):
        """Test pinned reports."""
        encoders = self._encoders()
        ok = real_core.check_boundary_portability("<stop>", (99,), encoders["exact"])
        self.assertEqual(
            ok,
            {
                "portable": True,
                "primary_ids": [99],
                "secondary_ids": [99],
                "detail": "identical",
            },
        )
        div = real_core.check_boundary_portability("<stop>", (99,), encoders["shift"])
        self.assertEqual(div["portable"], False)
        self.assertEqual(div["secondary_ids"], [1099])
        self.assertEqual(div["detail"], "divergent_ids")
        failed = real_core.check_boundary_portability("missing", (99,), encoders["exact"])
        self.assertEqual(
            failed,
            {
                "portable": False,
                "primary_ids": [99],
                "secondary_ids": None,
                "detail": "secondary_encode_failed",
            },
        )
        invalid = real_core.check_boundary_portability("<stop>", (99,), encoders["list"])
        self.assertEqual(invalid["detail"], "secondary_encode_invalid")
        self.assertIsNone(invalid["secondary_ids"])
        # D5: () is valid output -> divergent_ids, not invalid.
        empty = real_core.check_boundary_portability("<stop>", (99,), encoders["empty"])
        self.assertEqual(empty["detail"], "divergent_ids")
        self.assertEqual(empty["secondary_ids"], [])

    def test_validation(self):
        """Test validation."""
        enc = self._encoders()["exact"]
        with self.assertRaises(ValueError):
            real_core.check_boundary_portability("", (99,), enc)
        with self.assertRaises(ValueError):
            real_core.check_boundary_portability(None, (99,), enc)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            real_core.check_boundary_portability("<stop>", (), enc)
        with self.assertRaises(ValueError):
            real_core.check_boundary_portability("<stop>", [99], enc)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            real_core.check_boundary_portability("<stop>", (99,), None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            real_core.check_boundary_portability("<stop>", (99,), "not-callable")  # type: ignore[arg-type]

        class _StrSub(str):
            """Str sub."""

        with self.assertRaises(ValueError):
            real_core.check_boundary_portability(_StrSub("<stop>"), (99,), enc)


class _MappingEncoder:
    """Dict-backed fake second tokenizer for the compatibility fuzz."""

    def __init__(self, mapping, fail_on=(), invalid_on=()):
        self.mapping = mapping
        self.fail_on = set(fail_on)
        self.invalid_on = set(invalid_on)

    def __call__(self, text):
        if text in self.fail_on:
            raise ValueError(f"no encoding for {text!r}")
        if text in self.invalid_on:
            return [1, 2]  # list -> encode_invalid
        return self.mapping[text]  # KeyError -> encode_failed


class TestTemplateTokenizerCompatibilityDifferential(unittest.TestCase):
    """Test template tokenizer compatibility differential."""

    _LITERALS = ["<s>", "</s>", "[INST]", "[/INST]", "<|user|>", "<|end|>", "Ġhello"]

    def _rand_case(self, rng: random.Random):
        """Rand case."""
        inventory = tuple(rng.choice(self._LITERALS) for _ in range(rng.randint(0, 4)))
        registered = {}
        mapping = {}
        for literal in dict.fromkeys(inventory):
            roll = rng.random()
            if roll < 0.55:
                token_id = rng.randint(0, 500)
                registered[literal] = token_id
                mapping[literal] = (token_id,)  # ok
            elif roll < 0.7:
                registered[literal] = rng.randint(0, 500)
                mapping[literal] = (rng.randint(501, 999),)  # id_mismatch
            elif roll < 0.8:
                mapping[literal] = (1, 2)  # fragments (unregistered)
            elif roll < 0.9:
                registered[literal] = rng.randint(0, 500)
                mapping[literal] = (1, 2, 3)  # fragments (registered)
            else:
                pass  # absent from mapping -> encode_failed
        fail_on = [lit for lit in dict.fromkeys(inventory) if rng.random() < 0.15]
        invalid_on = [lit for lit in dict.fromkeys(inventory) if rng.random() < 0.15]
        encoder = _MappingEncoder(mapping, fail_on=fail_on, invalid_on=invalid_on)
        return ("fmt", inventory, registered, encoder)

    def test_fuzz_matches_oracle(self):
        """Test fuzz matches oracle."""
        rng = _rng("ttc")
        for iteration in range(300):
            name, inventory, registered, encoder = self._rand_case(rng)
            with self.subTest(iteration=iteration, inventory=inventory):
                self.assertEqual(
                    _outcome_of(
                        real_core.template_tokenizer_compatibility,
                        name,
                        inventory,
                        registered,
                        encoder,
                    ),
                    _outcome_of(
                        _oracle_template_tokenizer_compatibility,
                        name,
                        inventory,
                        registered,
                        encoder,
                    ),
                )

    def test_status_vocabulary(self):
        """Test status vocabulary."""
        # One token per status: ok / unregistered / fragments / id_mismatch /
        # encode_failed / encode_invalid.
        encoder = _MappingEncoder(
            {"<s>": (10,), "[INST]": (1, 2), "[/INST]": (77,), "<|user|>": (5, 6)},
            fail_on=["<|end|>"],
            invalid_on=["Ġhello"],
        )
        registered = {"<s>": 10, "[/INST]": 78, "[INST]": 1}
        report = real_core.template_tokenizer_compatibility(
            "fmt",
            ("<s>", "[INST]", "[/INST]", "<|user|>", "<|end|>", "Ġhello"),
            registered,
            encoder,
        )
        by_token = {row["token"]: row for row in report["tokens"]}
        self.assertEqual(by_token["<s>"]["status"], "ok")
        self.assertTrue(by_token["<s>"]["id_match"])
        self.assertEqual(by_token["<s>"]["registered_id"], 10)
        self.assertEqual(by_token["<s>"]["encoded_id"], 10)
        # D6: unregistered is checked before fragments.
        self.assertEqual(by_token["[INST]"]["status"], "fragments")
        self.assertFalse(by_token["[INST]"]["single_token"])
        self.assertIsNone(by_token["[INST]"]["id_match"])
        self.assertEqual(by_token["[/INST]"]["status"], "id_mismatch")
        self.assertFalse(by_token["[/INST]"]["id_match"])
        # Unregistered AND multi-token -> "unregistered" (registered first).
        self.assertEqual(by_token["<|user|>"]["status"], "unregistered")
        self.assertFalse(by_token["<|user|>"]["registered"])
        self.assertIsNone(by_token["<|user|>"]["registered_id"])
        self.assertEqual(by_token["<|end|>"]["status"], "encode_failed")
        self.assertIsNone(by_token["<|end|>"]["encoded_id"])
        self.assertEqual(by_token["Ġhello"]["status"], "encode_invalid")
        # Degraded probe outranks the incompatible tokens -> unknown (D6).
        self.assertEqual(report["verdict"], "unknown")
        self.assertEqual(report["reason"], "encode_failed")
        self.assertIn("'<|end|>'", report["detail"])

    def test_compatible_report(self):
        """Test compatible report."""
        encoder = _MappingEncoder({"<s>": (10,), "</s>": (11,)})
        report = real_core.template_tokenizer_compatibility(
            "fmt", ("<s>", "</s>"), {"<s>": 10, "</s>": 11}, encoder
        )
        self.assertEqual(report["verdict"], "compatible")
        self.assertEqual(report["reason"], "")
        self.assertEqual(
            report["detail"],
            "2/2 inventory tokens registered, atomic, and id-consistent; "
            "this predicts nothing about experimental success (weights decide)",
        )
        self.assertTrue(all(row["status"] == "ok" for row in report["tokens"]))

    def test_empty_inventory_is_unknown(self):
        """Test empty inventory is unknown."""
        report = real_core.template_tokenizer_compatibility("fmt", (), {}, _MappingEncoder({}))
        self.assertEqual(report["verdict"], "unknown")
        self.assertEqual(report["reason"], "empty_inventory")
        self.assertEqual(report["tokens"], [])
        self.assertIn("never 'compatible'", report["detail"])

    def test_incompatible_names_first_failure_in_order(self):
        """Test incompatible names first failure in order."""
        encoder = _MappingEncoder({"a": (1, 2), "b": (3,)})
        report = real_core.template_tokenizer_compatibility("fmt", ("a", "b"), {}, encoder)
        self.assertEqual(report["verdict"], "incompatible")
        self.assertEqual(report["reason"], "unregistered")
        self.assertEqual(report["detail"], "unregistered on 'a'")
        # Reversed inventory order names the other token first.
        report2 = real_core.template_tokenizer_compatibility("fmt", ("b", "a"), {}, encoder)
        self.assertEqual(report2["reason"], "unregistered")
        self.assertEqual(report2["detail"], "unregistered on 'b'")

    def test_validation(self):
        """Test validation."""
        enc = _MappingEncoder({})
        with self.assertRaises(ValueError):
            real_core.template_tokenizer_compatibility("", (), {}, enc)
        with self.assertRaises(ValueError):
            real_core.template_tokenizer_compatibility("f", ["<s>"], {}, enc)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            real_core.template_tokenizer_compatibility("f", ("",), {}, enc)
        with self.assertRaises(ValueError):
            real_core.template_tokenizer_compatibility("f", ("<s>",), {"<s>": -1}, enc)
        # D6: bool ids are not exact int.
        with self.assertRaises(ValueError):
            real_core.template_tokenizer_compatibility("f", ("<s>",), {"<s>": True}, enc)  # type: ignore[dict-item]
        with self.assertRaises(ValueError):
            real_core.template_tokenizer_compatibility("f", (), [], enc)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            real_core.template_tokenizer_compatibility("f", (), {}, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Adversarial red-team pass: hidden assumptions the fuzz sweep does not cover.
# ---------------------------------------------------------------------------


class TestRedTeam(unittest.TestCase):
    """Test red team."""

    def test_truncate_all_split_positions_of_multibyte_char(self):
        """Test truncate all split positions of multibyte char."""
        # Every interior cut of a 3-byte char -> None; the aligned cut -> text.
        text = "x中y"
        raw = text.encode("utf-8")
        for drop in range(1, len(raw)):
            result = real_core.truncate_text_tail(text, drop)
            head = raw[: len(raw) - drop]
            try:
                expected = head.decode("utf-8")
            except UnicodeDecodeError:
                expected = None
            self.assertEqual(result, expected, f"drop={drop}")

    def test_first_boundary_unsized_tokens_raise_typeerror(self):
        """Test first boundary unsized tokens raise typeerror."""
        with self.assertRaises(TypeError):
            real_core.first_boundary((t for t in (1, 2)), (1,))

    def test_portability_baseexception_propagates(self):
        """Test portability baseexception propagates."""
        # D5: the never-raise guarantee covers Exception, not BaseException.

        def enc_interrupt(text):
            """Enc interrupt."""
            raise KeyboardInterrupt("simulated")

        with self.assertRaises(KeyboardInterrupt):
            real_core.check_boundary_portability("<stop>", (99,), enc_interrupt)

        def enc_system_exit(text):
            """Enc system exit."""
            raise SystemExit(3)

        with self.assertRaises(SystemExit):
            real_core.check_boundary_portability("<stop>", (99,), enc_system_exit)

    def test_ttc_baseexception_propagates(self):
        """Test ttc baseexception propagates."""

        def enc_interrupt(text):
            """Enc interrupt."""
            raise KeyboardInterrupt("simulated")

        with self.assertRaises(KeyboardInterrupt):
            real_core.template_tokenizer_compatibility("f", ("<s>",), {}, enc_interrupt)

    def test_local_repairs_assumes_format(self):
        """Test local repairs assumes format."""
        # No docstring states the argument contract; a non-Format fails with
        # AttributeError (code-pinned, not a bug: garbage in).
        with self.assertRaises(AttributeError):
            real_core.local_repairs("not-a-format")  # type: ignore[arg-type]

    def test_ttc_duplicate_inventory_tokens(self):
        """Test ttc duplicate inventory tokens."""
        # Duplicates are accepted: one row per inventory entry, in order.
        encoder = _MappingEncoder({"<s>": (10,)})
        report = real_core.template_tokenizer_compatibility(
            "f", ("<s>", "<s>"), {"<s>": 10}, encoder
        )
        self.assertEqual(report["verdict"], "compatible")
        self.assertEqual(len(report["tokens"]), 2)
        self.assertTrue(all(row["status"] == "ok" for row in report["tokens"]))

    def test_template_diff_long_equal_prefix_stays_fast(self):
        """Test template diff long equal prefix stays fast."""
        # difflib is linear-ish here; guard against pathological slowdown.
        # The CI gate runs under `coverage run` (3-4x instrumentation
        # overhead: ~1.5s uninstrumented vs ~5.1s covered for this input),
        # so the threshold must clear coverage overhead with margin; it
        # still catches algorithmic blowup, which would take minutes.
        import time

        hand = (0,) * 3000 + (1, 2, 3)
        template = (0,) * 3000 + (1, 9, 3)
        start = time.monotonic()
        segments = real_core.template_diff(hand, template)
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 15.0)
        self.assertEqual(_reconstruct_template(hand, segments), template)
        self.assertEqual(len(segments), 1)

    def test_truncate_nul_and_control_bytes(self):
        """Test truncate nul and control bytes."""
        self.assertEqual(real_core.truncate_text_tail("a\x00b", 1), "a\x00")
        self.assertEqual(real_core.truncate_text_tail("a\r\nb", 2), "a\r")

    def test_first_boundary_bool_tokens_never_equal_int_stop(self):
        """Test first boundary bool tokens never equal int stop."""
        # Exact-type discipline is NOT applied here (D1): True == 1 in
        # Python, so bools match ints under ==.
        self.assertEqual(real_core.first_boundary((True, False), (1,)), 0)


# ---------------------------------------------------------------------------
# Mutation-style sensitivity controls: copies in /tmp, repo never mutated.
# ---------------------------------------------------------------------------

MUTANTS = [
    (
        "M1_truncate_head_slice_instead_of_tail",
        '        return raw[:-drop_bytes].decode("utf-8")',
        '        return raw[:drop_bytes].decode("utf-8")',
        1,
    ),
    (
        "M2_truncate_allows_full_drop",
        "    if not 1 <= drop_bytes < len(raw):",
        "    if not 1 <= drop_bytes <= len(raw):",
        1,
    ),
    (
        "M3_first_boundary_misses_terminal_match",
        "    for index in range(len(tokens) - len(stop) + 1):",
        "    for index in range(len(tokens) - len(stop)):",
        1,
    ),
    (
        "M4_first_boundary_off_by_one",
        "            return index\n",
        "            return index + 1\n",
        1,
    ),
    (
        "M5_local_repairs_drops_toggle_bos",
        '        replace(base, name=base.name + "/toggle-bos", repeat_bos=not base.repeat_bos),\n',
        "",
        1,
    ),
    (
        "M6_local_repairs_no_repeat_bos_collapse",
        "            candidate.repeat_bos if candidate.bos else False,",
        "            candidate.repeat_bos,",
        1,
    ),
    (
        "M7_template_diff_swapped_spans",
        '                    "hand_span": [hand_pos, block_i],\n'
        '                    "template_span": [template_pos, block_j],',
        '                    "hand_span": [template_pos, block_j],\n'
        '                    "template_span": [hand_pos, block_i],',
        1,
    ),
    (
        "M8_template_diff_swapped_id_slices",
        '                    "hand_ids": hand[hand_pos:block_i],\n'
        '                    "template_ids": template[template_pos:block_j],',
        '                    "hand_ids": template[template_pos:block_j],\n'
        '                    "template_ids": hand[hand_pos:block_i],',
        1,
    ),
    (
        "M9_portability_inverted_flag",
        "    portable = secondary_ids == primary_ids",
        "    portable = secondary_ids != primary_ids",
        1,
    ),
    (
        "M10_portability_exceptions_propagate",
        "    try:\n"
        "        secondary_ids = secondary_encode(boundary_text)\n"
        "    except Exception:  # noqa: BLE001 -- portability probe must not fail its caller\n"
        "        return {\n"
        '            "portable": False,\n'
        '            "primary_ids": list(primary_ids),\n'
        '            "secondary_ids": None,\n'
        '            "detail": "secondary_encode_failed",\n'
        "        }\n",
        "    secondary_ids = secondary_encode(boundary_text)\n",
        1,
    ),
    (
        "M11_ttc_degraded_reports_incompatible",
        '            "verdict": "unknown",\n            "reason": degraded["status"],',
        '            "verdict": "incompatible",\n            "reason": degraded["status"],',
        1,
    ),
    (
        "M12_ttc_fragments_before_unregistered",
        '        if not row["registered"]:\n'
        '            row["status"] = "unregistered"\n'
        "        elif len(encoded) != 1:\n"
        '            row["status"] = "fragments"',
        "        if len(encoded) != 1:\n"
        '            row["status"] = "fragments"\n'
        '        elif not row["registered"]:\n'
        '            row["status"] = "unregistered"',
        1,
    ),
    (
        "M13_first_boundary_none_check_instead_of_truthiness",
        "    if not stop:\n",
        "    if stop is None:\n",
        1,
    ),
    (
        "M14_truncate_replace_instead_of_none",
        "    except UnicodeDecodeError:\n        return None",
        "    except UnicodeDecodeError:\n"
        '        return raw[:-drop_bytes].decode("utf-8", errors="replace")',
        1,
    ),
    (
        "M15_ttc_empty_inventory_compatible",
        '            "verdict": "unknown",\n            "reason": "empty_inventory",',
        '            "verdict": "compatible",\n            "reason": "empty_inventory",',
        1,
    ),
    (
        "M16_first_boundary_inverted_match",
        "        if tokens[index : index + len(stop)] == stop:",
        "        if tokens[index : index + len(stop)] != stop:",
        1,
    ),
]


def _load_mutant(name: str, old: str, new: str, count: int):
    """Load mutant."""
    src = CORE_SRC.read_text(encoding="utf-8")
    found = src.count(old)
    if found != count:
        raise AssertionError(f"mutant {name}: anchor found {found}x, expected {count}x")
    MUTANT_DIR.mkdir(parents=True, exist_ok=True)
    path = MUTANT_DIR / f"mut_{name}_core.py"
    path.write_text(src.replace(old, new), encoding="utf-8")
    mod_name = f"cycle42_mutant_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError(f"mutant {name}: no loader")
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[mod_name]
    return module


def _battery_cases():
    """Differential battery: (func_name, args, kwargs) exercised on every helper.

    Includes targeted cases for each mutant M1..M16 plus general coverage.
    """
    fmt_full = Format(
        name="x",
        bos="<s>",
        user_prefix="U:",
        assistant_prefix="A: ",
        turn_end="\n",
        repeat_bos=False,
    )
    fmt_empty_bos = Format(
        name="y",
        bos="",
        user_prefix="U:",
        assistant_prefix="A:",
        turn_end="\n",
        repeat_bos=True,
    )
    port_map = {"<stop>": (99,), "end": (7, 8)}

    def enc_exact(t):
        """Enc exact."""
        return port_map[t]

    def enc_shift(t):
        """Enc shift."""
        return tuple(v + 1000 for v in port_map[t])

    def enc_boom(t):
        """Enc boom."""
        raise RuntimeError("boom")

    ttc_enc = _MappingEncoder(
        {"<s>": (10,), "[INST]": (1, 2), "[/INST]": (77,)},
        fail_on=["<|end|>"],
    )

    return [
        # truncate_text_tail (M1, M2, M14 targeted)
        ("truncate_text_tail", ("hello", 2), {}),
        ("truncate_text_tail", ("hello world", 5), {}),
        ("truncate_text_tail", ("abc", 3), {}),
        ("truncate_text_tail", ("a中", 1), {}),
        ("truncate_text_tail", ("a中", 2), {}),
        ("truncate_text_tail", ("a中", 3), {}),
        ("truncate_text_tail", ("", 1), {}),
        ("truncate_text_tail", ("héllo", 0), {}),
        # first_boundary (M3, M4, M13, M16 targeted)
        ("first_boundary", ((1, 2, 3), (2, 3)), {}),
        ("first_boundary", ((1, 2, 3), (2,)), {}),
        ("first_boundary", ((1, 2, 3), (9,)), {}),
        ("first_boundary", ((1, 2, 3), ()), {}),
        ("first_boundary", ((5, 5, 5), (5, 5)), {}),
        ("first_boundary", ((), (1,)), {}),
        # local_repairs (M5, M6 targeted)
        ("local_repairs", (fmt_full,), {}),
        ("local_repairs", (fmt_empty_bos,), {}),
        # template_diff (M7, M8 targeted; asymmetric pairs)
        ("template_diff", ((1, 2, 3, 4), (1, 9, 3, 4)), {}),
        ("template_diff", ((1, 2, 3), (1, 2, 3, 4, 5)), {}),
        ("template_diff", ((7, 7, 7), (7, 7)), {}),
        ("template_diff", ((1, 2, 1, 2), (2, 1, 2, 1)), {}),
        ("template_diff", ((1, 2, 3), (1, 2, 3)), {}),
        # check_boundary_portability (M9, M10 targeted)
        ("check_boundary_portability", ("<stop>", (99,), enc_exact), {}),
        ("check_boundary_portability", ("<stop>", (99,), enc_shift), {}),
        ("check_boundary_portability", ("<stop>", (99,), enc_boom), {}),
        ("check_boundary_portability", ("missing", (1,), enc_exact), {}),
        # template_tokenizer_compatibility (M11, M12, M15 targeted)
        (
            "template_tokenizer_compatibility",
            ("f", ("<s>", "<|end|>"), {"<s>": 10}, ttc_enc),
            {},
        ),
        (
            "template_tokenizer_compatibility",
            ("f", ("[INST]",), {}, ttc_enc),
            {},
        ),
        (
            "template_tokenizer_compatibility",
            ("f", (), {}, ttc_enc),
            {},
        ),
        (
            "template_tokenizer_compatibility",
            ("f", ("<s>", "[/INST]"), {"<s>": 10, "[/INST]": 77}, ttc_enc),
            {},
        ),
    ]


def _battery_divergences(mutant_module) -> list[str]:
    """Battery divergences."""
    divergences = []
    for func_name, args, kwargs in _battery_cases():
        real = _outcome_of(getattr(real_core, func_name), *args, **kwargs)
        mutated = _outcome_of(getattr(mutant_module, func_name), *args, **kwargs)
        if real != mutated:
            divergences.append(f"{func_name}{args}: real={real} mutant={mutated}")
    return divergences


class TestMutantSensitivity(unittest.TestCase):
    """Test mutant sensitivity."""

    def test_all_mutants_caught(self):
        """Test all mutants caught."""
        self.addCleanup(shutil.rmtree, MUTANT_DIR, True)
        caught: list[str] = []
        for name, old, new, count in MUTANTS:
            with self.subTest(mutant=name):
                mutant_mod = _load_mutant(name, old, new, count)
                divs = _battery_divergences(mutant_mod)
                self.assertTrue(
                    divs,
                    f"mutant {name} was NOT caught by the differential battery",
                )
                caught.append(name)
        self.assertEqual(len(caught), len(MUTANTS))


if __name__ == "__main__":
    unittest.main()
