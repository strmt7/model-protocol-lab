"""Cycle 32: differential verification of model_protocol_lab/core.py's pure data layer.

Covers the ``Message`` and ``Format`` dataclasses: constructor validation
invariants, default-fallback semantics (per-role turn terminators,
``open_suffix``), and the exact framing output of ``Format.render()``.

An independent oracle (``_OracleMessage`` / ``_OracleFormat``) is
re-implemented here FROM THE DOCSTRINGS and field comments -- deliberately
different structure and naming from the implementation, sharing no helpers
with it. The suite then:

* fuzzes the oracle against the implementation on generated inputs
  (deterministic string seeds), covering every ``Format`` field,
  empty-string vs ``None`` overrides, ``repeat_bos``, multi-turn
  alternation, and hostile inputs (wrong types, blank names, empty
  messages, non-alternating roles, non-tuple containers, subclass
  instances);
* renders real catalog families (``chatml``, ``llama3``, ``deepseek``,
  ``mistral``, ``glm4``, ``mistral_large``, ``kimi3`` -- imported from the
  real ``catalog.py``) through BOTH the oracle and the implementation;
* pins exact rendered strings for those families (1-turn and 3-turn);
* runs mutation-style sensitivity controls on COPIES of ``core.py`` in
  ``/tmp/cycle32_mutants_<pid>`` (the repo is never mutated) to prove the tests
  catch real bugs -- ``TestMutantSensitivity`` loads each mutated copy via
  importlib (``core.py`` is stdlib-only, so it loads standalone) and
  asserts the differential harness reports at least one divergence;
* red-teams hidden assumptions (exact-type discipline, ``str``-subclass
  asymmetry, frozen-ness, hashability, lone surrogates, huge contents).

BUGS FOUND
----------
None. The implementation matches the oracle on all fuzzed inputs (see
counts below), all pinned renders hold byte-exact, and all 15 mutants are
caught, with the docstring/code divergences documented below.

DOCSTRING / CODE DIVERGENCES (code behavior pinned; source untouched)
---------------------------------------------------------------------
The docstrings and the code agree on every framing/fallback rule; the
divergences are all under-specification -- the code enforces more than the
docstrings state. The oracle implements the code behavior in those places.

D1. ``Message`` has no docstring at all; its validation rules are
    code-pinned. The role check is equality-based
    (``role not in ("user", "assistant")``), so a ``str`` *subclass* equal
    to ``"user"`` is accepted, while ``content`` requires *exact* ``str``
    (``type(content) is not str``) -- an asymmetry no docstring mentions.
    Pinned by ``test_str_subclass_role_accepted`` and
    ``test_content_exact_str_required``.
D2. ``Format`` text-field validation: comments say "must be strings" but
    the code requires exact ``str`` (subclasses rejected); ``name`` must be
    non-blank where blank means ``name.strip()`` is empty -- i.e. ALL
    Unicode whitespace (``"\\u00a0"``, ``"\\u2003"``), not just ASCII.
    Pinned by ``test_text_fields_exact_str`` / ``test_blank_names_rejected``.
D3. ``render``'s docstring states the framing/fallback rules but not the
    acceptance rules: the container must be an *exact* ``tuple``
    (subclasses rejected), non-empty, of *exact* ``Message`` elements
    (subclasses rejected); roles must strictly alternate starting with
    ``user``; the final turn must be ``user``. Pinned by
    ``TestRenderValidation``.
D4. "repeat_bos means BOS before every message, including the open reply"
    is pinned literally: with ``repeat_bos=True`` the render contains
    exactly ``len(messages) + 1`` BOS chunks -- one immediately before
    each message frame and one before the open-reply suffix. Pinned by
    ``test_repeat_bos_synthetic_pin`` / ``test_repeat_bos_counts``.
D5. Empty-string vs ``None`` overrides: documented in the field comments
    and pinned behaviorally -- ``mistral_large`` sets ``open_suffix=""``
    and its renders end with the empty string (NOT ``assistant_prefix``);
    the 108-case override sweep pins ``""``-is-honored vs
    ``None``-falls-back for all three optional fields.

MUTATION CONTROLS
-----------------
15 mutants of ``core.py`` (M1..M15, see ``MUTANTS``), each applied to a
fresh copy in ``/tmp/cycle32_mutants_<pid>`` and loaded standalone. A mutant is
CAUGHT when the differential harness reports >= 1 divergence against it.
All 15 are caught.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import random
import shutil
import sys
import unittest
from pathlib import Path

from model_protocol_lab import catalog as catalog_mod
from model_protocol_lab import core as real_core

CORE_SRC = Path(real_core.__file__).resolve()
# PID-scoped: two pytest processes staging mutants concurrently must not share
# a directory (fixed names caused FileExistsError / half-written imports).
MUTANT_DIR = Path(f"/tmp/cycle32_mutants_{os.getpid()}")  # noqa: S108 -- mutant scratch copies, never the repo


# ---------------------------------------------------------------------------
# Independent oracle, written from the docstrings + field comments.
# Deliberately different structure/naming from the implementation; shares no
# helpers with it. Where the docstrings under-specify the code (D1..D4
# above), the oracle implements the pinned code behavior.
# ---------------------------------------------------------------------------


class _OracleMessage:
    """Oracle re-implementation of Message's contract."""

    __slots__ = ("role", "content")

    def __init__(self, role: object, content: object) -> None:
        # No docstring governs Message validation; code-pinned (D1): role is
        # an equality check, content must be exactly str.
        if role not in ("user", "assistant"):
            raise ValueError("oracle: role must be 'user' or 'assistant'")
        if type(content) is not str:
            raise ValueError("oracle: content must be exactly str")
        self.role = role
        self.content = content


class _OracleFormat:
    """Oracle re-implementation of Format's contract."""

    __slots__ = (
        "name",
        "bos",
        "user_prefix",
        "assistant_prefix",
        "turn_end",
        "repeat_bos",
        "user_turn_end",
        "assistant_turn_end",
        "open_suffix",
    )

    def __init__(
        self,
        name: object,
        bos: object,
        user_prefix: object,
        assistant_prefix: object,
        turn_end: object,
        repeat_bos: object = False,
        user_turn_end: object = None,
        assistant_turn_end: object = None,
        open_suffix: object = None,
    ) -> None:
        # Docstrings say "must be strings"; code pins exact str (D2).
        for label, value in (
            ("name", name),
            ("bos", bos),
            ("user_prefix", user_prefix),
            ("assistant_prefix", assistant_prefix),
            ("turn_end", turn_end),
        ):
            if type(value) is not str:
                raise ValueError(f"oracle: {label} must be exactly str")
        # Docstrings say "nonblank"; code pins name.strip() (D2).
        if not name.strip():  # type: ignore[union-attr]
            raise ValueError("oracle: name must be nonblank")
        if type(repeat_bos) is not bool:
            raise ValueError("oracle: repeat_bos must be exactly bool")
        # Field comments: None means "fall back"; "" is honored (D5).
        for label, value in (
            ("user_turn_end", user_turn_end),
            ("assistant_turn_end", assistant_turn_end),
            ("open_suffix", open_suffix),
        ):
            if value is not None and type(value) is not str:
                raise ValueError(f"oracle: {label} must be None or exactly str")
        self.name = name
        self.bos = bos
        self.user_prefix = user_prefix
        self.assistant_prefix = assistant_prefix
        self.turn_end = turn_end
        self.repeat_bos = repeat_bos
        self.user_turn_end = user_turn_end
        self.assistant_turn_end = assistant_turn_end
        self.open_suffix = open_suffix

    def spec_render(self, messages: object) -> str:
        """Render per the render() docstring + code-pinned acceptance rules."""
        # Docstring is silent on acceptance; code pins exact tuple,
        # non-empty, exact message elements (D3).
        if (
            type(messages) is not tuple
            or not messages
            or any(type(m) is not _OracleMessage for m in messages)
        ):
            raise ValueError("oracle: messages must be a nonempty tuple of messages")
        if messages[-1].role != "user":
            raise ValueError("oracle: final turn must be user")
        chunks = [self.bos]
        for index, message in enumerate(messages):
            # Docstring does not mention alternation; code pins it (D3).
            expected = "user" if index % 2 == 0 else "assistant"
            if message.role != expected:
                raise ValueError("oracle: roles must alternate starting with user")
            if index and self.repeat_bos:
                chunks.append(self.bos)
            chunks.append(self.user_prefix if message.role == "user" else self.assistant_prefix)
            chunks.append(message.content)
            # Docstring: per-role ends fall back to turn_end when unset (D5).
            role_end = self.user_turn_end if message.role == "user" else self.assistant_turn_end
            chunks.append(role_end if role_end is not None else self.turn_end)
        if self.repeat_bos:
            chunks.append(self.bos)
        # Docstring: open_suffix falls back to assistant_prefix when unset.
        tail = self.open_suffix if self.open_suffix is not None else self.assistant_prefix
        chunks.append(tail)
        return "".join(chunks)


# ---------------------------------------------------------------------------
# Differential harness: compare oracle vs implementation outcomes.
# Each probe returns a list of divergence descriptions (empty == match).
# ---------------------------------------------------------------------------


def _attempt(fn):
    """Run fn; return (accepted, value-or-exception-kind)."""
    try:
        return (True, fn())
    except ValueError:
        return (False, "ValueError")
    except Exception as exc:  # noqa: BLE001 -- harness must observe any exception kind
        return (False, f"{type(exc).__name__}")


class _StrSub(str):
    """A str subclass: rejected wherever the code demands exact str (D1, D2)."""


def _format_field_tuple(fmt) -> tuple:
    return (
        fmt.name,
        fmt.bos,
        fmt.user_prefix,
        fmt.assistant_prefix,
        fmt.turn_end,
        fmt.repeat_bos,
        fmt.user_turn_end,
        fmt.assistant_turn_end,
        fmt.open_suffix,
    )


# ---------------------------------------------------------------------------
# Fuzz input generators (deterministic string seeds).
# ---------------------------------------------------------------------------

_TEXT_POOL = [
    "",
    "a",
    " ",
    "\n",
    "\t",
    "<|im_start|>",
    "{content}",
    "%s",
    "\\n",
    "\x00",
    "héllo",
    "｜end｜",
    "\u200b",
    "\U0001f600",
    "x" * 64,
    "user",
    "assistant",
    "None",
    "[INST] ",
    "</s>",
]
_MAYBE_TEXT_POOL = [None, *_TEXT_POOL]
_VALID_NAMES = ["f", "g2", " chatml ", "ünï", "a/b", "x" * 40]
_BLANK_NAMES = ["", " ", "\t", "\n", "\r\n", " ", " "]
_HOSTILE_SCALARS = [123, 4.5, b"x", ["x"], ("x",), {"x": 1}, _StrSub("x")]


def _valid_format_kwargs() -> dict:
    return {
        "name": "f",
        "bos": "<bos>",
        "user_prefix": "<u>",
        "assistant_prefix": "<a>",
        "turn_end": "<e>",
    }


def _random_format_kwargs(rng: random.Random, hostile: bool = False) -> dict:
    """Random Format kwargs; hostile=True mixes in invalid values."""
    kw: dict = {
        "name": rng.choice(_VALID_NAMES),
        "bos": rng.choice(_TEXT_POOL),
        "user_prefix": rng.choice(_TEXT_POOL),
        "assistant_prefix": rng.choice(_TEXT_POOL),
        "turn_end": rng.choice(_TEXT_POOL),
        "repeat_bos": rng.choice([True, False]),
        "user_turn_end": rng.choice(_MAYBE_TEXT_POOL),
        "assistant_turn_end": rng.choice(_MAYBE_TEXT_POOL),
        "open_suffix": rng.choice(_MAYBE_TEXT_POOL),
    }
    if hostile and rng.random() < 0.6:
        field = rng.choice(
            [
                "name",
                "bos",
                "user_prefix",
                "assistant_prefix",
                "turn_end",
                "user_turn_end",
                "assistant_turn_end",
                "open_suffix",
            ]
        )
        if field == "name":
            kw[field] = rng.choice([*_BLANK_NAMES, *_HOSTILE_SCALARS])
        elif field in ("user_turn_end", "assistant_turn_end", "open_suffix"):
            # None is legal for the optionals; "" is honored (D5).
            kw[field] = rng.choice(_HOSTILE_SCALARS)
        else:
            kw[field] = rng.choice([*_HOSTILE_SCALARS, None])
        if rng.random() < 0.3:
            kw["repeat_bos"] = rng.choice([1, 0, "yes", None, 2.5])
    return kw


def _random_message_specs(
    rng: random.Random, n: int, break_alternation: bool = False
) -> list[tuple[str, str]]:
    """Message (role, content) specs; roles are always valid strings here."""
    specs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        if break_alternation and rng.random() < 0.35:
            role = rng.choice(["user", "assistant"])
        specs.append((role, rng.choice(_TEXT_POOL)))
    return specs


def _message_construction_probes(core_mod) -> list[tuple]:
    """Fixed hostile (role, content) pairs; impl and oracle must agree."""
    return [
        ("user", "x"),
        ("assistant", ""),
        ("User", "x"),
        ("SYSTEM", "x"),
        ("", "x"),
        (None, "x"),
        (0, "x"),
        (True, "x"),
        (b"user", "x"),
        (["user"], "x"),
        # str-subclass role: ACCEPTED by the code's equality check (D1).
        (_StrSub("user"), "x"),
        (_StrSub("assistant"), "y"),
        # Content must be exactly str (D1).
        ("user", _StrSub("x")),
        ("user", b"x"),
        ("user", None),
        ("user", 123),
        ("user", ["x"]),
        ("user", ("x",)),
        ("user", bytearray(b"x")),
    ]


def _format_construction_probes() -> list[dict]:
    """Fixed hostile Format kwargs; impl and oracle must agree."""
    base = _valid_format_kwargs()
    probes = []
    for bad_name in [*_BLANK_NAMES, 123, None, b"n", _StrSub("n"), ["n"]]:
        probe = dict(base)
        probe["name"] = bad_name
        probes.append(probe)
    for field in ("bos", "user_prefix", "assistant_prefix", "turn_end"):
        for bad in [*_HOSTILE_SCALARS, None]:
            probe = dict(base)
            probe[field] = bad
            probes.append(probe)
    for bad in [1, 0, "yes", None, 2.5, _StrSub("t")]:
        probe = dict(base)
        probe["repeat_bos"] = bad
        probes.append(probe)
    for field in ("user_turn_end", "assistant_turn_end", "open_suffix"):
        for bad in _HOSTILE_SCALARS:
            probe = dict(base)
            probe[field] = bad
            probes.append(probe)
        # None and "" are both legal (D5); "" must NOT fall back.
        for good in (None, ""):
            probe = dict(base)
            probe[field] = good
            probes.append(probe)
    return probes


def _render_container_probes(core_mod, fmt) -> list:
    """Fixed hostile render() containers; impl and oracle must agree."""
    msg = core_mod.Message("user", "x")

    class _MessageSub(core_mod.Message):
        pass

    class _TupleSub(tuple):
        pass

    return [
        None,
        [],
        (),
        [msg],
        _TupleSub((msg,)),
        (x for x in (msg,)),
        "usermsg",
        (None,),
        ("x",),
        (msg, "x"),
        (_MessageSub("user", "x"),),
        (msg, msg),  # (user, user): alternation violation
        (core_mod.Message("assistant", "y"),),  # final turn not user
        (
            core_mod.Message("user", "a"),
            core_mod.Message("assistant", "b"),
        ),  # 2-turn: final not user
    ]


# ---------------------------------------------------------------------------
# Divergence collectors: [] means oracle and implementation agree.
# ---------------------------------------------------------------------------


def _construction_divergences(core_mod, seed: str, trials: int) -> list[str]:
    """Format + Message construction parity: accept/reject and field values."""
    rng = random.Random(seed)  # noqa: S311 -- deterministic test seed
    divergences: list[str] = []
    cases = _format_construction_probes()
    cases += [_random_format_kwargs(rng, hostile=(i % 2 == 0)) for i in range(trials)]
    for kw in cases:
        impl_ok, impl_val = _attempt(lambda kw=kw: core_mod.Format(**kw))
        spec_ok, spec_val = _attempt(lambda kw=kw: _OracleFormat(**kw))
        if (impl_ok, impl_val if not impl_ok else None) != (
            spec_ok,
            spec_val if not spec_ok else None,
        ):
            divergences.append(f"Format construction mismatch for {kw!r}: {impl_ok}/{impl_val}")
        elif impl_ok and _format_field_tuple(impl_val) != _format_field_tuple(spec_val):
            divergences.append(f"Format field mismatch for {kw!r}")
    for role, content in _message_construction_probes(core_mod):
        impl_ok, impl_val = _attempt(
            lambda role=role, content=content: core_mod.Message(role, content)
        )
        spec_ok, spec_val = _attempt(
            lambda role=role, content=content: _OracleMessage(role, content)
        )
        if (impl_ok, impl_val if not impl_ok else None) != (
            spec_ok,
            spec_val if not spec_ok else None,
        ):
            divergences.append(f"Message construction mismatch for {(role, content)!r}")
        elif impl_ok and (impl_val.role, impl_val.content) != (spec_val.role, spec_val.content):
            divergences.append(f"Message field mismatch for {(role, content)!r}")
    return divergences


def _render_one(core_mod, fmt_kwargs: dict, msg_specs: list[tuple[str, str]]) -> str | None:
    """Single render-parity probe; returns a divergence description or None."""
    impl_ok, impl_fmt = _attempt(lambda: core_mod.Format(**fmt_kwargs))
    spec_ok, spec_fmt = _attempt(lambda: _OracleFormat(**fmt_kwargs))
    if impl_ok != spec_ok:
        return f"construction accept mismatch: {fmt_kwargs!r}"
    if not impl_ok:
        return None
    impl_msgs, spec_msgs = [], []
    for role, content in msg_specs:
        i_ok, i_m = _attempt(lambda role=role, content=content: core_mod.Message(role, content))
        s_ok, s_m = _attempt(lambda role=role, content=content: _OracleMessage(role, content))
        if i_ok != s_ok or (i_ok and (i_m.role, i_m.content) != (s_m.role, s_m.content)):
            return f"message mismatch: {(role, content)!r}"
        impl_msgs.append(i_m)
        spec_msgs.append(s_m)
    i_ok, i_out = _attempt(lambda: impl_fmt.render(tuple(impl_msgs)))
    s_ok, s_out = _attempt(lambda: spec_fmt.spec_render(tuple(spec_msgs)))
    if i_ok != s_ok:
        return f"render accept mismatch impl={i_ok}({i_out}) spec={s_ok}({s_out}): {msg_specs!r}"
    if i_ok and i_out != s_out:
        return f"render output mismatch:\n  impl {i_out!r}\n  spec {s_out!r}\n  fmt={fmt_kwargs!r}"
    return None


def _render_divergences(core_mod, seed: str, trials: int) -> list[str]:
    """Render parity on random formats x random conversations + container probes."""
    rng = random.Random(seed)  # noqa: S311 -- deterministic test seed
    divergences: list[str] = []
    for trial in range(trials):
        kw = _random_format_kwargs(rng, hostile=(trial % 3 == 0))
        specs = _random_message_specs(rng, rng.randint(1, 5), break_alternation=(trial % 2 == 0))
        div = _render_one(core_mod, kw, specs)
        if div is not None:
            divergences.append(f"trial {trial}: {div}")
    fmt = core_mod.Format(**_valid_format_kwargs())
    for container in _render_container_probes(core_mod, fmt):
        i_ok, i_out = _attempt(lambda container=container: fmt.render(container))
        oracle_msgs: list = []
        oracle_build_ok = True
        if type(container) is tuple:
            for element in container:
                if type(element) is not core_mod.Message:
                    oracle_build_ok = False
                    break
                # Rebuild the equivalent oracle message from the impl one.
                try:
                    oracle_msgs.append(_OracleMessage(element.role, element.content))
                except ValueError:
                    oracle_build_ok = False
                    break
        else:
            oracle_build_ok = False
        if not oracle_build_ok:
            s_ok, s_out = (False, "ValueError")
        else:
            oracle_tuple = tuple(oracle_msgs)
            oracle_fmt = _OracleFormat(**_valid_format_kwargs())
            s_ok, s_out = _attempt(lambda fmt=oracle_fmt, msgs=oracle_tuple: fmt.spec_render(msgs))
        if (i_ok, i_out if not i_ok else None) != (s_ok, s_out if not s_ok else None):
            divergences.append(f"container probe mismatch: {type(container).__name__}")
        elif i_ok and i_out != s_out:
            divergences.append(f"container probe output mismatch: {type(container).__name__}")
    return divergences


def _override_sweep_divergences(core_mod) -> list[str]:
    """Empty-string vs None override sweep: 3^3 optionals x repeat_bos x turns."""
    divergences: list[str] = []
    optionals = [None, "", "<X>"]
    turn_sets = [
        [("user", "U")],
        [("user", "U1"), ("assistant", "A1"), ("user", "U2")],
    ]
    for user_end in optionals:
        for asst_end in optionals:
            for open_sfx in optionals:
                for repeat in (False, True):
                    for specs in turn_sets:
                        kw = {
                            **_valid_format_kwargs(),
                            "user_turn_end": user_end,
                            "assistant_turn_end": asst_end,
                            "open_suffix": open_sfx,
                            "repeat_bos": repeat,
                        }
                        div = _render_one(core_mod, kw, specs)
                        if div is not None:
                            divergences.append(
                                f"override u={user_end!r} a={asst_end!r} o={open_sfx!r} "
                                f"rb={repeat} n={len(specs)}: {div}"
                            )
    return divergences


def _catalog_divergences(core_mod, catalog: dict) -> list[str]:
    """Every real catalog family through BOTH oracle and implementation."""
    divergences: list[str] = []
    rng = random.Random("cycle32-catalog")  # noqa: S311 -- deterministic test seed
    for name, fmt in catalog.items():
        oracle_fmt = _OracleFormat(
            fmt.name,
            fmt.bos,
            fmt.user_prefix,
            fmt.assistant_prefix,
            fmt.turn_end,
            repeat_bos=fmt.repeat_bos,
            user_turn_end=fmt.user_turn_end,
            assistant_turn_end=fmt.assistant_turn_end,
            open_suffix=fmt.open_suffix,
        )
        for trial in range(4):
            specs = _random_message_specs(
                rng, rng.choice([1, 3, 5]), break_alternation=(trial % 2 == 0)
            )
            impl_msgs = tuple(core_mod.Message(r, c) for r, c in specs)
            spec_msgs = tuple(_OracleMessage(r, c) for r, c in specs)
            i_ok, i_out = _attempt(lambda fmt=fmt, msgs=impl_msgs: fmt.render(msgs))
            s_ok, s_out = _attempt(lambda fmt=oracle_fmt, msgs=spec_msgs: fmt.spec_render(msgs))
            if i_ok != s_ok or (i_ok and i_out != s_out):
                divergences.append(f"catalog {name} trial {trial}: {specs!r}")
    return divergences


# ---------------------------------------------------------------------------
# Test suites.
# ---------------------------------------------------------------------------


class TestOracleDifferentialFuzz(unittest.TestCase):
    """Oracle vs implementation agreement on generated inputs."""

    def test_fuzz_construction(self):
        for seed in (f"cycle32-construct-{i}" for i in range(8)):
            with self.subTest(seed=seed):
                divs = _construction_divergences(real_core, seed, 30)
                self.assertEqual(divs, [], "divergences:\n" + "\n".join(divs[:5]))

    def test_fuzz_render(self):
        for seed in (f"cycle32-render-{i}" for i in range(8)):
            with self.subTest(seed=seed):
                divs = _render_divergences(real_core, seed, 40)
                self.assertEqual(divs, [], "divergences:\n" + "\n".join(divs[:5]))

    def test_fuzz_override_sweep(self):
        # 3^3 optionals x repeat_bos x {1-turn, 3-turn} = 108 deterministic cases.
        divs = _override_sweep_divergences(real_core)
        self.assertEqual(divs, [], "divergences:\n" + "\n".join(divs[:5]))

    def test_fuzz_catalog(self):
        # Every real catalog family through BOTH oracle and implementation.
        divs = _catalog_divergences(real_core, catalog_mod.CATALOG)
        self.assertEqual(divs, [], "divergences:\n" + "\n".join(divs[:5]))

    def test_catalog_has_expected_families(self):
        for name in (
            "chatml",
            "llama3",
            "deepseek",
            "mistral",
            "glm4",
            "mistral_large",
            "kimi3",
        ):
            with self.subTest(family=name):
                self.assertIn(name, catalog_mod.CATALOG)


class TestPinnedRenders(unittest.TestCase):
    """Byte-exact pinned renders for real catalog families (1-turn + 3-turn)."""

    def _check(self, family: str, expected_1t: str, expected_3t: str):
        fmt = catalog_mod.CATALOG[family]
        one = (real_core.Message("user", "Hello"),)
        three = (
            real_core.Message("user", "A"),
            real_core.Message("assistant", "B"),
            real_core.Message("user", "C"),
        )
        self.assertEqual(fmt.render(one), expected_1t, f"{family} 1-turn")
        self.assertEqual(fmt.render(three), expected_3t, f"{family} 3-turn")

    def test_chatml(self):
        self._check(
            "chatml",
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
            "<|im_start|>user\nA<|im_end|>\n"
            "<|im_start|>assistant\nB<|im_end|>\n"
            "<|im_start|>user\nC<|im_end|>\n"
            "<|im_start|>assistant\n",
        )

    def test_llama3(self):
        self._check(
            "llama3",
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "A<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            "B<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "C<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    def test_deepseek(self):
        # Per-role ends: user turns unterminated (""), assistant ends with
        # <end of sentence>; open suffix falls back to assistant_prefix.
        self._check(
            "deepseek",
            "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>"
            "Hello<\uff5cAssistant\uff5c>",
            "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>"
            "A<\uff5cAssistant\uff5c>B<\uff5cend\u2581of\u2581sentence\uff5c>"
            "<\uff5cUser\uff5c>C<\uff5cAssistant\uff5c>",
        )

    def test_mistral(self):
        self._check(
            "mistral",
            "<s>[INST] Hello [/INST]",
            "<s>[INST] A [/INST]B</s>[INST] C [/INST]",
        )

    def test_glm4(self):
        # open_suffix="<|assistant|>" (bare, no frame newline) overrides the
        # assistant_prefix fallback.
        self._check(
            "glm4",
            "[gMASK]<sop><|user|>\nHello<|assistant|>",
            "[gMASK]<sop><|user|>\nA<|assistant|>\nB<|user|>\nC<|assistant|>",
        )

    def test_mistral_large(self):
        # open_suffix="" is honored (D5): the render ends with the empty
        # string, NOT with assistant_prefix (" ").
        self._check(
            "mistral_large",
            "<s>[INST] Hello[/INST]",
            "<s>[INST] A[/INST] B</s>[INST] C[/INST]",
        )
        fmt = catalog_mod.CATALOG["mistral_large"]
        self.assertEqual(fmt.open_suffix, "")
        self.assertNotEqual(fmt.assistant_prefix, "")

    def test_kimi3(self):
        self._check(
            "kimi3",
            '<|open|>message role="user"<|sep|>Hello<|close|>message<|sep|>'
            '<|end_of_msg|><|open|>message role="assistant"<|sep|><|open|>think<|sep|>',
            '<|open|>message role="user"<|sep|>A<|close|>message<|sep|><|end_of_msg|>'
            '<|open|>message role="assistant"<|sep|><|open|>think<|sep|>'
            "<|close|>think<|sep|><|open|>response<|sep|>B<|close|>response<|sep|>"
            "<|close|>message<|sep|><|end_of_msg|>"
            '<|open|>message role="user"<|sep|>C<|close|>message<|sep|><|end_of_msg|>'
            '<|open|>message role="assistant"<|sep|><|open|>think<|sep|>',
        )

    def test_repeat_bos_synthetic_pin(self):
        # Docstring: "repeat_bos means BOS before every message, including
        # the open reply" (D4) -- pinned literally.
        fmt = real_core.Format("rb", "<B>", "[U]", "[A]", "<E>", repeat_bos=True)
        one = (real_core.Message("user", "H"),)
        three = (
            real_core.Message("user", "A"),
            real_core.Message("assistant", "B"),
            real_core.Message("user", "C"),
        )
        self.assertEqual(fmt.render(one), "<B>[U]H<E><B>[A]")
        self.assertEqual(fmt.render(three), "<B>[U]A<E><B>[A]B<E><B>[U]C<E><B>[A]")
        # Exactly len(messages)+1 BOS chunks, one before each frame + open reply.
        self.assertEqual(fmt.render(three).count("<B>"), 4)

    def test_repeat_bos_false_contrast(self):
        fmt = real_core.Format("norb", "<B>", "[U]", "[A]", "<E>", repeat_bos=False)
        one = (real_core.Message("user", "H"),)
        self.assertEqual(fmt.render(one), "<B>[U]H<E>[A]")
        self.assertEqual(fmt.render(one).count("<B>"), 1)

    def test_repeat_bos_with_empty_bos_is_noop(self):
        fmt = real_core.Format("emptybos", "", "[U]", "[A]", "<E>", repeat_bos=True)
        one = (real_core.Message("user", "H"),)
        self.assertEqual(fmt.render(one), "[U]H<E>[A]")


class TestMessageValidation(unittest.TestCase):
    """Adversarial probes of Message's constructor (D1)."""

    def test_valid_messages(self):
        cases = [("user", ""), ("user", "x"), ("assistant", ""), ("assistant", "y" * 1000)]
        for role, content in cases:
            with self.subTest(role=role, content=content[:8]):
                msg = real_core.Message(role, content)
                self.assertEqual((msg.role, msg.content), (role, content))

    def test_invalid_roles_rejected(self):
        bad_roles = [
            "User",
            "USER",
            "system",
            "",
            " user",
            "user ",
            None,
            0,
            1,
            True,
            False,
            b"user",
            ["user"],
        ]
        for bad in bad_roles:
            with self.subTest(role=bad), self.assertRaises(ValueError):
                real_core.Message(bad, "x")

    def test_str_subclass_role_accepted(self):
        # Code-pinned (D1): the role check is equality-based, NOT exact-type.
        msg = real_core.Message(_StrSub("user"), "x")
        self.assertEqual(msg.role, "user")
        self.assertIs(type(msg.role), _StrSub)

    def test_content_exact_str_required(self):
        bad_contents = [
            _StrSub("x"),
            b"x",
            None,
            123,
            4.5,
            ["x"],
            ("x",),
            bytearray(b"x"),
            {"x": 1},
        ]
        for bad in bad_contents:
            with self.subTest(content=type(bad).__name__), self.assertRaises(ValueError):
                real_core.Message("user", bad)

    def test_message_is_frozen(self):
        msg = real_core.Message("user", "x")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            msg.content = "y"  # type: ignore[misc]

    def test_message_equality_and_hash(self):
        self.assertEqual(real_core.Message("user", "x"), real_core.Message("user", "x"))
        self.assertNotEqual(real_core.Message("user", "x"), real_core.Message("assistant", "x"))
        self.assertEqual(len({real_core.Message("user", "x"), real_core.Message("user", "x")}), 1)


class TestFormatValidation(unittest.TestCase):
    """Adversarial probes of Format's constructor (D2, D5)."""

    @staticmethod
    def _base() -> dict:
        return {
            "name": "f",
            "bos": "<b>",
            "user_prefix": "<u>",
            "assistant_prefix": "<a>",
            "turn_end": "<e>",
        }

    def test_valid_format(self):
        fmt = real_core.Format(**self._base())
        self.assertEqual(_format_field_tuple(fmt)[:5], ("f", "<b>", "<u>", "<a>", "<e>"))
        self.assertIs(fmt.repeat_bos, False)
        self.assertIsNone(fmt.user_turn_end)
        self.assertIsNone(fmt.assistant_turn_end)
        self.assertIsNone(fmt.open_suffix)

    def test_text_fields_exact_str(self):
        for field in ("name", "bos", "user_prefix", "assistant_prefix", "turn_end"):
            for bad in [123, 4.5, None, b"x", ["x"], ("x",), _StrSub("x")]:
                with self.subTest(field=field, bad=type(bad).__name__):
                    kw = self._base()
                    kw[field] = bad
                    with self.assertRaises(ValueError):
                        real_core.Format(**kw)

    def test_blank_names_rejected(self):
        for blank in ["", " ", "\t", "\n", "\r\n", " ", " ", " "]:
            with self.subTest(name=repr(blank)):
                kw = self._base()
                kw["name"] = blank
                with self.assertRaises(ValueError):
                    real_core.Format(**kw)

    def test_nonblank_names_accepted(self):
        for name in ["f", " x ", "ünï", "a/b", "x" * 200, "0"]:
            with self.subTest(name=name[:12]):
                kw = self._base()
                kw["name"] = name
                self.assertEqual(real_core.Format(**kw).name, name)

    def test_repeat_bos_exact_bool(self):
        for bad in [1, 0, "yes", "", None, 2.5, [True]]:
            with self.subTest(bad=repr(bad)):
                kw = self._base()
                kw["repeat_bos"] = bad
                with self.assertRaises(ValueError):
                    real_core.Format(**kw)
        for good in (True, False):
            kw = self._base()
            kw["repeat_bos"] = good
            self.assertIs(real_core.Format(**kw).repeat_bos, good)

    def test_optional_overrides_none_or_exact_str(self):
        for field in ("user_turn_end", "assistant_turn_end", "open_suffix"):
            for bad in [123, 4.5, b"x", ["x"], _StrSub("x")]:
                with self.subTest(field=field, bad=type(bad).__name__):
                    kw = self._base()
                    kw[field] = bad
                    with self.assertRaises(ValueError):
                        real_core.Format(**kw)
            for good in (None, "", "<X>", "\n"):
                with self.subTest(field=field, good=repr(good)):
                    kw = self._base()
                    kw[field] = good
                    self.assertEqual(getattr(real_core.Format(**kw), field), good)

    def test_format_is_frozen(self):
        fmt = real_core.Format(**self._base())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            fmt.bos = "y"  # type: ignore[misc]

    def test_format_replace_and_hash(self):
        fmt = real_core.Format(**self._base())
        variant = dataclasses.replace(fmt, name="f2", repeat_bos=True)
        self.assertEqual(variant.name, "f2")
        self.assertIs(variant.repeat_bos, True)
        self.assertEqual(variant.bos, "<b>")
        # skipcq: PTC-W0050 - the duplicate is the assertion (frozen-value hash collapse)
        self.assertEqual(len({fmt, fmt}), 1)
        self.assertNotEqual(fmt, variant)


class TestRenderValidation(unittest.TestCase):
    """Adversarial probes of Format.render's acceptance rules (D3, D4)."""

    @staticmethod
    def _fmt(**kw) -> real_core.Format:
        base = {
            "name": "f",
            "bos": "<b>",
            "user_prefix": "<u>",
            "assistant_prefix": "<a>",
            "turn_end": "<e>",
        }
        base.update(kw)
        return real_core.Format(**base)

    def test_container_must_be_nonempty_exact_tuple(self):
        fmt = self._fmt()
        msg = real_core.Message("user", "x")

        class _TupleSub(tuple):
            pass

        class _MessageSub(real_core.Message):
            pass

        bad_containers = [
            None,
            [],
            (),
            [msg],
            _TupleSub((msg,)),
            (x for x in (msg,)),
            "usermsg",
            (None,),
            ("x",),
            (msg, "x"),
            (_MessageSub("user", "x"),),
            123,
        ]
        for container in bad_containers:
            with self.subTest(container=type(container).__name__), self.assertRaises(ValueError):
                fmt.render(container)

    def test_final_turn_must_be_user(self):
        fmt = self._fmt()
        u = real_core.Message("user", "u")
        a = real_core.Message("assistant", "a")
        for msgs in [(a,), (u, a), (u, a, u, a)]:
            with self.subTest(n=len(msgs)), self.assertRaises(ValueError):
                fmt.render(msgs)

    def test_alternation_starting_with_user(self):
        fmt = self._fmt()
        u = real_core.Message("user", "u")
        a = real_core.Message("assistant", "a")
        for msgs in [(a, u), (u, u), (u, a, a), (u, a, u, u, u)]:
            with self.subTest(roles=tuple(m.role for m in msgs)), self.assertRaises(ValueError):
                fmt.render(msgs)

    def test_valid_multiturn_renders(self):
        fmt = self._fmt()
        one = (real_core.Message("user", "H"),)
        self.assertEqual(fmt.render(one), "<b><u>H<e><a>")
        three = (
            real_core.Message("user", "A"),
            real_core.Message("assistant", "B"),
            real_core.Message("user", "C"),
        )
        self.assertEqual(fmt.render(three), "<b><u>A<e><a>B<e><u>C<e><a>")
        five = (
            real_core.Message("user", "1"),
            real_core.Message("assistant", "2"),
            real_core.Message("user", "3"),
            real_core.Message("assistant", "4"),
            real_core.Message("user", "5"),
        )
        self.assertEqual(fmt.render(five), "<b><u>1<e><a>2<e><u>3<e><a>4<e><u>5<e><a>")

    def test_hostile_contents_render_verbatim(self):
        fmt = self._fmt()
        hostile = ["", " ", "\x00", "\ud800", "\U0001f600", "{content}", "%s", "\\n", "x" * 100000]
        for content in hostile:
            with self.subTest(content=repr(content[:12])):
                out = fmt.render((real_core.Message("user", content),))
                self.assertEqual(out, f"<b><u>{content}<e><a>")

    def test_per_role_ends_and_open_suffix_fallbacks(self):
        # All three optionals None -> turn_end / assistant_prefix fallbacks.
        fmt = self._fmt()
        msgs = (
            real_core.Message("user", "U"),
            real_core.Message("assistant", "A"),
            real_core.Message("user", "V"),
        )
        self.assertEqual(fmt.render(msgs), "<b><u>U<e><a>A<e><u>V<e><a>")
        # Per-role overrides replace turn_end per role.
        fmt2 = self._fmt(user_turn_end="</u>", assistant_turn_end="</a>")
        self.assertEqual(fmt2.render(msgs), "<b><u>U</u><a>A</a><u>V</u><a>")
        # Empty-string overrides are honored, not treated as None (D5).
        fmt3 = self._fmt(user_turn_end="", assistant_turn_end="", open_suffix="")
        self.assertEqual(fmt3.render(msgs), "<b><u>U<a>A<u>V")
        # open_suffix=None falls back to assistant_prefix.
        fmt4 = self._fmt(open_suffix=None)
        self.assertTrue(fmt4.render((real_core.Message("user", "U"),)).endswith("<a>"))

    def test_repeat_bos_counts(self):
        for n in (1, 3, 5):
            with self.subTest(n=n):
                fmt = real_core.Format("rb", "<B>", "[U]", "[A]", "<E>", repeat_bos=True)
                msgs = tuple(
                    real_core.Message("user" if i % 2 == 0 else "assistant", str(i))
                    for i in range(n)
                )
                self.assertEqual(fmt.render(msgs).count("<B>"), n + 1)


# ---------------------------------------------------------------------------
# Mutation-style sensitivity controls: each mutant is a behavior-changing
# edit to a COPY of core.py in /tmp/cycle32_mutants_<pid> (the repo is never
# mutated). core.py is stdlib-only, so the copy loads standalone via
# importlib. A mutant is CAUGHT iff the differential harness reports >= 1
# divergence against it.
# ---------------------------------------------------------------------------

MUTANTS: list[tuple[str, str, str, int]] = [
    (
        "M1_open_suffix_fallback_to_empty",
        "suffix = self.open_suffix if self.open_suffix is not None else self.assistant_prefix",
        'suffix = self.open_suffix if self.open_suffix is not None else ""',
        1,
    ),
    (
        "M2_drop_repeat_bos_open_reply",
        "        if self.repeat_bos:\n"
        "            parts.append(self.bos)\n"
        "        suffix = self.open_suffix if self.open_suffix is not None "
        "else self.assistant_prefix",
        "        suffix = self.open_suffix if self.open_suffix is not None "
        "else self.assistant_prefix",
        1,
    ),
    (
        "M3_ignore_per_role_turn_ends",
        "turn_end = per_role_end if per_role_end is not None else self.turn_end",
        "turn_end = self.turn_end",
        1,
    ),
    (
        "M4_swap_role_prefixes",
        'prefix = self.user_prefix if message.role == "user" else self.assistant_prefix',
        'prefix = self.assistant_prefix if message.role == "user" else self.user_prefix',
        1,
    ),
    (
        "M5_drop_turn_end",
        "parts.extend((prefix, message.content, turn_end))",
        "parts.extend((prefix, message.content))",
        1,
    ),
    (
        "M6_drop_final_user_turn_check",
        '        if messages[-1].role != "user":\n'
        '            raise ValueError("This initial renderer requires a final user turn")\n',
        "",
        1,
    ),
    (
        "M7_flip_alternation_parity",
        'expected = "user" if index % 2 == 0 else "assistant"',
        'expected = "assistant" if index % 2 == 0 else "user"',
        1,
    ),
    (
        "M8_empty_override_falls_back",
        "turn_end = per_role_end if per_role_end is not None else self.turn_end",
        "turn_end = per_role_end if per_role_end else self.turn_end",
        1,
    ),
    (
        "M9_content_isinstance_instead_of_exact",
        "if type(self.content) is not str:",
        "if not isinstance(self.content, str):",
        1,
    ),
    (
        "M10_name_blank_check_without_strip",
        "if not self.name.strip():",
        "if not self.name:",
        1,
    ),
    (
        "M11_repeat_bos_accepts_int",
        "if type(self.repeat_bos) is not bool:",
        "if not isinstance(self.repeat_bos, (bool, int)):",
        1,
    ),
    (
        "M12_messages_isinstance_instead_of_exact_tuple",
        "            type(messages) is not tuple\n",
        "            not isinstance(messages, tuple)\n",
        1,
    ),
    (
        "M13_message_element_isinstance_instead_of_exact",
        "            or any(type(message) is not Message for message in messages)\n",
        "            or any(not isinstance(message, Message) for message in messages)\n",
        1,
    ),
    (
        "M14_role_exact_str_check_added",
        '        if self.role not in ("user", "assistant"):',
        '        if type(self.role) is not str or self.role not in ("user", "assistant"):',
        1,
    ),
    (
        "M15_open_suffix_or_instead_of_is_not_none",
        "suffix = self.open_suffix if self.open_suffix is not None else self.assistant_prefix",
        "suffix = self.open_suffix or self.assistant_prefix",
        1,
    ),
]


def _load_mutant(name: str, old: str, new: str, count: int):
    src = CORE_SRC.read_text(encoding="utf-8")
    found = src.count(old)
    if found != count:
        raise AssertionError(f"mutant {name}: anchor found {found}x, expected {count}x")
    MUTANT_DIR.mkdir(parents=True, exist_ok=True)
    path = MUTANT_DIR / f"mut_{name}_core.py"
    path.write_text(src.replace(old, new), encoding="utf-8")
    mod_name = f"cycle32_mutant_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError(f"mutant {name}: no loader")
    # dataclasses (3.12+) resolve annotations via sys.modules[cls.__module__],
    # so the synthetic module must be registered during exec.
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[mod_name]
    return module


class TestMutantSensitivity(unittest.TestCase):
    def test_all_mutants_caught(self):
        self.addCleanup(shutil.rmtree, MUTANT_DIR, True)
        caught: list[str] = []
        for name, old, new, count in MUTANTS:
            with self.subTest(mutant=name):
                mutant_mod = _load_mutant(name, old, new, count)
                divs = _construction_divergences(mutant_mod, f"cycle32-mut-{name}-c", 60)
                divs += _render_divergences(mutant_mod, f"cycle32-mut-{name}-r", 60)
                divs += _override_sweep_divergences(mutant_mod)
                self.assertTrue(
                    divs,
                    f"mutant {name} was NOT caught by the differential harness",
                )
                caught.append(name)
        self.assertEqual(len(caught), len(MUTANTS))
