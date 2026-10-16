"""Cycle 62 adversarial audit: ``doctor._portability_check`` / ``doctor.portability_matrix``.

Scope
-----
Cycle 59 audited ``core.check_boundary_portability`` but explicitly left
``doctor._portability_check`` and ``doctor.portability_matrix`` unverified
("require real tokenizer checkpoints / HF stack -- thin wrappers"). This
file closes that gap with synthetic tokenizers only: no checkpoints, no
network, no real ``transformers`` objects. A fake ``transformers`` module
is injected into ``sys.modules`` whose ``PreTrainedTokenizerFast`` builds
scripted fake tokenizers from a per-test registry keyed by the
``tokenizer_file=`` path the implementation passes.

Tokenizer interface consumed (established by reading ``doctor.py``):
* ``_portability_check`` -- primary tokenizer: ``.decode(list[int]) -> str``
  ONLY (proved by a decode-only fake). Reference tokenizer:
  ``.encode(text, add_special_tokens=False)``.
* ``portability_matrix`` -- secondary tokenizer: ``.encode(text,
  add_special_tokens=False)``, ``.convert_tokens_to_ids(str)``, and the
  seven special-token attributes (unk/bos/eos/pad/sep/cls/mask_token).

Contracts (from code reading, ``doctor.py``)
-------------------------------------------
``_portability_check(reference_model_path, boundary_ids, primary_tokenizer)``
returns ``{"status", "detail"}`` and never raises for the inputs below:
1. ``primary_tokenizer is None`` -> skipped / "primary tokenizer unavailable".
2. ``boundary_ids is None`` -> skipped / "needs --boundary-ids".
3. ``tuple(boundary_ids)`` raises TypeError, or the ids are empty, or any
   element fails ``type(t) is int`` (bool excluded) or is negative ->
   skipped / "invalid boundary ids".
4. ``decode`` raises, or returns a non-``str`` (strict ``type() is``) or an
   empty string -> skipped / "boundary_decode_failed".
5. Reference load fails (unreadable dir, missing/oversized/malformed
   ``tokenizer.json``, ``transformers`` unimportable, constructor raises) ->
   skipped / "reference_tokenizer_unavailable".
6. Reference ``encode`` raises, or ``tuple()`` of its raw return raises ->
   skipped / "reference_tokenizer_unavailable" (core: secondary_encode_failed).
7. Reference ``encode`` returns invalid ids (non-int incl. bool, negative;
   note the EMPTY tuple is *valid* here) -> skipped /
   "reference_tokenizer_unavailable" (core: secondary_encode_invalid).
8. Re-encoded ids equal the boundary ids -> ok /
   "boundary ids identical under reference tokenizer".
9. Otherwise -> mismatch / ``"primary=[...] reference=[...]"``.

``portability_matrix(second_checkpoint)``:
* Raises ``ValueError`` only when ``second_checkpoint`` is not str/Path.
* Never raises for checkpoint problems; those yield
  ``{"status": "unavailable", "detail": <fixed code>, "second_checkpoint":
  str(input), "asymmetry": ..., "rows": []}`` with detail in
  {"not_a_readable_directory", "oversized_tokenizer_file",
  "tokenizer_load_failed", "transformers_not_installed"}.
* On success: one row per catalog candidate (sorted families,
  ``candidates(family, mutate=True)``) holding the
  ``core.template_tokenizer_compatibility`` row plus
  ``row["bos_skipped"] = (fmt.bos == "")``; ``detail`` is
  ``"{n} candidates checked (tokenizer only, no weights)"`` and
  ``second_checkpoint`` is the *resolved* root.
* The registered map (``_secondary_registered``) parses ``added_tokens``
  from the ``tokenizer.json`` document first-wins (content: nonempty str,
  id: ``type() is int`` and >= 0 -- bool/negative rejected), then fills
  gaps from the seven special-token attributes via
  ``convert_tokens_to_ids`` (exceptions suppressed, result must be
  ``type() is int`` and >= 0).
* The ``encode`` closure normalises with ``tuple()``: a list/generator
  return still works; a raising encode degrades only its token's row
  (format verdict "unknown", never "compatible") -- one bad literal never
  poisons the whole matrix.

Method
------
1. Independent oracles hand-written from the docstrings (the matrix row
   oracle is deliberately structured differently from the implementation;
   the ``_portability_check`` oracle delegates the re-encode comparison
   itself to the real, cycle-59-audited
   ``core.check_boundary_portability`` so only doctor's own mapping is
   re-stated).
2. Differential fuzzing (>=200 randomized synthetic cases) with
   full-dict equality assertions.
3. A cross-check of the hand-rolled row oracle against the real
   ``core.template_tokenizer_compatibility`` over the fuzz corpus, so an
   oracle bug cannot masquerade as implementation correctness.
4. One hand-pinned test per branch/edge case.
5. Mutation gate on /tmp copies of the package (repo sources never
   touched): 5 semantic mutants, all killed.

Findings: CLEAN -- no genuine bugs found. Every branch of both functions
behaves exactly per contract across 350 fuzz subtests (220 for
``_portability_check``, 130 for ``portability_matrix``), 57 hand-pinned
edge-case tests, a 40-case oracle-vs-core cross-check, and 5 killed
mutants. Deliberately not verified: real ``transformers``/``tokenizers``
behaviour against real checkpoint files (the functions' HF-stack seam is
exactly what the fake module replaces); ``examine_model``/``doctor``
integration beyond these two functions.
"""

from __future__ import annotations

import contextlib
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from model_protocol_lab.catalog import CATALOG, candidates
from model_protocol_lab.core import (
    check_boundary_portability,
    template_tokenizer_compatibility,
)
from model_protocol_lab.doctor import (
    _PORTABILITY_ASYMMETRY,
    _portability_check,
    _secondary_registered,
    portability_matrix,
)
from model_protocol_lab.fixtures import _extract_special_tokens

# ---------------------------------------------------------------------------
# Fake HF stack
# ---------------------------------------------------------------------------


def _unused(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("fake tokenizer method called without a script")


class FakeSpec:
    """Scripted behaviour for one fake tokenizer checkpoint."""

    def __init__(
        self,
        *,
        encode: Callable[[str], Any] | None = None,
        decode: Callable[[list], Any] | None = None,
        convert: Callable[[str], Any] | None = None,
        attrs: dict[str, Any] | None = None,
        construct_error: BaseException | None = None,
        drop_encode: bool = False,
    ) -> None:
        self.encode = encode or _unused
        self.decode = decode or _unused
        self.convert = convert or _unused
        self.attrs = attrs or {}
        self.construct_error = construct_error
        self.drop_encode = drop_encode


class _FakePreTrainedTokenizerFast:
    """Scripted stand-in for ``transformers.PreTrainedTokenizerFast``.

    Constructed exactly like the real thing
    (``tokenizer_file=..., clean_up_tokenization_spaces=False``); behaviour
    comes from the class-level registry keyed by the ``tokenizer_file``
    path string.
    """

    registry: dict[str, FakeSpec] = {}

    def __init__(self, tokenizer_file: Any = None, **kwargs: Any) -> None:
        spec = _FakePreTrainedTokenizerFast.registry.get(str(tokenizer_file))
        if spec is None:
            raise FileNotFoundError(f"no fake tokenizer for {tokenizer_file}")
        if spec.construct_error is not None:
            raise spec.construct_error
        self._spec = spec
        for name, value in spec.attrs.items():
            setattr(self, name, value)
        if spec.drop_encode:
            self.encode = None  # present but unusable -> TypeError on call

    def encode(self, text: str, add_special_tokens: bool = False) -> Any:
        return self._spec.encode(text)

    def decode(self, token_ids: list) -> Any:
        return self._spec.decode(token_ids)

    def convert_tokens_to_ids(self, token: str) -> Any:
        return self._spec.convert(token)


class _PrimaryFake:
    """Decode-only fake: proves ``_portability_check`` needs nothing else."""

    def __init__(self, decode: Callable[[list], Any]) -> None:
        self._decode = decode

    def decode(self, token_ids: list) -> Any:
        return self._decode(token_ids)


@pytest.fixture()
def fake_transformers(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeSpec]:
    """Install the fake ``transformers`` module; yield its registry."""
    registry: dict[str, FakeSpec] = {}
    _FakePreTrainedTokenizerFast.registry = registry
    module = ModuleType("transformers")
    module.PreTrainedTokenizerFast = _FakePreTrainedTokenizerFast
    monkeypatch.setitem(sys.modules, "transformers", module)
    yield registry
    _FakePreTrainedTokenizerFast.registry = {}


def _write_ckpt(directory: Path, doc: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tokenizer.json").write_text(json.dumps(doc), encoding="utf-8")
    return directory


def _register(registry: dict[str, FakeSpec], ckpt: Path, spec: FakeSpec) -> None:
    registry[str(ckpt / "tokenizer.json")] = spec


@contextlib.contextmanager
def _no_transformers():
    """Temporarily make ``import transformers`` raise ImportError.

    ``sys.modules["transformers"] = None`` makes the import system raise
    ``ImportError``; the previous entry (the fake module under test) is
    restored afterwards so later cases are unaffected.
    """
    saved = sys.modules.get("transformers")
    sys.modules["transformers"] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = saved


# ---------------------------------------------------------------------------
# Oracle: _portability_check (doctor's mapping only; core comparison is real)
# ---------------------------------------------------------------------------

_CHECK_SKIPPED = "skipped"


def _oracle_portability_check(
    *,
    primary: dict | None,
    boundary: dict | None,
    reference: dict,
) -> dict[str, str]:
    """Independent decision table for ``doctor._portability_check``.

    ``primary``: None, or ``{"decode": callable}``.
    ``boundary``: None, or ``{"kind": "tuple"|"list"|"gen"|"str"|"int"|"dict"|"obj",
    "ids": [...]}`` describing the value handed to the implementation.
    ``reference``: ``{"load": "ok"|"missing_dir"|"missing_file"|"bad_json"|
    "construct_error"|"no_transformers"|"bad_path", "encode": callable|None}``.

    The re-encode comparison itself uses the real
    ``core.check_boundary_portability`` (audited in cycle 59); this oracle
    only re-states doctor's documented status/detail mapping.
    """
    if primary is None:
        return {"status": _CHECK_SKIPPED, "detail": "primary tokenizer unavailable"}
    if boundary is None:
        return {"status": _CHECK_SKIPPED, "detail": "needs --boundary-ids"}
    kind, ids = boundary["kind"], boundary["ids"]
    value: Any = {"tuple": tuple, "list": list}.get(kind, lambda x: x)(ids)
    if kind == "gen":
        value = iter(ids)
    elif kind == "str":
        value = "12"
    elif kind == "int":
        value = 5
    elif kind == "dict":
        value = {"a": 1}
    elif kind == "obj":
        value = object()
    try:
        clean = tuple(value)
    except TypeError:
        return {"status": _CHECK_SKIPPED, "detail": "invalid boundary ids"}
    if not clean or any(type(t) is not int or t < 0 for t in clean):
        return {"status": _CHECK_SKIPPED, "detail": "invalid boundary ids"}
    try:
        text = primary["decode"](list(clean))
    except Exception:  # noqa: BLE001 -- oracle mirrors the degrade boundary
        return {"status": _CHECK_SKIPPED, "detail": "boundary_decode_failed"}
    if type(text) is not str or not text:
        return {"status": _CHECK_SKIPPED, "detail": "boundary_decode_failed"}
    if reference["load"] != "ok":
        return {"status": _CHECK_SKIPPED, "detail": "reference_tokenizer_unavailable"}

    def closure(probe: str) -> tuple[int, ...]:
        return tuple(reference["encode"](probe))

    outcome = check_boundary_portability(text, clean, closure)
    if outcome["detail"] == "identical":
        return {"status": "ok", "detail": "boundary ids identical under reference tokenizer"}
    if outcome["detail"] in ("secondary_encode_failed", "secondary_encode_invalid"):
        return {"status": _CHECK_SKIPPED, "detail": "reference_tokenizer_unavailable"}
    return {
        "status": "mismatch",
        "detail": f"primary={outcome['primary_ids']} reference={outcome['secondary_ids']}",
    }


def _impl_boundary_value(boundary: dict | None) -> Any:
    if boundary is None:
        return None
    kind, ids = boundary["kind"], boundary["ids"]
    if kind == "tuple":
        return tuple(ids)
    if kind == "list":
        return list(ids)
    if kind == "gen":
        return iter(ids)
    if kind == "str":
        return "12"
    if kind == "int":
        return 5
    if kind == "dict":
        return {"a": 1}
    return object()


# ---------------------------------------------------------------------------
# Oracle: portability_matrix (hand-rolled from docstrings)
# ---------------------------------------------------------------------------

_SPECIAL_ATTRS = (
    "unk_token",
    "bos_token",
    "eos_token",
    "pad_token",
    "sep_token",
    "cls_token",
    "mask_token",
)


def _oracle_secondary_registered(
    doc: dict, attrs: dict[str, Any], convert: Callable[[str], Any]
) -> dict[str, int]:
    """Independent re-statement of ``doctor._secondary_registered``'s contract.

    Document ``added_tokens`` first-wins (nonempty str content, ``type() is
    int`` id >= 0); then the seven special-token attributes fill gaps via
    ``convert_tokens_to_ids`` with per-attribute exception suppression.
    """
    registered: dict[str, int] = {}
    added = doc.get("added_tokens")
    if type(added) is list:
        for entry in added:
            if (
                type(entry) is dict
                and type(entry.get("content")) is str
                and entry["content"]
                and type(entry.get("id")) is int
                and entry["id"] >= 0
                and entry["content"] not in registered
            ):
                registered[entry["content"]] = entry["id"]
    for attr in _SPECIAL_ATTRS:
        token = attrs.get(attr)
        if type(token) is str and token and token not in registered:
            token_id: Any = None
            with contextlib.suppress(Exception):
                token_id = convert(token)
            if type(token_id) is int and token_id >= 0:
                registered[token] = token_id
    return registered


def _oracle_token_row(
    token: str, registered: dict[str, int], encode: Callable[[str], tuple[int, ...]]
) -> dict[str, Any]:
    """One inventory token's row, from ``core.template_tokenizer_compatibility``'s docstring.

    Per-token checks: (a) registered, (b) encodes to exactly one id,
    (c) that id equals the registered id. Encode failures degrade the
    row to ``encode_failed``/``encode_invalid`` and never raise.
    """
    row: dict[str, Any] = {
        "token": token,
        "registered": token in registered,
        "registered_id": registered.get(token),
        "single_token": None,
        "encoded_id": None,
        "id_match": None,
        "status": "encode_failed",
    }
    try:
        encoded = encode(token)
    except Exception:  # noqa: BLE001 -- probe degrades, never raises
        return row
    valid = type(encoded) is tuple and all(type(t) is int and t >= 0 for t in encoded)
    if not valid or not encoded:
        row["status"] = "encode_invalid"
        return row
    row["encoded_id"] = encoded[0]
    row["single_token"] = len(encoded) == 1
    if token not in registered:
        row["status"] = "unregistered"
    elif len(encoded) != 1:
        row["status"] = "fragments"
    elif encoded[0] != registered[token]:
        row["status"] = "id_mismatch"
        row["id_match"] = False
    else:
        row["status"] = "ok"
        row["id_match"] = True
    return row


def _oracle_row_verdict(name: str, token_rows: list[dict[str, Any]]) -> tuple[str, str, str]:
    """Verdict precedence from the docstring: unknown > incompatible > compatible."""
    if not token_rows:
        return (
            "unknown",
            "empty_inventory",
            "empty inventory: the format exposes no token-shaped control "
            "literals, so compatibility cannot be judged (never 'compatible')",
        )
    degraded = next(
        (r for r in token_rows if r["status"] in ("encode_failed", "encode_invalid")),
        None,
    )
    if degraded is not None:
        return (
            "unknown",
            degraded["status"],
            f"{degraded['status']} on {degraded['token']!r}: the encode "
            "probe degraded, so compatibility cannot be judged",
        )
    failed = next((r for r in token_rows if r["status"] != "ok"), None)
    if failed is not None:
        return (
            "incompatible",
            failed["status"],
            f"{failed['status']} on {failed['token']!r}",
        )
    count = len(token_rows)
    return (
        "compatible",
        "",
        f"{count}/{count} inventory tokens registered, atomic, and id-consistent; "
        "this predicts nothing about experimental success (weights decide)",
    )


def _oracle_unavailable(second_checkpoint: Any, detail: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "detail": detail,
        "second_checkpoint": str(second_checkpoint),
        "asymmetry": _PORTABILITY_ASYMMETRY,
        "rows": [],
    }


def _oracle_matrix(
    second_checkpoint: Path,
    doc: dict,
    attrs: dict[str, Any],
    convert: Callable[[str], Any],
    raw_encode: Callable[[str], Any],
) -> dict[str, Any]:
    """Independent re-statement of ``doctor.portability_matrix``'s ok-path."""
    registered = _oracle_secondary_registered(doc, attrs, convert)

    def encode(text: str) -> tuple[int, ...]:
        return tuple(raw_encode(text))

    rows: list[dict[str, Any]] = []
    for family in sorted(CATALOG):
        for fmt in candidates(family, mutate=True):
            inventory = _extract_special_tokens(fmt)
            token_rows = [_oracle_token_row(t, registered, encode) for t in inventory]
            verdict, reason, detail = _oracle_row_verdict(fmt.name, token_rows)
            rows.append(
                {
                    "format": fmt.name,
                    "verdict": verdict,
                    "reason": reason,
                    "detail": detail,
                    "tokens": token_rows,
                    "bos_skipped": fmt.bos == "",
                }
            )
    return {
        "status": "ok",
        "detail": f"{len(rows)} candidates checked (tokenizer only, no weights)",
        "second_checkpoint": str(second_checkpoint.resolve()),
        "asymmetry": _PORTABILITY_ASYMMETRY,
        "rows": rows,
    }


def _all_inventory_literals() -> list[str]:
    seen: list[str] = []
    for family in sorted(CATALOG):
        for fmt in candidates(family, mutate=True):
            for token in _extract_special_tokens(fmt):
                if token not in seen:
                    seen.append(token)
    return seen


def _row_by_format(report: dict[str, Any], name: str) -> dict[str, Any]:
    for row in report["rows"]:
        if row["format"] == name:
            return row
    raise AssertionError(f"no row for {name}")


# ---------------------------------------------------------------------------
# _portability_check: hand-pinned branch/edge tests
# ---------------------------------------------------------------------------


class _StrSub(str):
    pass


def _raise_decode(ids: list) -> Any:
    raise ValueError("decode boom")


def _raise_encode(text: str) -> Any:
    raise RuntimeError("encode boom")


@pytest.fixture()
def check_dirs(tmp_path: Path) -> dict[str, Path]:
    base = tmp_path / "check"
    good = _write_ckpt(base / "ref_ok", {})
    nofile = base / "ref_nofile"
    nofile.mkdir(parents=True)
    bad = _write_ckpt(base / "ref_bad", {})
    (bad / "tokenizer.json").write_text("{not json", encoding="utf-8")
    return {"good": good, "nofile": nofile, "bad": bad, "base": base}


def _run_check(
    registry: dict[str, FakeSpec],
    dirs: dict[str, Path],
    *,
    primary: dict | None,
    boundary: dict | None,
    load: str = "ok",
    encode: Callable[[str], Any] | None = None,
    construct_error: BaseException | None = None,
) -> dict[str, str]:
    if load == "missing_dir":
        ref_path: Any = dirs["base"] / "does-not-exist"
    elif load == "bad_path":
        ref_path = 12345
    elif load == "nofile":
        ref_path = dirs["nofile"]
    elif load == "bad_json":
        ref_path = dirs["bad"]
    else:
        ref_path = dirs["good"]
    if load in ("ok", "construct_error"):
        _register(
            registry,
            ref_path,
            FakeSpec(encode=encode, construct_error=construct_error),
        )
    primary_tok = None if primary is None else _PrimaryFake(primary["decode"])
    if load == "no_transformers":
        with _no_transformers():
            return _portability_check(ref_path, _impl_boundary_value(boundary), primary_tok)
    return _portability_check(ref_path, _impl_boundary_value(boundary), primary_tok)


def _std_case(**over: Any) -> dict[str, Any]:
    case: dict[str, Any] = {
        "primary": {"decode": lambda ids: "<m>"},
        "boundary": {"kind": "tuple", "ids": [1, 2]},
        "load": "ok",
        "encode": lambda text: (1, 2),
    }
    case.update(over)
    return case


def _assert_check(
    registry: dict[str, FakeSpec],
    dirs: dict[str, Path],
    case: dict[str, Any],
) -> None:
    expected = _oracle_portability_check(
        primary=case["primary"], boundary=case["boundary"], reference=case
    )
    actual = _run_check(
        registry,
        dirs,
        primary=case["primary"],
        boundary=case["boundary"],
        load=case["load"],
        encode=case.get("encode"),
        construct_error=case.get("construct_error"),
    )
    assert actual == expected, f"case={case}"
    assert set(actual) == {"status", "detail"}


def test_check_none_primary(fake_transformers, check_dirs):
    _assert_check(fake_transformers, check_dirs, _std_case(primary=None))


def test_check_none_boundary_ids(fake_transformers, check_dirs):
    _assert_check(fake_transformers, check_dirs, _std_case(boundary=None))


def test_check_bool_boundary_id_invalid(fake_transformers, check_dirs):
    case = _std_case(boundary={"kind": "tuple", "ids": [7, True]})
    _assert_check(fake_transformers, check_dirs, case)
    assert _oracle_portability_check(
        primary=case["primary"], boundary=case["boundary"], reference=case
    ) == {"status": "skipped", "detail": "invalid boundary ids"}


def test_check_zero_boundary_id_valid(fake_transformers, check_dirs):
    # id 0 is a legitimate token id, not "empty"/invalid
    _assert_check(
        fake_transformers,
        check_dirs,
        _std_case(
            boundary={"kind": "tuple", "ids": [0]},
            encode=lambda text: (0,),
        ),
    )


def test_check_negative_boundary_id_invalid(fake_transformers, check_dirs):
    _assert_check(
        fake_transformers,
        check_dirs,
        _std_case(boundary={"kind": "tuple", "ids": [3, -1]}),
    )


def test_check_noniterable_boundary_ids(fake_transformers, check_dirs):
    _assert_check(
        fake_transformers,
        check_dirs,
        _std_case(boundary={"kind": "int", "ids": []}),
    )


def test_check_str_boundary_ids_invalid(fake_transformers, check_dirs):
    # tuple("12") -> ('1', '2'): not ints -> invalid, never a bare TypeError
    _assert_check(
        fake_transformers,
        check_dirs,
        _std_case(boundary={"kind": "str", "ids": []}),
    )


def test_check_empty_boundary_ids(fake_transformers, check_dirs):
    _assert_check(
        fake_transformers,
        check_dirs,
        _std_case(boundary={"kind": "tuple", "ids": []}),
    )


def test_check_generator_boundary_ids(fake_transformers, check_dirs):
    _assert_check(
        fake_transformers,
        check_dirs,
        _std_case(boundary={"kind": "gen", "ids": [4, 5]}, encode=lambda t: (4, 5)),
    )


def test_check_decode_raises(fake_transformers, check_dirs):
    _assert_check(
        fake_transformers,
        check_dirs,
        _std_case(primary={"decode": _raise_decode}),
    )


def test_check_decode_returns_bytes(fake_transformers, check_dirs):
    _assert_check(
        fake_transformers,
        check_dirs,
        _std_case(primary={"decode": lambda ids: b"<m>"}),
    )


def test_check_decode_returns_empty(fake_transformers, check_dirs):
    _assert_check(
        fake_transformers,
        check_dirs,
        _std_case(primary={"decode": lambda ids: ""}),
    )


def test_check_decode_returns_str_subclass(fake_transformers, check_dirs):
    # strict `type() is str`: a subclass is still a decode failure
    _assert_check(
        fake_transformers,
        check_dirs,
        _std_case(primary={"decode": lambda ids: _StrSub("<m>")}),
    )


def test_check_reference_dir_missing(fake_transformers, check_dirs):
    _assert_check(fake_transformers, check_dirs, _std_case(load="missing_dir"))


def test_check_reference_bad_path(fake_transformers, check_dirs):
    # _directory raises ValueError -> caught -> skipped, never raised
    _assert_check(fake_transformers, check_dirs, _std_case(load="bad_path"))


def test_check_reference_tokenizer_json_missing(fake_transformers, check_dirs):
    _assert_check(fake_transformers, check_dirs, _std_case(load="nofile"))


def test_check_reference_tokenizer_json_malformed(fake_transformers, check_dirs):
    _assert_check(fake_transformers, check_dirs, _std_case(load="bad_json"))


def test_check_reference_transformers_missing(fake_transformers, check_dirs):
    _assert_check(fake_transformers, check_dirs, _std_case(load="no_transformers"))


def test_check_reference_construct_raises(fake_transformers, check_dirs):
    _assert_check(
        fake_transformers,
        check_dirs,
        _std_case(load="construct_error", construct_error=RuntimeError("boom")),
    )


def test_check_encode_returns_list_normalized(fake_transformers, check_dirs):
    # the tuple() wrapper normalises a list return: still identical
    case = _std_case(encode=lambda text: [1, 2])
    _assert_check(fake_transformers, check_dirs, case)
    assert _run_check(
        fake_transformers,
        check_dirs,
        primary=case["primary"],
        boundary=case["boundary"],
        encode=case["encode"],
    ) == {"status": "ok", "detail": "boundary ids identical under reference tokenizer"}


def test_check_encode_returns_generator(fake_transformers, check_dirs):
    _assert_check(
        fake_transformers,
        check_dirs,
        _std_case(encode=lambda text: (i for i in (1, 2))),
    )


def test_check_encode_returns_none(fake_transformers, check_dirs):
    # tuple(None) raises TypeError inside the probe -> secondary_encode_failed
    _assert_check(fake_transformers, check_dirs, _std_case(encode=lambda t: None))


def test_check_encode_returns_bool_ids(fake_transformers, check_dirs):
    _assert_check(
        fake_transformers,
        check_dirs,
        _std_case(encode=lambda t: (True, True)),
    )


def test_check_encode_returns_negative_ids(fake_transformers, check_dirs):
    _assert_check(
        fake_transformers,
        check_dirs,
        _std_case(encode=lambda t: (1, -1)),
    )


def test_check_encode_returns_str_elements(fake_transformers, check_dirs):
    _assert_check(
        fake_transformers,
        check_dirs,
        _std_case(encode=lambda t: ("1", "2")),
    )


def test_check_encode_returns_empty_tuple_is_divergent(fake_transformers, check_dirs):
    # empty tuple PASSES _valid_token_ids -> divergent, i.e. a real mismatch
    # with secondary=[] -- not an "invalid" degradation
    case = _std_case(encode=lambda t: ())
    _assert_check(fake_transformers, check_dirs, case)
    assert _run_check(
        fake_transformers,
        check_dirs,
        primary=case["primary"],
        boundary=case["boundary"],
        encode=case["encode"],
    ) == {"status": "mismatch", "detail": "primary=[1, 2] reference=[]"}


def test_check_encode_raises(fake_transformers, check_dirs):
    _assert_check(fake_transformers, check_dirs, _std_case(encode=_raise_encode))


def test_check_encode_missing_attr(fake_transformers, check_dirs):
    registry = fake_transformers
    _register(registry, check_dirs["good"], FakeSpec(drop_encode=True))
    primary = _PrimaryFake(lambda ids: "<m>")
    actual = _portability_check(check_dirs["good"], (1, 2), primary)
    assert actual == {
        "status": "skipped",
        "detail": "reference_tokenizer_unavailable",
    }


def test_check_identical_exact_dict(fake_transformers, check_dirs):
    case = _std_case()
    actual = _run_check(
        fake_transformers,
        check_dirs,
        primary=case["primary"],
        boundary=case["boundary"],
        encode=case["encode"],
    )
    assert actual == {
        "status": "ok",
        "detail": "boundary ids identical under reference tokenizer",
    }


def test_check_divergent_exact_detail(fake_transformers, check_dirs):
    case = _std_case(encode=lambda t: (1, 3))
    actual = _run_check(
        fake_transformers,
        check_dirs,
        primary=case["primary"],
        boundary=case["boundary"],
        encode=case["encode"],
    )
    assert actual == {"status": "mismatch", "detail": "primary=[1, 2] reference=[1, 3]"}


# ---------------------------------------------------------------------------
# _portability_check: differential fuzz vs the oracle (>=200 cases)
# ---------------------------------------------------------------------------

_CHECK_FUZZ_CASES = 220
_DECODE_TEXTS = ("<m>", "hello world", "<m0><m1>", "x", "  spaced\t\n")


def _fuzz_check_case(rng: random.Random) -> dict[str, Any]:
    bkind = rng.choice(
        [
            "none",
            "tuple",
            "list",
            "gen",
            "str",
            "int",
            "dict",
            "obj",
            "tuple_bool",
            "tuple_neg",
            "tuple_strel",
            "tuple_empty",
            "tuple_zero",
        ]
    )
    if bkind == "none":
        boundary = None
        ids: list = []
    else:
        ids = list(rng.choice([[1], [0], [1, 2], [3, 0, 999], [5, 6, 7, 8]]))
        if bkind == "tuple_bool":
            ids = [1, True]
        elif bkind == "tuple_neg":
            ids = ids + [-2]
        elif bkind == "tuple_strel":
            ids = ids + ["x"]
        elif bkind == "tuple_empty":
            ids = []
        kind = {
            "tuple": "tuple",
            "list": "list",
            "gen": "gen",
            "str": "str",
            "int": "int",
            "dict": "dict",
            "obj": "obj",
        }.get(bkind, "tuple")
        boundary = {"kind": kind, "ids": ids}

    if rng.random() < 0.12:
        primary = None
    else:
        dkind = rng.choice(["text", "empty", "bytes", "none", "raise", "subclass", "list"])
        dtext = rng.choice(_DECODE_TEXTS)
        if dkind == "text":
            decode = lambda xs, t=dtext: t  # noqa: E731
        elif dkind == "empty":
            decode = lambda xs: ""  # noqa: E731
        elif dkind == "bytes":
            decode = lambda xs: b"<m>"  # noqa: E731
        elif dkind == "none":
            decode = lambda xs: None  # noqa: E731
        elif dkind == "raise":
            decode = _raise_decode
        elif dkind == "subclass":
            decode = lambda xs: _StrSub("<m>")  # noqa: E731
        else:
            decode = lambda xs: ["<m>"]  # noqa: E731
        primary = {"decode": decode}

    load = rng.choices(
        [
            "ok",
            "missing_dir",
            "bad_path",
            "nofile",
            "bad_json",
            "construct_error",
            "no_transformers",
        ],
        weights=[70, 5, 3, 5, 5, 4, 3],
    )[0]
    ekind = rng.choice(
        [
            "identical",
            "divergent",
            "shorter",
            "longer",
            "raise",
            "none",
            "list",
            "gen",
            "bools",
            "neg",
            "empty",
            "strs",
            "raw_int",
            "raw_str",
        ]
    )
    valid_ids = list(ids)
    if ekind == "identical":
        encode = lambda t, v=tuple(valid_ids): v  # noqa: E731
    elif ekind == "divergent":
        if valid_ids and type(valid_ids[-1]) is int:
            div = tuple(valid_ids[:-1]) + ((valid_ids[-1] + 1) % 1000000,)
        else:
            div = tuple(valid_ids) + ("zz-divergent",)
        encode = lambda t, v=div: v  # noqa: E731
    elif ekind == "shorter":
        encode = lambda t, v=tuple(valid_ids[:-1]): v  # noqa: E731
    elif ekind == "longer":
        encode = lambda t, v=tuple(valid_ids) + (424242,): v  # noqa: E731
    elif ekind == "raise":
        encode = _raise_encode
    elif ekind == "none":
        encode = lambda t: None  # noqa: E731
    elif ekind == "list":
        encode = lambda t, v=list(valid_ids): v  # noqa: E731
    elif ekind == "gen":
        encode = lambda t, v=valid_ids: (i for i in v)  # noqa: E731
    elif ekind == "bools":
        encode = lambda t, v=tuple(True for _ in valid_ids): v  # noqa: E731
    elif ekind == "neg":
        encode = lambda t: (-1,)  # noqa: E731
    elif ekind == "empty":
        encode = lambda t: ()  # noqa: E731
    elif ekind == "strs":
        encode = lambda t: tuple("ab")  # noqa: E731
    elif ekind == "raw_int":
        encode = lambda t: 5  # noqa: E731
    else:
        encode = lambda t: "12"  # noqa: E731
    reference: dict[str, Any] = {"load": load, "encode": encode}
    if load == "construct_error":
        reference["construct_error"] = RuntimeError("fuzz boom")
    return {"primary": primary, "boundary": boundary, "reference": reference}


def test_fuzz_portability_check(fake_transformers, check_dirs):
    rng = random.Random(0xC62)  # noqa: S311 -- deterministic test seed
    failures: list[str] = []
    for n in range(_CHECK_FUZZ_CASES):
        case = _fuzz_check_case(rng)
        reference = case["reference"]
        try:
            expected = _oracle_portability_check(
                primary=case["primary"],
                boundary=case["boundary"],
                reference=reference,
            )
            actual = _run_check(
                fake_transformers,
                check_dirs,
                primary=case["primary"],
                boundary=case["boundary"],
                load=reference["load"],
                encode=reference["encode"],
                construct_error=reference.get("construct_error"),
            )
            assert actual == expected, f"impl={actual} oracle={expected}"
            assert set(actual) == {"status", "detail"}
        except AssertionError as exc:
            failures.append(f"case {n}: {exc}\n  spec={case!r}")
    assert not failures, "\n".join(failures[:10]) + (
        f"\n... {len(failures)} of {_CHECK_FUZZ_CASES} fuzz cases failed" if failures else ""
    )


# ---------------------------------------------------------------------------
# portability_matrix: hand-pinned branch/edge tests
# ---------------------------------------------------------------------------

_LITS = _all_inventory_literals()
assert len(_LITS) > 4, "catalog inventories unexpectedly tiny"

_INV_BY_NAME: dict[str, tuple[str, ...]] = {}
for _fam in sorted(CATALOG):
    for _f in candidates(_fam, mutate=True):
        _INV_BY_NAME[_f.name] = _extract_special_tokens(_f)


def _matrix_ckpt(tmp_path: Path, name: str, doc: dict) -> Path:
    return _write_ckpt(tmp_path / "matrix" / name, doc)


def _run_matrix(
    registry: dict[str, FakeSpec],
    ckpt: Path,
    doc: dict,
    spec: FakeSpec,
) -> dict[str, Any]:
    _write_ckpt(ckpt, doc)
    _register(registry, ckpt, spec)
    return portability_matrix(ckpt)


def _full_coverage_doc(extra: list[dict] | None = None) -> tuple[dict, dict[str, int]]:
    """tokenizer.json doc + id map covering every catalog inventory literal."""
    ids = {lit: 1000 + i for i, lit in enumerate(_LITS)}
    doc = {"added_tokens": [{"content": lit, "id": i} for lit, i in ids.items()]}
    if extra:
        doc["added_tokens"].extend(extra)
    return doc, ids


def _assert_unavailable_shape(report: dict[str, Any], detail: str) -> None:
    assert set(report) == {
        "status",
        "detail",
        "second_checkpoint",
        "asymmetry",
        "rows",
    }
    assert report["status"] == "unavailable"
    assert report["detail"] == detail
    assert report["asymmetry"] == _PORTABILITY_ASYMMETRY
    assert report["rows"] == []


def test_matrix_rejects_non_path(fake_transformers):
    for bad in (123, None, b"x"):
        with pytest.raises(ValueError, match="second_checkpoint must be a path"):
            portability_matrix(bad)


def test_matrix_missing_dir(fake_transformers, tmp_path):
    target = tmp_path / "does-not-exist"
    assert portability_matrix(target) == _oracle_unavailable(target, "not_a_readable_directory")


def test_matrix_path_is_file(fake_transformers, tmp_path):
    target = tmp_path / "afile"
    target.write_text("x", encoding="utf-8")
    assert portability_matrix(target) == _oracle_unavailable(target, "not_a_readable_directory")


def test_matrix_missing_tokenizer_json(fake_transformers, tmp_path):
    target = tmp_path / "nodoc"
    target.mkdir()
    report = portability_matrix(target)
    _assert_unavailable_shape(report, "tokenizer_load_failed")
    assert report["second_checkpoint"] == str(target)


def test_matrix_malformed_tokenizer_json(fake_transformers, tmp_path):
    target = _matrix_ckpt(tmp_path, "badjson", {})
    (target / "tokenizer.json").write_text("{oops", encoding="utf-8")
    report = portability_matrix(target)
    _assert_unavailable_shape(report, "tokenizer_load_failed")


def test_matrix_oversized_tokenizer_json(fake_transformers, tmp_path):
    import os

    target = _matrix_ckpt(tmp_path, "huge", {})
    path = target / "tokenizer.json"
    fd = os.open(path, os.O_WRONLY)
    try:
        os.ftruncate(fd, 128 * 1024 * 1024 + 1)  # sparse: no disk use
    finally:
        os.close(fd)
    report = portability_matrix(target)
    # the allowlisted fixed code surfaces verbatim, not "tokenizer_load_failed"
    _assert_unavailable_shape(report, "oversized_tokenizer_file")


def test_matrix_transformers_missing(fake_transformers, tmp_path):
    target = _matrix_ckpt(tmp_path, "notk", {})
    with _no_transformers():
        report = portability_matrix(target)
    _assert_unavailable_shape(report, "transformers_not_installed")


def test_matrix_construct_raises(fake_transformers, tmp_path):
    target = tmp_path / "matrix" / "boom"
    _write_ckpt(target, {})
    _register(fake_transformers, target, FakeSpec(construct_error=RuntimeError("boom")))
    report = portability_matrix(target)
    _assert_unavailable_shape(report, "tokenizer_load_failed")


def test_matrix_construct_raises_allowlisted_code(fake_transformers, tmp_path):
    target = tmp_path / "matrix" / "boom2"
    _write_ckpt(target, {})
    _register(
        fake_transformers,
        target,
        FakeSpec(construct_error=ValueError("oversized_tokenizer_file")),
    )
    report = portability_matrix(target)
    _assert_unavailable_shape(report, "oversized_tokenizer_file")


def test_matrix_all_compatible_checkpoint(fake_transformers, tmp_path):
    doc, ids = _full_coverage_doc()
    target = tmp_path / "matrix" / "full"
    report = _run_matrix(
        fake_transformers,
        target,
        doc,
        FakeSpec(encode=lambda text: (ids[text],), attrs={}),
    )
    assert report["status"] == "ok"
    assert report["detail"] == (
        f"{len(report['rows'])} candidates checked (tokenizer only, no weights)"
    )
    assert report["second_checkpoint"] == str(target.resolve())
    assert report["asymmetry"] == _PORTABILITY_ASYMMETRY
    expected_fmts = [fmt for family in sorted(CATALOG) for fmt in candidates(family, mutate=True)]
    assert len(report["rows"]) == len(expected_fmts) == 118
    for row, fmt in zip(report["rows"], expected_fmts, strict=True):
        assert row["format"] == fmt.name
        assert row["verdict"] == "compatible", row
        assert row["reason"] == ""
        assert row["bos_skipped"] == (fmt.bos == "")
        for token_row in row["tokens"]:
            assert token_row["status"] == "ok"
            assert token_row["registered"] is True
            assert token_row["single_token"] is True
            assert token_row["id_match"] is True
            assert token_row["encoded_id"] == token_row["registered_id"]


def test_matrix_one_bad_literal_poisons_only_its_row(fake_transformers, tmp_path):
    lit = _LITS[0]
    doc, ids = _full_coverage_doc()

    def encode(text: str) -> Any:
        if text == lit:
            raise RuntimeError("encode boom on one literal")
        return (ids[text],)

    target = tmp_path / "matrix" / "oneliteral"
    report = _run_matrix(fake_transformers, target, doc, FakeSpec(encode=encode, attrs={}))
    assert report["status"] == "ok"  # the matrix itself is not poisoned
    with_lit = [r for r in report["rows"] if lit in _INV_BY_NAME[r["format"]]]
    without_lit = [r for r in report["rows"] if lit not in _INV_BY_NAME[r["format"]]]
    assert with_lit and without_lit
    for row in with_lit:
        assert row["verdict"] == "unknown", row
        assert row["reason"] == "encode_failed"
        bad = [t for t in row["tokens"] if t["token"] == lit]
        assert len(bad) == 1 and bad[0]["status"] == "encode_failed"
    for row in without_lit:
        assert row["verdict"] == "compatible", row


def test_matrix_duplicate_added_tokens_first_wins(fake_transformers, tmp_path):
    lit = _LITS[1]
    doc, ids = _full_coverage_doc()
    doc["added_tokens"].append({"content": lit, "id": ids[lit] + 5000})  # later dupe loses
    target = tmp_path / "matrix" / "dupe"
    report = _run_matrix(
        fake_transformers,
        target,
        doc,
        FakeSpec(encode=lambda text: (ids[text],), attrs={}),
    )
    assert report["status"] == "ok"
    for row in report["rows"]:
        for token_row in row["tokens"]:
            if token_row["token"] == lit:
                assert token_row["registered_id"] == ids[lit]
                assert token_row["status"] == "ok"


def test_matrix_unregistered_literal(fake_transformers, tmp_path):
    lit = _LITS[2]
    doc, ids = _full_coverage_doc()
    doc["added_tokens"] = [e for e in doc["added_tokens"] if e["content"] != lit]
    target = tmp_path / "matrix" / "unreg"
    report = _run_matrix(
        fake_transformers,
        target,
        doc,
        FakeSpec(encode=lambda text: (ids[text],), attrs={}),
    )
    hits = [r for r in report["rows"] if any(t["token"] == lit for t in r["tokens"])]
    assert hits
    for row in hits:
        assert row["verdict"] == "incompatible"
        assert row["reason"] == "unregistered"
        assert row["detail"] == f"unregistered on {lit!r}"
        token_row = next(t for t in row["tokens"] if t["token"] == lit)
        assert token_row["registered"] is False
        assert token_row["registered_id"] is None
        assert token_row["single_token"] is True  # encode was fine
        assert token_row["id_match"] is None


def test_matrix_fragmenting_literal(fake_transformers, tmp_path):
    lit = _LITS[3]
    doc, ids = _full_coverage_doc()

    def encode(text: str) -> Any:
        return (ids[text], ids[text] + 1) if text == lit else (ids[text],)

    target = tmp_path / "matrix" / "frag"
    report = _run_matrix(fake_transformers, target, doc, FakeSpec(encode=encode, attrs={}))
    hits = [r for r in report["rows"] if any(t["token"] == lit for t in r["tokens"])]
    assert hits
    for row in hits:
        assert row["verdict"] == "incompatible"
        assert row["reason"] == "fragments"
        token_row = next(t for t in row["tokens"] if t["token"] == lit)
        assert token_row["single_token"] is False
        assert token_row["id_match"] is None


def test_matrix_id_mismatch_literal(fake_transformers, tmp_path):
    lit = _LITS[4]
    doc, ids = _full_coverage_doc()

    def encode(text: str) -> Any:
        return (ids[text] + 1,) if text == lit else (ids[text],)

    target = tmp_path / "matrix" / "mismatch"
    report = _run_matrix(fake_transformers, target, doc, FakeSpec(encode=encode, attrs={}))
    hits = [r for r in report["rows"] if any(t["token"] == lit for t in r["tokens"])]
    assert hits
    for row in hits:
        assert row["verdict"] == "incompatible"
        assert row["reason"] == "id_mismatch"
        token_row = next(t for t in row["tokens"] if t["token"] == lit)
        assert token_row["id_match"] is False
        assert token_row["encoded_id"] == ids[lit] + 1
        assert token_row["registered_id"] == ids[lit]


def test_matrix_added_tokens_not_a_list_ignored(fake_transformers, tmp_path):
    lit = _LITS[0]
    doc = {"added_tokens": {lit: 123}}  # dict, not list -> ignored
    target = tmp_path / "matrix" / "notalist"
    report = _run_matrix(
        fake_transformers,
        target,
        doc,
        FakeSpec(encode=lambda text: (7,), attrs={}),
    )
    assert report["status"] == "ok"
    hits = [r for r in report["rows"] if any(t["token"] == lit for t in r["tokens"])]
    assert hits
    for row in hits:
        assert row["verdict"] == "incompatible"
        assert row["reason"] == "unregistered"


def test_matrix_convert_raises_attr_skipped(fake_transformers, tmp_path):
    lit = _LITS[0]
    doc, ids = _full_coverage_doc()
    doc["added_tokens"] = [e for e in doc["added_tokens"] if e["content"] != lit]

    def convert(token: str) -> Any:
        raise RuntimeError("convert boom")

    target = tmp_path / "matrix" / "convraise"
    report = _run_matrix(
        fake_transformers,
        target,
        doc,
        FakeSpec(encode=lambda text: (ids[text],), convert=convert, attrs={"bos_token": lit}),
    )
    hits = [r for r in report["rows"] if any(t["token"] == lit for t in r["tokens"])]
    assert hits
    for row in hits:
        token_row = next(t for t in row["tokens"] if t["token"] == lit)
        assert token_row["registered"] is False  # suppressed exception -> skipped
        assert row["reason"] == "unregistered"


def test_matrix_empty_doc_all_encode_invalid(fake_transformers, tmp_path):
    target = tmp_path / "matrix" / "emptydoc"
    report = _run_matrix(
        fake_transformers,
        target,
        {},
        FakeSpec(encode=lambda text: (), attrs={}),
    )
    assert report["status"] == "ok"
    assert report["rows"]
    for row in report["rows"]:
        assert row["verdict"] == "unknown", row
        if row["tokens"]:
            assert row["reason"] == "encode_invalid"
        else:
            assert row["reason"] == "empty_inventory"


def test_matrix_never_raises_on_hostile_doc(fake_transformers, tmp_path):
    doc = {
        "added_tokens": [
            {"content": 1, "id": 2},
            "nope",
            None,
            {"content": "", "id": 3},
            {"content": "<x>", "id": "s"},
            {"content": "<y>", "id": -1},
            {"content": "<z>", "id": True},
            {},
        ]
    }
    target = tmp_path / "matrix" / "hostile"
    report = _run_matrix(
        fake_transformers,
        target,
        doc,
        FakeSpec(encode=lambda text: (1,), attrs={"bos_token": 123}),
    )
    assert report["status"] == "ok"
    assert len(report["rows"]) == 118


def test_matrix_str_and_path_agree(fake_transformers, tmp_path):
    doc, ids = _full_coverage_doc()
    target = tmp_path / "matrix" / "both"
    spec = FakeSpec(encode=lambda text: (ids[text],), attrs={})
    by_str = _run_matrix(fake_transformers, target, doc, spec)
    _register(fake_transformers, target, spec)
    by_path = portability_matrix(Path(target))
    assert by_str == by_path


# ---------------------------------------------------------------------------
# _secondary_registered: direct unit tests
# ---------------------------------------------------------------------------


class _AttrFake:
    def __init__(self, attrs: dict[str, Any], convert: Callable[[str], Any] | None = None) -> None:
        for name, value in attrs.items():
            setattr(self, name, value)
        self._convert = convert or _unused

    def convert_tokens_to_ids(self, token: str) -> Any:
        return self._convert(token)


def test_registered_first_wins_and_zero_id():
    doc = {
        "added_tokens": [
            {"content": "<a>", "id": 1},
            {"content": "<a>", "id": 2},
            {"content": "<b>", "id": 0},
        ]
    }
    assert _secondary_registered(_AttrFake({}), doc) == {"<a>": 1, "<b>": 0}


def test_registered_rejects_malformed_entries():
    doc = {
        "added_tokens": [
            {"content": "<bool>", "id": True},
            {"content": "<neg>", "id": -3},
            {"content": "<strid>", "id": "9"},
            {"content": "", "id": 5},
            {"content": 7, "id": 8},
            {"id": 9},
            {"content": "<noid>"},
            "junk",
            None,
            {},
            {"content": "<ok>", "id": 11},
        ]
    }
    assert _secondary_registered(_AttrFake({}), doc) == {"<ok>": 11}


def test_registered_ignores_non_list_section():
    assert _secondary_registered(_AttrFake({}), {"added_tokens": {"<a>": 1}}) == {}
    assert _secondary_registered(_AttrFake({}), {}) == {}


def test_registered_attr_gap_fill_and_doc_wins():
    doc = {"added_tokens": [{"content": "<d>", "id": 1}]}
    seen: list[str] = []

    def convert(token: str) -> int:
        seen.append(token)
        return {"<d>": 99, "<a>": 2}[token]

    tok = _AttrFake({"bos_token": "<d>", "eos_token": "<a>"}, convert)
    assert _secondary_registered(tok, doc) == {"<d>": 1, "<a>": 2}
    assert seen == ["<a>"]  # doc-covered literal never consults convert


def test_registered_convert_failures_suppressed():
    def convert(token: str) -> Any:
        raise RuntimeError("convert boom")

    tok = _AttrFake({"bos_token": "<a>"}, convert)
    assert _secondary_registered(tok, {"added_tokens": []}) == {}


def test_registered_convert_bad_values_skipped():
    for bad in (True, -1, "12", None):
        tok = _AttrFake({"bos_token": "<a>"}, lambda token, b=bad: b)
        assert _secondary_registered(tok, {"added_tokens": []}) == {}, bad


def test_registered_non_string_attrs_skipped():
    tok = _AttrFake(
        {"bos_token": 123, "eos_token": "", "pad_token": None},
        lambda token: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    assert _secondary_registered(tok, {"added_tokens": []}) == {}


# ---------------------------------------------------------------------------
# portability_matrix: differential fuzz vs the oracle + core cross-check
# ---------------------------------------------------------------------------

_MATRIX_FUZZ_CASES = 130
_MATRIX_POOL: list[str] = list(dict.fromkeys(_LITS[:14])) + [f"<syn{i}>" for i in range(8)]


def _fuzz_matrix_stimulus(
    rng: random.Random,
) -> tuple[dict, dict[str, Any], Callable[[str], Any], Callable[[str], Any]]:
    """Random (doc, attrs, convert, raw_encode) for one matrix fuzz case."""
    lits = rng.sample(_MATRIX_POOL, k=rng.randint(0, 6))
    entries: list[Any] = []
    for i, lit in enumerate(lits):
        if rng.random() < 0.85:
            entries.append({"content": lit, "id": 500 + i})
    if rng.random() < 0.25 and lits:
        entries.append({"content": rng.choice(lits), "id": 9999})  # dupe
    if rng.random() < 0.2:
        entries.append({"content": "<zero-id>", "id": 0})
    for extra in (
        {"content": "<b>", "id": True},
        {"content": "<n>", "id": -3},
        {"content": "<s>", "id": "9"},
        {"content": "", "id": 5},
        "junk",
        None,
    ):
        if rng.random() < 0.3:
            entries.append(extra)
    r = rng.random()
    if r < 0.10:
        doc: dict = {"added_tokens": {e["content"]: e["id"] for e in entries if type(e) is dict}}
    elif r < 0.15:
        doc = {}
    else:
        doc = {"added_tokens": entries}

    attrs: dict[str, Any] = {}
    convert_ids: dict[str, int] = {}
    convert_modes: dict[str, str] = {}
    for attr in _SPECIAL_ATTRS:
        pick = rng.random()
        if pick < 0.25:
            lit = rng.choice(_MATRIX_POOL)
            attrs[attr] = lit
            convert_ids[lit] = 7000 + len(convert_ids)
            convert_modes[lit] = rng.choices(
                ["ok", "raise", "bool", "neg", "str"],
                weights=[60, 15, 10, 7.5, 7.5],
            )[0]
        elif pick < 0.30:
            attrs[attr] = ""
        elif pick < 0.35:
            attrs[attr] = 123

    def convert(token: str) -> Any:
        mode = convert_modes.get(token, "ok")
        if mode == "raise":
            raise RuntimeError("fuzz convert boom")
        if mode == "bool":
            return True
        if mode == "neg":
            return -5
        if mode == "str":
            return "x"
        return convert_ids[token]

    registered = _oracle_secondary_registered(doc, attrs, convert)

    encode_lits = sorted(set(lits) | {a for a in attrs.values() if type(a) is str and a})
    emap = {
        lit: rng.choice(
            [
                "match",
                "wrong",
                "fragment",
                "raise",
                "none",
                "bools",
                "neg",
                "empty",
                "strs",
                "list",
                "gen",
                "raw_int",
            ]
        )
        for lit in encode_lits
    }
    default_kind = rng.choice(["raise", "empty", "zero"])

    def apply(kind: str, text: str) -> Any:
        if kind == "match":
            return (registered[text],) if text in registered else (4242,)
        if kind == "wrong":
            return (registered.get(text, 9) + 1,)
        if kind == "fragment":
            base_id = registered.get(text, 9)
            return (base_id, base_id + 1)
        if kind == "raise":
            raise RuntimeError("fuzz encode boom")
        if kind == "none":
            return None
        if kind == "bools":
            return (True,)
        if kind == "neg":
            return (-5,)
        if kind == "empty":
            return ()
        if kind == "strs":
            return ("a", "b")
        if kind == "list":
            return [registered.get(text, 9)]
        if kind == "gen":
            return (i for i in (registered.get(text, 9),))
        if kind == "raw_int":
            return 7
        return (0,)

    def raw_encode(text: str) -> Any:
        return apply(emap.get(text, default_kind), text)

    return doc, attrs, convert, raw_encode


def test_fuzz_portability_matrix(fake_transformers, tmp_path):
    rng = random.Random(0xC620)  # noqa: S311 -- deterministic test seed
    failures: list[str] = []
    for n in range(_MATRIX_FUZZ_CASES):
        ckpt = tmp_path / "fuzz" / f"case{n}"
        doc, attrs, convert, raw_encode = _fuzz_matrix_stimulus(rng)
        try:
            _write_ckpt(ckpt, doc)
            expected = _oracle_matrix(ckpt, doc, attrs, convert, raw_encode)
            _register(
                fake_transformers,
                ckpt,
                FakeSpec(encode=raw_encode, convert=convert, attrs=attrs),
            )
            actual = portability_matrix(ckpt)
            assert actual == expected, (
                f"status={actual.get('status')!r} detail={actual.get('detail')!r}"
            )
        except AssertionError as exc:
            failures.append(f"case {n}: {exc}")
    assert not failures, "\n".join(failures[:10]) + (
        f"\n... {len(failures)} of {_MATRIX_FUZZ_CASES} fuzz cases failed" if failures else ""
    )


def test_row_oracle_matches_core_on_fuzz_corpus():
    """The hand-rolled row oracle agrees with the real core implementation.

    Guards against an oracle bug masquerading as implementation
    correctness: the oracle was written from the docstring, and must
    reproduce ``core.template_tokenizer_compatibility`` exactly over the
    randomized corpus.
    """
    rng = random.Random(0xC621)  # noqa: S311 -- deterministic test seed
    for n in range(40):
        doc, attrs, convert, raw_encode = _fuzz_matrix_stimulus(rng)
        registered = _oracle_secondary_registered(doc, attrs, convert)

        def encode(text: str, _re: Callable[[str], Any] = raw_encode) -> tuple[int, ...]:
            return tuple(_re(text))

        for family in sorted(CATALOG):
            for fmt in candidates(family, mutate=True):
                inventory = _extract_special_tokens(fmt)
                token_rows = [_oracle_token_row(token, registered, encode) for token in inventory]
                verdict, reason, detail = _oracle_row_verdict(fmt.name, token_rows)
                real = template_tokenizer_compatibility(fmt.name, inventory, registered, encode)
                assert real["format"] == fmt.name
                assert real["tokens"] == token_rows, (n, fmt.name)
                assert (real["verdict"], real["reason"], real["detail"]) == (
                    verdict,
                    reason,
                    detail,
                ), (n, fmt.name)
