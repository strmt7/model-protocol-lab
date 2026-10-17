"""Cycle 30: differential verification of model_protocol_lab/datasets.py.

An independent oracle is re-implemented here from the module *docstrings*
(structure and naming deliberately differ from the implementation; no
helpers are shared with it). The test-suite then:

* fuzzes the oracle against the implementation on generated JSON datasets
  (deterministic string seeds), covering the search/held-out split
  discipline, edge cases, boundary conditions, NFC normalization and size
  limits;
* pins behavioral facts the docstrings claim;
* runs mutation-style sensitivity controls on a COPY of the package in
  /tmp (the repo is never mutated) to prove the tests catch real bugs --
  see /tmp/cycle30_mutations/run_mutations.py;
* red-teams the module for hidden assumptions.

BUGS FOUND
----------
None. The implementation matches the oracle on 300 valid and 400 invalid
fuzzed JSON datasets plus targeted boundary sweeps, and all pinned
docstring claims hold, with the discrepancies below.

DOCSTRING / CODE DIVERGENCES (code behavior pinned; source untouched)
---------------------------------------------------------------------
D1. ``parse_dataset`` rejects a non-``str`` input (e.g. a ``str`` subclass)
    with the message "Dataset exceeds size limit": the ``type(text) is not
    str`` check shares the size-limit error branch, so the message is
    misleading even though the ``ValueError`` type is right. Pinned by
    ``TestPinnedBehavior::test_non_str_input_rejected``.
D2. A string containing lone surrogates fails the ``text.encode("utf-8")``
    size probe with ``UnicodeEncodeError`` (a ``ValueError`` subclass), so
    callers catching ``ValueError`` are unaffected, but the message is raw
    codec text rather than the module's error vocabulary. Pinned by
    ``TestPinnedBehavior::test_lone_surrogate_raises_valueerror``.
D3. The module docstring is a single sentence ("Bounded JSON fixture
    loading without executable configuration.") and the functions carry no
    docstrings, so the documented contract used for the oracle is that
    sentence plus the public bound constants and the module's uniform
    ``ValueError`` error vocabulary. Where the docstrings are silent (exact
    ``type(...) is ...`` checks instead of ``isinstance``, the
    ``RecursionError`` -> ``ValueError`` conversion, NFC-normalized split
    discipline inherited from ``core.Split``), the oracle mirrors the code
    and the tests pin the code behavior as fact rather than as documented
    promise.

MUTATION-SENSITIVITY GATE (documentation; the real run happens on copies
in /tmp -- see /tmp/cycle30_mutations/run_mutations.py).

The mutants below were verified caught by this test file on copies of the
package under /tmp/cycle30_mutations/mut_<n>/ (repo never mutated):
  M1  parse size check `>` -> `>=` (exactly-MAX_BYTES input rejected)
  M2  `MAX_MESSAGES = 21` -> `20`
  M3  `MAX_TEXT = 16_384` -> `16_385`
  M4  duplicate-key check reduced to `pass` (last wins)
  M5  role parity flipped (`index % 2 == 0` -> assistant first)
  M6  end-of-conversation check `!= "user"` -> `!= "assistant"`
  M7  `fd = -1` ownership handoff removed (double close -> OSError)
  M8  `type(text) is not str` -> `not isinstance(text, str)`
  M9  `stat.S_ISREG` -> `stat.S_ISDIR`
  M10 text length check `>` -> `>=` (exactly-MAX_TEXT rejected)
  M11 fixture count `<= MAX_CASES_PER_SPLIT` -> `<`
  M12 `MAX_CASES_PER_SPLIT = 100` -> `101`
  M13 `RecursionError` -> `ValueError` conversion removed
  M14 `_text` length check switched from characters to UTF-8 bytes
      (`len(value)` -> `len(value.encode("utf-8"))`)
  M15 `_fields` exact key-set check weakened to subset (extra keys allowed)
  M16 `expected` checked with `nonblank=True`
  M17 message-count lower bound `1` -> `0`
  M18 `UnicodeDecodeError` -> `ValueError` mapping removed
  M19 `OSError` -> "Unreadable fixture file" mapping removed
  M20 `handle.read(MAX_BYTES + 1)` -> `handle.read(MAX_BYTES)`
Result, verified 2026-10-15 via /tmp/cycle30_mutations/run_mutations.py:
20/20 mutants CAUGHT.
"""

import json
import os
import random
import stat
import tempfile
import threading
import unicodedata
import unittest
from pathlib import Path

from model_protocol_lab import datasets as datasets_mod
from model_protocol_lab.core import Fixture, Message, Split

# ---------------------------------------------------------------------------
# Independent oracle, written from the docstrings (module docstring: bounded
# JSON fixture loading without executable configuration; uniform ValueError
# vocabulary; the public bound constants below). Deliberately different
# structure and naming from model_protocol_lab/datasets.py; shares no
# helpers with it. Bound values are literals (not imported) so the oracle
# cannot silently follow a mutated implementation constant.
# ---------------------------------------------------------------------------

_REF_BYTE_CAP = 2_000_000  # documented bound on the UTF-8 dataset size
_REF_CASE_CAP = 100  # documented bound on fixtures per split
_REF_MESSAGE_CAP = 21  # documented bound on messages per fixture
_REF_TEXT_CAP = 16_384  # documented bound on any single text field


def _ref_reject_nonfinite(value):
    raise ValueError("reference oracle: non-finite JSON constant")


def _ref_object_hook(pairs):
    observed = set()
    for key, _ignored in pairs:
        if key in observed:
            raise ValueError("reference oracle: duplicate JSON field")
        observed.add(key)
    return dict(pairs)


def _ref_checked_text(value, *, allow_blank, where):
    if type(value) is not str or len(value) > _REF_TEXT_CAP:
        raise ValueError(f"reference oracle: invalid text at {where}")
    if not allow_blank and not value.strip():
        raise ValueError(f"reference oracle: blank identifier at {where}")
    return value


def _ref_fixture_from_mapping(mapping, label):
    if type(mapping) is not dict or set(mapping) != {"id", "group", "messages", "expected"}:
        raise ValueError(f"reference oracle: bad fixture fields at {label}")
    ident = _ref_checked_text(mapping["id"], allow_blank=False, where=f"{label}.id")
    group = _ref_checked_text(mapping["group"], allow_blank=False, where=f"{label}.group")
    expected = _ref_checked_text(mapping["expected"], allow_blank=True, where=f"{label}.expected")
    raw_messages = mapping["messages"]
    if type(raw_messages) is not list or not 1 <= len(raw_messages) <= _REF_MESSAGE_CAP:
        raise ValueError(f"reference oracle: bad message count at {label}")
    converted = []
    for position, item in enumerate(raw_messages):
        if type(item) is not dict or set(item) != {"role", "content"}:
            raise ValueError(f"reference oracle: bad message fields at {label}[{position}]")
        wanted = "user" if position % 2 == 0 else "assistant"
        if item["role"] != wanted:
            raise ValueError(f"reference oracle: alternation broken at {label}[{position}]")
        converted.append(
            Message(item["role"], _ref_checked_text(item["content"], allow_blank=True, where="msg"))
        )
    if converted[-1].role != "user":
        raise ValueError(f"reference oracle: conversation must end with user at {label}")
    return Fixture(ident, tuple(converted), expected, group)


def _ref_message_key(case):
    return tuple(
        (unicodedata.normalize("NFC", message.role), unicodedata.normalize("NFC", message.content))
        for message in case.messages
    )


def _ref_split_discipline(search_cases, held_cases):
    # Mirrors the documented core.Split contract (NFC-canonical uniqueness of
    # ids across the whole dataset; no group shared across the split; no
    # identical message inputs shared across the split), checked here
    # independently before the real Split constructor re-checks it.
    seen = set()
    for case in search_cases + held_cases:
        canon = unicodedata.normalize("NFC", case.id)
        if canon in seen:
            raise ValueError("reference oracle: duplicate fixture id")
        seen.add(canon)
    search_groups = {unicodedata.normalize("NFC", c.group) for c in search_cases}
    held_groups = {unicodedata.normalize("NFC", c.group) for c in held_cases}
    if search_groups & held_groups:
        raise ValueError("reference oracle: group crosses the split")
    search_keys = {_ref_message_key(c) for c in search_cases}
    held_keys = {_ref_message_key(c) for c in held_cases}
    if search_keys & held_keys:
        raise ValueError("reference oracle: identical message inputs cross the split")


def reference_parse_dataset(text):
    """Oracle re-implementation of datasets.parse_dataset from the docstrings."""
    if type(text) is not str or len(text.encode("utf-8")) > _REF_BYTE_CAP:
        raise ValueError("reference oracle: dataset too large or not text")
    try:
        document = json.loads(
            text,
            object_pairs_hook=_ref_object_hook,
            parse_constant=_ref_reject_nonfinite,
        )
    except RecursionError:
        raise ValueError("reference oracle: dataset nesting too deep") from None
    if type(document) is not dict or set(document) != {"search", "held_out"}:
        raise ValueError("reference oracle: top-level shape")
    partitions = []
    for name in ("search", "held_out"):
        entries = document[name]
        if type(entries) is not list or not 1 <= len(entries) <= _REF_CASE_CAP:
            raise ValueError("reference oracle: fixture count")
        partitions.append(
            tuple(
                _ref_fixture_from_mapping(entry, f"{name}[{i}]") for i, entry in enumerate(entries)
            )
        )
    search_cases, held_cases = partitions
    _ref_split_discipline(search_cases, held_cases)
    return Split(search_cases, held_cases)


def reference_load_dataset(path):
    """Oracle re-implementation of datasets.load_dataset from the docstrings."""
    if not isinstance(path, (str, Path)):
        raise ValueError("reference oracle: invalid fixture path")
    target = Path(path)
    try:
        handle_id = os.open(target, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        raise ValueError("reference oracle: unreadable fixture file") from None
    try:
        if not stat.S_ISREG(os.fstat(handle_id).st_mode):
            raise ValueError("reference oracle: fixture path is not a regular file")
        with os.fdopen(handle_id, "rb") as stream:
            handle_id = -1
            blob = stream.read(_REF_BYTE_CAP + 1)
    finally:
        if handle_id != -1:
            os.close(handle_id)
    if len(blob) > _REF_BYTE_CAP:
        raise ValueError("reference oracle: dataset exceeds size limit")
    try:
        return reference_parse_dataset(blob.decode("utf-8"))
    except UnicodeDecodeError:
        raise ValueError("reference oracle: fixture file is not valid UTF-8") from None


# ---------------------------------------------------------------------------
# Differential harness
# ---------------------------------------------------------------------------


def _capture(call):
    """Run ``call``; classify the outcome as value / ValueError / other."""
    try:
        return ("value", call())
    except ValueError:
        return ("verror", None)
    except Exception as exc:  # noqa: BLE001 -- a non-ValueError outcome is itself a finding
        return ("other", type(exc).__name__)


def _assert_agree(label, argument, *, expect_valid):
    kind_impl, value_impl = _capture(lambda: datasets_mod.parse_dataset(argument))
    kind_ref, value_ref = _capture(lambda: reference_parse_dataset(argument))
    if expect_valid:
        assert kind_impl == "value", f"{label}: implementation raised ({kind_impl})"
        assert kind_ref == "value", f"{label}: oracle raised ({kind_ref})"
        assert value_impl == value_ref, f"{label}: Split mismatch"
    else:
        assert kind_impl == "verror", (
            f"{label}: implementation -> {kind_impl} (expected ValueError)"
        )
        assert kind_ref == "verror", f"{label}: oracle -> {kind_ref} (expected ValueError)"


# ---------------------------------------------------------------------------
# Fuzz generators (deterministic string seeds)
# ---------------------------------------------------------------------------

_TEXT_POOL = [
    "",
    "hello",
    "What is 2+2?",
    "héllo wörld",
    "emoji 🌍 test",
    "tab\there",
    "newline\nin\ntext",
    'quote"q"',
    "back\\slash",
    "  padded  ",
    "café",
    "café",  # NFD form of café: distinct visual twin used for NFC checks
]


def _fresh_id(rng, seen_nfc, prefix):
    while True:
        base = f"{prefix}-{rng.randrange(1_000_000)}"
        roll = rng.random()
        if roll < 0.15:
            base = "caf\u00e9-" + base  # NFC form
        elif roll < 0.30:
            base = "cafe\u0301-" + base  # NFD form (NFC-distinct via suffix)
        canon = unicodedata.normalize("NFC", base)
        if canon not in seen_nfc:
            seen_nfc.add(canon)
            return base


def _random_messages(rng):
    count = rng.choice([1, 1, 3, 3, 5, 7])
    messages = []
    for index in range(count):
        content = rng.choice(_TEXT_POOL)
        if rng.random() < 0.2:
            content = "x" * rng.randint(0, 600)
        messages.append({"role": "user" if index % 2 == 0 else "assistant", "content": content})
    return messages


def _message_key(mapping_messages):
    return tuple(
        (unicodedata.normalize("NFC", m["role"]), unicodedata.normalize("NFC", m["content"]))
        for m in mapping_messages
    )


def _gen_valid_doc(rng, seq):
    seen_nfc = set()
    seen_keys = set()  # message-input keys must not repeat across the split
    search, held = [], []

    def make_fixture(prefix):
        for _attempt in range(60):
            messages = _random_messages(rng)
            key = _message_key(messages)
            if key not in seen_keys:
                seen_keys.add(key)
                break
        else:  # pragma: no cover -- virtually unreachable with the pools used
            messages = [{"role": "user", "content": f"unique-{prefix}-{rng.randrange(1 << 60)}"}]
            seen_keys.add(_message_key(messages))
        group = rng.choice(["sg-a", "sg-b"] if prefix == "s" else ["hg-a", "hg-b"])
        if rng.random() < 0.15:
            group = "gr\u00e9-" + prefix  # NFC group, confined to one split
        return {
            "id": _fresh_id(rng, seen_nfc, f"{prefix}{seq}"),
            "group": group,
            "messages": messages,
            "expected": "" if rng.random() < 0.25 else rng.choice(_TEXT_POOL),
        }

    for _ in range(rng.randint(1, 5)):
        search.append(make_fixture("s"))
    for _ in range(rng.randint(1, 5)):
        held.append(make_fixture("h"))
    return json.dumps({"search": search, "held_out": held}, ensure_ascii=False)


def _base_doc():
    # Note: the two fixtures differ in every discipline dimension (id, group
    # AND message inputs) so the base document is valid.
    def fixture(ident, group, content):
        return {
            "id": ident,
            "group": group,
            "messages": [{"role": "user", "content": content}],
            "expected": "e",
        }

    return {"search": [fixture("fs1", "gs", "cs")], "held_out": [fixture("fh1", "gh", "ch")]}


def _dump(doc):
    return json.dumps(doc, ensure_ascii=False)


# Each corruption takes (rng, doc-dict) and returns the fuzz input (usually a
# str; sometimes a non-str object for the type gate).
def _c_dup_key_fixture(rng, doc):
    return (
        '{"search": [{"id": "x", "id": "y", "group": "g", "messages": '
        '[{"role": "user", "content": "cs"}], "expected": "e"}], '
        '"held_out": [{"id": "z", "group": "h", "messages": '
        '[{"role": "user", "content": "ch"}], "expected": "e"}]}'
    )


def _c_dup_key_top(rng, doc):
    inner = _dump(doc["search"])
    return f'{{"search": {inner}, "search": {inner}, "held_out": {_dump(doc["held_out"])}}}'


def _c_dup_key_message(rng, doc):
    return (
        '{"search": [{"id": "x", "group": "g", "messages": '
        '[{"role": "user", "role": "assistant", "content": "cs"}], "expected": "e"}], '
        '"held_out": [{"id": "z", "group": "h", "messages": '
        '[{"role": "user", "content": "ch"}], "expected": "e"}]}'
    )


def _c_nan(rng, doc):
    return _dump(doc).replace('"expected": "e"', '"expected": NaN', 1)


def _c_infinity(rng, doc):
    return _dump(doc).replace('"id": "fs1"', '"id": Infinity', 1)


def _c_neg_infinity(rng, doc):
    return _dump(doc).replace('"expected": "e"', '"expected": -Infinity', 1)


def _c_missing_heldout(rng, doc):
    return _dump({"search": doc["search"]})


def _c_missing_search(rng, doc):
    return _dump({"held_out": doc["held_out"]})


def _c_extra_top(rng, doc):
    doc["zzz"] = 1
    return _dump(doc)


def _c_fixture_missing_group(rng, doc):
    del doc["search"][0]["group"]
    return _dump(doc)


def _c_fixture_extra_key(rng, doc):
    doc["search"][0]["note"] = "extra"
    return _dump(doc)


def _c_message_missing_role(rng, doc):
    del doc["search"][0]["messages"][0]["role"]
    return _dump(doc)


def _c_message_extra_key(rng, doc):
    doc["search"][0]["messages"][0]["lang"] = "en"
    return _dump(doc)


def _c_id_number(rng, doc):
    doc["search"][0]["id"] = 7
    return _dump(doc)


def _c_id_null(rng, doc):
    doc["search"][0]["id"] = None
    return _dump(doc)


def _c_content_null(rng, doc):
    doc["search"][0]["messages"][0]["content"] = None
    return _dump(doc)


def _c_content_number(rng, doc):
    doc["search"][0]["messages"][0]["content"] = 3.5
    return _dump(doc)


def _c_expected_list(rng, doc):
    doc["search"][0]["expected"] = ["e"]
    return _dump(doc)


def _c_group_bool(rng, doc):
    doc["search"][0]["group"] = True
    return _dump(doc)


def _c_search_dict(rng, doc):
    doc["search"] = {"oops": 1}
    return _dump(doc)


def _c_heldout_str(rng, doc):
    doc["held_out"] = "nope"
    return _dump(doc)


def _c_top_list(rng, doc):
    return _dump([doc])


def _c_top_str(rng, doc):
    return _dump("nope")


def _c_top_null(rng, doc):
    return _dump(None)


def _c_empty_search(rng, doc):
    doc["search"] = []
    return _dump(doc)


def _c_empty_heldout(rng, doc):
    doc["held_out"] = []
    return _dump(doc)


def _c_too_many_cases(rng, doc):
    doc["search"] = [
        {
            "id": f"bulk-{i}",
            "group": "gb",
            "messages": [{"role": "user", "content": "c"}],
            "expected": "e",
        }
        for i in range(101)
    ]
    return _dump(doc)


def _c_zero_messages(rng, doc):
    doc["search"][0]["messages"] = []
    return _dump(doc)


def _c_many_messages(rng, doc):
    doc["search"][0]["messages"] = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "c"} for i in range(22)
    ]
    return _dump(doc)


def _c_even_messages(rng, doc):
    doc["search"][0]["messages"] = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "c"}
        for i in range(rng.choice([2, 4, 6]))
    ]
    return _dump(doc)


def _c_bad_alternation(rng, doc):
    doc["search"][0]["messages"] = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    return _dump(doc)


def _c_bad_role(rng, doc):
    doc["search"][0]["messages"][0]["role"] = "system"
    return _dump(doc)


def _c_oversize_id(rng, doc):
    doc["search"][0]["id"] = "x" * 16_385
    return _dump(doc)


def _c_oversize_content(rng, doc):
    doc["search"][0]["messages"][0]["content"] = "y" * 16_385
    return _dump(doc)


def _c_blank_id(rng, doc):
    doc["search"][0]["id"] = "   "
    return _dump(doc)


def _c_blank_group(rng, doc):
    doc["search"][0]["group"] = ""
    return _dump(doc)


def _c_blank_group_ws(rng, doc):
    doc["search"][0]["group"] = "\t \n"
    return _dump(doc)


def _c_dup_id_across(rng, doc):
    doc["held_out"][0]["id"] = "fs1"
    return _dump(doc)


def _c_dup_id_within(rng, doc):
    doc["search"].append(dict(doc["search"][0], group="other"))
    return _dump(doc)


def _c_group_crosses(rng, doc):
    doc["held_out"][0]["group"] = "gs"
    return _dump(doc)


def _c_nfc_id_collision(rng, doc):
    doc["search"][0]["id"] = "caf\u00e9"
    doc["held_out"][0]["id"] = "cafe\u0301"
    return _dump(doc)


def _c_nfc_group_collision(rng, doc):
    doc["search"][0]["group"] = "gr\u00e9"
    doc["held_out"][0]["group"] = "gre\u0301"
    return _dump(doc)


def _c_identical_messages_across(rng, doc):
    # Same message inputs in both splits (ids and groups differ) -> rejected.
    doc["held_out"][0]["messages"] = [dict(m) for m in doc["search"][0]["messages"]]
    return _dump(doc)


def _c_nfc_message_collision(rng, doc):
    # NFC-equal message contents across the split -> rejected even though the
    # stored bytes differ.
    doc["search"][0]["messages"] = [{"role": "user", "content": "caf\u00e9"}]
    doc["held_out"][0]["messages"] = [{"role": "user", "content": "cafe\u0301"}]
    return _dump(doc)


def _c_truncated(rng, doc):
    return _dump(doc)[: len(_dump(doc)) // 2]


def _c_trailing_garbage(rng, doc):
    return _dump(doc) + " garbage"


def _c_trailing_comma(rng, doc):
    return _dump(doc).replace('{"search":', '{"search":,', 1)


def _c_empty_text(rng, doc):
    return ""


def _c_whitespace_text(rng, doc):
    return "   \n\t  "


def _c_none_input(rng, doc):
    return None


def _c_int_input(rng, doc):
    return 123


def _c_bytes_input(rng, doc):
    return b"{}"


def _c_list_input(rng, doc):
    return ["x"]


def _c_message_not_dict(rng, doc):
    doc["search"][0]["messages"][0] = "oops"
    return _dump(doc)


def _c_messages_not_list(rng, doc):
    doc["search"][0]["messages"] = {"role": "user", "content": "c"}
    return _dump(doc)


def _c_deep_nesting(rng, doc):
    return '{"search": ' + "[" * 20_000 + "]" * 20_000 + ', "held_out": []}'


CORRUPTIONS = [
    _c_dup_key_fixture,
    _c_dup_key_top,
    _c_dup_key_message,
    _c_nan,
    _c_infinity,
    _c_neg_infinity,
    _c_missing_heldout,
    _c_missing_search,
    _c_extra_top,
    _c_fixture_missing_group,
    _c_fixture_extra_key,
    _c_message_missing_role,
    _c_message_extra_key,
    _c_id_number,
    _c_id_null,
    _c_content_null,
    _c_content_number,
    _c_expected_list,
    _c_group_bool,
    _c_search_dict,
    _c_heldout_str,
    _c_top_list,
    _c_top_str,
    _c_top_null,
    _c_empty_search,
    _c_empty_heldout,
    _c_too_many_cases,
    _c_zero_messages,
    _c_many_messages,
    _c_even_messages,
    _c_bad_alternation,
    _c_bad_role,
    _c_oversize_id,
    _c_oversize_content,
    _c_blank_id,
    _c_blank_group,
    _c_blank_group_ws,
    _c_dup_id_across,
    _c_dup_id_within,
    _c_group_crosses,
    _c_nfc_id_collision,
    _c_nfc_group_collision,
    _c_identical_messages_across,
    _c_nfc_message_collision,
    _c_truncated,
    _c_trailing_garbage,
    _c_trailing_comma,
    _c_empty_text,
    _c_whitespace_text,
    _c_none_input,
    _c_int_input,
    _c_bytes_input,
    _c_list_input,
    _c_message_not_dict,
    _c_messages_not_list,
    _c_deep_nesting,
]


def _build_valid_cases():
    out = []
    for seq in range(300):
        rng = random.Random(f"cycle30-valid-{seq}")  # noqa: S311 -- deterministic test seed
        out.append(_gen_valid_doc(rng, seq))
    return out


def _build_invalid_cases():
    out = []
    for seq in range(400):
        rng = random.Random(f"cycle30-invalid-{seq}")  # noqa: S311 -- deterministic test seed
        corrupt = CORRUPTIONS[seq % len(CORRUPTIONS)]
        out.append((corrupt.__name__, corrupt(rng, _base_doc())))
    return out


VALID_CASES = _build_valid_cases()
INVALID_CASES = _build_invalid_cases()

# ---------------------------------------------------------------------------
# Differential fuzz tests: oracle vs implementation
# ---------------------------------------------------------------------------


class TestOracleDifferentialValid(unittest.TestCase):
    def test_valid_documents_agree(self):
        for doc in VALID_CASES:
            with self.subTest(doc=doc):
                _assert_agree("valid", doc, expect_valid=True)

    def setUp(self):
        self._td = tempfile.TemporaryDirectory(prefix="mpl-c107-")
        self.tmp_path = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()


class TestOracleDifferentialInvalid(unittest.TestCase):
    def test_invalid_documents_agree(self):
        for payload in [payload for _name, payload in INVALID_CASES]:
            with self.subTest(payload=payload):
                _assert_agree("invalid", payload, expect_valid=False)

    def setUp(self):
        self._td = tempfile.TemporaryDirectory(prefix="mpl-c107-")
        self.tmp_path = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


def _doc_with_byte_length(nbytes):
    # Distribute padding across many fixtures so every text field stays
    # within MAX_TEXT while the total UTF-8 byte length is exact.
    per_split = 64
    base_len = 1000

    def fixture(tag, i):
        suffix = f"-{tag}-{i:05d}"
        return {
            "id": f"pad-{tag}-{i:05d}",
            "group": f"pg-{tag}",
            "messages": [{"role": "user", "content": "x" * (base_len - len(suffix)) + suffix}],
            "expected": "e",
        }

    doc = {
        "search": [fixture("s", i) for i in range(per_split)],
        "held_out": [fixture("h", i) for i in range(per_split)],
    }
    text = _dump(doc)
    remaining = nbytes - len(text.encode("utf-8"))
    assert remaining >= 0, "base document already exceeds target size"
    buckets = doc["search"] + doc["held_out"]
    index = 0
    while remaining > 0:  # ASCII "x": one char == one byte
        content = buckets[index % len(buckets)]["messages"][0]["content"]
        room = datasets_mod.MAX_TEXT - len(content)
        assert room > 0, "padding headroom exhausted"
        add = min(room, remaining)
        buckets[index % len(buckets)]["messages"][0]["content"] = content + "x" * add
        remaining -= add
        index += 1
    out = _dump(doc)
    assert len(out.encode("utf-8")) == nbytes
    return out


def _doc_with_case_count(count):
    doc = _base_doc()

    def fixture(tag, i):
        # Content is unique per fixture so no message-input key repeats
        # across the split (the discipline under test here is the count).
        return {
            "id": f"case-{tag}{i}",
            "group": f"gb-{tag}",
            "messages": [{"role": "user", "content": f"c-{tag}{i}"}],
            "expected": "e",
        }

    doc["search"] = [fixture("s", i) for i in range(count)]
    doc["held_out"] = [fixture("h", i) for i in range(count)]
    return _dump(doc)


def _doc_with_message_count(count):
    doc = _base_doc()
    doc["search"][0]["messages"] = [
        # Search content "m" keeps the message-input key distinct from the
        # held-out fixture's "ch" even for a single-message conversation.
        {"role": "user" if i % 2 == 0 else "assistant", "content": "m"}
        for i in range(count)
    ]
    return _dump(doc)


def _set_field(doc, field, value):
    if field == "id":
        doc["search"][0]["id"] = value
    elif field == "group":
        doc["search"][0]["group"] = value
    elif field == "content":
        doc["search"][0]["messages"][0]["content"] = value
    elif field == "expected":
        doc["search"][0]["expected"] = value
    else:
        raise AssertionError(f"unknown field {field}")
    return _dump(doc)


class TestBoundaries(unittest.TestCase):
    def test_byte_limit_exact(self):
        doc = _doc_with_byte_length(datasets_mod.MAX_BYTES)
        _assert_agree("bytes-exact", doc, expect_valid=True)

    def test_byte_limit_plus_one(self):
        doc = _doc_with_byte_length(datasets_mod.MAX_BYTES + 1)
        _assert_agree("bytes-plus-one", doc, expect_valid=False)

    def test_case_count_ok(self):
        for count in [1, 2, 99, 100]:
            with self.subTest(count=count):
                _assert_agree(f"cases-{count}", _doc_with_case_count(count), expect_valid=True)

    def test_case_count_101_rejected(self):
        _assert_agree("cases-101", _doc_with_case_count(101), expect_valid=False)

    def test_message_count_ok(self):
        for count in [1, 3, 19, 21]:
            with self.subTest(count=count):
                _assert_agree(f"msgs-{count}", _doc_with_message_count(count), expect_valid=True)

    def test_message_count_22_rejected(self):
        _assert_agree("msgs-22", _doc_with_message_count(22), expect_valid=False)

    def test_message_count_20_rejected_ends_with_assistant(self):
        _assert_agree("msgs-20", _doc_with_message_count(20), expect_valid=False)

    def test_text_limit_exact_ok(self):
        for field in ["id", "group", "content", "expected"]:
            with self.subTest(field=field):
                doc = _set_field(_base_doc(), field, "x" * datasets_mod.MAX_TEXT)
                _assert_agree(f"text-exact-{field}", doc, expect_valid=True)

    def test_text_limit_plus_one_rejected(self):
        for field in ["id", "group", "content", "expected"]:
            with self.subTest(field=field):
                doc = _set_field(_base_doc(), field, "x" * (datasets_mod.MAX_TEXT + 1))
                _assert_agree(f"text-plus-one-{field}", doc, expect_valid=False)

    def test_text_limit_counts_characters_not_bytes(self):
        doc = _set_field(_base_doc(), "content", "\u00e9" * datasets_mod.MAX_TEXT)
        assert len(doc.encode("utf-8")) > datasets_mod.MAX_TEXT
        _assert_agree("text-multibyte", doc, expect_valid=True)

    def setUp(self):
        self._td = tempfile.TemporaryDirectory(prefix="mpl-c107-")
        self.tmp_path = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()


# ---------------------------------------------------------------------------
# Pinned behavioral facts (docstring claims + code facts)
# ---------------------------------------------------------------------------


class TestPinnedBehavior(unittest.TestCase):
    def test_non_str_input_rejected(self):
        class Sub(str):
            pass

        payload = Sub(_dump(_base_doc()))
        with self.assertRaisesRegex(ValueError, "Dataset exceeds size limit"):
            datasets_mod.parse_dataset(payload)
        with self.assertRaises(ValueError):
            reference_parse_dataset(payload)

    def test_lone_surrogate_raises_valueerror(self):
        doc = _dump(_base_doc()).replace("fs1", "a\ud800", 1)
        with self.assertRaises(ValueError) as info:
            datasets_mod.parse_dataset(doc)
        assert isinstance(info.exception, UnicodeEncodeError)

    def test_duplicate_id_across_splits_rejected(self):
        doc = _base_doc()
        doc["held_out"][0]["id"] = "fs1"
        with self.assertRaisesRegex(ValueError, "globally unique"):
            datasets_mod.parse_dataset(_dump(doc))

    def test_group_cannot_cross_split(self):
        doc = _base_doc()
        doc["held_out"][0]["group"] = "gs"
        with self.assertRaisesRegex(ValueError, "Related fixtures cannot cross the split"):
            datasets_mod.parse_dataset(_dump(doc))

    def test_conversation_must_end_with_user(self):
        doc = _base_doc()
        doc["search"][0]["messages"] = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        with self.assertRaisesRegex(ValueError, "must end with user"):
            datasets_mod.parse_dataset(_dump(doc))

    def test_alternation_must_start_with_user(self):
        doc = _base_doc()
        doc["search"][0]["messages"][0]["role"] = "assistant"
        with self.assertRaisesRegex(ValueError, "must alternate"):
            datasets_mod.parse_dataset(_dump(doc))

    def test_blank_id_rejected(self):
        doc = _base_doc()
        doc["search"][0]["id"] = "  \t "
        with self.assertRaisesRegex(ValueError, "Blank identifier"):
            datasets_mod.parse_dataset(_dump(doc))

    def test_blank_group_rejected(self):
        doc = _base_doc()
        doc["search"][0]["group"] = ""
        with self.assertRaisesRegex(ValueError, "Blank identifier"):
            datasets_mod.parse_dataset(_dump(doc))

    def test_whitespace_padded_id_accepted(self):
        doc = _base_doc()
        doc["search"][0]["id"] = "  padded-id  "
        split = datasets_mod.parse_dataset(_dump(doc))
        assert split.search[0].id == "  padded-id  "

    def test_empty_expected_accepted(self):
        doc = _base_doc()
        doc["search"][0]["expected"] = ""
        split = datasets_mod.parse_dataset(_dump(doc))
        assert split.search[0].expected == ""

    def test_top_level_key_order_irrelevant(self):
        doc = _base_doc()
        text = _dump({"held_out": doc["held_out"], "search": doc["search"]})
        split = datasets_mod.parse_dataset(text)
        assert split == reference_parse_dataset(text)

    def test_nfd_data_preserved_byte_exact(self):
        doc = _base_doc()
        doc["search"][0]["id"] = "cafe\u0301"
        doc["search"][0]["messages"][0]["content"] = "nai\u0308ve"
        split = datasets_mod.parse_dataset(_dump(doc))
        assert split.search[0].id == "cafe\u0301"
        assert split.search[0].messages[0].content == "nai\u0308ve"
        assert unicodedata.normalize("NFC", split.search[0].id) == "caf\u00e9"

    def test_nfc_id_collision_rejected(self):
        doc = _base_doc()
        doc["search"][0]["id"] = "caf\u00e9"  # NFC
        doc["held_out"][0]["id"] = "cafe\u0301"  # NFD, NFC-equal
        with self.assertRaisesRegex(ValueError, "globally unique"):
            datasets_mod.parse_dataset(_dump(doc))

    def test_nan_and_infinity_rejected(self):
        base = _dump(_base_doc())
        for token in ("NaN", "Infinity", "-Infinity"):
            text = base.replace('"expected": "e"', f'"expected": {token}', 1)
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                datasets_mod.parse_dataset(text)

    def test_deep_nesting_converted_to_valueerror(self):
        text = '{"search": ' + "[" * 20_000 + "]" * 20_000 + ', "held_out": []}'
        with self.assertRaisesRegex(ValueError, "nesting too deep"):
            datasets_mod.parse_dataset(text)

    def test_overflow_float_rejected(self):
        text = _dump(_base_doc()).replace('"id": "fs1"', '"id": 1e400', 1)
        with self.assertRaisesRegex(ValueError, "Invalid or oversized text field"):
            datasets_mod.parse_dataset(text)

    def test_empty_splits_rejected(self):
        doc = _base_doc()
        doc["search"] = []
        with self.assertRaisesRegex(ValueError, "Invalid fixture count"):
            datasets_mod.parse_dataset(_dump(doc))

    def test_zero_messages_rejected(self):
        doc = _base_doc()
        doc["search"][0]["messages"] = []
        with self.assertRaisesRegex(ValueError, "conversation must have 1-21 messages"):
            datasets_mod.parse_dataset(_dump(doc))

    def test_bom_rejected(self):
        with self.assertRaises(ValueError):
            datasets_mod.parse_dataset("\ufeff" + _dump(_base_doc()))

    def test_dunder_keys_are_inert_data(self):
        doc = _base_doc()
        doc["search"][0]["id"] = "__proto__"
        doc["search"][0]["messages"][0]["content"] = "${jndi:ldap://x}"
        split = datasets_mod.parse_dataset(_dump(doc))
        assert split.search[0].id == "__proto__"

    def test_parse_is_deterministic(self):
        text = VALID_CASES[7]
        assert datasets_mod.parse_dataset(text) == datasets_mod.parse_dataset(text)

    def setUp(self):
        self._td = tempfile.TemporaryDirectory(prefix="mpl-c107-")
        self.tmp_path = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


class TestLoadDataset(unittest.TestCase):
    def test_roundtrip_file(self):
        tmp_path = self.tmp_path
        text = VALID_CASES[3]
        target = tmp_path / "ds.json"
        target.write_text(text, encoding="utf-8")
        loaded = datasets_mod.load_dataset(target)
        assert loaded == datasets_mod.parse_dataset(text)
        assert loaded == reference_load_dataset(target)

    def test_fuzz_files_agree(self):
        tmp_path = self.tmp_path
        for index in [0, 11, 42, 101, 250]:
            with self.subTest(index=index):
                text = VALID_CASES[index]
                target = tmp_path / "ds.json"
                target.write_text(text, encoding="utf-8")
                assert datasets_mod.load_dataset(target) == reference_load_dataset(target)

    def test_missing_file(self):
        tmp_path = self.tmp_path
        with self.assertRaisesRegex(ValueError, "Unreadable fixture file"):
            datasets_mod.load_dataset(tmp_path / "nope.json")

    def test_directory_refused(self):
        tmp_path = self.tmp_path
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            datasets_mod.load_dataset(tmp_path)

    def test_dev_null_refused(self):
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            datasets_mod.load_dataset("/dev/null")

    def test_fifo_without_writer_refused(self):
        tmp_path = self.tmp_path
        fifo = tmp_path / "pipe"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            datasets_mod.load_dataset(fifo)

    def test_symlink_to_fifo_refused(self):
        tmp_path = self.tmp_path
        fifo = tmp_path / "pipe"
        os.mkfifo(fifo)
        link = tmp_path / "link"
        os.symlink(fifo, link)
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            datasets_mod.load_dataset(link)

    def test_invalid_utf8(self):
        tmp_path = self.tmp_path
        target = tmp_path / "bad.json"
        target.write_bytes(b'{"search": "\xed\xa0\x80"}')
        with self.assertRaisesRegex(ValueError, "not valid UTF-8"):
            datasets_mod.load_dataset(target)

    def test_bytes_path_rejected(self):
        tmp_path = self.tmp_path
        target = tmp_path / "ds.json"
        target.write_text(_dump(_base_doc()), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid fixture path"):
            datasets_mod.load_dataset(os.fsencode(target))

    def test_non_path_types_rejected(self):
        for bad in [None, 123, 4.5, ["x"], b"x"]:
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "Invalid fixture path"):
                datasets_mod.load_dataset(bad)

    def test_symlink_to_regular_file_accepted(self):
        tmp_path = self.tmp_path
        text = _dump(_base_doc())
        target = tmp_path / "ds.json"
        target.write_text(text, encoding="utf-8")
        link = tmp_path / "link.json"
        os.symlink(target, link)
        assert datasets_mod.load_dataset(link) == datasets_mod.parse_dataset(text)

    def test_dangling_symlink(self):
        tmp_path = self.tmp_path
        link = tmp_path / "dangling"
        os.symlink(tmp_path / "missing", link)
        with self.assertRaisesRegex(ValueError, "Unreadable fixture file"):
            datasets_mod.load_dataset(link)

    def test_symlink_loop(self):
        tmp_path = self.tmp_path
        link = tmp_path / "loop"
        os.symlink(link, link)
        with self.assertRaisesRegex(ValueError, "Unreadable fixture file"):
            datasets_mod.load_dataset(link)

    def test_trailing_slash_normalized_by_pathlib(self):
        tmp_path = self.tmp_path
        text = _dump(_base_doc())
        target = tmp_path / "ds.json"
        target.write_text(text, encoding="utf-8")
        assert datasets_mod.load_dataset(str(target) + "/") == datasets_mod.parse_dataset(text)
        with self.assertRaisesRegex(ValueError, "Unreadable fixture file"):
            datasets_mod.load_dataset(str(tmp_path / "missing.json") + "/")

    def test_exact_max_bytes_file_accepted(self):
        tmp_path = self.tmp_path
        text = _doc_with_byte_length(datasets_mod.MAX_BYTES)
        target = tmp_path / "ds.json"
        target.write_bytes(text.encode("utf-8"))
        assert target.stat().st_size == datasets_mod.MAX_BYTES
        assert datasets_mod.load_dataset(target) == reference_load_dataset(target)

    def test_max_bytes_plus_one_file_rejected(self):
        tmp_path = self.tmp_path
        text = _doc_with_byte_length(datasets_mod.MAX_BYTES + 1)
        target = tmp_path / "ds.json"
        target.write_bytes(text.encode("utf-8"))
        with self.assertRaisesRegex(ValueError, "Dataset exceeds size limit"):
            datasets_mod.load_dataset(target)

    def test_unreadable_file(self):
        tmp_path = self.tmp_path
        target = tmp_path / "noperm.json"
        target.write_text(_dump(_base_doc()), encoding="utf-8")
        target.chmod(0o000)
        try:
            if os.access(target, os.R_OK):
                self.skipTest("process can still read the file (e.g. running as root)")
            with self.assertRaisesRegex(ValueError, "Unreadable fixture file"):
                datasets_mod.load_dataset(target)
        finally:
            target.chmod(0o600)

    def test_empty_file(self):
        tmp_path = self.tmp_path
        target = tmp_path / "empty.json"
        target.write_bytes(b"")
        with self.assertRaises(ValueError):
            datasets_mod.load_dataset(target)

    def test_path_with_null_byte(self):
        with self.assertRaises(ValueError):
            datasets_mod.load_dataset("a\0b")

    def test_absurdly_long_path(self):
        with self.assertRaisesRegex(ValueError, "Unreadable fixture file"):
            datasets_mod.load_dataset("/" + "x" * 6000)

    def setUp(self):
        self._td = tempfile.TemporaryDirectory(prefix="mpl-c107-")
        self.tmp_path = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()


# ---------------------------------------------------------------------------
# Red team: hidden assumptions
# ---------------------------------------------------------------------------


class TestRedTeam(unittest.TestCase):
    def test_fifo_with_writer_still_refused(self):
        tmp_path = self.tmp_path
        fifo = tmp_path / "pipe"
        os.mkfifo(fifo)
        stop = threading.Event()

        def writer():
            fd = os.open(fifo, os.O_RDWR)  # O_RDWR on a fifo never blocks
            try:
                os.write(fd, _dump(_base_doc()).encode("utf-8"))
                stop.wait(timeout=15)
            finally:
                os.close(fd)

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        try:
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                datasets_mod.load_dataset(fifo)
        finally:
            stop.set()
            thread.join(timeout=15)
        assert not thread.is_alive()

    def test_oversize_file_never_fully_read(self):
        tmp_path = self.tmp_path
        target = tmp_path / "big.json"
        with open(target, "wb") as handle:
            handle.write(b'{"search": [{"id": "')
            handle.write(b"x" * (3 * datasets_mod.MAX_BYTES))
            handle.write(b'"}]}')
        with self.assertRaisesRegex(ValueError, "Dataset exceeds size limit"):
            datasets_mod.load_dataset(target)

    def test_deeply_nested_object_converted(self):
        text = '{"a":' * 30_000 + "null" + "}" * 30_000
        with self.assertRaisesRegex(ValueError, "nesting too deep"):
            datasets_mod.parse_dataset(text)

    def test_many_extra_keys_rejected(self):
        search_fixture = (
            '{"id": "x", "group": "g", '
            '"messages": [{"role": "user", "content": "c"}], "expected": "e"'
        )
        held_fixture = (
            '{"id": "y", "group": "h", '
            '"messages": [{"role": "user", "content": "c"}], "expected": "e"}'
        )
        pairs = ", ".join(f'"k{i}": {i}' for i in range(50))
        text = (
            '{"search": ['
            + search_fixture
            + ", "
            + pairs
            + '}], "held_out": ['
            + held_fixture
            + "]}"
        )
        with self.assertRaisesRegex(ValueError, "Unexpected or missing JSON fields"):
            datasets_mod.parse_dataset(text)

    def test_whitespace_only_splits_rejected(self):
        with self.assertRaises(ValueError):
            datasets_mod.parse_dataset('{"search": [], "held_out": []}')

    def test_groups_may_repeat_within_one_split(self):
        doc = _base_doc()
        doc["search"].append(
            {
                "id": "fs2",
                "group": "gs",
                "messages": [{"role": "user", "content": "c"}],
                "expected": "e",
            }
        )
        split = datasets_mod.parse_dataset(_dump(doc))
        assert len(split.search) == 2

    def test_ids_unique_within_split_enforced(self):
        doc = _base_doc()
        doc["search"].append(dict(doc["search"][0]))
        with self.assertRaisesRegex(ValueError, "globally unique"):
            datasets_mod.parse_dataset(_dump(doc))

    def test_single_message_conversation_valid(self):
        doc = _base_doc()  # one user message, no assistant turn yet
        split = datasets_mod.parse_dataset(_dump(doc))
        assert len(split.search[0].messages) == 1
        assert split.search[0].messages[0].role == "user"

    def test_maximal_conversation_shape(self):
        text = _doc_with_message_count(21)
        split = datasets_mod.parse_dataset(text)
        assert [m.role for m in split.search[0].messages] == ["user", "assistant"] * 10 + ["user"]

    def setUp(self):
        self._td = tempfile.TemporaryDirectory(prefix="mpl-c107-")
        self.tmp_path = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()


if __name__ == "__main__":
    unittest.main()
