"""Cycle 109: property-based audit of Split's anti-leakage guarantees.

The Split class (model_protocol_lab.core) is the lab's contamination
firewall: it enforces, on NFC-normalized keys, globally unique fixture IDs,
no shared groups across the search/held-out boundary, and no identical
message inputs across the boundary. Stored data stays byte-exact; only the
comparisons normalize.

Example-based tests pin specific cases (notably the NFD/NFC adversarial
cases in test_cycle66). This module uses Hypothesis to verify the
*universal* properties: for ARBITRARY strings, the NFC-evasion attacks
must always be caught, and valid constructions must always succeed.
"""

from __future__ import annotations

import unicodedata
import unittest

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from model_protocol_lab.core import Fixture, Message, Split


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _nfd(value: str) -> str:
    return unicodedata.normalize("NFD", value)


# Strings that are non-blank and have distinct NFC/NFD forms (i.e. contain
# at least one precomposed character that decomposes). These are the
# evasion candidates: byte-different but canonically equivalent.
def _e_nfc_nfd_pair() -> st.SearchStrategy[tuple[str, str]]:
    """Strategy yielding (nfc_form, nfd_form) of the same visual string."""
    base = st.sampled_from(["café", "naïve", "résumé", "Ångström", "Zürich", "piñata"])
    return base.map(lambda s: (_nfc(s), _nfd(s)))


_nonblank_text = st.text(min_size=1, max_size=20).filter(lambda s: s.strip())


def _message() -> st.SearchStrategy[Message]:
    return st.builds(
        Message,
        role=st.sampled_from(["user", "assistant"]),
        content=st.text(max_size=50),
    )


def _fixture(id_: str, group: str, messages: tuple[Message, ...]) -> Fixture:
    return Fixture(id=id_, messages=messages, expected="x", group=group)


def _simple_fixture(tag: str) -> Fixture:
    return _fixture(
        f"id-{tag}",
        f"group-{tag}",
        (Message(role="user", content=f"content-{tag}"),),
    )


class SplitNfcProperties(unittest.TestCase):
    @given(pair=_e_nfc_nfd_pair())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_nfc_nfd_id_collision_rejected(self, pair):
        """Byte-different but NFC-equal IDs are always rejected as duplicates."""
        nfc_id, nfd_id = pair
        assume(nfc_id != nfd_id)  # distinct byte sequences, same canonical form
        assume(_nfc(nfc_id) == _nfc(nfd_id))
        # The search fixture uses the NFC variant; held_out uses the NFD variant.
        search_nfc = (_fixture(nfc_id, "group-a", (Message(role="user", content="content-a"),)),)
        held = (_fixture(nfd_id, "group-b", (Message(role="user", content="other"),)),)
        with self.assertRaisesRegex(ValueError, "Fixture IDs must be globally unique"):
            Split(search=search_nfc, held_out=held)

    @given(pair=_e_nfc_nfd_pair())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_nfc_nfd_group_overlap_rejected(self, pair):
        """NFC-equal groups across the split are always rejected."""
        nfc_group, nfd_group = pair
        assume(nfc_group != nfd_group)
        search = (_fixture("id-a", nfc_group, (Message(role="user", content="q1"),)),)
        held = (_fixture("id-b", nfd_group, (Message(role="user", content="q2"),)),)
        with self.assertRaisesRegex(ValueError, "Related fixtures cannot cross the split"):
            Split(search=search, held_out=held)

    @given(pair=_e_nfc_nfd_pair())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_nfc_nfd_message_leakage_rejected(self, pair):
        """NFC-equal message content across the split is always rejected."""
        nfc_text, nfd_text = pair
        assume(nfc_text != nfd_text)
        search = (_fixture("id-a", "group-a", (Message(role="user", content=nfc_text),)),)
        held = (_fixture("id-b", "group-b", (Message(role="user", content=nfd_text),)),)
        with self.assertRaisesRegex(ValueError, "Identical message inputs cannot cross the split"):
            Split(search=search, held_out=held)

    @given(pair=_e_nfc_nfd_pair())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_byte_exact_storage_preserved(self, pair):
        """Valid Splits keep the original byte sequences; only comparisons normalize."""
        nfc_id, nfd_id = pair
        assume(nfc_id != nfd_id)
        # Use the NFD form in search and a DIFFERENT valid fixture in held_out.
        search = (_fixture(nfd_id, "group-a", (Message(role="user", content="alpha"),)),)
        held = (_fixture("id-b", "group-b", (Message(role="user", content="beta"),)),)
        split = Split(search=search, held_out=held)
        # The stored ID must be the exact NFD bytes, not normalized to NFC.
        self.assertEqual(split.search[0].id, nfd_id)
        self.assertNotEqual(split.search[0].id, nfc_id)

    @given(
        ids=st.lists(_nonblank_text, min_size=2, max_size=6, unique=True),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_unique_ids_groups_messages_always_construct(self, ids):
        """Distinct IDs, groups, and messages always build a valid Split."""
        # Ensure NFC-uniqueness too (unique=True is byte-level; NFC could collide).
        assume(len({_nfc(i) for i in ids}) == len(ids))
        n = len(ids) // 2
        search_ids, held_ids = ids[:n], ids[n:]
        assume(search_ids and held_ids)
        search = tuple(
            _fixture(f"s-{i}", f"gs-{i}", (Message(role="user", content=f"sq-{i}"),))
            for i in search_ids
        )
        held = tuple(
            _fixture(f"h-{i}", f"gh-{i}", (Message(role="user", content=f"hq-{i}"),))
            for i in held_ids
        )
        split = Split(search=search, held_out=held)
        self.assertEqual(len(split.search), len(search_ids))
        self.assertEqual(len(split.held_out), len(held_ids))


if __name__ == "__main__":
    unittest.main()
