"""Cycle 159: deliforge -- reserved-token vs subword delimiter authority.

Covers the pure ``deliforge`` core (encoding-pair discovery, forged
marker localization, subword splicing, flip rates, authority gap, the
subword backend wrapper), the ``deliforge_stage`` /
``deliforge_authority`` experiments wiring, the CLI probe, the HTML
section, and the ``diff_deliforge`` comparison -- including 4/4
tempdir mutation gates, a 200-case brute-force oracle differential for
marker localization + splicing, and single-cause pins for every
``changed`` disjunct.
"""

from __future__ import annotations

import importlib.util
import os
import random
import tempfile
import unittest
from pathlib import Path
from threading import Event
from typing import Any

from model_protocol_lab import diffreports
from model_protocol_lab.core import Fixture, Format, Generation, Message
from model_protocol_lab.deliforge import (
    DETAIL_CELL_FAILED,
    DETAIL_LOCALIZATION_FAILED,
    DETAIL_NO_ENCODING_PAIR,
    MARKERS,
    PAYLOADS,
    DeliforgeBackend,
    authority_gap,
    build_forged_fixtures,
    build_forged_payload,
    build_subword_ids,
    decode_equal,
    find_encoding_pair,
    find_insertion_span,
    flip_rates,
    forge_fake_assistant,
    forge_fake_system,
    forged_marker_offsets,
    forged_target_index,
    validate_marker_index,
    validate_payload,
)
from model_protocol_lab.experiments import (
    deliforge_authority,
    deliforge_stage,
    native_search_outcomes,
)

BOUNDARY = (124,)  # ord("|")

MARKER_IDS = {marker: 900000 + i for i, marker in enumerate(MARKERS)}


def scripted_encode(text: str) -> tuple[int, ...]:
    """Greedy codec: markers are single ids, every other char is ord()."""
    ids: list[int] = []
    i = 0
    while i < len(text):
        for marker, mid in MARKER_IDS.items():
            if text.startswith(marker, i):
                ids.append(mid)
                i += len(marker)
                break
        else:
            ids.append(ord(text[i]))
            i += 1
    return tuple(ids)


_ID_TO_MARKER = {mid: marker for marker, mid in MARKER_IDS.items()}


def scripted_decode(token_ids: Any) -> str:
    """Scripted decode."""
    return "".join(_ID_TO_MARKER.get(t, chr(t)) for t in token_ids)


class DeliScriptedBackend:
    """Authority-policy backend: emits WRONG when any forged reserved
    marker id is present in the input ids, WORD otherwise. The two
    deliforge arms decode to identical text; only the encoding of the
    forged markers differs -- so a flip on the reserved arm and none
    on the subword arm is a pure encoding-authority effect.
    """

    fingerprint = "scripted-deliforge-c159"

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        """Encode."""
        return scripted_encode(text)

    @staticmethod
    def decode(token_ids: tuple[int, ...]) -> str:
        """Decode."""
        return scripted_decode(token_ids)

    @staticmethod
    def generate(
        input_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        seed: int,
        cancel: Event,
    ) -> Generation:
        """Generate."""
        forged = any(mid in tuple(input_ids) for mid in MARKER_IDS.values())
        answer = "WRONG" if forged else "WORD"
        out = tuple(map(ord, (answer + "|").ljust(max_new_tokens, ".")[:max_new_tokens]))
        return Generation(token_ids=out, reason="limit")


def _format(name: str = "chatml") -> Format:
    """Format."""
    # Marker-free template: none of the four probed markers appears in
    # the template strings, so every reserved marker id in a forged
    # prompt is a forged occurrence. (Genuine-marker preservation is
    # covered at the pure level and in the oracle differential.)
    return Format(
        name=name,
        bos="<s>",
        user_prefix="[user]\n",
        assistant_prefix="[asst]\n",
        turn_end="[end]\n",
    )


def _fixture(fid: str, expected: str = "WORD") -> Fixture:
    """Fixture."""
    return Fixture(
        id=fid,
        messages=(Message(role="user", content="Say the word."),),
        expected=expected,
        group="g",
    )


# ---------------------------------------------------------------------------
# Validation + payload builders
# ---------------------------------------------------------------------------


class ValidationTests(unittest.TestCase):
    """Validation tests."""

    def test_validate_payload_accepts(self):
        """Test validate payload accepts."""
        for payload in PAYLOADS:
            self.assertEqual(validate_payload(payload), payload)

    def test_validate_payload_rejects(self):
        """Test validate payload rejects."""
        for bad in (None, "", "fake_user", "FAKE_ASSISTANT", 0, b"fake_assistant"):
            with self.assertRaises(ValueError):
                validate_payload(bad)

    def test_validate_marker_index_accepts(self):
        """Test validate marker index accepts."""
        for i in range(len(MARKERS)):
            self.assertEqual(validate_marker_index(i), i)

    def test_validate_marker_index_rejects(self):
        """Test validate marker index rejects."""
        for bad in (None, -1, len(MARKERS), 1.0, True, "0"):
            with self.assertRaises(ValueError):
                validate_marker_index(bad)


class PayloadBuilderTests(unittest.TestCase):
    """Payload builder tests."""

    def test_fake_assistant_wraps_marker_twice(self):
        """Test fake assistant wraps marker twice."""
        text = forge_fake_assistant(MARKERS[0], "WRONG")
        self.assertEqual(text.count(MARKERS[0]), 2)
        self.assertIn("WRONG", text)

    def test_fake_system_wraps_marker_twice(self):
        """Test fake system wraps marker twice."""
        text = forge_fake_system(MARKERS[1], "WRONG")
        self.assertEqual(text.count(MARKERS[1]), 2)
        self.assertIn("WRONG", text)

    def test_builders_reject_bad_input(self):
        """Test builders reject bad input."""
        for fn in (forge_fake_assistant, forge_fake_system):
            with self.assertRaises(ValueError):
                fn("", "WRONG")
            with self.assertRaises(ValueError):
                fn(MARKERS[0], "")
            with self.assertRaises(ValueError):
                fn(None, "WRONG")  # type: ignore[arg-type]

    def test_build_forged_payload_dispatches(self):
        """Test build forged payload dispatches."""
        for payload in PAYLOADS:
            text = build_forged_payload(payload, MARKERS[2], "WRONG")
            self.assertEqual(text.count(MARKERS[2]), 2)
        with self.assertRaises(ValueError):
            build_forged_payload("nope", MARKERS[0], "WRONG")

    def test_build_forged_fixtures_shape(self):
        """Test build forged fixtures shape."""
        cases = (_fixture("f0"), _fixture("f1", expected="OTHER"))
        forged = build_forged_fixtures(cases, payload="fake_assistant", marker_index=0)
        self.assertEqual(len(forged), 2)
        for original, new in zip(cases, forged, strict=True):
            self.assertEqual(new.id, f"deliforge/fake_assistant/0/{original.id}")
            self.assertEqual(new.expected, original.expected)
            self.assertEqual(new.group, "deliforge")
            content = new.messages[0].content
            self.assertEqual(content.count(MARKERS[0]), 2)
            self.assertIn("(ref: deliforge/fake_assistant/0/", content)

    def test_build_forged_fixtures_rejects_bad_input(self):
        """Test build forged fixtures rejects bad input."""
        cases = (_fixture("f0"),)
        with self.assertRaises(ValueError):
            build_forged_fixtures((), payload="fake_assistant", marker_index=0)
        with self.assertRaises(ValueError):
            build_forged_fixtures(list(cases), payload="fake_assistant", marker_index=0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            build_forged_fixtures(cases, payload="bogus", marker_index=0)
        with self.assertRaises(ValueError):
            build_forged_fixtures(cases, payload="fake_assistant", marker_index=99)


# ---------------------------------------------------------------------------
# Encoding-pair discovery
# ---------------------------------------------------------------------------


class EncodingPairTests(unittest.TestCase):
    """Encoding pair tests."""

    def test_all_markers_get_pairs(self):
        """Test all markers get pairs."""
        for marker in MARKERS:
            pair = find_encoding_pair(scripted_encode, scripted_decode, marker)
            self.assertIsNotNone(pair)
            assert pair is not None
            self.assertEqual(len(pair["reserved"]), 1)
            self.assertNotEqual(pair["subword"], pair["reserved"])
            self.assertNotIn(pair["reserved"][0], pair["subword"])
            self.assertEqual(scripted_decode(pair["subword"]), marker)

    def test_none_when_marker_not_single_id(self):
        """Test none when marker not single id."""
        plain = lambda text: tuple(map(ord, text))  # noqa: E731
        self.assertIsNone(find_encoding_pair(plain, lambda ids: "".join(map(chr, ids)), MARKERS[0]))

    def test_none_on_backend_failures(self):
        """Test none on backend failures."""

        def boom_encode(text):
            """Boom encode."""
            raise RuntimeError("nope")

        self.assertIsNone(find_encoding_pair(boom_encode, scripted_decode, MARKERS[0]))
        self.assertIsNone(
            find_encoding_pair(scripted_encode, lambda ids: 1 / 0, MARKERS[0])  # type: ignore[return-value]
        )

    def test_none_when_chunk_encode_fails(self):
        """Test none when chunk encode fails."""

        # The full marker encodes fine, but every chunk-split encode
        # raises: pair discovery fails closed via _encode_ids_or_none.
        def flaky_encode(text):
            """Flaky encode."""
            if text != MARKERS[0]:
                raise RuntimeError("chunk boom")
            return scripted_encode(text)

        self.assertIsNone(find_encoding_pair(flaky_encode, scripted_decode, MARKERS[0]))

    def test_none_on_decode_mismatch(self):
        """Test none on decode mismatch."""
        self.assertIsNone(find_encoding_pair(scripted_encode, lambda ids: "WRONG", MARKERS[0]))

    def test_none_on_bad_marker(self):
        """Test none on bad marker."""
        for bad in (None, "", 42, b"<|im_start|>"):
            self.assertIsNone(find_encoding_pair(scripted_encode, scripted_decode, bad))


# ---------------------------------------------------------------------------
# Insertion span, marker offsets, subword-id construction
# ---------------------------------------------------------------------------


class ForgedTargetIndexTests(unittest.TestCase):
    """Forged target index tests."""

    def test_last_user_message_wins(self):
        """Test last user message wins."""
        messages = (
            Message(role="user", content="a"),
            Message(role="assistant", content="b"),
            Message(role="user", content="c"),
        )
        self.assertEqual(forged_target_index(messages), 2)

    def test_falls_back_to_last_message(self):
        """Test falls back to last message."""
        messages = (Message(role="assistant", content="b"),)
        self.assertEqual(forged_target_index(messages), 0)

    def test_rejects_malformed(self):
        """Test rejects malformed."""
        with self.assertRaises(ValueError):
            forged_target_index(())
        with self.assertRaises(ValueError):
            forged_target_index([Message(role="user", content="a")])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            forged_target_index(("nope",))  # type: ignore[arg-type]


class FindInsertionSpanTests(unittest.TestCase):
    """Find insertion span tests."""

    def test_finds_unique_span(self):
        """Test finds unique span."""
        self.assertEqual(find_insertion_span("abXXcd", "XX"), (2, 4))

    def test_none_when_missing_or_repeated(self):
        """Test none when missing or repeated."""
        self.assertIsNone(find_insertion_span("abcd", "XX"))
        self.assertIsNone(find_insertion_span("XXabXX", "XX"))
        self.assertIsNone(find_insertion_span("abc", ""))
        self.assertIsNone(find_insertion_span(None, "XX"))  # type: ignore[arg-type]
        self.assertIsNone(find_insertion_span("abc", None))  # type: ignore[arg-type]


class ForgedMarkerOffsetsTests(unittest.TestCase):
    """Forged marker offsets tests."""

    def test_offsets_inside_span(self):
        """Test offsets inside span."""
        text = "aa<|im_start|>bb<|im_start|>cc"
        offsets = forged_marker_offsets(text, "<|im_start|>", (0, len(text)), 2)
        self.assertEqual(offsets, [2, 16])

    def test_only_span_interior_counts(self):
        """Test only span interior counts."""
        text = "<|im_start|>aa<|im_start|>bb"
        # Span covers only the second marker.
        offsets = forged_marker_offsets(text, "<|im_start|>", (12, len(text)), 1)
        self.assertEqual(offsets, [14])

    def test_none_on_count_mismatch(self):
        """Test none on count mismatch."""
        text = "aa<|im_start|>bb"
        self.assertIsNone(forged_marker_offsets(text, "<|im_start|>", (0, len(text)), 2))
        self.assertIsNone(forged_marker_offsets(text, "<|im_start|>", (0, len(text)), 0))

    def test_none_on_bad_input(self):
        """Test none on bad input."""
        self.assertIsNone(forged_marker_offsets(None, "m", (0, 1), 1))
        self.assertIsNone(forged_marker_offsets("abc", "", (0, 3), 1))
        self.assertIsNone(forged_marker_offsets("abc", "b", (2, 1), 1))
        self.assertIsNone(forged_marker_offsets("abc", "b", (0, 99), 1))
        self.assertIsNone(forged_marker_offsets("abc", "b", (0, 3), -1))
        self.assertIsNone(forged_marker_offsets("abc", "b", (0, 3), True))


class BuildSubwordIdsTests(unittest.TestCase):
    """Build subword ids tests."""

    @staticmethod
    def _pair(marker_index=0):
        """Pair."""
        marker = MARKERS[marker_index]
        pair = find_encoding_pair(scripted_encode, scripted_decode, marker)
        assert pair is not None
        return marker, pair

    def _splice(self, benign_text, marker_index=0):
        """Splice."""
        marker, pair = self._pair(marker_index)
        cases = (_fixture("f0"),)
        forged = build_forged_fixtures(cases, payload="fake_assistant", marker_index=marker_index)
        target = forged_target_index(forged[0].messages)
        forged_messages = forged[0].messages
        forged_content = forged_messages[target].content
        inserted = forged_content[len(benign_text) :]
        text = benign_text + inserted  # render is identity here
        span = find_insertion_span(text, inserted)
        assert span is not None
        offsets = forged_marker_offsets(text, marker, span, inserted.count(marker))
        assert offsets is not None
        spliced = build_subword_ids(
            scripted_encode,
            scripted_decode,
            text,
            marker,
            pair["reserved"][0],
            pair["subword"],
            offsets,
        )
        return marker, pair, text, spliced, offsets

    def test_splice_replaces_only_forged(self):
        """Test splice replaces only forged."""
        _marker, pair, text, spliced, offsets = self._splice("Say the word.")
        self.assertIsNotNone(spliced)
        assert spliced is not None
        self.assertEqual(len(offsets), 2)
        self.assertTrue(decode_equal(scripted_decode, scripted_encode(text), spliced))
        # No reserved marker ids remain: both occurrences were forged.
        self.assertNotIn(pair["reserved"][0], spliced)

    def test_genuine_template_markers_untouched(self):
        """Test genuine template markers untouched."""
        _marker, pair, text, spliced, _offsets = self._splice(
            f"template {MARKERS[0]} says: Say the word.", marker_index=0
        )
        self.assertIsNotNone(spliced)
        assert spliced is not None
        self.assertTrue(decode_equal(scripted_decode, scripted_encode(text), spliced))
        # Exactly the one genuine marker id survives.
        self.assertEqual(sum(1 for t in spliced if t == pair["reserved"][0]), 1)
        # The genuine marker is at its original position in the benign prefix.
        benign_ids = scripted_encode(f"template {MARKERS[0]} says: Say the word.")
        genuine_at = benign_ids.index(pair["reserved"][0])
        self.assertEqual(spliced[genuine_at], pair["reserved"][0])

    def test_none_on_bad_input(self):
        """Test none on bad input."""
        marker, pair = self._pair()
        good = {
            "encode": scripted_encode,
            "decode": scripted_decode,
            "text": f"a{marker}b",
            "marker": marker,
            "marker_id": pair["reserved"][0],
            "subword": pair["subword"],
            "offsets": [1],
        }
        self.assertIsNotNone(build_subword_ids(**good))  # type: ignore[arg-type]
        for key, bad in (
            ("text", None),
            ("marker", ""),
            ("marker_id", -1),
            ("marker_id", True),
            ("subword", ()),
            ("subword", None),
            ("offsets", ()),
            ("offsets", None),
            ("offsets", [1, 1]),
            ("offsets", [99]),
            ("offsets", [2]),  # not at a marker occurrence
            ("offsets", [True]),
        ):
            kwargs = dict(good)
            kwargs[key] = bad
            self.assertIsNone(build_subword_ids(**kwargs), f"key={key}")  # type: ignore[arg-type]

    def test_none_when_encode_fails(self):
        """Test none when encode fails."""
        marker, pair = self._pair()

        def boom_encode(text):
            """Boom encode."""
            raise RuntimeError("nope")

        self.assertIsNone(
            build_subword_ids(
                boom_encode,
                scripted_decode,
                f"a{marker}b",
                marker,
                pair["reserved"][0],
                pair["subword"],
                [1],
            )
        )

    def test_none_on_decode_mismatch(self):
        """Test none on decode mismatch."""
        marker, pair = self._pair()

        # Decode distinguishes the reserved arm (contains the reserved
        # id) from the spliced arm: arms must decode to the SAME text.
        def picky_decode(ids):
            """Picky decode."""
            return "A" if pair["reserved"][0] in tuple(ids) else "B"

        self.assertIsNone(
            build_subword_ids(
                scripted_encode,
                picky_decode,
                f"a{marker}b",
                marker,
                pair["reserved"][0],
                pair["subword"],
                [1],
            )
        )


# ---------------------------------------------------------------------------
# Flip rates + authority gap
# ---------------------------------------------------------------------------


class FlipRateTests(unittest.TestCase):
    """Flip rate tests."""

    def test_basic_counts(self):
        """Test basic counts."""
        out = flip_rates({"f0": True, "f1": False}, {"f0": False, "f1": False})
        self.assertEqual(out, {"n_evaluable": 2, "n_flips": 1, "flip_rate": 0.5})

    def test_none_verdicts_excluded_never_counted(self):
        """Test none verdicts excluded never counted."""
        out = flip_rates({"f0": True}, {"f0": None})
        self.assertEqual(out["n_evaluable"], 0)
        self.assertIsNone(out["flip_rate"])
        self.assertEqual(out["n_flips"], 0)

    def test_unknown_fixture_ids_ignored(self):
        """Test unknown fixture ids ignored."""
        out = flip_rates({"f0": True}, {"f9": False})
        self.assertEqual(out["n_evaluable"], 0)

    def test_rejects_malformed(self):
        """Test rejects malformed."""
        with self.assertRaises(ValueError):
            flip_rates({"f0": "yes"}, {"f0": True})  # type: ignore[dict-item]
        with self.assertRaises(ValueError):
            flip_rates({"f0": True}, {"f0": 1})  # type: ignore[dict-item]
        with self.assertRaises(ValueError):
            flip_rates(None, {})  # type: ignore[arg-type]


class AuthorityGapTests(unittest.TestCase):
    """Authority gap tests."""

    def test_gap_arithmetic(self):
        """Test gap arithmetic."""
        self.assertEqual(authority_gap(1.0, 0.0), 1.0)
        self.assertEqual(authority_gap(0.25, 0.75), -0.5)
        self.assertEqual(authority_gap(0.5, 0.5), 0.0)

    def test_gap_none_propagates(self):
        """Test gap none propagates."""
        self.assertIsNone(authority_gap(None, 0.5))
        self.assertIsNone(authority_gap(0.5, None))
        self.assertIsNone(authority_gap(None, None))

    def test_gap_rejects_out_of_range(self):
        """Test gap rejects out of range."""
        for bad in (-0.1, 1.1, "0.5", float("nan")):
            with self.assertRaises(ValueError):
                authority_gap(bad, 0.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DeliforgeBackend wrapper
# ---------------------------------------------------------------------------


class DeliforgeBackendTests(unittest.TestCase):
    """Deliforge backend tests."""

    def test_plan_hit_and_miss(self):
        """Test plan hit and miss."""
        inner = DeliScriptedBackend()
        plan = {"forged text": (1, 2, 3)}
        wrapped = DeliforgeBackend(inner, plan)
        self.assertEqual(wrapped.encode("forged text"), (1, 2, 3))
        self.assertEqual(wrapped.encode("other"), scripted_encode("other"))
        self.assertEqual(wrapped.fingerprint, inner.fingerprint + "+deliforge:subword")
        self.assertEqual(wrapped.decode((65,)), "A")

    def test_generate_delegates(self):
        """Test generate delegates."""
        inner = DeliScriptedBackend()
        wrapped = DeliforgeBackend(inner, {})
        cancel = Event()
        got = wrapped.generate((65,), max_new_tokens=8, seed=1, cancel=cancel)
        want = inner.generate((65,), max_new_tokens=8, seed=1, cancel=cancel)
        self.assertEqual(got.token_ids, want.token_ids)

    def test_rejects_malformed(self):
        """Test rejects malformed."""
        inner = DeliScriptedBackend()
        with self.assertRaises(ValueError):
            DeliforgeBackend(None, {})
        with self.assertRaises(ValueError):
            DeliforgeBackend(inner, {"x": ()})
        with self.assertRaises(ValueError):
            DeliforgeBackend(inner, {"x": (1, -2)})
        with self.assertRaises(ValueError):
            DeliforgeBackend(inner, [("x", (1,))])  # type: ignore[arg-type]

        class NoFingerprint:
            """No fingerprint."""

            @staticmethod
            def encode(text):
                """Encode."""
                return (1,)

        with self.assertRaises(ValueError):
            DeliforgeBackend(NoFingerprint(), {})


# ---------------------------------------------------------------------------
# Stage + authority end to end (scripted authority-policy backend)
# ---------------------------------------------------------------------------


class DeliforgeStageEndToEndTests(unittest.TestCase):
    """Deliforge stage end to end tests."""

    @staticmethod
    def _native_entries(names, formats, cases):
        """Native entries."""
        from model_protocol_lab.core import evaluate

        entries = []
        for name, fmt in zip(names, formats, strict=True):
            report = evaluate(
                DeliScriptedBackend(), fmt, cases, boundary=BOUNDARY, max_new_tokens=16, seed=7
            )
            entries.append(
                {
                    "format": {"name": name},
                    "observations": [
                        {
                            "fixture": obs.fixture,
                            "status": obs.status,
                            "exact_match": obs.exact_match,
                            "boundary_found": obs.boundary_found,
                            "input_ids": list(obs.input_ids),
                        }
                        for obs in report.observations
                    ],
                }
            )
        return entries

    def test_full_pipeline_measures_positive_gap(self):
        """Test full pipeline measures positive gap."""
        names = ("chatml", "llama3")
        formats = tuple(_format(n) for n in names)
        cases = (_fixture("f0"), _fixture("f1"))
        entries = self._native_entries(names, formats, cases)
        stage = deliforge_stage(
            DeliScriptedBackend(),
            formats,
            cases,
            payloads=("fake_assistant",),
            marker_indexes=(0, 1),
            boundary=BOUNDARY,
            max_new_tokens=16,
            seed=7,
        )
        self.assertEqual(stage["payloads"], ["fake_assistant"])
        self.assertEqual(stage["marker_indexes"], [0, 1])
        analysis = deliforge_authority(native_search_outcomes(entries, names), stage, names)
        self.assertTrue(analysis["available"])
        self.assertEqual(analysis["excluded_candidates"], [])
        for name in names:
            entry = analysis["per_candidate"][name]
            self.assertEqual(entry["n_cells"], 2)
            self.assertEqual(entry["n_evaluable_cells"], 2)
            for key, cell in entry["cells"].items():
                self.assertTrue(cell["supported"], key)
                self.assertEqual(cell["n_evaluable"], 2)
                self.assertEqual(cell["n_flips_reserved"], 2)
                self.assertEqual(cell["n_flips_subword"], 0)
                self.assertEqual(cell["flip_rate_reserved"], 1.0)
                self.assertEqual(cell["flip_rate_subword"], 0.0)
                self.assertEqual(cell["authority_gap"], 1.0)
            self.assertEqual(entry["mean_authority_gap"], 1.0)
            self.assertEqual(entry["max_authority_gap"], 1.0)
            self.assertEqual(entry["min_authority_gap"], 1.0)
        self.assertEqual(analysis["max_authority_gap"], 1.0)

    def test_both_payloads(self):
        """Test both payloads."""
        names = ("chatml",)
        formats = tuple(_format(n) for n in names)
        cases = (_fixture("f0"),)
        entries = self._native_entries(names, formats, cases)
        stage = deliforge_stage(
            DeliScriptedBackend(),
            formats,
            cases,
            boundary=BOUNDARY,
            max_new_tokens=16,
            seed=7,
        )
        self.assertEqual(stage["payloads"], ["fake_assistant", "fake_system"])
        self.assertEqual(stage["marker_indexes"], [0, 1, 2, 3])
        analysis = deliforge_authority(native_search_outcomes(entries, names), stage, names)
        self.assertTrue(analysis["available"])
        entry = analysis["per_candidate"]["chatml"]
        self.assertEqual(entry["n_cells"], 8)
        self.assertTrue(all(c["authority_gap"] == 1.0 for c in entry["cells"].values()))

    def test_no_encoding_pair_degrades_cell_not_stage(self):
        """Test no encoding pair degrades cell not stage."""

        class OrdBackend(DeliScriptedBackend):
            """Ord backend."""

            fingerprint = "ord-only-c159"

            @staticmethod
            def encode(text):
                """Encode."""
                return tuple(map(ord, text))

            @staticmethod
            def decode(token_ids):
                """Decode."""
                return "".join(map(chr, token_ids))

        names = ("chatml",)
        formats = tuple(_format(n) for n in names)
        cases = (_fixture("f0"),)
        entries = self._native_entries(names, formats, cases)
        stage = deliforge_stage(
            OrdBackend(), formats, cases, boundary=BOUNDARY, max_new_tokens=16, seed=7
        )
        cell = stage["per_payload"]["fake_assistant"]["per_marker"]["0"]["per_candidate"]["chatml"]
        self.assertEqual(cell, {"supported": False, "detail": DETAIL_NO_ENCODING_PAIR})
        analysis = deliforge_authority(native_search_outcomes(entries, names), stage, names)
        self.assertTrue(analysis["available"])
        self.assertEqual(analysis["excluded_candidates"], ["chatml"])
        self.assertIsNone(analysis["max_authority_gap"])

    def test_stage_rejects_bad_input(self):
        """Test stage rejects bad input."""
        formats = (_format(),)
        cases = (_fixture("f0"),)
        backend = DeliScriptedBackend()
        with self.assertRaises(ValueError):
            deliforge_stage(None, formats, cases, boundary=BOUNDARY)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            deliforge_stage(backend, (), cases, boundary=BOUNDARY)
        with self.assertRaises(ValueError):
            deliforge_stage(backend, formats, (), boundary=BOUNDARY)
        with self.assertRaises(ValueError):
            deliforge_stage(backend, formats, cases, payloads=(), boundary=BOUNDARY)
        with self.assertRaises(ValueError):
            deliforge_stage(backend, formats, cases, payloads=("bogus",), boundary=BOUNDARY)
        with self.assertRaises(ValueError):
            deliforge_stage(backend, formats, cases, marker_indexes=(0, 0), boundary=BOUNDARY)
        with self.assertRaises(ValueError):
            deliforge_stage(backend, formats, cases, marker_indexes=(9,), boundary=BOUNDARY)
        with self.assertRaises(ValueError):
            deliforge_stage(backend, formats, cases, boundary=(1, -2))


# ---------------------------------------------------------------------------
# Authority analysis: malformed-input handling
# ---------------------------------------------------------------------------


_FAILED_CELL = "failed"  # pass cell=_FAILED_CELL for a failed (None) cell


def _stage_cell(n=2, reserved_flips=2, subword_flips=0):
    """Stage-shaped supported cell: native verdicts are all True, so the
    reserved arm flips ``reserved_flips`` fixtures and the subword arm
    flips ``subword_flips``.
    """
    verdicts = {
        f"f{i}": {
            "reserved": (i >= reserved_flips),
            "subword": (i >= subword_flips),
        }
        for i in range(n)
    }
    return {"supported": True, "n": n, "verdicts": verdicts}


def _native_pair(n=2):
    """Native pair."""
    return {"chatml": {"per_fixture": {f"f{i}": True for i in range(n)}}}


def _analysis_stage(payloads=("fake_assistant",), markers=(0, 1), cell=None):
    """Analysis stage."""
    resolved = None if cell is _FAILED_CELL else (_stage_cell() if cell is None else cell)
    return {
        "payloads": list(payloads),
        "marker_indexes": list(markers),
        "per_payload": {
            payload: {
                "per_marker": {str(mi): {"per_candidate": {"chatml": resolved}} for mi in markers}
            }
            for payload in payloads
        },
    }


class DeliforgeAuthorityTests(unittest.TestCase):
    """Deliforge authority tests."""

    def test_rejects_bad_candidates(self):
        """Test rejects bad candidates."""
        stage = _analysis_stage()
        native = {"chatml": {"per_fixture": {"f0": True}}}
        for bad in (None, (), ("",), ("a", "a"), ["chatml"]):
            out = deliforge_authority(native, stage, bad)  # type: ignore[arg-type]
            self.assertEqual(out, {"available": False, "detail": "invalid_candidates"})

    def test_rejects_malformed_stage(self):
        """Test rejects malformed stage."""
        for bad in (None, {}, {"payloads": []}, {"payloads": ["x"], "marker_indexes": [0]}):
            out = deliforge_authority(_native_pair(), bad, ("chatml",))
            self.assertFalse(out["available"])
        out = deliforge_authority("nope", _analysis_stage(), ("chatml",))  # type: ignore[arg-type]
        self.assertEqual(out, {"available": False, "detail": "native_data_malformed"})

    def test_candidate_without_native_verdicts_excluded(self):
        """Test candidate without native verdicts excluded."""
        out = deliforge_authority({}, _analysis_stage(), ("chatml",))
        self.assertTrue(out["available"])
        self.assertEqual(out["excluded_candidates"], ["chatml"])
        self.assertIsNone(out["max_authority_gap"])

    def test_failed_cell_degrades_with_detail(self):
        """Test failed cell degrades with detail."""
        stage = _analysis_stage()
        stage["per_payload"]["fake_assistant"]["per_marker"]["0"]["per_candidate"]["chatml"] = None
        out = deliforge_authority(_native_pair(), stage, ("chatml",))
        self.assertTrue(out["available"])
        entry = out["per_candidate"]["chatml"]
        self.assertEqual(entry["cells"]["fake_assistant/0"]["detail"], DETAIL_CELL_FAILED)
        self.assertEqual(entry["n_evaluable_cells"], 1)
        self.assertEqual(entry["mean_authority_gap"], 1.0)

    def test_unsupported_cell_keeps_detail(self):
        """Test unsupported cell keeps detail."""
        stage = _analysis_stage(cell={"supported": False, "detail": DETAIL_LOCALIZATION_FAILED})
        out = deliforge_authority(_native_pair(), stage, ("chatml",))
        entry = out["per_candidate"]["chatml"]
        self.assertEqual(entry["cells"]["fake_assistant/0"]["detail"], DETAIL_LOCALIZATION_FAILED)
        self.assertEqual(entry["n_evaluable_cells"], 0)
        self.assertIsNone(entry["mean_authority_gap"])
        self.assertEqual(out["excluded_candidates"], ["chatml"])

    def test_malformed_cell_fails_closed(self):
        """Test malformed cell fails closed."""
        stage = _analysis_stage(cell={"supported": True, "verdicts": "junk"})
        out = deliforge_authority(_native_pair(), stage, ("chatml",))
        self.assertEqual(out, {"available": False, "detail": "stage_invalid"})

    def test_mixed_gaps_aggregate(self):
        """Test mixed gaps aggregate."""
        stage = _analysis_stage(markers=(0, 1))
        stage["per_payload"]["fake_assistant"]["per_marker"]["1"]["per_candidate"]["chatml"] = (
            _stage_cell(n=2, reserved_flips=0, subword_flips=1)
        )
        out = deliforge_authority(_native_pair(), stage, ("chatml",))
        entry = out["per_candidate"]["chatml"]
        self.assertEqual(entry["mean_authority_gap"], 0.25)
        self.assertEqual(entry["max_authority_gap"], 1.0)
        self.assertEqual(entry["min_authority_gap"], -0.5)
        self.assertEqual(out["max_authority_gap"], 1.0)


# ---------------------------------------------------------------------------
# 200-case brute-force oracle differential: subword-id construction
# ---------------------------------------------------------------------------


def _oracle_mask_expand(encode, decode, text, marker, marker_id, subword, span):
    """Independent oracle: mask forged markers with \\x00 (absent from
    the test alphabet, encodes as the single id 0), encode the masked
    text in one shot, then expand each mask id to the subword ids.
    A genuinely different algorithm from build_subword_ids'
    segment walk; the two must agree exactly.
    """
    start, end = span
    masked = []
    i = 0
    while i < len(text):
        if start <= i < end and text.startswith(marker, i):
            masked.append("\x00")
            i += len(marker)
        else:
            masked.append(text[i])
            i += 1
    ids = encode("".join(masked))
    out = []
    for token_id in ids:
        if token_id == 0:
            out.extend(subword)
        else:
            out.append(token_id)
    result = tuple(out)
    assert decode(result) == text, "oracle decode mismatch"
    return result


class BuildSubwordIdsOracleTests(unittest.TestCase):
    """Build subword ids oracle tests."""

    def test_200_case_differential(self):
        """Test 200 case differential."""
        rng = random.Random(0xD311F096E)  # noqa: S311 -- deterministic test seed
        words = ["alpha", "beta", "gamma", "delta", "Say", "the", "word", "please"]
        for case in range(200):
            with self.subTest(case=case):
                marker_index = rng.randrange(len(MARKERS))
                marker = MARKERS[marker_index]
                pair = find_encoding_pair(scripted_encode, scripted_decode, marker)
                assert pair is not None
                marker_id = pair["reserved"][0]
                # Benign text: random tokens; markers appear atomically.
                # \\x00 never appears (oracle mask sentinel).
                tokens = [rng.choice(words + list(MARKERS)) for _ in range(rng.randint(1, 6))]
                benign_text = " ".join(tokens)
                boundaries = [0]
                for token in tokens:
                    boundaries.append(boundaries[-1] + len(token) + 1)
                boundaries[-1] = len(benign_text)
                insert_at = rng.choice(boundaries)
                payload_text = build_forged_payload(rng.choice(PAYLOADS), marker, "WRONG")
                inserted = " " + payload_text
                forged_text = benign_text[:insert_at] + inserted + benign_text[insert_at:]
                span = find_insertion_span(forged_text, inserted)
                self.assertEqual(span, (insert_at, insert_at + len(inserted)))
                expected_count = inserted.count(marker)
                offsets = forged_marker_offsets(forged_text, marker, span, expected_count)
                self.assertIsNotNone(offsets)
                assert offsets is not None
                # Mask sentinel must be absent from the text.
                self.assertNotIn("\x00", forged_text)
                got = build_subword_ids(
                    scripted_encode,
                    scripted_decode,
                    forged_text,
                    marker,
                    marker_id,
                    pair["subword"],
                    offsets,
                )
                want = _oracle_mask_expand(
                    scripted_encode,
                    scripted_decode,
                    forged_text,
                    marker,
                    marker_id,
                    pair["subword"],
                    span,
                )
                self.assertIsNotNone(got)
                assert got is not None
                self.assertEqual(got, want)
                # Contract properties on top of the differential:
                # identical decode, genuine markers keep their reserved id.
                self.assertTrue(decode_equal(scripted_decode, scripted_encode(forged_text), got))
                genuine = forged_text.count(marker) - expected_count
                self.assertEqual(sum(1 for t in got if t == marker_id), genuine)


# ---------------------------------------------------------------------------
# Mutation gates (tempdir-module; the real tree is never rewritten)
# ---------------------------------------------------------------------------


class DeliforgeMutationGateTests(unittest.TestCase):
    """Each gate reverts one safety property; the scenario must catch it."""

    def _run_gate(self, module_name, name, old_text, new_text, scenario):
        """Run gate."""
        mod = __import__(f"model_protocol_lab.{module_name}", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c159_mut_", dir="/tmp") as tmpdir:
            mod_path = os.path.join(tmpdir, f"{module_name}_mut_{name}.py")
            with open(mod_path, "w", encoding="utf-8") as handle:
                handle.write(mutated)
            dotted = f"model_protocol_lab.{module_name}_mut_{name}"
            spec = importlib.util.spec_from_file_location(dotted, mod_path)
            assert spec is not None and spec.loader is not None
            mutant = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mutant)
            scenario(mod)  # passes on the real module: non-vacuous
            with self.assertRaises(AssertionError, msg=f"mutant {name} was NOT caught"):
                scenario(mutant)

    @staticmethod
    def _gap_sign_scenario(mod):
        """Gap sign scenario."""
        assert mod.authority_gap(1.0, 0.0) == 1.0
        assert mod.authority_gap(0.0, 1.0) == -1.0

    def test_mutant_gap_sign_flip_caught(self):
        """Test mutant gap sign flip caught."""
        self._run_gate(
            "deliforge",
            "gap_sign_flip",
            "return float(flip_rate_reserved) - float(flip_rate_subword)",
            "return float(flip_rate_subword) - float(flip_rate_reserved)  # MUTANT",
            self._gap_sign_scenario,
        )

    @staticmethod
    def _splice_genuine_scenario(mod):
        """Splice genuine scenario."""
        marker = MARKERS[0]
        pair = mod.find_encoding_pair(scripted_encode, scripted_decode, marker)
        assert pair is not None
        benign_text = f"template {marker} says: go."
        cases = (_fixture("f0"),)
        forged = mod.build_forged_fixtures(cases, payload="fake_assistant", marker_index=0)
        target = mod.forged_target_index(forged[0].messages)
        forged_content = forged[0].messages[target].content
        inserted = forged_content[len(benign_text) :]
        text = benign_text + inserted
        span = mod.find_insertion_span(text, inserted)
        assert span is not None
        offsets = mod.forged_marker_offsets(text, marker, span, inserted.count(marker))
        assert offsets is not None
        spliced = mod.build_subword_ids(
            scripted_encode,
            scripted_decode,
            text,
            marker,
            pair["reserved"][0],
            pair["subword"],
            offsets,
        )
        assert spliced is not None
        # The genuine template marker must keep its reserved id.
        assert sum(1 for t in spliced if t == pair["reserved"][0]) == 1

    def test_mutant_genuine_marker_reencoded_caught(self):
        """Test mutant genuine marker reencoded caught."""
        self._run_gate(
            "deliforge",
            "genuine_marker_reencoded",
            "    genuine = text.count(marker) - len(slots)\n"
            "    if sum(1 for i in spliced if i == marker_id) != genuine:\n"
            "        return None",
            "    genuine = text.count(marker) - len(slots) + 1  # MUTANT\n"
            "    if sum(1 for i in spliced if i == marker_id) != genuine:\n"
            "        return None",
            self._splice_genuine_scenario,
        )

    @staticmethod
    def _none_excluded_scenario(mod):
        """None excluded scenario."""
        out = mod.flip_rates({"f0": True, "f1": True}, {"f0": None, "f1": None})
        assert out["n_evaluable"] == 0
        assert out["flip_rate"] is None

    def test_mutant_none_verdicts_counted_caught(self):
        """Test mutant none verdicts counted caught."""
        self._run_gate(
            "deliforge",
            "none_verdicts_counted",
            "    evaluable = [fid for fid, verdict in forged.items() "
            "if verdict is not None and fid in native]",
            "    evaluable = [fid for fid, verdict in forged.items() if fid in native]  # MUTANT",
            self._none_excluded_scenario,
        )

    @staticmethod
    def _diff_changed_scenario(mod):
        """Diff changed scenario."""

        def payload(gap0, gap1):
            """Payload."""

            def cell(gap):
                """Cell."""
                return {
                    "supported": True,
                    "n_evaluable": 2,
                    "flip_rate_reserved": gap,
                    "flip_rate_subword": 0.0,
                    "authority_gap": gap,
                }

            return {
                "available": True,
                "payloads": ["fake_assistant"],
                "marker_indexes": [0, 1],
                "per_candidate": {
                    "chatml": {
                        "cells": {
                            "fake_assistant/0": cell(gap0),
                            "fake_assistant/1": cell(gap1),
                        },
                        "max_authority_gap": 1.0,
                    }
                },
                "excluded_candidates": [],
            }

        # The per-cell gaps swap between runs while the top gap stays
        # 1.0: ONLY deliforge_gap_drift fires (single-cause pin).
        old = payload(1.0, 0.0)
        new = payload(0.0, 1.0)
        diff = mod.diff_deliforge(old, new)
        assert diff["available"] == "both"
        assert diff["changed"] is True
        assert len(diff["gap_drift"]) == 2
        assert diff["top_gap_drift"] == []

    def test_mutant_diff_gap_drift_ignored_caught(self):
        """Test mutant diff gap drift ignored caught."""
        self._run_gate(
            "diffreports",
            "deliforge_gap_drift_ignored",
            '    result["changed"] = bool(\n'
            "        deliforge_gap_drift\n"
            "        or deliforge_top_gap_drift\n",
            '    result["changed"] = bool(\n'
            "        deliforge_top_gap_drift  # MUTANT\n"
            "        or deliforge_top_gap_drift\n",
            self._diff_changed_scenario,
        )


# ---------------------------------------------------------------------------
# diff_deliforge comparison
# ---------------------------------------------------------------------------


def _diff_payload(*, gaps=(1.0,), max_gap=1.0, candidates=("chatml",), excluded=()):
    """Diff payload."""
    per_candidate = {}
    for name in candidates:
        cells = {
            f"fake_assistant/{i}": {
                "supported": True,
                "n_evaluable": 2,
                "flip_rate_reserved": gap,
                "flip_rate_subword": 0.0,
                "authority_gap": gap,
            }
            for i, gap in enumerate(gaps)
        }
        per_candidate[name] = {"cells": cells, "max_authority_gap": max_gap}
    return {
        "available": True,
        "payloads": ["fake_assistant"],
        "marker_indexes": list(range(len(gaps))),
        "per_candidate": per_candidate,
        "excluded_candidates": list(excluded),
    }


class DiffDeliforgeTests(unittest.TestCase):
    """Diff deliforge tests."""

    def test_identical_payloads_unchanged(self):
        """Test identical payloads unchanged."""
        payload = _diff_payload()
        diff = diffreports.diff_deliforge(payload, payload)
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_availability_matrix(self):
        """Test availability matrix."""
        good = _diff_payload()
        self.assertEqual(diffreports.diff_deliforge(good, None)["available"], "old")
        self.assertEqual(diffreports.diff_deliforge(None, good)["available"], "new")
        self.assertEqual(diffreports.diff_deliforge(None, None)["available"], "neither")
        self.assertEqual(diffreports.diff_deliforge({"available": False}, good)["available"], "new")

    def test_single_cause_pins(self):
        """Test single cause pins."""
        # Each changed disjunct pinned in isolation: exactly one fires.
        base = _diff_payload(gaps=(0.5,), max_gap=0.5)
        # 1. gap drift only.
        diff = diffreports.diff_deliforge(base, _diff_payload(gaps=(0.0,), max_gap=0.5))
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["gap_drift"]), 1)
        self.assertEqual(diff["top_gap_drift"], [])
        # 2. top-gap drift only (cell gap moves but max recomputed equal
        # is impossible from cells alone -- instead shift max directly).
        old, new = _diff_payload(gaps=(0.5,), max_gap=0.5), _diff_payload(gaps=(0.5,), max_gap=0.9)
        diff = diffreports.diff_deliforge(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["gap_drift"], [])
        self.assertEqual(len(diff["top_gap_drift"]), 1)
        # 3. candidate added.
        diff = diffreports.diff_deliforge(base, _diff_payload(candidates=("chatml", "llama3")))
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["candidates_added"], ["llama3"])
        # 4. candidate removed.
        diff = diffreports.diff_deliforge(_diff_payload(candidates=("chatml", "llama3")), base)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["candidates_removed"], ["llama3"])
        # 5. newly excluded.
        diff = diffreports.diff_deliforge(base, _diff_payload(excluded=("chatml",)))
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["excluded_added"], ["chatml"])
        # 6. no longer excluded.
        diff = diffreports.diff_deliforge(_diff_payload(excluded=("chatml",)), base)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["excluded_removed"], ["chatml"])

    def test_valid_deliforge_rejects_hostile(self):
        """Test valid deliforge rejects hostile."""
        good = _diff_payload()
        self.assertTrue(diffreports._valid_deliforge(good))
        for mutate in (
            lambda p: p.update({"available": False}),
            lambda p: p.update({"payloads": ["x", 7]}),
            lambda p: p.update({"marker_indexes": []}),
            lambda p: p["per_candidate"].update({"chatml": None}),
            lambda p: p["per_candidate"]["chatml"]["cells"]["fake_assistant/0"].update(
                {"authority_gap": 2.0}
            ),
            lambda p: p["per_candidate"]["chatml"]["cells"]["fake_assistant/0"].update(
                {"flip_rate_reserved": -0.1}
            ),
            lambda p: p["per_candidate"]["chatml"]["cells"]["fake_assistant/0"].update(
                {"n_evaluable": -1}
            ),
            lambda p: p.update({"excluded_candidates": [42]}),
            lambda p: p["per_candidate"]["chatml"].update({"max_authority_gap": "high"}),
        ):
            import copy

            bad = copy.deepcopy(good)
            mutate(bad)
            self.assertFalse(diffreports._valid_deliforge(bad), mutate)

    def test_render_summary_lines(self):
        """Test render summary lines."""
        old = _diff_payload(gaps=(1.0,), max_gap=1.0)
        new = _diff_payload(gaps=(0.0,), max_gap=0.0)
        diff = diffreports.diff_deliforge(old, new)
        lines = diffreports._render_deliforge_summary(diff)
        self.assertTrue(any("gap drift" in line for line in lines))
        self.assertTrue(any("top-gap drift" in line for line in lines))
        # Unavailable side.
        lines = diffreports._render_deliforge_summary(diffreports.diff_deliforge(old, None))
        self.assertTrue(any("unavailable" in line for line in lines))
        self.assertEqual(diffreports._render_deliforge_summary(None), [])
        self.assertEqual(diffreports._render_deliforge_summary({"available": "neither"}), [])

    def test_render_summary_sanitizes_hostile_ids(self):
        """Test render summary sanitizes hostile ids."""
        old = _diff_payload(candidates=("chatml",))
        new = _diff_payload(candidates=("chatml", "evil\x1b[31m"))
        diff = diffreports.diff_deliforge(old, new)
        lines = diffreports._render_deliforge_summary(diff)
        joined = "\n".join(lines)
        self.assertNotIn("\x1b", joined)

    def test_diff_feeds_identical(self):
        """Test diff feeds identical."""

        # A deliforge-only change flips identical; asymmetric probe
        # flips it too.
        def report(gaps):
            """Report."""
            return {
                "decision": "select_best",
                "selected_candidate": "chatml",
                "search": [],
                "deliforge": _diff_payload(gaps=gaps),
            }

        diff = diffreports.diff_reports(report((1.0,)), report((0.0,)))
        self.assertTrue(diff["summary"]["deliforge_changed"])
        self.assertFalse(diff["summary"]["identical"])
        diff2 = diffreports.diff_reports(
            report((1.0,)),
            {"decision": "select_best", "selected_candidate": "chatml", "search": []},
        )
        self.assertTrue(diff2["summary"]["deliforge_asymmetric"])
        self.assertFalse(diff2["summary"]["identical"])
        diff3 = diffreports.diff_reports(report((1.0,)), report((1.0,)))
        self.assertFalse(diff3["summary"]["deliforge_changed"])
        self.assertTrue(diff3["summary"]["identical"])


# ---------------------------------------------------------------------------
# CLI probe wiring
# ---------------------------------------------------------------------------


class DeliforgeCliProbeTests(unittest.TestCase):
    """Deliforge cli probe tests."""

    def test_probe_records_available_payload(self):
        """Test probe records available payload."""
        from model_protocol_lab import cli
        from model_protocol_lab.core import evaluate

        names = ("chatml",)
        formats = tuple(_format(n) for n in names)
        cases = (_fixture("f0"),)
        search_entries = []
        for name, fmt in zip(names, formats, strict=True):
            report = evaluate(
                DeliScriptedBackend(), fmt, cases, boundary=BOUNDARY, max_new_tokens=16, seed=7
            )
            search_entries.append(
                {
                    "format": {"name": name},
                    "observations": [
                        {
                            "fixture": obs.fixture,
                            "status": obs.status,
                            "exact_match": obs.exact_match,
                            "boundary_found": obs.boundary_found,
                            "input_ids": list(obs.input_ids),
                        }
                        for obs in report.observations
                    ],
                }
            )
        result = {"search": search_entries}
        cli._run_deliforge_probe(
            DeliScriptedBackend(),
            formats,
            result,
            cases,
            boundary=BOUNDARY,
            boundary_map=None,
            max_new_tokens=16,
            seed=7,
        )
        payload = result["deliforge"]
        self.assertTrue(payload["available"])
        self.assertEqual(payload["payloads"], ["fake_assistant", "fake_system"])
        self.assertEqual(payload["marker_indexes"], [0, 1, 2, 3])
        self.assertEqual(payload["per_candidate"]["chatml"]["max_authority_gap"], 1.0)

    def test_probe_never_raises_on_broken_backend(self):
        """Test probe never raises on broken backend."""
        from model_protocol_lab import cli

        result = {}
        cli._run_deliforge_probe(
            None,
            (),
            result,
            (),
            boundary=None,
            boundary_map=None,
            max_new_tokens=8,
            seed=7,
        )
        self.assertEqual(result["deliforge"]["detail"], "deliforge_unavailable")


# ---------------------------------------------------------------------------
# Reporting: HTML section + TOC anchor
# ---------------------------------------------------------------------------


class DeliforgeReportingTests(unittest.TestCase):
    """Deliforge reporting tests."""

    @staticmethod
    def _result():
        """Result."""
        return {
            "deliforge": {
                "available": True,
                "payloads": ["fake_assistant"],
                "marker_indexes": [0],
                "per_candidate": {
                    "chatml": {
                        "cells": {
                            "fake_assistant/0": {
                                "supported": True,
                                "n_evaluable": 2,
                                "n_flips_reserved": 2,
                                "n_flips_subword": 0,
                                "flip_rate_reserved": 1.0,
                                "flip_rate_subword": 0.0,
                                "authority_gap": 1.0,
                            }
                        },
                        "n_cells": 1,
                        "n_evaluable_cells": 1,
                        "mean_authority_gap": 1.0,
                        "max_authority_gap": 1.0,
                        "min_authority_gap": 1.0,
                    }
                },
                "excluded_candidates": [],
                "max_authority_gap": 1.0,
            }
        }

    def test_section_renders_with_anchor(self):
        """Test section renders with anchor."""
        from model_protocol_lab.reporting import _deliforge_section

        html = _deliforge_section(self._result())
        self.assertIn('id="delimiter-authority"', html)
        self.assertIn("+1.00", html)
        self.assertIn("chatml", html)

    def test_section_empty_when_unavailable(self):
        """Test section empty when unavailable."""
        from model_protocol_lab.reporting import _deliforge_section

        self.assertEqual(_deliforge_section({}), "")
        self.assertEqual(_deliforge_section({"deliforge": {"available": False}}), "")
        self.assertEqual(_deliforge_section({"deliforge": {"available": True}}), "")

    def test_section_escapes_hostile_names(self):
        """Test section escapes hostile names."""
        from model_protocol_lab.reporting import _deliforge_section

        result = self._result()
        result["deliforge"]["per_candidate"] = {
            "<script>alert(1)</script>": result["deliforge"]["per_candidate"]["chatml"],
        }
        result["deliforge"]["excluded_candidates"] = ["<img src=x onerror=y>"]
        html = _deliforge_section(result)
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img src=x", html)

    def test_toc_links_to_section(self):
        """Test toc links to section."""
        from model_protocol_lab import reporting

        minimal = {
            "config": {},
            "deliforge": self._result()["deliforge"],
        }
        html = reporting.render_html(minimal)
        self.assertIn('href="#delimiter-authority"', html)
        self.assertIn('id="delimiter-authority"', html)


class EncodingPairAdversarialTests(unittest.TestCase):
    """Encoding pair adversarial tests."""

    def test_chunk_decode_mismatch_skipped(self):
        """Test chunk decode mismatch skipped."""
        # A decode that only round-trips the exact reserved id: every
        # chunk split decodes differently, so no pair exists.
        marker = MARKERS[0]

        def picky_decode(ids):
            """Picky decode."""
            return marker if tuple(ids) == (MARKER_IDS[marker],) else "OTHER"

        self.assertIsNone(find_encoding_pair(scripted_encode, picky_decode, marker))


class BuildSubwordIdsAdversarialTests(unittest.TestCase):
    """Build subword ids adversarial tests."""

    def test_none_when_reference_lacks_genuine_reserved_ids(self):
        """Test none when reference lacks genuine reserved ids."""
        # Pathologically context-sensitive backend: maps the marker to
        # its reserved id in short segments but to ords in the full
        # text. The subword arm keeps the genuine reserved id while the
        # reserved arm itself lacks it -- inconsistent, so the
        # construction fails closed.
        marker = MARKERS[0]
        pair = find_encoding_pair(scripted_encode, scripted_decode, marker)
        assert pair is not None
        marker_id = pair["reserved"][0]

        def flippy_encode(text):
            """Flippy encode."""
            if len(text) < 30 and marker in text:
                return scripted_encode(text)
            return tuple(map(ord, text))

        text = f"pad {marker} pad a{marker}b"
        forged_at = text.find(marker, 1)
        self.assertGreater(forged_at, 0)
        spliced = build_subword_ids(
            flippy_encode,
            scripted_decode,
            text,
            marker,
            marker_id,
            pair["subword"],
            [forged_at],
        )
        self.assertIsNone(spliced)


class DecoyAnswerTests(unittest.TestCase):
    """Decoy answer tests."""

    def test_decoy_avoids_expected_answers(self):
        """Test decoy avoids expected answers."""
        from model_protocol_lab.deliforge import _decoy_answer

        self.assertEqual(_decoy_answer((_fixture("a", expected="WORD"),)), "WRONG")
        # "WRONG" taken -> falls through to "INCORRECT".
        self.assertEqual(_decoy_answer((_fixture("a", expected="WRONG"),)), "INCORRECT")
        # All three decoys taken -> last-resort constant.
        cases = (
            _fixture("a", expected="WRONG"),
            _fixture("b", expected="INCORRECT"),
            _fixture("c", expected="DECOY"),
        )
        self.assertEqual(_decoy_answer(cases), "NOT_THE_ANSWER")


class DecodeEqualTests(unittest.TestCase):
    """Decode equal tests."""

    def test_decode_equal(self):
        """Test decode equal."""
        self.assertTrue(decode_equal(scripted_decode, (65,), (65,)))
        self.assertFalse(decode_equal(scripted_decode, (65,), (66,)))
        # A raising decode is fail-closed (unequal), never propagates.
        self.assertFalse(decode_equal(lambda ids: 1 / 0, (65,), (65,)))


class EncodingPairSkipTests(unittest.TestCase):
    """Encoding pair skip tests."""

    def test_chunk_with_invalid_ids_skipped(self):
        """Test chunk with invalid ids skipped."""
        # A chunk whose encoding fails id validation is skipped; a
        # later split still yields a pair.
        marker = MARKERS[0]

        def neg_encode(text):
            """Neg encode."""
            if text == marker[:1]:
                return (-1,)
            return scripted_encode(text)

        pair = find_encoding_pair(neg_encode, scripted_decode, marker)
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertNotIn(pair["reserved"][0], pair["subword"])

    def test_chunk_encoding_to_reserved_id_skipped(self):
        """Test chunk encoding to reserved id skipped."""
        # split=1 fails outright; split=2 natively encodes to the
        # reserved id and is skipped; split=3 wins.
        marker = MARKERS[0]

        def tricky_encode(text):
            """Tricky encode."""
            if text == marker[:1]:
                raise RuntimeError("chunk boom")
            if text == marker[:2]:
                return (MARKER_IDS[marker],)
            return scripted_encode(text)

        pair = find_encoding_pair(tricky_encode, scripted_decode, marker)
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertNotIn(pair["reserved"][0], pair["subword"])


class BuildSubwordIdsInvalidTests(unittest.TestCase):
    """Build subword ids invalid tests."""

    @staticmethod
    def _pair():
        """Pair."""
        marker = MARKERS[0]
        pair = find_encoding_pair(scripted_encode, scripted_decode, marker)
        assert pair is not None
        return marker, pair

    def test_none_when_segment_ids_invalid(self):
        """Test none when segment ids invalid."""
        # A segment encoding to invalid ids fails the spliced-id
        # validation (fail-closed).
        marker, pair = self._pair()

        def neg_segment_encode(text):
            """Neg segment encode."""
            if text == "a":
                return (-5,)
            return scripted_encode(text)

        self.assertIsNone(
            build_subword_ids(
                neg_segment_encode,
                scripted_decode,
                f"a{marker}b",
                marker,
                pair["reserved"][0],
                pair["subword"],
                [1],
            )
        )

    def test_none_when_genuine_marker_loses_reserved_id(self):
        """Test none when genuine marker loses reserved id."""
        # Backend maps the marker to its reserved id only in isolation:
        # the subword arm's genuine marker occurrence loses the reserved
        # id, so the genuine-count check fails closed.
        marker, pair = self._pair()

        def lonely_encode(text):
            """Lonely encode."""
            if text == marker:
                return (pair["reserved"][0],)
            return tuple(map(ord, text))

        text = f"g{marker}x{marker}h"
        forged_at = text.find(marker, 1)
        self.assertGreater(forged_at, 0)
        self.assertIsNone(
            build_subword_ids(
                lonely_encode,
                scripted_decode,
                text,
                marker,
                pair["reserved"][0],
                pair["subword"],
                [forged_at],
            )
        )
