"""Cycle 190: numform F3 — the validator re-parses the answer.

Before this cycle the diff validator trusted the stage's
stored parse (`answer_value`) when re-deriving the value/form
decomposition; red-team A proved by execution that a forged
`answer_value` with honestly recomputed aggregates validated
(F0, plus a None variant and a single-candidate variant).
Records now carry `answer_text` verbatim, bound by
`answer_digest == sha256(answer_text)`, and every validator
(diffreports mirror, cycle-27 hand mirror, cycle-67 oracle)
re-parses the text and requires equality with the stored
value. Compare answers stay digest-bound only (A2). The
amendments A1-A9 from the red-team briefs are pinned here:
the three-way differential parse battery, the fail-closed
surrogate digest, the explicit cell-absent clause, the
rambling-answer byte cap, and the reporting non-rendering
pin.
"""

from __future__ import annotations

import copy
import hashlib
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, "tests")

from test_cycle186_numform import FORMATS, _base_payload, _NumformBackend  # noqa: E402

from model_protocol_lab import diffreports, experiments, numform  # noqa: E402
from model_protocol_lab.core import Generation  # noqa: E402
from model_protocol_lab.diffreports import _valid_numform  # noqa: E402
from model_protocol_lab.reporting import render_html  # noqa: E402


def _honest_recompute(payload):
    """Splice honestly recomputed aggregates/contrasts into
    a mutated payload, so a rejection can only come from the
    record-level checks (the F0 attack shape).
    """
    recomputed = diffreports._numform_recompute(payload)
    assert recomputed is not None, "construction must pass for a splice"
    for name, expected in recomputed["aggregates"].items():
        payload["candidates"][name]["aggregates"] = expected["aggregates"]
        payload["candidates"][name]["contrasts"] = expected["contrasts"]
    return payload


def _producer_recompute(payload):
    """Recompute a mutated payload's readouts through the
    producer's own analysis path (which trusts the stored
    parse — that trust is exactly what F3 removes from the
    validator). After this splice every derived field is
    honest, so a validator rejection can only come from
    the new record-level binding.
    """
    stage = {
        "available": True,
        "detail": "",
        "protocol": payload["protocol"],
        "grouping": payload["grouping"],
        "candidates": {
            name: {
                "status": entry["status"],
                "detail": entry["detail"],
                "records": entry["records"],
            }
            for name, entry in payload["candidates"].items()
        },
    }
    fresh = experiments.numform_analysis(stage, tuple(payload["candidates"]))
    payload["candidates"] = fresh["candidates"]
    payload["grouping"] = fresh["grouping"]
    return payload


def _first_celled(payload, task=None):
    """First celled."""
    for entry in payload["candidates"].values():
        for record in entry["records"]:
            if record["cell"] is not None and (task is None or record["task"] == task):
                return record
    raise AssertionError("no celled record")


class TestStageRecords(unittest.TestCase):
    """Test stage records."""

    def test_answer_text_stored_and_bound(self):
        """Test answer text stored and bound."""
        payload = _base_payload()
        self.assertTrue(_valid_numform(payload))
        for entry in payload["candidates"].values():
            for record in entry["records"]:
                if record["cell"] is None:
                    self.assertIsNone(record["answer_text"])
                    self.assertIsNone(record["answer_value"])
                else:
                    self.assertIsInstance(record["answer_text"], str)
                    digest = hashlib.sha256(record["answer_text"].encode("utf-8")).hexdigest()
                    self.assertEqual(digest, record["cell"]["answer_digest"])
                    if record["task"] != "compare":
                        self.assertEqual(
                            numform.parse_value(record["answer_text"]),
                            record["answer_value"],
                        )

    def test_failed_cells_carry_no_text(self):
        """Test failed cells carry no text."""
        payload = _base_payload()
        backend = _NumformBackend(mode="faithful", fail_tasks={"copy"})
        stage = experiments.numform_stage(
            backend,
            FORMATS[:1],
            boundary=tuple(backend.encode("<|im_end|>")),
            max_new_tokens=64,
            max_context=8192,
            seed=0,
        )
        payload = experiments.numform_analysis(stage, (FORMATS[0].name,))
        self.assertTrue(_valid_numform(payload))
        for record in payload["candidates"][FORMATS[0].name]["records"]:
            if record["task"] == "copy":
                self.assertIsNone(record["cell"])
                self.assertIsNone(record["answer_text"])
                self.assertIsNone(record["answer_value"])


class _FixedAnswerBackend(_NumformBackend):
    """Answers every task with one fixed string."""

    def __init__(self, answer):
        super().__init__(mode="faithful")
        self.fixed = answer

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
        self.generate_calls += 1
        boundary = self.encode("<|im_end|>")[0]
        out = list(self.encode(self.fixed)) + [boundary]
        out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


class TestVerbatimStorage(unittest.TestCase):
    """A9: whitespace-padded and empty answers are stored
    verbatim; the payload still validates (the parse of a
    padded answer is None, which is a legal stored value).
    """

    @staticmethod
    def _payload_with(answer):
        """Payload with."""
        backend = _FixedAnswerBackend(answer)
        stage = experiments.numform_stage(
            backend,
            FORMATS[:1],
            boundary=tuple(backend.encode("<|im_end|>")),
            max_new_tokens=64,
            max_context=8192,
            seed=0,
        )
        return experiments.numform_analysis(stage, (FORMATS[0].name,))

    def test_padded_answer_verbatim(self):
        """Test padded answer verbatim."""
        # "1 000" with an ASCII space is not any grammar's
        # rendering (the space form uses U+202F): stored
        # verbatim, parses to None, verdict False, and the
        # payload is still a legal one.
        payload = self._payload_with("1 000")
        self.assertTrue(_valid_numform(payload))
        record = _first_celled(payload, task="copy")
        self.assertEqual(record["answer_text"], "1 000")
        self.assertIsNone(record["answer_value"])
        self.assertFalse(record["cell"]["verdict"])

    def test_empty_answer_verbatim(self):
        """Test empty answer verbatim."""
        payload = self._payload_with("")
        self.assertTrue(_valid_numform(payload))
        record = _first_celled(payload, task="copy")
        self.assertEqual(record["answer_text"], "")
        self.assertEqual(
            record["cell"]["answer_digest"],
            hashlib.sha256(b"").hexdigest(),
        )


class TestF0Regression(unittest.TestCase):
    """The executed cycle-190 hole, in all three variants:
    each forged payload has honestly recomputed aggregates,
    so only the new binding can reject it.
    """

    def test_forged_value_rejected(self):
        """Test forged value rejected."""
        payload = _base_payload()
        record = _first_celled(payload, task="copy")
        record["answer_value"] += 999
        _producer_recompute(payload)
        self.assertFalse(_valid_numform(payload))

    def test_none_variant_rejected(self):
        """Test none variant rejected."""
        payload = _base_payload()
        record = _first_celled(payload, task="copy")
        record["answer_value"] = None
        _producer_recompute(payload)
        self.assertFalse(_valid_numform(payload))

    def test_single_candidate_variant_rejected(self):
        """Test single candidate variant rejected."""
        payload = _base_payload()
        name = next(iter(payload["candidates"]))
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        record = next(
            r
            for r in payload["candidates"][name]["records"]
            if r["cell"] is not None and r["task"] == "copy"
        )
        record["answer_value"] += 999
        _producer_recompute(payload)
        self.assertFalse(_valid_numform(payload))


class TestBindingBattery(unittest.TestCase):
    """Test binding battery."""

    def test_answer_text_tampered(self):
        """Test answer text tampered."""
        payload = _base_payload()
        record = _first_celled(payload, task="copy")
        record["answer_text"] = f"{record['answer_text']}x"
        self.assertFalse(_valid_numform(payload))

    def test_text_and_digest_consistent_but_parse_forged(self):
        """Test text and digest consistent but parse forged."""
        # The forger does everything right except the parse:
        # text and digest agree with each other, the stored
        # value does not.
        payload = _base_payload()
        record = _first_celled(payload, task="copy")
        record["answer_text"] = "424242"
        record["cell"]["answer_digest"] = hashlib.sha256(b"424242").hexdigest()
        record["cell"]["verdict"] = False
        _producer_recompute(payload)
        self.assertFalse(_valid_numform(payload))

    def test_cell_absent_clauses(self):
        """Test cell absent clauses."""
        backend = _NumformBackend(mode="faithful", fail_tasks={"copy"})
        stage = experiments.numform_stage(
            backend,
            FORMATS[:1],
            boundary=tuple(backend.encode("<|im_end|>")),
            max_new_tokens=64,
            max_context=8192,
            seed=0,
        )
        base = experiments.numform_analysis(stage, (FORMATS[0].name,))
        for field, value in (("answer_value", 5), ("answer_text", "5")):
            payload = copy.deepcopy(base)
            # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
            record = next(
                r for r in payload["candidates"][FORMATS[0].name]["records"] if r["cell"] is None
            )
            record[field] = value
            self.assertFalse(_valid_numform(payload), field)

    def test_non_string_answer_text_rejected(self):
        """Test non string answer text rejected."""
        payload = _base_payload()
        record = _first_celled(payload, task="copy")
        record["answer_text"] = 123
        self.assertFalse(_valid_numform(payload))

    def test_surrogate_answer_text_fails_closed(self):
        """Test surrogate answer text fails closed."""
        # A3: a lone surrogate cannot be UTF-8 encoded; the
        # validator must answer False, never raise.
        payload = _base_payload()
        record = _first_celled(payload, task="copy")
        record["answer_text"] = "\ud800"
        self.assertFalse(_valid_numform(payload))

    def test_pre_f3_payload_rejected(self):
        """Test pre f3 payload rejected."""
        # A8: no migration shim — the exact record key set
        # rejects payloads written before answer_text existed.
        payload = _base_payload()
        for entry in payload["candidates"].values():
            for record in entry["records"]:
                del record["answer_text"]
        self.assertFalse(_valid_numform(payload))

    def test_compare_numeric_answer_is_legal(self):
        """Test compare numeric answer is legal."""
        # A2: a compare record answering "123" is a non_label
        # record; its stored value is None and the digest
        # binding alone governs it. A uniform parse check
        # would false-reject exactly this honest shape.
        payload = _base_payload()
        record = _first_celled(payload, task="compare")
        record["answer_text"] = "123"
        record["cell"]["answer_digest"] = hashlib.sha256(b"123").hexdigest()
        record["cell"]["verdict"] = False
        self.assertIsNone(record["answer_value"])
        _honest_recompute(payload)
        self.assertTrue(_valid_numform(payload))


class TestParseDifferential(unittest.TestCase):
    """A1: producer vs diffreports mirror over the grammar
    alphabet, plus the pattern-equality pin across all three
    implementations (the cycle-27 hand mirror is the third
    leg; its patterns are pinned textually here and its
    behaviour through the cycle-27 fuzz agreement).
    """

    @staticmethod
    def _battery():
        """Battery."""
        strings = set()
        for value in range(0, 2001):
            for form in numform.FORMS:
                strings.add(numform.render_form(value, form))
        strings.update(
            [
                "١٢٣",
                "１２３",
                "१२३",  # non-ASCII digits: int() accepts, grammar does not
                "123\n",
                "1,234\n",  # $-anchor trap: fullmatch rejects
                "1e0999",
                "1e+0999",
                "1e999",
                "1e005",
                "1e+999",
                "1.234e2",
                "1.230e2",
                "00.5e3",
                "007e2",
                "1,23,456",
                "0,001",
                "00,001",
                "1,234\u202f567",
                "",
                " ",
                "+42",
                "-42",
                "1_000",
                "1.5",
                "1e",
                "e3",
                "1e+",
                "9" * 20 + "e999",
                "0" * 21,
                "1" * 21,
                "12345678901234567890.5e3",
                "1,000,000,000,000,000,000,000",
                "1\u202f000\u202f000\u202f000\u202f000\u202f000\u202f000\u202f000",
                "123456789012345678901e3",
                "1e-3",
                "1E3",
                "1e+3",
                "12,34",
                "1,2345",
                "1 000",
                "1\u202f000",
                "0",
                "00",
            ]
        )
        # Deterministic pseudo-random grammar strings from a
        # fixed-seed LCG (no random module, per house style).
        alphabet = "0123456789,.eE+ \u202f"
        state = 190

        def _next(modulus):
            """Next."""
            nonlocal state
            state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
            return state % modulus

        for _ in range(3000):
            n = _next(13)
            strings.add("".join(alphabet[_next(len(alphabet))] for _ in range(n)))
        return sorted(strings)

    def test_mirror_matches_producer(self):
        """Test mirror matches producer."""
        for text in self._battery():
            self.assertEqual(
                numform.parse_value(text),
                diffreports._numform_parse_value(text),
                repr(text),
            )

    def test_pattern_texts_equal_across_implementations(self):
        """Test pattern texts equal across implementations."""
        producer = [
            numform._PLAIN_RE.pattern,
            numform._COMMA_RE.pattern,
            numform._SPACE_RE.pattern,
            numform._SCI_RE.pattern,
        ]
        mirror = [
            diffreports._NUMFORM_PLAIN_RE.pattern,
            diffreports._NUMFORM_COMMA_RE.pattern,
            diffreports._NUMFORM_SPACE_RE.pattern,
            diffreports._NUMFORM_SCI_RE.pattern,
        ]
        self.assertEqual(producer, mirror)
        source = Path("tests/test_cycle27_diffreports_verify.py").read_text(encoding="utf-8")
        oracle = re.findall(r'_re\.compile\(r"([^"]+)"\)', source)
        self.assertEqual(producer, oracle)


class TestRamblingCap(unittest.TestCase):
    """A5: answer text is byte-accounted; a checkpoint that
    rambles to a large token cap now fails the records cap
    closed instead of staying available. No cap change.
    """

    def test_rambling_answers_trip_records_cap(self):
        """Test rambling answers trip records cap."""
        # At the legal maximum (24 candidates x 192 records)
        # a ~6 KB answer per record projects to ~30 MB of
        # ground records — past the 12 MiB cap (B's F5
        # arithmetic: ~2,277 escaped bytes of headroom per
        # record over the faithful baseline).
        from dataclasses import replace

        formats = tuple(replace(FORMATS[0], name=f"nf-{i:02d}") for i in range(24))
        backend = _FixedAnswerBackend("x " * 3000)
        stage = experiments.numform_stage(
            backend,
            formats,
            boundary=tuple(backend.encode("<|im_end|>")),
            max_new_tokens=4096,
            max_context=100000,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "numform_records_too_large")


class TestReportingPin(unittest.TestCase):
    """A6: answer text is never rendered into the HTML
    report (counts and aggregates only).
    """

    def test_answer_text_not_rendered(self):
        """Test answer text not rendered."""
        payload = _base_payload()
        record = _first_celled(payload, task="copy")
        record["answer_text"] = "ZQXW-MARKER"
        record["cell"]["answer_digest"] = hashlib.sha256(b"ZQXW-MARKER").hexdigest()
        record["cell"]["verdict"] = False
        record["answer_value"] = None
        _honest_recompute(payload)
        self.assertTrue(_valid_numform(payload))
        html = render_html({"decision": "selected", "numform": payload})
        self.assertIn('id="numform"', html)
        self.assertNotIn("ZQXW-MARKER", html)


if __name__ == "__main__":
    unittest.main()
