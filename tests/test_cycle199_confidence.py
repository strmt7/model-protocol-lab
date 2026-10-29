"""Cycle 199: verbalized confidence (`run --confidence`, wave-8 rank 8).

A frozen suffix appended to each search fixture's last user message
asks the model to end its reply with ``Confidence: <integer 0-100>``;
the elicited generations are parsed under a frozen grammar and the
stated numbers are compared against the exact-match verdicts. A
generated pair -- planted-fact knowable cases vs their donor-key
controls -- tests whether the stated number drops when the answer
is not knowable from the prompt. This file pins the parser (a total
function with a mutation gate whose mutants are each killed by a
named battery case), the screens, the generated-case construction,
the stage's degradation rules, the diff validator's full-recompute
binding (including the carried line that derives the confidence
scalar), the diff section, the HTML section, and the CLI wrapper's
fail-closed behaviour. Design: docs/CONFIDENCE_DESIGN.md.
"""

from __future__ import annotations

import copy
import re
import unittest
from pathlib import Path
from threading import Event

from model_protocol_lab import cli, confidence, diffreports, experiments, reporting
from model_protocol_lab.core import Fixture, Format, Generation, Message

_FORMATS = (
    Format(name="alpha", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
    Format(name="beta", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
)


def _fx(fid, expected, group, content=None):
    """Fx."""
    return Fixture(
        id=fid,
        messages=(Message(role="user", content=content or f"question {fid}?"),),
        expected=expected,
        group=group,
    )


def _cases():
    """Cases."""
    return (
        _fx("s0", "42", "numbers"),
        _fx("s1", "Paris", "geo"),
        _fx("s2", "blue", "colors"),
        _fx("s3", "17", "numbers"),
    )


class _ScriptedBackend:
    """Answers from the prompt: operator fixtures get their
    expected answer plus a compliant line on treated prompts;
    generated cases are parsed (knowable -> stated value at 95,
    donor-key control -> "unknown" at the configured level).
    """

    def __init__(self, *, control_confidence=5, fingerprint="sha256:confidence-test"):
        self.control_confidence = control_confidence
        self.fingerprint = fingerprint
        self.dtype = "float32"
        self.declared_dtype = "float32"
        self.calls = 0

    @staticmethod
    def encode(text):
        """Encode."""
        return tuple(text.encode("utf-8"))

    @staticmethod
    def decode(ids):
        """Decode."""
        return bytes(int(i) % 256 for i in ids).decode("utf-8", "replace")

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        self.calls += 1
        prompt = self.decode(input_ids)
        answer = "?"
        bound = re.search(r"the code for (key-[0-9a-f]+) is ([0-9a-f]+)\.", prompt)
        asked = re.search(r"What is the code for (key-[0-9a-f]+)\?", prompt)
        if bound and asked:
            if bound.group(1) == asked.group(1):
                answer = f"{bound.group(2)}\nConfidence: 95"
            else:
                answer = f"unknown\nConfidence: {self.control_confidence}"
        else:
            expected = {"s0": "42", "s1": "Paris", "s2": "blue", "s3": "17"}
            for fid, exp in expected.items():
                if f"question {fid}?" in prompt:
                    answer = (
                        exp if "Confidence: <number>" not in prompt else f"{exp}\nConfidence: 90"
                    )
                    break
        ids = tuple((answer + "\x07").encode("utf-8"))
        ids = ids + (0,) * max(0, max_new_tokens - len(ids))
        return Generation(token_ids=ids, reason="limit")


def _native_result(backend, cases, *, max_new_tokens=64):
    """Native result."""
    return experiments.run_experiment(
        backend,
        _FORMATS,
        experiments.Split(search=cases, held_out=(_fx("h0", "held", "heldout"),)),
        boundary=(7,),
        max_new_tokens=max_new_tokens,
        seed=0,
    )


def _stage_payload(backend=None, cases=None, *, seed=3, result=None):
    """Stage payload."""
    backend = _ScriptedBackend() if backend is None else backend
    cases = _cases() if cases is None else cases
    result = _native_result(backend, cases) if result is None else result
    return experiments.confidence_stage(
        backend, _FORMATS, cases, result, boundary=(7,), max_new_tokens=64, seed=seed
    )


# ---------------------------------------------------------------------------
# The parser battery. Every entry is a named case; the mutation gate
# below names which case kills which mutant.
# ---------------------------------------------------------------------------

_BATTERY = [
    ("Paris\nConfidence: 85", ("ok", 85, "Paris")),
    ("Confidence: 50", ("ok", 50, "")),
    ("Confidence: 100", ("ok", 100, "")),
    ("Confidence: 0", ("ok", 0, "")),
    (" \nConfidence: 85", ("ok", 85, " ")),
    ("Paris\nConfidence: 85\n", ("ok", 85, "Paris")),
    ("", ("missing_line", None, "")),
    ("\n", ("missing_line", None, "\n")),
    ("Paris", ("missing_line", None, "Paris")),
    ("Paris Confidence: 85", ("missing_line", None, "Paris Confidence: 85")),
    ("Paris\nConfidence: +85", ("malformed_line", None, "Paris\nConfidence: +85")),
    ("Paris\nConfidence: 085", ("malformed_line", None, "Paris\nConfidence: 085")),
    ("Paris\nConfidence: ８５", ("malformed_line", None, "Paris\nConfidence: ８５")),
    ("Paris\nConfidence: 8_5", ("malformed_line", None, "Paris\nConfidence: 8_5")),
    ("Paris\nConfidence: 85 ", ("malformed_line", None, "Paris\nConfidence: 85 ")),
    ("Paris\nConfidence:85", ("malformed_line", None, "Paris\nConfidence:85")),
    ("Paris\nconfidence: 85", ("malformed_line", None, "Paris\nconfidence: 85")),
    ("Confidence: 101", ("malformed_line", None, "Confidence: 101")),
    ("Paris\r\nConfidence: 85\r", ("malformed_line", None, "Paris\r\nConfidence: 85\r")),
    (
        "Paris\nConfidence: 85\nConfidence: 90",
        ("multiple_lines", None, "Paris\nConfidence: 85\nConfidence: 90"),
    ),
    (
        "Paris\nConfidence: 85\nConfidence: ",
        ("malformed_line", None, "Paris\nConfidence: 85\nConfidence: "),
    ),
    (
        "Paris\nConfidence: 85\nConfidence: 90x",
        ("malformed_line", None, "Paris\nConfidence: 85\nConfidence: 90x"),
    ),
    (
        "Paris\nConfidence: 85\nSome prose",
        ("confidence_not_final", None, "Paris\nConfidence: 85\nSome prose"),
    ),
    (
        "Paris\nConfidence: 85\n\n",
        ("confidence_not_final", None, "Paris\nConfidence: 85\n\n"),
    ),
]


class TestParserBattery(unittest.TestCase):
    """Test parser battery."""

    def test_battery_exact(self):
        """Test battery exact."""
        for text, want in _BATTERY:
            with self.subTest(text=text):
                self.assertEqual(confidence.parse_verbalized_confidence(text), want)

    def test_kill_m1a_sign(self):
        """Test kill m1a sign."""
        # A grammar mutant admitting a sign must fail here.
        self.assertEqual(
            confidence.parse_verbalized_confidence("Paris\nConfidence: +85"),
            ("malformed_line", None, "Paris\nConfidence: +85"),
        )

    def test_kill_m1b_leading_zero(self):
        """Test kill m1b leading zero."""
        self.assertEqual(
            confidence.parse_verbalized_confidence("Paris\nConfidence: 085"),
            ("malformed_line", None, "Paris\nConfidence: 085"),
        )

    def test_kill_m1c_non_ascii_digits(self):
        """Test kill m1c non ascii digits."""
        self.assertEqual(
            confidence.parse_verbalized_confidence("Paris\nConfidence: ８５"),
            ("malformed_line", None, "Paris\nConfidence: ８５"),
        )

    def test_kill_m2_confidence_not_final(self):
        """Test kill m2 confidence not final."""
        self.assertEqual(
            confidence.parse_verbalized_confidence("Paris\nConfidence: 85\nSome prose")[0],
            "confidence_not_final",
        )

    def test_kill_m3_multiple_lines(self):
        """Test kill m3 multiple lines."""
        self.assertEqual(
            confidence.parse_verbalized_confidence("Paris\nConfidence: 85\nConfidence: 90")[0],
            "multiple_lines",
        )

    def test_kill_m4_line_separator(self):
        """Test kill m4 line separator."""
        self.assertEqual(
            confidence.parse_verbalized_confidence("Paris Confidence: 85")[0],
            "missing_line",
        )

    def test_kill_m5_underscore_int(self):
        """Test kill m5 underscore int."""
        self.assertEqual(
            confidence.parse_verbalized_confidence("Paris\nConfidence: 8_5")[0],
            "malformed_line",
        )

    def test_kill_m6_precedence(self):
        """Test kill m6 precedence."""
        # Final-line-first: a prefix-shaped final line is malformed
        # even when an earlier line matched the grammar.
        self.assertEqual(
            confidence.parse_verbalized_confidence("Paris\nConfidence: 85\nConfidence: ")[0],
            "malformed_line",
        )

    def test_parser_is_total_and_pure_str(self):
        """Test parser is total and pure str."""
        # A lone surrogate must not raise (the parser never encodes).
        code, value, span = confidence.parse_verbalized_confidence("x\ud800\nConfidence: 85")
        self.assertEqual((code, value), ("ok", 85))
        self.assertEqual(span, "x\ud800")

    def test_suffix_has_no_grammar_line(self):
        """Test suffix has no grammar line."""
        self.assertFalse(
            any(confidence.is_grammar_line(line) for line in confidence.SUFFIX.split("\n"))
        )


def _load_confidence_variant(source: str) -> dict:
    """Exec a (possibly mutated) confidence.py source under the
    package namespace so its relative imports resolve (the
    cycle-169 standalone-exec rule).
    """
    import types

    module = types.ModuleType("model_protocol_lab.confidence_variant")
    module.__package__ = "model_protocol_lab"
    # skipcq: PYL-W0122 - gate execs own module source mutated in memory only
    exec(compile(source, "confidence_variant", "exec"), module.__dict__)  # noqa: S102 -- the gate execs this repo's own module source, mutated in memory only
    return module.__dict__


class TestParserMutationGate(unittest.TestCase):
    """Each mutant is a plausible implementation slip; each must
    change the named killer case's classification. The control
    (unmutated source) must pass the whole battery.
    """

    @classmethod
    def setUpClass(cls):
        """Set up the test fixture."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import model_protocol_lab.confidence as _cf

        cls.source = Path(_cf.__file__).read_text(encoding="utf-8")

    def _mutant_parse(self, original: str, mutant: str):
        """Mutant parse."""
        self.assertEqual(
            self.source.count(original), 1, f"mutant anchor not unique: {original[:60]}"
        )
        namespace = _load_confidence_variant(self.source.replace(original, mutant))
        return namespace["parse_verbalized_confidence"]

    def test_control_unmutated_passes_battery(self):
        """Test control unmutated passes battery."""
        parse = _load_confidence_variant(self.source)["parse_verbalized_confidence"]
        for text, want in _BATTERY:
            self.assertEqual(parse(text), want)

    def test_m1a_sign_mutant_killed(self):
        """Test m1a sign mutant killed."""
        parse = self._mutant_parse("(0|[1-9][0-9]?|100)", r"(\+?[0-9]+)")
        self.assertEqual(parse("Paris\nConfidence: +85"), ("ok", 85, "Paris"))

    def test_m1b_leading_zero_mutant_killed(self):
        """Test m1b leading zero mutant killed."""
        parse = self._mutant_parse("(0|[1-9][0-9]?|100)", "([0-9]+)")
        self.assertEqual(parse("Paris\nConfidence: 085"), ("ok", 85, "Paris"))

    def test_m1c_unicode_digit_mutant_killed(self):
        """Test m1c unicode digit mutant killed."""
        parse = self._mutant_parse("(0|[1-9][0-9]?|100)", r"(\d+)")
        self.assertEqual(parse("Paris\nConfidence: ８５"), ("ok", 85, "Paris"))

    def test_m2_any_line_mutant_killed(self):
        """Test m2 any line mutant killed."""
        parse = self._mutant_parse(
            '    final = lines[-1] if lines else ""',
            "    final = next((line for line in lines if _GRAMMAR_RE.match(line)),"
            ' lines[-1] if lines else "")',
        )
        self.assertNotEqual(parse("Paris\nConfidence: 85\nSome prose")[0], "confidence_not_final")

    def test_m3_multiple_threshold_mutant_killed(self):
        """Test m3 multiple threshold mutant killed."""
        parse = self._mutant_parse(
            '        if grammar_lines >= 2:\n            return ("multiple_lines", None, answer)',
            '        if grammar_lines >= 3:\n            return ("multiple_lines", None, answer)',
        )
        self.assertEqual(
            parse("Paris\nConfidence: 85\nConfidence: 90"), ("ok", 90, "Paris\nConfidence: 85")
        )

    def test_m4_splitlines_mutant_killed(self):
        """Test m4 splitlines mutant killed."""
        parse = self._mutant_parse(
            '    lines = answer.split("\\n")\n'
            '    if answer.endswith("\\n"):\n'
            "        lines.pop()\n"
            '    final = lines[-1] if lines else ""',
            '    lines = answer.splitlines()\n    final = lines[-1] if lines else ""',
        )
        self.assertNotEqual(parse("Paris Confidence: 85")[0], "missing_line")

    def test_m5_loose_regex_mutant_killed(self):
        """Test m5 loose regex mutant killed."""
        parse = self._mutant_parse(
            'r"^Confidence: (0|[1-9][0-9]?|100)$"', 'r"^Confidence: (\\S+)$"'
        )
        self.assertEqual(parse("Paris\nConfidence: 8_5"), ("ok", 85, "Paris"))

    def test_m6_count_first_mutant_killed(self):
        """Test m6 count first mutant killed."""
        original = (
            '    if final.startswith("Confidence:") or final.lower().startswith("confidence:"):\n'
            '        return ("malformed_line", None, answer)\n'
            "    if grammar_lines >= 1:\n"
            '        return ("confidence_not_final", None, answer)'
        )
        mutant = (
            "    if grammar_lines >= 1:\n"
            '        return ("confidence_not_final", None, answer)\n'
            '    if final.startswith("Confidence:") or final.lower().startswith("confidence:"):\n'
            '        return ("malformed_line", None, answer)'
        )
        parse = self._mutant_parse(original, mutant)
        self.assertEqual(parse("Paris\nConfidence: 85\nConfidence: ")[0], "confidence_not_final")


class TestScreensAndSurgery(unittest.TestCase):
    """Test screens and surgery."""

    def test_screens(self):
        """Test screens."""
        self.assertEqual(confidence.screen_fixture(_fx("a", "  ", "g")), "expected_blank")
        self.assertEqual(
            confidence.screen_fixture(_fx("a", "Paris\nConfidence: 50", "g")),
            "expected_confidence_shaped",
        )
        self.assertEqual(
            confidence.screen_fixture(_fx("a", "Paris", "g", content="   ")),
            "vacuous_last_user",
        )
        assistant_only = Fixture(
            id="a",
            messages=(Message(role="assistant", content="hi"),),
            expected="Paris",
            group="g",
        )
        self.assertEqual(confidence.screen_fixture(assistant_only), "no_user_turn")
        self.assertEqual(confidence.screen_fixture(_fx("a", "Paris", "g")), "")

    def test_treated_fixture_only_last_user_changes(self):
        """Test treated fixture only last user changes."""
        case = Fixture(
            id="m",
            messages=(
                Message(role="user", content="first"),
                Message(role="assistant", content="reply"),
                Message(role="user", content="second"),
            ),
            expected="x",
            group="g",
        )
        treated = confidence.treated_fixture(case)
        self.assertEqual(treated.id, "m")
        self.assertEqual(treated.expected, "x")
        self.assertEqual(treated.group, "g")
        self.assertEqual(treated.messages[0].content, "first")
        self.assertEqual(
            treated.messages[2].content, "second" + confidence.JOINER + confidence.SUFFIX
        )

    def test_case_material_distinct_and_deterministic(self):
        """Test case material distinct and deterministic."""
        pairs = confidence.case_material(11)
        self.assertEqual(len(pairs), confidence.CASE_COUNT)
        self.assertEqual(len({k for k, _ in pairs}), confidence.CASE_COUNT)
        self.assertEqual(len({v for _, v in pairs}), confidence.CASE_COUNT)
        self.assertEqual(confidence.case_material(11), pairs)
        self.assertNotEqual(confidence.case_material(12), pairs)

    def test_case_protocol_matches_arms(self):
        """Test case protocol matches arms."""
        protocol = confidence.case_protocol(5)
        knowable = confidence.build_cases(5, control=False)
        control = confidence.build_cases(5, control=True)
        self.assertEqual(len(protocol), confidence.CASE_COUNT)
        for entry, kn, ct in zip(protocol, knowable, control, strict=True):
            self.assertEqual(entry["id"], kn.id)
            self.assertEqual(entry["expected_digest"], confidence.text_digest(kn.expected))
            self.assertEqual(
                entry["knowable_payload_digest"],
                confidence.text_digest(kn.messages[0].content),
            )
            self.assertEqual(
                entry["control_payload_digest"],
                confidence.text_digest(ct.messages[0].content),
            )
            # The control binds the donor key, never the asked key.
            key, _value = confidence.case_material(5)[int(entry["id"].rsplit("-", 1)[1])]
            donor_key, donor_value = confidence.case_material(5)[
                (int(entry["id"].rsplit("-", 1)[1]) + 1) % confidence.CASE_COUNT
            ]
            self.assertIn(f"the code for {donor_key} is {donor_value}", ct.messages[0].content)
            self.assertNotIn(f"the code for {key} is", ct.messages[0].content.split(".")[0])


class TestStage(unittest.TestCase):
    """Test stage."""

    def test_faithful_payload_validates(self):
        """Test faithful payload validates."""
        payload = _stage_payload()
        self.assertTrue(payload["available"])
        self.assertTrue(diffreports._valid_confidence(payload))
        alpha = payload["candidates"]["alpha"]
        self.assertEqual(alpha["operator"]["n_compliant"], 4)
        self.assertEqual(alpha["operator"]["gap_detail"], "no_incorrect_compliant")
        self.assertEqual(alpha["operator"]["flips"]["n_compared"], 4)
        knowability = alpha["generated"]["knowability"]
        self.assertEqual(knowability["n_paired_compliant"], 24)
        self.assertEqual(knowability["knowable_mean"], 95.0)
        self.assertEqual(knowability["control_mean"], 5.0)
        self.assertIs(knowability["falsified"], False)

    def test_theatre_backend_is_falsified(self):
        """Test theatre backend is falsified."""
        payload = _stage_payload(_ScriptedBackend(control_confidence=95))
        knowability = payload["candidates"]["alpha"]["generated"]["knowability"]
        self.assertIs(knowability["falsified"], True)
        self.assertTrue(diffreports._valid_confidence(payload))

    def test_screened_fixtures_carry_no_cells(self):
        """Test screened fixtures carry no cells."""
        cases = _cases() + (
            _fx("s4", "Confidence: 50\nParis", "geo"),
            _fx("s5", "   ", "geo"),
            _fx("s6", "Rome", "geo", content="  "),
        )
        payload = _stage_payload(cases=cases)
        self.assertTrue(payload["available"])
        self.assertTrue(diffreports._valid_confidence(payload))
        screens = {
            r["fixture"]: r["screen"] for r in payload["records"] if r["candidate"] == "alpha"
        }
        self.assertEqual(screens["s4"], "expected_confidence_shaped")
        self.assertEqual(screens["s5"], "expected_blank")
        self.assertEqual(screens["s6"], "vacuous_last_user")
        self.assertEqual(payload["candidates"]["alpha"]["operator"]["n_population"], 4)

    def test_candidate_degrades_alone(self):
        """Test candidate degrades alone."""

        class _BetaFails(_ScriptedBackend):
            """Beta fails."""

            def generate(self, input_ids, **kwargs):
                """Generate."""
                prompt = self.decode(input_ids)
                if prompt.startswith("U>"):
                    raise RuntimeError("boom")
                return super().generate(input_ids, **kwargs)

        cases = _cases()
        result = _native_result(_ScriptedBackend(), cases)
        payload = _stage_payload(_BetaFails(), cases, result=result)
        self.assertTrue(payload["available"])
        self.assertEqual(payload["unavailable_candidates"], ["beta"])
        self.assertTrue(diffreports._valid_confidence(payload))

    def test_budget_gate(self):
        """Test budget gate."""
        formats = tuple(
            Format(name=f"f{i}", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>")
            for i in range(25)
        )
        cases = tuple(_fx(f"c{i}", "x", "g") for i in range(100))
        payload = experiments.confidence_stage(
            _ScriptedBackend(), formats, cases, {"search": []}, boundary=(7,), seed=0
        )
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "confidence_cell_budget_exceeded")

    def test_cancelled(self):
        """Test cancelled."""
        cancel = Event()
        cancel.set()
        payload = experiments.confidence_stage(
            _ScriptedBackend(),
            _FORMATS,
            _cases(),
            {"search": []},
            boundary=(7,),
            seed=0,
            cancel=cancel,
        )
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "cancelled")

    def test_suffix_screen_fails_closed(self):
        """Test suffix screen fails closed."""
        formats = (
            Format(
                name="evil",
                bos="",
                user_prefix="q <number> ",
                assistant_prefix="a:",
                turn_end="</s>",
            ),
        )
        payload = experiments.confidence_stage(
            _ScriptedBackend(), formats, _cases(), {"search": []}, boundary=(7,), seed=0
        )
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "confidence_screen_failed")


class TestValidatorTamper(unittest.TestCase):
    """One break at a time over a real stage payload; the control
    payload must validate and every tampered one must not.
    """

    @classmethod
    def setUpClass(cls):
        """Set up the test fixture."""
        cls.payload = _stage_payload()

    def _tampered(self, mutate):
        """Tampered."""
        payload = copy.deepcopy(self.payload)
        mutate(payload)
        return payload

    def test_control_valid(self):
        """Test control valid."""
        self.assertTrue(diffreports._valid_confidence(self.payload))

    def test_scalar_edit_caught_by_carried_line(self):
        """Test scalar edit caught by carried line."""

        def mutate(p):
            """Mutate."""
            for record in p["records"]:
                cell = record["elicited"]
                if type(cell) is dict and cell["code"] == "ok":
                    cell["confidence"] = (
                        100 - cell["confidence"] if cell["confidence"] != 50 else 51
                    )
                    return

        self.assertFalse(diffreports._valid_confidence(self._tampered(mutate)))

    def test_breaks(self):
        """Test breaks."""
        breaks = {
            "verdict_flip": lambda p: p["records"][0]["elicited"].__setitem__(
                "verdict", not p["records"][0]["elicited"]["verdict"]
            ),
            "span_digest_swap": lambda p: p["records"][0]["elicited"].__setitem__(
                "span_digest", p["records"][0]["elicited"]["answer_digest"]
            ),
            "code_flip": lambda p: p["records"][0]["elicited"].__setitem__("code", "missing_line"),
            "carried_line_edit": lambda p: p["records"][0]["elicited"].__setitem__(
                "final_line", "Confidence: 1"
            ),
            "screen_mismatch": lambda p: p["records"][0].__setitem__("screen", "expected_blank"),
            "screened_with_cell": lambda p: p["protocol"]["screens"].__setitem__(
                p["records"][0]["fixture"], "expected_blank"
            ),
            "case_expected_edit": lambda p: p["control_records"][0].__setitem__(
                "expected_digest", "0" * 64
            ),
            "fingerprint_zero": lambda p: p["protocol"].__setitem__("fingerprint", "0" * 64),
            "case_set_edit": lambda p: p["protocol"]["control_cases"][0].__setitem__(
                "key_digest", "0" * 64
            ),
            "derived_mean_edit": lambda p: p["candidates"]["alpha"]["operator"].__setitem__(
                "mean_confidence", 1.0
            ),
            "falsified_flip": lambda p: p["candidates"]["alpha"]["generated"][
                "knowability"
            ].__setitem__("falsified", True),
            "unavailable_wrong": lambda p: p.__setitem__("unavailable_candidates", ["alpha"]),
            "bin_accuracy_edit": lambda p: p["candidates"]["alpha"]["operator"]["bins"][
                3
            ].__setitem__("accuracy", 0.0),
            "native_verdict_type": lambda p: p["records"][0].__setitem__("native_verdict", 1),
            "confidence_bool": lambda p: p["records"][0]["elicited"].__setitem__(
                "confidence", True
            ),
            "gap_detail_edit": lambda p: p["candidates"]["alpha"]["operator"].__setitem__(
                "gap_detail", ""
            ),
            "primary_winner_edit": lambda p: p.__setitem__("primary_winner", "beta"),
            "record_dropped": lambda p: p["records"].pop(),
            "control_record_dropped": lambda p: p["control_records"].pop(),
        }
        for name, mutate in breaks.items():
            with self.subTest(case=name):
                self.assertFalse(
                    diffreports._valid_confidence(self._tampered(mutate)),
                    f"tamper not caught: {name}",
                )


class TestDiffSection(unittest.TestCase):
    """Test diff section."""

    def test_self_diff_unchanged(self):
        """Test self diff unchanged."""
        payload = _stage_payload()
        diff = diffreports.diff_confidence(payload, payload)
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertTrue(diff["comparable"])

    def test_theatre_diff_changes_falsified(self):
        """Test theatre diff changes falsified."""
        faithful = _stage_payload()
        theatre = _stage_payload(_ScriptedBackend(control_confidence=95))
        diff = diffreports.diff_confidence(faithful, theatre)
        self.assertTrue(diff["changed"])
        metrics = {(d["candidate"], d["metric"]) for d in diff["candidate_drift"]}
        self.assertIn(("alpha", "knowability_falsified"), metrics)
        self.assertIn(("alpha", "control_mean_confidence"), metrics)

    def test_asymmetric_availability(self):
        """Test asymmetric availability."""
        payload = _stage_payload()
        diff = diffreports.diff_confidence(payload, {"available": False})
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])
        diff = diffreports.diff_confidence(None, None)
        self.assertEqual(diff["available"], "neither")

    def test_full_diff_reports_integration(self):
        """Test full diff reports integration."""
        payload = _stage_payload()
        old = {
            "decision": "compatible_on_fixture_set",
            "selected_candidate": "alpha",
            "confidence": payload,
        }
        new = {
            "decision": "compatible_on_fixture_set",
            "selected_candidate": "alpha",
            "confidence": payload,
        }
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(diff["confidence"]["available"], "both")
        self.assertFalse(diff["summary"]["confidence_changed"])
        self.assertFalse(diff["summary"]["confidence_asymmetric"])
        self.assertTrue(diff["summary"]["identical"])


class TestReportingSection(unittest.TestCase):
    """Test reporting section."""

    def test_section_renders(self):
        """Test section renders."""
        payload = _stage_payload()
        html = reporting._confidence_section({"confidence": payload})
        self.assertIn('id="confidence"', html)
        self.assertIn("alpha", html)
        self.assertIn("Knowability", html)
        self.assertIn("not a probability", html)
        self.assertIn("outside its", html)

    def test_section_empty_when_unavailable(self):
        """Test section empty when unavailable."""
        self.assertEqual(reporting._confidence_section({}), "")
        self.assertEqual(reporting._confidence_section({"confidence": {"available": False}}), "")

    def test_section_strips_hostile_candidate_name(self):
        """Test section strips hostile candidate name."""
        formats = (
            Format(
                name="a\x00b\ud800c",
                bos="",
                user_prefix="u:",
                assistant_prefix="a:",
                turn_end="</s>",
            ),
        )
        backend = _ScriptedBackend()
        cases = _cases()
        result = experiments.run_experiment(
            backend,
            formats,
            experiments.Split(search=cases, held_out=(_fx("h0", "held", "heldout"),)),
            boundary=(7,),
            max_new_tokens=64,
            seed=0,
        )
        payload = experiments.confidence_stage(
            backend, formats, cases, result, boundary=(7,), max_new_tokens=64, seed=3
        )
        html = reporting._confidence_section({"confidence": payload})
        self.assertNotIn("\x00", html)
        self.assertNotIn("\ud800", html)


class TestCliWrapper(unittest.TestCase):
    """Test cli wrapper."""

    @staticmethod
    def _result(decision="compatible_on_fixture_set"):
        """Result."""
        return {
            "decision": decision,
            "selected_candidate": "alpha",
            "search": [{"format": {"name": "alpha"}}, {"format": {"name": "beta"}}],
        }

    def test_gate_on_failed_native(self):
        """Test gate on failed native."""
        result = self._result(decision="failed")
        cli._run_confidence_probe(
            _ScriptedBackend(),
            _FORMATS,
            result,
            _cases(),
            boundary=(7,),
            boundary_map=None,
            max_new_tokens=64,
            seed=0,
            cancel=None,
        )
        self.assertEqual(
            result["confidence"],
            {"available": False, "detail": "native_search_incomplete"},
        )

    def test_stage_exception_degrades(self):
        """Test stage exception degrades."""
        result = self._result()
        original = experiments.confidence_stage

        def _boom(*args, **kwargs):
            """Boom."""
            raise RuntimeError("boom")

        experiments.confidence_stage = _boom
        try:
            cli._run_confidence_probe(
                _ScriptedBackend(),
                _FORMATS,
                result,
                _cases(),
                boundary=(7,),
                boundary_map=None,
                max_new_tokens=64,
                seed=0,
                cancel=None,
            )
        finally:
            experiments.confidence_stage = original
        self.assertEqual(
            result["confidence"],
            {"available": False, "detail": "confidence_unavailable"},
        )

    def test_wrapper_installs_payload(self):
        """Test wrapper installs payload."""
        backend = _ScriptedBackend()
        cases = _cases()
        result = _native_result(backend, cases)
        cli._run_confidence_probe(
            backend,
            _FORMATS,
            result,
            cases,
            boundary=(7,),
            boundary_map=None,
            max_new_tokens=64,
            seed=3,
            cancel=None,
        )
        self.assertTrue(result["confidence"]["available"])
        self.assertTrue(diffreports._valid_confidence(result["confidence"]))


class TestTinyCheckpointEndToEnd(unittest.TestCase):
    """One real tiny-checkpoint run through the CLI probe path."""

    @classmethod
    def setUpClass(cls):
        """Set up the test fixture."""
        import tempfile

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from pathlib import Path

        from tests.test_adapter_integration import build_tiny_gpt2

        cls._tmp = tempfile.TemporaryDirectory()
        cls.model_dir = Path(cls._tmp.name) / "tiny"
        build_tiny_gpt2(cls.model_dir)

    @classmethod
    def tearDownClass(cls):
        """Tear down the test fixture."""
        cls._tmp.cleanup()

    def test_tiny_checkpoint_confidence_run(self):
        """Test tiny checkpoint confidence run."""
        from model_protocol_lab.backends import LocalHFBackend

        # The tiny checkpoint's declared window is 64 tokens, so
        # the generation budget is small; the assertions are the
        # probe-path ones (payload produced, validator-clean,
        # record grids complete) -- what the random-weight model
        # happens to say is not asserted.
        backend = LocalHFBackend(self.model_dir, max_context=64)
        cases = _cases()
        result = _native_result(backend, cases, max_new_tokens=8)
        cli._run_confidence_probe(
            backend,
            _FORMATS,
            result,
            cases,
            boundary=(7,),
            boundary_map=None,
            max_new_tokens=8,
            seed=3,
            cancel=None,
        )
        payload = result["confidence"]
        self.assertTrue(payload["available"])
        self.assertTrue(diffreports._valid_confidence(payload))
        self.assertEqual(payload["totals"]["n_records"], 8)
        self.assertEqual(payload["totals"]["n_control_records"], 48)


if __name__ == "__main__":
    unittest.main()
