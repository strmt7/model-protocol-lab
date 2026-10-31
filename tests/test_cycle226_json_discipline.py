"""Cycle 226: the JSON-discipline class (B-05, B-07, B-14, B-16).

The readiness audit found four parsers lagging the tree's shared JSON
discipline — the strict form the hardened siblings (datasets,
diffreports' results loader, toolcall's answer parser, the MCP frame
parser) already enforce: duplicate keys refused, NaN/Infinity
literals and overflowing floats refused, lone surrogates refused,
and pathological nesting converted into the module's own error
vocabulary instead of escaping as a raw RecursionError.

- ``surfacenoise._load_probe_json`` accepted every one of those
  classes with a plain ``json.loads`` (B-05), and because it never
  checked that the payload was a probe report at all, the
  ``surface-noise diff`` CLI certified two garbage files as
  ``identical: true`` (B-14). The loader now enforces the discipline
  and the probe-report skeleton every real report — ok, cancelled,
  or unavailable — carries.
- ``recordnotation._parsed_hit`` and its diffreports mirror let a
  deeply nested model answer kill the candidate aggregation with a
  RecursionError (B-07); both now parse with the toolcall fail-closed
  hooks and treat the class as a miss.
- ``tools/agent_context.load_state`` used a plain ``json.loads``
  while the same file's ``json_report`` enforced the discipline;
  the state loader now routes through the file's own hooks (B-16).

Each test below is red on the pre-fix tree for the class it names:
reverting any one fix (the hooks, the RecursionError conversion,
the skeleton check, the load_state routing) re-breaks exactly the
tests for that class.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from model_protocol_lab import recordnotation as rn
from model_protocol_lab import surfacenoise as sn
from model_protocol_lab.cli import main as cli_main
from model_protocol_lab.core import Fixture, Message
from model_protocol_lab.diffreports import _recordnotation_parsed_hit
from tools import agent_context as context

DEEP = "[" * 10000 + "]" * 10000


def _write(root: Path, name: str, text: str) -> Path:
    """Write text."""
    target = root / name
    target.write_text(text, encoding="utf-8")
    return target


def _cancelled_report() -> dict:
    """A real probe report skeleton (the cancelled shape)."""
    return {
        "instrument": sn.PROBE_NAME,
        "protocol_version": sn.PROBE_VERSION,
        "status": "cancelled",
        "reason": "cancelled_before_baseline",
        "completed_variants": 0,
        "total_variants": 0,
        "prompt_sha256": "0" * 64,
        "protocol": {"instrument": sn.PROBE_NAME},
    }


class SurfaceNoiseProbeLoaderTests(unittest.TestCase):
    """B-05/B-14: the probe.json loader enforces the shared discipline."""

    def test_deep_nesting_is_a_loader_refusal_not_a_crash(self):
        """Test deep nesting is a loader refusal not a crash."""
        with tempfile.TemporaryDirectory() as td:
            path = _write(Path(td), "deep.json", DEEP)
            with self.assertRaises(ValueError):
                sn._load_probe_json(path)

    def test_nonfinite_literals_and_overflow_are_refused(self):
        """Test nonfinite literals and overflow are refused."""
        with tempfile.TemporaryDirectory() as td:
            for name, text in (
                ("nan.json", '{"x": NaN}'),
                ("inf.json", '{"x": Infinity}'),
                ("overflow.json", '{"x": 1e999}'),
            ):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    sn._load_probe_json(_write(Path(td), name, text))

    def test_duplicate_keys_are_refused(self):
        """Test duplicate keys are refused."""
        with tempfile.TemporaryDirectory() as td:
            path = _write(Path(td), "dup.json", '{"x": 1, "x": 2}')
            with self.assertRaises(ValueError):
                sn._load_probe_json(path)

    def test_lone_surrogate_is_refused(self):
        """Test lone surrogate is refused."""
        with tempfile.TemporaryDirectory() as td:
            path = _write(Path(td), "surr.json", '{"x": "\\ud800"}')
            with self.assertRaises(ValueError):
                sn._load_probe_json(path)

    def test_non_report_json_is_refused(self):
        """Test non report json is refused."""
        with tempfile.TemporaryDirectory() as td:
            for name, text in (("list.json", "[]"), ("foreign.json", '{"a": 1}')):
                with self.subTest(name=name):
                    with self.assertRaises(ValueError) as caught:
                        sn._load_probe_json(_write(Path(td), name, text))
                    self.assertIn("not a surface-noise probe report", str(caught.exception))

    def test_real_report_skeleton_still_loads(self):
        """Test real report skeleton still loads."""
        with tempfile.TemporaryDirectory() as td:
            path = _write(Path(td), "probe.json", json.dumps(_cancelled_report()))
            loaded = sn._load_probe_json(path)
            self.assertEqual(loaded["status"], "cancelled")
            self.assertEqual(loaded["instrument"], sn.PROBE_NAME)

    def test_invalid_utf8_is_refused(self):
        """Test invalid utf8 is refused."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bytes.json"
            path.write_bytes(b'{"x": \xff}')
            with self.assertRaises(ValueError):
                sn._load_probe_json(path)

    def test_finite_floats_still_load(self):
        """Test finite floats still load."""
        with tempfile.TemporaryDirectory() as td:
            payload = _cancelled_report()
            payload["score"] = 0.25
            path = _write(Path(td), "probe.json", json.dumps(payload))
            self.assertEqual(sn._load_probe_json(path)["score"], 0.25)

    def test_cli_diff_of_garbage_exits_1(self):
        """Test cli diff of garbage exits 1."""
        # B-14's headline: pre-fix this printed
        # {"available": "neither", "changed": false, "identical": true}
        # and exited 0 for two files containing just [].
        with tempfile.TemporaryDirectory() as td:
            old = _write(Path(td), "old.json", "[]")
            new = _write(Path(td), "new.json", "[]")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = cli_main(["surface-noise", "diff", str(old), str(new)])
            self.assertEqual(code, 1)

    def test_cli_diff_of_real_reports_still_works(self):
        """Test cli diff of real reports still works."""
        with tempfile.TemporaryDirectory() as td:
            backend = sn.SelfTestBackend()
            fixtures = (
                Fixture(
                    id="fx-1",
                    messages=(Message(role="user", content="Say the code."),),
                    expected="oak-1234",
                    group="probe",
                ),
            )
            protocol = sn.freeze_surface_noise_protocol(
                backend_fingerprint=backend.fingerprint, fixtures=fixtures
            )
            report = sn.probe_surface_noise("MAGICWORD", fixtures, backend, protocol, boundary=(7,))
            old = _write(Path(td), "old.json", json.dumps(report))
            new = _write(Path(td), "new.json", json.dumps(report))
            buf = io.StringIO()
            with redirect_stdout(buf), redirect_stderr(io.StringIO()):
                code = cli_main(["surface-noise", "diff", str(old), str(new)])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(buf.getvalue())["identical"])


class RecordNotationParsedHitTests(unittest.TestCase):
    """B-07: the parsed-hit diagnostic fails closed at both mirrors."""

    def test_deep_answer_is_a_miss_not_a_crash(self):
        """Test deep answer is a miss not a crash."""
        self.assertFalse(rn._parsed_hit(DEEP, "x"))
        self.assertFalse(_recordnotation_parsed_hit(DEEP, "x"))

    def test_nonfinite_answer_is_a_miss(self):
        """Test nonfinite answer is a miss."""
        self.assertFalse(rn._parsed_hit('{"code": NaN}', "x"))
        self.assertFalse(rn._parsed_hit('{"code": 1e999}', "x"))
        self.assertFalse(_recordnotation_parsed_hit('{"code": 1e999}', "x"))
        # A finite float parses under the discipline and is a miss
        # on value, exercising the float hook's accept path.
        self.assertFalse(rn._parsed_hit('{"code": 1.5}', "x"))

    def test_genuine_hit_is_unchanged(self):
        """Test genuine hit is unchanged."""
        self.assertTrue(rn._parsed_hit('{"code": "oak-1234"}', "oak-1234"))
        self.assertTrue(_recordnotation_parsed_hit('{"code": "oak-1234"}', "oak-1234"))

    def test_aggregate_survives_a_deep_answer(self):
        """Test aggregate survives a deep answer."""
        key = rn.HEADLINE_ARMS[0]
        record = {
            "fixture": "f1",
            "cells": {
                key: {
                    "verdict": False,
                    "answer_text": DEEP,
                    "boundary_found": False,
                    "answer_digest": "x",
                }
            },
            "cell_details": {},
        }
        cases = [{"fixture_id": "f1", "expected": "e"}]
        report = rn.aggregate_candidate([record], cases)
        self.assertIsInstance(report, dict)


class AgentContextStateTests(unittest.TestCase):
    """B-16: load_state enforces the discipline json_report already had."""

    def _state(self, root: Path, text: str) -> str:
        """Write a state file and return its state name."""
        name = context.state_name("notes", "discipline")
        target = context.safe_path(root, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return name

    def test_deep_state_is_a_value_error_not_a_crash(self):
        """Test deep state is a value error not a crash."""
        with tempfile.TemporaryDirectory() as td:
            name = self._state(Path(td), DEEP)
            with self.assertRaises(ValueError) as caught:
                context.load_state(Path(td), name)
            self.assertIn("decodable nesting depth", str(caught.exception))

    def test_duplicate_and_nonfinite_state_members_are_refused(self):
        """Test duplicate and nonfinite state members are refused."""
        with tempfile.TemporaryDirectory() as td:
            for text in ('{"schema": 1, "schema": 1}', '{"schema": NaN}', '{"schema": 1e999}'):
                with self.subTest(text=text):
                    name = self._state(Path(td), text)
                    with self.assertRaises(ValueError):
                        context.load_state(Path(td), name)

    def test_valid_state_round_trips(self):
        """Test valid state round trips."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            name = context.state_name("notes", "roundtrip")
            context.write_state(root, name, {"schema": 1, "note": "kept"})
            self.assertEqual(context.load_state(root, name)["note"], "kept")


if __name__ == "__main__":
    unittest.main()
