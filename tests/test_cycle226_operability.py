"""Cycle 226: the operability batch (B-10..B-13, B-15, B-17..B-20).

The readiness audit's lane-4 findings, each pinned at the surface it
was observed on:

- B-10: piping a command's stdout into a closing consumer killed the
  process with exit 120 and an "Exception ignored ... BrokenPipeError"
  dump; the CLI now dies quietly on SIGPIPE like a standard Unix
  filter (exit 141, nothing on stderr).
- B-11: two identical surface-noise probe runs differed in exactly
  one field — the protocol's wall-clock ``frozen_at``. The field is
  removed from both freeze functions; probe output is now a pure
  function of its inputs.
- B-12: verdict-bearing commands disagreed on the failed-verdict
  code (doctor 1, verify 2, run "failed" 1). One table now governs:
  0 success, 1 operational failure (no verdict produced),
  2 negative verdict (and argparse usage errors), 3 probe baseline
  unavailable, 130 cancelled/interrupted.
- B-13: the MCP probe tool mapped client-caused fixture defects to
  -32603 (internal error) or crashed on them; its parse layer now
  mirrors the fixture grammar (nonblank id, per-fixture message cap,
  per-message length cap, no lone surrogates) with -32602.
- B-15: recordnotation's generators accepted any seed type
  (``random.Random`` semantics leaked through); they now validate
  exactly like canonicalform's generator.
- B-17: ``debug --script`` hid command failures (errors on stdout,
  exit 0); script-mode errors go to stderr and the exit code is 1
  when any command failed. Interactive behaviour is unchanged.
- B-18/B-19: the MCP entry point ignored argv entirely and the CLI
  had no ``--version``; both entry points parse arguments now.
- B-20: every subcommand's ``--help`` states its exit codes and
  output contract in an epilog.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from model_protocol_lab import __version__
from model_protocol_lab import recordnotation as rn
from model_protocol_lab import surfacenoise as sn
from model_protocol_lab.cli import main as cli_main
from model_protocol_lab.core import Fixture, Message
from model_protocol_lab.diffreports import _recordnotation_generate
from model_protocol_lab.mcp_server import handle_request, serve

REPO_ROOT = Path(__file__).resolve().parents[1]
DEEP = "[" * 10000 + "]" * 10000


def _fixtures_payload(n_messages: int = 1, content: str = "Capital of France?") -> list:
    """One MCP fixture payload."""
    return [
        {
            "id": "fx-1",
            "messages": [{"role": "user", "content": content} for _ in range(n_messages)],
            "expected": "Paris",
            "group": "probe",
        }
    ]


def _probe_call(arguments: dict) -> dict:
    """Dispatch one MCP probe tools/call and return the response."""
    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "probe_surface_noise_stability", "arguments": arguments},
        }
    )
    assert response is not None
    return response


class SigpipeTests(unittest.TestCase):
    """B-10: a closing downstream consumer ends the CLI quietly."""

    def test_piped_stdout_closes_without_broken_pipe_dump(self):
        """Test piped stdout closes without broken pipe dump."""
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "model_protocol_lab",
                "fixtures",
                "fuzz",
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdout is not None
        proc.stdout.read(64)
        proc.stdout.close()
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        proc.wait(timeout=60)
        self.assertNotEqual(proc.returncode, 120)
        self.assertNotIn("Exception ignored", stderr)
        self.assertNotIn("BrokenPipeError", stderr)


class SurfaceNoiseDeterminismTests(unittest.TestCase):
    """B-11: probe output is a pure function of its inputs."""

    def _report(self) -> dict:
        """One selftest probe report."""
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
        return sn.probe_surface_noise("MAGICWORD", fixtures, backend, protocol, boundary=(7,))

    def test_two_identical_runs_are_byte_identical(self):
        """Test two identical runs are byte identical."""
        first = json.dumps(self._report(), indent=2, sort_keys=True)
        second = json.dumps(self._report(), indent=2, sort_keys=True)
        self.assertEqual(first, second)

    def test_protocols_carry_no_wall_clock_field(self):
        """Test protocols carry no wall clock field."""
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
        self.assertNotIn("frozen_at", protocol)
        template_protocol = sn.freeze_template_robustness_protocol(
            backend_fingerprint=backend.fingerprint, fixtures=fixtures
        )
        self.assertNotIn("frozen_at", template_protocol)


class ExitCodeTableTests(unittest.TestCase):
    """B-12: one documented exit-code table, reconciled deliberately."""

    def test_doctor_failed_verdict_is_2_like_verify(self):
        """Test doctor failed verdict is 2 like verify."""
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = cli_main(["doctor", "--model", "/nonexistent-ckpt"])
        self.assertEqual(code, 2)

    def test_verify_failed_verdict_stays_2(self):
        """Test verify failed verdict stays 2."""
        import shutil

        source = REPO_ROOT / "examples" / "sample-report"
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "report"
            shutil.copytree(source, target)
            results = json.loads((target / "results.json").read_text(encoding="utf-8"))
            results["selected_candidate"] = "nosuchcandidate"
            (target / "results.json").write_text(json.dumps(results), encoding="utf-8")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = cli_main(["verify", str(target)])
            self.assertEqual(code, 2)

    def test_usage_error_is_argparse_2(self):
        """Test usage error is argparse 2."""
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            cli_main(["doctor"])
        self.assertEqual(caught.exception.code, 2)


class McpFixtureGrammarTests(unittest.TestCase):
    """B-13: client-caused fixture defects are -32602, never -32603."""

    def _call(self, fixtures) -> dict:
        """Call the probe tool with the given fixtures payload."""
        return _probe_call(
            {
                "prompt": "MAGICWORD please",
                "fixtures": fixtures,
                "boundary": [7],
                "backend": {"kind": "selftest"},
            }
        )

    def test_whitespace_only_fixture_id_is_invalid_params(self):
        """Test whitespace only fixture id is invalid params."""
        payload = _fixtures_payload()
        payload[0]["id"] = "   "
        self.assertEqual(self._call(payload)["error"]["code"], -32602)

    def test_oversize_message_is_invalid_params(self):
        """Test oversize message is invalid params."""
        payload = _fixtures_payload(content="x" * (256 * 1024 + 1))
        self.assertEqual(self._call(payload)["error"]["code"], -32602)

    def test_message_count_cap_is_invalid_params(self):
        """Test message count cap is invalid params."""
        payload = _fixtures_payload(n_messages=22)
        self.assertEqual(self._call(payload)["error"]["code"], -32602)

    def test_surrogates_are_invalid_params(self):
        """Test surrogates are invalid params."""
        payload = _fixtures_payload()
        payload[0]["expected"] = "\ud800"
        self.assertEqual(self._call(payload)["error"]["code"], -32602)
        response = _probe_call(
            {
                "prompt": "MAGICWORD \ud800",
                "fixtures": _fixtures_payload(),
                "boundary": [7],
                "backend": {"kind": "selftest"},
            }
        )
        self.assertEqual(response["error"]["code"], -32602)

    def test_valid_fixture_still_probes(self):
        """Test valid fixture still probes."""
        response = self._call(_fixtures_payload())
        self.assertIn("result", response)


class RecordNotationSeedTests(unittest.TestCase):
    """B-15: generator seeds validate exactly like canonicalform's."""

    def test_non_int_seeds_are_value_errors(self):
        """Test non int seeds are value errors."""
        for bad in ("7", None, [7], {"s": 7}, 1.5, True):
            with self.subTest(seed=bad):
                with self.assertRaises(ValueError):
                    rn.generate_cases(bad)  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    rn.generate_case(bad, 0)  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    _recordnotation_generate(bad)  # type: ignore[arg-type]

    def test_negative_seed_is_accepted_like_canonicalform(self):
        """Test negative seed is accepted like canonicalform."""
        cases = rn.generate_cases(-1)
        self.assertEqual(len(cases), 24)
        self.assertEqual(rn.generate_cases(7), rn.generate_cases(7))


class EntryPointArgumentTests(unittest.TestCase):
    """B-18/B-19: both entry points parse their arguments."""

    def test_cli_version_flag(self):
        """Test cli version flag."""
        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as caught:
            cli_main(["--version"])
        self.assertEqual(caught.exception.code, 0)
        self.assertEqual(buf.getvalue().strip(), f"model-protocol-lab {__version__}")

    def test_mcp_help_exits_0_with_usage(self):
        """Test mcp help exits 0 with usage."""
        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as caught:
            serve(["--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("usage", buf.getvalue().lower())

    def test_mcp_unknown_argument_exits_2(self):
        """Test mcp unknown argument exits 2."""
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            serve(["--nonsense-arg"])
        self.assertEqual(caught.exception.code, 2)

    def test_mcp_version_flag(self):
        """Test mcp version flag."""
        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as caught:
            serve(["--version"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn(__version__, buf.getvalue())


class DebugScriptExitTests(unittest.TestCase):
    """B-17: a scripted debug run reports command failures in its exit."""

    def _shell(self):
        """A debug shell over the example fixtures."""
        from model_protocol_lab.datasets import load_dataset
        from model_protocol_lab.debug import DebugShell

        split = load_dataset(REPO_ROOT / "examples" / "fixtures.json")
        return DebugShell(split)

    def _run_script(self, body: str) -> tuple[int, str]:
        """Run one script body; return (exit code, captured stderr)."""
        from model_protocol_lab.debug import run_script

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "script.read"
            path.write_text(body, encoding="utf-8")
            err = io.StringIO()
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                code = run_script(self._shell(), path)
        return code, err.getvalue()

    def test_failing_command_exits_1_and_reports_on_stderr(self):
        """Test failing command exits 1 and reports on stderr."""
        code, err = self._run_script(".nonexistent\n")
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_clean_script_exits_0(self):
        """Test clean script exits 0."""
        code, err = self._run_script(".help\n")
        self.assertEqual(code, 0)
        self.assertEqual(err, "")

    def test_unreadable_script_exits_1(self):
        """Test unreadable script exits 1."""
        from model_protocol_lab.debug import run_script

        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            code = run_script(self._shell(), "/nonexistent-dir-xyz/script.read")
        self.assertEqual(code, 1)
        self.assertIn("debug:", err.getvalue())


class HelpContractTests(unittest.TestCase):
    """B-20: every parser's help states exit codes and output."""

    def test_every_parser_carries_an_exit_code_epilog(self):
        """Test every parser carries an exit code epilog."""
        import argparse

        from model_protocol_lab.cli import build_parser

        seen: list[argparse.ArgumentParser] = []
        stack = [build_parser()]
        while stack:
            parser = stack.pop()
            seen.append(parser)
            for action in parser._actions:
                if isinstance(action, argparse._SubParsersAction):
                    stack.extend(action.choices.values())
        self.assertGreater(len(seen), 10)
        for parser in seen:
            with self.subTest(prog=parser.prog):
                self.assertIsNotNone(parser.epilog)
                self.assertIn("Exit codes", parser.epilog or "")

    def test_usage_documents_the_table(self):
        """Test usage documents the table."""
        usage = (REPO_ROOT / "docs" / "USAGE.md").read_text(encoding="utf-8")
        self.assertIn("## 15. Exit codes and output formats", usage)
        self.assertIn("130", usage)


class PylockSelfEntryTests(unittest.TestCase):
    """B-26: the PEP 751 lock must not contain the project itself."""

    def test_lock_has_no_self_entry(self):
        """Test lock has no self entry."""
        text = (REPO_ROOT / "pylock.toml").read_text(encoding="utf-8")
        self.assertNotIn('name = "model-protocol-lab"', text)
        self.assertNotIn("[packages.directory]", text)
        self.assertIn("two-step install", text)


if __name__ == "__main__":
    unittest.main()
