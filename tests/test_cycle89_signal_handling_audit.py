"""Cycle 89: adversarial signal/interruption-handling audit.

METHOD
------
Fills the exact gap cycles 64/83 deliberately excluded ("exit-130 SIGINT
path not tested: needs live signals"). Two layers:

1. Real OS signals against real subprocesses (``sys.executable -m
   model_protocol_lab ...``) in PID-scoped temp dirs:
   - ``debug --script`` with a ~9000-command script gives a reliably
     interruptible multi-second window; readiness is polled (first stdout
     byte), never slept into. Pins: SIGINT -> exit 130, stderr's last
     line is the clean "Interrupted before a complete report was saved."
     (no traceback); double SIGINT -> still terminates promptly, no hang,
     no traceback; SIGTERM -> default disposition (killed, -SIGTERM).
   - ``run`` against a tiny REAL GPT-2 checkpoint built locally with
     torch/transformers (no network, ``local_files_only``) with a slow
     generate (``--max-new-tokens 1000``, ``use_cache=False`` recomputes
     the prefix per token, ~50s per generate) gives a long generation
     window. Readiness: the claimed output dir appearing, then the
     weights-loaded marker on stderr. Pins: SIGINT ->
     exit 130, no traceback, and the output directory is either removed
     (interrupted before any report) or holds a complete valid
     "cancelled" report -- never a half-written results.json/report.html.
     SIGTERM during the experiment now drives the same cooperative
     cancel as SIGINT (cycle-89 fix): exit 130 with a valid "cancelled"
     report instead of dying mid-run and leaking the claimed directory.
     NOTE (cycle 92): these two real-model tests are slow (~3min each:
     torch model build + 1000-token generation) and therefore gated behind
     ``MPL_RUN_SLOW_SIGNAL_TESTS=1``, skipped in the default CI gate; the
     deterministic in-process signal tests below stay in the gate.
     Readiness is deterministic, not timing-based: the child writes a
     marker file (``MODEL_PROTOCOL_LAB_TEST_SIGNAL_READY_FILE`` hook in cli.py) immediately
     after installing its signal handlers, so the test signals exactly
     inside the experiment window. (An earlier revision polled stderr for a
     "100%" weights-loaded marker that transformers never prints for local
     loads, then guessed with a fixed sleep -- the signal could land before
     the handlers were installed.)
2. In-process fault injection through the REAL ``cli.main`` signal
   machinery (fake backends that raise a real SIGINT/SIGTERM at their own
   process from inside ``generate()``, or raise KeyboardInterrupt):
   - A real SIGINT raised from inside ``generate()`` -> the production
     handler sets the cancel event -> decision "cancelled" -> valid
     report saved -> exit 130. Double SIGINT -> same, no hang.
   - A real SIGTERM raised from inside ``generate()`` -> same graceful
     path (cycle-89 fix), exit 130.
   - KeyboardInterrupt raised from ``generate()`` propagates out of
     ``run_experiment`` unchanged (``evaluate`` only catches
     ``Exception``; KeyboardInterrupt is a BaseException) -- never
     swallowed into a verdict -- and ``cli.main`` turns it into exit 130
     with the claimed output directory cleaned up.
   - ``save_report`` with a fault-injecting writer raising
     KeyboardInterrupt mid-write leaves no partial results.json /
     report.html (cycle-89 fix; ``save_report`` previously left a
     half-written results.json, unlike ``write_fixture_file`` which
     already unlinked on failure).

FINDINGS
--------
- REAL BUG (fixed in ``reporting.save_report``): a KeyboardInterrupt
  landing mid-write left a half-written ``results.json`` on disk.
  Fixed with the same unlink-on-``BaseException`` discipline
  ``fixtures.write_fixture_file`` already used, extended to also remove
  the first file when the second write fails (no half-built pair).
- REAL BUG (fixed in ``cli.main``): SIGTERM during ``run`` used the
  default disposition -- the process died immediately (exit -15),
  leaving the claimed output directory behind so a retry tripped
  "Output path already exists". Fixed by wiring SIGTERM to the same
  cooperative cancel event as SIGINT around ``run_experiment``
  (restored afterwards).
- CLEAN: SIGINT / double-SIGINT on ``debug --script`` -> exit 130,
  clean stderr, no hang, no traceback. SIGTERM on ``debug --script`` ->
  default disposition (killed); pinned as-is, since a scripted debugger
  session holds no output state worth draining.
- CLEAN: KeyboardInterrupt from ``generate()`` propagates unchanged
  through ``evaluate``/``run_experiment``.

DELIBERATELY NOT VERIFIED
-------------------------
- SIGTERM landing in the short diagnostics/save_report window after
  ``run_experiment`` returns (handlers already restored): default
  disposition still applies there, same as SIGINT's pre-existing
  exposure outside the experiment window. The long phase -- the
  experiment itself -- is covered.
- ``debug`` interactive REPL stdin behavior (out of scope; cycle-64
  read-only pins stand).

Verified by RUNNING this file: ``pytest -p no:cacheprovider
tests/test_cycle89_signal_handling_audit.py`` -- 2026-10-16.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from threading import Event
from unittest import mock

from model_protocol_lab import backends, cli
from model_protocol_lab import reporting as reporting_mod
from model_protocol_lab.catalog import candidates
from model_protocol_lab.core import Generation
from model_protocol_lab.datasets import load_dataset
from model_protocol_lab.experiments import run_experiment

_PID = os.getpid()
_INTERRUPTED_LINE = "Interrupted before a complete report was saved."
_POSIX = os.name == "posix"


def _scratch(prefix: str) -> Path:
    """PID-scoped temp dir (tempfile location, never hard-coded /tmp)."""
    return Path(tempfile.mkdtemp(prefix=f"mpl_c89_{_PID}_{prefix}_"))


def _write_fixture(directory: Path, *, search_cases: int = 2) -> Path:
    """Minimal valid fixture JSON, mirroring test_cycle31's helper."""
    path = directory / "fixture.json"
    path.write_text(
        json.dumps(
            {
                "search": [
                    {
                        "id": f"s{i}",
                        "group": "g1",
                        "messages": [{"role": "user", "content": "hi"}],
                        "expected": "yo",
                    }
                    for i in range(search_cases)
                ],
                "held_out": [
                    {
                        "id": "h1",
                        "group": "g2",
                        "messages": [{"role": "user", "content": "hey"}],
                        "expected": "yo",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _wait_until(predicate, deadline: float, poll: float = 0.1) -> bool:
    """Readiness polling: bounded wait, no blind sleeps."""
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(poll)
    return predicate()


class _Child:
    """Subprocess with background-drained pipes for incremental polling."""

    def __init__(self, argv: list[str], cwd: str, env: dict[str, str] | None = None) -> None:
        self._proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env
        )
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._lock = threading.Lock()
        self._threads = [
            threading.Thread(
                target=self._pump, args=(self._proc.stdout, self._stdout), daemon=True
            ),
            threading.Thread(
                target=self._pump, args=(self._proc.stderr, self._stderr), daemon=True
            ),
        ]
        for thread in self._threads:
            thread.start()
        self._t0 = time.monotonic()

    @staticmethod
    def _pump(pipe, buf: bytearray) -> None:
        try:
            while True:
                chunk = pipe.read(65536)
                if not chunk:
                    break
                buf.extend(chunk)
        finally:
            pipe.close()

    @property
    def pid(self) -> int:
        return self._proc.pid

    @property
    def age(self) -> float:
        return time.monotonic() - self._t0

    def stdout_bytes(self) -> bytes:
        with self._lock:
            return bytes(self._stdout)

    def stderr_bytes(self) -> bytes:
        with self._lock:
            return bytes(self._stderr)

    def wait(self, timeout: float) -> int:
        try:
            return self._proc.wait(timeout=timeout)
        finally:
            for thread in self._threads:
                thread.join(timeout=5)

    def kill(self) -> None:
        # Safe after wait(): Popen.kill skips signalling a reaped process,
        # and send_signal swallows ProcessLookupError for the narrow race.
        self._proc.kill()
        self.wait(10)

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode


def _send(pid: int, sig: int) -> bool:
    """Deliver a signal; False when the child already exited (prompt death)."""
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return False
    return True


# ---------------------------------------------------------------------------
# Layer 1a: real subprocess, debug --script (deterministic window via
# first-output-byte readiness polling).
# ---------------------------------------------------------------------------


@unittest.skipUnless(_POSIX, "sends POSIX signals")
class DebugScriptSignalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _scratch("dbg")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        _write_fixture(self.tmp)
        lines = [".render" if i % 2 == 0 else ".audit" for i in range(9000)]
        (self.tmp / "long.script").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _spawn(self) -> _Child:
        return _Child(
            [
                sys.executable,
                "-m",
                "model_protocol_lab",
                "debug",
                "--dataset",
                str(self.tmp / "fixture.json"),
                "--script",
                str(self.tmp / "long.script"),
            ],
            cwd=str(self.tmp),
        )

    def _in_window(self, child: _Child) -> _Child:
        ready = _wait_until(lambda: len(child.stdout_bytes()) > 0, 60)
        self.assertTrue(ready, "child printed nothing; no interruptible window observed")
        return child

    def _finish(self, child: _Child, timeout: float = 60) -> tuple[int, str]:
        try:
            rc = child.wait(timeout)
        except subprocess.TimeoutExpired:
            self.fail("child did not terminate after the signal (hang)")
        finally:
            child.kill()
        return rc, child.stderr_bytes().decode("utf-8", errors="replace")

    def test_sigint_exit_130_clean_stderr_no_traceback(self) -> None:
        child = self._in_window(self._spawn())
        _send(child.pid, signal.SIGINT)
        rc, err = self._finish(child)
        self.assertEqual(rc, 130)
        self.assertNotIn("Traceback", err)
        self.assertEqual(err.strip().splitlines()[-1], _INTERRUPTED_LINE)

    def test_double_sigint_still_terminates_no_hang_no_traceback(self) -> None:
        child = self._in_window(self._spawn())
        _send(child.pid, signal.SIGINT)
        time.sleep(0.05)
        # May already be gone: that itself proves prompt termination.
        _send(child.pid, signal.SIGINT)
        rc, err = self._finish(child)
        self.assertEqual(rc, 130)
        self.assertNotIn("Traceback", err)

    def test_sigterm_default_disposition_pinned(self) -> None:
        child = self._in_window(self._spawn())
        _send(child.pid, signal.SIGTERM)
        rc, err = self._finish(child)
        # A scripted debugger session holds no output state worth draining:
        # default disposition (killed by SIGTERM) is the pinned contract.
        self.assertEqual(rc, -signal.SIGTERM)
        self.assertNotIn("Traceback", err)


# ---------------------------------------------------------------------------
# Layer 1b: real subprocess, run against a tiny REAL local GPT-2.
# ---------------------------------------------------------------------------


def _heavy_deps_available() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("torch", "transformers", "safetensors", "tokenizers")
    )


def _build_tiny_gpt2(root: Path) -> None:
    """Minimal GPT-2 checkpoint satisfying LocalHFBackend's local checks."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=256,
        n_positions=1024,
        n_embd=128,
        n_layer=4,
        n_head=4,
        n_inner=512,
        bos_token_id=255,
        eos_token_id=254,
    )
    GPT2LMHeadModel(config).save_pretrained(str(root))
    vocab = {f"tok{i}": i for i in range(254)}
    vocab["[UNK]"] = 254
    vocab["<|endoftext|>"] = 255
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(["<|endoftext|>"])
    tokenizer.save(str(root / "tokenizer.json"))


@unittest.skipUnless(_POSIX, "sends POSIX signals")
@unittest.skipUnless(_heavy_deps_available(), "needs torch+transformers+safetensors+tokenizers")
@unittest.skipUnless(
    os.environ.get("MPL_RUN_SLOW_SIGNAL_TESTS") == "1",
    "real-model signal tests are slow (~3min each); "
    "run explicitly with MPL_RUN_SLOW_SIGNAL_TESTS=1",
)
class RunSignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = _scratch("run")
        _write_fixture(cls.tmp, search_cases=4)
        cls.model_dir = cls.tmp / "tiny-gpt2"
        cls.model_dir.mkdir()
        _build_tiny_gpt2(cls.model_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _spawn_run(self, outdir: Path) -> tuple[_Child, Path]:
        # Deterministic readiness: the child writes the marker file right
        # after installing its SIGINT/SIGTERM handlers (see the
        # MODEL_PROTOCOL_LAB_TEST_SIGNAL_READY_FILE hook in cli.py), so the test signals
        # exactly inside the experiment window -- no timing guesses.
        marker = self.tmp / f"ready-{outdir.name}.marker"
        if marker.exists():
            marker.unlink()
        env = {**os.environ, "MODEL_PROTOCOL_LAB_TEST_SIGNAL_READY_FILE": str(marker)}
        child = _Child(
            [
                sys.executable,
                "-m",
                "model_protocol_lab",
                "run",
                "--model",
                str(self.model_dir),
                "--dataset",
                str(self.tmp / "fixture.json"),
                "--family",
                "chatml",
                "--boundary-ids",
                "198,199",
                "--output",
                str(outdir),
                "--max-new-tokens",
                "1000",
                "--max-context",
                "1024",
                "--seed",
                "7",
            ],
            cwd=str(self.tmp),
            env=env,
        )
        return child, marker

    def _in_generation_window(self, child: _Child, outdir: Path, marker: Path) -> _Child:
        # The CLI claims the output dir before backend init: its appearance
        # proves we are at-or-past the claim.
        claimed = _wait_until(outdir.exists, 120)
        self.assertTrue(claimed, "output dir was never claimed; child stuck before backend init")
        # The marker file is written by the child immediately after the
        # signal handlers are installed: signalling now deterministically
        # lands inside the experiment window. The result is asserted, never
        # silently ignored -- a missing marker means the child never reached
        # the cancellable phase.
        ready = _wait_until(marker.exists, 180)
        self.assertTrue(
            ready,
            "signal-handler readiness marker never appeared within 180s; refusing "
            "to signal a child that may not have installed its signal handlers yet",
        )
        return child

    def _assert_no_partial_report(self, outdir: Path) -> None:
        results = outdir / "results.json"
        report = outdir / "report.html"
        if not outdir.exists():
            # Interrupted before any report: the claim was released; a
            # retry still sees a fresh path.
            self.assertFalse(results.exists())
            self.assertFalse(report.exists())
            return
        # A present results.json must be complete and valid, never
        # half-written: json.loads either succeeds fully or raises.
        doc = json.loads(results.read_text(encoding="utf-8"))
        self.assertEqual(doc["decision"], "cancelled")
        self.assertTrue(report.is_file())

    def test_sigint_during_run_exit_130_no_partial_output(self) -> None:
        outdir = self.tmp / "sigint-out"
        child, marker = self._spawn_run(outdir)
        child = self._in_generation_window(child, outdir, marker)
        _send(child.pid, signal.SIGINT)
        try:
            rc = child.wait(120)
        except subprocess.TimeoutExpired:
            self.fail("run did not terminate within 120s of SIGINT (hang)")
        finally:
            child.kill()
        self.assertEqual(rc, 130)
        err = child.stderr_bytes().decode("utf-8", errors="replace")
        self.assertNotIn("Traceback", err)
        self._assert_no_partial_report(outdir)

    def test_sigterm_during_run_drives_cooperative_cancel(self) -> None:
        outdir = self.tmp / "sigterm-out"
        child, marker = self._spawn_run(outdir)
        child = self._in_generation_window(child, outdir, marker)
        # No settle sleep needed: the readiness marker is written AFTER the
        # signal handlers are installed, so SIGTERM deterministically lands
        # inside the experiment window where the cooperative cancel applies.
        _send(child.pid, signal.SIGTERM)
        try:
            rc = child.wait(120)
        except subprocess.TimeoutExpired:
            self.fail("run did not terminate within 120s of SIGTERM (hang)")
        finally:
            child.kill()
        # Cycle-89 fix: SIGTERM cooperates like SIGINT instead of killing
        # the process and leaking the claimed output directory.
        self.assertEqual(rc, 130)
        err = child.stderr_bytes().decode("utf-8", errors="replace")
        self.assertNotIn("Traceback", err)
        self._assert_no_partial_report(outdir)


# ---------------------------------------------------------------------------
# Layer 2: in-process fault injection through the real cli.main machinery.
# ---------------------------------------------------------------------------


class _SignallingBackend:
    """Fake LocalHFBackend: deterministic signal/KeyboardInterrupt injection."""

    fingerprint = "cycle89-signal-audit"
    eos_token_id = None
    mode = "sigint-once"

    def __init__(self, model, device=None, max_context=None) -> None:
        self.calls = 0

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        return (1, 2, 3)

    @staticmethod
    def decode(token_ids: tuple[int, ...]) -> str:
        return "yo"

    def generate(self, input_ids, *, max_new_tokens: int, seed: int, cancel: Event) -> Generation:
        self.calls += 1
        if self.mode == "sigint-once":
            os.kill(os.getpid(), signal.SIGINT)
        elif self.mode == "sigint-twice":
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGINT)
        elif self.mode == "sigterm-once":
            os.kill(os.getpid(), signal.SIGTERM)
        elif self.mode == "raise-ki":
            raise KeyboardInterrupt("injected mid-generate")
        # Exactly the cap with reason "limit": the ONLY cancellation signal
        # is the event the signal handler sets, so the test proves the
        # handler -> cancel -> "cancelled" path (a short "limit" would be
        # an invalid generation and fail closed before reaching it).
        return Generation(tuple(range(max_new_tokens)), "limit")


def _backend_with(mode: str):
    cls = type(f"_Backend_{mode}", (_SignallingBackend,), {"mode": mode})
    return cls


class RunMainSignalTests(unittest.TestCase):
    """Real signals through the real cli.main handler installation."""

    def setUp(self) -> None:
        self.tmp = _scratch("main")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.fixture = _write_fixture(self.tmp)

    def _run_main(self, backend_cls) -> tuple[int, Path, str, str]:
        outdir = self.tmp / "report"
        argv = [
            "run",
            "--model",
            "ckpt-dir",
            "--dataset",
            str(self.fixture),
            "--family",
            "chatml",
            "--boundary-ids",
            "198,199",
            "--output",
            str(outdir),
            "--seed",
            "7",
        ]
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(backends, "LocalHFBackend", backend_cls),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            rc = cli.main(argv)
        return rc, outdir, stdout.getvalue(), stderr.getvalue()

    def _assert_cancelled_report(self, outdir: Path, err: str) -> None:
        self.assertNotIn("Traceback", err)
        doc = json.loads((outdir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["decision"], "cancelled")
        self.assertIsNone(doc["selected_candidate"])
        self.assertTrue((outdir / "report.html").is_file())

    def test_sigint_during_generate_yields_cancelled_report_and_130(self) -> None:
        rc, outdir, _out, err = self._run_main(_backend_with("sigint-once"))
        self.assertEqual(rc, 130)
        self._assert_cancelled_report(outdir, err)

    def test_double_sigint_during_generate_no_hang(self) -> None:
        rc, outdir, _out, err = self._run_main(_backend_with("sigint-twice"))
        self.assertEqual(rc, 130)
        self._assert_cancelled_report(outdir, err)

    def test_sigterm_during_generate_drives_cooperative_cancel(self) -> None:
        # Without the cycle-89 SIGTERM wiring this would kill the test
        # process itself; with it, SIGTERM behaves exactly like SIGINT.
        rc, outdir, _out, err = self._run_main(_backend_with("sigterm-once"))
        self.assertEqual(rc, 130)
        self._assert_cancelled_report(outdir, err)

    def test_keyboardinterrupt_in_generate_cleans_claimed_dir(self) -> None:
        rc, outdir, _out, err = self._run_main(_backend_with("raise-ki"))
        self.assertEqual(rc, 130)
        self.assertNotIn("Traceback", err)
        self.assertEqual(err.strip().splitlines()[-1], _INTERRUPTED_LINE)
        # No report was saved: the claimed directory must be released so a
        # retry still sees a fresh output path.
        self.assertFalse(outdir.exists())


class RunExperimentPropagationTests(unittest.TestCase):
    def test_keyboardinterrupt_in_generate_propagates_unchanged(self) -> None:
        tmp = _scratch("propexp")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        split = load_dataset(str(_write_fixture(tmp)))
        formats = candidates("chatml")
        # evaluate() catches only `Exception`; KeyboardInterrupt (a
        # BaseException) must propagate unchanged -- never swallowed into a
        # verdict, never converted to a confusing error code.
        with self.assertRaises(KeyboardInterrupt):
            run_experiment(
                _backend_with("raise-ki")("m"),
                formats,
                split,
                boundary=(198, 199),
                max_new_tokens=64,
                seed=7,
                cancel=Event(),
            )


class SaveReportAtomicityTests(unittest.TestCase):
    """Fault-injecting writer: KeyboardInterrupt mid-write must leave nothing."""

    def setUp(self) -> None:
        self.tmp = _scratch("report")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    @staticmethod
    def _result() -> dict:
        return {
            "schema_version": 1,
            "decision": "cancelled",
            "selected_candidate": None,
            "search": [],
            "held_out": None,
            "failure_taxonomy": {},
        }

    @staticmethod
    def _boom_fdopen(fail_on: int):
        real_fdopen = os.fdopen
        state = {"calls": 0}

        def fdopen(fd, *args, **kwargs):
            handle = real_fdopen(fd, *args, **kwargs)
            state["calls"] += 1
            if state["calls"] == fail_on:
                original_write = handle.write

                def write_and_boom(text: str) -> None:
                    original_write(text[:16])
                    raise KeyboardInterrupt("injected mid-write")

                handle.write = write_and_boom  # type: ignore[method-assign]
            return handle

        return fdopen

    def _assert_no_report_files(self, target: Path) -> None:
        self.assertFalse(
            (target / "results.json").exists(), "half-written results.json left behind"
        )
        self.assertFalse((target / "report.html").exists(), "half-written report.html left behind")

    def test_keyboardinterrupt_during_results_json_write_leaves_no_partial(self) -> None:
        target = self.tmp / "report"
        with (
            mock.patch.object(os, "fdopen", self._boom_fdopen(fail_on=1)),
            self.assertRaises(KeyboardInterrupt),
        ):
            reporting_mod.save_report(self._result(), target)
        self._assert_no_report_files(target)

    def test_keyboardinterrupt_during_report_html_write_removes_pair(self) -> None:
        target = self.tmp / "report"
        with (
            mock.patch.object(os, "fdopen", self._boom_fdopen(fail_on=2)),
            self.assertRaises(KeyboardInterrupt),
        ):
            reporting_mod.save_report(self._result(), target)
        # The fully-written results.json must go too: never a half-built pair.
        self._assert_no_report_files(target)


if __name__ == "__main__":
    unittest.main()
