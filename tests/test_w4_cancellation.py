"""W4 push 5 (ORCH-02): one cancellation lifetime for the whole run.

Regression-first pins for the single-lifetime design (readiness
brief section 4): the CLI's signal handlers stay installed from
the experiment through the final report save; the first signal
sets the shared event and a second one restores the default
disposition and re-raises; the diagnosis sections degrade to
their cancelled shapes without secondary loads; transfer and
batch stages receive the owner event instead of private ones;
the MCP server honours ``notifications/cancelled`` through a
per-request registry and sets every outstanding event at stdio
EOF. Expected values come from the brief and the audit's
acceptance text, never from the pre-fix behaviour.
"""

import io
import json
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from model_protocol_lab import cli as cli_module
from model_protocol_lab import core, doctor, experiments, mcp_server
from model_protocol_lab.backends import LocalHFBackend  # noqa: F401  (patch anchor)
from model_protocol_lab.core import Event, Fixture, Message

REPO_ROOT = Path(__file__).resolve().parent.parent
MPL = [sys.executable, "-m", "model_protocol_lab"]
BOUNDARY = (999,)
MAX_NEW_TOKENS = 40
EXPECTED = {"S1": "GREEN", "S2": "BLUE", "H1": "RED", "H2": "CYAN"}
BEHAVIOR = {("cand-B", "S2"): "fail"}


class ScriptedBackend:
    """Deterministic backend: ord-encode; pass text + boundary, padded."""

    fingerprint = "w4-cancellation"

    def __init__(self, expected, behavior=None):
        """Init."""
        self._expected = dict(expected)
        self._behavior = dict(behavior or {})
        self.generate_calls = 0
        self.encode_calls = 0

    def encode(self, text):
        """Encode."""
        self.encode_calls += 1
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids):
        """Decode."""
        return "".join(map(chr, token_ids))

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        self.generate_calls += 1
        prompt = self.decode(input_ids)
        candidate = next((n for n in ("cand-A", "cand-B") if f"[{n}]" in prompt), None)
        fixture = next((fid for fid in self._expected if f"question-{fid}" in prompt), None)
        if candidate is None or fixture is None:
            return core.Generation((), "error", "unknown_request")
        mode = self._behavior.get((candidate, fixture), "pass")
        expected = self._expected[fixture]
        text = expected + chr(BOUNDARY[0]) if mode == "pass" else expected + "WRONG"
        ids = tuple(map(ord, text))
        assert len(ids) <= max_new_tokens
        return core.Generation(ids + (1,) * (max_new_tokens - len(ids)), "limit")


def _formats():
    """Formats."""
    return tuple(
        core.Format(
            name=name,
            bos="",
            user_prefix=f"[{name}]U:",
            assistant_prefix="A:",
            turn_end="\n",
        )
        for name in ("cand-A", "cand-B")
    )


def _cases():
    """Cases."""
    return tuple(
        Fixture(
            id=fid,
            messages=(Message(role="user", content=f"question-{fid}"),),
            expected=EXPECTED[fid],
            group="g",
        )
        for fid in ("S1", "S2")
    )


class TransferStageCancelTests(unittest.TestCase):
    """Test transfer stage cancel."""

    def test_preset_event_means_zero_factory_calls(self):
        """Test preset event means zero factory calls."""
        cancel = Event()
        cancel.set()
        factory_calls = []

        def factory(ckpt):
            """Factory."""
            factory_calls.append(ckpt)
            return ScriptedBackend(EXPECTED, BEHAVIOR)

        outcomes = experiments.transfer_stage(
            factory,
            ("ck1", "ck2"),
            _formats(),
            _cases(),
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            cancel=cancel,
        )
        self.assertEqual(factory_calls, [])
        self.assertEqual(
            outcomes,
            {
                "ck1": {"cand-A": None, "cand-B": None},
                "ck2": {"cand-A": None, "cand-B": None},
            },
        )

    def test_event_set_mid_stage_stops_further_work(self):
        """Test event set mid stage stops further work."""
        cancel = Event()
        factory_calls = []
        backends = []

        def factory(ckpt):
            """Factory."""
            factory_calls.append(ckpt)
            if ckpt == "ck2":
                cancel.set()
            backend = ScriptedBackend(EXPECTED, BEHAVIOR)
            backends.append(backend)
            return backend

        outcomes = experiments.transfer_stage(
            factory,
            ("ck1", "ck2", "ck3"),
            _formats(),
            _cases(),
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            cancel=cancel,
        )
        # ck1 evaluated fully; ck2 was constructed (its factory set
        # the event) but never evaluated; ck3 was never constructed.
        self.assertEqual(factory_calls, ["ck1", "ck2"])
        self.assertEqual(backends[0].generate_calls, 4)
        self.assertEqual(backends[1].generate_calls, 0)
        self.assertEqual(outcomes["ck1"], {"cand-A": True, "cand-B": False})
        self.assertEqual(outcomes["ck2"], {"cand-A": None, "cand-B": None})
        self.assertEqual(outcomes["ck3"], {"cand-A": None, "cand-B": None})

    def test_no_cancel_keeps_existing_behaviour(self):
        """Test no cancel keeps existing behaviour."""
        outcomes = experiments.transfer_stage(
            lambda ckpt: ScriptedBackend(EXPECTED, BEHAVIOR),
            ("ck1",),
            _formats(),
            _cases(),
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        self.assertEqual(outcomes, {"ck1": {"cand-A": True, "cand-B": False}})


class _BatchBackend:
    """Backend with a batch path; records which event each call saw."""

    fingerprint = "w4-batch"

    def __init__(self, on_solo=None):
        """Init."""
        self.solo_calls = 0
        self.batch_calls = 0
        self.seen_events = []
        self._on_solo = on_solo

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        self.solo_calls += 1
        self.seen_events.append(cancel)
        if self._on_solo is not None:
            self._on_solo(self.solo_calls)
        return core.Generation((7, 8), "limit")

    def generate_batch(self, prompts, *, max_new_tokens, seed, cancel):
        """Generate batch."""
        self.batch_calls += 1
        self.seen_events.append(cancel)
        return tuple((7, 8) for _ in prompts)


class BatchStageCancelTests(unittest.TestCase):
    """Test batch stage cancel."""

    def test_preset_event_means_zero_generations(self):
        """Test preset event means zero generations."""
        cancel = Event()
        cancel.set()
        backend = _BatchBackend()
        out = experiments.batch_stage(
            backend, {"p1": (1, 2), "p2": (3, 4)}, max_new_tokens=8, seed=1, cancel=cancel
        )
        self.assertEqual(out, {"available": False, "detail": "cancelled"})
        self.assertEqual((backend.solo_calls, backend.batch_calls), (0, 0))

    def test_owner_event_reaches_every_call(self):
        """Test owner event reaches every call."""
        cancel = Event()

        def trip(call_no):
            """Trip."""
            if call_no == 1:
                cancel.set()

        backend = _BatchBackend(on_solo=trip)
        out = experiments.batch_stage(
            backend, {"p1": (1, 2), "p2": (3, 4)}, max_new_tokens=8, seed=1, cancel=cancel
        )
        self.assertEqual(out, {"available": False, "detail": "cancelled"})
        self.assertEqual(backend.solo_calls, 1)
        self.assertEqual(backend.batch_calls, 0)
        self.assertTrue(all(event is cancel for event in backend.seen_events))

    def test_no_cancel_keeps_existing_behaviour(self):
        """Test no cancel keeps existing behaviour."""
        backend = _BatchBackend()
        out = experiments.batch_stage(backend, {"p1": (1, 2)}, max_new_tokens=8, seed=1)
        self.assertTrue(out["available"])
        self.assertEqual((backend.solo_calls, backend.batch_calls), (1, 1))


class DoctorCancelTests(unittest.TestCase):
    """Test doctor cancel."""

    def test_conformance_preset_event_loads_nothing(self):
        """Test conformance preset event loads nothing."""
        cancel = Event()
        cancel.set()
        out = doctor.template_conformance(
            "/nonexistent-checkpoint", _formats(), _cases(), cancel=cancel
        )
        self.assertEqual(out, {"status": "unavailable", "detail": "cancelled", "diffs": []})

    def test_firewall_preset_event_loads_nothing(self):
        """Test firewall preset event loads nothing."""
        cancel = Event()
        cancel.set()
        out = doctor.template_firewall("/nonexistent-checkpoint", cancel=cancel)
        self.assertEqual(out, {"status": "unavailable", "detail": "cancelled", "findings": []})

    def test_marker_isolation_preset_event_loads_nothing(self):
        """Test marker isolation preset event loads nothing."""
        cancel = Event()
        cancel.set()
        out = doctor.marker_isolation(
            "/nonexistent-checkpoint", _formats(), _cases(), cancel=cancel
        )
        self.assertEqual(out, {"status": "unavailable", "detail": "cancelled", "probes": []})

    def test_conformance_cancelled_mid_sweep_discards_partial(self):
        """Test conformance cancelled mid sweep discards partial."""
        cancel = Event()

        class FakeSession:
            """Fake session."""

            def __init__(self, *args, **kwargs):
                """Init."""

            def start(self):
                """Start."""

            def __enter__(self):
                """Enter."""
                return self

            def __exit__(self, *exc):
                """Exit."""
                return False

            def close(self):
                """Close."""

            def encode(self, text):
                """Encode."""
                return (1, 2, 3)

            def render(self, messages):
                """Render."""
                cancel.set()
                return "rendered"

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tokenizer_config.json").write_text(
                json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
            )
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(doctor, "RenderSession", FakeSession):
                out = doctor.template_conformance(root, _formats(), _cases(), cancel=cancel)
        self.assertEqual(out, {"status": "unavailable", "detail": "cancelled", "diffs": []})


class CancelHandlerTests(unittest.TestCase):
    """Test cancel handler."""

    def test_first_signal_sets_second_restores_and_reraises(self):
        """Test first signal sets second restores and reraises."""
        cancel = Event()
        handler = cli_module._make_cancel_handler(cancel)
        handler(signal.SIGTERM, None)
        self.assertTrue(cancel.is_set())
        with (
            mock.patch.object(cli_module.signal, "signal") as sig_mock,
            mock.patch.object(cli_module.signal, "raise_signal") as raise_mock,
        ):
            handler(signal.SIGTERM, None)
        sig_mock.assert_called_once_with(signal.SIGTERM, signal.SIG_DFL)
        raise_mock.assert_called_once_with(signal.SIGTERM)


class _DispatchFakeBackend:
    """Backend double for in-process CLI dispatch tests."""

    fingerprint = "w4-cancel-dispatch"

    def __init__(self, *args, **kwargs):
        """Init."""


class CliCancelDispatchTests(unittest.TestCase):
    """Test cli cancel dispatch."""

    def _fixture_file(self, directory):
        """Fixture file."""
        path = Path(directory) / "fixture.json"
        path.write_text(
            json.dumps(
                {
                    "search": [
                        {
                            "id": "s1",
                            "group": "g1",
                            "messages": [{"role": "user", "content": "hi"}],
                            "expected": "yo",
                        }
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

    def _canned_cancelled_result(self):
        """Canned cancelled result."""
        search = tuple(
            Fixture(
                id=fid,
                messages=(Message(role="user", content=f"question-{fid}"),),
                expected=EXPECTED[fid],
                group="gs",
            )
            for fid in ("S1", "S2")
        )
        held = tuple(
            Fixture(
                id=fid,
                messages=(Message(role="user", content=f"question-{fid}"),),
                expected=EXPECTED[fid],
                group="gh",
            )
            for fid in ("H1", "H2")
        )
        split = core.Split(search=search, held_out=held)
        preset = Event()
        preset.set()
        result = experiments.run_experiment(
            ScriptedBackend(EXPECTED, BEHAVIOR),
            _formats(),
            split,
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=7,
            cancel=preset,
        )
        assert result["decision"] == "cancelled"
        return result

    def test_signal_during_experiment_skips_diagnosis_loads(self):
        """Test signal during experiment skips diagnosis loads."""
        from model_protocol_lab import backends as backends_module

        canned = self._canned_cancelled_result()
        saved = {}

        def stub_run(*args, **kwargs):
            """Stub run."""
            handler = signal.getsignal(signal.SIGTERM)
            handler(signal.SIGTERM, None)
            assert kwargs["cancel"].is_set()
            return canned

        def stub_save(result, *args, **kwargs):
            """Stub save."""
            saved["result"] = result
            saved["handler_at_save"] = signal.getsignal(signal.SIGINT)
            return (Path("results.json"), Path("report.html"))

        with TemporaryDirectory() as tmp:
            fixture = self._fixture_file(tmp)
            output = Path(tmp) / "out"
            argv = [
                "run",
                "--model",
                str(tmp),
                "--dataset",
                str(fixture),
                "--output",
                str(output),
                "--family",
                "chatml",
                "--boundary-ids",
                "999",
                "--max-new-tokens",
                "40",
                "--seed",
                "7",
            ]
            before = signal.getsignal(signal.SIGINT)
            out, err = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(backends_module, "LocalHFBackend", _DispatchFakeBackend),
                mock.patch.object(cli_module, "run_experiment", stub_run),
                mock.patch.object(cli_module, "save_report", stub_save),
                redirect_stdout(out),
                redirect_stderr(err),
            ):
                code = cli_module.main(argv)
            self.assertEqual(code, 130)
            result = saved["result"]
            self.assertEqual(result["template_conformance"]["detail"], "cancelled")
            self.assertEqual(result["template_firewall"]["detail"], "cancelled")
            self.assertEqual(result["marker_isolation"]["detail"], "cancelled")
            # The cooperative handler was still installed while the
            # report was being saved, and the previous disposition
            # is back afterwards: one lifetime, start to report.
            self.assertTrue(callable(saved["handler_at_save"]))
            self.assertIsNot(saved["handler_at_save"], before)
            self.assertEqual(signal.getsignal(signal.SIGINT), before)


class McpCancelTests(unittest.TestCase):
    """Test mcp cancel."""

    def _double(self, started, record):
        """Double."""

        def handler(arguments):
            """Handler."""
            started.set()
            event = mcp_server.current_call_event()
            record["saw_event"] = event is not None
            if event is not None:
                record["was_set"] = event.wait(timeout=10)
            return {"ok": True}

        return handler

    def test_notification_sets_the_registered_event(self):
        """Test notification sets the registered event."""
        started = threading.Event()
        record = {}
        frame = {
            "jsonrpc": "2.0",
            "id": "req-1",
            "method": "tools/call",
            "params": {"name": "probe_surface_noise_stability", "arguments": {}},
        }
        with mock.patch.dict(
            mcp_server.TOOLS["probe_surface_noise_stability"],
            {"handler": self._double(started, record)},
        ):
            worker = threading.Thread(target=lambda: mcp_server.handle_request(frame))
            worker.start()
            self.assertTrue(started.wait(timeout=10))
            reply = mcp_server.handle_request(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/cancelled",
                    "params": {"requestId": "req-1"},
                }
            )
            self.assertIsNone(reply)
            worker.join(timeout=10)
        self.assertFalse(worker.is_alive())
        self.assertEqual(record, {"saw_event": True, "was_set": True})

    def test_notification_for_unknown_id_is_a_noop(self):
        """Test notification for unknown id is a noop."""
        reply = mcp_server.handle_request(
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": "nobody"},
            }
        )
        self.assertIsNone(reply)

    def test_cancelled_frame_with_an_id_is_a_request(self):
        """Test cancelled frame with an id is a request."""
        reply = mcp_server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "notifications/cancelled",
                "params": {"requestId": 5},
            }
        )
        self.assertEqual(reply["error"]["code"], -32601)

    def test_eof_sets_outstanding_events(self):
        """Test eof sets outstanding events."""
        started = threading.Event()
        record = {}
        line = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "req-9",
                "method": "tools/call",
                "params": {"name": "probe_surface_noise_stability", "arguments": {}},
            }
        )
        fake_stdin = mock.Mock()
        fake_stdin.buffer = io.BytesIO((line + "\n").encode("utf-8"))
        fake_stdout = mock.Mock()
        fake_stdout.buffer = io.BytesIO()
        with (
            mock.patch.dict(
                mcp_server.TOOLS["probe_surface_noise_stability"],
                {"handler": self._double(started, record)},
            ),
            mock.patch.object(sys, "stdin", fake_stdin),
            mock.patch.object(sys, "stdout", fake_stdout),
        ):
            code = mcp_server.serve([])
        self.assertEqual(code, 0)
        self.assertEqual(record, {"saw_event": True, "was_set": True})


def _toolkit_available() -> bool:
    """Toolkit available."""
    import importlib.util

    return all(
        importlib.util.find_spec(name) is not None
        for name in ("torch", "transformers", "safetensors", "tokenizers", "numpy")
    )


def _build_checkpoint(root: Path) -> Path:
    """Build a tiny deterministic GPT-2 checkpoint (no network)."""
    import numpy as np
    from safetensors.numpy import save_file
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as _BLDec
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel

    root.mkdir(parents=True, exist_ok=True)
    n_embd, n_layer, n_pos, vocab = 64, 2, 1024, 267
    rng = np.random.default_rng(7)
    tensors: dict = {}
    tensors["transformer.wte.weight"] = rng.normal(0, 0.02, (vocab, n_embd)).astype(np.float32)
    tensors["transformer.wpe.weight"] = rng.normal(0, 0.02, (n_pos, n_embd)).astype(np.float32)
    for i in range(n_layer):
        p = f"transformer.h.{i}"
        tensors[f"{p}.ln_1.weight"] = np.ones(n_embd, dtype=np.float32)
        tensors[f"{p}.ln_1.bias"] = np.zeros(n_embd, dtype=np.float32)
        tensors[f"{p}.attn.c_attn.weight"] = rng.normal(0, 0.02, (n_embd, 3 * n_embd)).astype(
            np.float32
        )
        tensors[f"{p}.attn.c_attn.bias"] = np.zeros(3 * n_embd, dtype=np.float32)
        tensors[f"{p}.attn.c_proj.weight"] = rng.normal(0, 0.02, (n_embd, n_embd)).astype(
            np.float32
        )
        tensors[f"{p}.attn.c_proj.bias"] = np.zeros(n_embd, dtype=np.float32)
        tensors[f"{p}.ln_2.weight"] = np.ones(n_embd, dtype=np.float32)
        tensors[f"{p}.ln_2.bias"] = np.zeros(n_embd, dtype=np.float32)
        tensors[f"{p}.mlp.c_fc.weight"] = rng.normal(0, 0.02, (n_embd, 4 * n_embd)).astype(
            np.float32
        )
        tensors[f"{p}.mlp.c_fc.bias"] = np.zeros(4 * n_embd, dtype=np.float32)
        tensors[f"{p}.mlp.c_proj.weight"] = rng.normal(0, 0.02, (4 * n_embd, n_embd)).astype(
            np.float32
        )
        tensors[f"{p}.mlp.c_proj.bias"] = np.zeros(n_embd, dtype=np.float32)
    tensors["transformer.ln_f.weight"] = np.ones(n_embd, dtype=np.float32)
    tensors["transformer.ln_f.bias"] = np.zeros(n_embd, dtype=np.float32)
    save_file(tensors, str(root / "model.safetensors"))
    vocab_map = {bytes([i]).decode("latin-1"): i for i in range(256)}
    for i in range(10):
        vocab_map[f"tok{i}"] = 256 + i
    vocab_map["<|eos|>"] = 266
    tok = Tokenizer(BPE(vocab=vocab_map, merges=[]))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = _BLDec()
    tok.add_special_tokens(["<|eos|>"])
    tok.save(str(root / "tokenizer.json"))
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["GPT2LMHeadModel"],
                "model_type": "gpt2",
                "n_positions": n_pos,
                "n_ctx": n_pos,
                "n_embd": n_embd,
                "n_layer": n_layer,
                "n_head": 2,
                "vocab_size": vocab,
            }
        )
    )
    return root


def _write_dataset(path: Path, *, search: int = 24, held_out: int = 4) -> Path:
    """Write dataset."""

    def fixtures(prefix, count, word):
        """Fixtures."""
        return [
            {
                "id": f"{prefix}-{i}",
                "group": f"g-{prefix}",
                "messages": [
                    {"role": "user", "content": f"Reply with exactly {word}{i} and nothing else."}
                ],
                "expected": f"{word}{i}",
            }
            for i in range(count)
        ]

    path.write_text(
        json.dumps({"search": fixtures("s", search, "W"), "held_out": fixtures("h", held_out, "H")})
    )
    return path


@unittest.skipUnless(_toolkit_available(), "HF toolkit not installed")
class RealSignalCliTests(unittest.TestCase):
    """Brief 4.3: real SIGTERM against the real CLI, three arms."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test fixture."""
        cls._tmp = tempfile.mkdtemp(prefix="mpl_w4_cancel_")
        cls.model = _build_checkpoint(Path(cls._tmp) / "model")
        cls.dataset = _write_dataset(Path(cls._tmp) / "data.json")

    def _spawn(self, tmp: str, name: str):
        """Spawn."""
        import os

        output = Path(tmp) / name
        marker = Path(tmp) / f"{name}.ready"
        env = dict(os.environ)
        env["MODEL_PROTOCOL_LAB_TEST_SIGNAL_READY_FILE"] = str(marker)
        proc = subprocess.Popen(
            [
                *MPL,
                "run",
                "--model",
                str(self.model),
                "--dataset",
                str(self.dataset),
                "--family",
                "chatml",
                "--mutate",
                "--boundary-ids",
                "266",
                "--max-new-tokens",
                "48",
                "--max-context",
                "1024",
                "--output",
                str(output),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 180
        while not marker.exists():
            if proc.poll() is not None:
                self.fail(f"run exited {proc.returncode} before the ready marker")
            if time.monotonic() > deadline:
                proc.kill()
                self.fail("ready marker never appeared")
            time.sleep(0.02)
        return proc, output

    def test_sigterm_mid_run_drains_to_a_cancelled_report(self):
        """Test sigterm mid run drains to a cancelled report."""
        from model_protocol_lab.diffreports import load_results

        with TemporaryDirectory() as tmp:
            proc, output = self._spawn(tmp, "mid")
            time.sleep(8)
            sent = time.monotonic()
            proc.send_signal(signal.SIGTERM)
            returncode = proc.wait(timeout=120)
            drain = time.monotonic() - sent
            self.assertEqual(returncode, 130)
            self.assertLess(drain, 30)
            result = load_results(output)
            self.assertEqual(result["decision"], "cancelled")
            self.assertEqual(result["template_conformance"]["detail"], "cancelled")
            self.assertEqual(result["template_firewall"]["detail"], "cancelled")
            self.assertEqual(result["marker_isolation"]["detail"], "cancelled")

    def test_sigterm_before_generation_means_zero_completed_cells(self):
        """Test sigterm before generation means zero completed cells."""
        with TemporaryDirectory() as tmp:
            proc, output = self._spawn(tmp, "early")
            proc.send_signal(signal.SIGTERM)
            returncode = proc.wait(timeout=120)
            self.assertEqual(returncode, 130)
            result = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(result["decision"], "cancelled")
            completed = sum(
                1
                for entry in result["search"]
                for obs in entry.get("observations", [])
                if obs["status"] == "completed"
            )
            self.assertEqual(completed, 0)

    def test_second_sigterm_is_a_hard_exit(self):
        """Test second sigterm is a hard exit."""
        with TemporaryDirectory() as tmp:
            proc, output = self._spawn(tmp, "twice")
            time.sleep(8)
            proc.send_signal(signal.SIGTERM)
            time.sleep(0.3)
            proc.send_signal(signal.SIGTERM)
            returncode = proc.wait(timeout=120)
            self.assertEqual(returncode, -signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
