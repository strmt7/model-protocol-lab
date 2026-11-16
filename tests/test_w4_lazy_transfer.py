"""W4 push 6 (ORCH-03): lazy transfer backends, one live at a time.

The audit found the transfer stage's secondary backends were all
constructed eagerly and retained for the whole stage: with up to
four ``--transfer-model`` checkpoints, four full model loads were
live simultaneously, and nothing ever released them. The contract
pinned here:

- ``experiments.transfer_stage`` closes each checkpoint's backend
  (when it offers ``close()``) before the next checkpoint's factory
  call -- at most one secondary backend is live at a time -- and a
  failing ``close()`` never fails the stage.
- ``LocalHFBackend.close()`` releases the model/tokenizer
  references (and the CUDA allocator cache on CUDA devices),
  idempotently.
- The CLI constructs transfer backends lazily through the stage's
  factory -- a cancelled run constructs none at all -- and records
  each checkpoint's fingerprint at construction time.
"""

import io
import json
import signal
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, RLock
from types import SimpleNamespace
from unittest import mock

from model_protocol_lab import backends as backends_module
from model_protocol_lab import cli as cli_module
from model_protocol_lab import core, experiments
from model_protocol_lab.backends import LocalHFBackend
from model_protocol_lab.core import Fixture, Message
from model_protocol_lab.experiments import transfer_stage

BOUNDARY = (999,)
MAX_NEW_TOKENS = 40
EXPECTED = {"S1": "GREEN", "S2": "BLUE", "H1": "RED", "H2": "CYAN"}


class ScriptedCore:
    """Scripted evaluate-compatible backend surface (no close)."""

    fingerprint = "w4-lazy-transfer"

    @staticmethod
    def encode(text):
        """Encode."""
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids):
        """Decode."""
        return "".join(map(chr, token_ids))

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        prompt = self.decode(input_ids)
        fixture = next((fid for fid in EXPECTED if f"question-{fid}" in prompt), None)
        if fixture is None:
            return core.Generation((), "error", "unknown_request")
        ids = tuple(map(ord, EXPECTED[fixture] + chr(BOUNDARY[0])))
        return core.Generation(ids + (1,) * (max_new_tokens - len(ids)), "limit")


class RecordingBackend(ScriptedCore):
    """Scripted backend that logs its close into a shared event log."""

    def __init__(self, log, name, *, close_raises=False):
        """Init."""
        self._log = log
        self._name = name
        self._close_raises = close_raises

    def close(self):
        """Close."""
        self._log.append(("close", self._name))
        if self._close_raises:
            raise RuntimeError("close boom")


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


class TransferStageLifecycleTests(unittest.TestCase):
    """Test transfer stage lifecycle."""

    def test_one_secondary_live_at_a_time(self):
        """Test one secondary live at a time."""
        log = []

        def factory(ckpt):
            """Factory."""
            log.append(("construct", ckpt))
            return RecordingBackend(log, ckpt)

        outcomes = transfer_stage(
            factory,
            ("ck1", "ck2"),
            _formats(),
            _cases(),
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=7,
        )
        self.assertEqual(
            outcomes,
            {
                "ck1": {"cand-A": True, "cand-B": True},
                "ck2": {"cand-A": True, "cand-B": True},
            },
        )
        self.assertEqual(
            log,
            [
                ("construct", "ck1"),
                ("close", "ck1"),
                ("construct", "ck2"),
                ("close", "ck2"),
            ],
        )

    def test_close_failure_does_not_fail_the_stage(self):
        """Test close failure does not fail the stage."""
        log = []

        def factory(ckpt):
            """Factory."""
            log.append(("construct", ckpt))
            return RecordingBackend(log, ckpt, close_raises=True)

        outcomes = transfer_stage(
            factory,
            ("ck1", "ck2"),
            _formats(),
            _cases(),
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=7,
        )
        self.assertEqual(outcomes["ck2"], {"cand-A": True, "cand-B": True})
        self.assertEqual(log.count(("close", "ck1")), 1)

    def test_backend_without_close_is_supported(self):
        """Test backend without close is supported."""

        def factory(ckpt):
            """Factory."""
            return ScriptedCore()

        outcomes = transfer_stage(
            factory,
            ("ck1",),
            _formats(),
            _cases(),
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=7,
        )
        self.assertEqual(outcomes, {"ck1": {"cand-A": True, "cand-B": True}})

    def test_failed_construction_records_none_and_continues(self):
        """Test failed construction records none and continues."""
        log = []

        def factory(ckpt):
            """Factory."""
            log.append(("construct", ckpt))
            if ckpt == "ck1":
                raise RuntimeError("load boom")
            return RecordingBackend(log, ckpt)

        outcomes = transfer_stage(
            factory,
            ("ck1", "ck2"),
            _formats(),
            _cases(),
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=7,
        )
        self.assertEqual(outcomes["ck1"], {"cand-A": None, "cand-B": None})
        self.assertEqual(outcomes["ck2"], {"cand-A": True, "cand-B": True})
        self.assertEqual(
            log,
            [("construct", "ck1"), ("construct", "ck2"), ("close", "ck2")],
        )


class LocalHFBackendCloseTests(unittest.TestCase):
    """Test local hf backend close."""

    def _bare(self):
        """Bare."""
        backend = LocalHFBackend.__new__(LocalHFBackend)
        backend._lock = RLock()
        backend._model = object()
        backend._tokenizer = object()
        backend._torch = None
        backend._device = None
        return backend

    def test_close_releases_references_and_is_idempotent(self):
        """Test close releases references and is idempotent."""
        backend = self._bare()
        backend.close()
        self.assertIsNone(backend._model)
        self.assertIsNone(backend._tokenizer)
        backend.close()
        self.assertIsNone(backend._model)

    def test_close_empties_cuda_cache(self):
        """Test close empties cuda cache."""
        calls = []
        backend = self._bare()
        backend._torch = SimpleNamespace(
            cuda=SimpleNamespace(empty_cache=lambda: calls.append("emptied"))
        )
        backend._device = SimpleNamespace(type="cuda")
        backend.close()
        self.assertEqual(calls, ["emptied"])

    def test_close_swallows_cuda_cache_failure(self):
        """Test close swallows cuda cache failure."""

        def boom():
            """Boom."""
            raise RuntimeError("cuda boom")

        backend = self._bare()
        backend._torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=boom))
        backend._device = SimpleNamespace(type="cuda")
        backend.close()
        self.assertIsNone(backend._model)


class _CliFakeBackend:
    """Recording LocalHFBackend double for CLI dispatch tests."""

    fingerprint = "w4-lazy-cli"
    log: list = []

    def __init__(self, path, **kwargs):
        """Init."""
        self._name = Path(path).name
        type(self).log.append(("construct", self._name))

    @staticmethod
    def encode(text):
        """Encode."""
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids):
        """Decode."""
        return "".join(map(chr, token_ids))

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        ids = tuple(map(ord, "yo" + chr(999)))
        return core.Generation(ids + (1,) * (max_new_tokens - len(ids)), "limit")

    def close(self):
        """Close."""
        type(self).log.append(("close", self._name))


class CliLazyTransferTests(unittest.TestCase):
    """Test cli lazy transfer."""

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

    def _canned_result(self, *, cancelled):
        """Canned result."""
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
        backend = RecordingBackend([], "primary")
        cancel = Event()
        if cancelled:
            cancel.set()
        return experiments.run_experiment(
            backend,
            _formats(),
            split,
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=7,
            cancel=cancel,
        )

    def _dispatch(self, tmp, *, cancelled):
        """Dispatch."""
        _CliFakeBackend.log = []
        fixture = self._fixture_file(tmp)
        for name in ("ckA", "ckB"):
            (Path(tmp) / name).mkdir()
        canned = self._canned_result(cancelled=cancelled)
        saved = {}

        def stub_run(*args, **kwargs):
            """Stub run."""
            if cancelled:
                handler = signal.getsignal(signal.SIGTERM)
                handler(signal.SIGTERM, None)
                assert kwargs["cancel"].is_set()
            return canned

        def stub_save(result, *args, **kwargs):
            """Stub save."""
            saved["result"] = result
            return (Path("results.json"), Path("report.html"))

        argv = [
            "run",
            "--model",
            str(tmp),
            "--dataset",
            str(fixture),
            "--output",
            str(Path(tmp) / "out"),
            "--family",
            "chatml",
            "--boundary-ids",
            "999",
            "--max-new-tokens",
            "40",
            "--seed",
            "7",
            "--transfer-model",
            str(Path(tmp) / "ckA"),
            "--transfer-model",
            str(Path(tmp) / "ckB"),
        ]
        out, err = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(backends_module, "LocalHFBackend", _CliFakeBackend),
            mock.patch.object(cli_module, "run_experiment", stub_run),
            mock.patch.object(cli_module, "save_report", stub_save),
            redirect_stdout(out),
            redirect_stderr(err),
        ):
            code = cli_module.main(argv)
        return code, saved["result"]

    def test_transfer_backends_construct_lazily_and_close_in_order(self):
        """Test transfer backends construct lazily and close in order."""
        with TemporaryDirectory() as tmp:
            code, result = self._dispatch(tmp, cancelled=False)
        self.assertNotEqual(code, 130)
        transfer_events = [e for e in _CliFakeBackend.log if e[1] in ("ckA", "ckB")]
        self.assertEqual(
            transfer_events,
            [
                ("construct", "ckA"),
                ("close", "ckA"),
                ("construct", "ckB"),
                ("close", "ckB"),
            ],
        )
        # The primary backend is never closed by the transfer stage.
        self.assertNotIn(("close", Path(tmp).name), _CliFakeBackend.log)
        checkpoints = result["transfer"]["checkpoints"]
        self.assertEqual(
            [c["fingerprint"] for c in checkpoints],
            ["w4-lazy-cli", "w4-lazy-cli"],
        )
        self.assertTrue(result["transfer"]["available"])

    def test_cancelled_run_constructs_no_transfer_backends(self):
        """Test cancelled run constructs no transfer backends."""
        with TemporaryDirectory() as tmp:
            code, result = self._dispatch(tmp, cancelled=True)
        self.assertEqual(code, 130)
        transfer_events = [e for e in _CliFakeBackend.log if e[1] in ("ckA", "ckB")]
        self.assertEqual(transfer_events, [])
        checkpoints = result["transfer"]["checkpoints"]
        self.assertEqual([c["fingerprint"] for c in checkpoints], [None, None])
        self.assertTrue(result["transfer"]["available"])


if __name__ == "__main__":
    unittest.main()
