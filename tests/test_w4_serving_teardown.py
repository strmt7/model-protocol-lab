"""W4 push 7 (RTI-05 + RTI-08): serving-backend teardown and
transport classification.

RTI-05: ``run_cache_sequence`` ran its end teardown -- the final
slot erase and the end identity re-read -- only on normal
completion. An aborted sequence (a raised arm, cancellation, or a
failing ``on_record`` persistence callback, the audit's executed
reproducer) left the probed server's slot holding cached prompt
state and skipped the identity recheck. The teardown now runs on
every exit from the arm loop; the aborting exception still
propagates.

RTI-08: ``LlamacppHttpBackend._call`` classified transport faults
from ``opener.open`` as ``_StackUnreachable`` but let faults from
reading the body escape -- and ``http.client.IncompleteRead`` is
an ``HTTPException``, not an ``OSError``, so a truncated success
body or error body propagated raw instead of joining the
transport classification every caller handles. Body-read faults
now classify identically, on both the success path and the
HTTPError path.
"""

import http.client
import io
import threading
import unittest
import urllib.error

from model_protocol_lab.servingbackends import (
    ExecutionCancelled,
    LlamacppHttpBackend,
    StackIdentity,
    StackObservation,
    _StackUnreachable,
    run_cache_sequence,
)


def _identity():
    """Identity."""
    return StackIdentity(
        stack="llamacpp",
        stack_version="b12182",
        checkpoint_digest="d",
        checkpoint_bytes=1,
        tokenizer_fingerprint="t",
        hardware_class="cpu",
        declared_eos_ids=(2,),
        config={"posture": {"slot_erase_available": True}},
        config_fingerprint="f",
    )


class FakeCacheBackend:
    """Duck-typed cache-sequence backend counting its teardown."""

    def __init__(
        self, *, fail_on_mode=None, teardown_erase_raises=False, teardown_verify_raises=False
    ):
        """Init."""
        self.identity = _identity()
        self.erase_calls = 0
        self.verify_calls = 0
        self.identity_ok = True
        self._fail_on_mode = fail_on_mode
        self._teardown_erase_raises = teardown_erase_raises
        self._teardown_verify_raises = teardown_verify_raises

    def verify_identity(self):
        """Verify identity."""
        self.verify_calls += 1
        if self.verify_calls == 2 and self._teardown_verify_raises:
            raise RuntimeError("teardown verify boom")
        return self.identity_ok

    def erase_slot(self):
        """Erase slot."""
        self.erase_calls += 1
        if self.erase_calls == 2 and self._teardown_erase_raises:
            raise RuntimeError("teardown erase boom")
        return True

    def execute(self, request, *, cancel=None):
        """Execute."""
        if self._fail_on_mode is not None and request.cache_mode == self._fail_on_mode:
            raise RuntimeError("arm boom")
        return StackObservation(
            returned_ids=(2,),
            finish_class="eos",
            stop_winner=None,
            cache_n_reported=0,
            tokens_cached_reported=0,
            tokens_evaluated_reported=4,
            returned_text_digest=None,
            server_detok_digest=None,
            config_echo_digest=None,
        )


def _run_sequence(backend, **kwargs):
    """Run sequence."""
    return run_cache_sequence(
        backend,
        cold_ids=(8, 12),
        shared_ids=(8, 12, 9),
        divergent_ids=(1, 2),
        seed=3,
        decode=lambda ids: "",
        **kwargs,
    )


class CacheSequenceTeardownTests(unittest.TestCase):
    """Test cache sequence teardown."""

    def test_normal_completion_runs_teardown_once_at_each_end(self):
        """Test normal completion runs teardown once at each end."""
        backend = FakeCacheBackend()
        sequence = _run_sequence(backend)
        self.assertTrue(sequence["available"])
        self.assertEqual(sequence["detail"], "")
        self.assertEqual(backend.erase_calls, 2)
        self.assertEqual(backend.verify_calls, 2)

    def test_failing_on_record_still_tears_down(self):
        """Test failing on record still tears down."""
        backend = FakeCacheBackend()
        seen = []

        def on_record(record):
            """On record."""
            seen.append(record)
            if len(seen) == 2:
                raise RuntimeError("persistence boom")

        with self.assertRaises(RuntimeError):
            _run_sequence(backend, on_record=on_record)
        self.assertEqual(backend.erase_calls, 2)
        self.assertEqual(backend.verify_calls, 2)

    def test_cancellation_still_tears_down(self):
        """Test cancellation still tears down."""
        backend = FakeCacheBackend()
        cancel = threading.Event()

        def on_record(record):
            """On record."""
            cancel.set()

        with self.assertRaises(ExecutionCancelled):
            _run_sequence(backend, on_record=on_record, cancel=cancel)
        self.assertEqual(backend.erase_calls, 2)
        self.assertEqual(backend.verify_calls, 2)

    def test_raised_arm_still_tears_down(self):
        """Test raised arm still tears down."""
        backend = FakeCacheBackend(fail_on_mode="shared_prefix")
        with self.assertRaises(RuntimeError):
            _run_sequence(backend)
        self.assertEqual(backend.erase_calls, 2)
        self.assertEqual(backend.verify_calls, 2)

    def test_identity_change_at_end_still_reported(self):
        """Test identity change at end still reported."""
        backend = FakeCacheBackend()

        original_execute = backend.execute

        def execute_then_drift(request, *, cancel=None):
            """Execute then drift."""
            backend.identity_ok = False
            return original_execute(request, cancel=cancel)

        backend.execute = execute_then_drift
        sequence = _run_sequence(backend)
        self.assertEqual(sequence["detail"], "stack_identity_changed_during_run")
        self.assertEqual(backend.erase_calls, 2)
        self.assertEqual(backend.verify_calls, 2)

    def test_raising_teardown_erase_does_not_mask_arm_error(self):
        """Test raising teardown erase does not mask arm error (BH12-5)."""
        backend = FakeCacheBackend(fail_on_mode="shared_prefix", teardown_erase_raises=True)
        with self.assertRaises(RuntimeError) as caught:
            _run_sequence(backend)
        self.assertEqual(str(caught.exception), "arm boom")
        self.assertEqual(backend.erase_calls, 2)
        self.assertEqual(backend.verify_calls, 2)

    def test_raising_teardown_erase_on_clean_run_returns_sequence(self):
        """Test raising teardown erase on clean run returns sequence (BH12-5)."""
        backend = FakeCacheBackend(teardown_erase_raises=True)
        sequence = _run_sequence(backend)
        self.assertTrue(sequence["available"])
        self.assertEqual(backend.erase_calls, 2)
        self.assertEqual(backend.verify_calls, 2)

    def test_raising_teardown_verify_reports_identity_changed(self):
        """Test raising teardown verify reports identity changed (BH12-5)."""
        backend = FakeCacheBackend(teardown_verify_raises=True)
        sequence = _run_sequence(backend)
        self.assertEqual(sequence["detail"], "stack_identity_changed_during_run")
        self.assertEqual(backend.erase_calls, 2)
        self.assertEqual(backend.verify_calls, 2)


class _FakeResponse:
    """Minimal response context manager with a scriptable read."""

    status = 200

    def __init__(self, payload=b"{}", *, read_error=None):
        """Init."""
        self._payload = payload
        self._read_error = read_error

    def __enter__(self):
        """Enter."""
        return self

    def __exit__(self, *exc):
        """Exit."""
        return False

    def read(self, _limit):
        """Read."""
        if self._read_error is not None:
            raise self._read_error
        return self._payload


class _FakeOpener:
    """Opener whose open() runs a scripted behavior."""

    def __init__(self, behavior):
        """Init."""
        self._behavior = behavior

    def open(self, request, timeout=None):
        """Open."""
        return self._behavior()


class _FaultyBody:
    """File-like whose read raises the scripted fault."""

    def __init__(self, error):
        """Init."""
        self._error = error

    def read(self, _limit=-1):
        """Read."""
        raise self._error


def _bare_backend(opener):
    """Bare backend."""
    backend = LlamacppHttpBackend.__new__(LlamacppHttpBackend)
    backend._base = "http://127.0.0.1:1"
    backend._api_key = "k"
    backend._timeout_s = 1.0
    backend._opener = opener
    return backend


def _incomplete():
    """Incomplete."""
    return http.client.IncompleteRead(b"partial", 10)


class BodyReadClassificationTests(unittest.TestCase):
    """Test body read classification."""

    def test_success_body_incomplete_read_is_unreachable(self):
        """Test success body incomplete read is unreachable."""
        backend = _bare_backend(_FakeOpener(lambda: _FakeResponse(read_error=_incomplete())))
        with self.assertRaises(_StackUnreachable):
            backend._call("GET", "/health", None, authenticated=False)

    def test_error_body_incomplete_read_is_unreachable(self):
        """Test error body incomplete read is unreachable."""

        def behavior():
            """Behavior."""
            raise urllib.error.HTTPError(
                "http://127.0.0.1:1/x", 503, "unavailable", {}, _FaultyBody(_incomplete())
            )

        backend = _bare_backend(_FakeOpener(behavior))
        with self.assertRaises(_StackUnreachable):
            backend._call("GET", "/x", None)

    def test_success_body_remote_disconnected_is_unreachable(self):
        """Test success body remote disconnected is unreachable."""
        backend = _bare_backend(
            _FakeOpener(lambda: _FakeResponse(read_error=http.client.RemoteDisconnected("gone")))
        )
        with self.assertRaises(_StackUnreachable):
            backend._call("GET", "/health", None, authenticated=False)

    def test_normal_and_error_bodies_still_parse(self):
        """Test normal and error bodies still parse."""
        backend = _bare_backend(_FakeOpener(lambda: _FakeResponse(b'{"status": "ok"}')))
        self.assertEqual(
            backend._call("GET", "/health", None, authenticated=False),
            (200, {"status": "ok"}),
        )

        def behavior():
            """Behavior."""
            raise urllib.error.HTTPError(
                "http://127.0.0.1:1/x",
                503,
                "unavailable",
                {},
                io.BytesIO(b'{"error": "busy"}'),
            )

        backend = _bare_backend(_FakeOpener(behavior))
        self.assertEqual(backend._call("GET", "/x", None), (503, {"error": "busy"}))

    def test_open_failure_still_unreachable(self):
        """Test open failure still unreachable."""

        def behavior():
            """Behavior."""
            raise urllib.error.URLError("refused")

        backend = _bare_backend(_FakeOpener(behavior))
        with self.assertRaises(_StackUnreachable):
            backend._call("GET", "/x", None)


if __name__ == "__main__":
    unittest.main()
