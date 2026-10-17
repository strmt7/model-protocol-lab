"""Cycle 116: CPU budget for checkpoint-controlled Jinja renders.

``_render_with_memory_cap`` bounds memory bombs (RLIMIT_AS), but a hostile
``chat_template`` can burn CPU without allocating -- e.g.
``{% for i in range(10**12) %}{% set x = i %}{% endfor %}``. These tests pin
the SIGALRM wall-clock watchdog in ``_render_with_time_budget``: a CPU burn
aborts promptly with ``_RenderTimeBudgetExceeded`` (a ``ValueError``, so the
per-pair handler degrades to ``"unavailable"``), legitimate renders are
unaffected, and the previous SIGALRM handler/timer are always restored.
"""

from __future__ import annotations

import threading
import time
import unittest


class _BurnTokenizer:
    """Fake tokenizer whose render burns CPU without allocating memory."""

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds

    def apply_chat_template(
        self, conversation: object, tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        end = time.monotonic() + self._seconds
        while time.monotonic() < end:
            pass
        return "burned"


class _OkTokenizer:
    def apply_chat_template(
        self, conversation: object, tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        return "<|user|>hi<|assistant|>"


class TestRenderTimeBudget(unittest.TestCase):
    def test_cpu_burn_aborts_with_budget_exceeded(self) -> None:
        import signal

        from model_protocol_lab.doctor import _render_with_time_budget, _RenderTimeBudgetExceeded

        self.assertTrue(hasattr(signal, "setitimer"), "test requires setitimer (Unix)")
        start = time.monotonic()
        with self.assertRaises(_RenderTimeBudgetExceeded):
            _render_with_time_budget(_BurnTokenizer(60.0), [], budget_s=1.0)
        elapsed = time.monotonic() - start
        # Generous slack for loaded CI boxes; the point is ~1 s, not ~60 s.
        self.assertLess(elapsed, 20.0, f"budget did not abort the burn (took {elapsed:.1f}s)")

    def test_budget_exceeded_is_value_error_not_keyboard_interrupt(self) -> None:
        # The per-pair handler catches Exception; ValueError keeps a budget
        # expiry distinct from a user ^C.
        from model_protocol_lab.doctor import _RenderTimeBudgetExceeded

        self.assertTrue(issubclass(_RenderTimeBudgetExceeded, ValueError))
        self.assertFalse(issubclass(_RenderTimeBudgetExceeded, KeyboardInterrupt))

    def test_legitimate_render_unaffected(self) -> None:
        from model_protocol_lab.doctor import _render_with_time_budget

        self.assertEqual(
            _render_with_time_budget(
                _OkTokenizer(), [{"role": "user", "content": "hi"}], budget_s=5.0
            ),
            "<|user|>hi<|assistant|>",
        )

    def test_alarm_disarmed_after_render(self) -> None:
        # No stray SIGALRM may fire after the wrapper returns.
        import signal

        from model_protocol_lab.doctor import _render_with_time_budget

        _render_with_time_budget(_OkTokenizer(), [], budget_s=5.0)
        delay, _interval = signal.getitimer(signal.ITIMER_REAL)
        self.assertEqual(delay, 0.0)

    def test_previous_handler_and_timer_restored(self) -> None:
        import signal

        from model_protocol_lab.doctor import _render_with_time_budget

        def _handler(signum: int, frame: object) -> None:
            raise AssertionError("stray alarm fired")

        old = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _handler)
        try:
            _render_with_time_budget(_OkTokenizer(), [], budget_s=5.0)
            self.assertIs(signal.getsignal(signal.SIGALRM), _handler)
            old_delay, _ = signal.getitimer(signal.ITIMER_REAL)
            self.assertEqual(old_delay, 0.0)
        finally:
            signal.signal(signal.SIGALRM, old)

    def test_non_main_thread_runs_without_watchdog(self) -> None:
        # SIGALRM is delivered to the main thread only; off-main-thread the
        # wrapper must not arm a timer (it would never fire) and must still
        # return the render result.
        from model_protocol_lab.doctor import _render_with_time_budget

        results: list[str] = []

        def _work() -> None:
            results.append(_render_with_time_budget(_OkTokenizer(), [], budget_s=0.01))

        worker = threading.Thread(target=_work)
        worker.start()
        worker.join(timeout=30)
        self.assertFalse(worker.is_alive(), "worker thread hung")
        self.assertEqual(results, ["<|user|>hi<|assistant|>"])

    def test_memory_error_still_propagates_through_watchdog(self) -> None:
        # The time wrapper composes with the memory cap; it must not mask it.
        from model_protocol_lab.doctor import _render_with_time_budget

        class _MemBomb:
            def apply_chat_template(
                self,
                conversation: object,
                tokenize: bool = False,
                add_generation_prompt: bool = True,
            ) -> str:
                return "x" * 500_000_000

        with self.assertRaises(MemoryError):
            _render_with_time_budget(_MemBomb(), [], budget_s=30.0)

    def test_zero_budget_disables_watchdog(self) -> None:
        from model_protocol_lab.doctor import _render_with_time_budget

        self.assertEqual(
            _render_with_time_budget(_OkTokenizer(), [], budget_s=0),
            "<|user|>hi<|assistant|>",
        )

    def test_watchdog_setup_failure_falls_back_to_unbounded(self) -> None:
        # If the timer cannot be installed, the render must still run (the
        # pre-existing behavior), not fail.
        from unittest import mock

        from model_protocol_lab.doctor import _render_with_time_budget

        with mock.patch("signal.setitimer", side_effect=OSError("no timers")):
            self.assertEqual(
                _render_with_time_budget(_OkTokenizer(), [], budget_s=5.0),
                "<|user|>hi<|assistant|>",
            )


if __name__ == "__main__":
    unittest.main()
