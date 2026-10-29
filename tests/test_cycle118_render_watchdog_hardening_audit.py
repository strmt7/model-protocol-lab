"""Cycle 118: hardening audit of the render watchdog (cycle 116).

``_render_with_time_budget`` bounds checkpoint-controlled Jinja CPU burns
with a SIGALRM watchdog. Red-team review found two robustness gaps:

1. A non-finite ``budget_s`` (``float("inf")``) reached ``setitimer``,
   which raises ``OverflowError`` -- not covered by the ``except
   (OSError, ValueError)`` fallback, so the defensive "fall back to the
   unbounded render" contract was violated. Non-finite budgets now disable
   the watchdog, like non-positive ones.
2. Teardown disarmed the timer but did not block ``SIGALRM``: an alarm
   that became pending in that window would fire mid-restore, and because
   ``_RenderTimeBudgetExceeded`` is a ``ValueError`` the cleanup
   ``contextlib.suppress`` guards would swallow it, leaving the previous
   handler/timer inconsistent. Teardown now blocks ``SIGALRM`` across the
   restore (``pthread_sigmask``); a pending alarm is then delivered to the
   restored previous handler after unblocking -- honest Unix semantics.

These tests pin both fixes.
"""

from __future__ import annotations

import unittest


class _OkTokenizer:
    @staticmethod
    def apply_chat_template(
        conversation: object, tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        return "<|user|>hi<|assistant|>"


class TestWatchdogHardening(unittest.TestCase):
    def test_infinite_budget_disables_watchdog(self) -> None:
        # setitimer(inf) raises OverflowError, which the old except clause
        # did not catch. inf now means "no budget", like budget_s <= 0.
        from model_protocol_lab.doctor import _render_with_time_budget

        self.assertEqual(
            _render_with_time_budget(
                _OkTokenizer(), [{"role": "user", "content": "hi"}], budget_s=float("inf")
            ),
            "<|user|>hi<|assistant|>",
        )

    def test_nan_budget_disables_watchdog(self) -> None:
        # nan cannot arm setitimer either; it must not raise.
        from model_protocol_lab.doctor import _render_with_time_budget

        self.assertEqual(
            _render_with_time_budget(
                _OkTokenizer(), [{"role": "user", "content": "hi"}], budget_s=float("nan")
            ),
            "<|user|>hi<|assistant|>",
        )

    def test_signal_mask_not_leaked(self) -> None:
        # The pthread_sigmask used across teardown must be restored; the
        # process must not leave SIGALRM blocked.
        import signal

        from model_protocol_lab.doctor import _render_with_time_budget

        if not hasattr(signal, "pthread_sigmask"):
            self.skipTest("requires pthread_sigmask (Unix)")
        before = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        _render_with_time_budget(_OkTokenizer(), [], budget_s=5.0)
        after = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        self.assertEqual(before, after, "signal mask changed by the watchdog")
        self.assertNotIn(signal.SIGALRM, after, "SIGALRM left blocked after teardown")

    def test_stale_alarm_mid_render_restores_handler(self) -> None:
        # A SIGALRM that becomes pending during the render (e.g. a stale
        # alarm, not the watchdog's own) is delivered to the watchdog
        # handler and must not corrupt the handler restoration.
        import os
        import signal

        from model_protocol_lab.doctor import _render_with_time_budget, _RenderTimeBudgetExceeded

        if not hasattr(signal, "setitimer"):
            self.skipTest("requires setitimer (Unix)")

        def old_handler(signum: int, frame: object) -> None:
            """No-op stand-in for the previously installed SIGALRM handler."""

        prev = signal.signal(signal.SIGALRM, old_handler)
        try:

            class StaleAlarmTokenizer:
                @staticmethod
                def apply_chat_template(
                    conversation: object,
                    tokenize: bool = False,
                    add_generation_prompt: bool = True,
                ) -> str:
                    os.kill(os.getpid(), signal.SIGALRM)
                    return "ok"

            with self.assertRaises(_RenderTimeBudgetExceeded):
                _render_with_time_budget(StaleAlarmTokenizer(), [], budget_s=60.0)
            self.assertIs(
                signal.getsignal(signal.SIGALRM),
                old_handler,
                "previous SIGALRM handler not restored after mid-render alarm",
            )
        finally:
            signal.signal(signal.SIGALRM, prev)


if __name__ == "__main__":
    unittest.main()
