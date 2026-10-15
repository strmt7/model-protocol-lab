"""Cycle 14: differential verification of ``emulate_sglang_stop``.

Compares ``backends.emulate_sglang_stop`` against a line-faithful port of
SGLang's own algorithm -- ``Req._check_str_based_finish`` in
``python/sglang/srt/managers/schedule_batch.py`` (1865-1892) and
``DetokenizerManager.trim_matched_stop`` in
``python/sglang/srt/managers/detokenizer_manager.py`` (190-221), both
fetched 2026-10-15 -- over adversarial inputs plus a seeded fuzz sweep.

The fake backend below is purely concatenative --
``decode(ids[:k+1]) == decode(ids[:k]) + text(ids[k])`` -- so the emulation's
per-prefix batch decodes coincide exactly with upstream's incremental
``decoded_text`` accumulation on these inputs. This isolates the stop
algorithm (per-step search, leftmost trim, keep/strip, first-listed
multi-stop selection) from the documented full-prefix-decode approximation
of upstream's incremental detokenizer (which additionally holds back
incomplete UTF-8 behind its "�" recovery); that approximation is out of
scope here, exactly as in ``test_cycle14_holdback.py``.

Verified properties (each pinned by at least one case below):
- per-step search of the decoded output text; the first step whose decoded
  prefix contains the stop string fires (upstream's ``stop_str in tail_str
  or stop_str in decoded_text`` reduces to the full-text check under a
  concatenative decode: the tail is a suffix of the decoded text, so the
  window size provably cannot change the firing decision);
- ``trim_matched_stop`` cuts at the LEFTMOST occurrence
  (``output.find(matched)``), both in strip mode (``no_stop_trim=False``,
  the ``SamplingParams.no_stop_trim`` default) and in keep mode
  (``no_stop_trim=True`` returns through the stop *end*, not the text end);
- multi-stop: first-listed wins within one check; ``diffbackends``' caller
  rule -- min over ``(firing_step, list_index)`` -- is proved equivalent to
  upstream's per-step list-order loop and pinned against the reference run
  with the full stop list, including the research-log caveat that with
  1 token/step the earlier-completing stop still fires at its own step;
- ``truncated_ids`` keeps the ids through the firing step, matching
  upstream ``finished_len`` at 1 token/step (``_locate_str_stop_finished_len``
  returns ``len(output_ids)`` when ``new_accepted_len == 1``); the trim is
  text-level upstream, and the emulation agrees even on the
  ``finished_len is None`` path (stop outside the token-count tail window
  via empty-decoding tokens) because the returned text never depends on it;
- edge cases: stop at position 0 (empty stripped text), empty output,
  unicode (code-point ``find`` on both sides), overlapping occurrences
  (leftmost), stop longer than the output, stop absent, empty-decoding
  tokens, newline stops;
- invalid inputs collapse to ``invalid_emulation_input`` (empty/non-str
  stop, non-bool ``no_stop_trim``, bad ids); a non-str decode collapses to
  ``decode_unavailable``. Upstream would actually fire on an *empty* stop
  string (``"" in tail_str``), but the lab harness rejects empty stops in
  ``_checked_stop_texts`` before any emulation runs, so the guard is the
  correct unreachable-input contract.

Verified by source read (pinned here, not runnable offline): stop string
beats EOS in the same step (``update_finish_state`` runs the str-based
check before the token-based check -- the shipped SGLang #28802 fix), and
there is no ``min_tokens`` gating on the finish path (``min_new_tokens``
only clips the length budget).
"""

from __future__ import annotations

import random
import unittest
from threading import Event

from model_protocol_lab.backends import StopOutcome, emulate_sglang_stop
from model_protocol_lab.core import Backend, Generation

# ---------------------------------------------------------------------------
# Reference port of SGLang main (fetched 2026-10-15):
#   python/sglang/srt/managers/schedule_batch.py :: Req._check_str_based_finish
#   python/sglang/srt/managers/detokenizer_manager.py :: trim_matched_stop
# ---------------------------------------------------------------------------


def _upstream_check_str_based_finish(
    decoded_text: str, tail_str: str, stop_strs: list[str]
) -> str | None:
    """Port of ``Req._check_str_based_finish``, stop-strings half.

    Upstream (schedule_batch.py:1865-1892)::

        for stop_str in self.sampling_params.stop_strs:
            stop_str_in_tail = stop_str in tail_str
            if stop_str_in_tail or stop_str in self.decoded_text:
                self.finished_reason = FINISH_MATCHED_STR(matched=stop_str)
                ...
                return True

    Returns the matched stop string -- first-listed wins -- or None.
    ``stop_regex`` is out of scope (the lab's USAGE note); ``min_new_tokens``
    is never consulted on this path upstream.
    """
    if stop_strs:
        for stop_str in stop_strs:
            stop_str_in_tail = stop_str in tail_str
            if stop_str_in_tail or stop_str in decoded_text:
                return stop_str
    return None


def _upstream_trim_matched_stop(output: str, matched: str | None, no_stop_trim: bool) -> str:
    """Port of ``DetokenizerManager.trim_matched_stop``, str branch.

    Upstream (detokenizer_manager.py:190-221)::

        matched = finished_reason.get("matched", None)
        if matched is None:
            return output
        # TODO(lmzheng): handle the case where multiple stop strs are hit
        if isinstance(matched, str) and isinstance(output, str):
            pos = output.find(matched)
            if pos == -1:
                return output
            end = pos + len(matched)
            return output[:end] if no_stop_trim else output[:pos]

    ``matched=None`` mirrors an absent/unset finished reason; the ``-1``
    guard is unreachable on the fire path (a matched stop is in the text).
    """
    if matched is None:
        return output
    pos = output.find(matched)
    if pos == -1:
        return output
    end = pos + len(matched)
    return output[:end] if no_stop_trim else output[:pos]


def _upstream_request_lifecycle(
    token_texts: list[str], stop_strs: list[str], no_stop_trim: bool
) -> tuple[bool, int | None, str | None, str | None]:
    """Simulate one SGLang request's str-based finish path at 1 token/step.

    Incremental ``decoded_text`` accumulation; each step runs the str-based
    check (upstream ``update_finish_state`` orders it before the EOS/token
    check); on fire the detokenizer materializes the full decoded text and
    trims the matched stop. The tail window mirrors
    ``_stop_match_tail_len`` (``max(stop_str_max_len, stop_regex_max_len)+1``
    tokens); char length stands in for the fake backend's missing tokenizer.
    With a concatenative decode the tail is a suffix of ``decoded_text``, so
    ``stop in tail`` implies ``stop in decoded_text`` and the window size
    provably cannot change the firing decision -- the two-branch check keeps
    upstream's shape anyway.

    Returns ``(fired, firing_step_1based, returned_text, matched_stop)``;
    ``returned_text``/``matched_stop`` are None when nothing fired.
    """
    decoded_text = ""
    tail_tokens = max([len(stop) for stop in stop_strs] + [0]) + 1
    for step, piece in enumerate(token_texts, start=1):
        decoded_text += piece
        tail_str = "".join(token_texts[max(0, step - tail_tokens) : step])
        matched = _upstream_check_str_based_finish(decoded_text, tail_str, stop_strs)
        if matched is not None:
            # Finished: materialize once, trim the matched stop, emit the tail
            # (detokenizer_manager.py:413-416).
            return (
                True,
                step,
                _upstream_trim_matched_stop(decoded_text, matched, no_stop_trim),
                matched,
            )
    return False, None, None, None


# ---------------------------------------------------------------------------
# Concatenative fake backend: decode(prefix) == incremental accumulation.
# ---------------------------------------------------------------------------


class ConcatBackend:
    """Backend whose decode is exactly incremental concatenation.

    ``decode(ids[:k+1]) == decode(ids[:k]) + piece(ids[k])``, so the
    emulation's per-prefix batch decodes coincide exactly with upstream's
    incremental ``decoded_text`` accumulation on these inputs.
    """

    fingerprint = "cycle14-sglang-concat-backend-not-a-model"

    def __init__(self, texts: dict[int, str]) -> None:
        self._texts = dict(texts)

    def encode(self, text: str) -> tuple[int, ...]:
        raise AssertionError("encode unused in sglang differential tests")

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(self._texts[token] for token in token_ids)

    def generate(
        self,
        input_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        seed: int,
        cancel: Event,
    ) -> Generation:
        raise AssertionError("generate unused in sglang differential tests")


def _as_backend(backend: ConcatBackend) -> Backend:
    return backend  # structural check against the Backend protocol


class _NonStrDecodeBackend(ConcatBackend):
    """Decode returns None: pins the emulation's decode_unavailable path."""

    fingerprint = "cycle14-sglang-nonstr-decode-not-a-model"

    def decode(self, token_ids: tuple[int, ...]) -> str:  # type: ignore[override]
        return None  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Adversarial cases.
# ---------------------------------------------------------------------------

# (name, token_texts, stop, no_stop_trim)
_ADVERSARIAL_SINGLE: tuple[tuple[str, list[str], str, bool], ...] = (
    ("stop_at_position_zero_strip", ["<stop>", "rest"], "<stop>", False),
    ("stop_at_position_zero_keep", ["<stop>", "rest"], "<stop>", True),
    ("spans_two_tokens_strip", ["ab", "cd"], "bc", False),
    ("spans_two_tokens_keep", ["ab", "cd"], "bc", True),
    ("spans_three_tokens", ["a", "b", "c"], "abc", False),
    ("stop_is_whole_token", ["x", "STOP", "y"], "STOP", False),
    ("stop_longer_than_output", ["ab"], "abcdef", False),
    ("empty_output", [], "STOP", False),
    ("leftmost_occurrence_wins", ["xSTOP", "STOP"], "STOP", False),
    ("repeat_stop_strips_first", ["STOP", "x", "STOP"], "STOP", False),
    ("unicode_stop", ["hé", "llo ", "wörld"], "wörld", False),
    ("unicode_partial_stop", ["hé", "llo"], "éll", False),
    ("unicode_keep", ["héllo"], "éll", True),
    ("overlapping_stop", ["aa", "aa"], "aa", False),
    ("overlapping_mid_text", ["xaaa"], "aa", False),
    ("keep_with_trailing_text", ["a<stop>b"], "<stop>", True),
    ("strip_with_trailing_text", ["a<stop>b"], "<stop>", False),
    ("keep_stop_at_text_end", ["a<stop>"], "<stop>", True),
    ("single_char_stop", ["x", "y", "!"], "!", False),
    ("stop_absent", ["a", "b"], "zzz", False),
    ("partial_then_full", ["STO", "X", "STOP"], "STOP", False),
    ("empty_decoding_token_before_stop", ["a", "", "STOP"], "STOP", False),
    ("stop_spanning_empty_token", ["x", "a", "", "b"], "ab", False),
    ("newline_stop", ["line1\n", "line2"], "\n", False),
    ("multi_char_stop_single_token_keep", ["xxSTOPyy"], "STOP", True),
    ("stop_is_token_suffix", ["abSTOP"], "STOP", False),
    ("single_char_tokens", ["S", "T", "O", "P"], "STOP", False),
    ("stop_fires_last_step", ["a", "b", "STOP"], "STOP", False),
)

# (name, token_texts, stops, no_stop_trim)
_ADVERSARIAL_MULTI: tuple[tuple[str, list[str], list[str], bool], ...] = (
    # Same step: first-listed wins.
    ("first_listed_wins_same_step", ["ab"], ["b", "ab"], False),
    ("first_listed_wins_same_step_keep", ["ab"], ["b", "ab"], True),
    # 1 token/step caveat: the earlier-completing stop fires at its own step
    # even when listed second.
    ("earlier_completing_wins_despite_list_order", ["Xab", "c"], ["abc", "ab"], False),
    ("later_listed_earlier_fire", ["ab", "c"], ["bc", "ab"], False),
    ("first_listed_fires_second", ["XY", "a"], ["a", "XY"], False),
    ("second_listed_fires_first", ["a", "XY"], ["XY", "a"], False),
    ("no_stop_fires", ["a", "b"], ["zzz", "qqq"], False),
    ("three_stops_middle_wins", ["x", "M", "y"], ["zzz", "M", "aaa"], True),
    ("three_stops_first_listed_tie", ["ab"], ["ab", "b", "a"], False),
)


# ---------------------------------------------------------------------------
# Differential harness.
# ---------------------------------------------------------------------------


class SglangStopDifferentialTests(unittest.TestCase):
    def _backend(self, texts: list[str]) -> tuple[Backend, tuple[int, ...]]:
        backend = _as_backend(ConcatBackend({i + 1: text for i, text in enumerate(texts)}))
        return backend, tuple(range(1, len(texts) + 1))

    def check_single(self, name: str, texts: list[str], stop: str, no_stop_trim: bool) -> None:
        backend, ids = self._backend(texts)
        outcome = emulate_sglang_stop(backend, ids, stop, no_stop_trim=no_stop_trim)
        fired, step, returned, matched = _upstream_request_lifecycle(texts, [stop], no_stop_trim)
        self.assertTrue(outcome.available, name)
        self.assertEqual(outcome.fired, fired, name)
        if fired:
            self.assertIsNotNone(step, name)
            self.assertEqual(matched, stop, name)
            self.assertEqual(outcome.detail, f"fired_at_step_{step}", name)
            # Upstream finished_len at 1 token/step: _locate_str_stop_finished_len
            # returns len(output_ids) (its search range is empty when
            # new_accepted_len == 1), i.e. the ids through the firing step.
            self.assertEqual(outcome.truncated_ids, ids[:step], name)
            self.assertEqual(outcome.returned_text, returned, name)
        else:
            self.assertEqual(outcome.detail, "no_stop_within_cap", name)
            self.assertEqual(outcome.truncated_ids, ids, name)
            self.assertIsNone(outcome.returned_text, name)

    def check_multi(
        self, name: str, texts: list[str], stops: list[str], no_stop_trim: bool
    ) -> None:
        self.assertEqual(len(set(stops)), len(stops), f"{name}: stops must be unique")
        backend, ids = self._backend(texts)
        # Reference: one upstream request carrying the full stop list --
        # first-listed wins within each per-step check.
        ref_fired, ref_step, ref_text, ref_matched = _upstream_request_lifecycle(
            texts, stops, no_stop_trim
        )
        # Lab composition: per-stop cells, each pinned against its own
        # single-stop reference, combined by diffbackends' SGLang rule --
        # min over (firing_step, list_index). Proved equivalent to upstream's
        # per-step list-order loop (see module docstring).
        pick: tuple[tuple[int, int], str, str | None] | None = None
        for list_index, stop in enumerate(stops):
            outcome = emulate_sglang_stop(backend, ids, stop, no_stop_trim=no_stop_trim)
            s_fired, s_step, s_text, _ = _upstream_request_lifecycle(texts, [stop], no_stop_trim)
            self.assertTrue(outcome.available, f"{name}/{stop}")
            self.assertEqual(outcome.fired, s_fired, f"{name}/{stop}")
            if s_fired:
                assert s_step is not None
                self.assertEqual(outcome.detail, f"fired_at_step_{s_step}", f"{name}/{stop}")
                self.assertEqual(outcome.truncated_ids, ids[:s_step], f"{name}/{stop}")
                self.assertEqual(outcome.returned_text, s_text, f"{name}/{stop}")
                key = (s_step, list_index)
                if pick is None or key < pick[0]:
                    pick = (key, stop, outcome.returned_text)
        if ref_fired:
            self.assertIsNotNone(pick, name)
            assert pick is not None
            self.assertEqual(pick[1], ref_matched, name)
            self.assertEqual(pick[0][0], ref_step, name)
            self.assertEqual(pick[2], ref_text, name)
        else:
            self.assertIsNone(pick, name)

    def test_adversarial_single_matches_upstream(self) -> None:
        for name, texts, stop, no_stop_trim in _ADVERSARIAL_SINGLE:
            with self.subTest(name=name):
                self.check_single(name, texts, stop, no_stop_trim)

    def test_adversarial_multi_matches_upstream(self) -> None:
        for name, texts, stops, no_stop_trim in _ADVERSARIAL_MULTI:
            with self.subTest(name=name):
                self.check_multi(name, texts, stops, no_stop_trim)

    def test_invalid_inputs(self) -> None:
        backend, ids = self._backend(["a", "STOP"])

        def check_invalid(name: str, outcome: object) -> None:
            assert isinstance(outcome, StopOutcome)
            self.assertFalse(outcome.available, name)
            self.assertFalse(outcome.fired, name)
            self.assertEqual(outcome.detail, "invalid_emulation_input", name)

        check_invalid("empty_stop", emulate_sglang_stop(backend, ids, ""))
        check_invalid(
            "non_str_stop",
            emulate_sglang_stop(backend, ids, 123),  # type: ignore[arg-type]
        )
        check_invalid(
            "non_bool_no_stop_trim",
            emulate_sglang_stop(backend, ids, "STOP", no_stop_trim="yes"),  # type: ignore[arg-type]
        )
        check_invalid("negative_id", emulate_sglang_stop(backend, (-1, 2), "STOP"))
        check_invalid(
            "non_tuple_ids",
            emulate_sglang_stop(backend, "ab", "STOP"),  # type: ignore[arg-type]
        )

    def test_non_str_decode_unavailable(self) -> None:
        backend = _as_backend(_NonStrDecodeBackend({1: "a", 2: "STOP"}))
        outcome = emulate_sglang_stop(backend, (1, 2), "STOP")
        self.assertFalse(outcome.available)
        self.assertFalse(outcome.fired)
        self.assertEqual(outcome.detail, "decode_unavailable")
        self.assertEqual(outcome.truncated_ids, (1, 2))

    def test_fuzz_single_matches_upstream_reference(self) -> None:
        rng = random.Random(20261015)  # noqa: S311 -- seeded deterministic fuzz, not crypto
        alphabet = [
            "a",
            "b",
            "STOP",
            "<stop>",
            "",
            "x",
            "ab",
            "é",
            "STOPSTOP",
            " ",
            "\n",
            "ü",
            "xab",
            "b",
            "aa",
        ]
        stops = [
            "STOP",
            "ab",
            "a",
            "<stop>",
            "é",
            "STOPSTOP",
            "bST",
            "xyz",
            "ü",
            "\n",
            "aa",
            "aab",
            " ",
        ]
        for trial in range(500):
            texts = [rng.choice(alphabet) for _ in range(rng.randint(0, 6))]
            stop = rng.choice(stops)
            no_stop_trim = rng.choice([False, True])
            with self.subTest(trial=trial):
                self.check_single(f"fuzz-{trial}", texts, stop, no_stop_trim)

    def test_fuzz_multi_matches_upstream_reference(self) -> None:
        rng = random.Random(20261016)  # noqa: S311 -- seeded deterministic fuzz, not crypto
        alphabet = [
            "a",
            "b",
            "STOP",
            "<stop>",
            "",
            "x",
            "ab",
            "é",
            "M",
            " ",
            "\n",
            "ü",
            "xy",
            "STOPSTOP",
        ]
        stops = [
            "STOP",
            "ab",
            "a",
            "<stop>",
            "é",
            "M",
            "bST",
            "xyz",
            "ü",
            "\n",
            "xy",
            " ",
        ]
        for trial in range(300):
            texts = [rng.choice(alphabet) for _ in range(rng.randint(0, 6))]
            stop_list = rng.sample(stops, rng.randint(1, 3))
            no_stop_trim = rng.choice([False, True])
            with self.subTest(trial=trial):
                self.check_multi(f"fuzz-multi-{trial}", texts, stop_list, no_stop_trim)


if __name__ == "__main__":
    unittest.main()
