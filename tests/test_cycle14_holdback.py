"""Cycle 14: differential verification of ``emulate_vllm_holdback``.

Compares ``backends.emulate_vllm_holdback`` against a line-faithful port of
vLLM's own algorithm -- ``BaseIncrementalDetokenizer.update`` and the
module-level ``check_stop_strings`` in ``vllm/v1/engine/detokenizer.py``
(fetched 2026-10-15) -- over adversarial inputs plus a seeded fuzz sweep.

The reference port covers one stop string with one token per update, which
is exactly the lab's replay discipline; upstream's multi-token-per-update
path exists for speculative decoding and is specified (in the upstream
docstring) to match one-token-at-a-time appending, so the single-step port
*is* the reference semantics.

The fake backend below is purely concatenative --
``decode(ids[:k+1]) == decode(ids[:k]) + text(ids[k])`` -- so the emulation's
per-prefix batch decodes coincide exactly with vLLM's incremental
``DecodeStream`` accumulation. This isolates the stop algorithm (find
window, ``min_tokens`` gating, truncation, strip/keep) from the documented
DecodeStream-vs-batch-decode approximation, which is out of scope here.

Verified properties (each pinned by at least one case below):
- find window ``text.find(stop, 1 - new_char_count - len(stop))`` per step,
  including the negative-start "already-searched text is excluded" behavior;
- stop spanning token boundaries (2 and 3 tokens) fires at the completing step;
- stop at position 0 fires at step 1 with empty stripped text;
- ``include_stop_str_in_output=False`` strips to the stop start while the
  completing token stays in ``truncated_ids`` (token_ids/text overshoot);
- ``include_stop_str_in_output=True`` keeps through the stop *end*, including
  trailing text decoded in the same step and the no-truncation
  (``truncate_to == -1``) path when the stop ends the text;
- ``stop_buffer_length`` holdback does not touch the final text: the
  emulation's ``returned_text`` matches vLLM's finished-request output
  (upstream only withholds ``max(len(s)) - 1`` chars from *unfinished*
  streamed deltas);
- ``min_tokens`` gating: no check while ``num_output_tokens <= min_tokens``;
  a stop fully completed during the gate is NOT refound (the tail-only
  window at the first ungated step misses it); a stop spanning the gate
  boundary fires;
- ``new_char_count == 0`` steps (empty-decoding token): upstream's early
  return and the emulation's window provably agree -- ``find`` with start
  ``1 - len(stop)`` can never match a non-empty stop;
- empty output, single-char stops, overlapping occurrences, unicode
  (code-point lengths on both sides), stop longer than the output.
"""

from __future__ import annotations

import random
import unittest
from bisect import bisect_left
from threading import Event

from model_protocol_lab.backends import emulate_vllm_holdback
from model_protocol_lab.core import Backend, Generation

# ---------------------------------------------------------------------------
# Reference port of vLLM main v1/engine/detokenizer.py (fetched 2026-10-15).
# ---------------------------------------------------------------------------


def _upstream_check_stop_strings(
    output_text: str, new_char_count: int, stop_str: str, include_in_output: bool
) -> tuple[str, int] | None:
    """Port of vLLM's module-level ``check_stop_strings`` (one stop string).

    Returns ``(stop_string, offset)`` where offset is the length to truncate
    the output text to, or -1 for no truncation; None when no stop matched.
    """
    if not new_char_count or not stop_str:
        return None
    stop_string_len = len(stop_str)
    # "Avoid searching already-searched text."
    stop_index = output_text.find(stop_str, 1 - new_char_count - stop_string_len)
    if stop_index == -1:
        return None
    # "Prefer the stop string that completes earliest in the text."
    # (Trivial with a single stop string.)
    end = stop_index + stop_string_len
    if include_in_output:
        # "Truncate to end of stop string."
        if end >= len(output_text):
            # "No truncation required."
            return stop_str, -1
        return stop_str, end
    # "Truncate the output text to the beginning of the stop string."
    return stop_str, stop_index


def _upstream_update(
    token_texts: list[str], stop: str, include_in_output: bool, min_tokens: int
) -> tuple[bool, int | None, tuple[int, ...], str | None, bool]:
    """Port of ``BaseIncrementalDetokenizer.update``: one token per update.

    ``stop_terminated`` is False -- the lab replays plain recorded outputs,
    and upstream's skipped-stop-token path only applies to stop-token-id
    termination, which the lab's "exactly N tokens" discipline never uses.

    Returns ``(fired, firing_step_1based, kept_token_ids, final_text,
    any_check_ran)``; ``final_text`` is None when nothing fired.
    """
    output_text = ""
    token_ids: list[int] = []
    any_check = False
    for position, piece in enumerate(token_texts, start=1):
        num_prev_tokens = position - 1
        first_new_token_index = len(token_ids)
        # "Output text offsets at the boundaries of the new tokens."
        token_offsets = [len(output_text)]
        token_ids.append(position)
        output_text += piece
        token_offsets.append(len(output_text))
        # "Support min_tokens, see https://github.com/vllm-project/vllm/pull/22014"
        if position > min_tokens:
            any_check = True
            stop_check_offset = token_offsets[max(min_tokens - num_prev_tokens, 0)]
            found = _upstream_check_stop_strings(
                output_text,
                len(output_text) - stop_check_offset,
                stop,
                include_in_output,
            )
            if found is not None:
                stop_string, truncate_to = found
                stop_end = len(output_text) if truncate_to == -1 else truncate_to
                if not include_in_output:
                    stop_end += len(stop_string)
                # "Drop tokens generated after the one completing the stop
                # string (possible when multiple tokens are appended in one
                # step)."
                num_kept = bisect_left(token_offsets, stop_end, lo=1)
                del token_ids[first_new_token_index + num_kept :]
                if truncate_to != -1:
                    output_text = output_text[:truncate_to]
                return True, position, tuple(token_ids), output_text, any_check
    return False, None, tuple(token_ids), None, any_check


# ---------------------------------------------------------------------------
# Concatenative fake backend: decode(prefix) == incremental accumulation.
# ---------------------------------------------------------------------------


class ConcatBackend:
    """Backend whose decode is exactly incremental concatenation.

    ``decode(ids[:k+1]) == decode(ids[:k]) + piece(ids[k])``, so the
    emulation's per-prefix batch decodes coincide exactly with vLLM's
    incremental ``DecodeStream`` accumulation on these inputs.
    """

    fingerprint = "cycle14-concat-backend-not-a-model"

    def __init__(self, texts: dict[int, str]) -> None:
        self._texts = dict(texts)

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        """Encode."""
        raise AssertionError("encode unused in holdback differential tests")

    def decode(self, token_ids: tuple[int, ...]) -> str:
        """Decode."""
        return "".join(self._texts[token] for token in token_ids)

    @staticmethod
    def generate(
        input_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        seed: int,
        cancel: Event,
    ) -> Generation:
        """Generate."""
        raise AssertionError("generate unused in holdback differential tests")


def _as_backend(backend: ConcatBackend) -> Backend:
    """As backend."""
    return backend  # structural check against the Backend protocol


# ---------------------------------------------------------------------------
# Differential harness.
# ---------------------------------------------------------------------------

# (name, token_texts, stop, include_stop_str_in_output, min_tokens)
_ADVERSARIAL_CASES: tuple[tuple[str, list[str], str, bool, int], ...] = (
    ("stop_at_position_zero_strip", ["<stop>", "rest"], "<stop>", False, 0),
    ("stop_at_position_zero_keep", ["<stop>", "rest"], "<stop>", True, 0),
    ("spans_two_tokens", ["ab", "cd"], "bc", False, 0),
    ("spans_two_tokens_keep", ["ab", "cd"], "bc", True, 0),
    ("spans_three_tokens", ["a", "b", "c"], "abc", False, 0),
    ("stop_is_whole_token", ["x", "STOP", "y"], "STOP", False, 0),
    ("stop_longer_than_output", ["ab"], "abcdef", False, 0),
    ("empty_output", [], "STOP", False, 0),
    ("empty_output_gated", [], "STOP", False, 2),
    ("first_occurrence_wins", ["STOP", "x", "STOP"], "STOP", False, 0),
    ("gate_all_steps", ["S", "T", "O", "P"], "STOP", False, 4),
    ("gate_then_fire", ["a", "STOP"], "STOP", False, 1),
    ("completed_during_gate_not_refound", ["ST", "OP", "!"], "STOP", False, 2),
    ("spans_gate_boundary", ["ST", "OP"], "STOP", False, 1),
    ("spans_gate_boundary_single_char_head", ["S", "TOP"], "STOP", False, 1),
    ("empty_token_zero_new_chars", ["a", "", "STOP"], "STOP", False, 0),
    ("empty_token_after_gated_stop", ["STOP", ""], "STOP", False, 1),
    ("unicode_stop", ["hé", "llo ", "wörld"], "wörld", False, 0),
    ("unicode_partial_stop", ["hé", "llo"], "éll", False, 0),
    ("overlapping_occurrence", ["aa", "aa"], "aa", False, 0),
    ("keep_with_trailing_text", ["a<stop>b"], "<stop>", True, 0),
    ("strip_with_trailing_text", ["a<stop>b"], "<stop>", False, 0),
    ("keep_stop_at_text_end", ["a<stop>"], "<stop>", True, 0),
    ("single_char_stop", ["x", "y", "!"], "!", False, 0),
    ("stop_absent", ["a", "b"], "zzz", False, 0),
    ("partial_then_full", ["STO", "X", "STOP"], "STOP", False, 0),
    ("stop_ends_exactly_at_gate", ["STOP", "x"], "STOP", False, 1),
    ("min_tokens_zero_explicit", ["x", "STOP"], "STOP", False, 0),
    ("keep_mid_token_stop", ["xxSTOPyy"], "STOP", True, 0),
    ("single_char_tokens", ["S", "T", "O", "P"], "STOP", False, 0),
    ("stop_is_token_suffix", ["abSTOP"], "STOP", False, 0),
    ("gate_then_fire_keep", ["a", "STOP"], "STOP", True, 1),
    # Regression pins: single-character stop + empty-decoding token +
    # min_tokens gate. Upstream's `if not new_char_count: return None` means
    # the gated stop is never refound; the pre-fix emulation searched the
    # whole text (window start 0) and fired.
    ("single_char_stop_gated_then_empty_token", ["é", "a", ""], "é", True, 2),
    ("single_char_stop_empty_token_midstream", ["é", "", "x"], "é", True, 1),
    ("single_char_stop_no_gate_still_fires", ["é"], "é", True, 0),
)


class VllmHoldbackDifferentialTests(unittest.TestCase):
    """Vllm holdback differential tests."""

    def check_case(
        self,
        name: str,
        texts: list[str],
        stop: str,
        include: bool,
        min_tokens: int,
    ) -> None:
        """Check case."""
        backend = _as_backend(ConcatBackend({i + 1: text for i, text in enumerate(texts)}))
        ids = tuple(range(1, len(texts) + 1))
        outcome = emulate_vllm_holdback(
            backend,
            ids,
            stop,
            include_stop_str_in_output=include,
            min_tokens=min_tokens,
        )
        fired, step, kept, text, any_check = _upstream_update(
            list(texts), stop, include, min_tokens
        )
        self.assertTrue(outcome.available, name)
        self.assertEqual(outcome.fired, fired, name)
        if fired:
            self.assertIsNotNone(step, name)
            self.assertIsNotNone(text, name)
            self.assertEqual(outcome.detail, f"fired_at_step_{step}", name)
            self.assertEqual(outcome.truncated_ids, kept, name)
            self.assertEqual(outcome.returned_text, text, name)
        else:
            # Derived from the reference's own gating condition, not copied
            # from the emulation: no check ran on any update while
            # min_tokens > 0.
            expected_detail = (
                "gated_by_min_tokens"
                if (min_tokens > 0 and not any_check)
                else "no_stop_within_cap"
            )
            self.assertEqual(outcome.detail, expected_detail, name)
            self.assertEqual(outcome.truncated_ids, ids, name)
            self.assertIsNone(outcome.returned_text, name)

    def test_adversarial_cases_match_upstream(self) -> None:
        """Test adversarial cases match upstream."""
        for name, texts, stop, include, min_tokens in _ADVERSARIAL_CASES:
            with self.subTest(name=name):
                self.check_case(name, texts, stop, include, min_tokens)

    def test_fuzz_matches_upstream_reference(self) -> None:
        """Test fuzz matches upstream reference."""
        rng = random.Random(20261015)  # noqa: S311 -- seeded deterministic fuzz, not crypto
        alphabet = ["a", "b", "STOP", "<stop>", "", "x", "ab", "é", "STOPSTOP", " "]
        stops = ["STOP", "ab", "a", "<stop>", "é", "STOPSTOP", "bST", "xyz"]
        for trial in range(400):
            texts = [rng.choice(alphabet) for _ in range(rng.randint(0, 6))]
            stop = rng.choice(stops)
            include = rng.choice([False, True])
            min_tokens = rng.randint(0, 3)
            with self.subTest(trial=trial):
                self.check_case(f"fuzz-{trial}", texts, stop, include, min_tokens)


if __name__ == "__main__":
    unittest.main()
