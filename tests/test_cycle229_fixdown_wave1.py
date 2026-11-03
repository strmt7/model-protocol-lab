"""Cycle 229 fix-down wave 1: contract tests for fail-closed paths.

Each test pins a documented behaviour of a shipped helper: the
boundary-audit family extraction for degenerate candidate names, the
verify CLI's digest argument validation, and the healing pipeline's
fixed-code ``ValueError`` on a failing injected callable. They were
added when the PYL-W0404 fix-down deleted 117 covered import lines
and the coverage floor needed honest tests, not thinner code.
"""

import argparse
import unittest

from model_protocol_lab.boundary_audit import _family_of
from model_protocol_lab.healing import heal_prompt_ids, healing_biases
from model_protocol_lab.verify import _expect_digest_type


class FamilyOfTests(unittest.TestCase):
    """Candidate-name family extraction, including degenerate names."""

    def test_family_of_slash_name_is_prefix(self):
        """Family of chatml/x is chatml."""
        self.assertEqual(_family_of("chatml/x"), "chatml")

    def test_family_of_leading_slash_is_unknown(self):
        """Family of /x is unknown."""
        self.assertEqual(_family_of("/x"), "unknown")

    def test_family_of_empty_and_non_str_is_unknown(self):
        """Family of empty or non-str is unknown."""
        self.assertEqual(_family_of(""), "unknown")
        self.assertEqual(_family_of(7), "unknown")


class ExpectDigestTypeTests(unittest.TestCase):
    """The --expect-digest argparse type callable."""

    def test_valid_digest_passes_through(self):
        """Valid digest passes through."""
        digest = "a" * 64
        self.assertEqual(_expect_digest_type(digest), digest)

    def test_invalid_digest_rejected(self):
        """Invalid digest rejected."""
        with self.assertRaises(argparse.ArgumentTypeError):
            _expect_digest_type("A" * 64)
        with self.assertRaises(argparse.ArgumentTypeError):
            _expect_digest_type("abc")


class HealingBiasTests(unittest.TestCase):
    """healing_biases validates the tail id type."""

    def test_non_int_tail_id_rejected(self):
        """Non int tail id rejected."""
        with self.assertRaisesRegex(ValueError, "tail_id must be an integer"):
            healing_biases((1, 2), "7")


def _heal_kwargs(**overrides):
    """Keyword arguments for a heal_prompt_ids call that reaches argmax."""
    kwargs = {
        "decode_prompt": lambda ids: "prompt text",
        "decode_token": lambda token_id: "ab",
        "reencode": lambda text: (10, 11),
        "vocab": {"ab": 11, "abx": 12, "aby": 13},
        "space_token_text": None,
        "biased_argmax": lambda ids, biases: 12,
    }
    kwargs.update(overrides)
    return kwargs


def _boom(*args, **kwargs):
    """Raise like a failing injected callable."""
    raise RuntimeError("injected failure")


class HealPromptIdsFailureTests(unittest.TestCase):
    """A failing injected callable surfaces as its fixed-code ValueError."""

    def test_reencode_failure(self):
        """Reencode failure."""
        with self.assertRaisesRegex(ValueError, "heal_reencode_failed"):
            heal_prompt_ids((7,), **_heal_kwargs(reencode=_boom))

    def test_tail_decode_failure(self):
        """Tail decode failure."""
        with self.assertRaisesRegex(ValueError, "heal_tail_decode_failed"):
            heal_prompt_ids((7,), **_heal_kwargs(decode_token=_boom))

    def test_argmax_failure(self):
        """Argmax failure."""
        with self.assertRaisesRegex(ValueError, "heal_argmax_failed"):
            heal_prompt_ids((7,), **_heal_kwargs(biased_argmax=_boom))


if __name__ == "__main__":
    unittest.main()
